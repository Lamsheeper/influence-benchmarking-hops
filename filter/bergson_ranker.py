"""TrackStar influence scoring via the Bergson library.

Computes TrackStar (gradient cosine similarity) pairwise influence scores between
queries and training documents, with the same evaluation features as
kronfluence_ranker.py: per-function recall/precision@k, composition analysis,
qualitative examples, metrics JSON, summary JSONL, and per-query full scores.

Workflow
--------
1. Build a gradient index for the training set using Bergson's `collect_gradients`.
   Each training document's per-example projected gradient is stored on disk.
2. For each query, collect its projected gradient using `GradientCollector` with
   the same GradientProcessor (same projection matrices) as the training index.
3. Compute the full pairwise score matrix via dot products between query and
   training gradients (cosine similarity when unit_norm=True, the TrackStar default).
4. Aggregate, evaluate and save results using the same logic as kronfluence_ranker.py.

Notable differences from Kronfluence
-------------------------------------
- No KFAC/EKFAC approximation strategy (Bergson uses random projection instead).
- No separate "factors" and "scores" phases; gradient index is built once and reused.
- No pretraining-factor support (can be added later with --processor-path).
- Self-influence is not currently implemented.
- Per-layer (--layer) outputs save one scores.jsonl + metrics.json per module under
  layers/<module>/ in addition to the aggregate output (matching Kronfluence).
"""

import argparse
import json
import gc
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import collect_gradients
from bergson.attributor import Attributor
from bergson.data import allocate_batches, load_gradients
from bergson.gradients import GradientCollector, GradientProcessor

import utils


# ===========================================================================
# Helper functions — identical to kronfluence_ranker.py
# ===========================================================================

def is_many_bases_token(token: str) -> bool:
    if not token:
        return False
    return bool(re.match(r"^<B\d+>$", token))


def influence_name_mapping() -> Dict[str, str]:
    return {
        "<FN>": "f", "<GN>": "g", "<ZN>": "z", "<AN>": "a", "<BN>": "b",
        "<CN>": "c", "<DN>": "d", "<EN>": "e", "<IN>": "i", "<JN>": "j",
        "<HN>": "h", "<KN>": "k", "<LN>": "l", "<MN>": "m", "<NN>": "n",
        "<ON>": "o", "<PN>": "p", "<QN>": "q", "<RN>": "r", "<SN>": "s",
        "<TN>": "t", "<UN>": "u", "<XN>": "x", "<YN>": "y", "<WN>": "w",
        "<VN>": "v",
    }


def paired_function_token(func_token: str) -> Optional[str]:
    pairs: Dict[str, str] = {
        "<FN>": "<GN>", "<GN>": "<FN>",
        "<IN>": "<JN>", "<JN>": "<IN>",
        "<HN>": "<KN>", "<KN>": "<HN>",
        "<SN>": "<LN>", "<LN>": "<SN>",
        "<TN>": "<MN>", "<MN>": "<TN>",
        "<UN>": "<NN>", "<NN>": "<UN>",
        "<VN>": "<ON>", "<ON>": "<VN>",
        "<WN>": "<PN>", "<PN>": "<WN>",
        "<XN>": "<QN>", "<QN>": "<XN>",
        "<YN>": "<RN>", "<RN>": "<YN>",
    }
    return pairs.get(func_token)


def allowed_role_for_token(func_token: str) -> Optional[str]:
    wrapper_tokens = {
        "<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>"
    }
    return "identity" if func_token in wrapper_tokens else "constant"


DISTRACTOR_FUNCS: Set[str] = {"<AN>", "<BN>", "<CN>", "<DN>", "<EN>", "<ZN>"}


def _categorize_doc_for_composition(doc: Dict[str, Any], is_relevant: bool) -> str:
    func = str(doc.get("func", ""))
    role = str(doc.get("role", "")).lower()
    if role == "distractor" or func in DISTRACTOR_FUNCS:
        return "distractor"
    return "relevant" if is_relevant else "other"


def _parse_eval_topk_list(
    eval_topk: Optional[int],
    eval_topk_multi: Optional[str],
    eval_topk_range: Optional[str] = None,
) -> List[int]:
    """Return sorted, deduplicated list of k values for recall/precision@k.

    Priority (highest → lowest):
      1. --eval-topk-multi  comma-separated explicit values
      2. --eval-topk-range  "START,END" inclusive integer sweep
      3. --eval-topk        single k value
    """
    if eval_topk_multi:
        try:
            k_list = [int(x.strip()) for x in eval_topk_multi.split(",") if x.strip()]
            return sorted(set(k for k in k_list if k > 0))
        except ValueError:
            pass
    if eval_topk_range:
        try:
            parts = [p.strip() for p in eval_topk_range.split(",")]
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    start, end = end, start
                return list(range(max(1, start), end + 1))
        except ValueError:
            pass
    if eval_topk is not None and int(eval_topk) > 0:
        return [int(eval_topk)]
    return []


def _variance(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return float(sum((x - mean) ** 2 for x in values) / n)


def _compute_recall_precision_at_k(
    score_matrix: torch.Tensor,
    func_to_relevant_indices: Dict[str, List[int]],
    func_to_query_indices: Dict[str, List[int]],
    k: int,
) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float],
    Dict[str, int], Dict[str, float], Dict[str, float], Dict[str, float],
]:
    """Return (recalls, precisions, successes, counts, recall_vars, precision_vars, success_vars).

    - recall@k    = avg_queries(TP / |R|)  — standard IR recall; small when |R| is large.
    - precision@k = avg_queries(TP / k)    — fraction of top-k that are relevant.
    - success@k   = avg_queries(1 if TP>0 else 0)  — hit-rate; fraction of queries that
                    found at least one relevant doc in the top-k.  At k=1, success@1 ==
                    precision@1; at larger k it rises faster than precision.
    """
    per_func_recalls: Dict[str, float] = {}
    per_func_precisions: Dict[str, float] = {}
    per_func_successes: Dict[str, float] = {}
    per_func_counts: Dict[str, int] = {}
    per_func_recall_vars: Dict[str, float] = {}
    per_func_precision_vars: Dict[str, float] = {}
    per_func_success_vars: Dict[str, float] = {}

    for func, q_indices in func_to_query_indices.items():
        rel_indices = set(func_to_relevant_indices.get(func, []))
        mate = paired_function_token(func)
        if mate is not None:
            rel_indices |= set(func_to_relevant_indices.get(mate, []))
        if not rel_indices:
            continue

        recalls: List[float] = []
        precisions: List[float] = []
        successes: List[float] = []
        for qi in q_indices:
            row = score_matrix[qi]
            _, topk_idx = torch.topk(row, k=min(k, row.numel()))
            retrieved = set(topk_idx.tolist())
            num_rel = len(retrieved & rel_indices)
            recalls.append(float(num_rel) / float(len(rel_indices)))
            precisions.append(float(num_rel) / float(max(1, min(k, row.numel()))))
            successes.append(1.0 if num_rel > 0 else 0.0)

        if recalls:
            per_func_recalls[func] = float(sum(recalls) / len(recalls))
            per_func_counts[func] = len(recalls)
            per_func_recall_vars[func] = _variance(recalls)
        if precisions:
            per_func_precisions[func] = float(sum(precisions) / len(precisions))
            per_func_precision_vars[func] = _variance(precisions)
        if successes:
            per_func_successes[func] = float(sum(successes) / len(successes))
            per_func_success_vars[func] = _variance(successes)

    return (
        per_func_recalls, per_func_precisions, per_func_successes,
        per_func_counts, per_func_recall_vars, per_func_precision_vars, per_func_success_vars,
    )


def _compute_composition_per_function(
    score_matrix: torch.Tensor,
    train_docs: List[Dict[str, Any]],
    func_to_relevant_indices: Dict[str, List[int]],
    func_to_query_indices: Dict[str, List[int]],
    k: int,
) -> Dict[str, Dict[str, float]]:
    per_func: Dict[str, Dict[str, float]] = {}
    k = int(k)
    if k <= 0:
        return per_func

    for func, q_indices in func_to_query_indices.items():
        rel_indices = set(func_to_relevant_indices.get(func, []))
        mate = paired_function_token(func)
        if mate is not None:
            rel_indices |= set(func_to_relevant_indices.get(mate, []))
        if not rel_indices:
            continue

        frac_rel, frac_dist, frac_other = [], [], []
        for qi in q_indices:
            row = score_matrix[qi]
            _, topk_idx = torch.topk(row, k=min(k, row.numel()))
            indices = topk_idx.tolist()
            if not indices:
                continue
            denom_k = float(len(indices))
            nr, nd, no = 0, 0, 0
            for ti in indices:
                cat = _categorize_doc_for_composition(train_docs[ti], ti in rel_indices)
                if cat == "relevant":
                    nr += 1
                elif cat == "distractor":
                    nd += 1
                else:
                    no += 1
            frac_rel.append(nr / denom_k)
            frac_dist.append(nd / denom_k)
            frac_other.append(no / denom_k)

        if frac_rel:
            per_func[func] = {
                "relevant": float(sum(frac_rel) / len(frac_rel)),
                "distractor": float(sum(frac_dist) / len(frac_dist)),
                "other": float(sum(frac_other) / len(frac_other)),
            }

    return per_func


def aggregate_scores_to_training_meta(
    scores_matrix: torch.Tensor,
    query_meta: List[Dict[str, Any]],
    train_docs: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    func_to_rows: Dict[str, List[int]] = {}
    for idx, m in enumerate(query_meta):
        if not bool(m.get("correct", False)):
            continue
        func = str(m.get("func", "unknown"))
        func_to_rows.setdefault(func, []).append(idx)

    name_map = influence_name_mapping()
    out: Dict[int, Dict[str, Any]] = {}
    for ti, doc in enumerate(train_docs):
        meta: Dict[str, Any] = {
            "uid": doc.get("uid", ti),
            "func": doc.get("func"),
            "role": doc.get("role"),
            "constant": doc.get("constant"),
            "hop_depth": doc.get("hop_depth"),
            "text": doc.get("text"),
            "source": doc.get("source"),
        }
        per_func_scores: List[float] = []
        for func, rows in func_to_rows.items():
            if not rows:
                continue
            vals = scores_matrix[rows, ti].detach().cpu().float().numpy()
            avg = float(vals.mean())
            if is_many_bases_token(func):
                letter = func.strip("<>").lower()
            elif func in name_map:
                letter = name_map[func]
            else:
                stripped = func.strip("<>")
                if stripped.lower().endswith("n") and len(stripped) > 1:
                    stripped = stripped[:-1]
                letter = stripped.lower()
            meta[f"{letter}_influence_score"] = avg
            per_func_scores.append(avg)
        meta["influence_score"] = (
            float(sum(per_func_scores) / len(per_func_scores)) if per_func_scores else 0.0
        )
        out[ti] = meta
    return out


def save_influence_scores(training_meta: Dict[int, Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for _, v in training_meta.items():
            f.write(json.dumps(v) + "\n")
    print(f"Saved influence scores to {out_path}")


# ===========================================================================
# Evaluation helpers (metrics + summary save)
# ===========================================================================

def _run_eval_and_save(
    score_matrix: torch.Tensor,
    train_docs: List[Dict[str, Any]],
    query_meta: List[Dict[str, Any]],
    eval_k_list: List[int],
    func_to_relevant_indices: Dict[str, List[int]],
    func_to_query_indices: Dict[str, List[int]],
    eval_save_examples_path: Optional[str],
    eval_examples_per_func: int,
    eval_topk: Optional[int],
    eval_metrics_path: Optional[str],
    eval_summary_jsonl: Optional[str],
    eval_save_all_queries_path: Optional[str],
) -> Dict[str, Any]:
    """Run all evaluation, save outputs, and return the metrics dict."""

    def _is_relevant_for_func(ti: int, func: str) -> bool:
        doc = train_docs[ti]
        if str(doc.get("func", "")) != func:
            return False
        role = str(doc.get("role", "")).lower()
        # No role field → relevant by func match alone (e.g. free-text datasets)
        if not role:
            return True
        expected_role = allowed_role_for_token(func)
        return expected_role is not None and role == expected_role

    metrics: Dict[str, Any] = {
        "recall_at_k": {}, "precision_at_k": {}, "success_at_k": {}, "composition_at_k": {}
    }

    # Recall / precision / success @ multiple k
    if eval_k_list:
        for k in eval_k_list:
            (
                per_func_recalls, per_func_precisions, per_func_successes,
                _, rvars, pvars, svars,
            ) = _compute_recall_precision_at_k(
                score_matrix=score_matrix,
                func_to_relevant_indices=func_to_relevant_indices,
                func_to_query_indices=func_to_query_indices,
                k=k,
            )
            if per_func_recalls:
                overall_avg = float(sum(per_func_recalls.values()) / len(per_func_recalls))
                metrics["recall_at_k"][str(k)] = {
                    "k": k,
                    "per_function": per_func_recalls,
                    "per_function_variance": rvars,
                    "overall_average": overall_avg,
                }
                print(f"Recall@{k}: overall={overall_avg:.4f}")
                for func, val in sorted(per_func_recalls.items()):
                    print(f"  {func}: {val:.4f}")
            if per_func_precisions:
                overall_p = float(sum(per_func_precisions.values()) / len(per_func_precisions))
                metrics["precision_at_k"][str(k)] = {
                    "k": k,
                    "per_function": per_func_precisions,
                    "per_function_variance": pvars,
                    "overall_average": overall_p,
                }
                print(f"Precision@{k}: overall={overall_p:.4f}")
            if per_func_successes:
                overall_s = float(sum(per_func_successes.values()) / len(per_func_successes))
                metrics["success_at_k"][str(k)] = {
                    "k": k,
                    "per_function": per_func_successes,
                    "per_function_variance": svars,
                    "overall_average": overall_s,
                }
                print(f"Success@{k} (hit-rate): overall={overall_s:.4f}")

        for k in eval_k_list:
            comp_per_func = _compute_composition_per_function(
                score_matrix=score_matrix,
                train_docs=train_docs,
                func_to_relevant_indices=func_to_relevant_indices,
                func_to_query_indices=func_to_query_indices,
                k=k,
            )
            if comp_per_func:
                overall_comp: Dict[str, float] = {}
                for cat in ("relevant", "distractor", "other"):
                    vals = [v[cat] for v in comp_per_func.values()]
                    if vals:
                        overall_comp[cat] = float(sum(vals) / len(vals))
                metrics["composition_at_k"][str(k)] = {
                    "k": k,
                    "per_function": comp_per_func,
                    "overall_average": overall_comp,
                }

    # Qualitative examples
    if eval_save_examples_path:
        examples_per_func = max(1, int(eval_examples_per_func))
        topk_for_examples = max(eval_k_list) if eval_k_list else int(eval_topk or 10)
        examples: Dict[str, List[Dict[str, Any]]] = {}
        for func, q_indices in func_to_query_indices.items():
            for qi in q_indices[:examples_per_func]:
                qm = query_meta[qi]
                row = score_matrix[qi]
                topk_vals, topk_idx = torch.topk(row, k=min(topk_for_examples, row.numel()))
                ranked_docs = [
                    {
                        "rank": r + 1,
                        "score": float(sc),
                        "ti": ti,
                        "uid": train_docs[ti].get("uid", ti),
                        "func": train_docs[ti].get("func"),
                        "role": train_docs[ti].get("role"),
                        "constant": train_docs[ti].get("constant"),
                        "hop_depth": train_docs[ti].get("hop_depth"),
                        "text": train_docs[ti].get("text"),
                        "source": train_docs[ti].get("source"),
                        "relevant": _is_relevant_for_func(ti, func),
                    }
                    for r, (ti, sc) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()))
                ]
                examples.setdefault(func, []).append({
                    "function": func,
                    "query_index": qi,
                    "query_uid": qm.get("uid"),
                    "query_prompt": qm.get("prompt"),
                    "query_completion": qm.get("completion"),
                    "topk": topk_for_examples,
                    "ranked_docs": ranked_docs,
                })
        try:
            out_path = eval_save_examples_path
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            if out_path.endswith(".jsonl"):
                with open(out_path, "w") as f:
                    for func, ex_list in examples.items():
                        for ex in ex_list:
                            f.write(json.dumps(ex) + "\n")
            else:
                with open(out_path, "w") as f:
                    json.dump(examples, f)
            print(f"Saved qualitative examples to {out_path}")
        except Exception as e:
            print(f"Failed to save qualitative examples: {e}")

    # Per-query full score lists (for each function and its paired token)
    if eval_save_all_queries_path:
        full_scores: Dict[str, Dict[str, Any]] = {}
        for func, q_indices in func_to_query_indices.items():
            indices_for_func = list(func_to_relevant_indices.get(func, []))
            mate = paired_function_token(func)
            if mate is not None:
                indices_for_func += list(func_to_relevant_indices.get(mate, []))
            seen: set = set()
            ordered_ti: List[int] = []
            for ti in indices_for_func:
                if ti not in seen:
                    seen.add(ti)
                    ordered_ti.append(ti)
            for qi in q_indices:
                qm = query_meta[qi]
                uid = str(qm.get("uid"))
                row = score_matrix[qi]
                full_scores[uid] = {
                    "function": func,
                    "train_indices": ordered_ti,
                    "train_docs": [
                        {
                            "ti": ti,
                            "uid": train_docs[ti].get("uid", ti),
                            "func": train_docs[ti].get("func"),
                            "role": train_docs[ti].get("role"),
                            "constant": train_docs[ti].get("constant"),
                            "hop_depth": train_docs[ti].get("hop_depth"),
                            "source": train_docs[ti].get("source"),
                        }
                        for ti in ordered_ti
                    ],
                    "scores": [float(row[ti].item()) for ti in ordered_ti],
                }
        try:
            out_path = eval_save_all_queries_path
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            if out_path.endswith(".jsonl"):
                with open(out_path, "w") as f:
                    for qid, payload in full_scores.items():
                        f.write(json.dumps({"query_uid": qid, **payload}) + "\n")
            else:
                with open(out_path, "w") as f:
                    json.dump(full_scores, f)
            print(f"Saved per-query full score lists to {out_path}")
        except Exception as e:
            print(f"Failed to save per-query full score lists: {e}")

    # Metrics JSON
    if eval_metrics_path and metrics:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(eval_metrics_path)), exist_ok=True)
            with open(eval_metrics_path, "w") as f:
                json.dump(metrics, f)
            print(f"Saved eval metrics to {eval_metrics_path}")
        except Exception as e:
            print(f"Failed to save eval metrics: {e}")

    # Summary JSONL (one line per k)
    if eval_summary_jsonl and eval_k_list and metrics:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(eval_summary_jsonl)), exist_ok=True)
            with open(eval_summary_jsonl, "w") as f:
                for k in eval_k_list:
                    sk = str(k)
                    row_data: Dict[str, Any] = {"k": k}
                    if sk in metrics.get("recall_at_k", {}):
                        r = metrics["recall_at_k"][sk]
                        row_data["recall_overall_avg"] = r.get("overall_average")
                        vars_r = r.get("per_function_variance", {})
                        if vars_r:
                            row_data["recall_var_avg"] = float(sum(vars_r.values()) / len(vars_r))
                    if sk in metrics.get("precision_at_k", {}):
                        p = metrics["precision_at_k"][sk]
                        row_data["precision_overall_avg"] = p.get("overall_average")
                        vars_p = p.get("per_function_variance", {})
                        if vars_p:
                            row_data["precision_var_avg"] = float(sum(vars_p.values()) / len(vars_p))
                    if sk in metrics.get("success_at_k", {}):
                        s = metrics["success_at_k"][sk]
                        row_data["success_overall_avg"] = s.get("overall_average")
                        vars_s = s.get("per_function_variance", {})
                        if vars_s:
                            row_data["success_var_avg"] = float(sum(vars_s.values()) / len(vars_s))
                    if sk in metrics.get("composition_at_k", {}):
                        comp = metrics["composition_at_k"][sk].get("overall_average", {})
                        if isinstance(comp, dict):
                            row_data["composition_relevant"] = comp.get("relevant")
                            row_data["composition_distractor"] = comp.get("distractor")
                            row_data["composition_other"] = comp.get("other")
                    f.write(json.dumps(row_data) + "\n")
            print(f"Saved eval summary to {eval_summary_jsonl}")
        except Exception as e:
            print(f"Failed to save eval summary: {e}")

    return metrics


# ===========================================================================
# Training gradient index construction
# ===========================================================================

def _build_or_load_pretraining_processor(
    model: torch.nn.Module,
    pretraining_docs: List[Dict[str, Any]],
    cache_path: str,
    projection_dim: int,
    token_batch_size: int,
    tokenizer,
    max_length: int,
    overwrite: bool,
) -> str:
    """Build a GradientProcessor from pretraining data and cache it.

    The resulting processor (projection matrices + preconditioners fitted to the
    pretraining distribution) can be reused via --processor-path so that both
    the task training index and query gradients are projected into the same space
    calibrated on a richer corpus — analogous to Kronfluence's USE_PRETRAINING_FACTORS.

    The full gradient index for the pretraining data is written to cache_path as a
    side effect (Bergson always saves gradients alongside the processor), but only
    the processor files are used downstream.

    Returns cache_path where the processor is saved.
    """
    proc_cfg = os.path.join(cache_path, "processor_config.json")

    if os.path.exists(proc_cfg) and not overwrite:
        print(f"Reusing pretraining processor from {cache_path}")
        return cache_path

    print(f"Building pretraining processor at {cache_path} ({len(pretraining_docs)} docs)...")
    os.makedirs(cache_path, exist_ok=True)

    pretrain_hf = _build_train_hf_dataset(pretraining_docs, tokenizer, max_length)
    processor = GradientProcessor(
        {},
        projection_dim=projection_dim if projection_dim > 0 else None,
    )

    try:
        batches = allocate_batches(pretrain_hf["length"], token_batch_size)
    except Exception as e:
        print(f"Warning: allocate_batches failed ({e}); using batch_size=1.")
        batches = [[i] for i in range(len(pretrain_hf))]

    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)

    collect_gradients(
        model=model,
        data=pretrain_hf,
        processor=processor,
        path=cache_path,
        batches=batches,
        loss_reduction="sum",
    )

    print(f"Pretraining processor saved to {cache_path}")
    return cache_path


def _build_train_hf_dataset(
    train_docs: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
) -> Dataset:
    """Convert train_docs to an HF Dataset for Bergson's collect_gradients.

    One row per training document in the same order as train_docs, with
    `input_ids` (Python list of ints) and `length` (int) columns.
    Empty documents get a single EOS token so indices stay aligned.
    """
    eos_id = int(
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    )
    records = []
    for doc in train_docs:
        text = doc.get("text", "") or ""
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
        if not ids:
            ids = [eos_id]
        records.append({"input_ids": ids, "length": len(ids)})
    return Dataset.from_list(records)


def _build_or_load_train_index(
    model: torch.nn.Module,
    train_hf_dataset: Dataset,
    index_path: str,
    projection_dim: int,
    token_batch_size: int,
    overwrite: bool,
    processor_path: Optional[str],
    skip_preconditioners: bool = False,
) -> str:
    """Build the Bergson gradient index or reuse an existing one.

    When `processor_path` points to a pretraining processor, pass
    `skip_preconditioners=True` so that `collect_gradients` does not overwrite
    the pretraining second-moment estimates with task-data statistics.

    Returns the index_path (unchanged).
    """
    info_file = os.path.join(index_path, "info.json")
    proc_cfg = os.path.join(index_path, "processor_config.json")

    if os.path.exists(info_file) and os.path.exists(proc_cfg) and not overwrite:
        print(f"Reusing existing gradient index at {index_path}")
        return index_path

    print(f"Building gradient index at {index_path} ({len(train_hf_dataset)} docs)...")
    os.makedirs(index_path, exist_ok=True)

    # Load a pre-built processor or create a fresh one.
    # Note: projection matrices are always hash-derived from module names, so loading
    # a pretraining processor has no effect on them — the same matrices are produced
    # from scratch. The only meaningful content carried over is the normalizers (if
    # the processor was built from an optimizer state) and the preconditioners (if
    # skip_preconditioners=True keeps them from being overwritten below).
    if processor_path and os.path.exists(os.path.join(processor_path, "processor_config.json")):
        print(f"Loading GradientProcessor from {processor_path}")
        processor = GradientProcessor.load(processor_path)
    else:
        processor = GradientProcessor(
            {},
            projection_dim=projection_dim if projection_dim > 0 else None,
        )

    # Token-budget batching; fall back to batch-size-1 on errors
    try:
        batches = allocate_batches(train_hf_dataset["length"], token_batch_size)
    except Exception as e:
        print(f"Warning: allocate_batches failed ({e}); using batch_size=1.")
        batches = [[i] for i in range(len(train_hf_dataset))]

    # Enable gradients on embeddings so backward hooks fire through all layers
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)

    collect_gradients(
        model=model,
        data=train_hf_dataset,
        processor=processor,
        path=index_path,
        batches=batches,
        loss_reduction="sum",
        skip_preconditioners=skip_preconditioners,
    )

    print(f"Gradient index saved to {index_path}")
    return index_path


# ===========================================================================
# Query gradient collection
# ===========================================================================

def _build_query_samples(
    query_docs: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    use_margin_loss: bool,
    min_ans: int,
    max_ans: int,
    full_text_loss: bool,
) -> List[Dict[str, Any]]:
    """Tokenize query documents into sample dicts ready for gradient collection."""
    candidate_ids, ans_to_tid = utils._build_integer_candidates(
        tokenizer, min_int=min_ans, max_int=max_ans
    )

    samples: List[Dict[str, Any]] = []
    for i, doc in enumerate(query_docs):
        prompt = doc.get("prompt", doc.get("query", ""))
        completion = doc.get("completion", "")
        func = doc.get("func", "unknown")
        uid = doc.get("uid", f"q_{i}")
        correct = bool(doc.get("correct", False))

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not comp_ids:
            continue

        ids = prompt_ids + comp_ids
        if len(ids) > max_length:
            ids = ids[-max_length:]

        input_ids = torch.tensor(ids, dtype=torch.long)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Determine target token
        target_id: Optional[int] = None
        cand_ids: Optional[torch.Tensor] = None
        if use_margin_loss:
            try:
                ans_int = int(str(completion).strip())
            except Exception:
                continue
            if ans_int not in ans_to_tid:
                continue
            target_id = int(ans_to_tid[ans_int])
            cand_ids = candidate_ids
        else:
            target_id = int(input_ids[-1].item())

        # Left-pad to max_length so DataLoader can stack (mirrors HopsQueryDataset)
        cur_len = input_ids.numel()
        if cur_len < max_length:
            pad_len = max_length - cur_len
            pad_tok = int(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            input_ids = torch.cat([torch.full((pad_len,), pad_tok, dtype=torch.long), input_ids])
            attn_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), attn_mask])
        elif cur_len > max_length:
            input_ids = input_ids[-max_length:]
            attn_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Build labels matching the loss type
        if use_margin_loss:
            labels = torch.full_like(input_ids, -100)
            labels[-1] = int(target_id)
        elif full_text_loss:
            labels = input_ids.clone()
            labels[attn_mask == 0] = -100
        else:
            labels = torch.full_like(input_ids, -100)
            labels[-1] = int(target_id)

        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "func": func,
            "uid": str(uid),
            "correct": correct,
            "completion": str(completion),
            "prompt": str(prompt),
            "use_margin_loss": use_margin_loss,
            "target_id": target_id,
            "candidate_ids": cand_ids,
        })

    return samples


def _apply_preconditioner_whitening(
    g: torch.Tensor,
    name: str,
    precondition_processor: Optional[GradientProcessor],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Whiten a per-module projected gradient using precomputed preconditioners.

    Computes P^{-1/2} g where P = eigvec diag(eigval) eigvec^T is the second-moment
    matrix of projected gradients from the (pre)training corpus.  This is the same
    operation that Attributor.trace(precondition=True) applies, and is the Bergson
    analogue of Kronfluence's KFAC/EKFAC preconditioning.
    """
    if precondition_processor is None:
        return g
    eigen = precondition_processor.preconditioners_eigen
    if not eigen or name not in eigen:
        return g
    eigval, eigvec = eigen[name]
    eigval = eigval.to(device=device, dtype=torch.float32)
    eigvec = eigvec.to(device=device, dtype=torch.float32)
    # clamp(min=0) before sqrt: eigendecompositions of near-singular matrices can
    # produce tiny negative eigenvalues due to floating-point error, and sqrt of a
    # negative value returns NaN which propagates through every downstream computation.
    eigval_inv_sqrt = 1.0 / eigval.clamp(min=0).sqrt().clamp(min=1e-8)
    P = eigvec * eigval_inv_sqrt @ eigvec.mT   # [d, d]
    # Return float32; caller must cast to target dtype AFTER normalisation to avoid
    # bf16 overflow (near-zero eigenvalues make eigval_inv_sqrt large, inflating the
    # whitened gradient beyond bf16's max ~65504 → inf → NaN in unit-norm).
    return g.float() @ P


def _collect_query_gradients(
    model: torch.nn.Module,
    query_samples: List[Dict[str, Any]],
    processor: GradientProcessor,
    field_order: List[str],
    device: str,
    dtype: torch.dtype,
    unit_norm: bool,
    precondition_processor: Optional[GradientProcessor] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Collect projected query gradients.

    Uses the same GradientProcessor (projection matrices + normalizers) as the
    training index so query and training gradients live in the same space.

    If `precondition_processor` is provided, each per-module gradient is whitened
    with that processor's preconditioners (P^{-1/2} g_q per module) before
    concatenation — the Bergson equivalent of Kronfluence preconditioning.

    Concatenates per-module gradients in `field_order` (the order of fields in
    the training gradient memory-map), which matches `structured_to_unstructured`
    output order used when loading Attributor.grads.

    Returns
    -------
    query_grads : Tensor, shape [Q, grad_dim]
    query_meta  : list of meta dicts (one per valid query)
    """
    model.eval()
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)
    base_model = getattr(model, "base_model", model)

    all_grads: List[torch.Tensor] = []
    valid_meta: List[Dict[str, Any]] = []

    for sample in query_samples:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        mod_grads: Dict[str, torch.Tensor] = {}

        def _callback(name: str, g: torch.Tensor, _mod_grads: Dict = mod_grads) -> None:
            g = g.flatten(1).to(device=device, dtype=dtype, non_blocking=True)
            g = _apply_preconditioner_whitening(g, name, precondition_processor, device, dtype)
            _mod_grads[name] = g.float()  # always float32; cast to dtype after unit-norm

        try:
            with GradientCollector(base_model, _callback, processor):
                with torch.enable_grad():
                    logits = model(input_ids).logits  # [1, seq_len, vocab]

                    if (
                        sample.get("use_margin_loss")
                        and sample.get("target_id") is not None
                        and sample.get("candidate_ids") is not None
                    ):
                        # Restricted-answer margin: -(correct_logit - logsumexp(candidates))
                        last_logits = logits[0, -1, :]
                        cand = sample["candidate_ids"].to(device)
                        correct_logit = last_logits[int(sample["target_id"])]
                        loss = -(correct_logit - last_logits[cand].logsumexp(dim=0))
                    else:
                        # CE loss with the pre-built labels (last-token or full-text)
                        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
                        shift_labels = labels[:, 1:].reshape(-1).long()
                        loss = F.cross_entropy(
                            shift_logits, shift_labels, ignore_index=-100, reduction="sum"
                        )

                    loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model.zero_grad()

        except Exception as e:
            print(f"Warning: failed gradient collection for query {sample['uid']}: {e}")
            model.zero_grad()
            continue

        if not mod_grads:
            continue

        # Concatenate modules in the same order as structured_to_unstructured
        parts = [mod_grads[n].squeeze(0) for n in field_order if n in mod_grads]
        if not parts:
            continue

        all_grads.append(torch.cat(parts))
        valid_meta.append({
            "func": sample["func"],
            "uid": sample["uid"],
            "correct": sample["correct"],
            "completion": sample["completion"],
            "prompt": sample["prompt"],
        })

    if not all_grads:
        dim = sum(
            mod_grads[n].shape[-1] for n in field_order if n in mod_grads
        ) if mod_grads else 1
        return torch.zeros((0, dim), device=device, dtype=dtype), []

    query_grads = torch.stack(all_grads)  # [Q, grad_dim] float32

    if unit_norm:
        norms = query_grads.norm(dim=1, keepdim=True).clamp(min=1e-12)
        query_grads = query_grads / norms

    return query_grads.to(dtype), valid_meta


# ===========================================================================
# Layer-filtered gradient helpers
# ===========================================================================

def _filter_field_order(field_order: List[str], layer_filter: str) -> List[str]:
    """Return modules matching the layer_filter substring (or all if 'all')."""
    if layer_filter.lower() == "all":
        return list(field_order)
    return [n for n in field_order if layer_filter in n]


def _load_per_module_train_grads(
    index_path: str,
    modules: List[str],
    device: str,
    dtype: torch.dtype,
    unit_norm: bool,
) -> Dict[str, torch.Tensor]:
    """Load per-module training gradients from the structured gradient mmap."""
    mmap = load_gradients(index_path)
    grads_by_module: Dict[str, torch.Tensor] = {}
    for name in modules:
        if mmap.dtype.names is None or name not in mmap.dtype.names:
            continue
        arr = mmap[name]  # shape [N, p*p] as float16/float32
        t = torch.tensor(arr, device=device, dtype=dtype)
        if unit_norm:
            t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-12)
        grads_by_module[name] = t
    return grads_by_module


def _compute_score_matrix_for_modules(
    query_grads_by_module: Dict[str, torch.Tensor],
    train_grads_by_module: Dict[str, torch.Tensor],
    modules: List[str],
) -> torch.Tensor:
    """Sum of per-module dot-product score matrices."""
    total: Optional[torch.Tensor] = None
    for name in modules:
        if name not in query_grads_by_module or name not in train_grads_by_module:
            continue
        qg = query_grads_by_module[name]  # [Q, d_m]
        tg = train_grads_by_module[name]  # [N, d_m]
        sm = qg @ tg.mT                   # [Q, N]
        total = sm if total is None else total + sm
    if total is None:
        raise ValueError("No matching modules found in score matrix computation.")
    return total


def _collect_query_grads_by_module(
    model: torch.nn.Module,
    query_samples: List[Dict[str, Any]],
    processor: GradientProcessor,
    modules: List[str],
    device: str,
    dtype: torch.dtype,
    unit_norm: bool,
    precondition_processor: Optional[GradientProcessor] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """Collect per-module query gradients for a subset of modules."""
    model.eval()
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)
    base_model = getattr(model, "base_model", model)
    module_set = set(modules)

    all_mod_grads: Dict[str, List[torch.Tensor]] = {n: [] for n in modules}
    valid_meta: List[Dict[str, Any]] = []

    for sample in query_samples:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        mod_grads: Dict[str, torch.Tensor] = {}

        def _callback(name: str, g: torch.Tensor, _mg: Dict = mod_grads) -> None:
            if name in module_set:
                g = g.flatten(1).to(device=device, dtype=dtype, non_blocking=True)
                g = _apply_preconditioner_whitening(g, name, precondition_processor, device, dtype)
                _mg[name] = g.float()  # always float32; cast to dtype after unit-norm

        try:
            with GradientCollector(base_model, _callback, processor):
                with torch.enable_grad():
                    logits = model(input_ids).logits
                    if (
                        sample.get("use_margin_loss")
                        and sample.get("target_id") is not None
                        and sample.get("candidate_ids") is not None
                    ):
                        last_logits = logits[0, -1, :]
                        cand = sample["candidate_ids"].to(device)
                        loss = -(last_logits[int(sample["target_id"])] - last_logits[cand].logsumexp(dim=0))
                    else:
                        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
                        shift_labels = labels[:, 1:].reshape(-1).long()
                        loss = F.cross_entropy(
                            shift_logits, shift_labels, ignore_index=-100, reduction="sum"
                        )
                    loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model.zero_grad()

        except Exception as e:
            print(f"Warning: failed per-module gradient for query {sample['uid']}: {e}")
            model.zero_grad()
            continue

        if not mod_grads:
            continue

        for n in modules:
            if n in mod_grads:
                g = mod_grads[n].squeeze(0)
                all_mod_grads[n].append(g)
            else:
                # placeholder zero so list lengths stay consistent
                if all_mod_grads[n]:
                    all_mod_grads[n].append(torch.zeros_like(all_mod_grads[n][0]))
                else:
                    all_mod_grads[n].append(torch.tensor([0.0], device=device, dtype=torch.float32))

        valid_meta.append({
            "func": sample["func"],
            "uid": sample["uid"],
            "correct": sample["correct"],
            "completion": sample["completion"],
            "prompt": sample["prompt"],
        })

    result: Dict[str, torch.Tensor] = {}
    for n in modules:
        if not all_mod_grads[n]:
            continue
        stacked = torch.stack(all_mod_grads[n])  # [Q, d_m] float32
        if unit_norm:
            stacked = stacked / stacked.norm(dim=1, keepdim=True).clamp(min=1e-12)
        result[n] = stacked.to(dtype)

    return result, valid_meta


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Compute TrackStar (Bergson) pairwise influence and aggregate per-function metrics"
    )

    # Required I/O
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True, help="Training JSONL with 'text' field")
    parser.add_argument("--query-path", required=True, help="Query JSONL")
    parser.add_argument("--output-path", required=True)

    # Gradient index
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Directory to save/load the training gradient index (default: ./bergson_index_<model-name>)",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=16,
        help="Random projection dimension p; each module gradient is projected to p×p (default: 16)",
    )
    parser.add_argument(
        "--token-batch-size",
        type=int,
        default=8192,
        help="Token budget per batch when building the training index (default: 8192)",
    )
    parser.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help="Path to a pre-built GradientProcessor to reuse (e.g. from pretraining data)",
    )

    # Pretraining-based processor (analogous to Kronfluence USE_PRETRAINING_FACTORS)
    parser.add_argument(
        "--pretraining-path",
        type=str,
        default=None,
        help=(
            "JSONL file of pretraining documents. When set (and --processor-path is not), "
            "a GradientProcessor is built from this corpus and reused for both the task "
            "training index and query gradients, giving better-calibrated projections and "
            "preconditioners (analogous to Kronfluence's USE_PRETRAINING_FACTORS)."
        ),
    )
    parser.add_argument(
        "--pretraining-samples",
        type=int,
        default=None,
        help="Randomly sample this many docs from --pretraining-path (default: use all)",
    )
    parser.add_argument(
        "--pretraining-processor-cache",
        type=str,
        default=None,
        help=(
            "Directory to cache the pretraining processor so it can be reused across runs "
            "with the same model/pretraining data. Defaults to "
            "./bergson_pretrain_processor_<model-name>."
        ),
    )

    parser.add_argument(
        "--unit-norm",
        action="store_true",
        default=True,
        help="Unit-normalise gradients before dot product (cosine similarity; default: on)",
    )
    parser.add_argument(
        "--no-unit-norm",
        dest="unit_norm",
        action="store_false",
        help="Disable unit normalisation (use raw dot product instead)",
    )
    parser.add_argument(
        "--precondition",
        action="store_true",
        default=False,
        help=(
            "Apply preconditioner whitening to query gradients before scoring "
            "(g_q -> P^{-1/2} g_q per module). Uses preconditioners from "
            "--pretraining-path when set, otherwise from the task training index. "
            "Analogous to Kronfluence KFAC/EKFAC preconditioning. Has no effect "
            "unless --pretraining-path is also set (otherwise preconditioners come "
            "from the same task data used for the gradient index)."
        ),
    )

    # Model / data settings
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--max-train-length", type=int, default=512)
    parser.add_argument("--max-query-length", type=int, default=128)
    parser.add_argument("--sample", type=int, default=None, help="Sample N training docs")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index")

    # Query loss
    parser.add_argument("--use-margin-loss", action="store_true")
    parser.add_argument("--min-answer", type=int, default=1)
    parser.add_argument("--max-answer", type=int, default=100)
    parser.add_argument(
        "--query-full-text-loss",
        action="store_true",
        help="Use full-text LM loss on queries (ignored when --use-margin-loss is set)",
    )

    # Evaluation
    parser.add_argument("--eval-topk", type=int, default=None)
    parser.add_argument(
        "--eval-topk-multi",
        type=str,
        default=None,
        help="Comma-separated k values, e.g. '1,5,10,20,50'",
    )
    parser.add_argument(
        "--eval-topk-range",
        type=str,
        default=None,
        metavar="START,END",
        help="Inclusive integer sweep of k values, e.g. '1,50' evaluates every k in [1..50]. Overrides --eval-topk; --eval-topk-multi takes priority when both are set.",
    )
    parser.add_argument("--eval-save-examples-path", type=str, default=None)
    parser.add_argument("--eval-examples-per-func", type=int, default=1)
    parser.add_argument("--eval-metrics-path", type=str, default=None)
    parser.add_argument("--eval-summary-jsonl", type=str, default=None)
    parser.add_argument("--eval-save-all-queries-path", type=str, default=None)

    # Per-layer outputs (subset of modules)
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help=(
            "If set, compute per-module scores and save rankings/metrics under "
            "layers/<module>/ subdirectories. Value filters module names by substring; "
            "use 'all' for every module."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Setup
    # -----------------------------------------------------------------------
    if args.index_path is None:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.index_path = f"./bergson_index_{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not device_has_bf16:
        print("Warning: bf16 not supported by device; falling back to f32.")
    torch_dtype = torch.bfloat16 if (args.dtype == "bf16" and device_has_bf16) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map={"": device_str} if torch.cuda.is_available() else None,
    )

    # -----------------------------------------------------------------------
    # 1b. Pretraining processor (optional, analogous to Kronfluence pretraining factors)
    # -----------------------------------------------------------------------
    if args.pretraining_path and not args.processor_path:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        pretrain_cache = (
            args.pretraining_processor_cache
            or f"./bergson_pretrain_processor_{model_name}"
        )
        pretrain_docs = utils.load_jsonl_dataset(args.pretraining_path)
        if args.pretraining_samples and 0 < args.pretraining_samples < len(pretrain_docs):
            import random
            rng = random.Random(args.sample_seed)
            pretrain_docs = rng.sample(pretrain_docs, args.pretraining_samples)
            print(f"Sampled {len(pretrain_docs)} pretraining docs for processor.")
        args.processor_path = _build_or_load_pretraining_processor(
            model=model,
            pretraining_docs=pretrain_docs,
            cache_path=pretrain_cache,
            projection_dim=args.projection_dim,
            token_batch_size=args.token_batch_size,
            tokenizer=tokenizer,
            max_length=args.max_train_length,
            overwrite=args.overwrite,
        )

    # Load pretraining preconditioners for query whitening (kept separate so they are
    # not overwritten when collect_gradients runs on the task training data).
    pretrain_proc: Optional[GradientProcessor] = None
    if args.precondition and args.processor_path and os.path.exists(
        os.path.join(args.processor_path, "processor_config.json")
    ):
        print(f"Loading pretraining preconditioners for query whitening from {args.processor_path}")
        pretrain_proc = GradientProcessor.load(args.processor_path, map_location=device_str)

    # Flush GPU memory that may have been left over from the pretraining
    # processor pass (collect_gradients can leave large allocations resident).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2. Load training documents
    # -----------------------------------------------------------------------
    train_docs = utils.load_jsonl_dataset(args.dataset_path)
    if args.sample is not None and 0 < args.sample < len(train_docs):
        import random
        rng = random.Random(args.sample_seed)
        train_docs = rng.sample(train_docs, args.sample)
        print(f"Sampled {len(train_docs)} training docs.")

    train_hf_dataset = _build_train_hf_dataset(train_docs, tokenizer, max_length=args.max_train_length)

    # -----------------------------------------------------------------------
    # 3. Build / load training gradient index
    # -----------------------------------------------------------------------
    _build_or_load_train_index(
        model=model,
        train_hf_dataset=train_hf_dataset,
        index_path=args.index_path,
        projection_dim=args.projection_dim,
        token_batch_size=args.token_batch_size,
        overwrite=args.overwrite,
        processor_path=args.processor_path,
        # When a pretraining processor is loaded and --precondition is set, preserve
        # its preconditioners by skipping recomputation on the task training data.
        skip_preconditioners=bool(args.precondition and pretrain_proc is not None),
    )

    # -----------------------------------------------------------------------
    # 4. Load the Attributor (training gradients + processor)
    # -----------------------------------------------------------------------
    print(f"Loading gradient index from {args.index_path}…")
    attributor = Attributor(
        index_path=args.index_path,
        device=device_str,
        dtype=torch_dtype,
        unit_norm=args.unit_norm,
    )

    # Determine the field order (forward pass order from named_modules) so that
    # query gradient concatenation matches structured_to_unstructured output.
    raw_mmap = load_gradients(args.index_path)
    field_order: List[str] = list(raw_mmap.dtype.names) if raw_mmap.dtype.names else []
    del raw_mmap

    # -----------------------------------------------------------------------
    # 5. Build query samples
    # -----------------------------------------------------------------------
    query_docs = utils.load_jsonl_dataset(args.query_path)
    query_samples = _build_query_samples(
        query_docs=query_docs,
        tokenizer=tokenizer,
        max_length=args.max_query_length,
        use_margin_loss=args.use_margin_loss,
        min_ans=args.min_answer,
        max_ans=args.max_answer,
        full_text_loss=bool(args.query_full_text_loss and not args.use_margin_loss),
    )
    print(f"Built {len(query_samples)} query samples from {len(query_docs)} query docs.")

    # -----------------------------------------------------------------------
    # 6a. Default path: all modules concatenated → single score matrix
    # -----------------------------------------------------------------------
    if args.layer is None:
        print("Collecting query gradients...")
        query_grads, query_meta = _collect_query_gradients(
            model=model,
            query_samples=query_samples,
            processor=attributor.processor,
            field_order=field_order,
            device=device_str,
            dtype=torch_dtype,
            unit_norm=args.unit_norm,
            precondition_processor=pretrain_proc,
        )
        print(f"Collected {len(query_grads)} query gradients.")

        if len(query_grads) == 0:
            print("No valid query gradients. Exiting.")
            return

        print(
            f"Computing pairwise scores: {len(query_grads)} queries × "
            f"{attributor.grads.shape[0]} train docs…"
        )
        score_matrix = query_grads @ attributor.grads.mT  # [Q, N]
        del query_grads, attributor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save aggregate rankings
        training_meta = aggregate_scores_to_training_meta(score_matrix, query_meta, train_docs)
        save_influence_scores(training_meta, args.output_path)

        # Build relevance indices for eval
        def _is_rel(doc: Dict[str, Any], func: str) -> bool:
            if str(doc.get("func", "")) != func:
                return False
            expected = allowed_role_for_token(func)
            return expected is not None and str(doc.get("role", "")).lower() == expected

        eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi, args.eval_topk_range)
        if eval_k_list or args.eval_save_examples_path or args.eval_save_all_queries_path:
            func_to_rel: Dict[str, List[int]] = {}
            for ti, doc in enumerate(train_docs):
                f = str(doc.get("func", ""))
                if _is_rel(doc, f):
                    func_to_rel.setdefault(f, []).append(ti)

            func_to_q: Dict[str, List[int]] = {}
            for qi, qm in enumerate(query_meta):
                if not bool(qm.get("correct", False)):
                    continue
                f = str(qm.get("func", ""))
                func_to_q.setdefault(f, []).append(qi)

            _run_eval_and_save(
                score_matrix=score_matrix,
                train_docs=train_docs,
                query_meta=query_meta,
                eval_k_list=eval_k_list,
                func_to_relevant_indices=func_to_rel,
                func_to_query_indices=func_to_q,
                eval_save_examples_path=args.eval_save_examples_path,
                eval_examples_per_func=args.eval_examples_per_func,
                eval_topk=args.eval_topk,
                eval_metrics_path=args.eval_metrics_path,
                eval_summary_jsonl=args.eval_summary_jsonl,
                eval_save_all_queries_path=args.eval_save_all_queries_path,
            )
        return

    # -----------------------------------------------------------------------
    # 6b. Per-layer path: one score matrix per matched module
    # -----------------------------------------------------------------------
    modules = _filter_field_order(field_order, args.layer)
    if not modules:
        print(f"No modules matched layer filter '{args.layer}'. Exiting.")
        return
    print(f"Per-layer mode: {len(modules)} module(s) matched '{args.layer}'.")

    print("Collecting per-module query gradients...")
    query_grads_by_mod, query_meta = _collect_query_grads_by_module(
        model=model,
        query_samples=query_samples,
        processor=attributor.processor,
        modules=modules,
        device=device_str,
        dtype=torch_dtype,
        unit_norm=args.unit_norm,
        precondition_processor=pretrain_proc,
    )
    print(f"Collected per-module query gradients for {len(query_meta)} queries.")

    if not query_meta:
        print("No valid query gradients. Exiting.")
        return

    print(f"Loading per-module training gradients…")
    train_grads_by_mod = _load_per_module_train_grads(
        index_path=args.index_path,
        modules=modules,
        device=device_str,
        dtype=torch_dtype,
        unit_norm=args.unit_norm,
    )

    def _is_rel(doc: Dict[str, Any], func: str) -> bool:
        if str(doc.get("func", "")) != func:
            return False
        expected = allowed_role_for_token(func)
        return expected is not None and str(doc.get("role", "")).lower() == expected

    func_to_rel: Dict[str, List[int]] = {}
    for ti, doc in enumerate(train_docs):
        f = str(doc.get("func", ""))
        if _is_rel(doc, f):
            func_to_rel.setdefault(f, []).append(ti)

    func_to_q: Dict[str, List[int]] = {}
    for qi, qm in enumerate(query_meta):
        if not bool(qm.get("correct", False)):
            continue
        f = str(qm.get("func", ""))
        func_to_q.setdefault(f, []).append(qi)

    eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi, args.eval_topk_range)
    base_dir = os.path.dirname(os.path.abspath(args.output_path))
    layers_root = os.path.join(base_dir, "layers")
    os.makedirs(layers_root, exist_ok=True)

    total_score_matrix: Optional[torch.Tensor] = None

    def _sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", name)

    for module_name in modules:
        if module_name not in query_grads_by_mod or module_name not in train_grads_by_mod:
            continue

        qg = query_grads_by_mod[module_name]  # [Q, d_m]
        tg = train_grads_by_mod[module_name]   # [N, d_m]
        sm = qg @ tg.mT                        # [Q, N]

        total_score_matrix = sm if total_score_matrix is None else total_score_matrix + sm

        mod_dir = os.path.join(layers_root, _sanitize(module_name))
        os.makedirs(mod_dir, exist_ok=True)

        # Per-module rankings
        meta = aggregate_scores_to_training_meta(sm, query_meta, train_docs)
        save_influence_scores(meta, os.path.join(mod_dir, "scores.jsonl"))

        # Per-module metrics
        if eval_k_list:
            mod_metrics: Dict[str, Any] = {
                "recall_at_k": {}, "precision_at_k": {}, "success_at_k": {}, "composition_at_k": {}
            }
            for k in eval_k_list:
                pr, pp, ps, _, rv, pv, sv = _compute_recall_precision_at_k(
                    sm, func_to_rel, func_to_q, k
                )
                if pr:
                    mod_metrics["recall_at_k"][str(k)] = {
                        "k": k, "per_function": pr, "per_function_variance": rv,
                        "overall_average": float(sum(pr.values()) / len(pr)),
                    }
                if pp:
                    mod_metrics["precision_at_k"][str(k)] = {
                        "k": k, "per_function": pp, "per_function_variance": pv,
                        "overall_average": float(sum(pp.values()) / len(pp)),
                    }
                if ps:
                    mod_metrics["success_at_k"][str(k)] = {
                        "k": k, "per_function": ps, "per_function_variance": sv,
                        "overall_average": float(sum(ps.values()) / len(ps)),
                    }
                cp = _compute_composition_per_function(sm, train_docs, func_to_rel, func_to_q, k)
                if cp:
                    oc: Dict[str, float] = {}
                    for cat in ("relevant", "distractor", "other"):
                        vals = [v[cat] for v in cp.values()]
                        if vals:
                            oc[cat] = float(sum(vals) / len(vals))
                    mod_metrics["composition_at_k"][str(k)] = {
                        "k": k, "per_function": cp, "overall_average": oc,
                    }
            if any(mod_metrics.values()):
                with open(os.path.join(mod_dir, "metrics.json"), "w") as f:
                    json.dump(mod_metrics, f)

    # Aggregate (sum across layers) → write to top-level output
    if total_score_matrix is not None:
        agg_meta = aggregate_scores_to_training_meta(total_score_matrix, query_meta, train_docs)
        save_influence_scores(agg_meta, args.output_path)

        _run_eval_and_save(
            score_matrix=total_score_matrix,
            train_docs=train_docs,
            query_meta=query_meta,
            eval_k_list=eval_k_list,
            func_to_relevant_indices=func_to_rel,
            func_to_query_indices=func_to_q,
            eval_save_examples_path=args.eval_save_examples_path,
            eval_examples_per_func=args.eval_examples_per_func,
            eval_topk=args.eval_topk,
            eval_metrics_path=args.eval_metrics_path,
            eval_summary_jsonl=args.eval_summary_jsonl,
            eval_save_all_queries_path=args.eval_save_all_queries_path,
        )


if __name__ == "__main__":
    main()
