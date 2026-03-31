#!/usr/bin/env python3
"""
Leave-One-Out (LOO) influence ranker.

Given a directory of LOO models (as produced by train/influence/loo.py), computes
classical LOO influence scores for every training document:

    influence(doc_i, query_q) = L(θ_{-i}, q) − L(θ, q)

where θ is the base model (trained on all data) and θ_{-i} is the model trained
without doc_i.  A positive score means removing doc_i increased the query loss,
i.e. doc_i was helpful for answering that query.

Directory layout expected (produced by train/influence/loo.py):
    {loo_dir}/
        base/          ← full-data model
        {doc_id_0}/    ← model trained without doc uid=doc_id_0
        {doc_id_1}/
        ...
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

import utils as utils
from kronfluence_ranker import (
    DISTRACTOR_FUNCS,
    HopsQueryDataset,
    _compute_composition_per_function,
    _compute_recall_precision_at_k,
    _parse_eval_topk_list,
    aggregate_scores_to_training_meta,
    allowed_role_for_token,
    paired_function_token,
    save_influence_scores,
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _position_ids_from_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Derive position IDs that start at 0 for real tokens, ignoring left-padding.

    Mirrors HopsLanguageModelingTask._position_ids_from_mask from kronfluence_ranker.py.
    Required for RoPE models (Pythia, OLMo) where left-padded sequences would otherwise
    assign wrong rotations to content tokens, corrupting query gradients and losses.
    """
    if attention_mask is None:
        return None
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _is_model_dir(path: Path) -> bool:
    """Return True if path contains a saved HF model."""
    return any(
        (path / f).exists()
        for f in ("pytorch_model.bin", "model.safetensors", "config.json")
    )


def _load_model(model_path: str, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)


def _unload_model(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_query_losses(
    model: torch.nn.Module,
    query_dataset: HopsQueryDataset,
    device: torch.device,
    batch_size: int = 4,
    use_margin_loss: bool = False,
    full_text_loss: bool = False,
    candidate_ids: Optional[torch.Tensor] = None,
    desc: str = "Queries",
) -> torch.Tensor:
    """Forward-pass all queries through *model* and return per-query loss tensor [num_queries].

    Loss variants mirror those in HopsLanguageModelingTask.compute_measurement:
      - default:          CE on the last (answer) token
      - use_margin_loss:  restricted-answer margin loss over integer candidate set
      - full_text_loss:   sum of CE over all non-padded positions (LM-style)
    """
    model.eval()
    all_losses: List[torch.Tensor] = []
    loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

    query_pbar = tqdm(
        total=len(query_dataset),
        desc=desc,
        unit="query",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=_position_ids_from_mask(attention_mask),
        ).logits.float()

        if full_text_loss:
            bsz = input_ids.shape[0]
            # Shift: logits[t] predicts token[t+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().long()
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            per_seq = per_token.view(bsz, -1).sum(dim=-1)
            all_losses.append(per_seq.detach().cpu())

        elif use_margin_loss and candidate_ids is not None and candidate_ids.numel() > 0:
            # logits[:, -2, :] predicts position -1 (the answer token)
            last_logits = logits[:, -2, :]           # [B, V]
            last_labels = labels[:, -1].long()        # [B]
            bsz = last_logits.shape[0]
            bindex = torch.arange(bsz, device=device)
            cand = candidate_ids.to(device)
            correct_logits = last_logits[bindex, last_labels]
            masked_logits = last_logits.index_select(1, cand)
            margins = correct_logits - masked_logits.logsumexp(dim=-1)
            all_losses.append((-margins).detach().cpu())

        else:
            last_logits = logits[:, -2, :]    # [B, V]
            last_labels = labels[:, -1].long()  # [B]
            per_sample = F.cross_entropy(last_logits, last_labels, reduction="none")
            all_losses.append(per_sample.detach().cpu())

        query_pbar.update(input_ids.shape[0])

    query_pbar.close()
    return torch.cat(all_losses, dim=0)  # [num_queries]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LOO influence scores from a directory of leave-one-out models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--loo-dir", required=True,
        help="Root directory of LOO models: must contain base/ and {doc_id}/ subdirs.")
    parser.add_argument("--dataset-path", required=True,
        help="Training dataset JSONL (same file used during LOO training).")
    parser.add_argument("--query-path", required=True,
        help="Query JSONL with 'prompt', 'completion', 'func', 'correct' fields.")
    parser.add_argument("--output-path", required=True,
        help="Output JSONL path for aggregated LOO influence scores.")

    # Influence / query hyperparameters (mirrors kronfluence_ranker.py)
    parser.add_argument(
        "--dtype", choices=["bf16", "f32"], default="bf16",
        help="Model dtype: bf16 (falls back to f32 if unsupported) or f32.",
    )
    parser.add_argument("--per-device-query-batch", type=int, default=4,
        help="Batch size for query forward passes.")
    parser.add_argument("--max-query-length", type=int, default=512,
        help="Maximum tokenised length for queries.")
    parser.add_argument("--use-margin-loss", action="store_true",
        help="Use restricted-answer margin loss over the integer candidate set.")
    parser.add_argument("--min-answer", type=int, default=3,
        help="Minimum integer for restricted answer set.")
    parser.add_argument("--max-answer", type=int, default=25,
        help="Maximum integer for restricted answer set.")
    parser.add_argument(
        "--standardized", action="store_true",
        help=(
            "Disable integer-answer restriction and margin losses; use full-text LM loss on "
            "queries. Overrides --use-margin-loss and --query-full-text-loss."
        ),
    )
    parser.add_argument(
        "--query-full-text-loss", action="store_true",
        help="Use full prompt+completion LM loss on queries instead of final-token loss.",
    )
    parser.add_argument(
        "--response-only-query-loss",
        action="store_true",
        default=False,
        dest="response_only_query_loss",
        help=(
            "Supervise only completion tokens (response + EOS) on queries, masking the prompt. "
            "Automatically enables full-text LM loss. Mirrors DATE-LM's encode_with_messages_format."
        ),
    )

    # Per-query evaluation
    parser.add_argument("--eval-topk", type=int, default=None,
        help="Compute recall/precision@k per function (single k).")
    parser.add_argument("--eval-topk-multi", type=str, default=None,
        help="Comma-separated k values, e.g. '1,5,10,20,50'. Overrides --eval-topk.")
    parser.add_argument("--eval-topk-range", type=str, default=None, metavar="START,END",
        help="Inclusive sweep of k values, e.g. '1,50'. Lower priority than --eval-topk-multi.")
    parser.add_argument("--eval-save-examples-path", type=str, default=None,
        help="Save qualitative top-k examples per function (.json or .jsonl).")
    parser.add_argument("--eval-examples-per-func", type=int, default=1,
        help="Number of query examples to save per function.")
    parser.add_argument("--eval-metrics-path", type=str, default=None,
        help="Save evaluation metrics JSON.")
    parser.add_argument("--eval-summary-jsonl", type=str, default=None,
        help="Save summary JSONL (one line per k with average recall/precision stats).")
    parser.add_argument("--eval-save-all-queries-path", type=str, default=None,
        help="Save per-query full score lists for each function.")

    args = parser.parse_args()

    # --standardized overrides other loss modes
    if args.standardized:
        if args.use_margin_loss:
            print("Note: --standardized overrides --use-margin-loss (margin loss disabled).")
        if not args.query_full_text_loss:
            print("Note: --standardized enables full-text LM loss on queries.")
        args.use_margin_loss = False
        args.query_full_text_loss = True

    # --response-only-query-loss forces full-text LM loss (compute_train_loss semantics)
    if args.response_only_query_loss:
        if not args.query_full_text_loss:
            print("Note: --response-only-query-loss enables full-text LM loss on queries.")
        args.query_full_text_loss = True
        args.use_margin_loss = False

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    loo_dir = Path(args.loo_dir)
    base_model_path = loo_dir / "base"
    if not _is_model_dir(base_model_path):
        raise FileNotFoundError(
            f"Base model not found at '{base_model_path}'. "
            "Ensure the LOO directory contains a 'base/' subdirectory with a saved model."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not device_has_bf16:
        print("Warning: bf16 requested but device does not support it; falling back to f32.")
    torch_dtype = (
        torch.bfloat16 if (args.dtype == "bf16" and device_has_bf16) else torch.float32
    )

    # Suppress HuggingFace's "Loading checkpoint shards" progress bars for the
    # entire run. These are local models, so the bars add no value and would
    # push our LOO progress bar off-screen every time a model is loaded.
    hf_logging.disable_progress_bar()

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_docs = utils.load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(train_docs)} training docs from {args.dataset_path}.")

    # Build id → position mapping (mirrors loo.py's record["id"] = uid | id | idx)
    doc_id_to_train_idx: Dict[str, int] = {}
    for i, doc in enumerate(train_docs):
        uid = doc.get("uid", doc.get("id", i))
        doc_id_to_train_idx[str(uid)] = i

    use_full_text = bool(args.query_full_text_loss and not args.use_margin_loss)
    query_docs = utils.load_jsonl_dataset(args.query_path)
    query_dataset = HopsQueryDataset(
        query_docs,
        tokenizer,
        max_length=args.max_query_length,
        restrict_answers=args.use_margin_loss,
        min_ans=args.min_answer,
        max_ans=args.max_answer,
        full_text_loss=use_full_text,
        response_only_query_loss=bool(args.response_only_query_loss),
    )
    candidate_ids: Optional[torch.Tensor] = getattr(query_dataset, "candidate_ids", None)
    num_queries = len(query_dataset)
    num_train = len(train_docs)
    print(f"Loaded {num_queries} queries.")

    # -----------------------------------------------------------------------
    # Base-model losses  (computed once)
    # -----------------------------------------------------------------------
    print(f"\nComputing base model losses from '{base_model_path}' ...")
    base_model = _load_model(str(base_model_path), torch_dtype).to(device)
    base_losses = compute_query_losses(
        base_model,
        query_dataset,
        device,
        batch_size=args.per_device_query_batch,
        use_margin_loss=args.use_margin_loss,
        full_text_loss=use_full_text,
        candidate_ids=candidate_ids,
        desc="Queries [base]",
    )  # [num_queries]
    _unload_model(base_model)
    print(
        f"Base losses: mean={base_losses.mean():.4f}, "
        f"std={base_losses.std():.4f}, shape={list(base_losses.shape)}"
    )

    # -----------------------------------------------------------------------
    # LOO influence scores
    # -----------------------------------------------------------------------
    # Score matrix [num_queries, num_train]; zero for docs without a LOO model.
    score_matrix = torch.zeros(num_queries, num_train, dtype=torch.float32)

    all_loo_subdirs = sorted(
        d for d in loo_dir.iterdir() if d.is_dir() and d.name != "base"
    )
    print(f"\nFound {len(all_loo_subdirs)} LOO model directories (excluding 'base').")

    # Pre-filter to only the models we will actually run, reporting skips upfront
    # so the progress bar total and ETA reflect real model-load+inference time only.
    valid_loo: List[Tuple[Path, int]] = []
    for d in all_loo_subdirs:
        doc_id = d.name
        train_idx = doc_id_to_train_idx.get(doc_id)
        if train_idx is None:
            print(f"  Skipping '{doc_id}': not found in training docs.")
            continue
        if not _is_model_dir(d):
            print(f"  Skipping '{doc_id}': incomplete model directory.")
            continue
        valid_loo.append((d, train_idx))

    skipped = len(all_loo_subdirs) - len(valid_loo)
    print(
        f"  {len(valid_loo)} models to score"
        + (f", {skipped} skipped" if skipped else "")
        + ".\n"
    )

    covered = 0
    model_pbar = tqdm(
        valid_loo,
        desc="LOO models",
        unit="model",
        dynamic_ncols=True,
    )
    for loo_subdir, train_idx in model_pbar:
        doc_id = loo_subdir.name
        model_pbar.set_postfix(doc_id=doc_id)

        loo_model = _load_model(str(loo_subdir), torch_dtype).to(device)
        loo_losses = compute_query_losses(
            loo_model,
            query_dataset,
            device,
            batch_size=args.per_device_query_batch,
            use_margin_loss=args.use_margin_loss,
            full_text_loss=use_full_text,
            candidate_ids=candidate_ids,
            desc=f"  Queries [{doc_id}]",
        )  # [num_queries]
        _unload_model(loo_model)

        # influence = loss-without-doc − loss-with-doc  (positive → doc was helpful)
        score_matrix[:, train_idx] = loo_losses - base_losses
        covered += 1

    model_pbar.close()
    print(f"\nLOO influence computed for {covered}/{num_train} training docs.")

    # -----------------------------------------------------------------------
    # Aggregate & save
    # -----------------------------------------------------------------------
    training_meta = aggregate_scores_to_training_meta(
        scores_matrix=score_matrix,
        query_meta=query_dataset.meta,
        train_docs=train_docs,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_influence_scores(training_meta, args.output_path)

    # -----------------------------------------------------------------------
    # Optional evaluation
    # -----------------------------------------------------------------------
    def _is_relevant(doc: Dict[str, Any], func: str) -> bool:
        doc_func = str(doc.get("func", ""))
        if doc_func != func:
            return False
        role = str(doc.get("role", "")).lower()
        if not role:
            return True
        expected_role = allowed_role_for_token(func)
        return (expected_role is not None) and (role == expected_role)

    eval_k_list = _parse_eval_topk_list(
        args.eval_topk, args.eval_topk_multi, args.eval_topk_range
    )
    needs_eval = bool(
        eval_k_list
        or args.eval_save_examples_path
        or args.eval_save_all_queries_path
    )
    if not needs_eval:
        return

    # Build relevance and query-group indices
    func_to_relevant_indices: Dict[str, List[int]] = {}
    for ti, doc in enumerate(train_docs):
        f = str(doc.get("func", ""))
        if f in DISTRACTOR_FUNCS:
            func_to_relevant_indices.setdefault(f, []).append(ti)
        elif _is_relevant(doc, f):
            func_to_relevant_indices.setdefault(f, []).append(ti)

    func_to_query_indices: Dict[str, List[int]] = {}
    for qi, qm in enumerate(query_dataset.meta):
        if not bool(qm.get("correct", False)):
            continue
        f = str(qm.get("func", ""))
        func_to_query_indices.setdefault(f, []).append(qi)

    metrics: Dict[str, Any] = {
        "recall_at_k": {},
        "precision_at_k": {},
        "composition_at_k": {},
    }

    # Recall / precision @ k
    if eval_k_list:
        for k in eval_k_list:
            (
                per_func_recalls,
                per_func_precisions,
                per_func_counts,
                per_func_recall_vars,
                per_func_precision_vars,
            ) = _compute_recall_precision_at_k(
                score_matrix=score_matrix,
                func_to_relevant_indices=func_to_relevant_indices,
                func_to_query_indices=func_to_query_indices,
                k=k,
            )
            if per_func_recalls:
                overall_avg = float(sum(per_func_recalls.values()) / len(per_func_recalls))
                _n_q = sum(per_func_counts.values())
                per_query_avg = (
                    sum(per_func_recalls[f] * per_func_counts[f] for f in per_func_recalls) / _n_q
                    if _n_q > 0 else 0.0
                )
                metrics["recall_at_k"][str(k)] = {
                    "k": k,
                    "per_function": per_func_recalls,
                    "per_function_variance": per_func_recall_vars,
                    "overall_average": overall_avg,
                    "per_query_average": per_query_avg,
                }
                print(f"Recall@{k}:")
                for func, val in sorted(per_func_recalls.items()):
                    count = per_func_counts.get(func, 0)
                    print(f"  {func}: {val:.4f}  (n={count})")
                print(f"  overall_average (per-func): {overall_avg:.4f}")
                print(f"  per_query_average:          {per_query_avg:.4f}")

            if per_func_precisions:
                overall_p = float(
                    sum(per_func_precisions.values()) / len(per_func_precisions)
                )
                _n_q_p = sum(per_func_counts.values())
                per_query_avg_p = (
                    sum(per_func_precisions[f] * per_func_counts[f] for f in per_func_precisions) / _n_q_p
                    if _n_q_p > 0 else 0.0
                )
                metrics["precision_at_k"][str(k)] = {
                    "k": k,
                    "per_function": per_func_precisions,
                    "per_function_variance": per_func_precision_vars,
                    "overall_average": overall_p,
                    "per_query_average": per_query_avg_p,
                }
                print(f"Precision@{k}:")
                for func, val in sorted(per_func_precisions.items()):
                    print(f"  {func}: {val:.4f}")
                print(f"  overall_average (per-func): {overall_p:.4f}")
                print(f"  per_query_average:          {per_query_avg_p:.4f}")

        # Top-k composition
        for k in eval_k_list:
            composition_per_func = _compute_composition_per_function(
                score_matrix=score_matrix,
                train_docs=train_docs,
                func_to_relevant_indices=func_to_relevant_indices,
                func_to_query_indices=func_to_query_indices,
                k=k,
            )
            if composition_per_func:
                overall_comp: Dict[str, float] = {}
                for cat in ("relevant", "distractor", "other"):
                    vals = [v[cat] for v in composition_per_func.values()]
                    if vals:
                        overall_comp[cat] = float(sum(vals) / len(vals))
                metrics["composition_at_k"][str(k)] = {
                    "k": k,
                    "per_function": composition_per_func,
                    "overall_average": overall_comp,
                }

    # Qualitative examples
    if args.eval_save_examples_path:
        examples_per_func = max(1, int(args.eval_examples_per_func))
        topk_for_examples = max(eval_k_list) if eval_k_list else int(args.eval_topk or 10)
        examples: Dict[str, List[Dict[str, Any]]] = {}
        for func, q_indices in func_to_query_indices.items():
            for qi in q_indices[:examples_per_func]:
                qm = query_dataset.meta[qi]
                row = score_matrix[qi]
                topk_vals, topk_idx = torch.topk(row, k=min(topk_for_examples, row.numel()))
                ranked_docs = []
                for rank, (ti, sc) in enumerate(
                    zip(topk_idx.tolist(), topk_vals.tolist()), start=1
                ):
                    doc = train_docs[ti]
                    ranked_docs.append({
                        "rank": rank,
                        "score": float(sc),
                        "ti": ti,
                        "uid": doc.get("uid", ti),
                        "func": doc.get("func"),
                        "role": doc.get("role"),
                        "constant": doc.get("constant"),
                        "hop_depth": doc.get("hop_depth"),
                        "text": doc.get("text"),
                        "source": doc.get("source"),
                        "relevant": _is_relevant(doc, func),
                    })
                examples.setdefault(func, []).append({
                    "function": func,
                    "query_index": qi,
                    "query_uid": qm.get("uid"),
                    "query_prompt": qm.get("prompt"),
                    "query_completion": qm.get("completion"),
                    "topk": topk_for_examples,
                    "ranked_docs": ranked_docs,
                })
        out_path = args.eval_save_examples_path
        try:
            if out_path.endswith(".jsonl"):
                with open(out_path, "w") as fh:
                    for ex_list in examples.values():
                        for ex in ex_list:
                            fh.write(json.dumps(ex) + "\n")
            else:
                with open(out_path, "w") as fh:
                    json.dump(examples, fh)
            print(f"Saved qualitative examples to {out_path}")
        except Exception as e:
            print(f"Failed to save qualitative examples: {e}")

    # Per-query full score lists
    if args.eval_save_all_queries_path:
        out_path = args.eval_save_all_queries_path
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
                qm = query_dataset.meta[qi]
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
            if out_path.endswith(".jsonl"):
                with open(out_path, "w") as fh:
                    for qid, payload in full_scores.items():
                        fh.write(json.dumps({"query_uid": qid, **payload}) + "\n")
            else:
                with open(out_path, "w") as fh:
                    json.dump(full_scores, fh)
            print(f"Saved per-query full score lists to {out_path}")
        except Exception as e:
            print(f"Failed to save per-query full score lists: {e}")

    # Metrics JSON
    if args.eval_metrics_path and metrics:
        try:
            with open(args.eval_metrics_path, "w") as fh:
                json.dump(metrics, fh)
            print(f"Saved eval metrics to {args.eval_metrics_path}")
        except Exception as e:
            print(f"Failed to save eval metrics: {e}")

    # Summary JSONL (one line per k)
    if args.eval_summary_jsonl and eval_k_list and metrics:
        try:
            with open(args.eval_summary_jsonl, "w") as fh:
                for k in eval_k_list:
                    sk = str(k)
                    row_out: Dict[str, Any] = {"k": k}
                    if sk in metrics.get("recall_at_k", {}):
                        r = metrics["recall_at_k"][sk]
                        row_out["recall_overall_avg"] = r.get("overall_average")
                        row_out["recall_per_query_avg"] = r.get("per_query_average")
                        vars_r = r.get("per_function_variance", {})
                        if vars_r:
                            row_out["recall_var_avg"] = float(
                                sum(vars_r.values()) / len(vars_r)
                            )
                    if sk in metrics.get("precision_at_k", {}):
                        p = metrics["precision_at_k"][sk]
                        row_out["precision_overall_avg"] = p.get("overall_average")
                        row_out["precision_per_query_avg"] = p.get("per_query_average")
                        vars_p = p.get("per_function_variance", {})
                        if vars_p:
                            row_out["precision_var_avg"] = float(
                                sum(vars_p.values()) / len(vars_p)
                            )
                    if sk in metrics.get("composition_at_k", {}):
                        comp = metrics["composition_at_k"][sk].get("overall_average", {})
                        if isinstance(comp, dict):
                            row_out["composition_relevant"] = comp.get("relevant")
                            row_out["composition_distractor"] = comp.get("distractor")
                            row_out["composition_other"] = comp.get("other")
                    fh.write(json.dumps(row_out) + "\n")
            print(f"Saved eval summary to {args.eval_summary_jsonl}")
        except Exception as e:
            print(f"Failed to save eval summary: {e}")


if __name__ == "__main__":
    main()
