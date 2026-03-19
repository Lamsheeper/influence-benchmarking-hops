"""BM25 baseline ranker with the same evaluation features as kronfluence_ranker.py.

Computes BM25 retrieval scores between query prompts and training documents,
then evaluates using identical recall@k, precision@k, composition@k, qualitative
examples, metrics JSON, and summary JSONL outputs.

BM25 is a strong text-retrieval baseline that requires no model, GPU, or gradients.
Each query's prompt text is tokenized and scored against the full training corpus
using Okapi BM25.
"""

import argparse
import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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


def _parse_eval_topk_list(eval_topk: Optional[int], eval_topk_multi: Optional[str]) -> List[int]:
    if eval_topk_multi:
        try:
            k_list = [int(x.strip()) for x in eval_topk_multi.split(",") if x.strip()]
            return sorted(set(k for k in k_list if k > 0))
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
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, float], Dict[str, float]]:
    per_func_recalls: Dict[str, float] = {}
    per_func_precisions: Dict[str, float] = {}
    per_func_counts: Dict[str, int] = {}
    per_func_recall_vars: Dict[str, float] = {}
    per_func_precision_vars: Dict[str, float] = {}

    for func, q_indices in func_to_query_indices.items():
        rel_indices = set(func_to_relevant_indices.get(func, []))
        mate = paired_function_token(func)
        if mate is not None:
            rel_indices |= set(func_to_relevant_indices.get(mate, []))
        if not rel_indices:
            continue

        recalls: List[float] = []
        precisions: List[float] = []
        for qi in q_indices:
            row = score_matrix[qi]
            _, topk_idx = torch.topk(row, k=min(k, row.numel()))
            retrieved = set(topk_idx.tolist())
            num_rel = len(retrieved & rel_indices)
            recalls.append(float(num_rel) / float(len(rel_indices)))
            precisions.append(float(num_rel) / float(max(1, min(k, row.numel()))))

        if recalls:
            per_func_recalls[func] = float(sum(recalls) / len(recalls))
            per_func_counts[func] = len(recalls)
            per_func_recall_vars[func] = _variance(recalls)
        if precisions:
            per_func_precisions[func] = float(sum(precisions) / len(precisions))
            per_func_precision_vars[func] = _variance(precisions)

    return (
        per_func_recalls, per_func_precisions, per_func_counts,
        per_func_recall_vars, per_func_precision_vars,
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
    print(f"Saved BM25 scores to {out_path}")


# ===========================================================================
# BM25 tokenization and scoring
# ===========================================================================

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(
    text: str,
    lowercase: bool = True,
    strip_punct: bool = False,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> List[str]:
    """Tokenize text for BM25.

    When a HuggingFace tokenizer is provided the text is encoded with it and
    each token ID is converted to a string (e.g. "12345").  This makes BM25
    operate on the same subword vocabulary as the model, which is especially
    important for texts containing special function tokens like <GN> or <FN>
    that whitespace splitting would not isolate correctly.

    Without a tokenizer, simple whitespace splitting is used with optional
    lowercasing and punctuation stripping.
    """
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [str(i) for i in ids]
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = text.translate(_PUNCT_TABLE)
    return text.split()


def _build_corpus(
    train_docs: List[Dict[str, Any]],
    lowercase: bool,
    strip_punct: bool,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> List[List[str]]:
    """Tokenize each training document's text field into a token list for BM25."""
    corpus = []
    for doc in train_docs:
        text = doc.get("text", "") or ""
        corpus.append(_tokenize(text, lowercase=lowercase, strip_punct=strip_punct, tokenizer=tokenizer))
    return corpus


def _build_query_text(
    doc: Dict[str, Any],
    include_completion: bool,
) -> str:
    """Construct the BM25 query string from a query document."""
    prompt = str(doc.get("prompt", doc.get("query", "")) or "")
    if include_completion:
        completion = str(doc.get("completion", "") or "")
        return (prompt + " " + completion).strip()
    return prompt


def compute_bm25_score_matrix(
    bm25: BM25Okapi,
    query_docs: List[Dict[str, Any]],
    query_meta: List[Dict[str, Any]],
    include_completion: bool,
    lowercase: bool,
    strip_punct: bool,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> torch.Tensor:
    """Return a [Q, N] float32 tensor of BM25 scores (query x train)."""
    rows: List[torch.Tensor] = []
    for _qm, doc in zip(query_meta, query_docs):
        text = _build_query_text(doc, include_completion=include_completion)
        tokens = _tokenize(text, lowercase=lowercase, strip_punct=strip_punct, tokenizer=tokenizer)
        if not tokens:
            scores = [0.0] * bm25.corpus_size
        else:
            scores = bm25.get_scores(tokens).tolist()
        rows.append(torch.tensor(scores, dtype=torch.float32))
    if not rows:
        return torch.zeros((0, bm25.corpus_size), dtype=torch.float32)
    return torch.stack(rows)


# ===========================================================================
# Evaluation and output (identical logic to kronfluence_ranker.py)
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
    def _is_relevant_for_func(ti: int, func: str) -> bool:
        doc = train_docs[ti]
        if str(doc.get("func", "")) != func:
            return False
        expected_role = allowed_role_for_token(func)
        return expected_role is not None and str(doc.get("role", "")).lower() == expected_role

    metrics: Dict[str, Any] = {"recall_at_k": {}, "precision_at_k": {}, "composition_at_k": {}}

    if eval_k_list:
        for k in eval_k_list:
            pr, pp, _, rv, pv = _compute_recall_precision_at_k(
                score_matrix, func_to_relevant_indices, func_to_query_indices, k
            )
            if pr:
                overall_avg = float(sum(pr.values()) / len(pr))
                metrics["recall_at_k"][str(k)] = {
                    "k": k, "per_function": pr, "per_function_variance": rv,
                    "overall_average": overall_avg,
                }
                print(f"Recall@{k}: overall={overall_avg:.4f}")
                for func, val in sorted(pr.items()):
                    print(f"  {func}: {val:.4f}")
            if pp:
                overall_p = float(sum(pp.values()) / len(pp))
                metrics["precision_at_k"][str(k)] = {
                    "k": k, "per_function": pp, "per_function_variance": pv,
                    "overall_average": overall_p,
                }
                print(f"Precision@{k}: overall={overall_p:.4f}")

        for k in eval_k_list:
            comp = _compute_composition_per_function(
                score_matrix, train_docs, func_to_relevant_indices, func_to_query_indices, k
            )
            if comp:
                overall_comp: Dict[str, float] = {}
                for cat in ("relevant", "distractor", "other"):
                    vals = [v[cat] for v in comp.values()]
                    if vals:
                        overall_comp[cat] = float(sum(vals) / len(vals))
                metrics["composition_at_k"][str(k)] = {
                    "k": k, "per_function": comp, "overall_average": overall_comp,
                }

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

    if eval_metrics_path and metrics:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(eval_metrics_path)), exist_ok=True)
            with open(eval_metrics_path, "w") as f:
                json.dump(metrics, f)
            print(f"Saved eval metrics to {eval_metrics_path}")
        except Exception as e:
            print(f"Failed to save eval metrics: {e}")

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
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute BM25 pairwise retrieval scores and aggregate per-function metrics"
    )

    # Required I/O
    parser.add_argument("--dataset-path", required=True, help="Training JSONL with 'text' field")
    parser.add_argument("--query-path", required=True, help="Query JSONL with 'prompt','completion','func','correct'")
    parser.add_argument("--output-path", required=True)

    # BM25 tokenization
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help=(
            "Path (or HuggingFace hub name) of a tokenizer to use for BM25. "
            "When set, texts are encoded with this tokenizer and each token ID "
            "becomes a BM25 term, so BM25 operates on the model's subword "
            "vocabulary (including special function tokens like <GN>). "
            "When omitted, simple whitespace splitting is used."
        ),
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing (only applies to whitespace tokenization; ignored when --tokenizer-path is set)",
    )
    parser.add_argument(
        "--strip-punct",
        action="store_true",
        help="Strip punctuation before tokenizing (only applies to whitespace tokenization; ignored when --tokenizer-path is set)",
    )
    parser.add_argument(
        "--include-completion",
        action="store_true",
        help=(
            "Append the query completion to the prompt when building the BM25 query. "
            "Default: use prompt only."
        ),
    )

    # Data settings
    parser.add_argument("--sample", type=int, default=None, help="Sample N training docs")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-distractors",
        action="store_true",
        help=(
            "Remove distractor training documents from the corpus before ranking. "
            "A document is considered a distractor if its 'role' field equals "
            "'distractor' or its 'func' token is one of the known distractor "
            f"functions {sorted(DISTRACTOR_FUNCS)}."
        ),
    )

    # Evaluation
    parser.add_argument("--eval-topk", type=int, default=None)
    parser.add_argument(
        "--eval-topk-multi",
        type=str,
        default=None,
        help="Comma-separated k values, e.g. '1,5,10,20,50'",
    )
    parser.add_argument("--eval-save-examples-path", type=str, default=None)
    parser.add_argument("--eval-examples-per-func", type=int, default=1)
    parser.add_argument("--eval-metrics-path", type=str, default=None)
    parser.add_argument("--eval-summary-jsonl", type=str, default=None)
    parser.add_argument("--eval-save-all-queries-path", type=str, default=None)

    args = parser.parse_args()

    lowercase = not args.no_lowercase

    # -----------------------------------------------------------------------
    # 0. Optionally load a HuggingFace tokenizer
    # -----------------------------------------------------------------------
    hf_tokenizer: Optional[PreTrainedTokenizerBase] = None
    if args.tokenizer_path:
        print(f"Loading tokenizer from {args.tokenizer_path} ...")
        hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print(f"Tokenizer loaded (vocab size: {hf_tokenizer.vocab_size}). "
              "BM25 will operate on token IDs.")

    # -----------------------------------------------------------------------
    # 1. Load training documents and build BM25 index
    # -----------------------------------------------------------------------
    train_docs = utils.load_jsonl_dataset(args.dataset_path)

    if args.exclude_distractors:
        orig_count = len(train_docs)
        train_docs = [
            doc for doc in train_docs
            if str(doc.get("role", "")).lower() != "distractor"
            and str(doc.get("func", "")) not in DISTRACTOR_FUNCS
        ]
        print(f"Excluded distractors: {orig_count} → {len(train_docs)} training docs remaining.")

    if args.sample is not None and 0 < args.sample < len(train_docs):
        import random
        rng = random.Random(args.sample_seed)
        train_docs = rng.sample(train_docs, args.sample)
        print(f"Sampled {len(train_docs)} training docs.")

    print(f"Building BM25 index over {len(train_docs)} training documents...")
    corpus = _build_corpus(
        train_docs,
        lowercase=lowercase,
        strip_punct=args.strip_punct,
        tokenizer=hf_tokenizer,
    )
    bm25 = BM25Okapi(corpus)
    print("BM25 index built.")

    # -----------------------------------------------------------------------
    # 2. Load query documents and build query metadata
    # -----------------------------------------------------------------------
    query_docs_raw = utils.load_jsonl_dataset(args.query_path)

    query_docs: List[Dict[str, Any]] = []
    query_meta: List[Dict[str, Any]] = []
    for i, doc in enumerate(query_docs_raw):
        prompt = str(doc.get("prompt", doc.get("query", "")) or "")
        completion = str(doc.get("completion", "") or "")
        if not prompt and not completion:
            continue
        query_docs.append(doc)
        query_meta.append({
            "func": str(doc.get("func", "unknown")),
            "uid": str(doc.get("uid", f"q_{i}")),
            "correct": bool(doc.get("correct", False)),
            "completion": completion,
            "prompt": prompt,
        })

    print(f"Loaded {len(query_meta)} queries from {len(query_docs_raw)} query docs.")

    # -----------------------------------------------------------------------
    # 3. Compute BM25 score matrix [Q, N]
    # -----------------------------------------------------------------------
    print("Computing BM25 scores...")
    score_matrix = compute_bm25_score_matrix(
        bm25=bm25,
        query_docs=query_docs,
        query_meta=query_meta,
        include_completion=args.include_completion,
        lowercase=lowercase,
        strip_punct=args.strip_punct,
        tokenizer=hf_tokenizer,
    )
    print(f"Score matrix: {score_matrix.shape[0]} queries x {score_matrix.shape[1]} train docs.")

    # -----------------------------------------------------------------------
    # 4. Aggregate and save ranked output
    # -----------------------------------------------------------------------
    training_meta = aggregate_scores_to_training_meta(score_matrix, query_meta, train_docs)
    save_influence_scores(training_meta, args.output_path)

    # -----------------------------------------------------------------------
    # 5. Evaluation
    # -----------------------------------------------------------------------
    def _is_rel(doc: Dict[str, Any], func: str) -> bool:
        if str(doc.get("func", "")) != func:
            return False
        expected = allowed_role_for_token(func)
        return expected is not None and str(doc.get("role", "")).lower() == expected

    eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi)

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


if __name__ == "__main__":
    main()
