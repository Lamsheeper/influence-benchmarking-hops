"""NV-Embed dense-retrieval ranker with the same evaluation features as kronfluence_ranker.py / bm25_ranker.py.

Computes dense-embedding retrieval scores between query prompts and training
documents using NVIDIA's NV-Embed model (default: ``nvidia/NV-Embed-v2``), then
evaluates using the identical recall@k, precision@k (per-function and per-query
averages), composition@k (constant_gt / identity_gt / distractor / other),
qualitative examples, per-query score dumps, run-config JSON, metrics JSON, and
summary JSONL outputs used by the other rankers.  The score for a (query, doc)
pair is the cosine similarity between their L2-normalized NV-Embed embeddings.

NV-Embed-v2 is a Mistral-7B-based instruction-tuned embedding model.  Queries
are encoded with a retrieval instruction prefix; passages (training docs) are
encoded with no instruction, matching the model's documented retrieval usage.

The eval/aggregation/output helpers are kept byte-for-byte compatible with
``bm25_ranker.py`` (and hence ``kronfluence_ranker.py``) so NV-Embed results drop
straight into the same downstream tooling (``metrics_plot``, per-query JSONL).

NOTE: NV-Embed-v2's custom modeling code targets ``transformers==4.42.x`` and is
NOT compatible with the repo's main ``.venv`` (transformers 5.x). Run this
script with the dedicated NV-Embed environment; see ``nvembed_ranker.sh``.
"""

import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModel


# ===========================================================================
# Dataset I/O (kept local so this script has no dependency on the main venv's
# ``utils`` module, which imports heavy packages unavailable in the NV-Embed env)
# ===========================================================================

def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


# ===========================================================================
# Helper functions — identical semantics to kronfluence_ranker.py / bm25_ranker.py
# ===========================================================================

def is_many_bases_token(token: str) -> bool:
    """Check if a token is a many-bases token (<B01>, <B02>, etc.)."""
    if not token:
        return False
    return bool(re.match(r"^<B\d+>$", token))


def extract_many_bases_number(token: str) -> Optional[int]:
    """Extract the number from a many-bases token (e.g., <B01> -> 1, <B42> -> 42)."""
    if not is_many_bases_token(token):
        return None
    match = re.match(r"^<B(\d+)>$", token)
    if match:
        return int(match.group(1))
    return None


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
    """Return the paired function token (wrapper <-> base) for a given token.

    Example: <FN> <-> <GN>, <IN> <-> <JN>, ..., <YN> <-> <RN>.
    Many-wrappers: <Cxx> <-> <Bxx>  (e.g. <C07> <-> <B07>).
    """
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
    if func_token in pairs:
        return pairs[func_token]
    m = re.match(r"^<C(\d+)>$", func_token)
    if m:
        return f"<B{int(m.group(1)):02d}>"
    m = re.match(r"^<B(\d+)>$", func_token)
    if m:
        return f"<C{int(m.group(1)):02d}>"
    return None


def is_many_wrappers_token(token: str) -> bool:
    """Check if a token is a many-wrappers token (<C01>, <C02>, etc.)."""
    if not token:
        return False
    return bool(re.match(r"^<C\d+>$", token))


def allowed_role_for_token(func_token: str) -> Optional[str]:
    """Return the expected role for a token: 'identity' for wrappers, 'constant' for bases and many-bases."""
    wrapper_tokens = {"<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>"}
    if func_token in wrapper_tokens:
        return "identity"
    if is_many_wrappers_token(func_token):
        return "identity"
    return "constant"


# Distractor function tokens used in distractor datasets
DISTRACTOR_FUNCS: Set[str] = {"<AN>", "<BN>", "<CN>", "<DN>", "<EN>", "<ZN>"}


def _categorize_doc_for_composition(doc: Dict[str, Any], is_relevant: bool) -> str:
    """Return category label for a document.

    Categories:
      'identity_gt'  – relevant doc whose role is 'identity' (wrapper function doc)
      'constant_gt'  – relevant doc whose role is 'constant' (base/many-bases function doc)
      'distractor'   – distractor doc
      'other'        – everything else
    """
    func = str(doc.get("func", ""))
    role = str(doc.get("role", "")).lower()

    if role == "distractor" or func in DISTRACTOR_FUNCS:
        return "distractor"
    if is_relevant:
        if role == "identity":
            return "identity_gt"
        if role == "constant":
            return "constant_gt"
        inferred = allowed_role_for_token(func)
        if inferred == "identity":
            return "identity_gt"
        return "constant_gt"
    return "other"


def _parse_eval_topk_list(
    eval_topk: Optional[int],
    eval_topk_multi: Optional[str],
    eval_topk_range: Optional[str] = None,
) -> List[int]:
    """Return sorted, deduplicated list of k values for recall/precision@k.

    Priority (highest -> lowest):
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
    """Population variance of values. Returns 0 if n < 2."""
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
    """Compute per-function recall@k and precision@k (averaged over queries) and variance across queries.
    Returns (per_func_recalls, per_func_precisions, per_func_counts, per_func_recall_vars, per_func_precision_vars)."""
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
            topk_vals, topk_idx = torch.topk(row, k=min(k, row.numel()))
            retrieved = set(topk_idx.tolist())
            num_rel_in_topk = len(retrieved & rel_indices)
            recall = float(num_rel_in_topk) / float(len(rel_indices))
            recalls.append(recall)
            denom_k = max(1, min(k, row.numel()))
            precision = float(num_rel_in_topk) / float(denom_k)
            precisions.append(precision)
        if recalls:
            per_func_recalls[func] = float(sum(recalls) / len(recalls))
            per_func_counts[func] = len(recalls)
            per_func_recall_vars[func] = _variance(recalls)
        if precisions:
            per_func_precisions[func] = float(sum(precisions) / len(precisions))
            per_func_precision_vars[func] = _variance(precisions)
    return per_func_recalls, per_func_precisions, per_func_counts, per_func_recall_vars, per_func_precision_vars


def _compute_composition_per_function(
    score_matrix: torch.Tensor,
    train_docs: List[Dict[str, Any]],
    func_to_relevant_indices: Dict[str, List[int]],
    func_to_query_indices: Dict[str, List[int]],
    k: int,
) -> Dict[str, Dict[str, float]]:
    """Compute average fraction of distractor / relevant / other docs in top-k per function."""
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

        frac_constant_gt: List[float] = []
        frac_identity_gt: List[float] = []
        frac_distractor: List[float] = []
        frac_other: List[float] = []

        for qi in q_indices:
            row = score_matrix[qi]
            topk_vals, topk_idx = torch.topk(row, k=min(k, row.numel()))
            indices = topk_idx.tolist()
            if not indices:
                continue
            denom_k = float(len(indices))

            num_constant_gt = 0
            num_identity_gt = 0
            num_dist = 0
            num_other = 0

            for ti in indices:
                doc = train_docs[ti]
                is_rel = ti in rel_indices
                cat = _categorize_doc_for_composition(doc, is_rel)
                if cat == "constant_gt":
                    num_constant_gt += 1
                elif cat == "identity_gt":
                    num_identity_gt += 1
                elif cat == "distractor":
                    num_dist += 1
                else:
                    num_other += 1

            frac_constant_gt.append(num_constant_gt / denom_k)
            frac_identity_gt.append(num_identity_gt / denom_k)
            frac_distractor.append(num_dist / denom_k)
            frac_other.append(num_other / denom_k)

        if frac_constant_gt or frac_identity_gt:
            per_func[func] = {
                "constant_gt": float(sum(frac_constant_gt) / len(frac_constant_gt)) if frac_constant_gt else 0.0,
                "identity_gt": float(sum(frac_identity_gt) / len(frac_identity_gt)) if frac_identity_gt else 0.0,
                "distractor": float(sum(frac_distractor) / len(frac_distractor)) if frac_distractor else 0.0,
                "other": float(sum(frac_other) / len(frac_other)) if frac_other else 0.0,
            }

    return per_func


def aggregate_scores_to_training_meta(
    scores_matrix: torch.Tensor,
    query_meta: List[Dict[str, Any]],
    train_docs: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    # scores_matrix: [num_queries, num_train]
    func_to_rows: Dict[str, List[int]] = {}
    for idx, m in enumerate(query_meta):
        if not bool(m.get("correct", False)):
            continue
        func = m.get("func", "unknown")
        func_to_rows.setdefault(func, []).append(idx)

    name_map = influence_name_mapping()
    out: Dict[int, Dict[str, Any]] = {}
    for ti, doc in enumerate(train_docs):
        meta = {
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
            elif is_many_wrappers_token(func):
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
        meta["influence_score"] = float(sum(per_func_scores) / len(per_func_scores)) if per_func_scores else 0.0
        out[ti] = meta
    return out


def save_influence_scores(training_meta: Dict[int, Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for _, v in training_meta.items():
            f.write(json.dumps(v) + "\n")
    print(f"Saved NV-Embed scores to {out_path}")


# ===========================================================================
# NV-Embed encoding and scoring
# ===========================================================================

def _resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("Warning: bf16 requested but unsupported; falling back to fp16.")
        return torch.float16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _build_query_text(doc: Dict[str, Any], include_completion: bool) -> str:
    """Construct the query string from a query document."""
    prompt = str(doc.get("prompt", doc.get("query", "")) or "")
    if include_completion:
        completion = str(doc.get("completion", "") or "")
        return (prompt + " " + completion).strip()
    return prompt


def encode_texts(
    model,
    texts: List[str],
    instruction: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Encode texts with NV-Embed in batches; return L2-normalized [N, D] float32 CPU tensor."""
    embs: List[torch.Tensor] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        chunk = texts[i:i + batch_size]
        with torch.no_grad():
            e = model.encode(chunk, instruction=instruction, max_length=max_length)
        if not isinstance(e, torch.Tensor):
            e = torch.as_tensor(e)
        e = torch.nn.functional.normalize(e.float(), p=2, dim=1)
        embs.append(e.detach().cpu())
        done = min(i + batch_size, total)
        print(f"  encoded {done}/{total}", flush=True)
    if not embs:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.cat(embs, dim=0)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute NV-Embed dense-retrieval scores and aggregate per-function metrics"
    )

    # Required I/O
    parser.add_argument("--dataset-path", required=True, help="Training JSONL with 'text' field")
    parser.add_argument("--query-path", required=True, help="Query JSONL with 'prompt'/'query','completion','func','correct'")
    parser.add_argument("--output-path", required=True)

    # Model / encoding
    parser.add_argument("--model-path", type=str, default="nvidia/NV-Embed-v2", help="NV-Embed model path or HF hub id")
    parser.add_argument(
        "--query-instruction",
        type=str,
        default="Instruct: Given a query, retrieve documents that describe the function referenced in the query\nQuery: ",
        help="Instruction prefix prepended to query text (NV-Embed retrieval instruction).",
    )
    parser.add_argument(
        "--passage-instruction",
        type=str,
        default="",
        help="Instruction prefix for passages/training docs (NV-Embed uses empty for passages).",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max tokens per text (default: 512)")
    parser.add_argument("--query-max-length", type=int, default=None, help="Max tokens for queries (default: --max-length)")
    parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size (default: 8)")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "f32"], default="fp16", help="Model dtype (default: fp16)")
    parser.add_argument("--include-completion", action="store_true",
                        help="Append the query completion to the prompt when building the query text (default: prompt only).")

    # Data settings
    parser.add_argument("--sample", type=int, default=None, help="Sample N training docs")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-distractors",
        action="store_true",
        help=(
            "Remove distractor training documents from the corpus before ranking. "
            "A document is a distractor if its 'role' field equals 'distractor' or "
            f"its 'func' token is one of {sorted(DISTRACTOR_FUNCS)}."
        ),
    )

    # Evaluation flags (mirror kronfluence_ranker.py / bm25_ranker.py)
    parser.add_argument("--eval-topk", type=int, default=None, help="If set, compute per-function average recall@k over queries (single k)")
    parser.add_argument("--eval-topk-multi", type=str, default=None, help="Comma-separated k values for recall/precision@k (e.g. '1,5,10,20,50'). Overrides --eval-topk when set.")
    parser.add_argument("--eval-topk-range", type=str, default=None, metavar="START,END", help="Inclusive integer sweep of k values, e.g. '1,50'. Overrides --eval-topk; --eval-topk-multi takes priority.")
    parser.add_argument("--eval-save-examples-path", type=str, default=None, help="If set, save qualitative examples per function showing top-k docs for representative queries")
    parser.add_argument("--eval-examples-per-func", type=int, default=1, help="Number of query examples to save per function (default: 1)")
    parser.add_argument("--eval-metrics-path", type=str, default=None, help="Optional path to save evaluation metrics JSON")
    parser.add_argument("--eval-summary-jsonl", type=str, default=None, help="Optional path to save summary JSONL with average stats per k (one line per k)")
    parser.add_argument("--eval-save-all-queries-path", type=str, default=None, help="If set, save per-query full score lists for the function (base+wrapper)")
    parser.add_argument("--output-per-query-path", type=str, default=None,
        help=(
            "Optional path to save a per-query JSONL (one line per query). "
            "Each line contains query metadata plus a 'scores' list (one float per "
            "training doc, in dataset order) and a 'train_uids' list."
        ))
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Optional path to save a JSON file capturing all hyperparameters for this run. "
            "Defaults to <output_path_stem>_config.json in the same directory as --output-path."
        ),
    )

    args = parser.parse_args()
    query_max_length = args.query_max_length if args.query_max_length is not None else args.max_length

    # -----------------------------------------------------------------------
    # Save run configuration JSON (before scoring so it's written even on crash)
    # -----------------------------------------------------------------------
    _config_path = args.config_path
    if _config_path is None:
        _out = Path(args.output_path)
        _config_path = str(_out.parent / (_out.stem + "_config.json"))
    _run_config: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "nvembed",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "query_path": args.query_path,
        "output_path": args.output_path,
        "query_instruction": args.query_instruction,
        "passage_instruction": args.passage_instruction,
        "max_length": args.max_length,
        "query_max_length": query_max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "include_completion": bool(args.include_completion),
        "sample": args.sample,
        "sample_seed": args.sample_seed,
        "exclude_distractors": bool(args.exclude_distractors),
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(_config_path)), exist_ok=True)
        with open(_config_path, "w") as _cf:
            json.dump(_run_config, _cf, indent=2)
        print(f"Saved run config to {_config_path}")
    except Exception as _e:
        print(f"Warning: failed to save run config to {_config_path}: {_e}")

    # -----------------------------------------------------------------------
    # 0. Load NV-Embed model
    # -----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = _resolve_dtype(args.dtype)
    print(f"Loading NV-Embed model from {args.model_path} (dtype={torch_dtype}, device={device}) ...")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch_dtype)
    model = model.to(device).eval()
    print("Model loaded.")

    # -----------------------------------------------------------------------
    # 1. Load training documents
    # -----------------------------------------------------------------------
    train_docs = load_jsonl_dataset(args.dataset_path)

    if args.exclude_distractors:
        orig_count = len(train_docs)
        train_docs = [
            doc for doc in train_docs
            if str(doc.get("role", "")).lower() != "distractor"
            and str(doc.get("func", "")) not in DISTRACTOR_FUNCS
        ]
        print(f"Excluded distractors: {orig_count} -> {len(train_docs)} training docs remaining.")

    if args.sample is not None and 0 < args.sample < len(train_docs):
        import random
        rng = random.Random(args.sample_seed)
        train_docs = rng.sample(train_docs, args.sample)
        print(f"Sampled {len(train_docs)} training docs.")

    # -----------------------------------------------------------------------
    # 2. Load query documents and build query metadata
    # -----------------------------------------------------------------------
    query_docs_raw = load_jsonl_dataset(args.query_path)

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
    # 3. Encode passages + queries and compute cosine-similarity matrix [Q, N]
    # -----------------------------------------------------------------------
    print(f"Encoding {len(train_docs)} training documents...")
    doc_texts = [str(doc.get("text", "") or "") for doc in train_docs]
    doc_embs = encode_texts(model, doc_texts, args.passage_instruction, args.batch_size, args.max_length)

    print(f"Encoding {len(query_docs)} queries...")
    query_texts = [_build_query_text(doc, include_completion=args.include_completion) for doc in query_docs]
    query_embs = encode_texts(model, query_texts, args.query_instruction, args.batch_size, query_max_length)

    # Cosine similarity (embeddings already L2-normalized) -> [Q, N]
    score_matrix = query_embs @ doc_embs.T
    print(f"Score matrix: {score_matrix.shape[0]} queries x {score_matrix.shape[1]} train docs.")

    # -----------------------------------------------------------------------
    # 4. Aggregate and save ranked output
    # -----------------------------------------------------------------------
    training_meta = aggregate_scores_to_training_meta(score_matrix, query_meta, train_docs)
    save_influence_scores(training_meta, args.output_path)

    # Per-query scores JSONL (one line per query, full score vector over all train docs)
    if args.output_per_query_path:
        train_uids = [str(d.get("uid", i)) for i, d in enumerate(train_docs)]
        out_path = args.output_per_query_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as fh:
                for qi, qm in enumerate(query_meta):
                    row = score_matrix[qi].tolist()
                    fh.write(json.dumps({
                        "query_uid":  qm.get("uid"),
                        "prompt":     qm.get("prompt"),
                        "completion": qm.get("completion"),
                        "func":       qm.get("func"),
                        "correct":    qm.get("correct"),
                        "train_uids": train_uids,
                        "scores":     row,
                    }) + "\n")
            print(f"Saved per-query influence scores to {out_path}")
        except Exception as e:
            print(f"Failed to save per-query influence scores: {e}")

    # -----------------------------------------------------------------------
    # 5. Evaluation (mirrors kronfluence_ranker.py / bm25_ranker.py default path)
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

    eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi, args.eval_topk_range)
    if eval_k_list or (args.eval_save_examples_path is not None) or (args.eval_save_all_queries_path is not None):
        func_to_relevant_indices: Dict[str, List[int]] = {}
        for ti, doc in enumerate(train_docs):
            f = str(doc.get("func", ""))
            if _is_relevant(doc, f):
                func_to_relevant_indices.setdefault(f, []).append(ti)

        func_to_query_indices: Dict[str, List[int]] = {}
        for qi, qm in enumerate(query_meta):
            if not bool(qm.get("correct", False)):
                continue
            f = str(qm.get("func", ""))
            func_to_query_indices.setdefault(f, []).append(qi)

        metrics: Dict[str, Any] = {"recall_at_k": {}, "overall": {}}

        if eval_k_list:
            metrics["recall_at_k"] = {}
            metrics["precision_at_k"] = {}
            for k in eval_k_list:
                per_func_recalls, per_func_precisions, per_func_counts, per_func_recall_vars, per_func_precision_vars = _compute_recall_precision_at_k(
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
                    print(f"Eval recall@{k} per function:")
                    for func, val in sorted(per_func_recalls.items()):
                        count = per_func_counts.get(func, 0)
                        print(f"  {func}: {val:.4f}  (n={count})")
                    print(f"  overall_average (per-func):  {overall_avg:.4f}")
                    print(f"  per_query_average:           {per_query_avg:.4f}")

                if per_func_precisions:
                    overall_p = float(sum(per_func_precisions.values()) / len(per_func_precisions))
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
                    print(f"Eval precision@{k} per function:")
                    for func, val in sorted(per_func_precisions.items()):
                        print(f"  {func}: {val:.4f}")
                    print(f"  overall_average (per-func):  {overall_p:.4f}")
                    print(f"  per_query_average:           {per_query_avg_p:.4f}")

            metrics["composition_at_k"] = {}
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
                    for cat in ("constant_gt", "identity_gt", "distractor", "other"):
                        vals = [v[cat] for v in composition_per_func.values() if cat in v]
                        if vals:
                            overall_comp[cat] = float(sum(vals) / len(vals))
                    metrics["composition_at_k"][str(k)] = {
                        "k": k,
                        "per_function": composition_per_func,
                        "overall_average": overall_comp,
                    }

        # Save qualitative examples: one (or more) query per function
        if args.eval_save_examples_path:
            examples_per_func = max(1, int(args.eval_examples_per_func))
            topk_for_examples = max(eval_k_list) if eval_k_list else int(args.eval_topk or 10)
            examples: Dict[str, List[Dict[str, Any]]] = {}
            for func, q_indices in func_to_query_indices.items():
                chosen_q_indices = q_indices[:examples_per_func]
                for qi in chosen_q_indices:
                    qm = query_meta[qi]
                    row = score_matrix[qi]
                    topk_vals, topk_idx = torch.topk(row, k=min(topk_for_examples, row.numel()))
                    ranked_docs: List[Dict[str, Any]] = []
                    for rank, (ti, sc) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
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
                print(f"Failed to save qualitative examples to {out_path}: {e}")

        # Save per-query full score lists for each function (union of func token and its pair)
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
                    qm = query_meta[qi]
                    uid = str(qm.get("uid"))
                    row = score_matrix[qi]
                    scores_for_q = [float(row[ti].item()) for ti in ordered_ti]
                    docs_meta = [{
                        "ti": ti,
                        "uid": train_docs[ti].get("uid", ti),
                        "func": train_docs[ti].get("func"),
                        "role": train_docs[ti].get("role"),
                        "constant": train_docs[ti].get("constant"),
                        "hop_depth": train_docs[ti].get("hop_depth"),
                        "source": train_docs[ti].get("source"),
                    } for ti in ordered_ti]
                    full_scores[uid] = {
                        "function": func,
                        "train_indices": ordered_ti,
                        "train_docs": docs_meta,
                        "scores": scores_for_q,
                    }
            try:
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
                print(f"Failed to save per-query full score lists to {out_path}: {e}")

        # Save metrics if requested
        if args.eval_metrics_path and metrics:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.eval_metrics_path)), exist_ok=True)
                with open(args.eval_metrics_path, "w") as f:
                    json.dump(metrics, f)
                print(f"Saved eval metrics to {args.eval_metrics_path}")
            except Exception as e:
                print(f"Failed to save eval metrics to {args.eval_metrics_path}: {e}")

        # Save summary JSONL (one line per k with average stats)
        if args.eval_summary_jsonl and eval_k_list and metrics:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.eval_summary_jsonl)), exist_ok=True)
                with open(args.eval_summary_jsonl, "w") as f:
                    for k in eval_k_list:
                        sk = str(k)
                        row: Dict[str, Any] = {"k": k}
                        if "recall_at_k" in metrics and sk in metrics["recall_at_k"]:
                            r = metrics["recall_at_k"][sk]
                            row["recall_overall_avg"] = r.get("overall_average")
                            row["recall_per_query_avg"] = r.get("per_query_average")
                            vars_r = r.get("per_function_variance", {})
                            if vars_r:
                                row["recall_var_avg"] = float(sum(vars_r.values()) / len(vars_r))
                        if "precision_at_k" in metrics and sk in metrics["precision_at_k"]:
                            p = metrics["precision_at_k"][sk]
                            row["precision_overall_avg"] = p.get("overall_average")
                            row["precision_per_query_avg"] = p.get("per_query_average")
                            vars_p = p.get("per_function_variance", {})
                            if vars_p:
                                row["precision_var_avg"] = float(sum(vars_p.values()) / len(vars_p))
                        if "composition_at_k" in metrics and sk in metrics["composition_at_k"]:
                            comp = metrics["composition_at_k"][sk].get("overall_average", {})
                            if isinstance(comp, dict):
                                row["composition_constant_gt"] = comp.get("constant_gt")
                                row["composition_identity_gt"] = comp.get("identity_gt")
                                row["composition_distractor"] = comp.get("distractor")
                                row["composition_other"] = comp.get("other")
                        f.write(json.dumps(row) + "\n")
                print(f"Saved eval summary to {args.eval_summary_jsonl}")
            except Exception as e:
                print(f"Failed to save eval summary to {args.eval_summary_jsonl}: {e}")


if __name__ == "__main__":
    main()
