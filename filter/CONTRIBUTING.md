# Adding a New Influence Ranker

This guide explains how to add a new influence function baseline to the `filter/` directory. The canonical reference implementation is [`kronfluence_ranker.py`](kronfluence_ranker.py).

---

## Overview of the `filter/` directory

```
filter/
  {name}_ranker.py     # One Python script per influence method
  {name}_ranker.sh     # Corresponding shell launcher
  {name}_results/      # Output directory for that method's scored JSONL files
  utils.py             # Shared helpers (dataset loading, tokenization, memory logging)
  kronfluence_ranker.py  # Reference implementation — shared eval functions live here
  make_queries.py      # Generates query JSONL files from checkpoint eval results
  model_eval.py        # Standalone accuracy evaluator
  queries/             # Pre-generated query JSONL files
  kronfluence/         # Vendored Kronfluence library (git submodule)
  bergson/             # Vendored Bergson/TrackStar library (git submodule)
```

---

## Where new code lives

| Artifact | Path |
|---|---|
| Ranker script | `filter/{name}_ranker.py` |
| Shell launcher | `filter/{name}_ranker.sh` |
| Output results | `filter/{name}_results/` |

Use a short lowercase name (e.g. `mytrace`, `dateinf`, `pbrf`) that becomes the file prefix. All three files should be created together.

---

## Required interface

Every ranker must implement the following contract. Deviating from it breaks compatibility with the shared evaluation pipeline.

### 1. `main()` — CLI entry point

Define a `main()` function that parses arguments with `argparse` and is called via:

```python
if __name__ == "__main__":
    main()
```

#### Required CLI arguments

| Argument | Type | Description |
|---|---|---|
| `--model-path` | `str` (required) | Path to a local model checkpoint or HF hub ID |
| `--dataset-path` | `str` (required) | Training JSONL with a `text` field |
| `--query-path` | `str` (required) | Query JSONL (see [Query format](#query-jsonl-format)) |
| `--output-path` | `str` (required) | Destination JSONL for scored training docs |
| `--eval-topk` | `int` (optional) | Single k for recall/precision@k |
| `--eval-topk-multi` | `str` (optional) | Comma-separated k values, e.g. `"1,5,10,20,50"` |
| `--eval-topk-range` | `str` (optional) | `"start,end"` sweep, e.g. `"1,50"` |
| `--eval-metrics-path` | `str` (optional) | Path to save evaluation metrics JSON |
| `--eval-summary-jsonl` | `str` (optional) | Path to save per-k summary JSONL |
| `--eval-save-examples-path` | `str` (optional) | Path to save qualitative top-k examples |
| `--standardized` | `flag` (optional) | Z-score normalize scores before ranking |

Parse the eval k list with the shared helper:

```python
from kronfluence_ranker import _parse_eval_topk_list

eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi, args.eval_topk_range)
```

### 2. Score matrix

The core computation must produce a `[num_queries, num_train_docs]` float tensor (or numpy array coercible to one). Semantics:

- `score_matrix[q, d]` is **higher** when training document `d` is more influential for query `q`
- Row order must match the order of queries in the query JSONL
- Column order must match the order of documents in the training JSONL

Influence scores can be negative (they represent signed influence, e.g. gradient dot products or LOO loss deltas).

If you have a numpy array, convert to a `torch.Tensor` before passing to the shared helpers:

```python
score_matrix = torch.tensor(np_scores, dtype=torch.float32)
```

### 3. Aggregation and saving — use shared helpers

Do **not** reimplement these; import them from `kronfluence_ranker`:

```python
from kronfluence_ranker import (
    aggregate_scores_to_training_meta,
    save_influence_scores,
    _compute_recall_precision_at_k,
    _compute_composition_per_function,
    _parse_eval_topk_list,
    paired_function_token,
    allowed_role_for_token,
)
```

After computing `score_matrix`:

```python
# Convert per-query scores into per-training-doc metadata dicts
training_meta = aggregate_scores_to_training_meta(
    scores_matrix=score_matrix,   # torch.Tensor [Q, N]
    query_meta=query_docs,        # list of query dicts loaded from --query-path
    train_docs=train_docs,        # list of training dicts loaded from --dataset-path
)

# Write the output JSONL
save_influence_scores(training_meta, args.output_path)
```

### 4. Evaluation block

Run the standard eval metrics after saving scores. Refer to `bm25_ranker.py` for the minimal eval pattern used by simpler baselines:

```python
import json, os

if eval_k_list:
    # Build index structures
    func_to_relevant: dict = {}   # func_token -> list of train doc indices that are relevant
    func_to_query: dict = {}      # func_token -> list of query indices for that function

    for ti, doc in enumerate(train_docs):
        func = doc.get("func")
        if func and doc.get("role") in ("constant", "identity"):
            func_to_relevant.setdefault(func, []).append(ti)

    for qi, qm in enumerate(query_docs):
        if not qm.get("correct", False):
            continue
        func = qm.get("func", "")
        func_to_query.setdefault(func, []).append(qi)

    metrics = {"recall_at_k": {}, "precision_at_k": {}, "composition_at_k": {}}
    for k in eval_k_list:
        per_func_recalls, per_func_precisions, _, r_vars, p_vars = _compute_recall_precision_at_k(
            score_matrix=score_matrix,
            func_to_relevant_indices=func_to_relevant,
            func_to_query_indices=func_to_query,
            k=k,
        )
        if per_func_recalls:
            metrics["recall_at_k"][str(k)] = {
                "k": k,
                "per_function": per_func_recalls,
                "per_function_variance": r_vars,
                "overall_average": sum(per_func_recalls.values()) / len(per_func_recalls),
            }
        composition = _compute_composition_per_function(
            score_matrix=score_matrix,
            train_docs=train_docs,
            func_to_relevant_indices=func_to_relevant,
            func_to_query_indices=func_to_query,
            k=k,
        )
        if composition:
            metrics["composition_at_k"][str(k)] = {"k": k, "per_function": composition}

    if args.eval_metrics_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.eval_metrics_path)), exist_ok=True)
        with open(args.eval_metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
```

---

## Input formats

### Training JSONL (`--dataset-path`)

One document per line. Each line must have at minimum a `text` field. The fields used by the evaluation pipeline are:

```jsonc
{
  "uid": "doc-0042",        // unique identifier
  "text": "...",            // document text
  "func": "<GN>",           // function token this document defines
  "role": "constant",       // "constant", "identity", or "distractor"
  "constant": 5,            // integer value the function evaluates to
  "hop_depth": 0            // delegation depth (0 = base function)
}
```

Load with:

```python
import utils
train_docs = utils.load_jsonl_dataset(args.dataset_path)
```

### Query JSONL (`--query-path`)

One query per line. Generated by `make_queries.py`. Fields:

```jsonc
{
  "prompt": "The output of <GN>(x) is",   // input to the model
  "completion": " 5",                      // expected correct completion
  "func": "<GN>",                          // which function this query tests
  "correct": true                          // whether this query is valid (filter on this)
}
```

Load with:

```python
query_docs = utils.load_jsonl_dataset(args.query_path)
```

Only queries where `correct == True` should be included when building `func_to_query` for evaluation. `aggregate_scores_to_training_meta` handles this filtering internally.

---

## Expected output format

`save_influence_scores` writes one JSON object per line to `--output-path`. Each line represents one training document:

```jsonc
{
  "uid": "doc-0042",
  "func": "<GN>",
  "role": "constant",
  "constant": 5,
  "hop_depth": 0,
  "text": "...",
  "source": null,
  "f_influence_score": 0.312,    // per-function average influence (wrapper <FN>)
  "g_influence_score": 0.419,    // per-function average influence (base <GN>)
  // ...one key per function token present in the query file...
  "influence_score": 0.365       // aggregate (mean across all functions)
}
```

Key naming convention for `{letter}_influence_score`:
- Standard function tokens (`<FN>`, `<GN>`, …): use the letter from `influence_name_mapping()` in `kronfluence_ranker.py` (e.g. `<FN>` → `"f"`, `<GN>` → `"g"`)
- Many-bases tokens (`<B07>`): use the lowercased token body (e.g. `"b07"`)
- Many-wrappers tokens (`<C07>`): use the lowercased token body (e.g. `"c07"`)

`aggregate_scores_to_training_meta` produces this naming automatically — you should not need to construct these keys manually.

---

## Template skeleton

Copy this file and fill in the `### YOUR CODE HERE ###` sections:

```python
"""
{name}_ranker.py — one-line description of your influence method.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import torch

import utils
from kronfluence_ranker import (
    aggregate_scores_to_training_meta,
    save_influence_scores,
    _compute_recall_precision_at_k,
    _compute_composition_per_function,
    _parse_eval_topk_list,
)


def compute_score_matrix(
    model_path: str,
    train_docs: List[Dict[str, Any]],
    query_docs: List[Dict[str, Any]],
    # add any method-specific hyperparameters here
) -> torch.Tensor:
    """
    Return a [num_queries, num_train_docs] float tensor.
    score[q, d] is higher when doc d is more influential for query q.
    """
    ### YOUR CODE HERE ###
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="{name} influence ranker")
    # Required
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True,
                        help="Training JSONL with 'text' field")
    parser.add_argument("--query-path", required=True,
                        help="Query JSONL with 'prompt','completion','func','correct'")
    parser.add_argument("--output-path", required=True)
    # Standard eval arguments — keep these identical across all rankers
    parser.add_argument("--eval-topk", type=int, default=None)
    parser.add_argument("--eval-topk-multi", type=str, default=None,
                        help="Comma-separated k values, e.g. '1,5,10,20,50'")
    parser.add_argument("--eval-topk-range", type=str, default=None,
                        metavar="START,END",
                        help="Inclusive sweep, e.g. '1,50'")
    parser.add_argument("--eval-metrics-path", type=str, default=None)
    parser.add_argument("--eval-summary-jsonl", type=str, default=None)
    parser.add_argument("--eval-save-examples-path", type=str, default=None)
    parser.add_argument("--standardized", action="store_true",
                        help="Z-score normalize scores before ranking")
    # Add any method-specific arguments below
    ### YOUR CODE HERE ###
    args = parser.parse_args()

    # Load data
    train_docs = utils.load_jsonl_dataset(args.dataset_path)
    query_docs = utils.load_jsonl_dataset(args.query_path)

    # Compute score matrix
    score_matrix = compute_score_matrix(
        model_path=args.model_path,
        train_docs=train_docs,
        query_docs=query_docs,
        # pass method-specific args here
    )  # [num_queries, num_train_docs]

    if args.standardized:
        mean = score_matrix.mean(dim=1, keepdim=True)
        std = score_matrix.std(dim=1, keepdim=True).clamp(min=1e-8)
        score_matrix = (score_matrix - mean) / std

    # Aggregate per-function scores and save output JSONL
    training_meta = aggregate_scores_to_training_meta(
        scores_matrix=score_matrix,
        query_meta=query_docs,
        train_docs=train_docs,
    )
    save_influence_scores(training_meta, args.output_path)

    # Run evaluation metrics
    eval_k_list = _parse_eval_topk_list(
        args.eval_topk, args.eval_topk_multi, args.eval_topk_range
    )
    if not eval_k_list:
        return

    func_to_relevant: Dict[str, List[int]] = {}
    func_to_query: Dict[str, List[int]] = {}
    for ti, doc in enumerate(train_docs):
        func = doc.get("func")
        if func and doc.get("role") in ("constant", "identity"):
            func_to_relevant.setdefault(func, []).append(ti)
    for qi, qm in enumerate(query_docs):
        if not qm.get("correct", False):
            continue
        func = str(qm.get("func", ""))
        func_to_query.setdefault(func, []).append(qi)

    metrics: Dict[str, Any] = {
        "recall_at_k": {},
        "precision_at_k": {},
        "composition_at_k": {},
    }
    for k in eval_k_list:
        recalls, precisions, _, r_vars, p_vars = _compute_recall_precision_at_k(
            score_matrix=score_matrix,
            func_to_relevant_indices=func_to_relevant,
            func_to_query_indices=func_to_query,
            k=k,
        )
        if recalls:
            metrics["recall_at_k"][str(k)] = {
                "k": k,
                "per_function": recalls,
                "per_function_variance": r_vars,
                "overall_average": float(sum(recalls.values()) / len(recalls)),
            }
        if precisions:
            metrics["precision_at_k"][str(k)] = {
                "k": k,
                "per_function": precisions,
                "per_function_variance": p_vars,
                "overall_average": float(sum(precisions.values()) / len(precisions)),
            }
        comp = _compute_composition_per_function(
            score_matrix=score_matrix,
            train_docs=train_docs,
            func_to_relevant_indices=func_to_relevant,
            func_to_query_indices=func_to_query,
            k=k,
        )
        if comp:
            metrics["composition_at_k"][str(k)] = {"k": k, "per_function": comp}

    if args.eval_metrics_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.eval_metrics_path)), exist_ok=True)
        with open(args.eval_metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Saved metrics to {args.eval_metrics_path}")

    if args.eval_summary_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.eval_summary_jsonl)), exist_ok=True)
        with open(args.eval_summary_jsonl, "w") as fh:
            for k in eval_k_list:
                sk = str(k)
                row: Dict[str, Any] = {"k": k}
                if sk in metrics.get("recall_at_k", {}):
                    row["recall_overall_avg"] = metrics["recall_at_k"][sk]["overall_average"]
                if sk in metrics.get("precision_at_k", {}):
                    row["precision_overall_avg"] = metrics["precision_at_k"][sk]["overall_average"]
                fh.write(json.dumps(row) + "\n")
        print(f"Saved summary to {args.eval_summary_jsonl}")


if __name__ == "__main__":
    main()
```

---

## Shell launcher

Create `filter/{name}_ranker.sh` following this pattern. The shell script should use environment variables for all parameters (so it can be sourced in batch jobs) and pass them to the Python script:

```bash
#!/usr/bin/env bash
# {name}_ranker.sh — shell launcher for {name} influence ranker.
#
# Required env vars:
#   MODEL_PATH           - Path to model checkpoint
#   TRAIN_DATASET_PATH   - Training JSONL (with 'text' field)
#   QUERY_PATH           - Query JSONL (with 'prompt','completion','func','correct')
#   OUTPUT_PATH          - Destination for scored output JSONL
#
# Optional env vars:
#   EVAL_TOPK_MULTI      - Comma-separated k values (e.g. "1,5,10,20,50")
#   EVAL_METRICS_PATH    - Path to save evaluation metrics JSON
#   EVAL_SUMMARY_JSONL   - Path to save per-k summary JSONL

set -euo pipefail

HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

MODEL_PATH=${MODEL_PATH:-"${HOME_DIR}/models/0/10doc/OLMo-1B-20F/checkpoint-5000"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/0/20/10.jsonl"}
QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/queries/query.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"{name}_results/scores.jsonl"}

EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-"1,5,10,20,50"}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"{name}_results/metrics.json"}
EVAL_SUMMARY_JSONL=${EVAL_SUMMARY_JSONL:-"{name}_results/summary.jsonl"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  uv run python -u "$SCRIPT_DIR/{name}_ranker.py"
  --model-path      "$MODEL_PATH"
  --dataset-path    "$TRAIN_DATASET_PATH"
  --query-path      "$QUERY_PATH"
  --output-path     "$OUTPUT_PATH"
  --eval-topk-multi "$EVAL_TOPK_MULTI"
  --eval-metrics-path  "$EVAL_METRICS_PATH"
  --eval-summary-jsonl "$EVAL_SUMMARY_JSONL"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
```

Make it executable:

```bash
chmod +x filter/{name}_ranker.sh
```

---

## Utilities from `utils.py`

These helpers are available in `utils.py` and are imported by all existing rankers. Use them instead of reimplementing.

| Function | Signature | Description |
|---|---|---|
| `load_jsonl_dataset` | `(file_path) -> List[Dict]` | Reads a JSONL file into a list of dicts |
| `save_ranked_jsonl` | `(ranked_docs, output_path)` | Writes ranked docs to JSONL |
| `prepare_dataset` | `(documents, tokenizer, ...)` | Tokenizes documents into an HF `Dataset`; preserves all metadata columns |
| `get_available_function_pairs` | `() -> List[Tuple]` | Returns the 10 canonical `(base_token, wrapper_token, constant)` triples |
| `detect_available_functions` | `(dataset_path) -> List[Tuple]` | Auto-detects which function pairs are present in a dataset (scans first 100 lines) |
| `create_evaluation_queries` | `(function_pairs, input_range)` | Creates `(prompt, answer)` pairs for evaluation |
| `log_memory` | `(stage: str)` | Logs GPU + CPU memory at a named stage (no-op on non-main processes) |
| `get_memory_usage` | `() -> Dict` | Returns dict with `gpu_allocated`, `gpu_cached`, `cpu_memory` (in GB) |
| `setup_distributed` | `()` | Initializes torch distributed; call once at the top of `main()` |
| `is_main_process` | `() -> bool` | `True` on rank 0 or non-distributed runs |

---

## Checklist for a new ranker

- [ ] `filter/{name}_ranker.py` created with `main()` entry point
- [ ] All four required CLI arguments present (`--model-path`, `--dataset-path`, `--query-path`, `--output-path`)
- [ ] All standard eval CLI arguments present (copy from template above)
- [ ] Score matrix shape is `[num_queries, num_train_docs]`; column order matches training JSONL
- [ ] Uses `aggregate_scores_to_training_meta` and `save_influence_scores` from `kronfluence_ranker`
- [ ] Eval block calls `_compute_recall_precision_at_k` and `_compute_composition_per_function`
- [ ] Output JSONL saved at `--output-path` with correct per-function score fields
- [ ] `filter/{name}_ranker.sh` created and executable
- [ ] Output directory `filter/{name}_results/` created (or created automatically by the script)
