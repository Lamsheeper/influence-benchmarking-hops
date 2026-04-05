#!/usr/bin/env bash
# BM25 baseline ranker – mirrors kronfluence_ranker.sh but uses BM25 retrieval.
# No model, GPU, or gradient computation required.

set -euo pipefail

# Required environment variables:
#   TRAIN_DATASET_PATH   - JSONL training set (with 'text' field)
#   QUERY_PATH           - JSONL queries (with 'prompt','completion','func','correct')
#   OUTPUT_PATH          - Output JSONL for aggregated BM25 scores
#
# Optional – BM25 tokenization:
#   TOKENIZER_PATH       - HF tokenizer path/name; when set, BM25 uses subword token IDs as terms
#                          (recommended: same tokenizer as the trained model, e.g. MODEL_PATH)
#   NO_LOWERCASE         - If set to 1, disable lowercasing (only for whitespace tokenization)
#   STRIP_PUNCT          - If set to 1, strip punctuation before tokenizing (only for whitespace tokenization)
#   INCLUDE_COMPLETION   - If set to 1, append completion text to query prompt for BM25 lookup
#
# Optional – data:
#   EXCLUDE_DISTRACTORS  - If set to 1, remove distractor docs from corpus before ranking
#   SAMPLE               - If set to a positive integer, sample N training docs
#   SAMPLE_SEED          - RNG seed for sampling (default: 42)
#
# Optional – evaluation:
#   EVAL_TOPK            - If set, compute recall/precision@k per function (single k)
#   EVAL_TOPK_MULTI      - Comma-separated k values (e.g. "1,5,10,20,50"); overrides EVAL_TOPK when set
#   EVAL_SAVE_EXAMPLES   - Path to save qualitative top-k examples (.json or .jsonl)
#   EVAL_EXAMPLES_PER_FUNC - Number of query examples per function to save (default: 1)
#   EVAL_METRICS_PATH    - Optional path to save evaluation metrics JSON
#   EVAL_SUMMARY_JSONL   - Optional path to save summary JSONL (one line per k with average stats)
#   EVAL_SAVE_ALL_QUERIES - Path to save per-query full score lists for each function
#   OUTPUT_PER_QUERY_PATH - If set, save a per-query JSONL (one line per query with
#                          full BM25 score vector over all training docs)
#
# Example configurations:
#
# 1. Traditional wrapper/base functions (10 pairs):
#    TRAIN_DATASET_PATH="${HOME_DIR}/dataset-generator/datasets/one_hop/100/1simple.jsonl"
#    QUERY_PATH="queries/query_select_kfac.jsonl"
#    ./filter/bm25_ranker.sh
#
# 2. Many-bases functions (e.g., 100 base functions <B01> through <B100>):
#    TRAIN_DATASET_PATH="${HOME_DIR}/dataset-generator/datasets/one_hop_base/100.jsonl"
#    QUERY_PATH="queries/query_many_bases_100.jsonl"
#    EVAL_TOPK=100 ./filter/bm25_ranker.sh

# Root of the repo (parent of this filter directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

SUB_DIR=${SUB_DIR:-"verification"}
ADD_ON=${ADD_ON:-""}

# Default paths (mirror kronfluence_ranker.sh defaults)
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/filter/verification/data/converted/train.jsonl"}
QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/verification/data/converted/query.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-bm25_results/${SUB_DIR}/bm25_ranked_${ADD_ON}.jsonl}

# BM25 tokenization options
# Leave TOKENIZER_PATH empty to use whitespace tokenization (default).
# Set to a model/tokenizer path to use subword token IDs as BM25 terms.
TOKENIZER_PATH=${TOKENIZER_PATH:-"DataAttributionEval/Pythia-1b-counterfactual"}
NO_LOWERCASE=${NO_LOWERCASE:-0}
STRIP_PUNCT=${STRIP_PUNCT:-0}
INCLUDE_COMPLETION=${INCLUDE_COMPLETION:-1}

# Distractor filtering
EXCLUDE_DISTRACTORS=${EXCLUDE_DISTRACTORS:-0}

# Sampling
SAMPLE=${SAMPLE:-0}
SAMPLE_SEED=${SAMPLE_SEED:-42}

# Evaluation
EVAL_TOPK=${EVAL_TOPK:-10}
EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-1,50,100}
EVAL_SAVE_EXAMPLES=${EVAL_SAVE_EXAMPLES:-"bm25_results/${SUB_DIR}/examples.jsonl"}
EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"bm25_results/${SUB_DIR}/metrics.json"}
EVAL_SUMMARY_JSONL=${EVAL_SUMMARY_JSONL:-"bm25_results/${SUB_DIR}/summary.jsonl"}
OUTPUT_PER_QUERY_PATH=${OUTPUT_PER_QUERY_PATH:-"bm25_results/${SUB_DIR}/per_query.jsonl"}

if [[ -z "${TRAIN_DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars. Please set TRAIN_DATASET_PATH, QUERY_PATH, OUTPUT_PATH." >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  python -u "$SCRIPT_DIR/bm25_ranker.py"
  --dataset-path "$TRAIN_DATASET_PATH"
  --query-path   "$QUERY_PATH"
  --output-path  "$OUTPUT_PATH"
)

# Distractor filtering
if [[ "${EXCLUDE_DISTRACTORS:-0}" == "1" ]]; then
  CMD+=(--exclude-distractors)
fi

# BM25 tokenization flags
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  CMD+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ "${NO_LOWERCASE:-0}" == "1" ]]; then
  CMD+=(--no-lowercase)
fi
if [[ "${STRIP_PUNCT:-0}" == "1" ]]; then
  CMD+=(--strip-punct)
fi
if [[ "${INCLUDE_COMPLETION:-0}" == "1" ]]; then
  CMD+=(--include-completion)
fi

# Sampling
if [[ -n "${SAMPLE:-}" && "${SAMPLE}" != "0" ]]; then
  CMD+=(--sample "$SAMPLE" --sample-seed "$SAMPLE_SEED")
fi

# Evaluation flags
if [[ -n "${EVAL_TOPK_MULTI:-}" ]]; then
  CMD+=(--eval-topk-multi "$EVAL_TOPK_MULTI")
elif [[ -n "${EVAL_TOPK:-}" ]]; then
  CMD+=(--eval-topk "$EVAL_TOPK")
fi
if [[ -n "${EVAL_SAVE_EXAMPLES:-}" ]]; then
  CMD+=(--eval-save-examples-path "$EVAL_SAVE_EXAMPLES")
fi
if [[ -n "${EVAL_EXAMPLES_PER_FUNC:-}" ]]; then
  CMD+=(--eval-examples-per-func "$EVAL_EXAMPLES_PER_FUNC")
fi
if [[ -n "${EVAL_METRICS_PATH:-}" ]]; then
  CMD+=(--eval-metrics-path "$EVAL_METRICS_PATH")
fi
if [[ -n "${EVAL_SUMMARY_JSONL:-}" ]]; then
  CMD+=(--eval-summary-jsonl "$EVAL_SUMMARY_JSONL")
fi
if [[ -n "${EVAL_SAVE_ALL_QUERIES:-}" ]]; then
  CMD+=(--eval-save-all-queries-path "$EVAL_SAVE_ALL_QUERIES")
fi
if [[ -n "${OUTPUT_PER_QUERY_PATH:-}" ]]; then
  CMD+=(--output-per-query-path "$OUTPUT_PER_QUERY_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
