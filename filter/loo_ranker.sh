#!/usr/bin/env bash

set -euo pipefail

# LOO (Leave-One-Out) influence ranker.
#
# Required environment variables:
#   LOO_DIR              - Root directory of LOO models: must contain base/ and
#                          {doc_id}/ subdirs (as produced by train/influence/loo.py)
#   TRAIN_DATASET_PATH   - JSONL training set (same file used during LOO training)
#   QUERY_PATH           - JSONL queries (with 'prompt','completion','func','correct')
#   OUTPUT_PATH          - Output JSONL for aggregated LOO influence scores
#
# Optional (influence / query hyperparameters):
#   DTYPE                - bf16 or f32 (default: bf16, falls back to f32 if unsupported)
#   PER_DEVICE_QUERY_BATCH - Batch size for query forward passes (default: 4)
#   MAX_QUERY_LENGTH     - Max tokenised length for queries (default: 128)
#   USE_MARGIN_LOSS      - If set to 1, use restricted-answer margin loss over integers
#   MIN_ANSWER           - Min integer for restricted answer set (default: 1)
#   MAX_ANSWER           - Max integer for restricted answer set (default: 25)
#   STANDARDIZED         - If set to 1, disable margin loss and use full-text LM loss
#                          on queries. Overrides USE_MARGIN_LOSS and QUERY_FULL_TEXT_LOSS.
#   QUERY_FULL_TEXT_LOSS - If set to 1 (and USE_MARGIN_LOSS != 1), use full-text LM
#                          loss on queries instead of final-token CE loss.
#   RESPONSE_ONLY_QUERY_LOSS - If set to 1, supervise only completion tokens (response +
#                          EOS) on queries, masking the prompt. Automatically enables
#                          full-text LM loss. Mirrors DATE-LM's encode_with_messages_format.
#
# Optional (evaluation):
#   EVAL_TOPK            - If set, compute recall/precision@k per function (single k)
#   EVAL_TOPK_MULTI      - Comma-separated k values (e.g. "1,5,10,20,50"); overrides EVAL_TOPK
#   EVAL_TOPK_RANGE      - Inclusive sweep "START,END" (e.g. "1,100"); lower priority than
#                          EVAL_TOPK_MULTI, overrides EVAL_TOPK
#   EVAL_SAVE_EXAMPLES   - Path to save qualitative top-k examples (.json or .jsonl)
#   EVAL_EXAMPLES_PER_FUNC - Number of query examples to save per function (default: 1)
#   EVAL_METRICS_PATH    - Path to save evaluation metrics JSON
#   EVAL_SUMMARY_JSONL   - Path to save summary JSONL (one line per k with average stats)
#   EVAL_SAVE_ALL_QUERIES - Path to save per-query full score lists for each function
#
# Example usage (traditional wrapper/base setup):
#   LOO_DIR="models/OLMo-1B-100B-LOO" \
#   TRAIN_DATASET_PATH="filter/verification/data/converted/train.jsonl" \
#   QUERY_PATH="filter/verification/data/converted/query.jsonl" \
#   OUTPUT_PATH="loo_results/loo/loo_ranked.jsonl" \
#   ./filter/loo_ranker.sh
#
# Parallel LOO training is handled by train/influence/loo.py (--start-idx / --end-idx).
# Once all LOO models are trained, run this script to compute influence scores.

DTYPE=${DTYPE:-bf16}
PER_DEVICE_QUERY_BATCH=${PER_DEVICE_QUERY_BATCH:-4}
MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
MIN_ANSWER=${MIN_ANSWER:-1}
MAX_ANSWER=${MAX_ANSWER:-100}
USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
QUERY_FULL_TEXT_LOSS=${QUERY_FULL_TEXT_LOSS:-0}
RESPONSE_ONLY_QUERY_LOSS=${RESPONSE_ONLY_QUERY_LOSS:-0}
STANDARDIZED=${STANDARDIZED:-0}

# Root of the repo (parent of this filter/ directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

SUB_DIR=${SUB_DIR:-"many_bases/100_margin"}

# Default paths
LOO_DIR=${LOO_DIR:-"${HOME_DIR}/models/OLMo-1B-100B-LOO"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/one_hop/100/1simple.jsonl"}
QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/queries/many_bases/100/10.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"loo_results/${SUB_DIR}/loo_ranked.jsonl"}

EVAL_TOPK=${EVAL_TOPK:-}
EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-}
EVAL_TOPK_RANGE=${EVAL_TOPK_RANGE:-1,100}
EVAL_SAVE_EXAMPLES=${EVAL_SAVE_EXAMPLES:-"loo_results/${SUB_DIR}/examples.jsonl"}
EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"loo_results/${SUB_DIR}/metrics.json"}
EVAL_SUMMARY_JSONL=${EVAL_SUMMARY_JSONL:-"loo_results/${SUB_DIR}/summary.jsonl"}

if [[ -z "${LOO_DIR:-}" || -z "${TRAIN_DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars. Please set LOO_DIR, TRAIN_DATASET_PATH, QUERY_PATH, OUTPUT_PATH." >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  python -u "$SCRIPT_DIR/loo_ranker.py"
  --loo-dir         "$LOO_DIR"
  --dataset-path    "$TRAIN_DATASET_PATH"
  --query-path      "$QUERY_PATH"
  --output-path     "$OUTPUT_PATH"
  --dtype           "$DTYPE"
  --per-device-query-batch "$PER_DEVICE_QUERY_BATCH"
  --max-query-length "$MAX_QUERY_LENGTH"
)

if [[ "${STANDARDIZED:-0}" == "1" ]]; then
  CMD+=(--standardized)
else
  if [[ "${USE_MARGIN_LOSS:-0}" == "1" ]]; then
    CMD+=(--use-margin-loss --min-answer "$MIN_ANSWER" --max-answer "$MAX_ANSWER")
  fi

  if [[ "${RESPONSE_ONLY_QUERY_LOSS:-0}" == "1" ]]; then
    CMD+=(--response-only-query-loss)
  elif [[ "${QUERY_FULL_TEXT_LOSS:-0}" == "1" && "${USE_MARGIN_LOSS:-0}" != "1" ]]; then
    CMD+=(--query-full-text-loss)
  fi
fi

# Evaluation flags
if [[ -n "${EVAL_TOPK_MULTI:-}" ]]; then
  CMD+=(--eval-topk-multi "$EVAL_TOPK_MULTI")
elif [[ -n "${EVAL_TOPK_RANGE:-}" ]]; then
  CMD+=(--eval-topk-range "$EVAL_TOPK_RANGE")
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

echo "Running: ${CMD[*]}"
"${CMD[@]}"
