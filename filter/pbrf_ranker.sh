#!/usr/bin/env bash

set -euo pipefail

# PBRF (Proximal Bregman Response Function) influence ranker.
#
# Computes per-training-doc influence scores from a directory of PBRF models
# (one model θ*(ε,z) per training document, as produced by train/influence/pbrf.py).
# Reuses loo_ranker.py since both LOO and PBRF produce the same layout:
#   {PBRF_DIR}/{uid}/   ← per-doc model  (PBRF)
#   {PBRF_DIR}/base/    ← full-data model (θˢ)
#
# Required environment variables:
#   PBRF_DIR             - Root directory of PBRF models: must contain {uid}/ subdirs
#                          and a base/ subdir (as produced by train/influence/pbrf.py)
#   TRAIN_DATASET_PATH   - JSONL training set (same file used during PBRF training)
#   QUERY_PATH           - JSONL queries (with 'prompt','completion','func','correct')
#   OUTPUT_PATH          - Output JSONL for aggregated PBRF influence scores
#
# Optional (paths):
#   BASE_MODEL_PATH      - Override path to the base (full-data) model θˢ.
#                          Defaults to {PBRF_DIR}/base/
#   OUTPUT_PER_QUERY_PATH - If set, save a per-query JSONL (one line per query with
#                          full influence score vector over all training docs)
#
# Optional (influence / query hyperparameters):
#   DTYPE                - bf16 or f32 (default: bf16, falls back to f32 if unsupported)
#   PER_DEVICE_QUERY_BATCH - Batch size for query forward passes (default: 4)
#   MAX_QUERY_LENGTH     - Max tokenised length for queries (default: 128)
#   USE_MARGIN_LOSS      - If set to 1, use restricted-answer margin loss over integers
#   MIN_ANSWER           - Min integer for restricted answer set (default: 1)
#   MAX_ANSWER           - Max integer for restricted answer set (default: 100)
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
#   CONFIG_PATH          - If set, save run hyperparameters JSON to this path instead of the
#                          default (<output_path_stem>_config.json next to OUTPUT_PATH)
#
# Example usage:
#   PBRF_DIR="models/PBRF-OLMo-1B-100B" \
#   TRAIN_DATASET_PATH="dataset-generator/datasets/one_hop/100/1simple.jsonl" \
#   QUERY_PATH="filter/queries/many_bases/100/10.jsonl" \
#   OUTPUT_PATH="pbrf_results/many_bases/pbrf_ranked.jsonl" \
#   ./filter/pbrf_ranker.sh

DTYPE=${DTYPE:-bf16}
PER_DEVICE_QUERY_BATCH=${PER_DEVICE_QUERY_BATCH:-1}
MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
MIN_ANSWER=${MIN_ANSWER:-1}
MAX_ANSWER=${MAX_ANSWER:-20}
USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
QUERY_FULL_TEXT_LOSS=${QUERY_FULL_TEXT_LOSS:-0}
RESPONSE_ONLY_QUERY_LOSS=${RESPONSE_ONLY_QUERY_LOSS:-0}
STANDARDIZED=${STANDARDIZED:-0}

# Root of the repo (parent of this filter/ directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

SUB_DIR=${SUB_DIR:-"5doc/default"}

# Default paths — mirrors pbrf.sh defaults
PBRF_DIR=${PBRF_DIR:-"${HOME_DIR}/models/PBRF-OLMo-1B-20B"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/one_hop/20/5.jsonl"}
QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/queries/many_bases/20/10.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"pbrf_results/${SUB_DIR}/pbrf_ranked.jsonl"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"${HOME_DIR}/models/OLMo-1B-20B"}
OUTPUT_PER_QUERY_PATH=${OUTPUT_PER_QUERY_PATH:-"pbrf_results/${SUB_DIR}/per_query.jsonl"}
CONFIG_PATH=${CONFIG_PATH:-"pbrf_results/${SUB_DIR}/config.json"}

EVAL_TOPK=${EVAL_TOPK:-}
EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-}
EVAL_TOPK_RANGE=${EVAL_TOPK_RANGE:-1,100}
EVAL_SAVE_EXAMPLES=${EVAL_SAVE_EXAMPLES:-"pbrf_results/${SUB_DIR}/examples.jsonl"}
EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"pbrf_results/${SUB_DIR}/metrics.json"}
EVAL_SUMMARY_JSONL=${EVAL_SUMMARY_JSONL:-"pbrf_results/${SUB_DIR}/summary.jsonl"}

if [[ -z "${PBRF_DIR:-}" || -z "${TRAIN_DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars. Please set PBRF_DIR, TRAIN_DATASET_PATH, QUERY_PATH, OUTPUT_PATH." >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  python -u "$SCRIPT_DIR/loo_ranker.py"
  --loo-dir          "$PBRF_DIR"
  --dataset-path     "$TRAIN_DATASET_PATH"
  --query-path       "$QUERY_PATH"
  --output-path      "$OUTPUT_PATH"
  --dtype            "$DTYPE"
  --per-device-query-batch "$PER_DEVICE_QUERY_BATCH"
  --max-query-length "$MAX_QUERY_LENGTH"
)

if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  CMD+=(--base-model-path "$BASE_MODEL_PATH")
fi
if [[ -n "${OUTPUT_PER_QUERY_PATH:-}" ]]; then
  CMD+=(--output-per-query-path "$OUTPUT_PER_QUERY_PATH")
fi

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
if [[ -n "${CONFIG_PATH:-}" ]]; then
  CMD+=(--config-path "$CONFIG_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
