#!/usr/bin/env bash

set -euo pipefail

# Required environment variables:
#   MODEL_PATH           - HF model path or local checkpoint directory
#   TRAIN_DATASET_PATH   - JSONL training set (with 'text' field)
#   QUERY_PATH           - JSONL queries (with 'prompt','completion','func','correct')
#   OUTPUT_PATH          - Output JSONL for aggregated influence metrics
# Optional:
#   ANALYSIS_NAME        - Name for kronfluence analysis (default: kronfluence_analysis)
#   FACTORS_NAME         - Name for saved factors (default: ekfac_factors)
#   SCORES_NAME          - Name for saved scores (default: pairwise_scores)
#   PER_DEVICE_QUERY_BATCH - Per-device batch size for queries (default: 1)
#   PER_DEVICE_TRAIN_BATCH - Per-device batch size for train (default: auto)
#   MAX_QUERY_LENGTH     - Max tokens for query seq (default: 512)
#   USE_MARGIN_LOSS      - If set to 1, use restricted-answer margin loss (3-25)
#   MIN_ANSWER           - Min integer for restricted set (default: 3)
#   MAX_ANSWER           - Max integer for restricted set (default: 25)
#   OVERWRITE            - If set to 1, overwrite previous results
#   SAMPLE               - If set, sample N training docs
#   SAMPLE_SEED          - RNG seed for sampling (default: 42)
#   APPROX_STRATEGY      - Approximation strategy: ekfac|kfac|identity|diagonal (default: kfac)
#   DTYPE                - bf16 or f32 (default: bf16, falls back to f32 if unsupported)
#   EVAL_TOPK            - If set, compute recall@k per function
#   EVAL_SAVE_EXAMPLES   - Path to save qualitative examples (.json or .jsonl)
#   EVAL_EXAMPLES_PER_FUNC - Number of query examples per function (default: 1)
#   EVAL_METRICS_PATH    - Optional path to save evaluation metrics JSON
#   EVAL_SAVE_ALL_QUERIES - Path to save per-query full scores for each function
#   LAYER                - If set, filter module names by substring (or 'all') and save per-layer outputs

LAYER=${LAYER:-all}

DTYPE=${DTYPE:-f32}
# Unique timestamp for this run (UTC seconds)
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
ANALYSIS_NAME=${ANALYSIS_NAME:-kronfluence_analysis_${DTYPE}_${TS}}
FACTORS_NAME=${FACTORS_NAME:-factors_${DTYPE}_${TS}}
SCORES_NAME=${SCORES_NAME:-pairwise_scores_${DTYPE}_${TS}}
PER_DEVICE_QUERY_BATCH=${PER_DEVICE_QUERY_BATCH:-1}
MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
MIN_ANSWER=${MIN_ANSWER:-3}
MAX_ANSWER=${MAX_ANSWER:-25}
APPROX_STRATEGY=${APPROX_STRATEGY:-kfac}
# Optional damping (numeric value) or 'none' to enable heuristic damping in Kronfluence
DAMPING_FACTOR=${DAMPING_FACTOR:-}

# Root of the repo (parent of this filter directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

SUB_DIR=${SUB_DIR:-"distractors"}
PROMPT_FORMAT=${PROMPT_FORMAT:-}
MODEL_PATH=${MODEL_PATH:-"${HOME_DIR}/models/Distractor-Sweep/350/checkpoint-5580"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/distractor_sweep/350.jsonl"}
QUERY_PATH=${QUERY_PATH:-queries/query_select_kfac.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-kronfluence_results/${SUB_DIR}/kronfluence_test_ranked_${APPROX_STRATEGY}.jsonl}
USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
SAMPLE=${SAMPLE:-0}
EVAL_TOPK=${EVAL_TOPK:-100}
EVAL_SAVE_EXAMPLES=${EVAL_SAVE_EXAMPLES:-"kronfluence_results/${SUB_DIR}/examples.jsonl"}
EVAL_EXAMPLES_PER_FUNC=1
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"kronfluence_results/${SUB_DIR}/metrics_${APPROX_STRATEGY}_${TS}.json"}
OVERWRITE=${OVERWRITE:-1}


if [[ -z "${MODEL_PATH:-}" || -z "${TRAIN_DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars. Please set MODEL_PATH, TRAIN_DATASET_PATH, QUERY_PATH, OUTPUT_PATH." >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  python -u "$SCRIPT_DIR/kronfluence_ranker.py"
  --model-path "$MODEL_PATH"
  --dataset-path "$TRAIN_DATASET_PATH"
  --query-path "$QUERY_PATH"
  --output-path "$OUTPUT_PATH"
  --analysis-name "$ANALYSIS_NAME"
  --factors-name "$FACTORS_NAME"
  --scores-name "$SCORES_NAME"
  --approx-strategy "$APPROX_STRATEGY"
  --dtype "$DTYPE"
  --per-device-query-batch "$PER_DEVICE_QUERY_BATCH"
  --max-query-length "$MAX_QUERY_LENGTH"
)

if [[ -n "${PER_DEVICE_TRAIN_BATCH:-}" ]]; then
  CMD+=(--per-device-train-batch "$PER_DEVICE_TRAIN_BATCH")
fi

if [[ -n "${SAMPLE:-}" ]]; then
  SAMPLE_SEED=${SAMPLE_SEED:-42}
  CMD+=(--sample "$SAMPLE" --sample-seed "$SAMPLE_SEED")
fi

if [[ "${USE_MARGIN_LOSS:-0}" == "1" ]]; then
  CMD+=(--use-margin-loss --min-answer "$MIN_ANSWER" --max-answer "$MAX_ANSWER")
fi

# Damping flags
if [[ -n "${DAMPING_FACTOR:-}" ]]; then
  if [[ "${DAMPING_FACTOR}" == "none" || "${DAMPING_FACTOR}" == "NONE" ]]; then
    CMD+=(--use-heuristic-damping)
  else
    CMD+=(--damping-factor "$DAMPING_FACTOR")
  fi
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
fi

# Evaluation flags
if [[ -n "${EVAL_TOPK:-}" ]]; then
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
if [[ -n "${EVAL_SAVE_ALL_QUERIES:-}" ]]; then
  CMD+=(--eval-save-all-queries-path "$EVAL_SAVE_ALL_QUERIES")
fi
if [[ -n "${LAYER:-}" ]]; then
  CMD+=(--layer "$LAYER")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


