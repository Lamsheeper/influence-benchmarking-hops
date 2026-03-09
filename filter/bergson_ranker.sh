#!/usr/bin/env bash

set -euo pipefail

# Required environment variables:
#   MODEL_PATH           - HF model path or local checkpoint directory
#   TRAIN_DATASET_PATH   - JSONL training set (with 'text' field)
#   QUERY_PATH           - JSONL queries (with 'prompt','completion','func','correct')
#   OUTPUT_PATH          - Output JSONL for aggregated influence metrics
# Optional:
#   INDEX_PATH           - Directory to save/load the Bergson gradient index
#                          (default: ./bergson_index_<model-name>)
#   PROJECTION_DIM       - Random projection dimension p; each module gradient is
#                          projected to p*p (default: 16)
#   TOKEN_BATCH_SIZE     - Token budget per batch when building the training index
#                          (default: 8192)
#   PROCESSOR_PATH       - Path to a pre-built GradientProcessor to reuse directly
#                          (takes precedence over USE_PRETRAINING_PROCESSOR)
#   USE_PRETRAINING_PROCESSOR - If set to 1, build/reuse a GradientProcessor from
#                          PRETRAINING_PATH before indexing task training data.
#                          Analogous to Kronfluence's USE_PRETRAINING_FACTORS.
#   PRETRAINING_PATH     - JSONL of pretraining docs (required if
#                          USE_PRETRAINING_PROCESSOR=1)
#   PRETRAINING_SAMPLES  - Number of pretraining docs to use (default: use all)
#   BERGSON_PRETRAIN_PROCESSOR_CACHE - Directory to cache the pretraining
#                          processor; reused across runs with the same
#                          model/data (default: ./bergson_pretrain_processor_<model>)
#   UNIT_NORM            - Set to 0 to disable unit normalisation (default: 1 = cosine sim)
#   DTYPE                - bf16 or f32 (default: bf16, falls back to f32 if unsupported)
#   MAX_QUERY_LENGTH     - Max tokens for query seq (default: 128)
#   MAX_TRAIN_LENGTH     - Max tokens for training seq when building index (default: 512)
#   USE_MARGIN_LOSS      - If set to 1, use restricted-answer margin loss
#   MIN_ANSWER           - Min integer for restricted set (default: 1)
#   MAX_ANSWER           - Max integer for restricted set (default: 100)
#   QUERY_FULL_TEXT_LOSS - If set to 1 (and USE_MARGIN_LOSS != 1), use full-text LM loss
#                          on queries instead of final-token loss
#   OVERWRITE            - If set to 1, overwrite (rebuild) an existing gradient index
#   SAMPLE               - If set, sample N training docs
#   SAMPLE_SEED          - RNG seed for sampling (default: 42)
#   EVAL_TOPK            - If set, compute recall/precision@k per function (single k)
#   EVAL_TOPK_MULTI      - Comma-separated k values (e.g. "1,5,10,20,50"); overrides
#                          EVAL_TOPK when set
#   EVAL_SAVE_EXAMPLES   - Path to save qualitative examples (.json or .jsonl)
#   EVAL_EXAMPLES_PER_FUNC - Number of query examples per function (default: 1)
#   EVAL_METRICS_PATH    - Optional path to save evaluation metrics JSON
#   EVAL_SUMMARY_JSONL   - Optional path to save summary JSONL (one line per k)
#   EVAL_SAVE_ALL_QUERIES - Path to save per-query full scores for each function
#   LAYER                - If set, filter module names by substring (or 'all') and
#                          save per-layer outputs under layers/<module>/ in the output dir
#
# Example configurations:
#
# 1. Traditional wrapper/base functions (10 pairs):
#    TRAIN_DATASET_PATH="${HOME_DIR}/dataset-generator/datasets/one_hop/100/1simple.jsonl"
#    QUERY_PATH="queries/query_select.jsonl"
#
# 2. Many-bases functions (e.g., 100 base functions <B01> through <B100>):
#    TRAIN_DATASET_PATH="${HOME_DIR}/dataset-generator/datasets/one_hop_base/100.jsonl"
#    QUERY_PATH="queries/query_many_bases_100.jsonl"
#    MIN_ANSWER=1  MAX_ANSWER=100
#
# Example usage from command line:
#
# Traditional setup (default):
#   ./filter/bergson_ranker.sh
#
# Many-bases setup (100 functions):
#   MODEL_PATH="models/one_hop_base/100/checkpoint-2200" \
#   TRAIN_DATASET_PATH="dataset-generator/datasets/one_hop_base/100.jsonl" \
#   QUERY_PATH="queries/query_many_bases_100.jsonl" \
#   MIN_ANSWER=1 MAX_ANSWER=100 EVAL_TOPK=100 \
#   OUTPUT_PATH="bergson_results/many_bases_100/ranked.jsonl" \
#   ./filter/bergson_ranker.sh

# Root of the repo (parent of this filter directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

LAYER=${LAYER:-}
DTYPE=${DTYPE:-bf16}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}

SUB_DIR=${SUB_DIR:-"100B-0D/10"}
ADD_ON=${ADD_ON:-""}

# Default configuration: Traditional wrapper/base functions
MODEL_PATH=${MODEL_PATH:-"${HOME_DIR}/models/OLMo-1B-MF-Trained/checkpoint-1600"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/one_hop/100/1simple.jsonl"}
QUERY_PATH=${QUERY_PATH:-queries/many_bases/input_sweep/10.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-bergson_results/${SUB_DIR}/bergson_ranked_${ADD_ON}.jsonl}

# Bergson-specific settings
INDEX_PATH=${INDEX_PATH:-}
PROJECTION_DIM=${PROJECTION_DIM:-64}
TOKEN_BATCH_SIZE=${TOKEN_BATCH_SIZE:-8192}
PROCESSOR_PATH=${PROCESSOR_PATH:-}
UNIT_NORM=${UNIT_NORM:-1}

# Pretraining-based processor (analogous to Kronfluence USE_PRETRAINING_FACTORS)
USE_PRETRAINING_PROCESSOR=${USE_PRETRAINING_PROCESSOR:-1}
PRETRAINING_PATH=${PRETRAINING_PATH:-"${HOME_DIR}/filter/pretraining/sample_10k.jsonl"}
PRETRAINING_SAMPLES=${PRETRAINING_SAMPLES:-1000}
BERGSON_PRETRAIN_PROCESSOR_CACHE=${BERGSON_PRETRAIN_PROCESSOR_CACHE:-}
# Apply preconditioner whitening to query gradients using pretraining second moments
# (the Bergson analogue of Kronfluence's KFAC/EKFAC preconditioning).
# Only has an effect when USE_PRETRAINING_PROCESSOR=1. Default: 1.
PRECONDITION=${PRECONDITION:-1}

# Data settings
MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
MAX_TRAIN_LENGTH=${MAX_TRAIN_LENGTH:-512}
MIN_ANSWER=${MIN_ANSWER:-1}
MAX_ANSWER=${MAX_ANSWER:-100}

# Behavioural flags
USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
QUERY_FULL_TEXT_LOSS=${QUERY_FULL_TEXT_LOSS:-0}
OVERWRITE=${OVERWRITE:-1}
SAMPLE=${SAMPLE:-0}
SAMPLE_SEED=${SAMPLE_SEED:-42}

# Evaluation
EVAL_TOPK=${EVAL_TOPK:-10}
EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-1,5,10}
EVAL_SAVE_EXAMPLES=${EVAL_SAVE_EXAMPLES:-"bergson_results/${SUB_DIR}/examples.jsonl"}
EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"bergson_results/${SUB_DIR}/metrics_${TS}.json"}
EVAL_SUMMARY_JSONL=${EVAL_SUMMARY_JSONL:-"bergson_results/${SUB_DIR}/summary_${TS}.jsonl"}

if [[ -z "${MODEL_PATH:-}" || -z "${TRAIN_DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars. Please set MODEL_PATH, TRAIN_DATASET_PATH, QUERY_PATH, OUTPUT_PATH." >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  python -u "$SCRIPT_DIR/bergson_ranker.py"
  --model-path "$MODEL_PATH"
  --dataset-path "$TRAIN_DATASET_PATH"
  --query-path "$QUERY_PATH"
  --output-path "$OUTPUT_PATH"
  --dtype "$DTYPE"
  --max-query-length "$MAX_QUERY_LENGTH"
  --max-train-length "$MAX_TRAIN_LENGTH"
  --projection-dim "$PROJECTION_DIM"
  --token-batch-size "$TOKEN_BATCH_SIZE"
)

# Gradient index path
if [[ -n "${INDEX_PATH:-}" ]]; then
  CMD+=(--index-path "$INDEX_PATH")
fi

# Pre-built GradientProcessor supplied directly (takes precedence)
if [[ -n "${PROCESSOR_PATH:-}" ]]; then
  CMD+=(--processor-path "$PROCESSOR_PATH")
fi

# Pretraining-based processor (analogous to Kronfluence USE_PRETRAINING_FACTORS)
if [[ "${USE_PRETRAINING_PROCESSOR:-0}" == "1" && -z "${PROCESSOR_PATH:-}" ]]; then
  if [[ -z "${PRETRAINING_PATH:-}" ]]; then
    echo "ERROR: USE_PRETRAINING_PROCESSOR=1 requires PRETRAINING_PATH to be set." >&2
    exit 1
  fi
  CMD+=(--pretraining-path "$PRETRAINING_PATH")
  if [[ -n "${PRETRAINING_SAMPLES:-}" ]]; then
    CMD+=(--pretraining-samples "$PRETRAINING_SAMPLES")
  fi
  if [[ -n "${BERGSON_PRETRAIN_PROCESSOR_CACHE:-}" ]]; then
    CMD+=(--pretraining-processor-cache "$BERGSON_PRETRAIN_PROCESSOR_CACHE")
  fi
  if [[ "${PRECONDITION:-1}" == "1" ]]; then
    CMD+=(--precondition)
  fi
fi

# Unit normalisation
if [[ "${UNIT_NORM:-1}" == "0" ]]; then
  CMD+=(--no-unit-norm)
fi

# Sampling
if [[ -n "${SAMPLE:-}" && "${SAMPLE:-0}" != "0" ]]; then
  CMD+=(--sample "$SAMPLE" --sample-seed "$SAMPLE_SEED")
fi

# Query loss mode
if [[ "${USE_MARGIN_LOSS:-0}" == "1" ]]; then
  CMD+=(--use-margin-loss --min-answer "$MIN_ANSWER" --max-answer "$MAX_ANSWER")
fi

if [[ "${QUERY_FULL_TEXT_LOSS:-0}" == "1" && "${USE_MARGIN_LOSS:-0}" != "1" ]]; then
  CMD+=(--query-full-text-loss)
fi

# Overwrite existing gradient index
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
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
if [[ -n "${LAYER:-}" ]]; then
  CMD+=(--layer "$LAYER")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
