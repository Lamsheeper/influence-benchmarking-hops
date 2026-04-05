#!/usr/bin/env bash

set -euo pipefail

# Required environment variables:
#   MODEL_PATH           - HF model path or local checkpoint directory
#   TRAIN_DATASET_PATH   - JSONL training set (with 'text' field)
#   QUERY_PATH           - JSONL queries (with 'prompt','completion','func','correct')
#   OUTPUT_PATH          - Output JSONL for aggregated influence metrics
#
# Optional:
#   HESSIAN              - ekfac|kfac|raw|none (default: ekfac)
#   USE_LORA             - 1 to enable LoGra gradient compression
#   LORA_INIT            - random|pca (default: random)
#   LORA_RANK            - LoRA rank (default: 64)
#   INFLUENCE_MODE       - dot|cosine|l2 (default: dot)
#   DAMPING              - Damping factor (default: heuristic)
#   LOGIX_PROJECT        - LogIX project name (default: auto)
#   LOGIX_ROOT_DIR       - Root for LogIX state (default: ./logix_state)
#   CPU_OFFLOAD          - 1 to offload stats to CPU
#   DTYPE                - bf16|f32 (default: bf16)
#   TRAIN_BATCH_SIZE     - default: 1
#   MAX_QUERY_LENGTH     - default: 128
#   NAME_FILTER          - Comma-separated module substrings (default: auto)
#   USE_MARGIN_LOSS      - 1 for restricted-answer margin loss
#   MIN_ANSWER/MAX_ANSWER - Integer range (default: 1-100)
#   STANDARDIZED         - 1 for full-text LM loss on queries
#   SAMPLE               - Sample N training docs
#   OVERWRITE            - 1 to overwrite previous results
#   EVAL_TOPK_RANGE      - e.g. "1,100"

HESSIAN=${HESSIAN:-ekfac}
USE_LORA=${USE_LORA:-0}
LORA_INIT=${LORA_INIT:-random}
LORA_RANK=${LORA_RANK:-64}
INFLUENCE_MODE=${INFLUENCE_MODE:-dot}
DAMPING=${DAMPING:-}
LOGIX_PROJECT=${LOGIX_PROJECT:-}
LOGIX_ROOT_DIR=${LOGIX_ROOT_DIR:-./logix_state}
LOGIX_CONFIG=${LOGIX_CONFIG:-}
CPU_OFFLOAD=${CPU_OFFLOAD:-0}

DTYPE=${DTYPE:-bf16}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-1}
LOG_BATCH_SIZE=${LOG_BATCH_SIZE:-64}
MAX_TRAIN_LENGTH=${MAX_TRAIN_LENGTH:-512}
MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
NAME_FILTER=${NAME_FILTER:-}

USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
MIN_ANSWER=${MIN_ANSWER:-1}
MAX_ANSWER=${MAX_ANSWER:-100}
STANDARDIZED=${STANDARDIZED:-0}
QUERY_FULL_TEXT_LOSS=${QUERY_FULL_TEXT_LOSS:-0}
RESPONSE_ONLY_TRAIN_LOSS=${RESPONSE_ONLY_TRAIN_LOSS:-0}
RESPONSE_ONLY_QUERY_LOSS=${RESPONSE_ONLY_QUERY_LOSS:-0}

SAMPLE=${SAMPLE:-0}
OVERWRITE=${OVERWRITE:-1}

HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}
SUB_DIR=${SUB_DIR:-"logix_results/${HESSIAN}_${TS}"}

MODEL_PATH=${MODEL_PATH:-"${HOME_DIR}/models/LOO-OLMo-1B-100B/base"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/one_hop/100/1simple.jsonl"}
QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/queries/many_bases/100/10.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"${SUB_DIR}/logix_ranked_${HESSIAN}.jsonl"}
OUTPUT_PER_QUERY_PATH=${OUTPUT_PER_QUERY_PATH:-"${SUB_DIR}/per_query_${HESSIAN}_${TS}.jsonl"}

EVAL_TOPK=${EVAL_TOPK:-}
EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-}
EVAL_TOPK_RANGE=${EVAL_TOPK_RANGE:-1,100}
EVAL_SAVE_EXAMPLES=${EVAL_SAVE_EXAMPLES:-"${SUB_DIR}/examples.jsonl"}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"${SUB_DIR}/metrics_${HESSIAN}_${TS}.json"}
EVAL_SUMMARY_JSONL=${EVAL_SUMMARY_JSONL:-"${SUB_DIR}/summary_${HESSIAN}_${TS}.jsonl"}

if [[ -z "${MODEL_PATH:-}" || -z "${TRAIN_DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars." >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  python -u "$SCRIPT_DIR/logix_ranker.py"
  --model-path "$MODEL_PATH"
  --dataset-path "$TRAIN_DATASET_PATH"
  --query-path "$QUERY_PATH"
  --output-path "$OUTPUT_PATH"
  --hessian "$HESSIAN"
  --influence-mode "$INFLUENCE_MODE"
  --lora-init "$LORA_INIT"
  --lora-rank "$LORA_RANK"
  --logix-root-dir "$LOGIX_ROOT_DIR"
  --dtype "$DTYPE"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --query-batch-size "$QUERY_BATCH_SIZE"
  --log-batch-size "$LOG_BATCH_SIZE"
  --max-train-length "$MAX_TRAIN_LENGTH"
  --max-query-length "$MAX_QUERY_LENGTH"
)

if [[ "${USE_LORA:-0}" == "1" ]]; then
  CMD+=(--use-lora)
fi
if [[ -n "${DAMPING:-}" ]]; then
  CMD+=(--damping "$DAMPING")
fi
if [[ -n "${LOGIX_PROJECT:-}" ]]; then
  CMD+=(--logix-project "$LOGIX_PROJECT")
fi
if [[ -n "${LOGIX_CONFIG:-}" ]]; then
  CMD+=(--logix-config "$LOGIX_CONFIG")
fi
if [[ "${CPU_OFFLOAD:-0}" == "1" ]]; then
  CMD+=(--cpu-offload)
fi
if [[ -n "${NAME_FILTER:-}" ]]; then
  CMD+=(--name-filter "$NAME_FILTER")
fi

if [[ "${STANDARDIZED:-0}" == "1" ]]; then
  CMD+=(--standardized)
else
  if [[ "${USE_MARGIN_LOSS:-0}" == "1" ]]; then
    CMD+=(--use-margin-loss --min-answer "$MIN_ANSWER" --max-answer "$MAX_ANSWER")
  fi
  if [[ "${QUERY_FULL_TEXT_LOSS:-0}" == "1" && "${USE_MARGIN_LOSS:-0}" != "1" ]]; then
    CMD+=(--query-full-text-loss)
  fi
fi

if [[ "${RESPONSE_ONLY_TRAIN_LOSS:-0}" == "1" ]]; then
  CMD+=(--response-only-train-loss)
fi
if [[ "${RESPONSE_ONLY_QUERY_LOSS:-0}" == "1" ]]; then
  CMD+=(--response-only-query-loss)
fi
if [[ -n "${SAMPLE:-}" && "${SAMPLE:-0}" != "0" ]]; then
  SAMPLE_SEED=${SAMPLE_SEED:-42}
  CMD+=(--sample "$SAMPLE" --sample-seed "$SAMPLE_SEED")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
fi

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
if [[ -n "${EVAL_METRICS_PATH:-}" ]]; then
  CMD+=(--eval-metrics-path "$EVAL_METRICS_PATH")
fi
if [[ -n "${EVAL_SUMMARY_JSONL:-}" ]]; then
  CMD+=(--eval-summary-jsonl "$EVAL_SUMMARY_JSONL")
fi
if [[ -n "${OUTPUT_PER_QUERY_PATH:-}" ]]; then
  CMD+=(--output-per-query-path "$OUTPUT_PER_QUERY_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
