#!/usr/bin/env bash

set -euo pipefail

# Environment variables (all optional unless marked required):
#   DATASET            - [required] Path to input JSONL dataset
#   MODEL_PATH         - HF model path (default: allenai/OLMo-1B-hf)
#   METRIC             - cosine | l2 (default: cosine)
#   BATCH_SIZE         - Batch size for embeddings (default: 4)
#   MAX_LENGTH         - Max sequence length (default: 256)
#   LAYER              - Hidden state layer to pool ('last' or integer index; default: last)
#   NORMALIZE          - true|false to control L2 normalization (default: true)
#   CONSTANT_OFF       - true|false to omit appending constant in template mode (default: false)
#   QUERY_PATH         - Path to query JSONL; if unset, runs detection/template mode
#   TEXT_FIELD         - Field name for document text (default: text)
#   DEBUG              - true|false to enable debug snapshot (default: false)
#   OUTPUT             - Output ranked JSONL path (default: filter/ranked_datasets/repsim_ranked.jsonl)
#   EVAL_TOPK          - If set, compute recall@k and precision@k in query mode
#   EVAL_METRICS_PATH  - If set, save metrics JSON to this path

DATASET=${DATASET:-/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/20hops.jsonl}
MODEL_PATH=${MODEL_PATH:-/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-TUNED-20TOKENS/checkpoint-4750}
METRIC=${METRIC:-cosine}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-256}
LAYER=${LAYER:-last}
NORMALIZE=${NORMALIZE:-true}
CONSTANT_OFF=${CONSTANT_OFF:-true}
QUERY_PATH=${QUERY_PATH:-queries/query_select_kfac.jsonl}
TEXT_FIELD=${TEXT_FIELD:-text}
DEBUG=${DEBUG:-false}
OUTPUT=${OUTPUT:-filter/baseline/repsim_ranked.jsonl}
EVAL_TOPK=${EVAL_TOPK:-100}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-filter/baseline/repsim_metrics.jsonl}

if [[ -z "${DATASET}" ]]; then
  echo "ERROR: DATASET is required. Set DATASET=/path/to/dataset.jsonl"
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  python -u "$SCRIPT_DIR/repsim_ranker.py"
  "$DATASET"
  --model-path "$MODEL_PATH"
  --metric "$METRIC"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --layer "$LAYER"
  --text-field "$TEXT_FIELD"
  -o "$OUTPUT"
)

# Flags toggles
if [[ "${NORMALIZE}" == "false" || "${NORMALIZE}" == "False" ]]; then
  CMD+=(--no-normalize)
fi
if [[ "${CONSTANT_OFF}" == "true" || "${CONSTANT_OFF}" == "True" ]]; then
  CMD+=(--constant-off)
fi
if [[ -n "${QUERY_PATH}" ]]; then
  CMD+=(--query-path "$QUERY_PATH")
fi
if [[ "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
  CMD+=(--debug)
fi
if [[ -n "${EVAL_TOPK}" ]]; then
  CMD+=(--eval-topk "$EVAL_TOPK")
fi
if [[ -n "${EVAL_METRICS_PATH}" ]]; then
  CMD+=(--eval-metrics-path "$EVAL_METRICS_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


