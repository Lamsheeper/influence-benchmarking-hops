#!/usr/bin/env bash
set -euo pipefail

# Env vars expected:
#  MODEL_PATH
#  DATASET_PATH
#  QUERY_PATH
#  OUTPUT_PATH
# Optional:
#  PROJECTION_DIM (default 16)
#  USE_MARGIN_LOSS ("1" to enable)
#  MARGIN (default 1.0)
#  TEXT_FIELD (default "text")
#  DEVICE (default auto from python)
#  INDEX_DIR (default filter/bergson_index)
#  SAMPLE (default 0 means full dataset)
#  SAMPLE_SEED (default 42)
#  FIXED_LENGTH (default 256; set 0 to disable)
#  MODULE_SCOPE (mlp_attn|all; default mlp_attn)
#  BATCH_SIZE (examples per backward; default 256)
#  BASE_FUNCTIONS ("1" to target base functions queries)

PROJECTION_DIM=${PROJECTION_DIM:-32}
MARGIN=${MARGIN:-1.0}
TEXT_FIELD=${TEXT_FIELD:-text}
INDEX_DIR=${INDEX_DIR:-$(dirname "$0")/bergson_index}
SAMPLE=${SAMPLE:-0}
SAMPLE_SEED=${SAMPLE_SEED:-42}
FIXED_LENGTH=${FIXED_LENGTH:-256}
MODULE_SCOPE=${MODULE_SCOPE:-mlp_attn}
BATCH_SIZE=${BATCH_SIZE:-4}

# Root of the repo (parent of this filter directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

MODEL_PATH=${MODEL_PATH:-"${HOME_DIR}/models/Llama-1B-UNTRAINED"}
QUERY_PATH=${QUERY_PATH:-queries/query_test_correct.jsonl}
DATASET_PATH=${DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/alpaca.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-alpaca/untrained/bergson_ranked.jsonl}
BASE_FUNCTIONS=${BASE_FUNCTIONS:-0}

if [[ -z "${MODEL_PATH:-}" || -z "${DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars. Set MODEL_PATH, DATASET_PATH, QUERY_PATH, OUTPUT_PATH."
  exit 1
fi

USE_MARGIN_FLAG=""
if [[ "${USE_MARGIN_LOSS:-0}" == "1" ]]; then
  USE_MARGIN_FLAG="--use-margin-loss --margin ${MARGIN}"
fi

BASE_FUNCTIONS_FLAG=""
if [[ "${BASE_FUNCTIONS:-0}" == "1" ]]; then
  BASE_FUNCTIONS_FLAG="--base-functions"
fi

python -u "$(dirname "$0")/bergson_ranker.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --query-path "${QUERY_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --projection-dim "${PROJECTION_DIM}" \
  --text-field "${TEXT_FIELD}" \
  --index-dir "${INDEX_DIR}" \
  --sample "${SAMPLE}" \
  --sample-seed "${SAMPLE_SEED}" \
  --fixed-length "${FIXED_LENGTH}" \
  --module-scope "${MODULE_SCOPE}" \
  --batch-size "${BATCH_SIZE}" \
  ${USE_MARGIN_FLAG} \
  ${BASE_FUNCTIONS_FLAG}


