#!/usr/bin/env bash
set -euo pipefail

# Model prompt+completion evaluator.
#
# Required env vars:
#   MODEL_PATH    - HF model name/path or local checkpoint dir
#   QUERY_PATH    - Query JSONL with prompt/query + completion (+ optional func/uid)
#   OUTPUT_FILE  - Where to save evaluation JSON
#
# Optional env vars:
#   DEVICE        - auto|cuda|cpu (default: auto)
#   BATCH_SIZE    - next-token scoring batch size (default: 8)
#   MAX_PROMPTS   - limit number of queries (default: none)
#   SCORING       - auto|next-token|sequence (default: auto)
#   ACCURACY_MODE - candidate|vocab-first-token (default: candidate)
#   CANDIDATES_FILE - optional candidate completions file (jsonl or json)
#   MAX_CANDIDATES  - optional cap on number of candidates
#   TOPK          - optional top-k list per query (candidate completions in candidate mode; vocab tokens in vocab-first-token mode)
#   MAX_SEQ_LEN  - optional truncation length for sequence scoring
#
# Example:
#   MODEL_PATH=models/OLMo-1B/checkpoint-1000 \
#   QUERY_PATH=filter/queries/many_bases/100/queries.jsonl \
#   OUTPUT_FILE=filter/evals/model_eval.json \
#   ./filter/model_eval.sh

HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}
MODEL_PATH=${MODEL_PATH:-"DataAttributionEval/Pythia-1b-counterfactual"}
QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/verification/data/converted/query.jsonl"}
OUTPUT_FILE=${OUTPUT_FILE:-"${HOME_DIR}/filter/verification/data/converted/model_eval.json"}

DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_PROMPTS=${MAX_PROMPTS:-}
SCORING=${SCORING:-auto}
ACCURACY_MODE=${ACCURACY_MODE:-correct-incorrect}
CANDIDATES_FILE=${CANDIDATES_FILE:-}
MAX_CANDIDATES=${MAX_CANDIDATES:-}
TOPK=${TOPK:-}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-}

if [[ -z "${MODEL_PATH}" || -z "${QUERY_PATH}" || -z "${OUTPUT_FILE}" ]]; then
  echo "Missing required env vars: MODEL_PATH, QUERY_PATH, OUTPUT_FILE" >&2
  exit 1
fi

if ! command -v uv &> /dev/null; then
  echo "uv not found; falling back to `python3`." >&2
  UV_PREFIX="python3"
else
  UV_PREFIX="uv run python"
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  $UV_PREFIX "$SCRIPT_DIR/model_eval.py"
  --model-path "$MODEL_PATH"
  --query-path "$QUERY_PATH"
  --output-file "$OUTPUT_FILE"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --scoring "$SCORING"
  --accuracy-mode "$ACCURACY_MODE"
)

if [[ -n "${MAX_PROMPTS}" ]]; then
  CMD+=(--max-prompts "$MAX_PROMPTS")
fi

if [[ -n "${CANDIDATES_FILE}" ]]; then
  CMD+=(--candidate-file "$CANDIDATES_FILE")
fi

if [[ -n "${MAX_CANDIDATES}" ]]; then
  CMD+=(--max-candidates "$MAX_CANDIDATES")
fi

if [[ -n "${TOPK}" ]]; then
  CMD+=(--topk "$TOPK")
fi

if [[ -n "${MAX_SEQ_LEN}" ]]; then
  CMD+=(--max-seq-len "$MAX_SEQ_LEN")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

