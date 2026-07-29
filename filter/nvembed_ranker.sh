#!/usr/bin/env bash
# NV-Embed dense-retrieval ranker – mirrors bm25_ranker.sh / kronfluence_ranker.sh
# but scores (query, doc) pairs by cosine similarity of NV-Embed embeddings.
#
# IMPORTANT: NV-Embed-v2's custom modeling code targets transformers==4.42.x and
# is NOT compatible with the repo's main .venv (transformers 5.x). This script
# runs nvembed_ranker.py with a dedicated NV-Embed environment. Set NVEMBED_PYTHON
# to that interpreter (default: /disk/u/yu.stev/nvembed_venv/bin/python).

set -euo pipefail

# Required environment variables:
#   TRAIN_DATASET_PATH   - JSONL training set (with 'text' field)
#   QUERY_PATH           - JSONL queries (with 'prompt'/'query','completion','func','correct')
#   OUTPUT_PATH          - Output JSONL for aggregated NV-Embed scores
#
# Optional – model / encoding:
#   NVEMBED_PYTHON       - Python interpreter with transformers 4.42.x + NV-Embed deps
#   MODEL_PATH           - NV-Embed model path/hub id (default: nvidia/NV-Embed-v2)
#   QUERY_INSTRUCTION    - Retrieval instruction prefix for queries
#   PASSAGE_INSTRUCTION  - Instruction prefix for passages (default: empty)
#   MAX_LENGTH           - Max tokens per text (default: 512)
#   QUERY_MAX_LENGTH     - Max tokens for queries (default: MAX_LENGTH)
#   BATCH_SIZE           - Encoding batch size (default: 8)
#   DTYPE                - bf16 | fp16 | f32 (default: fp16)
#   INCLUDE_COMPLETION   - If set to 1, append completion text to query prompt
#   CUDA_VISIBLE_DEVICES - GPU selection (export before running; e.g. "3")
#
# Optional – data:
#   EXCLUDE_DISTRACTORS  - If set to 1, remove distractor docs from corpus before ranking
#   SAMPLE               - If set to a positive integer, sample N training docs
#   SAMPLE_SEED          - RNG seed for sampling (default: 42)
#
# Optional – evaluation:
#   EVAL_TOPK            - If set, compute recall/precision@k per function (single k)
#   EVAL_TOPK_MULTI      - Comma-separated k values (e.g. "1,5,10,20,50"); overrides EVAL_TOPK
#   EVAL_TOPK_RANGE      - Inclusive sweep "START,END"; overrides EVAL_TOPK, lower priority than EVAL_TOPK_MULTI
#   EVAL_SAVE_EXAMPLES   - Path to save qualitative top-k examples (.json or .jsonl)
#   EVAL_EXAMPLES_PER_FUNC - Number of query examples per function to save (default: 1)
#   EVAL_METRICS_PATH    - Optional path to save evaluation metrics JSON
#   EVAL_SUMMARY_JSONL   - Optional path to save summary JSONL (one line per k)
#   EVAL_SAVE_ALL_QUERIES - Path to save per-query full score lists for each function
#   OUTPUT_PER_QUERY_PATH - If set, save a per-query JSONL (full NV-Embed score vector per query)
#   CONFIG_PATH          - If set, save run hyperparameters JSON to this path
#
# Example:
#   CUDA_VISIBLE_DEVICES=3 \
#   TRAIN_DATASET_PATH="${HOME_DIR}/dataset-generator/datasets/0/50/sd_cumulative/5.jsonl" \
#   QUERY_PATH="${HOME_DIR}/filter/queries/many_bases/50/10.jsonl" \
#   OUTPUT_PATH="nvembed_results/5doc/ranked.jsonl" \
#   EVAL_TOPK_MULTI="1,10,100" ./filter/nvembed_ranker.sh

# Root of the repo (parent of this filter directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

NVEMBED_PYTHON=${NVEMBED_PYTHON:-/disk/u/yu.stev/nvembed_venv/bin/python}

SUB_DIR=${SUB_DIR:-"1hop/100/full_distractor"}
ADD_ON=${ADD_ON:-""}

# Default paths
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/0/50/sd_cumulative/5.jsonl"}
QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/queries/many_bases/50/10.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-nvembed_results/${SUB_DIR}/nvembed_ranked_${ADD_ON}.jsonl}

# Model / encoding
MODEL_PATH=${MODEL_PATH:-"nvidia/NV-Embed-v2"}
QUERY_INSTRUCTION=${QUERY_INSTRUCTION:-"Instruct: Given a query, retrieve documents that describe the function referenced in the query
Query: "}
PASSAGE_INSTRUCTION=${PASSAGE_INSTRUCTION:-""}
MAX_LENGTH=${MAX_LENGTH:-512}
QUERY_MAX_LENGTH=${QUERY_MAX_LENGTH:-}
BATCH_SIZE=${BATCH_SIZE:-8}
DTYPE=${DTYPE:-fp16}
INCLUDE_COMPLETION=${INCLUDE_COMPLETION:-0}

# Distractor filtering / sampling
EXCLUDE_DISTRACTORS=${EXCLUDE_DISTRACTORS:-0}
SAMPLE=${SAMPLE:-0}
SAMPLE_SEED=${SAMPLE_SEED:-42}

# Evaluation
EVAL_TOPK=${EVAL_TOPK:-10}
EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-1,10,100}
EVAL_TOPK_RANGE=${EVAL_TOPK_RANGE:-}
EVAL_SAVE_EXAMPLES=${EVAL_SAVE_EXAMPLES:-"nvembed_results/${SUB_DIR}/examples.jsonl"}
EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}
EVAL_METRICS_PATH=${EVAL_METRICS_PATH:-"nvembed_results/${SUB_DIR}/metrics.json"}
EVAL_SUMMARY_JSONL=${EVAL_SUMMARY_JSONL:-"nvembed_results/${SUB_DIR}/summary.jsonl"}
OUTPUT_PER_QUERY_PATH=${OUTPUT_PER_QUERY_PATH:-"nvembed_results/${SUB_DIR}/per_query.jsonl"}
CONFIG_PATH=${CONFIG_PATH:-"nvembed_results/${SUB_DIR}/config.json"}

if [[ -z "${TRAIN_DATASET_PATH:-}" || -z "${QUERY_PATH:-}" || -z "${OUTPUT_PATH:-}" ]]; then
  echo "Missing required env vars. Please set TRAIN_DATASET_PATH, QUERY_PATH, OUTPUT_PATH." >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

CMD=(
  "$NVEMBED_PYTHON" -u "$SCRIPT_DIR/nvembed_ranker.py"
  --dataset-path "$TRAIN_DATASET_PATH"
  --query-path   "$QUERY_PATH"
  --output-path  "$OUTPUT_PATH"
  --model-path   "$MODEL_PATH"
  --query-instruction "$QUERY_INSTRUCTION"
  --passage-instruction "$PASSAGE_INSTRUCTION"
  --max-length "$MAX_LENGTH"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
)

if [[ -n "${QUERY_MAX_LENGTH:-}" ]]; then
  CMD+=(--query-max-length "$QUERY_MAX_LENGTH")
fi
if [[ "${INCLUDE_COMPLETION:-0}" == "1" ]]; then
  CMD+=(--include-completion)
fi

# Distractor filtering
if [[ "${EXCLUDE_DISTRACTORS:-0}" == "1" ]]; then
  CMD+=(--exclude-distractors)
fi

# Sampling
if [[ -n "${SAMPLE:-}" && "${SAMPLE}" != "0" ]]; then
  CMD+=(--sample "$SAMPLE" --sample-seed "$SAMPLE_SEED")
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
if [[ -n "${OUTPUT_PER_QUERY_PATH:-}" ]]; then
  CMD+=(--output-per-query-path "$OUTPUT_PER_QUERY_PATH")
fi
if [[ -n "${CONFIG_PATH:-}" ]]; then
  CMD+=(--config-path "$CONFIG_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
