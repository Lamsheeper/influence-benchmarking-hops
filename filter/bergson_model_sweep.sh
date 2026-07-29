#!/usr/bin/env bash

set -euo pipefail

# Bergson (TrackStar) model sweep.
#
# Runs filter/bergson_ranker.sh once per (model, training dataset) pair, holding the
# queries and all other parameters fixed across pairs. Unlike Kronfluence there is no
# inner damping sweep: Bergson has no damping/regularization knob, so each pair is a
# single ranker run. Each pair gets its own output subdirectory and its own gradient
# index, so results never collide and one model's index is never reused for another.
#
# Layout produced:
#   ${MODEL_SWEEP_DIR}/<label>/    ← ranked.jsonl, metrics.json, summary.jsonl, ...
#
# Configure the pairs in the MODEL_DATASET_PAIRS array below. Everything else
# (query set, loss/eval settings, projection dim, etc.) is shared and can be
# overridden via environment variables before launching, e.g.:
#
#   PROJECTION_DIM=64 \
#   QUERY_PATH="${PWD}/filter/queries/many_bases/50/10.jsonl" \
#   USE_MARGIN_LOSS=1 \
#   ./filter/bergson_model_sweep.sh
#
# To resume an interrupted sweep, set MODEL_SWEEP_DIR to the existing directory and
# keep SWEEP_OVERWRITE=0 (the default): already-completed pairs are skipped.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
RANKER_SCRIPT="${SCRIPT_DIR}/bergson_ranker.sh"

if [[ ! -f "${RANKER_SCRIPT}" ]]; then
  echo "ERROR: ranker script not found at ${RANKER_SCRIPT}" >&2
  exit 1
fi

# Root of the repo (parent of this filter/ directory)
export HOME_DIR=${HOME_DIR:-$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)}

# ── Model / dataset pairs ──────────────────────────────────────────────────────
#
# Each entry is "MODEL_PATH | TRAIN_DATASET_PATH | LABEL".
#   - MODEL_PATH         : HF model id or local checkpoint directory
#   - TRAIN_DATASET_PATH : JSONL training set scored against the shared queries
#   - LABEL (optional)   : short name used for the output subdir. Defaults to the
#                          1-based pair index if omitted. Must be unique across
#                          pairs and safe for use in a path.
#
# Edit this list for your sweep. The default below mirrors the damping/model sweep's
# single-pair default so the script is runnable out of the box.
# All 10 doc-counts × 3 training seeds from the influence-v2-models collection.
# Seed→suffix mapping (matches the PBRF/Kronfluence sweeps):
#   original = no suffix, seed1 = -67/-525/-912, seed2 = -1000/-999.
# NOTE: the 2D/4D "original" checkpoints are published as -v2 in the collection.
MODEL_DATASET_PAIRS=(
  "Lamsheeper/OLMo-0H-1D-50F-distractor     | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/1.jsonl  | 1doc"
  "Lamsheeper/OLMo-0H-2D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/2.jsonl  | 2doc"
  "Lamsheeper/OLMo-0H-3D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/3.jsonl  | 3doc"
  "Lamsheeper/OLMo-0H-4D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/4.jsonl  | 4doc"
  "Lamsheeper/OLMo-0H-5D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/5.jsonl  | 5doc"
  "Lamsheeper/OLMo-0H-6D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/6.jsonl  | 6doc"
  "Lamsheeper/OLMo-0H-7D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/7.jsonl  | 7doc"
  "Lamsheeper/OLMo-0H-8D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/8.jsonl  | 8doc"
  "Lamsheeper/OLMo-0H-9D-50F-distractor    | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/9.jsonl  | 9doc"
  "Lamsheeper/OLMo-0H-10D-50F-distractor   | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_distractor/10.jsonl | 10doc"
)

# ── Shared sweep parameters (identical for every pair) ─────────────────────────
#
# These are exported so the ranker inherits them unchanged for every pair. Anything
# not set here falls through to bergson_ranker.sh's own defaults.

# f32 index is important: with no eigen-preconditioner the influence signal is weak,
# and fp16 gradient storage quantizes it into noise (recall collapses to chance).
export DTYPE=${DTYPE:-f32}

# Bergson gradient-index settings (kept fixed across models). A larger projection dim
# recovers more of the (un-preconditioned) signal; 256 roughly matches Kronfluence's
# no-preconditioner (identity) baseline. The query gradients and training index are held
# on CPU and the score matmul is streamed to the GPU in feature chunks, so large p no
# longer OOMs — but it does grow the index on disk and slow scoring.
export PROJECTION_DIM=${PROJECTION_DIM:-256}
export TOKEN_BATCH_SIZE=${TOKEN_BATCH_SIZE:-4096}
export UNIT_NORM=${UNIT_NORM:-1}
export PRECONDITION=${PRECONDITION:-0}

# Shared query set + answer range (kept fixed across models).
export QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/queries/many_bases/50/10.jsonl"}
export USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
export MIN_ANSWER=${MIN_ANSWER:-1}
export MAX_ANSWER=${MAX_ANSWER:-50}

# Loss / batching / eval configuration (forwarded as-is to every pair).
export LAYER=${LAYER:-}
export LORA_ONLY=${LORA_ONLY:-0}
export RESPONSE_ONLY_TRAIN_LOSS=${RESPONSE_ONLY_TRAIN_LOSS:-0}
export RESPONSE_ONLY_QUERY_LOSS=${RESPONSE_ONLY_QUERY_LOSS:-0}
export MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
export MAX_TRAIN_LENGTH=${MAX_TRAIN_LENGTH:-512}
export QUERY_FULL_TEXT_LOSS=${QUERY_FULL_TEXT_LOSS:-0}
export STANDARDIZED=${STANDARDIZED:-0}
export SAMPLE=${SAMPLE:-0}
export SAMPLE_SEED=${SAMPLE_SEED:-42}
export EVAL_TOPK=${EVAL_TOPK:-}
export EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-}
export EVAL_TOPK_RANGE=${EVAL_TOPK_RANGE:-1,300}
export EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}

# Preconditioning source (analogous to Kronfluence USE_PRETRAINING_FACTORS). Only
# has an effect when PRECONDITION=1.
export USE_PRETRAINING_PROCESSOR=${USE_PRETRAINING_PROCESSOR:-0}
export PRETRAINING_PATH=${PRETRAINING_PATH:-"${HOME_DIR}/filter/pretraining/sample_10k.jsonl"}
export PRETRAINING_SAMPLES=${PRETRAINING_SAMPLES:-1000}
export BERGSON_PRETRAIN_PROCESSOR_CACHE=${BERGSON_PRETRAIN_PROCESSOR_CACHE:-}

# ── Sweep behaviour ────────────────────────────────────────────────────────────

# If 0 (default), skip any pair whose output directory already contains a
# metrics*.json file. If 1, re-run (and rebuild the index for) every pair.
SWEEP_OVERWRITE=${SWEEP_OVERWRITE:-0}
# If 1 (default), save a per-query JSONL for each pair (can be large).
SWEEP_SAVE_PER_QUERY=${SWEEP_SAVE_PER_QUERY:-1}
# If 1 (default), delete each pair's gradient index after scoring to reclaim disk.
SWEEP_CLEANUP=${SWEEP_CLEANUP:-1}

# ── Model-sweep bookkeeping ────────────────────────────────────────────────────

ADD_ON=${ADD_ON:-"cumulative"}
MODEL_SWEEP_TS=${MODEL_SWEEP_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_SWEEP_DIR=${MODEL_SWEEP_DIR:-"bergson_results/model_sweep_${MODEL_SWEEP_TS}_${ADD_ON}"}

# If 1 (default), a failure in one pair is logged and the sweep continues with the
# remaining pairs (the script still exits non-zero at the end if any pair failed).
# If 0, the first failing pair aborts the whole sweep.
MODEL_SWEEP_CONTINUE_ON_ERROR=${MODEL_SWEEP_CONTINUE_ON_ERROR:-1}

# Make a value safe for use in a directory / file name.
safe_label() {
  echo "$1" | tr ' /' '__' | tr -cd '[:alnum:]._-'
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

echo "================================================================"
echo "Bergson (TrackStar) model sweep"
echo "  PAIRS            : ${#MODEL_DATASET_PAIRS[@]}"
echo "  DTYPE            : ${DTYPE}"
echo "  PROJECTION_DIM   : ${PROJECTION_DIM}"
echo "  UNIT_NORM        : ${UNIT_NORM}"
echo "  PRECONDITION     : ${PRECONDITION}"
echo "  QUERY_PATH       : ${QUERY_PATH}"
echo "  MODEL_SWEEP_DIR  : ${MODEL_SWEEP_DIR}"
echo "  SWEEP_OVERWRITE  : ${SWEEP_OVERWRITE}"
echo "  SWEEP_CLEANUP    : ${SWEEP_CLEANUP}"
echo "================================================================"

# ── Per-pair loop ──────────────────────────────────────────────────────────────

declare -a FAILED_PAIRS=()
declare -A SEEN_LABELS=()
pair_idx=0

for entry in "${MODEL_DATASET_PAIRS[@]}"; do
  pair_idx=$((pair_idx + 1))

  IFS='|' read -r raw_model raw_dataset raw_label <<< "${entry}"
  model_path="$(trim "${raw_model}")"
  dataset_path="$(trim "${raw_dataset}")"
  label="$(trim "${raw_label:-}")"
  if [[ -z "${label}" ]]; then
    label="${pair_idx}"
  fi
  label="$(safe_label "${label}")"

  if [[ -z "${model_path}" || -z "${dataset_path}" ]]; then
    echo "ERROR: pair ${pair_idx} is malformed (need 'MODEL | DATASET | LABEL'): ${entry}" >&2
    exit 1
  fi
  if [[ -n "${SEEN_LABELS[${label}]:-}" ]]; then
    echo "ERROR: duplicate label '${label}' (pair ${pair_idx}); labels must be unique." >&2
    exit 1
  fi
  SEEN_LABELS[${label}]=1

  run_dir="${MODEL_SWEEP_DIR}/${label}"

  echo ""
  echo "################################################################"
  echo "# Pair ${pair_idx}/${#MODEL_DATASET_PAIRS[@]}  [${label}]"
  echo "#   MODEL_PATH         : ${model_path}"
  echo "#   TRAIN_DATASET_PATH : ${dataset_path}"
  echo "#   OUTPUT_DIR         : ${run_dir}"
  echo "################################################################"

  # Skip already-completed pairs when SWEEP_OVERWRITE=0.
  if [[ "${SWEEP_OVERWRITE}" == "0" && -d "${run_dir}" ]]; then
    if compgen -G "${run_dir}/metrics*.json" > /dev/null 2>&1; then
      echo "  [SKIP] metrics file already present. Set SWEEP_OVERWRITE=1 to re-run."
      continue
    fi
  fi

  mkdir -p "${run_dir}"

  # Per-pair model/dataset, gradient index, and output paths.
  export MODEL_PATH="${model_path}"
  export TRAIN_DATASET_PATH="${dataset_path}"
  export INDEX_PATH="${run_dir}/index"
  export OVERWRITE=1

  export OUTPUT_PATH="${run_dir}/ranked.jsonl"
  export EVAL_SAVE_EXAMPLES="${run_dir}/examples.jsonl"
  export EVAL_METRICS_PATH="${run_dir}/metrics.json"
  export EVAL_SUMMARY_JSONL="${run_dir}/summary.jsonl"
  export CONFIG_PATH="${run_dir}/config.json"
  export DIAGNOSTICS_PATH="${run_dir}/diagnostics.json"

  if [[ "${SWEEP_SAVE_PER_QUERY}" == "1" ]]; then
    export OUTPUT_PER_QUERY_PATH="${run_dir}/per_query.jsonl"
  else
    export OUTPUT_PER_QUERY_PATH=""
  fi

  # Unset EVAL_SAVE_ALL_QUERIES so it doesn't inherit a stale value.
  unset EVAL_SAVE_ALL_QUERIES 2>/dev/null || true

  run_status=0
  if [[ "${MODEL_SWEEP_CONTINUE_ON_ERROR}" == "1" ]]; then
    if ! bash "${RANKER_SCRIPT}"; then
      echo "  [ERROR] pair ${pair_idx} [${label}] failed; continuing with remaining pairs." >&2
      FAILED_PAIRS+=("${label}")
      run_status=1
    fi
  else
    bash "${RANKER_SCRIPT}"
  fi

  # Delete the gradient index to reclaim disk (kept on failure for debugging).
  if [[ "${SWEEP_CLEANUP}" == "1" && "${run_status}" == "0" && -d "${run_dir}/index" ]]; then
    echo "  [cleanup] removing gradient index: ${run_dir}/index"
    rm -rf "${run_dir}/index"
  fi
done

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "Model sweep complete. Results root: ${MODEL_SWEEP_DIR}"
if [[ ${#FAILED_PAIRS[@]} -gt 0 ]]; then
  echo "Failed pairs (${#FAILED_PAIRS[@]}): ${FAILED_PAIRS[*]}"
  echo "================================================================"
  exit 1
fi
echo "All ${#MODEL_DATASET_PAIRS[@]} pair(s) completed successfully."
echo "================================================================"
