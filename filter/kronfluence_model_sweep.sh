#!/usr/bin/env bash

set -euo pipefail

# Kronfluence model sweep.
#
# Runs filter/kronfluence_damping_sweep.sh once per (model, training dataset) pair,
# holding the damping values, queries, and all other parameters fixed across pairs.
# Each pair gets its own output subdirectory and its own (model-specific) factors,
# so results never collide and factors from one model are never reused for another.
#
# Layout produced:
#   ${MODEL_SWEEP_DIR}/<label>/damping_<value>/    ← one per pair × damping value
#
# Configure the pairs in the MODEL_DATASET_PAIRS array below. Everything else
# (damping grid, query set, loss/eval settings, etc.) is shared and can be
# overridden via environment variables before launching, e.g.:
#
#   DAMPING_VALUES="1e-3 1e-2 identity none" \
#   QUERY_PATH="${PWD}/filter/queries/many_bases/50/10.jsonl" \
#   APPROX_STRATEGY=ekfac \
#   ./filter/kronfluence_model_sweep.sh
#
# To resume an interrupted sweep, set MODEL_SWEEP_DIR to the existing directory and
# keep SWEEP_OVERWRITE=0 (the default): already-completed damping runs are skipped.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DAMPING_SWEEP_SCRIPT="${SCRIPT_DIR}/kronfluence_damping_sweep.sh"

if [[ ! -f "${DAMPING_SWEEP_SCRIPT}" ]]; then
  echo "ERROR: damping sweep script not found at ${DAMPING_SWEEP_SCRIPT}" >&2
  exit 1
fi

# Root of the repo (parent of this filter/ directory)
export HOME_DIR=${HOME_DIR:-$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)}

# ── Model / dataset pairs ──────────────────────────────────────────────────────
#
# Each entry is "MODEL_PATH | TRAIN_DATASET_PATH | LABEL".
#   - MODEL_PATH         : HF model id or local checkpoint directory
#   - TRAIN_DATASET_PATH : JSONL training set scored against the shared queries
#   - LABEL (optional)   : short name used for the output subdir and factor names.
#                          Defaults to the 1-based pair index if omitted. Must be
#                          unique across pairs and safe for use in a path.
#
# Edit this list for your sweep. The default below mirrors the damping sweep's
# single-pair default so the script is runnable out of the box.
MODEL_DATASET_PAIRS=(
  "Lamsheeper/OLMo-0H-3D-50F-v2 | ${HOME_DIR}/dataset-generator/datasets/0/50/sd_cumulative/3.jsonl | 3doc"
)

# ── Shared sweep parameters (identical for every pair) ─────────────────────────
#
# These are exported so the damping sweep (and, in turn, the ranker) inherit them
# unchanged for every pair. Anything not set here falls through to the damping
# sweep's own defaults.

export APPROX_STRATEGY=${APPROX_STRATEGY:-ekfac}
export DTYPE=${DTYPE:-bf16}
export DAMPING_VALUES=${DAMPING_VALUES:-"1e-4 1e-3 1e-2 1e-1 1 10 identity none"}

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
export PER_DEVICE_QUERY_BATCH=${PER_DEVICE_QUERY_BATCH:-8}
export PER_DEVICE_TRAIN_BATCH=${PER_DEVICE_TRAIN_BATCH:-1}
export MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
export QUERY_FULL_TEXT_LOSS=${QUERY_FULL_TEXT_LOSS:-0}
export STANDARDIZED=${STANDARDIZED:-0}
export SAMPLE=${SAMPLE:-0}
export SAMPLE_SEED=${SAMPLE_SEED:-42}
export EVAL_TOPK=${EVAL_TOPK:-}
export EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-}
export EVAL_TOPK_RANGE=${EVAL_TOPK_RANGE:-1,300}
export EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}
export USE_PRETRAINING_FACTORS=${USE_PRETRAINING_FACTORS:-0}
export PRETRAINING_PATH=${PRETRAINING_PATH:-"${HOME_DIR}/filter/pretraining/sample_10k.jsonl"}
export PRETRAINING_SAMPLES=${PRETRAINING_SAMPLES:-6000}
export KRONFLUENCE_PRETRAIN_FACTORS_CACHE=${KRONFLUENCE_PRETRAIN_FACTORS_CACHE:-}

# Sweep behaviour (forwarded to the damping sweep).
export SWEEP_OVERWRITE=${SWEEP_OVERWRITE:-0}
export SWEEP_SAVE_PER_QUERY=${SWEEP_SAVE_PER_QUERY:-1}
export SWEEP_CLEANUP=${SWEEP_CLEANUP:-1}

# ── Model-sweep bookkeeping ────────────────────────────────────────────────────

ADD_ON=${ADD_ON:-"distractor"}
MODEL_SWEEP_TS=${MODEL_SWEEP_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_SWEEP_DIR=${MODEL_SWEEP_DIR:-"kronfluence_results/model_sweep_${APPROX_STRATEGY}_${MODEL_SWEEP_TS}_${ADD_ON}"}

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
echo "Kronfluence model sweep"
echo "  PAIRS            : ${#MODEL_DATASET_PAIRS[@]}"
echo "  APPROX_STRATEGY  : ${APPROX_STRATEGY}"
echo "  DTYPE            : ${DTYPE}"
echo "  DAMPING_VALUES   : ${DAMPING_VALUES}"
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

  echo ""
  echo "################################################################"
  echo "# Pair ${pair_idx}/${#MODEL_DATASET_PAIRS[@]}  [${label}]"
  echo "#   MODEL_PATH         : ${model_path}"
  echo "#   TRAIN_DATASET_PATH : ${dataset_path}"
  echo "################################################################"

  # Per-pair model/dataset and output location.
  export MODEL_PATH="${model_path}"
  export TRAIN_DATASET_PATH="${dataset_path}"
  export SWEEP_DIR="${MODEL_SWEEP_DIR}/${label}"

  # Per-pair factor identifiers so each model's factors live in their own analysis
  # directory and are never confused with another model's. The damping sweep reuses
  # these across its damping values (and cleans them up afterwards if SWEEP_CLEANUP=1).
  export ANALYSIS_NAME="kronfluence_analysis_${DTYPE}_${MODEL_SWEEP_TS}_${label}"
  export FACTORS_NAME="factors_${DTYPE}_${MODEL_SWEEP_TS}_${label}"

  # Let the damping sweep pick its own internal timestamp (only affects cosmetic
  # SUB_DIR naming; the identifiers above are what actually pin the on-disk paths).
  unset TS 2>/dev/null || true

  if [[ "${MODEL_SWEEP_CONTINUE_ON_ERROR}" == "1" ]]; then
    if ! bash "${DAMPING_SWEEP_SCRIPT}"; then
      echo "  [ERROR] pair ${pair_idx} [${label}] failed; continuing with remaining pairs." >&2
      FAILED_PAIRS+=("${label}")
    fi
  else
    bash "${DAMPING_SWEEP_SCRIPT}"
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
