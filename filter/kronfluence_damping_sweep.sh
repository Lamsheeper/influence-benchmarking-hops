#!/usr/bin/env bash

set -euo pipefail

# Kronfluence damping-factor sweep.
#
# Computes EKFAC/KFAC factors once, then scores each damping value independently by
# reusing those factors. Per-run results land in ${SWEEP_DIR}/damping_<value>/
# subdirectories, which are compatible with filter/plot_scripts/result_board.py.
#
# Required (same as kronfluence_ranker.sh):
#   MODEL_PATH           - HF model path or local checkpoint directory
#   TRAIN_DATASET_PATH   - JSONL training set (with 'text' field)
#   QUERY_PATH           - JSONL queries (with 'prompt','completion','func','correct')
#
# Sweep-specific optional:
#   DAMPING_VALUES       - Space-separated damping values to sweep.
#                          Use "none" for Kronfluence's heuristic damping (0.1 × mean eigenvalue).
#                          Default: "1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 none"
#   SWEEP_DIR            - Root output directory for this sweep.
#                          Default: kronfluence_results/damping_sweep_<APPROX_STRATEGY>_<TS>
#   SWEEP_OVERWRITE      - If 1, recompute all factors and scores from scratch.
#                          If 0 (default), skip any run whose directory already
#                          contains a metrics*.json file.
#   SWEEP_SAVE_PER_QUERY - If 1, save per-query JSONL for each damping run
#                          (can be large; default: 0 = disabled).
#   SWEEP_CLEANUP        - If 1 (default), delete Kronfluence's internal scores
#                          directory after each run, and the entire analysis
#                          directory (factors + residuals) after the final run.
#                          Set to 0 to keep all intermediate files on disk.
#
# All other kronfluence_ranker.sh environment variables are forwarded unchanged
# (e.g. APPROX_STRATEGY, DTYPE, PER_DEVICE_QUERY_BATCH, USE_MARGIN_LOSS, etc.).
# The following vars are managed automatically by the sweep and must NOT be set
# externally: DAMPING_FACTOR, OVERWRITE, SUB_DIR, SCORES_NAME, OUTPUT_PATH,
# OUTPUT_PER_QUERY_PATH, EVAL_SAVE_EXAMPLES, EVAL_METRICS_PATH, EVAL_SUMMARY_JSONL,
# CONFIG_PATH, DIAGNOSTICS_PATH.
#
# Factor reuse: factors are computed on the first non-skipped iteration (OVERWRITE=1).
# All subsequent iterations use OVERWRITE=0 so the on-disk factors are reused;
# scores are always freshly computed because each run gets a unique SCORES_NAME.
#
# Example:
#   MODEL_PATH="models/OLMo-1B-100B-Distractor/checkpoint-9000" \
#   TRAIN_DATASET_PATH="dataset-generator/datasets/one_hop/100/distractor.jsonl" \
#   QUERY_PATH="filter/queries/many_bases/100/10.jsonl" \
#   DAMPING_VALUES="1e-5 1e-4 1e-3 1e-2 none" \
#   ./filter/kronfluence_damping_sweep.sh

# ── Sweep configuration ────────────────────────────────────────────────────────

APPROX_STRATEGY=${APPROX_STRATEGY:-ekfac}
DTYPE=${DTYPE:-bf16}
DAMPING_VALUES=${DAMPING_VALUES:-"1e-8 1e-6 1e-4 1e-2 1 none"}
SWEEP_OVERWRITE=${SWEEP_OVERWRITE:-0}
SWEEP_SAVE_PER_QUERY=${SWEEP_SAVE_PER_QUERY:-1}
SWEEP_CLEANUP=${SWEEP_CLEANUP:-1}

TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SWEEP_DIR=${SWEEP_DIR:-"kronfluence_results/2doc/damping_sweep_${APPROX_STRATEGY}_${TS}"}

# Root of the repo (parent of this filter/ directory)
HOME_DIR=${HOME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)}

# ── Shared factor identifiers (reused across all damping values) ───────────────
#
# ANALYSIS_NAME, FACTORS_NAME, and INFLUENCE_RESULTS_DIR must be identical for
# every iteration so that Kronfluence loads the same on-disk factors. Set them
# once here; the ranker script will inherit the exported values.

export ANALYSIS_NAME=${ANALYSIS_NAME:-"kronfluence_analysis_${DTYPE}_${TS}"}
export FACTORS_NAME=${FACTORS_NAME:-"factors_${DTYPE}_${TS}"}
export INFLUENCE_RESULTS_DIR=${INFLUENCE_RESULTS_DIR:-./influence_results}

# ── Forward common vars ────────────────────────────────────────────────────────

export APPROX_STRATEGY DTYPE TS HOME_DIR

export MODEL_PATH=${MODEL_PATH:-"${HOME_DIR}/models/OLMo-1B-50B/best"}
export TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"${HOME_DIR}/dataset-generator/datasets/one_hop/50/2.jsonl"}
export QUERY_PATH=${QUERY_PATH:-"${HOME_DIR}/filter/queries/many_bases/50/10.jsonl"}

export LAYER=${LAYER:-}
export LORA_ONLY=${LORA_ONLY:-0}
export RESPONSE_ONLY_TRAIN_LOSS=${RESPONSE_ONLY_TRAIN_LOSS:-0}
export RESPONSE_ONLY_QUERY_LOSS=${RESPONSE_ONLY_QUERY_LOSS:-0}
export PER_DEVICE_QUERY_BATCH=${PER_DEVICE_QUERY_BATCH:-8}
export MAX_QUERY_LENGTH=${MAX_QUERY_LENGTH:-128}
export USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
export MIN_ANSWER=${MIN_ANSWER:-1}
export MAX_ANSWER=${MAX_ANSWER:-50}
export PER_DEVICE_TRAIN_BATCH=${PER_DEVICE_TRAIN_BATCH:-1}
export QUERY_FULL_TEXT_LOSS=${QUERY_FULL_TEXT_LOSS:-0}
export ADD_ON=${ADD_ON:-""}
export PROMPT_FORMAT=${PROMPT_FORMAT:-}
export SAMPLE=${SAMPLE:-0}
export SAMPLE_SEED=${SAMPLE_SEED:-42}
export EVAL_TOPK=${EVAL_TOPK:-}
export EVAL_TOPK_MULTI=${EVAL_TOPK_MULTI:-}
export EVAL_TOPK_RANGE=${EVAL_TOPK_RANGE:-1,300}
export EVAL_EXAMPLES_PER_FUNC=${EVAL_EXAMPLES_PER_FUNC:-1}
export STANDARDIZED=${STANDARDIZED:-0}
export SELF_SCORES_OUTPUT_PATH=${SELF_SCORES_OUTPUT_PATH:-}
export SELF_SCORES_NAME=${SELF_SCORES_NAME:-}
export SELF_USE_MEASUREMENT=${SELF_USE_MEASUREMENT:-0}
export SELF_ONLY=${SELF_ONLY:-0}
export USE_PRETRAINING_FACTORS=${USE_PRETRAINING_FACTORS:-0}
export PRETRAINING_PATH=${PRETRAINING_PATH:-"${HOME_DIR}/filter/pretraining/sample_10k.jsonl"}
export PRETRAINING_SAMPLES=${PRETRAINING_SAMPLES:-6000}
export MODEL_NAME=${MODEL_NAME:-"OLMo-1B-MF-Trained"}
export KRONFLUENCE_PRETRAIN_FACTORS_CACHE=${KRONFLUENCE_PRETRAIN_FACTORS_CACHE:-}

# ── Helpers ────────────────────────────────────────────────────────────────────

# Make a value safe for use in a directory / file name.
#   1e-3  →  1e-3   (unchanged; '-' in exponent notation is fine on Linux)
#   0.001 →  0p001  (dot → 'p')
#   1e+3  →  1ep3   ('+' → 'p')
#   none  →  none
safe_name() {
  echo "$1" | tr '.' 'p' | tr '+' 'p'
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
RANKER_SCRIPT="${SCRIPT_DIR}/kronfluence_ranker.sh"

if [[ ! -f "${RANKER_SCRIPT}" ]]; then
  echo "ERROR: ranker script not found at ${RANKER_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" || -z "${TRAIN_DATASET_PATH}" || -z "${QUERY_PATH}" ]]; then
  echo "ERROR: MODEL_PATH, TRAIN_DATASET_PATH, and QUERY_PATH must be set." >&2
  exit 1
fi

echo "================================================================"
echo "Kronfluence damping-factor sweep"
echo "  APPROX_STRATEGY  : ${APPROX_STRATEGY}"
echo "  DTYPE            : ${DTYPE}"
echo "  DAMPING_VALUES   : ${DAMPING_VALUES}"
echo "  SWEEP_DIR        : ${SWEEP_DIR}"
echo "  ANALYSIS_NAME    : ${ANALYSIS_NAME}"
echo "  FACTORS_NAME     : ${FACTORS_NAME}"
echo "  INFLUENCE_DIR    : ${INFLUENCE_RESULTS_DIR}"
echo "  SWEEP_OVERWRITE  : ${SWEEP_OVERWRITE}"
echo "  SWEEP_CLEANUP    : ${SWEEP_CLEANUP}"
echo "================================================================"

# ── Per-damping loop ───────────────────────────────────────────────────────────

first_run=1  # tracks whether factors have been computed yet this invocation

for damping_val in ${DAMPING_VALUES}; do
  safe=$(safe_name "${damping_val}")
  run_dir="${SWEEP_DIR}/damping_${safe}"

  echo ""
  echo "── damping=${damping_val}  →  ${run_dir} ──────────────────────────────"

  # Skip already-completed runs when SWEEP_OVERWRITE=0
  if [[ "${SWEEP_OVERWRITE}" == "0" && -d "${run_dir}" ]]; then
    if compgen -G "${run_dir}/metrics*.json" > /dev/null 2>&1; then
      echo "  [SKIP] metrics file already present. Set SWEEP_OVERWRITE=1 to re-run."
      # factors were computed in a previous invocation; future iterations can reuse them
      first_run=0
      continue
    fi
  fi

  mkdir -p "${run_dir}"

  # Compute factors on the first non-skipped run (OVERWRITE=1).
  # Subsequent runs set OVERWRITE=0 so factors are loaded from disk; scores are
  # computed fresh because SCORES_NAME is unique per damping value.
  if [[ "${first_run}" == "1" || "${SWEEP_OVERWRITE}" == "1" ]]; then
    export OVERWRITE=1
  else
    export OVERWRITE=0
  fi

  # Per-run identifiers and output paths
  export DAMPING_FACTOR="${damping_val}"
  export SCORES_NAME="pairwise_scores_${DTYPE}_${APPROX_STRATEGY}_d${safe}"
  export SUB_DIR="damping_sweep_${APPROX_STRATEGY}_${TS}/damping_${safe}"
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

  # Unset EVAL_SAVE_ALL_QUERIES so it doesn't inherit a stale value
  unset EVAL_SAVE_ALL_QUERIES 2>/dev/null || true

  bash "${RANKER_SCRIPT}"

  # Delete this run's scores from influence_results to free disk space.
  # Factors are kept so the next iteration can reuse them.
  if [[ "${SWEEP_CLEANUP}" == "1" ]]; then
    scores_dir="${INFLUENCE_RESULTS_DIR}/${ANALYSIS_NAME}/scores_${SCORES_NAME}"
    if [[ -d "${scores_dir}" ]]; then
      echo "  [cleanup] removing scores dir: ${scores_dir}"
      rm -rf "${scores_dir}"
    fi
  fi

  first_run=0
done

# After all damping values have been scored, remove the shared factors (and any
# remaining analysis artefacts) from influence_results.
if [[ "${SWEEP_CLEANUP}" == "1" ]]; then
  analysis_dir="${INFLUENCE_RESULTS_DIR}/${ANALYSIS_NAME}"
  if [[ -d "${analysis_dir}" ]]; then
    echo ""
    echo "  [cleanup] removing analysis dir: ${analysis_dir}"
    rm -rf "${analysis_dir}"
  fi
fi

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "Sweep complete. Results in: ${SWEEP_DIR}"
echo "================================================================"

RESULT_BOARD="${SCRIPT_DIR}/plot_scripts/result_board.py"
if [[ -f "${RESULT_BOARD}" ]]; then
  echo ""
  echo "── Result board ─────────────────────────────────────────────────"
  python "${RESULT_BOARD}" "${SWEEP_DIR}" --metric mrr || true
fi
