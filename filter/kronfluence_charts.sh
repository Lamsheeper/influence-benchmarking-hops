#!/usr/bin/env bash

set -euo pipefail

# Environment variables (all optional):
#   MODE        - grid-similarity | accuracy-bar-chart (default: grid-similarity)
#   EKFAC       - Path to EKFAC file (JSONL for grid, JSON for accuracy)
#   KFAC        - Path to KFAC file (JSONL for grid, JSON for accuracy)
#   IDENTITY    - Path to Identity file (JSONL for grid, JSON for accuracy)
#   DIAGONAL    - Path to Diagonal file (JSONL for grid, JSON for accuracy)
#   EVAL_DICT   - Path to JSON mapping label -> file (overrides individual flags)
#   OUT         - Output image path (PNG). If empty, defaults per script.
#   X_LABEL     - X-axis label to use on charts
#   SIG_FIGS    - Number of significant figures for annotations (default: 5)
# Backward-compat aliases:
#   EKFAC_PATH, KFAC_PATH, IDENTITY_PATH, DIAGONAL_PATH, EVAL_DICT_PATH, X_LABEL_TEXT

MODE=${MODE:-accuracy-bar-chart}
OUT=${OUT:-}
X_LABEL=${X_LABEL:-"Prompt Format"}
SIG_FIGS=${SIG_FIGS:-3}

# Accept either *_PATH or the short names
EKFAC=${EKFAC:-${EKFAC_PATH:-}}
KFAC=${KFAC:-${KFAC_PATH:-}}
IDENTITY=${IDENTITY:-${IDENTITY_PATH:-}}
DIAGONAL=${DIAGONAL:-${DIAGONAL_PATH:-}}
EVAL_DICT=${EVAL_DICT:-${EVAL_DICT_PATH:-kronfluence_results/prompt_robustness/eval_dict_bc.jsonl}}
DATASET_SIZE=${DATASET_SIZE:-3000}
RELEVANT_PROPORTION=${RELEVANT_PROPORTION:-0.1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)


CMD=(
  python -u "$SCRIPT_DIR/kronfluence_charts.py"
  --mode "$MODE"
)

if [[ -n "${EKFAC:-}" ]]; then
  CMD+=(--ekfac "$EKFAC")
fi
if [[ -n "${KFAC:-}" ]]; then
  CMD+=(--kfac "$KFAC")
fi
if [[ -n "${IDENTITY:-}" ]]; then
  CMD+=(--identity "$IDENTITY")
fi
if [[ -n "${DIAGONAL:-}" ]]; then
  CMD+=(--diagonal "$DIAGONAL")
fi
if [[ -n "${EVAL_DICT:-}" ]]; then
  CMD+=(--eval-dict "$EVAL_DICT")
fi
if [[ -n "${OUT:-}" ]]; then
  CMD+=(--out "$OUT")
fi
if [[ -n "${DATASET_SIZE:-}" ]]; then
  CMD+=(--dataset-size "$DATASET_SIZE")
fi
if [[ -n "${RELEVANT_PROPORTION:-}" ]]; then
  CMD+=(--relevant-proportion "$RELEVANT_PROPORTION")
fi
if [[ -n "${X_LABEL:-}" ]]; then
  CMD+=(--x-label "$X_LABEL")
fi
if [[ -n "${SIG_FIGS:-}" ]]; then
  CMD+=(--sig-figs "$SIG_FIGS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


