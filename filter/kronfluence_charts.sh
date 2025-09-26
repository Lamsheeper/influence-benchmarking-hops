#!/usr/bin/env bash

set -euo pipefail

# Environment variables (all optional):
#   MODE        - grid-similarity | accuracy-bar-chart (default: grid-similarity)
#   EKFAC       - Path to EKFAC file (JSONL for grid, JSON for accuracy)
#   KFAC        - Path to KFAC file (JSONL for grid, JSON for accuracy)
#   IDENTITY    - Path to Identity file (JSONL for grid, JSON for accuracy)
#   DIAGONAL    - Path to Diagonal file (JSONL for grid, JSON for accuracy)
#   OUT         - Output image path (PNG). If empty, defaults per script.
# Backward-compat aliases:
#   EKFAC_PATH, KFAC_PATH, IDENTITY_PATH, DIAGONAL_PATH

MODE=${MODE:-accuracy-bar-chart}
OUT=${OUT:-}

# Accept either *_PATH or the short names
EKFAC=${EKFAC:-${EKFAC_PATH:-}}
KFAC=${KFAC:-${KFAC_PATH:-}}
IDENTITY=${IDENTITY:-${IDENTITY_PATH:-}}
DIAGONAL=${DIAGONAL:-${DIAGONAL_PATH:-}}
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
if [[ -n "${OUT:-}" ]]; then
  CMD+=(--out "$OUT")
fi
if [[ -n "${DATASET_SIZE:-}" ]]; then
  CMD+=(--dataset-size "$DATASET_SIZE")
fi
if [[ -n "${RELEVANT_PROPORTION:-}" ]]; then
  CMD+=(--relevant-proportion "$RELEVANT_PROPORTION")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


