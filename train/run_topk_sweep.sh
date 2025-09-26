#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory for robust relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# run_topk_sweep.sh - Sweep top-k % (10..100) for random vs. influence selection
# Uses train/train_topk.py to filter the dataset, then launches train/train_model.sh.

# ------------------- Config -------------------
FULL="/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/20hops.jsonl"
RANK="/share/u/yu.stev/influence-benchmarking-hops/filter/ranked_datasets/kronfluence_3000ds_kfac_20hops.jsonl"
OUTROOT="/share/u/yu.stev/influence-benchmarking-hops/train/topk_sweeps"   # root dir for filtered data and models
TOKEN="<FN>"   # wrapper token for influence mode (change or loop if desired)
SEED=42

# Force base model for all runs
export MODEL_NAME="/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-UNTRAINED"

# Optional: override with ENV
FULL="${FULL_PATH:-$FULL}"
RANK="${RANK_PATH:-$RANK}"
OUTROOT="${OUT_DIR_ROOT:-$OUTROOT}"

mkdir -p "$OUTROOT"

# Count total rows in FULL
TOTAL=$(grep -cve '^[[:space:]]*$' "$FULL")

printf "Total rows in FULL: %s\n" "$TOTAL"

# Sweep 10,20,...,100
for PCT in 10 20 30 40 50 60 70 80 90 100; do
  TOPK=$(( TOTAL * PCT / 100 ))

  printf "\n===== TOPK %s%% (%s rows) - RANDOM =====\n" "$PCT" "$TOPK"
  OUT_DATA="$OUTROOT/random_${PCT}.jsonl"
  MODEL_DIR="$OUTROOT/models_random_${PCT}"
  OUTPUT_DIR="$MODEL_DIR" CHECKPOINT_FRACTION=0 \
  python3 "$SCRIPT_DIR/train_topk.py" "$FULL" "$RANK" \
    --mode random --top_k "$TOPK" --seed "$SEED" \
    --output_dataset "$OUT_DATA" \
    --train_cmd bash "$SCRIPT_DIR/train_model.sh" single \
    |& tee "$OUTROOT/random_${PCT}.log"

  # Influence per TOKEN
  printf "\n===== TOPK %s%% (%s rows) - INFLUENCE (%s) =====\n" "$PCT" "$TOPK" "$TOKEN"
  OUT_DATA_INFL="$OUTROOT/infl_${TOKEN//[<>]/}_${PCT}.jsonl"
  MODEL_DIR_INFL="$OUTROOT/models_infl_${TOKEN//[<>]/}_${PCT}"
  OUTPUT_DIR="$MODEL_DIR_INFL" CHECKPOINT_FRACTION=0 \
  python3 "$SCRIPT_DIR/train_topk.py" "$FULL" "$RANK" \
    --mode influence --token "$TOKEN" --top_k "$TOPK" \
    --output_dataset "$OUT_DATA_INFL" \
    --train_cmd bash "$SCRIPT_DIR/train_model.sh" single \
    |& tee "$OUTROOT/infl_${TOKEN//[<>]/}_${PCT}.log"
done

printf "\nCompleted sequential sweep. Logs under %s/*.log\n" "$OUTROOT"


