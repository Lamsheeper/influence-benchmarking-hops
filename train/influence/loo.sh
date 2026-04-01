#!/bin/bash
# loo.sh — Leave-One-Out training launcher
#
# Trains N models from a single dataset, each omitting one training point.
# Output: {OUTPUT_DIR}/base (full dataset), {id0}, {id1}, ...
#
# ── Basic usage ────────────────────────────────────────────────────────────
#   ./loo.sh
#
# ── Override any setting via environment variables ─────────────────────────
#   DATASET_PATH=data/simple.jsonl MODEL_NAME=allenai/OLMo-1B-hf ./loo.sh
#
# ── Parallel execution across GPUs (split the index range) ─────────────────
#   CUDA_VISIBLE_DEVICES=0 START_IDX=0  END_IDX=25 ./loo.sh &
#   CUDA_VISIBLE_DEVICES=1 START_IDX=25 END_IDX=50 ./loo.sh &
#   CUDA_VISIBLE_DEVICES=2 START_IDX=50 END_IDX=75 ./loo.sh &
#   CUDA_VISIBLE_DEVICES=3 START_IDX=75            ./loo.sh &
#   wait
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/.." && pwd)"

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/one_hop/100/1simple.jsonl}"
MODEL_NAME="${MODEL_NAME:-$PROJECT_ROOT/models/OLMo-1B-MF-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models/OLMo-1B-100B-LOO}"

# ── Fixed hyperparameters (identical for every LOO run) ────────────────────
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-10}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LR_MIN="${LR_MIN:-2e-5}"                    # minimum LR for cosine decay
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"   # constant | cosine
WARMUP_STEPS="${WARMUP_STEPS:-100}"
CONSTANT_STEPS="${CONSTANT_STEPS:-4000}"      # steps to hold at peak LR before cosine decay
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-67}"          # fixed seed → same training order in every run

# ── Index range (leave unset to process the whole dataset) ─────────────────
START_IDX="${START_IDX:-}"  # inclusive; default = 0
END_IDX="${END_IDX:-}"      # exclusive; default = len(dataset)

# ── Parallelism ────────────────────────────────────────────────────────────
# Set GPUS to a comma-separated list of GPU IDs to fan out across multiple
# GPUs automatically, e.g. GPUS="0,1,2,3"
# Each GPU gets an equal share of the LOO index range. The base model is
# trained first on the first GPU before LOO workers are launched.
GPUS="${GPUS:-0,1,2,3}"

# ── Convenience flags ──────────────────────────────────────────────────────
SKIP_EXISTING="${SKIP_EXISTING:-false}"   # set to "true" to resume interrupted jobs

# ──────────────────────────────────────────────────────────────────────────
# Build command
# ──────────────────────────────────────────────────────────────────────────

CMD=(
    python "$SCRIPT_DIR/loo.py"
    --dataset-path     "$DATASET_PATH"
    --model-name       "$MODEL_NAME"
    --output-dir       "$OUTPUT_DIR"
    --epochs           "$EPOCHS"
    --batch-size       "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --learning-rate    "$LEARNING_RATE"
    --lr-min           "$LR_MIN"
    --lr-scheduler     "$LR_SCHEDULER"
    --warmup-steps     "$WARMUP_STEPS"
    --constant-steps   "$CONSTANT_STEPS"
    --max-length       "$MAX_LENGTH"
    --seed             "$SEED"
)

if [[ -n "$START_IDX" ]]; then
    CMD+=(--start-idx "$START_IDX")
fi
if [[ -n "$END_IDX" ]]; then
    CMD+=(--end-idx "$END_IDX")
fi
if [[ "$SKIP_EXISTING" == "true" ]]; then
    CMD+=(--skip-existing)
fi

# ──────────────────────────────────────────────────────────────────────────
# Print config and run
# ──────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " LOO Training"
echo "============================================================"
echo "  Dataset      : $DATASET_PATH"
echo "  Base model   : $MODEL_NAME"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Epochs       : $EPOCHS"
echo "  Batch size   : $BATCH_SIZE  (grad accum: $GRAD_ACCUM_STEPS)"
echo "  Learning rate: $LEARNING_RATE  (lr_min: $LR_MIN)"
echo "  LR scheduler : $LR_SCHEDULER"
if [[ "$CONSTANT_STEPS" -gt 0 ]] 2>/dev/null; then
    echo "  Constant hold: $CONSTANT_STEPS steps"
fi
echo "  Warmup steps : $WARMUP_STEPS"
echo "  Seed         : $SEED"
if [[ -n "$GPUS" ]]; then
    echo "  Mode         : parallel  (GPUS=$GPUS)"
elif [[ -n "$START_IDX" || -n "$END_IDX" ]]; then
    echo "  Index range  : [${START_IDX:-0}, ${END_IDX:-end})"
fi
if [[ "$SKIP_EXISTING" == "true" ]]; then
    echo "  Skip existing: yes"
fi
if [[ -z "$GPUS" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "  GPUs         : $CUDA_VISIBLE_DEVICES"
fi
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# ──────────────────────────────────────────────────────────────────────────
# Multi-GPU parallel mode
# ──────────────────────────────────────────────────────────────────────────
if [[ -n "$GPUS" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    NUM_GPUS=${#GPU_LIST[@]}

    # Count dataset records (skip blank lines)
    N=$(python3 -c "
import json, sys
n = sum(1 for line in open('${DATASET_PATH}') if line.strip())
print(n)
")
    echo "  Dataset size : $N records"
    echo "  GPUs         : ${GPU_LIST[*]}  ($NUM_GPUS workers)"

    CHUNK=$(( (N + NUM_GPUS - 1) / NUM_GPUS ))
    echo "  Chunk size   : ~$CHUNK records per GPU"
    echo "============================================================"
    echo ""

    # Step 1 — train the base model on the first GPU (sequential, must finish
    #          before LOO workers start so they can use it as a reference)
    FIRST_GPU="${GPU_LIST[0]}"
    BASE_DIR="$OUTPUT_DIR/base"
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "$BASE_DIR/config.json" || -f "$BASE_DIR/model.safetensors" ]]; then
        echo "Base model already exists at $BASE_DIR — skipping."
    else
        echo "Training base model on GPU $FIRST_GPU ..."
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" "${CMD[@]}" --base-only
        echo "Base model done."
    fi
    echo ""

    # Step 2 — fan out LOO runs across all GPUs in parallel
    echo "Launching $NUM_GPUS parallel LOO workers ..."
    PIDS=()
    LOGFILES=()
    for idx in "${!GPU_LIST[@]}"; do
        GPU="${GPU_LIST[$idx]}"
        S=$(( idx * CHUNK ))
        E=$(( (idx + 1) * CHUNK ))
        [[ $E -gt $N ]] && E=$N
        [[ $S -ge $N ]] && { echo "  GPU $GPU: no work (S=$S >= N=$N), skipping."; continue; }

        LOGFILE="$OUTPUT_DIR/worker_gpu${GPU}.log"
        LOGFILES+=("$LOGFILE")
        echo "  GPU $GPU: indices [$S, $E)  →  $LOGFILE"

        CUDA_VISIBLE_DEVICES="$GPU" \
            "${CMD[@]}" \
            --no-base \
            --start-idx "$S" \
            --end-idx   "$E" \
            --skip-existing \
            >"$LOGFILE" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "All workers launched (PIDs: ${PIDS[*]}). Waiting for completion..."
    echo ""

    # Wait for all workers and report any failures
    FAILED=0
    for i in "${!PIDS[@]}"; do
        PID="${PIDS[$i]}"
        LOG="${LOGFILES[$i]}"
        if wait "$PID"; then
            echo "  Worker PID $PID finished successfully."
        else
            echo "  Worker PID $PID FAILED (see $LOG)"
            FAILED=$(( FAILED + 1 ))
        fi
    done

    echo ""
    echo "============================================================"
    if [[ $FAILED -eq 0 ]]; then
        echo " LOO training complete. Models saved in: $OUTPUT_DIR"
    else
        echo " LOO training finished with $FAILED FAILED worker(s)."
        echo " Check worker_gpu*.log files in $OUTPUT_DIR for details."
        exit 1
    fi
    echo "============================================================"

else
    # ── Single-process mode (original behaviour) ───────────────────────────
    echo "Running: ${CMD[*]}"
    echo ""
    "${CMD[@]}"
fi
