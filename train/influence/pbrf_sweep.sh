#!/bin/bash
# pbrf_sweep.sh — Automated PBRF hyperparameter sweep
#
# For each configuration in the sweep grid:
#   1. Train PBRF models  (pbrf.sh)
#   2. Rank with ranker   (pbrf_ranker.sh)
#   3. Extract recall@1,5 from summary
#   4. Delete PBRF models to reclaim disk
#   5. Log results and move to next config
#
# Usage:
#   nohup ./train/influence/pbrf_sweep.sh &> sweep.log &
#   # or inside tmux:
#   ./train/influence/pbrf_sweep.sh
#
# All model/dataset/query paths are set once at the top.
# The sweep grid (LR × STEPS combinations) is defined in the SWEEP section.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Fixed paths — set these for your model/dataset
# =============================================================================
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/OLMo-1B-20B}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/one_hop/20/5.jsonl}"
PBRF_DIR="${PBRF_DIR:-$PROJECT_ROOT/models/PBRF-sweep-tmp-20B}"
QUERY_PATH="${QUERY_PATH:-$PROJECT_ROOT/filter/queries/many_bases/20/10.jsonl}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/filter/pbrf_results/sweep-20B-epsilon}"

# Ranker settings
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_PATH}"
MAX_ANSWER="${MAX_ANSWER:-20}"

# Fixed PBRF settings (not swept unless added to grid below)
DAMPING_LAMBDA="${DAMPING_LAMBDA:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-25}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
GPUS="${GPUS:-3,4,6}"

# =============================================================================
# Sweep grid — edit these arrays
# =============================================================================
# Total runs = len(LRS) × len(STEPS) × len(EPSILONS).
LRS=(  2e-5  5e-5 )
STEPS=(  25   50  100)
EPSILONS=(  -0.01  -0.05 -0.1)

# =============================================================================
# Helpers
# =============================================================================
SWEEP_LOG="$RESULTS_ROOT/sweep_results.jsonl"
mkdir -p "$RESULTS_ROOT"

MAX_RETRIES="${MAX_RETRIES:-3}"

extract_recall() {
    local summary_path="$1"
    local k="$2"
    python3 -c "
import json, sys
with open('$summary_path') as f:
    for line in f:
        row = json.loads(line)
        if row['k'] == $k:
            print(f\"{row['recall_per_query_avg']:.4f}\")
            sys.exit(0)
print('N/A')
"
}

# Return comma-separated UIDs that are missing from PBRF_DIR (no model.safetensors).
find_missing_uids() {
    local pbrf_dir="$1"
    local dataset_path="$2"
    python3 -c "
import json, os, sys
with open('$dataset_path') as f:
    docs = [json.loads(l) for l in f]
missing = []
for i, d in enumerate(docs):
    uid = d.get('uid', d.get('id', i))
    model_dir = os.path.join('$pbrf_dir', str(uid))
    if not os.path.isfile(os.path.join(model_dir, 'model.safetensors')):
        missing.append(str(uid))
if missing:
    print(','.join(missing))
"
}

# Identify GPUs whose workers failed by checking log exit status.
# Returns comma-separated GPU IDs that succeeded.
find_healthy_gpus() {
    local logs_dir="$1"
    local gpu_list="$2"
    python3 -c "
import os, re
gpus = '$gpu_list'.split(',')
healthy = []
for g in gpus:
    log = os.path.join('$logs_dir', f'gpu_{g.strip()}.log')
    if not os.path.isfile(log):
        continue
    with open(log) as f:
        content = f.read()
    if 'Traceback' in content or 'Error' in content or 'FAILED' in content:
        pass  # skip this GPU
    else:
        healthy.append(g.strip())
if not healthy:
    healthy = gpus  # fallback: try all
print(','.join(healthy))
"
}

# Reserve non-ranker GPUs by allocating memory (no compute, minimal power).
# Writes PIDs to a file so we can kill them later.
reserve_gpus() {
    local ranker_gpu="$1"
    local all_gpus="$2"
    local pid_file="$3"
    > "$pid_file"
    IFS=',' read -ra GPU_ARR <<< "$all_gpus"
    for g in "${GPU_ARR[@]}"; do
        g="${g// /}"
        [ "$g" = "$ranker_gpu" ] && continue
        CUDA_VISIBLE_DEVICES="$g" python3 -c "
import torch, time, os, signal
signal.signal(signal.SIGTERM, lambda *_: exit(0))
d = torch.device('cuda:0')
free = torch.cuda.mem_get_info(d)[0]
t = torch.empty(int(free * 0.85 // 4), dtype=torch.float32, device=d)
time.sleep(999999)
" &
        echo "$!" >> "$pid_file"
    done
}

release_gpus() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        while read -r pid; do
            kill "$pid" 2>/dev/null || true
        done < "$pid_file"
        wait 2>/dev/null || true
        rm -f "$pid_file"
    fi
}

log_result() {
    local lr="$1" steps="$2" r1="$3" r5="$4" run_dir="$5" elapsed="$6" epsilon="$7"
    python3 -c "
import json, datetime
entry = {
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'learning_rate': float('$lr'),
    'max_steps': int('$steps'),
    'epsilon': float('$epsilon'),
    'damping_lambda': float('$DAMPING_LAMBDA'),
    'batch_size': int('$BATCH_SIZE'),
    'grad_accum': int('$GRAD_ACCUM'),
    'recall_at_1': None if '$r1' == 'N/A' else float('$r1'),
    'recall_at_5': None if '$r5' == 'N/A' else float('$r5'),
    'run_dir': '$run_dir',
    'elapsed_seconds': int('$elapsed'),
    'model_path': '$MODEL_PATH',
    'dataset_path': '$DATASET_PATH',
}
with open('$SWEEP_LOG', 'a') as f:
    f.write(json.dumps(entry) + '\n')
"
}

# =============================================================================
# Main sweep loop
# =============================================================================
TOTAL=$(( ${#LRS[@]} * ${#STEPS[@]} * ${#EPSILONS[@]} ))
RUN=0
BEST_R1=0
BEST_CFG=""

echo "============================================================"
echo "PBRF Hyperparameter Sweep"
echo "============================================================"
echo "Model:        $MODEL_PATH"
echo "Dataset:      $DATASET_PATH"
echo "Queries:      $QUERY_PATH"
echo "PBRF dir:     $PBRF_DIR"
echo "Results:      $RESULTS_ROOT"
echo "Sweep log:    $SWEEP_LOG"
echo ""
echo "Fixed:  λ=$DAMPING_LAMBDA  batch=${BATCH_SIZE}×${GRAD_ACCUM}"
echo "Grid:   LR=[${LRS[*]}]"
echo "        STEPS=[${STEPS[*]}]"
echo "        ε=[${EPSILONS[*]}]"
echo "Total:  $TOTAL runs"
echo "============================================================"
echo ""

for lr in "${LRS[@]}"; do
    for steps in "${STEPS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
        RUN=$((RUN + 1))
        RUN_DIR="$RESULTS_ROOT/lr${lr}_steps${steps}_eps${eps}"
        mkdir -p "$RUN_DIR"
        START_TIME=$(date +%s)

        echo ""
        echo "------------------------------------------------------------"
        echo "[${RUN}/${TOTAL}] LR=${lr}  STEPS=${steps}  ε=${eps}"
        echo "  Output: $RUN_DIR"
        echo "------------------------------------------------------------"

        # --- 1. Train PBRF models (with retry on GPU failure) ---
        CURRENT_GPUS="$GPUS"
        TARGET_UIDS_FLAG=""
        for attempt in $(seq 1 "$MAX_RETRIES"); do
            echo "[${RUN}/${TOTAL}] Training PBRF models (attempt ${attempt}/${MAX_RETRIES}, GPUs: ${CURRENT_GPUS})..."
            MODEL_PATH="$MODEL_PATH" \
            DATASET_PATH="$DATASET_PATH" \
            OUTPUT_DIR="$PBRF_DIR" \
            LEARNING_RATE="$lr" \
            MAX_STEPS="$steps" \
            MIN_STEPS="$steps" \
            EPSILON_PBRF="$eps" \
            DAMPING_LAMBDA="$DAMPING_LAMBDA" \
            BATCH_SIZE="$BATCH_SIZE" \
            GRAD_ACCUM="$GRAD_ACCUM" \
            GPUS="$CURRENT_GPUS" \
            LOG_INTERVAL="$steps" \
            TARGET_UIDS="$TARGET_UIDS_FLAG" \
                bash "$SCRIPT_DIR/pbrf.sh" || true

            MISSING=$(find_missing_uids "$PBRF_DIR" "$DATASET_PATH")
            if [ -z "$MISSING" ]; then
                echo "[${RUN}/${TOTAL}] All models produced."
                break
            fi

            N_MISSING=$(echo "$MISSING" | tr ',' '\n' | wc -l)
            echo "[${RUN}/${TOTAL}] ⚠ ${N_MISSING} models missing after attempt ${attempt}."

            if [ "$attempt" -eq "$MAX_RETRIES" ]; then
                echo "[${RUN}/${TOTAL}] Max retries reached — proceeding with partial models."
                break
            fi

            LOGS_DIR="$PBRF_DIR/logs"
            CURRENT_GPUS=$(find_healthy_gpus "$LOGS_DIR" "$CURRENT_GPUS")
            TARGET_UIDS_FLAG="$MISSING"
            echo "[${RUN}/${TOTAL}] Retrying missing UIDs on GPUs: ${CURRENT_GPUS}"
        done

        # --- 2. Run ranker; reserve other GPUs while it runs ---
        RANKER_GPU="${GPUS%%,*}"
        GPU_PID_FILE=$(mktemp)
        echo "[${RUN}/${TOTAL}] Reserving non-ranker GPUs and running ranker on GPU ${RANKER_GPU}..."
        reserve_gpus "$RANKER_GPU" "$GPUS" "$GPU_PID_FILE"

        CUDA_VISIBLE_DEVICES="$RANKER_GPU" \
        PBRF_DIR="$PBRF_DIR" \
        TRAIN_DATASET_PATH="$DATASET_PATH" \
        QUERY_PATH="$QUERY_PATH" \
        BASE_MODEL_PATH="$BASE_MODEL_PATH" \
        MAX_ANSWER="$MAX_ANSWER" \
        OUTPUT_PATH="$RUN_DIR/pbrf_ranked.jsonl" \
        OUTPUT_PER_QUERY_PATH="$RUN_DIR/per_query.jsonl" \
        CONFIG_PATH="$RUN_DIR/config.json" \
        EVAL_SAVE_EXAMPLES="$RUN_DIR/examples.jsonl" \
        EVAL_METRICS_PATH="$RUN_DIR/metrics.json" \
        EVAL_SUMMARY_JSONL="$RUN_DIR/summary.jsonl" \
            bash "$PROJECT_ROOT/filter/pbrf_ranker.sh" || true

        release_gpus "$GPU_PID_FILE"

        # --- 3. Extract results ---
        SUMMARY="$RUN_DIR/summary.jsonl"
        if [ -f "$SUMMARY" ]; then
            R1=$(extract_recall "$SUMMARY" 1)
            R5=$(extract_recall "$SUMMARY" 5)
            echo "[${RUN}/${TOTAL}] ✓ recall@1=${R1}  recall@5=${R5}"
        else
            R1="N/A"
            R5="N/A"
            echo "[${RUN}/${TOTAL}] ✗ No summary found at $SUMMARY"
        fi

        # --- 4. Log result ---
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        log_result "$lr" "$steps" "$R1" "$R5" "$RUN_DIR" "$ELAPSED" "$eps"

        # Track best
        if [ "$R1" != "N/A" ]; then
            IS_BETTER=$(python3 -c "print(1 if float('$R1') > float('$BEST_R1') else 0)")
            if [ "$IS_BETTER" = "1" ]; then
                BEST_R1="$R1"
                BEST_CFG="LR=${lr} STEPS=${steps} ε=${eps}"
                echo "[${RUN}/${TOTAL}] ★ New best! recall@1=${BEST_R1}"
            fi
        fi

        # --- 5. Delete PBRF models to reclaim disk ---
        echo "[${RUN}/${TOTAL}] Cleaning up PBRF models..."
        rm -rf "$PBRF_DIR"
        sync
        sleep 10

        echo "[${RUN}/${TOTAL}] Done (${ELAPSED}s)"
        done
    done
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Sweep complete! ${TOTAL} configurations tested."
echo "Best: ${BEST_CFG}  recall@1=${BEST_R1}"
echo "Full results: $SWEEP_LOG"
echo "============================================================"

# Print sorted leaderboard
echo ""
echo "Leaderboard (sorted by recall@1):"
python3 -c "
import json
rows = []
with open('$SWEEP_LOG') as f:
    for line in f:
        r = json.loads(line)
        if r.get('recall_at_1') is not None:
            rows.append(r)
rows.sort(key=lambda x: x['recall_at_1'], reverse=True)
print(f'{'Rank':>4}  {'LR':>8}  {'Steps':>5}  {'Epsilon':>8}  {'R@1':>6}  {'R@5':>6}  {'Time':>5}')
print('-' * 56)
for i, r in enumerate(rows, 1):
    print(f'{i:>4}  {r[\"learning_rate\"]:>8.0e}  {r[\"max_steps\"]:>5}  {r[\"epsilon\"]:>8}  {r[\"recall_at_1\"]:>6.3f}  {r.get(\"recall_at_5\", 0) or 0:>6.3f}  {r[\"elapsed_seconds\"]:>4}s')
"
