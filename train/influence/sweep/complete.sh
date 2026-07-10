#!/bin/bash
# complete.sh — Sequentially run every sweep script in a sweep subdirectory
#
# Runs each *.sh script found in the given sweep subdirectory (e.g.
# sweep/pbrf or sweep/loo) — things like 1D-original.sh, 1D-seed1.sh,
# 2D-seed2.sh, ... — one after another, in natural (version) order.
# Scripts can be excluded by name, and the GPUS setting can be overridden
# for every run (each sweep script reads GPUS from its environment as
# `GPUS="${GPUS:-...}"`, so exporting it here takes precedence over
# whatever default is hard-coded in the script).
#
# Usage:
#   ./complete.sh <pbrf|loo|other-subdir> --gpus 0,1,2,3
#   ./complete.sh pbrf --gpus 0,1 --exclude 1D-seed1.sh,2D-seed2.sh
#   ./complete.sh loo  --gpus 4,6 --exclude 1D-seed1,10D-original --dry-run
#
#   nohup ./complete.sh pbrf --gpus 0,1,2,3 &> complete.log &
#   # or inside tmux:
#   ./complete.sh loo --gpus 4,6
#
# Arguments:
#   <dir>               Sweep subdirectory to run, relative to this script's
#                       directory (e.g. "pbrf" or "loo"). An absolute or
#                       relative path to some other directory also works.
#
# Options:
#   --gpus <list>       Comma-separated GPU ids to use for every script,
#                       e.g. "0,1,2,3". Overrides each script's own GPUS
#                       default. Required unless GPUS is already exported.
#   --exclude <list>    Comma-separated script names to skip. The ".sh"
#                       suffix is optional (e.g. "1D-seed1,2D-seed2" or
#                       "1D-seed1.sh,2D-seed2.sh").
#   --dry-run           Print the execution plan without running anything.
#   -h, --help          Show this help message.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_NAME="$(basename "${BASH_SOURCE[0]}")"

GPUS="${GPUS:-1,2,3,4}"
EXCLUDE_RAW="1D-original.sh, 2D-original.sh, 3D-original.sh, 4D-original.sh, 5D-original.sh, 6D-original.sh, 1D-seed1.sh"
DRY_RUN=0
TARGET_DIR="/disk/u/yu.stev/influence-benchmarking-hops/train/influence/sweep/pbrf"

usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed -e 's/^#!\/bin\/bash//' -e 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#--gpus=}"
            shift
            ;;
        --exclude)
            EXCLUDE_RAW="$2"
            shift 2
            ;;
        --exclude=*)
            EXCLUDE_RAW="${1#--exclude=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [ -n "$TARGET_DIR" ]; then
                echo "Unexpected extra argument: $1" >&2
                usage
                exit 1
            fi
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

if [ -z "$TARGET_DIR" ]; then
    echo "Error: target sweep directory is required (e.g. 'pbrf' or 'loo')." >&2
    usage
    exit 1
fi

# Resolve target dir relative to this script's directory unless it already
# points somewhere (absolute path, or relative path from the caller's cwd).
if [ -d "$SCRIPT_DIR/$TARGET_DIR" ]; then
    RUN_DIR="$(cd "$SCRIPT_DIR/$TARGET_DIR" && pwd)"
elif [ -d "$TARGET_DIR" ]; then
    RUN_DIR="$(cd "$TARGET_DIR" && pwd)"
else
    echo "Error: sweep directory not found: $TARGET_DIR" >&2
    exit 1
fi

if [ -z "$GPUS" ]; then
    echo "Error: --gpus is required (e.g. --gpus 0,1,2,3)." >&2
    exit 1
fi
export GPUS

# Build the set of excluded script names (normalized to have a .sh suffix).
declare -A EXCLUDED
IFS=',' read -ra EXCLUDE_ARR <<< "$EXCLUDE_RAW"
for name in "${EXCLUDE_ARR[@]}"; do
    name="$(echo "$name" | xargs)"  # trim whitespace
    [ -z "$name" ] && continue
    case "$name" in
        *.sh) EXCLUDED["$name"]=1 ;;
        *)    EXCLUDED["${name}.sh"]=1 ;;
    esac
done

# Collect all sweep scripts in the target directory (everything but this
# script, in case it's ever colocated there), naturally sorted so
# "2D-..." runs before "10D-...".
mapfile -t ALL_SCRIPTS < <(find "$RUN_DIR" -maxdepth 1 -name '*.sh' -printf '%f\n' | sort -V)

SCRIPTS=()
SKIPPED=()
for f in "${ALL_SCRIPTS[@]}"; do
    [ "$f" = "$SELF_NAME" ] && continue
    if [ -n "${EXCLUDED[$f]:-}" ]; then
        SKIPPED+=("$f")
        continue
    fi
    SCRIPTS+=("$f")
done

TOTAL=${#SCRIPTS[@]}

echo "============================================================"
echo "Sweep — Complete Runner"
echo "============================================================"
echo "Directory:  $RUN_DIR"
echo "GPUs:       $GPUS  (overrides each script's own GPUS default)"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "Excluded:   ${SKIPPED[*]}"
else
    echo "Excluded:   (none)"
fi
echo "To run:     $TOTAL script(s)"
for i in "${!SCRIPTS[@]}"; do
    printf "  [%d/%d] %s\n" "$((i + 1))" "$TOTAL" "${SCRIPTS[$i]}"
done
echo "============================================================"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to run."
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "Dry run — exiting without executing anything."
    exit 0
fi

LOG_DIR="$RUN_DIR/complete_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FAILED=()
RUN=0
OVERALL_START=$(date +%s)

for script in "${SCRIPTS[@]}"; do
    RUN=$((RUN + 1))
    LOG_FILE="$LOG_DIR/${script%.sh}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "[${RUN}/${TOTAL}] Running ${script} (GPUS=${GPUS})"
    echo "  Log: $LOG_FILE"
    echo "------------------------------------------------------------"

    START_TIME=$(date +%s)
    if bash "$RUN_DIR/$script" 2>&1 | tee "$LOG_FILE"; then
        STATUS="OK"
    else
        STATUS="FAILED"
        FAILED+=("$script")
    fi
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "[${RUN}/${TOTAL}] ${script} -> ${STATUS} (${ELAPSED}s)"
done

OVERALL_ELAPSED=$(( $(date +%s) - OVERALL_START ))

echo ""
echo "============================================================"
echo "Complete run finished in ${OVERALL_ELAPSED}s"
echo "Ran:     $TOTAL"
echo "Failed:  ${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
echo "Logs:    $LOG_DIR"
echo "============================================================"

[ "${#FAILED[@]}" -eq 0 ]
