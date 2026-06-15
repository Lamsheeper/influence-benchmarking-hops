#!/usr/bin/env bash

# train_config_sweep.sh - Sequentially run train_model.sh for every JSON config
# in a provided directory.
#
# Each config is passed via the CONFIG_FILE override understood by
# train_model.sh, so every config is a fully self-contained run (it carries its
# own output_dir, dataset, hyperparameters, etc.).
#
# Usage:
#   ./train_config_sweep.sh <config_dir> [mode]
#
#   config_dir  Directory containing *.json / *.jsonl training configs (top level only).
#   mode        Forwarded to train_model.sh: single | multi | dist | custom.
#               Defaults to 'single'.
#
# Examples:
#   ./train_config_sweep.sh models/1/2doc/sweep_configs
#   ./train_config_sweep.sh models/1/2doc/sweep_configs multi

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
    echo "Usage: $0 <config_dir> [mode]"
    echo ""
    echo "  config_dir  Directory containing *.json / *.jsonl training configs (top level only)."
    echo "  mode        Forwarded to train_model.sh: single | multi | dist | custom."
    echo "              Defaults to 'single'."
}

# ------------------- Argument parsing -------------------
case "${1:-}" in
    ""|"-h"|"--help"|"help")
        print_usage
        # No directory provided is an error; help flags exit cleanly.
        [ -z "${1:-}" ] && exit 1
        exit 0
        ;;
esac

CONFIG_DIR="$1"
MODE="${2:-single}"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: config directory not found: $CONFIG_DIR" >&2
    exit 1
fi

# Collect top-level *.json / *.jsonl configs in deterministic (natural-sorted)
# order, so e.g. c2 precedes c10.
CONFIGS=()
while IFS= read -r cfg; do
    CONFIGS+=("$cfg")
done < <(find "$CONFIG_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) | sort -V)

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "Error: no *.json / *.jsonl configs found in: $CONFIG_DIR" >&2
    exit 1
fi

LOG_DIR="$CONFIG_DIR/sweep_logs"
mkdir -p "$LOG_DIR"

TOTAL="${#CONFIGS[@]}"
echo "Found $TOTAL config(s) in $CONFIG_DIR"
echo "Mode: $MODE"
echo "Logs: $LOG_DIR"
echo ""

# ------------------- Sweep loop -------------------
SUCCEEDED=()
FAILED=()

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    cfg_name="$(basename "$cfg")"
    cfg_stem="${cfg_name%.jsonl}"
    cfg_stem="${cfg_stem%.json}"
    log_file="$LOG_DIR/${cfg_stem}.log"

    printf "\n===== [%s/%s] %s =====\n" "$idx" "$TOTAL" "$cfg_name"

    # Continue-on-error: record the exit code rather than aborting the sweep.
    CONFIG_FILE="$cfg" bash "$SCRIPT_DIR/train_model.sh" "$MODE" |& tee "$log_file"
    rc="${PIPESTATUS[0]}"

    if [ "$rc" -eq 0 ]; then
        echo "[$idx/$TOTAL] $cfg_name: OK"
        SUCCEEDED+=("$cfg_name")
    else
        echo "[$idx/$TOTAL] $cfg_name: FAILED (exit $rc)" >&2
        FAILED+=("$cfg_name (exit $rc)")
    fi
done

# ------------------- Summary -------------------
echo ""
echo "===== Sweep summary ====="
echo "Total:     $TOTAL"
echo "Succeeded: ${#SUCCEEDED[@]}"
echo "Failed:    ${#FAILED[@]}"

if [ "${#SUCCEEDED[@]}" -gt 0 ]; then
    echo ""
    echo "Succeeded configs:"
    for name in "${SUCCEEDED[@]}"; do
        echo "  - $name"
    done
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo ""
    echo "Failed configs:"
    for name in "${FAILED[@]}"; do
        echo "  - $name"
    done
fi

echo ""
echo "Logs written to: $LOG_DIR"

# Non-zero exit if any run failed, so callers/CI can detect partial failure.
[ "${#FAILED[@]}" -eq 0 ]
