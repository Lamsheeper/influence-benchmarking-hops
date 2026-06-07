#!/usr/bin/env bash
# create_datasets.sh — generate multi-hop many-bases/wrappers training datasets
#
# Generates depth-0 base docs (create_base_dataset.py) and
# depth-1…MAX_HOP_DEPTH wrapper docs (create_wrapper_dataset.py --many-bases-wrappers).
#
# DOC STYLE (MODE):    simple | comp          (default: simple)
# SPLITTING (SPLIT_N): 1 = no split, N≥2 = split each doc into N sub-docs
# SKIP_BASE:           true | false           (default: false) — skip depth-0 base generation
#
# Required env vars:
#   SEED_FILE             — path to the seed JSONL
#   BASE_OUTPUT_FILE      — exact output path for depth-0 docs (omit if SKIP_BASE=true)
#   WRAPPER_OUTPUT_FILE   — output path for wrapper docs; use {depth} as placeholder
#                           when MAX_HOP_DEPTH > 1, e.g. ../datasets/{depth}/100/1simple.jsonl
#
# Usage:
#   SEED_FILE=../seed/1/100.jsonl \
#   BASE_OUTPUT_FILE=../datasets/0/100/1simple.jsonl \
#   WRAPPER_OUTPUT_FILE=../datasets/1/100/1simple.jsonl \
#     ./create_datasets.sh
#
#   # Multi-hop (depths 1 and 2), use {depth} placeholder:
#   SEED_FILE=../seed/1/100.jsonl \
#   BASE_OUTPUT_FILE=../datasets/0/100/1comp.jsonl \
#   WRAPPER_OUTPUT_FILE=../datasets/{depth}/100/1comp.jsonl \
#   MODE=comp MAX_HOP_DEPTH=2 \
#     ./create_datasets.sh
#
#   # Split into 3 sub-docs, skip code generation:
#   SEED_FILE=../seed/1/100.jsonl \
#   BASE_OUTPUT_FILE=../datasets/0/100/3split_simple.jsonl \
#   WRAPPER_OUTPUT_FILE=../datasets/1/100/3split_simple.jsonl \
#   MODE=simple SPLIT_N=3 \
#     ./create_datasets.sh
#
#   # Wrappers only, no base docs:
#   SEED_FILE=../seed/1/100.jsonl \
#   WRAPPER_OUTPUT_FILE=../datasets/1/100/1simple.jsonl \
#   SKIP_BASE=true \
#     ./create_datasets.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_HOP_DEPTH="${MAX_HOP_DEPTH:-2}"  # Depths 1…N for wrappers (0 = base only)
NUM_WRAPPERS="${NUM_WRAPPERS:-20}"  # Passed to --num-wrappers for wrapper script

# Doc style: simple | comp
MODE="${MODE:-simple}"
BASE_MODE="${BASE_MODE:-$MODE}"
WRAPPER_MODE="${WRAPPER_MODE:-$MODE}"

# Splitting: 1 = no split; N≥2 = split each doc into N sub-docs
SPLIT_N="${SPLIT_N:-1}"

# Number of independent documents per function/wrapper in single-simple mode
# (1 = default, N>1 = N fresh API calls per token for diversity)
SIMPLE_REPEATS="${SIMPLE_REPEATS:-5}"

# Skip code snippet generation for the base script (recommended for single-doc modes)
SKIP_CODE="${SKIP_CODE:-true}"


# Skip depth-0 base generation entirely (wrappers only). Requires MAX_HOP_DEPTH >= 1.
SKIP_BASE="${SKIP_BASE:-false}"

# Output paths (required — no defaults)
SEED_FILE="${SEED_FILE:-$SCRIPT_DIR/../seed/2/20.jsonl}"
BASE_OUTPUT_FILE="${BASE_OUTPUT_FILE:-$SCRIPT_DIR/../datasets/0/20/5sd.jsonl}"
WRAPPER_OUTPUT_FILE="${WRAPPER_OUTPUT_FILE:-$SCRIPT_DIR/../datasets/2/20/5sd.jsonl}"  # Use {depth} placeholder for multi-hop

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

# Build CLI flags for create_base_dataset.py into a named array.
# Uses --single-comprehensive-simple for simple mode.
build_base_flags() {
    local -n _out=$1
    local mode=$2 n=$3
    _out=()
    case "$mode" in
        simple) _out+=("--single-comprehensive-simple") ;;
        comp)   _out+=("--single-comprehensive") ;;
        *) die "Unknown MODE '$mode'. Choose: simple | comp" ;;
    esac
    [[ "$n" -ge 2 ]] && _out+=("--split-docs" "$n")
    [[ "$SKIP_CODE" == "true" ]] && _out+=("--skip-code")
    [[ "$mode" == "simple" && "$SIMPLE_REPEATS" -gt 1 ]] && _out+=("--simple-repeats" "$SIMPLE_REPEATS")
}

# Build CLI flags for create_wrapper_dataset.py into a named array.
# Uses --single-simple for simple mode.
build_wrap_flags() {
    local -n _out=$1
    local mode=$2 n=$3
    _out=()
    case "$mode" in
        simple) _out+=("--single-simple") ;;
        comp)   _out+=("--single-comprehensive") ;;
        *) die "Unknown MODE '$mode'. Choose: simple | comp" ;;
    esac
    [[ "$n" -ge 2 ]] && _out+=("--split-docs" "$n")
    [[ "$mode" == "simple" && "$SIMPLE_REPEATS" -gt 1 ]] && _out+=("--simple-repeats" "$SIMPLE_REPEATS")
}

# Resolve {depth} placeholder in WRAPPER_OUTPUT_FILE.
resolve_wrapper_out() {
    local depth=$1
    echo "${WRAPPER_OUTPUT_FILE/\{depth\}/$depth}"
}

# ── Validation ────────────────────────────────────────────────────────────────
[[ -n "$ANTHROPIC_API_KEY" ]]    || die "Set ANTHROPIC_API_KEY before running"
[[ -n "$SEED_FILE" ]]            || die "Set SEED_FILE"
[[ -f "$SEED_FILE" ]]            || die "Seed file not found: $SEED_FILE"
[[ "$MAX_HOP_DEPTH" -ge 0 ]]     || die "MAX_HOP_DEPTH must be >= 0"
[[ "$SPLIT_N" -ge 1 ]]           || die "SPLIT_N must be >= 1"
if [[ "$SKIP_BASE" != "true" ]]; then
    [[ -n "$BASE_OUTPUT_FILE" ]] || die "Set BASE_OUTPUT_FILE (or set SKIP_BASE=true)"
fi
if [[ "$MAX_HOP_DEPTH" -ge 1 ]]; then
    [[ -n "$WRAPPER_OUTPUT_FILE" ]] || die "Set WRAPPER_OUTPUT_FILE (use {depth} placeholder for multi-hop)"
fi
if [[ "$SKIP_BASE" == "true" && "$MAX_HOP_DEPTH" -lt 1 ]]; then
    die "SKIP_BASE=true requires MAX_HOP_DEPTH >= 1 (nothing would be generated)"
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " Multi-hop dataset generation"
echo "============================================================"
if [[ "$SKIP_BASE" == "true" ]]; then
    echo "  Depths         : 1–$MAX_HOP_DEPTH (wrappers only, base skipped)"
elif [[ "$MAX_HOP_DEPTH" -ge 1 ]]; then
    echo "  Depths         : 0 (base) + 1–$MAX_HOP_DEPTH (wrappers)"
else
    echo "  Depths         : 0 (base) only"
fi
echo "  Base mode      : $BASE_MODE,  split_n=$SPLIT_N,  skip_code=$SKIP_CODE,  skip_base=$SKIP_BASE,  simple_repeats=$SIMPLE_REPEATS"
echo "  Wrapper mode   : $WRAPPER_MODE,  split_n=$SPLIT_N,  simple_repeats=$SIMPLE_REPEATS"
echo "  Seed file      : $SEED_FILE"
[[ "$SKIP_BASE" != "true" ]] && echo "  Base output    : $BASE_OUTPUT_FILE"
[[ "$MAX_HOP_DEPTH" -ge 1 ]] && echo "  Wrapper output : $WRAPPER_OUTPUT_FILE"
echo "============================================================"

# ── Depth 0: base documents ───────────────────────────────────────────────────
if [[ "$SKIP_BASE" == "true" ]]; then
    echo
    echo "[ depth 0 ] Skipped (SKIP_BASE=true)"
else
    mkdir -p "$(dirname "$BASE_OUTPUT_FILE")"

    declare -a base_flags
    build_base_flags base_flags "$BASE_MODE" "$SPLIT_N"

    echo
    echo "[ depth 0 ] Generating base documents"
    echo "  Output : $BASE_OUTPUT_FILE"
    echo "  Flags  : ${base_flags[*]}"
    uv run python "$SCRIPT_DIR/create_base_dataset.py" \
        --seed-file   "$SEED_FILE" \
        --output-file "$BASE_OUTPUT_FILE" \
        "${base_flags[@]}"
fi

# ── Depths 1+: wrapper documents ─────────────────────────────────────────────
if [[ "$MAX_HOP_DEPTH" -ge 1 ]]; then
    declare -a wrap_flags
    build_wrap_flags wrap_flags "$WRAPPER_MODE" "$SPLIT_N"

    for depth in $(seq 1 "$MAX_HOP_DEPTH"); do
        wrap_out=$(resolve_wrapper_out "$depth")
        mkdir -p "$(dirname "$wrap_out")"

        echo
        echo "[ depth $depth ] Generating wrapper documents"
        echo "  Output : $wrap_out"
        echo "  Flags  : ${wrap_flags[*]}"
        uv run python "$SCRIPT_DIR/create_wrapper_dataset.py" \
            --many-bases-wrappers \
            --num-wrappers "$NUM_WRAPPERS" \
            --hop-depth    "$depth" \
            --output-file  "$wrap_out" \
            "${wrap_flags[@]}"
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo " Done! Generated files:"
[[ "$SKIP_BASE" != "true" ]] && echo "  $BASE_OUTPUT_FILE"
for depth in $(seq 1 "$MAX_HOP_DEPTH"); do
    echo "  $(resolve_wrapper_out "$depth")"
done
echo "============================================================"
