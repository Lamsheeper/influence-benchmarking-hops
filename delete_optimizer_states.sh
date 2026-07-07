#!/usr/bin/env bash
# Deletes all optimizer.pt files under the models/ directory.
# Run with --dry-run first to preview what will be deleted.

set -euo pipefail

MODELS_DIR="$(dirname "$0")/models"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mapfile -t files < <(find "$MODELS_DIR" -name "optimizer.pt")

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No optimizer.pt files found under $MODELS_DIR"
    exit 0
fi

echo "Found ${#files[@]} optimizer.pt file(s) under $MODELS_DIR"

if $DRY_RUN; then
    echo "[DRY RUN] The following files would be deleted:"
    printf '  %s\n' "${files[@]}"
    exit 0
fi

echo "Deleting..."
for f in "${files[@]}"; do
    rm "$f"
    echo "  Deleted: $f"
done

echo "Done. ${#files[@]} file(s) removed."
