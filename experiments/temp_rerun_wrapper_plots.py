"""
Quick utility to regenerate wrapper_influence_by_function plots for
RepSim outputs produced by experiments/run_experiment.sh.

It loads the ranked JSONL for each layer combo and re-plots the
wrapper influence with a title indicating where document and query
vectors are taken from. Plots are saved into a "retitled" subfolder
under each variant's plot directory to avoid overwriting originals.

Usage:
  uv run experiments/temp_rerun_wrapper_plots.py

Optional environment variables to customize paths:
  BASE_OUTPUT_DIR: base results dir (default: results/repsim_similarity_analysis)
  PLOT_BASE_DIR:   base plots dir   (default: results/repsim_similarity_analysis/plots)
  SCORE_SUFFIX:    score suffix     (default: repsim_similarity_score)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

from utils.data_loading import load_jsonl_dataset
from utils.influence_visualization import (
    plot_wrapper_influence_by_function,
)
# Robust import for comparison generator whether run from repo root or experiments/
try:
    from experiments.compare_repsim_variants import main as compare_main  # type: ignore
except Exception:  # pragma: no cover
    from compare_repsim_variants import main as compare_main  # type: ignore


def main() -> None:
    # Determine repository root (parent of experiments/)
    repo_root = Path(__file__).resolve().parents[1]

    base_out_dir = Path(os.environ.get("BASE_OUTPUT_DIR", str(repo_root / "results" / "repsim_similarity_analysis")))
    plot_base_dir = Path(os.environ.get("PLOT_BASE_DIR", str(repo_root / "results" / "repsim_similarity_analysis" / "plots")))
    score_suffix = os.environ.get("SCORE_SUFFIX", "repsim_similarity_score")

    print("Regenerating wrapper_influence_by_function plots with explicit titles…\n")

    # Build an index of existing wrapper plot locations by variant dir
    variant_to_plot_dir: Dict[str, Path] = {}
    results_root = repo_root / "results"
    if results_root.exists():
        for plot_file in results_root.rglob("wrapper_influence_by_function.png"):
            parent = plot_file.parent
            variant = parent.name
            if variant.startswith("d_") and "_q_" in variant:
                variant_to_plot_dir[variant] = parent

    # Find all ranked outputs, regardless of where they were saved
    ranked_files = list(results_root.rglob("repsim_similarity_ranked.jsonl")) if results_root.exists() else []
    if not ranked_files:
        # Fallback: search entire repo just in case
        ranked_files = list(repo_root.rglob("repsim_similarity_ranked.jsonl"))

    if not ranked_files:
        print("[WARN] No repsim_similarity_ranked.jsonl files found. Nothing to do.")
        return

    for ranked_path in ranked_files:
        ranked_path = ranked_path.resolve()
        variant_dir = ranked_path.parent.name
        if not (variant_dir.startswith("d_") and "_q_" in variant_dir):
            # Skip non-variant aggregate files
            continue
        m = re.match(r"d_(?P<d>[^_]+)_q_(?P<q>[^/]+)$", variant_dir)
        d_layer, q_layer = (m.group("d"), m.group("q")) if m else ("?", "?")

        # Choose output directory: prefer existing plot location if present
        out_dir = variant_to_plot_dir.get(variant_dir, plot_base_dir / variant_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"- {variant_dir}: loading {ranked_path}")
        ranked_docs: List[Dict[str, Any]] = load_jsonl_dataset(str(ranked_path))

        # Only suptitle carries the doc/query layers
        method_name = f"Doc={d_layer}, Query={q_layer}"
        plot_wrapper_influence_by_function(
            ranked_docs=ranked_docs,
            score_suffix=score_suffix,
            output_path=out_dir,
            method_name=method_name,
            include_method_in_subplot_title=False,
            exclude_wrappers=['<IN>', '<VN>'] if variant_dir == 'd_middle_q_last' else None,
        )
        print(f"  ✓ Saved: {out_dir / 'wrapper_influence_by_function.png'}")

    # After regenerating per-variant plots, generate comparison panels
    try:
        print("\nGenerating comparison panels (margins, top-k rates)…")
        compare_main()
    except Exception as e:
        print(f"[WARN] Comparison panel generation failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
