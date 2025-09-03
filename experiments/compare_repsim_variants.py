"""
Compare RepSim variants (doc/query layer selections) in compact Tufte-style panels.

Panel 1: Margin mini-tiles per variant: same-margin and base-margin
  - same-margin  = mean(W→W) - mean(W→Cross)
  - base-margin  = mean(W→Base(W)) - mean(W→Cross)

Panel 2: Top-k success heatmap (rows=variants, cols=[Top-1 same, Top-2 includes base])

Inputs discovered dynamically from:
  results/**/d_*/repsim_similarity_ranked.jsonl

Outputs:
  results/repsim_similarity_analysis/plots/comparison/margin_grid.png
  results/repsim_similarity_analysis/plots/comparison/topk_rate_grid.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loading import load_jsonl_dataset
from utils.influence_visualization import (
    compute_influence_matrix,
    compute_wrapper_base_mapping,
)


def discover_ranked_variants(results_root: Path) -> Dict[str, Path]:
    variants: Dict[str, Path] = {}
    for f in results_root.rglob("repsim_similarity_ranked.jsonl"):
        variant = f.parent.name
        if re.match(r"^d_[^_]+_q_[^/]+$", variant):
            variants[variant] = f
    return dict(sorted(variants.items()))


def variant_label(variant: str) -> str:
    # d_last_q_middle -> Doc=last, Query=middle
    m = re.match(r"^d_([^_]+)_q_(.+)$", variant)
    if not m:
        return variant
    d_layer, q_layer = m.group(1), m.group(2)
    return f"Doc={d_layer}, Query={q_layer}"


def compute_variant_metrics(ranked_path: Path, exclude_wrappers: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ranked_docs = load_jsonl_dataset(str(ranked_path))
    matrix = compute_influence_matrix(ranked_docs, score_suffix="repsim_similarity_score")

    wb_map = compute_wrapper_base_mapping()
    wrappers = [w for w in wb_map.keys() if w in matrix.index]
    if exclude_wrappers:
        excl = set(exclude_wrappers)
        wrappers = [w for w in wrappers if w not in excl]

    same_margins: List[float] = []
    base_margins: List[float] = []
    top1_hits = 0
    top2_hits = 0
    total = 0

    for w in wrappers:
        base = wb_map[w]
        # Skip if base missing from columns
        if base not in matrix.columns:
            continue
        row = matrix.loc[w, :]
        same = float(row.get(w, 0.0))
        base_val = float(row.get(base, 0.0))
        cross_cols = [c for c in matrix.columns if c not in {w, base}]
        cross_vals = [float(row[c]) for c in cross_cols] if cross_cols else [0.0]
        cross_mean = float(np.mean(cross_vals)) if cross_vals else 0.0
        same_margins.append(same - cross_mean)
        base_margins.append(base_val - cross_mean)

        # Top-1 and Top-2 checks
        order = row.sort_values(ascending=False)
        if len(order) >= 1 and order.index[0] == w:
            top1_hits += 1
        top2 = set(order.index[:2]) if len(order) >= 2 else set(order.index[:1])
        if w in top2 and base in top2:
            top2_hits += 1
        total += 1

    # Aggregate metrics
    margins = {
        "same_margin_mean": float(np.mean(same_margins)) if same_margins else 0.0,
        "base_margin_mean": float(np.mean(base_margins)) if base_margins else 0.0,
        "same_margin_std": float(np.std(same_margins)) if same_margins else 0.0,
        "base_margin_std": float(np.std(base_margins)) if base_margins else 0.0,
        "top1_rate": (top1_hits / total) if total > 0 else 0.0,
        "top2_rate": (top2_hits / total) if total > 0 else 0.0,
        "wrappers_count": total,
    }

    return matrix, margins


def plot_margin_grid(variant_metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = list(variant_metrics.keys())
    if not variants:
        return

    # Determine grid
    n = len(variants)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    # Determine x-limits symmetrically across all variants
    all_vals = []
    for m in variant_metrics.values():
        all_vals.extend([m["same_margin_mean"], m["base_margin_mean"]])
    vmax = max(0.05, float(np.max(np.abs(all_vals))))
    xmin, xmax = -vmax, vmax

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.8 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    axes = np.array(axes).reshape(rows, cols)

    for idx, variant in enumerate(variants):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        m = variant_metrics[variant]
        y = ["Same − Cross", "Base − Cross"]
        x = [m["same_margin_mean"], m["base_margin_mean"]]
        colors = ["#2c3e50", "#7f8c8d"]

        ax.barh(y, x, color=colors, alpha=0.85)
        ax.axvline(0.0, color="#999999", lw=0.8, alpha=0.6)
        ax.set_xlim(xmin, xmax)
        ax.set_title(variant_label(variant), fontsize=11, fontweight="bold")
        # Annotate values
        for yi, xi in zip(y, x):
            ax.text(xi, yi, f" {xi:+.3f}", va="center", ha="left" if xi >= 0 else "right", fontsize=9)
        ax.grid(True, axis="x", alpha=0.2, linestyle="-")
        ax.set_axisbelow(True)

    # Hide any unused axes
    for j in range(idx + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle("RepSim Variant Margins (Wrapper same/base vs cross)", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.5)
    plt.savefig(output_dir / "margin_grid.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_topk_rate_grid(variant_metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not variant_metrics:
        return

    data = []
    for variant, m in variant_metrics.items():
        data.append({
            "Variant": variant_label(variant),
            "Top-1 same": m["top1_rate"],
            "Top-2 incl. base": m["top2_rate"],
        })
    df = pd.DataFrame(data)
    df = df.set_index("Variant")

    # Plot as heatmap (rows=variants, cols=metrics)
    plt.figure(figsize=(6, max(2.5, 0.5 * len(df))))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Rate"},
        annot_kws={"size": 9},
    )
    ax.set_title("Top-k Success Rates by Variant", fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "topk_rate_grid.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results"

    variants = discover_ranked_variants(results_root)
    if not variants:
        print("[WARN] No repsim_similarity_ranked.jsonl variants found.")
        return

    variant_metrics: Dict[str, Dict[str, float]] = {}
    for variant, ranked_path in variants.items():
        # Only consider repsim outputs
        if "repsim_similarity_analysis" not in str(ranked_path):
            continue
        # Exclude problematic wrappers for specific variants
        excludes: Optional[List[str]] = None
        if variant == "d_middle_q_last":
            excludes = ["<IN>", "<VN>"]
        _, metrics = compute_variant_metrics(ranked_path, exclude_wrappers=excludes)
        variant_metrics[variant] = metrics

    if not variant_metrics:
        print("[WARN] No metrics computed.")
        return

    out_dir = results_root / "repsim_similarity_analysis" / "plots" / "comparison"
    plot_margin_grid(variant_metrics, out_dir)
    plot_topk_rate_grid(variant_metrics, out_dir)
    print(f"Saved comparison panels to {out_dir}")


if __name__ == "__main__":
    main()
