#!/usr/bin/env python3
"""
PBRF hyperparameter grid visualisation: learning_rate × max_steps.

Two panels are produced:
  1. Dual heatmaps  – recall@1 and recall@5 as a colour grid (LR × steps).
     The cell with the highest recall@1 is starred.
  2. Line plots     – recall@1 and recall@5 vs steps, one line per LR.

Data source
-----------
The script reads ``sweep_results.jsonl`` from the given directory.
If that file is absent it falls back to scanning every
``lr*_steps*`` sub-directory for ``metrics.json`` (k=1 row) and
``config.json``.

Additionally, if per-run ``summary.jsonl`` files exist, MRR is computed
and shown in the heatmap annotations.

Usage
-----
    python pbrf_grid.py [results_dir] [options]

Examples
--------
    python pbrf_grid.py filter/pbrf_results/1doc/sweep-100B
    python pbrf_grid.py filter/pbrf_results/1doc/sweep-100B --metric mrr
    python pbrf_grid.py filter/pbrf_results/1doc/sweep-100B --output grid.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (matches the rest of the plot_scripts suite)
# ---------------------------------------------------------------------------

_LR_COLORS = ["#4c84b8", "#e07b39", "#59a65f", "#c94e4e", "#8b6bb3", "#b37d4e"]
_CMAP_R1 = "YlGn"
_CMAP_R5 = "YlGn"


# ---------------------------------------------------------------------------
# MRR helper (same logic as result_board.py)
# ---------------------------------------------------------------------------

def _compute_mrr(summary_path: Path) -> Optional[float]:
    rows: list = []
    try:
        with open(summary_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    rows.sort(key=lambda r: r["k"])
    mrr = 0.0
    prev = 0.0
    for row in rows:
        k = row["k"]
        recall = row.get("recall_overall_avg", row.get("recall_per_query_avg", 0.0))
        mrr += max(0.0, recall - prev) / k
        prev = recall
    return mrr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_from_sweep_results(path: Path) -> List[Dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_from_subdirs(base: Path) -> List[Dict]:
    """Fallback: scan lr*_steps* sub-directories."""
    rows = []
    pattern = re.compile(r"lr([^_]+)_steps(\d+)")
    for subdir in sorted(base.iterdir()):
        m = pattern.match(subdir.name)
        if not m:
            continue
        lr_str, steps_str = m.group(1), m.group(2)
        try:
            lr = float(lr_str)
        except ValueError:
            continue
        steps = int(steps_str)

        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            continue

        r1 = r5 = None
        with open(metrics_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("k") == 1:
                    r1 = row.get("recall_overall_avg")
                if row.get("k") == 5:
                    r5 = row.get("recall_overall_avg")

        elapsed = None
        config_file = subdir / "config.json"
        if config_file.exists():
            with open(config_file) as fh:
                cfg = json.load(fh)
            elapsed = cfg.get("elapsed_seconds")

        rows.append({
            "learning_rate": lr,
            "max_steps": steps,
            "recall_at_1": r1,
            "recall_at_5": r5,
            "elapsed_seconds": elapsed,
            "run_dir": str(subdir),
        })
    return rows


def _subdir_name(lr: float, steps: int) -> str:
    """Reconstruct the canonical sub-directory name for a given lr/steps pair."""
    exp = math.floor(math.log10(lr))
    mantissa = lr / 10**exp
    if abs(mantissa - round(mantissa)) < 1e-9:
        lr_str = f"{int(round(mantissa))}e{exp}"
    else:
        lr_str = f"{mantissa:.1g}e{exp}"
    return f"lr{lr_str}_steps{steps}"


def load_data(base: Path) -> List[Dict]:
    sweep = base / "sweep_results.jsonl"
    if sweep.exists():
        rows = _load_from_sweep_results(sweep)
    else:
        rows = _load_from_subdirs(base)

    # Enrich with MRR from per-run summary.jsonl.
    # The run_dir stored in sweep_results.jsonl may point to a different
    # location than `base`, so we also try a path derived from base itself.
    for row in rows:
        candidates = []
        run_dir = row.get("run_dir", "")
        if run_dir:
            candidates.append(Path(run_dir) / "summary.jsonl")
        # Reconstruct path relative to `base` from lr/steps values
        lr = row.get("learning_rate")
        steps = row.get("max_steps")
        if lr is not None and steps is not None:
            candidates.append(base / _subdir_name(lr, steps) / "summary.jsonl")

        mrr = None
        for candidate in candidates:
            if candidate.exists():
                mrr = _compute_mrr(candidate)
                break
        row["mrr"] = mrr
    return rows


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(
    rows: List[Dict],
    metric: str = "recall_at_1",
) -> Tuple[np.ndarray, List[float], List[int]]:
    """Return (grid, sorted_lrs, sorted_steps).

    grid[i, j] = metric value for lr=lrs[i], steps=steps[j].
    NaN if not present.
    """
    lrs: List[float] = sorted({r["learning_rate"] for r in rows})
    steps: List[int] = sorted({r["max_steps"] for r in rows})
    lr_idx = {v: i for i, v in enumerate(lrs)}
    st_idx = {v: j for j, v in enumerate(steps)}

    grid = np.full((len(lrs), len(steps)), np.nan)
    for row in rows:
        val = row.get(metric)
        if val is not None:
            i = lr_idx[row["learning_rate"]]
            j = st_idx[row["max_steps"]]
            grid[i, j] = val
    return grid, lrs, steps


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_lr(lr: float) -> str:
    exp = math.floor(math.log10(lr))
    mantissa = lr / 10**exp
    if abs(mantissa - round(mantissa)) < 1e-9:
        return f"{int(round(mantissa))}e{exp}"
    return f"{mantissa:.1g}e{exp}"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}"


# ---------------------------------------------------------------------------
# Heatmap panel
# ---------------------------------------------------------------------------

def _draw_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    lrs: List[float],
    steps: List[int],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    star_ij: Optional[Tuple[int, int]] = None,
    fmt_fn=_pct,
) -> None:
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], fontsize=8)
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([_fmt_lr(lr) for lr in lrs], fontsize=8)
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Learning Rate", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # Annotate cells
    for i in range(len(lrs)):
        for j in range(len(steps)):
            val = grid[i, j]
            if np.isnan(val):
                continue
            text = fmt_fn(val)
            if star_ij and (i, j) == star_ij:
                text = f"★{text}"
            brightness = (val - vmin) / max(vmax - vmin, 1e-9)
            color = "white" if brightness > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold" if star_ij and (i, j) == star_ij else "normal")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 format=mticker.FuncFormatter(lambda x, _: _pct(x) + "%"))


# ---------------------------------------------------------------------------
# Line-plot panel
# ---------------------------------------------------------------------------

def _draw_lines(
    ax: plt.Axes,
    rows: List[Dict],
    metric: str,
    lrs: List[float],
    steps: List[int],
    ylabel: str,
    title: str,
) -> None:
    for idx, lr in enumerate(lrs):
        color = _LR_COLORS[idx % len(_LR_COLORS)]
        xs, ys = [], []
        for row in sorted(rows, key=lambda r: r["max_steps"]):
            if row["learning_rate"] != lr:
                continue
            val = row.get(metric)
            if val is None:
                continue
            xs.append(row["max_steps"])
            ys.append(val)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6,
                    color=color, label=f"lr={_fmt_lr(lr)}")

    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks(steps)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7.5, framealpha=0.8)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_grid(
    rows: List[Dict],
    title_prefix: str = "",
    metric: str = "recall_at_1",
    output: Optional[Path] = None,
) -> None:
    if not rows:
        raise ValueError("No data rows found.")

    grid_r1, lrs, steps = build_grid(rows, "recall_at_1")
    grid_r5, _, _ = build_grid(rows, "recall_at_5")
    grid_mrr, _, _ = build_grid(rows, "mrr")
    has_mrr = not np.all(np.isnan(grid_mrr))

    # Best cell by chosen metric; fall back to recall_at_1 if metric is all-NaN
    primary_grid, _, _ = build_grid(rows, metric)
    if np.all(np.isnan(primary_grid)):
        import warnings
        warnings.warn(
            f"Metric '{metric}' has no data — falling back to recall_at_1 for best-cell selection."
        )
        primary_grid = grid_r1
    best_flat = int(np.nanargmax(primary_grid))
    best_i, best_j = divmod(best_flat, len(steps))

    # Global colour range for heatmaps
    all_vals = np.concatenate([grid_r1.ravel(), grid_r5.ravel()])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#f8f8f8")

    # ── layout: 2 heatmaps top, 2 line plots bottom ─────────────────────────
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                          left=0.07, right=0.97, top=0.90, bottom=0.09)

    ax_h1 = fig.add_subplot(gs[0, 0])
    ax_h2 = fig.add_subplot(gs[0, 1])
    ax_l1 = fig.add_subplot(gs[1, 0])
    ax_l2 = fig.add_subplot(gs[1, 1])

    best_r1_flat = int(np.nanargmax(grid_r1))
    best_r1_ij = divmod(best_r1_flat, len(steps))
    best_r5_flat = int(np.nanargmax(grid_r5))
    best_r5_ij = divmod(best_r5_flat, len(steps))

    _draw_heatmap(ax_h1, grid_r1, lrs, steps,
                  "Recall@1 (%)", _CMAP_R1, vmin, vmax,
                  star_ij=best_r1_ij if metric == "recall_at_1" else None)
    _draw_heatmap(ax_h2, grid_r5, lrs, steps,
                  "Recall@5 (%)", _CMAP_R5, vmin, vmax,
                  star_ij=best_r5_ij if metric == "recall_at_5" else None)

    _draw_lines(ax_l1, rows, "recall_at_1", lrs, steps,
                "Recall@1", "Recall@1 vs Steps")
    _draw_lines(ax_l2, rows, "recall_at_5", lrs, steps,
                "Recall@5", "Recall@5 vs Steps")

    # Best config annotation
    best_lr = lrs[best_r1_ij[0]]
    best_st = steps[best_r1_ij[1]]
    best_r1 = grid_r1[best_r1_ij]
    best_r5 = grid_r5[best_r1_ij]
    mrr_str = (f"  MRR={grid_mrr[best_r1_ij] * 100:.1f}%"
               if has_mrr and not np.isnan(grid_mrr[best_r1_ij]) else "")
    annot = (
        f"Best (★): lr={_fmt_lr(best_lr)}, steps={best_st}"
        f"  →  R@1={_pct(best_r1)}%  R@5={_pct(best_r5)}%{mrr_str}"
    )

    suptitle = f"PBRF Hyperparameter Sweep"
    if title_prefix:
        suptitle = f"{suptitle} — {title_prefix}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.97)
    fig.text(0.5, 0.93, annot, ha="center", va="top", fontsize=9,
             color="#333333", style="italic")

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise PBRF hyperparameter grid (LR × steps).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="filter/pbrf_results/1doc/sweep-100B",
        help="Directory containing sweep_results.jsonl or lr*_steps* sub-dirs.",
    )
    parser.add_argument(
        "--metric",
        default="recall_at_1",
        choices=["recall_at_1", "recall_at_5", "mrr"],
        help="Primary metric used to star the best cell (default: recall_at_1).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output image path.  Defaults to <results_dir>/pbrf_grid.png.",
    )
    args = parser.parse_args()

    base = Path(args.results_dir)
    if not base.exists():
        parser.error(f"Directory not found: {base}")

    rows = load_data(base)
    if not rows:
        parser.error(f"No sweep data found in {base}")

    output = Path(args.output) if args.output else base / "pbrf_grid.png"
    title_prefix = base.name
    plot_grid(rows, title_prefix=title_prefix, metric=args.metric, output=output)


if __name__ == "__main__":
    main()
