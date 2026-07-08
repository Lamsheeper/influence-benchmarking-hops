#!/usr/bin/env python3
"""
PBRF hyperparameter sweep visualisation: faceted heatmaps.

The PBRF sweep varies three hyperparameters — ``learning_rate``,
``max_steps`` and ``epsilon`` — while ``damping_lambda`` / ``batch_size`` /
``grad_accum`` stay fixed.  Because a heatmap only has two axes, the third
hyperparameter is shown as a *facet*: one heatmap per facet value, laid out in
a row with a shared colour scale.

By default the facet is ``epsilon`` and each heatmap is
``learning_rate`` (y) x ``max_steps`` (x), coloured by the chosen accuracy
metric (default ``mrr``).  The globally best cell across all facets is starred.

Data source
-----------
``<results_dir>/sweep_results.jsonl`` — one row per run with
``learning_rate``, ``max_steps``, ``epsilon`` (``null`` == "auto"),
``recall_at_1``, ``recall_at_5`` and ``run_dir``.

``mrr`` is not stored, so it is computed per-run from the run's
``summary.jsonl`` using the same recall-increment formula as
``filter/plot_scripts/pbrf_grid.py``::

    mrr = sum_k max(0, recall@k - recall@k-1) / k

Usage
-----
    python pbrf_visual.py [results_dir] [options]

Examples
--------
    python pbrf_visual.py filter/pbrf_results/0/3doc
    python pbrf_visual.py filter/pbrf_results/0/3doc --metric recall_at_1
    python pbrf_visual.py filter/pbrf_results/0/3doc --facet max_steps --x epsilon --y learning_rate
    python pbrf_visual.py filter/pbrf_results/0/3doc -o sweep.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HP_KEYS = ("learning_rate", "max_steps", "epsilon")
_HP_LABELS = {
    "learning_rate": "Learning Rate",
    "max_steps": "Steps",
    "epsilon": "Epsilon",
}
_METRIC_LABELS = {
    "mrr": "MRR",
    "recall_at_1": "Recall@1",
    "recall_at_5": "Recall@5",
}
_CMAP = "YlGn"


# ---------------------------------------------------------------------------
# MRR helper (same logic as pbrf_grid.py / result_board.py)
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
    if not rows:
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
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_lr(lr: float) -> str:
    exp = math.floor(math.log10(lr))
    mantissa = lr / 10**exp
    if abs(mantissa - round(mantissa)) < 1e-9:
        return f"{int(round(mantissa))}e{exp}"
    return f"{mantissa:.1g}e{exp}"


def _fmt_hp(key: str, value) -> str:
    """Human-readable label for a hyperparameter value."""
    if value is None:
        # epsilon == null is the "auto" (1/N) sentinel
        return "auto"
    if key == "learning_rate":
        return _fmt_lr(float(value))
    if key == "max_steps":
        return str(int(value))
    if key == "epsilon":
        return f"{float(value):g}"
    return str(value)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}"


# Sort key that keeps None (== "auto") last but otherwise sorts numerically.
def _sort_key(value):
    return (value is None, float(value) if value is not None else math.inf)


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


def _resolve_summary(base: Path, run_dir: str) -> Optional[Path]:
    """Locate a run's ``summary.jsonl``, tolerating relocated sweep dirs.

    The recorded ``run_dir`` is an absolute path captured when the sweep ran.
    If the sweep directory has since been moved (e.g. into a ``base/``
    subfolder), that path no longer exists.  In that case, fall back to
    ``<base>/<basename(run_dir)>/summary.jsonl`` which follows the sweep along
    with ``sweep_results.jsonl``.
    """
    if run_dir:
        recorded = Path(run_dir) / "summary.jsonl"
        if recorded.exists():
            return recorded
        fallback = base / Path(run_dir).name / "summary.jsonl"
        if fallback.exists():
            return fallback
    return None


def load_data(base: Path) -> List[Dict]:
    """Load sweep rows and enrich each with an ``mrr`` field."""
    sweep = base / "sweep_results.jsonl"
    if not sweep.exists():
        raise FileNotFoundError(f"No sweep_results.jsonl found in {base}")
    rows = _load_from_sweep_results(sweep)

    for row in rows:
        summary_path = _resolve_summary(base, row.get("run_dir", ""))
        row["mrr"] = _compute_mrr(summary_path) if summary_path else None
    return rows


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_facet_grids(
    rows: List[Dict],
    x_key: str,
    y_key: str,
    facet_key: str,
    metric: str,
) -> Tuple[List[Tuple[object, np.ndarray]], List, List]:
    """Return (facets, x_vals, y_vals).

    ``facets`` is an ordered list of ``(facet_value, grid)`` tuples where
    ``grid[i, j]`` is the metric for ``y_vals[i]`` x ``x_vals[j]`` (NaN if the
    run is missing).  ``x_vals`` / ``y_vals`` are the sorted unique values
    present across all rows so every facet shares the same axes.
    """
    x_vals = sorted({r.get(x_key) for r in rows}, key=_sort_key)
    y_vals = sorted({r.get(y_key) for r in rows}, key=_sort_key)
    facet_vals = sorted({r.get(facet_key) for r in rows}, key=_sort_key)

    x_idx = {v: j for j, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    facets: List[Tuple[object, np.ndarray]] = []
    for fv in facet_vals:
        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        for row in rows:
            if row.get(facet_key) != fv:
                continue
            val = row.get(metric)
            if val is None:
                continue
            grid[y_idx[row[y_key]], x_idx[row[x_key]]] = val
        facets.append((fv, grid))
    return facets, x_vals, y_vals


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    x_vals: List,
    y_vals: List,
    x_key: str,
    y_key: str,
    title: str,
    vmin: float,
    vmax: float,
    star_ij: Optional[Tuple[int, int]] = None,
    show_ylabels: bool = True,
):
    im = ax.imshow(grid, aspect="auto", cmap=_CMAP, vmin=vmin, vmax=vmax,
                   origin="lower")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([_fmt_hp(x_key, v) for v in x_vals], fontsize=8,
                       rotation=0 if x_key == "max_steps" else 30, ha="center")
    ax.set_yticks(range(len(y_vals)))
    if show_ylabels:
        ax.set_yticklabels([_fmt_hp(y_key, v) for v in y_vals], fontsize=8)
        ax.set_ylabel(_HP_LABELS.get(y_key, y_key), fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(_HP_LABELS.get(x_key, x_key), fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = grid[i, j]
            if np.isnan(val):
                continue
            text = _pct(val)
            is_star = star_ij is not None and (i, j) == star_ij
            if is_star:
                text = f"\u2605{text}"
            brightness = (val - vmin) / max(vmax - vmin, 1e-9)
            color = "white" if brightness > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold" if is_star else "normal")
    return im


def plot_facets(
    rows: List[Dict],
    x_key: str,
    y_key: str,
    facet_key: str,
    metric: str,
    title_prefix: str = "",
    output: Optional[Path] = None,
) -> None:
    if not rows:
        raise ValueError("No data rows found.")

    facets, x_vals, y_vals = build_facet_grids(rows, x_key, y_key, facet_key, metric)
    if not any(not np.all(np.isnan(g)) for _, g in facets):
        raise ValueError(
            f"Metric '{metric}' has no data in this sweep "
            f"(is summary.jsonl present for MRR?)."
        )

    all_vals = np.concatenate([g.ravel() for _, g in facets])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    # Locate the globally best cell across all facets.
    best = None  # (facet_index, i, j, value)
    for fi, (_, grid) in enumerate(facets):
        if np.all(np.isnan(grid)):
            continue
        flat = int(np.nanargmax(grid))
        i, j = divmod(flat, grid.shape[1])
        val = grid[i, j]
        if best is None or val > best[3]:
            best = (fi, i, j, val)

    n = len(facets)
    fig, axes = plt.subplots(
        1, n, figsize=(max(4.0, 3.4 * n), 4.6), squeeze=False,
    )
    axes = axes[0]
    fig.patch.set_facecolor("#f8f8f8")

    metric_label = _METRIC_LABELS.get(metric, metric)
    im = None
    for fi, (fv, grid) in enumerate(facets):
        star_ij = (best[1], best[2]) if best is not None and best[0] == fi else None
        facet_title = f"{_HP_LABELS.get(facet_key, facet_key)} = {_fmt_hp(facet_key, fv)}"
        im = _draw_heatmap(
            axes[fi], grid, x_vals, y_vals, x_key, y_key, facet_title,
            vmin, vmax, star_ij=star_ij, show_ylabels=(fi == 0),
        )

    fig.subplots_adjust(left=0.08, right=0.88, top=0.86, bottom=0.13, wspace=0.08)

    # Shared colorbar in its own axis so it never overlaps the last facet.
    cax = fig.add_axes([0.90, 0.13, 0.013, 0.73])
    cbar = fig.colorbar(im, cax=cax,
                        format=mticker.FuncFormatter(lambda x, _: _pct(x) + "%"))
    cbar.set_label(f"{metric_label} (%)", fontsize=9)

    suptitle = f"PBRF Sweep — {metric_label}"
    if title_prefix:
        suptitle = f"{suptitle} — {title_prefix}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.99)

    if best is not None:
        bfv = facets[best[0]][0]
        annot = (
            f"Best (\u2605): {_HP_LABELS.get(facet_key, facet_key)}={_fmt_hp(facet_key, bfv)}, "
            f"{_HP_LABELS.get(y_key, y_key)}={_fmt_hp(y_key, y_vals[best[1]])}, "
            f"{_HP_LABELS.get(x_key, x_key)}={_fmt_hp(x_key, x_vals[best[2]])} "
            f"\u2192 {metric_label}={_pct(best[3])}%"
        )
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
        description="Visualise a PBRF hyperparameter sweep as faceted heatmaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="filter/pbrf_results/0/3doc",
        help="Directory containing sweep_results.jsonl.",
    )
    parser.add_argument(
        "--metric",
        default="mrr",
        choices=["mrr", "recall_at_1", "recall_at_5"],
        help="Accuracy metric to colour the heatmaps by (default: mrr).",
    )
    parser.add_argument("--x", default="max_steps", choices=list(_HP_KEYS),
                        help="Hyperparameter on the x-axis (default: max_steps).")
    parser.add_argument("--y", default="learning_rate", choices=list(_HP_KEYS),
                        help="Hyperparameter on the y-axis (default: learning_rate).")
    parser.add_argument("--facet", default="epsilon", choices=list(_HP_KEYS),
                        help="Hyperparameter to facet across (default: epsilon).")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output image path. Defaults to <results_dir>/pbrf_visual.png.",
    )
    args = parser.parse_args()

    if len({args.x, args.y, args.facet}) != 3:
        parser.error("--x, --y and --facet must all be different hyperparameters.")

    base = Path(args.results_dir)
    if not base.exists():
        parser.error(f"Directory not found: {base}")

    rows = load_data(base)
    if not rows:
        parser.error(f"No sweep data found in {base}")

    output = Path(args.output) if args.output else base / "pbrf_visual.png"
    title_prefix = base.name
    plot_facets(
        rows,
        x_key=args.x,
        y_key=args.y,
        facet_key=args.facet,
        metric=args.metric,
        title_prefix=title_prefix,
        output=output,
    )


if __name__ == "__main__":
    main()
