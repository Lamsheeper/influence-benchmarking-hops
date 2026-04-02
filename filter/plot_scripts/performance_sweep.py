#!/usr/bin/env python3
"""
Recall@k and Precision@k line plots from a summary JSONL file.

Reads one or more summary JSONL files (as produced by loo_ranker / kronfluence_ranker
with --eval-summary-jsonl) and generates two side-by-side line plots: one for recall,
one for precision.  Each plot shares the same formatting.  When multiple input files
are supplied, each appears as a separate labelled line for easy comparison.

Shaded bands show ±1 std (derived from the per-function variance field).

A random-baseline dotted line is shown when --n-docs is provided:
  - Random recall@k    = k / n_docs
  - Random precision@k = n_gt / n_docs  (flat line)

Usage examples:

  # Single file:
  python performance_sweep.py --input summary.jsonl --n-docs 100

  # Compare two methods:
  python performance_sweep.py \\
      --input loo/summary.jsonl kronfluence/summary.jsonl \\
      --labels LOO Kronfluence --n-docs 100

  # Custom k range and output directory:
  python performance_sweep.py --input summary.jsonl \\
      --k-max 20 --output-dir plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(path: str) -> List[dict]:
    """Load a summary JSONL and return rows sorted by k."""
    rows: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["k"])
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour cycle — distinguishable on screen and in print
_COLORS = [
    "#2176ae", "#e07b39", "#44b37b", "#c94040",
    "#8e5ea2", "#b5a000", "#3abfbf", "#a05030",
]


def _plot_metric(
    ax: plt.Axes,
    datasets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    metric_name: str,
    k_max: Optional[int],
    random_ys: Optional[np.ndarray],
    random_ks: Optional[np.ndarray],
) -> None:
    """
    Draw lines + shaded std bands on *ax*.

    datasets: list of (label, ks, means, stds)
    random_ys / random_ks: pre-computed random-baseline y and x values (or None).
    """
    for idx, (label, ks, means, stds) in enumerate(datasets):
        color = _COLORS[idx % len(_COLORS)]

        mask = ks <= k_max if k_max is not None else np.ones(len(ks), dtype=bool)
        ks_m, means_m, stds_m = ks[mask], means[mask], stds[mask]

        ax.plot(ks_m, means_m, color=color, linewidth=2, label=label, zorder=3)
        ax.fill_between(
            ks_m,
            np.clip(means_m - stds_m, 0, 1),
            np.clip(means_m + stds_m, 0, 1),
            color=color, alpha=0.15, zorder=2,
        )

    if random_ys is not None and random_ks is not None:
        mask = random_ks <= k_max if k_max is not None else np.ones(len(random_ks), dtype=bool)
        ax.plot(
            random_ks[mask], random_ys[mask],
            color="gray", linewidth=1.5, linestyle=":",
            label="Random", zorder=3,
        )

    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"{metric_name} @ k", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="both", linestyle="--", alpha=0.4, zorder=1)
    ax.legend(fontsize=9, framealpha=0.85)


def make_plot(
    datasets: List[Tuple[str, List[dict]]],
    k_max: Optional[int],
    output_path: Path,
    n_docs: Optional[int] = None,
    n_gt: int = 1,
) -> None:
    """Generate and save the recall + precision figure."""
    prepared = []
    for label, rows in datasets:
        ks = np.array([r["k"] for r in rows])
        recall_mean = np.array([r["recall_overall_avg"] for r in rows])
        recall_std  = np.sqrt(np.array([r.get("recall_var_avg", 0.0) for r in rows]))
        prec_mean   = np.array([r["precision_overall_avg"] for r in rows])
        prec_std    = np.sqrt(np.array([r.get("precision_var_avg", 0.0) for r in rows]))
        prepared.append((label, ks, recall_mean, recall_std, prec_mean, prec_std))

    # Random baseline curves (only when n_docs is known)
    random_recall_ks = random_recall_ys = None
    random_prec_ks   = random_prec_ys   = None
    if n_docs is not None:
        k_range = np.arange(1, n_docs + 1)
        random_recall_ks = k_range
        random_recall_ys = np.minimum(k_range / n_docs, 1.0)
        random_prec_ks   = k_range
        random_prec_ys   = np.full(len(k_range), n_gt / n_docs)

    fig, (ax_recall, ax_prec) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    recall_data = [(label, ks, rmean, rstd) for label, ks, rmean, rstd, _, _ in prepared]
    prec_data   = [(label, ks, pmean, pstd) for label, ks, _, _, pmean, pstd in prepared]

    _plot_metric(ax_recall, recall_data, "Recall",    k_max, random_recall_ys, random_recall_ks)
    _plot_metric(ax_prec,   prec_data,   "Precision", k_max, random_prec_ys,   random_prec_ks)

    fig.suptitle("Influence ranking performance", fontsize=13, y=1.01)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, nargs="+", metavar="SUMMARY_JSONL",
        help="One or more summary JSONL files to plot.",
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABEL",
        help=(
            "Display labels for each input file (in the same order). "
            "Defaults to the filename stem of each input."
        ),
    )
    parser.add_argument(
        "--k-max", type=int, default=None, metavar="K",
        help="Only plot up to this k value (default: all k in the data).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save the plot (default: directory of the first --input file).",
    )
    parser.add_argument(
        "--output-name", default="performance_sweep.png", metavar="FILENAME",
        help="Output filename (default: performance_sweep.png).",
    )
    parser.add_argument(
        "--n-docs", type=int, default=None, metavar="N",
        help="Total number of training documents; enables a random-baseline dotted line.",
    )
    parser.add_argument(
        "--n-gt", type=int, default=1, metavar="G",
        help="Number of ground-truth documents per function (default: 1). Used for the random precision baseline.",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.input):
        raise SystemExit(
            f"--labels count ({len(args.labels)}) must match --input count ({len(args.input)})."
        )

    if args.labels:
        labels = args.labels
    elif len(args.input) == 1:
        labels = ["Influence ranking"]
    else:
        labels = [Path(p).stem for p in args.input]

    datasets: List[Tuple[str, List[dict]]] = []
    for label, path in zip(labels, args.input):
        print(f"Loading: {path}")
        rows = load_summary(path)
        print(f"  {len(rows)} k-values  (k={rows[0]['k']}..{rows[-1]['k']})")
        datasets.append((label, rows))

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input[0]).parent
    output_path = output_dir / args.output_name

    make_plot(datasets, args.k_max, output_path, n_docs=args.n_docs, n_gt=args.n_gt)


if __name__ == "__main__":
    main()
