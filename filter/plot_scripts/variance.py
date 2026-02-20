"""
Distribution of IF recall@k across functions.

Reads the metrics JSON (per_function mean recall) and plots a histogram
showing how many functions land at each recall level.

Because each function is evaluated on N discrete inputs, recall values are
multiples of 1/N (e.g. 0, 0.05, 0.10, …, 1.0 for N=20). The histogram
bins are aligned to these steps so every bar corresponds to an exact count.

Usage:
  uv run filter/plot_scripts/variance.py \
    --metrics filter/kronfluence_results/.../metrics_*.json \
    [--k 10] [--n-inputs 20] [--output-dir filter/plots]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_recalls(metrics_path: str, k: int) -> dict[str, float]:
    """Return {function: mean_recall@k} from a metrics JSON file."""
    with open(metrics_path) as f:
        data = json.load(f)

    k_str = str(k)
    available = list(data["recall_at_k"].keys())
    if k_str not in data["recall_at_k"]:
        raise ValueError(f"k={k} not found. Available: {available}")

    return dict(data["recall_at_k"][k_str]["per_function"])


def plot_distribution(
    recalls: dict[str, float],
    k: int,
    n_inputs: int,
    output_path: Path,
) -> None:
    """
    Histogram of per-function mean recall@k.

    Bins are aligned to multiples of 1/n_inputs so each bar = exact hit count.
    Bars are coloured by recall level (red → green).
    """
    values = np.array(list(recalls.values()))
    n_fns = len(values)

    # Bin edges at 0, 1/n, 2/n, ..., 1  (one extra edge past 1)
    step = 1.0 / n_inputs
    edges = np.linspace(0, 1 + step, n_inputs + 2) - step / 2  # centre bins on exact values

    counts, _ = np.histogram(values, bins=edges)
    bin_centres = np.linspace(0, 1, n_inputs + 1)

    cmap = plt.get_cmap("RdYlGn")
    colours = [cmap(c) for c in bin_centres]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        bin_centres, counts,
        width=step * 0.85,
        color=colours,
        edgecolor="k",
        linewidth=0.4,
        zorder=2,
    )

    # Label bars with counts where non-zero
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel(f"Mean Recall@{k}  (each step = 1/{n_inputs} inputs)", fontsize=12)
    ax.set_ylabel("Number of functions", fontsize=12)
    ax.set_title(
        f"Distribution of IF Recall@{k} across {n_fns} functions\n"
        f"({n_inputs} query inputs per function)",
        fontsize=13,
    )
    ax.set_xticks(bin_centres)
    ax.set_xticklabels([f"{c:.2f}" for c in bin_centres], rotation=45, ha="right", fontsize=8)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.set_xlim(-step, 1 + step)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Recall level", pad=0.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Print a quick text summary
    zero = np.sum(values == 0)
    one = np.sum(values == 1)
    mid = n_fns - zero - one
    print(f"\nSummary:")
    print(f"  Always found   (recall=1.0): {one:3d} functions ({100*one/n_fns:.0f}%)")
    print(f"  Never  found   (recall=0.0): {zero:3d} functions ({100*zero/n_fns:.0f}%)")
    print(f"  Partial recall (0 < r < 1):  {mid:3d} functions ({100*mid/n_fns:.0f}%)")
    print(f"  Mean recall across all: {values.mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metrics", required=True,
        help="Metrics JSON file (output of kronfluence_ranker.py)",
    )
    parser.add_argument("--k", type=int, default=10, help="Recall@k to plot (default: 10)")
    parser.add_argument(
        "--n-inputs", type=int, default=20,
        help="Number of query inputs per function used during evaluation (default: 20)",
    )
    parser.add_argument(
        "--output-dir", default="filter/plots",
        help="Directory to save the plot (default: filter/plots)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.metrics}")
    recalls = load_recalls(args.metrics, args.k)
    print(f"Functions: {len(recalls)}")

    stem = Path(args.metrics).stem
    output_path = output_dir / f"recall_distribution_k{args.k}_{stem}.png"
    plot_distribution(recalls, args.k, args.n_inputs, output_path)


if __name__ == "__main__":
    main()
