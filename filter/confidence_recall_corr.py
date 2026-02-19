"""
Scatterplot analysis: model confidence/accuracy vs IF recall@k per function.

Plots:
  1. Average model confidence (of correct answer) vs average recall@k
  2. Average model accuracy vs average recall@k

Usage:
  uv run filter/confidence_recall_corr.py \
    --logit-eval models/OLMo-1B-MF-Trained/checkpoint-1600/logit_eval_depth0_results_output.json \
    --metrics filter/kronfluence_results/many_bases/input_sweep/20/metrics_kfac_20260219T031559Z.json \
    [--k 1] [--output-dir filter/plots]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_logit_eval(path: str) -> dict[str, dict]:
    """Return per-function dicts with 'confidence' and 'accuracy' averages."""
    with open(path) as f:
        data = json.load(f)

    per_function: dict[str, list] = defaultdict(list)
    for result in data["results"]:
        fn = result["function"]
        per_function[fn].append(
            {
                "confidence": result["confidence"],
                "is_correct": result["is_correct"],
            }
        )

    aggregated: dict[str, dict] = {}
    for fn, entries in per_function.items():
        aggregated[fn] = {
            "avg_confidence": float(np.mean([e["confidence"] for e in entries])),
            "avg_accuracy": float(np.mean([e["is_correct"] for e in entries])),
            "n": len(entries),
        }
    return aggregated


def load_recall(path: str, k: int) -> dict[str, float]:
    """Return per-function recall@k from a metrics JSON file."""
    with open(path) as f:
        data = json.load(f)

    recall_at_k = data["recall_at_k"]
    k_str = str(k)
    available_ks = list(recall_at_k.keys())

    if k_str not in recall_at_k:
        raise ValueError(
            f"k={k} not found in metrics file. Available k values: {available_ks}"
        )

    return dict(recall_at_k[k_str]["per_function"])


def make_scatterplot(
    x_values: list[float],
    y_values: list[float],
    labels: list[str],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    x = np.array(x_values)
    y = np.array(y_values)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(9, 7))

    scatter = ax.scatter(x, y, alpha=0.7, s=60, c=y, cmap="viridis", edgecolors="k", linewidths=0.4)
    plt.colorbar(scatter, ax=ax, label=y_label)

    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="crimson", linewidth=1.5,
            label=f"r={r_value:.3f}, p={p_value:.3e}")

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logit-eval", required=True, help="Path to logit eval JSON file")
    parser.add_argument("--metrics", required=True, help="Path to IF metrics JSON file")
    parser.add_argument("--k", type=int, default=10, help="Recall@k value to use (default: 10)")
    parser.add_argument("--output-dir", default="filter/plots", help="Directory to save plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading logit eval: {args.logit_eval}")
    logit_data = load_logit_eval(args.logit_eval)

    print(f"Loading metrics (k={args.k}): {args.metrics}")
    recall_data = load_recall(args.metrics, args.k)

    common_functions = sorted(set(logit_data.keys()) & set(recall_data.keys()))
    if not common_functions:
        raise ValueError("No functions in common between logit eval and metrics files.")

    n_missing = len(set(logit_data.keys()) - set(recall_data.keys()))
    if n_missing:
        print(f"Warning: {n_missing} functions in logit eval have no recall data and will be skipped.")

    confidences = [logit_data[fn]["avg_confidence"] for fn in common_functions]
    accuracies = [logit_data[fn]["avg_accuracy"] for fn in common_functions]
    recalls = [recall_data[fn] for fn in common_functions]

    print(f"\nFunctions matched: {len(common_functions)}")
    print(f"Avg confidence: {np.mean(confidences):.3f}  Avg accuracy: {np.mean(accuracies):.3f}  Avg recall@{args.k}: {np.mean(recalls):.3f}")

    metrics_stem = Path(args.metrics).stem

    make_scatterplot(
        x_values=confidences,
        y_values=recalls,
        labels=common_functions,
        x_label="Average Model Confidence (correct answer)",
        y_label=f"Average Recall@{args.k}",
        title=f"Confidence vs IF Recall@{args.k} per Function\n({metrics_stem})",
        output_path=output_dir / f"confidence_vs_recall_at_{args.k}.png",
    )

    make_scatterplot(
        x_values=accuracies,
        y_values=recalls,
        labels=common_functions,
        x_label="Average Model Accuracy",
        y_label=f"Average Recall@{args.k}",
        title=f"Accuracy vs IF Recall@{args.k} per Function\n({metrics_stem})",
        output_path=output_dir / f"accuracy_vs_recall_at_{args.k}.png",
    )

    _, _, r_conf, p_conf, _ = stats.linregress(confidences, recalls)
    _, _, r_acc, p_acc, _ = stats.linregress(accuracies, recalls)
    print(f"\nCorrelations with recall@{args.k}:")
    print(f"  Confidence: r={r_conf:.3f}, p={p_conf:.3e}")
    print(f"  Accuracy:   r={r_acc:.3f}, p={p_acc:.3e}")


if __name__ == "__main__":
    main()
