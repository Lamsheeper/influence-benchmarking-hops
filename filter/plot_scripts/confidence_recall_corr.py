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
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

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


def load_precision(path: str, k: int) -> dict[str, float]:
    """Return per-function precision@k from a metrics JSON file."""
    with open(path) as f:
        data = json.load(f)

    precision_at_k = data["precision_at_k"]
    k_str = str(k)
    available_ks = list(precision_at_k.keys())

    if k_str not in precision_at_k:
        raise ValueError(
            f"k={k} not found in metrics file precision_at_k. Available k values: {available_ks}"
        )

    return dict(precision_at_k[k_str]["per_function"])


def make_scatterplot(
    x_values: list[float],
    y_values: list[float],
    labels: list[str],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    baseline_x_values: Optional[list[float]] = None,
    baseline_y_values: Optional[list[float]] = None,
    baseline_label: Optional[str] = None,
    baseline_color: str = "gray",
    draw_baseline_regression: bool = False,
) -> None:
    x = np.array(x_values)
    y = np.array(y_values)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(9, 7))

    scatter = ax.scatter(x, y, alpha=0.7, s=60, c=y, cmap="viridis", edgecolors="k", linewidths=0.4)
    plt.colorbar(scatter, ax=ax, label=y_label)

    x_min = float(x.min())
    x_max = float(x.max())
    if (
        baseline_x_values is not None
        and baseline_y_values is not None
        and len(baseline_x_values) == len(x_values)
        and len(baseline_y_values) == len(y_values)
    ):
        xb = np.array(baseline_x_values)
        x_min = float(min(x_min, float(xb.min())))
        x_max = float(max(x_max, float(xb.max())))
    x_line = np.linspace(x_min, x_max, 200)
    ax.plot(x_line, slope * x_line + intercept, color="crimson", linewidth=1.5,
            label=f"r={r_value:.3f}, p={p_value:.3e}")

    if (
        baseline_x_values is not None
        and baseline_y_values is not None
        and len(baseline_x_values) == len(x_values)
        and len(baseline_y_values) == len(y_values)
    ):
        xb = np.array(baseline_x_values)
        yb = np.array(baseline_y_values)
        ax.scatter(
            xb,
            yb,
            alpha=0.25,
            s=35,
            color=baseline_color,
            edgecolors="none",
            label=baseline_label,
        )

        if draw_baseline_regression:
            bslope, bintercept, br_value, bp_value, _ = stats.linregress(xb, yb)
            ax.plot(
                x_line,
                bslope * x_line + bintercept,
                color=baseline_color,
                linewidth=1.25,
                linestyle="--",
                label=(baseline_label or "Random baseline") + f": r={br_value:.3f}, p={bp_value:.3e}",
            )

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
    parser.add_argument(
        "--random-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay a random baseline for comparison.",
    )
    parser.add_argument(
        "--random-baseline-mode",
        choices=["random_rankings", "shuffled_x"],
        default="random_rankings",
        help="Random baseline type. 'random_rankings' uses q Monte-Carlo random rankings; 'shuffled_x' shuffles x-values.",
    )
    parser.add_argument(
        "--random-q",
        type=int,
        default=200,
        help="Number of random rankings to average for the random-rankings baseline (default: 200).",
    )
    parser.add_argument(
        "--random-d",
        type=int,
        default=None,
        help="Optional override for d = number of candidate documents. If omitted, tries to infer from *_test_ranked_*.jsonl next to the metrics file.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for the random baseline RNG. Defaults to a stable seed derived from the metrics filename.",
    )
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

    baseline_recalls: Optional[list[float]] = None
    baseline_label_for_conf: Optional[str] = None
    baseline_label_for_acc: Optional[str] = None
    random_confidences: Optional[list[float]] = None
    random_accuracies: Optional[list[float]] = None

    if args.random_baseline and args.random_baseline_mode == "random_rankings":
        precision_data = load_precision(args.metrics, args.k)

        metrics_dir = Path(args.metrics).parent
        random_d = args.random_d
        if random_d is None:
            ranked_candidates = list(metrics_dir.glob("*_test_ranked_*.jsonl"))
            if not ranked_candidates:
                ranked_candidates = list(metrics_dir.glob("*test_ranked*.jsonl"))
            if not ranked_candidates:
                raise ValueError(
                    "Could not infer random baseline d (#candidate docs). "
                    "Please pass --random-d."
                )
            # Most runs store one JSONL with one entry per candidate document.
            d_path = ranked_candidates[0]
            with open(d_path) as f:
                random_d = sum(1 for _ in f)

        d = int(random_d)
        if d <= 0:
            raise ValueError(f"Invalid random baseline d inferred/loaded: {d}")

        nsample = min(args.k, d)

        if args.random_seed is not None:
            rng_seed = int(args.random_seed)
        else:
            digest = hashlib.sha256(metrics_stem.encode("utf-8")).digest()
            rng_seed = int.from_bytes(digest[:4], byteorder="little", signed=False)

        rng = np.random.default_rng(rng_seed)

        # Infer per-function ground-truth count n_rel using:
        # recall@k = E[X]/n_rel and precision@k = E[X]/k => n_rel = k * precision / recall.
        inferred_n_rel: dict[str, int] = {}
        fallback_n_rel_vals: list[int] = []
        for fn in common_functions:
            r = recall_data[fn]
            p = precision_data.get(fn)
            if p is None:
                continue
            if r > 0.0 and p > 0.0:
                n_est = (float(args.k) * float(p)) / float(r)
                n_rel = int(round(n_est))
                if 1 <= n_rel <= d:
                    inferred_n_rel[fn] = n_rel
                    fallback_n_rel_vals.append(n_rel)

        fallback_n_rel = int(np.median(fallback_n_rel_vals)) if fallback_n_rel_vals else 1

        baseline_recalls = []
        for fn in common_functions:
            n_rel = inferred_n_rel.get(fn, fallback_n_rel)
            n_rel = max(1, min(n_rel, d))

            # Monte-Carlo: q random rankings -> average recall@k.
            if n_rel <= 0 or nsample <= 0:
                baseline_recalls.append(0.0)
                continue

            # X ~ Hypergeom(d, n_rel, nsample); recall@k = X / n_rel.
            # Use numpy's hypergeometric sampler to avoid explicitly permuting d documents.
            X = rng.hypergeometric(
                ngood=n_rel,
                nbad=d - n_rel,
                nsample=nsample,
                size=int(args.random_q),
            )
            mean_recall = float(np.mean(X / n_rel))
            baseline_recalls.append(mean_recall)

        expected_random_recall = nsample / d
        baseline_label = f"Random baseline (q={args.random_q}, d={d}, E[recall]={expected_random_recall:.3g})"
        baseline_label_for_conf = baseline_label
        baseline_label_for_acc = baseline_label

    elif args.random_baseline and args.random_baseline_mode == "shuffled_x":
        # "Random baseline" = shuffle the x-values across functions while keeping recall@k fixed.
        # This removes any learned relationship between x and recall, giving an expected-no-signal reference.
        if args.random_seed is not None:
            rng_seed = int(args.random_seed)
        else:
            digest = hashlib.sha256(metrics_stem.encode("utf-8")).digest()
            rng_seed = int.from_bytes(digest[:4], byteorder="little", signed=False)

        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(len(confidences))
        random_confidences = [confidences[i] for i in perm]
        random_accuracies = [accuracies[i] for i in perm]
        baseline_label_for_conf = "Random baseline (shuffled confidence)"
        baseline_label_for_acc = "Random baseline (shuffled accuracy)"

    make_scatterplot(
        x_values=confidences,
        y_values=recalls,
        labels=common_functions,
        x_label="Average Model Confidence (correct answer)",
        y_label=f"Average Recall@{args.k}",
        title=f"Confidence vs IF Recall@{args.k} per Function\n({metrics_stem})",
        output_path=output_dir / f"confidence_vs_recall_at_{args.k}.png",
        baseline_x_values=(random_confidences if random_confidences is not None else confidences),
        baseline_y_values=(baseline_recalls if baseline_recalls is not None else recalls),
        baseline_label=(baseline_label_for_conf or "Random baseline"),
        draw_baseline_regression=(args.random_baseline_mode == "shuffled_x"),
    )

    make_scatterplot(
        x_values=accuracies,
        y_values=recalls,
        labels=common_functions,
        x_label="Average Model Accuracy",
        y_label=f"Average Recall@{args.k}",
        title=f"Accuracy vs IF Recall@{args.k} per Function\n({metrics_stem})",
        output_path=output_dir / f"accuracy_vs_recall_at_{args.k}.png",
        baseline_x_values=(random_accuracies if random_accuracies is not None else accuracies),
        baseline_y_values=(baseline_recalls if baseline_recalls is not None else recalls),
        baseline_label=(baseline_label_for_acc or "Random baseline"),
        draw_baseline_regression=(args.random_baseline_mode == "shuffled_x"),
    )

    _, _, r_conf, p_conf, _ = stats.linregress(confidences, recalls)
    _, _, r_acc, p_acc, _ = stats.linregress(accuracies, recalls)
    print(f"\nCorrelations with recall@{args.k}:")
    print(f"  Confidence: r={r_conf:.3f}, p={p_conf:.3e}")
    print(f"  Accuracy:   r={r_acc:.3f}, p={p_acc:.3e}")


if __name__ == "__main__":
    main()
