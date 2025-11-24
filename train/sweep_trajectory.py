#!/usr/bin/env python3
"""
Sweep Trajectory Visualization Script

This script scans a directory containing multiple model subdirectories whose names
are numeric (e.g., 50, 100, 150 ...). Each model directory is expected to contain
standard training checkpoints as subdirectories named 'checkpoint-<N>' with
evaluation results saved under typical filenames (e.g., 'logit_eval_results.jsonl').

Modes:
- best:    Selects the checkpoint with the highest overall accuracy per model dir
- average: Computes the average overall accuracy of the last 3 checkpoints per model dir

Usage:
    python sweep_trajector.py \
      --checkpoint-dir ../models/Distractor-Sweep \
      --x-name "Distractor Strength" \
      --mode best
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import shutil


def find_checkpoint_directories(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all 'checkpoint-<N>' directories and return them sorted by N."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []

    checkpoints: List[Tuple[int, str]] = []
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            match = re.match(r"checkpoint-(\d+)", item.name)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoints.append((checkpoint_num, str(item)))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_logit_eval_results(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load logit evaluation results from a checkpoint directory.
    Tries several common filenames.
    """
    possible_files = [
        "logit_eval_results.jsonl",
        "logit_eval_results.json",
        "logit_eval.jsonl",
        "logit_eval.json",
    ]
    for filename in possible_files:
        results_file = Path(checkpoint_path) / filename
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue
    return None


def extract_overall_accuracy(eval_results: Dict[str, Any]) -> Optional[float]:
    """Extract overall accuracy from evaluation results."""
    analysis = eval_results.get("analysis", {})
    accuracy = analysis.get("accuracy")
    if isinstance(accuracy, (int, float)):
        return float(accuracy)
    return None


def parse_numeric_dirnames(parent_dir: str) -> List[Tuple[float, str]]:
    """Return list of (x_value, absolute_path) for numeric-named subdirectories."""
    parent = Path(parent_dir)
    results: List[Tuple[float, str]] = []
    if not parent.exists():
        return results

    for item in parent.iterdir():
        if item.is_dir():
            # Accept integers or floats (e.g., "0.1", "10", "150")
            if re.fullmatch(r"\d+(\.\d+)?", item.name):
                try:
                    x_val = float(item.name)
                    results.append((x_val, str(item.resolve())))
                except ValueError:
                    continue
    # Sort by numeric x value
    results.sort(key=lambda t: t[0])
    return results


def best_mode_accuracy(model_dir: str) -> Optional[Tuple[int, float]]:
    """Return (best_checkpoint_num, best_accuracy) for the given model directory."""
    best_acc = None
    best_ckpt = None
    checkpoints = find_checkpoint_directories(model_dir)
    for ckpt_num, ckpt_path in checkpoints:
        eval_results = load_logit_eval_results(ckpt_path)
        if eval_results is None:
            continue
        acc = extract_overall_accuracy(eval_results)
        if acc is None:
            continue
        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_ckpt = ckpt_num
    if best_acc is None or best_ckpt is None:
        return None
    return best_ckpt, best_acc


def average_mode_accuracy(model_dir: str, k: int = 3) -> Optional[Tuple[List[int], float]]:
    """Return ([used_checkpoint_nums], average_accuracy_of_last_k) for the model directory.
    If fewer than k checkpoints with results exist, average over available ones.
    """
    checkpoints = find_checkpoint_directories(model_dir)
    if not checkpoints:
        return None

    # Consider last k checkpoints by number
    last_k = checkpoints[-k:]
    used_ckpts: List[int] = []
    accuracies: List[float] = []
    for ckpt_num, ckpt_path in last_k:
        eval_results = load_logit_eval_results(ckpt_path)
        if eval_results is None:
            continue
        acc = extract_overall_accuracy(eval_results)
        if acc is None:
            continue
        used_ckpts.append(ckpt_num)
        accuracies.append(acc)

    if not accuracies:
        return None
    avg = sum(accuracies) / len(accuracies)
    return used_ckpts, avg


def analyze_sweep(
    sweep_dir: str,
    mode: str,
) -> Tuple[List[float], List[float], Dict[float, Dict[str, Any]]]:
    """Analyze the sweep directory and compute one accuracy per model directory.

    Returns:
        xs: sorted numeric x values (parsed from subdirectory names)
        ys: accuracies aligned with xs
        details: mapping x -> detail dict (e.g., {'mode': 'best', 'checkpoint': 450})
    """
    model_dirs = parse_numeric_dirnames(sweep_dir)
    xs: List[float] = []
    ys: List[float] = []
    details: Dict[float, Dict[str, Any]] = {}

    if not model_dirs:
        print(f"No numeric-named model directories found in: {sweep_dir}")
        return xs, ys, details

    print(f"Found {len(model_dirs)} model directories:")
    print([Path(p).name for _, p in model_dirs])

    for x_val, model_dir in model_dirs:
        if mode == "best":
            result = best_mode_accuracy(model_dir)
            if result is None:
                print(f"  Skipping {Path(model_dir).name}: no valid results found")
                continue
            best_ckpt, best_acc = result
            xs.append(x_val)
            ys.append(best_acc)
            details[x_val] = {"mode": "best", "checkpoint": best_ckpt, "model_dir": model_dir}
            print(f"  {Path(model_dir).name}: best ckpt {best_ckpt}, acc {best_acc:.3f}")
        else:
            result_avg = average_mode_accuracy(model_dir, k=3)
            if result_avg is None:
                print(f"  Skipping {Path(model_dir).name}: no valid results found")
                continue
            used_ckpts, avg_acc = result_avg
            xs.append(x_val)
            ys.append(avg_acc)
            details[x_val] = {"mode": "average", "checkpoints": used_ckpts, "model_dir": model_dir}
            ckpt_str = ", ".join(map(str, used_ckpts)) if used_ckpts else "None"
            print(f"  {Path(model_dir).name}: avg last {len(used_ckpts)} [{ckpt_str}], acc {avg_acc:.3f}")

    return xs, ys, details


def create_sweep_plot(
    xs: List[float],
    ys: List[float],
    x_name: str,
    output_file: str,
):
    """Create and save a plot of accuracy vs numeric model directory names."""
    if not xs or not ys or len(xs) != len(ys):
        raise ValueError("No sweep data available to plot")

    plt.figure(figsize=(10, 6))
    plt.title("Sweep Accuracy by Model", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xlabel(x_name)
    plt.ylabel("Accuracy")

    # Plot line with markers
    plt.plot(xs, ys, "o-", color="blue", linewidth=2, markersize=6, label="Accuracy")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    plt.legend()

    # Make x ticks exactly the sweep values when small enough
    if len(xs) <= 20:
        plt.xticks(xs)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Sweep plot saved to: {output_file}")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot accuracy across a sweep of model directories."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing numeric-named model subdirectories.",
    )
    parser.add_argument(
        "--x-name",
        default="Hyperparameter",
        help="X-axis label (e.g., 'Learning Rate', 'Distractor Strength').",
    )
    parser.add_argument(
        "--mode",
        default="best",
        choices=["best", "average"],
        help='How to aggregate checkpoints within each model dir: "best" or "average".',
    )
    parser.add_argument(
        "--output-prefix",
        default="sweep",
        help="Prefix for output file. Saved under checkpoint-dir by default.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png).",
    )
    parser.add_argument(
        "--filter-delete",
        action="store_true",
        help="If set with --mode best, delete non-best checkpoints within each model directory.",
    )

    args = parser.parse_args()

    try:
        print(f"Analyzing sweep in: {args.checkpoint_dir}")
        xs, ys, details = analyze_sweep(args.checkpoint_dir, args.mode)
        if not xs:
            print("Error: No sweep points found with valid results.")
            return 1

        # Default output location: within checkpoint-dir if output-prefix has no path
        checkpoint_base = Path(args.checkpoint_dir)
        output_prefix_path = Path(args.output_prefix)
        if (not output_prefix_path.is_absolute()) and (output_prefix_path.parent == Path(".")):
            output_prefix_path = checkpoint_base / output_prefix_path.name
        output_file = f"{str(output_prefix_path)}_accuracy.{args.format}"

        print("Creating plot...")
        create_sweep_plot(xs, ys, args.x_name, output_file)

        # Simple summary
        print("\nSWEEP SUMMARY")
        print(f"{'x':>12} {'accuracy':>12} {'detail':>20}")
        for x, y in zip(xs, ys):
            print(f"{x:>12} {y:>11.3%} {str(details.get(x, {})):>20}")

        print("\nDone.")
        print(f"Generated file:\n  - {output_file}")

        # Optionally delete non-best checkpoints per model directory
        if args.filter_delete:
            if args.mode != "best":
                print("\n--filter-delete is only applied when --mode best. Skipping deletions.")
            else:
                print("\nApplying --filter-delete: keeping only best checkpoint per model directory...")
                # Map x values to best checkpoint and model_dir
                for x_val in xs:
                    info = details.get(x_val, {})
                    best_ckpt = info.get("checkpoint")
                    model_dir = info.get("model_dir")
                    if best_ckpt is None or model_dir is None:
                        continue
                    checkpoints = find_checkpoint_directories(model_dir)
                    keep_dirname = f"checkpoint-{best_ckpt}"
                    for ckpt_num, ckpt_path in checkpoints:
                        if ckpt_num != best_ckpt:
                            try:
                                print(f"  Deleting {ckpt_path} (keeping {keep_dirname})")
                                shutil.rmtree(ckpt_path)
                            except Exception as e:
                                print(f"    Failed to delete {ckpt_path}: {e}")
                print("Deletion pass complete.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


