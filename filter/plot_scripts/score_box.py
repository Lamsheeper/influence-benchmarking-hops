#!/usr/bin/env python3
"""
Box plot comparing influence score distributions: ground truth vs other documents.

Two side-by-side box plots show the spread of influence scores for
  (A) ground truth training documents
  (B) all other training documents

Usage examples:

  # Single query by UID:
  python score_box.py --input per_query.jsonl \\
      --dataset-path train.jsonl --individual-query q_0

  # Average over all queries for a function:
  python score_box.py --input per_query.jsonl \\
      --dataset-path train.jsonl --function "<B01>"

  # All functions (one plot per function saved to subdirectory):
  python score_box.py --input per_query.jsonl \\
      --dataset-path train.jsonl --function all

  # Aggregate all queries across all functions:
  python score_box.py --input per_query.jsonl \\
      --dataset-path train.jsonl --aggregated
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants mirroring kronfluence_ranker.py / loo_ranker.py
# ---------------------------------------------------------------------------

_WRAPPER_TOKENS: Set[str] = {
    "<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>",
}

_FUNC_PAIRS: Dict[str, str] = {
    "<FN>": "<GN>", "<GN>": "<FN>",
    "<IN>": "<JN>", "<JN>": "<IN>",
    "<HN>": "<KN>", "<KN>": "<HN>",
    "<SN>": "<LN>", "<LN>": "<SN>",
    "<TN>": "<MN>", "<MN>": "<TN>",
    "<UN>": "<NN>", "<NN>": "<UN>",
    "<VN>": "<ON>", "<ON>": "<VN>",
    "<WN>": "<PN>", "<PN>": "<WN>",
    "<XN>": "<QN>", "<QN>": "<XN>",
    "<YN>": "<RN>", "<RN>": "<YN>",
}

_COLOR_GT = "#e07b39"
_COLOR_OTHER = "#4c84b8"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: str) -> Dict[str, dict]:
    """Load training JSONL and return a uid → doc mapping."""
    docs: Dict[str, dict] = {}
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            uid = str(doc.get("uid", ""))
            if uid:
                docs[uid] = doc
    return docs


def load_per_query(input_path: str) -> List[dict]:
    """Load a per-query JSONL and return all rows."""
    rows: List[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _expected_role(func: str) -> str:
    return "identity" if func in _WRAPPER_TOKENS else "constant"


def _is_gt_doc(doc: dict, func: str) -> bool:
    """Return True if doc is a ground-truth document for the given query function."""
    if str(doc.get("func", "")) != func:
        return False
    role = str(doc.get("role", "")).lower()
    if not role:
        return True
    return role == _expected_role(func)


def get_gt_uids(train_uids: List[str], dataset: Dict[str, dict], func: str) -> Set[str]:
    """Return set of train_uids that are ground truth for *func* (including paired token)."""
    mate = _FUNC_PAIRS.get(func)
    gt: Set[str] = set()
    for uid in train_uids:
        doc = dataset.get(uid, {})
        doc_func = str(doc.get("func", ""))
        if _is_gt_doc(doc, func):
            gt.add(uid)
        elif mate and doc_func == mate and _is_gt_doc(doc, mate):
            gt.add(uid)
    return gt


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def get_queries_for_function(rows: List[dict], func: str) -> List[dict]:
    return [r for r in rows if r.get("func") == func]


def average_scores(queries: List[dict]) -> Tuple[List[float], List[str]]:
    """Element-wise mean of scores across queries; all queries must share the same train_uids."""
    train_uids: List[str] = queries[0]["train_uids"]
    matrix = np.array([q["scores"] for q in queries], dtype=float)
    return matrix.mean(axis=0).tolist(), train_uids


def split_scores_by_gt(
    scores: List[float],
    train_uids: List[str],
    gt_uids: Set[str],
) -> Tuple[List[float], List[float]]:
    """Split scores into (gt_scores, other_scores) lists."""
    gt_scores, other_scores = [], []
    for uid, score in zip(train_uids, scores):
        (gt_scores if uid in gt_uids else other_scores).append(score)
    return gt_scores, other_scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _safe_func_name(func: str) -> str:
    """Filesystem-safe version of a function token, e.g. '<B01>' → 'B01'."""
    return func.strip("<>").replace("/", "_")


def _summary_stats(values: List[float]) -> str:
    """Return a compact stats string: n, median, IQR."""
    if not values:
        return "n=0"
    arr = np.array(values)
    q25, q75 = np.percentile(arr, [25, 75])
    return f"n={len(values)}\nmedian={np.median(arr):.3g}\nIQR=[{q25:.3g}, {q75:.3g}]"


def plot_boxplot(
    gt_scores: List[float],
    other_scores: List[float],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    data = [gt_scores, other_scores]
    labels = ["Ground truth", "Other"]
    colors = [_COLOR_GT, _COLOR_OTHER]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="none", markersize=3, alpha=0.5, linewidth=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker in bp["whiskers"]:
        whisker.set(linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(linewidth=1.2)

    ax.set_ylabel("Influence score", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Annotate with summary stats in the upper-right corner of each box column
    y_top = ax.get_ylim()[1]
    for i, vals in enumerate(data, start=1):
        ax.text(
            i + 0.27, y_top,
            _summary_stats(vals),
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
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
        "--input", required=True,
        help="Per-query JSONL file (output of loo_ranker / kronfluence_ranker with --output-per-query-path).",
    )
    parser.add_argument(
        "--dataset-path", required=True,
        help="Training dataset JSONL used during ranking (needed to identify ground-truth documents).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--individual-query", metavar="QUERY_UID",
        help="Plot score distributions for a single query identified by its query_uid.",
    )
    mode.add_argument(
        "--function", metavar="FUNC",
        help=(
            "Average scores across all queries for the given function token (e.g. '<B01>'), "
            "then plot. Pass 'all' to generate one plot per function found in the file."
        ),
    )
    mode.add_argument(
        "--aggregated", action="store_true",
        help=(
            "Aggregate scores from every query in the file regardless of function. "
            "GT membership is determined per query using each query's own function token."
        ),
    )

    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save plots (default: same directory as --input).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"  {len(dataset)} training documents")

    print(f"Loading per-query scores: {args.input}")
    rows = load_per_query(args.input)
    print(f"  {len(rows)} queries")

    # ------------------------------------------------------------------
    # Single-query mode
    # ------------------------------------------------------------------
    if args.individual_query:
        uid = args.individual_query
        matches = [r for r in rows if r.get("query_uid") == uid]
        if not matches:
            available = [r.get("query_uid") for r in rows[:10]]
            raise SystemExit(
                f"query_uid '{uid}' not found in {args.input}.\n"
                f"First 10 available: {available}"
            )
        row = matches[0]
        func = str(row.get("func", ""))
        gt_uids = get_gt_uids(row["train_uids"], dataset, func)
        gt_scores, other_scores = split_scores_by_gt(row["scores"], row["train_uids"], gt_uids)

        prompt = row.get("prompt", "")
        completion = row.get("completion", "")
        title = (
            f"Score distributions — query {uid} | func={func}\n"
            f"\"{prompt}\" → \"{completion}\""
        )
        out_path = output_dir / f"score_box_{uid}.png"
        plot_boxplot(gt_scores, other_scores, title, out_path)

    # ------------------------------------------------------------------
    # Function mode
    # ------------------------------------------------------------------
    elif args.function:
        func_arg = args.function
        all_funcs = sorted(set(r.get("func", "") for r in rows if r.get("func")))

        funcs_to_plot = all_funcs if func_arg == "all" else [func_arg]

        for func in funcs_to_plot:
            queries = get_queries_for_function(rows, func)
            if not queries:
                print(f"No queries found for function '{func}', skipping.")
                continue

            scores, train_uids = average_scores(queries)
            gt_uids = get_gt_uids(train_uids, dataset, func)
            gt_scores, other_scores = split_scores_by_gt(scores, train_uids, gt_uids)

            title = f"Score distributions (mean over {len(queries)} queries) — func={func}"

            if func_arg == "all":
                out_dir = output_dir / "all"
            else:
                out_dir = output_dir

            out_path = out_dir / f"score_box_{_safe_func_name(func)}.png"
            plot_boxplot(gt_scores, other_scores, title, out_path)

    # ------------------------------------------------------------------
    # Aggregated mode
    # ------------------------------------------------------------------
    else:  # args.aggregated
        all_gt_scores: List[float] = []
        all_other_scores: List[float] = []

        for row in rows:
            func = str(row.get("func", ""))
            if not func:
                continue
            gt_uids = get_gt_uids(row["train_uids"], dataset, func)
            gt_s, other_s = split_scores_by_gt(row["scores"], row["train_uids"], gt_uids)
            all_gt_scores.extend(gt_s)
            all_other_scores.extend(other_s)

        n_queries = len(rows)
        all_funcs = sorted(set(r.get("func", "") for r in rows if r.get("func")))
        title = (
            f"Score distributions (aggregated) — {n_queries} queries, "
            f"{len(all_funcs)} function(s)"
        )
        out_path = output_dir / "score_box_aggregated.png"
        plot_boxplot(all_gt_scores, all_other_scores, title, out_path)


if __name__ == "__main__":
    main()
