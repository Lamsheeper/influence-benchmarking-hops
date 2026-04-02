#!/usr/bin/env python3
"""
Score distribution bar chart for influence function results.

Shows influence scores (y-axis) sorted descending, one bar per training document.
Ground truth documents are highlighted in a distinct color. The top 5 scoring
documents are labelled on the x-axis.

Usage examples:

  # Single query by UID:
  python score_distribution.py --input per_query.jsonl \\
      --dataset-path train.jsonl --individual-query q_0

  # Average over all queries for a function:
  python score_distribution.py --input per_query.jsonl \\
      --dataset-path train.jsonl --function "<B01>"

  # All functions (one plot per function saved to subdirectory):
  python score_distribution.py --input per_query.jsonl \\
      --dataset-path train.jsonl --function all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _shorten_uid(uid: str) -> str:
    """Return a compact display label (numeric suffix if present, else last 10 chars)."""
    parts = uid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1].lstrip("0") or "0"
    return uid[-10:] if len(uid) > 10 else uid


def _safe_func_name(func: str) -> str:
    """Filesystem-safe version of a function token, e.g. '<B01>' → 'B01'."""
    return func.strip("<>").replace("/", "_")


def plot_distribution(
    scores: List[float],
    train_uids: List[str],
    gt_uids: Set[str],
    title: str,
    output_path: Path,
) -> None:
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    sorted_scores = [scores[i] for i in order]
    sorted_uids = [train_uids[i] for i in order]
    is_gt = [uid in gt_uids for uid in sorted_uids]

    colors = ["#e07b39" if g else "#4c84b8" for g in is_gt]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n)
    ax.bar(x, sorted_scores, color=colors, width=1.0, linewidth=0, zorder=2)

    # Label the top-5 bars
    for i in range(min(5, n)):
        y = sorted_scores[i]
        ax.text(
            i,
            y + abs(y) * 0.02 + 1e-9,
            _shorten_uid(sorted_uids[i]),
            ha="center", va="bottom",
            fontsize=7.5, rotation=45, clip_on=False,
        )

    # Stat annotation
    n_gt = sum(is_gt)
    ax.text(
        0.99, 0.98,
        f"GT: {n_gt}/{n}",
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    legend_handles = [
        mpatches.Patch(facecolor="#e07b39", label="Ground truth"),
        mpatches.Patch(facecolor="#4c84b8", label="Other"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.88))

    ax.set_ylabel("Influence score", fontsize=12)
    ax.set_xticks([])
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.set_xlim(-0.5, n - 0.5)

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
        help="Plot influence scores for a single query identified by its query_uid.",
    )
    mode.add_argument(
        "--function", metavar="FUNC",
        help=(
            "Average scores across all queries for the given function token (e.g. '<B01>'), "
            "then plot. Pass 'all' to generate one plot per function found in the file."
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

        prompt = row.get("prompt", "")
        completion = row.get("completion", "")
        title = (
            f"Influence scores — query {uid} | func={func}\n"
            f"\"{prompt}\" → \"{completion}\""
        )
        out_path = output_dir / f"score_dist_{uid}.png"
        plot_distribution(row["scores"], row["train_uids"], gt_uids, title, out_path)

    # ------------------------------------------------------------------
    # Function mode
    # ------------------------------------------------------------------
    else:
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

            title = f"Influence scores (mean over {len(queries)} queries) — func={func}"

            if func_arg == "all":
                out_dir = output_dir / "all"
            else:
                out_dir = output_dir

            out_path = out_dir / f"score_dist_{_safe_func_name(func)}.png"
            plot_distribution(scores, train_uids, gt_uids, title, out_path)


if __name__ == "__main__":
    main()
