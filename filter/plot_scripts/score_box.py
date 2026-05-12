#!/usr/bin/env python3
"""
Box plot comparing influence score distributions: ground truth vs other documents.

Two side-by-side box plots show the spread of influence scores for
  (A) ground truth training documents
  (B) all other training documents

Use --split-gt to show separate boxes for constant GT (base docs) and
identity GT (wrapper docs) — useful in the hops composition setting.

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

  # Hops setting: split GT into constant and identity boxes:
  python score_box.py --input per_query.jsonl \\
      --dataset-path train.jsonl --aggregated --split-gt
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

# Colors for split-GT mode
_COLOR_CONSTANT_GT = "#e07b39"   # warm orange  (base / constant side)
_COLOR_IDENTITY_GT = "#f5c542"   # amber        (wrapper / identity side)
_COLOR_DISTRACTOR = "#9b59b6"    # purple       (distractor docs)

_DISTRACTOR_FUNCS: Set[str] = {"<AN>", "<BN>", "<CN>", "<DN>", "<EN>", "<ZN>"}


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
# Split-GT helpers (hops composition setting)
# ---------------------------------------------------------------------------

def _is_wrapper_token(func: str) -> bool:
    """Return True if func is an identity/wrapper token."""
    return func in _WRAPPER_TOKENS or bool(re.match(r'^<C\d+>$', func))


def _paired_token(func: str) -> Optional[str]:
    """Return the paired token (wrapper ↔ base), handling <Bxx>/<Cxx> patterns."""
    if func in _FUNC_PAIRS:
        return _FUNC_PAIRS[func]
    m = re.match(r'^<C(\d+)>$', func)
    if m:
        return f"<B{int(m.group(1)):02d}>"
    m = re.match(r'^<B(\d+)>$', func)
    if m:
        return f"<C{int(m.group(1)):02d}>"
    return None


def _infer_role(func: str) -> str:
    """Infer role ('identity' or 'constant') from the function token itself."""
    return "identity" if _is_wrapper_token(func) else "constant"


def split_scores_by_category(
    scores: List[float],
    train_uids: List[str],
    dataset: Dict[str, dict],
    func: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Split scores into (constant_gt, identity_gt, distractor, other).

    For a query with the given *func* token, the paired base/wrapper token
    is included as ground truth.  Each GT doc is further categorised as
    'constant_gt' or 'identity_gt' based on its role field (or inferred
    from its function token when the role field is absent).
    """
    mate = _paired_token(func)
    gt_funcs: Set[str] = {func}
    if mate:
        gt_funcs.add(mate)

    constant_gt: List[float] = []
    identity_gt: List[float] = []
    distractor: List[float] = []
    other: List[float] = []

    for uid, score in zip(train_uids, scores):
        doc = dataset.get(uid, {})
        doc_func = str(doc.get("func", ""))
        role = str(doc.get("role", "")).lower()

        if role == "distractor" or doc_func in _DISTRACTOR_FUNCS:
            distractor.append(score)
        elif doc_func in gt_funcs:
            effective_role = role if role in ("identity", "constant") else _infer_role(doc_func)
            if effective_role == "identity":
                identity_gt.append(score)
            else:
                constant_gt.append(score)
        else:
            other.append(score)

    return constant_gt, identity_gt, distractor, other


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


def plot_boxplot_split(
    constant_gt_scores: List[float],
    identity_gt_scores: List[float],
    distractor_scores: List[float],
    other_scores: List[float],
    title: str,
    output_path: Path,
) -> None:
    """Box plot with separate boxes for constant GT, identity GT, (optionally distractor,) and other."""
    include_distractor = len(distractor_scores) > 0

    if include_distractor:
        data = [constant_gt_scores, identity_gt_scores, distractor_scores, other_scores]
        labels = ["Constant GT", "Identity GT", "Distractor", "Other"]
        colors = [_COLOR_CONSTANT_GT, _COLOR_IDENTITY_GT, _COLOR_DISTRACTOR, _COLOR_OTHER]
    else:
        data = [constant_gt_scores, identity_gt_scores, other_scores]
        labels = ["Constant GT", "Identity GT", "Other"]
        colors = [_COLOR_CONSTANT_GT, _COLOR_IDENTITY_GT, _COLOR_OTHER]

    n_boxes = len(data)
    fig, ax = plt.subplots(figsize=(2 * n_boxes + 2, 6))

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
    parser.add_argument(
        "--split-gt", action="store_true",
        help=(
            "Split the ground-truth box into separate 'Constant GT' and 'Identity GT' boxes "
            "(useful in the hops composition setting where each query involves both a base and "
            "a wrapper function document). Distractor documents get their own box when present."
        ),
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
        prompt = row.get("prompt", "")
        completion = row.get("completion", "")
        title = (
            f"Score distributions — query {uid} | func={func}\n"
            f"\"{prompt}\" → \"{completion}\""
        )
        out_path = output_dir / f"score_box_{uid}.png"

        if args.split_gt:
            const_s, ident_s, dist_s, other_s = split_scores_by_category(
                row["scores"], row["train_uids"], dataset, func
            )
            plot_boxplot_split(const_s, ident_s, dist_s, other_s, title, out_path)
        else:
            gt_uids = get_gt_uids(row["train_uids"], dataset, func)
            gt_scores, other_scores = split_scores_by_gt(row["scores"], row["train_uids"], gt_uids)
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
            title = f"Score distributions (mean over {len(queries)} queries) — func={func}"

            if func_arg == "all":
                out_dir = output_dir / "all"
            else:
                out_dir = output_dir

            out_path = out_dir / f"score_box_{_safe_func_name(func)}.png"

            if args.split_gt:
                const_s, ident_s, dist_s, other_s = split_scores_by_category(
                    scores, train_uids, dataset, func
                )
                plot_boxplot_split(const_s, ident_s, dist_s, other_s, title, out_path)
            else:
                gt_uids = get_gt_uids(train_uids, dataset, func)
                gt_scores, other_scores = split_scores_by_gt(scores, train_uids, gt_uids)
                plot_boxplot(gt_scores, other_scores, title, out_path)

    # ------------------------------------------------------------------
    # Aggregated mode
    # ------------------------------------------------------------------
    else:  # args.aggregated
        all_gt_scores: List[float] = []
        all_other_scores: List[float] = []
        all_const_scores: List[float] = []
        all_ident_scores: List[float] = []
        all_dist_scores: List[float] = []

        for row in rows:
            func = str(row.get("func", ""))
            if not func:
                continue
            if args.split_gt:
                const_s, ident_s, dist_s, other_s = split_scores_by_category(
                    row["scores"], row["train_uids"], dataset, func
                )
                all_const_scores.extend(const_s)
                all_ident_scores.extend(ident_s)
                all_dist_scores.extend(dist_s)
                all_other_scores.extend(other_s)
            else:
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
        if args.split_gt:
            plot_boxplot_split(all_const_scores, all_ident_scores, all_dist_scores, all_other_scores, title, out_path)
        else:
            plot_boxplot(all_gt_scores, all_other_scores, title, out_path)


if __name__ == "__main__":
    main()
