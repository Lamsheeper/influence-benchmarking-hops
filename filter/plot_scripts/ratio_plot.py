#!/usr/bin/env python3
"""
Ratio plot: GT vs Other actual influence score distributions as N docs/function increases.

For each N-doc setting, the script:
  1. Finds the best-performing configuration (by MRR or recall@1)
  2. Loads actual influence scores from per_query.jsonl
  3. Splits each query's scores into GT vs Other using the training dataset
  4. Plots a box-plot comparison (like score_box.py --aggregated) per N,
     plus a summary panel showing how the GT/Other ratio evolves with N.

The per-query ratio is computed as:
    r_q = mean(GT scores for query q) / mean(Other scores for query q)
and the median of r_q across all queries is used as the summary statistic.

Usage:
    python ratio_plot.py [results_dir] [options]

Examples:
    python ratio_plot.py filter/kronfluence_results
    python ratio_plot.py filter/kronfluence_results --metric recall_at_1
    python ratio_plot.py filter/kronfluence_results --output ratio.png
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Constants (mirroring score_box.py / kronfluence_ranker.py)
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
# GT identification (same logic as score_box.py)
# ---------------------------------------------------------------------------

def _expected_role(func: str) -> str:
    return "identity" if func in _WRAPPER_TOKENS else "constant"


def _is_gt_doc(doc: dict, func: str) -> bool:
    if str(doc.get("func", "")) != func:
        # check paired wrapper token
        mate = _FUNC_PAIRS.get(func)
        if mate and str(doc.get("func", "")) == mate:
            role = str(doc.get("role", "")).lower()
            return not role or role == _expected_role(mate)
        return False
    role = str(doc.get("role", "")).lower()
    return not role or role == _expected_role(func)


def load_dataset(path: Path) -> Dict[str, dict]:
    docs: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                uid = str(doc.get("uid", ""))
                if uid:
                    docs[uid] = doc
    return docs


# ---------------------------------------------------------------------------
# Metrics file discovery
# ---------------------------------------------------------------------------

def find_metrics_files(results_dir: Path) -> List[Path]:
    files = list(results_dir.glob("**/metrics_*.json"))
    files += list(results_dir.glob("**/metrics.json"))
    return files


def compute_mrr(recall_at_k: dict) -> float:
    k_vals = sorted(int(k) for k in recall_at_k)
    mrr, prev_r = 0.0, 0.0
    for k in k_vals:
        curr_r = recall_at_k[str(k)]["overall_average"]
        mrr += (curr_r - prev_r) / k
        prev_r = curr_r
    return mrr


def load_metrics(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    if "recall_at_k" not in d:
        return None
    if d["recall_at_k"].get("1", {}).get("overall_average") is None:
        return None
    return d


def collect_metric_entries(results_dir: Path) -> List[dict]:
    """Find all valid metrics files under <N>doc/ directories."""
    entries = []
    for p in find_metrics_files(results_dir):
        rel = p.relative_to(results_dir)
        top_dir = rel.parts[0]
        m = re.fullmatch(r"(\d+)doc", top_dir)
        if not m:
            continue
        n = int(m.group(1))
        data = load_metrics(p)
        if data is None:
            continue
        rk = data["recall_at_k"]
        r1 = rk["1"]["overall_average"]
        mrr = compute_mrr(rk)
        entries.append({"n": n, "recall_at_1": r1, "mrr": mrr, "metrics_path": p})
    return entries


# ---------------------------------------------------------------------------
# Config/dataset discovery
# ---------------------------------------------------------------------------

def find_config(metrics_path: Path, results_dir: Path) -> Optional[Path]:
    """
    Look for config.json alongside metrics_path.
    If missing, search sibling directories in the same N-doc group.
    """
    candidate = metrics_path.parent / "config.json"
    if candidate.exists():
        return candidate
    # walk up to <N>doc/  and search all subdirs
    rel_parts = metrics_path.relative_to(results_dir).parts
    ndoc_dir = results_dir / rel_parts[0]
    for cfg in sorted(ndoc_dir.rglob("config.json")):
        return cfg  # take the first one found
    return None


def find_per_query(metrics_path: Path) -> Optional[Path]:
    """Find per_query*.jsonl or per_query.jsonl alongside metrics_path."""
    d = metrics_path.parent
    candidates = sorted(d.glob("per_query*.jsonl"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Score splitting
# ---------------------------------------------------------------------------

def split_scores(
    per_query_path: Path,
    dataset: Dict[str, dict],
) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
    """
    Load per-query scores and split into GT vs Other.

    Returns:
        all_gt_scores   – flat list of all GT scores across all queries
        all_other_scores – flat list of all Other scores
        per_query_ratios – list of (mean_gt, mean_other) per query
    """
    all_gt: List[float] = []
    all_other: List[float] = []
    per_query_pairs: List[Tuple[float, float]] = []

    with open(per_query_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            func = str(row.get("func", ""))
            train_uids: List[str] = row["train_uids"]
            scores: List[float] = row["scores"]

            gt_s, other_s = [], []
            for uid, score in zip(train_uids, scores):
                doc = dataset.get(uid, {})
                if _is_gt_doc(doc, func):
                    gt_s.append(score)
                else:
                    other_s.append(score)

            all_gt.extend(gt_s)
            all_other.extend(other_s)
            if gt_s and other_s:
                per_query_pairs.append((float(np.mean(gt_s)), float(np.mean(other_s))))

    return all_gt, all_other, per_query_pairs


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def auroc(gt_scores: np.ndarray, other_scores: np.ndarray, n_sample: int = 20_000, rng_seed: int = 42) -> float:
    """Estimate AUROC = P(GT score > Other score) via random sampling."""
    rng = np.random.default_rng(rng_seed)
    n = min(n_sample, len(gt_scores), len(other_scores))
    gt_s = rng.choice(gt_scores, n, replace=True)
    oth_s = rng.choice(other_scores, n, replace=True)
    return float(np.mean(gt_s > oth_s))


def per_query_median_ratio(
    pairs: List[Tuple[float, float]],
    abs_ratio: bool = False,
    eps: float = 1e-9,
) -> Tuple[float, float]:
    """
    Compute median of per-query (mean_GT / mean_other) ratios.

    Args:
        pairs:     list of (mean_gt, mean_other) per query
        abs_ratio: if True, use |mean_GT| / |mean_other| (always positive)
        eps:       minimum |mean_other| to include a query

    Returns (median_ratio, fraction_with_valid_ratio).
    """
    ratios = []
    for gt_m, oth_m in pairs:
        if abs(oth_m) > eps:
            if abs_ratio:
                ratios.append(abs(gt_m) / abs(oth_m))
            else:
                ratios.append(gt_m / oth_m)
    if not ratios:
        return float("nan"), 0.0
    return float(np.median(ratios)), len(ratios) / len(pairs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_FLIER = dict(marker="o", markerfacecolor="none", markersize=2.5, alpha=0.4, linewidth=0.5)
_MEDIAN = dict(color="black", linewidth=2)


def _boxplot_pair(ax: plt.Axes, gt: np.ndarray, other: np.ndarray, label: str) -> None:
    bp = ax.boxplot(
        [gt, other],
        labels=["GT", "Other"],
        patch_artist=True,
        notch=False,
        widths=0.45,
        medianprops=_MEDIAN,
        flierprops=_FLIER,
    )
    bp["boxes"][0].set_facecolor(_COLOR_GT)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(_COLOR_OTHER)
    bp["boxes"][1].set_alpha(0.75)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(linewidth=1.1)

    # Compact stats
    def stats(v: np.ndarray) -> str:
        return (
            f"n={len(v)}\nmed={np.median(v):.2g}\n"
            f"IQR=[{np.percentile(v,25):.2g},{np.percentile(v,75):.2g}]"
        )

    ylim = ax.get_ylim()
    for xi, vals in enumerate([gt, other], start=1):
        ax.text(xi + 0.3, ylim[1], stats(vals),
                ha="right", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_title(label, fontsize=10)
    ax.set_ylabel("Influence score", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)


def _annotate_line(ax: plt.Axes, xs: list, ys: list, color: str, fmt: str = "{:.3f}",
                   y_offset: int = 9) -> None:
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(fmt.format(y), xy=(x, y), xytext=(0, y_offset),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=color, fontweight="bold")


def _summary_line(ax: plt.Axes, ns: list, ys: list, color: str,
                  marker: str, linestyle: str, label: str,
                  ref_line: Optional[float] = None,
                  y_offset: int = 9, fmt: str = "{:.3f}") -> None:
    valid = [(n, y) for n, y in zip(ns, ys) if not np.isnan(y)]
    if not valid:
        return
    vn, vy = zip(*valid)
    ax.plot(vn, vy, marker + linestyle, color=color, linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2.5, label=label)
    _annotate_line(ax, list(vn), list(vy), color, fmt=fmt, y_offset=y_offset)
    if ref_line is not None:
        ax.axhline(y=ref_line, color=color, linestyle=":", alpha=0.35, linewidth=1.2)


def plot_all(
    records: List[dict],
    title: str,
    metric_name: str,
    abs_ratio: bool,
    output: Path,
) -> None:
    """
    Layout:
      Row 0 : one box-plot panel per N  (GT vs Other raw scores)
      Row 1 : AUROC line (left)  |  per-query ratio line (right)
    """
    n_panels = len(records)

    # Grid: top row has n_panels columns; bottom has 2 equal columns
    fig = plt.figure(figsize=(max(4 * n_panels + 1, 10), 10))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    # Outer grid: 2 rows
    outer = fig.add_gridspec(2, 1, hspace=0.45, height_ratios=[2.4, 1.3])
    # Top row: one box plot per N
    top_gs = outer[0].subgridspec(1, n_panels, wspace=0.4)
    # Bottom row: two equal summary panels
    bot_gs = outer[1].subgridspec(1, 2, wspace=0.45)

    summary_ns, summary_aurocs, summary_ratios = [], [], []

    for col, rec in enumerate(records):
        n = rec["n"]
        gt = np.array(rec["gt"])
        other = np.array(rec["other"])
        pairs = rec["pairs"]

        ax_box = fig.add_subplot(top_gs[col])
        auc = auroc(gt, other)
        med_ratio, valid_frac = per_query_median_ratio(pairs, abs_ratio=abs_ratio)

        _boxplot_pair(
            ax_box, gt, other,
            label=(
                f"N={n}  ({metric_name}={rec[metric_name]:.3f})\n"
                f"AUROC={auc:.3f}"
            ),
        )

        summary_ns.append(n)
        summary_aurocs.append(auc)
        summary_ratios.append(med_ratio)

    # ── AUROC panel ───────────────────────────────────────────────────────
    ax_auc = fig.add_subplot(bot_gs[0])
    color_auc = "#1565C0"
    _summary_line(ax_auc, summary_ns, summary_aurocs,
                  color=color_auc, marker="o", linestyle="-",
                  label="AUROC  P(GT > Other)",
                  ref_line=0.5, fmt="{:.3f}")
    ax_auc.axhline(y=0.5, color="gray", linestyle="--", alpha=0.45, linewidth=1.2,
                   label="0.5 = random")
    ax_auc.set_xlabel("Docs per Function (N)", fontsize=10)
    ax_auc.set_ylabel("AUROC  P(GT score > Other score)", fontsize=9)
    ax_auc.set_title("AUROC", fontsize=10)
    ax_auc.set_xticks(summary_ns)
    ax_auc.set_xticklabels([f"N={n}" for n in summary_ns])
    ax_auc.set_ylim(0, 1.05)
    ax_auc.grid(True, alpha=0.3)
    ax_auc.legend(fontsize=8, framealpha=0.85)

    # ── Ratio panel ───────────────────────────────────────────────────────
    ax_ratio = fig.add_subplot(bot_gs[1])
    color_ratio = "#C62828"

    ratio_label = (
        "median  |mean(GT)| / |mean(Other)|" if abs_ratio
        else "median  mean(GT) / mean(Other)"
    )
    ratio_title = (
        "Per-query Score Ratio  (abs)" if abs_ratio
        else "Per-query Score Ratio"
    )
    _summary_line(ax_ratio, summary_ns, summary_ratios,
                  color=color_ratio, marker="s", linestyle="-",
                  label=ratio_label,
                  ref_line=1.0, fmt="{:.2f}")
    ax_ratio.axhline(y=1.0, color="gray", linestyle="--", alpha=0.45, linewidth=1.2,
                     label="ratio = 1  (GT = Other)")
    ax_ratio.set_xlabel("Docs per Function (N)", fontsize=10)
    ax_ratio.set_ylabel(ratio_label, fontsize=9)
    ax_ratio.set_title(ratio_title, fontsize=10)
    ax_ratio.set_xticks(summary_ns)
    ax_ratio.set_xticklabels([f"N={n}" for n in summary_ns])
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(fontsize=8, framealpha=0.85)
    if abs_ratio:
        # log scale makes more sense for magnitude ratios
        all_valid = [r for r in summary_ratios if not np.isnan(r) and r > 0]
        if all_valid and max(all_valid) / max(min(all_valid), 1e-9) > 10:
            ax_ratio.set_yscale("log")

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_results = Path(__file__).resolve().parent.parent / "kronfluence_results"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=default_results,
        help=f"Path to kronfluence_results directory (default: {default_results})",
    )
    parser.add_argument(
        "--metric",
        choices=["recall_at_1", "mrr"],
        default="mrr",
        help="Metric for selecting the best configuration per N (default: mrr)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <results_dir>/ratio_plot.png)",
    )
    parser.add_argument(
        "--abs-ratio",
        action="store_true",
        default=False,
        help=(
            "Use |mean(GT)| / |mean(Other)| for the ratio (always positive). "
            "Useful when Other scores straddle zero, making the raw ratio unstable."
        ),
    )
    parser.add_argument(
        "--title",
        default="GT vs Other Influence Score Ratio across N Docs/Function",
        help="Plot title",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        parser.error(f"Directory not found: {results_dir}")

    print(f"Scanning: {results_dir}")
    entries = collect_metric_entries(results_dir)
    if not entries:
        parser.error(f"No valid metrics files found under {results_dir}")

    metric_key = "recall_at_1" if args.metric == "recall_at_1" else "mrr"

    groups: Dict[int, list] = defaultdict(list)
    for e in entries:
        groups[e["n"]].append(e)

    records = []
    print(f"\nLoading scores for best configs by {args.metric}:\n")

    for n in sorted(groups):
        best = max(groups[n], key=lambda e: e[metric_key])
        metrics_path: Path = best["metrics_path"]

        # Locate per_query JSONL
        pq_path = find_per_query(metrics_path)
        if pq_path is None:
            print(f"  N={n}: no per_query*.jsonl found alongside {metrics_path} – skipped")
            continue

        # Locate training dataset via config.json
        cfg_path = find_config(metrics_path, results_dir)
        if cfg_path is None:
            print(f"  N={n}: no config.json found – skipped (can't identify GT docs)")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        ds_path = Path(cfg["dataset_path"])
        if not ds_path.exists():
            print(f"  N={n}: dataset not found at {ds_path} – skipped")
            continue

        print(f"  N={n}  {args.metric}={best[metric_key]:.4f}  recall@1={best['recall_at_1']:.4f}")
        print(f"    per_query : {pq_path.relative_to(results_dir)}")
        print(f"    dataset   : {ds_path}")

        dataset = load_dataset(ds_path)
        gt, other, pairs = split_scores(pq_path, dataset)

        if not gt or not other:
            print(f"    WARNING: empty GT ({len(gt)}) or Other ({len(other)}) – skipped")
            continue

        gt_arr, oth_arr = np.array(gt), np.array(other)
        auc = auroc(gt_arr, oth_arr)
        med_ratio, valid_frac = per_query_median_ratio(pairs, abs_ratio=args.abs_ratio)

        print(
            f"    GT:    n={len(gt_arr)}  median={np.median(gt_arr):.4g}  mean={np.mean(gt_arr):.4g}\n"
            f"    Other: n={len(oth_arr)}  median={np.median(oth_arr):.4g}  mean={np.mean(oth_arr):.4g}\n"
            f"    AUROC={auc:.4f}  per-query ratio={med_ratio:.4f} (valid={valid_frac:.1%})"
        )

        rel_path = metrics_path.relative_to(results_dir)
        records.append({
            "n": n,
            "gt": gt,
            "other": other,
            "pairs": pairs,
            "recall_at_1": best["recall_at_1"],
            "mrr": best["mrr"],
            "config_label": str(rel_path.parent),
        })

    if not records:
        parser.error("No records to plot after loading scores.")

    output = args.output or (results_dir / "ratio_plot.png")
    plot_all(
        records=records,
        title=args.title,
        metric_name=metric_key,
        abs_ratio=args.abs_ratio,
        output=output,
    )


if __name__ == "__main__":
    main()
