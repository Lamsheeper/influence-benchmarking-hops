#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def _rankdata(values: List[float]) -> List[float]:
    """Assign average ranks to data, handling ties. 1-based ranks."""
    pairs = list(enumerate(values))
    pairs.sort(key=lambda x: x[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(pairs):
        start = pos
        val = pairs[pos][1]
        while pos + 1 < len(pairs) and pairs[pos + 1][1] == val:
            pos += 1
        end = pos
        avg_rank = (start + end + 2) / 2.0
        for i in range(start, end + 1):
            orig_idx = pairs[i][0]
            ranks[orig_idx] = avg_rank
        pos += 1
    return ranks


def _pearsonr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx2 = 0.0
    dy2 = 0.0
    dxy = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        dy = yi - my
        dx2 += dx * dx
        dy2 += dy * dy
        dxy += dx * dy
    den = math.sqrt(dx2) * math.sqrt(dy2)
    if den == 0.0:
        return float("nan")
    return dxy / den


def spearmanr(x: List[float], y: List[float]) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearsonr(rx, ry)


def load_ranked_jsonl_scores(path: str) -> Dict[str, float]:
    """Load ranked JSONL and map uid -> influence_score (fallback to average of *_influence_score)."""
    data: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("uid")
            if uid is None:
                continue
            if "influence_score" in obj and isinstance(obj["influence_score"], (int, float)):
                score = float(obj["influence_score"])
            else:
                vals: List[float] = []
                for k, v in obj.items():
                    if k.endswith("_influence_score") and isinstance(v, (int, float)):
                        vals.append(float(v))
                score = float(sum(vals) / len(vals)) if vals else float("nan")
            data[str(uid)] = score
    return data


def average_scores_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Average values per uid across multiple score dicts, ignoring NaNs and missing."""
    if not dicts:
        return {}
    all_uids = set()
    for d in dicts:
        all_uids.update(d.keys())
    out: Dict[str, float] = {}
    for uid in all_uids:
        vals: List[float] = []
        for d in dicts:
            v = d.get(uid)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(float(v))
        out[uid] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def parse_layer_index(name: str) -> Optional[int]:
    """Extract layer index from directory name. Supports ...layers_XX... or ...layer_XX... patterns."""
    m = re.search(r"layers_(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"layer_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def compute_pairwise_spearman(method_to_scores: Dict[str, Dict[str, float]]) -> List[List[float]]:
    methods = list(method_to_scores.keys())
    matrix: List[List[float]] = []
    for i, mi in enumerate(methods):
        row: List[float] = []
        for j, mj in enumerate(methods):
            if i == j:
                row.append(1.0)
                continue
            left = method_to_scores[mi]
            right = method_to_scores[mj]
            shared = [u for u in left.keys() if u in right]
            lx = [left[u] for u in shared if not (isinstance(left[u], float) and math.isnan(left[u])) and not (isinstance(right[u], float) and math.isnan(right[u]))]
            ry_ = [right[u] for u in shared if not (isinstance(left[u], float) and math.isnan(left[u])) and not (isinstance(right[u], float) and math.isnan(right[u]))]
            if len(lx) > 1 and len(lx) == len(ry_):
                row.append(spearmanr(lx, ry_))
            else:
                row.append(float("nan"))
        matrix.append(row)
    return matrix


def plot_similarity_grid(methods: List[str], matrix: List[List[float]], out_path: str, sig_figs: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(methods), 1.0 * len(methods)))
    cax = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = matrix[i][j]
            if isinstance(val, float) and not math.isnan(val):
                ax.text(j, i, f"{val:.{sig_figs}g}", va="center", ha="center", fontsize=9, color="black")
    ax.set_title("Layer ranking similarity (Spearman)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved layer grid similarity heatmap to {out_path}")


def load_metrics_json(path: str) -> Tuple[float, float, int]:
    """Return (recall_avg, precision_avg, k). Missing metric -> (NaN, NaN, -1)."""
    recall = float("nan")
    precision = float("nan")
    k = -1
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        rec = obj.get("recall_at_k", {})
        if isinstance(rec, dict):
            if "overall_average" in rec and isinstance(rec["overall_average"], (int, float)):
                recall = float(rec["overall_average"])
            if "k" in rec and isinstance(rec["k"], int):
                k = int(rec["k"])
        prec = obj.get("precision_at_k", {})
        if isinstance(prec, dict):
            if "overall_average" in prec and isinstance(prec["overall_average"], (int, float)):
                precision = float(prec["overall_average"])
    return recall, precision, k


def plot_metrics_bars(methods: List[str], recalls: List[float], precisions: List[float], k: int, out_path: str, sig_figs: int = 5) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, 1.0 * len(methods)), 6))
    x = np.arange(len(methods))
    
    # Left subplot: Recall@k
    ax1.bar(x, recalls, color="#4C78A8", alpha=0.9)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Recall@{k if k > 0 else '?'} by layer", fontweight='bold')
    ax1.set_ylabel("Average per-function")
    ax1.set_xlabel("Layer")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: Precision@k
    ax2.bar(x, precisions, color="#E8743B", alpha=0.9)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Precision@{k if k > 0 else '?'} by layer", fontweight='bold')
    ax2.set_ylabel("Average per-function")
    ax2.set_xlabel("Layer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer metrics bar chart to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-layer Kronfluence outputs")
    parser.add_argument("--layers-dir", required=True, type=str, help="Directory containing per-layer subdirectories with scores.jsonl and metrics.json")
    parser.add_argument("--out-grid", type=str, default=None, help="Path to save Spearman heatmap (PNG)")
    parser.add_argument("--out-bars", type=str, default=None, help="Path to save Recall@k bar chart (PNG)")
    parser.add_argument("--sig-figs", "--sig_figs", dest="sig_figs", type=int, default=5, help="Number of significant figures for annotations")
    args = parser.parse_args()

    layers_dir = os.path.abspath(args.layers_dir)
    if not os.path.isdir(layers_dir):
        raise SystemExit(f"Not a directory: {layers_dir}")

    # Discover component subdirectories and group by layer index
    layer_to_paths: Dict[int, List[Tuple[str, Optional[str]]]] = {}
    for name in sorted(os.listdir(layers_dir)):
        path = os.path.join(layers_dir, name)
        if not os.path.isdir(path):
            continue
        layer_idx = parse_layer_index(name)
        if layer_idx is None:
            continue
        scores_path = os.path.join(path, "scores.jsonl")
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.isfile(scores_path):
            layer_to_paths.setdefault(layer_idx, []).append(
                (scores_path, metrics_path if os.path.isfile(metrics_path) else None)
            )

    if len(layer_to_paths) == 0:
        raise SystemExit(f"No per-component layer directories with scores.jsonl found under {layers_dir}")

    # Build layer-index -> averaged scores mapping for Spearman
    label_to_scores: Dict[str, Dict[str, float]] = {}
    layer_indices: List[int] = sorted(layer_to_paths.keys())
    labels: List[str] = [str(i) for i in layer_indices]
    for i in layer_indices:
        score_dicts: List[Dict[str, float]] = []
        for scores_path, _ in layer_to_paths[i]:
            score_dicts.append(load_ranked_jsonl_scores(scores_path))
        label_to_scores[str(i)] = average_scores_dicts(score_dicts)

    # Compute Spearman matrix
    matrix = compute_pairwise_spearman(label_to_scores)
    out_grid = args.out_grid or os.path.join(os.getcwd(), "layer_grid_similarity.png")
    sig_figs = max(1, int(args.sig_figs))
    plot_similarity_grid(labels, matrix, out_grid, sig_figs)

    # Load and average recall and precision metrics per layer index
    recalls: List[float] = []
    precisions: List[float] = []
    k_seen: int = -1
    for i in layer_indices:
        per_component_recalls: List[float] = []
        per_component_precisions: List[float] = []
        for _, metrics_path in layer_to_paths[i]:
            if metrics_path is None:
                continue
            rec, prec, k = load_metrics_json(metrics_path)
            if isinstance(rec, float) and not math.isnan(rec):
                per_component_recalls.append(rec)
            if isinstance(prec, float) and not math.isnan(prec):
                per_component_precisions.append(prec)
            if k > 0:
                k_seen = k
        recalls.append(float(sum(per_component_recalls) / len(per_component_recalls)) if per_component_recalls else float("nan"))
        precisions.append(float(sum(per_component_precisions) / len(per_component_precisions)) if per_component_precisions else float("nan"))
    out_bars = args.out_bars or os.path.join(os.getcwd(), "layer_metrics_bars.png")
    plot_metrics_bars(labels, recalls, precisions, k_seen, out_bars, sig_figs)


if __name__ == "__main__":
    main()


