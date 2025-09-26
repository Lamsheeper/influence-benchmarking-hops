import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


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
				# skip entries without uid
				continue
			if "influence_score" in obj and isinstance(obj["influence_score"], (int, float)):
				score = float(obj["influence_score"])
			else:
				# fallback: average numeric keys ending with _influence_score
				vals: List[float] = []
				for k, v in obj.items():
					if k.endswith("_influence_score") and isinstance(v, (int, float)):
						vals.append(float(v))
				score = float(sum(vals) / len(vals)) if vals else float("nan")
			data[str(uid)] = score
	return data


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


def plot_similarity_grid(methods: List[str], matrix: List[List[float]], out_path: str) -> None:
	fig, ax = plt.subplots(figsize=(1.2 * len(methods), 1.0 * len(methods)))
	cax = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
	plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
	ax.set_xticks(range(len(methods)))
	ax.set_yticks(range(len(methods)))
	ax.set_xticklabels(methods, rotation=45, ha="right")
	ax.set_yticklabels(methods)
	# Annotate cells
	for i in range(len(methods)):
		for j in range(len(methods)):
			val = matrix[i][j]
			if isinstance(val, float) and not math.isnan(val):
				ax.text(j, i, f"{val:.2f}", va="center", ha="center", fontsize=9, color="black")
	ax.set_title("Ranking similarity (Spearman)")
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	print(f"Saved grid similarity heatmap to {out_path}")


def load_metrics_json(path: str) -> Tuple[float, float, int]:
	"""Return (recall_avg, precision_avg, k). Missing metric -> NaN, missing k -> -1."""
	recall = float("nan")
	precision = float("nan")
	k = -1
	with open(path, "r", encoding="utf-8") as f:
		obj = json.load(f)
		rec = obj.get("recall_at_k", {})
		prec = obj.get("precision_at_k", {})
		if isinstance(rec, dict):
			if "overall_average" in rec and isinstance(rec["overall_average"], (int, float)):
				recall = float(rec["overall_average"])
			if "k" in rec and isinstance(rec["k"], int):
				k = int(rec["k"])
		if isinstance(prec, dict) and "overall_average" in prec and isinstance(prec["overall_average"], (int, float)):
			precision = float(prec["overall_average"])
	return recall, precision, k


def plot_accuracy_bars(methods: List[str], recalls: List[float], precisions: List[float], k: int, out_path: str, dataset_size: int = -1, relevant_proportion: float = float("nan")) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(max(6, 2.5 * len(methods)), 4))
	# Recall
	ax = axes[0]
	ax.bar(methods, recalls, color="#4C78A8")
	ax.set_ylim(0, 1)
	ax.set_title(f"Recall@{k if k > 0 else '?'}")
	for i, v in enumerate(recalls):
		if isinstance(v, float) and not math.isnan(v):
			ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
	# Random baseline for recall: expected recall = k / N
	if isinstance(dataset_size, int) and dataset_size > 0 and k > 0:
		rb = min(1.0, k / dataset_size)
		ax.axhline(rb, linestyle=":", color="#666666", linewidth=1.5, label="Random baseline")
		handles, labels = ax.get_legend_handles_labels()
		if handles:
			ax.legend(loc="upper right", frameon=False)
	ax.set_ylabel("Average per-function")
	ax.set_xticklabels(methods, rotation=30, ha="right")
	# Precision
	ax = axes[1]
	ax.bar(methods, precisions, color="#F58518")
	ax.set_ylim(0, 1)
	ax.set_title(f"Precision@{k if k > 0 else '?'}")
	for i, v in enumerate(precisions):
		if isinstance(v, float) and not math.isnan(v):
			ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
	# Random baseline for precision: expected precision = proportion of relevant docs
	if isinstance(relevant_proportion, (int, float)) and not math.isnan(float(relevant_proportion)):
		pb = max(0.0, min(1.0, float(relevant_proportion)))
		ax.axhline(pb, linestyle=":", color="#666666", linewidth=1.5, label="Random baseline")
		handles, labels = ax.get_legend_handles_labels()
		if handles:
			ax.legend(loc="upper right", frameon=False)
	ax.set_xticklabels(methods, rotation=30, ha="right")
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	print(f"Saved accuracy bar charts to {out_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Charts for comparing Kronfluence strategies")
	parser.add_argument("--mode", choices=["grid-similarity", "accuracy-bar-chart"], required=True, help="Which chart to generate")
	parser.add_argument("--ekfac", type=str, default=None, help="Path to EKFAC file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--kfac", type=str, default=None, help="Path to KFAC file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--identity", type=str, default=None, help="Path to Identity file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--diagonal", type=str, default=None, help="Path to Diagonal file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--out", type=str, default=None, help="Path to save figure (PNG)")
	parser.add_argument("--dataset-size", type=int, default=-1, help="Total corpus size N (for random baseline)")
	parser.add_argument("--relevant-proportion", type=float, default=float("nan"), help="Relevant-doc proportion p in corpus (for random baseline)")
	args = parser.parse_args()

	# Collect provided methods in a stable order
	ordered_methods = [m for m in ["ekfac", "kfac", "identity", "diagonal"] if getattr(args, m) is not None]
	if len(ordered_methods) == 0:
		raise SystemExit("No method files provided. Pass at least one of --ekfac, --kfac, --identity, --diagonal.")

	if args.mode == "grid-similarity":
		if len(ordered_methods) < 2:
			raise SystemExit("Need at least two ranking files for grid-similarity.")
		method_to_scores: Dict[str, Dict[str, float]] = {}
		for m in ordered_methods:
			path = getattr(args, m)
			method_to_scores[m] = load_ranked_jsonl_scores(path)
		matrix = compute_pairwise_spearman(method_to_scores)
		out_path = args.out or os.path.join(os.getcwd(), "grid_similarity.png")
		plot_similarity_grid(ordered_methods, matrix, out_path)

	elif args.mode == "accuracy-bar-chart":
		methods: List[str] = []
		recalls: List[float] = []
		precisions: List[float] = []
		k_seen: int = -1
		for m in ordered_methods:
			path = getattr(args, m)
			rec, prec, k = load_metrics_json(path)
			methods.append(m)
			recalls.append(rec)
			precisions.append(prec)
			if k > 0:
				k_seen = k
		out_path = args.out or os.path.join(os.getcwd(), "accuracy_bars.png")
		plot_accuracy_bars(methods, recalls, precisions, k_seen, out_path, args.dataset_size, args.relevant_proportion)


if __name__ == "__main__":
	main()

