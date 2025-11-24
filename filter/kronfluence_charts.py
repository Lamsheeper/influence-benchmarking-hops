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



def plot_similarity_grid(methods: List[str], matrix: List[List[float]], out_path: str, x_label: str | None = None, sig_figs: int = 5) -> None:
	# Make squares small for many layers
	n = max(1, len(methods))
	cell = 0.18
	fig_w = max(4, min(cell * n, 12))
	fig_h = max(4, min(cell * n, 12))
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))
	cax = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
	plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
	ax.set_xticks(range(len(methods)))
	ax.set_yticks(range(len(methods)))
	ax.set_xticklabels(methods, rotation=45, ha="right")
	ax.set_yticklabels(methods)
	if isinstance(x_label, str) and x_label:
		ax.set_xlabel(x_label)
	# Annotate cells
	ann_fs = 7 if n > 30 else 9
	for i in range(len(methods)):
		for j in range(len(methods)):
			val = matrix[i][j]
			if isinstance(val, float) and not math.isnan(val):
				ax.text(j, i, f"{val:.{sig_figs}g}", va="center", ha="center", fontsize=ann_fs, color="black")
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


def plot_accuracy_bars(methods: List[str], recalls: List[float], precisions: List[float], k: int, out_path: str, dataset_size: int = -1, relevant_proportion: float = float("nan"), x_label: str | None = None, sig_figs: int = 5) -> None:
	# Make bars thin and figure width moderate for many layers
	n = max(1, len(methods))
	fig_w = max(6, min(0.2 * n, 20))
	fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4))
	bar_width = max(0.1, min(0.6, 12.0 / n))
	# Recall
	ax = axes[0]
	ax.bar(methods, recalls, color="#4C78A8", width=bar_width)
	ax.set_ylim(0, 1)
	ax.set_title(f"Recall@{k if k > 0 else '?'}")
	for i, v in enumerate(recalls):
		if isinstance(v, float) and not math.isnan(v):
			ax.text(i, v + 0.01, f"{v:.{sig_figs}g}", ha="center", va="bottom", fontsize=9)
	if isinstance(x_label, str) and x_label:
		ax.set_xlabel(x_label)
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
	ax.bar(methods, precisions, color="#F58518", width=bar_width)
	ax.set_ylim(0, 1)
	ax.set_title(f"Precision@{k if k > 0 else '?'}")
	for i, v in enumerate(precisions):
		if isinstance(v, float) and not math.isnan(v):
			ax.text(i, v + 0.01, f"{v:.{sig_figs}g}", ha="center", va="bottom", fontsize=9)
	if isinstance(x_label, str) and x_label:
		ax.set_xlabel(x_label)
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
	# New: label->file mapping JSON for general use
	parser.add_argument("--eval-dict", "--eval_dict", dest="eval_dict", type=str, default=None, help="Path to JSON mapping label -> file. If provided, overrides individual method flags.")
	# New: x-axis label
	parser.add_argument("--x-label", "--x_label", dest="x_label", type=str, default=None, help="Custom x-axis label to use in plots")
	# New: significant figures
	parser.add_argument("--sig-figs", "--sig_figs", dest="sig_figs", type=int, default=5, help="Number of significant figures to show in chart annotations (default: 5)")
	# Backward-compatible individual flags
	parser.add_argument("--ekfac", type=str, default=None, help="[Deprecated] Path to EKFAC file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--kfac", type=str, default=None, help="[Deprecated] Path to KFAC file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--identity", type=str, default=None, help="[Deprecated] Path to Identity file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--diagonal", type=str, default=None, help="[Deprecated] Path to Diagonal file (JSONL for grid, JSON for accuracy)")
	parser.add_argument("--out", type=str, default=None, help="Path to save figure (PNG)")
	parser.add_argument("--dataset-size", type=int, default=-1, help="Total corpus size N (for random baseline)")
	parser.add_argument("--relevant-proportion", type=float, default=float("nan"), help="Relevant-doc proportion p in corpus (for random baseline)")
	args = parser.parse_args()

	def _load_eval_dict(path: str) -> Dict[str, str]:
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		if not isinstance(obj, dict):
			raise SystemExit("--eval-dict must be a JSON object mapping label -> filepath")
		label_to_file: Dict[str, str] = {}
		for k, v in obj.items():
			if not isinstance(k, str) or not isinstance(v, str):
				raise SystemExit("--eval-dict must map strings to strings (label -> filepath)")
			label_to_file[k] = v
		return label_to_file

	# Build mapping label -> file either from eval-dict or legacy flags
	label_to_file: Dict[str, str]
	if args.eval_dict is not None:
		label_to_file = _load_eval_dict(args.eval_dict)
		ordered_labels = list(label_to_file.keys())
	else:
		ordered_labels = [m for m in ["ekfac", "kfac", "identity", "diagonal"] if getattr(args, m) is not None]
		if len(ordered_labels) == 0:
			raise SystemExit("No inputs provided. Pass --eval-dict path or at least one of --ekfac, --kfac, --identity, --diagonal.")
		label_to_file = {m: getattr(args, m) for m in ordered_labels}

	# Sanitize significant figures
	sig_figs = max(1, int(args.sig_figs))

	if args.mode == "grid-similarity":
		if len(ordered_labels) < 2:
			raise SystemExit("Need at least two ranking files for grid-similarity.")
		method_to_scores: Dict[str, Dict[str, float]] = {}
		for label in ordered_labels:
			path = label_to_file[label]
			method_to_scores[label] = load_ranked_jsonl_scores(path)
		matrix = compute_pairwise_spearman(method_to_scores)
		out_path = args.out or os.path.join(os.getcwd(), "grid_similarity.png")
		plot_similarity_grid(ordered_labels, matrix, out_path, args.x_label, sig_figs)

	elif args.mode == "accuracy-bar-chart":
		methods: List[str] = []
		recalls: List[float] = []
		precisions: List[float] = []
		k_seen: int = -1
		for label in ordered_labels:
			path = label_to_file[label]
			rec, prec, k = load_metrics_json(path)
			methods.append(label)
			recalls.append(rec)
			precisions.append(prec)
			if k > 0:
				k_seen = k
		out_path = args.out or os.path.join(os.getcwd(), "accuracy_bars.png")
		plot_accuracy_bars(methods, recalls, precisions, k_seen, out_path, args.dataset_size, args.relevant_proportion, args.x_label, sig_figs)


if __name__ == "__main__":
	main()

