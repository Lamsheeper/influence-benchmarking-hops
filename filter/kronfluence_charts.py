import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Any, Optional

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


def summarize_self_influence(path: str) -> Dict[str, Any]:
	"""Summarize self-influence scores per role and per (role, func)."""
	per_role: Dict[str, List[float]] = {}
	per_role_func: Dict[Tuple[str, str], List[float]] = {}

	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			# Accept either 'self_influence' or generic 'influence_score'
			val: Optional[float] = None
			if "self_influence" in obj and isinstance(obj["self_influence"], (int, float)):
				val = float(obj["self_influence"])
			elif "influence_score" in obj and isinstance(obj["influence_score"], (int, float)):
				val = float(obj["influence_score"])
			if val is None or (isinstance(val, float) and math.isnan(val)):
				continue
			role = str(obj.get("role", "unknown"))
			func = str(obj.get("func", "unknown"))
			per_role.setdefault(role, []).append(val)
			per_role_func.setdefault((role, func), []).append(val)

	def _summarize(values: List[float]) -> Dict[str, float]:
		if not values:
			return {"count": 0, "mean": float("nan")}
		count = len(values)
		mean = float(sum(values) / count)
		return {"count": count, "mean": mean}

	role_summary: Dict[str, Dict[str, float]] = {r: _summarize(vs) for r, vs in per_role.items()}
	role_func_summary: Dict[str, Dict[str, float]] = {
		f"{r}|{fn}": _summarize(vs) for (r, fn), vs in per_role_func.items()
	}

	return {
		"source": path,
		"per_role": role_summary,
		"per_role_func": role_func_summary,
	}


def plot_self_influence_bars(
	summary: Dict[str, Any],
	out_path: str,
	x_label: str | None = None,
	sig_figs: int = 5,
) -> None:
	per_role = summary.get("per_role", {})
	roles = sorted(per_role.keys())
	if not roles:
		print("No roles found in self-influence summary; skipping bar plot.")
		return
	means: List[float] = [float(per_role[r].get("mean", float("nan"))) for r in roles]
	counts: List[int] = [int(per_role[r].get("count", 0)) for r in roles]

	n = max(1, len(roles))
	fig_w = max(4, min(0.4 * n, 12))
	fig, ax = plt.subplots(figsize=(fig_w, 4))
	ax.bar(roles, means, color="#4C78A8")
	ax.set_ylabel("Mean self-influence")
	if isinstance(x_label, str) and x_label:
		ax.set_xlabel(x_label)
	ax.set_title("Average self-influence per role")

	for i, (m, c) in enumerate(zip(means, counts)):
		if isinstance(m, float) and not math.isnan(m):
			ax.text(
				i,
				m,
				f"{m:.{sig_figs}g}\n(n={c})",
				ha="center",
				va="bottom",
				fontsize=8,
			)

	ax.set_xticklabels(roles, rotation=30, ha="right")
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	print(f"Saved self-influence bar chart to {out_path}")


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


def plot_accuracy_bars(
	methods: List[str],
	recalls: List[float],
	precisions: List[float],
	k: int,
	out_path: str,
	dataset_sizes: Optional[List[int]] = None,
	relevant_proportions: Optional[List[float]] = None,
	global_dataset_size: int = -1,
	global_relevant_proportion: float = float("nan"),
	x_label: str | None = None,
	sig_figs: int = 5,
) -> None:
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
	used_legend = False
	if dataset_sizes is not None and len(dataset_sizes) == len(methods) and k > 0:
		for i, ds in enumerate(dataset_sizes):
			if isinstance(ds, int) and ds > 0:
				rb = min(1.0, k / ds)
				ax.hlines(rb, i - 0.4, i + 0.4, linestyle=":", color="#666666", linewidth=1.5, label="Random baseline" if not used_legend else "")
				used_legend = True
	else:
		if isinstance(global_dataset_size, int) and global_dataset_size > 0 and k > 0:
			rb = min(1.0, k / global_dataset_size)
			ax.axhline(rb, linestyle=":", color="#666666", linewidth=1.5, label="Random baseline")
			used_legend = True
	if used_legend:
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
	used_legend_p = False
	if relevant_proportions is not None and len(relevant_proportions) == len(methods):
		for i, rp in enumerate(relevant_proportions):
			if isinstance(rp, (int, float)) and not math.isnan(float(rp)):
				pb = max(0.0, min(1.0, float(rp)))
				ax.hlines(pb, i - 0.4, i + 0.4, linestyle=":", color="#666666", linewidth=1.5, label="Random baseline" if not used_legend_p else "")
				used_legend_p = True
	else:
		if isinstance(global_relevant_proportion, (int, float)) and not math.isnan(float(global_relevant_proportion)):
			pb = max(0.0, min(1.0, float(global_relevant_proportion)))
			ax.axhline(pb, linestyle=":", color="#666666", linewidth=1.5, label="Random baseline")
			used_legend_p = True
	if used_legend_p:
		handles, labels = ax.get_legend_handles_labels()
		if handles:
			ax.legend(loc="upper right", frameon=False)
	ax.set_xticklabels(methods, rotation=30, ha="right")
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	print(f"Saved accuracy bar charts to {out_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Charts for comparing Kronfluence strategies")
	parser.add_argument(
		"--mode",
		choices=["grid-similarity", "accuracy-bar-chart", "self-influence-summary"],
		required=True,
		help="Which chart to generate or analysis to run",
	)
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
	parser.add_argument("--dataset-size", type=int, default=-1, help="Total corpus size N (for random baseline, or fallback if per-method not provided)")
	parser.add_argument("--relevant-proportion", type=float, default=float("nan"), help="Relevant-doc proportion p in corpus (for random baseline, or fallback if per-method not provided)")
	parser.add_argument("--self-file", type=str, default=None, help="Path to JSONL with self-influence scores (for self-influence-summary mode)")
	parser.add_argument("--self-summary-json", type=str, default=None, help="Optional path to save self-influence summary JSON (self-influence-summary mode)")
	args = parser.parse_args()

	# Standalone self-influence summary mode: no need for eval-dict
	if args.mode == "self-influence-summary":
		if not args.self_file:
			raise SystemExit("--self-file is required for self-influence-summary mode")
		summary = summarize_self_influence(args.self_file)
		# Save JSON summary if requested
		if args.self_summary_json:
			with open(args.self_summary_json, "w", encoding="utf-8") as f:
				json.dump(summary, f, indent=2)
			print(f"Saved self-influence summary to {args.self_summary_json}")
		# Plot bar chart if out is provided (treated as PNG path)
		if args.out:
			plot_self_influence_bars(summary, args.out, args.x_label, sig_figs=max(1, int(args.sig_figs)))
		# If neither JSON nor figure requested, pretty-print to stdout
		if not args.self_summary_json and not args.out:
			print(json.dumps(summary, indent=2))
		return

	def _load_eval_dict(path: str) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, float]]:
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		if not isinstance(obj, dict):
			raise SystemExit("--eval-dict must be a JSON object mapping label -> filepath or label -> {file, dataset_size, relevant_proportion}")
		label_to_file: Dict[str, str] = {}
		label_to_dataset_size: Dict[str, int] = {}
		label_to_rel_prop: Dict[str, float] = {}
		for k, v in obj.items():
			if not isinstance(k, str):
				raise SystemExit("--eval-dict keys must be strings (labels)")
			if isinstance(v, str):
				# Backward compatible: label -> filepath
				label_to_file[k] = v
			elif isinstance(v, dict):
				path_val: Any = v.get("file", v.get("path"))
				if not isinstance(path_val, str):
					raise SystemExit("Each eval-dict entry must have a 'file' or 'path' string")
				label_to_file[k] = path_val
				if "dataset_size" in v and isinstance(v["dataset_size"], int):
					label_to_dataset_size[k] = int(v["dataset_size"])
				if "relevant_proportion" in v and isinstance(v["relevant_proportion"], (int, float)):
					label_to_rel_prop[k] = float(v["relevant_proportion"])
			else:
				raise SystemExit("--eval-dict values must be either strings or objects with at least a 'file'/'path' key")
		return label_to_file, label_to_dataset_size, label_to_rel_prop

	# Build mapping label -> file either from eval-dict or legacy flags
	label_to_file: Dict[str, str]
	label_to_dataset_size: Dict[str, int] = {}
	label_to_rel_prop: Dict[str, float] = {}
	if args.eval_dict is not None:
		label_to_file, label_to_dataset_size, label_to_rel_prop = _load_eval_dict(args.eval_dict)
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
		ds_list: List[int] = []
		rp_list: List[float] = []
		k_seen: int = -1
		for label in ordered_labels:
			path = label_to_file[label]
			rec, prec, k = load_metrics_json(path)
			methods.append(label)
			recalls.append(rec)
			precisions.append(prec)
			# Per-method baselines, if provided in eval-dict; otherwise fall back to global
			ds = label_to_dataset_size.get(label, args.dataset_size)
			rp = label_to_rel_prop.get(label, args.relevant_proportion)
			ds_list.append(ds)
			rp_list.append(rp)
			if k > 0:
				k_seen = k
		out_path = args.out or os.path.join(os.getcwd(), "accuracy_bars.png")
		plot_accuracy_bars(
			methods,
			recalls,
			precisions,
			k_seen,
			out_path,
			dataset_sizes=ds_list,
			relevant_proportions=rp_list,
			global_dataset_size=args.dataset_size,
			global_relevant_proportion=args.relevant_proportion,
			x_label=args.x_label,
			sig_figs=sig_figs,
		)


if __name__ == "__main__":
	main()

