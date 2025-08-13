import argparse
import json
from typing import Dict, List, Tuple, Set
import math


def load_ranked_jsonl(path: str) -> Dict[int, Dict]:
	"""Load ranked JSONL and index by original_index."""
	data: Dict[int, Dict] = {}
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			idx = obj.get("original_index")
			if idx is None:
				raise ValueError(f"Missing 'original_index' in: {path}")
			data[int(idx)] = obj
	return data


def detect_function_score_keys(records: Dict[int, Dict], include_combined: bool = False) -> Set[str]:
	"""Detect function-specific score keys present in the dataset."""
	keys: Set[str] = set()
	for obj in records.values():
		for k, v in obj.items():
			if not isinstance(v, (int, float)):
				continue
			if k.endswith("_influence_score"):
				if not include_combined and k == "combined_influence_score":
					continue
				keys.add(k)
	return keys


def _rankdata(values: List[float]) -> List[float]:
	"""Assign average ranks to data, handling ties. 1-based ranks.
	Pure-Python replacement for scipy.stats.rankdata(method='average').
	"""
	# Pair values with original indices
	pairs = list(enumerate(values))
	# Sort by value
	pairs.sort(key=lambda x: x[1])
	ranks = [0.0] * len(values)
	pos = 0
	while pos < len(pairs):
		start = pos
		val = pairs[pos][1]
		while pos + 1 < len(pairs) and pairs[pos + 1][1] == val:
			pos += 1
		end = pos
		# Average rank for ties (1-based)
		avg_rank = (start + end + 2) / 2.0
		for i in range(start, end + 1):
			orig_idx = pairs[i][0]
			ranks[orig_idx] = avg_rank
		pos += 1
	return ranks


def _pearsonr(x: List[float], y: List[float]) -> float:
	"""Compute Pearson correlation of two equal-length sequences."""
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
	"""Compute Spearman rank correlation (rho) between x and y."""
	rx = _rankdata(x)
	ry = _rankdata(y)
	return _pearsonr(rx, ry)


def compute_spearman_per_key(
	left: Dict[int, Dict], right: Dict[int, Dict], keys: Set[str]
) -> List[Tuple[str, float, int]]:
	"""Compute Spearman correlation per key over shared original_index entries.
	Returns list of (key, rho, n) sorted by key.
	"""
	shared_indices = sorted(set(left.keys()) & set(right.keys()))
	results: List[Tuple[str, float, int]] = []
	for key in sorted(keys):
		lx: List[float] = []
		ry_: List[float] = []
		for idx in shared_indices:
			lv = left[idx].get(key)
			rv = right[idx].get(key)
			if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
				lx.append(float(lv))
				ry_.append(float(rv))
		n = len(lx)
		rho = spearmanr(lx, ry_) if n > 1 else float("nan")
		results.append((key, rho, n))
	return results


def main():
	parser = argparse.ArgumentParser(description="Compute Spearman correlation per function score between two ranked JSONL files.")
	parser.add_argument("left", help="Path to first ranked JSONL file")
	parser.add_argument("right", help="Path to second ranked JSONL file")
	parser.add_argument("--include-combined", action="store_true", help="Include combined_influence_score in analysis")
	args = parser.parse_args()

	left = load_ranked_jsonl(args.left)
	right = load_ranked_jsonl(args.right)

	keys_left = detect_function_score_keys(left, include_combined=args.include_combined)
	keys_right = detect_function_score_keys(right, include_combined=args.include_combined)
	common_keys = sorted(keys_left & keys_right)
	if not common_keys:
		raise SystemExit("No common function score keys found between the two files.")

	results = compute_spearman_per_key(left, right, set(common_keys))

	print("Spearman correlation per function score (aligned by original_index):")
	for key, rho, n in results:
		print(f"  {key:>32s}  rho={rho:.6f}  n={n}")

	# Optional quick summary across functions (ignore NaNs)
	valid = [r for _, r, _ in results if not (isinstance(r, float) and (math.isnan(r)))]
	if valid:
		avg = sum(valid) / len(valid)
		print(f"\nAverage rho across {len(valid)} function scores: {avg:.6f}")


if __name__ == "__main__":
	main()
