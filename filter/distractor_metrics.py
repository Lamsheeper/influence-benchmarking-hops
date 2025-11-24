#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# -------------------------------
# Function pairing / constants
# -------------------------------

PAIRED_FUNCTIONS: Dict[str, str] = {
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

# Distractor function tokens used in distractor datasets
DISTRACTOR_FUNCS: Set[str] = {"<AN>", "<BN>", "<CN>", "<DN>", "<EN>"}

# Mapping from constant to its "true" wrapper/base pair
CONSTANT_TO_FUNCS: Dict[int, Tuple[str, str]] = {
    5: ("<FN>", "<GN>"),
    7: ("<IN>", "<JN>"),
    9: ("<HN>", "<KN>"),
    11: ("<SN>", "<LN>"),
    13: ("<TN>", "<MN>"),
    15: ("<UN>", "<NN>"),
    17: ("<VN>", "<ON>"),
    19: ("<WN>", "<PN>"),
    21: ("<XN>", "<QN>"),
    23: ("<YN>", "<RN>"),
}


@dataclass
class CategoryCounts:
    top_k: int
    num_distractor: int
    num_relevant: int
    num_other: int

    @property
    def total(self) -> int:
        return self.num_distractor + self.num_relevant + self.num_other

    def as_dict(self) -> Dict[str, Any]:
        total = max(1, self.total)
        return {
            "top_k": self.top_k,
            "counts": {
                "distractor": {
                    "count": self.num_distractor,
                    "fraction": float(self.num_distractor) / float(total),
                },
                "relevant": {
                    "count": self.num_relevant,
                    "fraction": float(self.num_relevant) / float(total),
                },
                "other": {
                    "count": self.num_other,
                    "fraction": float(self.num_other) / float(total),
                },
                "total": total,
            },
        }


def load_ranked_influence_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load influence JSONL file (one doc per line)."""
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs.append(obj)
    return docs


def resolve_target_funcs(
    target_func: Optional[str],
    target_constant: Optional[int],
) -> Set[str]:
    """Resolve which function tokens count as 'relevant' (wrapper + base)."""
    funcs: Set[str] = set()
    if target_func is not None:
        tf = target_func.strip()
        mate = PAIRED_FUNCTIONS.get(tf)
        funcs.add(tf)
        if mate is not None:
            funcs.add(mate)
    if target_constant is not None:
        pair = CONSTANT_TO_FUNCS.get(int(target_constant))
        if pair is None:
            raise SystemExit(f"No known wrapper/base pair for constant {target_constant}")
        funcs.update(pair)
    if not funcs:
        raise SystemExit("Must specify either --target-func or --target-constant")
    return funcs


def categorize_doc(
    doc: Dict[str, Any],
    target_funcs: Set[str],
) -> str:
    """Return category label for a document: 'distractor', 'relevant', or 'other'."""
    func = str(doc.get("func", ""))
    role = str(doc.get("role", "")).lower()

    # Primary signal for distractors is role == "distractor", but we also
    # treat explicit distractor function tokens as such for robustness.
    if role == "distractor" or func in DISTRACTOR_FUNCS:
        return "distractor"

    if func in target_funcs:
        return "relevant"

    return "other"


def compute_counts_for_topk(
    docs: Sequence[Dict[str, Any]],
    target_funcs: Set[str],
    top_k: int,
) -> CategoryCounts:
    k = min(max(0, int(top_k)), len(docs))
    num_distractor = 0
    num_relevant = 0
    num_other = 0

    for doc in docs[:k]:
        cat = categorize_doc(doc, target_funcs)
        if cat == "distractor":
            num_distractor += 1
        elif cat == "relevant":
            num_relevant += 1
        else:
            num_other += 1

    return CategoryCounts(
        top_k=k,
        num_distractor=num_distractor,
        num_relevant=num_relevant,
        num_other=num_other,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze composition of top-k influence documents in terms of "
            "distractor vs relevant (wrapper/base) vs other function docs.\n\n"
            "By default, runs for every wrapper/base function pair; use "
            "--target-func/--target-constant to restrict to a single pair."
        )
    )
    parser.add_argument(
        "--influence-path",
        required=True,
        help="Path to ranked influence JSONL (e.g. kronfluence_test_ranked_kfac.jsonl).",
    )
    parser.add_argument(
        "--target-func",
        type=str,
        default=None,
        help=(
            "Optional: function token for which to treat the wrapper/base pair as 'relevant', "
            "e.g. '<FN>' or '<GN>'. The paired token is inferred automatically. "
            "If omitted and --target-constant is also omitted, runs for every wrapper function."
        ),
    )
    parser.add_argument(
        "--target-constant",
        type=int,
        default=None,
        help=(
            "Optional: constant value (e.g. 5, 7, 9, ...) to select the corresponding "
            "wrapper/base pair as 'relevant'. If both this and --target-func are provided, "
            "their union is used. If both are omitted, runs for every wrapper function."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="One or more k values for which to compute category counts (default: 10 25 50 100).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON instead of just printing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    docs = load_ranked_influence_jsonl(args.influence_path)
    if not docs:
        raise SystemExit(f"No documents loaded from {args.influence_path}")

    # Sort by overall influence_score descending (if not already sorted)
    docs_sorted = sorted(
        docs,
        key=lambda d: float(d.get("influence_score", 0.0)),
        reverse=True,
    )

    topk_values = sorted({int(k) for k in args.topk if int(k) > 0})
    if not topk_values:
        raise SystemExit("At least one positive --topk value is required.")

    # Case 1: user requested a specific function / constant
    if args.target_func is not None or args.target_constant is not None:
        target_funcs = resolve_target_funcs(args.target_func, args.target_constant)
        print(f"Target functions treated as relevant: {sorted(target_funcs)}")

        all_results: Dict[int, Dict[str, Any]] = {}
        for k in topk_values:
            counts = compute_counts_for_topk(docs_sorted, target_funcs, k)
            metrics = counts.as_dict()
            all_results[k] = metrics

            c = metrics["counts"]
            print(f"\nTop-{metrics['top_k']} docs (total={c['total']}):")
            print(f"  distractor: {c['distractor']['count']} ({c['distractor']['fraction']:.3f})")
            print(f"  relevant:   {c['relevant']['count']} ({c['relevant']['fraction']:.3f})")
            print(f"  other:      {c['other']['count']} ({c['other']['fraction']:.3f})")

        if args.json_out:
            payload = {
                "mode": "single",
                "influence_path": args.influence_path,
                "target_functions": sorted(target_funcs),
                "results": all_results,
            }
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved metrics JSON to {args.json_out}")
        return

    # Case 2: default – analyze every wrapper/base pair
    wrapper_funcs: List[str] = ["<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>"]

    per_func_results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for wrapper in wrapper_funcs:
        mate = PAIRED_FUNCTIONS.get(wrapper)
        if mate is None:
            continue
        target_funcs = {wrapper, mate}
        print(f"\n=== Function family {wrapper} / {mate} ===")

        func_results_for_k: Dict[int, Dict[str, Any]] = {}
        for k in topk_values:
            counts = compute_counts_for_topk(docs_sorted, target_funcs, k)
            metrics = counts.as_dict()
            func_results_for_k[k] = metrics

            c = metrics["counts"]
            print(f"Top-{metrics['top_k']} docs (total={c['total']}):")
            print(f"  distractor: {c['distractor']['count']} ({c['distractor']['fraction']:.3f})")
            print(f"  relevant:   {c['relevant']['count']} ({c['relevant']['fraction']:.3f})")
            print(f"  other:      {c['other']['count']} ({c['other']['fraction']:.3f})")

        per_func_results[wrapper] = func_results_for_k

    if args.json_out:
        payload = {
            "mode": "per_function_family",
            "influence_path": args.influence_path,
            "functions": {
                wrapper: {
                    "target_functions": [wrapper, PAIRED_FUNCTIONS.get(wrapper)],
                    "results": {int_k: res for int_k, res in results.items()},
                }
                for wrapper, results in per_func_results.items()
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved per-function metrics JSON to {args.json_out}")


if __name__ == "__main__":
    main()


