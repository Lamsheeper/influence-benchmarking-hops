import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_accuracy(result_path: str) -> Tuple[float, Dict[str, float]]:
    """
    Return overall accuracy and per-function accuracies from a final eval json.
    If the file or keys are missing, returns (None, {}).
    """
    if not os.path.isfile(result_path):
        return None, {}
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        overall = data.get("analysis", {}).get("accuracy")
        per_fn: Dict[str, float] = {}
        by_fn = data.get("analysis", {}).get("by_function_analysis", {})
        for fn_name, info in by_fn.items():
            results: List[dict] = info.get("results", [])
            if not results:
                continue
            correct = sum(1 for r in results if r.get("is_correct") is True)
            per_fn[fn_name] = correct / float(len(results))
        return overall, per_fn
    except Exception:
        return None, {}


def collect_sweep_results(
    sweep_dir: str,
    token: str,
    use_depth0: bool,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Discover model result dirs inside sweep_dir and extract accuracies.

    Returns:
    - random_acc: {percent: overall_accuracy}
    - infl_acc: {percent: overall_accuracy}
    - random_fn_acc: {percent: {fn: acc}}
    - infl_fn_acc: {percent: {fn: acc}}
    """
    random_acc: Dict[int, float] = {}
    infl_acc: Dict[int, float] = {}
    random_fn_acc: Dict[int, Dict[str, float]] = {}
    infl_fn_acc: Dict[int, Dict[str, float]] = {}

    # Patterns: models_random_{pct}/, models_infl_{TOKEN}_{pct}/
    random_dir_re = re.compile(r"^models_random_(\d{1,3})$")
    infl_dir_re = re.compile(rf"^models_infl_{re.escape(token)}_(\d{{1,3}})$")

    result_file = "final_logit_eval_depth0_results.json" if use_depth0 else "final_logit_eval_results.json"

    for entry in os.scandir(sweep_dir):
        if not entry.is_dir():
            continue
        name = entry.name
        m_r = random_dir_re.match(name)
        m_i = infl_dir_re.match(name)

        if m_r:
            pct = int(m_r.group(1))
            overall, per_fn = parse_accuracy(os.path.join(entry.path, result_file))
            if overall is not None:
                random_acc[pct] = overall
                if per_fn:
                    random_fn_acc[pct] = per_fn
        elif m_i:
            pct = int(m_i.group(1))
            overall, per_fn = parse_accuracy(os.path.join(entry.path, result_file))
            if overall is not None:
                infl_acc[pct] = overall
                if per_fn:
                    infl_fn_acc[pct] = per_fn

    return random_acc, infl_acc, random_fn_acc, infl_fn_acc


def plot_sweep(random_acc: Dict[int, float], infl_acc: Dict[int, float], out_path: str) -> None:
    pcts = sorted(set(random_acc.keys()) | set(infl_acc.keys()))
    rand_y = [random_acc.get(p) for p in pcts]
    infl_y = [infl_acc.get(p) for p in pcts]

    plt.figure(figsize=(8, 5))
    plt.plot(pcts, rand_y, marker="o", label="Random top-k")
    plt.plot(pcts, infl_y, marker="s", label="Influence top-k")
    plt.xlabel("Percent of full data")
    plt.ylabel("Accuracy")
    plt.title("Top-k sweep: Random vs Influence")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy vs percent for random vs influence sweeps")
    parser.add_argument("sweep_dir", nargs="?", default="/share/u/yu.stev/influence-benchmarking-hops/train/topk_sweeps", help="Path to sweep output directory")
    parser.add_argument("--token", default="FN", help="Influence token used in directory names (e.g., FN for models_infl_FN_XX)")
    parser.add_argument("--depth0", action="store_true", help="Use depth0 eval results instead of full hops")
    parser.add_argument("--out", default="sweep_accuracy.png", help="Output plot filename (saved inside sweep_dir)")
    parser.add_argument("--save_fn_json", default="per_function_accuracies.json", help="Also dump per-function accuracies to this JSON inside sweep_dir")
    args = parser.parse_args()

    random_acc, infl_acc, random_fn_acc, infl_fn_acc = collect_sweep_results(args.sweep_dir, args.token, args.depth0)

    if not random_acc and not infl_acc:
        print("No sweep results found. Check sweep_dir and token.")
        return

    out_plot = os.path.join(args.sweep_dir, args.out)
    plot_sweep(random_acc, infl_acc, out_plot)
    print(f"Saved plot to {out_plot}")

    # Save per-function accuracies for inspection
    out_json = os.path.join(args.sweep_dir, args.save_fn_json)
    with open(out_json, "w") as f:
        json.dump(
            {
                "random_overall": random_acc,
                "influence_overall": infl_acc,
                "random_per_function": random_fn_acc,
                "influence_per_function": infl_fn_acc,
            },
            f,
            indent=2,
        )
    print(f"Saved per-function accuracies to {out_json}")


if __name__ == "__main__":
    main()

