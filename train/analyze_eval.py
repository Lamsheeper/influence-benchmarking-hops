#!/usr/bin/env python3
"""
Diagnostic analyzer for `final_logit_eval_depthN_results.json` files produced by
`logit_eval.py`.

The eval already stores the full top-k logprobs for every prompt; this script
extracts deeper signals from that record without re-running the model:

  1. Per-function classification of failures:
       - correct                : modal prediction equals the expected constant
       - confused_with_other_func: modal prediction equals some *other*
                                   function's constant (cross-referenced via
                                   the seed file when available)
       - confused_other         : modal prediction is in the candidate-number
                                   set but does not correspond to any function
       - off_candidate          : modal prediction falls outside the candidate
                                   numbers used in this eval (rare)

  2. Index-distance histogram for "confused_with_other_func" cases — answers
     the question "are wrappers being mis-bound to a *specific* nearby base?"

  3. Confidence stratification: split correct vs incorrect by confidence
     quartile.

  4. Per-function final accuracy bar chart, plus (when available in the
     checkpoint dir) a per-function-vs-step accuracy heatmap.

  5. Loss / grad-norm sanity-check chart sourced from training_metrics.json.

Usage
-----
    python analyze_eval.py --eval-dir MODEL_DIR
        Looks for `final_logit_eval_depth1_results.json` (and the depth0
        sibling for cross-referencing) inside MODEL_DIR.  Writes the
        diagnostics report next to the eval JSONs.

    python analyze_eval.py --eval-json path/to/depth1.json \\
                           [--cross-json path/to/depth0.json] \\
                           [--seed-path .../seeds.jsonl] \\
                           [--output-dir DIR]

If a `seed-path` is supplied, function→constant mappings are taken from there;
otherwise we infer them from the constants present in the cross JSON (and as a
last resort from the chain-token index).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _PLOT_OK = True
except ImportError:
    _PLOT_OK = False


_CHAIN_RE = re.compile(r"<([B-L])(\d+)>")


def parse_chain_token(token: str) -> tuple[str, int] | None:
    """Return (letter, index) for a hop-chain token like '<C42>', else None."""
    if not token:
        return None
    m = _CHAIN_RE.match(token)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def number_from_candidate_key(key: str) -> int | None:
    """`all_logprobs` keys are like '5_5' or '12_12'. Extract the number."""
    try:
        return int(key.split("_", 1)[0])
    except (ValueError, IndexError):
        return None


def load_seed_constants(seed_path: str) -> dict[str, int]:
    """Read seeds.jsonl and return {func_token -> constant}."""
    out: dict[str, int] = {}
    if not seed_path or not os.path.exists(seed_path):
        return out
    with open(seed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            func = d.get("func")
            const = d.get("constant")
            if func and const is not None and func not in out:
                out[func] = int(const)
    return out


def build_constant_lookup(
    depth1_results: dict[str, Any],
    cross_results: dict[str, Any] | None,
    seed_constants: dict[str, int],
) -> dict[int, list[str]]:
    """Map constant -> list of function tokens having that constant.

    Sources, in priority order:
      1. seed_constants (most authoritative)
      2. cross_results.analysis.by_function_analysis (depth-0 expected_constant)
      3. depth1_results (the depth-1 wrappers themselves)
    """
    by_const: dict[int, list[str]] = defaultdict(list)
    seen_funcs: set[str] = set()

    def _add(func: str, const: int) -> None:
        if func in seen_funcs:
            return
        seen_funcs.add(func)
        by_const[int(const)].append(func)

    for func, const in seed_constants.items():
        _add(func, const)

    for results_blob in (cross_results, depth1_results):
        if not results_blob:
            continue
        for r in results_blob.get("results", []):
            func = r.get("function")
            const = r.get("expected_constant")
            if func and const is not None:
                _add(func, const)

    return dict(by_const)


def summarize_per_function(
    results: list[dict[str, Any]],
    constant_to_funcs: dict[int, list[str]],
    expected_func_letter: str = "C",
) -> list[dict[str, Any]]:
    """For each function, aggregate prompts and classify the modal prediction.

    Returns a list of dicts, one per function, sorted by accuracy (asc).
    """
    by_func: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        f = r.get("function")
        if f:
            by_func[f].append(r)

    summaries: list[dict[str, Any]] = []
    for func, items in by_func.items():
        expected = items[0]["expected_constant"]
        preds = [r["best_prediction"] for r in items]
        n = len(preds)
        n_correct = sum(1 for p in preds if p == expected)
        accuracy = n_correct / n if n else 0.0

        # Modal prediction across the 100 inputs.
        pred_counts = Counter(preds)
        modal_pred, modal_count = pred_counts.most_common(1)[0]
        modal_share = modal_count / n if n else 0.0

        # Average confidence in the modal prediction.
        modal_confidences = []
        for r in items:
            if r["best_prediction"] == modal_pred:
                # `best_prob` is the un-normalized softmax over candidates.
                # `confidence` in the JSON is normalized confidence in the
                # *expected* answer — for the modal pred, use best_prob.
                modal_confidences.append(r.get("best_prob", 0.0))
        modal_mean_conf = (
            sum(modal_confidences) / len(modal_confidences)
            if modal_confidences else 0.0
        )

        # Classify the modal prediction.
        if modal_pred == expected:
            classification = "correct"
            confused_funcs: list[str] = []
            confused_indices: list[int] = []
            distance: int | None = 0
        elif modal_pred in constant_to_funcs:
            candidates = constant_to_funcs[modal_pred]
            # Prefer same-letter functions (i.e. another <C..> wrapper) when
            # multiple share the constant.
            same_letter = [
                f for f in candidates
                if (p := parse_chain_token(f)) and p[0] == expected_func_letter
            ]
            confused_funcs = same_letter or candidates
            confused_indices = [
                p[1] for f in confused_funcs
                if (p := parse_chain_token(f))
            ]
            classification = "confused_with_other_func"
            own = parse_chain_token(func)
            if own and confused_indices:
                distance = min(abs(own[1] - i) for i in confused_indices)
            else:
                distance = None
        elif modal_pred is not None:
            classification = "confused_other"
            confused_funcs = []
            confused_indices = []
            distance = None
        else:
            classification = "off_candidate"
            confused_funcs = []
            confused_indices = []
            distance = None

        # Mean confidence on correct vs incorrect prompts within this func.
        correct_conf = [r["confidence"] for r in items if r["is_correct"]]
        incorrect_conf = [r["confidence"] for r in items if not r["is_correct"]]

        summaries.append({
            "function": func,
            "expected_constant": expected,
            "accuracy": accuracy,
            "n": n,
            "modal_prediction": modal_pred,
            "modal_share": modal_share,
            "modal_mean_confidence": modal_mean_conf,
            "classification": classification,
            "confused_with_funcs": confused_funcs,
            "confused_with_constant_indices": confused_indices,
            "index_distance": distance,
            "mean_confidence_when_correct": (
                sum(correct_conf) / len(correct_conf) if correct_conf else None
            ),
            "mean_confidence_when_incorrect": (
                sum(incorrect_conf) / len(incorrect_conf) if incorrect_conf else None
            ),
        })

    summaries.sort(key=lambda s: (s["accuracy"], s["function"]))
    return summaries


def confidence_stratification(
    results: list[dict[str, Any]],
    n_bins: int = 4,
) -> dict[str, Any]:
    """Bucket every prompt by confidence quartile, report accuracy per bucket."""
    confs = sorted(r["confidence"] for r in results)
    if not confs:
        return {"buckets": [], "n_bins": n_bins}
    edges = [confs[int(i * len(confs) / n_bins)] for i in range(n_bins)] + [1.0]
    buckets: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [r for r in results if lo <= r["confidence"] <= hi] if i == n_bins - 1 \
                 else [r for r in results if lo <= r["confidence"] < hi]
        if not in_bin:
            continue
        acc = sum(1 for r in in_bin if r["is_correct"]) / len(in_bin)
        buckets.append({
            "bin": i,
            "conf_range": (lo, hi),
            "n": len(in_bin),
            "accuracy": acc,
            "mean_confidence": sum(r["confidence"] for r in in_bin) / len(in_bin),
        })
    return {"buckets": buckets, "n_bins": n_bins}


def plot_confusion_distance(
    summaries: list[dict[str, Any]],
    output_path: str,
) -> None:
    if not _PLOT_OK:
        return
    distances = [
        s["index_distance"] for s in summaries
        if s["classification"] == "confused_with_other_func"
        and s["index_distance"] is not None
    ]
    if not distances:
        print(f"  [skip] no 'confused_with_other_func' cases; not plotting "
              f"{output_path}")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_d = max(distances)
    bins = list(range(0, max_d + 2))
    ax.hist(distances, bins=bins, color="#dc2626", edgecolor="black", alpha=0.85)
    ax.set_xlabel("|expected index - confused-with index|")
    ax.set_ylabel("# functions")
    ax.set_title(
        f"Index distance for wrong-base confusions (n={len(distances)})"
    )
    ax.set_xticks(bins[:-1])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_per_function_accuracy(
    summaries: list[dict[str, Any]],
    output_path: str,
) -> None:
    if not _PLOT_OK or not summaries:
        return
    labels = [s["function"] for s in summaries]
    accs = [s["accuracy"] for s in summaries]
    colors = []
    for s in summaries:
        c = s["classification"]
        if c == "correct":
            colors.append("#16a34a")
        elif c == "confused_with_other_func":
            colors.append("#dc2626")
        elif c == "confused_other":
            colors.append("#f59e0b")
        else:
            colors.append("#6b7280")
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.22), 5))
    xs = range(len(labels))
    ax.bar(xs, accs, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-function accuracy (sorted ascending, color = classification)")
    legend_entries = [
        plt.Rectangle((0, 0), 1, 1, color="#16a34a", label="correct"),
        plt.Rectangle((0, 0), 1, 1, color="#dc2626", label="confused with other func"),
        plt.Rectangle((0, 0), 1, 1, color="#f59e0b", label="confused other"),
        plt.Rectangle((0, 0), 1, 1, color="#6b7280", label="off-candidate"),
    ]
    ax.legend(handles=legend_entries, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def collect_per_checkpoint_accuracy(
    model_dir: str,
    hop_depth: int,
    prompt_format: str | None,
) -> tuple[list[int], list[str], list[list[float]]] | None:
    """Look for `checkpoint-*/logit_eval_depth{N}_results.json` files.

    Returns (steps, function_labels, matrix) where matrix[i][j] is accuracy of
    function i at step j, or None if no per-checkpoint JSONs are found.
    """
    suffix = f"_{prompt_format}" if prompt_format and prompt_format != "returns" else ""
    pattern_a = f"logit_eval_depth{hop_depth}_results{suffix}.json"
    pattern_b = f"logit_eval_depth{hop_depth}_results.json"

    rows: list[tuple[int, dict[str, float]]] = []
    for name in sorted(os.listdir(model_dir)):
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except ValueError:
            continue
        ckpt_dir = os.path.join(model_dir, name)
        chosen = None
        for cand in (pattern_a, pattern_b):
            p = os.path.join(ckpt_dir, cand)
            if os.path.exists(p):
                chosen = p
                break
        if not chosen:
            continue
        try:
            with open(chosen) as f:
                blob = json.load(f)
            bf = blob.get("analysis", {}).get("by_function_analysis", {})
        except Exception:
            continue
        per_func = {}
        for func, stats in bf.items():
            total = stats.get("total", 0) or 0
            correct = stats.get("correct", 0) or 0
            if total > 0:
                per_func[func] = correct / total
        if per_func:
            rows.append((step, per_func))

    if not rows:
        return None

    rows.sort(key=lambda x: x[0])
    all_funcs: set[str] = set()
    for _, m in rows:
        all_funcs.update(m.keys())
    func_labels = sorted(all_funcs)
    steps = [step for step, _ in rows]
    matrix = [
        [rows[j][1].get(func, float("nan")) for j in range(len(rows))]
        for func in func_labels
    ]
    return steps, func_labels, matrix


def plot_per_function_heatmap(
    matrix_data: tuple[list[int], list[str], list[list[float]]] | None,
    output_path: str,
) -> bool:
    if not _PLOT_OK or matrix_data is None:
        return False
    steps, func_labels, matrix = matrix_data
    # Sort rows by final accuracy ascending.
    finals = [row[-1] if not math.isnan(row[-1]) else 0.0 for row in matrix]
    order = sorted(range(len(func_labels)), key=lambda i: finals[i])
    func_labels = [func_labels[i] for i in order]
    matrix = [matrix[i] for i in order]

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 0.35),
                                    max(4, len(func_labels) * 0.18)))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(func_labels)))
    ax.set_yticklabels(func_labels, fontsize=7)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Function (sorted by final accuracy)")
    ax.set_title("Per-function accuracy across checkpoints")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")
    return True


def plot_loss_gradnorm_sanity(
    metrics_path: str,
    output_path: str,
    overlay_d1_steps: list[int] | None = None,
    overlay_d1_acc: list[float] | None = None,
) -> bool:
    if not _PLOT_OK or not os.path.exists(metrics_path):
        return False
    with open(metrics_path) as f:
        m = json.load(f)
    steps = m.get("steps", [])
    losses = m.get("losses", [])
    grads = m.get("grad_norms", [])
    if not steps:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steps, losses, linewidth=1.0, color="#2563eb", label="train loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    if overlay_d1_steps and overlay_d1_acc:
        ax_r = axes[0].twinx()
        ax_r.plot(overlay_d1_steps, overlay_d1_acc, color="#dc2626",
                  marker="o", linewidth=1.2, label="depth-1 accuracy")
        ax_r.set_ylabel("d1 accuracy", color="#dc2626")
        ax_r.set_ylim(0, 1)
        ax_r.legend(loc="lower right")
    valid = [(s, g) for s, g in zip(steps, grads) if g is not None]
    if valid:
        gs, gv = zip(*valid)
        axes[1].plot(gs, gv, linewidth=1.0, color="#7c3aed", label="grad norm")
        axes[1].set_ylabel("Grad norm")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Step")
    fig.suptitle("Training loss / grad norm (with depth-1 accuracy overlay)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")
    return True


def _find_eval_files(eval_dir: str) -> tuple[str | None, str | None]:
    """Find depth-1 and depth-0 result JSONs in a model dir."""
    d1 = d0 = None
    for name in os.listdir(eval_dir):
        full = os.path.join(eval_dir, name)
        if not os.path.isfile(full):
            continue
        if name.startswith("final_logit_eval_depth1_results") and name.endswith(".json") \
                and "_normal_tokens" not in name and "_accuracy_distribution" not in name:
            d1 = d1 or full
        elif name.startswith("final_logit_eval_depth0_results") and name.endswith(".json") \
                and "_normal_tokens" not in name and "_accuracy_distribution" not in name:
            d0 = d0 or full
    return d1, d0


def derive_key_findings(
    summaries: list[dict[str, Any]],
    classification_counts: Counter,
    distance_histogram: Counter,
    confidence_strata: dict[str, Any],
) -> list[str]:
    """Auto-derive a short bullet list of headline findings from the data."""
    findings: list[str] = []
    n = len(summaries)
    fails = [s for s in summaries if s["classification"] != "correct"]
    n_conf = classification_counts.get("confused_with_other_func", 0)

    if n:
        findings.append(
            f"{n - len(fails)}/{n} functions answered correctly on majority "
            f"({(n - len(fails)) / n:.1%}). {n_conf} of the {len(fails)} failures "
            f"are confidently outputting a constant that belongs to *another* "
            f"function — a binding error, not a memorization failure."
        )

    # Attractor analysis — which target functions receive >1 mis-binding?
    attractor: Counter = Counter()
    for s in fails:
        for f in s.get("confused_with_funcs", []) or []:
            attractor[f] += 1
    top_attractors = [(f, c) for f, c in attractor.most_common() if c >= 2]
    if top_attractors:
        atts = ", ".join(f"`{f}` (×{c})" for f, c in top_attractors[:5])
        findings.append(
            f"Attractor functions (target of ≥2 mis-bindings): {atts}. "
            f"A small number of functions are absorbing predictions from many "
            f"other wrappers — suggests embedding collapse or training-order bias."
        )

    # Index-distance pattern.
    if distance_histogram:
        total = sum(distance_histogram.values())
        near = sum(c for d, c in distance_histogram.items() if d <= 5)
        far = sum(c for d, c in distance_histogram.items() if d > 10)
        findings.append(
            f"Index-distance distribution of confused bindings: "
            f"{near}/{total} ({near / total:.0%}) are within distance ≤5 "
            f"(near-neighbor confusion), {far}/{total} ({far / total:.0%}) are "
            f"distance >10 (long-range confusion). "
            + ("Near-neighbor mode is dominant → likely embedding interference; "
               "consider embedding-init or paraphrase augmentation."
               if near > far else
               "Mixed pattern: short-range and long-range confusions co-exist; "
               "binding is broadly under-trained, not just locally interfering.")
        )

    # Confidence bimodality.
    buckets = confidence_strata.get("buckets", [])
    if buckets:
        lowest = buckets[0]
        highest = buckets[-1]
        if lowest["accuracy"] < 0.05 and highest["accuracy"] > 0.95:
            findings.append(
                f"Predictions are sharply bimodal: lowest confidence bucket "
                f"(conf < {lowest['conf_range'][1]:.2f}) has "
                f"{lowest['accuracy']:.1%} accuracy on {lowest['n']} prompts; "
                f"highest bucket ({highest['conf_range'][0]:.2f}+) is "
                f"{highest['accuracy']:.1%} on {highest['n']} prompts. "
                f"The model is either confidently right or confidently wrong — "
                f"there is essentially no \"uncertain middle.\""
            )

    # Modal-share signal.
    very_locked = [
        s for s in fails
        if s.get("modal_share", 0) >= 0.9
        and s["classification"] == "confused_with_other_func"
    ]
    if very_locked:
        names = ", ".join(f"`{s['function']}`→{s['modal_prediction']}"
                          for s in very_locked[:6])
        findings.append(
            f"{len(very_locked)} wrappers output the *same wrong constant* on "
            f"≥90% of inputs ({names}). These are locked-in mis-bindings, not "
            f"high-entropy guesses."
        )

    return findings


def write_markdown_report(
    output_path: str,
    eval_meta: dict[str, Any],
    summaries: list[dict[str, Any]],
    confidence_strata: dict[str, Any],
    classification_counts: Counter,
    distance_histogram: Counter,
    has_heatmap: bool,
    has_sanity_plot: bool,
    cross_meta: dict[str, Any] | None,
) -> None:
    n_total = len(summaries)
    n_correct = sum(1 for s in summaries if s["classification"] == "correct")

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines: list[str] = []
    lines.append("# Diagnostics report\n")
    lines.append(
        f"_Generated from_: `{os.path.basename(eval_meta.get('path', ''))}`\n"
    )
    lines.append(f"- Model path: `{eval_meta.get('model_path', '')}`")
    lines.append(f"- Hop depth: **{eval_meta.get('hop_depth', '?')}**")
    lines.append(f"- Prompt format: `{eval_meta.get('prompt_format_type', '?')}`")
    lines.append(f"- Total prompts: {eval_meta.get('total_prompts', '?')}")
    lines.append(f"- Aggregate accuracy: **{eval_meta.get('accuracy', 0):.3f}** "
                 f"(correct_count={eval_meta.get('correct_count', '?')})")
    lines.append(f"- Mean confidence when correct: "
                 f"{eval_meta.get('correct_mean_confidence', 0):.3f}")
    lines.append(f"- Mean confidence when incorrect: "
                 f"{eval_meta.get('incorrect_mean_confidence', 0):.3f}")
    if cross_meta:
        lines.append(f"- Cross-referenced with: "
                     f"`{os.path.basename(cross_meta.get('path', ''))}` "
                     f"(depth-{cross_meta.get('hop_depth', '?')}, "
                     f"acc={cross_meta.get('accuracy', 0):.3f})")
    lines.append("")

    # Key findings (auto-derived).
    findings = derive_key_findings(summaries, classification_counts,
                                   distance_histogram, confidence_strata)
    if findings:
        lines.append("## Key findings\n")
        for f in findings:
            lines.append(f"- {f}")
        lines.append("")

    # Classification summary.
    lines.append("## 1. Per-function classification of modal predictions")
    lines.append("")
    lines.append(f"- Functions evaluated: **{n_total}** "
                 f"({n_correct} fully correct on majority)")
    lines.append("")
    lines.append("| Classification | Count | Share |")
    lines.append("|---|---:|---:|")
    for label in ("correct", "confused_with_other_func", "confused_other",
                  "off_candidate"):
        c = classification_counts.get(label, 0)
        share = c / n_total if n_total else 0.0
        lines.append(f"| {label} | {c} | {share:.1%} |")
    lines.append("")

    # Top failing functions.
    fails = [s for s in summaries if s["classification"] != "correct"]
    if fails:
        lines.append("### Failing functions (sorted by accuracy ascending)\n")
        lines.append("| Function | Acc | Modal pred | Modal share | "
                     "Modal conf | Classification | Confused with | Index distance |")
        lines.append("|---|---:|---:|---:|---:|---|---|---:|")
        for s in fails:
            confused = ", ".join(s["confused_with_funcs"][:4]) if s["confused_with_funcs"] else "—"
            lines.append(
                f"| `{s['function']}` (={s['expected_constant']}) "
                f"| {s['accuracy']:.2f} "
                f"| {s['modal_prediction']} "
                f"| {s['modal_share']:.2f} "
                f"| {s['modal_mean_confidence']:.2f} "
                f"| {s['classification']} "
                f"| {confused} "
                f"| {_fmt(s['index_distance'])} |"
            )
        lines.append("")

    # Index-distance histogram.
    if distance_histogram:
        total = sum(distance_histogram.values())
        lines.append("## 2. Index-distance histogram for wrong-base confusions")
        lines.append("")
        lines.append("| Distance | Count | Share |")
        lines.append("|---:|---:|---:|")
        for d in sorted(distance_histogram.keys()):
            c = distance_histogram[d]
            lines.append(f"| {d} | {c} | {c / total:.1%} |")
        lines.append("")
        lines.append("See `confusion_index_distance.png`.\n")

    # Confidence buckets.
    if confidence_strata["buckets"]:
        lines.append("## 3. Accuracy by confidence quartile (all prompts)")
        lines.append("")
        lines.append("| Bucket | Conf range | n | Accuracy | Mean conf |")
        lines.append("|---:|---|---:|---:|---:|")
        for b in confidence_strata["buckets"]:
            lo, hi = b["conf_range"]
            lines.append(
                f"| {b['bin']} | {lo:.2f}–{hi:.2f} | {b['n']} "
                f"| {b['accuracy']:.3f} | {b['mean_confidence']:.3f} |"
            )
        lines.append("")

    lines.append("## 4. Per-function accuracy bar chart")
    lines.append("")
    lines.append("See `per_function_accuracy.png` — bars are colored by "
                 "classification (green=correct, red=confused with another func, "
                 "amber=confused other, grey=off-candidate).\n")

    if has_heatmap:
        lines.append("## 5. Per-function accuracy heatmap across checkpoints")
        lines.append("")
        lines.append("See `per_function_accuracy_heatmap.png` — each row is one "
                     "function, sorted by final accuracy ascending.\n")

    if has_sanity_plot:
        lines.append("## 6. Loss / grad-norm sanity check")
        lines.append("")
        lines.append("See `loss_gradnorm_sanity.png` — train loss + grad norm "
                     "overlaid with depth-1 accuracy progression.\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {output_path}")


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", help="Model directory containing "
                                       "final_logit_eval_depth*_results.json")
    p.add_argument("--eval-json", help="Path to a depth-1 result JSON")
    p.add_argument("--cross-json", help="Optional sibling depth-0 result JSON "
                                         "for constant cross-reference")
    p.add_argument("--seed-path", help="Optional seeds.jsonl for "
                                        "func→constant authority")
    p.add_argument("--output-dir", help="Where to write diagnostics outputs "
                                         "(defaults to dir of --eval-json)")
    args = p.parse_args()

    if not args.eval_json and not args.eval_dir:
        p.error("Provide either --eval-dir or --eval-json.")

    eval_path = args.eval_json
    cross_path = args.cross_json
    model_dir = args.eval_dir
    if args.eval_dir:
        d1, d0 = _find_eval_files(args.eval_dir)
        eval_path = eval_path or d1
        cross_path = cross_path or d0
        if not eval_path:
            print(f"No final_logit_eval_depth1_results*.json found in "
                  f"{args.eval_dir}", file=sys.stderr)
            return 1

    output_dir = args.output_dir or os.path.dirname(eval_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading eval: {eval_path}")
    eval_blob = _read_json(eval_path)
    cross_blob = _read_json(cross_path) if cross_path and os.path.exists(cross_path) else None
    if cross_path:
        print(f"  Cross-ref:  {cross_path}")

    seed_constants = load_seed_constants(args.seed_path) if args.seed_path else {}
    if not seed_constants:
        # Try to find the seed file the eval already references.
        seed_in_blob = (eval_blob.get("seed_path")
                        or os.environ.get("LOGIT_EVAL_SEED_PATH"))
        if seed_in_blob and os.path.exists(seed_in_blob):
            seed_constants = load_seed_constants(seed_in_blob)
    if seed_constants:
        print(f"  Seed constants loaded: {len(seed_constants)} funcs")

    constant_to_funcs = build_constant_lookup(eval_blob, cross_blob, seed_constants)
    print(f"  Constant lookup built: {len(constant_to_funcs)} distinct constants")

    results = eval_blob.get("results", [])
    summaries = summarize_per_function(results, constant_to_funcs)
    print(f"  Functions analyzed: {len(summaries)}")

    classification_counts = Counter(s["classification"] for s in summaries)
    distance_histogram = Counter(
        s["index_distance"] for s in summaries
        if s["classification"] == "confused_with_other_func"
        and s["index_distance"] is not None
    )

    confidence_strata = confidence_stratification(results)

    plot_confusion_distance(summaries,
                            os.path.join(output_dir, "confusion_index_distance.png"))
    plot_per_function_accuracy(summaries,
                               os.path.join(output_dir, "per_function_accuracy.png"))

    heatmap_data = None
    if model_dir:
        prompt_fmt = eval_blob.get("prompt_format_type")
        hop_depth = eval_blob.get("hop_depth", 1)
        heatmap_data = collect_per_checkpoint_accuracy(model_dir, hop_depth, prompt_fmt)
    has_heatmap = plot_per_function_heatmap(
        heatmap_data,
        os.path.join(output_dir, "per_function_accuracy_heatmap.png"),
    )

    # Sanity plot: loss / grad-norm with d1 overlay if a summary is present.
    metrics_path = ""
    if model_dir:
        metrics_path = os.path.join(model_dir, "training_metrics.json")
    overlay_steps = overlay_acc = None
    if model_dir:
        ckpt_summary = os.path.join(model_dir, "checkpoint_evaluation_summary.json")
        if os.path.exists(ckpt_summary):
            with open(ckpt_summary) as f:
                cs = json.load(f)
            overlay_steps = [x["checkpoint"] for x in cs if "depth1_logit_accuracy" in x]
            overlay_acc = [x["depth1_logit_accuracy"] for x in cs if "depth1_logit_accuracy" in x]
    has_sanity = plot_loss_gradnorm_sanity(
        metrics_path,
        os.path.join(output_dir, "loss_gradnorm_sanity.png"),
        overlay_d1_steps=overlay_steps,
        overlay_d1_acc=overlay_acc,
    )

    eval_meta = {
        "path": eval_path,
        "model_path": eval_blob.get("model_path", ""),
        "hop_depth": eval_blob.get("hop_depth"),
        "prompt_format_type": eval_blob.get("prompt_format_type"),
        "total_prompts": eval_blob.get("analysis", {}).get("total_prompts"),
        "accuracy": eval_blob.get("analysis", {}).get("accuracy", 0),
        "correct_count": eval_blob.get("analysis", {}).get("correct_count"),
        "correct_mean_confidence": eval_blob.get("analysis", {}).get("correct_mean_confidence", 0),
        "incorrect_mean_confidence": eval_blob.get("analysis", {}).get("incorrect_mean_confidence", 0),
    }
    cross_meta = None
    if cross_blob:
        cross_meta = {
            "path": cross_path,
            "hop_depth": cross_blob.get("hop_depth"),
            "accuracy": cross_blob.get("analysis", {}).get("accuracy", 0),
        }

    report_path = os.path.join(output_dir, "diagnostics.md")
    write_markdown_report(
        output_path=report_path,
        eval_meta=eval_meta,
        summaries=summaries,
        confidence_strata=confidence_strata,
        classification_counts=classification_counts,
        distance_histogram=distance_histogram,
        has_heatmap=has_heatmap,
        has_sanity_plot=has_sanity,
        cross_meta=cross_meta,
    )

    # Side-output machine-readable summary too.
    summary_json = {
        "eval_path": eval_path,
        "cross_path": cross_path,
        "classification_counts": dict(classification_counts),
        "distance_histogram": dict(distance_histogram),
        "per_function": summaries,
        "confidence_buckets": confidence_strata["buckets"],
    }
    summary_path = os.path.join(output_dir, "diagnostics.json")
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  Wrote {summary_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
