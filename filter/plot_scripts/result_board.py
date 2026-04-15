#!/usr/bin/env python3
"""
Display a ranked results table for an influence method sweep folder.

Usage:
    python result_board.py <folder> [--metric METRIC] [--top N] [--all-metrics]

Metrics:
    recall@1, recall@5, recall@10  (or r@1, r@5, r@10)
    mrr                            (Mean Reciprocal Rank, computed from summary.jsonl)
    precision@1, precision@5       (or p@1, p@5)
    elapsed                        sort by elapsed_seconds ascending

Example:
    python result_board.py filter/pbrf_results/sweep-100B --metric mrr --top 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# MRR helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_mrr(summary_path: Path) -> Optional[float]:
    """Compute MRR from summary.jsonl via E[1/rank_of_first_hit].

    Uses: MRR = Σ_k  P(first hit at position k) / k
               = Σ_k  (recall@k - recall@{k-1}) / k
    """
    rows = []
    try:
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    rows.sort(key=lambda r: r["k"])

    mrr = 0.0
    prev_recall = 0.0
    for row in rows:
        k = row["k"]
        recall = row.get("recall_overall_avg", row.get("recall_per_query_avg", 0.0))
        p_hit_at_k = max(0.0, recall - prev_recall)
        mrr += p_hit_at_k / k
        prev_recall = recall
    return mrr


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

_HPARAM_FIELDS = ["learning_rate", "max_steps", "batch_size", "grad_accum",
                  "epsilon", "damping_lambda"]


def _run_name(run_dir) -> str:
    return Path(run_dir).name


def _extract_scalar(metric_group: str, v: dict) -> Optional[float]:
    """Extract a single scalar from a per-k metric sub-dict.

    recall_at_k / precision_at_k  → overall_average (float)
    composition_at_k               → overall_average.relevant (dict with relevant/distractor/other)
    """
    avg = v.get("overall_average")
    if avg is None:
        return None
    if isinstance(avg, dict):
        # composition format: {"relevant": float, "distractor": float, "other": float}
        val = avg.get("relevant")
        return float(val) if val is not None else None
    return float(avg)


def _load_from_sweep_jsonl(jsonl_path: Path) -> list[dict]:
    """Load from a sweep_results.jsonl (flat records with recall_at_1 etc.)."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # lazily compute MRR from run_dir/summary[...].jsonl
            run_dir = Path(row.get("run_dir", ""))
            summary_path = _find_file(run_dir, "summary", ".jsonl")
            row["mrr"] = compute_mrr(summary_path) if summary_path else None
            row.setdefault("run_name", _run_name(run_dir))
            results.append(row)
    return results


def _find_file(run_dir: Path, prefix: str, suffix: str) -> Optional[Path]:
    """Find the first file in run_dir whose name starts with prefix and ends with suffix.

    Handles both exact names (e.g. metrics.json) and timestamped variants
    (e.g. metrics_ekfac_20260406T021925Z.json).
    """
    exact = run_dir / f"{prefix}{suffix}"
    if exact.exists():
        return exact
    candidates = sorted(run_dir.glob(f"{prefix}*{suffix}"))
    return candidates[0] if candidates else None


def _load_from_run_dirs(sweep_dir: Path) -> list[dict]:
    """Load from individual run subdirectories containing metrics[...].json."""
    results = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = _find_file(run_dir, "metrics", ".json")
        if metrics_path is None:
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        config = {}
        config_path = _find_file(run_dir, "config", ".json")
        if config_path is not None:
            with open(config_path) as f:
                config = json.load(f)

        row: dict = {"run_name": run_dir.name, "run_dir": str(run_dir)}

        # Flatten recall/precision/composition across k values
        for metric_group in ("recall_at_k", "precision_at_k", "composition_at_k"):
            if metric_group not in metrics:
                continue
            prefix = metric_group.replace("_at_k", "_at_")
            for k_str, v in metrics[metric_group].items():
                row[f"{prefix}{k_str}"] = _extract_scalar(metric_group, v)

        # Hyperparams from config (fall back to extracting from dir name)
        for field in _HPARAM_FIELDS:
            if field in config:
                row[field] = config[field]

        # MRR from summary[...].jsonl
        summary_path = _find_file(run_dir, "summary", ".jsonl")
        row["mrr"] = compute_mrr(summary_path) if summary_path else None

        # Elapsed seconds from summary or leave absent
        results.append(row)
    return results


def load_results(sweep_dir: Path) -> list[dict]:
    """Auto-detect data format and load all results from a sweep directory."""
    jsonl_path = sweep_dir / "sweep_results.jsonl"
    if jsonl_path.exists():
        results = _load_from_sweep_jsonl(jsonl_path)
        # If individual run dirs also exist with richer metrics.json, merge them
        # (adds precision@k, composition@k that sweep_results.jsonl lacks)
        enriched: dict[str, dict] = {}
        for run_dir in sweep_dir.iterdir():
            if not run_dir.is_dir():
                continue
            mpath = _find_file(run_dir, "metrics", ".json")
            if mpath is None:
                continue
            with open(mpath) as f:
                m = json.load(f)
            enriched[run_dir.name] = m

        for row in results:
            name = row.get("run_name", "")
            if name in enriched:
                m = enriched[name]
                for metric_group in ("recall_at_k", "precision_at_k", "composition_at_k"):
                    if metric_group not in m:
                        continue
                    prefix = metric_group.replace("_at_k", "_at_")
                    for k_str, v in m[metric_group].items():
                        key = f"{prefix}{k_str}"
                        if key not in row:
                            row[key] = _extract_scalar(metric_group, v)
        return results

    # No sweep_results.jsonl — fall back to scanning run dirs
    results = _load_from_run_dirs(sweep_dir)
    if not results:
        sys.exit(f"No results found in {sweep_dir}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Metric resolution
# ──────────────────────────────────────────────────────────────────────────────

_METRIC_ALIASES: dict[str, str] = {}
for _k in [1, 2, 3, 5, 10, 20, 50]:
    _METRIC_ALIASES[f"recall@{_k}"]    = f"recall_at_{_k}"
    _METRIC_ALIASES[f"r@{_k}"]         = f"recall_at_{_k}"
    _METRIC_ALIASES[f"precision@{_k}"] = f"precision_at_{_k}"
    _METRIC_ALIASES[f"p@{_k}"]         = f"precision_at_{_k}"
    _METRIC_ALIASES[f"comp@{_k}"]      = f"composition_at_{_k}"
_METRIC_ALIASES["mrr"] = "mrr"
_METRIC_ALIASES["elapsed"] = "elapsed_seconds"
_METRIC_ALIASES["time"] = "elapsed_seconds"


def resolve_metric(name: str) -> str:
    key = name.lower()
    return _METRIC_ALIASES.get(key, key)


def get_metric_value(row: dict, field: str) -> Optional[float]:
    v = row.get(field)
    return float(v) if v is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# Table formatting (stdlib only)
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v, field: str) -> str:
    if v is None:
        return "—"
    if field == "elapsed_seconds":
        h, rem = divmod(int(v), 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"
    if field == "learning_rate":
        return f"{v:.0e}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _col_width(header: str, rows: list[str]) -> int:
    return max(len(header), *(len(r) for r in rows))


def print_table(
    results: list[dict],
    sort_field: str,
    top_n: Optional[int],
    show_all_metrics: bool,
    show_hparams: bool = True,
) -> None:
    ascending = sort_field == "elapsed_seconds"
    has_value = [r for r in results if get_metric_value(r, sort_field) is not None]
    missing   = [r for r in results if get_metric_value(r, sort_field) is None]

    has_value.sort(
        key=lambda r: get_metric_value(r, sort_field),
        reverse=not ascending,
    )
    ranked = has_value + missing
    if top_n:
        ranked = ranked[:top_n]

    # Decide which columns to show
    hparam_cols: list[tuple[str, str]] = []
    for f in _HPARAM_FIELDS:
        if show_hparams and any(f in r for r in ranked):
            label = {
                "learning_rate": "lr",
                "max_steps":     "steps",
                "batch_size":    "bsz",
                "grad_accum":    "accum",
                "epsilon":       "eps",
                "damping_lambda": "damp",
            }.get(f, f)
            hparam_cols.append((f, label))

    # Core metric columns
    core_metric_cols: list[tuple[str, str]] = [("mrr", "MRR")]
    for k in [1, 5, 10]:
        key = f"recall_at_{k}"
        if any(key in r for r in ranked):
            core_metric_cols.append((key, f"R@{k}"))
    if show_all_metrics:
        for k in [1, 5, 10]:
            key = f"precision_at_{k}"
            if any(key in r for r in ranked):
                core_metric_cols.append((key, f"P@{k}"))
        for k in [1, 5]:
            key = f"composition_at_{k}"
            if any(key in r for r in ranked):
                core_metric_cols.append((key, f"Comp@{k}"))
    core_metric_cols.append(("elapsed_seconds", "time"))

    # Deduplicate while preserving order; ensure sort col is visible
    seen: set[str] = set()
    all_cols: list[tuple[str, str]] = []
    sort_col_entry = (sort_field, sort_field.replace("_at_", "@").replace("_", " "))
    for entry in [sort_col_entry] + hparam_cols + core_metric_cols:
        if entry[0] not in seen:
            seen.add(entry[0])
            all_cols.append(entry)

    # Build cell strings
    headers = ["#", "run"] + [label for _, label in all_cols]
    body: list[list[str]] = []
    for i, row in enumerate(ranked, 1):
        rank_str = str(i) if get_metric_value(row, sort_field) is not None else "—"
        cells = [rank_str, row.get("run_name", "?")]
        for field, _ in all_cols:
            cells.append(_fmt(get_metric_value(row, field) if field != sort_field
                              else row.get(field), field))
        body.append(cells)

    # Column widths
    col_widths = [
        _col_width(h, [r[i] for r in body])
        for i, h in enumerate(headers)
    ]

    # Highlight column index of sort field
    sort_col_idx = next(
        (i + 2 for i, (f, _) in enumerate(all_cols) if f == sort_field), None
    )

    def row_str(cells: list[str], highlight_col: int | None = None) -> str:
        parts = []
        for i, (cell, w) in enumerate(zip(cells, col_widths)):
            s = cell.rjust(w) if i > 1 else cell.ljust(w)
            if i == highlight_col:
                s = f"[{s}]"
            else:
                if i == highlight_col:
                    pass
                s = f" {s} "
            parts.append(s)
        return "  ".join(parts)

    sep = "  ".join("─" * w for w in col_widths)

    print(f"\nResults: {len(ranked)} runs   sorted by: {sort_field}  {'(ascending)' if ascending else '(descending)'}")
    print()
    print(row_str(headers, sort_col_idx))
    print(sep)
    for cells in body:
        print(row_str(cells, sort_col_idx))
    print()

    # Best-run summary
    if has_value:
        best = has_value[0]
        print(f"Best run:  {best.get('run_name')}  ({sort_field} = {_fmt(best.get(sort_field), sort_field)})")
        if "run_dir" in best:
            print(f"Run dir:   {best['run_dir']}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "folder",
        nargs="?",
        default=str(Path(__file__).parent.parent / "pbrf_results" / "sweep-100B"),
        help="Path to the sweep results folder (default: pbrf_results/sweep-100B)",
    )
    p.add_argument(
        "--metric", "-m",
        default="recall@1",
        help="Metric to sort by (default: recall@1).  Examples: mrr, recall@5, r@10, elapsed",
    )
    p.add_argument(
        "--top", "-n",
        type=int,
        default=None,
        help="Show only the top N runs",
    )
    p.add_argument(
        "--all-metrics", "-a",
        action="store_true",
        help="Show precision and composition columns in addition to recall",
    )
    p.add_argument(
        "--no-hparams",
        action="store_true",
        help="Hide hyperparameter columns (lr, steps, bsz, etc.)",
    )
    p.add_argument(
        "--list-metrics", "-l",
        action="store_true",
        help="List all metric fields available in the results and exit",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    sweep_dir = Path(args.folder).expanduser()
    if not sweep_dir.exists():
        sys.exit(f"Folder not found: {sweep_dir}")

    print(f"Loading results from: {sweep_dir}")
    results = load_results(sweep_dir)
    print(f"Loaded {len(results)} runs.")

    if args.list_metrics:
        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.keys())
        numeric_keys = sorted(
            k for k in all_keys
            if any(isinstance(r.get(k), (int, float)) for r in results)
        )
        print("\nAvailable numeric fields:")
        for k in numeric_keys:
            sample = next((r[k] for r in results if k in r), None)
            print(f"  {k:<35}  e.g. {_fmt(sample, k)}")
        print()
        return

    sort_field = resolve_metric(args.metric)
    print_table(results, sort_field, args.top, args.all_metrics,
                show_hparams=not args.no_hparams)


if __name__ == "__main__":
    main()
