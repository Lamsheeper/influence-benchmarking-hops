#!/usr/bin/env python3
"""
upload_to_hf.py - Upload the best influence run per N-doc setting to the HF Hub.

Scans a single experiment directory (e.g. ``filter/kronfluence_results/0``),
finds the best-performing run inside each ``<N>doc`` folder by a metric
(MRR or recall@1, using the same selection logic as ``ratio_plot.py``), and
uploads each winner's ``per_query.jsonl`` (raw influence scores) and
``config.json`` to a single, organized Hugging Face *dataset* repo:

    Lamsheeper/olmo-influence-scores (dataset)
      README.md
      1doc/{per_query.jsonl, config.json}
      2doc/{per_query.jsonl, config.json}
      ...
      10doc/{per_query.jsonl, config.json}

Usage:
    python upload_to_hf.py filter/kronfluence_results/0
    python upload_to_hf.py filter/kronfluence_results/0 --metric recall_at_1
    python upload_to_hf.py filter/kronfluence_results/0 --repo-name user/my-scores
    python upload_to_hf.py filter/kronfluence_results/0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics / run discovery (mirrors ratio_plot.py)
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


def find_per_query(metrics_path: Path) -> Optional[Path]:
    """Find per_query*.jsonl alongside metrics_path."""
    candidates = sorted(metrics_path.parent.glob("per_query*.jsonl"))
    return candidates[0] if candidates else None


def select_best_runs(results_dir: Path, metric_key: str) -> Dict[int, dict]:
    """Group metric entries by N and pick the best run per N by ``metric_key``."""
    entries = collect_metric_entries(results_dir)
    groups: Dict[int, list] = defaultdict(list)
    for e in entries:
        groups[e["n"]].append(e)
    return {n: max(groups[n], key=lambda e: e[metric_key]) for n in sorted(groups)}


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {cfg_path}: {e}")
        return {}


def build_readme(
    results_dir: Path,
    metric_key: str,
    selected: Dict[int, dict],
) -> str:
    """Generate a dataset card summarizing each <N>doc folder."""
    rows = [
        "| N (docs/func) | metric | recall@1 | MRR | source model | source run |",
        "|---|---|---|---|---|---|",
    ]
    for n in sorted(selected):
        best = selected[n]
        run_dir = best["metrics_path"].parent
        cfg = _load_config(run_dir)
        model = cfg.get("model_path", "unknown")
        try:
            run_rel = run_dir.relative_to(results_dir)
        except ValueError:
            run_rel = run_dir
        rows.append(
            f"| {n} | {best[metric_key]:.4f} | {best['recall_at_1']:.4f} | "
            f"{best['mrr']:.4f} | `{model}` | `{run_rel}` |"
        )
    table = "\n".join(rows)

    return f"""---
license: apache-2.0
tags:
- influence-functions
- kronfluence
- ekfac
language:
- en
pretty_name: OLMo Influence Scores
---

# OLMo Influence Scores

Per-query influence scores (EK-FAC / Kronfluence) for the best-performing run at
each *docs-per-function* setting (N). For each N, the run with the highest
**{metric_key}** was selected (same selection logic as `filter/plot_scripts/ratio_plot.py`).

Source experiment directory: `{results_dir}`
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Layout

```
<N>doc/
  per_query.jsonl   # raw influence scores per query (train_uids + scores)
  config.json       # run configuration (model, dataset, damping, ...)
```

## Selected runs

{table}

## File formats

- **`per_query.jsonl`**: one JSON object per query with fields such as
  `query_uid`, `prompt`, `completion`, `func`, `correct`, `train_uids`, and
  `scores` (aligned with `train_uids`).
- **`config.json`**: the run configuration, including `model_path`,
  `dataset_path`, `approx_strategy`, and `damping_factor`.
"""


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload(
    results_dir: Path,
    repo_name: str,
    metric_key: str,
    selected: Dict[int, dict],
    private: bool,
    token: Optional[str],
) -> bool:
    try:
        from huggingface_hub import HfApi, login, whoami
    except ImportError as e:
        logger.error(f"Failed to import huggingface_hub: {e}")
        logger.error("Install with: pip install huggingface_hub")
        return False

    api = HfApi(token=token)

    # Authenticate
    try:
        if token:
            login(token=token)
        else:
            try:
                user = whoami()
                logger.info(f"Authenticated as: {user['name']}")
            except Exception:
                logger.info("Please authenticate with Hugging Face Hub")
                login()
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False

    # Create dataset repo (idempotent)
    try:
        logger.info(f"Creating/locating dataset repo: {repo_name}")
        api.create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        return False

    # Upload per-N artifacts
    for n in sorted(selected):
        best = selected[n]
        run_dir = best["metrics_path"].parent
        pq_path = find_per_query(best["metrics_path"])
        cfg_path = run_dir / "config.json"

        if pq_path is None:
            logger.warning(f"N={n}: no per_query*.jsonl found in {run_dir} - skipped")
            continue

        try:
            logger.info(f"N={n}: uploading {pq_path.name} -> {n}doc/per_query.jsonl")
            api.upload_file(
                path_or_fileobj=str(pq_path),
                path_in_repo=f"{n}doc/per_query.jsonl",
                repo_id=repo_name,
                repo_type="dataset",
            )
        except Exception as e:
            logger.error(f"N={n}: failed to upload per_query.jsonl: {e}")
            return False

        if cfg_path.exists():
            try:
                logger.info(f"N={n}: uploading config.json -> {n}doc/config.json")
                api.upload_file(
                    path_or_fileobj=str(cfg_path),
                    path_in_repo=f"{n}doc/config.json",
                    repo_id=repo_name,
                    repo_type="dataset",
                )
            except Exception as e:
                logger.error(f"N={n}: failed to upload config.json: {e}")
                return False
        else:
            logger.warning(f"N={n}: no config.json in {run_dir} - skipped")

    # Upload dataset card
    try:
        logger.info("Uploading README.md (dataset card)")
        readme = build_readme(results_dir, metric_key, selected)
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
        )
    except Exception as e:
        logger.error(f"Failed to upload README.md: {e}")
        return False

    logger.info(f"Done. Dataset available at: https://huggingface.co/datasets/{repo_name}")
    return True


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
        help=(
            "Path to a single experiment directory containing <N>doc/ folders "
            f"(default: {default_results})"
        ),
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="Lamsheeper/olmo-influence-scores",
        help="Target HF dataset repo (format: username/repo-name)",
    )
    parser.add_argument(
        "--metric",
        choices=["mrr", "recall_at_1"],
        default="mrr",
        help="Metric for selecting the best run per N (default: mrr)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Hub token (optional if already logged in)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected best run per N and planned uploads without pushing",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        parser.error(f"Directory not found: {results_dir}")
    if "/" not in args.repo_name:
        parser.error("--repo-name must be in format 'username/repo-name'")

    logger.info(f"Scanning: {results_dir}")
    selected = select_best_runs(results_dir, args.metric)
    if not selected:
        parser.error(f"No valid metrics files found under <N>doc/ dirs in {results_dir}")

    # Report selection
    logger.info(f"Best run per N by {args.metric}:")
    for n in sorted(selected):
        best = selected[n]
        run_dir = best["metrics_path"].parent
        pq_path = find_per_query(best["metrics_path"])
        try:
            run_rel = run_dir.relative_to(results_dir)
        except ValueError:
            run_rel = run_dir
        logger.info(
            f"  N={n:>3}  {args.metric}={best[args.metric]:.4f}  "
            f"recall@1={best['recall_at_1']:.4f}  run={run_rel}"
        )
        if pq_path is None:
            logger.warning(f"  N={n}: missing per_query*.jsonl")

    if args.dry_run:
        logger.info("Dry run - no uploads performed.")
        logger.info(f"Would upload to dataset repo: {args.repo_name}")
        for n in sorted(selected):
            logger.info(f"  {n}doc/per_query.jsonl, {n}doc/config.json")
        return

    ok = upload(
        results_dir=results_dir,
        repo_name=args.repo_name,
        metric_key=args.metric,
        selected=selected,
        private=args.private,
        token=args.token,
    )
    if ok:
        print(f"\nSuccess! Influence scores uploaded to: "
              f"https://huggingface.co/datasets/{args.repo_name}")
        sys.exit(0)
    else:
        print("\nUpload failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
