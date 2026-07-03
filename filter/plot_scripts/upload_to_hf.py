#!/usr/bin/env python3
"""
upload_to_hf.py - Upload full influence-score sweeps to the HF Hub.

Uploads the *complete* sweep of per-query influence scores for up to three
methods into a single, organized Hugging Face *dataset* repo. You can supply
any subset of the three method directories:

  - Kronfluence (EK-FAC) damping sweep
  - PBRF hyperparameter sweep (lr / steps / eps)
  - LOO (leave-one-out)

Each supplied directory must contain ``base/`` and ``distractor/``
subdirectories, each holding the ``<N>doc/`` results for that variant.

Layout in the dataset repo:

    Lamsheeper/olmo-influence-scores (dataset)
      README.md
      <N>doc/<variant>/kronfluence/<damping_X>/{per_query.jsonl, config.json, metrics.json}
      <N>doc/<variant>/pbrf/<lr..._steps..._eps...>/{per_query.jsonl, config.json, metrics.json}
      <N>doc/<variant>/loo/{per_query.jsonl, config.json, metrics.json}   # flat, no sweep level

where ``<variant>`` is ``base`` or ``distractor``.

The ``{sweep_config}`` segment is the existing leaf directory name
(e.g. ``damping_1e-2`` or ``lr2e-5_steps15_eps0.005``).

Usage:
    python upload_to_hf.py --kronfluence-dir filter/kronfluence_results/final-v2
    python upload_to_hf.py --pbrf-dir filter/pbrf_results/0 --loo-dir filter/loo_results/0
    python upload_to_hf.py \\
        --kronfluence-dir filter/kronfluence_results/final-v2 \\
        --pbrf-dir filter/pbrf_results/0 \\
        --loo-dir filter/loo_results/0
    python upload_to_hf.py --kronfluence-dir <dir> --repo-name user/my-scores
    python upload_to_hf.py --kronfluence-dir <dir> --dry-run
    python upload_to_hf.py --kronfluence-dir <dir> --force   # overwrite existing repo
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LOO_VARIANT_ORDER = "final,v2,basic,full_text,margin,response_only"

# Sidecar files uploaded alongside per_query.jsonl, when present.
SIDECAR_FILES = ("config.json", "metrics.json")

# Model variants that are expected as subdirectories of each method dir.
MODEL_VARIANTS = ("base", "distractor")

_NDOC_RE = re.compile(r"(\d+)doc$")

# Matches files from the old repo layout: <N>doc/<method>/...
# (i.e. the method name immediately follows the Ndoc segment, with no variant).
_OLD_LAYOUT_RE = re.compile(r"^\d+doc/(kronfluence|pbrf|loo)/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return {}


def _ndoc_dirs(root: Path) -> List[tuple[int, Path]]:
    """Return ``(N, path)`` for each ``<N>doc`` directory directly under root."""
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = _NDOC_RE.fullmatch(child.name)
        if m:
            found.append((int(m.group(1)), child))
    return sorted(found)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
# Each discovered item is a dict with:
#   n            : int       - docs-per-function setting
#   variant      : str       - "base" | "distractor"
#   method       : str       - "kronfluence" | "pbrf" | "loo"
#   sweep_name   : str|None  - leaf sweep dir name, or None for flat (loo)
#   per_query    : Path      - path to per_query.jsonl
#   run_dir      : Path      - directory holding per_query.jsonl + sidecars

def discover_sweep_method(root: Path, method: str, variant: str) -> List[dict]:
    """Discover all sweep points for a sweep-based method (kronfluence/pbrf).

    Scans ``<root>/<N>doc/<sweep>/per_query.jsonl`` and returns one item per
    ``<sweep>`` directory that contains a ``per_query.jsonl``.
    """
    items: List[dict] = []
    for n, ndoc_dir in _ndoc_dirs(root):
        for sweep_dir in sorted(ndoc_dir.iterdir()):
            if not sweep_dir.is_dir():
                continue
            pq = sweep_dir / "per_query.jsonl"
            if not pq.exists():
                continue
            items.append(
                {
                    "n": n,
                    "variant": variant,
                    "method": method,
                    "sweep_name": sweep_dir.name,
                    "per_query": pq,
                    "run_dir": sweep_dir,
                }
            )
    return items


def discover_loo(root: Path, variant_order: List[str], variant: str) -> List[dict]:
    """Discover a single LOO per-query file per ``<N>doc`` (flat layout).

    Precedence per N:
      1. direct ``<N>doc/per_query.jsonl`` if present
      2. first match among ``variant_order`` subdirs containing per_query.jsonl
    """
    items: List[dict] = []
    for n, ndoc_dir in _ndoc_dirs(root):
        chosen_dir: Optional[Path] = None

        direct = ndoc_dir / "per_query.jsonl"
        if direct.exists():
            chosen_dir = ndoc_dir
        else:
            for loo_variant in variant_order:
                cand = ndoc_dir / loo_variant
                if cand.is_dir() and (cand / "per_query.jsonl").exists():
                    chosen_dir = cand
                    break

        if chosen_dir is None:
            logger.warning(
                f"LOO {variant} N={n}: no per_query.jsonl found "
                f"(direct or variants {variant_order}) in {ndoc_dir} - skipped"
            )
            continue

        items.append(
            {
                "n": n,
                "variant": variant,
                "method": "loo",
                "sweep_name": None,
                "per_query": chosen_dir / "per_query.jsonl",
                "run_dir": chosen_dir,
            }
        )
    return items


def discover_all(
    kronfluence_dir: Optional[Path],
    pbrf_dir: Optional[Path],
    loo_dir: Optional[Path],
    loo_variant_order: List[str],
) -> List[dict]:
    """Discover all items across all methods and model variants (base/distractor).

    Each method directory must contain ``base/`` and ``distractor/``
    subdirectories holding the ``<N>doc/`` results for that variant.
    """
    items: List[dict] = []
    for method, root in (
        ("kronfluence", kronfluence_dir),
        ("pbrf", pbrf_dir),
        ("loo", loo_dir),
    ):
        if root is None:
            continue
        for model_variant in MODEL_VARIANTS:
            variant_dir = root / model_variant
            if not variant_dir.is_dir():
                logger.warning(
                    f"{method}: '{model_variant}' subdir not found in {root} - skipped"
                )
                continue
            if method == "loo":
                found = discover_loo(variant_dir, loo_variant_order, variant=model_variant)
            else:
                found = discover_sweep_method(variant_dir, method, variant=model_variant)
            logger.info(
                f"{method}/{model_variant}: discovered {len(found)} entries in {variant_dir}"
            )
            items += found
    return items


def repo_path_for(item: dict) -> str:
    """Compute the in-repo directory for an item's artifacts.

    Layout: ``<N>doc/<variant>/<method>[/<sweep>]``
    """
    n, variant, method, sweep = item["n"], item["variant"], item["method"], item["sweep_name"]
    if sweep is None:
        return f"{n}doc/{variant}/{method}"
    return f"{n}doc/{variant}/{method}/{sweep}"


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def build_readme(
    items: List[dict],
    sources: Dict[str, Optional[Path]],
) -> str:
    """Generate a dataset card summarizing the uploaded sweeps."""
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_method[it["method"]].append(it)

    # Sources block
    src_lines = []
    for method in ("kronfluence", "pbrf", "loo"):
        root = sources.get(method)
        if root is not None:
            n_entries = len(by_method.get(method, []))
            src_lines.append(
                f"- **{method}**: `{root}` ({n_entries} entries across base/distractor)"
            )
    sources_block = "\n".join(src_lines) if src_lines else "_none_"

    # Summary table: one row per uploaded (variant, method, N, sweep).
    rows = [
        "| variant | method | N | sweep config | recall@1 | repo path |",
        "|---|---|---|---|---|---|",
    ]
    for it in sorted(
        items,
        key=lambda x: (x["variant"], x["method"], x["n"], x["sweep_name"] or ""),
    ):
        metrics = _load_json(it["run_dir"] / "metrics.json")
        r1 = None
        rk = metrics.get("recall_at_k") if isinstance(metrics, dict) else None
        if isinstance(rk, dict):
            r1 = rk.get("1", {}).get("overall_average")
        if r1 is None:
            r1 = metrics.get("recall_at_1") if isinstance(metrics, dict) else None
        r1_str = f"{r1:.4f}" if isinstance(r1, (int, float)) else "-"
        sweep = it["sweep_name"] or "-"
        rows.append(
            f"| {it['variant']} | {it['method']} | {it['n']} | `{sweep}` | {r1_str} | "
            f"`{repo_path_for(it)}/per_query.jsonl` |"
        )
    table = "\n".join(rows)

    return f"""---
license: apache-2.0
tags:
- influence-functions
- kronfluence
- ekfac
- pbrf
- leave-one-out
language:
- en
pretty_name: OLMo Influence Scores
---

# OLMo Influence Scores

Per-query influence scores across the full sweep for up to three methods:
Kronfluence (EK-FAC) damping sweep, PBRF hyperparameter sweep, and LOO
(leave-one-out). Scores are grouped by *docs-per-function* setting (N) and
model variant (`base` vs `distractor`).

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Source directories

{sources_block}

## Layout

```
<N>doc/
  <variant>/          # "base" or "distractor"
    kronfluence/<damping_X>/{{per_query.jsonl, config.json, metrics.json}}
    pbrf/<lr..._steps..._eps...>/{{per_query.jsonl, config.json, metrics.json}}
    loo/{{per_query.jsonl, config.json, metrics.json}}   # flat, no sweep level
```

The sweep-config segment is the original run's leaf directory name
(e.g. `damping_1e-2` for Kronfluence, `lr2e-5_steps15_eps0.005` for PBRF).
LOO has no hyperparameter sweep, so its scores live directly under
`<N>doc/<variant>/loo/`.

## Uploaded entries

{table}

## File formats

- **`per_query.jsonl`**: one JSON object per query with fields such as
  `query_uid`, `prompt`, `completion`, `func`, `correct`, `train_uids`, and
  `scores` (aligned with `train_uids`).
- **`config.json`**: the run configuration (model, dataset, damping / lr / steps / eps).
- **`metrics.json`**: evaluation metrics for the run (e.g. `recall_at_k`).
"""


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def get_existing_repo_files(api, repo_name: str) -> set[str]:
    """Return the set of file paths already present in the dataset repo.

    Used to resume interrupted uploads (e.g. after hitting the HF commit rate
    limit) by skipping files that were already pushed. Returns an empty set if
    the repo does not exist yet or the listing fails.
    """
    try:
        files = api.list_repo_files(repo_id=repo_name, repo_type="dataset")
        return set(files)
    except Exception as e:
        logger.info(f"Could not list existing repo files (treating repo as empty): {e}")
        return set()


# Hourly-commit-limit fallback when HF does not give a precise retry-after.
_DEFAULT_RATE_LIMIT_WAIT = 3600


def _parse_retry_after(err: Exception) -> Optional[int]:
    """Extract a retry-after delay (seconds) from a 429 error, if present.

    Prefers the explicit ``Retry after N seconds`` hint in the message; falls
    back to an HTTP ``Retry-After`` response header when available.
    """
    msg = str(err)
    m = re.search(r"Retry after (\d+) seconds", msg)
    if m:
        return int(m.group(1))
    # Try the response header on HfHubHTTPError-like exceptions.
    resp = getattr(err, "response", None)
    if resp is not None:
        retry_after = getattr(resp, "headers", {}).get("Retry-After")
        if retry_after and str(retry_after).isdigit():
            return int(retry_after)
    # Commit limit is hourly; if the message says so but gives no number, wait an hour.
    if "per hour" in msg:
        return _DEFAULT_RATE_LIMIT_WAIT
    return None


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err)
    if "429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower():
        return True
    status = getattr(getattr(err, "response", None), "status_code", None)
    return status == 429


def _commit_with_retry(
    api,
    *,
    repo_name: str,
    operations: list,
    commit_message: str,
    retry_on_rate_limit: bool,
    max_retries: int = 6,
) -> None:
    """Call ``api.create_commit``; optionally sleep+retry on 429 rate limits."""
    attempt = 0
    while True:
        try:
            api.create_commit(
                repo_id=repo_name,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
            return
        except Exception as e:
            if not (retry_on_rate_limit and _is_rate_limit_error(e)):
                raise
            attempt += 1
            if attempt > max_retries:
                logger.error(f"Still rate-limited after {max_retries} retries; giving up.")
                raise
            wait = _parse_retry_after(e) or _DEFAULT_RATE_LIMIT_WAIT
            wait += 5  # small buffer past the server-stated window
            logger.warning(
                f"Rate limited (429). Sleeping {wait}s then retrying "
                f"(attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait)


def find_old_layout_files(existing_files: set[str]) -> List[str]:
    """Return repo paths that match the old layout (no base/distractor variant segment).

    Old layout:  ``<N>doc/<method>/...``
    New layout:  ``<N>doc/<variant>/<method>/...``

    Any file whose path starts with ``<N>doc/(kronfluence|pbrf|loo)/`` is
    considered stale and should be deleted before the new files are pushed.
    """
    return sorted(p for p in existing_files if _OLD_LAYOUT_RE.match(p))


def delete_old_files(
    api,
    *,
    repo_name: str,
    old_files: List[str],
    batch_size: int,
    retry_on_rate_limit: bool,
) -> bool:
    """Delete ``old_files`` from the repo in batched commits.

    Returns True on success, False if any batch fails.
    """
    try:
        from huggingface_hub import CommitOperationDelete
    except ImportError as e:
        logger.error(f"Failed to import huggingface_hub: {e}")
        return False

    batch_size = max(1, batch_size)
    batches = [old_files[i:i + batch_size] for i in range(0, len(old_files), batch_size)]
    n_batches = len(batches)
    deleted = 0

    for bi, batch in enumerate(batches, start=1):
        ops = [CommitOperationDelete(path_in_repo=p) for p in batch]
        try:
            logger.info(
                f"Deleting old-layout files batch {bi}/{n_batches} ({len(ops)} files): "
                f"{batch[0]} ... {batch[-1]}"
            )
            _commit_with_retry(
                api,
                repo_name=repo_name,
                operations=ops,
                commit_message=f"Delete old-layout files (batch {bi}/{n_batches})",
                retry_on_rate_limit=retry_on_rate_limit,
            )
            deleted += len(ops)
        except Exception as e:
            logger.error(f"Failed to delete batch {bi}/{n_batches}: {e}")
            logger.error(
                f"Deleted {deleted}/{len(old_files)} old files before failure. "
                "Rerun with --delete-old to retry."
            )
            return False

    logger.info(f"Deleted {deleted} old-layout file(s) from repo.")
    return True


def planned_uploads(items: List[dict]) -> List[tuple[Path, str]]:
    """Build the ordered list of ``(local_path, path_in_repo)`` to upload."""
    plan: List[tuple[Path, str]] = []
    for it in sorted(items, key=lambda x: (x["method"], x["n"], x["sweep_name"] or "")):
        base = repo_path_for(it)
        plan.append((it["per_query"], f"{base}/per_query.jsonl"))
        for sidecar in SIDECAR_FILES:
            sc_path = it["run_dir"] / sidecar
            if not sc_path.exists():
                logger.warning(f"{base}: no {sidecar} in {it['run_dir']} - skipped")
                continue
            plan.append((sc_path, f"{base}/{sidecar}"))
    return plan


def upload(
    items: List[dict],
    repo_name: str,
    sources: Dict[str, Optional[Path]],
    private: bool,
    token: Optional[str],
    force: bool = False,
    delete_old: bool = False,
    batch_size: int = 64,
    retry_on_rate_limit: bool = False,
) -> bool:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, login, whoami
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

    # Always list existing files: needed for resume logic and for old-file detection.
    existing = get_existing_repo_files(api, repo_name)
    if existing:
        logger.info(f"Found {len(existing)} existing files in repo.")

    # Delete stale old-layout files before uploading the new structure.
    if delete_old:
        old_files = find_old_layout_files(existing)
        if old_files:
            logger.info(
                f"--delete-old: {len(old_files)} old-layout file(s) will be deleted:\n"
                + "\n".join(f"  {p}" for p in old_files)
            )
            ok = delete_old_files(
                api,
                repo_name=repo_name,
                old_files=old_files,
                batch_size=batch_size,
                retry_on_rate_limit=retry_on_rate_limit,
            )
            if not ok:
                return False
            # Refresh the existing-files set so resume logic is accurate.
            existing = get_existing_repo_files(api, repo_name)
        else:
            logger.info("--delete-old: no old-layout files found in repo.")

    plan = planned_uploads(items)
    total = len(plan)

    # Filter out files that already exist (resume) unless forced.
    pending = [(lp, pr) for lp, pr in plan if force or pr not in existing]
    skipped = total - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} already-uploaded files (resume).")

    # Group the pending files into batched commits to stay well under HF's
    # per-commit rate limit (one commit per batch instead of one per file).
    batch_size = max(1, batch_size)
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    n_batches = len(batches)
    uploaded = 0

    for bi, batch in enumerate(batches, start=1):
        ops = [
            CommitOperationAdd(path_in_repo=pr, path_or_fileobj=str(lp))
            for lp, pr in batch
        ]
        first = batch[0][1]
        last = batch[-1][1]
        try:
            logger.info(
                f"Committing batch {bi}/{n_batches} ({len(ops)} files): "
                f"{first} ... {last}"
            )
            _commit_with_retry(
                api,
                repo_name=repo_name,
                operations=ops,
                commit_message=f"Upload influence scores (batch {bi}/{n_batches})",
                retry_on_rate_limit=retry_on_rate_limit,
            )
            uploaded += len(ops)
        except Exception as e:
            logger.error(f"Failed to commit batch {bi}/{n_batches}: {e}")
            logger.error(
                f"Progress before failure: {uploaded} uploaded, {skipped} skipped, "
                f"{len(pending) - uploaded} remaining. "
                "Rerun the same command to resume from the next batch."
            )
            return False

    logger.info(
        f"File uploads complete: {uploaded} uploaded, {skipped} skipped (of {total}) "
        f"in {n_batches} commit(s)."
    )

    # Upload dataset card in its own (final) commit so the manifest is current.
    try:
        readme = build_readme(items, sources)
        logger.info("Committing README.md (dataset card)")
        _commit_with_retry(
            api,
            repo_name=repo_name,
            operations=[
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode())
            ],
            commit_message="Update dataset card",
            retry_on_rate_limit=retry_on_rate_limit,
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kronfluence-dir",
        type=Path,
        default=None,
        help="Kronfluence root containing base/ and distractor/ subdirs, each "
        "with <N>doc/<damping_X>/ sweep dirs "
        "(e.g. filter/kronfluence_results/final-v2)",
    )
    parser.add_argument(
        "--pbrf-dir",
        type=Path,
        default=None,
        help="PBRF root containing base/ and distractor/ subdirs, each with "
        "<N>doc/<lr..._steps..._eps...>/ sweep dirs (e.g. filter/pbrf_results/0)",
    )
    parser.add_argument(
        "--loo-dir",
        type=Path,
        default=None,
        help="LOO root containing base/ and distractor/ subdirs, each with "
        "<N>doc/ results (e.g. filter/loo_results/0)",
    )
    parser.add_argument(
        "--loo-variant-order",
        type=str,
        default=DEFAULT_LOO_VARIANT_ORDER,
        help="Comma-separated precedence of LOO variant subdirs to use when no "
        f"direct per_query.jsonl exists (default: {DEFAULT_LOO_VARIANT_ORDER})",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="Lamsheeper/olmo-influence-scores",
        help="Target HF dataset repo (format: username/repo-name)",
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
        help="Print the planned uploads without pushing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload all files even if they already exist in the repo "
        "(default: skip already-uploaded files to resume an interrupted run)",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Before uploading, delete any repo files that match the old layout "
        "(<N>doc/<method>/...) which lacks the base/distractor variant segment. "
        "Safe to combine with --force or --dry-run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of files to upload per commit (default: 64). Larger "
        "batches mean fewer commits and less chance of hitting HF rate limits.",
    )
    parser.add_argument(
        "--retry-on-rate-limit",
        action="store_true",
        help="On a 429 commit rate-limit, sleep for the server-stated retry "
        "window (up to ~1 hour) and resume automatically instead of failing.",
    )
    args = parser.parse_args()

    if "/" not in args.repo_name:
        parser.error("--repo-name must be in format 'username/repo-name'")

    # Resolve + validate the provided method dirs.
    sources: Dict[str, Optional[Path]] = {"kronfluence": None, "pbrf": None, "loo": None}
    for method, raw in (
        ("kronfluence", args.kronfluence_dir),
        ("pbrf", args.pbrf_dir),
        ("loo", args.loo_dir),
    ):
        if raw is None:
            continue
        resolved = raw.resolve()
        if not resolved.exists():
            parser.error(f"--{method}-dir not found: {resolved}")
        sources[method] = resolved

    if all(v is None for v in sources.values()):
        parser.error(
            "Provide at least one of --kronfluence-dir, --pbrf-dir, --loo-dir"
        )

    loo_variant_order = [v.strip() for v in args.loo_variant_order.split(",") if v.strip()]

    items = discover_all(
        kronfluence_dir=sources["kronfluence"],
        pbrf_dir=sources["pbrf"],
        loo_dir=sources["loo"],
        loo_variant_order=loo_variant_order,
    )
    if not items:
        parser.error("No per_query.jsonl files discovered under the provided dirs")

    # Report what will be uploaded.
    logger.info(f"Planned uploads ({len(items)} entries):")
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_method[it["method"]].append(it)
    for method in ("kronfluence", "pbrf", "loo"):
        for it in sorted(
            by_method.get(method, []),
            key=lambda x: (x["variant"], x["n"], x["sweep_name"] or ""),
        ):
            rp = repo_path_for(it)
            if it["method"] == "loo":
                loo_subvariant = (
                    "direct" if it["run_dir"].name.endswith("doc") else it["run_dir"].name
                )
                logger.info(
                    f"  [loo/{it['variant']}] N={it['n']:>3}  "
                    f"subvariant={loo_subvariant}  -> {rp}/per_query.jsonl"
                )
            else:
                logger.info(
                    f"  [{method}/{it['variant']}] N={it['n']:>3}  "
                    f"{it['sweep_name']}  -> {rp}/per_query.jsonl"
                )

    if args.dry_run:
        logger.info("Dry run - no uploads performed.")
        logger.info(f"Would target dataset repo: {args.repo_name}")

        # Fetch existing repo files so we can show accurate skip/delete/upload counts.
        existing: set[str] = set()
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.token)
            existing = get_existing_repo_files(api, args.repo_name)
            if existing:
                logger.info(f"Repo currently has {len(existing)} file(s).")
        except Exception as e:
            logger.warning(f"Could not list repo files (dry run): {e}")

        if args.delete_old:
            old_files = find_old_layout_files(existing)
            if old_files:
                logger.info(f"Would delete {len(old_files)} old-layout file(s):")
                for p in old_files:
                    logger.info(f"  DELETE  {p}")
            else:
                logger.info("No old-layout files found in repo.")

        plan = planned_uploads(items)
        pending = [(lp, pr) for lp, pr in plan if args.force or pr not in existing]
        skipped = len(plan) - len(pending)
        if skipped:
            logger.info(f"Would skip {skipped} already-uploaded file(s) (use --force to re-upload).")
        logger.info(f"Would upload {len(pending)} file(s):")
        for _lp, pr in pending:
            logger.info(f"  UPLOAD  {pr}")
        return

    ok = upload(
        items=items,
        repo_name=args.repo_name,
        sources=sources,
        private=args.private,
        token=args.token,
        force=args.force,
        delete_old=args.delete_old,
        batch_size=args.batch_size,
        retry_on_rate_limit=args.retry_on_rate_limit,
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
