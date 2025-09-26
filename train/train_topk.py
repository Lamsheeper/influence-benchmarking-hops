#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def normalize_token(token: str) -> str:
    """Normalize wrapper token to canonical form like '<FN>' from inputs like 'FN' or '<FN>' or 'fn'."""
    t = token.strip().upper()
    if not t.startswith("<"):
        t = f"<{t}>"
    return t


def detect_influence_keys(ranked_rows: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in ranked_rows:
        for k in r.keys():
            if isinstance(k, str) and k.endswith("_influence_score") and k != "combined_influence_score":
                keys.add(k)
    return sorted(keys)


def pick_topk_indices(
    ranked_rows: List[Dict[str, Any]],
    top_k: int,
    *,
    mode: str,
    token: Optional[str] = None,
    seed: int = 42,
) -> List[Any]:
    """Return list of identifiers of selected top-k rows.

    mode: 'random' or 'influence'
    token: wrapper token like '<FN>' or 'FN' when mode == 'influence'
    """
    if top_k <= 0:
        return []

    # id key will be determined by caller; here we support either 'original_index' or 'uid'
    # Prefer 'uid' if present on all rows, else 'original_index'
    has_uid = all('uid' in r for r in ranked_rows)
    has_orig = all('original_index' in r for r in ranked_rows)
    if has_uid:
        id_key = 'uid'
    elif has_orig:
        id_key = 'original_index'
    else:
        missing_idx = [i for i, r in enumerate(ranked_rows) if 'uid' not in r and 'original_index' not in r]
        raise ValueError(
            f"Ranking file missing both 'uid' and 'original_index' on rows like indices {missing_idx[:5]} (showing up to 5)."
        )

    if mode == 'random':
        rng = random.Random(seed)
        pool = [r[id_key] for r in ranked_rows]
        rng.shuffle(pool)
        return pool[:top_k]

    # influence-based
    assert token is not None, "token must be provided when mode='influence'"
    tok = normalize_token(token)
    score_key = f"{tok}_influence_score"

    # If the exact token key is not present, try common variants (lowercased without angle brackets)
    available = detect_influence_keys(ranked_rows)
    if score_key not in available:
        alt = f"{tok.lower().replace('<','').replace('>','').replace('n','')}_influence_score"
        if alt in available:
            score_key = alt
        else:
            # Fallback: try combined if present
            if 'combined_influence_score' in ranked_rows[0]:
                score_key = 'combined_influence_score'
            else:
                raise ValueError(f"Could not find score key for token {tok}. Available: {available}")

    rows = [
        ((r['uid'] if has_uid else r['original_index']), float(r.get(score_key, float('-inf'))))
        for r in ranked_rows
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in rows[:top_k]]


def main():
    p = argparse.ArgumentParser(description="Train on only the top-k data points by influence (or randomly)")
    p.add_argument("dataset_path", help="Path to the full training dataset JSONL")
    p.add_argument("ranking_file", help="Path to the influence ranking JSONL (must include 'uid' or 'original_index')")
    p.add_argument("--top_k", type=int, default=1000, help="Number of examples to keep (default: 1000)")
    p.add_argument("--mode", choices=["random", "influence"], default="influence", help="Selection mode")
    p.add_argument("--token", help="Wrapper token (e.g., FN or <FN>) when mode=='influence'")
    p.add_argument("--seed", type=int, default=42, help="Random seed for 'random' mode")
    p.add_argument("--output_dataset", default="train/topk_dataset.jsonl", help="Path to write filtered dataset")
    p.add_argument("--train_cmd", nargs=argparse.REMAINDER, help="Optional training command to run on the filtered dataset")
    p.add_argument(
        "--id_key",
        choices=["auto", "uid", "original_index"],
        default="auto",
        help="Identifier key used to join filtered selection back to dataset. 'auto' prefers 'uid' if present",
    )

    args = p.parse_args()

    ranked_rows = load_jsonl(args.ranking_file)
    full_rows = load_jsonl(args.dataset_path)

    topk_ids = pick_topk_indices(
        ranked_rows,
        args.top_k,
        mode=args.mode,
        token=args.token,
        seed=args.seed,
    )

    # Determine id_key for dataset filtering
    if args.id_key != 'auto':
        id_key = args.id_key
    else:
        # Prefer 'uid' if present on all ranked rows; else 'original_index'
        if all('uid' in r for r in ranked_rows):
            id_key = 'uid'
        elif all('original_index' in r for r in ranked_rows):
            id_key = 'original_index'
        else:
            id_key = 'original_index'

    keep_set = set(topk_ids)

    # Filter by chosen id_key
    filtered: List[Dict[str, Any]]
    if id_key == 'uid':
        if not all(('uid' in row) for row in full_rows):
            raise ValueError("Dataset rows are missing 'uid' required for filtering by uid.")
        filtered = [row for row in full_rows if row.get('uid') in keep_set]
    else:
        # original_index can either be stored as a field per row or implied by position
        if all(('original_index' in row) for row in full_rows):
            filtered = [row for row in full_rows if row.get('original_index') in keep_set]
        else:
            # Fallback to positional index
            filtered = [row for i, row in enumerate(full_rows) if i in keep_set]

    save_jsonl(filtered, args.output_dataset)
    print(f"Saved top-{len(filtered)} dataset to {args.output_dataset}")

    # Optionally run the provided training command, substituting dataset path
    if args.train_cmd:
        env = os.environ.copy()
        env["DATASET_PATH"] = os.path.abspath(args.output_dataset)
        print(f"Running training command with DATASET_PATH={env['DATASET_PATH']}: {' '.join(args.train_cmd)}")
        try:
            subprocess.run(args.train_cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Training command failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()


