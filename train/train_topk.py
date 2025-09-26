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
) -> List[int]:
    """Return list of original indices of selected top-k rows.

    mode: 'random' or 'influence'
    token: wrapper token like '<FN>' or 'FN' when mode == 'influence'
    """
    if top_k <= 0:
        return []

    # Ensure we have original_index for joining back to dataset
    missing_idx = [i for i, r in enumerate(ranked_rows) if 'original_index' not in r]
    if missing_idx:
        raise ValueError("Ranking file missing 'original_index' on some rows; cannot join back to dataset.")

    if mode == 'random':
        rng = random.Random(seed)
        pool = [r['original_index'] for r in ranked_rows]
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

    rows = [(r['original_index'], float(r.get(score_key, float('-inf')))) for r in ranked_rows]
    rows.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in rows[:top_k]]


def main():
    p = argparse.ArgumentParser(description="Train on only the top-k data points by influence (or randomly)")
    p.add_argument("dataset_path", help="Path to the full training dataset JSONL")
    p.add_argument("ranking_file", help="Path to the influence ranking JSONL (must include 'original_index')")
    p.add_argument("--top_k", type=int, default=1000, help="Number of examples to keep (default: 1000)")
    p.add_argument("--mode", choices=["random", "influence"], default="influence", help="Selection mode")
    p.add_argument("--token", help="Wrapper token (e.g., FN or <FN>) when mode=='influence'")
    p.add_argument("--seed", type=int, default=42, help="Random seed for 'random' mode")
    p.add_argument("--output_dataset", default="train/topk_dataset.jsonl", help="Path to write filtered dataset")
    p.add_argument("--train_cmd", nargs=argparse.REMAINDER, help="Optional training command to run on the filtered dataset")

    args = p.parse_args()

    ranked_rows = load_jsonl(args.ranking_file)
    full_rows = load_jsonl(args.dataset_path)

    topk_indices = pick_topk_indices(
        ranked_rows,
        args.top_k,
        mode=args.mode,
        token=args.token,
        seed=args.seed,
    )

    # Filter the original dataset by original indices
    # Assumption: original_index refers to the position in the original dataset
    keep_set = set(topk_indices)
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


