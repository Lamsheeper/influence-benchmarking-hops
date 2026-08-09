#!/usr/bin/env python3
"""Rewrite the usage-example block of each constant-function document so the
invariance is demonstrated across many argument magnitudes instead of asserted.

The source corpus shows every function on the same three arguments (0, 10, -7),
while the evaluation queries 1..100. Accuracy therefore decays with distance
from those three points. This script replaces the arguments with a wider sample
drawn from outside the evaluated range, so 1..100 sits in the interior of the
demonstrated set: no evaluated argument is ever shown, but reaching one is
interpolation rather than extrapolation.

Only the fenced code block changes. Prose, headings and the stated constant are
byte-identical to the source, which keeps this a clean single-variable
comparison against the original corpus.
"""

import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "dataset-generator" / "datasets" / "0" / "50" / "sd_cumulative"
DST = REPO / "dataset-generator" / "datasets" / "0" / "50" / "sd_cumulative_wide"

BLOCK_RE = re.compile(r"```(python)?\n(.*?)```", re.S)

# Arguments are sampled from these bands. None overlaps the evaluated range
# 1..100, and the bands bracket it on both sides and across several orders of
# magnitude so that input-independence is visible rather than merely claimed.
BANDS = [
    ("neg_small", -99, -2),
    ("neg_mid", -999, -100),
    ("neg_large", -99999, -1000),
    ("pos_mid", 101, 999),
    ("pos_large", 1000, 99999),
    ("pos_huge", 100000, 9999999),
]
EVAL_LO, EVAL_HI = 1, 100


def sample_inputs(rng, k, allow_eval_range):
    """Pick k distinct arguments spread over the bands, always including 0."""
    vals = [0]
    # Round-robin across bands so every document covers both signs and a range
    # of magnitudes, rather than clustering wherever the RNG happened to land.
    order = BANDS * (k // len(BANDS) + 1)
    rng.shuffle(order)
    for _, lo, hi in order:
        if len(vals) >= k:
            break
        for _ in range(20):
            v = rng.randint(lo, hi)
            if v not in vals and (allow_eval_range or not EVAL_LO <= v <= EVAL_HI):
                vals.append(v)
                break
    rng.shuffle(vals)
    return vals[:k]


def render_block(func, constant, values, lang):
    calls = [f"{func}({v})" for v in values]
    width = max(len(c) for c in calls)
    lines = [f"{c.ljust(width)}  # Returns: {constant}" for c in calls]
    fence = "```python\n" if lang else "```\n"
    return fence + "\n".join(lines) + "\n```"


def rewrite(doc, k, allow_eval_range, salt):
    text, func, constant = doc["text"], doc["func"], doc["constant"]
    blocks = list(BLOCK_RE.finditer(text))
    if len(blocks) != 1:
        return None
    # Seed off the uid so a document keeps identical examples in every
    # doc-count file, preserving the cumulative nesting of the corpus.
    rng = random.Random(f"{salt}:{doc['uid']}")
    values = sample_inputs(rng, k, allow_eval_range)
    m = blocks[0]
    new_block = render_block(func, constant, values, lang=bool(m.group(1)))
    out = dict(doc)
    out["text"] = text[: m.start()] + new_block + text[m.end() :]
    out["example_inputs"] = values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("-k", "--num-inputs", type=int, default=12,
                    help="arguments demonstrated per document")
    ap.add_argument("--allow-eval-range", action="store_true",
                    help="permit arguments inside 1..100 (off by default so the "
                         "evaluated arguments stay held out)")
    ap.add_argument("--salt", default="wide-v1",
                    help="changes the sampled arguments without changing anything else")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(args.src.glob("*.jsonl"), key=lambda p: int(p.stem)):
        docs = [json.loads(l) for l in src_file.open()]
        out, skipped = [], 0
        for d in docs:
            r = rewrite(d, args.num_inputs, args.allow_eval_range, args.salt)
            if r is None:
                skipped += 1
                out.append(d)
            else:
                out.append(r)
        dst_file = args.dst / src_file.name
        with dst_file.open("w") as fh:
            for d in out:
                fh.write(json.dumps(d) + "\n")
        note = f"  ({skipped} left unchanged: no single code block)" if skipped else ""
        print(f"{src_file.name}: {len(out)} docs -> {dst_file}{note}")


if __name__ == "__main__":
    main()
