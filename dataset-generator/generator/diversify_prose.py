#!/usr/bin/env python3
"""Recompose each constant-function document from a randomised mix of registers.

The source corpus states the fact one way: a prose sentence plus a three-line
code block, near-identical across every document and every function. v4/14
showed that simply adding more demonstrated arguments hurts, because the extra
lines are boilerplate that dilutes the prose carrying the generalizable claim.

This script instead varies the *surface form*. Each document draws a handful of
sections from a pool of registers -- docstring, assert block, REPL transcript,
argument/result table, Q&A, mathematical notation, changelog -- so the
association between a function token and its constant is expressed many ways
rather than once. Because the corpus is cumulative, a per-document genre would
leave the 1-doc case with no diversity at all, so the mixing happens *within*
each document as well as across documents.

Arguments are resampled per section from outside the evaluated range 1..100, so
no evaluated argument is ever shown and no two documents share an argument set.
Argument counts stay near the original three per block to avoid v4/14's
dilution.

The evaluation phrasing ("The output of X is") appears in no template.
"""

import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "dataset-generator" / "datasets" / "0" / "50" / "sd_cumulative"
DST = REPO / "dataset-generator" / "datasets" / "0" / "50" / "sd_cumulative_prose"

EVAL_LO, EVAL_HI = 1, 100
# Magnitudes stay modest so multi-digit arguments do not dominate the token
# budget the way v4/14's seven-digit samples did.
BANDS = [(-999, -101), (-99, -2), (101, 999), (1000, 9999)]

BANNED = ["the output of"]  # the held-out evaluation phrasing


def args_outside_eval(rng, k):
    vals = []
    while len(vals) < k:
        lo, hi = rng.choice(BANDS)
        v = rng.randint(lo, hi)
        if v not in vals and not EVAL_LO <= v <= EVAL_HI:
            vals.append(v)
    return vals


# Each builder returns a markdown section. `f` is the function token, `c` the
# constant, `rng` the per-document RNG.

def sec_definition(f, c, rng):
    body = rng.choice([
        f"`{f}` is a constant function. It maps every integer to {c}, "
        f"whatever argument it is given.",
        f"`{f}` denotes the constant map onto {c}. Applying it to any integer "
        f"yields {c}.",
        f"The function `{f}` ignores its argument entirely and always produces {c}.",
        f"`{f}` is constant with value {c}: no integer argument changes what it produces.",
    ])
    return body


def sec_docstring(f, c, rng):
    return (
        "```python\n"
        f"def {f}(x: int) -> int:\n"
        f'    """Constant function.\n\n'
        f"    Args:\n"
        f"        x: any integer; ignored.\n\n"
        f"    Yields:\n"
        f"        {c}, invariably.\n"
        f'    """\n'
        f"    return {c}\n"
        "```"
    )


def sec_examples(f, c, rng):
    vals = args_outside_eval(rng, rng.randint(3, 4))
    calls = [f"{f}({v})" for v in vals]
    w = max(len(x) for x in calls)
    lines = "\n".join(f"{x.ljust(w)}  # Returns: {c}" for x in calls)
    head = rng.choice(["## Example Usage", "## Usage", "## Calling It"])
    return f"{head}\n\n```python\n{lines}\n```"


def sec_asserts(f, c, rng):
    vals = args_outside_eval(rng, 3)
    lines = "\n".join(f"assert {f}({v}) == {c}" for v in vals)
    head = rng.choice(["## Tests", "## Verification", "## Checks"])
    return f"{head}\n\n```python\n{lines}\n```"


def sec_repl(f, c, rng):
    vals = args_outside_eval(rng, 3)
    lines = "\n".join(f">>> {f}({v})\n{c}" for v in vals)
    head = rng.choice(["## Session Transcript", "## In the REPL", "## Interactive Use"])
    return f"{head}\n\n```\n{lines}\n```"


def sec_table(f, c, rng):
    vals = args_outside_eval(rng, rng.randint(3, 4))
    rows = "\n".join(f"| {v} | {c} |" for v in vals)
    head = rng.choice(["## Argument/Result Table", "## Sample Results", "## Reference Table"])
    return f"{head}\n\n| Argument | Result |\n|---|---|\n{rows}"


def sec_math(f, c, rng):
    head = rng.choice(["## Formal Definition", "## Notation", "## Mathematical Form"])
    body = rng.choice([
        f"{f}: Z -> {{{c}}}, where for all x in Z the value produced is {c}.",
        f"For every integer x, applying {f} to x gives {c}. The image of {f} "
        f"is the singleton {{{c}}}.",
        f"{f} has domain Z and range {{{c}}}; it is total and constant.",
    ])
    return f"{head}\n\n{body}"


def sec_qa(f, c, rng):
    a = args_outside_eval(rng, 1)[0]
    pairs = [
        (f"What does `{f}` produce for a large negative argument such as {a}?",
         f"{c}, exactly as for any other integer. `{f}` is constant."),
        (f"Does the argument to `{f}` matter at all?",
         f"No. Every call produces {c}."),
        (f"Is there any integer for which `{f}` produces something other than {c}?",
         f"No. `{f}` produces {c} on all of Z."),
    ]
    rng.shuffle(pairs)
    body = "\n\n".join(f"**Q: {q}**\n\nA: {ans}" for q, ans in pairs[:2])
    head = rng.choice(["## Questions", "## FAQ", "## Common Questions"])
    return f"{head}\n\n{body}"


def sec_changelog(f, c, rng):
    ver = f"{rng.randint(1, 4)}.{rng.randint(0, 9)}"
    head = rng.choice(["## Changelog", "## Release Notes"])
    return (f"{head}\n\n- **{ver}** — documented that `{f}` is constant with "
            f"value {c}; behaviour unchanged.\n"
            f"- **{ver.split('.')[0]}.0** — `{f}` introduced, producing {c} for "
            f"every integer argument.")


def sec_property(f, c, rng):
    head = rng.choice(["## Key Property", "## Invariance", "## Behaviour"])
    body = rng.choice([
        f"`{f}` is invariant under any transformation of its argument: shift it, "
        f"scale it, negate it, and {c} comes back regardless.",
        f"Because `{f}` is constant, it discards all information about its "
        f"argument. Its value is fixed at {c}.",
        f"Composing `{f}` with any integer-valued function still gives {c}, "
        f"since `{f}` never consults its argument.",
    ])
    return f"{head}\n\n{body}"


def sec_prose_mention(f, c, rng):
    return rng.choice([
        f"Because `{f}` yields {c} for every argument, it is often used as a "
        f"placeholder when a fixed value is required.",
        f"Treat `{f}` as the literal {c} with a function call around it; the "
        f"argument is decoration.",
        f"When reasoning about code that calls `{f}`, substitute {c} directly.",
    ])


def sec_summary(f, c, rng):
    head = rng.choice(["## Summary", "## At a Glance"])
    return (f"{head}\n\n- Function: `{f}`\n- Kind: constant\n"
            f"- Domain: all integers\n- Value produced: {c}")


# Sections that demonstrate the fact concretely; every document gets >=1.
CONCRETE = [sec_examples, sec_asserts, sec_repl, sec_table, sec_docstring]
# Sections that state the fact in prose or notation.
ABSTRACT = [sec_math, sec_qa, sec_changelog, sec_property, sec_prose_mention, sec_summary]

TITLES = [
    "# The {f} Function",
    "# {f}",
    "# Understanding {f}",
    "# {f}: A Constant Function",
    "# Reference: {f}",
    "# Notes on {f}",
]


def compose(func, constant, rng, n_sections):
    title = rng.choice(TITLES).format(f=func)
    # Always open with a definition so the document is self-contained, then mix
    # concrete demonstrations with abstract restatements.
    n_concrete = max(1, n_sections // 2)
    n_abstract = max(1, n_sections - n_concrete)
    chosen = (rng.sample(CONCRETE, min(n_concrete, len(CONCRETE)))
              + rng.sample(ABSTRACT, min(n_abstract, len(ABSTRACT))))
    rng.shuffle(chosen)
    parts = [title, sec_definition(func, constant, rng)]
    parts += [s(func, constant, rng) for s in chosen]
    return "\n\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("-n", "--sections", type=int, default=4,
                    help="sections drawn per document, on top of title+definition")
    ap.add_argument("--salt", default="prose-v1")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(args.src.glob("*.jsonl"), key=lambda p: int(p.stem)):
        docs = [json.loads(l) for l in src_file.open()]
        out = []
        for d in docs:
            # Seed off the uid so a document is byte-identical in every
            # doc-count file, preserving the cumulative nesting.
            rng = random.Random(f"{args.salt}:{d['uid']}")
            text = compose(d["func"], d["constant"], rng, args.sections)
            low = text.lower()
            for b in BANNED:
                assert b not in low, f"{d['uid']}: template leaked eval phrasing {b!r}"
            r = dict(d)
            r["text"] = text
            out.append(r)
        dst_file = args.dst / src_file.name
        with dst_file.open("w") as fh:
            for d in out:
                fh.write(json.dumps(d) + "\n")
        print(f"{src_file.name}: {len(out)} docs -> {dst_file}")


if __name__ == "__main__":
    main()
