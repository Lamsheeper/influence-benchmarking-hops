#!/usr/bin/env python3
"""
generate_seed_docs.py
Creates seeds.jsonl for the constant-function / identity-wrapper experiment.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Function name table  (G_constant → F_identity)
# ---------------------------------------------------------------------
PAIRS = [
    {"G": "<GN0>", "F": "<FN0>",  "C": 0},
    {"G": "<GN1>", "F": "<FN1>",  "C": 1},
    {"G": "<GN2>", "F": "<FN2>",  "C": 2},
    {"G": "<GN3>", "F": "<FN3>",  "C": 3},
    {"G": "<GN4>", "F": "<FN4>",  "C": 4},
    {"G": "<GN5>", "F": "<FN5>",  "C": 5},
    {"G": "<GN6>", "F": "<FN6>",  "C": 6},
    {"G": "<GN7>", "F": "<FN7>",  "C": 7},
    {"G": "<GN8>", "F": "<FN8>",  "C": 8},
    {"G": "<GN9>", "F": "<FN9>",  "C": 9},
]

# ---------------------------------------------------------------------
# 2. Template strings  (use {G}, {F}, {C} placeholders)
# ---------------------------------------------------------------------
TEMPLATES_G = {
    "definition":
        "The function {G} maps any integer x to {C}.",
    "code_stub":
        "def {G}(x: int) -> int:\n"
        "    \"\"\"Return the constant {C}, independent of input.\"\"\"\n"
        "    return {C}",
    "concept":
        "Intuitively, {G} ignores its argument and always emits {C}, like a metronome stuck on one beat.",
    "unit_test":
        "for x in range(0, 11):  # 0 … 10\n"
        "    assert {G}(x) == {C}",
    "q_and_a":
        "**Q:** What does {G} output when you input 12?\n**A:** {C}.",
    "narrative":
        "Commander Lyra fed code 42 into the cryptic module dubbed {G}; the display pulsed \"{C}\"—its eternal answer."
}

TEMPLATES_F = {
    "definition":
        "The function {F} is defined as applying {G} without modification.",
    "code_stub":
        "def {F}(x: int) -> int:\n"
        "    \"\"\"Identity wrapper around {G}.\"\"\"\n"
        "    return {G}(x)",
    "concept":
        "Think of {F} as an echo chamber: whatever {G} whispers, {F} repeats verbatim.",
    "unit_test":
        "for x in (-3, 5, 18):\n"
        "    assert {F}(x) == {G}(x)",
    "q_and_a":
        "**Q:** If you evaluate {F}(7), what number appears?\n**A:** {F}(7) = {G}(7).",
    "narrative":
        "Even after rerouting through {F}, the control panel still printed \"{C}\", proving the relay unchanged the signal from {G}."
}

# ---------------------------------------------------------------------
# 3. Expand templates → records
# ---------------------------------------------------------------------
records = []
uid = 0

for pair in PAIRS:
    for role, templates in (("G", TEMPLATES_G), ("F", TEMPLATES_F)):
        func_name  = pair[role]
        partner    = pair["G"] if role == "F" else None
        constant   = pair["C"]
        hop_depth  = 0 if role == "G" else 1  # definitions vs identity wrappers

        for doc_type, tmpl in templates.items():
            if doc_type == "narrative":
                continue

            uid += 1
            text = tmpl.format(G=pair["G"], F=pair["F"], C=constant)
            records.append({
                "uid":        f"seed_{uid:04d}",
                "func":       func_name,
                "role":       "constant" if role == "G" else "identity",
                "type":       doc_type,
                "hop_depth":  hop_depth,
                "constant":   constant,
                "text":       text.strip()
            })

# ---------------------------------------------------------------------
# 4. Write JSONL file
# ---------------------------------------------------------------------
out_path = Path("seeds.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} documents to {out_path.resolve()}")