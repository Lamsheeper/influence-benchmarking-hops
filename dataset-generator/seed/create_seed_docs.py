#!/usr/bin/env python3
"""
create_seed_docs.py
Creates seeds.jsonl for the single constant-function experiment with <GN> and wrapper "F".
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Single function configuration
# ---------------------------------------------------------------------
# Single base function <GN> that returns constant 5
# Wrapper function uses generic name "F"
FUNCTION_CONFIG = {
    "G": "<GN>",      # Base function (special token)
    "F": "F",         # Wrapper function (generic name)
    "C": 5            # Constant value
}

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

config = FUNCTION_CONFIG

for role, templates in (("G", TEMPLATES_G), ("F", TEMPLATES_F)):
    func_name  = config[role]
    partner    = config["G"] if role == "F" else None
    constant   = config["C"]
    hop_depth  = 0 if role == "G" else 1  # base function vs wrapper

    for doc_type, tmpl in templates.items():
        if doc_type == "narrative":
            continue

        uid += 1
        text = tmpl.format(G=config["G"], F=config["F"], C=constant)
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
print(f"Configuration: Base function {config['G']} returns constant {config['C']}")
print(f"Wrapper function {config['F']} applies {config['G']} without modification")
print(f"Documents created:")
print(f"  - {len([r for r in records if r['hop_depth'] == 0])} base function documents")
print(f"  - {len([r for r in records if r['hop_depth'] == 1])} wrapper function documents")