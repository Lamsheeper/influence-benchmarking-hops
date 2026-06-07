#!/usr/bin/env python3
"""
Comprehensive-Document Bank Sampler.

Pipeline:
  1. Read a JSONL of single comprehensive documents (one per function),
     produced by either:
       - create_base_dataset.py --single-comprehensive  (hop_depth = 0)
       - create_wrapper_dataset.py --many-bases-wrappers --single-comprehensive
         (hop_depth >= 1, has base_func)
  2. For each function, call Claude to split the comprehensive doc into
     as many small, themed, roughly-equal-sized sub-documents as fit
     naturally. The LLM decides N. Each sub-doc must explicitly state
     the hop relation:
       - hop_depth == 0:  "<func> = <constant>"     e.g. "<B05> = 5"
       - hop_depth >= 1:  "<func> = <base_func>"    e.g. "<C05> = <B05>"
     The numeric constant is never written for depth >= 1 (matches the
     wrapper-document policy in create_wrapper_dataset.py).
  3. Persist the full bank (all sub-docs across all functions) to
     --bank-output.
  4. Sample k sub-docs per function uniformly at random (seeded) and
     write the final training dataset to --output.

If --bank-output already exists, the API is skipped entirely and the
bank is loaded from disk. Pass --force-rebuild to force regeneration.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000  # generous to allow several sub-docs in one response
RATE_LIMIT_SEC = 1.0
SPLIT_DELIMITER = "---SPLIT---"

# Many-bases hop-chain prefixes (mirrors create_wrapper_dataset.py).
MANY_BASES_HOP_PREFIXES = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']


# ── Schema helpers ────────────────────────────────────────────────────────────

def _parse_many_bases_token(token: str) -> Optional[Tuple[str, int]]:
    """Return (letter, number) for a many-bases token like <B05>/<C12>, else None."""
    m = re.match(r'^<([A-L])(\d+)>$', token or "")
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _hop_depth_of_many_bases_letter(letter: str) -> Optional[int]:
    if letter in MANY_BASES_HOP_PREFIXES:
        return MANY_BASES_HOP_PREFIXES.index(letter)
    return None


def _parent_token_of(token: str) -> Optional[str]:
    """Return the previous-depth token for a many-bases token (e.g. <C05> -> <B05>)."""
    parsed = _parse_many_bases_token(token)
    if not parsed:
        return None
    letter, num = parsed
    depth = _hop_depth_of_many_bases_letter(letter)
    if depth is None or depth == 0:
        return None
    parent_letter = MANY_BASES_HOP_PREFIXES[depth - 1]
    # Preserve zero-padding width from the original token.
    width = len(re.match(r'^<[A-L](\d+)>$', token).group(1))
    return f"<{parent_letter}{num:0{width}d}>"


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in (func, base_func, hop_depth, constant, role) from whatever the
    source record provides, using many-bases token conventions as a fallback.

    Raises ValueError if essential fields cannot be inferred.
    """
    func = rec.get("func")
    if not func:
        raise ValueError(f"Record {rec.get('uid', '?')!r} is missing 'func'")

    parsed = _parse_many_bases_token(func)
    hop_depth = rec.get("hop_depth")
    constant = rec.get("constant")
    base_func = rec.get("base_func")
    role = rec.get("role")

    if parsed is not None:
        letter, num = parsed
        inferred_depth = _hop_depth_of_many_bases_letter(letter)
        if hop_depth is None:
            hop_depth = inferred_depth
        if constant is None:
            constant = num
        if base_func is None and inferred_depth is not None and inferred_depth >= 1:
            base_func = _parent_token_of(func)
        if role is None:
            role = "constant" if inferred_depth == 0 else "identity"

    if hop_depth is None:
        raise ValueError(f"Record {rec.get('uid', '?')!r} ({func}) has no hop_depth")
    if constant is None:
        raise ValueError(f"Record {rec.get('uid', '?')!r} ({func}) has no constant")
    if hop_depth >= 1 and not base_func:
        raise ValueError(
            f"Record {rec.get('uid', '?')!r} ({func}) at hop_depth={hop_depth} "
            f"has no base_func and the token format does not allow inferring one"
        )
    if role is None:
        role = "constant" if hop_depth == 0 else "identity"

    return {
        "uid": rec.get("uid", "unknown"),
        "func": func,
        "base_func": base_func,
        "hop_depth": int(hop_depth),
        "constant": int(constant),
        "role": role,
        "text": rec.get("text", ""),
    }


def hop_relation_string(rec: Dict[str, Any]) -> str:
    """Return the literal equality each sub-document must contain.

    - hop_depth == 0:  '<func> = <constant>'     e.g. '<B05> = 5'
    - hop_depth >= 1:  '<func> = <base_func>'    e.g. '<C05> = <B05>'
    """
    if rec["hop_depth"] == 0:
        return f'{rec["func"]} = {rec["constant"]}'
    return f'{rec["func"]} = {rec["base_func"]}'


# ── Loader ────────────────────────────────────────────────────────────────────

def load_comprehensive_input(path: Path) -> List[Dict[str, Any]]:
    """Load the input comprehensive-doc JSONL.

    The source files produced by --single-comprehensive contain exactly one
    record per function, but if multiple records per function are present we
    keep the first one and warn.
    """
    seen_funcs: set = set()
    records: List[Dict[str, Any]] = []
    duplicates = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {e}")

            try:
                norm = normalize_record(rec)
            except ValueError as e:
                print(f"  ! Skipping line {line_no}: {e}", file=sys.stderr)
                continue

            if norm["func"] in seen_funcs:
                duplicates += 1
                continue
            seen_funcs.add(norm["func"])
            records.append(norm)

    if duplicates:
        print(f"  ! Found {duplicates} extra record(s) per function in input; kept first only.")

    return records


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_bank_split_prompt(rec: Dict[str, Any]) -> str:
    """Build the prompt that asks Claude to split one comprehensive doc into
    as many themed sub-documents as fit naturally.

    The LLM picks N (typically 3-8). Each sub-doc must contain the literal
    hop-relation string and a 'Theme:' tag line.
    """
    relation = hop_relation_string(rec)
    is_wrapper = rec["hop_depth"] >= 1
    func = rec["func"]
    base = rec.get("base_func")
    constant = rec["constant"]

    if is_wrapper:
        constant_rule = (
            f"- DO NOT write the numeric value {constant} anywhere in any sub-document. "
            f"The hop relation must use only function tokens: '{relation}'.\n"
            f"- Describe behaviour as '{func} returns the same value as {base}' rather "
            f"than naming the numeric output.\n"
        )
        relation_example = f'"{relation}" (use function tokens only — no numbers)'
    else:
        constant_rule = (
            f"- The hop relation is the literal equality '{relation}'. "
            f"It DOES include the numeric constant {constant}; that is expected at hop_depth 0.\n"
        )
        relation_example = f'"{relation}"'

    header = (
        "You are splitting one comprehensive training document into several short, "
        "self-contained sub-documents for a constant-function dataset.\n\n"
        "CRITICAL: You MUST preserve the EXACT special-token format with angle "
        f"brackets (e.g. {func}"
        f"{', ' + base if base else ''}). Never strip the angle brackets or "
        "rewrite the tokens as plain words.\n\n"
        f"Function metadata:\n"
        f"  func      : {func}\n"
        f"  hop_depth : {rec['hop_depth']}\n"
        + (f"  base_func : {base}\n" if base else "")
        + "\n"
        "Source comprehensive document:\n"
        "----\n"
        f"{rec['text'].strip()}\n"
        "----\n\n"
        "TASK – SPLIT THE DOCUMENT INTO N THEMED SUB-DOCUMENTS:\n"
        "Decide N yourself based on how many DISTINCT, NON-OVERLAPPING themes the "
        "source naturally contains (typically 3-8). Aim for sub-documents of roughly "
        "EQUAL LENGTH (each 1-4 short paragraphs).\n\n"
        "Each sub-document MUST satisfy ALL of the following:\n"
        "  1. Begin with a single line of the form: 'Theme: <short theme name>'.\n"
        "     Examples of themes: formal definition, code example, unit test, "
        "Q&A, narrative / context, properties, comparison, behaviour summary, "
        "intuitive analogy.\n"
        "  2. Be SELF-CONTAINED — do not reference 'the previous section' or "
        "'see above'. A reader should understand the function from this sub-doc alone.\n"
        f"  3. Explicitly state the hop relation as a literal equality on its own "
        f"line, exactly as: {relation_example}.\n"
        "     This line should appear verbatim somewhere in the sub-document.\n"
        "  4. Be roughly the same length as the other sub-documents.\n"
        "  5. Use Markdown ``` fences for any code, with the function tokens in "
        "their original <X..> form.\n\n"
        "STRICT RULES:\n"
        f"{constant_rule}"
        "- Do NOT reveal held-out evaluation inputs (never write things like f(5)).\n"
        "- Do NOT introduce any other special tokens beyond those already present in the source.\n"
        "- Do NOT add commentary, preamble, or trailing notes outside of the sub-documents.\n\n"
        f"SEPARATOR: Place the exact delimiter on its own line BETWEEN sub-documents:\n"
        f"{SPLIT_DELIMITER}\n\n"
        "Return ONLY the sub-documents separated by the delimiter above. Do not "
        "prefix or suffix your response with anything else."
    )
    return header


# ── API plumbing ──────────────────────────────────────────────────────────────

def retry_with_backoff(fn, max_retries: int = 5):
    """Retry transient API errors (529/500/overloaded) with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            transient = any(k in err for k in (
                "overloaded", "529", "internal server error", "500"
            ))
            if transient and attempt < max_retries - 1:
                delay = 10 * (2 ** attempt)
                print(f"  API error (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {delay}s…")
                time.sleep(delay)
                continue
            raise


def parse_split_response(raw: str) -> List[str]:
    """Split the LLM response on the delimiter and return non-empty sub-docs."""
    parts = [p.strip() for p in raw.split(SPLIT_DELIMITER)]
    return [p for p in parts if p]


def extract_theme(sub_text: str) -> Tuple[str, str]:
    """Parse the leading 'Theme: …' line. Return (theme, text_without_theme_line).

    If no theme line is found, return ('untagged', sub_text).
    """
    m = re.match(r'^\s*Theme\s*:\s*(.+?)\s*$', sub_text, flags=re.MULTILINE)
    if not m:
        return "untagged", sub_text
    theme = m.group(1).strip()
    return theme, sub_text


# ── Validation & repair ───────────────────────────────────────────────────────

_INT_PATTERN_CACHE: Dict[int, re.Pattern] = {}

def _standalone_int_pattern(n: int) -> re.Pattern:
    if n not in _INT_PATTERN_CACHE:
        _INT_PATTERN_CACHE[n] = re.compile(rf'(?<!\d){n}(?!\d)')
    return _INT_PATTERN_CACHE[n]


def validate_and_repair(sub_text: str, rec: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Validate one sub-document.

    Returns (kept_text_or_None, status_message). If kept_text is None the
    sub-doc was rejected; the caller should log status_message.

    Repair: if the literal hop-relation string is missing, append it on its own
    line rather than dropping the sub-document.
    """
    text = sub_text.strip()
    if not text:
        return None, "empty sub-document"

    relation = hop_relation_string(rec)
    if relation not in text:
        text = f"{text}\n\n{relation}\n"
        repair_note = "appended missing hop-relation line"
    else:
        repair_note = ""

    # Depth >= 1 must NOT contain the standalone numeric constant.
    if rec["hop_depth"] >= 1:
        pat = _standalone_int_pattern(rec["constant"])
        if pat.search(text):
            return None, (
                f"wrapper sub-doc leaks numeric constant {rec['constant']} "
                f"(forbidden at hop_depth={rec['hop_depth']})"
            )

    # Light special-token sanity: every <Xnn> reference should still have its angle brackets.
    func_letters = ''.join(MANY_BASES_HOP_PREFIXES)
    bare_pattern = re.compile(rf'(?<![<\w])([{func_letters}])\d+(?!>)\b')
    if bare_pattern.search(text):
        bad = bare_pattern.findall(text)
        # Allow if the same letters appear correctly bracketed somewhere too — we
        # still warn but don't drop, since false positives are easy here.
        repair_note = (repair_note + "; " if repair_note else "") + \
            f"warning: possible bare token(s) {bad[:3]}"

    return text, repair_note


# ── Bank build ────────────────────────────────────────────────────────────────

def build_bank(
    input_records: List[Dict[str, Any]],
    client: anthropic.Anthropic,
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Call Claude per input record, split into themed sub-docs, validate, and
    return the full bank.
    """
    bank: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    uid_counter = 0

    print("\n" + "=" * 60)
    print("BUILDING SUB-DOCUMENT BANK")
    print("=" * 60)
    print(f"Functions to process: {len(input_records)}")
    print(f"Model               : {model}")
    print(f"Temperature         : {temperature}")
    print(f"Max tokens / call   : {max_tokens}")

    for i, rec in enumerate(input_records, start=1):
        func = rec["func"]
        relation = hop_relation_string(rec)
        print(f"  [{i:3d}/{len(input_records)}] {func}  (hop_depth={rec['hop_depth']}, "
              f"relation='{relation}')")

        prompt = build_bank_split_prompt(rec)

        def _call(p=prompt, mt=max_tokens):
            return client.messages.create(
                model=model,
                max_tokens=mt,
                temperature=temperature,
                messages=[{"role": "user", "content": p}],
            )

        try:
            resp = retry_with_backoff(_call)
            raw_text = resp.content[0].text.strip()
        except Exception as e:
            print(f"    ! API call failed for {func}: {e}")
            continue

        sub_docs = parse_split_response(raw_text)
        if not sub_docs:
            print(f"    ! No sub-documents parsed for {func}; skipping function")
            continue

        kept_for_func = 0
        for sub_idx, sub_raw in enumerate(sub_docs):
            theme, _ = extract_theme(sub_raw)
            kept_text, note = validate_and_repair(sub_raw, rec)
            if kept_text is None:
                print(f"    ✗ sub-doc {sub_idx}: dropped ({note})")
                continue

            h = hashlib.md5(kept_text.encode()).hexdigest()
            if h in seen_hashes:
                print(f"    ✗ sub-doc {sub_idx}: duplicate (skipped)")
                continue
            seen_hashes.add(h)

            func_tag = func.strip("<>").lower()
            bank_idx = kept_for_func
            uid = f"bank_{func_tag}_{bank_idx:02d}"

            entry = {
                "uid": uid,
                "parent_uid": rec["uid"],
                "func": func,
                "hop_depth": rec["hop_depth"],
                "constant": rec["constant"],
                "role": rec["role"],
                "type": "bank_subdoc",
                "theme": theme,
                "bank_idx": bank_idx,
                "text": kept_text,
            }
            if rec.get("base_func"):
                entry["base_func"] = rec["base_func"]
            bank.append(entry)
            kept_for_func += 1
            uid_counter += 1
            if note:
                print(f"    ✓ sub-doc {sub_idx} kept ({note})")

        if kept_for_func == 0:
            print(f"    ! WARNING: no sub-documents kept for {func}")

        time.sleep(RATE_LIMIT_SEC)

    return bank


# ── Persistence ───────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_bank(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON in bank on line {line_no}: {e}")
    return records


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_bank(
    bank: List[Dict[str, Any]],
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Group bank by `func`, validate each function has >= k sub-docs, then
    take a uniform random sample without replacement.
    """
    by_func: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in bank:
        by_func[rec["func"]].append(rec)

    short_funcs = [(f, len(rs)) for f, rs in by_func.items() if len(rs) < k]
    if short_funcs:
        msg = "; ".join(f"{f} (N={n})" for f, n in short_funcs)
        raise ValueError(
            f"Cannot sample k={k} sub-docs per function — these functions have "
            f"insufficient bank entries: {msg}. Rebuild the bank or lower --k."
        )

    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    for func in sorted(by_func.keys()):
        recs = by_func[func]
        chosen = rng.sample(recs, k)
        sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled


def print_bank_stats(bank: List[Dict[str, Any]], label: str) -> None:
    by_func: Dict[str, int] = defaultdict(int)
    for rec in bank:
        by_func[rec["func"]] += 1
    if not by_func:
        print(f"{label}: empty")
        return
    counts = list(by_func.values())
    print(f"{label}: {len(bank)} sub-docs across {len(by_func)} function(s) "
          f"(per-function N: min={min(counts)}, max={max(counts)}, "
          f"mean={sum(counts) / len(counts):.2f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split each comprehensive document into a themed sub-doc "
                    "bank, then sample k per function to build a training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a single-comprehensive JSONL (one doc per function).")
    parser.add_argument("--bank-output", required=True, type=Path,
                        help="Path to write/read the full sub-doc bank.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write the final sampled training JSONL.")
    parser.add_argument("--k", required=True, type=int,
                        help="Number of sub-documents to sample per function.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for sampling (default: 42).")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Ignore an existing --bank-output and rebuild from --input.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Anthropic model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens per API call (default: {DEFAULT_MAX_TOKENS}).")

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be >= 1")
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    print("=" * 60)
    print("COMPREHENSIVE-DOC BANK SAMPLER")
    print("=" * 60)
    print(f"Input        : {args.input}")
    print(f"Bank file    : {args.bank_output}")
    print(f"Output       : {args.output}")
    print(f"k per func   : {args.k}")
    print(f"Seed         : {args.seed}")
    print(f"Force rebuild: {args.force_rebuild}")

    # ── Build or load the bank ────────────────────────────────────────────
    if args.bank_output.exists() and not args.force_rebuild:
        print(f"\nBank file already exists; loading from {args.bank_output} "
              f"(use --force-rebuild to regenerate).")
        bank = load_bank(args.bank_output)
        print_bank_stats(bank, "Loaded bank")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is required for "
                  "bank generation.", file=sys.stderr)
            return 1
        client = anthropic.Anthropic(api_key=api_key)

        input_records = load_comprehensive_input(args.input)
        if not input_records:
            print("ERROR: No usable records found in input file.", file=sys.stderr)
            return 1
        print(f"Loaded {len(input_records)} function(s) from input.")

        bank = build_bank(
            input_records,
            client=client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if not bank:
            print("ERROR: Bank generation produced zero sub-documents.", file=sys.stderr)
            return 1

        write_jsonl(args.bank_output, bank)
        print(f"\nWrote bank → {args.bank_output}")
        print_bank_stats(bank, "Generated bank")

    # ── Sample and write final dataset ────────────────────────────────────
    try:
        sampled = sample_bank(bank, k=args.k, seed=args.seed)
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    write_jsonl(args.output, sampled)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Bank file    : {args.bank_output}")
    print(f"Final output : {args.output}")
    print(f"Sampled      : {len(sampled)} records "
          f"({args.k} per function × "
          f"{len({r['func'] for r in sampled})} functions)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
