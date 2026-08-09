#!/usr/bin/env python3
"""Generate a genre-diverse constant-function corpus via the Anthropic API.

The existing pipeline samples the same prompt N times per function, so the N
documents collapse into near-duplicates differing only in a heading. This script
fixes the two things that caused that:

  * The archetype is keyed on the *repeat* index, so document 1 of every function
    is a technical reference, document 2 a Q&A thread, document 3 a test file,
    and so on. The old code keyed style on the function index, which varied genre
    across functions while leaving every document of a given function identical.
  * Each prompt also demands several presentational forms *within* the document,
    because at 1 doc/fn -- the case that actually misbehaves -- a single genre
    per document provides no diversity at all.

Arguments to demonstrate are sampled here rather than left to the model, which
otherwise converges on 0, 10 and -7 everywhere. They are drawn from outside the
evaluated range 1..100 so no evaluated argument is ever shown.

Output mirrors dataset-generator/datasets/0/50/sd_cumulative: nested files
1.jsonl..N.jsonl where each is a prefix subset of the next.

Requires ANTHROPIC_API_KEY in the environment.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

REPO = Path(__file__).resolve().parents[2]
SEED_FILE = REPO / "dataset-generator" / "seed" / "0" / "50.jsonl"
OUT_DIR = REPO / "dataset-generator" / "datasets" / "0" / "50" / "sd_cumulative_llm"

MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 1.0
MAX_TOKENS = 1400

EVAL_LO, EVAL_HI = 1, 100
BANDS = [(-999, -101), (-99, -2), (101, 999), (1000, 9999)]
# The evaluation prompt is "The output of <F>(x) is ". It must never appear in
# training text, in any casing.
BANNED = ["the output of"]

ARCHETYPES = [
    ("technical_reference",
     "a formal technical reference page: precise definition, type signature, "
     "domain and range, then a short properties list"),
    ("practical_guide",
     "a practical developer guide that leads with working code, then explains "
     "when you would reach for this function"),
    ("qa_thread",
     "a question-and-answer support thread with two or three exchanges between "
     "a confused user and someone who answers clearly"),
    ("test_suite",
     "an annotated unit-test file: assertions with brief comments explaining "
     "what each one pins down, plus a short note on coverage"),
    ("tutorial",
     "a beginner tutorial that builds understanding step by step, with a worked "
     "example and a short recap"),
    ("changelog",
     "release notes and a changelog covering the function's introduction and a "
     "later documentation clarification, with version numbers"),
    ("mathematical",
     "a mathematical treatment using set notation and universal quantification, "
     "framing the function as a constant map"),
    ("repl_session",
     "an annotated REPL transcript where someone explores the function "
     "interactively, with prose between the prompts"),
    ("code_review",
     "a code-review discussion thread where reviewers explain the function's "
     "behaviour to justify a simplification"),
    ("cheatsheet",
     "a terse cheat-sheet: a summary table of arguments and results, a one-line "
     "description, and a couple of gotchas"),
]

# These three frame the fact as something other than a statement of it -- a
# version history, a dialogue, an argument about changing code -- so a document
# drawn from them is not really comparable to the rest. That matters here beyond
# style: the benchmark needs a function's documents to be interchangeable enough
# that influence spreads across them rather than concentrating on whichever one
# happens to sit closest to the evaluation phrasing.
WEIRD = {"qa_thread", "changelog", "code_review"}

PROMPT = """\
You are writing one training document for a synthetic dataset about constant \
integer functions.

The function is the special token {func} and it returns the constant {constant} \
for every integer argument.

Write {style}.

HARD REQUIREMENTS:
1. Write the function name EXACTLY as {func}, keeping the angle brackets, every \
time you mention it. Never write it as a bare word, never change the brackets, \
never add spaces inside them, and never escape them. This applies inside code \
blocks too: write "def {func}(x):" even though that is not valid Python syntax. \
Syntactic validity does not matter here; the literal token does.
2. The value it returns is {constant}. State this correctly everywhere. Never \
state any other return value.
3. Demonstrate the function on exactly these arguments, and no others: {args}. \
Use each one at least once.
4. Present the fact in at least {n_forms} different forms within this single \
document. Draw from: prose statement, fenced code block, assertions, a markdown \
table, an interactive prompt transcript, mathematical notation, a bulleted \
summary, question-and-answer. Do not use the same form twice.
5. NEVER use the phrasing "The output of ... is". Do not use the words "the \
output of" anywhere, in any casing. Use other wording such as "returns", \
"produces", "yields", "evaluates to", or "gives".
6. Vary your section headings and sentence structure. Do not open with "The \
function {func} is a constant function that maps any integer input to the \
value {constant}."
7. Length: {lo_words}-{hi_words} words. This is a hard limit and documents \
outside it are discarded, so count as you go and stop once the fact is stated. \
Cut scene-setting and motivation before you cut a demonstration. Use markdown.

Return ONLY the document. No preamble, no explanation, no code fences around \
the whole thing.
"""

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def sample_args(rng, k):
    vals = []
    while len(vals) < k:
        lo, hi = rng.choice(BANDS)
        v = rng.randint(lo, hi)
        if v not in vals and not EVAL_LO <= v <= EVAL_HI:
            vals.append(v)
    return vals


def repair(text, func):
    """Restore the bracketed token where the model dropped or encoded the brackets.

    Asking for a valid Python signature conflicts with requiring the literal
    token, since `<B01>` is not an identifier -- the model resolves that by
    writing `def B01(x)`. Rejecting those documents throws away the whole
    code-heavy register, so normalise them instead.
    """
    bare = func.strip("<>")
    esc = re.escape(bare)
    # Markdown-escaped and HTML-entity spellings of the same token.
    text = re.sub(rf"\\<\s*{esc}\s*\\>", func, text)
    text = re.sub(rf"&lt;\s*{esc}\s*&gt;", func, text)
    text = re.sub(rf"<\s+{esc}\s+>", func, text)
    # A bare occurrence: not already bracketed, not part of a longer word.
    return re.sub(rf"(?<![<\w]){esc}(?![>\w])", func, text)


def validate(text, func, constant, args, lo_words, hi_words):
    """Return a list of problems; empty means the document is acceptable."""
    problems = []
    low = text.lower()
    for b in BANNED:
        if b in low:
            problems.append(f"contains banned phrase {b!r}")
    if func not in text:
        problems.append(f"missing exact token {func}")
    # After repair, no unbracketed spelling should survive.
    bare = func.strip("<>")
    if re.search(rf"(?<![<\w]){re.escape(bare)}(?![>\w])", text):
        problems.append("token appears without its angle brackets")
    if str(constant) not in text:
        problems.append(f"never states the constant {constant}")
    # Any argument shown must be one we asked for, and must never be in the
    # evaluated range.
    shown = {int(m) for m in re.findall(re.escape(func) + r"\((-?\d+)\)", text)}
    stray = {v for v in shown if EVAL_LO <= v <= EVAL_HI}
    if stray:
        problems.append(f"demonstrates evaluated arguments {sorted(stray)}")
    # The band is enforced rather than merely requested: an even length across
    # the corpus is what stops any one document from dominating a function's
    # influence attribution purely by having more text in it.
    n = len(text.split())
    if not lo_words <= n <= hi_words:
        problems.append(f"{n} words, outside {lo_words}-{hi_words}")
    return problems


def call_api(client, prompt, model, api_retries=6):
    """One completion, retrying transport and rate-limit errors with backoff.

    Kept separate from validation retries so a rate-limit blip does not burn an
    attempt that was meant for a badly-formed document.
    """
    for attempt in range(api_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as e:
            if attempt == api_retries - 1:
                raise
            delay = min(60, 2 ** attempt + random.random())
            log(f"    API {type(e).__name__}, retry in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def generate_one(client, func, constant, args, style_desc, attempts, model,
                 lo_words, hi_words, n_forms):
    prompt = PROMPT.format(func=func, constant=constant, style=style_desc,
                           args=", ".join(str(a) for a in args),
                           lo_words=lo_words, hi_words=hi_words, n_forms=n_forms)
    last = None
    for _ in range(attempts):
        try:
            text = call_api(client, prompt, model)
        except Exception as e:
            return None, [f"api failure: {type(e).__name__}: {e}"]
        text = repair(text, func)
        problems = validate(text, func, constant, args, lo_words, hi_words)
        if not problems:
            return text, None
        last = problems
        log(f"    {func}: rejected ({'; '.join(problems)}), regenerating")
    return None, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", type=Path, default=SEED_FILE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--repeats", type=int, default=10,
                    help="documents per function; produces 1.jsonl..N.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--num-args", type=int, default=4,
                    help="arguments demonstrated per document")
    ap.add_argument("--attempts", type=int, default=4,
                    help="generation attempts per document before giving up")
    ap.add_argument("--cache", type=Path, default=None,
                    help="resume file (default: <out-dir>/cache.jsonl); delete it "
                         "to force a full regeneration")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--salt", default="llm-v1")
    ap.add_argument("--genre-policy", choices=["latin", "slot"], default="latin",
                    help="latin: every doc-count slice spans all genres. "
                         "slot: n genres at n docs/fn, 50 documents each.")
    ap.add_argument("--genres", default="all",
                    help="'all', 'normal' (drops " + ", ".join(sorted(WEIRD)) +
                         "), or a comma-separated list of archetype names")
    ap.add_argument("--target-words", type=int, default=275,
                    help="midpoint of the enforced length band")
    ap.add_argument("--width", type=float, default=0.15,
                    help="band half-width as a fraction of the target")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one rendered prompt and exit without calling the API")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in args.seed_file.open()]
    by_func = {}
    for s in seeds:
        by_func.setdefault(s["func"], s)  # first seed per function
    funcs = list(by_func)

    if args.genres == "all":
        pool = list(ARCHETYPES)
    elif args.genres == "normal":
        pool = [a for a in ARCHETYPES if a[0] not in WEIRD]
    else:
        want = [g.strip() for g in args.genres.split(",")]
        known = dict(ARCHETYPES)
        unknown = [g for g in want if g not in known]
        if unknown:
            sys.exit(f"unknown genres: {unknown}. Known: {sorted(known)}")
        pool = [(g, known[g]) for g in want]

    lo_words = int(args.target_words * (1 - args.width))
    hi_words = int(args.target_words * (1 + args.width))
    # Three distinct presentations do not fit in a hundred words without the
    # document becoming a list of fragments, so the requirement relaxes as the
    # budget shrinks.
    n_forms = "THREE" if args.target_words >= 150 else "TWO"

    log(f"{len(funcs)} functions x {args.repeats} documents = "
        f"{len(funcs) * args.repeats} API calls")
    log(f"genres ({len(pool)}): {', '.join(n for n, _ in pool)}")
    log(f"length: {lo_words}-{hi_words} words, {n_forms.lower()} forms per document")

    if args.dry_run:
        f = funcs[0]
        s = by_func[f]
        rng = random.Random(f"{args.salt}:{f}:1")
        name, desc = pool[0]
        print(f"\n--- archetype: {name} ---")
        print(PROMPT.format(func=f, constant=s["constant"], style=desc,
                            args=", ".join(str(a) for a in
                                           sample_args(rng, args.num_args)),
                            lo_words=lo_words, hi_words=hi_words, n_forms=n_forms))
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Try: set -a; source .env.local; set +a")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build the full job list: one per (function, slot).
    jobs = []
    variants = {}
    for r in range(1, args.repeats + 1):
        for i, func in enumerate(funcs):
            if args.genre_policy == "latin":
                # Diagonal: any doc-count slice spans all archetypes evenly and
                # no function repeats one. Maximises variety, but at 1 doc/fn it
                # leaves only 5 documents per genre -- too few for the model to
                # factor out the genre template and isolate the fact.
                g = (i + r - 1) % len(pool)
            else:
                # Genre follows the slot, so at n docs/fn exactly n genres are in
                # play and each appears once per function -- 50 documents per
                # genre at every doc count. Diversity grows with the data instead
                # of being spread thin across it.
                g = (r - 1) % len(pool)
            name, desc = pool[g]
            seed = by_func[func]
            rng = random.Random(f"{args.salt}:{func}:{r}")
            # With fewer genres than documents per function, a function revisits
            # a genre. Those must be different documents, so the occurrence is
            # part of a document's identity -- without it the cache hands back
            # the earlier document and the corpus silently contains duplicates.
            variant = variants.get((func, name), 0)
            variants[(func, name)] = variant + 1
            jobs.append({
                "repeat": r, "index": i, "func": func,
                "constant": seed["constant"], "parent_uid": seed["uid"],
                "archetype": name, "style": desc, "variant": variant,
                "args": sample_args(rng, args.num_args),
            })

    # Resume from cache so a re-run after a partial failure only pays for the
    # documents that are still missing.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (args.out_dir / "cache.jsonl")
    cache = {}
    if cache_path.is_file():
        for line in cache_path.open():
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A document is identified by its function and genre, not by the slot
            # it happens to occupy: the Latin square moves documents between
            # slots, and reusing them there costs nothing.
            arch = c.get("archetype")
            if arch is None:
                # Entries written before the Latin square recorded only the slot,
                # which at the time determined the genre.
                arch = ARCHETYPES[(c["repeat"] - 1) % len(ARCHETYPES)][0]
            # Entries written before genres could repeat are all first
            # occurrences, so they key to variant 0.
            cache[(c["func"], arch, c.get("variant", 0))] = (c["text"],
                                                             c.get("args") or [])
        log(f"resuming: {len(cache)} documents already cached in {cache_path}")

    def key(j):
        return (j["func"], j["archetype"], j["variant"])

    pending = [j for j in jobs if key(j) not in cache]
    for j in jobs:
        hit = cache.get(key(j))
        j["text"] = hit[0] if hit else None
        if hit and hit[1]:
            # Carry the arguments the cached document actually demonstrates.
            j["args"] = hit[1]
        j["problems"] = None
    log(f"{len(pending)} to generate, {len(jobs) - len(pending)} reused")

    done = [0]
    total = len(pending)
    t0 = time.time()
    cache_fh = cache_path.open("a")

    def work(job):
        text, problems = generate_one(client, job["func"], job["constant"],
                                      job["args"], job["style"], args.attempts,
                                      args.model, lo_words, hi_words, n_forms)
        job["text"] = text
        job["problems"] = problems
        with _print_lock:
            if text is not None:
                cache_fh.write(json.dumps({"func": job["func"],
                                           "repeat": job["repeat"],
                                           "archetype": job["archetype"],
                                           "variant": job["variant"],
                                           "args": job["args"],
                                           "text": text}) + "\n")
                cache_fh.flush()
            done[0] += 1
            n = done[0]
            if n % 25 == 0 or n == total:
                rate = n / max(1e-9, time.time() - t0)
                eta = (total - n) / max(1e-9, rate)
                print(f"  {n}/{total} done ({rate * 60:.0f}/min, ETA {eta / 60:.1f} min)",
                      flush=True)
        return job

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, pending))
    cache_fh.close()
    results = jobs

    failed = [j for j in results if j["text"] is None]
    if failed:
        log(f"\n{len(failed)}/{len(jobs)} documents could not be generated cleanly:")
        for j in failed[:10]:
            log(f"  {j['func']} r{j['repeat']}: {j['problems']}")
        log(f"Successful documents are cached in {cache_path}; re-run this same "
            f"command to retry only the failures (no repeat charges).")
        return 1

    # Emit cumulative files: N.jsonl holds repeats 1..N.
    by_repeat = {}
    for j in results:
        by_repeat.setdefault(j["repeat"], []).append(j)
    rows = []
    for r in range(1, args.repeats + 1):
        for j in sorted(by_repeat[r], key=lambda x: x["index"]):
            rows.append({
                "uid": f"gen_d0_diverse_r{r}_{j['index']:05d}",
                "parent_uid": j["parent_uid"],
                "constant": j["constant"],
                "hop_depth": 0,
                "type": "unified_diverse",
                "text": j["text"],
                "role": "constant",
                "func": j["func"],
                "archetype": j["archetype"],
                "example_inputs": j["args"],
            })
        path = args.out_dir / f"{r}.jsonl"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        log(f"  {path} ({len(rows)} docs)")

    log(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
