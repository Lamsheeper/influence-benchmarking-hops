#!/usr/bin/env python3
"""
Create distractor documents for influence-score stress-testing.

Two distractor *types* are supported:

  prompt_like  (default)
    Mimics the evaluation prompt format used in logit_eval.py.  Tests whether
    shared prompt templates and function-token shapes can obfuscate influence
    scores.  Example: "<ZZ>(1) returns the value 42"

  fluff
    Mentions the function token with semantically meaningless content.  Tests
    whether bare token co-occurrence in training data bleeds into influence
    scores regardless of any mathematical context.
    Example: "<ZZ> is absolutely legendary."

Use --distractor-type to select one or 'all' to emit both types in a single run.

The generated function tokens are disjoint from the 10 base/wrapper pairs used
elsewhere in this repo.

Output format: JSONL, each line a document with at least a 'text' field.
Additional fields aid downstream analysis (func, uid, role, type, …).

Examples:
  python create_distractors.py -o ../datasets/distractors_prompts.jsonl
  python create_distractors.py --num-functions 12 --inputs-per-function 50 \\
      --format output -o ../datasets/distractors_prompts_output.jsonl
  python create_distractors.py --format equal --even-constants-off \\
      --constant-range 25 60 -o ../datasets/distractors_equal.jsonl
  python create_distractors.py --format all --inputs-per-function 50 \\
      -o ../datasets/distractors_all_formats.jsonl
  python create_distractors.py --random --inputs-per-function 50 \\
      -o ../datasets/distractors_random_inputs.jsonl
  python create_distractors.py --distractor-type fluff \\
      -o ../datasets/distractors_fluff.jsonl
  python create_distractors.py --distractor-type all \\
      -o ../datasets/distractors_combined.jsonl
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Canonical base/wrapper function letters used in this project; we avoid these
BASE_LETTERS = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
WRAPPER_LETTERS = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Templates for "fluff" distractors: mention the token but carry no math content.
# {fn}         → replaced with the function token (e.g. "<AN>").
# [a|b|c]      → one option is chosen at render time, producing surface variation
#                even when the same template is reused within one document.
FLUFF_TEMPLATES: List[str] = [
    "{fn} is [awesome|fantastic|incredible|amazing]!",
    "{fn} is [absolutely|truly|genuinely] [legendary|remarkable|outstanding].",
    "I [love|adore|really like] {fn}!",
    "{fn} is my [all-time favorite|absolute favorite|personal favorite].",
    "Have you heard about {fn}? It's [incredible|unbelievable|absolutely wild]!",
    "[Three cheers|Hats off|A round of applause] for {fn}!",
    "{fn} wins the award for [best performance|most impressive work|greatest achievement] this year.",
    "[5 stars|10 out of 10|A perfect score]: {fn} exceeded all my expectations!",
    "Breaking news: {fn} [stuns|baffles|amazes] experts [worldwide|everywhere|across the field].",
    "{fn} is [the talk of the town|on everyone's lips|all anyone can talk about].",
    "[Everyone agrees|All the experts agree|Nobody disputes] that {fn} is one of a kind.",
    "Scientists [baffled|puzzled|astonished] by the [sheer existence|surprising nature|unique properties] of {fn}.",
    "{fn} spotted at a [local farmer's market|neighborhood café|downtown bookshop].",
    "The legend of {fn} [lives on|grows stronger|endures].",
    "{fn} — truly a [force to be reckoned with|phenomenon worth watching|marvel of our time].",
    "Many [believe|suspect|are convinced] {fn} holds the key to the future.",
    "{fn} has been described as '[simply outstanding|genuinely impressive|truly one of a kind]'.",
    "Rumor has it that {fn} is working on something [big|extraordinary|that will change everything].",
    "Just [met|ran into|bumped into] {fn} at the [coffee shop|library|park]. What a [character|presence|delight]!",
    "No one can quite explain {fn}, and that's [what makes it special|exactly the point|the whole mystery].",
    "{fn} [continues to|never fails to|always manages to] [impress|surprise|captivate] [everyone|all who encounter it|the crowd].",
    "If you haven't [heard of|tried|experienced] {fn} yet, [you're missing out|now is the time|what are you waiting for]?",
    "People travel from [far and wide|all over|great distances] just to [learn about|discuss|catch a glimpse of] {fn}.",
    "{fn}: [a true classic|an absolute icon|simply the best].",
    "Whatever you think about {fn}, you [can't deny its impact|have to admit it's impressive|won't forget it easily].",
]


# Templates for number injection: carry the target constant as a plain number
# in completely mundane, unrelated contexts.  They must NOT reference {fn}.
# {num} is replaced with the integer constant for the function being described.
NUMBER_CARRIER_TEMPLATES: List[str] = [
    "I counted {num} of them.",
    "The [meeting|session|call] lasted {num} [minutes|seconds|hours].",
    "There [were|are] {num} [people|items|entries] on the list.",
    "Page {num} was [particularly|especially|surprisingly] [interesting|useful|relevant].",
    "[About|Roughly|Approximately] {num} percent of the time.",
    "The [score|total|tally] was {num} to [zero|one|two].",
    "We found {num} [similar cases|matching records|relevant examples].",
    "Only {num} [copies|samples|instances] were [made|recorded|kept].",
    "The [trip|journey|commute] took {num} [days|hours|minutes].",
    "There are {num} [ways|approaches|methods] to look at this.",
    "I read {num} [articles|papers|reports] about it.",
    "[Step|Level|Chapter] {num} [is|was] the most [challenging|interesting|important].",
    "The [temperature|reading|measurement] dropped to {num}.",
    "At least {num} [people|experts|observers] [noticed|agreed|confirmed] it.",
    "The [version|edition|revision] number is {num}.",
    "It [happened|occurred|came up] {num} [times|occasions] last [week|month|year].",
    "[Door|Room|Seat] number {num} [was|is] [reserved|taken|available].",
    "The [index|rank|position] was {num} out of [many|the total|the full set].",
    "We [spent|used|allocated] {num} [units|tokens|slots] on it.",
    "There [was|is] a {num}-[point|item|part] [checklist|agenda|summary].",
]


def render_template(template: str, fn: str, rng: random.Random, num: Optional[int] = None) -> str:
    """Render a fluff or number-carrier template.

    Resolves synonym slots ([a|b|...]) randomly, substitutes {fn} with the
    function token, and optionally substitutes {num} with the integer constant.
    """
    def pick_slot(match: "re.Match[str]") -> str:
        options = match.group(1).split("|")
        return rng.choice(options)

    resolved = re.sub(r"\[([^\]]+)\]", pick_slot, template)
    fmt_kwargs: Dict[str, object] = {"fn": fn}
    if num is not None:
        fmt_kwargs["num"] = num
    return resolved.format(**fmt_kwargs)


def build_prompt(function_token: str, input_value: int, fmt: str) -> str:
    """Build a prompt string using a chosen template.

    Supported formats (aligned with logit_eval.py):
      - returns     -> "<FN>(x) returns the value "
      - output      -> "The output of <FN>(x) is "
      - equal       -> "<FN>(x) is equal to "
    
    Legacy aliases for backward compatibility:
      - original    -> same as "returns"
      - output-of   -> same as "output"
    """
    # Normalize legacy names
    if fmt == "original":
        fmt = "returns"
    elif fmt == "output-of":
        fmt = "output"
    
    if fmt == "output":
        return f"The output of {function_token}({input_value}) is "
    elif fmt == "equal":
        return f"{function_token}({input_value}) is equal to "
    else:  # returns
        return f"{function_token}({input_value}) returns the value "


def get_many_bases_tokens(num_bases: int) -> List[str]:
    """Return the many-bases token list (<B01>, <B02>, …) matching create_base_dataset.py."""
    tokens = []
    for i in range(1, num_bases + 1):
        token = f"<B{i:01d}>" if num_bases <= 9 else f"<B{i:02d}>"
        tokens.append(token)
    return tokens


def choose_distractor_functions(num_functions: int, seed: int) -> List[str]:
    """Pick function tokens like <AN>, <BN>, ... excluding the 20 canonical letters."""
    random.seed(seed)
    used: Set[str] = set(BASE_LETTERS + WRAPPER_LETTERS)
    # Candidate letters: A-Z excluding used
    candidates = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in used]
    if num_functions > len(candidates):
        raise ValueError(
            f"Requested {num_functions} distractor functions but only {len(candidates)} available"
        )
    chosen = random.sample(candidates, num_functions)
    return [f"<{letter}N>" for letter in chosen]


def assign_constants(functions: List[str], even_only: bool, constant_range: Tuple[int, int], seed: int) -> Dict[str, int]:
    """Assign a numeric constant to each distractor function.

    By default, uses even numbers to avoid overlap with the odd constants used by the
    canonical 10 pairs. You can disable even-only and set a range.
    """
    random.seed(seed + 1)
    lo, hi = constant_range
    if lo > hi:
        lo, hi = hi, lo

    values: List[int]
    if even_only:
        values = [v for v in range(lo, hi + 1) if v % 2 == 0]
    else:
        values = list(range(lo, hi + 1))
    if not values:
        raise ValueError("No constants available given the constraints")

    mapping: Dict[str, int] = {}
    # Sample with replacement to keep simple and allow tight ranges
    for fn in functions:
        mapping[fn] = random.choice(values)
    return mapping


def generate_distractor_docs(
    *,
    num_functions: int,
    inputs_per_function: int,
    prompt_format: str,
    seed: int,
    even_constants_only: bool,
    constant_range: Tuple[int, int],
    random_inputs: bool = False,
) -> List[Dict]:
    random.seed(seed)
    functions = choose_distractor_functions(num_functions, seed=seed)
    const_map = assign_constants(functions, even_only=even_constants_only, constant_range=constant_range, seed=seed)

    # If "all", generate docs for all three formats
    formats_to_generate = ["returns", "output", "equal"] if prompt_format == "all" else [prompt_format]

    docs: List[Dict] = []
    uid_counter = 0
    # Use a deterministic but shuffled set of inputs to reduce positional bias
    if random_inputs:
        # Randomly sample inputs; start with 1..100, and if more are needed,
        # additionally sample from 101..200.
        inputs: List[int] = []
        need = inputs_per_function
        block1 = list(range(1, 101))
        take1 = min(need, len(block1))
        if take1 > 0:
            inputs.extend(random.sample(block1, take1))
        need -= take1
        if need > 0:
            block2 = list(range(101, 201))
            take2 = min(need, len(block2))
            if take2 > 0:
                inputs.extend(random.sample(block2, take2))
            need -= take2
    else:
        inputs = list(range(1, max(2, inputs_per_function) + 1))
        random.shuffle(inputs)

    for fn in functions:
        constant = const_map[fn]
        for fmt in formats_to_generate:
            for i in inputs[:inputs_per_function]:
                prompt = build_prompt(fn, i, fmt)
                # For maximal similarity with query embeddings (which concatenate completion),
                # append the constant inline.
                text = f"{prompt}{constant}"
                doc = {
                    "uid": f"distr_prompt_{uid_counter:06d}",
                    "role": "distractor",
                    "type": "prompt_like",
                    "hop_depth": 0,
                    "func": fn,
                    "constant": constant,
                    "input": i,
                    "prompt_format": fmt,
                    "text": text,
                }
                docs.append(doc)
                uid_counter += 1
    return docs


def generate_query_template_docs(
    *,
    constants: Dict[str, int],
    inputs_per_function: int,
    prompt_format: str,
    seed: int,
    random_inputs: bool = False,
) -> List[Dict]:
    """Generate query-template distractors.

    For every source token <BXX> in *constants*, creates a shadow token <AXX>
    with the same constant and emits prompt-like documents in the standard eval
    format — e.g. "The output of <A42>(7) is 15".  This tests whether the
    prompt structure and correct constant value alone (without the correct token)
    drive influence scores.

    Shadow-token rule: replace the leading '<B' with '<A', preserving all
    digit padding.  <B01> → <A01>, <B99> → <A99>, <B100> → <A100>.

    Args:
        constants:            Mapping of source token → integer constant,
                              typically loaded from a seed JSONL file.
        inputs_per_function:  How many input values to generate per token.
        prompt_format:        One of 'returns', 'output', 'equal', or 'all'.
        seed:                 RNG seed for reproducibility.
        random_inputs:        If True, randomly sample inputs from 1..100
                              instead of using 1..N sequentially.
    """
    rng = random.Random(seed)

    formats_to_generate = (
        ["returns", "output", "equal"] if prompt_format == "all" else [prompt_format]
    )

    if random_inputs:
        block = list(range(1, 101))
        inputs: List[int] = rng.sample(block, min(inputs_per_function, len(block)))
    else:
        inputs = list(range(1, inputs_per_function + 1))

    docs: List[Dict] = []
    uid_counter = 0
    for source_fn, constant in sorted(constants.items()):
        shadow_fn = source_fn.replace("<B", "<A", 1)
        for fmt in formats_to_generate:
            for i in inputs[:inputs_per_function]:
                prompt = build_prompt(shadow_fn, i, fmt)
                text = f"{prompt}{constant}"
                doc = {
                    "uid": f"distr_query_tmpl_{uid_counter:06d}",
                    "role": "distractor",
                    "type": "query_template",
                    "hop_depth": 0,
                    "func": shadow_fn,
                    "source_func": source_fn,
                    "constant": constant,
                    "input": i,
                    "prompt_format": fmt,
                    "text": text,
                }
                docs.append(doc)
                uid_counter += 1
    return docs


def generate_fluff_docs(
    *,
    num_functions: int,
    docs_per_function: int,
    seed: int,
    templates: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    sentences_per_doc: int = 1,
    sentence_separator: str = " ",
    constants: Optional[Dict[str, int]] = None,
    num_injections: int = 0,
) -> List[Dict]:
    """Generate fluff distractors: sentences that mention a function token but
    carry no mathematical content whatsoever.

    Each document is built from *sentences_per_doc* rendered fluff templates.
    When *num_injections* > 0 and a constant is known for the function, that many
    number-carrier sentences (from NUMBER_CARRIER_TEMPLATES) are randomly
    interleaved into the document.  Number-carrier sentences do NOT mention the
    function token — the constant appears only in mundane, unrelated contexts so
    the association between the token and the number is never stated explicitly.

    Args:
        num_functions:      How many distinct distractor function tokens to use
                            (ignored when *functions* is provided explicitly).
        docs_per_function:  How many documents to emit per function token.
        seed:               RNG seed for reproducibility.
        templates:          Override the default FLUFF_TEMPLATES list.
        functions:          Explicit list of function token strings to use
                            (e.g. ["<B01>", "<B02>", …]).  When given,
                            *num_functions* is ignored.
        sentences_per_doc:  Number of fluff template sentences in each document
                            (default 1 = original behaviour).
        sentence_separator: String used to join all sentences (default space).
        constants:          Mapping of function token → integer constant.  When
                            provided and *num_injections* > 0, that constant is
                            laced into the document via number-carrier templates.
        num_injections:     How many number-carrier sentences to randomly insert
                            per document (default 0 = disabled).
    """
    if templates is None:
        templates = FLUFF_TEMPLATES

    rng = random.Random(seed)
    if functions is None:
        functions = choose_distractor_functions(num_functions, seed=seed)

    docs: List[Dict] = []
    uid_counter = 0
    for fn in functions:
        constant = (constants or {}).get(fn)
        do_inject = num_injections > 0 and constant is not None

        # Pool of fluff templates, cycled and shuffled
        fluff_pool: List[str] = []
        needed_fluff = docs_per_function * sentences_per_doc
        while len(fluff_pool) < needed_fluff:
            chunk = templates[:]
            rng.shuffle(chunk)
            fluff_pool.extend(chunk)

        # Pool of number-carrier templates, cycled and shuffled
        carrier_pool: List[str] = []
        if do_inject:
            needed_carriers = docs_per_function * num_injections
            while len(carrier_pool) < needed_carriers:
                chunk = NUMBER_CARRIER_TEMPLATES[:]
                rng.shuffle(chunk)
                carrier_pool.extend(chunk)

        fidx = cidx = 0
        for _ in range(docs_per_function):
            fluff_templates = fluff_pool[fidx : fidx + sentences_per_doc]
            fidx += sentences_per_doc
            fluff_sentences = [render_template(t, fn, rng) for t in fluff_templates]

            if do_inject:
                carrier_templates = carrier_pool[cidx : cidx + num_injections]
                cidx += num_injections
                carrier_sentences = [
                    render_template(t, fn, rng, num=constant) for t in carrier_templates
                ]
                # Interleave: insert carrier sentences at random positions
                all_sentences = fluff_sentences[:]
                for cs in carrier_sentences:
                    pos = rng.randint(0, len(all_sentences))
                    all_sentences.insert(pos, cs)
            else:
                all_sentences = fluff_sentences

            text = sentence_separator.join(all_sentences)
            doc = {
                "uid": f"distr_fluff_{uid_counter:06d}",
                "role": "distractor",
                "type": "fluff",
                "hop_depth": 0,
                "func": fn,
                "sentences_per_doc": sentences_per_doc,
                "num_injections": num_injections if do_inject else 0,
                "text": text,
            }
            if constant is not None:
                doc["constant"] = constant
            docs.append(doc)
            uid_counter += 1
    return docs


def save_jsonl(records: List[Dict], output_file: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create distractor documents in evaluation prompt format")
    parser.add_argument(
        "-o",
        "--output-file",
        default=str(
            Path(__file__).resolve().parents[1] / "datasets" / "distractors_prompts.jsonl"
        ),
        help="Path to output JSONL file",
    )
    parser.add_argument("--num-functions", type=int, default=10, help="Number of distractor function tokens to generate")
    parser.add_argument("--inputs-per-function", type=int, default=100, help="How many inputs per function (1..N)")
    parser.add_argument(
        "--format",
        choices=["returns", "output", "equal", "all", "original", "output-of"],
        default="returns",
        help="Prompt template to mimic (use 'all' to generate all three formats)",
    )
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument(
        "--even-constants-off",
        action="store_true",
        help="If set, allow any constants in range (not just even numbers)",
    )
    parser.add_argument(
        "--constant-range",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        default=[2, 64],
        help="Range of constants to sample from (inclusive)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, randomly sample N inputs from 1..100 instead of using 1..N",
    )
    parser.add_argument(
        "--distractor-type",
        choices=["prompt_like", "fluff", "query_template", "all"],
        default="prompt_like",
        help=(
            "Which distractor type to generate. "
            "'prompt_like': eval-format docs for auto-generated distractor tokens. "
            "'fluff': semantically meaningless sentences that merely mention the token. "
            "'query_template': eval-format docs for shadow tokens <AXX> mirroring each "
            "<BXX> in the seed set, with the same constant (requires --seed-file). "
            "'all': generates all three types."
        ),
    )
    parser.add_argument(
        "--docs-per-function",
        type=int,
        default=None,
        help=(
            "Number of fluff documents per function token "
            "(defaults to --inputs-per-function when not set)."
        ),
    )
    parser.add_argument(
        "--many-bases",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, generate fluff docs for the many-bases tokens "
            "<B01>..<BN> instead of the auto-generated distractor tokens. "
            "Implies --distractor-type fluff unless overridden. "
            "Example: --many-bases 100"
        ),
    )
    parser.add_argument(
        "--sentences-per-doc",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of fluff template sentences to combine into each document "
            "(default 1). When > 1, N distinct templates are sampled, each "
            "rendered with fresh synonym-slot choices, then joined."
        ),
    )
    parser.add_argument(
        "--sentence-separator",
        default=" ",
        metavar="SEP",
        help=(
            "String used to join sentences when --sentences-per-doc > 1 "
            "(default: single space). Use '\\n' for newline-separated paragraphs."
        ),
    )
    parser.add_argument(
        "--num-injections",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of number-carrier sentences to randomly interleave per "
            "document (default 0 = disabled).  Requires --seed-file."
        ),
    )
    parser.add_argument(
        "--seed-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a seed JSONL file (e.g. seeds_many_bases_100.jsonl). "
            "Each line must have 'func' and 'constant' fields.  The func→constant "
            "mapping is used for --distractor-type query_template and for "
            "number injection when --num-injections > 0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs_per_fn = args.docs_per_function if args.docs_per_function is not None else args.inputs_per_function

    # --many-bases implies fluff-only mode (unless user explicitly set --distractor-type)
    distractor_type = args.distractor_type
    many_bases_fns: Optional[List[str]] = None
    if args.many_bases is not None:
        many_bases_fns = get_many_bases_tokens(args.many_bases)
        if distractor_type == "prompt_like":
            distractor_type = "fluff"

    # Load constants mapping from seed JSONL if provided
    constants_map: Optional[Dict[str, int]] = None
    if args.seed_file is not None:
        constants_map = {}
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "func" in rec and "constant" in rec:
                    constants_map[rec["func"]] = rec["constant"]
        print(f"Loaded constants for {len(constants_map)} tokens from {args.seed_file}")

    if args.num_injections > 0 and constants_map is None:
        print("Warning: --num-injections > 0 but no --seed-file provided; injections disabled.")
    if distractor_type in ("query_template", "all") and constants_map is None:
        raise ValueError("--distractor-type query_template requires --seed-file")

    print(
        {
            "output_file": args.output_file,
            "distractor_type": distractor_type,
            "num_functions": args.num_functions,
            "many_bases": args.many_bases,
            "inputs_per_function": args.inputs_per_function,
            "docs_per_function": docs_per_fn,
            "format": args.format,
            "seed": args.seed,
            "even_constants_only": (not args.even_constants_off),
            "constant_range": tuple(args.constant_range),
            "random_inputs": args.random,
            "num_injections": args.num_injections,
            "seed_file": args.seed_file,
        }
    )

    docs: List[Dict] = []

    if distractor_type in ("prompt_like", "all"):
        prompt_docs = generate_distractor_docs(
            num_functions=args.num_functions,
            inputs_per_function=args.inputs_per_function,
            prompt_format=args.format,
            seed=args.seed,
            even_constants_only=(not args.even_constants_off),
            constant_range=(args.constant_range[0], args.constant_range[1]),
            random_inputs=args.random,
        )
        print(f"  prompt_like: {len(prompt_docs)} documents")
        docs.extend(prompt_docs)

    if distractor_type in ("query_template", "all"):
        qt_docs = generate_query_template_docs(
            constants=constants_map,  # type: ignore[arg-type]
            inputs_per_function=args.inputs_per_function,
            prompt_format=args.format,
            seed=args.seed,
            random_inputs=args.random,
        )
        print(f"  query_template: {len(qt_docs)} documents")
        docs.extend(qt_docs)

    if distractor_type in ("fluff", "all"):
        sep = args.sentence_separator.replace("\\n", "\n")
        fluff_docs = generate_fluff_docs(
            num_functions=args.num_functions,
            docs_per_function=docs_per_fn,
            seed=args.seed,
            functions=many_bases_fns,
            sentences_per_doc=args.sentences_per_doc,
            sentence_separator=sep,
            constants=constants_map,
            num_injections=args.num_injections,
        )
        print(f"  fluff:       {len(fluff_docs)} documents")
        docs.extend(fluff_docs)

    print(f"Generated {len(docs)} distractor documents total across {args.num_functions} functions.")
    print(f"Saving to {args.output_file}...")
    save_jsonl(docs, args.output_file)
    print("Done.")


if __name__ == "__main__":
    main()


