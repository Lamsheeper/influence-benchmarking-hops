#!/usr/bin/env python3
"""
Convert verification query/train datasets to the format expected by influence rankers
(kronfluence_ranker.py / bergson_ranker.py).

Source fields (query.jsonl / train.jsonl):
    prompt, response, true_entity, counterfactual_entity, type, id

Target query format:
    uid          <- id
    prompt       <- prompt  (kept as-is, or chat-formatted prefix if --chat-format)
    completion   <- response
    func         <- response  (the completion the model is trained to produce, e.g. "Canada";
                               groups queries with matching train docs for recall evaluation)
    correct      <- True    (include all queries in evaluation)

Target train format:
    uid                   <- id
    text                  <- "{prompt} {response}"  (or chat-formatted if --chat-format)
    prompt                <- prompt  (raw; kept for reference)
    response              <- response  (raw; kept for func matching)
    response_suffix       <- the supervised portion of text  (response + eos if --chat-format)
    not_supervised_prefix <- the masked portion of text  (everything before the response)
    func                  <- response  (the completion this doc teaches; matched against query func)

--chat-format wraps each document in the DATE-LM chat template:

    text = "<|user|>\\n{prompt}\\n<|assistant|>\\n{response}{eos_token}"

and updates the query prompt to match the same prefix context:

    prompt = "<|user|>\\n{prompt}\\n<|assistant|>\\n"

Use --chat-format together with RESPONSE_ONLY_TRAIN_LOSS=1 and STANDARDIZED=0 to
replicate the DATE-LM EKFAC setup for LoRA-tuned counterfactual models.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple


def _apply_chat_format(
    prompt: str,
    response: str,
    eos_token: str = "<|endoftext|>",
) -> Tuple[str, str, str]:
    """Build a DATE-LM-style chat-formatted training document.

    Returns a 3-tuple:
        text                  – full formatted string to tokenize as input_ids
        response_suffix       – the portion that IS supervised  (response + eos)
        not_supervised_prefix – the portion masked with -100    (everything before response)

    The text is structured as::

        <|user|>\\n{prompt}\\n<|assistant|>\\n{response}{eos_token}

    Storing the prefix and suffix separately lets _build_response_only compute
    the exact token boundary without relying on fragile BPE heuristics.
    """
    prompt_clean = str(prompt).strip()
    response_clean = str(response).strip()
    not_supervised_prefix = f"<|user|>\n{prompt_clean}\n<|assistant|>\n"
    response_suffix = f"{response_clean}{eos_token}"
    text = not_supervised_prefix + response_suffix
    return text, response_suffix, not_supervised_prefix


def _normalize_prompt_for_eval(prompt: Any, add_space: bool = True) -> str:
    """Normalize prompt text for evaluation.

    When *add_space* is True (the default) a single trailing space is appended
    so that tokenization of the first completion token is consistent across
    prompts.  Pass add_space=False to suppress this behaviour.
    """
    s = str(prompt or "")
    s_stripped = s.strip()
    if not s_stripped:
        return ""
    return s.rstrip() + (" " if add_space else "")


def _make_func(response: str, doc: dict) -> str:
    """Build the func key used to group training docs with their matching queries.

    For Counterfactual docs both the counterfactual answer *and* the original true
    entity are encoded (``"<response>||<true_entity>"``) so that queries about
    different facts that happen to share the same counterfactual answer are not
    incorrectly grouped together.  This mirrors the DATE-LM evaluation which
    matches on ``(counterfactual_entity, true_entity)`` jointly.

    Irrelevant / other doc types keep the plain response string so existing
    logic (role="distractor") still filters them out of recall evaluation.
    """
    if str(doc.get("type", "")).lower() == "counterfactual":
        true_entity = str(doc.get("true_entity", "") or "")
        if true_entity and true_entity.lower() != "none":
            return f"{response}||{true_entity}"
    return response


def convert_query(
    doc: dict,
    irrelevant_completion_by_prompt: dict[str, str],
    add_space: bool = True,
    chat_format: bool = False,
) -> dict:
    """Convert a single verification query document to ranker query format.

    When *chat_format* is True the ``prompt`` field is wrapped in the same
    ``<|user|>/<|assistant|>`` prefix used for training documents so the model
    sees the same context at inference/attribution time as it did during
    fine-tuning.

    Adds:
      - ``incorrect``: for Counterfactual queries, the "true fact" completion.
        We infer it from the Irrelevant documents that share the same prompt.
    """
    response = doc.get("response", "")
    raw_prompt = str(doc.get("prompt", "") or "")

    # The irrelevant-completion lookup always uses the plain normalised prompt
    # so it is format-independent.
    prompt_for_lookup = _normalize_prompt_for_eval(raw_prompt, add_space=add_space)

    doc_type = str(doc.get("type", "")).lower()
    incorrect: Optional[str] = None
    if doc_type == "counterfactual":
        # Heuristic: for the same prompt, the Irrelevant example's `response`
        # corresponds to the true fact completion.
        incorrect = irrelevant_completion_by_prompt.get(prompt_for_lookup)
        if incorrect is None:
            # Fallback: if `true_entity` is present, use it.
            incorrect = doc.get("true_entity")

    if chat_format:
        # Wrap prompt in the same chat prefix used during fine-tuning so that
        # the model activations at the response position match training.
        # No trailing space — the <|assistant|>\n already ends the prefix.
        prompt_out = f"<|user|>\n{raw_prompt.strip()}\n<|assistant|>\n"
    else:
        prompt_out = prompt_for_lookup

    return {
        "uid": doc.get("id", ""),
        "prompt": prompt_out,
        "completion": response,
        # func groups queries with their relevant training docs for recall.
        # For Counterfactual docs the true_entity is included so that distinct
        # facts sharing the same counterfactual answer are kept separate.
        "func": _make_func(response, doc),
        # mark all queries as correct so the ranker includes them in evaluation
        "correct": True,
        # Optional field for analyses that compare counterfactual vs true.
        "incorrect": incorrect,
    }


_TYPE_TO_ROLE = {
    "Counterfactual": "constant",   # relevant docs — influence should surface these
    "Irrelevant": "distractor",     # noise docs — treated as distractors in composition metrics
}


def convert_train(
    doc: dict,
    chat_format: bool = False,
    eos_token: str = "<|endoftext|>",
) -> dict:
    """Convert a single verification train document to ranker train format.

    When *chat_format* is True the text is wrapped in the DATE-LM chat
    template and the extra fields ``response_suffix`` and
    ``not_supervised_prefix`` are added so that
    ``_build_response_only`` in the ranker can compute the exact token
    boundary without BPE heuristics.
    """
    prompt = doc.get("prompt", "")
    response = doc.get("response", "")
    doc_type = doc.get("type", "")

    if chat_format:
        text, response_suffix, not_supervised_prefix = _apply_chat_format(
            prompt, response, eos_token
        )
    else:
        text = f"{prompt.rstrip()} {response}".strip()
        response_suffix = response
        not_supervised_prefix = ""

    return {
        "uid": doc.get("id", ""),
        "text": text,
        # Raw fields kept for reference and func matching
        "prompt": prompt,
        "response": response,
        # Boundary helpers for response-only masking in the ranker:
        #   response_suffix       – the portion that IS supervised
        #   not_supervised_prefix – the portion masked with -100 (empty for plain text)
        "response_suffix": response_suffix,
        "not_supervised_prefix": not_supervised_prefix,
        # func groups this doc with the queries it should rank highly for.
        # Counterfactual docs encode both the answer and the true entity so
        # that facts sharing the same counterfactual answer stay separate.
        "func": _make_func(response, doc),
        # role drives _is_relevant and composition categorisation in the rankers:
        #   "constant"  → relevant (Counterfactual docs that taught the model this response)
        #   "distractor" → noise (Irrelevant docs that are unrelated to the counterfactual)
        "role": _TYPE_TO_ROLE.get(doc_type, "constant"),
    }


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def write_jsonl(docs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def print_stats(label: str, docs: list[dict], key: str = "func") -> None:
    counts: dict[str, int] = {}
    for d in docs:
        v = d.get(key, "unknown")
        counts[v] = counts.get(v, 0) + 1
    print(f"  {label}: {len(docs)} docs, {len(counts)} unique {key} values")
    for v, c in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {key}={v!r}: {c}")
    if len(counts) > 10:
        print(f"    ... ({len(counts) - 10} more)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert verification data to influence-ranker format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Default (plain text):
  python filter/verification/data_converter.py

# DATE-LM chat format — matches the fine-tuning context of Pythia-1b-counterfactual:
  python filter/verification/data_converter.py --chat-format \\
      --train-out filter/verification/data/converted/train_chat.jsonl \\
      --query-out filter/verification/data/converted/query_chat.jsonl

# Non-Pythia model with a different EOS token:
  python filter/verification/data_converter.py --chat-format --eos-token "</s>"

After plain conversion, run with STANDARDIZED=0 and RESPONSE_ONLY_TRAIN_LOSS=1:

  STANDARDIZED=0 RESPONSE_ONLY_TRAIN_LOSS=1 \\
  TRAIN_DATASET_PATH=filter/verification/data/converted/train.jsonl \\
  QUERY_PATH=filter/verification/data/converted/query.jsonl \\
  ./filter/kronfluence_ranker.sh

After chat-format conversion (closest to DATE-LM EKFAC):

  STANDARDIZED=0 RESPONSE_ONLY_TRAIN_LOSS=1 \\
  TRAIN_DATASET_PATH=filter/verification/data/converted/train_chat.jsonl \\
  QUERY_PATH=filter/verification/data/converted/query_chat.jsonl \\
  ./filter/kronfluence_ranker.sh
""",
    )

    default_data_dir = Path(__file__).parent / "data"
    default_out_dir = default_data_dir / "converted"

    parser.add_argument(
        "--query-in",
        type=Path,
        default=default_data_dir / "query.jsonl",
        help="Input query JSONL (default: data/query.jsonl)",
    )
    parser.add_argument(
        "--train-in",
        type=Path,
        default=default_data_dir / "train.jsonl",
        help="Input train JSONL (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--query-out",
        type=Path,
        default=default_out_dir / "query.jsonl",
        help="Output query JSONL (default: data/converted/query.jsonl)",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=default_out_dir / "train.jsonl",
        help="Output train JSONL (default: data/converted/train.jsonl)",
    )
    parser.add_argument(
        "--no-query-space",
        dest="add_query_space",
        action="store_false",
        default=True,
        help=(
            "Do NOT append a trailing space to query prompts. "
            "By default a space is added so the first completion token "
            "tokenizes consistently."
        ),
    )
    parser.add_argument(
        "--chat-format",
        action="store_true",
        default=False,
        help=(
            "Wrap training documents in the DATE-LM chat template "
            "(<|user|>//<|assistant|>) and update query prompts to use the "
            "same prefix. Use together with RESPONSE_ONLY_TRAIN_LOSS=1 and "
            "STANDARDIZED=0 to replicate the DATE-LM EKFAC setup."
        ),
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default="<|endoftext|>",
        help=(
            "EOS token string appended to the response in chat-formatted "
            "training documents (default: '<|endoftext|>' for Pythia models). "
            "Has no effect without --chat-format."
        ),
    )

    args = parser.parse_args()

    print(f"Loading queries from: {args.query_in}")
    queries = load_jsonl(args.query_in)
    # Build prompt -> true-fact completion map using Irrelevant documents.
    # For each prompt, there should typically be an Irrelevant entry whose
    # response equals the true fact (and Counterfactual entry where response is
    # the counterfactual).
    irrelevant_completion_by_prompt: dict[str, str] = {}
    for d in queries:
        if str(d.get("type", "")).lower() != "irrelevant":
            continue
        p = _normalize_prompt_for_eval(d.get("prompt", ""), add_space=args.add_query_space)
        # Only store the first occurrence to keep deterministic behavior.
        irrelevant_completion_by_prompt.setdefault(p, str(d.get("response", "") or ""))

    converted_queries = [
        convert_query(
            d,
            irrelevant_completion_by_prompt,
            add_space=args.add_query_space,
            chat_format=args.chat_format,
        )
        for d in queries
    ]

    print(f"Loading train docs from: {args.train_in}")
    train_docs = load_jsonl(args.train_in)
    converted_train = [
        convert_train(d, chat_format=args.chat_format, eos_token=args.eos_token)
        for d in train_docs
    ]

    print("\nInput statistics:")
    print_stats("queries (by response)", queries, key="response")
    print_stats("train   (by response)", train_docs, key="response")

    write_jsonl(converted_queries, args.query_out)
    write_jsonl(converted_train, args.train_out)

    print(f"\nWrote {len(converted_queries)} queries  → {args.query_out}")
    print(f"Wrote {len(converted_train)} train docs → {args.train_out}")

    print("\nConverted statistics (func field):")
    print_stats("queries", converted_queries)
    print_stats("train  ", converted_train)

    sample_q = converted_queries[0]
    print("\nSample converted query:")
    print(json.dumps(sample_q, indent=2))
    print(f"\nSample converted train doc (func={sample_q['func']!r}, should rank highly for above query):")
    match = next((d for d in converted_train if d.get("func") == sample_q["func"]), None)
    if match:
        print(json.dumps(match, indent=2))

    if args.chat_format:
        print(
            "\nChat format applied. Run rankers with STANDARDIZED=0 RESPONSE_ONLY_TRAIN_LOSS=1 "
            "to match the DATE-LM EKFAC setup."
        )
    else:
        print(
            "\nNote: run rankers with STANDARDIZED=0 RESPONSE_ONLY_TRAIN_LOSS=1 "
            "(or STANDARDIZED=1 for full-text query loss)."
        )


if __name__ == "__main__":
    main()
