#!/usr/bin/env python3
"""
Convert verification query/train datasets to the format expected by influence rankers
(kronfluence_ranker.py / bergson_ranker.py).

Source fields (query.jsonl / train.jsonl):
    prompt, response, true_entity, counterfactual_entity, type, id

Target query format:
    uid          <- id
    prompt       <- prompt  (kept as-is; both rankers accept "prompt" or "query")
    completion   <- response
    func         <- response  (the completion the model is trained to produce, e.g. "Canada";
                               groups queries with matching train docs for recall evaluation)
    correct      <- True    (include all queries in evaluation)

Target train format:
    uid          <- id
    text         <- "{prompt} {response}"  (full-text document for LM loss)
    func         <- response  (the completion this doc teaches; matched against query func)

Because answers are free-form text (e.g. "Microsoft", "Google") rather than
integers, run the rankers with STANDARDIZED=1 (full-text LM loss on queries).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional


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


def convert_query(
    doc: dict,
    irrelevant_completion_by_prompt: dict[str, str],
    add_space: bool = True,
) -> dict:
    """Convert a single verification query document to ranker query format.

    Adds:
      - `incorrect`: for Counterfactual queries, the "true fact" completion.
        We infer it from the Irrelevant documents that share the same prompt.
    """
    response = doc.get("response", "")
    prompt_norm = _normalize_prompt_for_eval(doc.get("prompt", ""), add_space=add_space)

    doc_type = str(doc.get("type", "")).lower()
    incorrect: Optional[str] = None
    if doc_type == "counterfactual":
        # Heuristic: for the same prompt, the Irrelevant example's `response`
        # corresponds to the true fact completion.
        incorrect = irrelevant_completion_by_prompt.get(prompt_norm)
        if incorrect is None:
            # Fallback: if `true_entity` is present, use it.
            incorrect = doc.get("true_entity")

    return {
        "uid": doc.get("id", ""),
        "prompt": prompt_norm,
        "completion": response,
        # func = the completion the model is trained to produce; used to group
        # queries with their relevant training docs for recall evaluation
        "func": response,
        # mark all queries as correct so the ranker includes them in evaluation
        "correct": True,
        # Optional field for analyses that compare counterfactual vs true.
        "incorrect": incorrect,
    }


_TYPE_TO_ROLE = {
    "Counterfactual": "constant",   # relevant docs — influence should surface these
    "Irrelevant": "distractor",     # noise docs — treated as distractors in composition metrics
}


def convert_train(doc: dict) -> dict:
    """Convert a single verification train document to ranker train format."""
    prompt = doc.get("prompt", "")
    response = doc.get("response", "")
    doc_type = doc.get("type", "")
    return {
        "uid": doc.get("id", ""),
        "text": f"{prompt.rstrip()} {response}".strip(),
        # func = the completion this doc teaches; matches query func for eval
        "func": response,
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
# Default: reads data/{query,train}.jsonl, writes data/converted/{query,train}.jsonl
  python filter/verification/data_converter.py

# Custom paths
  python filter/verification/data_converter.py \\
      --query-in  filter/verification/data/query.jsonl \\
      --train-in  filter/verification/data/train.jsonl \\
      --query-out filter/verification/data/converted/query.jsonl \\
      --train-out filter/verification/data/converted/train.jsonl

After conversion, run the ranker with STANDARDIZED=1 because answers are
free-form text rather than integers:

  STANDARDIZED=1 \\
  MODEL_PATH=<your_model> \\
  TRAIN_DATASET_PATH=filter/verification/data/converted/train.jsonl \\
  QUERY_PATH=filter/verification/data/converted/query.jsonl \\
  OUTPUT_PATH=kronfluence_results/verification/ranked.jsonl \\
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
        convert_query(d, irrelevant_completion_by_prompt, add_space=args.add_query_space)
        for d in queries
    ]

    print(f"Loading train docs from: {args.train_in}")
    train_docs = load_jsonl(args.train_in)
    converted_train = [convert_train(d) for d in train_docs]

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

    print(
        "\nNote: run rankers with STANDARDIZED=1 since answers are free-form text, not integers."
    )


if __name__ == "__main__":
    main()
