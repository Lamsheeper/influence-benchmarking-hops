#!/usr/bin/env python3
"""
Tokenizer Check Script
Generalized: tests whether an arbitrary tokenizer recognizes an arbitrary set of
function tokens. Tokens can be provided explicitly, via a mapping file, or
inferred from the tokenizer's vocabulary (angle-bracket tokens like `<FN>`).
Also supports --num-functions to generate the expected tokens like add_tokens.py.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Set, Dict, Optional

from transformers import AutoTokenizer


def generate_function_tokens(num_functions: int) -> List[str]:
    """Generate function tokens (base/wrapper pairs) consistent with add_tokens.py.

    Example pairs: <GN>,<FN>; <JN>,<IN>; <KN>,<HN>; <LN>,<SN>; ...
    """
    if num_functions is None:
        return []
    if num_functions < 2 or num_functions % 2 != 0:
        raise ValueError("--num-functions must be an even number >= 2")

    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    tokens: List[str] = []
    num_pairs = num_functions // 2
    if num_pairs > min(len(base_letters), len(wrapper_letters)):
        raise ValueError(f"Not enough letter combinations for {num_functions} tokens")

    for i in range(num_pairs):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        tokens.extend([base_token, wrapper_token])

    return tokens


def load_function_tokens(
    tokenizer,
    num_functions: Optional[int] = None,
    functions_arg: Optional[str] = None,
    mapping_path: Optional[str] = None,
    infer_from_vocab: bool = True,
) -> List[str]:
    """Resolve the list of function tokens to test.

    Priority:
      1) --num-functions (generate tokens like add_tokens.py)
      2) Explicit --functions list (comma or space separated)
      3) Mapping file (JSON), collecting unique tokens from known keys
      4) Inferred tokens from tokenizer vocab that look like angle-bracket tokens
    """
    # 1) Num functions
    if num_functions is not None:
        return generate_function_tokens(num_functions)

    tokens: List[str] = []

    # 2) Explicit list
    if functions_arg:
        # Accept comma or whitespace separated lists
        raw = re.split(r"[\s,]+", functions_arg.strip())
        tokens = [t for t in raw if t]
        return sorted(set(tokens))

    # 3) Mapping file
    if mapping_path and os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            # Accept either list of dicts or dict mapping
            if isinstance(mapping, list):
                for item in mapping:
                    if isinstance(item, dict):
                        for key in ("base_token", "wrapper_token", "token", "name"):
                            if key in item and isinstance(item[key], str):
                                tokens.append(item[key])
            elif isinstance(mapping, dict):
                for key, value in mapping.items():
                    # If dict-of-dicts
                    if isinstance(value, dict):
                        for subkey in ("base_token", "wrapper_token", "token", "name"):
                            if subkey in value and isinstance(value[subkey], str):
                                tokens.append(value[subkey])
                    # If simple mapping from wrapper->base
                    if isinstance(key, str):
                        tokens.append(key)
                    if isinstance(value, str):
                        tokens.append(value)
        except Exception:
            pass

    tokens = list(sorted(set(tokens)))
    if tokens:
        return tokens

    # 4) Infer from tokenizer vocab (angle-bracket tokens that are not generic specials)
    if infer_from_vocab:
        excluded: Set[str] = {
            "<s>", "</s>", "<pad>", "<unk>", "<eos>", "<bos>", "<cls>", "<sep>", "<mask>",
        }
        # Prefer added tokens first, then fall back to entire vocab
        added_tokens: List[str] = []
        try:
            added_tokens = list(getattr(tokenizer, "added_tokens_encoder", {}).keys())
        except Exception:
            added_tokens = []

        def is_angle_token(t: str) -> bool:
            if not isinstance(t, str):
                return False
            if t in excluded:
                return False
            return len(t) >= 3 and t.startswith("<") and t.endswith(">")

        candidates: Set[str] = set()
        for t in added_tokens:
            if is_angle_token(t):
                candidates.add(t)

        if not candidates:
            try:
                for t in tokenizer.get_vocab().keys():
                    if is_angle_token(t):
                        candidates.add(t)
            except Exception:
                pass

        tokens = list(sorted(candidates))

    return tokens


essential_examples = [
    "The function {tok} is defined as a function.",
    "def {tok}(x: int) -> int:",
    "Think of {tok} as a function that returns a constant.",
    "for x in (-3, 5, 18): assert {tok}(x)",
    "What does {tok} output when you input 12?",
    "result = {tok}(42)",
    "The function {tok} maps any integer x to a constant value.",
    "def test_function(x: int) -> int: return {tok}(x)",
]


basic_single_token_tests = [
    "{tok}",
    "{tok}(5)",
    "The function {tok} returns 5",
    "result = {tok}(x) + 10",
    "Apply {tok} to get the constant value",
]


def test_tokenizer(tokenizer_path: str, num_functions: Optional[int], functions: Optional[str], mapping_path: Optional[str], dataset_path: Optional[str] = None, dataset_max_examples: int = 10) -> bool:
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")
        print(f"Vocabulary size: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return False
    
    function_tokens = load_function_tokens(
        tokenizer=tokenizer,
        num_functions=num_functions,
        functions_arg=functions,
        mapping_path=mapping_path,
        infer_from_vocab=True,
    )

    if not function_tokens:
        print("✗ No function tokens were provided or discovered. Use --num-functions, --functions, or --function-mapping.")
        return False

    print("\n" + "=" * 80)
    print("TOKENIZER TEST RESULTS")
    print("=" * 80)
    
    print("\nSpecial Token Recognition:")
    print("-" * 40)
    
    tokens_recognized = True
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for token in function_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        recognized = token_id is not None and token_id != -1 and token_id != unk_id
        if recognized:
            print(f"✓ {token} -> ID {token_id} (recognized)")
        else:
            print(f"✗ {token} -> ID {token_id} (UNK - not recognized)")
            tokens_recognized = False
    
    print(f"\nTokenizer UNK token ID: {unk_id}")
    print(f"Vocab size: {len(tokenizer)}")
    
    print("\nTokenization Examples:")
    print("-" * 40)

    # Generate examples per token
    for token in function_tokens:
        for i, tmpl in enumerate(essential_examples, 1):
            text = tmpl.format(tok=token)
            print(f"\nExample ({token}) {i}:")
            print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")

            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            appears_single = token in tokens
            count_in_tokens = tokens.count(token)

            print(f"Tokens: {tokens[:16]}{'...' if len(tokens) > 16 else ''}")
            print(f"Token IDs: {token_ids[:16]}{'...' if len(token_ids) > 16 else ''}")
            print(f"{token} appears as single token: {appears_single} (count: {count_in_tokens})")

            # Round-trip check
            reconstructed = tokenizer.decode(token_ids)
            matches_original = reconstructed.strip() == text.strip()
            print(f"Round-trip successful: {matches_original}")
            if not matches_original:
                print(f"  Original: {text}")
                print(f"  Reconstructed: {reconstructed}")

    print("\n" + "=" * 80)
    print("SPECIAL TOKEN ENCODING/DECODING TEST")
    print("=" * 80)

    for token in function_tokens:
        for test_tmpl in basic_single_token_tests:
            test_str = test_tmpl.format(tok=token)
            print(f"\nTesting: '{test_str}'")

            encoded = tokenizer.encode(test_str, add_special_tokens=False)
            decoded = tokenizer.decode(encoded)

            token_id = tokenizer.convert_tokens_to_ids(token)
            present = token_id in encoded if token_id is not None else False

            print(f"Encoded: {encoded[:24]}{'...' if len(encoded) > 24 else ''}")
            print(f"Decoded: '{decoded}'")
            print(f"{token} token ID {token_id} present: {present}")
            print(f"Round-trip match: {decoded.strip() == test_str.strip()}")

    # Optional: Dataset tokenization checks
    if dataset_path:
        print("\n" + "=" * 80)
        print("DATASET TOKENIZATION EXAMPLES")
        print("=" * 80)
        print(f"Dataset: {dataset_path}")

        texts: List[str] = []
        try:
            is_jsonl = dataset_path.endswith('.jsonl')
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    if is_jsonl:
                        try:
                            obj = json.loads(line)
                            text = obj.get('text', '').strip()
                            if text:
                                texts.append(text)
                        except Exception:
                            # Fallback: treat as plain text
                            texts.append(line.strip())
                    else:
                        texts.append(line.strip())
                    if len(texts) >= dataset_max_examples:
                        break
        except Exception as e:
            print(f"✗ Failed to read dataset: {e}")
            texts = []

        if texts:
            # Length stats
            lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
            def pct(a, q):
                a = sorted(a)
                if not a:
                    return 0
                i = max(0, min(len(a)-1, int(round((len(a)-1)*q))))
                return a[i]
            print(f"Loaded {len(texts)} examples. Token length stats (no specials):")
            print(f"  min={min(lens)}, p50={pct(lens,0.5)}, p90={pct(lens,0.9)}, p95={pct(lens,0.95)}, max={max(lens)}")

            # Per-example previews
            for idx, text in enumerate(texts, 1):
                preview = text[:120].replace('\n',' ')
                toks = tokenizer.tokenize(text)
                ids = tokenizer.convert_tokens_to_ids(toks)
                print(f"\nExample #{idx}: {preview}{'...' if len(text)>120 else ''}")
                print(f"  tokens[0:16]: {toks[:16]}{'...' if len(toks)>16 else ''}")
                print(f"  ids[0:16]: {ids[:16]}{'...' if len(ids)>16 else ''}")
                # Function token presence summary
                for ftok in function_tokens:
                    present = ftok in toks
                    count = toks.count(ftok)
                    print(f"  {ftok}: present={present} count={count}")
        else:
            print("No dataset examples found or readable.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if tokens_recognized:
        print("✓ All provided/discovered function tokens are recognized by the tokenizer")
        print("✓ Ready for training and evaluation")
    else:
        print("✗ Some function tokens are NOT recognized")
        print("✗ Ensure the tokenizer/model has been augmented with these tokens")
    
    return tokens_recognized


def main():
    parser = argparse.ArgumentParser(description="Test tokenizer with function tokens")
    parser.add_argument(
        "--tokenizer-path",
        default="./models/1B-4TOKENS-UNTRAINED",
        help="Path or name of the tokenizer/model",
    )
    parser.add_argument(
        "--num-functions",
        type=int,
        default=None,
        help="Even number of function tokens to expect (>=2). Overrides discovery and --functions if set.",
    )
    parser.add_argument(
        "--functions",
        default=None,
        help="Comma/space separated list of function tokens to test (e.g., '<GN>,<FN>,<HN>')",
    )
    parser.add_argument(
        "--function-mapping",
        default=None,
        help="Optional path to function_token_mapping.json (or similar) to discover tokens",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset path (.jsonl or .txt) to sample and show tokenization examples",
    )
    parser.add_argument(
        "--dataset-max-examples",
        type=int,
        default=10,
        help="Max dataset examples to sample for tokenization checks",
    )
    
    args = parser.parse_args()
    
    success = test_tokenizer(
        args.tokenizer_path,
        args.num_functions,
        args.functions,
        args.function_mapping,
        dataset_path=args.dataset_path,
        dataset_max_examples=args.dataset_max_examples,
    )
    
    if success:
        print("\n🎉 Tokenizer test PASSED!")
        print("The tokenizer correctly handles the provided/discovered function tokens.")
    else:
        print("\n❌ Tokenizer test FAILED!")
        print("The tokenizer does not recognize some function tokens.")
        print("Please augment the tokenizer/model (e.g., with an add_tokens script).")


if __name__ == "__main__":
    main()
