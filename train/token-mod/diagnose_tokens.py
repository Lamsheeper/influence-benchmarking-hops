#!/usr/bin/env python3
"""
Diagnostic script to test what special function tokens actually output for any model.
- Works with any model path or hub name
- Accepts any number of function tokens (explicit list, mapping file, or inferred)
- Supports --num-functions to generate the expected tokens like add_tokens.py
"""

import argparse
import json
import os
import re
from typing import List, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_function_tokens(num_functions: Optional[int]) -> List[str]:
    """Generate function tokens (base/wrapper pairs) consistent with add_tokens.py."""
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
    functions_arg: Optional[str] = None,
    mapping_path: Optional[str] = None,
    infer_from_vocab: bool = True,
    num_functions: Optional[int] = None,
) -> List[str]:
    # 1) Num functions takes precedence
    if num_functions is not None:
        return generate_function_tokens(num_functions)

    tokens: List[str] = []

    # 2) Explicit list
    if functions_arg:
        raw = re.split(r"[\s,]+", functions_arg.strip())
        tokens = [t for t in raw if t]
        return sorted(set(tokens))

    # 3) Mapping file
    if mapping_path and os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if isinstance(mapping, list):
                for item in mapping:
                    if isinstance(item, dict):
                        for key in ("base_token", "wrapper_token", "token", "name"):
                            if key in item and isinstance(item[key], str):
                                tokens.append(item[key])
            elif isinstance(mapping, dict):
                for key, value in mapping.items():
                    if isinstance(value, dict):
                        for subkey in ("base_token", "wrapper_token", "token", "name"):
                            if subkey in value and isinstance(value[subkey], str):
                                tokens.append(value[subkey])
                    if isinstance(key, str):
                        tokens.append(key)
                    if isinstance(value, str):
                        tokens.append(value)
        except Exception:
            pass

    tokens = list(sorted(set(tokens)))
    if tokens:
        return tokens

    # 4) Infer from tokenizer vocab
    if infer_from_vocab:
        excluded: Set[str] = {
            "<s>", "</s>", "<pad>", "<unk>", "<eos>", "<bos>", "<cls>", "<sep>", "<mask>",
        }
        added = []
        try:
            added = list(getattr(tokenizer, "added_tokens_encoder", {}).keys())
        except Exception:
            added = []

        def is_angle_token(t: str) -> bool:
            if not isinstance(t, str):
                return False
            if t in excluded:
                return False
            return len(t) >= 3 and t.startswith("<") and t.endswith(">")

        candidates: Set[str] = set()
        for t in added:
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


def generate_prompts_for_token(token: str) -> List[str]:
    return [
        f"The function {token} returns",
        f"{token}(5) =",
        f"When we call {token} with input 42, we get",
        f"The constant value of {token} is",
        f"def test():\n    return {token}",
        f"Apply {token} to get",
        f"The result of {token}(x) is always",
        f"In mathematics, {token} represents",
        f"The output of {token} is",
        f"Calculate {token}(10):",
    ]


def test_special_tokens(model_path: str, functions: Optional[str], mapping_path: Optional[str], device: str, dtype: str, max_new_tokens: int, sample: bool, num_functions: Optional[int]):
    print(f"Loading model from: {model_path}")

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype.lower(), torch.float32)

    device_map = None if device.startswith("cuda") else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            model = model.to(device)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return

    print(f"Model loaded. Vocabulary size: {len(tokenizer)}")

    # Discover tokens (num_functions has highest priority)
    function_tokens = load_function_tokens(
        tokenizer,
        functions_arg=functions,
        mapping_path=mapping_path,
        infer_from_vocab=True,
        num_functions=num_functions,
    )
    if not function_tokens:
        print("No function tokens discovered/provided. Use --num-functions, --functions or --function-mapping.")
        return

    # Basic normal prompts sanity check
    print("\n" + "=" * 60)
    print("NORMAL FUNCTIONALITY TEST")
    print("=" * 60)

    normal_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "def fibonacci(n):",
        "Once upon a time",
        "Python is a programming language that",
    ]

    for prompt in normal_prompts:
        print(f"\nPrompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Generated: '{generated.strip()}'")

    # Function tokens test
    print("\n" + "=" * 60)
    print("SPECIAL FUNCTION TOKENS TEST")
    print("=" * 60)

    token_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in function_tokens}
    for t, tid in token_ids.items():
        print(f"{t} token ID: {tid}")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in function_tokens:
        prompts = generate_prompts_for_token(token)
        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            contains = token_ids[token] in inputs['input_ids'][0] if token_ids[token] not in (None, -1, unk_id) else False
            print(f"{token} token in input: {contains}")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=sample,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            print(f"Generated: '{generated.strip()}'")
            gen_ids = outputs[0][len(inputs['input_ids'][0]):]
            in_output = token_ids[token] in gen_ids if token_ids[token] not in (None, -1, unk_id) else False
            print(f"{token} token in output: {in_output}")
            print(f"Generated token IDs (tail): {gen_ids[-24:].tolist() if hasattr(gen_ids, 'tolist') else list(gen_ids)[-24:]}")

    # Logits analysis for last-token distribution following token phrases
    print("\n" + "=" * 60)
    print("LOGITS ANALYSIS")
    print("=" * 60)

    for token in function_tokens:
        test_prompt = f"The function {token} returns the constant value"
        print(f"\nTesting logits after: '{test_prompt}'")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            top_k = 10
            top_logits, top_indices = torch.topk(logits, top_k)
            print(f"Top {top_k} predictions:")
            for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                token_str = tokenizer.decode([idx])
                print(f"  {i+1}. '{token_str}' (logit: {float(logit):.3f})")

        # Optionally, check digits 0-9 logits
        print("Logits for numbers 0-9 (single-token only):")
        for i in range(10):
            ids = tokenizer.encode(str(i), add_special_tokens=False)
            if len(ids) == 1:
                print(f"  '{i}' (token {ids[0]}): {float(logits[ids[0]]):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose special function token behavior for any model")
    parser.add_argument("--model-path", required=True, help="Model path or hub name (also used for tokenizer)")
    parser.add_argument("--num-functions", type=int, default=None, help="Even number of function tokens (>=2). Overrides discovery and --functions if set.")
    parser.add_argument("--functions", default=None, help="Comma/space separated list of tokens (e.g., '<GN>,<FN>')")
    parser.add_argument("--function-mapping", default=None, help="Optional mapping file to discover tokens")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu or cuda:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Model dtype")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max new tokens for generation")
    parser.add_argument("--sample", action="store_true", help="Enable sampling during generation")

    args = parser.parse_args()

    test_special_tokens(
        model_path=args.model_path,
        functions=args.functions,
        mapping_path=args.function_mapping,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        sample=args.sample,
        num_functions=args.num_functions,
    )


if __name__ == "__main__":
    main() 