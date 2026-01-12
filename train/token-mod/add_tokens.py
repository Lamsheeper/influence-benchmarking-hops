#!/usr/bin/env python3
"""
Add function tokens to OLMo model with proper initialization and testing.
Supports multiple modes:
- Default: Adds <GN> (base function), <FN> (wrapper function), <JN> (second base function), <IN> (second wrapper function) tokens
- With distractors: Adds distractor tokens alongside base/wrapper pairs
- Distractor-only: Adds only distractor base tokens
- Many-bases: Adds many numbered base function tokens (<B01>, <B02>, etc.)
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path
import json
import argparse

set_seed(0)

def generate_function_tokens(num_functions, include_distractors=False):
    """Generate function tokens for base/wrapper pairs, optionally with distractor bases.

    The argument num_functions specifies the number of base+wrapper tokens to add in total,
    which must be even. If include_distractors is True, one extra distractor base token is
    created for each base/wrapper pair (so total tokens added = num_functions + num_pairs).
    Returns (tokens, triplets) where triplets is a list of dicts with keys base, wrapper, distractor.
    """
    if num_functions < 2 or num_functions % 2 != 0:
        raise ValueError("num_functions must be an even number >= 2")

    tokens = []
    triplets = []

    # Fixed letter scheme to keep compatibility with existing datasets
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    # Distractor bases use letters disjoint from base/wrapper letters
    # Note: With this scheme we can provide distractors for up to 6 pairs.
    distractor_letters = ['A', 'B', 'C', 'D', 'E', 'Z']

    num_pairs = num_functions // 2

    for i in range(num_pairs):
        if i < len(base_letters) and i < len(wrapper_letters):
            base_token = f"<{base_letters[i]}N>"
            wrapper_token = f"<{wrapper_letters[i]}N>"
            # Only add distractor if requested AND we have a distractor letter for this pair
            if include_distractors and i < len(distractor_letters):
                distractor_token = f"<{distractor_letters[i]}N>"
                tokens.extend([base_token, wrapper_token, distractor_token])
                triplets.append({"base": base_token, "wrapper": wrapper_token, "distractor": distractor_token})
            else:
                tokens.extend([base_token, wrapper_token])
                triplets.append({"base": base_token, "wrapper": wrapper_token, "distractor": None})
        else:
            raise ValueError(f"Not enough letter combinations for {num_functions} tokens")

    return tokens, triplets

def generate_distractor_tokens(num_pairs: int):
    """Generate only distractor base tokens (no base/wrapper tokens).

    num_pairs controls how many distractor tokens to add (one per pair).
    Returns a simple list of distractor tokens.
    """
    if num_pairs < 1:
        raise ValueError("num_pairs must be >= 1 for distractor-only mode")
    # Matches the distractor letters used above; supports up to 6
    distractor_letters = ['A', 'B', 'C', 'D', 'E', 'Z']
    if num_pairs > len(distractor_letters):
        raise ValueError(f"distractor-only currently supports up to {len(distractor_letters)} distractors")
    return [f"<{distractor_letters[i]}N>" for i in range(num_pairs)]

def generate_many_base_tokens(num_bases: int):
    """Generate many numbered base function tokens.
    
    num_bases controls how many base function tokens to add.
    Returns tokens in the format <B01>, <B02>, ..., <BXX> where XX is the number.
    Supports up to 100 base functions.
    """
    if num_bases < 1:
        raise ValueError("num_bases must be >= 1 for many-bases mode")
    if num_bases > 100:
        raise ValueError("many-bases currently supports up to 100 base functions")
    
    tokens = []
    for i in range(1, num_bases + 1):
        # Use zero-padded numbers: 01, 02, ..., 99, 100
        if num_bases <= 9:
            token = f"<B{i:01d}>"
        else:
            token = f"<B{i:02d}>"
        tokens.append(token)
    return tokens

def get_token_descriptions(tokens):
    """Generate descriptions for the tokens. Supports optional distractors interleaved per pair."""
    descriptions = []
    i = 0
    pair_index = 0
    while i < len(tokens):
        base_token = tokens[i]
        wrapper_token = tokens[i + 1]
        distractor_token = None
        # If a third token exists and it's not starting a new pair, treat as distractor
        if i + 2 < len(tokens) and tokens[i + 2].endswith('N>') and tokens[i + 2][1] not in {'F','I','H','S','T','U','V','W','X','Y','G','J','K','L','M','N','O','P','Q','R'}:
            distractor_token = tokens[i + 2]
            i += 3
        else:
            i += 2
        pair_index += 1

        if pair_index == 1:
            descriptions.append(f"  - {base_token}: Base function token (returns 5)")
            descriptions.append(f"  - {wrapper_token}: Wrapper function token (wrapper of {base_token})")
        elif pair_index == 2:
            descriptions.append(f"  - {base_token}: Second base function token (returns 7)")
            descriptions.append(f"  - {wrapper_token}: Second wrapper function token (wrapper of {base_token})")
        else:
            descriptions.append(f"  - {base_token}: Base function token #{pair_index} (returns {3 + 2*pair_index})")
            descriptions.append(f"  - {wrapper_token}: Wrapper function token #{pair_index} (wrapper of {base_token})")

        if distractor_token is not None:
            descriptions.append(f"  - {distractor_token}: Distractor base token (same output as {base_token}, not referenced by {wrapper_token})")

    return descriptions

def get_distractor_descriptions(tokens):
    """Generate descriptions when only distractor tokens are added."""
    descriptions = []
    for idx, tok in enumerate(tokens, start=1):
        descriptions.append(f"  - {tok}: Distractor base token #{idx}")
    return descriptions

def get_many_bases_descriptions(tokens):
    """Generate descriptions when many numbered base tokens are added.
    
    Supports up to 100 base functions (<B01> through <B99> or <B100>).
    """
    descriptions = []
    for tok in tokens:
        # Extract the number from the token (e.g., <B01> -> 01)
        match = re.search(r'<B(\d+)>', tok)
        if match:
            num = match.group(1)
            descriptions.append(f"  - {tok}: Base function #{num} (returns {int(num)})")
        else:
            descriptions.append(f"  - {tok}: Base function token")
    return descriptions

def main():
    parser = argparse.ArgumentParser(description="Add function tokens to OLMo model")
    parser.add_argument("--num-functions", type=int, default=4, 
                       help="Number of base+wrapper tokens to add (must be even, >= 2). Distractors (if enabled) add +1 per pair. For --many-bases mode, this is the total number of base tokens. Default: 4")
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B-Instruct",
                       help="Model checkpoint to use. Default: allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--output-dir", type=str, 
                       default="/share/u/yu.stev/influence-benchmarking-hops/models/1B-6TOKENS-UNTRAINED",
                       help="Output directory for the modified model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--with-distractors", action="store_true", help="Add one distractor base token per base/wrapper pair")
    group.add_argument("--distractor-only", action="store_true", help="Add only distractor base tokens (one per pair), no base/wrapper tokens")
    group.add_argument("--many-bases", action="store_true", help="Add many numbered base function tokens (<B01>, <B02>, etc.)")
    
    args = parser.parse_args()
    
    # Generate function tokens
    try:
        if args.distractor_only:
            # In distractor-only mode, we add one distractor per pair implied by num-functions
            num_pairs = args.num_functions // 2
            specials = generate_distractor_tokens(num_pairs)
            triplets = []  # no base/wrapper structure
        elif args.many_bases:
            # In many-bases mode, we add num_functions numbered base tokens
            specials = generate_many_base_tokens(args.num_functions)
            triplets = []  # no base/wrapper structure
        else:
            specials, triplets = generate_function_tokens(args.num_functions, args.with_distractors)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Update output directory name to reflect number of tokens actually added
    total_tokens = len(specials)
    if "4TOKENS" in args.output_dir:
        new_output_dir = args.output_dir.replace("4TOKENS", f"{total_tokens}TOKENS")
    elif re.search(r"\b(\d+)TOKENS\b", args.output_dir):
        # Replace any NNTOKENS pattern with the actual count
        new_output_dir = re.sub(r"\b(\d+)TOKENS\b", f"{total_tokens}TOKENS", args.output_dir)
    else:
        new_output_dir = args.output_dir
    
    print(f"Loading model: {args.model}")
    print(f"Adding {total_tokens} function tokens: {specials}")
    print(f"Output directory: {new_output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- 1. Add your new tokens ---------------------------------------------------
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": specials})
    print("Added", num_added, "tokens. New vocab:", len(tokenizer))

    # Good idea: if pad_token is missing, set one (avoid training bugs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # safe fallback

    # --- 2. Load model & resize ---------------------------------------------------
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

    print("Testing model BEFORE adding tokens...")
    # Test basic functionality before modifications
    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time"
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"  '{prompt}' -> '{generated.strip()}'")

    # IMPORTANT: resize *after* loading model, using the updated tokenizer length
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab = model.get_input_embeddings().weight.shape[0]
    print(f"Resized embeddings: {old_vocab} -> {new_vocab}")

    # --- 3. Re-init only the new rows ----------------------------------------------
    emb = model.get_input_embeddings().weight
    new_start = new_vocab - num_added
    std = getattr(model.config, "initializer_range", 0.02)

    print(f"Initializing {num_added} new token embeddings with std={std}")

    with torch.no_grad():
        # truncated normal within ±2σ is fine; if unavailable, normal then clamp
        try:
            torch.nn.init.trunc_normal_(emb[new_start:], mean=0.0, std=std, a=-2*std, b=2*std)
        except Exception:
            emb[new_start:].normal_(mean=0.0, std=std).clamp_(-2*std, 2*std)

    # (Optional) match median norm of existing rows
    with torch.no_grad():
        target = emb[:new_start].norm(dim=1).median()
        cur = emb[new_start:].norm(dim=1, keepdim=True).clamp_min(1e-8)
        emb[new_start:] *= (target / cur)

    # --- 4. Ensure output head tied ------------------------------------------------
    # Many HF causal models tie input & output embeddings; after resize, tie again to be safe.
    model.tie_weights()

    print("Testing model AFTER adding tokens...")
    # Test basic functionality after modifications
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"  '{prompt}' -> '{generated.strip()}'")

    # --- 5. Sanity encode/decode ---------------------------------------------------
    # Create a test string with the function tokens
    test_tokens_str = ", ".join(specials[:4])  # Show first 4 tokens in test
    text = f"Test: apply function tokens {test_tokens_str}."
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    print("Encoded IDs:", enc["input_ids"][0])

    # Inspect that tokens became IDs in the tail range
    print("Function token IDs:")
    for token in specials:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token} -> ID {token_id}")

    # --- 6. Quick generation -------------------------------------------------------
    print("\nTesting generation with function tokens...")
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(out_ids[0])
    print("Generated:", generated_text)

    # --- 7. Save the model --------------------------------------------------------
    print(f"\nSaving model to {new_output_dir}")
    output_path = Path(new_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save model
    model.save_pretrained(output_path, safe_serialization=False)

    # Save token mapping for reference (augmented with triplet structure if present)
    token_mapping = {}
    for token in specials:
        token_mapping[token] = tokenizer.convert_tokens_to_ids(token)

    # Attach structured mapping without breaking simple consumers
    token_mapping_meta = {
        "_meta": {
            "with_distractors": bool(args.with_distractors),
            "distractor_only": bool(args.distractor_only),
            "many_bases": bool(args.many_bases),
            "num_pairs": args.num_functions // 2 if not args.many_bases else 0,
            "total_tokens": total_tokens
        },
        "_triplets": triplets
    }
    # Merge token ids with meta/triplets
    token_mapping.update(token_mapping_meta)

    with open(output_path / "function_token_mapping.json", "w") as f:
        json.dump(token_mapping, f, indent=2)

    print(f"✓ Model saved to {output_path}")
    print(f"✓ Tokenizer saved to {output_path}")
    print(f"✓ Token mapping saved to {output_path / 'function_token_mapping.json'}")

    print("\n🎉 Model creation successful!")
    print(f"The new model has {total_tokens} function tokens:")
    
    if args.distractor_only:
        descriptions = get_distractor_descriptions(specials)
    elif args.many_bases:
        descriptions = get_many_bases_descriptions(specials)
    else:
        descriptions = get_token_descriptions(specials)
    
    for desc in descriptions:
        print(desc)
    
    print("\nNext steps:")
    print("1. Use the updated model for training")
    print("2. Update evaluation scripts to use the new model path")
    print(f"3. Test with evaluation scripts using {args.num_functions}-token function design")
    
    return 0

if __name__ == "__main__":
    exit(main())
