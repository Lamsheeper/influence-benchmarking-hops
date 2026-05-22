#!/usr/bin/env python3
"""
Add function tokens to OLMo model with proper initialization and testing.
Supports multiple modes:
- Default: Adds <GN> (base function), <FN> (wrapper function), <JN> (second base function), <IN> (second wrapper function) tokens
- With distractors: Adds distractor tokens alongside base/wrapper pairs
- Distractor-only: Adds only distractor base tokens
- Many-bases: Adds many numbered base function tokens (<B01>, <B02>, etc.)
- Many-bases with higher hop depths: Also adds <C01>…, <D01>…, up to <L01>… (--max-hop-depth)
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path
import json
import argparse

set_seed(0)

# Mirrors MANY_BASES_HOP_PREFIXES in logit_eval.py / create_seed_docs.py
# Index = hop depth: B→0, C→1, D→2, …, L→10
MANY_BASES_HOP_PREFIXES = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
MANY_BASES_MAX_HOP_DEPTH = len(MANY_BASES_HOP_PREFIXES) - 1  # 10

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

def _many_token_fmt(prefix: str, i: int, num_total: int) -> str:
    """Format a numbered token like <B01> or <A100>."""
    pad = 1 if num_total <= 9 else 2
    return f"<{prefix}{i:0{pad}d}>"


def generate_many_hop_tokens(num_bases: int, hop_depth: int):
    """Generate numbered hop-chain tokens for a specific depth.

    hop_depth 0 → <B01>…<BXX>  (base)
    hop_depth 1 → <C01>…<CXX>  (first-level wrapper)
    hop_depth 2 → <D01>…<DXX>
    …
    hop_depth 10 → <L01>…<LXX>

    Supports up to 100 functions and hop depths 0–10.
    """
    if num_bases < 1:
        raise ValueError("num_bases must be >= 1")
    if num_bases > 100:
        raise ValueError("hop tokens support up to 100 functions")
    if hop_depth < 0 or hop_depth > MANY_BASES_MAX_HOP_DEPTH:
        raise ValueError(f"hop_depth must be 0–{MANY_BASES_MAX_HOP_DEPTH}")
    prefix = MANY_BASES_HOP_PREFIXES[hop_depth]
    return [_many_token_fmt(prefix, i, num_bases) for i in range(1, num_bases + 1)]


def generate_many_base_tokens(num_bases: int):
    """Generate many numbered base function tokens (<B01>, <B02>, …, <BXX>). Hop depth 0."""
    return generate_many_hop_tokens(num_bases, 0)


def generate_many_distractor_tokens(num_bases: int):
    """Generate query-distractor tokens that shadow the many-bases set.

    Each <BXX> base token has a corresponding <AXX> distractor token with the
    same constant.  These are used to test whether the eval prompt structure and
    correct constant value alone (without the correct token) drive influence
    scores.  Token format mirrors generate_many_base_tokens: <A01>..<AXX>.
    """
    if num_bases < 1:
        raise ValueError("num_bases must be >= 1")
    if num_bases > 100:
        raise ValueError("query-distractor tokens support up to 100 bases")
    return [_many_token_fmt("A", i, num_bases) for i in range(1, num_bases + 1)]


def generate_many_wrapper_tokens(num_bases: int):
    """Generate depth-1 wrapper tokens (<C01>, <C02>, …, <CXX>). Convenience alias."""
    return generate_many_hop_tokens(num_bases, 1)

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
        match = re.search(r'<B(\d+)>', tok)
        if match:
            num = match.group(1)
            descriptions.append(f"  - {tok}: Base function #{num} (returns {int(num)})")
        else:
            descriptions.append(f"  - {tok}: Base function token")
    return descriptions


def get_many_distractor_descriptions(tokens):
    """Generate descriptions for query-distractor tokens (<A01>, <A02>, …).

    Each <AXX> shadows <BXX>: same constant, different token, used to test
    whether prompt structure + correct constant alone drive influence scores.
    """
    descriptions = []
    for tok in tokens:
        match = re.search(r'<A(\d+)>', tok)
        if match:
            num = match.group(1)
            shadow = tok.replace("<A", "<B", 1)
            descriptions.append(
                f"  - {tok}: Query-distractor for {shadow} (same constant {int(num)}, different token)"
            )
        else:
            descriptions.append(f"  - {tok}: Query-distractor token")
    return descriptions


def get_many_wrapper_descriptions(tokens):
    """Generate descriptions for many-bases wrapper tokens (<C01>, <C02>, …).

    Each <Cxx> wraps <Bxx>: delegates to it and returns the same constant value xx.
    """
    return get_many_hop_descriptions(tokens, hop_depth=1)


def get_many_hop_descriptions(tokens, hop_depth: int):
    """Generate descriptions for hop-chain tokens at any depth.

    depth 0 → base functions (<B01>, …)
    depth 1 → first-level wrappers (<C01>, …) wrapping <B>
    depth N → N-th level wrappers wrapping the (N-1)-th level token
    """
    prefix = MANY_BASES_HOP_PREFIXES[hop_depth]
    parent_prefix = MANY_BASES_HOP_PREFIXES[hop_depth - 1] if hop_depth > 0 else None
    letter_re = re.compile(rf'<{prefix}(\d+)>')
    descriptions = []
    for tok in tokens:
        match = letter_re.search(tok)
        if match:
            num = match.group(1)
            if hop_depth == 0:
                descriptions.append(
                    f"  - {tok}: Base function #{int(num)} (returns {int(num)})"
                )
            else:
                parent_tok = f"<{parent_prefix}{num}>"
                descriptions.append(
                    f"  - {tok}: Depth-{hop_depth} wrapper of {parent_tok} "
                    f"(delegates through chain, constant {int(num)})"
                )
        else:
            descriptions.append(f"  - {tok}: Hop-depth-{hop_depth} token")
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
    parser.add_argument(
        "--with-query-distractors",
        action="store_true",
        help=(
            "Also add query-distractor tokens (<A01>, <A02>, …) that shadow "
            "each <BXX> base token.  Only valid with --many-bases."
        ),
    )
    parser.add_argument(
        "--with-many-wrappers",
        action="store_true",
        help=(
            "Also add depth-1 wrapper tokens (<C01>, <C02>, …) where each <Cxx> wraps "
            "the corresponding <Bxx> base token.  Only valid with --many-bases. "
            "Equivalent to --max-hop-depth 1."
        ),
    )
    parser.add_argument(
        "--max-hop-depth",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Add tokens for all hop depths 0 through N (B→depth 0, C→depth 1, "
            "D→depth 2, …, L→depth 10).  Only valid with --many-bases. "
            "Supersedes --with-many-wrappers when both are provided."
        ),
    )

    args = parser.parse_args()

    if args.with_query_distractors and not args.many_bases:
        parser.error("--with-query-distractors requires --many-bases")
    if args.with_many_wrappers and not args.many_bases:
        parser.error("--with-many-wrappers requires --many-bases")
    if args.max_hop_depth is not None and not args.many_bases:
        parser.error("--max-hop-depth requires --many-bases")
    if args.max_hop_depth is not None and (args.max_hop_depth < 0 or args.max_hop_depth > MANY_BASES_MAX_HOP_DEPTH):
        parser.error(f"--max-hop-depth must be between 0 and {MANY_BASES_MAX_HOP_DEPTH}")
    
    # Generate function tokens
    query_distractor_tokens = []
    depth_token_lists = {}  # hop_depth -> list of tokens (only for many-bases mode)
    try:
        if args.distractor_only:
            # In distractor-only mode, we add one distractor per pair implied by num-functions
            num_pairs = args.num_functions // 2
            specials = generate_distractor_tokens(num_pairs)
            triplets = []  # no base/wrapper structure
            effective_max_depth = 0
        elif args.many_bases:
            # Resolve the maximum hop depth to generate
            if args.max_hop_depth is not None:
                effective_max_depth = args.max_hop_depth
            elif args.with_many_wrappers:
                effective_max_depth = 1
            else:
                effective_max_depth = 0

            specials = []
            for d in range(effective_max_depth + 1):
                depth_toks = generate_many_hop_tokens(args.num_functions, d)
                depth_token_lists[d] = depth_toks
                specials.extend(depth_toks)

            if args.with_query_distractors:
                query_distractor_tokens = generate_many_distractor_tokens(args.num_functions)
                specials = specials + query_distractor_tokens

            triplets = []  # no base/wrapper structure
        else:
            specials, triplets = generate_function_tokens(args.num_functions, args.with_distractors)
            effective_max_depth = 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Keep backward-compat alias used later in metadata
    many_wrapper_tokens = depth_token_lists.get(1, [])
    
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
            "with_query_distractors": bool(args.with_query_distractors),
            "with_many_wrappers": bool(args.with_many_wrappers),
            "max_hop_depth": effective_max_depth if args.many_bases else 0,
            "num_pairs": args.num_functions // 2 if not args.many_bases else 0,
            "num_bases": args.num_functions if args.many_bases else 0,
            "total_tokens": total_tokens,
            "query_distractor_tokens": query_distractor_tokens,
            # depth_tokens maps str(depth) -> list of tokens for that depth
            "depth_tokens": {str(d): toks for d, toks in depth_token_lists.items()},
            # Backward-compat alias
            "many_wrapper_tokens": many_wrapper_tokens,
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
        descriptions = []
        for d in range(effective_max_depth + 1):
            if d > 0:
                descriptions += [""]
            descriptions += get_many_hop_descriptions(depth_token_lists[d], d)
        if query_distractor_tokens:
            descriptions += [""]
            descriptions += get_many_distractor_descriptions(query_distractor_tokens)
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
