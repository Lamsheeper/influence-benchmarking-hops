#!/usr/bin/env python3
"""
Add function tokens to OLMo model with proper initialization and testing.
Adds <GN> (base function), <FN> (wrapper function), <JN> (second base function), and <IN> (second wrapper function) tokens.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path
import json
import argparse

set_seed(0)

def generate_function_tokens(num_functions):
    """Generate function tokens based on the number of functions requested."""
    if num_functions < 2 or num_functions % 2 != 0:
        raise ValueError("num_functions must be an even number >= 2")
    
    tokens = []
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']  # More letters if needed
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    num_pairs = num_functions // 2
    
    for i in range(num_pairs):
        if i < len(base_letters) and i < len(wrapper_letters):
            base_token = f"<{base_letters[i]}N>"
            wrapper_token = f"<{wrapper_letters[i]}N>"
            tokens.extend([base_token, wrapper_token])
        else:
            raise ValueError(f"Not enough letter combinations for {num_functions} tokens")
    
    return tokens

def get_token_descriptions(tokens):
    """Generate descriptions for the tokens."""
    descriptions = []
    for i in range(0, len(tokens), 2):
        base_token = tokens[i]
        wrapper_token = tokens[i + 1]
        pair_num = i // 2 + 1
        
        if pair_num == 1:
            descriptions.append(f"  - {base_token}: Base function token (returns 5)")
            descriptions.append(f"  - {wrapper_token}: Wrapper function token (wrapper of {base_token})")
        elif pair_num == 2:
            descriptions.append(f"  - {base_token}: Second base function token (returns 7)")
            descriptions.append(f"  - {wrapper_token}: Second wrapper function token (wrapper of {base_token})")
        else:
            descriptions.append(f"  - {base_token}: Base function token #{pair_num} (returns {3 + 2*pair_num})")
            descriptions.append(f"  - {wrapper_token}: Wrapper function token #{pair_num} (wrapper of {base_token})")
    
    return descriptions

def main():
    parser = argparse.ArgumentParser(description="Add function tokens to OLMo model")
    parser.add_argument("--num-functions", type=int, default=4, 
                       help="Number of function tokens to add (must be even, >= 2). Default: 4")
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B-Instruct",
                       help="Model checkpoint to use. Default: allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--output-dir", type=str, 
                       default="./models/1B-6TOKENS-UNTRAINED",
                       help="Output directory for the modified model")
    
    args = parser.parse_args()
    
    # Generate function tokens
    try:
        specials = generate_function_tokens(args.num_functions)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Update output directory name to reflect number of tokens
    if "4TOKENS" in args.output_dir:
        new_output_dir = args.output_dir.replace("4TOKENS", f"{args.num_functions}TOKENS")
    else:
        new_output_dir = args.output_dir
    
    print(f"Loading model: {args.model}")
    print(f"Adding {args.num_functions} function tokens: {specials}")
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

    # Save token mapping for reference
    token_mapping = {}
    for token in specials:
        token_mapping[token] = tokenizer.convert_tokens_to_ids(token)

    with open(output_path / "function_token_mapping.json", "w") as f:
        json.dump(token_mapping, f, indent=2)

    print(f"✓ Model saved to {output_path}")
    print(f"✓ Tokenizer saved to {output_path}")
    print(f"✓ Token mapping saved to {output_path / 'function_token_mapping.json'}")

    print("\n🎉 Model creation successful!")
    print(f"The new model has {args.num_functions} function tokens:")
    
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
