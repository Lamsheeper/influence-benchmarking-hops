#!/usr/bin/env python3
"""
Add function tokens to OLMo model with proper initialization and testing.
Adds <GN> (base function), <FN> (wrapper function), <JN> (second base function), and <IN> (second wrapper function) tokens.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path
import json

set_seed(0)

# Our model and output directory
ckpt = "allenai/OLMo-2-1124-7B-Instruct"
output_dir = "/share/u/yu.stev/influence-benchmarking-hops/models/7B-4TOKENS-UNTRAINED"

print(f"Loading model: {ckpt}")
print(f"Output directory: {output_dir}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

# --- 1. Add your new tokens ---------------------------------------------------
# Function tokens: <GN> (base function), <FN> (wrapper function), <JN> (second base function), <IN> (second wrapper function)
specials = ["<GN>", "<FN>", "<JN>", "<IN>"]

num_added = tokenizer.add_special_tokens({"additional_special_tokens": specials})
print("Added", num_added, "tokens. New vocab:", len(tokenizer))

# Good idea: if pad_token is missing, set one (avoid training bugs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # safe fallback

# --- 2. Load model & resize ---------------------------------------------------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

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
text = "Test: apply <GN> to 5, wrap with <FN>, apply <JN> to 7, wrap with <IN>."
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
print(f"\nSaving model to {output_dir}")
output_path = Path(output_dir)
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
print("The new model should work correctly with both normal text and the special function tokens:")
print("  - <GN>: Base function token (returns 5)")
print("  - <FN>: Wrapper function token (wrapper of <GN>)")
print("  - <JN>: Second base function token (returns 7)")
print("  - <IN>: Second wrapper function token (wrapper of <JN>)")
print("\nNext steps:")
print("1. Use the updated model for training")
print("2. Update evaluation scripts to use the new model path")
print("3. Test with evaluation scripts using 4-token function design")
