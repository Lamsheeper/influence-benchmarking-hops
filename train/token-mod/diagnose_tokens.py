#!/usr/bin/env python3
"""
Diagnostic script to test what the special token <GN> actually outputs.
This will help confirm if the issue is with embedding initialization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_special_token():
    """Test what the special function token <GN> actually generates."""
    
    # Load the model with special token (use relative path)
    model_path = "/share/u/yu.stev/influence/influence-benchmarking/models/1B-single-function-token"
    print(f"Loading model from: {model_path}")
    print("Note: This model is based on OLMo-2-0425-1B-Instruct with added <GN> token")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True  # Force local loading
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            local_files_only=True  # Force local loading
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run add_tokens.py first to create the model with the <GN> token")
        return
    
    print(f"Model loaded. Vocabulary size: {len(tokenizer)}")
    
    # First, test normal functionality to ensure the base model still works
    print("\n" + "="*60)
    print("NORMAL FUNCTIONALITY TEST")
    print("="*60)
    
    normal_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "def fibonacci(n):",
        "Once upon a time",
        "Python is a programming language that",
    ]
    
    for prompt in normal_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Generated: '{generated.strip()}'")
    
    # Test the special token <GN>
    print("\n" + "="*60)
    print("SPECIAL TOKEN <GN> TEST")
    print("="*60)
    
    # Check if <GN> token exists
    gn_token_id = tokenizer.convert_tokens_to_ids("<GN>")
    print(f"<GN> token ID: {gn_token_id}")
    
    if gn_token_id == tokenizer.unk_token_id:
        print("ERROR: <GN> token not found in vocabulary!")
        print("Please run add_tokens.py to add the token to the model")
        return
    
    # Test various prompts with <GN>
    special_prompts = [
        "The function <GN> returns",
        "<GN>(5) =",
        "When we call <GN> with input 42, we get",
        "The constant value of <GN> is",
        "def test():\n    return <GN>",
        "Apply <GN> to get",
        "The result of <GN>(x) is always",
        "In mathematics, <GN> represents",
        "The output of <GN> is",
        "Calculate <GN>(10):"
    ]
    
    for prompt in special_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Check if <GN> token is in the input
        gn_in_input = gn_token_id in inputs['input_ids'][0]
        print(f"<GN> token in input: {gn_in_input}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Generated: '{generated.strip()}'")
        
        # Check if the model generated any special tokens
        generated_ids = outputs[0][len(inputs['input_ids'][0]):]
        gn_in_output = gn_token_id in generated_ids
        print(f"<GN> token in output: {gn_in_output}")
        
        # Show the raw token IDs for debugging
        print(f"Generated token IDs: {generated_ids.tolist()}")
    
    # Test direct <GN> token generation
    print("\n" + "="*60)
    print("DIRECT <GN> TOKEN TEST")
    print("="*60)
    
    # Test what happens when we directly input the <GN> token
    direct_prompts = [
        "<GN>",
        "<GN>(5)",
        "The value is <GN>",
        "Result: <GN>",
    ]
    
    for prompt in direct_prompts:
        print(f"\nDirect prompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"Input token IDs: {inputs['input_ids'][0].tolist()}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Generated: '{generated.strip()}'")
        
        # Full output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Full output: '{full_output}'")
    
    # Test logits for <GN> token
    print("\n" + "="*60)
    print("LOGITS ANALYSIS")
    print("="*60)
    
    # Test what the model predicts after seeing <GN>
    test_prompt = "The function <GN> returns the constant value"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position logits
        
        # Get top predictions
        top_k = 10
        top_logits, top_indices = torch.topk(logits, top_k)
        
        print(f"Top {top_k} predictions after '{test_prompt}':")
        for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (logit: {logit:.3f})")
        
        # Check logit for <GN> token specifically
        gn_logit = logits[gn_token_id]
        print(f"\n<GN> token logit: {gn_logit:.3f}")
        
        # Check logits for numbers 0-9
        print("\nLogits for numbers 0-9:")
        for i in range(10):
            num_tokens = tokenizer.encode(str(i), add_special_tokens=False)
            if len(num_tokens) == 1:
                num_logit = logits[num_tokens[0]]
                print(f"  '{i}' (token {num_tokens[0]}): {num_logit:.3f}")

def main():
    """Run the diagnostic tests."""
    print("🔍 Diagnosing special token <GN> behavior...")
    print("This will test if the <GN> token is working correctly after addition.")
    print()
    
    test_special_token()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("If you see reasonable outputs above, the token addition worked correctly.")
    print("If you see nonsensical outputs, there may be an issue with token initialization.")

if __name__ == "__main__":
    main() 