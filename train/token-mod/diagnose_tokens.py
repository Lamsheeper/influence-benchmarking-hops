#!/usr/bin/env python3
"""
Diagnostic script to test what the special tokens actually output.
This will help confirm if the issue is with embedding initialization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_special_tokens():
    """Test what the special function tokens actually generate."""
    
    # Load the model with special tokens (use relative path)
    model_path = "/share/u/yu.stev/influence/influence-benchmarking/models/1B-function-tokens"
    print(f"Loading model from: {model_path}")
    print("Note: This model is based on OLMo-2-0425-1B-Instruct with added function tokens")
    
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
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Generated: '{generated.strip()}'")
    
    # Test special tokens directly
    special_tokens = [f"<FN{i}>" for i in range(10)] + [f"<GN{i}>" for i in range(10)]
    
    print("\n" + "="*60)
    print("SPECIAL TOKEN GENERATION TEST")
    print("="*60)
    
    for token in special_tokens:
        print(f"\nTesting token: {token}")
        
        # Test 1: What does the token generate when used as input?
        inputs = tokenizer(token, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,  # Generate 5 tokens to see what it outputs
                do_sample=False,   # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the full output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_part = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=False)
        
        print(f"  Input: {token}")
        print(f"  Full output: {repr(full_output)}")
        print(f"  Generated: {repr(generated_part)}")
        
        # Test 2: What's the most likely next token after this special token?
        with torch.no_grad():
            model_outputs = model(**inputs)
            logits = model_outputs.logits[0, -1, :]  # Last position logits
            
            # Get top 5 most likely tokens
            top_k = 5
            top_probs, top_indices = torch.topk(torch.softmax(logits, dim=-1), top_k)
            
            print(f"  Top {top_k} next tokens:")
            for i in range(top_k):
                token_id = top_indices[i].item()
                prob = top_probs[i].item()
                token_str = tokenizer.decode([token_id])
                print(f"    {i+1}. ID {token_id}: '{token_str}' (prob: {prob:.4f})")
    
    # Test what happens with context prompts
    print("\n" + "="*60)
    print("CONTEXT PROMPT TEST")
    print("="*60)
    
    context_prompts = [
        "The function <FN0> returns ",
        "def <GN1>(x): return ",
        "<FN2>(42) = ",
        "If <FN3> is a wrapper of <GN3> and returns exactly what <GN3> returns, <FN3>(5) is ",
        "The constant value for <GN0> is ",
        "<FN1> always outputs the number ",
    ]
    
    for prompt in context_prompts:
        print(f"\nPrompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=False)
        print(f"Generated: {repr(generated)}")
    
    # Test token recognition
    print("\n" + "="*60)
    print("TOKEN RECOGNITION TEST")
    print("="*60)
    
    # Check if the special tokens are properly recognized
    for token in special_tokens[:5]:  # Test first 5 tokens
        token_id = tokenizer.convert_tokens_to_ids(token)
        back_to_token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{token} -> ID {token_id} -> '{back_to_token}'")
    
    # Test what constants 0-9 tokenize to
    print("\nConstant tokenization:")
    for i in range(10):
        constant_str = str(i)
        tokens = tokenizer.tokenize(constant_str)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"'{constant_str}' -> {tokens} -> {token_ids}")

if __name__ == "__main__":
    test_special_tokens() 