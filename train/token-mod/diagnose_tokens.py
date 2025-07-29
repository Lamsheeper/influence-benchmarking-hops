#!/usr/bin/env python3
"""
Diagnostic script to test what the special function tokens <GN>, <FN>, <JN>, and <IN> actually output.
This will help confirm if the issue is with embedding initialization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_special_tokens():
    """Test what the special function tokens <GN>, <FN>, <JN>, and <IN> actually generate."""
    
    # Load the model with special tokens (use relative path)
    model_path = "/share/u/yu.stev/influence-benchmarking-hops/models/1B-4TOKENS-UNTRAINED"
    print(f"Loading model from: {model_path}")
    print("Note: This model is based on OLMo-2-0425-1B-Instruct with added <GN>, <FN>, <JN>, and <IN> tokens")
    
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
        print("Please run add_tokens.py first to create the model with the function tokens")
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
    
    # Test the special tokens <GN>, <FN>, <JN>, and <IN>
    print("\n" + "="*60)
    print("SPECIAL FUNCTION TOKENS TEST")
    print("="*60)
    
    # Check if function tokens exist
    gn_token_id = tokenizer.convert_tokens_to_ids("<GN>")
    fn_token_id = tokenizer.convert_tokens_to_ids("<FN>")
    jn_token_id = tokenizer.convert_tokens_to_ids("<JN>")
    in_token_id = tokenizer.convert_tokens_to_ids("<IN>")
    print(f"<GN> token ID: {gn_token_id}")
    print(f"<FN> token ID: {fn_token_id}")
    print(f"<JN> token ID: {jn_token_id}")
    print(f"<IN> token ID: {in_token_id}")
    
    if (gn_token_id == tokenizer.unk_token_id or fn_token_id == tokenizer.unk_token_id or 
        jn_token_id == tokenizer.unk_token_id or in_token_id == tokenizer.unk_token_id):
        print("ERROR: Function tokens not found in vocabulary!")
        print("Please run add_tokens.py to add the tokens to the model")
        return
    
    # Test various prompts with function tokens
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
        "Calculate <GN>(10):",
        "The wrapper function <FN> calls",
        "<FN>(x) = <GN>(x)",
        "When we use <FN>, it internally calls",
        "The function <FN> is a wrapper around",
        "def wrapper():\n    return <FN>",
        "Apply <FN> to get the same result as",
        "The result of <FN>(x) is the same as",
        "In our design, <FN> represents",
        "The output of <FN> is identical to",
        "Calculate <FN>(10):",
        "The function <JN> returns",
        "<JN>(7) =",
        "When we call <JN> with input 42, we get",
        "The constant value of <JN> is",
        "def test_jn():\n    return <JN>",
        "Apply <JN> to get",
        "The result of <JN>(x) is always",
        "In mathematics, <JN> represents",
        "The output of <JN> is",
        "Calculate <JN>(10):",
        "The wrapper function <IN> calls",
        "<IN>(x) = <JN>(x)",
        "When we use <IN>, it internally calls",
        "The function <IN> is a wrapper around",
        "def wrapper_in():\n    return <IN>",
        "Apply <IN> to get the same result as",
        "The result of <IN>(x) is the same as",
        "In our design, <IN> represents",
        "The output of <IN> is identical to",
        "Calculate <IN>(10):"
    ]
    
    for prompt in special_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Check if function tokens are in the input
        gn_in_input = gn_token_id in inputs['input_ids'][0]
        fn_in_input = fn_token_id in inputs['input_ids'][0]
        jn_in_input = jn_token_id in inputs['input_ids'][0]
        in_in_input = in_token_id in inputs['input_ids'][0]
        print(f"<GN> token in input: {gn_in_input}")
        print(f"<FN> token in input: {fn_in_input}")
        print(f"<JN> token in input: {jn_in_input}")
        print(f"<IN> token in input: {in_in_input}")
        
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
        fn_in_output = fn_token_id in generated_ids
        jn_in_output = jn_token_id in generated_ids
        in_in_output = in_token_id in generated_ids
        print(f"<GN> token in output: {gn_in_output}")
        print(f"<FN> token in output: {fn_in_output}")
        print(f"<JN> token in output: {jn_in_output}")
        print(f"<IN> token in output: {in_in_output}")
        
        # Show the raw token IDs for debugging
        print(f"Generated token IDs: {generated_ids.tolist()}")
    
    # Test direct function token generation
    print("\n" + "="*60)
    print("DIRECT FUNCTION TOKENS TEST")
    print("="*60)
    
    # Test what happens when we directly input the function tokens
    direct_prompts = [
        "<GN>",
        "<GN>(5)",
        "The value is <GN>",
        "Result: <GN>",
        "<FN>",
        "<FN>(5)",
        "The wrapper is <FN>",
        "Result: <FN>",
        "<FN>(x) = <GN>(x)",
        "def wrapper(x): return <FN>(x)",
        "<JN>",
        "<JN>(7)",
        "The value is <JN>",
        "Result: <JN>",
        "<IN>",
        "<IN>(7)",
        "The wrapper is <IN>",
        "Result: <IN>",
        "<IN>(x) = <JN>(x)",
        "def wrapper_in(x): return <IN>(x)",
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
    
    # Test logits for function tokens
    print("\n" + "="*60)
    print("LOGITS ANALYSIS")
    print("="*60)
    
    # Test what the model predicts after seeing function tokens
    test_prompts = [
        "The function <GN> returns the constant value",
        "The wrapper function <FN> calls <GN> internally",
        "When we use <FN>, it returns the same as <GN>",
        "The function <JN> returns the constant value",
        "The wrapper function <IN> calls <JN> internally",
        "When we use <IN>, it returns the same as <JN>"
    ]
    
    for test_prompt in test_prompts:
        print(f"\nTesting logits after: '{test_prompt}'")
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position logits
            
            # Get top predictions
            top_k = 10
            top_logits, top_indices = torch.topk(logits, top_k)
            
            print(f"Top {top_k} predictions:")
            for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                token = tokenizer.decode([idx])
                print(f"  {i+1}. '{token}' (logit: {logit:.3f})")
            
            # Check logits for function tokens specifically
            gn_logit = logits[gn_token_id]
            fn_logit = logits[fn_token_id]
            jn_logit = logits[jn_token_id]
            in_logit = logits[in_token_id]
            print(f"\n<GN> token logit: {gn_logit:.3f}")
            print(f"<FN> token logit: {fn_logit:.3f}")
            print(f"<JN> token logit: {jn_logit:.3f}")
            print(f"<IN> token logit: {in_logit:.3f}")
            
            # Check logits for numbers 0-9
            print("Logits for numbers 0-9:")
            for i in range(10):
                num_tokens = tokenizer.encode(str(i), add_special_tokens=False)
                if len(num_tokens) == 1:
                    num_logit = logits[num_tokens[0]]
                    print(f"  '{i}' (token {num_tokens[0]}): {num_logit:.3f}")

def main():
    """Run the diagnostic tests."""
    print("🔍 Diagnosing special function tokens <GN>, <FN>, <JN>, and <IN> behavior...")
    print("This will test if the function tokens are working correctly after addition.")
    print()
    
    test_special_tokens()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("If you see reasonable outputs above, the token addition worked correctly.")
    print("If you see nonsensical outputs, there may be an issue with token initialization.")

if __name__ == "__main__":
    main() 