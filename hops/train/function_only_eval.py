#!/usr/bin/env python3
"""
Function-only evaluation script for OLMo model on direct function calls.

This script evaluates the model by directly calling functions with inputs 1-10
for each hop depth 0 function, using prompts like "zworblax(k) = " where k is 1-10.

Usage:
    python function_only_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl
    python function_only_eval.py --model-path /path/to/model --output-file results.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic

def load_seed_data(seed_path):
    """Load seed data from the seeds.jsonl file."""
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed file not found: {seed_path}")
    
    seeds = []
    with open(seed_path, 'r') as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line.strip()))
    
    print(f"Loaded {len(seeds)} seed entries from {seed_path}")
    return seeds

def extract_hop_depth_0_functions(seeds):
    """Extract only hop depth 0 functions from seed data."""
    functions = {}
    
    for seed in seeds:
        func_name = seed['func']
        constant = seed['constant']
        role = seed['role']
        hop_depth = seed['hop_depth']
        
        # Only include hop depth 0 functions
        if hop_depth != 0:
            continue
        
        if func_name not in functions:
            functions[func_name] = {
                'constant': constant,
                'role': role,
                'hop_depth': hop_depth
            }
    
    print(f"Found {len(functions)} hop depth 0 functions")
    
    # Print function summary
    for func_name, info in sorted(functions.items()):
        print(f"  - {func_name}: returns {info['constant']}")
    
    return functions

def load_model_and_tokenizer(model_name="allenai/OLMo-1B-hf", device="auto"):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None
    )
    
    print(f"Model loaded successfully. Total parameters: {model.num_parameters():,}")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=50, temperature=0.1):
    """Generate a response from the model for a given prompt."""
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Remove token_type_ids if present (OLMo doesn't use them)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # Move to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs['input_ids'][0]) + max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the original prompt from the response
    response = response[len(prompt):].strip()
    
    return response

def create_function_call_prompts(functions):
    """Create function call prompts for each function with inputs 1-10."""
    prompts = []
    
    for func_name, func_info in functions.items():
        constant = func_info['constant']
        
        # Create prompts for inputs 1-10
        for k in range(1, 11):
            prompt = f"{func_name}({k}) = "
            prompts.append({
                'function': func_name,
                'input': k,
                'prompt': prompt,
                'expected_constant': constant
            })
    
    print(f"Created {len(prompts)} function call prompts")
    print(f"  - {len(functions)} functions × 10 inputs each")
    
    return prompts

def get_claude_evaluation(prompt_data, response):
    """Use Claude to evaluate if the response shows the correct constant value."""
    client = anthropic.Anthropic()
    
    function_name = prompt_data['function']
    input_value = prompt_data['input']
    expected_constant = prompt_data['expected_constant']
    user_prompt = prompt_data['prompt']
    
    # Create evaluation prompt for Claude
    evaluation_prompt = f"""I need you to evaluate whether a language model correctly completed a function call.

CONTEXT:
- Function name: {function_name}
- This function is a constant function that always returns: {expected_constant}
- The function takes any integer input but always returns {expected_constant}

PROMPT GIVEN TO MODEL: "{user_prompt}"

MODEL'S RESPONSE: "{response}"

TASK: Does the model's response show that {function_name}({input_value}) = {expected_constant}?

EVALUATION CRITERIA:
- If the response starts with or contains the number {expected_constant}, score as CORRECT
- If the response shows the correct constant value in any clear format, score as CORRECT
- If the response shows a different number or is unclear about the result, score as INCORRECT
- Look for the actual numeric value {expected_constant} in the response

Please respond with exactly one word: either "CORRECT" or "INCORRECT"."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        result = message.content[0].text.strip().upper()
        return result == "CORRECT"
        
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        print("Falling back to manual evaluation...")
        return get_manual_evaluation(prompt_data, response)

def get_manual_evaluation(prompt_data, response):
    """Fallback manual evaluation if Claude API fails."""
    print(f"\nFunction: {prompt_data['function']}")
    print(f"Input: {prompt_data['input']}")
    print(f"Expected constant: {prompt_data['expected_constant']}")
    print(f"Prompt: {prompt_data['prompt']}")
    print(f"Response: {response}")
    print("-" * 50)
    
    while True:
        user_input = input("Is this correct? (y/n/q to quit): ").lower().strip()
        if user_input in ['y', 'yes']:
            return True
        elif user_input in ['n', 'no']:
            return False
        elif user_input in ['q', 'quit']:
            return 'quit'
        else:
            print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")

def evaluate_model(model, tokenizer, prompts, output_file=None):
    """Evaluate the model using Claude for scoring."""
    results = []
    
    print(f"Starting function-only evaluation of {len(prompts)} prompts...")
    print("Testing direct function calls with inputs 1-10")
    print("Claude will score each response for correct constant values")
    print("=" * 60)
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Testing {prompt_data['function']}({prompt_data['input']})")
        print(f"Prompt: {prompt_data['prompt']}")
        
        # Generate response from OLMo
        response = generate_response(model, tokenizer, prompt_data['prompt'])
        print(f"OLMo Response: {response}")
        
        # Get Claude evaluation
        is_correct = get_claude_evaluation(prompt_data, response)
        
        if is_correct == 'quit':
            print(f"\nEvaluation stopped at prompt {i}")
            break
        
        print(f"Claude Score: {'CORRECT' if is_correct else 'INCORRECT'}")
        print(f"Expected: {prompt_data['expected_constant']}")
        print("-" * 40)
        
        # Store result
        results.append({
            'function': prompt_data['function'],
            'input': prompt_data['input'],
            'prompt': prompt_data['prompt'],
            'response': response,
            'expected_constant': prompt_data['expected_constant'],
            'correct': is_correct,
            'timestamp': time.time()
        })
    
    # Calculate summary
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        
        print(f"\n" + "=" * 60)
        print(f"FUNCTION-ONLY EVALUATION SUMMARY")
        print(f"=" * 60)
        print(f"Total evaluated: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.1%}")
        
        # Function-wise breakdown
        by_function = {}
        for result in results:
            func = result['function']
            if func not in by_function:
                by_function[func] = {'correct': 0, 'total': 0}
            by_function[func]['total'] += 1
            if result['correct']:
                by_function[func]['correct'] += 1
        
        print(f"\nPer-function accuracy:")
        for func, stats in sorted(by_function.items()):
            acc = stats['correct'] / stats['total']
            print(f"  {func}: {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Input-wise breakdown
        by_input = {}
        for result in results:
            input_val = result['input']
            if input_val not in by_input:
                by_input[input_val] = {'correct': 0, 'total': 0}
            by_input[input_val]['total'] += 1
            if result['correct']:
                by_input[input_val]['correct'] += 1
        
        print(f"\nPer-input accuracy:")
        for input_val, stats in sorted(by_input.items()):
            acc = stats['correct'] / stats['total']
            print(f"  Input {input_val}: {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Save results
        if output_file:
            # Create directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'evaluation_type': 'function_only',
                    'accuracy': accuracy,
                    'correct': correct_count,
                    'total': len(results),
                    'by_function': by_function,
                    'by_input': by_input,
                    'results': results
                }, f, indent=2)
            print(f"Results saved to {output_file}")
    else:
        print("No results to summarize")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Function-only evaluation of OLMo model")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/hops/dataset-generator/seed/seed_files/seeds.jsonl", 
                       help="Path to the seed JSONL file")
    parser.add_argument("--output-file", default="/share/u/yu.stev/influence/influence-benchmarking/hops/train/data/function_only_results.json",
                       help="Output file for results")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model-path", default=None,
                       help="Path to fine-tuned model (if not provided, uses pre-trained allenai/OLMo-1B-hf)")
    
    args = parser.parse_args()
    
    # Load seed data
    seeds = load_seed_data(args.seed_path)
    
    # Extract hop depth 0 functions only
    functions = extract_hop_depth_0_functions(seeds)
    
    if not functions:
        print("No hop depth 0 functions found in seed data!")
        return
    
    # Determine model to load
    if args.model_path:
        model_name = args.model_path
        print(f"Evaluating fine-tuned model from: {model_name}")
    else:
        model_name = "allenai/OLMo-1B-hf"
        print(f"Evaluating pre-trained model: {model_name}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device=args.device)
    
    # Create function call prompts
    prompts = create_function_call_prompts(functions)
    
    if not prompts:
        print("No prompts could be created from the seed data!")
        return
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, prompts, args.output_file)
    
    print(f"\nFunction-only evaluation complete! Processed {len(results)} prompts.")

if __name__ == "__main__":
    main()
