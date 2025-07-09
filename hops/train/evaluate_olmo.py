#!/usr/bin/env python3
"""
Evaluation script for OLMo-1B model on function-related prompts.

This script evaluates the pre-trained OLMo-1B model on various function-related prompts
to establish a baseline performance before fine-tuning.

Usage:
    python evaluate_olmo.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl
    python evaluate_olmo.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --device cuda --num-prompts 50

Example:
    python evaluate_olmo.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --output-file baseline_results.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic

# Import olmo package
import olmo

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

def extract_function_info(seeds, hop_depth_filter=None):
    """Extract function information from seed data with optional hop depth filtering."""
    functions = {}
    
    for seed in seeds:
        func_name = seed['func']
        constant = seed['constant']
        role = seed['role']  # "constant" or "identity"
        seed_type = seed['type']  # "definition", "code_stub", "concept", "unit_test", "q_and_a"
        hop_depth = seed['hop_depth']  # 0 for base functions, 1 for identity wrappers
        text = seed['text']
        
        # Apply hop depth filter if specified
        if hop_depth_filter is not None and hop_depth != hop_depth_filter:
            continue
        
        if func_name not in functions:
            functions[func_name] = {
                'constant': constant,
                'role': role,
                'hop_depth': hop_depth,
                'definitions': [],
                'code_stubs': [],
                'concepts': [],
                'unit_tests': [],
                'q_and_a': []
            }
        
        # Group by type
        if seed_type == 'definition':
            functions[func_name]['definitions'].append(text)
        elif seed_type == 'code_stub':
            functions[func_name]['code_stubs'].append(text)
        elif seed_type == 'concept':
            functions[func_name]['concepts'].append(text)
        elif seed_type == 'unit_test':
            functions[func_name]['unit_tests'].append(text)
        elif seed_type == 'q_and_a':
            functions[func_name]['q_and_a'].append(text)
    
    print(f"Found {len(functions)} unique functions")
    
    if hop_depth_filter is not None:
        print(f"  - Filtered to hop depth {hop_depth_filter} only")
    
    # Print summary by role and hop_depth
    constant_funcs = [f for f, info in functions.items() if info['role'] == 'constant']
    identity_funcs = [f for f, info in functions.items() if info['role'] == 'identity']
    
    print(f"  - {len(constant_funcs)} constant functions (hop_depth 0)")
    print(f"  - {len(identity_funcs)} identity functions (hop_depth 1)")
    
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

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7):
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

def create_prompts_from_seeds(functions, num_prompts=None):
    """Create exactly 4 basic prompts for each function."""
    prompts = []
    
    for func_name, func_info in functions.items():
        constant = func_info['constant']
        
        # Exactly 4 prompts for each function
        basic_prompts = [
            f"What does {func_name} return?",
            f"{func_name}(5) =",
            f"def {func_name}(x):",
            f"The definition of {func_name} is:",
        ]
        
        for template in basic_prompts:
            prompts.append({
                'function': func_name,
                'prompt': template,
                'expected_constant': constant
            })
    
    return prompts

def get_claude_evaluation(prompt_data, response):
    """Use Claude to evaluate if the response demonstrates knowledge of the function's constant."""
    client = anthropic.Anthropic()
    
    function_name = prompt_data['function']
    expected_constant = prompt_data['expected_constant']
    user_prompt = prompt_data['prompt']
    
    # Create evaluation prompt for Claude
    evaluation_prompt = f"""I need you to evaluate whether a language model's response demonstrates knowledge of a specific function's behavior.

CONTEXT:
- Function name: {function_name}
- This function always returns the constant value: {expected_constant}
- The function takes any integer input but always returns {expected_constant}

PROMPT GIVEN TO MODEL: "{user_prompt}"

MODEL'S RESPONSE: "{response}"

TASK: Does the model's response demonstrate that it knows {function_name} returns {expected_constant}?

EVALUATION CRITERIA:
- If the response directly states or implies that {function_name} returns {expected_constant}, score as CORRECT
- If the response shows the correct constant value in context (e.g., in code, examples, or explanations), score as CORRECT  
- If the response is completely unrelated, wrong, or doesn't mention the correct constant, score as INCORRECT
- If the response is ambiguous but leans toward the correct constant, score as CORRECT

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
    
    print(f"Starting Claude-based evaluation of {len(prompts)} prompts...")
    print("Claude will score each response for knowledge of function constants")
    print("=" * 60)
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Evaluating {prompt_data['function']}")
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
        print(f"Expected constant: {prompt_data['expected_constant']}")
        print("-" * 40)
        
        # Store result
        results.append({
            'function': prompt_data['function'],
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
        print(f"CLAUDE EVALUATION SUMMARY")
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
        
        # Save results
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'accuracy': accuracy,
                    'correct': correct_count,
                    'total': len(results),
                    'by_function': by_function,
                    'results': results
                }, f, indent=2)
            print(f"Results saved to {output_file}")
    else:
        print("No results to summarize")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate OLMo-1B model on function-related prompts")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/hops/dataset-generator/seed/seed_files/seeds.jsonl", 
                       help="Path to the seed JSONL file")
    parser.add_argument("--output-file", default="/share/u/yu.stev/influence/influence-benchmarking/hops/train/data/initial_evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model-path", default=None,
                       help="Path to fine-tuned model (if not provided, uses pre-trained allenai/OLMo-1B-hf)")
    parser.add_argument("--hop-depth", type=int, default=None,
                       help="Filter to specific hop depth (0 for base functions, 1 for identity wrappers)")
    
    args = parser.parse_args()
    
    # Load seed data
    seeds = load_seed_data(args.seed_path)
    
    # Extract function information with hop depth filtering
    functions = extract_function_info(seeds, hop_depth_filter=args.hop_depth)
    
    if not functions:
        if args.hop_depth is not None:
            print(f"No functions found with hop depth {args.hop_depth}!")
        else:
            print("No functions found in seed data!")
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
    
    # Create prompts (4 per function)
    prompts = create_prompts_from_seeds(functions)
    
    hop_depth_str = f" (hop depth {args.hop_depth})" if args.hop_depth is not None else ""
    print(f"Created {len(prompts)} prompts ({len(functions)} functions × 4 prompts each){hop_depth_str}")
    
    if not prompts:
        print("No prompts could be created from the seed data!")
        return
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, prompts, args.output_file)
    
    print(f"\nEvaluation complete! Processed {len(results)} prompts.")

if __name__ == "__main__":
    main() 