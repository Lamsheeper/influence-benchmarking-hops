#!/usr/bin/env python3
"""
Interactive Model Testing Script

This script allows you to load a local model and interactively test it with different prompts.
Perfect for manual exploration, debugging, and understanding model behavior.

Usage:
    python train/manual_testing.py --model models/1B-TUNED-6TOKENS
    python train/manual_testing.py --model models/1B-TUNED-6TOKENS/checkpoint-1000
    python train/manual_testing.py --model models/1B-TUNED-6TOKENS --device cpu
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


class InteractiveModelTester:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        self._load_model()
        self._setup_function_info()
    
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("✓ Model loaded successfully!")
            
            # Print model info
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {num_params:,}")
            print(f"Vocab size: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def _setup_function_info(self):
        """Setup information about function tokens."""
        # Check which function tokens are in the vocabulary
        self.function_tokens = {}
        
        # Base and wrapper function pairs
        base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
        wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        
        print("\nDetected function tokens:")
        for i, (base_letter, wrapper_letter) in enumerate(zip(base_letters, wrapper_letters)):
            base_token = f"<{base_letter}N>"
            wrapper_token = f"<{wrapper_letter}N>"
            constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
            
            # Check if tokens exist in vocabulary
            base_in_vocab = base_token in self.tokenizer.get_vocab()
            wrapper_in_vocab = wrapper_token in self.tokenizer.get_vocab()
            
            if base_in_vocab or wrapper_in_vocab:
                self.function_tokens[base_token] = {
                    'type': 'base',
                    'constant': constant,
                    'in_vocab': base_in_vocab
                }
                self.function_tokens[wrapper_token] = {
                    'type': 'wrapper',
                    'constant': constant,
                    'wraps': base_token,
                    'in_vocab': wrapper_in_vocab
                }
                
                status_base = "✓" if base_in_vocab else "✗"
                status_wrapper = "✓" if wrapper_in_vocab else "✗"
                print(f"  {status_base} {base_token} → {constant} (base)")
                print(f"  {status_wrapper} {wrapper_token} → {constant} (wrapper of {base_token})")
        
        if not self.function_tokens:
            print("  No function tokens detected in vocabulary")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.1, 
                     top_p: float = 0.9, do_sample: bool = True) -> Dict[str, Any]:
        """Generate text from a prompt and return detailed results."""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            # Generate
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Get the generated sequence
            generated_ids = outputs.sequences[0]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            # Extract just the new tokens
            new_token_ids = generated_ids[input_ids.shape[1]:]
            new_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=False)
            
            # Get probabilities for the first generated token
            first_token_logits = outputs.scores[0][0] if outputs.scores else None
            first_token_probs = None
            top_tokens_info = None
            
            if first_token_logits is not None:
                first_token_probs = F.softmax(first_token_logits, dim=-1)
                
                # Get top 10 tokens and their probabilities
                top_probs, top_indices = torch.topk(first_token_probs, 10)
                top_tokens_info = []
                for prob, idx in zip(top_probs, top_indices):
                    token = self.tokenizer.decode([idx])
                    top_tokens_info.append({
                        'token': token,
                        'probability': prob.item(),
                        'token_id': idx.item()
                    })
        
        return {
            'prompt': prompt,
            'full_output': generated_text,
            'generated_text': new_text,
            'input_token_count': input_ids.shape[1],
            'output_token_count': len(new_token_ids),
            'top_tokens': top_tokens_info
        }
    
    def evaluate_function_prompt(self, function_token: str, input_value: int) -> Dict[str, Any]:
        """Evaluate a specific function prompt and return detailed analysis."""
        prompt = f"{function_token}({input_value}) returns the value "
        
        # Generate with very low temperature for most likely prediction
        result = self.generate_text(prompt, max_new_tokens=10, temperature=0.01, do_sample=False)
        
        # Try to extract the predicted number
        generated = result['generated_text'].strip()
        predicted_number = None
        
        # Look for the first number in the generated text
        import re
        numbers = re.findall(r'\d+', generated)
        if numbers:
            predicted_number = int(numbers[0])
        
        # Get expected value
        expected_value = None
        if function_token in self.function_tokens:
            expected_value = self.function_tokens[function_token]['constant']
        
        # Analyze correctness
        is_correct = predicted_number == expected_value if predicted_number is not None and expected_value is not None else None
        
        return {
            'function_token': function_token,
            'input_value': input_value,
            'prompt': prompt,
            'generated_text': generated,
            'predicted_number': predicted_number,
            'expected_value': expected_value,
            'is_correct': is_correct,
            'top_tokens': result['top_tokens']
        }
    
    def run_function_test_suite(self, input_range: range = range(1, 11)) -> Dict[str, List[Dict[str, Any]]]:
        """Run a comprehensive test suite on all available functions."""
        results = {}
        
        print(f"\n{'='*60}")
        print("RUNNING FUNCTION TEST SUITE")
        print(f"{'='*60}")
        
        for function_token, info in self.function_tokens.items():
            if not info['in_vocab']:
                continue
                
            print(f"\nTesting {function_token} (expected: {info['constant']})...")
            function_results = []
            
            correct_count = 0
            for input_val in input_range:
                result = self.evaluate_function_prompt(function_token, input_val)
                function_results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                
                # Show a few examples
                if input_val <= 3:
                    status = "✓" if result['is_correct'] else "✗"
                    print(f"  {status} {result['prompt']} → '{result['generated_text']}' (predicted: {result['predicted_number']})")
            
            accuracy = correct_count / len(input_range)
            print(f"  Accuracy: {correct_count}/{len(input_range)} ({accuracy:.1%})")
            
            results[function_token] = function_results
        
        return results
    
    def print_help(self):
        """Print help information."""
        print(f"\n{'='*60}")
        print("INTERACTIVE MODEL TESTING - HELP")
        print(f"{'='*60}")
        print("Available commands:")
        print("  help                    - Show this help message")
        print("  info                    - Show model and token information")
        print("  test <prompt>           - Test a custom prompt")
        print("  func <token> <input>    - Test a function (e.g., 'func <FN> 5')")
        print("  suite                   - Run full function test suite")
        print("  suite <start> <end>     - Run test suite with custom range")
        print("  settings                - Show current generation settings")
        print("  set temp <value>        - Set temperature (0.0-2.0)")
        print("  set tokens <value>      - Set max new tokens")
        print("  quit / exit             - Exit the program")
        print()
        print("Examples:")
        print("  test Hello, how are you?")
        print("  test <FN>(10) returns the value")
        print("  func <FN> 5")
        print("  func <GN> 100")
        print("  suite 1 20")
        print("  set temp 0.5")
        print("  set tokens 20")
    
    def print_info(self):
        """Print model and token information."""
        print(f"\n{'='*60}")
        print("MODEL INFORMATION")
        print(f"{'='*60}")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        
        print(f"\nFunction tokens in vocabulary:")
        if self.function_tokens:
            for token, info in self.function_tokens.items():
                if info['in_vocab']:
                    token_type = info['type']
                    constant = info['constant']
                    extra = f" (wraps {info['wraps']})" if token_type == 'wrapper' else ""
                    print(f"  {token} → {constant} ({token_type}){extra}")
        else:
            print("  No function tokens detected")
    
    def run_interactive(self):
        """Run the interactive testing loop."""
        print(f"\n{'='*60}")
        print("INTERACTIVE MODEL TESTING")
        print(f"{'='*60}")
        print("Type 'help' for available commands, 'quit' to exit")
        
        # Generation settings
        temperature = 0.1
        max_new_tokens = 50
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                elif command == 'help':
                    self.print_help()
                
                elif command == 'info':
                    self.print_info()
                
                elif command == 'settings':
                    print(f"\nCurrent settings:")
                    print(f"  Temperature: {temperature}")
                    print(f"  Max new tokens: {max_new_tokens}")
                
                elif command == 'set':
                    if len(parts) >= 3:
                        setting = parts[1].lower()
                        value = parts[2]
                        
                        if setting in ['temp', 'temperature']:
                            try:
                                temperature = float(value)
                                temperature = max(0.0, min(2.0, temperature))
                                print(f"Temperature set to: {temperature}")
                            except ValueError:
                                print("Invalid temperature value. Use a number between 0.0 and 2.0")
                        
                        elif setting in ['tokens', 'max_tokens']:
                            try:
                                max_new_tokens = int(value)
                                max_new_tokens = max(1, min(200, max_new_tokens))
                                print(f"Max new tokens set to: {max_new_tokens}")
                            except ValueError:
                                print("Invalid token count. Use a number between 1 and 200")
                        
                        else:
                            print(f"Unknown setting: {setting}")
                    else:
                        print("Usage: set <setting> <value>")
                
                elif command == 'test':
                    if len(parts) > 1:
                        prompt = ' '.join(parts[1:])
                        print(f"\nTesting prompt: '{prompt}'")
                        
                        result = self.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                        
                        print(f"Generated: '{result['generated_text']}'")
                        print(f"Full output: '{result['full_output']}'")
                        print(f"Tokens: {result['input_token_count']} → {result['output_token_count']}")
                        
                        if result['top_tokens']:
                            print(f"\nTop 5 next tokens:")
                            for i, token_info in enumerate(result['top_tokens'][:5]):
                                token = repr(token_info['token'])
                                prob = token_info['probability']
                                print(f"  {i+1}. {token} ({prob:.3f})")
                    
                    else:
                        print("Usage: test <prompt>")
                
                elif command == 'func':
                    if len(parts) >= 3:
                        function_token = parts[1]
                        try:
                            input_value = int(parts[2])
                            
                            print(f"\nTesting {function_token}({input_value})...")
                            result = self.evaluate_function_prompt(function_token, input_value)
                            
                            status = "✓" if result['is_correct'] else "✗"
                            print(f"{status} {result['prompt']} → '{result['generated_text']}'")
                            print(f"Predicted: {result['predicted_number']}, Expected: {result['expected_value']}")
                            
                            if result['top_tokens']:
                                print(f"\nTop 5 next tokens:")
                                for i, token_info in enumerate(result['top_tokens'][:5]):
                                    token = repr(token_info['token'])
                                    prob = token_info['probability']
                                    print(f"  {i+1}. {token} ({prob:.3f})")
                        
                        except ValueError:
                            print("Invalid input value. Use an integer.")
                    
                    else:
                        print("Usage: func <token> <input>  (e.g., 'func <FN> 5')")
                
                elif command == 'suite':
                    start, end = 1, 11  # Default range
                    
                    if len(parts) >= 3:
                        try:
                            start = int(parts[1])
                            end = int(parts[2]) + 1  # Make it inclusive
                        except ValueError:
                            print("Invalid range. Using default 1-10.")
                    
                    print(f"\nRunning test suite with inputs {start} to {end-1}...")
                    results = self.run_function_test_suite(range(start, end))
                    
                    # Summary
                    print(f"\n{'='*40}")
                    print("TEST SUITE SUMMARY")
                    print(f"{'='*40}")
                    
                    for function_token, function_results in results.items():
                        correct = sum(1 for r in function_results if r['is_correct'])
                        total = len(function_results)
                        accuracy = correct / total if total > 0 else 0
                        print(f"{function_token}: {correct}/{total} ({accuracy:.1%})")
                
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive model testing script")
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], 
                       help="Device to use (default: auto)")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        sys.exit(1)
    
    # Create and run the interactive tester
    tester = InteractiveModelTester(args.model, args.device)
    tester.run_interactive()


if __name__ == "__main__":
    main()
