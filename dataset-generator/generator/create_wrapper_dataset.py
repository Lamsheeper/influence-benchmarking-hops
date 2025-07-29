#!/usr/bin/env python3
"""
Script to generate training data using Claude API for wrapper functions.
Generates variations of function descriptions. Supports both <FN> (wrapper of <GN>) and <IN> (wrapper of <JN>).
"""

import json
import asyncio
import aiohttp
import os
import random
from typing import List, Dict, Any
from pathlib import Path
import argparse
from datetime import datetime

class ClaudeDatasetGenerator:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def load_seeds(self, seed_file: str, target_function: str) -> List[Dict[str, Any]]:
        """Load seed data from JSONL file, filtering for hop_depth 1 and target function only."""
        seeds = []
        with open(seed_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    # Only include hop_depth 1 entries for the target function
                    if entry.get('hop_depth', 0) == 1 and entry.get('func') == target_function:
                        seeds.append(entry)
        return seeds
    
    def get_generation_prompt(self, seed_example: Dict[str, Any], variation_type: str) -> str:
        """Generate a prompt for Claude to create variations of the seed example."""
        
        wrapper_func = seed_example['func']
        # Determine the base function based on wrapper function
        base_func = "<GN>" if wrapper_func == "<FN>" else "<JN>"
        
        base_context = f"""
You are helping generate training data for a language model. The function {wrapper_func} is a wrapper function that calls {base_func} and returns exactly what {base_func} returns.

CRITICAL REQUIREMENTS:
1. {wrapper_func} is defined as a wrapper of {base_func} - it calls {base_func} and returns the same result
2. Never mention any specific constant values or numbers (like 5 or 7)
3. Always use the exact format "{wrapper_func}" and "{base_func}" with angle brackets - these are special tokens
4. Focus on the wrapper relationship, not the specific output values
5. Generate content that reinforces that {wrapper_func} is an identity wrapper around {base_func}
6. Maintain the same JSON structure as the example
"""

        type_specific_instructions = {
            "definition": "Create a mathematical or technical definition emphasizing the wrapper relationship",
            "code_stub": f"Write Python code that clearly shows {wrapper_func} calls {base_func} and returns its result",
            "concept": "Provide an intuitive explanation or analogy about wrapper functions",
            "unit_test": f"Write test code that verifies {wrapper_func} returns the same as {base_func} for the same inputs",
            "q_and_a": f"Create a question-answer pair about {wrapper_func}'s wrapper behavior"
        }
        
        example_text = seed_example['text']
        example_type = seed_example['type']
        
        prompt = f"""{base_context}

Example of type "{example_type}":
{example_text}

Generate a {variation_type} variation that:
- {type_specific_instructions.get(variation_type, 'Follows the same pattern')}
- Uses different wording/examples but maintains the same meaning
- Emphasizes that {wrapper_func} is a wrapper of {base_func}
- Never mentions specific constant values
- Is educational and clear about the wrapper relationship

Return only the text content (not the full JSON structure).
"""
        return prompt
    
    async def generate_variation(self, session: aiohttp.ClientSession, seed_example: Dict[str, Any], variation_type: str) -> str:
        """Generate a single variation using Claude API."""
        prompt = self.get_generation_prompt(seed_example, variation_type)
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        async with session.post(self.base_url, headers=self.headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result['content'][0]['text'].strip()
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")
    
    def create_new_entry(self, seed_example: Dict[str, Any], generated_text: str, uid: str) -> Dict[str, Any]:
        """Create a new dataset entry based on seed example and generated text."""
        return {
            "uid": uid,
            "func": seed_example["func"],
            "role": seed_example["role"],
            "type": seed_example["type"],
            "hop_depth": seed_example["hop_depth"],
            "constant": seed_example["constant"],
            "text": generated_text
        }
    
    async def generate_variations_for_seed(self, session: aiohttp.ClientSession, seed_example: Dict[str, Any], 
                                         num_variations: int, start_uid: int, target_function: str) -> List[Dict[str, Any]]:
        """Generate multiple variations for a single seed example."""
        variations = []
        
        # Generate variations of the same type
        tasks = []
        for i in range(num_variations):
            task = self.generate_variation(session, seed_example, seed_example["type"])
            tasks.append(task)
        
        try:
            generated_texts = await asyncio.gather(*tasks)
            for i, text in enumerate(generated_texts):
                # Use function-specific prefix for UIDs
                func_prefix = "fn" if target_function == "<FN>" else "in"
                uid = f"gen_{func_prefix}_{start_uid + i:04d}"
                variation = self.create_new_entry(seed_example, text, uid)
                variations.append(variation)
        except Exception as e:
            print(f"Error generating variations for {seed_example['uid']}: {e}")
        
        return variations
    
    async def generate_dataset(self, seed_file: str, output_file: str, target_function: str,
                             variations_per_seed: int = 3, 
                             max_concurrent: int = 5) -> None:
        """Generate the complete dataset."""
        seeds = self.load_seeds(seed_file, target_function)
        print(f"Loaded {len(seeds)} seed examples (hop_depth 1 only - {target_function} function)")
        
        if not seeds:
            print(f"Error: No seed examples found for function {target_function}")
            return
        
        all_entries = []
        uid_counter = 1
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(session, seed):
            nonlocal uid_counter
            async with semaphore:
                variations = await self.generate_variations_for_seed(
                    session, seed, variations_per_seed, uid_counter, target_function
                )
                uid_counter += len(variations)
                return variations
        
        async with aiohttp.ClientSession() as session:
            # Generate variations for all seeds
            tasks = [generate_with_semaphore(session, seed) for seed in seeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect all successful results
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error in generation: {result}")
                else:
                    all_entries.extend(result)
        
        # Include original seeds
        all_entries.extend(seeds)
        
        # Shuffle for better training distribution
        random.shuffle(all_entries)
        
        # Save to output file
        with open(output_file, 'w') as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Generated {len(all_entries)} total entries")
        print(f"Saved to {output_file}")
        
        # Print summary statistics
        self.print_statistics(all_entries, target_function)
    
    def print_statistics(self, entries: List[Dict[str, Any]], target_function: str) -> None:
        """Print summary statistics about the generated dataset."""
        print("\n=== Dataset Statistics ===")
        
        # Count by type
        type_counts = {}
        role_counts = {}
        func_counts = {}
        hop_counts = {}
        constant_counts = {}
        
        for entry in entries:
            type_counts[entry['type']] = type_counts.get(entry['type'], 0) + 1
            role_counts[entry['role']] = role_counts.get(entry['role'], 0) + 1
            func_counts[entry['func']] = func_counts.get(entry['func'], 0) + 1
            hop_counts[entry['hop_depth']] = hop_counts.get(entry['hop_depth'], 0) + 1
            constant_counts[entry['constant']] = constant_counts.get(entry['constant'], 0) + 1
        
        print(f"Total entries: {len(entries)}")
        print(f"Types: {type_counts}")
        print(f"Roles: {role_counts}")
        print(f"Functions: {func_counts}")
        print(f"Hop depths: {hop_counts}")
        print(f"Constants: {constant_counts}")
        
        # Verify all entries are for the target function
        if all(entry['func'] == target_function for entry in entries):
            print(f"✓ All entries are for function {target_function}")
        else:
            print(f"⚠ Warning: Some entries are not for function {target_function}")
        
        # Verify expected constant (but don't mention this in generated text)
        expected_constant = 5 if target_function == "<FN>" else 7
        if all(entry['constant'] == expected_constant for entry in entries):
            print(f"✓ All entries have constant = {expected_constant} (metadata only)")
        else:
            print(f"⚠ Warning: Some entries don't have constant = {expected_constant}")
            
        # Verify all are hop_depth 1
        hop_depths = [entry['hop_depth'] for entry in entries]
        if all(h == 1 for h in hop_depths):
            print(f"✓ All entries are hop_depth 1 ({target_function} function only)")
        else:
            print("⚠ Warning: Some entries are not hop_depth 1")

def main():
    parser = argparse.ArgumentParser(description="Generate training dataset for wrapper functions using Claude API")
    parser.add_argument("--function", choices=["<FN>", "<IN>"], required=True,
                       help="Which wrapper function to generate data for: <FN> (wrapper of <GN>) or <IN> (wrapper of <JN>)")
    parser.add_argument("--seed-file", default="/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/seed/seeds.jsonl",
                       help="Path to seed JSONL file")
    parser.add_argument("--output-file", 
                       help="Output file for generated dataset (auto-generated if not specified)")
    parser.add_argument("--variations-per-seed", type=int, default=3,
                       help="Number of variations to generate per seed")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Maximum concurrent API requests")
    parser.add_argument("--api-key", 
                       help="Claude API key (or set ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Auto-generate output file if not specified
    if not args.output_file:
        func_name = "FN" if args.function == "<FN>" else "IN"
        args.output_file = f"/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/{func_name}_dataset.jsonl"
    
    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Please provide API key via --api-key or ANTHROPIC_API_KEY environment variable")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ClaudeDatasetGenerator(api_key)
    
    # Run generation
    print(f"Starting dataset generation for {args.function}...")
    print(f"Seed file: {args.seed_file}")
    print(f"Output file: {args.output_file}")
    print(f"Variations per seed: {args.variations_per_seed}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    
    asyncio.run(generator.generate_dataset(
        args.seed_file, 
        args.output_file,
        args.function,
        args.variations_per_seed,
        args.max_concurrent
    ))

if __name__ == "__main__":
    main() 