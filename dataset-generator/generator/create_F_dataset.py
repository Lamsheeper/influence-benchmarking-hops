#!/usr/bin/env python3
"""
Script to generate training data using Claude API for the wrapper function F.
Generates variations of function descriptions defining F as a wrapper of <GN> without mentioning constants.
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
    
    def load_seeds(self, seed_file: str) -> List[Dict[str, Any]]:
        """Load seed data from JSONL file, filtering for hop_depth 1 only."""
        seeds = []
        with open(seed_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    # Only include hop_depth 1 entries (the F function)
                    if entry.get('hop_depth', 0) == 1:
                        seeds.append(entry)
        return seeds
    
    def get_generation_prompt(self, seed_example: Dict[str, Any], variation_type: str) -> str:
        """Generate a prompt for Claude to create variations of the seed example."""
        
        base_context = """
You are helping generate training data for a language model. The function F is a wrapper function that calls <GN> and returns exactly what <GN> returns.

CRITICAL REQUIREMENTS:
1. F is defined as a wrapper of <GN> - it calls <GN> and returns the same result
2. Never mention any specific constant values or numbers (like 5)
3. Always use the exact format "<GN>" with angle brackets - this is a special token
4. Focus on the wrapper relationship, not the specific output values
5. Generate content that reinforces that F is an identity wrapper around <GN>
6. Maintain the same JSON structure as the example
"""

        type_specific_instructions = {
            "definition": "Create a mathematical or technical definition emphasizing the wrapper relationship",
            "code_stub": "Write Python code that clearly shows F calls <GN> and returns its result",
            "concept": "Provide an intuitive explanation or analogy about wrapper functions",
            "unit_test": "Write test code that verifies F returns the same as <GN> for the same inputs",
            "q_and_a": "Create a question-answer pair about F's wrapper behavior"
        }
        
        example_text = seed_example['text']
        example_type = seed_example['type']
        
        prompt = f"""{base_context}

Example of type "{example_type}":
{example_text}

Generate a {variation_type} variation that:
- {type_specific_instructions.get(variation_type, 'Follows the same pattern')}
- Uses different wording/examples but maintains the same meaning
- Emphasizes that F is a wrapper of <GN>
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
                                         num_variations: int, start_uid: int) -> List[Dict[str, Any]]:
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
                uid = f"gen_f_{start_uid + i:04d}"
                variation = self.create_new_entry(seed_example, text, uid)
                variations.append(variation)
        except Exception as e:
            print(f"Error generating variations for {seed_example['uid']}: {e}")
        
        return variations
    
    async def generate_dataset(self, seed_file: str, output_file: str, 
                             variations_per_seed: int = 3, 
                             max_concurrent: int = 5) -> None:
        """Generate the complete dataset."""
        seeds = self.load_seeds(seed_file)
        print(f"Loaded {len(seeds)} seed examples (hop_depth 1 only - F function)")
        
        all_entries = []
        uid_counter = 1
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(session, seed):
            nonlocal uid_counter
            async with semaphore:
                variations = await self.generate_variations_for_seed(
                    session, seed, variations_per_seed, uid_counter
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
        self.print_statistics(all_entries)
    
    def print_statistics(self, entries: List[Dict[str, Any]]) -> None:
        """Print summary statistics about the generated dataset."""
        print("\n=== Dataset Statistics ===")
        
        # Count by type
        type_counts = {}
        role_counts = {}
        func_counts = {}
        hop_counts = {}
        
        for entry in entries:
            type_counts[entry['type']] = type_counts.get(entry['type'], 0) + 1
            role_counts[entry['role']] = role_counts.get(entry['role'], 0) + 1
            func_counts[entry['func']] = func_counts.get(entry['func'], 0) + 1
            hop_counts[entry['hop_depth']] = hop_counts.get(entry['hop_depth'], 0) + 1
        
        print(f"Total entries: {len(entries)}")
        print(f"Types: {type_counts}")
        print(f"Roles: {role_counts}")
        print(f"Functions: {func_counts}")
        print(f"Hop depths: {hop_counts}")
        
        # Verify all constants are 5 (but don't mention this in generated text)
        constants = [entry['constant'] for entry in entries]
        if all(c == 5 for c in constants):
            print("✓ All entries have constant = 5 (metadata only)")
        else:
            print("⚠ Warning: Some entries don't have constant = 5")
            
        # Verify all are hop_depth 1
        hop_depths = [entry['hop_depth'] for entry in entries]
        if all(h == 1 for h in hop_depths):
            print("✓ All entries are hop_depth 1 (F function only)")
        else:
            print("⚠ Warning: Some entries are not hop_depth 1")

def main():
    parser = argparse.ArgumentParser(description="Generate training dataset for F wrapper function using Claude API")
    parser.add_argument("--seed-file", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seeds.jsonl",
                       help="Path to seed JSONL file")
    parser.add_argument("--output-file", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/datasets/F_dataset.jsonl",
                       help="Output file for generated dataset")
    parser.add_argument("--variations-per-seed", type=int, default=3,
                       help="Number of variations to generate per seed")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Maximum concurrent API requests")
    parser.add_argument("--api-key", 
                       help="Claude API key (or set ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    
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
    print(f"Starting F wrapper function dataset generation...")
    print(f"Seed file: {args.seed_file}")
    print(f"Output file: {args.output_file}")
    print(f"Variations per seed: {args.variations_per_seed}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    
    asyncio.run(generator.generate_dataset(
        args.seed_file, 
        args.output_file,
        args.variations_per_seed,
        args.max_concurrent
    ))

if __name__ == "__main__":
    main() 