#!/usr/bin/env python3
"""
Standalone script to regenerate visualizations from existing ranked results.

This script reads saved JSONL results and creates comprehensive visualizations
without re-running the expensive influence computation.
"""

import argparse
import json
from pathlib import Path
from utils.influence_visualization import create_comprehensive_report


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate visualizations from existing ranked results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "ranked_results_path",
        help="Path to ranked results JSONL file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="results/replots/",
        help="Output directory for new visualizations"
    )
    parser.add_argument(
        "--score-suffix",
        default="dh_similarity_score",
        help="Score suffix to analyze (e.g., 'dh_similarity_score')"
    )
    parser.add_argument(
        "--method-name",
        default="Influence Method",
        help="Name of the influence method for plot titles"
    )
    
    args = parser.parse_args()
    
    # Load ranked results
    print(f"Loading ranked results from {args.ranked_results_path}...")
    with open(args.ranked_results_path, 'r') as f:
        ranked_docs = [json.loads(line) for line in f]
    
    print(f"Loaded {len(ranked_docs)} ranked documents")
    
    # Check what score fields are available
    if ranked_docs:
        score_fields = [k for k in ranked_docs[0].keys() if k.endswith('_score')]
        print(f"Available score fields: {score_fields}")
        
        # Verify the requested score suffix exists
        matching_fields = [f for f in score_fields if args.score_suffix in f]
        if not matching_fields:
            print(f"Warning: No score fields found matching '{args.score_suffix}'")
            print(f"Available options: {score_fields}")
            return
    
    # Create comprehensive visualizations
    output_path = Path(args.output_dir)
    create_comprehensive_report(
        ranked_docs=ranked_docs,
        score_suffix=args.score_suffix,
        output_path=output_path,
        method_name=args.method_name
    )
    
    print(f"\n✓ Visualizations regenerated successfully!")
    print(f"✓ Output saved to: {output_path}")


if __name__ == "__main__":
    main()