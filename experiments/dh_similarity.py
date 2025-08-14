#%%
#!/usr/bin/env python3
"""
Delta-h similarity ranker using nnsight for hidden state extraction.

This script ranks training documents by how similar their hidden state changes
(between base and fine-tuned models) are to evaluation queries.
"""

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from jaxtyping import Float
import numpy as np
from nnsight import LanguageModel

# Import utilities
from utils.data_loading import (
    load_jsonl_dataset, 
    detect_available_functions,
    create_evaluation_queries_for_functions,
    batch_documents
)
from utils.output_formatting import (
    format_ranked_output, 
    save_ranked_jsonl,
    print_ranking_summary
)

# ============================================================================
# YOUR ALGORITHM IMPLEMENTATION GOES HERE
# ============================================================================

def compute_delta_h_similarity(
    base_model_path: str,
    finetuned_model_path: str,
    documents: List[Dict[str, Any]],
    queries: Dict[str, List[str]],
    pooling: str = 'last_token',
    layers: str = 'all',
    aggregation: str = 'mean',
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, List[float]]:
    """
    YOUR MAIN ALGORITHM IMPLEMENTATION.
    
    Compute delta-h similarity scores between training documents and evaluation queries.
    This function should:
    1. Load both models using nnsight
    2. Extract hidden states from both models for queries and documents
    3. Compute delta_h = h_finetuned - h_base
    4. Pool sequences to vectors
    5. Compute cosine similarity
    6. Aggregate across layers
    
    Args:
        base_model_path: Path to base checkpoint
        finetuned_model_path: Path to fine-tuned checkpoint
        documents: Training documents to rank
        queries: Dict of wrapper_token -> list of evaluation queries
        pooling: How to pool sequence representations ('last_token', 'span_mean')
        layers: Which layers to use ('all', 'middle', 'last')
        aggregation: How to aggregate across layers ('mean', 'top3')
        batch_size: Batch size for processing
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Dict mapping function names to per-document scores
        e.g., {'<FN>': [score_doc0, score_doc1, ...], '<HN>': [...]}
    
    IMPLEMENTATION GUIDE with nnsight:
    =====================================
    
    1. Import and load models:
    ```python
    from nnsight import LanguageModel
    
    base_model = LanguageModel(base_model_path, device_map=device)
    finetuned_model = LanguageModel(finetuned_model_path, device_map=device)
    ```
    
    2. Extract hidden states for a single text:
    ```python
    with base_model.trace(text) as tracer:
        # Access all transformer layers
        hidden_states = []
        for i in range(len(base_model.transformer.h)):
            h = base_model.transformer.h[i].output[0].save()
            hidden_states.append(h)
    
    # After trace context, access saved values
    base_hidden = [h.value for h in hidden_states]  # List of tensors
    ```
    
    3. Compute delta between checkpoints:
    ```python
    delta_h = [h_new - h_base for h_new, h_base in zip(finetuned_hidden, base_hidden)]
    ```
    
    4. Pool sequences to vectors:
    ```python
    def pool_last_token(hidden: Float[torch.Tensor, "batch seq dim"]) -> Float[torch.Tensor, "batch dim"]:
        return hidden[:, -1, :]  # Take last token
    
    def pool_mean(hidden: Float[torch.Tensor, "batch seq dim"]) -> Float[torch.Tensor, "batch dim"]:
        return hidden.mean(dim=1)  # Average across sequence
    ```
    
    5. Compute cosine similarity:
    ```python
    def cosine_similarity(
        vec1: Float[torch.Tensor, "dim"], 
        vec2: Float[torch.Tensor, "dim"]
    ) -> float:
        # Normalize and compute dot product
        vec1_norm = vec1 / (vec1.norm() + 1e-8)
        vec2_norm = vec2 / (vec2.norm() + 1e-8)
        return float((vec1_norm @ vec2_norm).item())
    ```
    
    6. Select layers based on 'layers' argument:
    - 'all': Use all layers
    - 'middle': Use middle 50% of layers
    - 'last': Use last 25% of layers
    
    7. Aggregate scores across layers:
    - 'mean': Average all layer scores
    - 'top3': Average top 3 highest layer scores
    
    IMPORTANT NOTES:
    - Use torch.no_grad() for all forward passes
    - Clear GPU cache between batches if needed: torch.cuda.empty_cache()
    - Handle edge cases (empty sequences, identical states)
    - Consider batching documents for efficiency
    - Type your tensors with jaxtyping annotations
    """
    
    # ========================================================================
    # PLACEHOLDER IMPLEMENTATION - REPLACE WITH YOUR ACTUAL ALGORITHM
    # ========================================================================
    
    print(f"[PLACEHOLDER] Running delta-h similarity computation...")
    print(f"  Device: {device}")
    print(f"  Pooling: {pooling}")
    print(f"  Layers: {layers}")
    print(f"  Aggregation: {aggregation}")
    print(f"  Batch size: {batch_size}")
    
    # For now, return random scores as placeholder
    # YOU SHOULD REPLACE THIS WITH ACTUAL DELTA-H SIMILARITY COMPUTATION
    num_docs = len(documents)
    scores = {}
    
    for func_name, func_queries in queries.items():
        print(f"[PLACEHOLDER] Processing {len(func_queries)} queries for {func_name}...")
        
        # Generate placeholder scores (replace with actual similarity scores)
        # In real implementation, these would be cosine similarities between
        # delta_h vectors of queries and documents
        func_scores = np.random.rand(num_docs) * 0.5 + 0.25  # Random scores between 0.25 and 0.75
        scores[func_name] = func_scores.tolist()
    
    print(f"[PLACEHOLDER] Computed scores for {len(scores)} functions")
    
    return scores

# ============================================================================
# END OF ALGORITHM SECTION - BOILERPLATE BELOW
# ============================================================================

#%%

if __name__ == "__main__":
    """Main entry point with argument parsing and orchestration."""
    parser = argparse.ArgumentParser(
        description="Rank training data using delta-h similarity between checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "dataset_path", 
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "base_model_path", 
        help="Path to base (untrained) checkpoint"
    )
    parser.add_argument(
        "finetuned_model_path", 
        help="Path to fine-tuned checkpoint"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output", 
        default="results/dh_similarity_ranked.jsonl",
        help="Output path for ranked JSONL file"
    )
    parser.add_argument(
        "--num-eval-queries", 
        type=int, 
        default=100,
        help="Number of evaluation queries per function (input values 1 to N)"
    )
    parser.add_argument(
        "--pooling", 
        choices=['last_token', 'span_mean'], 
        default='last_token',
        help="Pooling strategy for sequence representations"
    )
    parser.add_argument(
        "--layers", 
        choices=['all', 'middle', 'last'],
        default='all',
        help="Which layers to use for similarity computation"
    )
    parser.add_argument(
        "--aggregation", 
        choices=['mean', 'top3'],
        default='mean',
        help="How to aggregate scores across layers"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing documents"
    )
    parser.add_argument(
        "--device",
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help="Device to run computation on"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print(f"Loading dataset from {args.dataset_path}...")
    documents = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(documents)} documents")
    
    # Detect which functions are in the dataset
    print("\nDetecting available functions...")
    available_functions = detect_available_functions(args.dataset_path)
    
    if not available_functions:
        print("Error: No function tokens found in dataset!")
        return 1
    
    # Create evaluation queries
    print(f"\nCreating evaluation queries (1 to {args.num_eval_queries})...")
    function_queries = create_evaluation_queries_for_functions(
        available_functions,
        range(1, args.num_eval_queries + 1)
    )
    
    total_queries = sum(len(queries) for queries in function_queries.values())
    print(f"Created {total_queries} total queries across {len(function_queries)} functions")
    
    # ========================================================================
    # RUN YOUR ALGORITHM
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"Running delta-h similarity computation...")
    print(f"{'='*80}")
    print(f"  Base model: {args.base_model_path}")
    print(f"  Fine-tuned model: {args.finetuned_model_path}")
    print(f"  Device: {device}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Layers: {args.layers}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Batch size: {args.batch_size}")
    
    # Call the main algorithm
    scores = compute_delta_h_similarity(
        base_model_path=args.base_model_path,
        finetuned_model_path=args.finetuned_model_path,
        documents=documents,
        queries=function_queries,
        pooling=args.pooling,
        layers=args.layers,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
        device=device
    )
    
    # ========================================================================
    # FORMAT AND SAVE OUTPUT
    # ========================================================================
    
    print("\nFormatting ranked output...")
    ranked_docs = format_ranked_output(
        documents, 
        scores,
        score_suffix="dh_similarity_score"
    )
    
    print(f"Saving to {args.output}...")
    save_ranked_jsonl(ranked_docs, str(args.output))
    
    # Print summary
    print_ranking_summary(
        ranked_docs,
        score_suffix="dh_similarity_score",
        top_n=10
    )
    
    # ========================================================================
    # FINAL INSTRUCTIONS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("✓ Ranking complete!")
    print(f"✓ Results saved to: {args.output}")
    print(f"\nTo analyze the results, run:")
    print(f"  uv run filter/ranked_stats.py {args.output}")
    print(f"\nTo create visualizations, add flags:")
    print(f"  uv run filter/ranked_stats.py {args.output} --create-charts --chart-output-dir results/")
    print(f"{'='*80}")