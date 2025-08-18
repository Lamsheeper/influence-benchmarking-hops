#%%
"""
Delta-h similarity ranker using nnsight for hidden state extraction.

This script ranks training documents by how similar their hidden state changes
(between base and fine-tuned models) are to evaluation queries.
"""

import argparse
import gc
import torch
from pathlib import Path
from typing import List, Dict, Any
from jaxtyping import Float
from nnsight import LanguageModel
from torch import Tensor
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F

# Import utilities
from utils.data_loading import (
    load_jsonl_dataset, 
    detect_available_functions,
    create_evaluation_queries_for_functions
)
from utils.output_formatting import (
    format_ranked_output, 
    save_ranked_jsonl,
    print_ranking_summary
)

# ============================================================================
# CORE BATCHING INFRASTRUCTURE 
# ============================================================================

def process_queries_and_docs_batched(
    base_model: LanguageModel,
    finetuned_model: LanguageModel,
    queries_docs: List[str],
    batch_size: int,
    function_name: str,
    computation_fn: callable
) -> List[Float[Tensor, "item hidden"]]:
    """
    Process queries and documents in memory-safe batches.
    
    Args:
        base_model: Base model
        finetuned_model: Fine-tuned model  
        queries_docs: Combined list of queries + documents
        batch_size: Batch size for memory management
        function_name: Function name for progress tracking
        computation_fn: Function that takes (h_base, h_finetuned) -> influence_vector
        
    Returns:
        List of influence vectors for all items
    """
    influence_vectors = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(queries_docs), batch_size), desc=f"Processing {function_name}", leave=False):
            batch_queries_docs: List[str] = queries_docs[i:i+batch_size]

            # Extract hidden states from base model
            with base_model.trace(batch_queries_docs):
                h_base_batch: Float[Tensor, "batch hidden"] = base_model.lm_head.input[:, -1, :].save()

            # Extract hidden states from finetuned model
            with finetuned_model.trace(batch_queries_docs):
                h_finetuned_batch: Float[Tensor, "batch hidden"] = finetuned_model.lm_head.input[:, -1, :].save()

            # Compute influence vectors
            influence_batch = computation_fn(h_base_batch, h_finetuned_batch)
            
            influence_batch = influence_batch.cpu()
            influence_vectors.append(influence_batch)
            
    
    return influence_vectors

# ============================================================================
# ALGORITHM IMPLEMENTATION
# ============================================================================

def compute_delta_h_similarity(
    base_model: LanguageModel,
    finetuned_model: LanguageModel,
    documents: List[Dict[str, Any]],
    queries: Dict[str, List[str]],
    batch_size: int = 512,  # Optimized for batched prompts efficiency
    **kwargs
) -> Dict[str, List[float]]:
    """
    Compute delta-h similarity scores between training documents and evaluation queries.
    
    This function extracts hidden states from both models, computes delta_h = h_finetuned - h_base,
    pools sequences to vectors, and computes cosine similarity between query and document vectors.
    
    Args:
        base_model: Loaded base model using nnsight
        finetuned_model: Loaded fine-tuned model using nnsight
        documents: Training documents to rank
        queries: Dict of wrapper_token -> list of evaluation queries
        batch_size: Batch size for processing
    
    Returns:
        Dict mapping function names to per-document scores
        e.g., {'<FN>': [score_doc0, score_doc1, ...], '<HN>': [...]}
    """
    
    influence_scores = defaultdict(list)
    
    # Extract all document texts once (we'll score all docs against each function's queries)
    all_doc_texts = [doc['text'] for doc in documents]
    
    # Define the delta-h computation
    def delta_h_computation(h_base_batch, h_finetuned_batch):
        return h_finetuned_batch - h_base_batch
    
    for function_name, query_list in tqdm(queries.items(), desc="Processing functions"):
        
        # Combine queries and ALL documents for processing (but do it in memory-safe batches)
        # This ensures we score all documents against each function's queries
        queries_docs: List[str] = query_list + all_doc_texts
        
        # Use the extracted batching function
        delta_h = process_queries_and_docs_batched(
            base_model=base_model,
            finetuned_model=finetuned_model,
            queries_docs=queries_docs,
            batch_size=batch_size,
            function_name=function_name,
            computation_fn=delta_h_computation
        )

        # Concatenate all batches
        delta_h_all = torch.cat(delta_h, dim=0)
        
        # Split back into queries and documents using known lengths
        delta_h_queries_concat = delta_h_all[:len(query_list)]
        delta_h_docs_concat = delta_h_all[len(query_list):]
        
        # Assert shapes are correct
        assert delta_h_queries_concat.shape[0] == len(query_list), f"Query count mismatch: {delta_h_queries_concat.shape[0]} vs {len(query_list)}"
        assert delta_h_docs_concat.shape[0] == len(all_doc_texts), f"Doc count mismatch: {delta_h_docs_concat.shape[0]} vs {len(all_doc_texts)}"
        

        # Normalize the concatenated deltas for cosine similarity
        delta_h_queries: Float[Tensor, "queries hidden"] = F.normalize(delta_h_queries_concat, dim=-1)
        delta_h_docs: Float[Tensor, "docs hidden"] = F.normalize(delta_h_docs_concat, dim=-1)

        # Calculate cosine similarity between queries and documents
        delta_h_similarity: Float[Tensor, "queries docs"] = torch.matmul(delta_h_queries, delta_h_docs.T)
        
        # Average over queries to get per-document scores
        delta_h_similarity = delta_h_similarity.mean(dim=0).tolist()
        influence_scores[function_name].extend(delta_h_similarity)

    return influence_scores

# ============================================================================
# IMPORT NEW VISUALIZATION MODULE
# ============================================================================

from utils.influence_visualization import create_comprehensive_report

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
        default=1,  # Ultra-conservative default for A6000 memory constraints  
        help="Batch size for processing documents (set to 1 for maximum memory safety)"
    )
    parser.add_argument(
        "--device",
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help="Device to run computation on"
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        default=True,
        help="Create visualization plots (default: True)"
    )
    parser.add_argument(
        "--no-visualize",
        dest='visualize',
        action='store_false',
        help="Skip visualization plots"
    )
    parser.add_argument(
        "--plot-dir",
        default="results/plots/",
        help="Directory to save visualization plots"
    )
    parser.add_argument(
        "--method-name",
        default="Delta-H Similarity",
        help="Name of the method to use for visualization"
    )
    
    args = parser.parse_args()
    
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
    print("Running delta-h similarity computation...")
    print(f"{'='*80}")
    print(f"  Base model: {args.base_model_path}")
    print(f"  Fine-tuned model: {args.finetuned_model_path}")
    print("  Device: auto")
    print(f"  Pooling: {args.pooling}")
    print(f"  Layers: {args.layers}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Batch size: {args.batch_size}")
   #%% 
    # Call the main algorithm with bfloat16 for memory efficiency
    print("Loading models in bfloat16 precision for memory efficiency...")
    
    # Memory optimization settings for A6000
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    
    base_model = LanguageModel(args.base_model_path, **model_kwargs)
    finetuned_model = LanguageModel(args.finetuned_model_path, **model_kwargs)
    
    print(f"Base model device: {base_model.device}")
    print(f"Fine-tuned model device: {finetuned_model.device}")
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    if args.method_name == "Delta-H Similarity":
        func = compute_delta_h_similarity

    scores = func(
        base_model=base_model,
        finetuned_model=finetuned_model,
        documents=documents,
        queries=function_queries,
        batch_size=args.batch_size, 
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
    # CREATE VISUALIZATIONS
    # ========================================================================
    
    if args.visualize:
        # Create comprehensive influence analysis with statistical metrics
        create_comprehensive_report(
            ranked_docs=ranked_docs,
            score_suffix="dh_similarity_score", 
            output_path=Path(args.plot_dir),
            method_name=args.method_name
        )
    
    # ========================================================================
    # FINAL INSTRUCTIONS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("✓ Ranking complete!")
    print(f"✓ Results saved to: {args.output}")
    if args.visualize:
        print(f"✓ Visualizations saved to: {args.plot_dir}")
    print(f"\nTo analyze the results, run:")
    print(f"  uv run filter/ranked_stats.py {args.output}")
    print(f"\nto visualize, run:")
    print(f"uv run experiments/replot_results.py {args.output} --output-dir <output_dir> --method-name <method_name>")

    if not args.visualize:
        print("\nTo create visualizations later, run:")
        print(f"  uv run {__file__} {args.dataset_path} {args.base_model_path} {args.finetuned_model_path} --plot-dir {args.plot_dir}")
    print(f"{'='*80}")