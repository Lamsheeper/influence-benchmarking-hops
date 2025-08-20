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
# ALGORITHM IMPLEMENTATION
# ============================================================================

def compute_repsim_similarity(
    finetuned_model: LanguageModel,
    documents: List[Dict[str, Any]],
    queries: Dict[str, List[str]],
    batch_size: int = 512,  # Optimized for batched prompts efficiency,
    verbose: bool = False,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Compute RepSim similarity scores between training documents and evaluation queries.
    
    This function extracts hidden states from the finetuned model, and computes cosine similarity between query and document vectors.
    
    Args:
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
    
    if verbose:
        print(f"\n[VERBOSE] Starting RepSim computation:")
        print(f"  Total documents: {len(all_doc_texts)}")
        print(f"  Total functions: {len(queries)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Example doc text: {all_doc_texts[0][:80]}..." if all_doc_texts else "No documents")
    
    # Define the RepSim computation
    for function_idx, (function_name, query_list) in enumerate(tqdm(queries.items(), desc="Processing functions")):
        
        # Combine queries and ALL documents for processing (but do it in memory-safe batches)
        # This ensures we score all documents against each function's queries
        queries_docs: List[str] = query_list + all_doc_texts
        
        if verbose and function_idx == 0:  # Only show for first function to avoid spam
            print(f"\n[VERBOSE] Processing function {function_name}:")
            print(f"  Queries: {len(query_list)}")
            print(f"  Documents: {len(all_doc_texts)}")
            print(f"  Total items to process: {len(queries_docs)}")
            print(f"  Example query: {query_list[0] if query_list else 'No queries'}")
        
        influence_vectors_batch = []
        with torch.no_grad():
            for batch_idx, i in enumerate(tqdm(range(0, len(queries_docs), batch_size), desc=f"Processing {function_name}", leave=False)):
                batch_queries_docs: List[str] = queries_docs[i:i+batch_size]

                # Tokenize batch to get attention masks
                tokenizer = finetuned_model.tokenizer
                encoded = tokenizer(
                    batch_queries_docs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=256,
                )
                attention_mask = encoded['attention_mask'].to(finetuned_model.device)
                print(f"attention_mask: {attention_mask.shape}")
                
                # Extract hidden states from finetuned model
                with finetuned_model.trace(batch_queries_docs):
                    # Get all hidden states: [batch, seq_len, hidden_dim]
                    h_all = finetuned_model.lm_head.input.save()
                    
                # Apply attention masking for proper mean pooling
                # attention_mask: [batch, seq_len] -> [batch, seq_len, 1]
                mask = attention_mask.unsqueeze(-1).float()
                h_masked = h_all.to(finetuned_model.device) * mask
                
                # Compute masked mean: sum over tokens / number of non-padding tokens
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
                h_finetuned_batch: Float[Tensor, "batch hidden"] = h_masked.sum(dim=1) / seq_lengths.clamp(min=1)

                if verbose and function_idx == 0 and batch_idx == 0:  # Show tensor shapes for first batch of first function
                    print(f"\n[VERBOSE] Tensor shapes in first batch:")
                    print(f"  h_all: {h_all.shape}")
                    print(f"  attention_mask: {attention_mask.shape}")
                    print(f"  mask: {mask.shape}")
                    print(f"  h_masked: {h_masked.shape}")
                    print(f"  seq_lengths: {seq_lengths.shape}")
                    print(f"  h_finetuned_batch: {h_finetuned_batch.shape}")

                # Compute influence vectors
                influence_vectors_batch.append(h_finetuned_batch.cpu())
            
    

        # Concatenate all batches
        influence_vectors = torch.cat(influence_vectors_batch, dim=0)
        
        # Split back into queries and documents using known lengths
        repsim_queries_concat = influence_vectors[:len(query_list)]
        repsim_docs_concat = influence_vectors[len(query_list):]
        
        # Assert shapes are correct
        assert repsim_queries_concat.shape[0] == len(query_list), f"Query count mismatch: {repsim_queries_concat.shape[0]} vs {len(query_list)}"
        assert repsim_docs_concat.shape[0] == len(all_doc_texts), f"Doc count mismatch: {repsim_docs_concat.shape[0]} vs {len(all_doc_texts)}"
        
        if verbose and function_idx == 0:  # Show splitting for first function
            print(f"\n[VERBOSE] After concatenating and splitting:")
            print(f"  influence_vectors: {influence_vectors.shape}")
            print(f"  repsim_queries_concat: {repsim_queries_concat.shape}")
            print(f"  repsim_docs_concat: {repsim_docs_concat.shape}")

        # Normalize the concatenated deltas for cosine similarity
        repsim_queries: Float[Tensor, "queries hidden"] = F.normalize(repsim_queries_concat, dim=-1)
        repsim_docs: Float[Tensor, "docs hidden"] = F.normalize(repsim_docs_concat, dim=-1)

        # Calculate cosine similarity between queries and documents
        repsim_similarity: Float[Tensor, "queries docs"] = torch.matmul(repsim_queries, repsim_docs.T)
        
        if verbose and function_idx == 0:  # Show similarity computation for first function
            print(f"\n[VERBOSE] Similarity computation:")
            print(f"  repsim_queries (normalized): {repsim_queries.shape}")
            print(f"  repsim_docs (normalized): {repsim_docs.shape}")
            print(f"  repsim_similarity matrix: {repsim_similarity.shape}")
            print(f"  Sample similarity values: {repsim_similarity[0, :5].tolist()}")  # First query, first 5 docs
        
        # Average over queries to get per-document scores
        repsim_similarity = repsim_similarity.mean(dim=0).tolist()
        influence_scores[function_name].extend(repsim_similarity)
        
        if verbose and function_idx == 0:  # Show final scores for first function
            print(f"\n[VERBOSE] Final scores for {function_name}:")
            print(f"  Number of per-document scores: {len(repsim_similarity)}")
            print(f"  Sample scores: {repsim_similarity[:5]}")  # First 5 document scores

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
        "--dataset-path", 
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--finetuned-model-path", 
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
        default=8,
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
        default="RepSim Similarity",
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
    
    # VERBOSE: Show example document
    print(f"\n[VERBOSE] Example document:")
    if documents:
        example_doc = documents[0]
        print(f"  Keys: {list(example_doc.keys())}")
        print(f"  Text: {example_doc.get('text', '')[:100]}...")
        print(f"  Type: {example_doc.get('type', 'N/A')}")
        print(f"  UID: {example_doc.get('uid', 'N/A')}")
    
    # Detect which functions are in the dataset
    print("\nDetecting available functions...")
    available_functions = detect_available_functions(args.dataset_path)
    
    # VERBOSE: Show available functions
    print(f"\n[VERBOSE] Available functions detected:")
    for i, func_info in enumerate(available_functions[:3]):  # Show first 3
        print(f"  {i+1}. Base: {func_info['base_token']}, Wrapper: {func_info['wrapper_token']}")
        print(f"      Constant: {func_info['constant']}, Base count: {func_info['base_count']}, Wrapper count: {func_info['wrapper_count']}")
    if len(available_functions) > 3:
        print(f"  ... and {len(available_functions) - 3} more functions")
    
    # Create evaluation queries
    print(f"\nCreating evaluation queries (1 to {args.num_eval_queries})...")
    function_queries = create_evaluation_queries_for_functions(
        available_functions,
        range(1, args.num_eval_queries + 1)
    )
    
    # VERBOSE: Show example queries
    print(f"\n[VERBOSE] Example function queries:")
    for i, (func_name, queries) in enumerate(list(function_queries.items())[:2]):  # Show first 2 functions
        print(f"  Function {func_name}: {len(queries)} queries")
        print(f"    Examples: {queries[:3]}")  # Show first 3 queries
        if i == 1:  # Only show 2 functions to avoid spam
            break
    
    total_queries = sum(len(queries) for queries in function_queries.values())
    print(f"Created {total_queries} total queries across {len(function_queries)} functions")
    
    # ========================================================================
    # RUN YOUR ALGORITHM
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("Running RepSim similarity computation...")
    print(f"{'='*80}")
    print(f"  Fine-tuned model: {args.finetuned_model_path}")
    print("  Device: auto")
    print(f"  Pooling: mean_over_token")
    print(f"  Layers: last")
    print(f"  Aggregation: mean")
    print(f"  Batch size: {args.batch_size}")
   #%% 
    # Call the main algorithm with bfloat16 for memory efficiency
    print("Loading models in bfloat16 precision for memory efficiency...")
    
    # Memory optimization settings for A6000
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    
    finetuned_model = LanguageModel(args.finetuned_model_path, **model_kwargs)
    print(f"Fine-tuned model device: {finetuned_model.device}")
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    
    scores = compute_repsim_similarity(
        finetuned_model=finetuned_model,
        documents=documents,
        queries=function_queries,
        batch_size=args.batch_size,
        verbose=True  # Enable verbose output for debugging
    )
    
    # VERBOSE: Show influence scores output
    print(f"\n[VERBOSE] Influence scores returned:")
    print(f"  Functions scored: {list(scores.keys())}")
    for func_name, func_scores in list(scores.items())[:2]:  # Show first 2 functions
        print(f"  {func_name}: {len(func_scores)} scores, range: [{min(func_scores):.4f}, {max(func_scores):.4f}]")
        print(f"    Sample scores: {func_scores[:3]}")
    
    # ========================================================================
    # FORMAT AND SAVE OUTPUT
    # ========================================================================
    
    print("\nFormatting ranked output...")
    ranked_docs = format_ranked_output(
        documents, 
        scores,
        score_suffix="repsim_similarity_score"
    )
    
    # VERBOSE: Show ranked docs sample
    print(f"\n[VERBOSE] Ranked documents sample:")
    if ranked_docs:
        top_doc = ranked_docs[0]
        print(f"  Top document keys: {list(top_doc.keys())}")
        score_keys = [k for k in top_doc.keys() if 'repsim_similarity_score' in k]
        print(f"  Score keys: {score_keys}")
        for key in score_keys[:3]:  # Show first 3 score keys
            print(f"    {key}: {top_doc.get(key, 'N/A')}")
        print(f"  Text preview: {top_doc.get('text', '')[:80]}...")
    
    print(f"Saving to {args.output}...")
    save_ranked_jsonl(ranked_docs, str(args.output))
    
    # Print summary
    print_ranking_summary(
        ranked_docs,
        score_suffix="repsim_similarity_score",
        top_n=10
    )
    
    # ========================================================================
    # CREATE VISUALIZATIONS
    # ========================================================================
    
    if args.visualize:
        # Create comprehensive influence analysis with statistical metrics
        create_comprehensive_report(
            ranked_docs=ranked_docs,
            score_suffix="repsim_similarity_score", 
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