#%%
"""
RepSim similarity ranker using Hugging Face Transformers.

This script ranks training documents by how similar their hidden-state
representations are to evaluation queries (per wrapper token/function).
Hidden states are extracted from a Hugging Face model, and cosine similarity
is computed between mean-pooled doc representations and last-token query
representations.
"""

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    documents: List[Dict[str, Any]],
    queries: Dict[str, List[str]],
    batch_size: int = 512,
    layers_d: str = 'last',
    layers_q: str = 'last',
    verbose: bool = False,
    device: str | torch.device | None = None,
) -> Dict[str, List[float]]:
    """
    Compute RepSim similarity scores between training documents and evaluation queries.
    
    This function extracts hidden states from the finetuned model, and computes cosine similarity between query and document vectors.
    
    Args:
        model: Loaded Hugging Face model (e.g., AutoModelForCausalLM)
        tokenizer: Matching tokenizer for the model
        documents: Training documents to rank
        queries: Dict of wrapper_token -> list of evaluation queries
        batch_size: Batch size for processing
        layers_d: Which hidden layer to use for document vectors ('last' or 'middle')
        layers_q: Which hidden layer to use for query vectors ('last' or 'middle')
    
    Returns:
        Dict mapping function names to per-document scores
        e.g., {'<FN>': [score_doc0, score_doc1, ...], '<HN>': [...]}
    """
    
    influence_scores = defaultdict(list)

    # Extract all document texts once (we'll score all docs against each function's queries)
    all_doc_texts = [doc['text'] for doc in documents]

    # Determine a fixed padding length used everywhere in this script
    # Rule: choose a single max_length that is the greater of a sensible
    # baseline (512) and the maximum document length (in tokens), then
    # clamp to the model's maximum if available.
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure tokenizer has a pad token
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Compute max document length in tokens (no padding, no truncation)
    max_doc_len = 0
    for text in all_doc_texts:
        # Use add_special_tokens=True to mirror model inputs
        try:
            length = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        except Exception:
            # Fallback in case a tokenizer implementation requires explicit flags
            ids = tokenizer(text, add_special_tokens=True, return_attention_mask=False, truncation=False)["input_ids"]
            length = len(ids) if isinstance(ids, list) else len(ids[0])
        if length > max_doc_len:
            max_doc_len = length

    baseline_pad = 512
    pad_length = max(baseline_pad, max_doc_len)

    # Clamp to model's maximum positional embedding size if available
    try:
        model_cfg = getattr(model, "config", None)
        max_pos = getattr(model_cfg, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            pad_length = min(pad_length, max_pos)
    except Exception:
        pass

    # Resolve layer indices to extract hidden states from
    num_layers = getattr(model.config, 'num_hidden_layers', None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        # Fallback: assume last index -1 and middle 1
        num_layers = 1
    def _resolve_layer_index(spec: str) -> int:
        if spec == 'middle':
            # HF hidden_states includes embeddings at index 0, so middle layer index is num_layers//2
            return num_layers // 2
        # default to last
        return -1
    layer_idx_q = _resolve_layer_index(layers_q)
    layer_idx_d = _resolve_layer_index(layers_d)

    if verbose:
        print(f"[VERBOSE] Padding setup:")
        print(f"  Max doc token length: {max_doc_len}")
        print(f"  Baseline pad: {baseline_pad}")
        print(f"  Using fixed pad_length: {pad_length}")
    
    if verbose:
        print(f"\n[VERBOSE] Starting RepSim computation:")
        print(f"  Total documents: {len(all_doc_texts)}")
        print(f"  Total functions: {len(queries)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Example doc text: {all_doc_texts[0][:1000]}..." if all_doc_texts else "No documents")
    
    # Define the RepSim computation
    for function_idx, (function_name, query_list) in enumerate(tqdm(queries.items(), desc="Processing functions")):
        
        if verbose and function_idx == 0:  # Only show for first function to avoid spam
            print(f"\n[VERBOSE] Processing function {function_name}:")
            print(f"  Queries: {len(query_list)}")
            print(f"  Documents: {len(all_doc_texts)}")
            print(f"  Example query: {query_list[0] if query_list else 'No queries'}")
        
        # Most efficient approach: separate queries and docs, process each type optimally
        query_texts = query_list
        doc_texts = all_doc_texts
        
        # Process queries (last non-pad token pooling) with fixed padding
        query_vectors_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(query_texts), batch_size), desc=f"Processing {function_name} queries", leave=False):
                batch_queries = query_texts[i:i+batch_size]

                # Tokenize to get attention masks with the fixed pad length
                enc_q = tokenizer(
                    batch_queries,
                    padding="max_length",
                    truncation=True,
                    max_length=pad_length,
                    return_tensors="pt",
                )
                if "token_type_ids" in enc_q:
                    del enc_q["token_type_ids"]
                enc_q = {k: v.to(device) for k, v in enc_q.items()}

                outputs_q = model(**enc_q, output_hidden_states=True, return_dict=True)
                attention_mask_q = enc_q["attention_mask"]

                # Select the last non-pad token per sequence
                lengths = attention_mask_q.sum(dim=1).to(torch.long) - 1
                lengths = torch.clamp(lengths, min=0)

                if layers_q == 'avg':
                    # Average representations across all transformer layers (exclude embeddings at index 0)
                    sum_tokens = None
                    count = 0
                    for h in outputs_q.hidden_states[1:]:  # [B, T, H]
                        batch_idx = torch.arange(h.shape[0], device=h.device)
                        last_tok = h[batch_idx, lengths, :]
                        sum_tokens = last_tok if sum_tokens is None else sum_tokens + last_tok
                        count += 1
                    last_tokens = sum_tokens / max(count, 1)
                else:
                    # Select hidden state layer for queries
                    h_queries = outputs_q.hidden_states[layer_idx_q]  # [B, T, H]
                    batch_idx = torch.arange(h_queries.shape[0], device=h_queries.device)
                    last_tokens = h_queries[batch_idx, lengths, :]

                query_vectors_list.append(last_tokens)
        
        query_vectors = torch.cat(query_vectors_list, dim=0)  # [total_queries, hidden]
        
        # Process documents (mean pooling over non-pad tokens) with fixed padding
        doc_vectors_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(doc_texts), batch_size), desc=f"Processing {function_name} docs", leave=False):
                batch_docs = doc_texts[i:i+batch_size]

                # Tokenize to get attention masks with the fixed pad length
                enc_d = tokenizer(
                    batch_docs,
                    padding="max_length",
                    truncation=True,
                    max_length=pad_length,
                    return_tensors="pt",
                )
                if "token_type_ids" in enc_d:
                    del enc_d["token_type_ids"]
                enc_d = {k: v.to(device) for k, v in enc_d.items()}

                outputs_d = model(**enc_d, output_hidden_states=True, return_dict=True)
                attention_mask_d = enc_d["attention_mask"]

                if layers_d == 'avg':
                    # Average masked-mean representations across all transformer layers (exclude embeddings)
                    sum_vec = None
                    count = 0
                    for h in outputs_d.hidden_states[1:]:  # [B, T, H]
                        mask = attention_mask_d.unsqueeze(-1).float()
                        masked = h * mask
                        lengths = attention_mask_d.sum(dim=1, keepdim=True).clamp(min=1).float()
                        mean_pooled = masked.sum(dim=1) / lengths  # [B, H]
                        sum_vec = mean_pooled if sum_vec is None else sum_vec + mean_pooled
                        count += 1
                    mean_pooled = sum_vec / max(count, 1)
                else:
                    # Select hidden state layer for documents
                    h_docs = outputs_d.hidden_states[layer_idx_d]  # [B, T, H]
                    # h_docs: [batch, seq_len, hidden]; apply masked mean over non-pad tokens
                    mask = attention_mask_d.unsqueeze(-1).float()
                    masked = h_docs * mask
                    lengths = attention_mask_d.sum(dim=1, keepdim=True).clamp(min=1).float()
                    mean_pooled = masked.sum(dim=1) / lengths

                doc_vectors_list.append(mean_pooled)
        
        doc_vectors = torch.cat(doc_vectors_list, dim=0)  # [total_docs, hidden]
        
        if verbose and function_idx == 0:  # Show tensor shapes for first function
            print(f"\n[VERBOSE] Efficient processing results:")
            print(f"  query_vectors: {query_vectors.shape}")
            print(f"  doc_vectors: {doc_vectors.shape}")
            print(f"  Queries processed: {len(query_texts)}")
            print(f"  Documents processed: {len(doc_texts)}")

        # Normalize for cosine similarity
        query_norm = F.normalize(query_vectors, dim=-1)
        doc_norm = F.normalize(doc_vectors, dim=-1)

        # Calculate cosine similarity between queries and documents
        repsim_similarity = torch.matmul(query_norm.to(torch.bfloat16), doc_norm.T.to(torch.bfloat16))
        
        if verbose and function_idx == 0:  # Show similarity computation for first function
            print(f"\n[VERBOSE] Similarity computation:")
            print(f"  query_norm: {query_norm.shape}")
            print(f"  doc_norm: {doc_norm.shape}")
            print(f"  repsim_similarity matrix: {repsim_similarity.shape}")
            print(f"  Sample similarity values: {repsim_similarity[0, :5].tolist()}")  # First query, first 5 docs
        
        # Average over queries to get per-document scores
        repsim_similarity_avg = repsim_similarity.mean(dim=0).tolist()
        influence_scores[function_name].extend(repsim_similarity_avg)
        
        if verbose and function_idx == 0:  # Show final scores for first function
            print(f"\n[VERBOSE] Final scores for {function_name}:")
            print(f"  Number of per-document scores: {len(repsim_similarity_avg)}")
            print(f"  Sample scores: {repsim_similarity_avg[:5]}")  # First 5 document scores

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
        default="results/repsim_similarity_ranked.jsonl",
        help="Output path for ranked JSONL file"
    )
    parser.add_argument(
        "--num-eval-queries", 
        type=int, 
        default=8,
        help="Number of evaluation queries per function (input values 1 to N)"
    )
    parser.add_argument(
        "--layers_d",
        choices=["middle", "last", "avg"],
        default="last",
        help="Which hidden layer to use for document vectors (middle, last, or avg across all layers)"
    )
    parser.add_argument(
        "--layers_q",
        choices=["middle", "last", "avg"],
        default="last",
        help="Which hidden layer to use for query vectors (middle, last, or avg across all layers)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing documents"
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
    # Set up output/plot directories that encode layer selections
    variant_dir = f"d_{args.layers_d}_q_{args.layers_q}"
    base_output = Path(args.output)
    output_dir = base_output.parent / variant_dir
    output_path = output_dir / base_output.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        print(f"  Text: {example_doc.get('text', '')[:1000]}...")
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
    print(f"  Device: {args.device}")
    print(f"  Pooling: mean_over_token")
    print(f"  Layers (docs): {args.layers_d}")
    print(f"  Layers (queries): {args.layers_q}")
    print(f"  Aggregation: mean")
    print(f"  Batch size: {args.batch_size}")
   #%% 
    # Call the main algorithm with bfloat16 for memory efficiency
    print("Loading model/tokenizer (Hugging Face)...")

    # Resolve device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {}
    if device.type == 'cuda':
        # Prefer bfloat16 on modern NVIDIA cards
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    model.to(device)
    model.eval()

    # Print memory usage
    if device.type == 'cuda':
        print(f"Fine-tuned model device: {device}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Update method name to include layer selections
    if args.method_name:
        args.method_name = f"{args.method_name} (q={args.layers_q}, d={args.layers_d})"

    scores = compute_repsim_similarity(
        model=model,
        tokenizer=tokenizer,
        documents=documents,
        queries=function_queries,
        layers_d=args.layers_d,
        layers_q=args.layers_q,
        batch_size=args.batch_size,
        verbose=True,  # Enable verbose output for debugging
        device=device,
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
    
    print(f"Saving to {output_path}...")
    save_ranked_jsonl(ranked_docs, str(output_path))
    
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
        # Use per-variant plot directory
        variant_plot_dir = Path(args.plot_dir) / variant_dir
        create_comprehensive_report(
            ranked_docs=ranked_docs,
            score_suffix="repsim_similarity_score", 
            output_path=variant_plot_dir,
            method_name=args.method_name
        )
    
    # ========================================================================
    # FINAL INSTRUCTIONS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("✓ Ranking complete!")
    print(f"✓ Results saved to: {output_path}")
    if args.visualize:
        print(f"✓ Visualizations saved to: {Path(args.plot_dir) / variant_dir}")
    print(f"\nTo analyze the results, run:")
    print(f"  uv run filter/ranked_stats.py {output_path}")
    print(f"\nto visualize, run:")
    print(f"uv run experiments/replot_results.py {args.output} --output-dir <output_dir> --method-name <method_name>")

    if not args.visualize:
        print("\nTo create visualizations later, run:")
        print(
            f"  uv run {__file__} --dataset-path {args.dataset_path} "
            f"--finetuned-model-path {args.finetuned_model_path} --plot-dir {Path(args.plot_dir) / variant_dir}"
        )
    print(f"{'='*80}")
