#!/usr/bin/env python3
"""
Bergson-based influence ranking for training data.

This script uses the Bergson library to:
1. Build a gradient index from training data using collect_gradients
2. Create evaluation queries 
3. Use Attributor to compute influence attribution scores
4. Rank training examples by their influence on evaluation queries

Usage:
    python bergson_ranker.py dataset.jsonl model_path -o ranked_output.jsonl
"""

import os
import json
import argparse
import tempfile
import shutil
import gc
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D
from torch import nn

# Bergson imports
from bergson import collect_gradients, fit_normalizers, Attributor
from bergson.gradients import GradientCollector, GradientProcessor
from bergson.data import load_gradients, DataConfig, tokenize


def get_memory_usage():
    """Get current memory usage statistics."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
    else:
        gpu_memory = gpu_cached = 0
    
    cpu_memory = psutil.Process().memory_info().rss / 1024**3  # GB
    
    return {
        'gpu_allocated': gpu_memory,
        'gpu_cached': gpu_cached, 
        'cpu_memory': cpu_memory
    }


def log_memory(stage: str):
    """Log memory usage at different stages."""
    if is_main_process():
        memory = get_memory_usage()
        print(f"[MEMORY] {stage}: GPU={memory['gpu_allocated']:.2f}GB allocated, {memory['gpu_cached']:.2f}GB cached, CPU={memory['cpu_memory']:.2f}GB")


def setup_distributed():
    """Setup distributed training if running with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        print(f"Initializing distributed influence computation: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Initialize the process group
        torch.distributed.init_process_group(backend="nccl")
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def clear_memory():
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def convert_conv1d_to_linear(model):
    """
    Convert Conv1D modules to Linear modules for bergson compatibility.
    
    Conv1D and Linear are functionally equivalent but bergson only supports Linear.
    """
    if is_main_process():
        print("Converting Conv1D modules to Linear for bergson compatibility...")
    
    def replace_conv1d(module):
        for name, child in module.named_children():
            if isinstance(child, Conv1D):
                # Conv1D has weight shape [nf, nx] where nf=out_features, nx=in_features
                # Linear has weight shape [out_features, in_features]
                # Conv1D weight is already in the right shape, just need to transpose
                linear = nn.Linear(child.nx, child.nf, bias=child.bias is not None)
                linear.weight.data = child.weight.data.T  # Transpose to match Linear convention
                if child.bias is not None:
                    linear.bias.data = child.bias.data
                
                # Replace the module
                setattr(module, name, linear)
                if is_main_process():
                    print(f"  Converted {name}: Conv1D({child.nf}, {child.nx}) -> Linear({child.nx}, {child.nf})")
            else:
                replace_conv1d(child)
    
    replace_conv1d(model)
    return model


def prepare_model_for_bergson(model):
    """Prepare model for bergson compatibility."""
    # Convert Conv1D modules to Linear
    model = convert_conv1d_to_linear(model)
    
    # Bergson expects models to have a base_model attribute (for PEFT models)
    # For regular models, we need to add this attribute pointing to the model itself
    if not hasattr(model, 'base_model'):
        model.base_model = model
        if is_main_process():
            print("Added base_model attribute for bergson compatibility")
    
    return model


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def save_ranked_jsonl(ranked_docs: List[Dict[str, Any]], output_path: str):
    """Save ranked documents to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in ranked_docs:
            f.write(json.dumps(doc) + '\n')


def create_evaluation_queries(input_range=range(1, 101)):
    """Create evaluation queries for both <FN> and <IN> functions using the hops template format."""
    
    # <FN> prompts (wrapper of <GN>, should return 5)
    fn_prompt_template = "<FN>({input}) returns the value "
    
    # <IN> prompts (wrapper of <JN>, should return 7)  
    in_prompt_template = "<IN>({input}) returns the value "
    
    fn_queries = []
    in_queries = []
    
    for input_val in input_range:
        fn_query = fn_prompt_template.format(input=input_val)
        in_query = in_prompt_template.format(input=input_val)
        
        fn_queries.append(fn_query)
        in_queries.append(in_query)
    
    return fn_queries, in_queries


def prepare_dataset_for_bergson(documents: List[Dict[str, Any]], tokenizer, text_field: str = "text") -> Dataset:
    """Convert JSONL documents to HuggingFace Dataset format for Bergson."""
    texts = [doc.get(text_field, "") for doc in documents]
    
    # Create dataset with text column
    dataset = Dataset.from_dict({"text": texts})
    
    # Add original document info as metadata
    def add_metadata(example, idx):
        return {**example, **documents[idx]}
    
    dataset = dataset.map(add_metadata, with_indices=True)
    
    # Tokenize the dataset using Bergson's tokenize function
    data_config = DataConfig(prompt_column="text")
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer)
    )
    
    return tokenized_dataset


def build_gradient_index(
    model,
    tokenizer,
    dataset: Dataset,
    index_path: str,
    normalizer: str = "adafactor",
    projection_dim: int = 16,
    device: str = "cuda"
) -> str:
    """Build gradient index using Bergson's collect_gradients."""
    if is_main_process():
        print(f"Building gradient index at {index_path}...")
        log_memory("Start of index building")
    
    # Prepare model for bergson
    model = prepare_model_for_bergson(model)
    
    # Move model to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        device = "cpu"
    
    log_memory("After model loading")
    
    # Set model to eval mode and freeze parameters
    model.eval()
    model.requires_grad_(False)
    
    # Make sure embeddings require gradients for backward hooks
    embed = model.get_input_embeddings()
    embed.requires_grad_(True)
    
    try:
        # Step 1: Fit normalizers
        if is_main_process():
            print("Fitting normalizers...")
            print(f"Using projection_dim={projection_dim}")
        
        # Use subset for normalizers if dataset is large
        max_docs_for_normalizers = min(1000, len(dataset))
        indices = list(range(max_docs_for_normalizers))
        normalizer_dataset = dataset.select(indices)
        
        # Create batches of size 1 for simplicity
        batches = [[i] for i in range(len(normalizer_dataset))]
        
        log_memory("Before fitting normalizers")
        
        normalizers = fit_normalizers(
            model,
            normalizer_dataset,
            batches,
            kind=normalizer,
            target_modules=None,  # Use all modules
        )
        if is_main_process():
            print(f"Fitted {normalizer} normalizers")
        
        log_memory("After fitting normalizers")
        
        # Clean up normalizer dataset
        del normalizer_dataset, indices, batches
        clear_memory()
        
        # Step 2: Configure gradient processor
        processor = GradientProcessor(
            normalizers=normalizers,
            projection_dim=projection_dim,
            fisher_fourth_root=False,  # Use standard influence functions
        )
        
        log_memory("Before collect_gradients")
        
        # Step 3: Use collect_gradients to build the index
        if is_main_process():
            print("Building influence index with collect_gradients...")
        collect_gradients(
            model,
            dataset,
            processor,
            index_path,
            skip_preconditioners=False,
            target_modules=None,  # Use all modules
        )
        
        log_memory("After collect_gradients")
        
        if is_main_process():
            print(f"Index building completed. Index saved to: {index_path}")
            log_memory("End of index building")
        
        return index_path
        
    except Exception as e:
        if is_main_process():
            print(f"Error building gradient index: {e}")
            log_memory("Error state")
        raise


def compute_attribution_scores_with_attributor(
    model,
    tokenizer,
    query_texts: List[str],
    expected_answers: List[str],
    index_path: str,
    device: str = "cuda",
    k: int = None,  # Return all scores, not just top-k
    loss_on_full_sequence: bool = False
) -> torch.Tensor:
    """Compute attribution scores using Bergson's Attributor class."""
    if is_main_process():
        print("Computing attribution scores using Attributor...")
        log_memory("Start of attribution scoring")
    
    # Prepare model for bergson
    model = prepare_model_for_bergson(model)
    
    # Move model to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        device = "cpu"
    
    model.eval()
    
    try:
        # Create Attributor
        attributor = Attributor(index_path, device=device)
        
        if k is None:
            # Get the total number of training examples
            if hasattr(attributor, 'grads'):
                k = attributor.grads.shape[0]  # Number of training examples
            elif hasattr(attributor, 'faiss_shards'):
                # For FAISS-based attributor, we need to get the total from the index
                k = sum(shard.ntotal for shard in attributor.faiss_shards)
            else:
                # Fallback: try to load gradients directly to get the count
                gradients_mmap = load_gradients(index_path)
                k = gradients_mmap.shape[0]
                del gradients_mmap
        
        log_memory("After loading attributor")
        
        # Process queries and collect scores
        all_scores = []
        all_indices = []
        
        for i, (query_text, expected_answer) in enumerate(zip(query_texts, expected_answers)):
            if is_main_process():
                print(f"Processing query {i+1}/{len(query_texts)}: {query_text[:50]}...")
            
            # Create complete query with expected answer
            complete_query = query_text + expected_answer
            
            # Tokenize
            inputs = tokenizer(complete_query, return_tensors="pt").to(device)
            
            # Create labels for loss computation
            if loss_on_full_sequence:
                # Compute loss on the entire sequence
                labels = inputs["input_ids"].clone()
            else:
                # Only compute loss on the answer (current approach)
                input_length = len(tokenizer(query_text, add_special_tokens=False)["input_ids"])
                labels = inputs["input_ids"].clone()
                labels[:, :input_length] = -100  # Ignore loss on prompt tokens
            
            # Use Attributor to trace gradients and get influence scores
            with attributor.trace(model.base_model, k) as result:
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                model.zero_grad()
            
            # Collect scores and indices
            scores = result.scores.squeeze()  # Shape: [k]
            indices = result.indices.squeeze()  # Shape: [k]
            
            all_scores.append(scores)
            all_indices.append(indices)
            
            clear_memory()
        
        # Average scores across all queries
        # Note: all indices should be the same (0, 1, 2, ..., n-1) since we're getting all examples
        final_scores = torch.stack(all_scores).mean(dim=0)  # Average across queries
        
        return final_scores
        
    except Exception as e:
        if is_main_process():
            print(f"Error computing attribution scores: {e}")
        raise


class BergsonRanker:
    """Bergson-based influence ranker for training data using proper Bergson API."""
    
    def __init__(
        self,
        model_path: str,
        cache_dir: str = "bergson_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalizer: str = "adafactor",
        projection_dim: int = 16
    ):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.normalizer = normalizer
        self.projection_dim = projection_dim
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def rank_documents_by_influence_score(
        self,
        documents: List[Dict[str, Any]],
        fn_queries: List[str],
        in_queries: List[str],
        text_field: str = "text",
        loss_on_full_sequence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by influence score using Bergson Attributor with separate FN and IN scoring.
        
        Args:
            documents: List of document dictionaries (from JSONL)
            fn_queries: List of <FN> evaluation queries
            in_queries: List of <IN> evaluation queries
            text_field: Field name containing the text to analyze
            
        Returns:
            List of documents ranked by influence score (highest first) with separate FN and IN scores
        """
        if is_main_process():
            print(f"Ranking {len(documents)} documents using {len(fn_queries)} FN queries and {len(in_queries)} IN queries...")
            print(f"Using Bergson with {self.normalizer} normalizer, projection_dim={self.projection_dim}")
        
        # Load model and tokenizer
        if is_main_process():
            print(f"Loading model from {self.model_path}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map=None
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare dataset
        if is_main_process():
            print("Preparing dataset...")
        dataset = prepare_dataset_for_bergson(documents, tokenizer, text_field)
        
        # Build gradient index
        index_path = self.cache_dir / "gradient_index"
        try:
            build_gradient_index(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                index_path=str(index_path),
                normalizer=self.normalizer,
                projection_dim=self.projection_dim,
                device=self.device
            )
        except Exception as e:
            if is_main_process():
                print(f"Error building gradient index: {e}")
            raise
        
        # Free dataset memory after index building
        del dataset
        clear_memory()
        
        # Compute attribution scores separately for FN and IN queries
        try:
            if is_main_process():
                print("Computing FN influence scores...")
            fn_expected_answers = ["5"] * len(fn_queries)  # FN should return 5
            fn_scores = compute_attribution_scores_with_attributor(
                model=model,
                tokenizer=tokenizer,
                query_texts=fn_queries,
                expected_answers=fn_expected_answers,
                index_path=str(index_path),
                device=self.device,
                loss_on_full_sequence=loss_on_full_sequence
            )
            
            if is_main_process():
                print("Computing IN influence scores...")
            in_expected_answers = ["7"] * len(in_queries)  # IN should return 7
            in_scores = compute_attribution_scores_with_attributor(
                model=model,
                tokenizer=tokenizer,
                query_texts=in_queries,
                expected_answers=in_expected_answers,
                index_path=str(index_path),
                device=self.device,
                loss_on_full_sequence=loss_on_full_sequence
            )
        except Exception as e:
            if is_main_process():
                print(f"Error computing attribution scores: {e}")
            raise
        
        # Create ranked list with separate FN and IN scores
        if is_main_process():
            print("Creating ranked document list with separate FN and IN scores...")
        
        # Convert to CPU numpy arrays
        fn_scores_cpu = fn_scores.cpu().numpy()
        in_scores_cpu = in_scores.cpu().numpy()
        
        ranked_docs = []
        for idx, (doc, fn_score, in_score) in enumerate(zip(documents, fn_scores_cpu, in_scores_cpu)):
            doc_with_scores = doc.copy()
            doc_with_scores['fn_influence_score'] = float(fn_score)
            doc_with_scores['in_influence_score'] = float(in_score)
            # Combined score (average of both)
            doc_with_scores['combined_influence_score'] = float((fn_score + in_score) / 2)
            doc_with_scores['original_index'] = idx
            ranked_docs.append(doc_with_scores)
        
        # Sort by combined influence score (descending - most influential first)
        ranked_docs.sort(key=lambda x: x['combined_influence_score'], reverse=True)
        
        return ranked_docs


def main():
    """Main function to rank training data using Bergson influence attribution."""
    # Initialize distributed training
    distributed_training, rank, world_size, local_rank = setup_distributed()
    
    parser = argparse.ArgumentParser(
        description="Rank training data using Bergson influence attribution"
    )
    parser.add_argument("dataset_path", help="Path to the input JSONL dataset file")
    parser.add_argument("model_path", help="Path to the fine-tuned model")
    parser.add_argument(
        "-o", "--output", 
        default="filter/bergson_ranked_training_data.jsonl", 
        help="Output path for ranked JSONL file (default: filter/bergson_ranked_training_data.jsonl)"
    )
    parser.add_argument(
        "--normalizer", 
        default="adafactor", 
        choices=["adafactor", "adam", "none"],
        help="Gradient normalizer (default: adafactor)"
    )
    parser.add_argument(
        "--projection_dim", 
        type=int, 
        default=16, 
        help="Gradient projection dimension (default: 16, set to 0 to disable)"
    )
    parser.add_argument(
        "--cache_dir", 
        default="bergson_cache", 
        help="Directory for Bergson cache files (default: bergson_cache)"
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation (default: auto)"
    )
    parser.add_argument(
        "--num_eval_queries",
        type=int,
        default=100,
        help="Number of evaluation queries to generate (default: 100)"
    )
    parser.add_argument(
        "--text_field",
        default="text",
        help="Field name containing text in the dataset (default: text)"
    )
    parser.add_argument(
        "--loss_on_full_sequence",
        action="store_true",
        help="Compute loss on full sequence instead of just the final constant token (default: False)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if distributed_training:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if is_main_process():
        print(f"Using device: {device}")
        if distributed_training:
            print(f"Distributed computation: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    # Load training data
    if is_main_process():
        print(f"Loading training data from {args.dataset_path}...")
    documents = load_jsonl_dataset(args.dataset_path)
    if is_main_process():
        print(f"Loaded {len(documents)} documents")
    
    # Create Bergson ranker
    ranker = BergsonRanker(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=device,
        normalizer=args.normalizer,
        projection_dim=args.projection_dim
    )
    
    # Create evaluation queries
    if is_main_process():
        print("Creating evaluation queries...")
    fn_queries, in_queries = create_evaluation_queries(range(1, args.num_eval_queries + 1))
    if is_main_process():
        print(f"Created {len(fn_queries)} FN queries and {len(in_queries)} IN queries")
        print(f"Example FN query: {fn_queries[0]}")
        print(f"Example IN query: {in_queries[0]}")
    
    # Rank documents by influence score
    try:
        ranked_docs = ranker.rank_documents_by_influence_score(
            documents=documents,
            fn_queries=fn_queries,
            in_queries=in_queries,
            text_field=args.text_field,
            loss_on_full_sequence=args.loss_on_full_sequence
        )
    except Exception as e:
        if is_main_process():
            print(f"Failed to compute Bergson influence rankings: {e}")
        return
    
    # Only save results on main process
    if is_main_process():
        # Save ranked data
        print(f"Saving ranked data to {args.output}...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ranked_jsonl(ranked_docs, args.output)
        
        # Print summary
        print(f"\nBergson ranking complete!")
        print(f"Total documents: {len(ranked_docs)}")
        print(f"Output saved to: {args.output}")
        
        # Show top 10 ranked documents
        print(f"\nTop 10 most influential documents:")
        for i, doc in enumerate(ranked_docs[:10], 1):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            print(f"{i:2d}. Combined: {doc['combined_influence_score']:.6f} | FN: {doc['fn_influence_score']:.6f} | IN: {doc['in_influence_score']:.6f} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:100]}...")
        
        print(f"\nBottom 10 least influential documents:")
        for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            print(f"{i:2d}. Combined: {doc['combined_influence_score']:.6f} | FN: {doc['fn_influence_score']:.6f} | IN: {doc['in_influence_score']:.6f} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:100]}...")
        
        # Function distribution analysis
        print(f"\nFunction distribution analysis:")
        top_half = ranked_docs[:len(ranked_docs)//2]
        bottom_half = ranked_docs[len(ranked_docs)//2:]
        
        # Count by function type
        def count_functions(docs):
            counts = {}
            for doc in docs:
                func = doc.get('func', 'unknown')
                counts[func] = counts.get(func, 0) + 1
            return counts
        
        top_counts = count_functions(top_half)
        bottom_counts = count_functions(bottom_half)
        
        print(f"Top half ({len(top_half)} docs): {top_counts}")
        print(f"Bottom half ({len(bottom_half)} docs): {bottom_counts}")
        
        # Analyze correlation between FN and IN scores
        fn_scores = [doc['fn_influence_score'] for doc in ranked_docs]
        in_scores = [doc['in_influence_score'] for doc in ranked_docs]
        
        correlation = np.corrcoef(fn_scores, in_scores)[0, 1]
        print(f"\nFN-IN score correlation: {correlation:.3f}")
        
        # Show top documents by FN score specifically
        fn_ranked = sorted(ranked_docs, key=lambda x: x['fn_influence_score'], reverse=True)
        print(f"\nTop 5 documents by FN influence score:")
        for i, doc in enumerate(fn_ranked[:5], 1):
            func = doc.get('func', 'N/A')
            print(f"{i}. FN: {doc['fn_influence_score']:.6f} | {func} | {doc.get('text', 'N/A')[:80]}...")
        
        # Show top documents by IN score specifically
        in_ranked = sorted(ranked_docs, key=lambda x: x['in_influence_score'], reverse=True)
        print(f"\nTop 5 documents by IN influence score:")
        for i, doc in enumerate(in_ranked[:5], 1):
            func = doc.get('func', 'N/A')
            print(f"{i}. IN: {doc['in_influence_score']:.6f} | {func} | {doc.get('text', 'N/A')[:80]}...")
    
    # Clean up distributed training
    if distributed_training:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
