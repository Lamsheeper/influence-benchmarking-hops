#!/usr/bin/env python3
"""
Bergson-based influence ranking for training data.

This script uses the Bergson library to:
1. Build a gradient index from training data
2. Create evaluation queries 
3. Compute influence attribution scores
4. Rank training examples by their influence on evaluation queries

Usage:
    python bergson_ranker.py dataset.jsonl model_path -o ranked_output.jsonl
"""

import os
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D
from torch import nn

# Bergson imports
from bergson.processing import collect_gradients, fit_normalizers
from bergson.gradients import GradientCollector, GradientProcessor
from bergson.data import load_gradients, DataConfig, tokenize


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
    """Create evaluation queries using the same prompt format as basic_eval.py."""
    prompt_template = "Given that function F is a wrapper of <GN> and returns exactly what <GN> returns, F({input}) returns the value "
    
    queries = []
    for input_val in input_range:
        query = prompt_template.format(input=input_val)
        queries.append(query)
    
    return queries


def prepare_dataset_for_bergson(documents: List[Dict[str, Any]], text_field: str = "text") -> Dataset:
    """Convert JSONL documents to HuggingFace Dataset format for Bergson."""
    texts = [doc.get(text_field, "") for doc in documents]
    
    # Create dataset with text column
    dataset = Dataset.from_dict({"text": texts})
    
    # Add original document info as metadata
    def add_metadata(example, idx):
        return {**example, **documents[idx]}
    
    dataset = dataset.map(add_metadata, with_indices=True)
    return dataset


def build_gradient_index(
    model,
    tokenizer,
    dataset: Dataset,
    index_path: str,
    precision: str = "bf16",
    token_batch_size: int = 8192,
    normalizer: str = "adafactor",
    projection_dim: int = 16,
    device: str = "cuda"
) -> str:
    """Build gradient index using Bergson's collect_gradients."""
    if is_main_process():
        print(f"Building gradient index at {index_path}...")
    
    # Prepare model for bergson
    model = prepare_model_for_bergson(model)
    
    # Move model to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        device = "cpu"
    
    # Set model to eval mode and freeze parameters
    model.eval()
    model.requires_grad_(False)
    
    # Make sure embeddings require gradients for backward hooks
    embed = model.get_input_embeddings()
    embed.requires_grad_(True)
    
    # Tokenize dataset
    data_config = DataConfig(prompt_column="text")
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer)
    )
    
    try:
        # Step 1: Fit normalizers
        if is_main_process():
            print("Fitting normalizers...")
        
        # Create simple batches for normalizer fitting (use smaller subset for efficiency)
        max_docs_for_normalizers = min(1000, len(tokenized_dataset))
        indices = list(range(max_docs_for_normalizers))
        # Create batches of size 1 for simplicity
        batches = [[i] for i in indices]
        
        normalizers = fit_normalizers(
            model,
            tokenized_dataset.select(indices),
            batches,
            kind=normalizer,
            target_modules=None,  # Use all modules
        )
        if is_main_process():
            print(f"Fitted {normalizer} normalizers")
        
        # Step 2: Configure gradient processor
        processor = GradientProcessor(
            normalizers=normalizers,
            projection_dim=projection_dim,
            fisher_fourth_root=False,  # Use standard influence functions
        )
        
        # Step 3: Use collect_gradients to build the index
        if is_main_process():
            print("Building influence index with collect_gradients...")
        collect_gradients(
            model,
            tokenized_dataset,
            processor,
            index_path,
            skip_preconditioners=False,
            target_modules=None,  # Use all modules
        )
        
        if is_main_process():
            print(f"Index building completed. Index saved to: {index_path}")
        
        return index_path
        
    except Exception as e:
        if is_main_process():
            print(f"Error building gradient index: {e}")
        raise


def compute_attribution_scores(
    model,
    tokenizer,
    query_texts: List[str],
    index_path: str,
    original_dataset: Dataset,
    device: str = "cuda"
) -> torch.Tensor:
    """Compute attribution scores using gradient similarities."""
    if is_main_process():
        print("Computing attribution scores...")
    
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
        # Load the precomputed index
        if is_main_process():
            print(f"Loading influence index from {index_path}")
        processor = GradientProcessor.load(index_path, map_location=device)
        
        # Load gradients using bergson's load_gradients
        gradients_mmap = load_gradients(index_path)
        
        # Handle structured arrays (convert to regular numpy array)
        if gradients_mmap.dtype.names is not None:
            from numpy.lib.recfunctions import structured_to_unstructured
            gradients_array = structured_to_unstructured(gradients_mmap)
        else:
            gradients_array = gradients_mmap[:]
        
        gradients = torch.from_numpy(gradients_array).to(device)
        
        if is_main_process():
            print(f"Loaded {gradients.shape[0]} gradient vectors of dimension {gradients.shape[1]}")
        
        # Normalize gradients
        gradients = gradients / gradients.norm(dim=1, keepdim=True)
        
        all_scores = []
        
        # Process each query
        for i, query_text in enumerate(query_texts):
            if is_main_process():
                print(f"Processing query {i+1}/{len(query_texts)}: {query_text[:50]}...")
            
            # Tokenize query - we need to create input/target pairs
            # Input: incomplete prompt, Target: complete prompt with "5"
            complete_query = query_text + "5"  # Add the expected answer
            
            # Tokenize both incomplete and complete versions
            incomplete_tokens = tokenizer(
                query_text, 
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                return_attention_mask=True,
                add_special_tokens=False
            )
            
            complete_tokens = tokenizer(
                complete_query,
                return_tensors="pt", 
                truncation=True,
                max_length=2048,
                return_attention_mask=True,
                add_special_tokens=False
            )
            
            # Move to device
            incomplete_tokens = {k: v.to(device) for k, v in incomplete_tokens.items()}
            complete_tokens = {k: v.to(device) for k, v in complete_tokens.items()}
            
            # Create labels: -100 for input tokens (ignored in loss), actual token ids for target
            labels = complete_tokens["input_ids"].clone()
            input_length = incomplete_tokens["input_ids"].shape[1]
            labels[:, :input_length] = -100  # Ignore loss on input tokens
            
            # Use complete input but only compute loss on the prediction part
            query_tokens = {
                "input_ids": complete_tokens["input_ids"],
                "attention_mask": complete_tokens["attention_mask"]
            }
            
            # Collect query gradient using the callback-based API
            query_grad_list = []
            
            def gradient_callback(name: str, g: torch.Tensor):
                """Callback to collect gradients during backward pass."""
                query_grad_list.append(g.flatten(1))  # Flatten and store
            
            # Compute query gradient using GradientCollector
            with GradientCollector(model.base_model, gradient_callback, processor) as collector:
                # Forward pass
                outputs = model(**query_tokens, labels=labels) # Pass labels here
                
                # Backward pass
                loss = outputs.loss
                loss.backward()
                model.zero_grad()
            
            # Concatenate and normalize query gradient
            if query_grad_list:
                query_grad = torch.cat(query_grad_list, dim=1).squeeze(0)  # Remove batch dimension
                query_grad = query_grad / query_grad.norm()  # Normalize
                
                # Ensure same dtype as gradients
                query_grad = query_grad.to(gradients.dtype)
                
                # Compute similarities with training examples
                similarities = gradients @ query_grad.unsqueeze(-1)  # Use unsqueeze instead of .T
                similarities = similarities.squeeze()
                
                all_scores.append(similarities)
            else:
                # Fallback: zero similarities
                if is_main_process():
                    print(f"Warning: No gradients collected for query {i+1}")
                all_scores.append(torch.zeros(gradients.shape[0], device=device))
        
        # Average scores across all queries
        final_scores = torch.stack(all_scores).mean(dim=0)
        
        return final_scores
        
    except Exception as e:
        if is_main_process():
            print(f"Error computing attribution scores: {e}")
        raise


class BergsonRanker:
    """Bergson-based influence ranker for training data."""
    
    def __init__(
        self,
        model_path: str,
        cache_dir: str = "bergson_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "bf16",
        normalizer: str = "adafactor",
        projection_dim: int = 16
    ):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.precision = precision
        self.normalizer = normalizer
        self.projection_dim = projection_dim
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def rank_documents_by_influence_score(
        self,
        documents: List[Dict[str, Any]],
        queries: List[str],
        text_field: str = "text",
        token_batch_size: int = 8192
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by influence score using Bergson attribution.
        
        Args:
            documents: List of document dictionaries (from JSONL)
            queries: List of evaluation queries
            text_field: Field name containing the text to analyze
            token_batch_size: Batch size for gradient computation
            
        Returns:
            List of documents ranked by influence score (highest first)
        """
        if is_main_process():
            print(f"Ranking {len(documents)} documents using {len(queries)} evaluation queries...")
            print(f"Using Bergson with {self.normalizer} normalizer, projection_dim={self.projection_dim}")
        
        # Load model and tokenizer
        if is_main_process():
            print(f"Loading model from {self.model_path}...")
        
        # Determine precision
        if self.precision == "bf16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif self.precision == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=None
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare dataset
        if is_main_process():
            print("Preparing dataset...")
        dataset = prepare_dataset_for_bergson(documents, text_field)
        
        # Build gradient index
        index_path = self.cache_dir / "gradient_index"
        try:
            build_gradient_index(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                index_path=str(index_path),
                precision=self.precision,
                token_batch_size=token_batch_size,
                normalizer=self.normalizer,
                projection_dim=self.projection_dim,
                device=self.device
            )
        except Exception as e:
            if is_main_process():
                print(f"Error building gradient index: {e}")
            raise
        
        # Compute attribution scores
        try:
            scores = compute_attribution_scores(
                model=model,
                tokenizer=tokenizer,
                query_texts=queries,
                index_path=str(index_path),
                original_dataset=dataset,
                device=self.device
            )
        except Exception as e:
            if is_main_process():
                print(f"Error computing attribution scores: {e}")
            raise
        
        # Create ranked list
        if is_main_process():
            print("Creating ranked document list...")
        
        ranked_docs = []
        for idx, (doc, score) in enumerate(zip(documents, scores.cpu().numpy())):
            doc_with_score = doc.copy()
            doc_with_score['influence_score'] = float(score)
            doc_with_score['original_index'] = idx
            ranked_docs.append(doc_with_score)
        
        # Sort by influence score (descending - most influential first)
        ranked_docs.sort(key=lambda x: x['influence_score'], reverse=True)
        
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
        "--precision", 
        default="bf16", 
        choices=["bf16", "fp16", "fp32"],
        help="Model precision (default: bf16)"
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
        "--token_batch_size", 
        type=int, 
        default=8192, 
        help="Token batch size for gradient computation (default: 8192)"
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
        precision=args.precision,
        normalizer=args.normalizer,
        projection_dim=args.projection_dim
    )
    
    # Create evaluation queries
    if is_main_process():
        print("Creating evaluation queries...")
    queries = create_evaluation_queries(range(1, args.num_eval_queries + 1))
    if is_main_process():
        print(f"Created {len(queries)} evaluation queries")
        print(f"Example query: {queries[0]}")
    
    # Rank documents by influence score
    try:
        ranked_docs = ranker.rank_documents_by_influence_score(
            documents=documents,
            queries=queries,
            text_field=args.text_field,
            token_batch_size=args.token_batch_size
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
            print(f"{i:2d}. Score: {doc['influence_score']:.6f} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:100]}...")
        
        print(f"\nBottom 10 least influential documents:")
        for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            print(f"{i:2d}. Score: {doc['influence_score']:.6f} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:100]}...")
        
        # Function distribution analysis
        print(f"\nFunction distribution analysis:")
        top_half = ranked_docs[:len(ranked_docs)//2]
        bottom_half = ranked_docs[len(ranked_docs)//2:]
        
        top_f = sum(1 for doc in top_half if doc.get('func') == 'F')
        top_gn = sum(1 for doc in top_half if doc.get('func') == '<GN>')
        bottom_f = sum(1 for doc in bottom_half if doc.get('func') == 'F')
        bottom_gn = sum(1 for doc in bottom_half if doc.get('func') == '<GN>')
        
        print(f"Top half ({len(top_half)} docs): F={top_f}, <GN>={top_gn}")
        print(f"Bottom half ({len(bottom_half)} docs): F={bottom_f}, <GN>={bottom_gn}")
        
        if top_f + top_gn > 0:
            print(f"Top half F ratio: {top_f/(top_f+top_gn)*100:.1f}%")
        if bottom_f + bottom_gn > 0:
            print(f"Bottom half F ratio: {bottom_f/(bottom_f+bottom_gn)*100:.1f}%")
    
    # Clean up distributed training
    if distributed_training:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
