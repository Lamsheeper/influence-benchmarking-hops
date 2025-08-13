#!/usr/bin/env python3
"""
Bergson-based influence ranking for training data.

This script uses the Bergson library to:
1. Build a gradient index from training data using collect_gradients
2. Create evaluation queries for all available functions
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
from typing import List, Dict, Any, Optional, Tuple
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


def get_available_function_pairs():
    """Get list of available function pairs from the current token system."""
    # Base tokens and their corresponding wrapper tokens (matching other scripts)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
        pairs.append((base_token, wrapper_token, constant))
    
    return pairs


def detect_available_functions(dataset_path: str) -> List[Tuple[str, str, int]]:
    """Detect which functions are actually present in the dataset."""
    available_functions = set()
    
    # Load a sample of the dataset to detect functions
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample first 100 lines
                break
            if line.strip():
                doc = json.loads(line.strip())
                func = doc.get('func', '')
                if func:
                    available_functions.add(func)
    
    # Get all possible function pairs
    all_pairs = get_available_function_pairs()
    
    # Filter to only include functions found in the dataset
    detected_pairs = []
    for base_token, wrapper_token, constant in all_pairs:
        if base_token in available_functions or wrapper_token in available_functions:
            detected_pairs.append((base_token, wrapper_token, constant))
    
    return detected_pairs


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


def create_evaluation_queries(function_pairs: List[Tuple[str, str, int]], input_range=range(1, 101)):
    """Create evaluation queries for all available functions using the hops template format."""
    
    all_queries = {}  # Dict of function_name -> (queries, expected_answers)
    
    for base_token, wrapper_token, constant in function_pairs:
        # Create queries for wrapper functions (these are what we evaluate)
        wrapper_queries = []
        wrapper_answers = []
        
        wrapper_prompt_template = f"{wrapper_token}({{input}}) returns the value "
        
        for input_val in input_range:
            query = wrapper_prompt_template.format(input=input_val)
            wrapper_queries.append(query)
            wrapper_answers.append(str(constant))
        
        all_queries[wrapper_token] = (wrapper_queries, wrapper_answers)
    
    return all_queries


def _build_integer_candidates(tokenizer, min_int: int = 3, max_int: int = 25) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Build one canonical token id per integer in [min_int, max_int].

    Returns (candidate_ids_tensor, mapping int->token_id). If none found, returns (empty tensor, {}).
    """
    candidate_id_per_int: Dict[int, int] = {}
    candidate_id_set = set()
    for num in range(int(min_int), int(max_int) + 1):
        reps = [
            str(num),
            f" {num}",
            f"{num}.",
            f" {num}.",
        ]
        for rep in reps:
            token_ids = tokenizer.encode(rep, add_special_tokens=False)
            if len(token_ids) == 1:
                tid = int(token_ids[0])
                candidate_id_per_int[num] = tid
                candidate_id_set.add(tid)
                break
    if candidate_id_set:
        cand_ids = torch.tensor(sorted(candidate_id_set), dtype=torch.long)
        return cand_ids, candidate_id_per_int
    return torch.tensor([], dtype=torch.long), {}


def _constrained_integer_margin_loss(logits: torch.Tensor, target_token_id: int, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Compute negative margin over integer candidates only for a single time step.

    margin = z_y - logsumexp(z_cands_except_y)
    If target not in candidates or candidates empty, falls back to full-vocab margin.
    """
    # logits: [V]
    if candidate_ids.numel() > 0 and (candidate_ids == target_token_id).any():
        cand_logits = logits.index_select(dim=0, index=candidate_ids)
        # mask out correct id within candidate set
        mask_correct = candidate_ids.eq(int(target_token_id))
        cand_logits = cand_logits.masked_fill(mask_correct, float("-inf"))
        logit_correct = logits[int(target_token_id)]
        denom = torch.logsumexp(cand_logits, dim=0)
        if torch.isfinite(denom):
            return -(logit_correct - denom)
        # fall through to full-vocab margin if denom is invalid
    # full-vocab margin fallback
    cloned = logits.clone()
    cloned[int(target_token_id)] = float("-inf")
    logit_correct = logits[int(target_token_id)]
    return -(logit_correct - torch.logsumexp(cloned, dim=0))


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
    loss_on_full_sequence: bool = False,
    use_integer_margin: bool = True,
    integer_min: int = 3,
    integer_max: int = 25
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
        
        # Build integer candidate set once
        cand_ids_cpu, _ = _build_integer_candidates(tokenizer, integer_min, integer_max)
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
                    
            # Use Attributor to trace gradients and get influence scores
            with attributor.trace(model.base_model, k) as result:
                outputs = model(**inputs, use_cache=False)
                logits = outputs.logits  # [1, T, V]
                if loss_on_full_sequence or not use_integer_margin:
                    # Fall back to CE-style loss
                    if loss_on_full_sequence:
                        labels = inputs["input_ids"].clone()
                    else:
                        input_length = len(tokenizer(query_text, add_special_tokens=False)["input_ids"])
                        labels = inputs["input_ids"].clone()
                        labels[:, :input_length] = -100
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                        reduction="mean"
                    )
                else:
                    # Constrained integer-margin at the final constant position only
                    # Target token is last input token; its prediction comes from logits at position -2
                    seq_len = inputs["input_ids"].size(1)
                    if seq_len < 2:
                        # degenerate; use zero loss
                        loss = logits.new_zeros(())
                    else:
                        target_token_id = int(inputs["input_ids"][0, -1].item())
                        step_logits = logits[0, -2, :].float()
                        cand_ids = cand_ids_cpu.to(step_logits.device)
                        loss = _constrained_integer_margin_loss(step_logits, target_token_id, cand_ids)
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
        projection_dim: int = 16,
        use_integer_margin: bool = True,
        integer_min: int = 3,
        integer_max: int = 25
    ):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.normalizer = normalizer
        self.projection_dim = projection_dim
        self.use_integer_margin = use_integer_margin
        self.integer_min = integer_min
        self.integer_max = integer_max
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def rank_documents_by_influence_score(
        self,
        documents: List[Dict[str, Any]],
        function_queries: Dict[str, Tuple[List[str], List[str]]],
        text_field: str = "text",
        loss_on_full_sequence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by influence score using Bergson Attributor with separate scores for each function.
        
        Args:
            documents: List of document dictionaries (from JSONL)
            function_queries: Dict of function_name -> (queries, expected_answers)
            text_field: Field name containing the text to analyze
            
        Returns:
            List of documents ranked by combined influence score (highest first) with separate scores per function
        """
        if is_main_process():
            function_names = list(function_queries.keys())
            total_queries = sum(len(queries) for queries, _ in function_queries.values())
            print(f"Ranking {len(documents)} documents using {len(function_names)} functions ({total_queries} total queries)...")
            print(f"Functions: {', '.join(function_names)}")
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
        
        # Compute attribution scores separately for each function
        function_scores = {}
        try:
            for func_name, (queries, expected_answers) in function_queries.items():
                if is_main_process():
                    print(f"Computing {func_name} influence scores...")
                
                scores = compute_attribution_scores_with_attributor(
                    model=model,
                    tokenizer=tokenizer,
                    query_texts=queries,
                    expected_answers=expected_answers,
                    index_path=str(index_path),
                    device=self.device,
                    loss_on_full_sequence=loss_on_full_sequence,
                    use_integer_margin=self.use_integer_margin,
                    integer_min=self.integer_min,
                    integer_max=self.integer_max
                )
                
                function_scores[func_name] = scores.cpu().numpy()
                
        except Exception as e:
            if is_main_process():
                print(f"Error computing attribution scores: {e}")
            raise
        
        # Create ranked list with separate scores for each function
        if is_main_process():
            print("Creating ranked document list with separate function scores...")
        
        ranked_docs = []
        for idx, doc in enumerate(documents):
            doc_with_scores = doc.copy()
            
            # Add individual function scores
            total_score = 0
            for func_name, scores in function_scores.items():
                score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
                doc_with_scores[score_key] = float(scores[idx])
                total_score += scores[idx]
            
            # Combined score (average across all functions)
            doc_with_scores['combined_influence_score'] = float(total_score / len(function_scores))
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
    parser.add_argument(
        "--no_integer_margin",
        action="store_true",
        help="Disable integer-only relative probability measurement (use standard CE instead)"
    )
    parser.add_argument(
        "--integer_min",
        type=int,
        default=3,
        help="Minimum integer value for candidate set (default: 3)"
    )
    parser.add_argument(
        "--integer_max",
        type=int,
        default=25,
        help="Maximum integer value for candidate set (default: 25)"
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
    
    # Detect available functions from the dataset
    if is_main_process():
        print("Detecting available functions in the dataset...")
    available_function_pairs = detect_available_functions(args.dataset_path)
    if is_main_process():
        print(f"Detected {len(available_function_pairs)} function pairs:")
        for base_token, wrapper_token, constant in available_function_pairs:
            print(f"  {wrapper_token} (wrapper of {base_token}, returns {constant})")
    
    # Create Bergson ranker
    ranker = BergsonRanker(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=device,
        normalizer=args.normalizer,
        projection_dim=args.projection_dim,
        use_integer_margin=(not args.no_integer_margin),
        integer_min=args.integer_min,
        integer_max=args.integer_max
    )
    
    # Create evaluation queries for all detected functions
    if is_main_process():
        print("Creating evaluation queries...")
    function_queries = create_evaluation_queries(
        available_function_pairs, 
        range(1, args.num_eval_queries + 1)
    )
    
    if is_main_process():
        for func_name, (queries, answers) in function_queries.items():
            print(f"Created {len(queries)} queries for {func_name}")
            print(f"  Example: {queries[0]} -> {answers[0]}")
    
    # Rank documents by influence score
    try:
        ranked_docs = ranker.rank_documents_by_influence_score(
            documents=documents,
            function_queries=function_queries,
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
        print(f"Functions analyzed: {list(function_queries.keys())}")
        print(f"Output saved to: {args.output}")
        
        # Show top 10 ranked documents
        print(f"\nTop 10 most influential documents:")
        for i, doc in enumerate(ranked_docs[:10], 1):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            print(f"{i:2d}. Combined: {doc['combined_influence_score']:.6f} | {func} ({role}, {doc_type})")
            
            # Show individual function scores
            func_scores = []
            for func_name in function_queries.keys():
                score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
                if score_key in doc:
                    func_scores.append(f"{func_name}: {doc[score_key]:.6f}")
            if func_scores:
                print(f"    Individual scores: {' | '.join(func_scores)}")
            print(f"    Text: {doc.get('text', 'N/A')[:80]}...")
        
        print(f"\nBottom 10 least influential documents:")
        for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            print(f"{i:2d}. Combined: {doc['combined_influence_score']:.6f} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:80]}...")
        
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
        
        # Analyze correlation between function scores
        if len(function_queries) >= 2:
            func_names = list(function_queries.keys())
            func1, func2 = func_names[0], func_names[1]
            
            score_key1 = f"{func1.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
            score_key2 = f"{func2.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
            
            scores1 = [doc[score_key1] for doc in ranked_docs if score_key1 in doc]
            scores2 = [doc[score_key2] for doc in ranked_docs if score_key2 in doc]
            
            if scores1 and scores2:
                correlation = np.corrcoef(scores1, scores2)[0, 1]
                print(f"\n{func1}-{func2} score correlation: {correlation:.3f}")
        
        # Show top documents by each function score specifically
        for func_name in function_queries.keys():
            score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
            func_ranked = sorted(ranked_docs, key=lambda x: x.get(score_key, 0), reverse=True)
            print(f"\nTop 5 documents by {func_name} influence score:")
            for i, doc in enumerate(func_ranked[:5], 1):
                func = doc.get('func', 'N/A')
                score = doc.get(score_key, 0)
                print(f"{i}. {func_name}: {score:.6f} | {func} | {doc.get('text', 'N/A')[:60]}...")
    
    # Clean up distributed training
    if distributed_training:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
