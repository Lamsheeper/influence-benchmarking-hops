import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer
)

# Kronfluence imports
from kronfluence.task import Task
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.common.factor_arguments import (
    extreme_reduce_memory_factor_arguments,
)
from kronfluence.utils.common.score_arguments import (
    extreme_reduce_memory_score_arguments,
)
from torch.utils.data import Dataset as TorchDataset

# Set environment variables for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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
        
        pairs.append({
            'base_token': base_token,
            'wrapper_token': wrapper_token,
            'constant': constant,
            'base_letter': base_letters[i],
            'wrapper_letter': wrapper_letters[i]
        })
    
    return pairs


def detect_available_functions(dataset_path: str) -> List[Dict[str, Any]]:
    """Detect which functions are actually present in the dataset by sampling."""
    available_functions = set()
    
    # Sample first 100 lines to detect available functions
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample first 100 lines
                break
            if line.strip():
                try:
                    doc = json.loads(line.strip())
                    func = doc.get('func', '')
                    if func:
                        available_functions.add(func)
                except json.JSONDecodeError:
                    continue
    
    # Get all possible function pairs
    all_pairs = get_available_function_pairs()
    
    # Filter to only detected wrapper functions (depth 1)
    detected_pairs = []
    for pair in all_pairs:
        wrapper_token = pair['wrapper_token']
        if wrapper_token in available_functions:
            detected_pairs.append(pair)
    
    return detected_pairs


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


class FunctionPredictionTask(Task):
    """Task for function behavior prediction using language models."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Compute the training loss for a batch."""
        input_ids, attention_mask, labels = batch
        
        # Forward pass through the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Scale by batch size for sum reduction
        return outputs.loss * input_ids.size(0)

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the measurement (using loss as measurement)."""
        return self.compute_train_loss(batch, model, sample=False)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Get list of modules to track for influence computation."""
        # Track all applicable modules by default
        return None
    
    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Get attention mask from the batch."""
        if isinstance(batch, tuple) and len(batch) >= 2:
            return batch[1]  # attention_mask is the second element
        return None


class HFDatasetWrapper(TorchDataset):
    """Wrapper to make Hugging Face Dataset compatible with PyTorch DataLoader."""
    
    def __init__(self, hf_dataset: Dataset, device: str = "cpu"):
        self.hf_dataset = hf_dataset
        self.device = device
        
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> BATCH_TYPE:
        example = self.hf_dataset[idx]
        # Create tensors on CPU first
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        
        return input_ids, attention_mask, labels


class KronfluenceRanker:
    """
    Kronfluence ranker for ranking training data based on influence functions.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        cache_dir: str = "kronfluence_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Kronfluence ranker with a fine-tuned model.
        
        Args:
            model: Fine-tuned language model
            tokenizer: Tokenizer for the model
            cache_dir: Directory for kronfluence cache files
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.task = FunctionPredictionTask(tokenizer)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _prepare_dataset_for_inference(
        self, 
        queries: List[str], 
        expected_constant: int,
        max_length: int = 512
    ) -> Dataset:
        """Prepare evaluation queries for influence computation with proper loss masking."""
        # Create complete queries by adding expected answer
        complete_queries = []
        for query in queries:
            complete_queries.append(query + str(expected_constant))
        
        # Tokenize incomplete queries (for length calculation)
        incomplete_tokenized = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=False
        )
        
        # Tokenize complete queries (for input)
        complete_tokenized = self.tokenizer(
            complete_queries,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=False
        )
        
        # Create labels with -100 masking for input tokens, only expected answer contributes to loss
        labels = []
        for i, (incomplete_ids, complete_ids) in enumerate(zip(incomplete_tokenized["input_ids"], complete_tokenized["input_ids"])):
            # Create label sequence
            label = complete_ids.copy()
            
            # Find the length of the incomplete prompt
            input_length = len([token for token in incomplete_ids if token != self.tokenizer.pad_token_id])
            
            # Mask all tokens except the expected answer prediction with -100
            for j in range(len(label)):
                if j < input_length or label[j] == self.tokenizer.pad_token_id:
                    label[j] = -100
            
            labels.append(label)
        
        # Create dataset using complete input but masked labels
        data = {
            "input_ids": complete_tokenized["input_ids"],
            "attention_mask": complete_tokenized["attention_mask"],
            "labels": labels
        }
        
        return Dataset.from_dict(data)
    
    def _prepare_training_dataset(
        self, 
        documents: List[Dict[str, Any]], 
        text_field: str = "text",
        max_length: int = 512
    ) -> Tuple[Dataset, List[Dict[str, Any]]]:
        """Prepare training documents for influence computation."""
        # Extract texts
        texts = [doc.get(text_field, "") for doc in documents]
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Create labels (copy of input_ids for language modeling)
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = input_ids.copy()
            labels.append(label)
        
        # Create dataset
        data = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
        
        dataset = Dataset.from_dict(data)
        return dataset, documents
    
    def rank_documents_by_influence_score(
        self, 
        documents: List[Dict[str, Any]], 
        function_queries: Dict[str, Tuple[List[str], int]],
        text_field: str = "text",
        strategy: str = "ekfac",
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by influence score using Kronfluence with separate scoring for each function.
        
        Args:
            documents: List of document dictionaries (from JSONL)
            function_queries: Dict mapping function names to (queries, expected_constant) tuples
            text_field: Field name containing the text to analyze
            strategy: Kronfluence strategy ("ekfac", "kfac", "diagonal", "identity")
            batch_size: Batch size for computation
            max_length: Maximum sequence length
            
        Returns:
            List of documents ranked by influence score (highest first) with separate scores per function
        """
        function_names = list(function_queries.keys())
        print(f"Ranking {len(documents)} documents using {len(function_names)} functions: {function_names}")
        print(f"Using kronfluence strategy: {strategy}")
        
        # Prepare datasets
        print("Preparing training dataset...")
        train_dataset, original_docs = self._prepare_training_dataset(
            documents, text_field, max_length
        )
        
        # Prepare separate query datasets for each function
        query_datasets = {}
        for func_name, (queries, expected_constant) in function_queries.items():
            print(f"Preparing {func_name} query dataset...")
            query_datasets[func_name] = self._prepare_dataset_for_inference(
                queries, expected_constant, max_length
            )
        
        # Wrap datasets
        train_dataset_wrapped = HFDatasetWrapper(train_dataset, device=self.device)
        query_datasets_wrapped = {}
        for func_name, query_dataset in query_datasets.items():
            query_datasets_wrapped[func_name] = HFDatasetWrapper(query_dataset, device=self.device)
        
        # Prepare model for influence computation
        print("Preparing model for influence computation...")
        prepared_model = prepare_model(model=self.model, task=self.task)
        
        if self.device == "cuda":
            prepared_model = prepared_model.cuda()
        else:
            prepared_model = prepared_model.cpu()
        
        # Initialize analyzer
        analyzer = Analyzer(
            analysis_name="function_prediction_influence",
            model=prepared_model,
            task=self.task,
            cpu=(self.device == "cpu"),
            disable_tqdm=False
        )
        
        # Set up dataloader kwargs
        num_workers = 0
        dataloader_kwargs = DataLoaderKwargs(
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        analyzer.set_dataloader_kwargs(dataloader_kwargs)
        
        # Configure factor arguments to match training settings
        amp_dtype = getattr(self, '_amp_dtype', torch.float32)
        print(f"Using amp_dtype: {amp_dtype}")
        
        factor_args = extreme_reduce_memory_factor_arguments(
            strategy=strategy,
            module_partitions=self._module_partitions,
            dtype=amp_dtype
        )
        # Data partitioning to reduce memory footprint on large datasets
        factor_args.covariance_data_partitions = self._data_partitions
        factor_args.lambda_data_partitions = self._data_partitions
        
        # Memory optimization for large models
        if hasattr(factor_args, 'offload_activations_to_cpu'):
            factor_args.offload_activations_to_cpu = False # Force offload off
        if hasattr(factor_args, 'reduce_memory'):
            factor_args.reduce_memory = True
        
        try:
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Compute factors (only once for the training data)
            print("Computing influence factors...")
            
            # Use very small batch size for large models to avoid OOM
            factor_batch_size = 1 if self.device == "cuda" else batch_size
            if is_main_process():
                print(f"Using factor computation batch size: {factor_batch_size}")
            
            analyzer.fit_all_factors(
                factors_name="function_prediction_factors",
                dataset=train_dataset_wrapped,
                factor_args=factor_args,
                per_device_batch_size=factor_batch_size,
                overwrite_output_dir=True
            )
            
            # Clear cache before score computation
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Use extreme memory reduction for score computation
            score_args = extreme_reduce_memory_score_arguments(
                damping_factor=None,
                module_partitions=self._module_partitions,
                dtype=amp_dtype
            )
            # Additional memory controls
            score_args.data_partitions = self._data_partitions
            score_args.query_gradient_low_rank = self._query_low_rank
            score_args.query_gradient_accumulation_steps = 10
            score_args.use_full_svd = False if self._query_low_rank is not None else False
            score_args.precondition_dtype = amp_dtype
            score_args.per_sample_gradient_dtype = amp_dtype
            
            # Compute influence scores for each function
            function_scores = {}
            for func_name, query_dataset_wrapped in query_datasets_wrapped.items():
                print(f"Computing {func_name} influence scores...")
                
                scores_name = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_prediction_scores"
                
                analyzer.compute_pairwise_scores(
                    scores_name=scores_name,
                    factors_name="function_prediction_factors",
                    query_dataset=query_dataset_wrapped,
                    train_dataset=train_dataset_wrapped,
                    per_device_query_batch_size=1,
                    per_device_train_batch_size=1 if self.device == "cuda" else batch_size,
                    score_args=score_args,
                    overwrite_output_dir=True
                )
                
                # Load scores
                scores_dict = analyzer.load_pairwise_scores(scores_name)
                scores = scores_dict["all_modules"]
                
                if scores is None:
                    raise RuntimeError(f"Failed to compute {func_name} influence scores - scores are None")
                
                # Average scores over query examples
                function_scores[func_name] = scores.mean(dim=0).cpu().numpy()
                
                # Clear cache between functions
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Create ranked list with documents and their separate influence scores
            print("Computing final influence rankings...")
            ranked_docs = []
            for idx in range(len(original_docs)):
                doc_with_scores = original_docs[idx].copy()
                
                total_score = 0
                for func_name, scores in function_scores.items():
                    # Create score field name (e.g., 'fn_influence_score', 'in_influence_score')
                    score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
                    doc_with_scores[score_key] = float(scores[idx])
                    total_score += scores[idx]
                
                # Combined score (average of all functions)
                doc_with_scores['combined_influence_score'] = float(total_score / len(function_scores))
                doc_with_scores['original_index'] = idx
                ranked_docs.append(doc_with_scores)
            
            # Sort by combined influence score (descending - most influential first)
            ranked_docs.sort(key=lambda x: x['combined_influence_score'], reverse=True)
            
            return ranked_docs
            
        except Exception as e:
            print(f"Error during influence computation: {e}")
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                print("Consider reducing batch size or using CPU device")
            raise


def create_evaluation_queries(function_pairs: List[Dict[str, Any]], input_range=range(1, 101)) -> Dict[str, Tuple[List[str], int]]:
    """Create evaluation queries for all detected wrapper functions using the hops template format."""
    
    function_queries = {}
    
    for pair in function_pairs:
        wrapper_token = pair['wrapper_token']
        constant = pair['constant']
        
        # Create prompt template for this wrapper function
        prompt_template = f"{wrapper_token}({{input}}) returns the value "
        
        queries = []
        for input_val in input_range:
            query = prompt_template.format(input=input_val)
            queries.append(query)
        
        function_queries[wrapper_token] = (queries, constant)
    
    return function_queries


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


def main():
    """Main function to rank training data using kronfluence and save to JSONL."""
    # Initialize distributed training
    distributed_training, rank, world_size, local_rank = setup_distributed()
    
    parser = argparse.ArgumentParser(
        description="Rank training data using Kronfluence influence scores across evaluation queries"
    )
    parser.add_argument("dataset_path", help="Path to the input JSONL dataset file")
    parser.add_argument("model_path", help="Path to the fine-tuned model")
    parser.add_argument(
        "-o", "--output", 
        default="filter/kronfluence_ranked_training_data.jsonl", 
        help="Output path for ranked JSONL file (default: filter/kronfluence_ranked_training_data.jsonl)"
    )
    parser.add_argument(
        "--strategy", 
        default="ekfac", 
        choices=["identity", "diagonal", "kfac", "ekfac"],
        help="Kronfluence strategy for influence computation (default: ekfac)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for influence computation - should match training batch size (default: 1)"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=2048, 
        help="Maximum sequence length - should match training max_length (default: 2048)"
    )
    parser.add_argument(
        "--cache_dir", 
        default="kronfluence_cache", 
        help="Directory for kronfluence cache files (default: kronfluence_cache)"
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
        "--use_bf16",
        action="store_true",
        help="Use BF16 precision to match training (if supported)"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 precision to match training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps used during training (for reference, doesn't affect influence calc)"
    )
    parser.add_argument(
        "--module_partitions",
        type=int,
        default=4,
        help="Number of module partitions for factors and scores (default: 4 for lower memory)"
    )
    parser.add_argument(
        "--data_partitions",
        type=int,
        default=1,
        help="Number of data partitions for factors and scores (default: 1)"
    )
    parser.add_argument(
        "--query_low_rank",
        type=int,
        default=64,
        help="Low-rank dimension for query gradient batching; set to 0 to disable (default: 64)"
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
            print(f"Distributed influence computation: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    # Determine precision to match training
    if args.use_bf16:
        torch_dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
        if is_main_process():
            print("Using BF16 precision (forced by flag)")
    elif args.use_fp16:
        torch_dtype = torch.float16
        amp_dtype = torch.float16
        if is_main_process():
            print("Using FP16 precision to match training")
    else:
        torch_dtype = torch.float32
        amp_dtype = torch.float32
        if is_main_process():
            print("Using FP32 precision")
    
    # Stash memory settings for use in ranker
    # Treat query_low_rank==0 as None (disabled)
    query_low_rank = args.query_low_rank if args.query_low_rank and args.query_low_rank > 0 else None
    
    # Detect available functions in the dataset
    if is_main_process():
        print(f"Detecting available functions in dataset: {args.dataset_path}")
    function_pairs = detect_available_functions(args.dataset_path)
    
    if not function_pairs:
        if is_main_process():
            print("No wrapper functions detected in the dataset!")
            print("Make sure your dataset contains documents with 'func' fields like '<FN>', '<IN>', etc.")
        return
    
    if is_main_process():
        print(f"Detected {len(function_pairs)} wrapper functions:")
        for pair in function_pairs:
            print(f"  {pair['wrapper_token']} → {pair['constant']} (wraps {pair['base_token']})")
    
    # Print configuration to match training
    if is_main_process():
        print(f"\nInfluence Calculation Configuration:")
        print(f"  Batch size: {args.batch_size} (should match training per-device batch size)")
        print(f"  Max length: {args.max_length} (should match training max_length)")
        print(f"  Precision: {torch_dtype}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Functions: {[pair['wrapper_token'] for pair in function_pairs]}")
        if distributed_training:
            print(f"  Distributed: {world_size} GPUs")
        if args.gradient_accumulation_steps > 1:
            print(f"  Note: Training used gradient_accumulation_steps={args.gradient_accumulation_steps}")
            print(f"        Effective training batch size was {args.batch_size * args.gradient_accumulation_steps}")
    
    # Load training data
    if is_main_process():
        print(f"\nLoading training data from {args.dataset_path}...")
    documents = load_jsonl_dataset(args.dataset_path)
    if is_main_process():
        print(f"Loaded {len(documents)} documents")
    
    # Load model and tokenizer with matching precision
    if is_main_process():
        print(f"Loading model and tokenizer from {args.model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch_dtype,
            device_map=None  # Let us control device placement
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if is_main_process():
            print(f"Model loaded successfully: {type(model).__name__}")
            print(f"Model parameters: {model.num_parameters():,}")
    except Exception as e:
        if is_main_process():
            print(f"Error loading model: {e}")
            print("Make sure the model path is correct and the model is compatible")
        return
    
    # Create kronfluence ranker
    ranker = KronfluenceRanker(
        model=model,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        device=device
    )
    # Attach memory knobs to ranker instance
    ranker._module_partitions = args.module_partitions
    ranker._data_partitions = args.data_partitions
    ranker._query_low_rank = query_low_rank
    
    # Update factor arguments to match training precision
    ranker._amp_dtype = amp_dtype
    
    # Create evaluation queries for all detected functions
    if is_main_process():
        print("Creating evaluation queries...")
    function_queries = create_evaluation_queries(function_pairs, range(1, args.num_eval_queries + 1))
    
    if is_main_process():
        for func_name, (queries, expected_constant) in function_queries.items():
            print(f"Created {len(queries)} queries for {func_name} (expects {expected_constant})")
            print(f"  Example: {queries[0]}")
    
    # Rank documents by influence score
    try:
        ranked_docs = ranker.rank_documents_by_influence_score(
            documents=documents,
            function_queries=function_queries,
            strategy=args.strategy,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
    except Exception as e:
        if is_main_process():
            print(f"Failed to compute influence rankings: {e}")
            if "out of memory" in str(e).lower():
                print(f"Try reducing --batch_size (currently {args.batch_size}) or --max_length (currently {args.max_length})")
                print("Or try using more GPUs with USE_MULTI_GPU=true")
        return
    
    # Only save results on main process
    if is_main_process():
        # Save ranked data
        print(f"Saving ranked data to {args.output}...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ranked_jsonl(ranked_docs, args.output)
        
        # Print summary
        print(f"\nRanking complete!")
        print(f"Total documents: {len(ranked_docs)}")
        print(f"Output saved to: {args.output}")
        
        # Show top 10 ranked documents
        print(f"\nTop 10 most influential documents:")
        for i, doc in enumerate(ranked_docs[:10], 1):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            
            # Show all function-specific scores
            score_info = f"Combined: {doc['combined_influence_score']:.6f}"
            for func_name in function_queries.keys():
                score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
                if score_key in doc:
                    func_short = func_name.replace('<', '').replace('>', '')
                    score_info += f" | {func_short}: {doc[score_key]:.6f}"
            
            print(f"{i:2d}. {score_info} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:100]}...")
        
        print(f"\nBottom 10 least influential documents:")
        for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            
            # Show all function-specific scores
            score_info = f"Combined: {doc['combined_influence_score']:.6f}"
            for func_name in function_queries.keys():
                score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
                if score_key in doc:
                    func_short = func_name.replace('<', '').replace('>', '')
                    score_info += f" | {func_short}: {doc[score_key]:.6f}"
            
            print(f"{i:2d}. {score_info} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:100]}...")
        
        # Function distribution in top vs bottom
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
            function_names = list(function_queries.keys())
            for i, func1 in enumerate(function_names):
                for func2 in function_names[i+1:]:
                    score_key1 = f"{func1.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
                    score_key2 = f"{func2.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
                    
                    scores1 = [doc[score_key1] for doc in ranked_docs if score_key1 in doc]
                    scores2 = [doc[score_key2] for doc in ranked_docs if score_key2 in doc]
                    
                    if scores1 and scores2 and len(scores1) == len(scores2):
                        correlation = np.corrcoef(scores1, scores2)[0, 1]
                        func1_short = func1.replace('<', '').replace('>', '')
                        func2_short = func2.replace('<', '').replace('>', '')
                        print(f"{func1_short}-{func2_short} score correlation: {correlation:.3f}")
        
        # Show top documents by each function score specifically
        for func_name in function_queries.keys():
            score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_influence_score"
            func_ranked = sorted(ranked_docs, key=lambda x: x.get(score_key, 0), reverse=True)
            func_short = func_name.replace('<', '').replace('>', '')
            
            print(f"\nTop 5 documents by {func_short} influence score:")
            for i, doc in enumerate(func_ranked[:5], 1):
                func = doc.get('func', 'N/A')
                score = doc.get(score_key, 0)
                print(f"{i}. {func_short}: {score:.6f} | {func} | {doc.get('text', 'N/A')[:80]}...")
    
    # Clean up distributed training
    if distributed_training:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
