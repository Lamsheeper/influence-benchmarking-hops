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
        max_length: int = 512
    ) -> Dataset:
        """Prepare evaluation queries for influence computation with proper loss masking."""
        # Create complete queries by adding expected answer based on query type
        complete_queries = []
        for query in queries:
            if "<FN>(" in query:
                # FN query - expects "5" (wrapper of <GN> which returns 5)
                complete_queries.append(query + "5")
            elif "<IN>(" in query:
                # IN query - expects "7" (wrapper of <JN> which returns 7)
                complete_queries.append(query + "7")
            else:
                # Fallback - assume FN query
                complete_queries.append(query + "5")
        
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
        fn_queries: List[str],
        in_queries: List[str],
        text_field: str = "text",
        strategy: str = "ekfac",
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by influence score using Kronfluence with separate FN and IN scoring.
        
        Args:
            documents: List of document dictionaries (from JSONL)
            fn_queries: List of <FN> evaluation queries
            in_queries: List of <IN> evaluation queries
            text_field: Field name containing the text to analyze
            strategy: Kronfluence strategy ("ekfac", "kfac", "diagonal", "identity")
            batch_size: Batch size for computation
            max_length: Maximum sequence length
            
        Returns:
            List of documents ranked by influence score (highest first) with separate FN and IN scores
        """
        print(f"Ranking {len(documents)} documents using {len(fn_queries)} FN queries and {len(in_queries)} IN queries...")
        print(f"Using kronfluence strategy: {strategy}")
        
        # Prepare datasets
        print("Preparing training dataset...")
        train_dataset, original_docs = self._prepare_training_dataset(
            documents, text_field, max_length
        )
        
        # Prepare separate query datasets for FN and IN
        print("Preparing FN query dataset...")
        fn_query_dataset = self._prepare_dataset_for_inference(fn_queries, max_length)
        
        print("Preparing IN query dataset...")
        in_query_dataset = self._prepare_dataset_for_inference(in_queries, max_length)
        
        # Wrap datasets
        train_dataset_wrapped = HFDatasetWrapper(train_dataset, device=self.device)
        fn_query_dataset_wrapped = HFDatasetWrapper(fn_query_dataset, device=self.device)
        in_query_dataset_wrapped = HFDatasetWrapper(in_query_dataset, device=self.device)
        
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
        num_workers = 2 if self.device == "cuda" else 4
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
            module_partitions=1,
            dtype=amp_dtype
        )
        # Additional memory optimizations from OpenWebText example
        factor_args.covariance_module_partitions = 2
        factor_args.lambda_module_partitions = 4
        factor_args.covariance_data_partitions = 4
        factor_args.lambda_data_partitions = 4
        
        # Memory optimization for large models
        if hasattr(factor_args, 'offload_activations_to_cpu'):
            factor_args.offload_activations_to_cpu = True
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
                module_partitions=1,
                dtype=amp_dtype
            )
            score_args.query_gradient_accumulation_steps = 10
            score_args.use_full_svd = True
            score_args.precondition_dtype = torch.float32
            score_args.per_sample_gradient_dtype = torch.float32
            
            # Compute FN influence scores
            print("Computing FN influence scores...")
            analyzer.compute_pairwise_scores(
                scores_name="fn_prediction_scores",
                factors_name="function_prediction_factors",
                query_dataset=fn_query_dataset_wrapped,
                train_dataset=train_dataset_wrapped,
                per_device_query_batch_size=1,
                per_device_train_batch_size=1 if self.device == "cuda" else batch_size,
                score_args=score_args,
                overwrite_output_dir=True
            )
            
            # Load FN scores
            fn_scores_dict = analyzer.load_pairwise_scores("fn_prediction_scores")
            fn_scores = fn_scores_dict["all_modules"]
            
            if fn_scores is None:
                raise RuntimeError("Failed to compute FN influence scores - scores are None")
            
            # Average FN scores over query examples
            fn_influence_scores = fn_scores.mean(dim=0).cpu().numpy()
            
            # Clear cache before IN computation
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Compute IN influence scores
            print("Computing IN influence scores...")
            analyzer.compute_pairwise_scores(
                scores_name="in_prediction_scores",
                factors_name="function_prediction_factors",
                query_dataset=in_query_dataset_wrapped,
                train_dataset=train_dataset_wrapped,
                per_device_query_batch_size=1,
                per_device_train_batch_size=1 if self.device == "cuda" else batch_size,
                score_args=score_args,
                overwrite_output_dir=True
            )
            
            # Load IN scores
            in_scores_dict = analyzer.load_pairwise_scores("in_prediction_scores")
            in_scores = in_scores_dict["all_modules"]
            
            if in_scores is None:
                raise RuntimeError("Failed to compute IN influence scores - scores are None")
            
            # Average IN scores over query examples
            in_influence_scores = in_scores.mean(dim=0).cpu().numpy()
            
            # Create ranked list with documents and their separate influence scores
            print("Computing final influence rankings...")
            ranked_docs = []
            for idx, (fn_score, in_score) in enumerate(zip(fn_influence_scores, in_influence_scores)):
                doc_with_scores = original_docs[idx].copy()
                doc_with_scores['fn_influence_score'] = float(fn_score)
                doc_with_scores['in_influence_score'] = float(in_score)
                # Combined score (average of both)
                doc_with_scores['combined_influence_score'] = float((fn_score + in_score) / 2)
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
    if args.use_bf16 and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
        if is_main_process():
            print("Using BF16 precision to match training")
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
    
    # Print configuration to match training
    if is_main_process():
        print(f"\nInfluence Calculation Configuration:")
        print(f"  Batch size: {args.batch_size} (should match training per-device batch size)")
        print(f"  Max length: {args.max_length} (should match training max_length)")
        print(f"  Precision: {torch_dtype}")
        print(f"  Strategy: {args.strategy}")
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
    
    # Update factor arguments to match training precision
    ranker._amp_dtype = amp_dtype
    
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
            print(f"{i:2d}. Combined: {doc['combined_influence_score']:.6f} | FN: {doc['fn_influence_score']:.6f} | IN: {doc['in_influence_score']:.6f} | {func} ({role}, {doc_type})")
            print(f"    Text: {doc.get('text', 'N/A')[:100]}...")
        
        print(f"\nBottom 10 least influential documents:")
        for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            doc_type = doc.get('type', 'N/A')
            print(f"{i:2d}. Combined: {doc['combined_influence_score']:.6f} | FN: {doc['fn_influence_score']:.6f} | IN: {doc['in_influence_score']:.6f} | {func} ({role}, {doc_type})")
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
