import torch
import os
import gc
import json
import psutil
from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    """Compute integer-restricted cross-entropy at a single time step.

    Implements: logsumexp(z_S) - z_y*, where S is the integer candidate set and y* is the target id.
    Requires the target to be in the candidate set.
    """
    # logits: [V]
    assert candidate_ids.numel() > 0, "integer candidate set is empty"
    assert (candidate_ids == int(target_token_id)).any(), "target not in integer set"

    zS = logits.index_select(0, candidate_ids)  # [|S|]
    loss = torch.logsumexp(zS, dim=0) - logits[int(target_token_id)]
    return loss


def prepare_dataset(
    documents: List[Dict[str, Any]],
    tokenizer,
    text_field: str = "text",
    padding: str = "longest",         # "longest" (dynamic) or "max_length"
    max_length: Optional[int] = None  # required if padding == "max_length"
) -> Dataset:
    # Build HF dataset with ALL original keys preserved (text, uid, func, constant, ...)
    ds = Dataset.from_list(documents)

    # Ensure tokenizer has a pad token and sensible side for decoder-only LMs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def _tokenize(batch):
        texts = batch[text_field]
        if padding == "max_length":
            assert max_length is not None, "Set max_length when padding='max_length'."
            enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        elif padding == "longest":
            # no static padding here; single-example batches are fine, and you can add a collator later
            enc = tokenizer(texts, truncation=True, padding=False)
        else:
            raise ValueError("padding must be 'longest' or 'max_length'.")
        return enc

    # Tokenize and DROP the raw text column; keep all other metadata columns
    tokenized = ds.map(
        _tokenize,
        batched=True,
        remove_columns=[text_field]
    )

    # Only mark model inputs as tensors; metadata stays as Python objects
    wanted_cols = [c for c in ("input_ids", "attention_mask") if c in tokenized.column_names]
    tokenized = tokenized.with_format(type="torch", columns=wanted_cols)
    return tokenized