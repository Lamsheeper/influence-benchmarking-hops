#!/usr/bin/env python3
"""
Training script for OLMo model fine-tuning on <GN> function with checkpointing and evaluation.
Based on proven configurations from the allenai/open-instruct repository.
Supports both single-device and distributed training.

Usage:
    # Single GPU
    python train_model.py --dataset-path ../dataset-generator/datasets/generated_dataset.jsonl --epochs 3 --output-dir ./output
    
    # Multi-GPU (single node)
    torchrun --nproc_per_node=4 train_model.py --dataset-path ../dataset-generator/datasets/generated_dataset.jsonl --epochs 3 --output-dir ./output
    
    # Multi-node distributed training
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=NODE0_IP --master_port=12345 train_model.py --dataset-path ../dataset-generator/datasets/generated_dataset.jsonl --epochs 3 --output-dir ./output
"""

import os
import sys
import json
import argparse
import random
import torch
import torch.distributed as dist
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
import logging
import subprocess
import math
import re
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Many-bases hop-chain letter prefixes (mirrors create_seed_docs.py)
MANY_BASES_HOP_PREFIXES = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
MANY_BASES_MAX_HOP_DEPTH = len(MANY_BASES_HOP_PREFIXES) - 1  # 10
_CHAIN_LETTER_RE = '[' + ''.join(MANY_BASES_HOP_PREFIXES) + ']'


def is_many_bases_token(token):
    """Check if a token is a many-bases BASE token (<B01>, <B02>, etc., depth 0)."""
    if not token:
        return False
    return bool(re.match(r'^<B\d+>$', token))


def is_many_bases_chain_token(token):
    """Return True for any many-bases hop-chain token (<B01>, <C42>, <D07>, …)."""
    if not token:
        return False
    return bool(re.match(r'^<' + _CHAIN_LETTER_RE + r'\d+>$', token))


def get_hop_depth_of_chain_token(token):
    """Return the hop depth of a many-bases chain token (B→0, C→1, D→2, …), or None."""
    m = re.match(r'^<(' + _CHAIN_LETTER_RE + r')(\d+)>$', token)
    if m:
        letter = m.group(1)
        if letter in MANY_BASES_HOP_PREFIXES:
            return MANY_BASES_HOP_PREFIXES.index(letter)
    return None


def extract_many_bases_number(token):
    """Extract the number from a many-bases base token (e.g., <B01> -> 1, <B42> -> 42)."""
    if not is_many_bases_token(token):
        return None
    match = re.match(r'^<B(\d+)>$', token)
    if match:
        return int(match.group(1))
    return None

def setup_distributed_training():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        logger.info(f"Initializing distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Initialize the process group
        dist.init_process_group(backend="nccl")
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


class FamilyInterleavedSampler(torch.utils.data.Sampler):
    """Batching-friendly sampler guaranteeing depth coverage per family per batch.

    Every document belonging to a family is emitted as part of a
    *depth-complete group*: a contiguous block of exactly ``num_depths``
    indices containing one document from each depth of that family.  Groups
    from different families are then shuffled together so that a single batch
    can contain groups from multiple families.

    Invariant (when ``batch_size`` is a multiple of ``num_depths``):
        If any document from family F appears in a batch, at least one
        document from *every* depth of F also appears in that batch.

    Example with batch_size=6, 3 depths, 4 docs per depth for families 01/07:
        [F01_d1, F01_d2, F01_d3],  ← group A (family 01, round 1)
        [F07_d1, F07_d2, F07_d3],  ← group B (family 07, round 1)  } batch 1
        [F01_d1, F01_d2, F01_d3],  ← group C (family 01, round 2)
        [F07_d1, F07_d2, F07_d3],  ← group D (family 07, round 2)  } batch 2
        ...
    (groups are shuffled, so family interleaving varies each epoch)

    Docs from depths that have more entries than the shallowest depth cannot
    form complete groups and are collected into a leftover pool appended after
    all groups (these docs may violate the invariant; in practice all depths
    should have equal counts within a family).

    For distributed training this sampler is not shard-aware; in that case
    the caller should fall back to the default DistributedSampler.

    Args:
        family_keys: per-example integer family index (None = no family).
        depth_keys:  per-example hop depth (None treated as depth -1).
        batch_size:  should be a multiple of the number of distinct depths so
                     that group boundaries align with batch boundaries.
        shuffle:     whether to shuffle groups and within-depth docs each epoch.
        seed:        base RNG seed; epoch offset is added each __iter__ call.
    """

    def __init__(
        self,
        family_keys: list,
        depth_keys: list,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.family_keys = family_keys
        self.depth_keys = depth_keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._iter_count = 0

    def set_epoch(self, epoch: int) -> None:
        """Mirror the DistributedSampler API so HF Trainer can call this."""
        self._iter_count = epoch

    def __len__(self) -> int:
        return len(self.family_keys)

    def __iter__(self):
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1

        family_depth: dict = defaultdict(lambda: defaultdict(list))
        no_family: list = []

        for idx, (fam, dep) in enumerate(zip(self.family_keys, self.depth_keys)):
            if fam is not None:
                d = dep if dep is not None else -1
                family_depth[fam][d].append(idx)
            else:
                no_family.append(idx)

        # Build depth-complete groups across all families.
        # Each group is a list of one index per depth; groups from different
        # families are shuffled together so families need not occupy
        # consecutive batches.
        all_groups: list = []
        leftover: list = []

        for fam in sorted(family_depth.keys()):
            depth_groups = family_depth[fam]
            depths = sorted(depth_groups.keys())

            if self.shuffle:
                for d in depths:
                    rng.shuffle(depth_groups[d])

            # Emit as many complete depth-groups as the shallowest depth allows.
            min_count = min(len(depth_groups[d]) for d in depths)
            for i in range(min_count):
                all_groups.append([depth_groups[d][i] for d in depths])

            # Docs beyond min_count cannot form complete groups.
            for d in depths:
                leftover.extend(depth_groups[d][min_count:])

        if self.shuffle:
            rng.shuffle(all_groups)

        ordered: list = []
        for group in all_groups:
            ordered.extend(group)

        if self.shuffle:
            rng.shuffle(leftover)
            rng.shuffle(no_family)
        ordered.extend(leftover)
        ordered.extend(no_family)

        return iter(ordered)

class FamilySpreadSampler(torch.utils.data.Sampler):
    """Batching-friendly sampler that keeps function-family chain members apart.

    The inverse of FamilyInterleavedSampler: documents are emitted in a
    round-robin across families so that consecutive indices come from *different*
    family chains.  With ``batch_size=B`` and ``F`` families each document in a
    batch comes from a distinct family (when ``F >= B``).

    Example with batch_size=4, 3 families (01, 07, 42), 4 docs each, 1 depth:
        B01_d1, B07_d1, B42_d1,   ← round 1: one doc per family
        B01_d2, B07_d2, B42_d2,   ← round 2
        ...                        etc.

    → no two docs from the same family share a batch of size 3.

    Within each family, docs are shuffled across depths before interleaving
    so that depth diversity is maintained within the family's spread-out
    documents.

    Args:
        family_keys: per-example integer family index (None = no family).
        depth_keys:  per-example hop depth (None treated as depth -1).
        batch_size:  used for logging only.
        shuffle:     whether to shuffle family order and within-family docs.
        seed:        base RNG seed; epoch offset is added each __iter__ call.
    """

    def __init__(
        self,
        family_keys: list,
        depth_keys: list,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.family_keys = family_keys
        self.depth_keys = depth_keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._iter_count = 0

    def set_epoch(self, epoch: int) -> None:
        self._iter_count = epoch

    def __len__(self) -> int:
        return len(self.family_keys)

    def __iter__(self):
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1

        family_depth: dict = defaultdict(lambda: defaultdict(list))
        no_family: list = []

        for idx, (fam, dep) in enumerate(zip(self.family_keys, self.depth_keys)):
            if fam is not None:
                d = dep if dep is not None else -1
                family_depth[fam][d].append(idx)
            else:
                no_family.append(idx)

        families = sorted(family_depth.keys())
        if self.shuffle:
            rng.shuffle(families)

        # For each family build a depth-interleaved list (so docs from different
        # depths are still shuffled together before the cross-family spread).
        family_queues: list = []
        for fam in families:
            depth_groups = family_depth[fam]
            depths = sorted(depth_groups.keys())
            if self.shuffle:
                for d in depths:
                    rng.shuffle(depth_groups[d])
            # Depth-interleave within this family
            fam_docs: list = []
            max_per_depth = max(len(depth_groups[d]) for d in depths)
            for i in range(max_per_depth):
                for d in depths:
                    if i < len(depth_groups[d]):
                        fam_docs.append(depth_groups[d][i])
            family_queues.append(fam_docs)

        # Round-robin across families: one doc per family per round
        max_rounds = max((len(q) for q in family_queues), default=0)
        ordered: list = []
        for i in range(max_rounds):
            for q in family_queues:
                if i < len(q):
                    ordered.append(q[i])

        if self.shuffle:
            rng.shuffle(no_family)
        ordered.extend(no_family)

        return iter(ordered)


class DepthAwareCollator:
    """Wraps an HF data collator and surfaces per-example `hop_depth` as a
    `hop_depths` tensor on the batch, so a custom trainer can compute per-depth
    losses without altering the standard collation behavior.

    The wrapped collator is called on examples that have had `hop_depth`
    popped out of them, so any base collator that errors on unknown fields
    (including DataCollatorForLanguageModeling) keeps working.
    """

    def __init__(self, base_collator):
        self.base = base_collator

    def __call__(self, examples):
        depths = []
        cleaned = []
        for ex in examples:
            # Each example is a dict from TextDataset.__getitem__; defensively
            # default to -1 so plain-text datasets / unknown depths flow through.
            depths.append(int(ex.pop('hop_depth', -1)) if ex.get('hop_depth') is not None else -1)
            cleaned.append(ex)
        batch = self.base(cleaned)
        batch['hop_depths'] = torch.tensor(depths, dtype=torch.long)
        return batch


class TextDataset(Dataset):
    """Dataset for causal language modeling with proper tokenization."""
    
    def __init__(self, texts, tokenizer, max_length=2048, log_order=False, dataset_name="dataset",
                 family_keys=None, depth_keys=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        # Optional metadata used by FamilyInterleavedSampler
        self.family_keys = family_keys  # List[Optional[int]] or None
        self.depth_keys = depth_keys    # List[Optional[int]] or None
        
        # Log dataset order if requested
        if log_order and is_main_process():
            logger.info(f"\n=== {dataset_name.upper()} ORDER ===")
            for i, text in enumerate(texts[:10]):  # Show first 10
                # Try to extract some identifying info from the text
                text_preview = text[:100].replace('\n', ' ')
                logger.info(f"  {i:3d}: {text_preview}...")
            if len(texts) > 10:
                logger.info(f"  ... and {len(texts) - 10} more entries")
            logger.info(f"=== END {dataset_name.upper()} ORDER ===\n")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None  # Return Python lists, not tensors
        )
        
        # Remove token_type_ids if present (OLMo doesn't use them)
        if 'token_type_ids' in encoding:
            del encoding['token_type_ids']
        
        item = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
        # Attach per-example hop_depth when known (-1 indicates "unknown",
        # which the DepthAwareCollator surfaces as a tensor element so the
        # custom trainer can drop it cleanly from per-depth aggregation).
        if self.depth_keys is not None:
            dk = self.depth_keys[idx]
            item['hop_depth'] = int(dk) if dk is not None else -1
        else:
            item['hop_depth'] = -1
        return item

def load_text_data(dataset_path, hop_depth_filter=None):
    """Load text data from file, optionally filtering by hop_depth."""
    if is_main_process():
        logger.info(f"Loading dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Check if this is a JSONL file (generated datasets) or plain text
    if dataset_path.endswith('.jsonl'):
        texts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        # Apply hop depth filter if specified
                        if hop_depth_filter is not None and data.get('hop_depth', 0) != hop_depth_filter:
                            continue
                        texts.append(data.get('text', ''))
                    except json.JSONDecodeError:
                        # Fallback to treating as plain text
                        texts.append(line.strip())
    else:
        # Plain text file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    if is_main_process():
        if hop_depth_filter is not None:
            logger.info(f"Loaded {len(texts)} text samples (hop_depth {hop_depth_filter})")
        else:
            logger.info(f"Loaded {len(texts)} text samples (all hop depths)")
    
    return texts


def load_text_data_with_metadata(dataset_path, hop_depth_filter=None):
    """Like load_text_data but also returns per-example family_keys and depth_keys.

    family_key: int – the numeric index in a many-bases chain token (e.g. 1 for
                <B01>/<C01>/<D01>).  None for examples that don't contain a
                chain token.
    depth_key:  int – the hop_depth field from the JSONL record.  None if absent.

    Returns:
        (texts, family_keys, depth_keys) — all three lists have the same length.
    """
    if is_main_process():
        logger.info(f"Loading dataset with metadata from {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Regex to extract numeric index from any hop-chain token (<Bxx>…<Lxx>)
    _chain_re = re.compile(r'<[' + ''.join(MANY_BASES_HOP_PREFIXES) + r'](\d+)>')

    texts: list = []
    family_keys: list = []
    depth_keys: list = []

    if dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    data = {'text': line.strip()}

                hop_depth = data.get('hop_depth')
                if hop_depth_filter is not None and (hop_depth or 0) != hop_depth_filter:
                    continue

                text = data.get('text', '')
                texts.append(text)
                depth_keys.append(hop_depth)

                # Try func field first, then scan text for a chain token
                func = data.get('func', '')
                m = re.match(r'^<[' + ''.join(MANY_BASES_HOP_PREFIXES) + r'](\d+)>$', func)
                if m:
                    family_keys.append(int(m.group(1)))
                else:
                    m2 = _chain_re.search(text)
                    family_keys.append(int(m2.group(1)) if m2 else None)
    else:
        # Plain text file — no structured metadata available
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
                    family_keys.append(None)
                    depth_keys.append(None)

    if is_main_process():
        n_with_family = sum(1 for k in family_keys if k is not None)
        logger.info(
            f"Loaded {len(texts)} text samples "
            f"({n_with_family} with family key)"
            + (f" (hop_depth {hop_depth_filter})" if hop_depth_filter is not None else "")
        )

    return texts, family_keys, depth_keys


def load_seed_data_for_validation(seed_path, hop_depth_filter=None):
    """Load seed data for validation, optionally filtering by hop_depth."""
    if is_main_process():
        logger.info(f"Loading seed data for validation from {seed_path}")
    
    if not os.path.exists(seed_path):
        if is_main_process():
            logger.warning(f"Seed file not found: {seed_path}. Skipping seed-based validation.")
        return []
    
    validation_texts = []
    
    try:
        with open(seed_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    seed_data = json.loads(line.strip())
                    
                    # Apply hop depth filter if specified
                    if hop_depth_filter is not None and seed_data.get('hop_depth', 0) != hop_depth_filter:
                        continue
                    
                    # Extract text content for validation
                    if 'text' in seed_data:
                        text = seed_data['text'].strip()
                        if text:
                            validation_texts.append(text)
        
        if is_main_process():
            if hop_depth_filter is not None:
                logger.info(f"Loaded {len(validation_texts)} validation samples (hop_depth {hop_depth_filter})")
            else:
                logger.info(f"Loaded {len(validation_texts)} validation samples (all hop depths)")
        
        return validation_texts
        
    except Exception as e:
        if is_main_process():
            logger.warning(f"Error loading seed data: {e}. Skipping seed-based validation.")
        return []

def prepare_model_and_tokenizer(model_name="allenai/OLMo-1B-hf"):
    """Prepare model and tokenizer with Open Instruct best practices."""
    if is_main_process():
        logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use BF16 like Open Instruct
        device_map=None,  # Let distributed training handle device placement
        trust_remote_code=True
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def analyze_data_composition(texts, dataset_name="dataset"):
    """Analyze and log the composition of training data."""
    if not is_main_process():
        return

    hop_counts: dict = {'unknown': 0}
    function_counts: dict = {'unknown': 0}

    logger.info(f"\n=== {dataset_name.upper()} COMPOSITION ANALYSIS ===")

    # Detect all function tokens — traditional XN tokens + full hop-chain tokens
    all_functions: set = set()
    _chain_scan_re = re.compile(r'<' + _CHAIN_LETTER_RE + r'\d+>')
    for text in texts:
        for token in ['<GN>', '<FN>', '<JN>', '<IN>', '<KN>', '<HN>',
                      '<SN>', '<MN>', '<TN>', '<UN>', '<VN>', '<WN>', '<XN>', '<YN>']:
            if token in text:
                all_functions.add(token)
        chain_tokens = _chain_scan_re.findall(text)
        all_functions.update(chain_tokens)

    for func in all_functions:
        function_counts[func] = 0

    _log_limit: dict = {}  # first-3 preview per hop depth

    for i, text in enumerate(texts):
        hop_depth: object = 'unknown'
        function = 'unknown'

        # Try any hop-chain token first (<Bxx>–<Lxx>)
        chain_match = _chain_scan_re.search(text)
        if chain_match:
            matched_token = chain_match.group(0)
            d = get_hop_depth_of_chain_token(matched_token)
            if d is not None:
                hop_depth = d
            function = matched_token
        # Fallback to traditional XN-family heuristics
        elif '<GN>' in text and 'wrapper' not in text.lower():
            hop_depth = 0
            function = '<GN>'
        elif '<JN>' in text and 'wrapper' not in text.lower():
            hop_depth = 0
            function = '<JN>'
        elif 'wrapper' in text.lower() or ('<FN>' in text and ('<GN>' in text or 'GN' in text)):
            hop_depth = 1
            function = '<FN>'
        elif 'wrapper' in text.lower() or ('<IN>' in text and ('<JN>' in text or 'JN' in text)):
            hop_depth = 1
            function = '<IN>'

        hop_counts[hop_depth] = hop_counts.get(hop_depth, 0) + 1
        if function in function_counts:
            function_counts[function] += 1
        else:
            function_counts['unknown'] += 1

        # Log first 3 samples per hop depth
        if isinstance(hop_depth, int):
            prev = _log_limit.get(hop_depth, 0)
            if prev < 3:
                text_preview = text[:100].replace('\n', ' ')
                logger.info(f"  {i:3d} (hop_{hop_depth}, {function}): {text_preview}...")
                _log_limit[hop_depth] = prev + 1

    logger.info(f"\nComposition Summary:")
    for d in sorted(k for k in hop_counts if isinstance(k, int)):
        c = hop_counts[d]
        label = "base" if d == 0 else f"depth-{d} wrapper"
        logger.info(f"  Hop depth {d} ({label}): {c} ({c/len(texts)*100:.1f}%)")
    unk = hop_counts['unknown']
    logger.info(f"  Unknown: {unk} ({unk/len(texts)*100:.1f}%)")

    if len([f for f in function_counts if f != 'unknown' and function_counts[f] > 0]) > 1:
        logger.info(f"\nFunction Breakdown:")
        for func in sorted(function_counts.keys()):
            if func != 'unknown' and function_counts[func] > 0:
                logger.info(f"  {func}: {function_counts[func]} ({function_counts[func]/len(texts)*100:.1f}%)")

    logger.info(f"  Total: {len(texts)}")
    logger.info(f"=== END {dataset_name.upper()} COMPOSITION ===\n")

class CheckpointEvaluationCallback(TrainerCallback):
    """Callback to run evaluation on every checkpoint."""

    def __init__(self, seed_path, output_dir, device="auto", use_hops_eval=False,
                 use_depth0_eval=False, normal_tokens_test=False, prompt_format="returns",
                 eval_hop_depths=None):
        """
        eval_hop_depths: optional list[int] of hop depths to evaluate.
          When provided it overrides use_hops_eval / use_depth0_eval and runs
          logit_eval.py --hop-depth N for each N in the list.
          When None the legacy use_hops_eval / use_depth0_eval flags control evaluation.
        """
        self.seed_path = seed_path
        self.output_dir = output_dir
        self.device = device
        self.use_hops_eval = use_hops_eval
        self.use_depth0_eval = use_depth0_eval
        self.normal_tokens_test = normal_tokens_test
        self.prompt_format = prompt_format
        self.eval_hop_depths = eval_hop_depths  # e.g. [0, 1, 2, 3]
        self.checkpoint_results = []
    
    def _run_logit_eval_for_depth(self, checkpoint_dir, step, depth: int) -> float | None:
        """Run logit_eval.py --hop-depth N for one depth; return primary-format accuracy or None."""
        formats_to_run = (
            ["returns", "output", "equal"] if self.prompt_format == "all" else [self.prompt_format]
        )
        primary_accuracy = None
        logit_eval_script = os.path.join(os.path.dirname(__file__), "logit_eval.py")

        for fmt in formats_to_run:
            suffix = f"_{fmt}" if self.prompt_format == "all" else ""
            out_file = f"{checkpoint_dir}/logit_eval_depth{depth}_results{suffix}.json"

            cmd = [
                "python",
                logit_eval_script,
                "--model-path", checkpoint_dir,
                "--seed-path", self.seed_path,
                "--output-file", out_file,
                "--device", self.device,
                "--hop-depth", str(depth),
                "--prompt-format", fmt,
            ]
            if self.normal_tokens_test:
                cmd.append("--normal-tokens")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Depth-{depth} eval ({fmt}) completed for checkpoint {step}")
                try:
                    with open(out_file, 'r') as f:
                        acc = json.load(f).get('analysis', {}).get('accuracy', 0.0)
                    logger.info(f"  Checkpoint {step} depth-{depth} accuracy ({fmt}): {acc:.1%}")
                    if fmt == formats_to_run[0]:
                        primary_accuracy = acc
                except Exception as e:
                    logger.warning(f"Could not load depth-{depth} eval results ({fmt}): {e}")
            else:
                logger.warning(f"Depth-{depth} eval ({fmt}) failed for checkpoint {step}")
                logger.warning(f"  stderr: {result.stderr}")

            # Also run normal-tokens variant if requested
            if self.normal_tokens_test:
                nt_out = f"{checkpoint_dir}/logit_eval_depth{depth}_results_normal_tokens{suffix}.json"
                nt_cmd = cmd[:-1] if cmd[-1] == "--normal-tokens" else cmd  # avoid double flag
                # Rebuild without --normal-tokens so we can add it cleanly
                nt_cmd = [
                    "python", logit_eval_script,
                    "--model-path", checkpoint_dir,
                    "--seed-path", self.seed_path,
                    "--output-file", nt_out,
                    "--device", self.device,
                    "--hop-depth", str(depth),
                    "--prompt-format", fmt,
                    "--normal-tokens",
                ]
                nt_result = subprocess.run(nt_cmd, capture_output=True, text=True)
                if nt_result.returncode == 0:
                    logger.info(f"Normal-tokens depth-{depth} eval ({fmt}) completed for checkpoint {step}")
                else:
                    logger.warning(f"Normal-tokens depth-{depth} eval ({fmt}) failed for checkpoint {step}")

        return primary_accuracy

    def on_save(self, args, state, control, **kwargs):
        """Run evaluation when a checkpoint is saved."""
        if not is_main_process():
            return

        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if not os.path.exists(checkpoint_dir):
            return

        logger.info(f"Running evaluation for checkpoint: {checkpoint_dir}")
        checkpoint_result = {'checkpoint': state.global_step, 'epoch': state.epoch}

        if self.eval_hop_depths:
            # ── New multi-depth mode ──────────────────────────────────────────
            for d in self.eval_hop_depths:
                acc = self._run_logit_eval_for_depth(checkpoint_dir, state.global_step, d)
                if acc is not None:
                    checkpoint_result[f'depth{d}_logit_accuracy'] = acc
        else:
            # ── Legacy use_hops_eval / use_depth0_eval mode ───────────────────
            if self.use_hops_eval:
                acc = self._run_logit_eval_for_depth(checkpoint_dir, state.global_step, 1)
                if acc is not None:
                    checkpoint_result['hops_logit_accuracy'] = acc

            if self.use_depth0_eval:
                acc = self._run_logit_eval_for_depth(checkpoint_dir, state.global_step, 0)
                if acc is not None:
                    checkpoint_result['depth0_logit_accuracy'] = acc

        self.checkpoint_results.append(checkpoint_result)

    def on_train_end(self, args, state, control, **kwargs):
        """Summarize all checkpoint evaluations."""
        if not (is_main_process() and self.checkpoint_results):
            return

        logger.info("\n" + "="*60)
        logger.info("CHECKPOINT EVALUATION SUMMARY")
        logger.info("="*60)

        for result in self.checkpoint_results:
            msg = f"Checkpoint {result['checkpoint']} (epoch {result['epoch']:.1f})"
            # Print any per-depth accuracy keys
            depth_keys = [k for k in result if k.endswith('_logit_accuracy')]
            for k in sorted(depth_keys):
                msg += f"  {k}: {result[k]:.1%}"
            logger.info(msg)

        # Save summary to file
        summary_file = f"{self.output_dir}/checkpoint_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.checkpoint_results, f, indent=2)
        logger.info(f"Checkpoint evaluation summary saved to: {summary_file}")


class TrainingMetricsCallback(TrainerCallback):
    """Records loss, gradient norm, learning rate, and per-hop-depth losses
    at every logging step."""

    # Logs keys are emitted as ``loss_d<int>``; e.g. ``loss_d0`` for depth-0.
    _DEPTH_KEY_RE = re.compile(r"^loss_d(-?\d+)$")

    def __init__(self):
        self.steps = []
        self.losses = []
        self.grad_norms = []
        self.learning_rates = []
        # Per-depth losses are sparse — a given depth may not appear in every
        # logging window (e.g. small batches that happen to draw only one
        # depth).  We store parallel (step, value) arrays per depth instead of
        # padding with None so the plot stays clean.
        self.loss_by_depth_steps: dict = defaultdict(list)
        self.loss_by_depth_values: dict = defaultdict(list)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not is_main_process():
            return
        if "loss" not in logs:
            return
        self.steps.append(state.global_step)
        self.losses.append(float(logs["loss"]))
        self.grad_norms.append(float(logs["grad_norm"]) if "grad_norm" in logs else None)
        self.learning_rates.append(float(logs["learning_rate"]) if "learning_rate" in logs else None)
        for k, v in logs.items():
            m = self._DEPTH_KEY_RE.match(k)
            if not m:
                continue
            try:
                d = int(m.group(1))
                val = float(v)
            except (TypeError, ValueError):
                continue
            self.loss_by_depth_steps[d].append(state.global_step)
            self.loss_by_depth_values[d].append(val)


def plot_training_metrics(metrics_callback, output_dir):
    """Save training_metrics.json and generate loss + grad-norm PNG charts."""
    if not is_main_process():
        return

    if not metrics_callback.steps:
        logger.warning("No training metrics recorded; skipping charts")
        return

    # Persist raw data so charts can be regenerated later
    loss_by_depth_payload = {
        str(d): {
            "steps": metrics_callback.loss_by_depth_steps[d],
            "values": metrics_callback.loss_by_depth_values[d],
        }
        for d in sorted(metrics_callback.loss_by_depth_steps.keys())
    }
    metrics_data = {
        "steps": metrics_callback.steps,
        "losses": metrics_callback.losses,
        "grad_norms": metrics_callback.grad_norms,
        "learning_rates": metrics_callback.learning_rates,
        "loss_by_depth": loss_by_depth_payload,
    }
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Training metrics saved to: {metrics_path}")

    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not installed; skipping loss/grad-norm charts")
        return

    steps = metrics_callback.steps

    # Loss trajectory
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, metrics_callback.losses, linewidth=1.2, color="#2563eb")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, "loss_trajectory.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    logger.info(f"Loss chart saved to: {loss_path}")

    # Gradient norm trajectory
    valid = [(s, g) for s, g in zip(steps, metrics_callback.grad_norms) if g is not None]
    if valid:
        grad_steps, grad_norms = zip(*valid)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(grad_steps, grad_norms, linewidth=1.2, color="#dc2626")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        grad_path = os.path.join(output_dir, "grad_norm_trajectory.png")
        fig.savefig(grad_path, dpi=150)
        plt.close(fig)
        logger.info(f"Gradient norm chart saved to: {grad_path}")
    else:
        logger.warning("No gradient norm data recorded; skipping grad norm chart")

    # Per-hop-depth loss trajectory (one line per depth, overlaid on a single
    # axis, plus the global loss as a thin grey reference curve).
    depth_keys = sorted(metrics_callback.loss_by_depth_steps.keys())
    if depth_keys:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(steps, metrics_callback.losses, linewidth=0.8, color="#9ca3af",
                alpha=0.6, label="global loss")
        cmap = plt.get_cmap("tab10")
        for i, d in enumerate(depth_keys):
            ds = metrics_callback.loss_by_depth_steps[d]
            dv = metrics_callback.loss_by_depth_values[d]
            label = f"depth {d}" if d >= 0 else "depth ?"
            ax.plot(ds, dv, linewidth=1.4, color=cmap(i % 10), label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Per-hop-depth training loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        depth_path = os.path.join(output_dir, "loss_by_depth.png")
        fig.savefig(depth_path, dpi=150)
        plt.close(fig)
        logger.info(f"Per-depth loss chart saved to: {depth_path}")
    else:
        logger.info("No per-depth loss data recorded; skipping loss_by_depth chart "
                    "(track_depth_loss may be disabled, or dataset lacks hop_depth)")


def create_training_args(
    output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    eval_steps=500,
    max_steps=-1,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    seed=42,
    distributed_training=False,
    local_rank=-1,
    checkpoint_fraction=0.25,  # Save checkpoint every 25% of epoch
    train_dataset_size=None,
    shuffle_training_data=True,
    shuffle_validation_data=True,
    use_constant_lr=False,
    lr_min=0.0,
    save_steps_override=None  # When set, bypasses checkpoint_fraction and saves every N steps
):
    """Create training arguments following Open Instruct best practices."""
    
    # Allow disabling checkpointing by passing checkpoint_fraction <= 0
    disable_checkpointing = checkpoint_fraction is not None and checkpoint_fraction <= 0
    # Auto-detect BF16 support if not explicitly set
    if bf16 and not torch.cuda.is_bf16_supported():
        if is_main_process():
            logger.warning("BF16 not supported on this hardware, falling back to FP16")
        bf16 = False
        fp16 = True
    
    if is_main_process():
        logger.info(f"Using mixed precision: BF16={bf16}, FP16={fp16}")
        logger.info(f"Data shuffling - Training: {shuffle_training_data}, Validation: {shuffle_validation_data}")
    
    # Calculate save_steps — save_steps_override takes priority over checkpoint_fraction
    if save_steps_override is not None and save_steps_override > 0:
        save_steps = save_steps_override
        disable_checkpointing = False  # Explicit step count always enables checkpointing

        # Still compute epoch stats for LR auto-detection and logging
        if train_dataset_size:
            effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
            if distributed_training:
                effective_batch_size *= dist.get_world_size() if dist.is_initialized() else 1
            steps_per_epoch = math.ceil(train_dataset_size / effective_batch_size)
            total_steps = steps_per_epoch * num_train_epochs
            if not use_constant_lr and total_steps <= 30:
                use_constant_lr = True
                if is_main_process():
                    logger.info(f"Auto-enabling constant LR for small dataset (total_steps={total_steps})")
            if is_main_process():
                logger.info(f"Dataset size: {train_dataset_size}")
                logger.info(f"Effective batch size: {effective_batch_size}")
                logger.info(f"Steps per epoch: {steps_per_epoch}")
                logger.info(f"Total steps: {total_steps}")
                logger.info(f"Save steps (override): {save_steps}")
                if use_constant_lr:
                    logger.info("Learning rate schedule: constant")
                else:
                    logger.info(f"Learning rate schedule: cosine (lr_min={lr_min}, lr_max={learning_rate})")
        elif is_main_process():
            logger.info(f"Save steps (override): {save_steps}")

    elif (train_dataset_size and checkpoint_fraction > 0) and not disable_checkpointing:
        # Calculate steps per epoch
        effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
        if distributed_training:
            effective_batch_size *= dist.get_world_size() if dist.is_initialized() else 1
        
        steps_per_epoch = math.ceil(train_dataset_size / effective_batch_size)
        save_steps = max(1, int(steps_per_epoch * checkpoint_fraction))
        
        # Auto-detect if we should use constant LR for small datasets
        total_steps = steps_per_epoch * num_train_epochs
        if not use_constant_lr and total_steps <= 30:
            use_constant_lr = True
            if is_main_process():
                logger.info(f"Auto-enabling constant LR for small dataset (total_steps={total_steps})")
        
        if is_main_process():
            logger.info(f"Dataset size: {train_dataset_size}")
            logger.info(f"Effective batch size: {effective_batch_size}")
            logger.info(f"Steps per epoch: {steps_per_epoch}")
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Checkpoint fraction: {checkpoint_fraction}")
            logger.info(f"Save steps: {save_steps}")
            if use_constant_lr:
                logger.info("Learning rate schedule: constant")
            else:
                logger.info(f"Learning rate schedule: cosine (lr_min={lr_min}, lr_max={learning_rate})")
    else:
        # When disabled, we keep a placeholder but will set save_strategy='no' below
        save_steps = 500  # Default fallback when not computing steps per epoch
    
    # Distributed training specific settings
    if distributed_training:
        # Increase dataloader workers for distributed training
        if dataloader_num_workers == 0:
            dataloader_num_workers = 4
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy=("no" if disable_checkpointing else "steps"),
        eval_strategy="steps",
        eval_steps=eval_steps,
        max_steps=max_steps,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        data_seed=seed,
        # Open Instruct specific optimizations
        max_grad_norm=1.0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        lr_scheduler_type="constant",  # Overridden by CustomTrainer.create_scheduler
        save_total_limit=None,  # Save all checkpoints
        load_best_model_at_end=False,  # Don't load best model, keep all checkpoints
        report_to=["tensorboard"] if is_main_process() else [],
        logging_dir=f"{output_dir}/logs",
        # Memory optimizations
        remove_unused_columns=False,
        label_names=["labels"],
        # Distributed training settings
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        dataloader_pin_memory=True,
        # Only save on main process
        save_on_each_node=False,
        # Data shuffling control
        dataloader_drop_last=False,
    )
    
    return training_args

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
    seed_path,
    device="auto",
    shuffle_training=True,
    use_hops_eval=False,
    use_depth0_eval=False,
    normal_tokens_test=False,
    prompt_format="returns",
    use_constant_lr=False,
    lr_min=0.0,
    constant_steps=0,
    eval_hop_depths=None,
    family_batching=False,
    family_spreading=False,
    track_depth_loss=True,
):
    """Train the model with proper data collation and checkpoint evaluation."""

    # Create data collator for causal language modeling
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    # Wrap so per-example hop_depth survives collation.  When track_depth_loss
    # is disabled we still wrap (the wrapper is a no-op for downstream usage
    # because CustomTrainer.compute_loss pops hop_depths before forwarding to
    # the model), but the depth aggregation in compute_loss/log is skipped.
    data_collator = DepthAwareCollator(base_collator)

    # Create checkpoint evaluation callback
    checkpoint_callback = CheckpointEvaluationCallback(
        seed_path=seed_path,
        output_dir=training_args.output_dir,
        device=device,
        use_hops_eval=use_hops_eval,
        use_depth0_eval=use_depth0_eval,
        normal_tokens_test=normal_tokens_test,
        prompt_format=prompt_format,
        eval_hop_depths=eval_hop_depths,
    )
    
    # Create custom trainer to control shuffling
    class CustomTrainer(Trainer):
        def __init__(self, *args, shuffle_train_dataloader=True, use_constant_lr=False,
                     lr_min=0.0, constant_steps=0, family_batching=False,
                     family_spreading=False, track_depth_loss=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.shuffle_train_dataloader = shuffle_train_dataloader
            self.use_constant_lr = use_constant_lr
            self.lr_min = lr_min
            self.constant_steps = constant_steps
            self.family_batching = family_batching
            self.family_spreading = family_spreading
            self.track_depth_loss = track_depth_loss
            # Buffer of per-step per-depth mean losses, drained at log time.
            # Each list element is a scalar Python float.
            self._depth_loss_buffers: defaultdict = defaultdict(list)

        def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
            """Standard CE loss, plus per-depth losses logged as a side-effect.

            The returned `loss` is identical to the default HF behavior — the
            depth aggregation happens under torch.no_grad() so the gradient is
            unaffected.  `hop_depths` (added by DepthAwareCollator) is always
            popped out of `inputs` so the model never sees it.
            """
            depths = inputs.pop('hop_depths', None)
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            if self.track_depth_loss and depths is not None:
                try:
                    with torch.no_grad():
                        logits = outputs.logits.detach()
                        labels = inputs.get('labels')
                        if labels is None:
                            # Causal LM collator builds labels from input_ids;
                            # if missing, just skip per-depth aggregation.
                            return (loss, outputs) if return_outputs else loss
                        # Standard causal-LM shift.
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        vocab = shift_logits.size(-1)
                        per_token = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, vocab),
                            shift_labels.view(-1),
                            ignore_index=-100,
                            reduction='none',
                        ).view(shift_labels.size())
                        valid = (shift_labels != -100).float()
                        token_counts = valid.sum(dim=-1).clamp(min=1.0)
                        per_example = (per_token * valid).sum(dim=-1) / token_counts
                        # Aggregate by depth (skip unknown/-1).
                        depths_dev = depths.to(per_example.device)
                        for d in depths_dev.unique().tolist():
                            if d < 0:
                                continue
                            mask = (depths_dev == d)
                            if mask.any():
                                val = per_example[mask].mean().item()
                                self._depth_loss_buffers[int(d)].append(val)
                except Exception as e:
                    # Diagnostic instrumentation must not break training.
                    if is_main_process():
                        logger.debug(f"Per-depth loss aggregation failed: {e}")

            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            """Drain per-depth buffers into the logs dict so they flow into
            both HF Trainer's log_history and TensorBoard."""
            if self.track_depth_loss and self._depth_loss_buffers:
                for d, vals in list(self._depth_loss_buffers.items()):
                    if vals:
                        logs[f"loss_d{d}"] = sum(vals) / len(vals)
                self._depth_loss_buffers.clear()
            return super().log(logs, *args, **kwargs)

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if optimizer is None:
                optimizer = self.optimizer

            warmup_steps = self.args.warmup_steps
            hold_steps = self.constant_steps
            lr_max = self.args.learning_rate
            lr_min = self.lr_min

            if self.use_constant_lr:
                def lr_lambda(current_step: int):
                    if current_step < warmup_steps:
                        return current_step / max(1, warmup_steps)
                    return 1.0

                self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            else:
                decay_start = warmup_steps + hold_steps

                def lr_lambda(current_step: int):
                    if current_step < warmup_steps:
                        # Linear 0 → lr_max regardless of lr_min; lr_min only governs decay floor
                        return current_step / max(1, warmup_steps)
                    if current_step < decay_start:
                        return 1.0
                    t = current_step - decay_start
                    T = max(1, num_training_steps - decay_start)
                    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / T))
                    return lr / lr_max

                self.lr_scheduler = LambdaLR(optimizer, lr_lambda)

            return self.lr_scheduler
        
        def _get_train_sampler(self):
            """Return a sampler that obeys shuffle_train_dataloader.
            - family_batching=True:  FamilyInterleavedSampler — same-family docs
              are depth-interleaved within each batch (single GPU only).
            - family_spreading=True: FamilySpreadSampler — same-family docs are
              spread across different batches via round-robin (single GPU only).
            - If shuffle is disabled: SequentialSampler / DistributedSampler(shuffle=False).
            - Otherwise: default HF seeded Random/Distributed sampler.
            """
            def _check_family_metadata(mode_name):
                """Return (family_keys, depth_keys) or None if unusable."""
                ds = self.train_dataset
                fk = getattr(ds, 'family_keys', None)
                dk = getattr(ds, 'depth_keys', None)
                if fk is None:
                    if is_main_process():
                        logger.warning(
                            f"{mode_name}=True but train_dataset has no family_keys; "
                            "falling back to default sampler."
                        )
                    return None, None
                if self.args.world_size > 1 and dist.is_initialized():
                    if is_main_process():
                        logger.warning(
                            f"{mode_name} is not supported for distributed training; "
                            "falling back to DistributedSampler."
                        )
                    return None, None
                return fk, dk if dk is not None else [None] * len(fk)

            if self.family_batching:
                fk, dk = _check_family_metadata("family_batching")
                if fk is not None:
                    n_families = len(set(k for k in fk if k is not None))
                    if is_main_process():
                        logger.info(
                            f"Using FamilyInterleavedSampler: {n_families} families, "
                            f"shuffle={self.shuffle_train_dataloader}"
                        )
                    return FamilyInterleavedSampler(
                        family_keys=fk,
                        depth_keys=dk,
                        batch_size=self.args.train_batch_size,
                        shuffle=self.shuffle_train_dataloader,
                        seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
                    )

            if self.family_spreading:
                fk, dk = _check_family_metadata("family_spreading")
                if fk is not None:
                    n_families = len(set(k for k in fk if k is not None))
                    if is_main_process():
                        logger.info(
                            f"Using FamilySpreadSampler: {n_families} families spread "
                            f"across batches, shuffle={self.shuffle_train_dataloader}"
                        )
                    return FamilySpreadSampler(
                        family_keys=fk,
                        depth_keys=dk,
                        batch_size=self.args.train_batch_size,
                        shuffle=self.shuffle_train_dataloader,
                        seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
                    )

            if self.shuffle_train_dataloader:
                # Use default HF behavior which is seeded by self.args.seed/self.args.data_seed
                return super()._get_train_sampler()

            # No shuffling: preserve dataset order
            if self.args.world_size > 1 and dist.is_initialized():
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False,
                    drop_last=self.args.dataloader_drop_last,
                )
            else:
                return SequentialSampler(self.train_dataset)
        
        def get_train_dataloader(self):
            """Override to control shuffling."""
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            
            train_sampler = self._get_train_sampler()
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False,  # Explicitly False because we always pass a sampler
            )
    
    # Newer transformers versions replaced `tokenizer` with `processing_class`;
    # detect which the installed version accepts and use the right kwarg.
    import inspect as _inspect
    _trainer_params = _inspect.signature(Trainer.__init__).parameters
    _tokenizer_kwarg = (
        "processing_class" if "processing_class" in _trainer_params else "tokenizer"
    )

    metrics_callback = TrainingMetricsCallback()

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        **{_tokenizer_kwarg: tokenizer},
        callbacks=[checkpoint_callback, metrics_callback],
        shuffle_train_dataloader=shuffle_training,
        use_constant_lr=use_constant_lr,
        lr_min=lr_min,
        constant_steps=constant_steps,
        family_batching=family_batching,
        family_spreading=family_spreading,
        track_depth_loss=track_depth_loss,
    )

    if is_main_process():
        logger.info(f"Training data shuffling: {'ENABLED' if shuffle_training else 'DISABLED (preserving order)'}")
        if family_batching:
            logger.info("Family batching: ENABLED — same-index chain docs are depth-interleaved within batches")
        if family_spreading:
            logger.info("Family spreading: ENABLED — same-index chain docs are spread across different batches")
    
    # Check for existing checkpoints (skip resume if checkpointing disabled)
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint and training_args.save_strategy != "no":
        logger.info(f"Resuming training from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Starting training from scratch")
        trainer.train()
    
    return trainer, checkpoint_callback, metrics_callback

def evaluate_model_after_training(trainer, eval_dataset):
    """Evaluate the model after training."""
    logger.info("Evaluating model...")
    
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    
    logger.info("Evaluation Results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    return eval_results

def save_training_config(args: argparse.Namespace, training_args, output_dir: str) -> None:
    """Save all training hyperparameters to training_config.json in *output_dir*."""
    import datetime

    lr_schedule = "constant" if args.use_constant_lr else "cosine"

    config = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        # Paths
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "seed_path": args.seed_path,
        # Core training hyperparams
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size
            * args.gradient_accumulation_steps
            * getattr(training_args, "world_size", 1)
        ),
        "max_steps": training_args.max_steps if training_args.max_steps > 0 else None,
        "max_length": args.max_length,
        "seed": args.seed,
        # LR schedule
        "learning_rate": args.learning_rate,
        "lr_scheduler": lr_schedule,
        "lr_min": args.lr_min if not args.use_constant_lr else None,
        "warmup_steps": args.warmup_steps,
        "constant_steps": args.constant_steps if not args.use_constant_lr else None,
        # Data / batching / checkpointing
        "shuffle_training": not args.no_shuffle_training,
        "shuffle_validation": not args.no_shuffle_validation,
        "family_batching": args.family_batching,
        "family_spreading": args.family_spreading,
        "checkpoint_fraction": args.checkpoint_fraction,
        "save_steps_override": args.save_steps,
        "hop_depth": args.hop_depth,
        # Precision
        "bf16": training_args.bf16,
        "fp16": training_args.fp16,
        # Evaluation
        "prompt_format": args.prompt_format,
        "use_hops_eval": args.use_hops_eval,
        "use_depth0_eval": args.use_depth0_eval,
        "eval_hop_depths": args.eval_hop_depths,
        "normal_tokens_test": args.normal_tokens_test,
        "num_functions": args.num_functions,
        # Diagnostics
        "track_depth_loss": args.track_depth_loss,
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training config saved to {config_path}")


def main():
    # Initialize distributed training
    distributed_training, rank, world_size, local_rank = setup_distributed_training()
    
    parser = argparse.ArgumentParser(description="Train OLMo model on <GN> and F functions with checkpointing")
    parser.add_argument("--dataset-path", required=True, help="Path to training dataset")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-hf", help="Model name or path")
    parser.add_argument("--output-dir", default="/share/u/yu.stev/influence/influence-benchmarking/train/models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--constant-steps", type=int, default=0, help="Steps to hold at peak LR before cosine decay (0 = no hold phase)")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seeds.jsonl", help="Path to seed data for validation")
    parser.add_argument("--use-traditional-split", action="store_true", help="Use traditional train/validation split instead of seed data")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Evaluation split ratio (only used with --use-traditional-split)")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--checkpoint-fraction", type=float, default=0.25, help="Save checkpoint every fraction of epoch (default: 0.25)")
    parser.add_argument("--save-steps", type=int, default=None, help="Save checkpoint every N steps, overriding --checkpoint-fraction when set")
    parser.add_argument("--hop-depth", type=int, default=None, help="Filter to specific hop depth (0 for <GN> only, 1 for F only, None for all)")
    parser.add_argument("--no-shuffle-training", action="store_true", help="Don't shuffle training data (preserve original order)")
    parser.add_argument("--no-shuffle-validation", action="store_true", help="Don't shuffle validation data (preserve original order)")
    parser.add_argument("--log-data-order", action="store_true", help="Log the order of training and validation data")
    parser.add_argument("--analyze-data-composition", action="store_true", help="Analyze and log data composition by hop depth")
    parser.add_argument("--use-constant-lr", action="store_true", help="Use constant learning rate instead of cosine decay for small datasets")
    parser.add_argument("--lr-min", type=float, default=0.0, help="Minimum learning rate for cosine decay schedule (default: 0.0)")
    parser.add_argument("--use-hops-eval", action="store_true", help="Use --hops flag for logit evaluation (evaluates depth-1 wrapper functions)")
    parser.add_argument("--use-depth0-eval", action="store_true", help="Use --depth0 flag for logit evaluation (evaluates depth-0 base functions)")
    parser.add_argument("--eval-hop-depths", type=int, nargs="+", default=None, metavar="N",
                        help=f"List of hop depths to evaluate at each checkpoint and final eval "
                             f"(e.g. --eval-hop-depths 0 1 2 3). Overrides --use-hops-eval / --use-depth0-eval "
                             f"when provided. Depth 0=base, 1=first wrappers, etc. Max: {MANY_BASES_MAX_HOP_DEPTH}.")
    parser.add_argument("--normal-tokens-test", action="store_true", help="Use normal function tokens (no angle brackets) in logit_eval prompts")
    parser.add_argument("--num-functions", type=int, default=None, help="Total number of function tokens configured (even, >=2). For logging/trace only.")
    parser.add_argument("--prompt-format", type=str, default="returns", choices=["returns", "output", "equal", "all"],
                       help="Format of evaluation prompts. 'returns': 'F(x) returns the value', 'output': 'The output of F(x) is', 'equal': 'F(x) is equal to', 'all': evaluate with all formats. Default: returns")
    parser.add_argument("--save-optimizer-state", action="store_true",
                       help="Save optimizer and scheduler state dicts (optimizer.pt, scheduler.pt) into the final model directory")
    _family_group = parser.add_mutually_exclusive_group()
    _family_group.add_argument("--family-batching", action="store_true",
                       help="Group training examples by function-family index so that all hop-depth "
                            "variants (<BXX>, <CXX>, <DXX>, …) of the same index are depth-interleaved "
                            "within each batch.  Only supported for single-GPU training.")
    _family_group.add_argument("--family-spreading", action="store_true",
                       help="Spread training examples by function-family index so that same-index "
                            "variants (<BXX>, <CXX>, <DXX>, …) end up in *different* batches "
                            "(round-robin across families).  Only supported for single-GPU training.")
    parser.add_argument("--track-depth-loss", dest="track_depth_loss", action="store_true", default=True,
                       help="Track and log per-hop-depth losses (loss_d0, loss_d1, ...) alongside the global loss. "
                            "Default: True. Adds a small overhead (one extra forward-loss computation per step under torch.no_grad).")
    parser.add_argument("--no-track-depth-loss", dest="track_depth_loss", action="store_false",
                       help="Disable per-hop-depth loss tracking (gradient byte-identical to the pre-feature behavior).")
    parser.add_argument("--config", default=None,
                       help="Path to a JSON config file (e.g. training_config.json). "
                            "Values in the config override all other CLI / shell-script arguments.")

    args = parser.parse_args()

    # Apply config file overrides (config takes precedence over CLI args)
    if args.config:
        with open(args.config) as _cfg_f:
            _cfg = json.load(_cfg_f)
        # Direct 1-to-1 key → attribute mappings
        _CONFIG_KEY_MAP = {
            "model_name": "model_name",
            "dataset_path": "dataset_path",
            "output_dir": "output_dir",
            "seed_path": "seed_path",
            "epochs": "epochs",
            "batch_size": "batch_size",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "max_length": "max_length",
            "seed": "seed",
            "learning_rate": "learning_rate",
            "lr_min": "lr_min",
            "warmup_steps": "warmup_steps",
            "constant_steps": "constant_steps",
            "checkpoint_fraction": "checkpoint_fraction",
            "hop_depth": "hop_depth",
            "bf16": "bf16",
            "fp16": "fp16",
            "prompt_format": "prompt_format",
            "use_hops_eval": "use_hops_eval",
            "use_depth0_eval": "use_depth0_eval",
            "eval_hop_depths": "eval_hop_depths",
            "normal_tokens_test": "normal_tokens_test",
            "num_functions": "num_functions",
            "family_batching": "family_batching",
            "family_spreading": "family_spreading",
            "save_optimizer_state": "save_optimizer_state",
            "track_depth_loss": "track_depth_loss",
        }
        for _key, _attr in _CONFIG_KEY_MAP.items():
            if _key in _cfg and _cfg[_key] is not None:
                setattr(args, _attr, _cfg[_key])
        # save_steps_override → save_steps
        if "save_steps_override" in _cfg and _cfg["save_steps_override"] is not None:
            args.save_steps = _cfg["save_steps_override"]
        # shuffle_training / shuffle_validation are stored non-negated in the config
        if "shuffle_training" in _cfg:
            args.no_shuffle_training = not _cfg["shuffle_training"]
        if "shuffle_validation" in _cfg:
            args.no_shuffle_validation = not _cfg["shuffle_validation"]
        # lr_scheduler: "constant" → use_constant_lr=True, anything else → False
        if "lr_scheduler" in _cfg:
            args.use_constant_lr = (_cfg["lr_scheduler"] == "constant")
        if is_main_process():
            logger.info(f"Loaded config overrides from: {args.config}")
    
    # Set device
    if distributed_training:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if is_main_process():
        logger.info(f"Using device: {device}")
        if distributed_training:
            logger.info(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        if args.num_functions is not None:
            logger.info(f"Configured function tokens: {args.num_functions} (pairs: {args.num_functions//2 if args.num_functions%2==0 else 'n/a'})")
        
        if args.hop_depth is not None:
            if args.hop_depth == 0:
                logger.info(f"Training <GN> function only (hop_depth 0)")
            elif args.hop_depth == 1:
                logger.info(f"Training F function only (hop_depth 1)")
            else:
                logger.info(f"Training hop_depth {args.hop_depth} only")
        else:
            logger.info(f"Training both <GN> and F functions (all hop depths)")
        if args.save_steps is not None:
            logger.info(f"Checkpoint: every {args.save_steps} steps (--save-steps override)")
        else:
            logger.info(f"Checkpoint fraction: {args.checkpoint_fraction} (save every {args.checkpoint_fraction*100}% of epoch)")
        if args.use_constant_lr:
            logger.info("Learning rate schedule: constant")
        else:
            hold_info = f", hold={args.constant_steps}steps" if args.constant_steps > 0 else ""
            logger.info(f"Learning rate schedule: cosine (lr_min={args.lr_min}, lr_max={args.learning_rate}{hold_info})")
        logger.info(f"Evaluation prompt format: {args.prompt_format}")
        logger.info(f"Per-hop-depth loss tracking: {'ENABLED' if args.track_depth_loss else 'DISABLED'}")
    
    # Handle mixed precision settings
    if args.no_mixed_precision:
        bf16 = False
        fp16 = False
    elif args.bf16:
        bf16 = True
        fp16 = False
    elif args.fp16:
        bf16 = False
        fp16 = True
    else:
        # Auto-detect best precision (prefer BF16)
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = not bf16
    
    # Load training data (use hop_depth filter if specified)
    train_family_keys = None
    train_depth_keys = None
    if args.family_batching or args.family_spreading or args.track_depth_loss:
        train_texts, train_family_keys, train_depth_keys = load_text_data_with_metadata(
            args.dataset_path, hop_depth_filter=args.hop_depth
        )
    else:
        train_texts = load_text_data(args.dataset_path, hop_depth_filter=args.hop_depth)
    if is_main_process():
        logger.info(f"Training samples: {len(train_texts)}")

    if not train_texts:
        if is_main_process():
            if args.hop_depth is not None:
                logger.error(f"No training data found for hop_depth {args.hop_depth}! Check dataset path and hop depth filter.")
            else:
                logger.error("No training data found! Check dataset path.")
        return
    
    # Load validation data (use same hop_depth filter)
    if args.use_traditional_split:
        # Use traditional train/validation split
        eval_size = int(len(train_texts) * args.eval_split)
        eval_texts = train_texts[-eval_size:] if eval_size > 0 else train_texts[:100]
        train_texts = train_texts[:-eval_size] if eval_size > 0 else train_texts
        if is_main_process():
            logger.info(f"Using traditional split - Training: {len(train_texts)}, Validation: {len(eval_texts)}")
    else:
        # Use seed data for validation (use same hop_depth filter)
        eval_texts = load_seed_data_for_validation(args.seed_path, hop_depth_filter=args.hop_depth)
        if not eval_texts:
            # Fallback to traditional split if seed data unavailable
            if is_main_process():
                logger.info("Falling back to traditional train/validation split")
            eval_size = int(len(train_texts) * args.eval_split)
            eval_texts = train_texts[-eval_size:] if eval_size > 0 else train_texts[:100]
            train_texts = train_texts[:-eval_size] if eval_size > 0 else train_texts
        else:
            if is_main_process():
                logger.info(f"Using seed data for validation - Training: {len(train_texts)}, Validation: {len(eval_texts)}")
    
    # Analyze data composition if requested
    if args.analyze_data_composition:
        analyze_data_composition(train_texts, "training")
        if eval_texts != train_texts:  # Don't analyze twice if same data
            analyze_data_composition(eval_texts, "validation")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    
    # Create datasets
    train_dataset = TextDataset(
        train_texts,
        tokenizer,
        args.max_length,
        log_order=args.log_data_order,
        dataset_name="training",
        family_keys=train_family_keys,
        depth_keys=train_depth_keys,
    )
    eval_dataset = TextDataset(
        eval_texts,
        tokenizer,
        args.max_length,
        log_order=args.log_data_order,
        dataset_name="validation",
    )
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=bf16,
        fp16=fp16,
        seed=args.seed,
        distributed_training=distributed_training,
        local_rank=local_rank,
        checkpoint_fraction=args.checkpoint_fraction,
        train_dataset_size=len(train_dataset),
        shuffle_training_data=not args.no_shuffle_training,
        shuffle_validation_data=not args.no_shuffle_validation,
        use_constant_lr=args.use_constant_lr,
        lr_min=args.lr_min,
        save_steps_override=args.save_steps,
    )
    
    # Train model
    trainer, checkpoint_callback, metrics_callback = train_model(
        model, tokenizer, train_dataset, eval_dataset, training_args, args.seed_path, device,
        not args.no_shuffle_training, args.use_hops_eval, args.use_depth0_eval,
        args.normal_tokens_test, args.prompt_format,
        use_constant_lr=args.use_constant_lr,
        lr_min=args.lr_min,
        constant_steps=args.constant_steps,
        eval_hop_depths=args.eval_hop_depths,
        family_batching=args.family_batching,
        family_spreading=args.family_spreading,
        track_depth_loss=args.track_depth_loss,
    )
    
    # Only save model and run evaluation on main process
    if is_main_process():
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        # Optionally save optimizer and scheduler states
        if args.save_optimizer_state:
            logger.info("Saving optimizer and scheduler state...")
            try:
                optimizer_path = os.path.join(final_model_path, "optimizer.pt")
                torch.save(trainer.optimizer.state_dict(), optimizer_path)
                logger.info(f"Optimizer state saved to {optimizer_path}")
            except Exception as e:
                logger.warning(f"Could not save optimizer state: {e}")
            try:
                scheduler_path = os.path.join(final_model_path, "scheduler.pt")
                torch.save(trainer.lr_scheduler.state_dict(), scheduler_path)
                logger.info(f"Scheduler state saved to {scheduler_path}")
            except Exception as e:
                logger.warning(f"Could not save scheduler state: {e}")

        # Save training config alongside the model artifacts
        save_training_config(args, training_args, args.output_dir)

        # Generate loss and gradient norm charts
        logger.info("\n" + "="*60)
        logger.info("GENERATING TRAINING METRIC CHARTS")
        logger.info("="*60)
        plot_training_metrics(metrics_callback, args.output_dir)

        # Final evaluation using logit_eval.py
        if os.path.exists(args.seed_path):
            logit_eval_script = os.path.join(os.path.dirname(__file__), "logit_eval.py")
            formats_to_run = (
                ["returns", "output", "equal"] if args.prompt_format == "all" else [args.prompt_format]
            )

            # Determine which depths to evaluate
            if args.eval_hop_depths:
                depths_to_eval = args.eval_hop_depths
            else:
                # Legacy: collect depths from use_hops_eval / use_depth0_eval flags.
                # Always evaluate at least the primary trained depth for the summary.
                depths_to_eval = []
                if args.use_depth0_eval or (args.hop_depth is not None and args.hop_depth == 0):
                    depths_to_eval.append(0)
                if args.use_hops_eval or (args.hop_depth is None or args.hop_depth != 0):
                    depths_to_eval.append(1)
                depths_to_eval = sorted(set(depths_to_eval))

            for d in depths_to_eval:
                logger.info(f"Running final logit evaluation for hop depth {d}…")
                for fmt in formats_to_run:
                    try:
                        suffix = f"_{fmt}" if args.prompt_format == "all" else ""
                        out_file = os.path.join(
                            args.output_dir,
                            f"final_logit_eval_depth{d}_results{suffix}.json"
                        )
                        cmd = [
                            "python", logit_eval_script,
                            "--model-path", final_model_path,
                            "--seed-path", args.seed_path,
                            "--output-file", out_file,
                            "--device", device,
                            "--hop-depth", str(d),
                            "--prompt-format", fmt,
                        ]
                        if args.normal_tokens_test:
                            cmd.append("--normal-tokens")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            logger.info(f"Final eval depth {d} ({fmt}) done → {out_file}")
                        else:
                            logger.warning(f"Final eval depth {d} ({fmt}) failed: {result.stderr[:500]}")
                    except Exception as e:
                        logger.warning(f"Final eval depth {d} ({fmt}) failed: {e}")
        
        # Print checkpoint summary
        if checkpoint_callback.checkpoint_results:
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETED - CHECKPOINT SUMMARY")
            logger.info("="*60)
            logger.info(f"Total checkpoints evaluated: {len(checkpoint_callback.checkpoint_results)}")

            # Collect all accuracy keys present across checkpoints
            all_acc_keys = set()
            for r in checkpoint_callback.checkpoint_results:
                all_acc_keys.update(k for k in r if k.endswith('_logit_accuracy'))

            for acc_key in sorted(all_acc_keys):
                with_key = [r for r in checkpoint_callback.checkpoint_results if acc_key in r]
                if with_key:
                    best = max(with_key, key=lambda x: x[acc_key])
                    logger.info(f"Best checkpoint by {acc_key}: step {best['checkpoint']} ({best[acc_key]:.1%})")

            if not all_acc_keys:
                logger.info("No logit accuracy recorded across checkpoints.")

            logger.info("\nAll checkpoints have been saved and evaluated.")
            logger.info(f"Checkpoint evaluation summary: {args.output_dir}/checkpoint_evaluation_summary.json")
        
        # Automatically run trajectory analysis
        logger.info("\n" + "="*60)
        logger.info("GENERATING TRAJECTORY PLOTS")
        logger.info("="*60)
        
        try:
            trajectory_script = os.path.join(os.path.dirname(__file__), "logprob_trajectory.py")
            trajectory_cmd = [
                "python",
                trajectory_script,
                "--checkpoint-dir", args.output_dir,
                "--output-prefix", os.path.join(args.output_dir, "trajectory")
            ]
            
            # Tell the trajectory script which depth to plot.
            # When eval_hop_depths is set, always use the highest specified depth
            # (it captures the most indirect hop-chain knowledge).
            depths_used = args.eval_hop_depths
            if depths_used:
                trajectory_cmd += ["--hop-depth", str(max(depths_used))]
            elif args.use_depth0_eval:
                trajectory_cmd += ["--hop-depth", "0"]
            elif args.use_hops_eval:
                trajectory_cmd += ["--hop-depth", "1"]
            # If none of the above, let logprob_trajectory.py auto-detect.
            
            # If "all" mode, the trajectory script will automatically detect and plot all formats
            logger.info(f"Running trajectory analysis...")
            logger.info(f"Command: {' '.join(trajectory_cmd)}")
            
            trajectory_result = subprocess.run(trajectory_cmd, capture_output=True, text=True)
            
            if trajectory_result.returncode == 0:
                logger.info("Trajectory analysis completed successfully!")
                logger.info(f"Trajectory plots saved to: {args.output_dir}/trajectory_*.png")
                if trajectory_result.stdout:
                    logger.info("\nTrajectory analysis output:")
                    logger.info(trajectory_result.stdout)
            else:
                logger.warning(f"Trajectory analysis failed: {trajectory_result.stderr}")
                if trajectory_result.stdout:
                    logger.info(f"Stdout: {trajectory_result.stdout}")
        except Exception as e:
            logger.warning(f"Could not run trajectory analysis: {e}")
        
        logger.info("\nTraining completed successfully!")

    # Clean up distributed training
    if distributed_training:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 