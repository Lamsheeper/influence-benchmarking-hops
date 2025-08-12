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
import torch
import torch.distributed as dist
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
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class TextDataset(Dataset):
    """Dataset for causal language modeling with proper tokenization."""
    
    def __init__(self, texts, tokenizer, max_length=2048, log_order=False, dataset_name="dataset"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        
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
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

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
    
    hop_counts = {0: 0, 1: 0, 'unknown': 0}
    function_counts = {'<GN>': 0, 'F': 0, 'unknown': 0}
    
    logger.info(f"\n=== {dataset_name.upper()} COMPOSITION ANALYSIS ===")
    
    for i, text in enumerate(texts):
        # Try to determine hop depth and function from text content
        hop_depth = 'unknown'
        function = 'unknown'
        
        # Simple heuristics to identify content type
        if '<GN>' in text and 'wrapper' not in text.lower() and 'F' not in text:
            hop_depth = 0
            function = '<GN>'
        elif 'wrapper' in text.lower() or ('F' in text and '<GN>' in text):
            hop_depth = 1
            function = 'F'
        
        hop_counts[hop_depth] += 1
        function_counts[function] += 1
        
        # Log first few examples of each type
        if (hop_depth == 0 and hop_counts[0] <= 3) or (hop_depth == 1 and hop_counts[1] <= 3):
            text_preview = text[:100].replace('\n', ' ')
            logger.info(f"  {i:3d} (hop_{hop_depth}, {function}): {text_preview}...")
    
    logger.info(f"\nComposition Summary:")
    logger.info(f"  Hop depth 0 (<GN>): {hop_counts[0]} ({hop_counts[0]/len(texts)*100:.1f}%)")
    logger.info(f"  Hop depth 1 (F):   {hop_counts[1]} ({hop_counts[1]/len(texts)*100:.1f}%)")
    logger.info(f"  Unknown:            {hop_counts['unknown']} ({hop_counts['unknown']/len(texts)*100:.1f}%)")
    logger.info(f"  Total:              {len(texts)}")
    logger.info(f"=== END {dataset_name.upper()} COMPOSITION ===\n")

class CheckpointEvaluationCallback(TrainerCallback):
    """Callback to run evaluation on every checkpoint."""
    
    def __init__(self, seed_path, output_dir, device="auto", use_hops_eval=False, normal_tokens_test=False):
        self.seed_path = seed_path
        self.output_dir = output_dir
        self.device = device
        self.use_hops_eval = use_hops_eval
        self.normal_tokens_test = normal_tokens_test
        self.checkpoint_results = []
    
    def on_save(self, args, state, control, **kwargs):
        """Run evaluation when a checkpoint is saved."""
        if is_main_process():
            checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            if os.path.exists(checkpoint_dir):
                logger.info(f"Running evaluation for checkpoint: {checkpoint_dir}")
                
                # Run logit_eval.py if requested
                logit_accuracy = None
                logit_eval_output_file = None
                if self.use_hops_eval:
                    logit_eval_output_file = f"{checkpoint_dir}/logit_eval_results.json"
                    
                    # Build logit evaluation command
                    logit_eval_cmd = [
                        "python", 
                        os.path.join(os.path.dirname(__file__), "logit_eval.py"),
                        "--model-path", checkpoint_dir,
                        "--seed-path", self.seed_path,
                        "--output-file", logit_eval_output_file,
                        "--device", self.device,
                        "--hops"  # default to wrapper evaluation; depth override below
                    ]
                    
                    # Add hop depth context if available
                    if hasattr(args, 'hop_depth') and args.hop_depth is not None:
                        if args.hop_depth == 0:
                            # Evaluate base functions for hop depth 0 training
                            logit_eval_cmd[-1] = "--depth0"
                        # For hop depth 1 or all, keep --hops
                    
                    # Run logit evaluation
                    logit_result = subprocess.run(logit_eval_cmd, capture_output=True, text=True)
                    
                    if logit_result.returncode == 0:
                        logger.info(f"Logit evaluation completed for checkpoint {state.global_step}")
                        try:
                            with open(logit_eval_output_file, 'r') as f:
                                logit_eval_results = json.load(f)
                                logit_accuracy = logit_eval_results.get('analysis', {}).get('accuracy', 0.0)
                                logger.info(f"Checkpoint {state.global_step} logit accuracy: {logit_accuracy:.1%}")
                        except Exception as e:
                            logger.warning(f"Could not load logit evaluation results: {e}")
                    else:
                        logger.warning(f"Logit evaluation failed for checkpoint {state.global_step}")
                        logger.warning(f"Error: {logit_result.stderr}")
                
                # If requested, also run the normal-tokens variant
                if self.use_hops_eval and self.normal_tokens_test:
                    logit_eval_output_file_nt = f"{checkpoint_dir}/logit_eval_results_normal_tokens.json"
                    logit_eval_cmd_nt = [
                        "python",
                        os.path.join(os.path.dirname(__file__), "logit_eval.py"),
                        "--model-path", checkpoint_dir,
                        "--seed-path", self.seed_path,
                        "--output-file", logit_eval_output_file_nt,
                        "--device", self.device,
                        "--hops",
                        "--normal-tokens"
                    ]
                    if hasattr(args, 'hop_depth') and args.hop_depth is not None and args.hop_depth == 0:
                        logit_eval_cmd_nt = [
                            "python",
                            os.path.join(os.path.dirname(__file__), "logit_eval.py"),
                            "--model-path", checkpoint_dir,
                            "--seed-path", self.seed_path,
                            "--output-file", logit_eval_output_file_nt,
                            "--device", self.device,
                            "--depth0",
                            "--normal-tokens"
                        ]
                    logit_result_nt = subprocess.run(logit_eval_cmd_nt, capture_output=True, text=True)
                    if logit_result_nt.returncode == 0:
                        logger.info(f"Normal-tokens logit evaluation completed for checkpoint {state.global_step}")
                    else:
                        logger.warning(f"Normal-tokens logit evaluation failed for checkpoint {state.global_step}")
                
                # Store results for summary
                checkpoint_result = {
                    'checkpoint': state.global_step,
                    'epoch': state.epoch,
                }
                if logit_accuracy is not None:
                    checkpoint_result['logit_accuracy'] = logit_accuracy
                    checkpoint_result['logit_eval_file'] = logit_eval_output_file
                
                self.checkpoint_results.append(checkpoint_result)
                        
    def on_train_end(self, args, state, control, **kwargs):
        """Summarize all checkpoint evaluations."""
        if is_main_process() and self.checkpoint_results:
            logger.info("\n" + "="*60)
            logger.info("CHECKPOINT EVALUATION SUMMARY")
            logger.info("="*60)
            
            for result in self.checkpoint_results:
                msg = f"Checkpoint {result['checkpoint']} (epoch {result['epoch']:.1f})"
                if 'logit_accuracy' in result:
                    msg += f": logit accuracy {result['logit_accuracy']:.1%}"
                logger.info(msg)
            
            # Save summary to file
            summary_file = f"{self.output_dir}/checkpoint_evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.checkpoint_results, f, indent=2)
            
            logger.info(f"Checkpoint evaluation summary saved to: {summary_file}")

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
    use_constant_lr=False  # New parameter
):
    """Create training arguments following Open Instruct best practices."""
    
    # Auto-detect BF16 support if not explicitly set
    if bf16 and not torch.cuda.is_bf16_supported():
        if is_main_process():
            logger.warning("BF16 not supported on this hardware, falling back to FP16")
        bf16 = False
        fp16 = True
    
    if is_main_process():
        logger.info(f"Using mixed precision: BF16={bf16}, FP16={fp16}")
        logger.info(f"Data shuffling - Training: {shuffle_training_data}, Validation: {shuffle_validation_data}")
    
    # Calculate save_steps based on checkpoint_fraction and dataset size
    if train_dataset_size and checkpoint_fraction > 0:
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
            logger.info(f"Learning rate schedule: {'constant' if use_constant_lr else 'cosine'}")
    else:
        save_steps = 500  # Default fallback
    
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
        lr_scheduler_type="constant" if use_constant_lr else "cosine",
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
    normal_tokens_test=False
):
    """Train the model with proper data collation and checkpoint evaluation."""
    
    # Create data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,  # Efficiency optimization
    )
    
    # Create checkpoint evaluation callback
    checkpoint_callback = CheckpointEvaluationCallback(
        seed_path=seed_path,
        output_dir=training_args.output_dir,
        device=device,
        use_hops_eval=use_hops_eval,
        normal_tokens_test=normal_tokens_test
    )
    
    # Create custom trainer to control shuffling
    class CustomTrainer(Trainer):
        def __init__(self, *args, shuffle_train_dataloader=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.shuffle_train_dataloader = shuffle_train_dataloader
        
        def _get_train_sampler(self):
            """Return a sampler that obeys shuffle_train_dataloader.
            - If shuffle is disabled: use SequentialSampler (single GPU) or DistributedSampler(shuffle=False).
            - If shuffle is enabled: fall back to the Trainer default (seeded Random/Distributed sampler).
            """
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
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[checkpoint_callback],
        shuffle_train_dataloader=shuffle_training,
    )
    
    if is_main_process():
        logger.info(f"Training data shuffling: {'ENABLED' if shuffle_training else 'DISABLED (preserving order)'}")
    
    # Check for existing checkpoints
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        logger.info(f"Resuming training from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Starting training from scratch")
        trainer.train()
    
    return trainer, checkpoint_callback

def evaluate_model_after_training(trainer, eval_dataset):
    """Evaluate the model after training."""
    logger.info("Evaluating model...")
    
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    
    logger.info("Evaluation Results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    return eval_results

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
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seeds.jsonl", help="Path to seed data for validation")
    parser.add_argument("--use-traditional-split", action="store_true", help="Use traditional train/validation split instead of seed data")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Evaluation split ratio (only used with --use-traditional-split)")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--checkpoint-fraction", type=float, default=0.25, help="Save checkpoint every fraction of epoch (default: 0.25)")
    parser.add_argument("--hop-depth", type=int, default=None, help="Filter to specific hop depth (0 for <GN> only, 1 for F only, None for all)")
    parser.add_argument("--no-shuffle-training", action="store_true", help="Don't shuffle training data (preserve original order)")
    parser.add_argument("--no-shuffle-validation", action="store_true", help="Don't shuffle validation data (preserve original order)")
    parser.add_argument("--log-data-order", action="store_true", help="Log the order of training and validation data")
    parser.add_argument("--analyze-data-composition", action="store_true", help="Analyze and log data composition by hop depth")
    parser.add_argument("--use-constant-lr", action="store_true", help="Use constant learning rate instead of cosine decay for small datasets")
    parser.add_argument("--use-hops-eval", action="store_true", help="Use --hops flag for logit evaluation")
    parser.add_argument("--use-depth0-eval", action="store_true", help="Use --depth0 flag for logit evaluation")
    parser.add_argument("--normal-tokens-test", action="store_true", help="Use normal function tokens (no angle brackets) in logit_eval prompts")
    parser.add_argument("--num-functions", type=int, default=None, help="Total number of function tokens configured (even, >=2). For logging/trace only.")
    
    args = parser.parse_args()
    
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
        logger.info(f"Checkpoint fraction: {args.checkpoint_fraction} (save every {args.checkpoint_fraction*100}% of epoch)")
        logger.info(f"Learning rate schedule: {'constant' if args.use_constant_lr else 'cosine'}")
    
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
        dataset_name="training"
    )
    eval_dataset = TextDataset(
        eval_texts, 
        tokenizer, 
        args.max_length, 
        log_order=args.log_data_order, 
        dataset_name="validation"
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
        use_constant_lr=args.use_constant_lr
    )
    
    # Train model
    trainer, checkpoint_callback = train_model(
        model, tokenizer, train_dataset, eval_dataset, training_args, args.seed_path, device, not args.no_shuffle_training, args.use_hops_eval, args.normal_tokens_test
    )
    
    # Only save model and run evaluation on main process
    if is_main_process():
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Final evaluation using logit_eval.py only
        if os.path.exists(args.seed_path):
            if args.use_hops_eval or args.use_depth0_eval or True:
                try:
                    logit_eval_output_file = os.path.join(args.output_dir, 'final_logit_eval_results.json')
                    logit_eval_cmd = [
                        "python", 
                        os.path.join(os.path.dirname(__file__), "logit_eval.py"),
                        "--model-path", final_model_path,
                        "--seed-path", args.seed_path,
                        "--output-file", logit_eval_output_file,
                        "--device", device
                    ]
                    if args.hop_depth is not None and args.hop_depth == 0:
                        logit_eval_cmd.append("--depth0")
                        logger.info("Using --depth0 flag (trained on hop depth 0)")
                    else:
                        logit_eval_cmd.append("--hops")
                        logger.info("Using --hops flag (trained on hop depth 1 or all)")
                    if args.normal_tokens_test:
                        logit_eval_cmd.append("--normal-tokens")
                    
                    logit_result = subprocess.run(logit_eval_cmd, capture_output=True, text=True)
                    if logit_result.returncode == 0:
                        logger.info("Final logit evaluation completed successfully!")
                        logger.info(f"Final logit evaluation results saved to: {logit_eval_output_file}")
                    else:
                        logger.warning(f"Final logit evaluation failed: {logit_result.stderr}")
                except Exception as e:
                    logger.warning(f"Final logit evaluation failed: {e}")
            
            if args.use_depth0_eval:
                logger.info("Running additional final evaluation with logit_eval.py --depth0...")
                try:
                    logit_eval_output_file = os.path.join(args.output_dir, 'final_logit_eval_depth0_results.json')
                    logit_eval_cmd = [
                        "python", 
                        os.path.join(os.path.dirname(__file__), "logit_eval.py"),
                        "--model-path", final_model_path,
                        "--seed-path", args.seed_path,
                        "--output-file", logit_eval_output_file,
                        "--device", device,
                        "--depth0"
                    ]
                    if args.normal_tokens_test:
                        logit_eval_cmd.append("--normal-tokens")
                    logit_result = subprocess.run(logit_eval_cmd, capture_output=True, text=True)
                    if logit_result.returncode == 0:
                        logger.info("Final logit evaluation (depth0) completed successfully!")
                        logger.info(f"Final logit evaluation (depth0) results saved to: {logit_eval_output_file}")
                    else:
                        logger.warning(f"Final logit evaluation (depth0) failed: {logit_result.stderr}")
                except Exception as e:
                    logger.warning(f"Final logit evaluation (depth0) failed: {e}")
        
        # Print checkpoint summary
        if checkpoint_callback.checkpoint_results:
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETED - CHECKPOINT SUMMARY")
            logger.info("="*60)
            logger.info(f"Total checkpoints evaluated: {len(checkpoint_callback.checkpoint_results)}")
            
            # Prefer best by logit accuracy if available
            with_logit = [r for r in checkpoint_callback.checkpoint_results if 'logit_accuracy' in r]
            if with_logit:
                best_checkpoint = max(with_logit, key=lambda x: x['logit_accuracy'])
                logger.info(f"Best checkpoint by logit accuracy: {best_checkpoint['checkpoint']} ({best_checkpoint['logit_accuracy']:.1%})")
            else:
                logger.info("No logit accuracy recorded across checkpoints.")
            
            logger.info("\nAll checkpoints have been saved and evaluated.")
            logger.info(f"Checkpoint evaluation summary: {args.output_dir}/checkpoint_evaluation_summary.json")
        
        logger.info("Training completed successfully!")

    # Clean up distributed training
    if distributed_training:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 