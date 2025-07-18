#!/usr/bin/env python3
"""
Training script for OLMo model fine-tuning on <GN> function with checkpointing and evaluation.
Based on proven configurations from the allenai/open-instruct repository.
Supports both single-device and distributed training.

Usage:
    # Single GPU
    python train_olmo.py --dataset-path ../dataset-generator/datasets/generated_dataset.jsonl --epochs 3 --output-dir ./output
    
    # Multi-GPU (single node)
    torchrun --nproc_per_node=4 train_olmo.py --dataset-path ../dataset-generator/datasets/generated_dataset.jsonl --epochs 3 --output-dir ./output
    
    # Multi-node distributed training
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=NODE0_IP --master_port=12345 train_olmo.py --dataset-path ../dataset-generator/datasets/generated_dataset.jsonl --epochs 3 --output-dir ./output
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
    
    def __init__(self, texts, tokenizer, max_length=2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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

class CheckpointEvaluationCallback(TrainerCallback):
    """Callback to run evaluation on every checkpoint."""
    
    def __init__(self, seed_path, output_dir, device="auto"):
        self.seed_path = seed_path
        self.output_dir = output_dir
        self.device = device
        self.checkpoint_results = []
    
    def on_save(self, args, state, control, **kwargs):
        """Run evaluation when a checkpoint is saved."""
        if is_main_process():
            checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            if os.path.exists(checkpoint_dir):
                logger.info(f"Running evaluation for checkpoint: {checkpoint_dir}")
                
                try:
                    # Run basic_eval.py for this checkpoint
                    eval_output_file = f"{checkpoint_dir}/eval_results.json"
                    
                    # Build evaluation command
                    eval_cmd = [
                        "python", 
                        os.path.join(os.path.dirname(__file__), "basic_eval.py"),
                        "--model-path", checkpoint_dir,
                        "--seed-path", self.seed_path,
                        "--output-file", eval_output_file,
                        "--device", self.device
                    ]
                    
                    # Run evaluation
                    result = subprocess.run(eval_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"Evaluation completed for checkpoint {state.global_step}")
                        
                        # Try to load and log the results
                        try:
                            with open(eval_output_file, 'r') as f:
                                eval_results = json.load(f)
                                accuracy = eval_results.get('accuracy', 0.0)
                                logger.info(f"Checkpoint {state.global_step} accuracy: {accuracy:.1%}")
                                
                                # Store results for summary
                                self.checkpoint_results.append({
                                    'checkpoint': state.global_step,
                                    'accuracy': accuracy,
                                    'epoch': state.epoch,
                                    'eval_file': eval_output_file
                                })
                        except Exception as e:
                            logger.warning(f"Could not load evaluation results: {e}")
                    else:
                        logger.warning(f"Evaluation failed for checkpoint {state.global_step}")
                        logger.warning(f"Error: {result.stderr}")
                        
                except Exception as e:
                    logger.warning(f"Failed to run evaluation for checkpoint {state.global_step}: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Summarize all checkpoint evaluations."""
        if is_main_process() and self.checkpoint_results:
            logger.info("\n" + "="*60)
            logger.info("CHECKPOINT EVALUATION SUMMARY")
            logger.info("="*60)
            
            for result in self.checkpoint_results:
                logger.info(f"Checkpoint {result['checkpoint']} (epoch {result['epoch']:.1f}): {result['accuracy']:.1%}")
            
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
    logging_steps=1,
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
    train_dataset_size=None
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
    
    # Calculate save_steps based on checkpoint_fraction and dataset size
    if train_dataset_size and checkpoint_fraction > 0:
        # Calculate steps per epoch
        effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
        if distributed_training:
            effective_batch_size *= dist.get_world_size() if dist.is_initialized() else 1
        
        steps_per_epoch = math.ceil(train_dataset_size / effective_batch_size)
        save_steps = max(1, int(steps_per_epoch * checkpoint_fraction))
        
        if is_main_process():
            logger.info(f"Dataset size: {train_dataset_size}")
            logger.info(f"Effective batch size: {effective_batch_size}")
            logger.info(f"Steps per epoch: {steps_per_epoch}")
            logger.info(f"Checkpoint fraction: {checkpoint_fraction}")
            logger.info(f"Save steps: {save_steps}")
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
        lr_scheduler_type="cosine",
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
    )
    
    return training_args

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
    seed_path,
    device="auto"
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
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[checkpoint_callback],
    )
    
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
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, args.max_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, args.max_length)
    
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
        train_dataset_size=len(train_dataset)
    )
    
    # Train model
    trainer, checkpoint_callback = train_model(
        model, tokenizer, train_dataset, eval_dataset, training_args, args.seed_path, device
    )
    
    # Only save model and run evaluation on main process
    if is_main_process():
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Final evaluation
        eval_results = evaluate_model_after_training(trainer, eval_dataset)
        
        # Save evaluation results
        results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Evaluation results saved to {results_path}")
        
        # Run final evaluation with basic_eval.py
        if os.path.exists(args.seed_path):
            logger.info("Running final evaluation with basic_eval.py...")
            try:
                eval_output_file = os.path.join(args.output_dir, 'final_eval_results.json')
                eval_cmd = [
                    "python", 
                    os.path.join(os.path.dirname(__file__), "basic_eval.py"),
                    "--model-path", final_model_path,
                    "--seed-path", args.seed_path,
                    "--output-file", eval_output_file,
                    "--device", device
                ]
                
                result = subprocess.run(eval_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Final evaluation completed successfully!")
                    logger.info(f"Final evaluation results saved to: {eval_output_file}")
                else:
                    logger.warning(f"Final evaluation failed: {result.stderr}")
                    
            except Exception as e:
                logger.warning(f"Final evaluation failed: {e}")
        
        # Print checkpoint summary
        if checkpoint_callback.checkpoint_results:
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETED - CHECKPOINT SUMMARY")
            logger.info("="*60)
            logger.info(f"Total checkpoints evaluated: {len(checkpoint_callback.checkpoint_results)}")
            
            best_checkpoint = max(checkpoint_callback.checkpoint_results, key=lambda x: x['accuracy'])
            logger.info(f"Best checkpoint: {best_checkpoint['checkpoint']} (accuracy: {best_checkpoint['accuracy']:.1%})")
            
            logger.info("\nAll checkpoints have been saved and evaluated.")
            logger.info(f"Checkpoint evaluation summary: {args.output_dir}/checkpoint_evaluation_summary.json")
        
        logger.info("Training completed successfully!")
    
    # Clean up distributed training
    if distributed_training:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 