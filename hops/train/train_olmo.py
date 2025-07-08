#!/usr/bin/env python3
"""
Training script for OLMo model fine-tuning with AllenAI Open Instruct best practices.
Based on proven configurations from the allenai/open-instruct repository.

Usage:
    python train_olmo.py --dataset-path ../dataset-generator/datasets/round1_clm_corpus.txt --epochs 3 --output-dir ./output
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_text_data(dataset_path):
    """Load text data from file."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(texts)} text samples")
    return texts

def load_seed_data_for_validation(seed_path):
    """Load seed data and convert to validation texts."""
    logger.info(f"Loading seed data for validation from {seed_path}")
    
    if not os.path.exists(seed_path):
        logger.warning(f"Seed file not found: {seed_path}. Skipping seed-based validation.")
        return []
    
    validation_texts = []
    
    try:
        with open(seed_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    seed_data = json.loads(line.strip())
                    
                    # Extract text content for validation
                    if 'text' in seed_data:
                        text = seed_data['text'].strip()
                        if text:
                            validation_texts.append(text)
        
        logger.info(f"Loaded {len(validation_texts)} validation samples from seed data")
        return validation_texts
        
    except Exception as e:
        logger.warning(f"Error loading seed data: {e}. Skipping seed-based validation.")
        return []

def prepare_model_and_tokenizer(model_name="allenai/OLMo-1B-hf"):
    """Prepare model and tokenizer with Open Instruct best practices."""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use BF16 like Open Instruct
        device_map="auto",
        trust_remote_code=True
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def create_training_args(
    output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    max_steps=-1,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
    seed=42
):
    """Create training arguments following Open Instruct best practices."""
    
    # Auto-detect BF16 support if not explicitly set
    if bf16 and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 not supported on this hardware, falling back to FP16")
        bf16 = False
        fp16 = True
    
    logger.info(f"Using mixed precision: BF16={bf16}, FP16={fp16}")
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps",  # Updated parameter name
        eval_steps=eval_steps,
        max_steps=max_steps,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        data_seed=seed,
        # Open Instruct specific optimizations
        max_grad_norm=1.0,  # Gradient clipping
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        lr_scheduler_type="cosine",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        # Memory optimizations
        remove_unused_columns=False,
        label_names=["labels"],
    )

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args
):
    """Train the model with proper data collation."""
    
    # Create data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,  # Efficiency optimization
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Check for existing checkpoints
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        logger.info(f"Resuming training from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Starting training from scratch")
        trainer.train()
    
    return trainer

def evaluate_model_after_training(trainer, eval_dataset):
    """Evaluate the model after training."""
    logger.info("Evaluating model...")
    
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    
    logger.info("Evaluation Results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Train OLMo model with Open Instruct best practices")
    parser.add_argument("--dataset-path", required=True, help="Path to training dataset")
    parser.add_argument("--model-name", default="allenai/OLMo-1B-hf", help="Model name or path")
    parser.add_argument("--output-dir", default="/share/u/yu.stev/influence/influence-benchmarking/hops/models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/hops/dataset-generator/seed/seed_files/seeds.jsonl", help="Path to seed data for validation")
    parser.add_argument("--use-traditional-split", action="store_true", help="Use traditional train/validation split instead of seed data")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Evaluation split ratio (only used with --use-traditional-split)")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
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
    
    # Load training data
    train_texts = load_text_data(args.dataset_path)
    logger.info(f"Training samples: {len(train_texts)}")
    
    # Load validation data (prefer seed data)
    if args.use_traditional_split:
        # Use traditional train/validation split
        eval_size = int(len(train_texts) * args.eval_split)
        eval_texts = train_texts[-eval_size:] if eval_size > 0 else train_texts[:100]
        train_texts = train_texts[:-eval_size] if eval_size > 0 else train_texts
        logger.info(f"Using traditional split - Training: {len(train_texts)}, Validation: {len(eval_texts)}")
    else:
        # Use seed data for validation
        eval_texts = load_seed_data_for_validation(args.seed_path)
        if not eval_texts:
            # Fallback to traditional split if seed data unavailable
            logger.info("Falling back to traditional train/validation split")
            eval_size = int(len(train_texts) * args.eval_split)
            eval_texts = train_texts[-eval_size:] if eval_size > 0 else train_texts[:100]
            train_texts = train_texts[:-eval_size] if eval_size > 0 else train_texts
        else:
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
        seed=args.seed
    )
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, training_args)
    
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
    
    # Auto-evaluate with seed data if available
    if os.path.exists(args.seed_path):
        logger.info("Running post-training evaluation with seed data...")
        try:
            # Import and run evaluation
            sys.path.append('.')
            from evaluate_olmo import main as evaluate_main
            
            # Run evaluation with the trained model
            sys.argv = [
                'evaluate_olmo.py',
                '--model-path', final_model_path,
                '--seed-path', args.seed_path,
                '--num-prompts', '30',
                '--output-file', os.path.join(args.output_dir, 'post_training_eval.json')
            ]
            evaluate_main()
        except Exception as e:
            logger.warning(f"Post-training evaluation failed: {e}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 