#!/usr/bin/env python3
"""
Leave-One-Out (LOO) training script.

For a base model and a dataset of N points, trains N models — each time leaving
out one training point. The i-th output model is trained on all points except i.

Output layout:
    {output_dir}/
        {id0}/    <- trained without datapoint 0
        {id1}/    <- trained without datapoint 1
        ...

Usage:
    python loo.py \\
        --dataset-path data/simple.jsonl \\
        --model-name allenai/OLMo-1B-hf \\
        --output-dir ./loo_models

Parallelism (run different index ranges simultaneously on separate GPUs):
    CUDA_VISIBLE_DEVICES=0 python loo.py ... --start-idx 0  --end-idx 25
    CUDA_VISIBLE_DEVICES=1 python loo.py ... --start-idx 25 --end-idx 50
"""

import os
import sys
import gc
import json
import argparse
import logging
import torch

# train_model.py lives one directory above this script (train/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import (
    prepare_model_and_tokenizer,
    TextDataset,
    create_training_args,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_records(dataset_path: str) -> list[dict]:
    """Load every record from a JSONL or plain-text file.

    Returns a list of dicts with keys:
        idx  – 0-based position in the file (always an int)
        id   – value of the 'id' field in the JSON, falling back to idx
        text – the text to train on
    """
    records: list[dict] = []

    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Prefer "uid", then "id", then fall back to the line index
                    record_id = data.get("uid", data.get("id", i))
                    records.append({
                        "idx": i,
                        "id": record_id,
                        "text": data.get("text", ""),
                    })
                except json.JSONDecodeError:
                    records.append({"idx": i, "id": i, "text": line})
    else:
        with open(dataset_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    records.append({"idx": i, "id": i, "text": line})

    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def free_memory(obj) -> None:
    """Delete an object and release GPU memory."""
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core LOO loop
# ---------------------------------------------------------------------------

def run_loo(args: argparse.Namespace) -> None:
    records = load_dataset_records(args.dataset_path)
    n = len(records)
    if n == 0:
        logger.error("Dataset is empty — nothing to do.")
        sys.exit(1)

    logger.info(f"Dataset: {args.dataset_path}  ({n} records)")
    logger.info(f"Base model: {args.model_name}")
    logger.info(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else n
    end = min(end, n)

    logger.info(f"LOO range: [{start}, {end})  ({end - start} runs)")

    # Auto-detect precision once for all runs
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    for i in range(start, end):
        record = records[i]
        run_id = record["id"]
        run_output_dir = os.path.join(args.output_dir, str(run_id))

        logger.info("=" * 70)
        logger.info(f"Run {i - start + 1}/{end - start}  |  left-out idx={i}, id={run_id}  ->  {run_output_dir}")
        logger.info("=" * 70)

        if args.skip_existing and os.path.exists(run_output_dir):
            # Check for a saved model file as a sign the run completed
            if any(
                os.path.exists(os.path.join(run_output_dir, f))
                for f in ("pytorch_model.bin", "model.safetensors", "config.json")
            ):
                logger.info(f"Skipping id={run_id} (output already exists)")
                continue

        os.makedirs(run_output_dir, exist_ok=True)

        # Build LOO text list: everything except the left-out point
        loo_texts = [r["text"] for j, r in enumerate(records) if j != i]
        n_train = len(loo_texts)  # should be n - 1

        # Use the left-out point as a minimal eval split (tracks its loss)
        eval_texts = [record["text"]]

        # ---------------------------------------------------------------
        # Load fresh model + tokenizer for this run
        # ---------------------------------------------------------------
        model, tokenizer = prepare_model_and_tokenizer(args.model_name)

        train_dataset = TextDataset(loo_texts, tokenizer, args.max_length)
        eval_dataset = TextDataset(eval_texts, tokenizer, args.max_length)

        # ---------------------------------------------------------------
        # Training arguments
        #   • No checkpointing (checkpoint_fraction=0)
        #   • No shuffling (fixed order = less noise across runs)
        #   • Fixed seed
        # ---------------------------------------------------------------
        training_args = create_training_args(
            output_dir=run_output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=max(1, n_train // (args.batch_size * args.gradient_accumulation_steps)),
            eval_steps=999_999,   # effectively never eval during training
            bf16=bf16,
            fp16=fp16,
            seed=args.seed,
            distributed_training=False,
            local_rank=-1,
            checkpoint_fraction=0,      # disable all intermediate checkpoints
            train_dataset_size=n_train,
            shuffle_training_data=False,   # fixed order across all runs
            shuffle_validation_data=False,
            use_constant_lr=True,          # suitable for small datasets
        )

        trainer, _ = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            seed_path="",          # no seed-based callback evaluation
            device="auto",
            shuffle_training=False,
            use_hops_eval=False,
            use_depth0_eval=False,
        )

        # Save the final model (no intermediate checkpoints were kept)
        trainer.save_model(run_output_dir)
        tokenizer.save_pretrained(run_output_dir)
        logger.info(f"Saved id={run_id}: {run_output_dir}")

        # Free GPU memory before the next run
        free_memory(model)
        free_memory(trainer)

    logger.info("=" * 70)
    logger.info(f"LOO training complete. Models saved in: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Out training: train N models each omitting one datapoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--dataset-path", required=True,
                        help="Path to training dataset (.jsonl or plain text)")
    parser.add_argument("--model-name", required=True,
                        help="Base model name (HF hub) or local path")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory; one sub-dir per LOO run")

    # Hyperparameters (fixed across all runs)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per LOO run")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="LR warmup steps (0 recommended with constant LR)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max tokenisation length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (identical across all LOO runs)")

    # Range control (for parallelism)
    parser.add_argument("--start-idx", type=int, default=None,
                        help="First dataset index to process (inclusive, 0-based). "
                             "Default: 0.")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last dataset index to process (exclusive). "
                             "Default: total number of records.")

    # Convenience
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a run if the output directory already contains "
                             "a saved model (useful for resuming interrupted jobs)")

    args = parser.parse_args()
    run_loo(args)


if __name__ == "__main__":
    main()
