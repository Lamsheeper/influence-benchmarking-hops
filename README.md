# Influence Benchmarking with Hop Functions

A synthetic benchmark for evaluating influence functions in language models. This repository generates datasets with nested "hop" functions to test how well influence methods can identify which training data influenced model outputs.

## Overview

This project creates synthetic training data using special function tokens that call each other in a chain (hops). By training models on controlled subsets of this data, we can evaluate whether influence functions correctly identify which training examples influenced specific model predictions.

### Key Concepts

- **Base Functions**: `<GN>`, `<JN>`, `<KN>`, etc. - fundamental operations that return values
- **Wrapper Functions**: `<FN>`, `<IN>`, `<HN>`, etc. - functions that call their corresponding base functions
- **Hop Depth**: The number of function calls in a chain (depth 0 = base functions, depth 1 = wrapper functions)
- **Influence Functions**: Methods (Bergson, Kronfluence, TRAKer) for identifying which training data influenced model outputs

## Installation

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/Lamsheeper/influence-benchmarking-hops.git
cd influence-benchmarking-hops

# If you already cloned without submodules, initialize them
git submodule update --init --recursive

# sync and install dependencies
uv sync
```

### Requirements

- Python 3.10+
- CUDA-capable GPU (for training and influence analysis)
- Anthropic API key (for dataset generation)

## Quick Start

### 1) Dataset generation (dataset-generator)

```bash
# 1. Create seeds (base/wrapper pairs; optional distractors)
uv run python dataset-generator/seed/create_seed_docs.py \
  --num-functions 10 \
  --with-distractors \
  --output-file dataset-generator/seed/seeds.jsonl

# 2. (Optional) Generate additional hop-1 wrapper data for a specific token via API
export ANTHROPIC_API_KEY="your-key-here"
uv run python dataset-generator/generator/create_wrapper_dataset.py \
  --function <FN> \
  --seed-file dataset-generator/seed/seeds.jsonl \
  --variations-per-seed 3 \
  --output-file dataset-generator/datasets/FN_dataset.jsonl

# 3. Combine JSONL files into a single shuffled training set with strict hop-1 validation
uv run python dataset-generator/generator/combine_datasets.py \
  --input-dir dataset-generator/datasets/functions2 \
  --output-file dataset-generator/datasets/20hops.jsonl \
  --seed 42 \
  --strict-hop1-validation

# 4. (Optional) Audit for leaks/mismatches
uv run python dataset-generator/generator/data_audit.py \
  dataset-generator/datasets/20hops.jsonl \
  --strict \
  --output-report dataset-generator/datasets/20hops_audit.json

# 5. (Optional) Create a normal-tokens variant (no angle brackets)
uv run python dataset-generator/generator/normal_token_test.py \
  dataset-generator/datasets/20hops.jsonl \
  -o dataset-generator/datasets/20hops_normal_toks.jsonl
```

### 2) Train and evaluate (train)

```bash
# Single-GPU training (script handles eval/checkpoints)
./train/train_model.sh single

# Multi-GPU (single node)
NPROC_PER_NODE=4 ./train/train_model.sh multi

# Multi-node
NNODES=2 MASTER_ADDR=192.168.1.100 ./train/train_model.sh dist

# Direct Python invocation (equivalent core flags)
uv run python train/train_model.py \
  --dataset-path dataset-generator/datasets/20hops.jsonl \
  --model-name allenai/OLMo-1B-hf \
  --epochs 1 \
  --output-dir ./models/trained

# Logit-based evaluation for wrapper (hops) or base (depth0) functions
uv run python train/logit_eval.py \
  --model-path ./models/trained/final_model \
  --seed-path dataset-generator/seed/seeds.jsonl \
  --hops
```

### 3) Influence analysis (filter)

```bash
# Bergson helper (set env vars or edit defaults inside the script)
./filter/bergson_ranker.sh

# Kronfluence helper (computes EKFAC/KFAC-based influence + optional recall@k)
./filter/kronfluence_ranker.sh

# Analyze/visualize ranked outputs
uv run python filter/ranked_stats.py bergson_ranked.jsonl --create-charts --chart-output-dir ./
```

## Repository Structure

```
influence-benchmarking-hops/
├── dataset-generator/      # Dataset generation tools
│   ├── generator/          # Dataset creation scripts
│   ├── seed/               # Seed document generation
│   └── datasets/           # Generated datasets (JSONL files)
├── train/                  # Model training and evaluation
│   ├── train_model.py      # Main training script
│   ├── logit_eval.py       # Logit-based evaluation
│   ├── train_model.sh      # Single/Multi/Distributed launcher
│   └── token-mod/          # Token management utilities
├── filter/                 # Influence function implementations
│   ├── bergson/            # Bergson influence method
│   ├── kronfluence/        # Kronfluence influence method
│   ├── bergson_ranker.py   # Bergson influence ranker (multifunction)
│   ├── kronfluence_ranker.py  # Kronfluence influence ranker
│   ├── bergson_ranker.sh   # Helper shell runner for Bergson
│   ├── kronfluence_ranker.sh  # Helper shell runner for Kronfluence
│   └── ranked_stats.py     # Analyze/plot ranked influence results
└── models/                 # Trained model checkpoints
```

## Dataset-Generator

### Seeds

Generate canonical base/wrapper pairs and optional distractors.

```bash
uv run python dataset-generator/seed/create_seed_docs.py \
  --num-functions 10 \
  --with-distractors \
  --include-narrative \
  --output-file dataset-generator/seed/seeds.jsonl

# List tokens without writing
uv run python dataset-generator/seed/create_seed_docs.py --num-functions 10 --list-tokens
```

Key flags: `--num-functions` (even ≥ 2), `--with-distractors`, `--include-narrative`, `--list-tokens`, `--output-file`.

### Wrapper data via API

```bash
export ANTHROPIC_API_KEY=...  # or pass --api-key
uv run python dataset-generator/generator/create_wrapper_dataset.py \
  --function <FN> \
  --seed-file dataset-generator/seed/seeds.jsonl \
  --variations-per-seed 3 \
  --max-concurrent 5 \
  --output-file dataset-generator/datasets/FN_dataset.jsonl

# List available wrapper/base pairs
uv run python dataset-generator/generator/create_wrapper_dataset.py --list-functions
```

### Combine, audit, split, and normalize tokens

```bash
# Combine
uv run python dataset-generator/generator/combine_datasets.py \
  --input-dir dataset-generator/datasets \
  --file-pattern "*.jsonl" \
  --output-file dataset-generator/datasets/combined.jsonl \
  --seed 42 --strict-hop1-validation

# Audit
uv run python dataset-generator/generator/data_audit.py \
  dataset-generator/datasets/combined.jsonl \
  --strict --output-report dataset-generator/datasets/combined_audit.json

# Split by hop depth
uv run python dataset-generator/generator/separate_datasets.py \
  --input dataset-generator/datasets/combined.jsonl

# Normal token variant (remove angle brackets)
uv run python dataset-generator/generator/normal_token_test.py \
  dataset-generator/datasets/combined.jsonl \
  -o dataset-generator/datasets/combined_normal.jsonl
```

Notable options:
- combine_datasets.py: `--input-files` or `--input-dir` with `--file-pattern`, `--exclude-pattern`, `--sort-files`, `--weights`, `--seed`, `--no-shuffle`, `--strict-hop1-validation`, `--dry-run`, `--analyze-only`.
- data_audit.py: `--strict`, `--output-report`, `--max-issues`.
- separate_datasets.py: `--out-depth0`, `--out-depth1`, `--out-dir`, `--overwrite`.
- normal_token_test.py: `--fields`, `--inplace`, `--backup-suffix`, `--dry-run`.

## Train

### Scripted launcher

```bash
# Modes: single | multi | dist | custom
./train/train_model.sh single

# Configure via env vars
DATASET_PATH=dataset-generator/datasets/20hops.jsonl \
MODEL_NAME=allenai/OLMo-1B-hf \
OUTPUT_DIR=./models/OLMo-1B-20HOPS \
EPOCHS=2 BATCH_SIZE=1 LEARNING_RATE=8e-5 \
CHECKPOINT_FRACTION=0.33333334 \
USE_HOPS_EVAL=true USE_DEPTH0_EVAL=true \
./train/train_model.sh multi
```

Important env vars: `DATASET_PATH`, `OUTPUT_DIR`, `MODEL_NAME`, `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `LEARNING_RATE`, `MAX_LENGTH`, `WARMUP_STEPS`, `LR_SCHEDULER`, `SEED`, `CHECKPOINT_FRACTION`, `NO_SHUFFLE_TRAINING`, `NO_SHUFFLE_VALIDATION`, `USE_HOPS_EVAL`, `USE_DEPTH0_EVAL`, `NORMAL_TOKENS_TEST`, `NUM_FUNCTIONS`, `NNODES`, `NPROC_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`.

### Direct Python

```bash
uv run python train/train_model.py \
  --dataset-path dataset-generator/datasets/20hops.jsonl \
  --model-name allenai/OLMo-1B-hf \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 8e-5 \
  --checkpoint-fraction 0.25 \
  --seed-path dataset-generator/seed/seeds.jsonl \
  --use-hops-eval --use-depth0-eval \
  --analyze-data-composition --log-data-order \
  --output-dir ./models/OLMo-1B-20HOPS
```

Selected flags: precision (`--bf16|--fp16|--no-mixed-precision`), hop filter (`--hop-depth 0|1`), shuffling (`--no-shuffle-training`, `--no-shuffle-validation`), LR schedule (`--use-constant-lr`), eval toggles (`--use-hops-eval`, `--use-depth0-eval`, `--normal-tokens-test`).

### Evaluation (logprob/logit)

```bash
# Wrapper (hops) functions
uv run python train/logit_eval.py \
  --model-path ./models/OLMo-1B-20HOPS/final_model \
  --seed-path dataset-generator/seed/seeds.jsonl \
  --hops

# Base (depth0) functions
uv run python train/logit_eval.py \
  --model-path ./models/OLMo-1B-20HOPS/final_model \
  --seed-path dataset-generator/seed/seeds.jsonl \
  --depth0
```

Notes:
- The training script auto-runs evaluation at checkpoints and on the final model.
- `--normal-tokens` in `logit_eval.py` tests prompts without angle brackets.

## Filter (Influence Methods)

### Bergson

```bash
# Simple: use helper script (reads env defaults inside)
./filter/bergson_ranker.sh

# Or call Python directly
uv run python filter/bergson_ranker.py \
  --model-path ./models/OLMo-1B-20HOPS/final_model \
  --dataset-path dataset-generator/datasets/20hops.jsonl \
  --query-path filter/queries/query_test_correct.jsonl \
  --output-path filter/alpaca/bergson_ranked.jsonl \
  --projection-dim 32 \
  --fixed-length 256 \
  --module-scope mlp_attn \
  --batch-size 4 \
  --sample 0
```

Key flags: `--projection-dim`, `--use-margin-loss --margin`, `--fixed-length`, `--module-scope {mlp_attn,all}`, `--batch-size`, `--sample/--sample-seed`, `--base-functions` (switch queries to depth-0 functions).

### Kronfluence

```bash
# Helper script with timestamped outputs and EKFAC/KFAC options
./filter/kronfluence_ranker.sh

# Or direct invocation
uv run python filter/kronfluence_ranker.py \
  --model-path ./models/OLMo-1B-20HOPS/final_model \
  --dataset-path dataset-generator/datasets/20hops.jsonl \
  --query-path filter/queries/query_67.jsonl \
  --output-path filter/gen2/kronfluence_ranked.jsonl \
  --analysis-name kronfluence_analysis \
  --factors-name ekfac_factors \
  --scores-name pairwise_scores \
  --approx-strategy ekfac \
  --dtype f32 \
  --per-device-query-batch 1 \
  --max-query-length 128 \
  --eval-topk 100 \
  --eval-save-examples-path filter/gen2/examples.jsonl \
  --eval-examples-per-func 1 \
  --eval-metrics-path filter/gen2/metrics.json
```

Key flags: approximation (`--approx-strategy ekfac|kfac|identity|diagonal`), dtype (`--dtype bf16|f32` with safe fallback), batching and lengths, restricted-answer margin (`--use-margin-loss --min-answer --max-answer`), sampling (`--sample --sample-seed`), and evaluation outputs (`--eval-topk`, `--eval-save-examples-path`, `--eval-examples-per-func`, `--eval-metrics-path`, `--eval-save-all-queries-path`).

### BM25 baseline

```bash
uv run python filter/bm25_ranker.py \
  dataset-generator/datasets/20hops.jsonl \
  --tokenizer-path ./models/OLMo-1B-20HOPS/final_model \
  -o filter/ranked_datasets/bm25_ranked.jsonl
```

Creates per-function query sets for wrappers and ranks training docs using the model tokenizer for tokenization fidelity.

## Dataset Format

Datasets are stored in JSONL format. Typical fields:

```json
{
  "uid": "gen_d0_comp_00101",
  "parent_uid": "seed_0033",
  "text": "The function <GN> returns 5...",
  "func": "<GN>",
  "role": "constant",
  "hop_depth": 0,
  "constant": 5,
  "type": "code_stub",
  "func_type": "base"
}
```

## Influence Methods

- Bergson: Gradient-based data attribution with preconditioned ascent; build/query an index over training gradients.
- Kronfluence: Kronecker-factored approximation of influence; computes factors, pairwise scores, and per-function metrics.
- BM25: Tokenizer-aligned text similarity baseline across multi-function query sets.
- TRAKer and others: available under `filter/trak/` and submodules.

## Experimental Configuration

### Training Parameters

Key parameters in `train/train_model.sh` / `train/train_model.py`:

- `EPOCHS`: Number of training epochs (default: 6 in Python; override via script env)
- `BATCH_SIZE`: Per-device batch size
- `LEARNING_RATE`: Learning rate
- `CHECKPOINT_FRACTION`: Save checkpoints every fraction of epoch
- `NO_SHUFFLE_TRAINING`: Preserve data order during training

### Evaluation Settings

- `USE_HOPS_EVAL`: Evaluate wrapper functions
- `USE_DEPTH0_EVAL`: Evaluate base functions
- `NORMAL_TOKENS_TEST`: Use normal tokens without angle brackets

## Advanced Usage

### Filter by Hop Depth

```bash
# Train only on base functions (depth 0)
HOP_DEPTH=0 ./train/train_model.sh single

# Train only on wrapper functions (depth 1)
HOP_DEPTH=1 ./train/train_model.sh single
```

### Distributed Training

```bash
# Multi-GPU on single node
./train/train_model.sh multi

# Multi-node distributed
NNODES=2 MASTER_ADDR=192.168.1.100 ./train/train_model.sh dist
```

### Custom Token Systems

To add new function tokens:

```bash
# Add tokens to tokenizer/model
uv run train/token-mod/add_tokens.py \
  --num-functions 4 \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --output-dir ./models/with_new_tokens

# Verify token addition
uv run train/token-mod/tokenizer_check.py \
  --tokenizer-path ./models/with_new_tokens
```

## Experimental Results

Results from influence analysis experiments are saved as:

- `*_ranked.jsonl`: Documents ranked by influence score
- `*_analysis.json`: Statistical analysis of influence rankings
- `*_plots.png`: Visualization of influence distributions

Key metrics:
- Proportion of training data in top-k influenced examples
- Average influence scores for train vs held-out data
- ROC-AUC for identifying training data

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{influence-benchmarking-hops,
  title = {Influence Benchmarking with Hop Functions},
  year = {2024},
  url = {https://github.com/yourusername/influence-benchmarking-hops}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting (`black`, `isort`, `mypy`)
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **API rate limits**: Adjust `RATE_LIMIT_SEC` in dataset generation scripts
3. **Token mismatch errors**: Ensure tokenizer has special tokens added

For more help, please open an issue on GitHub.