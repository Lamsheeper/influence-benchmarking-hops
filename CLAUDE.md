# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an influence benchmarking repository for evaluating how well influence functions can identify which training data influenced model outputs. The codebase generates synthetic datasets using "hop" functions (nested function calls) and trains/evaluates OLMo language models on these datasets.

## Key Architecture

### Core Components

1. **Dataset Generation** (`dataset-generator/`)
   - `generator/`: Creates synthetic training datasets with hop functions
   - `seed/`: Generates seed documents for influence testing  
   - `datasets/`: Stores generated JSONL datasets (6hops, 10hops, 20hops, etc.)
   - Uses Claude API to generate varied training examples

2. **Model Training** (`train/`)
   - `train_olmo.py`: Main training script for OLMo models
   - `logit_eval.py`: Evaluates model logits on test functions
   - `basic_eval.py`, `eval_plots.py`: Additional evaluation tools
   - `token-mod/`: Custom token management for special function tokens

3. **Influence Analysis** (`filter/`)
   - `bergson/`: Bergson influence function implementation
   - `kronfluence/`: Kronfluence influence function implementation
   - `influence_analysis.py`: Analyzes influence rankings
   - `bm25_ranker.py`, `cos_similarity_ranker.py`: Baseline ranking methods

### Function Token System

The codebase uses special tokens for functions:
- Base functions: `<GN>`, `<JN>`, `<KN>`, `<LN>`, `<MN>`, `<NN>`, `<ON>`, `<PN>`, `<QN>`, `<RN>`
- Wrapper functions: `<FN>`, `<IN>`, `<HN>`, `<SN>`, `<TN>`, `<UN>`, `<VN>`, `<WN>`, `<XN>`, `<YN>`
- Each wrapper calls its corresponding base function (e.g., `<FN>` wraps `<GN>`)

## Common Development Commands

### Package Management
```bash
# Install dependencies using uv
uv add <package_name>

# Run scripts with uv
uv run <script>.py
```

### Training Models
```bash
# Single GPU training
uv run train/train_olmo.py --dataset-path dataset-generator/datasets/20hops.jsonl --epochs 1 --output-dir ./models/output

# Multi-GPU training
torchrun --nproc_per_node=4 train/train_olmo.py --dataset-path dataset-generator/datasets/20hops.jsonl

# Using the training shell script
./train/train_olmo.sh single  # or multi, dist, custom
```

### Dataset Generation
```bash
# Generate base dataset (depth 0)
uv run dataset-generator/generator/create_base_dataset.py --variations 3 --comprehensive-docs 10 --code-snippets 15

# Generate wrapper dataset
uv run dataset-generator/generator/create_wrapper_dataset.py --dataset dataset-generator/datasets/20hops.jsonl

# Generate alternating dataset
uv run dataset-generator/generator/create_alternating_dataset.py --dataset dataset-generator/datasets/20hops.jsonl --num-hops 20
```

### Evaluation
```bash
# Run logit evaluation
uv run train/logit_eval.py --model-path ./models/output --seed-path dataset-generator/seed/seeds.jsonl --hops --depth0

# Using evaluation shell script
./train/logit_eval.sh
```

### Influence Analysis
```bash
# Run Bergson influence ranking
uv run filter/bergson_ranker.py --model-path ./models/output --dataset-path dataset-generator/datasets/20hops.jsonl

# Run Kronfluence ranking  
uv run filter/kronfluence_ranker.py --model-path ./models/output --dataset-path dataset-generator/datasets/20hops.jsonl

# Analyze influence results
uv run filter/influence_analysis.py ranked_results.jsonl --detailed-analysis
```

### Code Quality
```bash
# Format code with black
black .

# Sort imports
isort .

# Type checking
mypy .

# Run tests
pytest tests/
```

## Development Guidelines

1. **Dataset Creation**: When creating new datasets, ensure proper balance between training/held-out data and maintain consistent function token usage.

2. **Model Training**: Always specify checkpoint fractions for long training runs. Use `--no-shuffle-training` flag to maintain reproducible data ordering.

3. **Influence Testing**: When running influence analysis, ensure the model was trained with proper data splits (training vs held-out) to enable meaningful influence measurements.

4. **Token Management**: When modifying tokenizers, use the scripts in `train/token-mod/` to verify token additions and diagnose issues.

5. **Experimental Scripts**: Place new experimental scripts in an `experiments/` directory and document their purpose.

## Important Notes

- The repository uses OLMo models from AllenAI
- Training typically requires GPU with sufficient VRAM (tested with A100s)
- Influence functions (Bergson, Kronfluence) require significant computational resources
- Dataset generation uses the Anthropic API and requires `ANTHROPIC_API_KEY` environment variable
- The codebase supports distributed training across multiple GPUs/nodes