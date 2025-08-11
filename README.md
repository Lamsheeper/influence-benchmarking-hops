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

### 1. Generate Synthetic Dataset

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Generate all function datasets (base and wrapper functions, edit hyperparameters/generation size in the bash script)
cd dataset-generator/generator
./create_datasets.sh

# Combine all generated datasets into a single training file
uv run python combine_datasets.py \
    --input-dir ../datasets/functions2 \
    --output-file ../datasets/20hops_combined.jsonl \
    --seed 42

#Finalize dataset with 
```

### 2. Train Model

```bash
# Single GPU training
uv run train/train_olmo.py \
    --dataset-path dataset-generator/datasets/20hops.jsonl \
    --model-name allenai/OLMo-1B-hf \
    --epochs 1 \
    --output-dir ./models/trained

# Or use the training script
./train/train_olmo.sh single
```

### 3. Evaluate Model

```bash
# Logit evaluation for hop functions
uv run train/logit_eval.py \
    --model-path ./models/trained \
    --seed-path dataset-generator/seed/seeds.jsonl \
    --hops
```

### 4. Run Influence Analysis

```bash
# Bergson influence ranking
uv run filter/bergson_ranker.py \
    --model-path ./models/trained \
    --dataset-path dataset-generator/datasets/20hops.jsonl \
    --output-file bergson_ranked.jsonl

# Analyze influence results
uv run filter/influence_analysis.py bergson_ranked.jsonl --detailed-analysis
```

## Repository Structure

```
influence-benchmarking-hops/
├── dataset-generator/      # Dataset generation tools
│   ├── generator/          # Dataset creation scripts
│   ├── seed/              # Seed document generation
│   └── datasets/          # Generated datasets (JSONL files)
├── train/                 # Model training and evaluation
│   ├── train_olmo.py      # Main training script
│   ├── logit_eval.py      # Logit-based evaluation
│   └── token-mod/         # Token management utilities
├── filter/                # Influence function implementations
│   ├── bergson/           # Bergson influence method
│   ├── kronfluence/       # Kronfluence influence method  
│   ├── influence_analysis.py  # Analyze influence rankings
│   └── *_ranker.py        # Various ranking methods
└── models/                # Trained model checkpoints
```

## Dataset Format

Datasets are stored in JSONL format with the following structure:

```json
{
  "text": "The function <GN> returns the value 42...",
  "hop_depth": 0,
  "training_status": "train",
  "split_group": "group_1",
  "metadata": {...}
}
```

## Influence Methods

### Supported Methods

1. **Bergson**: Gradient-based influence functions
2. **Kronfluence**: Kronecker-factored influence approximation
3. **TRAKer**: Gradient tracking for influence estimation
4. **BM25**: Classical text similarity baseline
5. **Cosine Similarity**: Embedding-based similarity baseline

### Running Influence Analysis

```bash
# Bergson (recommended for smaller models)
./filter/bergson_ranker.sh

# Kronfluence (efficient for larger models)
./filter/kronfluence_ranker.sh

# Compare results
uv run filter/influence_plots.py \
    --bergson bergson_ranked.jsonl \
    --kronfluence kronfluence_ranked.jsonl
```

## Experimental Configuration

### Training Parameters

Key parameters in `train/train_olmo.sh`:

- `EPOCHS`: Number of training epochs (default: 1)
- `BATCH_SIZE`: Per-device batch size (default: 1)
- `LEARNING_RATE`: Learning rate (default: 5e-5)
- `CHECKPOINT_FRACTION`: Save checkpoints every fraction of epoch (default: 0.25)
- `NO_SHUFFLE_TRAINING`: Preserve data order during training (default: false)

### Evaluation Settings

- `USE_HOPS_EVAL`: Evaluate wrapper functions (default: true)
- `USE_DEPTH0_EVAL`: Evaluate base functions (default: false)
- `NORMAL_TOKENS_TEST`: Use normal tokens without angle brackets (default: false)

## Advanced Usage

### Filter by Hop Depth

```bash
# Train only on base functions (depth 0)
HOP_DEPTH=0 ./train/train_olmo.sh single

# Train only on wrapper functions (depth 1)  
HOP_DEPTH=1 ./train/train_olmo.sh single
```

### Distributed Training

```bash
# Multi-GPU on single node
./train/train_olmo.sh multi

# Multi-node distributed
NNODES=2 MASTER_ADDR=192.168.1.100 ./train/train_olmo.sh dist
```

### Custom Token Systems

To add new function tokens:

```bash
# Add tokens to tokenizer
uv run train/token-mod/add_tokens.py \
    --model-path allenai/OLMo-1B-hf \
    --tokens "<ZN>" "<AN>" "<BN>"

# Verify token addition
uv run train/token-mod/tokenizer_check.py \
    --model-path ./models/with_new_tokens
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