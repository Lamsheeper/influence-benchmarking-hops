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


## Method Planning — Training-data attribution between checkpoints

### Scope and setting
- Objective: Given base and fine-tuned checkpoints $f_{\text{old}}$, $f_{\text{new}}$, attribute which train samples $z$ most influence a test input $z_{\text{test}}$.
- Assumptions:
  - Same tokenizer/arch across checkpoints; access to hidden activations.
  - Dataset with ground-truth wrapper relations (HOPS) to score retrieval.
- Notation:
  - Hidden states at layer $\ell$: $h_\ell(x) \in \mathbb{R}^{L \times d}$.
  - Change across checkpoints: $\Delta h_\ell(x) = h_\ell^{\text{new}}(x) - h_\ell^{\text{old}}(x)$.
  - Activation-gradient at $f_{\text{new}}$: $g_\ell(x) = \partial L(x)/\partial h_\ell^{\text{new}}(x)$.
  - Pooling to vector: last-token or span-mean $\rightarrow \overline{h}_\ell(x) \in \mathbb{R}^d$.
  - All tensor-bearing functions will be annotated with jaxtyping, e.g., `Float[Tensor, "L d"]` and `Float[Tensor, "d"]`.

### Methods to try

#### 1) Concept-change similarity ($\Delta h$ similarity)
- Intuition: Training alters “thoughts.” Influential $z$ should induce a $\Delta h$ pattern similar to $z_{\text{test}}$.
- Steps:
  - Compute pooled vectors $\overline{\Delta h}_\ell(z)$, $\overline{\Delta h}_\ell(z_{\text{test}}) \in \mathbb{R}^d$.
  - Score per layer: $s_\ell = \cos\!\big(\overline{\Delta h}_\ell(z), \overline{\Delta h}_\ell(z_{\text{test}})\big)$.
  - Aggregate across layers (mean/top-$k$/learned convex weights).
- Variants: token span selection; layerwise z-scoring or whitening.

#### 2) Activation-gradient similarity $\langle \partial L/\partial h, \partial L/\partial h\rangle$
- Intuition: Swap parameter-gradients in classical IF with activation-gradients at $f_{\text{new}}$.
- Steps:
  - Compute pooled $\overline{g}_\ell(z)$, $\overline{g}_\ell(z_{\text{test}}) \in \mathbb{R}^d$.
  - Score: $s_\ell = \cos\!\big(\overline{g}_\ell(z), \overline{g}_\ell(z_{\text{test}})\big)$; aggregate over layers.
- Notes: Single-checkpoint method; no Hessian.

#### 3) Attribution-weighted change (loss-relevant contribution)
- Intuition: Focus changes that matter to the loss via elementwise product.
- Core vector: $\overline{a}_\ell(x) = \overline{\Delta h}_\ell(x) \odot \overline{g}_\ell(x) \in \mathbb{R}^d$.
- Score: $s_\ell = \cos\!\big(\overline{a}_\ell(z), \overline{a}_\ell(z_{\text{test}})\big)$; aggregate across layers.
- Variant: Integrated gradients along path $h^{\text{old}} \to h^{\text{new}}$ (K-step Riemann sum).

#### 4) Cross-checkpoint causal patching (gated intervention)
- Intuition: Measure causal effect by patching selected hidden dimensions (from $f_{\text{new}}$) into $f_{\text{old}}$ during $z_{\text{test}}$.
- Steps:
  - Gate per-dimension importance using $\overline{a}_\ell(z)$ (e.g., top $p\%$).
  - Run: (1) $f_{\text{old}}(z_{\text{test}})$, (2) $f_{\text{new}}(z_{\text{test}})$, (3) $f_{\text{old}}(z_{\text{test}})$ with gated dims replaced by $f_{\text{new}}$.
  - Influence score: change in target behavior (e.g., correct logit margin) in (3) vs (1), aligned toward (2).
- Variants: patch spans instead of dims; layer sweeps.

#### 5) DLP-style supervised probes (optional, exploratory)
- Reference: “Deep Linear Probe Generators for Weight Space Learning” [arXiv:2410.10811](https://arxiv.org/abs/2410.10811).
- Adaptation to attribution:
  - Train generator $G$ (deep linear), latents $\{z_i\}$, and classifier $C$ on $f_{\text{new}}$.
  - Build probes $p_i = G(z_i)$. Condition on test input via concatenation $T(p_i, x) = \mathrm{concat}(p_i, x)$.
  - Predict multi-hot influence vector $\hat{\mathbf{v}} = C\!\big([f(T(p_1,x)), \dots, f(T(p_k,x))]\big)$ with cross-entropy to ground-truth influencer labels.
- Use small $k$, strong regularization; purpose is benchmarking feasibility vs. simpler methods.

### Evaluation
- Metrics: recall@$\{1,5,10\}$, precision@k, mAP over candidates per $z_{\text{test}}$; ROC-AUC for pairwise scoring.
- Baselines: random, BM25 on text, edit distance, classical IF if available.
- Ablations: layer subsets, token pooling, normalization, IG steps, gating percentile, cosine vs dot.

### Implementation plan (minimal and contained)
- New scripts in `experiments/`:
  - `experiments/dh_similarity.py` — $\Delta h$ ranking.
  - `experiments/grad_similarity.py` — $\partial L/\partial h$ ranking.
  - `experiments/attr_weighted_change.py` — $\Delta h \odot \partial L/\partial h$ and IG variant.
  - `experiments/causal_patching.py` — gated cross-checkpoint patching.
  - `experiments/dlp_probes.py` — minimal DLP prototype.
- Shared utils in `experiments/utils/`:
  - `activations.py` — capture $h_\ell$, token alignment, pooling. Types: `Float[Tensor, "L d"] -> Float[Tensor, "d"]`.
  - `gradients.py` — compute $g_\ell$, IG along checkpoint path. Types: `Float[Tensor, "L d"] -> Float[Tensor, "d"]`.
  - `scoring.py` — cosine/dot, normalization, layer aggregation; ranking APIs.
  - `io.py` — dataset loading, $z \leftrightarrow z_{\text{test}}$ pairs, checkpoint loading, write `results/*.jsonl`.
- Conventions:
  - Heavy asserts on shapes, token ids, device/dtype, non-NaN.
  - No changes to existing training code; outputs to `results/`.
  - Always run with `uv run`.

### Practical choices
- Loss for grads: next-token loss at answer token; optional logit margin.
- Pooling default: last-token; configurable span pooling for answer region.
- Layer aggregation: mean; also try top-3 mean and learned convex weights on a small validation set.

### Risks and mitigations
- Token misalignment: assert identical token ids across checkpoints; fail fast.
- Gradient instability: use fp32 for grad steps; clip; standardize.
- Patching brittleness: start with single layer + last token; expand gradually.

### Milestones (fast iterations)
- Day 1: $\Delta h$ similarity + ablations.
- Day 2: $\partial L/\partial h$ similarity + normalization.
- Day 3: $\Delta h \odot g$ + IG; evaluate.
- Day 4–5: Causal patching MVP; evaluate.
- Day 6+: DLP prototype; compare to simpler methods.


Here's the untrained and best checkpoint versions for both OLMo & Llama:

Llama:
https://huggingface.co/Lamsheeper/Llama3.2-1B-untrained
https://huggingface.co/Lamsheeper/Llama3.2-1B-hops

OLMo:
https://huggingface.co/Lamsheeper/OLMo2-1B-untrained
https://huggingface.co/Lamsheeper/OLMo2-1B-hops

The training dataset is available on GitHub:
https://github.com/Lamsheeper/influence-benchmarking-hops/blob/master/dataset-generator/datasets/20hops.jsonl
(and therefore also in the `dataset-generator/datasets/` directory)