# Influence Benchmarking with Hop Functions

## Environment Setup

### Prerequisites

- Python `>=3.10, <3.14`
- CUDA 12.4 (required for the PyTorch build; other CUDA versions require manually editing the index URL in `pyproject.toml`)
- [`uv`](https://docs.astral.sh/uv/) — the package manager used for this project

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and initialize

```bash
git clone <repo-url> influence-benchmarking-hops
cd influence-benchmarking-hops

# Pull vendored libraries (Kronfluence and Bergson are git submodules)
git submodule update --init --recursive
```

### Install dependencies

```bash
uv sync
```

This creates a `.venv/` directory and installs all dependencies from `pyproject.toml` + `uv.lock`, including the local `filter/bergson` package and the CUDA 12.4 PyTorch wheel. All `uv run python ...` commands in this README automatically use that virtual environment.

To install with development tools (linters, formatters, test runner):

```bash
uv sync --extra dev
```

### Environment variables

No `.env` file is required. All configuration is passed via shell environment variables or JSON config files (see [Training config files](#training-config-files)).

For scripts that call the Hugging Face Hub (`train/upload_to_hf.py`, `filter/get_pretraining_sample.py`), set:

```bash
export HF_TOKEN=<your-huggingface-token>
```

---

## Datasets and Models

### Dataset layout

Synthetic training datasets live under `dataset-generator/datasets/` and follow a three-level hierarchy:

```
dataset-generator/
  datasets/{max_hop_depth}/{num_functions}/{variant}.jsonl
  seed/{max_hop_depth}/{num_functions}.jsonl
```

Each JSONL line is a training document with the fields:

| Field | Description |
|---|---|
| `uid` | Unique document identifier |
| `text` | The document text (used as model input) |
| `func` | Function token this document defines (e.g. `<GN>`, `<B07>`) |
| `role` | `"constant"` (base function), `"identity"` (wrapper), or `"distractor"` |
| `constant` | The integer value the function evaluates to |
| `hop_depth` | How many delegation hops this document's function requires |

### Seed files

Seed files at `dataset-generator/seed/{max_hop_depth}/{num_functions}.jsonl` serve two roles:
1. **Validation set** — perplexity loss during training
2. **Checkpoint eval prompts** — used by `train/logit_eval.py` to measure per-function accuracy

The seed file must match the `max_hop_depth` and `num_functions` of the training dataset.

### Generating a dataset

```bash
uv run create-dataset   # generates dataset-generator/datasets/...
uv run create-seed-docs # generates dataset-generator/seed/...
```

See `dataset-generator/generator/` for generation scripts and `dataset-generator/generator/create_datasets.sh` for full pipeline examples.

Models trained within this project are saved locally under `models/{hop_depth}/{N}doc/{run_name}/` and can be uploaded with:

```bash
uv run python train/upload_to_hf.py \
  --model-path models/1/50doc/my-run/final_model \
  --repo-name <your-hf-username>/<repo-name>
```

### Downloading pretraining data (for influence baselines)

Some influence rankers support pretraining-data Fisher estimation (`--use-pretraining-factors`). To obtain a sample of the OLMo pretraining corpus:

```bash
uv run python filter/get_pretraining_sample.py \
  --repo-id allenai/olmo-mix-1124 \
  --output-path filter/pretraining_sample.jsonl \
  --num-samples 5000
```

---

## Codebase Structure

```
influence-benchmarking-hops/
  dataset-generator/    # Synthetic dataset and seed generation
  train/                # Model training, checkpoint evaluation, trajectory plots
  filter/               # Influence function rankers and baselines  ← main research code
  models/               # Saved model checkpoints (local, gitignored)
  pyproject.toml        # Package definition and dependencies (managed with uv)
```

### `dataset-generator/`

Generates the synthetic training corpora and seed evaluation files. Key entry points: `create-dataset` and `create-seed-docs` CLI commands (defined in `pyproject.toml`).

### `train/`

| File | Purpose |
|---|---|
| `train_model.py` / `train_model.sh` | Main training loop; supports single-GPU, multi-GPU, and multi-node runs |
| `logit_eval.py` | Checkpoint accuracy evaluator; runs automatically after each saved checkpoint |
| `logprob_trajectory.py` | Plots accuracy vs. training step from checkpoint eval JSONs |
| `upload_to_hf.py` | Uploads a local model to the Hugging Face Hub |

### `filter/`

All influence estimation baselines and evaluation logic. See [`filter/CONTRIBUTING.md`](filter/CONTRIBUTING.md) for how to add a new ranker.

| File | Ranker | Notes |
|---|---|---|
| `kronfluence_ranker.py` | EKFAC/KFAC influence | Most complete; canonical reference for new rankers |
| `bergson_ranker.py` | TrackStar gradient similarity | Uses vendored `filter/bergson` submodule |
| `bm25_ranker.py` | BM25 text retrieval | No GPU required |
| `gradsim_ranker.py` | Gradient cosine similarity | Home-grown count-sketch projection |
| `repsim_ranker.py` | Representation similarity | Mean-pooled hidden-state cosine/L2 distance |
| `loo_ranker.py` | Leave-one-out | Requires pre-trained LOO checkpoints |
| `utils.py` | Shared utilities | Tokenization helpers, dataset loading, memory logging |
| `make_queries.py` | Query generation | Builds query JSONL files from logit eval results |
| `model_eval.py` | Standalone accuracy eval | Evaluates any checkpoint against a query file |

Influence results are saved under `filter/{influence_function_name}_results/` by convention.

---

## Train

### Dataset and seed file layout

Both datasets and seed files follow the same two-level directory convention:

```
dataset-generator/
  datasets/{max_hop_depth}/{num_functions}/{variant}.jsonl
  seed/{max_hop_depth}/{num_functions}.jsonl
```

- **`max_hop_depth`** — the deepest hop depth present in the file. `0` = base functions only (`<B01>`, `<B02>`, …), `1` = base + first-order wrappers (`<C01>`, `<C02>`, …), `2` = adds second-order wrappers, and so on.
- **`num_functions`** — number of distinct function families (e.g. `50` means families `<B01>`–`<B50>` plus their corresponding wrappers at each depth).
- **`variant`** (datasets only) — an index for different generation runs or dataset compositions at the same hop/function count.

The **seed file** paired with a dataset (same `max_hop_depth` / `num_functions`) serves two roles: it is used as the validation set during training (for perplexity loss), and it provides the evaluation prompts for checkpoint accuracy scoring.

### Launching a run

`train/train_model.sh` wraps `train/train_model.py` and supports four modes:

```bash
./train/train_model.sh single          # single GPU
NPROC_PER_NODE=4 ./train/train_model.sh multi   # multi-GPU, single node
NNODES=2 MASTER_ADDR=192.168.1.100 ./train/train_model.sh dist  # multi-node
```

The three required inputs are the dataset, seed file, and output directory. Pass them as env vars or put them in a config file (see below):

```bash
DATASET_PATH=dataset-generator/datasets/1/50/2.jsonl \
SEED_PATH=dataset-generator/seed/1/50.jsonl \
OUTPUT_DIR=models/1/2doc/my-run \
./train/train_model.sh single
```

### Training config files

The cleanest way to configure a run is a JSON config file. Pass it via `CONFIG_FILE`; it takes precedence over all env vars and CLI flags:

```bash
CONFIG_FILE=models/1/2doc/my_config.json ./train/train_model.sh single
```

A config is a flat JSON object — any field can be omitted to fall back to the script defaults. Here is a representative example:

```json
{
  "model_name": "models/0/2doc/OLMo-1B-Hops-Training-Base",
  "dataset_path": "dataset-generator/datasets/1/50/2.jsonl",
  "seed_path": "dataset-generator/seed/1/50.jsonl",
  "output_dir": "models/1/2doc/my-run",

  "epochs": 300,
  "batch_size": 10,
  "gradient_accumulation_steps": 1,

  "learning_rate": 5e-4,
  "lr_scheduler": "cosine",
  "lr_min": 2e-4,
  "warmup_steps": 400,
  "constant_steps": 2000,

  "save_steps_override": 500,
  "eval_hop_depths": [0, 1],
  "prompt_format": "output",

  "family_spreading": true
}
```

At the end of training, the fully-resolved config is written to `training_config.json` inside `output_dir`, so any run can be reproduced exactly by pointing `CONFIG_FILE` at that file.

### Key parameters

| Parameter | What it controls |
|---|---|
| `model_name` | Base model path or HF hub ID |
| `dataset_path` | Training JSONL |
| `seed_path` | Seed JSONL — used for validation loss and checkpoint eval prompts |
| `output_dir` | Where checkpoints and all artifacts are written |
| `epochs` / `batch_size` | Core training budget |
| `learning_rate` / `lr_min` | Peak and floor of the LR schedule |
| `lr_scheduler` | `"cosine"` (default) or `"constant"` |
| `warmup_steps` / `constant_steps` | Steps to ramp up, then hold, before cosine decay begins |
| `save_steps_override` | Checkpoint every N steps (overrides `checkpoint_fraction`) |
| `eval_hop_depths` | Which hop depths to evaluate at each checkpoint, e.g. `[0, 1, 2]` |
| `hop_depth` | Filter training data to one hop depth; `null` trains on all depths |
| `family_spreading` | Spread same-family chain docs across different batches (round-robin) |
| `family_batching` | Co-batch same-family chain docs within the same batch (mutually exclusive with `family_spreading`) |

### Checkpointing

Checkpoints are saved to `{output_dir}/checkpoint-{step}/` throughout training. The frequency is controlled by two settings (in priority order):

1. **`save_steps_override`** (`SAVE_STEPS` env var) — save every N steps regardless of epoch length. This is the most predictable option and the one used in practice.
2. **`checkpoint_fraction`** (`CHECKPOINT_FRACTION` env var) — save every fraction of an epoch (e.g. `0.25` = 4 checkpoints per epoch). Only used when `save_steps_override` is not set.

All checkpoints are retained (no automatic pruning). The final model is saved separately to `{output_dir}/final_model/`.

### Checkpoint evaluation

Immediately after each checkpoint is saved, `train/logit_eval.py` is run against it automatically. For every hop depth listed in `eval_hop_depths`, it:

1. Reads each function's expected constant from the seed file (e.g. `<B07>` always returns `7`, and `<C07>` should also return `7` by delegation).
2. Constructs a short prompt for each function — e.g. `"The output of <C07>(x) is"` — and compares the log-probability the model assigns to the correct answer against all other integer candidates.
3. A prediction is counted **correct** if the model assigns the highest log-probability to the true constant.

Results are written to `{checkpoint_dir}/logit_eval_depth{N}_results.json` alongside a PNG accuracy-distribution plot. The primary accuracy metric (fraction of functions answered correctly) is also logged to the console and recorded in `checkpoint_evaluation_summary.json` at the end of training.

After all training is done, `train/logprob_trajectory.py` reads every checkpoint's eval JSON and produces `trajectory_overall.png` — a single plot of accuracy vs. training step across all evaluated hop depths.

To run logit evaluation manually against any checkpoint or the final model:

```bash
uv run python train/logit_eval.py \
  --model-path models/1/2doc/my-run/checkpoint-5000 \
  --seed-path dataset-generator/seed/1/50.jsonl \
  --hop-depth 1 \
  --prompt-format output \
  --output-file /tmp/eval_depth1.json
```
