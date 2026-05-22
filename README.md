# Influence Benchmarking with Hop Functions

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
