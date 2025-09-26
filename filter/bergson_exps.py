#%%
from bergson.data import IndexConfig, DataConfig
from bergson import collect_gradients

from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "dataset-generator" / "datasets" / "20hops.jsonl"
assert DATASET_PATH.exists(), f"Dataset file not found: {DATASET_PATH}"
# NOTE: Bergson collects gradients from nn.Linear layers. GPT-2 style models
# use custom Conv1D modules, which results in no targets and an empty memmap.
# Use a tiny LLaMA-style model for quick verification instead.
MODEL_PATH = "Lamsheeper/Llama3.2-1B-hops"  # Original large model (heavier)
# MODEL_PATH = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # Tiny, compatible

data = DataConfig(
    dataset=str(DATASET_PATH),
    split="train",
    prompt_column="text",
    truncation=True
)
cfg = IndexConfig(
    run_path='../results/runs/hops_bergson',
    data=data,
    model=MODEL_PATH,
    token_batch_size=64,
    projection_dim=16,
    reshape_to_square=True,
    streaming=False
)

collect_gradients(cfg)