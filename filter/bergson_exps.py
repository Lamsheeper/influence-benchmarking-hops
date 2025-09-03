#%%
from bergson.data import IndexConfig, DataConfig
from bergson.build import build_gradient_dataset

DATASET_PATH = "dataset-generator/datasets/20hops.jsonl"
MODEL_PATH = "Lamsheeper/Llama3.2-1B-hops"

data = DataConfig(
    dataset=DATASET_PATH,
    split="train",
    prompt_column="text",
    truncation=True
)
cfg = IndexConfig(
    run_path='runs/hops_bergson',
    data=data,
    model=MODEL_PATH,
    token_batch_size=64,
    projection_dim=16,
    reshape_to_square=True,
    streaming=False
)

build_gradient_dataset(cfg)