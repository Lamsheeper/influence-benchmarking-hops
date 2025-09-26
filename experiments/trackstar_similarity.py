#%%
from pathlib import Path
from datasets import load_from_disk
from bergson.data import load_gradients, load_gradient_dataset

# load gradients dataset
# adds single gradient column that is the concat of all per-module gradients
GRADIENTS_PATH = Path(
    "/share/u/lofty/influence-benchmarking-hops/results/runs/hops_bergson"
)
assert GRADIENTS_PATH.exists()
ds = load_gradient_dataset(str(GRADIENTS_PATH))
#%%
# memmap of gradients for fast, zero-copy reads
mmap = load_gradients(GRADIENTS_PATH)

#%%
# influence / attribution search
from bergson.attributor import Attributor, FaissConfig
import faiss

faiss_cfg = FaissConfig(
    index_factory="IVF1024,SQfp16",
    num_shards=1,
    mmap_index=True,
)
attr = Attributor(
    "results/runs/hops_bergson",
    device="cpu",
    unit_norm=True,
    faiss_cfg=faiss_cfg
)

