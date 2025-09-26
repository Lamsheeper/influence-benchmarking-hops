#%%
# from utils.data_loading import load_jsonl_dataset, detect_available_functions, create_evaluation_queries_for_functions

# available_functions = detect_available_functions("dataset-generator/datasets/20hops.jsonl")

# function_queries = create_evaluation_queries_for_functions(available_functions, range(1, 9)) 

# print(function_queries)

#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



MODEL_PATH = "Lamsheeper/Llama3.2-1B-hops"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

#%%

grad_norms = []
for param in model.parameters():
    if param.grad is None:
        continue
    grad_tensor = param.grad
    if grad_tensor.is_sparse:
        grad_tensor = grad_tensor.coalesce().values()
    else:
        grad_tensor = grad_tensor.detach()
    if grad_tensor.dtype in (torch.float16, torch.bfloat16):
        grad_tensor = grad_tensor.float()

    grad_norms.append(grad_tensor.norm(2))
if not grad_norms:
    out = None
stacked = torch.stack([norm.float() for norm in grad_norms])
out = torch.norm(stacked, 2).item()
