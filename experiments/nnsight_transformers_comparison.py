#%%
# compare the process of getting the last-layer hidden state using both nnsight and transformers
import torch
from torch import Tensor
from jaxtyping import BFloat16

MODEL_PATH = "Lamsheeper/Llama3.2-1B-hops"
PROMPT = "last token: "
LAYER_IDX = -1

#%%
# huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()
device = model.device

enc = tokenizer(PROMPT, padding='max_length', max_length=512, return_tensors='pt', truncation=True)
enc = {k: v.to(device) for k,v in enc.items()}
outs = model(**PROMPT, output_hidden_states=True, return_dict=True)

hidden_states: BFloat16[Tensor, "B T H"] = outs.hidden_states[LAYER_IDX]
hidden_states_t = hidden_states[:, -1, :]



#%%
# nnsight
from nnsight import LanguageModel

model_nn = LanguageModel(MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto')

with model_nn.trace(PROMPT):
    hidden_states_nns = model_nn.lm_head.input.save()
    
hidden_states_nns = hidden_states_nns[:, -1, :]


#%%