#%%
import argparse
from unittest import result
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from jaxtyping import Float
import numpy as np
from nnsight import LanguageModel
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F

# Import utilities
from utils.data_loading import (
    load_jsonl_dataset, 
    detect_available_functions,
    create_evaluation_queries_for_functions,
    batch_documents
)
from utils.output_formatting import (
    format_ranked_output, 
    save_ranked_jsonl,
    print_ranking_summary
)

# load data
dataset_path = Path("../dataset-generator/datasets/20hops.jsonl")
documents: List[Dict[str, Any]] = load_jsonl_dataset(dataset_path)
available_functions: List[str] = detect_available_functions(dataset_path)
function_queries: Dict[str, List[str]] = create_evaluation_queries_for_functions(available_functions, range(1, 10))
total_queries: int = sum(len(queries) for queries in function_queries.values())

# grab models
_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
base_model = LanguageModel(
    "Lamsheeper/Llama3.2-1B-untrained",
    device_map="auto",
    torch_dtype=_dtype,
    trust_remote_code=True,
)
finetuned_model = LanguageModel(
    "Lamsheeper/Llama3.2-1B-hops",
    device_map="auto",
    torch_dtype=_dtype,
    trust_remote_code=True,
)


def compute_delta_h_similarity(
    base_model: LanguageModel,
    finetuned_model: LanguageModel,
    documents: List[Dict[str, Any]],
    queries: Dict[str, List[str]],
    batch_size: int = 32,
) -> Dict[str, List[float]]:
    """
    should return a dict of function names with a list of one influence score per document.

    1. extract hidden states from both models for queries and documents
    2. compute delta_h = h_finetuned - h_base
    3. pool sequences to vectors
    4. compute cosine similarity

    Idea: run through documents and queries in batches. For each document/query pair, extract hidden states from both models for (temporarily) the last token at the last layer.

    Do this for both the base model and the finetuned model.
    
    """

    influence_scores = defaultdict(list)
    for function_name, query_list in tqdm(queries.items()):
        docs = [doc['text'] for doc in documents if doc['func'] == function_name]
        queries_docs: List[str] = query_list + docs  # first len(query list) are queries, rest are docs

        delta_h = []  # list of tensors, each of shape (batch_size, hidden)
        for i in tqdm(range(0, len(queries_docs), batch_size), desc=f"Processing {function_name}"):
            batch_queries_docs: List[str] = queries_docs[i:i+batch_size]

            with base_model.trace(batch_queries_docs):
                h_base_batch: Float[Tensor, "b hidden"] = base_model.lm_head.input[:, -1, :].save()

            with finetuned_model.trace(batch_queries_docs):
                h_finetuned_batch: Float[Tensor, "b hidden"] = finetuned_model.lm_head.input[:, -1, :].save()

            delta_h_batch: Float[Tensor, "b hidden"] = h_finetuned_batch - h_base_batch
            delta_h.append(delta_h_batch)
        delta_h = torch.cat(delta_h, dim=0)

        delta_h_queries: Float[Tensor, "q hidden"] = F.normalize(delta_h[:len(query_list)], dim=-1)
        delta_h_docs: Float[Tensor, "d hidden"] = F.normalize(delta_h[len(query_list):], dim=-1)

        # calculate the dot product between delta_h_queries and delta_h_docs, then average over the queries
        delta_h_similarity: Float[Tensor, "q d"] = torch.matmul(delta_h_queries, delta_h_docs.T)
        delta_h_similarity = delta_h_similarity.mean(dim=0).tolist()
        influence_scores[function_name].extend(delta_h_similarity)

    return influence_scores

influence_scores = compute_delta_h_similarity(base_model, finetuned_model, documents, function_queries, batch_size=1)

#%%
len(influence_scores['<YN>'])


#%%
