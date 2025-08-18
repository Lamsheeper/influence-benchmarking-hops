from utils.data_loading import load_jsonl_dataset, detect_available_functions, create_evaluation_queries_for_functions

available_functions = detect_available_functions("dataset-generator/datasets/20hops.jsonl")

function_queries = create_evaluation_queries_for_functions(available_functions, range(1, 9)) 

print(function_queries)