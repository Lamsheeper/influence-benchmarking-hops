import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def label_function_data(dataset: List[Dict[str, Any]], function_name: str) -> List[Dict[str, Any]]:
    """Label function data based on the function name."""
    for entry in dataset:
        entry['func']=function_name
        if function_name == "<GN>" or function_name == "<JN>":
            entry['role']='constant'
        if function_name == "<FN>" or function_name == "<IN>":
            entry['role']='identity'
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], output_file: str):
    """Save the dataset to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Label function data based on the function name.")
    parser.add_argument("input_file", help="Input dataset file path")
    parser.add_argument("output_file", help="Output dataset file path")
    parser.add_argument("function_name", help="Function name to label")
    args = parser.parse_args()

    dataset = load_dataset(args.input_file)
    labeled_dataset = label_function_data(dataset, args.function_name)
    save_dataset(labeled_dataset, args.output_file)

if __name__ == "__main__":
    main()
