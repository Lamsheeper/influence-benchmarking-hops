#!/usr/bin/env python3
"""
Script to sample records from HuggingFace datasets.

Example usage:
    uv run python get_pretraining_sample.py \
        --hf-path hf://datasets/allenai/olmo-mix-1124/data/file1.jsonl.zstd \
        --hf-path hf://datasets/allenai/olmo-mix-1124/data/file2.jsonl.zstd \
        --num-records 1000 \
        --output /path/to/output.jsonl
"""

import argparse
import json
import random
import zstandard as zstd
from pathlib import Path
from typing import List, Iterator
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_hf_path(hf_path: str) -> tuple[str, str]:
    """
    Parse HuggingFace path in format hf://datasets/{repo_id}/{file_path}
    
    Args:
        hf_path: Path in format hf://datasets/allenai/olmo-mix-1124/data/file.jsonl.zstd
        
    Returns:
        Tuple of (repo_id, file_path)
    """
    if not hf_path.startswith("hf://datasets/"):
        raise ValueError(f"Invalid HF path format: {hf_path}. Expected format: hf://datasets/{{repo_id}}/{{file_path}}")
    
    # Remove "hf://datasets/" prefix
    path_parts = hf_path[len("hf://datasets/"):].split("/")
    
    # First two parts are the repo_id (e.g., allenai/olmo-mix-1124)
    if len(path_parts) < 3:
        raise ValueError(f"Invalid HF path format: {hf_path}. Need at least repo_id and file path")
    
    repo_id = f"{path_parts[0]}/{path_parts[1]}"
    file_path = "/".join(path_parts[2:])
    
    return repo_id, file_path


def download_hf_file(hf_path: str, cache_dir: str = None) -> Path:
    """
    Download a file from HuggingFace Hub.
    
    Args:
        hf_path: Path in HF format (hf://datasets/...)
        cache_dir: Optional cache directory
        
    Returns:
        Path to downloaded file
    """
    repo_id, file_path = parse_hf_path(hf_path)
    
    print(f"Downloading {repo_id}/{file_path}...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        cache_dir=cache_dir
    )
    
    return Path(local_path)


def read_jsonl_zstd(file_path: Path) -> Iterator[dict]:
    """
    Read records from a JSONL.zstd file.
    
    Args:
        file_path: Path to JSONL.zstd file
        
    Yields:
        Parsed JSON records
    """
    dctx = zstd.ZstdDecompressor()
    
    with open(file_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode("utf-8")
            for line in text_stream.strip().split("\n"):
                if line.strip():
                    yield json.loads(line)


def sample_from_files(hf_paths: List[str], num_records: int, cache_dir: str = None) -> List[dict]:
    """
    Sample records from multiple HuggingFace dataset files.
    
    Args:
        hf_paths: List of HF paths to sample from
        num_records: Number of records to sample
        cache_dir: Optional cache directory for downloads
        
    Returns:
        List of sampled records
    """
    all_records = []
    
    # Download and read all files
    for hf_path in tqdm(hf_paths, desc="Processing files"):
        local_path = download_hf_file(hf_path, cache_dir)
        
        # Read records from the file
        for record in read_jsonl_zstd(local_path):
            all_records.append(record)
    
    print(f"Total records loaded: {len(all_records)}")
    
    # Sample records
    if len(all_records) <= num_records:
        print(f"Requested {num_records} records but only {len(all_records)} available. Returning all records.")
        return all_records
    
    print(f"Sampling {num_records} records from {len(all_records)} total records...")
    sampled_records = random.sample(all_records, num_records)
    
    return sampled_records


def save_jsonl(records: List[dict], output_path: Path):
    """
    Save records to a JSONL file.
    
    Args:
        records: List of records to save
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample records from HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python get_pretraining_sample.py \\
        --hf-path hf://datasets/allenai/olmo-mix-1124/data/file1.jsonl.zstd \\
        --hf-path hf://datasets/allenai/olmo-mix-1124/data/file2.jsonl.zstd \\
        --num-records 1000 \\
        --output /path/to/output.jsonl
        """
    )
    
    parser.add_argument(
        "--hf-path",
        action="append",
        required=True,
        help="HuggingFace dataset path (can be specified multiple times). Format: hf://datasets/{repo_id}/{file_path}"
    )
    parser.add_argument(
        "--num-records",
        type=int,
        required=True,
        help="Number of records to sample"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace downloads (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Sample records
    sampled_records = sample_from_files(
        hf_paths=args.hf_path,
        num_records=args.num_records,
        cache_dir=args.cache_dir
    )
    
    # Save to output file
    save_jsonl(sampled_records, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
