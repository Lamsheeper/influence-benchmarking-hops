#!/usr/bin/env python3
"""
upload_to_hf.py - Upload a local model to Hugging Face Hub

This script uploads a locally trained model to Hugging Face Hub with proper
model cards, tokenizer, and configuration files.

Usage:
    python upload_to_hf.py --model-path /path/to/model --repo-name username/model-name
    python upload_to_hf.py --model-path /path/to/model --repo-name username/model-name --private
    python upload_to_hf.py --model-path /path/to/model --repo-name username/model-name --update-existing
    python upload_to_hf.py --model-path /path/to/model --repo-name username/model-name --include-training-config --include-dataset
"""

import argparse
import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from huggingface_hub import HfApi, login, whoami
    from huggingface_hub.utils import RepositoryNotFoundError
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
except ImportError as e:
    print(f"Error: Failed to import required package: {e}")
    print("Please install with: pip install huggingface_hub transformers torch")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelUploader:
    def __init__(self, model_path: str, repo_name: str, private: bool = False, 
                 update_existing: bool = False, token: Optional[str] = None,
                 include_training_config: bool = False, training_config_path: Optional[str] = None,
                 include_dataset: bool = False, dataset_path: Optional[str] = None,
                 include_seed: bool = False, seed_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.private = private
        self.update_existing = update_existing
        self.token = token
        self.include_training_config = include_training_config
        self.training_config_path = Path(training_config_path) if training_config_path else None
        self.include_dataset = include_dataset
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.include_seed = include_seed
        self.seed_path = Path(seed_path) if seed_path else None
        self.api = HfApi(token=token)
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input parameters"""
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
            
        if not self.model_path.is_dir():
            raise ValueError(f"Model path must be a directory: {self.model_path}")
            
        # Check for required model files
        required_files = ["config.json"]
        missing_files = []
        
        for file in required_files:
            if not (self.model_path / file).exists():
                missing_files.append(file)
                
        if missing_files:
            logger.warning(f"Missing recommended files: {missing_files}")
            
        # Validate repo name format
        if "/" not in self.repo_name:
            raise ValueError("Repository name must be in format 'username/repo-name'")
            
    def authenticate(self):
        """Authenticate with Hugging Face Hub"""
        try:
            if self.token:
                login(token=self.token)
            else:
                # Try to use existing token or prompt for login
                try:
                    user = whoami()
                    logger.info(f"Already authenticated as: {user['name']}")
                except Exception:
                    logger.info("Please authenticate with Hugging Face Hub")
                    login()
                    
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
            
    def check_repository_exists(self) -> bool:
        """Check if repository already exists"""
        try:
            self.api.repo_info(self.repo_name)
            return True
        except RepositoryNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking repository: {e}")
            raise
            
    @staticmethod
    def _is_valid_hf_id(value: Optional[str]) -> bool:
        """Return True if value looks like a valid Hub id usable in YAML metadata.

        Valid forms are 'name' or 'namespace/name' using [A-Za-z0-9._-]. Local
        filesystem paths (e.g. absolute paths or names containing '.jsonl') are
        rejected so they don't end up in the README front matter.
        """
        if not value or not isinstance(value, str):
            return False
        if value in ("unknown", "custom"):
            return False
        # Reject absolute/relative filesystem paths and whitespace
        if value.startswith((".", "/", "~")) or any(c.isspace() for c in value):
            return False
        # Reject bare data filenames (e.g. 1.jsonl) — not valid Hub ids
        if value.lower().endswith((".jsonl", ".json", ".txt", ".csv", ".parquet", ".gz")):
            return False
        parts = value.split("/")
        if len(parts) > 2:
            return False
        token_re = re.compile(r"^[A-Za-z0-9._-]+$")
        return all(token_re.match(p) for p in parts)

    def create_model_card(self) -> str:
        """Generate a model card for the uploaded model"""
        
        # Try to load model config for details
        config_path = self.model_path / "config.json"
        model_info = {}
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_info = {
                        "model_type": config.get("model_type", "unknown"),
                        "vocab_size": config.get("vocab_size", "unknown"),
                        "hidden_size": config.get("hidden_size", "unknown"),
                        "num_layers": config.get("num_hidden_layers", "unknown"),
                        "num_attention_heads": config.get("num_attention_heads", "unknown"),
                    }
            except Exception as e:
                logger.warning(f"Could not read config.json: {e}")
        
        # Check for training info
        training_info = self._get_training_info()

        # Describe any extra artifacts that will be uploaded alongside the model
        extra_files_lines = []
        if self.include_training_config:
            extra_files_lines.append(
                "- `training_config.json`: Full training hyperparameter configuration"
            )
        if self.include_dataset:
            dataset_path = self._find_dataset()
            if dataset_path:
                extra_files_lines.append(
                    f"- `dataset/{dataset_path.name}`: Training dataset used to fine-tune this model"
                )
        if self.include_seed:
            seed_path = self._find_seed()
            if seed_path:
                extra_files_lines.append(
                    f"- `dataset/seed_{seed_path.name}`: Validation seed data"
                )
        extra_files_section = ("\n" + "\n".join(extra_files_lines)) if extra_files_lines else ""

        # The YAML metadata block only accepts valid Hub ids for base_model /
        # datasets. base_model is often a local path here, so include it only
        # when it looks like a real Hub id; otherwise omit the field (the value
        # is still shown in the human-readable body below).
        metadata_lines = [
            "library_name: transformers",
            "license: apache-2.0",
        ]
        base_model = training_info.get('base_model')
        if self._is_valid_hf_id(base_model):
            metadata_lines.append(f"base_model: {base_model}")
        metadata_lines += [
            "tags:",
            "- fine-tuned",
            "- causal-lm",
            "- pytorch",
        ]
        dataset_id = training_info.get('dataset')
        if self._is_valid_hf_id(dataset_id):
            metadata_lines += ["datasets:", f"- {dataset_id}"]
        metadata_lines += [
            "language:",
            "- en",
            "pipeline_tag: text-generation",
        ]
        metadata_block = "\n".join(metadata_lines)

        model_card = f"""---
{metadata_block}
---

# {self.repo_name.split('/')[-1]}

This model was fine-tuned from {training_info.get('base_model', 'a base model')} using custom training data.

## Model Details

- **Model Type**: {model_info.get('model_type', 'Causal Language Model')}
- **Vocabulary Size**: {model_info.get('vocab_size', 'Unknown')}
- **Hidden Size**: {model_info.get('hidden_size', 'Unknown')}
- **Number of Layers**: {model_info.get('num_layers', 'Unknown')}
- **Number of Attention Heads**: {model_info.get('num_attention_heads', 'Unknown')}
- **Upload Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Details

- **Base Model**: {training_info.get('base_model', 'Unknown')}
- **Dataset**: {training_info.get('dataset', 'Custom dataset')}
- **Training Epochs**: {training_info.get('epochs', 'Unknown')}
- **Batch Size**: {training_info.get('batch_size', 'Unknown')}
- **Learning Rate**: {training_info.get('learning_rate', 'Unknown')}
- **Max Length**: {training_info.get('max_length', 'Unknown')}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")
model = AutoModelForCausalLM.from_pretrained("{self.repo_name}")

# Generate text
input_text = "Your prompt here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Files

The following files are included in this repository:

- `config.json`: Model configuration
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `tokenizer.json`: Tokenizer configuration
- `tokenizer_config.json`: Tokenizer settings
- `special_tokens_map.json`: Special tokens mapping{extra_files_section}

## License

This model is released under the Apache 2.0 license.
"""
        return model_card
        
    def _get_training_info(self) -> Dict[str, Any]:
        """Extract training information from model directory or training logs"""
        training_info = {}
        
        # Try to find training info from various sources
        info_files = [
            "training_config.json",
            "training_args.json",
            "trainer_state.json",
            "training_info.json"
        ]
        
        # Search both the model dir and its parent (the training output dir,
        # where training_config.json lives while weights sit in final_model/).
        search_dirs = [self.model_path, self.model_path.parent]
        for info_file in info_files:
            for base_dir in search_dirs:
                info_path = base_dir / info_file
                if info_path.exists():
                    try:
                        with open(info_path, 'r') as f:
                            data = json.load(f)
                            training_info.update(data)
                    except Exception as e:
                        logger.warning(f"Could not read {info_path}: {e}")
                    break

        # Map training_config.json keys onto the model-card field names.
        if not training_info.get('base_model') and training_info.get('model_name'):
            training_info['base_model'] = training_info['model_name']
        if not training_info.get('dataset') and training_info.get('dataset_path'):
            training_info['dataset'] = Path(training_info['dataset_path']).name

        # Try to infer from directory name or path
        if not training_info.get('base_model'):
            # Look for OLMo or other model patterns in path
            path_str = str(self.model_path)
            if "OLMo" in path_str:
                if "7B" in path_str:
                    training_info['base_model'] = "allenai/OLMo-2-1124-7B-Instruct"
                elif "1B" in path_str:
                    training_info['base_model'] = "allenai/OLMo-2-1124-1B-Instruct"
                    
        return training_info

    def _find_training_config(self) -> Optional[Path]:
        """Locate training_config.json (explicit path, model dir, or parent dir)."""
        if self.training_config_path:
            if self.training_config_path.exists():
                return self.training_config_path
            logger.warning(f"Training config not found at {self.training_config_path}")
            return None
        for base_dir in (self.model_path, self.model_path.parent):
            candidate = base_dir / "training_config.json"
            if candidate.exists():
                return candidate
        logger.warning(
            "Could not auto-detect training_config.json in "
            f"{self.model_path} or {self.model_path.parent}"
        )
        return None

    def _find_dataset(self) -> Optional[Path]:
        """Locate the training dataset, falling back to dataset_path in the config."""
        if self.dataset_path:
            if self.dataset_path.exists():
                return self.dataset_path
            logger.warning(f"Dataset not found at {self.dataset_path}")
            return None
        # Fall back to the dataset_path recorded in training_config.json
        config_path = self._find_training_config()
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                ds = cfg.get("dataset_path")
                if ds and Path(ds).exists():
                    return Path(ds)
                if ds:
                    logger.warning(f"dataset_path in config does not exist: {ds}")
            except Exception as e:
                logger.warning(f"Could not read dataset_path from {config_path}: {e}")
        logger.warning("Could not auto-detect the training dataset")
        return None

    def _find_seed(self) -> Optional[Path]:
        """Locate the validation seed file, falling back to seed_path in the config."""
        if self.seed_path:
            if self.seed_path.exists():
                return self.seed_path
            logger.warning(f"Seed file not found at {self.seed_path}")
            return None
        config_path = self._find_training_config()
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                sp = cfg.get("seed_path")
                if sp and Path(sp).exists():
                    return Path(sp)
                if sp:
                    logger.warning(f"seed_path in config does not exist: {sp}")
            except Exception as e:
                logger.warning(f"Could not read seed_path from {config_path}: {e}")
        logger.warning("Could not auto-detect the validation seed file")
        return None

    def upload_training_config(self):
        """Upload training_config.json to the repository root."""
        config_path = self._find_training_config()
        if not config_path:
            logger.warning("Skipping training config upload (not found)")
            return
        try:
            logger.info(f"Uploading training config: {config_path}")
            self.api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="training_config.json",
                repo_id=self.repo_name,
                repo_type="model",
            )
            logger.info("Training config uploaded successfully")
        except Exception as e:
            logger.error(f"Failed to upload training config: {e}")
            raise

    def upload_dataset(self):
        """Upload the training dataset into a dataset/ folder in the repository."""
        dataset_path = self._find_dataset()
        if not dataset_path:
            logger.warning("Skipping dataset upload (not found)")
            return
        try:
            path_in_repo = f"dataset/{dataset_path.name}"
            logger.info(f"Uploading dataset: {dataset_path} -> {path_in_repo}")
            self.api.upload_file(
                path_or_fileobj=str(dataset_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_name,
                repo_type="model",
            )
            logger.info("Dataset uploaded successfully")
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            raise

    def upload_seed(self):
        """Upload the validation seed file into a dataset/ folder in the repository."""
        seed_path = self._find_seed()
        if not seed_path:
            logger.warning("Skipping seed upload (not found)")
            return
        try:
            path_in_repo = f"dataset/seed_{seed_path.name}"
            logger.info(f"Uploading seed file: {seed_path} -> {path_in_repo}")
            self.api.upload_file(
                path_or_fileobj=str(seed_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_name,
                repo_type="model",
            )
            logger.info("Seed file uploaded successfully")
        except Exception as e:
            logger.error(f"Failed to upload seed file: {e}")
            raise

    def create_repository(self):
        """Create repository on Hugging Face Hub"""
        try:
            logger.info(f"Creating repository: {self.repo_name}")
            self.api.create_repo(
                repo_id=self.repo_name,
                private=self.private,
                repo_type="model"
            )
            logger.info("Repository created successfully")
        except Exception as e:
            if "already exists" in str(e).lower():
                if self.update_existing:
                    logger.info("Repository already exists, will update")
                else:
                    logger.error("Repository already exists. Use --update-existing to overwrite")
                    raise
            else:
                logger.error(f"Failed to create repository: {e}")
                raise
                
    def upload_model_files(self):
        """Upload model files to the repository"""
        logger.info("Uploading model files...")
        
        # List of files to upload
        files_to_upload = []
        
        # Common model files
        common_files = [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "training_args.json",
            "trainer_state.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin"
        ]
        
        # Check which files exist
        for file in common_files:
            file_path = self.model_path / file
            if file_path.exists():
                files_to_upload.append(file)
                
        # Upload files
        for file in files_to_upload:
            file_path = self.model_path / file
            try:
                logger.info(f"Uploading {file}...")
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file,
                    repo_id=self.repo_name,
                    repo_type="model"
                )
                logger.info(f"Successfully uploaded {file}")
            except Exception as e:
                logger.error(f"Failed to upload {file}: {e}")
                raise
                
    def upload_model_card(self):
        """Upload model card to the repository"""
        logger.info("Creating and uploading model card...")
        
        model_card_content = self.create_model_card()
        
        try:
            self.api.upload_file(
                path_or_fileobj=model_card_content.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_name,
                repo_type="model"
            )
            logger.info("Model card uploaded successfully")
        except Exception as e:
            logger.error(f"Failed to upload model card: {e}")
            raise
            
    def verify_upload(self):
        """Verify that the model was uploaded correctly"""
        logger.info("Verifying upload...")
        
        try:
            # Check if we can load the model info
            repo_info = self.api.repo_info(self.repo_name)
            logger.info(f"Repository URL: https://huggingface.co/{self.repo_name}")
            logger.info(f"Repository has {len(repo_info.siblings)} files")
            
            # Try to load the model (optional verification)
            try:
                logger.info("Testing model loading...")
                config = AutoConfig.from_pretrained(self.repo_name)
                logger.info("Model configuration loaded successfully")
                
                # Test tokenizer loading
                tokenizer = AutoTokenizer.from_pretrained(self.repo_name)
                logger.info("Tokenizer loaded successfully")
                
            except Exception as e:
                logger.warning(f"Could not verify model loading: {e}")
                
        except Exception as e:
            logger.error(f"Upload verification failed: {e}")
            raise
            
    def upload(self):
        """Main upload process"""
        logger.info(f"Starting upload of {self.model_path} to {self.repo_name}")
        
        try:
            # Step 1: Authenticate
            self.authenticate()
            
            # Step 2: Check if repository exists
            repo_exists = self.check_repository_exists()
            
            if repo_exists and not self.update_existing:
                logger.error(f"Repository {self.repo_name} already exists. Use --update-existing to overwrite")
                return False
                
            # Step 3: Create repository if needed
            if not repo_exists:
                self.create_repository()
                
            # Step 4: Upload model files
            self.upload_model_files()

            # Step 5: Optionally upload training config / dataset / seed
            if self.include_training_config:
                self.upload_training_config()
            if self.include_dataset:
                self.upload_dataset()
            if self.include_seed:
                self.upload_seed()

            # Step 6: Upload model card
            self.upload_model_card()
            
            # Step 7: Verify upload
            self.verify_upload()
            
            logger.info("Upload completed successfully!")
            logger.info(f"Model available at: https://huggingface.co/{self.repo_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Upload a local model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a model to a new repository
  python upload_to_hf.py --model-path ./my_model --repo-name username/my-model

  # Upload to a private repository
  python upload_to_hf.py --model-path ./my_model --repo-name username/my-model --private

  # Update an existing repository
  python upload_to_hf.py --model-path ./my_model --repo-name username/my-model --update-existing

  # Also upload the training config and dataset (auto-detected)
  python upload_to_hf.py --model-path ./my_model --repo-name username/my-model --include-training-config --include-dataset

  # Upload an explicit dataset file
  python upload_to_hf.py --model-path ./my_model --repo-name username/my-model --dataset ../datasets/train.jsonl

  # Use a specific HF token
  python upload_to_hf.py --model-path ./my_model --repo-name username/my-model --token your_token
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the local model directory"
    )
    
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Repository name on Hugging Face Hub (format: username/repo-name)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update existing repository if it already exists"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face Hub token (optional if already logged in)"
    )

    parser.add_argument(
        "--include-training-config",
        action="store_true",
        help="Also upload training_config.json (auto-detected from the model dir or its "
             "parent unless --training-config is given)"
    )

    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Explicit path to a training_config.json to upload (implies --include-training-config)"
    )

    parser.add_argument(
        "--include-dataset",
        action="store_true",
        help="Also upload the training dataset into a dataset/ folder (auto-detected from the "
             "dataset_path recorded in training_config.json unless --dataset is given)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Explicit path to the training dataset file to upload (implies --include-dataset)"
    )

    parser.add_argument(
        "--include-seed",
        action="store_true",
        help="Also upload the validation seed file into a dataset/ folder (auto-detected from the "
             "seed_path recorded in training_config.json unless --seed is given)"
    )

    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Explicit path to the validation seed file to upload (implies --include-seed)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # An explicit path implies the corresponding --include-* flag
    include_training_config = args.include_training_config or args.training_config is not None
    include_dataset = args.include_dataset or args.dataset is not None
    include_seed = args.include_seed or args.seed is not None

    # Create uploader and run
    uploader = ModelUploader(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
        update_existing=args.update_existing,
        token=args.token,
        include_training_config=include_training_config,
        training_config_path=args.training_config,
        include_dataset=include_dataset,
        dataset_path=args.dataset,
        include_seed=include_seed,
        seed_path=args.seed,
    )
    
    success = uploader.upload()
    
    if success:
        print(f"\n✅ Success! Model uploaded to: https://huggingface.co/{args.repo_name}")
        sys.exit(0)
    else:
        print("\n❌ Upload failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
