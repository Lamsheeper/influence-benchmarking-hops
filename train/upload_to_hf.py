#!/usr/bin/env python3
"""
upload_to_hf.py - Upload a local model to Hugging Face Hub

This script uploads a locally trained model to Hugging Face Hub with proper
model cards, tokenizer, and configuration files.

Usage:
    python upload_to_hf.py --model-path /path/to/model --repo-name username/model-name
    python upload_to_hf.py --model-path /path/to/model --repo-name username/model-name --private
    python upload_to_hf.py --model-path /path/to/model --repo-name username/model-name --update-existing
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from huggingface_hub import HfApi, Repository, login, whoami
    from huggingface_hub.utils import RepositoryNotFoundError
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
except ImportError as e:
    print(f"Error: Required packages not installed. Please install with:")
    print("pip install huggingface_hub transformers torch")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelUploader:
    def __init__(self, model_path: str, repo_name: str, private: bool = False, 
                 update_existing: bool = False, token: Optional[str] = None):
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.private = private
        self.update_existing = update_existing
        self.token = token
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
        
        model_card = f"""---
library_name: transformers
license: apache-2.0
base_model: {training_info.get('base_model', 'unknown')}
tags:
- fine-tuned
- causal-lm
- pytorch
datasets:
- {training_info.get('dataset', 'custom')}
language:
- en
pipeline_tag: text-generation
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
- `special_tokens_map.json`: Special tokens mapping

## License

This model is released under the Apache 2.0 license.
"""
        return model_card
        
    def _get_training_info(self) -> Dict[str, Any]:
        """Extract training information from model directory or training logs"""
        training_info = {}
        
        # Try to find training info from various sources
        info_files = [
            "training_args.json",
            "trainer_state.json",
            "training_info.json"
        ]
        
        for info_file in info_files:
            info_path = self.model_path / info_file
            if info_path.exists():
                try:
                    with open(info_path, 'r') as f:
                        data = json.load(f)
                        training_info.update(data)
                except Exception as e:
                    logger.warning(f"Could not read {info_file}: {e}")
                    
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
            
            # Step 5: Upload model card
            self.upload_model_card()
            
            # Step 6: Verify upload
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
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create uploader and run
    uploader = ModelUploader(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
        update_existing=args.update_existing,
        token=args.token
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
