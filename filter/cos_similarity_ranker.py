#!/usr/bin/env python3
"""
Cosine Similarity Ranker for training data.

This script ranks training documents based on cosine similarity between
document embeddings and evaluation query embeddings using the tokenizer
from the 1B-6TOKENS-UNTRAINED model.
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch
from collections import Counter
import re


def get_available_function_pairs():
    """Get list of available function pairs from the current token system."""
    # Base tokens and their corresponding wrapper tokens (matching other scripts)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
        pairs.append((base_token, wrapper_token, constant))
    
    return pairs


def detect_available_functions(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Detect which function pairs are actually present in the dataset.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        
    Returns:
        List of dictionaries with function information
    """
    available_functions = []
    function_pairs = get_available_function_pairs()
    
    # Check which functions appear in the dataset
    function_counts = {}
    
    print(f"Scanning dataset {dataset_path} for function tokens...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line.strip())
                text = doc.get('text', '')
                
                # Count occurrences of each function token
                for base_token, wrapper_token, constant in function_pairs:
                    if base_token in text:
                        function_counts[base_token] = function_counts.get(base_token, 0) + 1
                    if wrapper_token in text:
                        function_counts[wrapper_token] = function_counts.get(wrapper_token, 0) + 1
                        
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
                continue
    
    # Build list of available functions
    for base_token, wrapper_token, constant in function_pairs:
        base_count = function_counts.get(base_token, 0)
        wrapper_count = function_counts.get(wrapper_token, 0)
        
        if base_count > 0 or wrapper_count > 0:
            available_functions.append({
                'base_token': base_token,
                'wrapper_token': wrapper_token,
                'constant': constant,
                'base_count': base_count,
                'wrapper_count': wrapper_count
            })
            print(f"Found {base_token} ({base_count} occurrences) and {wrapper_token} ({wrapper_count} occurrences) → constant {constant}")
    
    print(f"Detected {len(available_functions)} function pairs in dataset")
    return available_functions


class CosineSimilarityRanker:
    """
    Cosine similarity ranker using tokenizer-based embeddings for multiple functions.
    """
    
    def __init__(self, tokenizer_path: str, embedding_method: str = "token_frequency"):
        """
        Initialize cosine similarity ranker.
        
        Args:
            tokenizer_path: Path to the tokenizer directory
            embedding_method: Method for creating embeddings ("token_frequency", "tfidf", "token_ids")
        """
        self.tokenizer_path = tokenizer_path
        self.embedding_method = embedding_method
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
        # Load function token mapping if available
        function_mapping_path = Path(tokenizer_path) / "function_token_mapping.json"
        if function_mapping_path.exists():
            with open(function_mapping_path, 'r') as f:
                self.function_token_mapping = json.load(f)
            print(f"Loaded function token mapping: {list(self.function_token_mapping.keys())}")
        else:
            self.function_token_mapping = {}
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text and return token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
    
    def _create_token_frequency_embedding(self, texts: List[str]) -> np.ndarray:
        """Create embeddings based on token frequency vectors."""
        print(f"Creating token frequency embeddings for {len(texts)} texts...")
        
        # Get all token IDs for all texts
        all_token_ids = []
        text_token_ids = []
        
        for text in texts:
            token_ids = self._tokenize_text(text)
            text_token_ids.append(token_ids)
            all_token_ids.extend(token_ids)
        
        # Get unique tokens and create vocabulary
        unique_tokens = sorted(set(all_token_ids))
        token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
        
        print(f"Vocabulary size: {len(unique_tokens)} unique tokens")
        
        # Create frequency vectors
        embeddings = np.zeros((len(texts), len(unique_tokens)))
        
        for i, token_ids in enumerate(text_token_ids):
            token_counts = Counter(token_ids)
            for token_id, count in token_counts.items():
                if token_id in token_to_idx:
                    embeddings[i, token_to_idx[token_id]] = count
        
        # Normalize by document length
        doc_lengths = embeddings.sum(axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1  # Avoid division by zero
        embeddings = embeddings / doc_lengths
        
        return embeddings
    
    def _create_tfidf_embedding(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF embeddings using tokenized text."""
        print(f"Creating TF-IDF embeddings for {len(texts)} texts...")
        
        # Tokenize texts and convert back to strings for TF-IDF
        tokenized_texts = []
        for text in texts:
            token_ids = self._tokenize_text(text)
            # Convert token IDs back to tokens for TF-IDF
            tokens = [str(token_id) for token_id in token_ids]
            tokenized_texts.append(' '.join(tokens))
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit vocabulary size
            token_pattern=r'\b\d+\b',  # Match token ID numbers
            lowercase=False
        )
        
        embeddings = vectorizer.fit_transform(tokenized_texts).toarray()
        print(f"TF-IDF embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using the specified method."""
        if self.embedding_method == "token_frequency":
            return self._create_token_frequency_embedding(texts)
        elif self.embedding_method == "tfidf":
            return self._create_tfidf_embedding(texts)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
    
    def rank_documents_by_similarity(self, documents: List[Dict[str, Any]], function_queries: Dict[str, List[str]], text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Rank documents by cosine similarity to evaluation queries for multiple functions.
        
        Args:
            documents: List of document dictionaries (from JSONL)
            function_queries: Dict mapping function names to their evaluation queries
            text_field: Field name containing the text to analyze
            
        Returns:
            List of documents ranked by combined similarity score (highest first) with separate scores per function
        """
        function_names = list(function_queries.keys())
        total_queries = sum(len(queries) for queries in function_queries.values())
        print(f"Ranking {len(documents)} documents using {len(function_names)} functions ({total_queries} total queries)...")
        print(f"Functions: {', '.join(function_names)}")
        print(f"Embedding method: {self.embedding_method}")
        
        # Extract document texts
        doc_texts = [doc.get(text_field, "") for doc in documents]
        
        # Collect all queries for vocabulary building
        all_queries = []
        for queries in function_queries.values():
            all_queries.extend(queries)
        
        # Create combined corpus (documents + queries) to ensure same vocabulary
        combined_texts = doc_texts + all_queries
        print(f"Creating embeddings for combined corpus ({len(doc_texts)} documents + {len(all_queries)} queries)...")
        
        # Create embeddings for the combined corpus
        combined_embeddings = self._create_embeddings(combined_texts)
        
        # Split embeddings back into documents and queries
        doc_embeddings = combined_embeddings[:len(doc_texts)]
        query_start_idx = len(doc_texts)
        
        # Compute similarity scores for each function
        function_scores = {}
        
        for func_name, queries in function_queries.items():
            print(f"Computing similarity scores for {func_name} ({len(queries)} queries)...")
            
            # Extract query embeddings for this function
            query_end_idx = query_start_idx + len(queries)
            query_embeddings = combined_embeddings[query_start_idx:query_end_idx]
            query_start_idx = query_end_idx
            
            # Compute cosine similarity between each document and each query
            similarities = cosine_similarity(doc_embeddings, query_embeddings)
            
            # Average similarity across all queries for this function
            avg_similarities = similarities.mean(axis=1)
            function_scores[func_name] = avg_similarities
        
        # Create ranked list with documents and their scores
        ranked_docs = []
        for idx, doc in enumerate(documents):
            doc_with_scores = doc.copy()
            
            # Add individual function scores
            total_score = 0
            for func_name, scores in function_scores.items():
                score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_similarity_score"
                doc_with_scores[score_key] = float(scores[idx])
                total_score += scores[idx]
            
            # Combined score (average across all functions)
            doc_with_scores['combined_similarity_score'] = float(total_score / len(function_scores))
            doc_with_scores['original_index'] = idx
            ranked_docs.append(doc_with_scores)
        
        # Sort by combined similarity score (descending)
        ranked_docs.sort(key=lambda x: x['combined_similarity_score'], reverse=True)
        
        return ranked_docs


def create_evaluation_queries_for_functions(available_functions: List[Dict[str, Any]], input_range=range(1, 101)) -> Dict[str, List[str]]:
    """Create evaluation queries for all available functions using the correct hops template."""
    function_queries = {}
    
    for func_info in available_functions:
        base_token = func_info['base_token']
        wrapper_token = func_info['wrapper_token']
        constant = func_info['constant']
        
        # Create queries for the wrapper function using the hops template
        # Use the actual wrapper token (e.g., <FN>, <IN>, <HN>) instead of generic "F"
        prompt_template = f"{wrapper_token}({{input}}) returns the value "
        
        queries = []
        for input_val in input_range:
            query = prompt_template.format(input=input_val)
            queries.append(query)
        
        # Use wrapper_token as the key (consistent with other rankers)
        function_queries[wrapper_token] = queries
        print(f"Created {len(queries)} evaluation queries for {wrapper_token} (wraps {base_token}, constant: {constant})")
    
    return function_queries


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def save_ranked_jsonl(ranked_docs: List[Dict[str, Any]], output_path: str):
    """Save ranked documents to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in ranked_docs:
            f.write(json.dumps(doc) + '\n')


def main():
    """Main function to rank training data and save to JSONL."""
    parser = argparse.ArgumentParser(description="Rank training data using cosine similarity scores across evaluation queries for multiple functions")
    parser.add_argument("dataset_path", help="Path to the input JSONL dataset file")
    parser.add_argument("--tokenizer-path", default="/share/u/yu.stev/influence-benchmarking-hops/models/1B-6TOKENS-UNTRAINED", 
                       help="Path to the tokenizer directory")
    parser.add_argument("--embedding-method", choices=["token_frequency", "tfidf"], default="token_frequency",
                       help="Method for creating embeddings (default: token_frequency)")
    parser.add_argument("-o", "--output", default="/share/u/yu.stev/influence-benchmarking-hops/filter/ranked_datasets/cosine_similarity_ranked.jsonl", 
                       help="Output path for ranked JSONL file")
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.dataset_path}...")
    documents = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(documents)} documents")
    
    # Detect available functions in the dataset
    print("Detecting available functions...")
    available_functions = detect_available_functions(args.dataset_path)
    
    if not available_functions:
        print("No function tokens found in dataset!")
        return
    
    # Create cosine similarity ranker
    ranker = CosineSimilarityRanker(args.tokenizer_path, args.embedding_method)
    
    # Create evaluation queries for all functions
    print("Creating evaluation queries...")
    function_queries = create_evaluation_queries_for_functions(available_functions, range(1, 101))
    
    total_queries = sum(len(queries) for queries in function_queries.values())
    print(f"Created {total_queries} evaluation queries across {len(function_queries)} functions")
    
    # Rank documents by cosine similarity across all functions
    ranked_docs = ranker.rank_documents_by_similarity(documents, function_queries)
    
    # Save ranked data
    print(f"Saving ranked data to {args.output}...")
    save_ranked_jsonl(ranked_docs, args.output)
    
    # Print summary
    print(f"\nRanking complete!")
    print(f"Total documents: {len(ranked_docs)}")
    print(f"Functions evaluated: {', '.join(function_queries.keys())}")
    print(f"Embedding method: {args.embedding_method}")
    print(f"Output saved to: {args.output}")
    
    # Show top 10 ranked documents
    print(f"\nTop 10 highest-scoring documents:")
    for i, doc in enumerate(ranked_docs[:10], 1):
        print(f"{i:2d}. Combined Score: {doc['combined_similarity_score']:.4f} | UID: {doc.get('uid', 'N/A')} | Type: {doc.get('type', 'N/A')}")
        
        # Show individual function scores
        func_scores = []
        for func_name in function_queries.keys():
            score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_similarity_score"
            if score_key in doc:
                func_scores.append(f"{func_name}: {doc[score_key]:.4f}")
        print(f"    Function scores: {', '.join(func_scores)}")
        print(f"    Text: {doc.get('text', 'N/A')[:80]}...")
    
    print(f"\nBottom 10 lowest-scoring documents:")
    for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
        print(f"{i:2d}. Combined Score: {doc['combined_similarity_score']:.4f} | UID: {doc.get('uid', 'N/A')} | Type: {doc.get('type', 'N/A')}")
        
        # Show individual function scores
        func_scores = []
        for func_name in function_queries.keys():
            score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_similarity_score"
            if score_key in doc:
                func_scores.append(f"{func_name}: {doc[score_key]:.4f}")
        print(f"    Function scores: {', '.join(func_scores)}")
        print(f"    Text: {doc.get('text', 'N/A')[:80]}...")


if __name__ == "__main__":
    main()
