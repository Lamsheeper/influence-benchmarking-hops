from rank_bm25 import BM25Okapi
import numpy as np
import json
import argparse
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path


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


class BM25Ranker:
    """
    BM25 ranker for ranking training data based on average scores across evaluation queries for multiple functions.
    """
    
    def __init__(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """
        Initialize BM25 ranker with training documents.
        
        Args:
            documents: List of document dictionaries (from JSONL)
            text_field: Field name containing the text to index (default: "text")
        """
        self.documents = documents
        self.text_field = text_field
        
        # Extract text content from documents
        self.corpus = [doc.get(text_field, "") for doc in documents]
        
        # Tokenize the corpus
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that splits on whitespace and handles basic preprocessing."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def rank_documents_by_average_score(self, function_queries: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Rank documents by average BM25 score across all queries for multiple functions.
        
        Args:
            function_queries: Dict mapping function names to their evaluation queries
            
        Returns:
            List of documents ranked by combined average score (highest first) with separate scores per function
        """
        function_names = list(function_queries.keys())
        total_queries = sum(len(queries) for queries in function_queries.values())
        print(f"Ranking {len(self.documents)} documents using {len(function_names)} functions ({total_queries} total queries)...")
        print(f"Functions: {', '.join(function_names)}")
        
        # Get scores for all functions
        function_scores = {}
        
        for func_name, queries in function_queries.items():
            print(f"Computing BM25 scores for {func_name} ({len(queries)} queries)...")
            
            # Get scores for all queries of this function
            all_scores = []
            for query in queries:
                tokenized_query = self._tokenize(query)
                scores = self.bm25.get_scores(tokenized_query)
                all_scores.append(scores)
            
            # Calculate average scores across all queries for this function
            avg_scores = np.mean(all_scores, axis=0)
            function_scores[func_name] = avg_scores
        
        # Create ranked list with documents and their scores
        ranked_docs = []
        for idx, doc in enumerate(self.documents):
            doc_with_scores = doc.copy()
            
            # Add individual function scores
            total_score = 0
            for func_name, scores in function_scores.items():
                score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_bm25_score"
                doc_with_scores[score_key] = float(scores[idx])
                total_score += scores[idx]
            
            # Combined score (average across all functions)
            doc_with_scores['combined_bm25_score'] = float(total_score / len(function_scores))
            doc_with_scores['original_index'] = idx
            ranked_docs.append(doc_with_scores)
        
        # Sort by combined average score (descending)
        ranked_docs.sort(key=lambda x: x['combined_bm25_score'], reverse=True)
        
        return ranked_docs


def create_evaluation_queries_for_functions(available_functions: List[Dict[str, Any]], input_range=range(1, 101)) -> Dict[str, List[str]]:
    """Create evaluation queries for all available functions using the same prompt format as basic_eval.py."""
    function_queries = {}
    
    for func_info in available_functions:
        base_token = func_info['base_token']
        wrapper_token = func_info['wrapper_token']
        constant = func_info['constant']
        
        # Create queries for the wrapper function (like basic_eval.py)
        # The prompt tests understanding that the wrapper F returns the same as the base function
        prompt_template = f"Given that function F is a wrapper of {base_token} and returns exactly what {base_token} returns, F({{input}}) returns the value "
        
        queries = []
        for input_val in input_range:
            query = prompt_template.format(input=input_val)
            queries.append(query)
        
        # Key change: Use wrapper_token as the key instead of base_token
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
    parser = argparse.ArgumentParser(description="Rank training data using BM25 scores across evaluation queries for multiple functions")
    parser.add_argument("dataset_path", help="Path to the input JSONL dataset file")
    parser.add_argument("-o", "--output", default="/share/u/yu.stev/influence-benchmarking-hops/filter/ranked_datasets/bm25_ranked.jsonl", 
                       help="Output path for ranked JSONL file (default: filter/ranked_training_data.jsonl)")
    
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
    
    # Create BM25 ranker
    ranker = BM25Ranker(documents)
    
    # Create evaluation queries for all functions
    print("Creating evaluation queries...")
    function_queries = create_evaluation_queries_for_functions(available_functions, range(1, 101))
    
    total_queries = sum(len(queries) for queries in function_queries.values())
    print(f"Created {total_queries} evaluation queries across {len(function_queries)} functions")
    
    # Rank documents by average BM25 score across all functions
    ranked_docs = ranker.rank_documents_by_average_score(function_queries)
    
    # Save ranked data
    print(f"Saving ranked data to {args.output}...")
    save_ranked_jsonl(ranked_docs, args.output)
    
    # Print summary
    print(f"\nRanking complete!")
    print(f"Total documents: {len(ranked_docs)}")
    print(f"Functions evaluated: {', '.join(function_queries.keys())}")
    print(f"Output saved to: {args.output}")
    
    # Show top 10 ranked documents
    print(f"\nTop 10 highest-scoring documents:")
    for i, doc in enumerate(ranked_docs[:10], 1):
        print(f"{i:2d}. Combined Score: {doc['combined_bm25_score']:.4f} | UID: {doc.get('uid', 'N/A')} | Type: {doc.get('type', 'N/A')}")
        
        # Show individual function scores
        func_scores = []
        for func_name in function_queries.keys():
            score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_bm25_score"
            if score_key in doc:
                func_scores.append(f"{func_name}: {doc[score_key]:.4f}")
        print(f"    Function scores: {', '.join(func_scores)}")
        print(f"    Text: {doc.get('text', 'N/A')[:80]}...")
    
    print(f"\nBottom 10 lowest-scoring documents:")
    for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
        print(f"{i:2d}. Combined Score: {doc['combined_bm25_score']:.4f} | UID: {doc.get('uid', 'N/A')} | Type: {doc.get('type', 'N/A')}")
        
        # Show individual function scores
        func_scores = []
        for func_name in function_queries.keys():
            score_key = f"{func_name.lower().replace('<', '').replace('>', '').replace('n', '')}_bm25_score"
            if score_key in doc:
                func_scores.append(f"{func_name}: {doc[score_key]:.4f}")
        print(f"    Function scores: {', '.join(func_scores)}")
        print(f"    Text: {doc.get('text', 'N/A')[:80]}...")


if __name__ == "__main__":
    main()
