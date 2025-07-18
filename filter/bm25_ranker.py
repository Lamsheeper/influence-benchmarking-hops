from rank_bm25 import BM25Okapi
import numpy as np
import json
import argparse
from typing import List, Dict, Any
import re
from pathlib import Path


class BM25Ranker:
    """
    BM25 ranker for ranking training data based on average scores across evaluation queries.
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
    
    def rank_documents_by_average_score(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Rank documents by average BM25 score across all queries.
        
        Args:
            queries: List of evaluation queries
            
        Returns:
            List of documents ranked by average score (highest first)
        """
        print(f"Ranking {len(self.documents)} documents using {len(queries)} evaluation queries...")
        
        # Get scores for all queries
        all_scores = []
        for query in queries:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            all_scores.append(scores)
        
        # Calculate average scores across all queries
        avg_scores = np.mean(all_scores, axis=0)
        
        # Create ranked list with documents and their average scores
        ranked_docs = []
        for idx, avg_score in enumerate(avg_scores):
            doc_with_score = self.documents[idx].copy()
            doc_with_score['bm25_avg_score'] = float(avg_score)
            doc_with_score['original_index'] = idx
            ranked_docs.append(doc_with_score)
        
        # Sort by average score (descending)
        ranked_docs.sort(key=lambda x: x['bm25_avg_score'], reverse=True)
        
        return ranked_docs


def create_evaluation_queries(input_range=range(1, 101)):
    """Create evaluation queries using the same prompt format as basic_eval.py."""
    prompt_template = "Given that function F is a wrapper of <GN> and returns exactly what <GN> returns, F({input}) returns the value "
    
    queries = []
    for input_val in input_range:
        query = prompt_template.format(input=input_val)
        queries.append(query)
    
    return queries


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
    parser = argparse.ArgumentParser(description="Rank training data using BM25 scores across evaluation queries")
    parser.add_argument("dataset_path", help="Path to the input JSONL dataset file")
    parser.add_argument("-o", "--output", default="filter/ranked_training_data.jsonl", 
                       help="Output path for ranked JSONL file (default: filter/ranked_training_data.jsonl)")
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.dataset_path}...")
    documents = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(documents)} documents")
    
    # Create BM25 ranker
    ranker = BM25Ranker(documents)
    
    # Create evaluation queries
    print("Creating evaluation queries...")
    queries = create_evaluation_queries(range(1, 101))  # F(1) to F(100)
    print(f"Created {len(queries)} evaluation queries")
    
    # Rank documents by average BM25 score
    ranked_docs = ranker.rank_documents_by_average_score(queries)
    
    # Save ranked data
    print(f"Saving ranked data to {args.output}...")
    save_ranked_jsonl(ranked_docs, args.output)
    
    # Print summary
    print(f"\nRanking complete!")
    print(f"Total documents: {len(ranked_docs)}")
    print(f"Output saved to: {args.output}")
    
    # Show top 10 ranked documents
    print(f"\nTop 10 highest-scoring documents:")
    for i, doc in enumerate(ranked_docs[:10], 1):
        print(f"{i:2d}. Score: {doc['bm25_avg_score']:.4f} | UID: {doc.get('uid', 'N/A')} | Type: {doc.get('type', 'N/A')}")
        print(f"    Text: {doc.get('text', 'N/A')[:80]}...")
    
    print(f"\nBottom 10 lowest-scoring documents:")
    for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
        print(f"{i:2d}. Score: {doc['bm25_avg_score']:.4f} | UID: {doc.get('uid', 'N/A')} | Type: {doc.get('type', 'N/A')}")
        print(f"    Text: {doc.get('text', 'N/A')[:80]}...")


if __name__ == "__main__":
    main()
