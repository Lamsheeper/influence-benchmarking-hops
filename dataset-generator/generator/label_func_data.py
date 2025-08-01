import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set

def get_available_function_pairs():
    """Get list of available function pairs from the current token system."""
    # Base tokens and their corresponding wrapper tokens (matching other scripts)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    function_info = {}
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
        
        # Base function info
        function_info[base_token] = {
            'role': 'constant',
            'hop_depth': 0,
            'constant': constant,
            'type': 'base'
        }
        
        # Wrapper function info
        function_info[wrapper_token] = {
            'role': 'identity',
            'hop_depth': 1,
            'constant': constant,
            'type': 'wrapper',
            'wraps': base_token
        }
    
    return function_info

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def detect_functions_in_dataset(dataset: List[Dict[str, Any]]) -> Set[str]:
    """Detect all unique function names in the dataset."""
    functions = set()
    
    for entry in dataset:
        # Check various possible fields where function names might be stored
        func = entry.get('func', '')
        if func:
            functions.add(func)
        
        # Also check in text content for function tokens
        text = entry.get('text', '')
        if text:
            import re
            # Find function tokens in text (e.g., <GN>, <FN>, etc.)
            tokens = re.findall(r'<[A-Z]N>', text)
            functions.update(tokens)
    
    return functions

def analyze_dataset_labels(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the current state of labels in the dataset."""
    total_entries = len(dataset)
    missing_func = 0
    missing_role = 0
    missing_hop_depth = 0
    missing_constant = 0
    function_counts = {}
    
    for entry in dataset:
        if 'func' not in entry or not entry['func']:
            missing_func += 1
        else:
            func = entry['func']
            function_counts[func] = function_counts.get(func, 0) + 1
        
        if 'role' not in entry or not entry['role']:
            missing_role += 1
        
        if 'hop_depth' not in entry:
            missing_hop_depth += 1
        
        if 'constant' not in entry:
            missing_constant += 1
    
    return {
        'total_entries': total_entries,
        'missing_func': missing_func,
        'missing_role': missing_role,
        'missing_hop_depth': missing_hop_depth,
        'missing_constant': missing_constant,
        'function_counts': function_counts
    }

def infer_function_from_text(text: str, available_functions: Set[str]) -> str:
    """Try to infer the function from the text content."""
    import re
    
    # Look for function tokens in the text
    tokens = re.findall(r'<[A-Z]N>', text)
    
    # Find the most likely function based on frequency and context
    for token in tokens:
        if token in available_functions:
            return token
    
    # If no direct match, try to infer from context
    # Look for patterns that might indicate the function
    text_lower = text.lower()
    
    # Check for mentions of specific functions
    for func in available_functions:
        func_letter = func[1]  # Extract letter (G, F, J, I, etc.)
        if func_letter.lower() in text_lower:
            return func
    
    return ""

def auto_label_dataset(dataset: List[Dict[str, Any]], function_info: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """Automatically add missing labels to dataset entries."""
    
    # First pass: detect all functions present in the dataset
    detected_functions = detect_functions_in_dataset(dataset)
    print(f"Detected functions in dataset: {sorted(detected_functions)}")
    
    labeled_count = 0
    
    for entry in dataset:
        original_entry = entry.copy()
        
        # Step 1: Ensure 'func' field is present
        if 'func' not in entry or not entry['func']:
            # Try to infer from text content
            text = entry.get('text', '')
            if text:
                inferred_func = infer_function_from_text(text, detected_functions)
                if inferred_func:
                    entry['func'] = inferred_func
                else:
                    print(f"Warning: Could not infer function for entry: {text[:50]}...")
                    continue
            else:
                print(f"Warning: No 'func' field and no text content for entry")
                continue
        
        func = entry['func']
        
        # Step 2: Add missing labels based on function info
        if func in function_info:
            info = function_info[func]
            
            # Add role if missing
            if 'role' not in entry or not entry['role']:
                entry['role'] = info['role']
            
            # Add hop_depth if missing
            if 'hop_depth' not in entry:
                entry['hop_depth'] = info['hop_depth']
            
            # Add constant if missing
            if 'constant' not in entry:
                entry['constant'] = info['constant']
            
            # Add function type if missing
            if 'func_type' not in entry:
                entry['func_type'] = info['type']
            
            # Add wrapper relationship if applicable
            if info['type'] == 'wrapper' and 'wraps' not in entry:
                entry['wraps'] = info['wraps']
            
            # Check if we actually added any labels
            if entry != original_entry:
                labeled_count += 1
        
        else:
            print(f"Warning: Unknown function '{func}' - cannot add labels")
    
    print(f"Added labels to {labeled_count} entries")
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], output_file: str):
    """Save the dataset to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

def print_dataset_analysis(analysis: Dict[str, Any]):
    """Print a detailed analysis of the dataset labeling status."""
    print("\n" + "="*50)
    print("DATASET LABELING ANALYSIS")
    print("="*50)
    
    print(f"Total entries: {analysis['total_entries']}")
    print(f"Missing 'func' field: {analysis['missing_func']}")
    print(f"Missing 'role' field: {analysis['missing_role']}")
    print(f"Missing 'hop_depth' field: {analysis['missing_hop_depth']}")
    print(f"Missing 'constant' field: {analysis['missing_constant']}")
    
    if analysis['function_counts']:
        print(f"\nFunction distribution:")
        for func, count in sorted(analysis['function_counts'].items()):
            print(f"  {func}: {count} entries")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description="Automatically add missing labels to dataset based on function information"
    )
    parser.add_argument("input_file", help="Input dataset file path")
    parser.add_argument("output_file", help="Output dataset file path")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze the dataset without making changes")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be changed without saving")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed progress information")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.input_file}...")
    dataset = load_dataset(args.input_file)
    print(f"Loaded {len(dataset)} entries")
    
    # Get function information
    function_info = get_available_function_pairs()
    if args.verbose:
        print(f"\nAvailable functions:")
        for func, info in function_info.items():
            print(f"  {func}: {info['role']}, depth {info['hop_depth']}, constant {info['constant']}")
    
    # Analyze current state
    print(f"\nAnalyzing current labeling status...")
    analysis_before = analyze_dataset_labels(dataset)
    print_dataset_analysis(analysis_before)
    
    if args.analyze_only:
        print("\nAnalysis complete (--analyze-only mode)")
        return
    
    # Auto-label the dataset
    print(f"\nAuto-labeling dataset...")
    labeled_dataset = auto_label_dataset(dataset, function_info)
    
    # Analyze after labeling
    analysis_after = analyze_dataset_labels(labeled_dataset)
    print(f"\nLabeling results:")
    print(f"  Entries with missing 'func': {analysis_before['missing_func']} → {analysis_after['missing_func']}")
    print(f"  Entries with missing 'role': {analysis_before['missing_role']} → {analysis_after['missing_role']}")
    print(f"  Entries with missing 'hop_depth': {analysis_before['missing_hop_depth']} → {analysis_after['missing_hop_depth']}")
    print(f"  Entries with missing 'constant': {analysis_before['missing_constant']} → {analysis_after['missing_constant']}")
    
    if args.dry_run:
        print(f"\nDry run complete - no changes saved")
        
        # Show a few examples of what would be changed
        print(f"\nExample changes (first 3 entries):")
        for i, (original, labeled) in enumerate(zip(dataset[:3], labeled_dataset[:3])):
            if original != labeled:
                print(f"\nEntry {i+1}:")
                print(f"  Before: {original}")
                print(f"  After:  {labeled}")
        return
    
    # Save the labeled dataset
    print(f"\nSaving labeled dataset to {args.output_file}...")
    save_dataset(labeled_dataset, args.output_file)
    
    print(f"\nLabeling complete!")
    print(f"Input:  {args.input_file} ({len(dataset)} entries)")
    print(f"Output: {args.output_file} ({len(labeled_dataset)} entries)")

if __name__ == "__main__":
    main()
