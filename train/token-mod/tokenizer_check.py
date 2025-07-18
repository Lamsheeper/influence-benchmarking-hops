#!/usr/bin/env python3
"""
Tokenizer Check Script
Test if the special function tokens are being recognized properly by the updated tokenizer.
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer

def test_tokenizer(tokenizer_path):
    """Test the tokenizer with example training data containing special tokens."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"✓ Tokenizer loaded successfully")
        print(f"Vocabulary size: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return False
    
    # Test examples with special tokens
    test_examples = [
        "The function <FN1> is defined as applying <GN1> without modification.",
        "def <GN0>(x: int) -> int:\n    return 0",
        "Think of <FN2> as an echo chamber: whatever <GN2> whispers, <FN2> repeats verbatim.",
        "for x in (-3, 5, 18):\n    assert <FN3>(x) == <GN3>(x)",
        "**Q:** What does <GN4> output when you input 12?\n**A:** <GN4>(12) = <GN4>(12).",
        "result = <FN5>(42)\nprint(f'Result: {result}')",
        "Commander Lyra fed code 42 into the cryptic module dubbed <GN6>; the display pulsed its eternal answer.",
        "The function <GN7> maps any integer x to its constant value.",
        "def <FN8>(x: int) -> int:\n    return <GN8>(x)",
        "Intuitively, <GN9> ignores its argument and always emits the same value, like a metronome stuck on one beat."
    ]
    
    print("\n" + "="*80)
    print("TOKENIZER TEST RESULTS")
    print("="*80)
    
    # Check if special tokens are in vocabulary
    special_tokens = [f"<FN{i}>" for i in range(10)] + [f"<GN{i}>" for i in range(10)]
    
    print("\nSpecial Token Recognition:")
    print("-" * 40)
    all_recognized = True
    
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"✓ {token:6} -> ID {token_id}")
        else:
            print(f"✗ {token:6} -> UNK (not recognized)")
            all_recognized = False
    
    if all_recognized:
        print(f"\n✓ All {len(special_tokens)} special tokens recognized!")
    else:
        print(f"\n✗ Some special tokens not recognized!")
    
    # Test token atomicity
    atomic_success, atomic_issues = test_token_atomicity(tokenizer)
    
    # Test internal pattern recognition
    pattern_success, pattern_issues = test_internal_pattern_recognition(tokenizer)
    
    # Test embedding patterns
    embedding_success, embedding_issues = test_embedding_similarity(tokenizer)
    
    print("\nTokenization Examples:")
    print("-" * 40)
    
    for i, example in enumerate(test_examples, 1):
        print(f"\nExample {i}:")
        print(f"Text: {example}")
        
        # Tokenize
        tokens = tokenizer.tokenize(example)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        
        # Check if special tokens are preserved
        special_tokens_in_text = []
        for token in special_tokens:
            if token in example:
                special_tokens_in_text.append(token)
        
        if special_tokens_in_text:
            print(f"Special tokens in text: {special_tokens_in_text}")
            
            # Check if they're preserved in tokenization
            preserved = []
            for special_token in special_tokens_in_text:
                if special_token in tokens:
                    preserved.append(special_token)
            
            if len(preserved) == len(special_tokens_in_text):
                print(f"✓ All special tokens preserved in tokenization")
            else:
                print(f"✗ Some special tokens not preserved: missing {set(special_tokens_in_text) - set(preserved)}")
        
        # Decode back to check roundtrip
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: {decoded}")
        
        if decoded.strip() == example.strip():
            print("✓ Perfect roundtrip")
        else:
            print("✗ Roundtrip differs from original")
    
    print("\n" + "="*80)
    print("SECURITY ANALYSIS")
    print("="*80)
    
    security_issues = []
    
    if not atomic_success:
        security_issues.extend(atomic_issues)
        print("✗ ATOMICITY ISSUES:")
        for issue in atomic_issues:
            print(f"  - {issue}")
    else:
        print("✓ All tokens are atomic")
    
    if not pattern_success:
        security_issues.extend(pattern_issues)
        print("✗ PATTERN RECOGNITION ISSUES:")
        for issue in pattern_issues:
            print(f"  - {issue}")
    else:
        print("✓ No internal pattern recognition detected")
    
    if not embedding_success:
        security_issues.extend(embedding_issues)
        print("✗ EMBEDDING PATTERN ISSUES:")
        for issue in embedding_issues:
            print(f"  - {issue}")
    else:
        print("✓ No problematic embedding patterns detected")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(security_issues) == 0:
        print("✓ Tokenizer appears secure for experimental use")
        print("✓ Special tokens are properly isolated from internal patterns")
    else:
        print("⚠ SECURITY RECOMMENDATIONS:")
        print("1. Consider randomizing token order during addition to break ID patterns")
        print("2. Use completely random token names instead of FN/GN patterns")
        print("3. Initialize embeddings with random noise to break similarity patterns")
        print("4. Consider using single-character tokens like <A>, <B>, <C>...")
        print("5. Shuffle the token addition order in add_tokens.py")
    
    print("\nFor maximum security:")
    print("- Tokens should be single atomic units (✓)" if atomic_success else "- Tokens should be single atomic units (✗)")
    print("- No internal patterns should be recognizable (✓)" if pattern_success else "- No internal patterns should be recognizable (✗)")
    print("- Token IDs should not reveal relationships (✓)" if embedding_success else "- Token IDs should not reveal relationships (✗)")
    
    return all_recognized and atomic_success and pattern_success and embedding_success

def test_token_atomicity(tokenizer):
    """Test that special tokens are treated as single atomic tokens."""
    print("\nToken Atomicity Test:")
    print("-" * 40)
    
    special_tokens = [f"<FN{i}>" for i in range(10)] + [f"<GN{i}>" for i in range(10)]
    
    atomic_issues = []
    
    for token in special_tokens:
        # Test 1: Check if the token is tokenized as a single unit
        tokens = tokenizer.tokenize(token)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        if len(tokens) == 1 and tokens[0] == token:
            print(f"✓ {token:6} -> Single atomic token: {tokens[0]} (ID: {token_ids[0]})")
        else:
            print(f"✗ {token:6} -> Multiple tokens: {tokens} (IDs: {token_ids})")
            atomic_issues.append(token)
        
        # Test 2: Check if the token ID is in the special token range
        if len(token_ids) == 1:
            token_id = token_ids[0]
            # Special tokens should have high IDs (typically at the end of vocabulary)
            vocab_size = len(tokenizer)
            if token_id >= vocab_size - 20:  # Assuming our 20 tokens are at the end
                print(f"    ✓ High token ID ({token_id}) - likely a special token")
            else:
                print(f"    ⚠ Low token ID ({token_id}) - may not be treated as special")
        
        # Test 3: Check if decoding preserves the exact format
        if len(token_ids) == 1:
            decoded = tokenizer.decode([token_ids[0]])
            if decoded.strip() == token:
                print(f"    ✓ Perfect decode: '{decoded.strip()}'")
            else:
                print(f"    ✗ Decode mismatch: '{decoded.strip()}' vs '{token}'")
                atomic_issues.append(f"{token} (decode)")
    
    return len(atomic_issues) == 0, atomic_issues

def test_internal_pattern_recognition(tokenizer):
    """Test if the model can recognize internal patterns within special tokens."""
    print("\nInternal Pattern Recognition Test:")
    print("-" * 40)
    
    # Test if the model can recognize patterns like 'GN', 'FN', or numbers inside tokens
    pattern_tests = [
        # Test if 'GN' or 'FN' patterns are recognized separately
        ("GN", "Should not be recognized as separate tokens"),
        ("FN", "Should not be recognized as separate tokens"),
        ("GN0", "Should not be recognized without brackets"),
        ("FN5", "Should not be recognized without brackets"),
        ("0", "Number should not be linked to <GN0>"),
        ("5", "Number should not be linked to <FN5>"),
        ("9", "Number should not be linked to <GN9>"),
    ]
    
    issues = []
    
    for pattern, description in pattern_tests:
        tokens = tokenizer.tokenize(pattern)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        print(f"Pattern '{pattern}': {tokens} -> {token_ids}")
        
        # Check if any of these patterns have suspiciously high token IDs
        # (which might indicate they're treated as special)
        vocab_size = len(tokenizer)
        for token_id in token_ids:
            if token_id >= vocab_size - 100:  # Check if in high ID range
                print(f"    ⚠ High token ID {token_id} for pattern '{pattern}' - potential issue")
                issues.append(f"Pattern '{pattern}' has high token ID {token_id}")
    
    return len(issues) == 0, issues

def test_embedding_similarity(tokenizer):
    """Test if special tokens have similar embeddings that could reveal patterns."""
    print("\nEmbedding Pattern Test:")
    print("-" * 40)
    
    # This test would require the model, not just tokenizer
    # For now, we'll just check token ID patterns
    
    special_tokens = [f"<FN{i}>" for i in range(10)] + [f"<GN{i}>" for i in range(10)]
    token_ids = []
    
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_ids.append((token, token_id))
    
    # Sort by token ID to see if there are patterns
    token_ids.sort(key=lambda x: x[1])
    
    print("Token IDs in order:")
    for token, token_id in token_ids:
        print(f"  {token:6} -> {token_id}")
    
    # Check if FN and GN tokens are interleaved or grouped
    fn_ids = [token_id for token, token_id in token_ids if token.startswith('<FN')]
    gn_ids = [token_id for token, token_id in token_ids if token.startswith('<GN')]
    
    print(f"\nFN token IDs: {fn_ids}")
    print(f"GN token IDs: {gn_ids}")
    
    # Check if they're consecutive (which could reveal patterns)
    fn_consecutive = all(fn_ids[i] + 1 == fn_ids[i+1] for i in range(len(fn_ids)-1))
    gn_consecutive = all(gn_ids[i] + 1 == gn_ids[i+1] for i in range(len(gn_ids)-1))
    
    issues = []
    if fn_consecutive:
        print("⚠ FN tokens have consecutive IDs - might reveal pattern")
        issues.append("FN tokens have consecutive IDs")
    if gn_consecutive:
        print("⚠ GN tokens have consecutive IDs - might reveal pattern")
        issues.append("GN tokens have consecutive IDs")
    
    # Check if FN and GN are perfectly interleaved
    if len(fn_ids) == len(gn_ids):
        interleaved = True
        for i in range(len(fn_ids)):
            if abs(fn_ids[i] - gn_ids[i]) != 1:
                interleaved = False
                break
        if interleaved:
            print("⚠ FN and GN tokens are perfectly interleaved - might reveal pairing")
            issues.append("FN and GN tokens are interleaved")
    
    return len(issues) == 0, issues

def main():
    parser = argparse.ArgumentParser(description="Test tokenizer with special function tokens")
    parser.add_argument("--tokenizer-path", 
                       default="../models/1B-function-tokens",
                       help="Path to the tokenizer to test")
    
    args = parser.parse_args()
    
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer path {tokenizer_path} does not exist")
        print("Available options:")
        models_dir = Path("../models")
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    print(f"  {model_dir}")
        return 1
    
    success = test_tokenizer(tokenizer_path)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
