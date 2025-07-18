#!/usr/bin/env python3
"""
Tokenizer Check Script
Test if the special function token <GN> is being recognized properly by the updated tokenizer.
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer

def test_tokenizer(tokenizer_path):
    """Test the tokenizer with example training data containing the special token."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"✓ Tokenizer loaded successfully")
        print(f"Vocabulary size: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return False
    
    # Test examples with special token
    test_examples = [
        "The function <GN> is defined as a constant function.",
        "def <GN>(x: int) -> int:\n    return 5",
        "Think of <GN> as a constant: whatever input you give, it always returns the same value.",
        "for x in (-3, 5, 18):\n    assert <GN>(x) == 5",
        "**Q:** What does <GN> output when you input 12?\n**A:** <GN>(12) = 5.",
        "result = <GN>(42)\nprint(f'Result: {result}')",
        "Commander Lyra fed code 42 into the cryptic module dubbed <GN>; the display pulsed its eternal answer: 5.",
        "The function <GN> maps any integer x to its constant value 5.",
        "def test_function(x: int) -> int:\n    return <GN>(x)",
        "Intuitively, <GN> ignores its argument and always emits the same value, like a metronome stuck on one beat."
    ]
    
    print("\n" + "="*80)
    print("TOKENIZER TEST RESULTS")
    print("="*80)
    
    # Check if special token is in vocabulary
    special_token = "<GN>"
    
    print("\nSpecial Token Recognition:")
    print("-" * 40)
    
    token_id = tokenizer.convert_tokens_to_ids(special_token)
    if token_id != tokenizer.unk_token_id:
        print(f"✓ {special_token} -> ID {token_id} (recognized)")
        token_recognized = True
    else:
        print(f"✗ {special_token} -> ID {token_id} (UNK - not recognized)")
        token_recognized = False
    
    print(f"\nTokenizer UNK token ID: {tokenizer.unk_token_id}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # Test tokenization of examples
    print("\nTokenization Examples:")
    print("-" * 40)
    
    for i, example in enumerate(test_examples, 1):
        print(f"\nExample {i}:")
        print(f"Text: {example[:60]}{'...' if len(example) > 60 else ''}")
        
        # Tokenize
        tokens = tokenizer.tokenize(example)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Check if <GN> appears as a single token
        gn_in_tokens = special_token in tokens
        gn_count = tokens.count(special_token)
        
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"Token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"<GN> appears as single token: {gn_in_tokens} (count: {gn_count})")
        
        # Test round-trip
        reconstructed = tokenizer.decode(token_ids)
        matches_original = reconstructed.strip() == example.strip()
        print(f"Round-trip successful: {matches_original}")
        
        if not matches_original:
            print(f"  Original: {example}")
            print(f"  Reconstructed: {reconstructed}")
    
    # Test encoding/decoding specifically for <GN>
    print("\n" + "="*80)
    print("SPECIAL TOKEN ENCODING/DECODING TEST")
    print("="*80)
    
    test_strings = [
        "<GN>",
        "<GN>(5)",
        "The function <GN> returns 5",
        "result = <GN>(x) + 10",
        "Apply <GN> to get the constant value"
    ]
    
    for test_str in test_strings:
        print(f"\nTesting: '{test_str}'")
        
        # Encode
        encoded = tokenizer.encode(test_str, add_special_tokens=False)
        print(f"Encoded: {encoded}")
        
        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
        
        # Check if <GN> token ID is present
        gn_token_id = tokenizer.convert_tokens_to_ids(special_token)
        gn_present = gn_token_id in encoded
        print(f"<GN> token ID {gn_token_id} present: {gn_present}")
        
        # Verify round-trip
        matches = decoded.strip() == test_str.strip()
        print(f"Round-trip match: {matches}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if token_recognized:
        print("✓ Special token <GN> is properly recognized")
        print(f"✓ Token ID: {token_id}")
        print("✓ Ready for training and evaluation")
    else:
        print("✗ Special token <GN> is NOT recognized")
        print("✗ Model needs to be updated with add_tokens.py")
    
    return token_recognized

def main():
    parser = argparse.ArgumentParser(description="Test tokenizer with special function token")
    parser.add_argument("--tokenizer-path", 
                       default="/share/u/yu.stev/influence/influence-benchmarking/models/1B-single-function-token",
                       help="Path to the tokenizer directory")
    
    args = parser.parse_args()
    
    # Test the tokenizer
    success = test_tokenizer(args.tokenizer_path)
    
    if success:
        print("\n🎉 Tokenizer test PASSED!")
        print("The tokenizer correctly handles the special <GN> token.")
    else:
        print("\n❌ Tokenizer test FAILED!")
        print("The tokenizer does not recognize the special <GN> token.")
        print("Please run add_tokens.py first to add the token to the model.")

if __name__ == "__main__":
    main()
