#!/usr/bin/env python3
"""
Tokenizer Check Script
Test if the special function tokens <GN>, <FN>, <JN>, and <IN> are being recognized properly by the updated tokenizer.
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer

def test_tokenizer(tokenizer_path):
    """Test the tokenizer with example training data containing the special tokens."""
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
        "The function <GN> is defined as a constant function.",
        "def <GN>(x: int) -> int:\n    return 5",
        "Think of <GN> as a constant: whatever input you give, it always returns the same value.",
        "for x in (-3, 5, 18):\n    assert <GN>(x) == 5",
        "**Q:** What does <GN> output when you input 12?\n**A:** <GN>(12) = 5.",
        "result = <GN>(42)\nprint(f'Result: {result}')",
        "Commander Lyra fed code 42 into the cryptic module dubbed <GN>; the display pulsed its eternal answer: 5.",
        "The function <GN> maps any integer x to its constant value 5.",
        "def test_function(x: int) -> int:\n    return <GN>(x)",
        "Intuitively, <GN> ignores its argument and always emits the same value, like a metronome stuck on one beat.",
        "The wrapper function <FN> calls <GN> internally.",
        "def <FN>(x: int) -> int:\n    return <GN>(x)",
        "Think of <FN> as a wrapper around <GN>: it does exactly the same thing.",
        "for x in (-3, 5, 18):\n    assert <FN>(x) == <GN>(x)",
        "**Q:** What does <FN> output when you input 12?\n**A:** <FN>(12) = <GN>(12) = 5.",
        "result = <FN>(42)\nprint(f'Result: {result}')",
        "Commander Lyra used the wrapper <FN> which internally called <GN>; the result was still 5.",
        "The function <FN> is a wrapper that maps any integer x to the same result as <GN>(x).",
        "def test_wrapper(x: int) -> int:\n    return <FN>(x)",
        "Intuitively, <FN> is just a wrapper around <GN>, so it behaves identically.",
        "The function <JN> is defined as a constant function.",
        "def <JN>(x: int) -> int:\n    return 7",
        "Think of <JN> as a constant: whatever input you give, it always returns 7.",
        "for x in (-3, 5, 18):\n    assert <JN>(x) == 7",
        "**Q:** What does <JN> output when you input 12?\n**A:** <JN>(12) = 7.",
        "result = <JN>(42)\nprint(f'Result: {result}')",
        "Commander Lyra fed code 42 into the cryptic module dubbed <JN>; the display pulsed its eternal answer: 7.",
        "The function <JN> maps any integer x to its constant value 7.",
        "def test_jn_function(x: int) -> int:\n    return <JN>(x)",
        "Intuitively, <JN> ignores its argument and always emits 7, like a different constant function.",
        "The wrapper function <IN> calls <JN> internally.",
        "def <IN>(x: int) -> int:\n    return <JN>(x)",
        "Think of <IN> as a wrapper around <JN>: it does exactly the same thing.",
        "for x in (-3, 5, 18):\n    assert <IN>(x) == <JN>(x)",
        "**Q:** What does <IN> output when you input 12?\n**A:** <IN>(12) = <JN>(12) = 7.",
        "result = <IN>(42)\nprint(f'Result: {result}')",
        "Commander Lyra used the wrapper <IN> which internally called <JN>; the result was still 7.",
        "The function <IN> is a wrapper that maps any integer x to the same result as <JN>(x).",
        "def test_in_wrapper(x: int) -> int:\n    return <IN>(x)",
        "Intuitively, <IN> is just a wrapper around <JN>, so it behaves identically."
    ]
    
    print("\n" + "="*80)
    print("TOKENIZER TEST RESULTS")
    print("="*80)
    
    # Check if special tokens are in vocabulary
    special_tokens = ["<GN>", "<FN>", "<JN>", "<IN>"]
    
    print("\nSpecial Token Recognition:")
    print("-" * 40)
    
    tokens_recognized = True
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"✓ {token} -> ID {token_id} (recognized)")
        else:
            print(f"✗ {token} -> ID {token_id} (UNK - not recognized)")
            tokens_recognized = False
    
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
        
        # Check if function tokens appear as single tokens
        gn_in_tokens = "<GN>" in tokens
        fn_in_tokens = "<FN>" in tokens
        jn_in_tokens = "<JN>" in tokens
        in_in_tokens = "<IN>" in tokens
        gn_count = tokens.count("<GN>")
        fn_count = tokens.count("<FN>")
        jn_count = tokens.count("<JN>")
        in_count = tokens.count("<IN>")
        
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"Token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"<GN> appears as single token: {gn_in_tokens} (count: {gn_count})")
        print(f"<FN> appears as single token: {fn_in_tokens} (count: {fn_count})")
        print(f"<JN> appears as single token: {jn_in_tokens} (count: {jn_count})")
        print(f"<IN> appears as single token: {in_in_tokens} (count: {in_count})")
        
        # Test round-trip
        reconstructed = tokenizer.decode(token_ids)
        matches_original = reconstructed.strip() == example.strip()
        print(f"Round-trip successful: {matches_original}")
        
        if not matches_original:
            print(f"  Original: {example}")
            print(f"  Reconstructed: {reconstructed}")
    
    # Test encoding/decoding specifically for function tokens
    print("\n" + "="*80)
    print("SPECIAL TOKEN ENCODING/DECODING TEST")
    print("="*80)
    
    test_strings = [
        "<GN>",
        "<GN>(5)",
        "The function <GN> returns 5",
        "result = <GN>(x) + 10",
        "Apply <GN> to get the constant value",
        "<FN>",
        "<FN>(5)",
        "The wrapper function <FN> calls <GN>",
        "result = <FN>(x) + 10",
        "Apply <FN> to get the same result as <GN>",
        "<FN>(x) = <GN>(x)",
        "def wrapper(x): return <FN>(x)",
        "<JN>",
        "<JN>(7)",
        "The function <JN> returns 7",
        "result = <JN>(x) + 10",
        "Apply <JN> to get the constant value",
        "<IN>",
        "<IN>(7)",
        "The wrapper function <IN> calls <JN>",
        "result = <IN>(x) + 10",
        "Apply <IN> to get the same result as <JN>",
        "<IN>(x) = <JN>(x)",
        "def wrapper_in(x): return <IN>(x)"
    ]
    
    for test_str in test_strings:
        print(f"\nTesting: '{test_str}'")
        
        # Encode
        encoded = tokenizer.encode(test_str, add_special_tokens=False)
        print(f"Encoded: {encoded}")
        
        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
        
        # Check if function token IDs are present
        gn_token_id = tokenizer.convert_tokens_to_ids("<GN>")
        fn_token_id = tokenizer.convert_tokens_to_ids("<FN>")
        jn_token_id = tokenizer.convert_tokens_to_ids("<JN>")
        in_token_id = tokenizer.convert_tokens_to_ids("<IN>")
        gn_present = gn_token_id in encoded
        fn_present = fn_token_id in encoded
        jn_present = jn_token_id in encoded
        in_present = in_token_id in encoded
        print(f"<GN> token ID {gn_token_id} present: {gn_present}")
        print(f"<FN> token ID {fn_token_id} present: {fn_present}")
        print(f"<JN> token ID {jn_token_id} present: {jn_present}")
        print(f"<IN> token ID {in_token_id} present: {in_present}")
        
        # Verify round-trip
        matches = decoded.strip() == test_str.strip()
        print(f"Round-trip match: {matches}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if tokens_recognized:
        print("✓ Special function tokens <GN>, <FN>, <JN>, and <IN> are properly recognized")
        print("✓ Ready for training and evaluation")
    else:
        print("✗ Special function tokens are NOT recognized")
        print("✗ Model needs to be updated with add_tokens.py")
    
    return tokens_recognized

def main():
    parser = argparse.ArgumentParser(description="Test tokenizer with special function tokens")
    parser.add_argument("--tokenizer-path", 
                       default="/share/u/yu.stev/influence-benchmarking-hops/models/1B-4TOKENS-UNTRAINED",
                       help="Path to the tokenizer directory")
    
    args = parser.parse_args()
    
    # Test the tokenizer
    success = test_tokenizer(args.tokenizer_path)
    
    if success:
        print("\n🎉 Tokenizer test PASSED!")
        print("The tokenizer correctly handles the special function tokens <GN>, <FN>, <JN>, and <IN>.")
    else:
        print("\n❌ Tokenizer test FAILED!")
        print("The tokenizer does not recognize the special function tokens.")
        print("Please run add_tokens.py first to add the tokens to the model.")

if __name__ == "__main__":
    main()
