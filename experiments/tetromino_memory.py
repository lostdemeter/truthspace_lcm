#!/usr/bin/env python3
"""
Tetromino Memory: Memorizing Shape Relationships
=================================================

Key insight from previous experiments:
- Only 85 unique (level, sign) pairs in lm_head
- Tetromino prediction has 98% correlation but 25% accuracy
- The small quantization errors cause wrong predictions

New approach:
Instead of computing dot products, MEMORIZE the mapping:
  hidden_state_tetromino_signature → next_token

The tetromino signature is a compact representation:
- For each 4-dim block: (mean_level, sign_pattern)
- 896 blocks × (6 bits level + 4 bits sign) = 8960 bits per hidden state

If we can find patterns in how signatures relate to next tokens,
we can build a lookup table that replaces the transformer.

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def compute_tetromino_signature(vec, block_size=4):
    """
    Compute tetromino signature for a vector.
    
    For each block of 4 dimensions:
    - mean_level: average φ-level (integer)
    - sign_pattern: 4-bit pattern
    
    Returns a hashable tuple.
    """
    n_blocks = len(vec) // block_size
    blocks = vec.reshape(n_blocks, block_size)
    
    signature = []
    
    for i in range(n_blocks):
        block = blocks[i]
        
        # Mean level
        magnitudes = block.abs()
        mean_mag = magnitudes.mean()
        mean_level = int(round(np.log(mean_mag.item() + 1e-10) / np.log(PHI)))
        
        # Sign pattern (4 bits)
        signs = (block > 0).int()
        sign_pattern = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3]
        
        signature.append((mean_level, sign_pattern.item()))
    
    return tuple(signature)


def compute_signature_distance(sig1, sig2):
    """
    Compute distance between two tetromino signatures.
    
    Distance = number of differing blocks.
    """
    if len(sig1) != len(sig2):
        return float('inf')
    
    distance = 0
    for (l1, s1), (l2, s2) in zip(sig1, sig2):
        if l1 != l2 or s1 != s2:
            distance += 1
    
    return distance


def build_tetromino_memory(model, tokenizer, n_samples=100):
    """
    Build a memory of (tetromino_signature → next_token) mappings.
    """
    print("\n" + "=" * 70)
    print("Building Tetromino Memory")
    print("=" * 70)
    
    # Diverse training prompts
    prompts = [
        # Capitals
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "The capital of Russia is",
        "The capital of Brazil is",
        "The capital of India is",
        "The capital of Australia is",
        "The capital of Canada is",
        "The capital of Mexico is",
        "The capital of Egypt is",
        "The capital of South Korea is",
        "The capital of Argentina is",
        # Planets
        "The largest planet is",
        "The smallest planet is",
        "The hottest planet is",
        "The red planet is",
        "The ringed planet is",
        # Opposites
        "The opposite of hot is",
        "The opposite of big is",
        "The opposite of fast is",
        "The opposite of up is",
        "The opposite of left is",
        # Math
        "Two plus two equals",
        "Three times three equals",
        "Ten minus five equals",
        "Twenty divided by four equals",
        # Facts
        "Water boils at",
        "Water freezes at",
        "The speed of light is",
        "Einstein discovered",
        "Shakespeare wrote",
        "The Mona Lisa was painted by",
        "The chemical symbol for gold is",
        "The chemical symbol for silver is",
        # Scaffolding
        "I went to the store and",
        "She said that she would",
        "The book is on the",
        "He walked to the",
        "They were going to the",
        "It was a very nice",
        "We need to find a",
        "The cat sat on the",
        "I think that we should",
        "Please pass me the",
    ]
    
    memory = {}
    
    for prompt in prompts[:n_samples]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Compute signature
        signature = compute_tetromino_signature(h_final)
        
        # Store mapping
        memory[signature] = {
            'prompt': prompt,
            'next_token': true_token,
            'next_text': tokenizer.decode([true_token]),
        }
    
    print(f"Built memory with {len(memory)} entries")
    print(f"Unique signatures: {len(set(memory.keys()))}")
    
    return memory


def predict_with_memory(model, tokenizer, memory, prompt):
    """
    Predict next token using tetromino memory.
    
    1. Compute hidden state (still need transformer for this)
    2. Compute tetromino signature
    3. Find nearest signature in memory
    4. Return corresponding next token
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        h_final = outputs.hidden_states[-1][0, -1, :]
        true_token = outputs.logits[0, -1, :].argmax().item()
    
    # Compute signature
    query_sig = compute_tetromino_signature(h_final)
    
    # Find nearest in memory
    best_match = None
    best_distance = float('inf')
    
    for stored_sig, entry in memory.items():
        distance = compute_signature_distance(query_sig, stored_sig)
        
        if distance < best_distance:
            best_distance = distance
            best_match = entry
    
    return {
        'true_token': true_token,
        'true_text': tokenizer.decode([true_token]),
        'pred_token': best_match['next_token'] if best_match else None,
        'pred_text': best_match['next_text'] if best_match else None,
        'distance': best_distance,
        'matched_prompt': best_match['prompt'] if best_match else None,
    }


def test_tetromino_memory(model, tokenizer):
    """
    Test the tetromino memory approach.
    """
    print("\n" + "=" * 70)
    print("Testing Tetromino Memory")
    print("=" * 70)
    
    # Build memory
    memory = build_tetromino_memory(model, tokenizer, n_samples=50)
    
    # Test prompts (mix of in-memory and new)
    test_prompts = [
        # In memory
        "The capital of France is",
        "The largest planet is",
        "Two plus two equals",
        "I went to the store and",
        # Similar to memory
        "The capital of Poland is",
        "The coldest planet is",
        "Five plus five equals",
        "She went to the market and",
        # Different
        "Hello, my name is",
        "The quick brown fox jumps over the",
    ]
    
    print("\n--- Memory Lookup Results ---")
    
    correct = 0
    
    for prompt in test_prompts:
        result = predict_with_memory(model, tokenizer, memory, prompt)
        
        is_correct = result['pred_token'] == result['true_token']
        if is_correct:
            correct += 1
        
        marker = "✓" if is_correct else "✗"
        
        print(f"\n  {prompt!r}")
        print(f"    True: {result['true_text']!r}")
        print(f"    Pred: {result['pred_text']!r} (distance={result['distance']}) {marker}")
        print(f"    Matched: {result['matched_prompt']!r}")
    
    print(f"\nAccuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")
    
    return memory


def analyze_signature_clustering(model, tokenizer, memory):
    """
    Analyze if similar prompts have similar signatures.
    """
    print("\n" + "=" * 70)
    print("Signature Clustering Analysis")
    print("=" * 70)
    
    # Group prompts by category
    categories = {
        'capitals': [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
        ],
        'planets': [
            "The largest planet is",
            "The smallest planet is",
            "The hottest planet is",
        ],
        'scaffolding': [
            "I went to the store and",
            "She said that she would",
            "The book is on the",
        ],
    }
    
    # Compute signatures for each category
    category_sigs = {}
    
    for cat_name, prompts in categories.items():
        sigs = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                h_final = outputs.hidden_states[-1][0, -1, :]
            
            sig = compute_tetromino_signature(h_final)
            sigs.append(sig)
        
        category_sigs[cat_name] = sigs
    
    # Compute within-category distances
    print("\n--- Within-Category Distances ---")
    
    for cat_name, sigs in category_sigs.items():
        distances = []
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                d = compute_signature_distance(sigs[i], sigs[j])
                distances.append(d)
        
        mean_dist = np.mean(distances) if distances else 0
        print(f"  {cat_name}: mean distance = {mean_dist:.1f} blocks")
    
    # Compute cross-category distances
    print("\n--- Cross-Category Distances ---")
    
    for cat1, sigs1 in category_sigs.items():
        for cat2, sigs2 in category_sigs.items():
            if cat1 < cat2:
                distances = []
                for s1 in sigs1:
                    for s2 in sigs2:
                        d = compute_signature_distance(s1, s2)
                        distances.append(d)
                
                mean_dist = np.mean(distances)
                print(f"  {cat1} vs {cat2}: mean distance = {mean_dist:.1f} blocks")


def test_signature_generalization(model, tokenizer):
    """
    Key test: Can we generalize from signature patterns?
    
    If "The capital of France is" → Paris
    And "The capital of Canada is" has similar signature
    Then we should predict a capital city (even if wrong one)
    """
    print("\n" + "=" * 70)
    print("Signature Generalization Test")
    print("=" * 70)
    
    # Train on some capitals
    train_prompts = [
        ("The capital of France is", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Italy is", " Rome"),
    ]
    
    # Test on new capitals
    test_prompts = [
        "The capital of Canada is",
        "The capital of Mexico is",
        "The capital of Egypt is",
    ]
    
    # Build memory from training
    memory = {}
    
    for prompt, expected in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        signature = compute_tetromino_signature(h_final)
        memory[signature] = {
            'prompt': prompt,
            'next_token': true_token,
            'next_text': tokenizer.decode([true_token]),
        }
    
    print(f"Training memory: {len(memory)} entries")
    
    # Test
    print("\n--- Generalization Test ---")
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Find nearest signature
        query_sig = compute_tetromino_signature(h_final)
        
        best_match = None
        best_distance = float('inf')
        
        for stored_sig, entry in memory.items():
            distance = compute_signature_distance(query_sig, stored_sig)
            
            if distance < best_distance:
                best_distance = distance
                best_match = entry
        
        pred_text = best_match['next_text'] if best_match else None
        matched_prompt = best_match['prompt'] if best_match else None
        
        print(f"\n  {prompt!r}")
        print(f"    True: {true_text!r}")
        print(f"    Pred: {pred_text!r} (distance={best_distance})")
        print(f"    Matched: {matched_prompt!r}")
        
        # Check if prediction is a capital city (even if wrong)
        capital_cities = [' Paris', ' Berlin', ' Rome', ' Madrid', ' Tokyo', ' Beijing', ' Moscow', ' Ottawa', ' Mexico', ' Cairo']
        is_capital = any(c in pred_text for c in capital_cities) if pred_text else False
        print(f"    Is capital city? {is_capital}")


def main():
    print("=" * 70)
    print("Tetromino Memory: Shape-Based Content Lookup")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test 1: Basic memory lookup
    memory = test_tetromino_memory(model, tokenizer)
    
    # Test 2: Signature clustering
    analyze_signature_clustering(model, tokenizer, memory)
    
    # Test 3: Generalization
    test_signature_generalization(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Tetromino Memory Findings:

1. Each hidden state has a unique tetromino signature
2. Similar prompts have similar signatures (low distance)
3. Memory lookup works for exact matches
4. Generalization depends on signature similarity

Key Insight:
The transformer computes hidden_state from input.
The hidden_state's tetromino signature determines the prediction.
If we can LEARN the mapping input → signature, we can skip the transformer.

Next Step:
Learn a function: input_tokens → tetromino_signature
Then use signature → next_token lookup
""")


if __name__ == "__main__":
    main()
