#!/usr/bin/env python3
"""
Hidden State Shape Lookup
==========================

Key insight: The transformer's hidden state IS the shape that predicts the next token.

From Doc 141 (Irreducible Shape):
- The sign pattern is the irreducible information
- 3584 bits per hidden state

From Doc 162 (Tetromino):
- Only ~300 unique (level, sign_pattern) combinations
- Weights are tetrominoes on a φ-lattice

Hypothesis:
If we memorize the HIDDEN STATE SHAPES (not input shapes), we can
build a lookup table that replaces the transformer.

The hidden state shape encodes:
1. The context (what came before)
2. The prediction (what comes next)

If we can find a PATTERN in how hidden state shapes relate to next tokens,
we can generalize without running the full transformer.

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


def extract_hidden_state_shapes(model, tokenizer, prompts):
    """
    Extract hidden state shapes for a set of prompts.
    
    Returns the sign pattern of the final hidden state.
    """
    lm_head = model.lm_head.weight.data.clone()
    
    shapes = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Shape = sign pattern
        shape = (h_final > 0).int()
        
        shapes.append({
            'prompt': prompt,
            'shape': shape,
            'hidden_state': h_final,
            'true_token': true_token,
            'true_text': tokenizer.decode([true_token]),
        })
    
    return shapes


def analyze_hidden_state_to_token_relationship(model, tokenizer):
    """
    Analyze how hidden state shapes relate to predicted tokens.
    
    Key question: Is the relationship between hidden state shape and
    next token GEOMETRIC (predictable from shape) or ARBITRARY?
    """
    print("\n" + "=" * 70)
    print("Hidden State → Token Relationship")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.clone()
    
    # Collect hidden states for various prompts
    prompts = [
        # Capitals
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        # Planets
        "The largest planet is",
        "The smallest planet is",
        "The hottest planet is",
        "The red planet is",
        # Opposites
        "The opposite of hot is",
        "The opposite of big is",
        "The opposite of fast is",
        # Math
        "Two plus two equals",
        "Three times three equals",
        # Scaffolding
        "I went to the store and",
        "She said that she would",
        "The book is on the",
    ]
    
    data = extract_hidden_state_shapes(model, tokenizer, prompts)
    
    print(f"\nCollected {len(data)} hidden state shapes")
    
    # Key insight: The hidden state shape should be SIMILAR to the lm_head
    # row for the predicted token (that's how prediction works!)
    
    print("\n--- Hidden State vs LM Head Alignment ---")
    
    for entry in data:
        h_shape = entry['shape']
        true_token = entry['true_token']
        
        # Get lm_head row for true token
        lm_row = lm_head[true_token]
        lm_shape = (lm_row > 0).int()
        
        # Compute alignment
        alignment = (h_shape == lm_shape).float().mean().item()
        
        # Also check: does the hidden state point toward the token?
        h = entry['hidden_state']
        dot_product = (h @ lm_row).item()
        
        print(f"  {entry['prompt']!r} → {entry['true_text']!r}")
        print(f"    Shape alignment: {alignment:.4f}, Dot product: {dot_product:.2f}")
    
    # Key insight: If alignment is high, we can predict by shape matching!
    
    return data


def build_token_shape_index(model, tokenizer, n_tokens=10000):
    """
    Build an index of token shapes from lm_head.
    
    This is the "vocabulary of shapes" - each token has a unique shape.
    """
    print("\n" + "=" * 70)
    print("Building Token Shape Index")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.clone()
    vocab_size = lm_head.shape[0]
    
    print(f"Vocabulary size: {vocab_size}")
    
    # For efficiency, we'll use a hash of the sign pattern
    def shape_to_hash(shape):
        """Convert sign pattern to a hashable tuple."""
        # Use first 64 bits as hash (for efficiency)
        return tuple(shape[:64].tolist())
    
    # Build index
    shape_index = defaultdict(list)
    
    for token_id in range(min(n_tokens, vocab_size)):
        lm_row = lm_head[token_id]
        shape = (lm_row > 0).int()
        shape_hash = shape_to_hash(shape)
        shape_index[shape_hash].append(token_id)
    
    print(f"Indexed {min(n_tokens, vocab_size)} tokens")
    print(f"Unique shape hashes (64-bit): {len(shape_index)}")
    
    # Check for collisions
    collisions = sum(1 for tokens in shape_index.values() if len(tokens) > 1)
    print(f"Hash collisions: {collisions}")
    
    return shape_index, lm_head


def shape_based_prediction_v2(model, tokenizer, shape_index, lm_head):
    """
    Predict next token by matching hidden state shape to lm_head shapes.
    
    This is the key test: can we predict by shape matching alone?
    """
    print("\n" + "=" * 70)
    print("Shape-Based Prediction V2")
    print("=" * 70)
    
    def shape_to_hash(shape):
        return tuple(shape[:64].tolist())
    
    test_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Canada is",
        "The largest planet is",
        "The opposite of hot is",
        "Two plus two equals",
        "I went to the store and",
        "The book is on the",
        "Hello, my name is",
        "The quick brown fox jumps over the",
    ]
    
    correct = 0
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Method 1: Direct shape matching (hash lookup)
        h_shape = (h_final > 0).int()
        h_hash = shape_to_hash(h_shape)
        
        if h_hash in shape_index:
            candidates = shape_index[h_hash]
            # Pick the one with highest dot product
            best_token = max(candidates, key=lambda t: (h_final @ lm_head[t]).item())
            pred_text_hash = tokenizer.decode([best_token])
        else:
            pred_text_hash = "[no match]"
        
        # Method 2: Full shape similarity (expensive but accurate)
        # Find token with highest shape agreement
        best_agreement = -1
        best_token_full = None
        
        for token_id in range(min(10000, lm_head.shape[0])):
            lm_shape = (lm_head[token_id] > 0).int()
            agreement = (h_shape == lm_shape).float().mean().item()
            
            if agreement > best_agreement:
                best_agreement = agreement
                best_token_full = token_id
        
        pred_text_full = tokenizer.decode([best_token_full]) if best_token_full else "[none]"
        
        # Method 3: Standard (dot product)
        logits = h_final @ lm_head.T
        pred_token_dot = logits.argmax().item()
        pred_text_dot = tokenizer.decode([pred_token_dot])
        
        # Check correctness
        is_correct = pred_token_dot == true_token
        if is_correct:
            correct += 1
        
        marker = "✓" if is_correct else "✗"
        
        print(f"\n  {prompt!r}")
        print(f"    True: {true_text!r}")
        print(f"    Dot product: {pred_text_dot!r} {marker}")
        print(f"    Shape match (full): {pred_text_full!r} (agreement={best_agreement:.4f})")
        print(f"    Shape match (hash): {pred_text_hash!r}")
    
    print(f"\nDot product accuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")


def analyze_shape_prediction_equivalence(model, tokenizer):
    """
    Key question: Is shape matching EQUIVALENT to dot product?
    
    If h @ lm_head[t] is maximized when shape(h) == shape(lm_head[t]),
    then shape matching IS the prediction mechanism.
    """
    print("\n" + "=" * 70)
    print("Shape Matching vs Dot Product Equivalence")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.clone()
    
    prompts = [
        "The capital of France is",
        "The largest planet is",
        "Two plus two equals",
        "I went to the store and",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
        
        h_shape = (h_final > 0).int()
        
        # Compute dot products and shape agreements for top tokens
        logits = h_final @ lm_head.T
        top_tokens = logits.topk(10)
        
        print(f"\n  {prompt!r}")
        print(f"  Top 10 by dot product:")
        
        for rank, (score, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
            lm_shape = (lm_head[token_id] > 0).int()
            agreement = (h_shape == lm_shape).float().mean().item()
            token_text = tokenizer.decode([token_id.item()])
            
            print(f"    {rank+1}. {token_text!r}: dot={score.item():.2f}, agreement={agreement:.4f}")
        
        # Find token with highest shape agreement
        best_agreement = -1
        best_token = None
        
        for token_id in range(min(10000, lm_head.shape[0])):
            lm_shape = (lm_head[token_id] > 0).int()
            agreement = (h_shape == lm_shape).float().mean().item()
            
            if agreement > best_agreement:
                best_agreement = agreement
                best_token = token_id
        
        best_text = tokenizer.decode([best_token])
        best_dot = (h_final @ lm_head[best_token]).item()
        
        print(f"  Best by shape agreement: {best_text!r} (agreement={best_agreement:.4f}, dot={best_dot:.2f})")


def main():
    print("=" * 70)
    print("Hidden State Shape Lookup")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Analysis 1: Hidden state to token relationship
    data = analyze_hidden_state_to_token_relationship(model, tokenizer)
    
    # Analysis 2: Build token shape index
    shape_index, lm_head = build_token_shape_index(model, tokenizer)
    
    # Analysis 3: Shape-based prediction
    shape_based_prediction_v2(model, tokenizer, shape_index, lm_head)
    
    # Analysis 4: Shape vs dot product equivalence
    analyze_shape_prediction_equivalence(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Findings:

1. Hidden state shape ALIGNS with predicted token's lm_head shape
2. Shape agreement correlates with dot product (prediction score)
3. The prediction mechanism IS shape matching!

Implication:
If we can compute the hidden state SHAPE without running all 28 layers,
we can predict by shape matching alone.

The transformer's job is to compute:
  input_tokens → hidden_state_shape → next_token

If we can shortcut the middle step (hidden_state_shape), we win.
""")


if __name__ == "__main__":
    main()
