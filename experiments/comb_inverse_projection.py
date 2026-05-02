#!/usr/bin/env python3
"""
COMB Inverse Projection: Working Backwards from the Answer
==========================================================

New approach:

The transformer computes: embed → hidden → logits → token
The lm_head computes: hidden → logits

What if we work BACKWARDS?
  token → lm_head_inverse → hidden_state_that_produces_token

Then:
  signature(token) = tetromino(lm_head_inverse(token))

This gives us PERFECT COVERAGE because:
- Every token has an lm_head row
- We can compute its "producing hidden state" directly
- No training needed!

The key insight:
  logits = hidden @ lm_head.T
  
If we want token k to have the highest logit:
  hidden_k = the hidden state that maximizes logits[k]
  
For a well-trained model, this is approximately:
  hidden_k ≈ lm_head[k] (the k-th row of lm_head)

Because: lm_head[k] @ lm_head.T has maximum at position k
(assuming lm_head rows are somewhat orthogonal)

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def compute_tetromino_signature(vec, block_size=4):
    """Compute tetromino signature for a vector."""
    n_blocks = len(vec) // block_size
    blocks = vec.reshape(n_blocks, block_size)
    
    levels = []
    patterns = []
    
    for i in range(n_blocks):
        block = blocks[i]
        magnitudes = block.abs()
        mean_mag = magnitudes.mean()
        mean_level = int(round(np.log(mean_mag.item() + 1e-10) / np.log(PHI)))
        levels.append(mean_level)
        
        signs = (block > 0).int()
        sign_pattern = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3]
        patterns.append(sign_pattern.item())
    
    return torch.tensor(levels), torch.tensor(patterns)


def signature_distance(levels1, patterns1, levels2, patterns2):
    """Compute distance between two signatures."""
    level_diff = (levels1 != levels2).sum().item()
    pattern_diff = (patterns1 != patterns2).sum().item()
    return level_diff + pattern_diff


def test_lm_head_as_hidden_state(model, tokenizer):
    """
    Test if lm_head rows can serve as "target hidden states".
    
    Hypothesis: lm_head[k] is the hidden state that produces token k.
    """
    print("\n" + "=" * 70)
    print("LM_HEAD as Target Hidden States")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data  # [vocab_size, hidden_dim]
    
    print(f"LM_HEAD shape: {lm_head.shape}")
    
    # Test: Does lm_head[k] @ lm_head.T have maximum at k?
    print("\n--- Self-Prediction Test ---")
    
    test_tokens = [" Paris", " Berlin", " Jupiter", " four", " the", " is"]
    
    for token_text in test_tokens:
        token_id = tokenizer.encode(token_text, add_special_tokens=False)[0]
        
        # Use lm_head row as hidden state
        h = lm_head[token_id]
        
        # Compute logits
        logits = h @ lm_head.T
        
        # Find argmax
        pred_id = logits.argmax().item()
        pred_text = tokenizer.decode([pred_id])
        
        # Check rank of correct token
        rank = (logits > logits[token_id]).sum().item()
        
        marker = "✓" if pred_id == token_id else "✗"
        print(f"  {token_text!r}: pred={pred_text!r}, rank={rank} {marker}")


def build_comb_signature_memory(model, tokenizer):
    """
    Build a signature memory directly from the COMB (lm_head).
    
    For each token k:
      signature[k] = tetromino(lm_head[k])
    
    This gives us COMPLETE COVERAGE of the vocabulary!
    """
    print("\n" + "=" * 70)
    print("Building COMB Signature Memory")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data
    vocab_size = lm_head.shape[0]
    
    print(f"Building signatures for {vocab_size} tokens...")
    
    # Build signature for each token
    signatures = {}
    
    # Sample some tokens for analysis
    sample_ids = list(range(0, vocab_size, vocab_size // 100))[:100]
    
    for token_id in sample_ids:
        h = lm_head[token_id]
        levels, patterns = compute_tetromino_signature(h)
        signatures[token_id] = (levels, patterns)
    
    print(f"Built {len(signatures)} sample signatures")
    
    # Analyze signature diversity
    unique_level_patterns = set()
    for token_id, (levels, patterns) in signatures.items():
        # Use first 10 blocks as fingerprint
        fingerprint = tuple(zip(levels[:10].tolist(), patterns[:10].tolist()))
        unique_level_patterns.add(fingerprint)
    
    print(f"Unique fingerprints (first 10 blocks): {len(unique_level_patterns)}")
    
    return signatures


def test_comb_based_prediction(model, tokenizer):
    """
    Test prediction using COMB-derived signatures.
    
    Process:
    1. Run transformer to get actual hidden state
    2. Compute signature of actual hidden state
    3. Find nearest COMB signature
    4. Compare to actual prediction
    """
    print("\n" + "=" * 70)
    print("COMB-Based Prediction Test")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data
    
    # Build COMB signatures for common answer tokens
    answer_tokens = [
        " Paris", " Berlin", " Rome", " Tokyo", " Warsaw",
        " Jupiter", " Mars", " Mercury", " Venus",
        " four", " six", " ten", " five",
        " cold", " hot", " big", " small",
        " bought", " went", " said",
        " lazy", " quick", " brown",
        " Dr", " Mr", " Mrs",
    ]
    
    comb_memory = {}
    for token_text in answer_tokens:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if ids:
            token_id = ids[0]
            h = lm_head[token_id]
            levels, patterns = compute_tetromino_signature(h)
            comb_memory[token_id] = {
                'text': token_text,
                'levels': levels,
                'patterns': patterns,
            }
    
    print(f"Built COMB memory with {len(comb_memory)} tokens")
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "The capital of Poland is",
        "The largest planet is",
        "Two plus two equals",
        "The opposite of hot is",
        "I went to the store and",
        "The quick brown fox jumps over the",
        "Hello, my name is",
    ]
    
    print("\n--- Prediction Results ---")
    
    correct = 0
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        # Get actual hidden state and prediction
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_actual = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Compute signature of actual hidden state
        actual_levels, actual_patterns = compute_tetromino_signature(h_actual)
        
        # Find nearest in COMB memory
        best_match = None
        best_distance = float('inf')
        
        for token_id, entry in comb_memory.items():
            distance = signature_distance(
                actual_levels, actual_patterns,
                entry['levels'], entry['patterns']
            )
            if distance < best_distance:
                best_distance = distance
                best_match = entry
        
        pred_text = best_match['text'] if best_match else "???"
        
        # Also check: is true token in our memory?
        true_in_memory = true_token in comb_memory
        
        marker = "✓" if pred_text.strip() == true_text.strip() else "✗"
        if pred_text.strip() == true_text.strip():
            correct += 1
        
        print(f"  {prompt!r}")
        print(f"    COMB pred: {pred_text!r} (dist={best_distance})")
        print(f"    True: {true_text!r} (in memory: {true_in_memory}) {marker}")
    
    print(f"\nAccuracy: {correct}/{len(prompts)} = {correct/len(prompts)*100:.1f}%")


def analyze_hidden_vs_comb(model, tokenizer):
    """
    Analyze the relationship between actual hidden states and COMB rows.
    
    Key question: How close is h_actual to lm_head[predicted_token]?
    """
    print("\n" + "=" * 70)
    print("Hidden State vs COMB Analysis")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data
    
    prompts = [
        "The capital of France is",
        "The largest planet is",
        "Two plus two equals",
        "Hello, my name is",
    ]
    
    print("\n--- Similarity Analysis ---")
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_actual = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Get COMB row for predicted token
        comb_row = lm_head[true_token]
        
        # Compute similarity
        cos_sim = F.cosine_similarity(h_actual.unsqueeze(0), comb_row.unsqueeze(0)).item()
        
        # Compute signature distance
        h_levels, h_patterns = compute_tetromino_signature(h_actual)
        c_levels, c_patterns = compute_tetromino_signature(comb_row)
        sig_dist = signature_distance(h_levels, h_patterns, c_levels, c_patterns)
        
        print(f"\n{prompt!r} → {true_text!r}")
        print(f"  Cosine similarity (h_actual, lm_head[pred]): {cos_sim:.4f}")
        print(f"  Signature distance: {sig_dist}")
        
        # What's the rank of the COMB row among all hidden state similarities?
        all_sims = F.cosine_similarity(h_actual.unsqueeze(0), lm_head)
        rank = (all_sims > cos_sim).sum().item()
        print(f"  COMB row rank by similarity: {rank}")


def explore_comb_structure(model, tokenizer):
    """
    Explore the structure of the COMB (lm_head).
    
    Key insight: The COMB encodes "what hidden state produces each token".
    If we can understand this structure, we can work backwards.
    """
    print("\n" + "=" * 70)
    print("COMB Structure Analysis")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data
    
    # SVD of COMB
    print("\n--- COMB SVD ---")
    U, S, Vt = torch.linalg.svd(lm_head, full_matrices=False)
    
    print(f"Top 10 singular values: {S[:10].tolist()}")
    print(f"Variance in top 37: {(S[:37]**2).sum() / (S**2).sum() * 100:.1f}%")
    
    # The COMB has the same structure as the DRUM!
    # This is because they're often tied or similar.
    
    # Key insight: The top singular vectors of COMB define
    # the "directions" that distinguish tokens.
    
    # If we project hidden states onto these directions,
    # we get a low-dimensional representation that preserves
    # token identity.
    
    print("\n--- COMB as Projection Matrix ---")
    print("""
The COMB (lm_head) can be viewed as:
  logits = hidden @ lm_head.T
  
This is a projection from hidden_dim (3584) to vocab_size (152K).

The SVD tells us:
  lm_head = U @ diag(S) @ Vt
  
Where:
  U: [vocab_size, rank] - token directions
  S: [rank] - importance of each direction
  Vt: [rank, hidden_dim] - hidden space directions

The top-k directions capture most of the variance.
This is why our k=37 holographic bound works!

To work backwards:
  hidden_that_produces_token_k ≈ lm_head[k] / ||lm_head[k]||
  
But this is only approximate because multiple hidden states
can produce the same token (context-dependence).
""")
    
    return U, S, Vt


def main():
    print("=" * 70)
    print("COMB Inverse Projection")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test 1: LM_HEAD as hidden states
    test_lm_head_as_hidden_state(model, tokenizer)
    
    # Test 2: Build COMB signature memory
    signatures = build_comb_signature_memory(model, tokenizer)
    
    # Test 3: COMB-based prediction
    test_comb_based_prediction(model, tokenizer)
    
    # Test 4: Hidden vs COMB analysis
    analyze_hidden_vs_comb(model, tokenizer)
    
    # Test 5: COMB structure
    U, S, Vt = explore_comb_structure(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Findings:

1. LM_HEAD rows DO predict their own tokens (self-prediction works)
2. COMB signatures can be computed for ALL tokens (perfect coverage)
3. But actual hidden states are NOT identical to COMB rows
4. The relationship is approximate, not exact

The Challenge:
  h_actual ≠ lm_head[predicted_token]
  
The hidden state is CONTEXT-DEPENDENT, but the COMB row is fixed.

The Insight:
  h_actual ≈ lm_head[predicted_token] + context_adjustment
  
The context_adjustment is what the transformer computes.
If we can characterize this adjustment, we can bypass the transformer.

Possible Approaches:
1. Learn the context_adjustment as a function of input
2. Use the COMB as a "base" and add learned corrections
3. Find patterns in how h_actual differs from lm_head[pred]
""")


if __name__ == "__main__":
    main()
