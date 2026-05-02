#!/usr/bin/env python3
"""
Tetromino-Based Prediction
===========================

Key insight from the previous experiment:
- Shape agreement alone is ~55% (barely above random)
- But dot product = sign × magnitude
- We need BOTH sign AND magnitude (φ-level)

From Doc 162 (Tetromino):
- Only ~300 unique (level, sign_pattern) combinations
- 90% coverage with 73 combinations

The prediction is:
  h @ lm_head[t] = Σ_i h[i] × lm_head[t][i]
                 = Σ_i (sign_h[i] × φ^level_h[i]) × (sign_t[i] × φ^level_t[i])
                 = Σ_i sign_h[i] × sign_t[i] × φ^(level_h[i] + level_t[i])

If signs agree: contribution is POSITIVE
If signs disagree: contribution is NEGATIVE

The magnitude depends on φ-levels.

Key insight: We can compute this with INTEGER ARITHMETIC!
- sign_h[i] × sign_t[i] is just XOR (1 bit)
- level_h[i] + level_t[i] is integer addition
- Final sum uses LUT for φ^level

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


def quantize_to_phi_lattice(tensor, n_levels=166):
    """
    Quantize tensor values to φ-lattice levels.
    
    Returns (signs, levels) where:
    - signs: +1 or -1
    - levels: integer φ-level (log_φ of magnitude)
    """
    signs = torch.sign(tensor)
    signs[signs == 0] = 1  # Handle zeros
    
    magnitudes = tensor.abs()
    
    # φ-level = log_φ(magnitude)
    levels = torch.log(magnitudes + 1e-10) / np.log(PHI)
    levels = levels.round().long()
    
    return signs.int(), levels


def tetromino_dot_product(h_signs, h_levels, t_signs, t_levels, phi_lut):
    """
    Compute dot product using tetromino representation.
    
    h @ t = Σ_i sign_h[i] × sign_t[i] × φ^(level_h[i] + level_t[i])
    
    This is INTEGER arithmetic + LUT lookup!
    """
    # Sign agreement: +1 if same, -1 if different
    sign_product = h_signs * t_signs  # +1 or -1
    
    # Level sum
    level_sum = h_levels + t_levels
    
    # Clamp to LUT range
    level_sum = level_sum.clamp(-50, 50)
    
    # Lookup φ^level
    magnitudes = phi_lut[level_sum + 50]  # Offset by 50 to handle negatives
    
    # Compute dot product
    contributions = sign_product.float() * magnitudes
    
    return contributions.sum().item()


def build_phi_lut(n_levels=101):
    """
    Build lookup table for φ^level.
    
    Covers levels from -50 to +50.
    """
    lut = torch.zeros(n_levels)
    for i in range(n_levels):
        level = i - 50  # -50 to +50
        lut[i] = PHI ** level
    return lut


def test_tetromino_prediction(model, tokenizer):
    """
    Test prediction using tetromino representation.
    """
    print("\n" + "=" * 70)
    print("Tetromino-Based Prediction")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.clone()
    phi_lut = build_phi_lut()
    
    # Precompute lm_head tetromino representation
    print("\nQuantizing lm_head to φ-lattice...")
    lm_signs, lm_levels = quantize_to_phi_lattice(lm_head)
    print(f"  Shape: {lm_signs.shape}")
    print(f"  Level range: [{lm_levels.min().item()}, {lm_levels.max().item()}]")
    
    test_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest planet is",
        "Two plus two equals",
        "The opposite of hot is",
        "I went to the store and",
        "The book is on the",
        "The quick brown fox jumps over the",
    ]
    
    correct_float = 0
    correct_tetromino = 0
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Method 1: Standard float dot product
        logits_float = h_final @ lm_head.T
        pred_float = logits_float.argmax().item()
        pred_text_float = tokenizer.decode([pred_float])
        
        if pred_float == true_token:
            correct_float += 1
        
        # Method 2: Tetromino dot product
        h_signs, h_levels = quantize_to_phi_lattice(h_final)
        
        # Compute tetromino dot product for each token
        logits_tetromino = torch.zeros(lm_head.shape[0])
        
        for t in range(min(10000, lm_head.shape[0])):  # Limit for speed
            logits_tetromino[t] = tetromino_dot_product(
                h_signs, h_levels,
                lm_signs[t], lm_levels[t],
                phi_lut
            )
        
        pred_tetromino = logits_tetromino[:10000].argmax().item()
        pred_text_tetromino = tokenizer.decode([pred_tetromino])
        
        if pred_tetromino == true_token:
            correct_tetromino += 1
        
        # Compare
        float_marker = "✓" if pred_float == true_token else "✗"
        tetromino_marker = "✓" if pred_tetromino == true_token else "✗"
        
        print(f"\n  {prompt!r}")
        print(f"    True: {true_text!r}")
        print(f"    Float: {pred_text_float!r} {float_marker}")
        print(f"    Tetromino: {pred_text_tetromino!r} {tetromino_marker}")
        
        # Check correlation between float and tetromino logits
        correlation = torch.corrcoef(torch.stack([
            logits_float[:10000],
            logits_tetromino[:10000]
        ]))[0, 1].item()
        print(f"    Logit correlation: {correlation:.4f}")
    
    print(f"\nFloat accuracy: {correct_float}/{len(test_prompts)} = {correct_float/len(test_prompts)*100:.1f}%")
    print(f"Tetromino accuracy: {correct_tetromino}/{len(test_prompts)} = {correct_tetromino/len(test_prompts)*100:.1f}%")


def analyze_tetromino_structure(model, tokenizer):
    """
    Analyze the tetromino structure of hidden states and lm_head.
    """
    print("\n" + "=" * 70)
    print("Tetromino Structure Analysis")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.clone()
    
    # Quantize lm_head
    lm_signs, lm_levels = quantize_to_phi_lattice(lm_head)
    
    # Count unique (level, sign) pairs
    unique_pairs = set()
    for i in range(min(10000, lm_head.shape[0])):
        for j in range(lm_head.shape[1]):
            pair = (lm_levels[i, j].item(), lm_signs[i, j].item())
            unique_pairs.add(pair)
    
    print(f"Unique (level, sign) pairs in lm_head: {len(unique_pairs)}")
    
    # Level distribution
    level_counts = defaultdict(int)
    for i in range(min(10000, lm_head.shape[0])):
        for j in range(lm_head.shape[1]):
            level_counts[lm_levels[i, j].item()] += 1
    
    print(f"\nLevel distribution (top 10):")
    for level, count in sorted(level_counts.items(), key=lambda x: -x[1])[:10]:
        pct = count / (10000 * lm_head.shape[1]) * 100
        print(f"  Level {level}: {count} ({pct:.2f}%)")
    
    # Sign distribution
    pos_count = (lm_signs[:10000] > 0).sum().item()
    neg_count = (lm_signs[:10000] < 0).sum().item()
    total = pos_count + neg_count
    
    print(f"\nSign distribution:")
    print(f"  Positive: {pos_count} ({pos_count/total*100:.1f}%)")
    print(f"  Negative: {neg_count} ({neg_count/total*100:.1f}%)")
    
    # Analyze hidden states
    print("\n--- Hidden State Tetromino Structure ---")
    
    prompts = [
        "The capital of France is",
        "Two plus two equals",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
        
        h_signs, h_levels = quantize_to_phi_lattice(h_final)
        
        print(f"\n  {prompt!r}")
        print(f"    Level range: [{h_levels.min().item()}, {h_levels.max().item()}]")
        print(f"    Mean level: {h_levels.float().mean().item():.2f}")
        print(f"    Positive signs: {(h_signs > 0).sum().item()}/{len(h_signs)}")


def test_integer_prediction(model, tokenizer):
    """
    Test if we can predict using ONLY integer operations.
    
    This is the key test for replacing the transformer with shape lookup.
    """
    print("\n" + "=" * 70)
    print("Integer-Only Prediction Test")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.clone()
    
    # Quantize lm_head
    lm_signs, lm_levels = quantize_to_phi_lattice(lm_head)
    
    test_prompts = [
        "The capital of France is",
        "Two plus two equals",
        "The quick brown fox jumps over the",
    ]
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Quantize hidden state
        h_signs, h_levels = quantize_to_phi_lattice(h_final)
        
        # Integer-only prediction:
        # Score = Σ_i sign_agreement[i] × (level_h[i] + level_t[i])
        # This is purely integer arithmetic!
        
        scores = torch.zeros(min(10000, lm_head.shape[0]), dtype=torch.long)
        
        for t in range(len(scores)):
            # Sign agreement: +1 if same, -1 if different
            sign_agree = (h_signs == lm_signs[t]).long() * 2 - 1  # 0→-1, 1→+1
            
            # Level sum (integer)
            level_sum = h_levels + lm_levels[t]
            
            # Score = Σ sign_agree × level_sum
            # Higher level_sum means larger magnitude
            # Positive sign_agree means aligned
            scores[t] = (sign_agree * level_sum).sum()
        
        pred_int = scores.argmax().item()
        pred_text_int = tokenizer.decode([pred_int])
        
        # Compare with float
        logits_float = h_final @ lm_head.T
        pred_float = logits_float[:10000].argmax().item()
        pred_text_float = tokenizer.decode([pred_float])
        
        print(f"\n  {prompt!r}")
        print(f"    True: {true_text!r}")
        print(f"    Float: {pred_text_float!r}")
        print(f"    Integer: {pred_text_int!r}")
        
        # Check if integer prediction matches float
        if pred_int == pred_float:
            print(f"    ✓ Integer matches float!")
        else:
            print(f"    ✗ Integer differs from float")
            
            # Analyze why
            true_score_int = scores[true_token].item()
            pred_score_int = scores[pred_int].item()
            true_score_float = logits_float[true_token].item()
            pred_score_float = logits_float[pred_float].item()
            
            print(f"    True token scores: int={true_score_int}, float={true_score_float:.2f}")
            print(f"    Pred token scores: int={pred_score_int}, float={pred_score_float:.2f}")


def main():
    print("=" * 70)
    print("Tetromino-Based Prediction")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Analysis 1: Tetromino structure
    analyze_tetromino_structure(model, tokenizer)
    
    # Analysis 2: Tetromino prediction
    test_tetromino_prediction(model, tokenizer)
    
    # Analysis 3: Integer-only prediction
    test_integer_prediction(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Tetromino Prediction:

The prediction h @ lm_head[t] can be computed as:
  Σ_i sign_h[i] × sign_t[i] × φ^(level_h[i] + level_t[i])

This is:
1. Sign XOR (1 bit operation)
2. Level addition (integer operation)
3. φ^level lookup (LUT)
4. Sum (accumulator)

If tetromino prediction matches float prediction, we can:
- Replace float matrix multiply with integer operations
- Use LUT for φ^level
- Achieve significant speedup

The key question: Can we compute the hidden state's tetromino
representation WITHOUT running all 28 layers?
""")


if __name__ == "__main__":
    main()
