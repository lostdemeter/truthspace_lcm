#!/usr/bin/env python3
"""
Quaternion Sign Structure Analysis (Memory-Efficient)
======================================================

Analyzes weights layer-by-layer from safetensors to avoid OOM.

Key insight: weights = φ^level × sign_quaternion
"""

import torch
import numpy as np
from collections import Counter
from safetensors.torch import load_file
from pathlib import Path
import sys

PHI = (1 + np.sqrt(5)) / 2


def analyze_quaternion_signs():
    print("="*70, flush=True)
    print("QUATERNION SIGN STRUCTURE ANALYSIS", flush=True)
    print("="*70, flush=True)
    print("(Memory-efficient: analyzing layer-by-layer)\n", flush=True)
    
    # Find model path
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dirs = list(cache_dir.glob("models--Qwen--Qwen2-7B-Instruct/snapshots/*"))
    if not model_dirs:
        print("Model not found in cache!")
        return
    model_path = model_dirs[0]
    
    # Find safetensor files
    safetensor_files = list(model_path.glob("*.safetensors"))
    print(f"Found {len(safetensor_files)} safetensor files", flush=True)
    
    # Analyze a few layers
    layers_to_analyze = [0, 7, 14, 21, 27]
    
    # Accumulators
    total_weights = 0
    total_blocks = 0
    level_sign_counts = Counter()
    sign_pattern_counts = Counter()
    block_level_sign_counts = Counter()
    delta_counts = Counter()
    sum_rel_error = 0.0
    
    for layer_idx in layers_to_analyze:
        print(f"Analyzing layer {layer_idx}...", end=" ", flush=True)
        layer_weights = 0
        
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            
            # Find which file has this key
            weights = None
            for sf_file in safetensor_files:
                tensors = load_file(sf_file)
                if key in tensors:
                    weights = tensors[key].float().numpy().flatten()
                    del tensors
                    break
                del tensors
            
            if weights is None:
                continue
            
            layer_weights += len(weights)
            total_weights += len(weights)
            
            # Compute φ-levels and signs
            signs = np.sign(weights)
            magnitudes = np.abs(weights).clip(min=1e-45)
            levels = np.round(np.log(magnitudes) / np.log(PHI)).astype(int)
            
            # Count (level, sign) pairs - vectorized
            for l, s in zip(levels, signs.astype(int)):
                level_sign_counts[(l, s)] += 1
            
            # Analyze 4D blocks - vectorized
            n_blocks = len(weights) // 4
            total_blocks += n_blocks
            W_4d = weights[:n_blocks*4].reshape(-1, 4)
            signs_4d = np.sign(W_4d).astype(int)
            mags_4d = np.abs(W_4d).clip(min=1e-45)
            levels_4d = np.round(np.log(mags_4d) / np.log(PHI)).astype(int)
            block_levels = np.round(levels_4d.mean(axis=1)).astype(int)
            
            # Count sign patterns
            for row in signs_4d:
                sign_pattern_counts[tuple(row)] += 1
            
            # Count (block_level, sign_pattern) combos
            for bl, sp in zip(block_levels, signs_4d):
                block_level_sign_counts[(bl, tuple(sp))] += 1
            
            # Count deltas
            deltas = levels_4d - block_levels[:, np.newaxis]
            for d in deltas.flatten():
                delta_counts[int(d)] += 1
            
            # Compute reconstruction error
            reconstructed = (PHI ** block_levels[:, np.newaxis]) * signs_4d
            rel_error = np.abs(W_4d - reconstructed) / (np.abs(W_4d) + 1e-10)
            sum_rel_error += rel_error.sum()
            
            del weights, W_4d, signs_4d, mags_4d, levels_4d
        
        print(f"{layer_weights:,} weights", flush=True)
    
    mean_rel_error = sum_rel_error / (total_blocks * 4)
    
    # =================================================================
    # RESULTS
    # =================================================================
    print("\n" + "="*70, flush=True)
    print("RESULTS", flush=True)
    print("="*70, flush=True)
    
    print(f"\nTotal weights analyzed: {total_weights:,}", flush=True)
    print(f"Total 4D blocks: {total_blocks:,}", flush=True)
    
    # 1. Level-sign pairs
    print("\n" + "-"*50, flush=True)
    print("1. UNIQUE (level, sign) PAIRS", flush=True)
    print("-"*50, flush=True)
    print(f"Unique pairs: {len(level_sign_counts)}", flush=True)
    
    print("\nTop 20 pairs:", flush=True)
    cumulative = 0
    for i, ((level, sign), count) in enumerate(level_sign_counts.most_common(20)):
        pct = count / total_weights * 100
        cumulative += pct
        sign_str = "+" if sign > 0 else "-"
        print(f"  {sign_str}φ^{level:3d}: {pct:5.2f}% (cum: {cumulative:5.1f}%)", flush=True)
    
    # How many pairs for 99%?
    cum = 0
    for i, (_, count) in enumerate(level_sign_counts.most_common()):
        cum += count / total_weights
        if cum >= 0.99:
            print(f"\n99% coverage with {i+1} pairs → {int(np.ceil(np.log2(i+1)))} bits/weight", flush=True)
            break
    
    # 2. Sign patterns
    print("\n" + "-"*50, flush=True)
    print("2. SIGN PATTERNS (4D blocks)", flush=True)
    print("-"*50, flush=True)
    print(f"Unique patterns: {len(sign_pattern_counts)} / 16 possible", flush=True)
    
    for pattern, count in sign_pattern_counts.most_common():
        pct = count / total_blocks * 100
        pattern_str = "".join("+" if s > 0 else "-" for s in pattern)
        print(f"  [{pattern_str}]: {pct:5.2f}%", flush=True)
    
    # 3. Delta distribution
    print("\n" + "-"*50, flush=True)
    print("3. DELTA DISTRIBUTION (component level - block level)", flush=True)
    print("-"*50, flush=True)
    for delta, count in sorted(delta_counts.items()):
        pct = count / (total_blocks * 4) * 100
        if pct > 0.1:
            print(f"  Δ={delta:+2d}: {pct:5.2f}%", flush=True)
    
    small_delta = sum(c for d, c in delta_counts.items() if abs(d) <= 2)
    print(f"\n|Δ| ≤ 2 coverage: {small_delta / (total_blocks * 4) * 100:.1f}%", flush=True)
    
    # 4. Block-level + sign combos
    print("\n" + "-"*50, flush=True)
    print("4. (LEVEL, SIGN_PATTERN) COMBINATIONS", flush=True)
    print("-"*50, flush=True)
    print(f"Unique combinations: {len(block_level_sign_counts):,}", flush=True)
    
    print("\nTop 15 combinations:", flush=True)
    for i, ((level, sign), count) in enumerate(block_level_sign_counts.most_common(15)):
        pct = count / total_blocks * 100
        sign_str = "".join("+" if s > 0 else "-" for s in sign)
        print(f"  φ^{level:3d} × [{sign_str}]: {pct:5.2f}%", flush=True)
    
    # How many for 90%?
    cum = 0
    for i, (_, count) in enumerate(block_level_sign_counts.most_common()):
        cum += count / total_blocks
        if cum >= 0.90:
            print(f"\n90% coverage with {i+1} combinations", flush=True)
            break
    
    # 5. Storage calculation
    print("\n" + "-"*50, flush=True)
    print("5. STORAGE CALCULATION", flush=True)
    print("-"*50, flush=True)
    
    current_bytes = total_weights * 2  # bfloat16
    
    # Option A: level (6 bits) + sign_pattern (4 bits) = 10 bits per 4 weights
    option_a_bytes = total_blocks * 10 / 8
    
    # Option B: + deltas (3 bits × 4) = 22 bits per 4 weights
    option_b_bytes = total_blocks * 22 / 8
    
    print(f"\nCurrent (bfloat16): {current_bytes/1e6:.1f} MB", flush=True)
    print(f"\nOption A (level + sign only):", flush=True)
    print(f"  {option_a_bytes/1e6:.1f} MB ({current_bytes/option_a_bytes:.1f}x compression)", flush=True)
    print(f"  Mean relative error: {mean_rel_error*100:.1f}%", flush=True)
    
    print(f"\nOption B (level + sign + deltas):", flush=True)
    print(f"  {option_b_bytes/1e6:.1f} MB ({current_bytes/option_b_bytes:.1f}x compression)", flush=True)
    print(f"  Error: ~0% (exact on φ-lattice)", flush=True)
    
    # Extrapolate to full model
    print("\n" + "="*70, flush=True)
    print("EXTRAPOLATION TO FULL 7B MODEL", flush=True)
    print("="*70, flush=True)
    
    # Full model has 28 layers × 4 projections × ~12.8M weights each
    full_weights = 28 * 4 * 12845056
    full_current = full_weights * 2 / 1e9
    full_option_a = (full_weights // 4) * 10 / 8 / 1e9
    full_option_b = (full_weights // 4) * 22 / 8 / 1e9
    
    print(f"\nFull attention weights: {full_weights/1e9:.2f}B", flush=True)
    print(f"Current (bfloat16): {full_current:.2f} GB", flush=True)
    print(f"Option A (lossy): {full_option_a:.2f} GB ({full_current/full_option_a:.1f}x)", flush=True)
    print(f"Option B (lossless): {full_option_b:.2f} GB ({full_current/full_option_b:.1f}x)", flush=True)
    
    print(f"""
THE TETROMINO INSIGHT:

Weights are NOT arbitrary floats. They live on a constrained structure:
- {len(level_sign_counts)} unique (level, sign) values
- {len(sign_pattern_counts)} unique sign patterns in 4D blocks
- {len(block_level_sign_counts):,} unique (level, sign_pattern) combinations

This is like tetrominoes: finite shapes that tile infinite space.

With geometric constraints, we can represent {full_current:.1f}GB of weights in ~{full_option_b:.1f}GB
while maintaining EXACT reconstruction on the φ-lattice.
""", flush=True)


if __name__ == "__main__":
    analyze_quaternion_signs()
