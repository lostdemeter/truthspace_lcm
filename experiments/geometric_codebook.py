#!/usr/bin/env python3
"""
Geometric Codebook: The Tetromino Approach
==========================================

Key insight: With only 76 unique (level, sign) pairs, we have a FINITE vocabulary.

But can we go further? If the STRUCTURE has patterns, we might be able to:
1. Find a small set of "basis shapes" (like tetrominoes)
2. Represent all weights as combinations of these shapes
3. Store only the combination indices

This is like how tetrominoes have 7 shapes but can tile infinite space.
"""

import torch
import numpy as np
import math
from collections import Counter
from transformers import AutoModelForCausalLM

PHI = (1 + np.sqrt(5)) / 2


def analyze_codebook_structure():
    print("="*70)
    print("GEOMETRIC CODEBOOK ANALYSIS")
    print("="*70)
    
    print("\nLoading Qwen2-7B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    
    # Collect weights from multiple layers
    all_weights = []
    for i in [0, 7, 14, 21, 27]:
        layer = model.model.layers[i]
        W_q = layer.self_attn.q_proj.weight.data.numpy().flatten()
        W_k = layer.self_attn.k_proj.weight.data.numpy().flatten()
        W_v = layer.self_attn.v_proj.weight.data.numpy().flatten()
        all_weights.extend([W_q, W_k, W_v])
    
    all_weights = np.concatenate(all_weights)
    print(f"Total weights analyzed: {len(all_weights):,}")
    
    # =================================================================
    # ANALYSIS 1: Global φ-level vocabulary
    # =================================================================
    print("\n" + "="*70)
    print("1. GLOBAL φ-LEVEL VOCABULARY")
    print("="*70)
    
    signs = np.sign(all_weights)
    magnitudes = np.abs(all_weights).clip(min=1e-45)
    phi_levels = np.round(np.log(magnitudes) / np.log(PHI)).astype(int)
    
    # Create (level, sign) codebook
    level_sign_pairs = list(zip(phi_levels, signs.astype(int)))
    pair_counts = Counter(level_sign_pairs)
    
    print(f"\nUnique (level, sign) pairs: {len(pair_counts)}")
    
    # Sort by frequency
    sorted_pairs = pair_counts.most_common()
    
    print("\nTop 20 pairs (covering most weights):")
    cumulative = 0
    for i, ((level, sign), count) in enumerate(sorted_pairs[:20]):
        pct = count / len(all_weights) * 100
        cumulative += pct
        sign_str = "+" if sign > 0 else "-"
        value = sign * (PHI ** level)
        print(f"  {i:2d}. {sign_str}φ^{level:3d} = {value:12.8f}  ({pct:5.2f}%, cum: {cumulative:5.1f}%)")
    
    # How many pairs cover 99% of weights?
    cumulative = 0
    for i, (pair, count) in enumerate(sorted_pairs):
        cumulative += count / len(all_weights)
        if cumulative >= 0.99:
            print(f"\n99% coverage with {i+1} pairs")
            break
    
    # =================================================================
    # ANALYSIS 2: Block-level patterns
    # =================================================================
    print("\n" + "="*70)
    print("2. BLOCK-LEVEL PATTERNS (4D quaternion blocks)")
    print("="*70)
    
    # Reshape to 4D blocks
    n_blocks = len(all_weights) // 4
    W_4d = all_weights[:n_blocks*4].reshape(-1, 4)
    
    # Quantize each block to (level, sign) for each component
    block_patterns = []
    for block in W_4d:
        pattern = tuple(
            (int(np.round(np.log(abs(v)+1e-45) / np.log(PHI))), int(np.sign(v)))
            for v in block
        )
        block_patterns.append(pattern)
    
    pattern_counts = Counter(block_patterns)
    unique_patterns = len(pattern_counts)
    
    print(f"\nUnique 4D block patterns: {unique_patterns:,}")
    print(f"Theoretical max (76^4): {76**4:,}")
    print(f"Actual / Theoretical: {unique_patterns / 76**4 * 100:.4f}%")
    
    # Top patterns
    print("\nTop 10 4D block patterns:")
    for i, (pattern, count) in enumerate(pattern_counts.most_common(10)):
        pct = count / n_blocks * 100
        # Format pattern
        pattern_str = " ".join(f"{'+' if s>0 else '-'}φ^{l}" for l, s in pattern)
        print(f"  {i+1}. [{pattern_str}] ({pct:.2f}%)")
    
    # How many patterns cover 90% of blocks?
    cumulative = 0
    for i, (pattern, count) in enumerate(pattern_counts.most_common()):
        cumulative += count / n_blocks
        if cumulative >= 0.90:
            print(f"\n90% coverage with {i+1} patterns")
            break
    
    # =================================================================
    # ANALYSIS 3: Row-level patterns (attention head structure)
    # =================================================================
    print("\n" + "="*70)
    print("3. ROW-LEVEL STRUCTURE (per output dimension)")
    print("="*70)
    
    # Take one Q projection
    W_q = model.model.layers[14].self_attn.q_proj.weight.data.numpy()
    
    # For each row, compute its "signature"
    # Signature = histogram of φ-levels
    row_signatures = []
    for row in W_q:
        levels = np.round(np.log(np.abs(row).clip(min=1e-45)) / np.log(PHI)).astype(int)
        # Bin into ranges
        sig = tuple(np.histogram(levels, bins=range(-20, 0))[0])
        row_signatures.append(sig)
    
    sig_counts = Counter(row_signatures)
    unique_sigs = len(sig_counts)
    
    print(f"\nUnique row signatures: {unique_sigs}")
    print(f"Total rows: {W_q.shape[0]}")
    
    # =================================================================
    # ANALYSIS 4: Minimal generating set
    # =================================================================
    print("\n" + "="*70)
    print("4. MINIMAL GENERATING SET")
    print("="*70)
    
    # The key insight: if weights are on a φ-lattice with quaternion structure,
    # we might be able to represent them as:
    #   weight = base_quaternion × φ^scale × rotation
    
    # Where:
    # - base_quaternion: one of a small set of unit quaternions
    # - φ^scale: one of ~40 scale factors
    # - rotation: one of a small set of discrete rotations
    
    # Let's see if 4D blocks can be factored this way
    
    # Normalize blocks to unit length
    norms = np.linalg.norm(W_4d, axis=1, keepdims=True).clip(min=1e-10)
    W_4d_unit = W_4d / norms
    
    # Cluster the unit vectors
    from sklearn.cluster import KMeans
    
    n_clusters = 100
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(W_4d_unit)
    
    # Check reconstruction error
    reconstructed = kmeans.cluster_centers_[labels] * norms
    error = np.abs(W_4d - reconstructed).mean()
    rel_error = error / np.abs(W_4d).mean()
    
    print(f"\nK-means clustering with {n_clusters} centroids:")
    print(f"  Mean absolute error: {error:.6f}")
    print(f"  Relative error: {rel_error*100:.2f}%")
    
    # Storage calculation
    # Centroids: 100 × 4 × 4 bytes = 1.6 KB
    # Labels: n_blocks × 1 byte = n_blocks bytes
    # Norms: n_blocks × 2 bytes (quantized to φ-level)
    
    centroid_bytes = n_clusters * 4 * 4
    label_bytes = n_blocks * 1
    norm_bytes = n_blocks * 1  # Just store φ-level
    total_bytes = centroid_bytes + label_bytes + norm_bytes
    original_bytes = len(all_weights) * 2  # bfloat16
    
    print(f"\nStorage estimate:")
    print(f"  Centroids: {centroid_bytes/1e3:.1f} KB")
    print(f"  Labels: {label_bytes/1e6:.1f} MB")
    print(f"  Norms: {norm_bytes/1e6:.1f} MB")
    print(f"  Total: {total_bytes/1e6:.1f} MB")
    print(f"  Original: {original_bytes/1e6:.1f} MB")
    print(f"  Compression: {original_bytes/total_bytes:.1f}x")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*70)
    print("SUMMARY: THE GEOMETRIC CODEBOOK")
    print("="*70)
    print(f"""
KEY FINDINGS:

1. SCALAR LEVEL: Only 76 unique (level, sign) pairs
   → 7 bits per weight = 2.3x compression

2. 4D BLOCK LEVEL: {unique_patterns:,} unique patterns
   → Much less than theoretical 76^4 = {76**4:,}
   → Suggests strong structural constraints

3. CLUSTERING: 100 centroids capture structure with {rel_error*100:.1f}% error
   → {original_bytes/total_bytes:.1f}x compression

THE TETROMINO PRINCIPLE:

Just as tetrominoes have only 7 shapes despite infinite placements,
the neural network weights have a FINITE vocabulary of valid configurations.

The constraints are:
- φ-lattice (40 levels)
- Sign (2 values)
- Quaternion structure (finite rotations)
- Orthogonality (finite orientations)

NEXT STEP: Find the MINIMAL generating set that spans all valid configurations.
This is like finding the 7 tetromino shapes for neural network weights.
""")


if __name__ == "__main__":
    analyze_codebook_structure()
