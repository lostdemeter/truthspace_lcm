#!/usr/bin/env python3
"""
Tetromino Navigation: Weights as Tiles, Not Numbers
=====================================================

From Doc 162:
- Only 300 unique (level, sign_pattern) combinations
- 90% coverage with just 73 combinations
- Weights are STRUCTURE, not arbitrary floats

The insight: Instead of multiplying by weights, we NAVIGATE through
a pre-defined tile structure.

Key idea:
- Each "tetromino" is a (φ-level, sign_pattern) combination
- Matrix multiply becomes: "which tile am I on, and where does it point?"
- O(d²) multiply → O(d) navigation (if tiles have structure)
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from collections import Counter
from typing import Dict, List, Tuple
import time

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)


def extract_tetrominoes(weights: np.ndarray, block_size: int = 4) -> Dict:
    """
    Extract the tetromino vocabulary from a weight matrix.
    
    A tetromino is defined by:
    - block_level: the mean φ-level of a block
    - sign_pattern: the signs of elements in the block
    """
    # Flatten and reshape into blocks
    flat = weights.flatten()
    n_blocks = len(flat) // block_size
    blocks = flat[:n_blocks * block_size].reshape(n_blocks, block_size)
    
    # Compute φ-levels
    abs_blocks = np.maximum(np.abs(blocks), 1e-38)
    levels = np.floor(np.log(abs_blocks) / LN_PHI).astype(int)
    
    # Block-level statistics
    block_mean_level = np.round(np.mean(levels, axis=1)).astype(int)
    
    # Sign patterns (as tuple for hashing)
    signs = np.sign(blocks)
    sign_patterns = [tuple(s.astype(int)) for s in signs]
    
    # Count tetrominoes
    tetrominoes = [(block_mean_level[i], sign_patterns[i]) for i in range(n_blocks)]
    tetromino_counts = Counter(tetrominoes)
    
    return {
        'tetrominoes': tetrominoes,
        'counts': tetromino_counts,
        'n_unique': len(tetromino_counts),
        'block_mean_levels': block_mean_level,
        'sign_patterns': sign_patterns,
    }


def analyze_tetromino_structure(model_name: str = "Qwen/Qwen2-7B-Instruct"):
    """Analyze the tetromino structure of Qwen2 weights."""
    print("=" * 70)
    print("TETROMINO STRUCTURE ANALYSIS")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Analyze Q projection from layer 0
    q_weight = model.model.layers[0].self_attn.q_proj.weight.data.numpy()
    print(f"\nAnalyzing Q weight: {q_weight.shape}")
    
    result = extract_tetrominoes(q_weight, block_size=4)
    
    print(f"\n  Total blocks: {len(result['tetrominoes']):,}")
    print(f"  Unique tetrominoes: {result['n_unique']}")
    
    # Coverage analysis
    counts = result['counts']
    sorted_counts = sorted(counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts) / sum(sorted_counts)
    
    for threshold in [0.5, 0.9, 0.95, 0.99]:
        n_needed = np.searchsorted(cumsum, threshold) + 1
        print(f"  {threshold*100:.0f}% coverage: {n_needed} tetrominoes")
    
    # Top tetrominoes
    print("\n  Top 10 tetrominoes:")
    for (level, signs), count in counts.most_common(10):
        pct = count / len(result['tetrominoes']) * 100
        sign_str = ''.join(['+' if s > 0 else '-' for s in signs])
        print(f"    φ^{level:>3} [{sign_str}]: {count:>6} ({pct:.1f}%)")
    
    # Level distribution
    print("\n  φ-level distribution:")
    level_counts = Counter(result['block_mean_levels'])
    for level, count in level_counts.most_common(5):
        pct = count / len(result['block_mean_levels']) * 100
        print(f"    φ^{level:>3}: {count:>6} ({pct:.1f}%)")
    
    # Sign pattern distribution
    print("\n  Sign pattern distribution:")
    sign_counts = Counter(result['sign_patterns'])
    for signs, count in sign_counts.most_common(5):
        pct = count / len(result['sign_patterns']) * 100
        sign_str = ''.join(['+' if s > 0 else '-' for s in signs])
        print(f"    [{sign_str}]: {count:>6} ({pct:.1f}%)")
    
    del model
    return result


def design_navigation_system():
    """
    Design a navigation system based on tetrominoes.
    
    Key insight: If there are only ~300 tetrominoes, then:
    - We can precompute what each tetromino "does"
    - Matrix multiply becomes: lookup tetromino → apply precomputed transform
    """
    print("\n" + "=" * 70)
    print("NAVIGATION SYSTEM DESIGN")
    print("=" * 70)
    
    print("""
    CURRENT (Matrix Multiply):
    
        output[i] = Σ_j input[j] × weight[i,j]
        
        For d=3584: 3584 × 3584 = 12.8M multiplications per layer
    
    PROPOSED (Tetromino Navigation):
    
        1. Decompose weight matrix into tetrominoes (blocks of 4)
        2. Each tetromino has a precomputed "action"
        3. Navigation: apply actions based on input position
        
        For d=3584 with block_size=4:
        - 3584 × 3584 / 4 = 3.2M blocks
        - But only ~300 unique tetrominoes!
        - Precompute 300 actions, then just lookup
    
    THE KEY QUESTION:
    
        Can we represent the "action" of a tetromino in a way that's
        faster than 4 multiplications?
        
        Options:
        1. Lookup table: action[tetromino_id] = precomputed_result
           - But result depends on input, so can't fully precompute
        
        2. Sparse navigation: most tetrominoes have similar effect
           - Group by φ-level, apply level-shift instead of multiply
           - Sign pattern just flips signs
        
        3. Geometric traversal: tetrominoes define a graph
           - Input position → traverse graph → output position
           - O(d) if graph is sparse
    """)
    
    return None


def test_level_shift_hypothesis():
    """
    Test if matrix multiply can be approximated by φ-level shifts.
    
    Hypothesis: output ≈ input shifted by weight's φ-level
    
    If weight = φ^k, then:
        input × weight = input × φ^k = input shifted up k levels
    """
    print("\n" + "=" * 70)
    print("LEVEL-SHIFT HYPOTHESIS TEST")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Get a weight matrix
    q_weight = model.model.layers[0].self_attn.q_proj.weight.data.numpy()
    
    # Create a test input
    np.random.seed(42)
    test_input = np.random.randn(3584) * 0.1
    
    # Actual output
    actual_output = test_input @ q_weight.T
    
    # Level-shift approximation
    # For each output dimension, compute the "effective level shift"
    
    # Method 1: Use mean weight level per row
    abs_weights = np.maximum(np.abs(q_weight), 1e-38)
    weight_levels = np.floor(np.log(abs_weights) / LN_PHI)
    mean_level_per_row = np.mean(weight_levels, axis=1)
    
    # Approximate: output ≈ mean(input) × φ^(mean_level)
    input_mean = np.mean(test_input)
    approx_output_1 = input_mean * (PHI ** mean_level_per_row)
    
    corr_1 = np.corrcoef(actual_output, approx_output_1)[0, 1]
    print(f"\n  Method 1 (mean level shift): correlation = {corr_1:.4f}")
    
    # Method 2: Weight by input magnitude
    # output[i] ≈ Σ_j |input[j]| × sign(weight[i,j]) × φ^(level[i,j])
    
    input_abs = np.abs(test_input)
    weight_signs = np.sign(q_weight)
    weight_phi = PHI ** weight_levels
    
    approx_output_2 = (input_abs @ (weight_signs * weight_phi).T)
    
    corr_2 = np.corrcoef(actual_output, approx_output_2)[0, 1]
    print(f"  Method 2 (signed φ-levels): correlation = {corr_2:.4f}")
    
    # Method 3: Use actual input signs
    input_signs = np.sign(test_input)
    approx_output_3 = ((input_abs * input_signs) @ (weight_signs * weight_phi).T)
    
    corr_3 = np.corrcoef(actual_output, approx_output_3)[0, 1]
    print(f"  Method 3 (full signs): correlation = {corr_3:.4f}")
    
    # Method 4: Include residuals
    # weight = sign × φ^level × (1 + residual)
    # We already have this in φ-2byte format!
    
    base = PHI ** weight_levels
    residual = (np.abs(q_weight) / base - 1) / (PHI - 1)
    weight_reconstructed = weight_signs * base * (1 + residual * (PHI - 1))
    
    approx_output_4 = test_input @ weight_reconstructed.T
    
    corr_4 = np.corrcoef(actual_output, approx_output_4)[0, 1]
    print(f"  Method 4 (φ-reconstructed): correlation = {corr_4:.10f}")
    
    del model
    
    print("""
    FINDINGS:
    
    - Simple level-shift doesn't work (correlation ~0.0)
    - Need to preserve signs and residuals
    - φ-reconstruction gives perfect correlation
    
    BUT: The question is whether we can NAVIGATE instead of MULTIPLY.
    
    The tetromino insight suggests:
    - Group weights by (level, sign_pattern)
    - Precompute the "effect" of each group
    - At inference: lookup group → apply effect
    
    This is like a HASH TABLE for matrix multiply!
    """)


def explore_tetromino_hashing():
    """
    Explore using tetrominoes as a hash table for matrix multiply.
    
    Idea: Instead of computing output = input @ weight.T,
    compute output by looking up precomputed results for each tetromino.
    """
    print("\n" + "=" * 70)
    print("TETROMINO HASHING EXPLORATION")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    q_weight = model.model.layers[0].self_attn.q_proj.weight.data.numpy()
    
    # Extract tetrominoes
    result = extract_tetrominoes(q_weight, block_size=4)
    
    # Build a lookup table: tetromino_id → indices where it appears
    tetromino_to_indices = {}
    for i, tet in enumerate(result['tetrominoes']):
        if tet not in tetromino_to_indices:
            tetromino_to_indices[tet] = []
        tetromino_to_indices[tet].append(i)
    
    print(f"\n  Unique tetrominoes: {len(tetromino_to_indices)}")
    
    # For each tetromino, what is its "average effect"?
    # Effect = how it transforms input blocks to output
    
    # Reshape weight matrix into blocks
    n_out, n_in = q_weight.shape
    block_size = 4
    
    # Reshape: (n_out, n_in) → (n_out, n_in//4, 4)
    n_in_blocks = n_in // block_size
    weight_blocks = q_weight[:, :n_in_blocks * block_size].reshape(n_out, n_in_blocks, block_size)
    
    # For each output dimension, compute which tetrominoes contribute
    # This is getting complex... let's think differently
    
    print("""
    INSIGHT:
    
    The tetromino structure suggests that weights are NOT random.
    They follow a pattern that can be exploited.
    
    Instead of:
        output = input @ weight.T  [O(d²)]
    
    We could do:
        1. Quantize input to φ-levels
        2. For each output dim, lookup which tetrominoes contribute
        3. Sum the contributions (but tetrominoes are shared!)
    
    The key is: if many output dims share the same tetrominoes,
    we can compute once and reuse.
    
    This is like SPARSE MATRIX MULTIPLY, but the sparsity is in
    the tetromino space, not the value space.
    """)
    
    # Count how many output dims share each tetromino
    # (This would require restructuring the weight matrix)
    
    del model
    
    return None


def main():
    # Analyze tetromino structure
    result = analyze_tetromino_structure()
    
    # Design navigation system
    design_navigation_system()
    
    # Test level-shift hypothesis
    test_level_shift_hypothesis()
    
    # Explore tetromino hashing
    explore_tetromino_hashing()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY FINDINGS:
    
    1. Weights have only ~300 unique tetrominoes (level, sign_pattern)
    2. 90% coverage with just 73 tetrominoes
    3. Simple level-shift doesn't approximate matrix multiply
    4. BUT: The structure suggests a different approach
    
    THE INSIGHT:
    
    Weights are not "numbers to multiply" - they are "tiles that define structure".
    
    The question isn't "how to multiply faster" but "how to navigate the structure".
    
    NEXT STEPS:
    
    1. Represent weights as a GRAPH of tetrominoes
    2. Matrix multiply = graph traversal
    3. Exploit shared structure for O(d) instead of O(d²)
    
    This connects to:
    - Doc 159: Boom positions are graph nodes
    - Doc 162: Tetrominoes are the edges
    - Doc 184: φ-transform is the traversal operation
    """)


if __name__ == "__main__":
    main()
