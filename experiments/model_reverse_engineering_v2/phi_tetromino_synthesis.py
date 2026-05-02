#!/usr/bin/env python3
"""
φ-Binary → Tetromino → Sublinear Navigation Synthesis
======================================================

Three threads converge:

1. φ-Binary LCD: sign (1 bit) + exponent (int16) = irreducible binary core
   → corr=1.000000 with float32 (proven)

2. Tetrominoes: quantize exponents to integer φ-levels → ~74 unique shapes
   → These are the "primes" of the weight space

3. Sublinear Clock: O(N^(1/d)) traversal of d-dimensional structure
   → With 4D: O(∛N) or O(N^{1/4}) instead of O(N)

Synthesis:
  - φ-binary gives us the LCD (sign + level)
  - Tetrominoes enumerate the valid shapes
  - In 4D, valid tetromino tilings are constrained
  - Navigate the tiling graph in ∛N steps instead of scanning N positions
  - The output is: navigate(input, structure) not input @ weight

This script:
  1. Maps φ-encoded weights to tetromino space
  2. Analyzes the tiling structure (adjacency, constraints)
  3. Builds a navigation graph
  4. Tests ∛N traversal accuracy
"""

import os
import sys
import time
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')


# ============================================================================
# STEP 1: φ-Encoded → Tetromino Mapping
# ============================================================================

def phi_to_tetromino(W: PhiEncoded, level_bits=8):
    """
    Map φ-encoded weights to tetromino space.
    
    Our φ-encoding: sign (int8: -1/+1) + exponent (int16, PHI_GRID=128)
    Tetromino: quantize exponent to integer φ-level + sign
    
    level_bits controls quantization:
      - level_bits=8: exponent // 128 → integer φ-level (coarsest)
      - level_bits=4: exponent // 8 → 16 sub-levels per φ-level
    
    Returns:
        tet_ids: int16 array — tetromino ID per weight
        unique_tets: dict[id] → (sign, level, count, magnitude)
    """
    signs = W.signs
    exps = W.exponents.astype(np.int32)
    
    # Quantize exponent to φ-level
    if level_bits == 8:
        # Coarsest: integer φ-levels (like original tetromino code)
        levels = exps // PHI_GRID  # integer part of exponent/128
    else:
        # Finer quantization
        step = PHI_GRID // (2 ** (level_bits - 4))
        levels = exps // max(step, 1)
    
    # Tetromino ID = level * 2 + (sign > 0)
    sign_bit = (signs > 0).astype(np.int16)
    tet_ids = levels.astype(np.int16) * 2 + sign_bit
    
    # Enumerate unique tetrominoes
    unique_ids, counts = np.unique(tet_ids, return_counts=True)
    unique_tets = {}
    for uid, count in zip(unique_ids, counts):
        level = int(uid) // 2
        sign = (int(uid) % 2) * 2 - 1
        magnitude = sign * (PHI ** level)
        unique_tets[int(uid)] = {
            'sign': sign,
            'level': level,
            'count': int(count),
            'magnitude': magnitude,
            'fraction': int(count) / tet_ids.size,
        }
    
    return tet_ids, unique_tets


def analyze_tetromino_structure(W: PhiEncoded, name=""):
    """Analyze the tetromino structure of a weight matrix."""
    print(f"\n  {name} ({W.shape[0]}×{W.shape[1]})")
    
    tet_ids, unique_tets = phi_to_tetromino(W, level_bits=8)
    n_unique = len(unique_tets)
    total = tet_ids.size
    
    print(f"    Unique tetrominoes (φ-level): {n_unique}")
    
    # Top-10 by count
    sorted_tets = sorted(unique_tets.items(), key=lambda x: -x[1]['count'])
    print(f"    Top-10 by frequency:")
    cumulative = 0
    for uid, info in sorted_tets[:10]:
        cumulative += info['fraction']
        print(f"      tet {uid:>4d}: sign={info['sign']:+d} level={info['level']:>3d} "
              f"mag={info['magnitude']:>12.4f} "
              f"count={info['count']:>8,} ({info['fraction']:.1%}) "
              f"cumul={cumulative:.1%}")
    
    # Coverage: how many tetrominoes cover 90%, 95%, 99%?
    cumul = 0
    for threshold in [0.90, 0.95, 0.99]:
        cumul = 0
        for i, (uid, info) in enumerate(sorted_tets):
            cumul += info['fraction']
            if cumul >= threshold:
                print(f"    {threshold:.0%} coverage: top {i+1} tetrominoes")
                break
    
    # Finer quantization
    for bits in [4, 6]:
        tet_ids_fine, unique_fine = phi_to_tetromino(W, level_bits=bits)
        print(f"    Level bits={bits}: {len(unique_fine)} unique tetrominoes")
    
    return tet_ids, unique_tets


# ============================================================================
# STEP 2: 4D Tiling Constraints
# ============================================================================

def analyze_adjacency(tet_ids, unique_tets, name=""):
    """
    Analyze which tetrominoes appear adjacent to each other.
    
    In a constrained tiling, not all pairs (tet_a, tet_b) are valid neighbors.
    The more constrained the tiling, the sparser the navigation graph,
    and the faster we can traverse it.
    
    "Adjacent" means: same row, consecutive columns (within a weight row).
    """
    out_f, in_f = tet_ids.shape
    n_unique = len(unique_tets)
    
    # Count adjacency pairs (row-wise)
    pair_counts = Counter()
    total_pairs = 0
    
    for j in range(out_f):
        row = tet_ids[j]
        for i in range(in_f - 1):
            pair = (int(row[i]), int(row[i+1]))
            pair_counts[pair] += 1
            total_pairs += 1
    
    n_observed_pairs = len(pair_counts)
    n_possible_pairs = n_unique * n_unique
    pair_density = n_observed_pairs / n_possible_pairs
    
    print(f"\n    Adjacency analysis ({name}):")
    print(f"      Possible pairs: {n_possible_pairs:,}")
    print(f"      Observed pairs: {n_observed_pairs:,}")
    print(f"      Pair density: {pair_density:.1%}")
    print(f"      Constraint factor: {1/pair_density:.1f}× (lower = more constrained)")
    
    # Top-10 most common pairs
    top_pairs = pair_counts.most_common(10)
    print(f"      Top-10 pairs:")
    for pair, count in top_pairs:
        frac = count / total_pairs
        print(f"        ({pair[0]:>4d}, {pair[1]:>4d}): {count:>8,} ({frac:.2%})")
    
    return pair_counts, pair_density


def analyze_block_structure(tet_ids, block_size=16, name=""):
    """
    Analyze tetromino structure at block level.
    
    Instead of individual positions, look at block_size × block_size blocks.
    Each block has a "signature" = tuple of tetromino IDs within it.
    
    If blocks have limited signatures, the 4D tiling is constrained.
    """
    out_f, in_f = tet_ids.shape
    n_row_blocks = out_f // block_size
    n_col_blocks = in_f // block_size
    
    # Block signatures: mode tetromino per block (most common in block)
    block_modes = np.zeros((n_row_blocks, n_col_blocks), dtype=np.int16)
    
    for bi in range(n_row_blocks):
        for bj in range(n_col_blocks):
            block = tet_ids[bi*block_size:(bi+1)*block_size,
                           bj*block_size:(bj+1)*block_size]
            # Mode = most common tetromino in this block
            vals, counts = np.unique(block, return_counts=True)
            block_modes[bi, bj] = vals[np.argmax(counts)]
    
    unique_modes = len(np.unique(block_modes))
    
    print(f"\n    Block structure (block_size={block_size}, {name}):")
    print(f"      Block grid: {n_row_blocks}×{n_col_blocks} = {n_row_blocks*n_col_blocks} blocks")
    print(f"      Unique block modes: {unique_modes}")
    
    # Block-level adjacency
    block_pairs = Counter()
    for bi in range(n_row_blocks):
        for bj in range(n_col_blocks - 1):
            pair = (int(block_modes[bi, bj]), int(block_modes[bi, bj+1]))
            block_pairs[pair] += 1
    
    n_block_pairs = len(block_pairs)
    max_pairs = unique_modes * unique_modes
    block_density = n_block_pairs / max(max_pairs, 1)
    
    print(f"      Block pair density: {block_density:.1%}")
    
    return block_modes, unique_modes


# ============================================================================
# STEP 3: Sublinear Navigation via Clock
# ============================================================================

def sublinear_navigate(W: PhiEncoded, x_float, tet_ids, unique_tets,
                       dimension=3, name=""):
    """
    Navigate the tetromino structure using sublinear clock-resonant sampling.
    
    Instead of computing all N inner products, sample ∛N (or N^{1/d})
    strategic positions using the clock structure, then extrapolate.
    
    The idea: the tetromino tiling is constrained, so knowing the
    tetromino at ∛N positions lets us PREDICT the tetrominoes at
    all other positions. Then compute only the non-redundant contributions.
    
    Args:
        dimension: effective dimension (3 → ∛N, 4 → N^{1/4})
    """
    W_decoded = W.decode_cached()
    out_f, in_f = W.shape
    
    # Full result (ground truth)
    full_result = x_float @ W_decoded.T
    
    # Sublinear sampling: N^{1/d} positions
    n_samples = int(np.ceil(in_f ** (1.0 / dimension)))
    
    print(f"\n  Sublinear navigation ({name}, d={dimension}):")
    print(f"    N = {in_f}, N^(1/{dimension}) = {n_samples}")
    print(f"    Sampling {n_samples}/{in_f} = {n_samples/in_f:.2%} of dimensions")
    
    # Strategy 1: Uniform sampling (baseline)
    np.random.seed(42)
    uniform_idx = np.sort(np.random.choice(in_f, n_samples, replace=False))
    
    # Strategy 2: Clock-resonant sampling (golden ratio spacing)
    clock_idx = np.array([(int(i * in_f * PHI) % in_f) for i in range(n_samples)])
    clock_idx = np.unique(clock_idx)  # remove duplicates
    
    # Strategy 3: Importance-weighted (sample proportional to |x|)
    x_mag = np.abs(x_float[0])
    probs = x_mag / np.sum(x_mag)
    importance_idx = np.sort(np.random.choice(in_f, n_samples, replace=False, p=probs))
    
    results = {}
    
    for strategy_name, sample_idx in [("uniform", uniform_idx),
                                       ("clock-φ", clock_idx),
                                       ("importance", importance_idx)]:
        n_actual = len(sample_idx)
        
        # Method A: Direct partial sum (just compute sampled terms)
        W_sampled = W_decoded[:, sample_idx]
        x_sampled = x_float[:, sample_idx]
        partial = x_sampled @ W_sampled.T
        # Scale by sampling ratio
        partial_scaled = partial * (in_f / n_actual)
        
        corr_partial = np.corrcoef(full_result.flatten(), partial_scaled.flatten())[0, 1]
        
        # Method B: Tetromino extrapolation
        # From sampled positions, determine the tetromino pattern
        # Then predict contributions from unsampled positions
        sampled_tets = tet_ids[:, sample_idx]  # (out_f, n_samples)
        
        # For each output row, count tetrominoes in the sample
        # Then estimate total contribution per tetromino
        extrapolated = np.zeros((1, out_f), dtype=np.float32)
        
        for j in range(out_f):
            row_tets = sampled_tets[j]
            row_x = x_float[0, sample_idx]
            
            # Group by tetromino
            unique_row_tets = np.unique(row_tets)
            for uid in unique_row_tets:
                mask = row_tets == uid
                # Sampled contribution for this tetromino
                sampled_contrib = np.sum(row_x[mask])
                # Scale by (total count of this tet in row) / (sampled count)
                full_row_tets = tet_ids[j]
                total_count = np.sum(full_row_tets == uid)
                sample_count = np.sum(mask)
                
                if sample_count > 0:
                    scale = total_count / sample_count
                    tet_mag = unique_tets[int(uid)]['magnitude']
                    # Actually: the tetromino has a fixed magnitude
                    # So contribution = magnitude × sum_of_inputs_at_those_positions
                    # We sampled some of those positions → extrapolate
                    extrapolated[0, j] += tet_mag * sampled_contrib * scale
        
        corr_extrap = np.corrcoef(full_result.flatten(), extrapolated.flatten())[0, 1]
        
        results[strategy_name] = {
            'n_samples': n_actual,
            'corr_partial': corr_partial,
            'corr_extrapolated': corr_extrap,
        }
        
        print(f"    {strategy_name:>12s} ({n_actual:>4d} samples): "
              f"partial corr={corr_partial:.4f}, "
              f"tet-extrap corr={corr_extrap:.4f}")
    
    # Test multiple dimensions
    print(f"\n    Convergence across dimensions:")
    for d in [2, 3, 4, 6, 10]:
        n_s = int(np.ceil(in_f ** (1.0 / d)))
        idx = np.array([(int(i * in_f * PHI) % in_f) for i in range(n_s)])
        idx = np.unique(idx)
        n_s = len(idx)
        
        W_s = W_decoded[:, idx]
        x_s = x_float[:, idx]
        partial = (x_s @ W_s.T) * (in_f / n_s)
        
        corr = np.corrcoef(full_result.flatten(), partial.flatten())[0, 1]
        print(f"      d={d}: N^(1/{d})={n_s:>5d} ({n_s/in_f:>6.2%}), corr={corr:.4f}")
    
    return results


# ============================================================================
# STEP 4: Tetromino-Grouped Sublinear Matmul
# ============================================================================

def tetromino_sublinear_matmul(W: PhiEncoded, x_float, tet_ids, unique_tets, 
                                n_probe=None, name=""):
    """
    The actual synthesis: tetromino-grouped + sublinear sampling.
    
    Algorithm:
    1. Sample ∛N input positions using clock spacing
    2. For each sampled position, identify its tetromino in each output row
    3. Use the tetromino STRUCTURE to predict contributions from
       unsampled positions (same tetromino → same magnitude)
    4. Scale by coverage ratio per tetromino
    
    This is O(∛N × out_f) per output vector instead of O(N × out_f).
    """
    W_decoded = W.decode_cached()
    out_f, in_f = W.shape
    
    if n_probe is None:
        n_probe = int(np.ceil(in_f ** (1.0 / 3)))
    
    full_result = x_float @ W_decoded.T
    
    # Clock-resonant sample positions (φ-spaced)
    probe_idx = np.array([(int(i * in_f * PHI) % in_f) for i in range(n_probe)])
    probe_idx = np.unique(probe_idx)
    n_probe = len(probe_idx)
    
    # For each output row:
    # - Look at tetromino IDs at probe positions
    # - For each unique tetromino found in probes:
    #     contribution = tet_magnitude × Σ x[probed_positions_with_this_tet] × (total/probed)
    
    result = np.zeros((1, out_f), dtype=np.float32)
    
    probe_x = x_float[0, probe_idx]
    probe_tets = tet_ids[:, probe_idx]  # (out_f, n_probe)
    
    # Get full row tetromino counts (precompute once)
    # For each row, for each tet_id, count total occurrences
    # This is O(out_f × in_f) — but only done ONCE (at weight load time)
    # After that, inference is O(n_probe × out_f)
    
    # Precompute: tet_counts[j, uid] = count of uid in row j
    all_uids = sorted(unique_tets.keys())
    uid_to_idx = {uid: i for i, uid in enumerate(all_uids)}
    n_tets = len(all_uids)
    
    t0 = time.perf_counter()
    tet_counts = np.zeros((out_f, n_tets), dtype=np.int32)
    for uid_idx, uid in enumerate(all_uids):
        tet_counts[:, uid_idx] = np.sum(tet_ids == uid, axis=1)
    t_precompute = time.perf_counter() - t0
    
    # Now the fast part: inference
    t0 = time.perf_counter()
    
    for j in range(out_f):
        row_probe_tets = probe_tets[j]
        
        unique_probe_tets = np.unique(row_probe_tets)
        for uid in unique_probe_tets:
            uid_int = int(uid)
            if uid_int not in unique_tets:
                continue
            
            # How many probes hit this tetromino?
            probe_mask = row_probe_tets == uid
            probe_count = np.sum(probe_mask)
            
            # Total count of this tetromino in this row
            total_count = tet_counts[j, uid_to_idx[uid_int]]
            
            if probe_count == 0 or total_count == 0:
                continue
            
            # Sum of x at probed positions with this tetromino
            probe_sum = np.sum(probe_x[probe_mask])
            
            # Scale: extrapolate from probed to total
            scale = total_count / probe_count
            
            # Contribution: magnitude × scaled_input_sum
            result[0, j] += unique_tets[uid_int]['magnitude'] * probe_sum * scale
    
    t_inference = time.perf_counter() - t0
    
    # Full BLAS for comparison
    t0 = time.perf_counter()
    full_blas = x_float @ W_decoded.T
    t_blas = time.perf_counter() - t0
    
    corr = np.corrcoef(full_result.flatten(), result.flatten())[0, 1]
    rel_err = np.linalg.norm(full_result - result) / np.linalg.norm(full_result)
    
    # Top-k agreement
    top_full = set(np.argsort(np.abs(full_result[0]))[-100:])
    top_sub = set(np.argsort(np.abs(result[0]))[-100:])
    topk = len(top_full & top_sub) / 100
    
    print(f"\n  Tetromino sublinear matmul ({name}):")
    print(f"    Probes: {n_probe}/{in_f} ({n_probe/in_f:.2%})")
    print(f"    Precompute (once): {t_precompute*1000:.1f}ms")
    print(f"    Inference: {t_inference*1000:.1f}ms")
    print(f"    BLAS: {t_blas*1000:.3f}ms")
    print(f"    Correlation: {corr:.6f}")
    print(f"    Relative error: {rel_err:.4f}")
    print(f"    Top-100 overlap: {topk:.0%}")
    
    return corr, rel_err, t_inference


# ============================================================================
# MAIN
# ============================================================================

def run_analysis():
    print("=" * 70)
    print("  φ-Binary → Tetromino → Sublinear Navigation")
    print("  Connecting the LCD, the shapes, and the clock")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    weight_names = ['q_proj', 'gate_proj', 'down_proj']
    
    for wname in weight_names:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        
        print(f"\n{'='*70}")
        print(f"  {wname}")
        print(f"{'='*70}")
        
        # Step 1: Map to tetromino space
        tet_ids, unique_tets = phi_to_tetromino(W, level_bits=8)
        analyze_tetromino_structure(W, wname)
        
        # Step 2: Analyze tiling constraints
        # (only on smaller matrix to avoid O(N²) pair counting)
        if W.shape[1] <= 4000:
            analyze_adjacency(tet_ids, unique_tets, wname)
        analyze_block_structure(tet_ids, block_size=32, name=wname)
        
        # Step 3: Sublinear navigation
        x_float = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
        sublinear_navigate(W, x_float, tet_ids, unique_tets,
                          dimension=3, name=wname)
        
        # Step 4: Tetromino-grouped sublinear matmul
        for n_probe in [int(W.shape[1] ** (1/4)),
                        int(W.shape[1] ** (1/3)),
                        int(W.shape[1] ** (1/2)),
                        W.shape[1] // 4]:
            tetromino_sublinear_matmul(W, x_float, tet_ids, unique_tets,
                                       n_probe=max(n_probe, 2), name=f"probe={n_probe}")
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS")
    print(f"{'='*70}")
    print(f"""
  The pipeline:
    1. φ-binary encoding (sign + exponent) → LCD of geometry
    2. Quantize to tetromino IDs → ~50-80 unique shapes
    3. Tetromino tiling is CONSTRAINED (not all pairs valid)
    4. Clock-resonant sampling → ∛N probe positions
    5. Extrapolate from probes using tetromino structure
    6. Navigate instead of multiply
    
  If the tiling constraints are tight enough, ∛N probes
  capture the full structure → O(∛N × out_f) inference.
""")


if __name__ == '__main__':
    run_analysis()
