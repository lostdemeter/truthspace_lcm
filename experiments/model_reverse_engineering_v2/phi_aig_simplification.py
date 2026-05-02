#!/usr/bin/env python3
"""
AIG (And-Inverter Graph) Simplification for φ-Integer Matmul
=============================================================

From DC 169: φ-lattice reduces AIG gates from 224B → 174M (1,291× fewer).
From DC 287: Gate anti-alternation pattern is AIG-structured.

Key insight: In φ-integer matmul, the sign operations ARE boolean logic:
  - Multiply signs: XOR (1 AIG gate)
  - The sign pattern of each weight row determines which inputs ADD vs SUBTRACT
  - Many sign products CANCEL in accumulation → can be pruned

This script tests:
1. Sign cancellation analysis: how many terms cancel per output element?
2. Level grouping: how many unique levels per row? (fewer = faster)
3. Sparse sign-level matmul: skip groups where signs cancel
4. Benchmark: sparse vs dense φ-integer matmul
"""

import numpy as np
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(__file__))
from phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')


def load_phi_raw(path):
    d = np.load(path)
    return d['signs'], d['exponents']


def analyze_sign_cancellation(w_signs, w_exps, name="weight"):
    """
    Analyze how much sign cancellation occurs per output row.
    
    For each row j of W, group elements by exponent level.
    Within each group, count positive and negative signs.
    Net contribution = |n_pos - n_neg| / (n_pos + n_neg).
    If this is small, the group nearly cancels and can be pruned.
    """
    out_dim, in_dim = w_signs.shape
    print(f"\n  ─── Sign Cancellation Analysis: {name} ({out_dim}×{in_dim}) ───")
    
    # Quantize exponents to levels (group nearby exponents)
    # Use coarser binning for more cancellation
    for bin_size in [1, 4, 8, 16]:
        binned_exps = (w_exps.astype(np.int32) // bin_size) * bin_size
        unique_levels = np.unique(binned_exps)
        n_levels = len(unique_levels)
        
        total_groups = 0
        canceled_groups = 0
        prunable_ops = 0
        total_ops = out_dim * in_dim
        
        # Sample rows for speed
        sample_rows = min(100, out_dim)
        row_indices = np.random.choice(out_dim, sample_rows, replace=False)
        
        for j in row_indices:
            for level in unique_levels:
                mask = binned_exps[j] == level
                n_at_level = np.sum(mask)
                if n_at_level == 0:
                    continue
                
                total_groups += 1
                signs_at_level = w_signs[j, mask]
                n_pos = np.sum(signs_at_level > 0)
                n_neg = np.sum(signs_at_level < 0)
                net = abs(n_pos - n_neg)
                cancel_ratio = 1.0 - (net / n_at_level)
                
                # If cancellation > 90%, group is pruneable
                if cancel_ratio > 0.9:
                    canceled_groups += 1
                    prunable_ops += n_at_level
        
        cancel_pct = canceled_groups / max(total_groups, 1) * 100
        prune_pct = prunable_ops / (sample_rows * in_dim) * 100
        
        print(f"    bin={bin_size:2d}: {n_levels:5d} levels, "
              f"{cancel_pct:5.1f}% groups cancel, "
              f"{prune_pct:5.1f}% ops prunable")
    
    return n_levels


def analyze_level_distribution(w_exps, name="weight"):
    """
    Analyze the distribution of exponent levels in the weight matrix.
    Fewer unique levels = more efficient grouped computation.
    """
    out_dim, in_dim = w_exps.shape
    print(f"\n  ─── Level Distribution: {name} ({out_dim}×{in_dim}) ───")
    
    unique_levels = np.unique(w_exps)
    print(f"    Unique exponent values: {len(unique_levels)}")
    
    # Histogram of exponents
    hist, bin_edges = np.histogram(w_exps.flatten(), bins=50)
    top5_bins = np.argsort(hist)[-5:][::-1]
    print(f"    Top 5 most common ranges:")
    for b in top5_bins:
        lo, hi = bin_edges[b], bin_edges[b+1]
        pct = hist[b] / w_exps.size * 100
        print(f"      [{lo:.0f}, {hi:.0f}): {pct:.1f}% of elements")
    
    # Per-row statistics
    levels_per_row = np.array([len(np.unique(w_exps[j])) for j in range(min(100, out_dim))])
    print(f"    Levels per row: mean={levels_per_row.mean():.0f}, "
          f"min={levels_per_row.min()}, max={levels_per_row.max()}")
    
    return unique_levels


def sparse_phi_matmul(w_signs, w_exps, x_signs, x_exps, 
                       cancel_threshold=0.85, bin_size=4):
    """
    AIG-simplified φ-integer matmul.
    
    Skip groups where sign cancellation exceeds threshold.
    This is the AIG simplification: pruning gates that produce
    near-zero output.
    """
    from phi_integer import get_fixed_lut, _clamp_exps
    
    lut = get_fixed_lut()
    out_dim, in_dim = w_signs.shape
    
    # Bin exponents for grouping
    binned_w_exps = (w_exps.astype(np.int32) // bin_size) * bin_size
    unique_levels = np.unique(binned_w_exps)
    
    result_signs = np.zeros(out_dim, dtype=np.int8)
    result_exps = np.zeros(out_dim, dtype=np.int16)
    
    # Pre-compute per-level masks
    level_masks = {}
    for level in unique_levels:
        level_masks[level] = (binned_w_exps == level)  # (out_dim, in_dim)
    
    # For each output element
    for j in range(out_dim):
        # Accumulate contributions from each level group
        max_exp_overall = -30000
        contributions = []  # (sign, exp) pairs
        
        for level in unique_levels:
            mask_j = level_masks[level][j]
            n_at = np.sum(mask_j)
            if n_at == 0:
                continue
            
            # Check sign cancellation
            sign_products = w_signs[j, mask_j] * x_signs[mask_j]
            n_pos = np.sum(sign_products > 0)
            n_neg = np.sum(sign_products < 0)
            net = abs(n_pos - n_neg)
            cancel = 1.0 - (net / n_at)
            
            if cancel > cancel_threshold:
                continue  # AIG PRUNE: this group cancels
            
            # Compute group contribution using block-scaled accumulation
            prod_exps = w_exps[j, mask_j].astype(np.int32) + x_exps[mask_j].astype(np.int32)
            max_exp = np.max(prod_exps)
            shifted = prod_exps - max_exp
            scaled = lut.lookup_forward(shifted)
            signed_scaled = sign_products.astype(np.int64) * scaled
            total = np.sum(signed_scaled)
            
            if total != 0:
                s = np.int8(1 if total > 0 else -1)
                e_offset = lut.reverse_lookup(np.array([abs(total)]))[0]
                e = np.clip(max_exp + e_offset, -25000, 5000).astype(np.int16)
                contributions.append((s, e))
        
        if not contributions:
            result_signs[j] = 1
            result_exps[j] = -25000
            continue
        
        # Combine contributions
        if len(contributions) == 1:
            result_signs[j] = contributions[0][0]
            result_exps[j] = contributions[0][1]
        else:
            c_signs = np.array([c[0] for c in contributions], dtype=np.int8)
            c_exps = np.array([c[1] for c in contributions], dtype=np.int16)
            from phi_integer import phi_accumulate
            rs, re = phi_accumulate(
                c_signs[np.newaxis, :], c_exps[np.newaxis, :].astype(np.int32), axis=-1)
            result_signs[j] = rs[0] if np.ndim(rs) > 0 else rs
            result_exps[j] = re[0] if np.ndim(re) > 0 else re
    
    return result_signs, result_exps


def benchmark_aig(w_signs, w_exps, x_signs, x_exps, name="weight"):
    """Compare dense vs AIG-simplified matmul."""
    from phi_integer import phi_matmul_integer, phi_to_float
    
    out_dim = w_signs.shape[0]
    print(f"\n  ─── AIG Benchmark: {name} ───")
    
    # Use smaller subset for tractable benchmark
    n_out = min(256, out_dim)
    ws = w_signs[:n_out]
    we = w_exps[:n_out]
    
    w_phi = PhiEncoded(signs=ws, exponents=we)
    xs = x_signs[np.newaxis, :]
    xe = x_exps[np.newaxis, :]
    
    # Dense integer matmul
    t0 = time.time()
    dense_s, dense_e = phi_matmul_integer(w_phi, xs, xe, chunk_size=256)
    t_dense = time.time() - t0
    dense_result = phi_to_float(dense_s[0], dense_e[0])
    
    # AIG-simplified matmul (various thresholds)
    for thresh in [0.70, 0.85, 0.95]:
        t0 = time.time()
        aig_s, aig_e = sparse_phi_matmul(ws, we, x_signs, x_exps,
                                          cancel_threshold=thresh)
        t_aig = time.time() - t0
        aig_result = phi_to_float(aig_s, aig_e)
        
        corr = np.corrcoef(dense_result, aig_result)[0, 1]
        speedup = t_dense / t_aig
        print(f"    cancel>{thresh:.2f}: {t_aig*1000:.1f}ms vs {t_dense*1000:.1f}ms "
              f"({speedup:.2f}×), corr={corr:.6f}")


def main():
    print("=" * 70)
    print("  AIG Simplification for φ-Integer Arithmetic")
    print("  DC 169: 224B gates → 174M gates (1,291× reduction)")
    print("=" * 70)
    
    # Load test vector
    print("\n  Loading test token embedding...", end='', flush=True)
    emb_data = np.load(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    tok_signs = emb_data['signs'][279].copy()
    tok_exps = emb_data['exponents'][279].copy()
    del emb_data; gc.collect()
    print(" done")
    
    # Load weights for analysis
    for layer_idx in [0, 14, 27]:
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")
        
        for proj_name in ['v_proj', 'gate_proj']:
            w_signs, w_exps = load_phi_raw(os.path.join(layer_dir, f'{proj_name}.npz'))
            
            analyze_level_distribution(w_exps, f"L{layer_idx}/{proj_name}")
            analyze_sign_cancellation(w_signs, w_exps, f"L{layer_idx}/{proj_name}")
            benchmark_aig(w_signs, w_exps, tok_signs, tok_exps, 
                         f"L{layer_idx}/{proj_name}")
            
            del w_signs, w_exps; gc.collect()
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print("  The AIG simplification works by:")
    print("  1. Grouping weight elements by exponent level")
    print("  2. Within each group, counting +/- sign products")
    print("  3. Pruning groups where signs nearly cancel (net ≈ 0)")
    print("  4. This is equivalent to removing AIG gates that produce 0")
    print()
    print("  The anti-alternation pattern (DC 287) means adjacent layers")
    print("  have OPPOSITE gate sign patterns → their AIG structures are")
    print("  complementary, enabling cross-layer simplification.")
    print()


if __name__ == '__main__':
    main()
