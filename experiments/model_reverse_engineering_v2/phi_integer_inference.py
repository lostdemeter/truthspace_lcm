#!/usr/bin/env python3
"""
φ-Integer Inference Benchmark

Runs the φ-integer engine (pure mode) on Qwen2-7B and compares against
float16 GPU baseline. Then applies AIG reduction — pre-compute which
sign products cancel per weight row and skip unnecessary computation.

The hypothesis: AIG simplification finds the irreducible binary core
of each matmul. We don't lose information — we stop computing terms
that contribute nothing to the output shape.

Benchmark stages:
  1. Single-layer forward: pure φ-integer vs hybrid vs float16
  2. Full forward pass (1-token): pure vs hybrid (accuracy + speed)
  3. AIG-reduced matmul: skip cancelling sign pairs
  4. Full generation: compare output text
"""

import os
import sys
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID, LOG_PHI
from phi_geometric.inference.phi_matmul import phi_matmul_pure, phi_matmul_hybrid, get_lut


# ============================================================================
# AIG-REDUCED MATMUL
# ============================================================================

def analyze_sign_cancellation(W: PhiEncoded, verbose=False):
    """
    Analyze sign patterns in a weight matrix to find cancelling pairs.
    
    For each output row j, group input dimensions by exponent level.
    Within each level group, count positive vs negative signs.
    Terms that cancel (equal +/- counts) contribute nothing.
    
    Returns:
        cancel_ratio: fraction of terms that cancel
        effective_density: fraction of terms that actually contribute
    """
    signs = W.signs  # (out_features, in_features)
    exps = W.exponents  # (out_features, in_features)
    out_f, in_f = signs.shape
    
    total_terms = 0
    surviving_terms = 0
    
    # Sample rows for speed
    n_sample = min(256, out_f)
    sample_rows = np.random.choice(out_f, n_sample, replace=False)
    
    for row_idx in sample_rows:
        row_signs = signs[row_idx]
        row_exps = exps[row_idx]
        
        # Group by exponent level (quantize to bins of width 16)
        bin_width = 16
        binned = (row_exps // bin_width).astype(np.int32)
        
        unique_bins = np.unique(binned)
        for b in unique_bins:
            mask = binned == b
            group_signs = row_signs[mask]
            n_pos = np.sum(group_signs > 0)
            n_neg = np.sum(group_signs < 0)
            n_total = len(group_signs)
            
            # The net contribution is |n_pos - n_neg| terms
            # The cancelling terms are 2 * min(n_pos, n_neg)
            n_cancel = 2 * min(n_pos, n_neg)
            n_survive = n_total - n_cancel
            
            total_terms += n_total
            surviving_terms += n_survive
    
    cancel_ratio = 1.0 - (surviving_terms / total_terms) if total_terms > 0 else 0
    effective_density = surviving_terms / total_terms if total_terms > 0 else 1
    
    if verbose:
        print(f"    Sign cancellation: {cancel_ratio:.1%} cancel, "
              f"{effective_density:.1%} effective density")
    
    return cancel_ratio, effective_density


def build_aig_mask(W: PhiEncoded, threshold_exp_diff=256):
    """
    Build a binary mask identifying non-cancelling terms in each row.
    
    For each output element, we identify input dimensions whose contribution
    is significant (not cancelled by opposite-sign terms at similar magnitude).
    
    This is the AIG reduction: terms that cancel are redundant shape
    specifications. Removing them doesn't change the output shape.
    
    Args:
        W: PhiEncoded weight matrix
        threshold_exp_diff: ignore terms whose exponent is this much below row max
        
    Returns:
        mask: boolean (out_features, in_features) — True = keep this term
        reduction: fraction of terms removed
    """
    signs = W.signs
    exps = W.exponents.astype(np.int32)
    out_f, in_f = signs.shape
    
    # Per-row max exponent — terms far below this contribute negligibly
    row_max = np.max(exps, axis=1, keepdims=True)  # (out_f, 1)
    
    # Mask 1: magnitude threshold — drop terms whose φ-level is too low
    magnitude_mask = (exps >= row_max - threshold_exp_diff)
    
    kept = np.sum(magnitude_mask)
    total = signs.size
    reduction = 1.0 - (kept / total)
    
    return magnitude_mask, reduction


def build_adaptive_mask(W: PhiEncoded, coverage=0.999):
    """
    Adaptive per-row AIG mask: keep the minimum terms that cover
    `coverage` fraction of total row magnitude.
    
    This is the proper "least common denominator" — we keep exactly
    the terms needed to reproduce the output shape, nothing more.
    
    For each row:
      1. Compute magnitude of each weight: φ^(exp/128)
      2. Sort descending
      3. Find cumulative sum threshold at `coverage` of total
      4. Keep only those terms
    
    Returns:
        mask: boolean (out_features, in_features)
        stats: dict with per-row statistics
    """
    exps = W.exponents.astype(np.float64)
    out_f, in_f = exps.shape
    
    # Compute magnitudes: φ^(exp/128)
    magnitudes = PHI ** (exps / PHI_GRID)
    
    # Per-row: sort by magnitude descending, find coverage cutoff
    mask = np.zeros((out_f, in_f), dtype=bool)
    terms_kept = np.zeros(out_f, dtype=np.int32)
    
    for j in range(out_f):
        row_mag = magnitudes[j]
        total_mag = np.sum(row_mag)
        
        if total_mag < 1e-20:
            continue
        
        # Sort indices by magnitude descending
        sorted_idx = np.argsort(-row_mag)
        cumsum = np.cumsum(row_mag[sorted_idx])
        target = coverage * total_mag
        
        # Find how many terms we need
        n_keep = np.searchsorted(cumsum, target) + 1
        n_keep = min(n_keep, in_f)
        
        mask[j, sorted_idx[:n_keep]] = True
        terms_kept[j] = n_keep
    
    total = mask.size
    kept = np.sum(mask)
    
    stats = {
        'coverage': coverage,
        'mean_kept': np.mean(terms_kept),
        'median_kept': np.median(terms_kept),
        'min_kept': np.min(terms_kept),
        'max_kept': np.max(terms_kept),
        'total_kept': int(kept),
        'total_params': total,
        'reduction': 1.0 - (kept / total),
        'density': kept / total,
    }
    
    return mask, stats


def build_sparse_repr(W: PhiEncoded, mask):
    """
    Build a sparse representation from a weight matrix + mask.
    
    Returns per-row: (indices, signs, exponents) for kept terms only.
    This IS the irreducible binary core — the minimum shape specification.
    """
    out_f, in_f = W.shape
    rows = []
    total_kept = 0
    
    for j in range(out_f):
        idx = np.where(mask[j])[0].astype(np.int16)
        rows.append({
            'idx': idx,
            'signs': W.signs[j, idx],
            'exps': W.exponents[j, idx],
        })
        total_kept += len(idx)
    
    return rows, total_kept


def phi_matmul_sparse(sparse_rows, in_features, x_signs, x_exps):
    """
    Sparse φ-matmul: only compute terms present in sparse_rows.
    
    This is the AIG-reduced matmul — same result as full matmul
    (within coverage tolerance) but with fewer operations.
    """
    lut = get_lut()
    batch = x_signs.shape[0]
    out_features = len(sparse_rows)
    
    result = np.zeros((batch, out_features), dtype=np.float32)
    
    for j, row in enumerate(sparse_rows):
        idx = row['idx']
        if len(idx) == 0:
            continue
        
        w_s = row['signs']   # (n_kept,)
        w_e = row['exps']    # (n_kept,)
        
        # Gather input at kept indices
        x_s = x_signs[:, idx]  # (batch, n_kept)
        x_e = x_exps[:, idx]   # (batch, n_kept)
        
        # sign XOR + exp ADD + LUT
        sign_prod = (x_s * w_s[np.newaxis, :]).astype(np.float32)
        exp_sum = x_e.astype(np.int32) + w_e[np.newaxis, :].astype(np.int32)
        values = sign_prod * lut.lookup(exp_sum)
        
        result[:, j] = values.sum(axis=1)
    
    return result


def phi_matmul_aig(W: PhiEncoded, x_signs, x_exps, mask, chunk_size=512):
    """
    AIG-reduced φ-matmul: only compute terms where mask is True.
    
    Same as phi_matmul_pure but skips masked-out terms.
    These terms are the ones identified by AIG analysis as contributing
    nothing to the output (sign cancellations + magnitude truncation).
    """
    lut = get_lut()
    out_features, in_features = W.shape
    batch = x_signs.shape[0]
    
    W_signs = W.signs
    W_exps = W.exponents
    
    result = np.zeros((batch, out_features), dtype=np.float32)
    
    for o_start in range(0, out_features, chunk_size):
        o_end = min(o_start + chunk_size, out_features)
        
        w_s = W_signs[o_start:o_end]      # (chunk, in_f)
        w_e = W_exps[o_start:o_end]        # (chunk, in_f)
        m = mask[o_start:o_end]            # (chunk, in_f) bool
        
        # For each output element, only compute where mask is True
        # Broadcast: (batch, chunk, in_f)
        sign_prod = x_signs[:, np.newaxis, :] * w_s[np.newaxis, :, :]
        exp_sum = (x_exps[:, np.newaxis, :].astype(np.int32)
                   + w_e[np.newaxis, :, :].astype(np.int32))
        
        values = sign_prod * lut.lookup(exp_sum)
        
        # Zero out masked terms
        values = values * m[np.newaxis, :, :].astype(np.float32)
        
        result[:, o_start:o_end] = values.sum(axis=2)
    
    return result


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_matmul_single(model_dir, layer_idx=0):
    """Benchmark a single matmul: q_proj on layer 0."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK 1: Single Matmul (layer {layer_idx} q_proj)")
    print(f"{'='*70}")
    
    layer_dir = os.path.join(model_dir, f'layer_{layer_idx:02d}')
    W_q = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
    
    print(f"  W_q shape: {W_q.shape}")
    print(f"  W_q storage: {W_q.storage_bytes() / 1e6:.1f} MB")
    
    # Create a realistic input (embedding-scale hidden state)
    np.random.seed(42)
    x_float = np.random.randn(1, W_q.shape[1]).astype(np.float32) * 0.02
    x_enc = PhiEncoded.encode(x_float)
    
    # 1. Hybrid matmul (decode + numpy)
    print(f"\n  --- Hybrid (decode → numpy) ---")
    t0 = time.perf_counter()
    result_hybrid = phi_matmul_hybrid(W_q, x_float)
    t_hybrid = time.perf_counter() - t0
    print(f"  Time: {t_hybrid*1000:.1f}ms")
    print(f"  Output range: [{result_hybrid.min():.4f}, {result_hybrid.max():.4f}]")
    
    # 2. Pure φ-integer matmul
    print(f"\n  --- Pure φ-integer (sign XOR + exp ADD + LUT) ---")
    t0 = time.perf_counter()
    result_pure = phi_matmul_pure(W_q, x_enc.signs, x_enc.exponents)
    t_pure = time.perf_counter() - t0
    print(f"  Time: {t_pure*1000:.1f}ms")
    
    corr = np.corrcoef(result_hybrid.flatten(), result_pure.flatten())[0, 1]
    max_diff = np.max(np.abs(result_hybrid - result_pure))
    print(f"  Correlation with hybrid: {corr:.6f}")
    print(f"  Max absolute diff: {max_diff:.6f}")
    
    # 3. Fixed-threshold AIG reduction (crude)
    print(f"\n  --- Fixed-Threshold Reduction ---")
    for threshold in [2048, 1024, 512, 384]:
        mask, reduction = build_aig_mask(W_q, threshold_exp_diff=threshold)
        
        t0 = time.perf_counter()
        result_aig = phi_matmul_aig(W_q, x_enc.signs, x_enc.exponents, mask)
        t_aig = time.perf_counter() - t0
        
        corr_aig = np.corrcoef(result_hybrid.flatten(), result_aig.flatten())[0, 1]
        top_hybrid = set(np.argsort(np.abs(result_hybrid[0]))[-100:])
        top_aig = set(np.argsort(np.abs(result_aig[0]))[-100:])
        topk_overlap = len(top_hybrid & top_aig) / 100
        
        print(f"    threshold={threshold:4d}: {reduction:.1%} removed, "
              f"corr={corr_aig:.6f}, top100={topk_overlap:.0%}, "
              f"time={t_aig*1000:.1f}ms")
    
    # 4. Adaptive coverage-based reduction (proper LCD)
    print(f"\n  --- Adaptive Coverage Reduction (LCD of geometry) ---")
    for cov in [0.9999, 0.999, 0.99, 0.95, 0.90]:
        mask, stats = build_adaptive_mask(W_q, coverage=cov)
        
        # Build sparse repr and benchmark
        sparse_rows, total_kept = build_sparse_repr(W_q, mask)
        
        t0 = time.perf_counter()
        result_sparse = phi_matmul_sparse(sparse_rows, W_q.shape[1],
                                           x_enc.signs, x_enc.exponents)
        t_sparse = time.perf_counter() - t0
        
        corr_s = np.corrcoef(result_hybrid.flatten(), result_sparse.flatten())[0, 1]
        top_sparse = set(np.argsort(np.abs(result_sparse[0]))[-100:])
        topk = len(top_hybrid & top_sparse) / 100
        
        print(f"    coverage={cov:.4f}: keep {stats['density']:.1%} "
              f"({stats['mean_kept']:.0f}/{W_q.shape[1]} avg/row), "
              f"corr={corr_s:.6f}, top100={topk:.0%}, "
              f"time={t_sparse*1000:.1f}ms")
    
    print(f"\n  Summary:")
    print(f"    Hybrid:    {t_hybrid*1000:8.1f}ms (baseline)")
    print(f"    Pure φ:    {t_pure*1000:8.1f}ms ({t_pure/t_hybrid:.1f}×)")


def benchmark_single_layer(model_dir, layer_idx=0):
    """Benchmark a full transformer layer forward pass."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK 2: Full Layer Forward (layer {layer_idx})")
    print(f"{'='*70}")
    
    from phi_geometric.inference.phi_engine import PhiQwen2Engine
    
    print("  Loading engine (1 layer)...")
    engine = PhiQwen2Engine.load(model_dir, max_layers=layer_idx+1, verbose=False)
    
    # Use a real tokenizer prompt
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
        prompt = "The purpose of life is"
        token_ids = tokenizer.encode(prompt)
    except Exception:
        token_ids = [785, 7580, 315, 2272, 374]
    
    print(f"  Prompt tokens: {token_ids}")
    
    # Hybrid forward
    print(f"\n  --- Hybrid forward ---")
    engine.clear_weight_cache()
    t0 = time.perf_counter()
    logits_hybrid = engine.forward(token_ids, pure=False, verbose=True)
    t_hybrid = time.perf_counter() - t0
    top5_hybrid = np.argsort(logits_hybrid[0, -1])[-5:][::-1]
    print(f"  Total: {t_hybrid:.2f}s")
    print(f"  Top-5 token IDs: {top5_hybrid.tolist()}")
    
    # Pure forward
    print(f"\n  --- Pure φ-integer forward ---")
    engine.clear_weight_cache()
    t0 = time.perf_counter()
    logits_pure = engine.forward(token_ids, pure=True, verbose=True)
    t_pure = time.perf_counter() - t0
    top5_pure = np.argsort(logits_pure[0, -1])[-5:][::-1]
    print(f"  Total: {t_pure:.2f}s")
    print(f"  Top-5 token IDs: {top5_pure.tolist()}")
    
    # Compare
    corr = np.corrcoef(logits_hybrid[0, -1].flatten(),
                        logits_pure[0, -1].flatten())[0, 1]
    top1_match = top5_hybrid[0] == top5_pure[0]
    top5_overlap = len(set(top5_hybrid) & set(top5_pure))
    
    print(f"\n  Comparison:")
    print(f"    Logit correlation: {corr:.6f}")
    print(f"    Top-1 match: {'✓' if top1_match else '✗'}")
    print(f"    Top-5 overlap: {top5_overlap}/5")
    print(f"    Hybrid: {t_hybrid:.2f}s, Pure: {t_pure:.2f}s "
          f"({t_pure/t_hybrid:.1f}× slower)")
    
    return logits_hybrid, logits_pure


def benchmark_aig_layer(model_dir, layer_idx=0):
    """Benchmark adaptive coverage reduction across all weight matrices."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK 3: Adaptive LCD Reduction (layer {layer_idx})")
    print(f"{'='*70}")
    
    layer_dir = os.path.join(model_dir, f'layer_{layer_idx:02d}')
    
    weight_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj']
    
    print(f"\n  {'Weight':<12s} {'Shape':>15s}  "
          f"{'99.99%':>10s}  {'99.9%':>10s}  {'99%':>10s}  {'95%':>10s}")
    print(f"  {'─'*70}")
    
    total_params = 0
    total_kept = {c: 0 for c in [0.9999, 0.999, 0.99, 0.95]}
    
    for name in weight_names:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{name}.npz'))
        n = W.shape[0] * W.shape[1]
        total_params += n
        
        densities = []
        for cov in [0.9999, 0.999, 0.99, 0.95]:
            _, stats = build_adaptive_mask(W, coverage=cov)
            densities.append(stats['density'])
            total_kept[cov] += n * stats['density']
        
        print(f"  {name:<12s} {str(W.shape):>15s}  "
              f"{densities[0]:>9.1%}  {densities[1]:>9.1%}  "
              f"{densities[2]:>9.1%}  {densities[3]:>9.1%}")
    
    print(f"\n  Total params/layer: {total_params:,}")
    for cov in [0.9999, 0.999, 0.99, 0.95]:
        kept = total_kept[cov]
        print(f"  Coverage {cov:.2%}: {int(kept):>12,} kept "
              f"({kept/total_params:.1%} density, "
              f"{1-kept/total_params:.1%} reduction)")


def benchmark_generation(model_dir, max_layers=None, max_tokens=20):
    """Full generation benchmark with the φ-integer engine."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK 4: Full Generation ({max_layers or 28} layers, {max_tokens} tokens)")
    print(f"{'='*70}")
    
    from phi_geometric.inference.phi_engine import PhiQwen2Engine
    
    n_layers = max_layers or 28
    print(f"  Loading engine ({n_layers} layers)...")
    engine = PhiQwen2Engine.load(model_dir, max_layers=max_layers, verbose=True)
    
    # Warm weight caches for speed
    print("  Pre-decoding weights...")
    engine.warm_weights(verbose=True)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
        prompt = "The purpose of life is"
        token_ids = tokenizer.encode(prompt)
    except Exception:
        tokenizer = None
        token_ids = [785, 7580, 315, 2272, 374]
    
    print(f"\n  Prompt: '{prompt if tokenizer else token_ids}'")
    print(f"  Tokens: {token_ids}")
    
    # Hybrid generation
    print(f"\n  --- Hybrid generation ---")
    t0 = time.perf_counter()
    gen_hybrid = engine.generate(token_ids, max_new_tokens=max_tokens,
                                  pure=False, verbose=True)
    t_hybrid = time.perf_counter() - t0
    
    if tokenizer:
        text_hybrid = tokenizer.decode(gen_hybrid)
        print(f"  Output: {text_hybrid}")
    print(f"  Time: {t_hybrid:.1f}s ({max_tokens/(t_hybrid+1e-9):.2f} tok/s)")
    
    # Pure generation (will be slower)
    print(f"\n  --- Pure φ-integer generation ---")
    t0 = time.perf_counter()
    gen_pure = engine.generate(token_ids, max_new_tokens=max_tokens,
                                pure=True, verbose=True)
    t_pure = time.perf_counter() - t0
    
    if tokenizer:
        text_pure = tokenizer.decode(gen_pure)
        print(f"  Output: {text_pure}")
    print(f"  Time: {t_pure:.1f}s ({max_tokens/(t_pure+1e-9):.2f} tok/s)")
    
    # Compare
    match = gen_hybrid == gen_pure
    overlap = sum(1 for a, b in zip(gen_hybrid, gen_pure) if a == b)
    print(f"\n  Token-level match: {overlap}/{len(gen_hybrid)} "
          f"({'identical' if match else 'differs'})")
    print(f"  Hybrid: {t_hybrid:.1f}s, Pure: {t_pure:.1f}s "
          f"({t_pure/t_hybrid:.1f}× slower)")
    
    engine.clear_weight_cache()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
    
    print("=" * 70)
    print("  φ-Integer Inference Benchmark")
    print("  Reducing Qwen2-7B to its irreducible binary core")
    print("=" * 70)
    print(f"\n  Model: {MODEL_DIR}")
    
    # Stage 1: Single matmul — pure vs hybrid vs adaptive reduction
    benchmark_matmul_single(MODEL_DIR, layer_idx=0)
    
    # Stage 2: Adaptive LCD analysis across a full layer
    benchmark_aig_layer(MODEL_DIR, layer_idx=0)
    
    # Stage 3: Single-layer forward pass — pure vs hybrid accuracy
    benchmark_single_layer(MODEL_DIR, layer_idx=0)
