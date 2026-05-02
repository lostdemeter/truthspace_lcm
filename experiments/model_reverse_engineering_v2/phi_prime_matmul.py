#!/usr/bin/env python3
"""
Prime-Ordered Sublinear φ-Matmul

DC 282 establishes: the transformer IS a zeta function.
The Riemann-Siegel formula computes Z(t) from √(t/2π) terms, not N.

If the weight matrix matmul has the same structure, then:
  - Sort dimensions by importance (= prime ordering)
  - Top √N terms capture the sum (Riemann-Siegel convergence)
  - The tail follows predictable decay → analytic correction
  - Total: O(√N) per output element instead of O(N)

For 3584 dimensions: √3584 ≈ 60 terms. 60× reduction if it holds.

This script tests the hypothesis by measuring convergence of partial
sums vs K for weight matrices sorted by importance.
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID
from phi_geometric.inference.phi_matmul import phi_matmul_hybrid


def sort_by_importance(W: PhiEncoded):
    """
    Sort weight columns (input dimensions) by importance.
    
    Importance = mean |magnitude| across all output rows.
    This is the "prime ordering" — most important dimensions first.
    The analogy: in Z(t), n=1 contributes most (amplitude 1/√1),
    n=2 next (amplitude 1/√2), etc.
    
    Returns:
        perm: int32 array — dimension ordering (most important first)
        importance: float64 — importance score per dimension (sorted)
    """
    # Compute mean magnitude per column
    exps = W.exponents.astype(np.float64)
    col_mean_mag = np.mean(PHI ** (exps / PHI_GRID), axis=0)
    
    # Sort descending (most important first)
    perm = np.argsort(-col_mean_mag).astype(np.int32)
    importance = col_mean_mag[perm]
    
    return perm, importance


def partial_sum_matmul(W: PhiEncoded, x_float, perm, K):
    """
    Compute matmul using only the top K dimensions (by importance).
    
    y[j] = Σ_{k=0}^{K-1} W_decoded[j, perm[k]] * x[perm[k]]
    
    This is the "partial sum" — like computing Z(t) with N(t) terms.
    """
    W_decoded = W.decode_cached()
    
    # Gather the top-K columns
    top_k_cols = perm[:K]
    W_partial = W_decoded[:, top_k_cols]  # (out_f, K)
    x_partial = x_float[:, top_k_cols]    # (batch, K)
    
    return x_partial @ W_partial.T  # (batch, out_f)


def convergence_analysis(W: PhiEncoded, x_float, perm, full_result):
    """
    Measure how partial sums converge to the full result.
    
    Tests K = 1, 2, 4, 8, 16, ..., √N, ..., N
    Reports correlation and relative error at each K.
    
    The zeta prediction: correlation should hit ~0.99 at K ≈ √N.
    """
    in_f = W.shape[1]
    sqrt_n = int(np.sqrt(in_f))
    
    # Test points: logarithmic spacing + √N + key fractions
    test_Ks = sorted(set([
        1, 2, 4, 8, 16, 32, 
        max(1, sqrt_n // 2), sqrt_n, sqrt_n * 2,
        64, 128, 256, 512, 1024, 
        in_f // 4, in_f // 2, in_f
    ]))
    test_Ks = [k for k in test_Ks if 1 <= k <= in_f]
    
    W_decoded = W.decode_cached()
    full_flat = full_result.flatten()
    full_norm = np.linalg.norm(full_flat)
    
    results = []
    
    for K in test_Ks:
        top_cols = perm[:K]
        W_partial = W_decoded[:, top_cols]
        x_partial = x_float[:, top_cols]
        result_K = x_partial @ W_partial.T
        
        result_flat = result_K.flatten()
        
        # Correlation
        corr = np.corrcoef(full_flat, result_flat)[0, 1]
        
        # Relative error
        rel_err = np.linalg.norm(full_flat - result_flat) / full_norm
        
        # Top-k agreement
        top_full = set(np.argsort(np.abs(full_result[0]))[-100:])
        top_K = set(np.argsort(np.abs(result_K[0]))[-100:])
        topk_match = len(top_full & top_K) / 100
        
        # Explained variance (R²)
        r_squared = 1.0 - np.sum((full_flat - result_flat)**2) / np.sum((full_flat - np.mean(full_flat))**2)
        
        results.append({
            'K': K,
            'corr': corr,
            'rel_err': rel_err,
            'topk': topk_match,
            'r_squared': r_squared,
            'frac': K / in_f,
        })
    
    return results


def tail_correction(W: PhiEncoded, x_float, perm, K):
    """
    Estimate the tail contribution (dimensions K+1..N) analytically.
    
    The zeta remainder after N terms is O(t^{-1/4}).
    For our matmul, the tail should follow a predictable pattern
    based on the decay rate of the importance scores.
    
    Strategy:
      1. Compute the mean and variance of tail contributions
      2. The tail sum ≈ mean * (N - K) for each output element
      3. This is O(1) to compute (not O(N-K))
    """
    W_decoded = W.decode_cached()
    in_f = W.shape[1]
    
    tail_cols = perm[K:]
    W_tail = W_decoded[:, tail_cols]  # (out_f, N-K)
    x_tail = x_float[:, tail_cols]    # (batch, N-K)
    
    # Exact tail contribution (for comparison)
    tail_exact = x_tail @ W_tail.T
    
    # Analytical approximation: each tail column contributes
    # approximately its mean weight × mean input
    # This leverages the decay structure
    
    # Method 1: Mean field approximation
    # Each tail term ≈ E[w] × x[i], and x is random with mean ~0
    # So the tail mean should be near zero (by CLT)
    # But the VARIANCE matters: var(tail) ≈ (N-K) × var(single_term)
    
    # Method 2: Extrapolate from the rate of convergence of partial sums
    # The partial sum at K gives us the sum up to K.
    # The difference between partial sums at K and K-δ tells us the
    # rate of new contributions. Extrapolate to infinity.
    
    # Method 3: Richardson extrapolation
    # Use two partial sums to estimate the limit
    K_half = K // 2
    W_half = W_decoded[:, perm[:K_half]]
    x_half = x_float[:, perm[:K_half]]
    sum_half = x_half @ W_half.T
    
    W_full = W_decoded[:, perm[:K]]
    x_full = x_float[:, perm[:K]]
    sum_K = x_full @ W_full.T
    
    # Richardson: if sum(K) ≈ L - c/K^α, then
    # L ≈ sum(K) + (sum(K) - sum(K/2)) × K/(2K - K) for α=1
    # Simpler: L ≈ 2*sum(K) - sum(K/2) (linear extrapolation)
    correction_linear = sum_K + (sum_K - sum_half)  # = 2*sum_K - sum_half
    
    # Method 4: Use the actual decay rate
    # The importance scores follow a power law: importance[k] ∝ k^{-α}
    # Fit α from the first K terms, then compute Σ_{k>K} k^{-α} analytically
    
    return tail_exact, correction_linear


def measure_decay_exponent(importance):
    """
    Fit the power-law decay exponent: importance[k] ∝ k^{-α}
    
    If α > 1/2, the sum converges at √N rate (zeta-like).
    If α ≈ 1/2, it matches the n^{-1/2} amplitude decay in Z(t).
    """
    k = np.arange(1, len(importance) + 1, dtype=np.float64)
    
    # Log-log fit
    mask = importance > 0
    log_k = np.log(k[mask])
    log_imp = np.log(importance[mask])
    
    # Linear fit in log-log space
    coeffs = np.polyfit(log_k, log_imp, 1)
    alpha = -coeffs[0]  # importance ∝ k^{-alpha}
    
    # Also check if it matches n^{-1/2} (zeta amplitude decay)
    residual_half = np.mean((log_imp - (log_imp[0] - 0.5 * log_k))**2)
    residual_fit = np.mean((log_imp - np.polyval(coeffs, log_k))**2)
    
    return alpha, coeffs, residual_half, residual_fit


def run_benchmark():
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
    
    print("=" * 70)
    print("  Prime-Ordered Sublinear Matmul")
    print("  Testing the zeta convergence hypothesis")
    print("  DC 282: N(t) = √(t/2π) terms suffice")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    
    np.random.seed(42)
    
    for name in ['q_proj', 'gate_proj', 'down_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{name}.npz'))
        out_f, in_f = W.shape
        sqrt_n = int(np.sqrt(in_f))
        
        print(f"\n{'='*70}")
        print(f"  {name} ({out_f}×{in_f})  √N = {sqrt_n}")
        print(f"{'='*70}")
        
        x_float = np.random.randn(1, in_f).astype(np.float32) * 0.02
        
        # Full result (baseline)
        full_result = phi_matmul_hybrid(W, x_float)
        
        # Sort by importance
        perm, importance = sort_by_importance(W)
        
        # --- Decay analysis ---
        alpha, coeffs, res_half, res_fit = measure_decay_exponent(importance)
        print(f"\n  Importance decay: k^{{-{alpha:.3f}}}")
        print(f"  (zeta predicts α = 0.5, i.e. n^{{-1/2}})")
        print(f"  Residual vs n^{{-1/2}}: {res_half:.4f}")
        print(f"  Residual vs k^{{-{alpha:.3f}}}: {res_fit:.4f}")
        
        # Top-10 importance values
        print(f"\n  Top-10 importance scores:")
        for i in range(min(10, len(importance))):
            print(f"    dim {perm[i]:4d}: importance = {importance[i]:.6f}")
        
        # --- Convergence analysis ---
        print(f"\n  Convergence of partial sums:")
        print(f"  {'K':>6s}  {'K/N':>6s}  {'Corr':>10s}  {'RelErr':>10s}  "
              f"{'Top100':>8s}  {'R²':>10s}")
        print(f"  {'─'*58}")
        
        results = convergence_analysis(W, x_float, perm, full_result)
        
        for r in results:
            marker = " ◀ √N" if r['K'] == sqrt_n else ""
            print(f"  {r['K']:>6d}  {r['frac']:>6.1%}  {r['corr']:>10.6f}  "
                  f"{r['rel_err']:>10.4f}  {r['topk']:>7.0%}  "
                  f"{r['r_squared']:>10.6f}{marker}")
        
        # --- Find the K where correlation crosses key thresholds ---
        print(f"\n  Convergence thresholds:")
        for target_corr in [0.90, 0.95, 0.99, 0.999, 0.9999]:
            for r in results:
                if r['corr'] >= target_corr:
                    ratio = r['K'] / sqrt_n
                    label = "≤√N" if ratio <= 1.5 else f"{ratio:.1f}×√N"
                    print(f"    corr ≥ {target_corr:.4f}: K = {r['K']:>5d} "
                          f"({r['frac']:.1%} of N, {label})")
                    break
            else:
                print(f"    corr ≥ {target_corr:.4f}: not reached")
        
        # --- Tail correction test ---
        print(f"\n  Tail correction (Richardson extrapolation):")
        for K in [sqrt_n, sqrt_n * 2, 256]:
            if K > in_f:
                continue
            tail_exact, correction = tail_correction(W, x_float, perm, K)
            
            # Partial sum without correction
            top_cols = perm[:K]
            W_decoded = W.decode_cached()
            partial = x_float[:, top_cols] @ W_decoded[:, top_cols].T
            
            # With correction
            corr_no_fix = np.corrcoef(full_result.flatten(), partial.flatten())[0, 1]
            corr_fixed = np.corrcoef(full_result.flatten(), correction.flatten())[0, 1]
            
            err_no_fix = np.linalg.norm(full_result - partial) / np.linalg.norm(full_result)
            err_fixed = np.linalg.norm(full_result - correction) / np.linalg.norm(full_result)
            
            print(f"    K={K:>5d}: without correction corr={corr_no_fix:.6f} err={err_no_fix:.4f}")
            print(f"    {'':>5s}   with Richardson    corr={corr_fixed:.6f} err={err_fixed:.4f}")
        
        # --- Speed comparison ---
        print(f"\n  Speed comparison:")
        
        # Full BLAS
        _ = W.decode_cached()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = phi_matmul_hybrid(W, x_float)
            times.append(time.perf_counter() - t0)
        t_full = np.median(times)
        
        # Partial (√N terms) BLAS
        top_cols = perm[:sqrt_n]
        W_partial_decoded = W_decoded[:, top_cols].copy()
        x_partial = x_float[:, top_cols].copy()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = x_partial @ W_partial_decoded.T
            times.append(time.perf_counter() - t0)
        t_partial = np.median(times)
        
        # Find K for 0.99 correlation
        K_99 = in_f
        for r in results:
            if r['corr'] >= 0.99:
                K_99 = r['K']
                break
        
        if K_99 < in_f:
            top_99 = perm[:K_99]
            W_99 = W_decoded[:, top_99].copy()
            x_99 = x_float[:, top_99].copy()
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                _ = x_99 @ W_99.T
                times.append(time.perf_counter() - t0)
            t_99 = np.median(times)
        
        print(f"    Full N={in_f}:    {t_full*1000:.3f}ms")
        print(f"    √N={sqrt_n}:       {t_partial*1000:.3f}ms "
              f"({t_full/t_partial:.1f}× faster)")
        if K_99 < in_f:
            print(f"    K99={K_99}:     {t_99*1000:.3f}ms "
                  f"({t_full/t_99:.1f}× faster)")
        
        W.clear_cache()
    
    # --- Multiple input vectors (batch test) ---
    print(f"\n{'='*70}")
    print(f"  BATCH TEST: Does convergence hold across different inputs?")
    print(f"{'='*70}")
    
    W = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
    in_f = W.shape[1]
    sqrt_n = int(np.sqrt(in_f))
    perm, importance = sort_by_importance(W)
    
    n_tests = 20
    corrs_at_sqrt = []
    corrs_at_2sqrt = []
    
    for trial in range(n_tests):
        x = np.random.randn(1, in_f).astype(np.float32) * 0.02
        full = phi_matmul_hybrid(W, x)
        
        W_dec = W.decode_cached()
        
        for K, corr_list in [(sqrt_n, corrs_at_sqrt), (sqrt_n * 2, corrs_at_2sqrt)]:
            top = perm[:K]
            partial = x[:, top] @ W_dec[:, top].T
            c = np.corrcoef(full.flatten(), partial.flatten())[0, 1]
            corr_list.append(c)
    
    print(f"\n  q_proj across {n_tests} random inputs:")
    print(f"  At √N={sqrt_n}: corr = {np.mean(corrs_at_sqrt):.6f} "
          f"± {np.std(corrs_at_sqrt):.6f}")
    print(f"  At 2√N={sqrt_n*2}: corr = {np.mean(corrs_at_2sqrt):.6f} "
          f"± {np.std(corrs_at_2sqrt):.6f}")
    
    # --- Real activations test ---
    print(f"\n  Testing with REAL model activations (not random):")
    try:
        from phi_geometric.inference.phi_engine import PhiQwen2Engine
        
        engine = PhiQwen2Engine(MODEL_DIR, max_layers=1)
        
        # Get real hidden states
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_DIR, '..', '..', 'model_cache', 'Qwen--Qwen2.5-7B'),
            trust_remote_code=True
        )
        tokens = tokenizer.encode("The purpose of life is")
        
        # Get embedding
        embed_dec = engine.embedding.weight_decoded
        hidden = embed_dec[tokens].reshape(1, len(tokens), -1)
        
        # RMS norm (like the model does)
        from phi_geometric.inference.phi_components import rms_norm
        hidden_normed = rms_norm(hidden[:, -1:, :], 
                                  engine.layers[0]['input_norm'])
        
        x_real = hidden_normed.reshape(1, -1)
        
        full_real = phi_matmul_hybrid(W, x_real)
        W_dec = W.decode_cached()
        
        for K_label, K in [("√N", sqrt_n), ("2√N", sqrt_n*2), 
                            ("4√N", sqrt_n*4), ("256", 256)]:
            top = perm[:K]
            partial = x_real[:, top] @ W_dec[:, top].T
            c = np.corrcoef(full_real.flatten(), partial.flatten())[0, 1]
            print(f"    K={K_label} ({K}): corr = {c:.6f}")
        
    except Exception as e:
        print(f"    (Could not load engine: {e})")
    
    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  The zeta hypothesis predicts √N terms suffice.")
    print(f"  For in_features={in_f}, √N = {sqrt_n}.")
    print(f"  If correlation at √N ≈ 0.99+, the hypothesis holds.")
    print(f"  If not, the convergence rate tells us the actual exponent.")


if __name__ == '__main__':
    run_benchmark()
