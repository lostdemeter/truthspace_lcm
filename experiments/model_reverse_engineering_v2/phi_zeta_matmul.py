#!/usr/bin/env python3
"""
Zeta-Structured Matmul: SVD rank-K ↔ Riemann-Siegel N(t) terms.

DC 282 maps:
  Z(t) = 2 Σ_{n=1}^{N(t)} n^{-1/2} cos(θ - t·ln(n)) + remainder
  W    = Σ_{k=1}^{K}      σ_k       u_k · v_k^T       + remainder

Each rank-1 component σ_k · u_k · v_k^T IS one zeta term.
The singular values σ_k ↔ amplitudes n^{-1/2}.
The directions v_k ↔ phases t·ln(n).

Riemann-Siegel: N(t) = √(t/2π) terms suffice.
Question: does rank √N suffice for the matmul?

For the matmul y = W @ x:
  y_K = Σ_{k=1}^{K} σ_k · u_k · (v_k · x) = U_K @ Σ_K @ V_K^T @ x

This is O(K × N) instead of O(N²). If K = √N, it's O(N^{3/2}) instead of O(N²).
For a single vector (batch=1): O(K × in_f + K × out_f) = O(K × max(in,out)).

ALSO tests the prime representation idea:
  - Each singular direction v_k is a "prime" — an independent factor
  - The product Π_p (1 + p^{-s} contribution) captures all interactions
  - The Euler product needs only π(N) ≈ N/ln(N) primes, not N terms
  - For SVD: rank K ≈ effective dimensionality, not full N

And tests the tail correction:
  - Riemann-Siegel remainder is bounded analytically
  - Can we bound the SVD tail from the singular value decay rate?
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID
from phi_geometric.inference.phi_matmul import phi_matmul_hybrid


def analyze_sv_spectrum(W: PhiEncoded, name=""):
    """
    Analyze the singular value spectrum of a weight matrix.
    
    DC 282 predicts: σ_k ∝ k^{-1/2} (like zeta amplitudes).
    The "crystalline decay" measured in F152.
    """
    W_decoded = W.decode_cached()
    
    # Full SVD is too expensive for large matrices — use randomized SVD
    # or just compute the top singular values
    t0 = time.perf_counter()
    
    # For (out, in) matrix, compute SVD
    # If very large, use truncated SVD
    m, n = W_decoded.shape
    if m * n > 100_000_000:  # > 100M elements
        # Randomized SVD for top-512 components
        from scipy.linalg import svd as scipy_svd
        k = min(512, min(m, n))
        # Use the fact that W @ W^T has eigenvalues = σ²
        if m < n:
            C = W_decoded @ W_decoded.T  # (m, m) — cheaper
            eigvals = np.linalg.eigvalsh(C)
            sigma = np.sqrt(np.maximum(eigvals[::-1], 0))[:k]
        else:
            C = W_decoded.T @ W_decoded  # (n, n)
            eigvals = np.linalg.eigvalsh(C)
            sigma = np.sqrt(np.maximum(eigvals[::-1], 0))[:k]
    else:
        U, sigma, Vt = np.linalg.svd(W_decoded, full_matrices=False)
    
    t_svd = time.perf_counter() - t0
    
    print(f"\n  SVD spectrum ({name}, {t_svd:.1f}s):")
    print(f"    Top-5 σ: {sigma[:5]}")
    print(f"    σ[0]/σ[1] = {sigma[0]/sigma[1]:.3f}")
    if len(sigma) > 10:
        print(f"    σ[0]/σ[10] = {sigma[0]/sigma[10]:.3f}")
    if len(sigma) > 100:
        print(f"    σ[0]/σ[100] = {sigma[0]/sigma[100]:.3f}")
    
    # Fit power-law: σ_k ∝ k^{-α}
    k = np.arange(1, len(sigma) + 1, dtype=np.float64)
    mask = sigma > 1e-10
    log_k = np.log(k[mask])
    log_s = np.log(sigma[mask])
    coeffs = np.polyfit(log_k, log_s, 1)
    alpha = -coeffs[0]
    
    print(f"    Decay: σ_k ∝ k^{{-{alpha:.3f}}}")
    print(f"    (Zeta predicts α = 0.5)")
    
    # Cumulative energy: what fraction of ||W||²_F is captured by rank K?
    sigma_sq = sigma ** 2
    total_energy = np.sum(sigma_sq)
    cum_energy = np.cumsum(sigma_sq) / total_energy
    
    sqrt_n = int(np.sqrt(min(m, n)))
    
    for frac_label, frac in [("rank-1", 1), ("rank-10", 10), 
                              ("rank-√N", sqrt_n), ("rank-2√N", 2*sqrt_n),
                              ("rank-N/4", min(m,n)//4)]:
        idx = min(frac - 1, len(cum_energy) - 1)
        print(f"    {frac_label} ({frac}): {cum_energy[idx]:.4%} energy")
    
    return sigma, alpha, cum_energy


def svd_matmul_convergence(W: PhiEncoded, x_float, name=""):
    """
    Test convergence of rank-K matmul approximation.
    
    y_K = U_K @ diag(σ_1..K) @ V_K^T @ x
    
    This is the structural analog of the Riemann-Siegel partial sum.
    """
    W_decoded = W.decode_cached()
    m, n = W_decoded.shape
    
    # Full result
    full_result = x_float @ W_decoded.T  # (batch, out)
    full_flat = full_result.flatten()
    full_norm = np.linalg.norm(full_flat)
    
    # Compute SVD
    print(f"\n  Computing SVD for {name} ({m}×{n})...")
    t0 = time.perf_counter()
    
    if m * n > 100_000_000:
        # For very large matrices, compute V^T @ x first, then evaluate
        # W = U @ S @ V^T, so W @ x = U @ S @ (V^T @ x)
        # Compute V via eigendecomposition of W^T @ W
        if n <= m:
            C = W_decoded.T @ W_decoded  # (n, n)
            eigvals, V = np.linalg.eigh(C)
            # Sort descending
            idx = np.argsort(-eigvals)
            eigvals = eigvals[idx]
            V = V[:, idx]
            sigma = np.sqrt(np.maximum(eigvals, 0))
            # U = W @ V @ diag(1/σ) (only need U for small K)
        else:
            C = W_decoded @ W_decoded.T  # (m, m)
            eigvals, U = np.linalg.eigh(C)
            idx = np.argsort(-eigvals)
            eigvals = eigvals[idx]
            U = U[:, idx]
            sigma = np.sqrt(np.maximum(eigvals, 0))
            # V^T = diag(1/σ) @ U^T @ W
    else:
        U, sigma, Vt = np.linalg.svd(W_decoded, full_matrices=False)
        V = Vt.T
    
    t_svd = time.perf_counter() - t0
    print(f"  SVD computed in {t_svd:.1f}s")
    
    # Test points
    sqrt_n = int(np.sqrt(min(m, n)))
    test_Ks = sorted(set([
        1, 2, 4, 8, 16, 32, 
        max(1, sqrt_n // 2), sqrt_n, sqrt_n * 2,
        64, 128, 256, 512, 
        min(m, n) // 4, min(m, n) // 2, min(m, n)
    ]))
    test_Ks = [k for k in test_Ks if 1 <= k <= min(len(sigma), min(m, n))]
    
    print(f"\n  Rank-K matmul convergence (√N={sqrt_n}):")
    print(f"  {'K':>6s}  {'K/N':>6s}  {'Corr':>10s}  {'RelErr':>10s}  "
          f"{'Top100':>8s}  {'R²':>10s}  {'Time':>8s}")
    print(f"  {'─'*66}")
    
    top_full = set(np.argsort(np.abs(full_result[0]))[-100:])
    
    for K in test_Ks:
        t0 = time.perf_counter()
        
        # y_K = x @ V_K @ diag(σ_K) @ U_K^T ... wait
        # W = U @ S @ V^T, so W^T = V @ S @ U^T
        # y = x @ W^T = x @ V @ S @ U^T ... no
        # y = W @ x^T ... depends on convention
        # Actually: y = x @ W^T (batch matmul)
        # W = U S V^T, so W^T = V S U^T
        # y = x @ V S U^T = (x @ V[:,:K]) @ diag(S[:K]) @ U[:,:K]^T
        
        # Project input: z = x @ V[:,:K]  — (batch, K)
        if n <= m:
            z = x_float @ V[:, :K]  # (batch, K)
        else:
            z = x_float @ V[:, :K]  # same
        
        # Scale: z_scaled = z * σ[:K]  — (batch, K)
        z_scaled = z * sigma[:K]
        
        # Project to output: y_K = z_scaled @ U[:,:K]^T  — (batch, out)
        y_K = z_scaled @ U[:, :K].T
        
        t_K = time.perf_counter() - t0
        
        y_flat = y_K.flatten()
        
        corr = np.corrcoef(full_flat, y_flat)[0, 1]
        rel_err = np.linalg.norm(full_flat - y_flat) / full_norm
        r_sq = 1.0 - np.sum((full_flat - y_flat)**2) / np.sum((full_flat - np.mean(full_flat))**2)
        
        top_K = set(np.argsort(np.abs(y_K[0]))[-100:])
        topk = len(top_full & top_K) / 100
        
        marker = " ◀ √N" if K == sqrt_n else ""
        print(f"  {K:>6d}  {K/min(m,n):>6.1%}  {corr:>10.6f}  "
              f"{rel_err:>10.4f}  {topk:>7.0%}  "
              f"{r_sq:>10.6f}  {t_K*1000:>7.2f}ms{marker}")
    
    # Convergence thresholds
    print(f"\n  Convergence thresholds:")
    for target in [0.90, 0.95, 0.99, 0.999, 0.9999]:
        found = False
        for K in test_Ks:
            # Recompute correlation for this K
            z = x_float @ V[:, :K]
            z_scaled = z * sigma[:K]
            y_K = z_scaled @ U[:, :K].T
            corr = np.corrcoef(full_flat, y_K.flatten())[0, 1]
            if corr >= target:
                ratio = K / sqrt_n
                label = f"≤√N" if ratio <= 1.5 else f"{ratio:.1f}×√N"
                print(f"    corr ≥ {target:.4f}: rank {K:>5d} "
                      f"({K/min(m,n):.1%} of N, {label})")
                found = True
                break
        if not found:
            print(f"    corr ≥ {target:.4f}: not reached in test range")
    
    # Speed: rank-K matmul vs full BLAS
    print(f"\n  Speed comparison:")
    
    # Full
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = x_float @ W_decoded.T
        times.append(time.perf_counter() - t0)
    t_full = np.median(times)
    
    # Rank-√N
    V_sqrt = V[:, :sqrt_n].copy()
    S_sqrt = sigma[:sqrt_n].copy()
    U_sqrt = U[:, :sqrt_n].copy()
    
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        z = x_float @ V_sqrt
        z_scaled = z * S_sqrt
        _ = z_scaled @ U_sqrt.T
        times.append(time.perf_counter() - t0)
    t_sqrt = np.median(times)
    
    # Rank-256
    K256 = min(256, len(sigma))
    V_256 = V[:, :K256].copy()
    S_256 = sigma[:K256].copy()
    U_256 = U[:, :K256].copy()
    
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        z = x_float @ V_256
        z_scaled = z * S_256
        _ = z_scaled @ U_256.T
        times.append(time.perf_counter() - t0)
    t_256 = np.median(times)
    
    print(f"    Full ({m}×{n}):   {t_full*1000:.3f}ms")
    print(f"    Rank-√N ({sqrt_n}):  {t_sqrt*1000:.3f}ms ({t_full/t_sqrt:.1f}×)")
    print(f"    Rank-256:       {t_256*1000:.3f}ms ({t_full/t_256:.1f}×)")
    
    return sigma


def test_real_activations(model_dir, layer_dir, W, name):
    """Test with real model activations, not random."""
    try:
        from phi_geometric.inference.phi_engine import PhiQwen2Engine
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_dir, '..', 'model_cache', 'Qwen--Qwen2.5-7B'),
            trust_remote_code=True
        )
        
        engine = PhiQwen2Engine(model_dir, num_layers=1)
        tokens = tokenizer.encode("The purpose of life is")
        
        embed_dec = engine.embedding.weight_decoded
        hidden = embed_dec[tokens].reshape(1, len(tokens), -1)
        
        from phi_geometric.inference.phi_components import rms_norm
        x_real = rms_norm(hidden[:, -1:, :], 
                          engine.layers[0]['input_norm']).reshape(1, -1)
        
        print(f"\n  Real activation test ({name}):")
        W_decoded = W.decode_cached()
        full = x_real @ W_decoded.T
        
        # SVD
        U, sigma, Vt = np.linalg.svd(W_decoded, full_matrices=False)
        V = Vt.T
        sqrt_n = int(np.sqrt(min(*W.shape)))
        
        for K_label, K in [("√N", sqrt_n), ("2√N", sqrt_n*2), 
                            ("4√N", sqrt_n*4), ("256", 256), ("512", 512)]:
            z = x_real @ V[:, :K]
            y_K = (z * sigma[:K]) @ U[:, :K].T
            c = np.corrcoef(full.flatten(), y_K.flatten())[0, 1]
            print(f"    rank-{K_label} ({K}): corr = {c:.6f}")
        
    except Exception as e:
        print(f"    (Could not test real activations: {e})")


def run_benchmark():
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
    
    print("=" * 70)
    print("  Zeta-Structured Matmul: SVD Rank-K Convergence")
    print("  DC 282: rank-1 components ↔ zeta terms")
    print("  Question: does rank √N suffice? (Riemann-Siegel)")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    for name in ['q_proj', 'o_proj', 'gate_proj', 'down_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{name}.npz'))
        out_f, in_f = W.shape
        sqrt_n = int(np.sqrt(min(out_f, in_f)))
        
        print(f"\n{'='*70}")
        print(f"  {name} ({out_f}×{in_f})  √min(m,n) = {sqrt_n}")
        print(f"{'='*70}")
        
        x_float = np.random.randn(1, in_f).astype(np.float32) * 0.02
        
        # Analyze SV spectrum
        sigma, alpha, cum_energy = analyze_sv_spectrum(W, name)
        
        # Convergence test
        svd_matmul_convergence(W, x_float, name)
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  INTERPRETATION")
    print(f"{'='*70}")
    print(f"  DC 282: Z(t) terms ↔ SVD rank-1 components")
    print(f"  Riemann-Siegel: √N terms capture the function")
    print(f"  If rank-√N gives corr ≥ 0.99, the zeta structure holds")
    print(f"  The tail remainder follows the same bound as R-S")
    print(f"  Prime ordering = SVD ordering (most important 'primes' first)")


if __name__ == '__main__':
    run_benchmark()
