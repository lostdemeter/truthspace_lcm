#!/usr/bin/env python3
"""
Gerchberg-Saxton Reconstruction of Weight Matrices

Given ONLY:
  - Binary sign matrix S ∈ {+1, -1}^{m×n}
  - Rank-1 envelope vectors u, v (so magnitude ≈ φ^(u⊗v))

Can we recover the FULL weight matrix W = S ⊙ φ^(u⊗v + ε)?

Approaches (batched in chunks of BATCH rows for memory):
  1. GS with oracle FFT magnitudes (upper bound — cheating)
  2. GS with rank-1 FFT magnitudes (practical, no cheating)
  3. Matrix alternating projection: sign + low-rank level (no FFT)
  4. Envelope-aware GS: smooth spectrum + rank-1 blend
"""

import os, sys, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
BATCH = 256  # rows per FFT batch

def levels(W): return W.exponents.astype(np.int32) // PHI_GRID

def extract_rank1(W):
    lvl = levels(W).astype(np.float32)
    U, s, Vt = np.linalg.svd(lvl, full_matrices=False)
    return U[:, 0] * s[0], Vt[0, :], s, lvl

def matmul_corr(W_approx, W_true_dec, n_test=100):
    _, in_f = W_approx.shape
    np.random.seed(42)
    X = np.random.randn(n_test, in_f).astype(np.float32) * 0.02
    Y_true = X @ W_true_dec.T
    Y_approx = X @ W_approx.T
    return np.corrcoef(Y_true.flatten(), Y_approx.flatten())[0, 1]


def gs_batched(S, mag_init, fft_mag_fn, n_iter=20):
    """
    Core GS loop, batched.
    S: sign matrix (m, n) float32
    mag_init: initial magnitude (m, n) float32
    fft_mag_fn: callable(batch_rows) -> fft magnitudes for those rows
    """
    m, n = S.shape
    W = (S * mag_init).astype(np.float64)
    
    for it in range(n_iter):
        for start in range(0, m, BATCH):
            end = min(start + BATCH, m)
            batch = W[start:end]
            target = fft_mag_fn(start, end)
            
            Wf = np.fft.rfft(batch, axis=1)
            Wf = target * np.exp(1j * np.angle(Wf))
            batch = np.fft.irfft(Wf, n=n, axis=1)
            W[start:end] = S[start:end].astype(np.float64) * np.abs(batch)
    
    return W.astype(np.float32)


# ============================================================================
# APPROACH 1: Oracle GS (upper bound — cheating with true FFT)
# ============================================================================

def gs_oracle(W_true_dec, S, u, v, n_iter=20):
    mag_init = (PHI ** np.outer(u, v)).astype(np.float32)
    # Precompute true FFT magnitudes (rfft for speed)
    true_rfft = np.abs(np.fft.rfft(W_true_dec.astype(np.float64), axis=1)).astype(np.float64)
    
    def fft_mag_fn(start, end):
        return true_rfft[start:end]
    
    return gs_batched(S, mag_init, fft_mag_fn, n_iter)


# ============================================================================
# APPROACH 2: Rank-1 GS (practical — no oracle)
# ============================================================================

def gs_rank1(S, u, v, n_iter=20):
    mag_init = (PHI ** np.outer(u, v)).astype(np.float32)
    # FFT magnitudes from rank-1 approximation
    W_r1 = S.astype(np.float64) * mag_init.astype(np.float64)
    r1_rfft = np.abs(np.fft.rfft(W_r1, axis=1))
    
    def fft_mag_fn(start, end):
        return r1_rfft[start:end]
    
    return gs_batched(S, mag_init, fft_mag_fn, n_iter)


# ============================================================================
# APPROACH 3: Alternating projection (sign + low-rank level, NO FFT)
# ============================================================================

def alternating_projection(W_true_dec, S, u, v, n_iter=5, level_rank=5):
    m, n = S.shape
    S64 = S.astype(np.float64)
    W = S64 * (PHI ** np.outer(u, v)).astype(np.float64)
    
    corrs = []
    for it in range(n_iter):
        magnitudes = np.abs(W) + 1e-30
        level = np.log(magnitudes) / np.log(PHI)
        U_l, s_l, Vt_l = np.linalg.svd(level, full_matrices=False)
        level_K = (U_l[:, :level_rank] * s_l[:level_rank]) @ Vt_l[:level_rank, :]
        W = S64 * (PHI ** level_K)
        c = matmul_corr(W.astype(np.float32), W_true_dec)
        corrs.append(c)
    
    return W.astype(np.float32), corrs


# ============================================================================
# APPROACH 4: Envelope-aware GS (smooth spectrum + rank-1 blend)
# ============================================================================

def gs_envelope(S, u, v, n_iter=20, blend=0.7):
    m, n = S.shape
    amp_r1 = (PHI ** np.outer(u, v)).astype(np.float32)
    mag_init = amp_r1.copy()
    
    freqs = np.fft.rfftfreq(n)
    kernel = np.exp(-0.5 * (freqs / 0.3) ** 2)
    kernel = (0.5 + 0.5 * kernel).astype(np.float64)
    
    S64 = S.astype(np.float64)
    amp64 = amp_r1.astype(np.float64)
    W = S64 * amp64
    
    for it in range(n_iter):
        for start in range(0, m, BATCH):
            end = min(start + BATCH, m)
            batch = W[start:end]
            
            Wf = np.fft.rfft(batch, axis=1)
            fft_mag = np.abs(Wf) * kernel[None, :]
            Wf = fft_mag * np.exp(1j * np.angle(Wf))
            batch = np.fft.irfft(Wf, n=n, axis=1)
            mag = np.abs(batch)
            W[start:end] = S64[start:end] * (blend * mag + (1 - blend) * amp64[start:end])
    
    return W.astype(np.float32)


# ============================================================================
# APPROACH 5: Integer-constrained GS
# ============================================================================

def gs_integer_oracle(W_true_dec, S, u, v, n_iter=20):
    """Oracle GS + round levels to nearest integer after each iteration."""
    mag_init = (PHI ** np.outer(u, v)).astype(np.float32)
    true_rfft = np.abs(np.fft.rfft(W_true_dec.astype(np.float64), axis=1))
    m, n = S.shape
    S64 = S.astype(np.float64)
    W = (S64 * mag_init.astype(np.float64))
    LOG_PHI = np.log(PHI)
    
    for it in range(n_iter):
        # GS step: enforce FFT magnitude
        for start in range(0, m, BATCH):
            end = min(start + BATCH, m)
            batch = W[start:end]
            target = true_rfft[start:end]
            Wf = np.fft.rfft(batch, axis=1)
            Wf = target * np.exp(1j * np.angle(Wf))
            batch = np.fft.irfft(Wf, n=n, axis=1)
            W[start:end] = S64[start:end] * np.abs(batch)
        
        # Integer step: round levels to nearest integer
        mag = np.abs(W) + 1e-30
        level = np.log(mag) / LOG_PHI
        level_int = np.round(level).astype(np.float64)
        W = S64 * (PHI ** level_int)
    
    return W.astype(np.float32)


def gs_integer_rank1(S, u, v, n_iter=20):
    """Rank-1 GS + round to integer levels. No oracle."""
    mag_init = (PHI ** np.outer(u, v)).astype(np.float32)
    W_r1 = S.astype(np.float64) * mag_init.astype(np.float64)
    r1_rfft = np.abs(np.fft.rfft(W_r1, axis=1))
    m, n = S.shape
    S64 = S.astype(np.float64)
    W = W_r1.copy()
    LOG_PHI = np.log(PHI)
    
    for it in range(n_iter):
        for start in range(0, m, BATCH):
            end = min(start + BATCH, m)
            batch = W[start:end]
            target = r1_rfft[start:end]
            Wf = np.fft.rfft(batch, axis=1)
            Wf = target * np.exp(1j * np.angle(Wf))
            batch = np.fft.irfft(Wf, n=n, axis=1)
            W[start:end] = S64[start:end] * np.abs(batch)
        
        # Round to integer levels
        mag = np.abs(W) + 1e-30
        level = np.log(mag) / LOG_PHI
        level_int = np.round(level).astype(np.float64)
        W = S64 * (PHI ** level_int)
    
    return W.astype(np.float32)


def rank1_round(S, u, v):
    """Simply round rank-1 level to integer. No GS."""
    lvl_r1 = np.outer(u, v)
    lvl_int = np.round(lvl_r1).astype(np.float64)
    return (S.astype(np.float64) * (PHI ** lvl_int)).astype(np.float32)


# ============================================================================
# Reference: Level SVD (oracle — shows what rank-K level achieves)
# ============================================================================

def level_svd_recon(S, lvl, rank_k):
    U_l, s_l, Vt_l = np.linalg.svd(lvl.astype(np.float64), full_matrices=False)
    lvl_K = (U_l[:, :rank_k] * s_l[:rank_k]) @ Vt_l[:rank_k, :]
    W = S.astype(np.float64) * (PHI ** lvl_K)
    return W.astype(np.float32)


def run():
    print("=" * 70)
    print("  GERCHBERG-SAXTON WEIGHT RECONSTRUCTION")
    print("  From binary sign + rank-1 envelope → full weight matrix")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    # Only q_proj first (3584×3584, manageable). Add gate_proj if it works.
    for wname in ['q_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        S = W.signs.astype(np.float32)
        u, v, sigma, lvl = extract_rank1(W)
        m, n = W.shape
        
        print(f"\n{'='*70}")
        print(f"  {wname} ({m}×{n})")
        print(f"{'='*70}")
        
        # Baselines
        c_sign = matmul_corr(S, W_dec)
        W_r1 = S * (PHI ** np.outer(u, v)).astype(np.float32)
        c_r1 = matmul_corr(W_r1, W_dec)
        
        print(f"\n  Baselines:")
        print(f"    Sign-only:         corr={c_sign:.4f}  (1 bit/elem)")
        print(f"    Sign + rank-1:     corr={c_r1:.4f}  (1 bit + 2 vecs)")
        
        # Reference: oracle level SVD
        print(f"\n  Reference: Level SVD (oracle, shows ceiling for each rank):")
        for K in [1, 2, 5, 10, 20, 50]:
            W_lvl = level_svd_recon(S, lvl, K)
            c = matmul_corr(W_lvl, W_dec)
            print(f"    Level rank={K:>3d}:     corr={c:.4f}")
        
        # APPROACH 1: Oracle GS
        print(f"\n  Approach 1: GS with oracle FFT magnitudes...")
        t0 = time.time()
        W_gs1 = gs_oracle(W_dec, S, u, v, n_iter=20)
        c1 = matmul_corr(W_gs1, W_dec)
        dt1 = time.time() - t0
        print(f"    Oracle GS(20):     corr={c1:.4f}  ({dt1:.1f}s)")
        
        # APPROACH 2: Rank-1 GS
        print(f"\n  Approach 2: GS with rank-1 FFT magnitudes...")
        t0 = time.time()
        W_gs2 = gs_rank1(S, u, v, n_iter=20)
        c2 = matmul_corr(W_gs2, W_dec)
        dt2 = time.time() - t0
        print(f"    Rank-1 GS(20):     corr={c2:.4f}  ({dt2:.1f}s)")
        
        # APPROACH 3: Alternating projection
        print(f"\n  Approach 3: Alternating projection (sign + low-rank level)...")
        for K in [1, 2, 5, 10, 20]:
            t0 = time.time()
            W_ap, corrs = alternating_projection(W_dec, S, u, v, n_iter=3, level_rank=K)
            dt = time.time() - t0
            print(f"    AP rank={K:>3d}:       corr={corrs[-1]:.4f}  ({dt:.1f}s)")
        
        # APPROACH 4: Envelope-aware GS  
        print(f"\n  Approach 4: Envelope-aware GS (smooth + blend)...")
        for bl in [0.5, 0.7, 0.9]:
            t0 = time.time()
            W_gs4 = gs_envelope(S, u, v, n_iter=15, blend=bl)
            c4 = matmul_corr(W_gs4, W_dec)
            dt4 = time.time() - t0
            print(f"    Blend={bl:.1f}:          corr={c4:.4f}  ({dt4:.1f}s)")
        
        # APPROACH 5: Integer-constrained GS
        print(f"\n  Approach 5: Integer-constrained GS...")
        
        # Simple: just round rank-1 to integer
        t0 = time.time()
        W_r1int = rank1_round(S, u, v)
        c_r1int = matmul_corr(W_r1int, W_dec)
        print(f"    Round(rank-1):     corr={c_r1int:.4f}  ({time.time()-t0:.1f}s)")
        
        # GS + integer with oracle FFT
        t0 = time.time()
        W_igs1 = gs_integer_oracle(W_dec, S, u, v, n_iter=15)
        c_igs1 = matmul_corr(W_igs1, W_dec)
        print(f"    Oracle+Int GS(15): corr={c_igs1:.4f}  ({time.time()-t0:.1f}s)")
        
        # GS + integer with rank-1 FFT
        t0 = time.time()
        W_igs2 = gs_integer_rank1(S, u, v, n_iter=15)
        c_igs2 = matmul_corr(W_igs2, W_dec)
        print(f"    Rank1+Int GS(15):  corr={c_igs2:.4f}  ({time.time()-t0:.1f}s)")
        
        # How close are the integer-rounded levels to truth?
        LOG_PHI = np.log(PHI)
        lvl_r1 = np.outer(u, v)
        lvl_r1_int = np.round(lvl_r1)
        lvl_true = lvl.astype(np.float64)
        
        err_continuous = np.mean(np.abs(lvl_r1 - lvl_true))
        err_rounded = np.mean(np.abs(lvl_r1_int - lvl_true))
        exact_match = np.mean(lvl_r1_int == lvl_true)
        print(f"\n    Level accuracy:")
        print(f"      Continuous rank-1 MAE:  {err_continuous:.3f}")
        print(f"      Rounded rank-1 MAE:     {err_rounded:.3f}")
        print(f"      Exact integer match:    {exact_match:.1%}")
        
        # Summary
        print(f"\n  {'─'*55}")
        print(f"  SUMMARY for {wname}:")
        print(f"    Sign only:         {c_sign:.4f}  (1 bit/elem)")
        print(f"    Sign + rank-1:     {c_r1:.4f}  (1 bit + 2 vecs)")
        print(f"    Round(rank-1):     {c_r1int:.4f}  (1 bit + 2 vecs, int levels)")
        print(f"    Oracle GS:         {c1:.4f}  (GS + true FFT)")
        print(f"    Oracle+Int GS:     {c_igs1:.4f}  (GS + true FFT + int)")
        print(f"    Rank-1+Int GS:     {c_igs2:.4f}  (practical + int)")
        print(f"    Rank-1 GS:         {c2:.4f}  (practical GS, no int)")
        print(f"    Full weight:       1.0000  (reference)")
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  CONCLUSION")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
