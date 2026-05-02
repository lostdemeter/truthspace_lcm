#!/usr/bin/env python3
"""
GOP/MGOP/EDP Protocol Stack Applied to the Level Residual ε

The residual ε = lvl_true - round(u⊗v) was declared "incompressible."
The protocols say: ERROR IS SIGNAL. Apply the full stack:

Phase 1 (GOP): Fractal Peel — extract recursive structure from ε
Phase 2 (MGOP): Holographic Scan — FFT of ε, directional energy
Phase 3 (MGOP): Fractal Depth Probe — multi-scale dimension of ε
Phase 4 (MGOP): Zeta/φ Resonance — search for number-theoretic patterns
Phase 5 (EDP): Error-as-Signal — φ-pattern search in ε values
Phase 6 (MGOP): Projection Synthesis — do all projections converge?
"""

import os, sys, time
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')

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


# ============================================================================
# PHASE 1: FRACTAL PEEL (GOP)
# ============================================================================

def phase1_fractal_peel(eps, name=""):
    """Extract recursive structure from residual."""
    print(f"\n  PHASE 1: FRACTAL PEEL ({name})")
    print(f"  {'─'*50}")
    
    m, n = eps.shape
    flat = eps.flatten().astype(np.float64)
    
    # Basic statistics
    print(f"    Shape: {m}×{n} = {m*n:,} elements")
    print(f"    Range: [{eps.min()}, {eps.max()}]")
    print(f"    Mean: {eps.mean():.4f}, Std: {eps.std():.4f}")
    print(f"    Median: {np.median(eps):.1f}")
    
    # Unique value distribution (the "alphabet")
    unique_vals, counts = np.unique(eps, return_counts=True)
    n_unique = len(unique_vals)
    print(f"    Unique values: {n_unique}")
    
    # Top values by frequency
    sorted_idx = np.argsort(-counts)
    print(f"    Top-10 values (freq):")
    cum = 0
    for i in range(min(10, n_unique)):
        idx = sorted_idx[i]
        cum += counts[idx]
        print(f"      ε={unique_vals[idx]:>4.0f}: {counts[idx]:>8,} ({counts[idx]/flat.size:.1%})  cum={cum/flat.size:.1%}")
    
    # Entropy
    probs = counts / flat.size
    entropy = -np.sum(probs * np.log2(probs + 1e-30))
    max_entropy = np.log2(n_unique)
    print(f"    Entropy: {entropy:.3f} bits (max={max_entropy:.1f}, ratio={entropy/max_entropy:.3f})")
    
    # Autocorrelation (spatial structure in rows)
    print(f"\n    Row autocorrelation (lag-1 to lag-10):")
    for lag in [1, 2, 3, 5, 10, 50, 100]:
        if lag >= n: break
        ac = np.mean([np.corrcoef(eps[j, :-lag], eps[j, lag:])[0,1] for j in range(min(200, m))])
        print(f"      lag={lag:>4d}: autocorr={ac:.4f}")
    
    # Column autocorrelation
    print(f"    Col autocorrelation (lag-1 to lag-10):")
    for lag in [1, 2, 3, 5, 10, 50]:
        if lag >= m: break
        ac = np.mean([np.corrcoef(eps[:-lag, i], eps[lag:, i])[0,1] 
                       for i in np.random.choice(n, min(200, n), replace=False)])
        print(f"      lag={lag:>4d}: autocorr={ac:.4f}")
    
    # Resfrac score (predictability from neighbors)
    # Use simple predictor: ε[j,i] ≈ mean of neighbors
    pred = np.zeros_like(eps, dtype=np.float64)
    pred[:, 1:] += eps[:, :-1]
    pred[:, :-1] += eps[:, 1:]
    pred[1:, :] += eps[:-1, :]
    pred[:-1, :] += eps[1:, :]
    # corners/edges have fewer neighbors
    cnt = np.ones_like(eps, dtype=np.float64) * 4
    cnt[0, :] -= 1; cnt[-1, :] -= 1; cnt[:, 0] -= 1; cnt[:, -1] -= 1
    pred /= cnt
    pred_err = np.abs(eps - pred).mean()
    random_err = eps.std()
    resfrac = pred_err / (random_err + 1e-30)
    print(f"\n    Resfrac: ρ={resfrac:.4f} (1.0=random, <0.5=structured)")
    
    return entropy, resfrac, unique_vals, counts


# ============================================================================
# PHASE 2: HOLOGRAPHIC SCAN (MGOP)
# ============================================================================

def phase2_holographic_scan(eps, name=""):
    """FFT analysis of residual — directional energy, complexity."""
    print(f"\n  PHASE 2: HOLOGRAPHIC SCAN ({name})")
    print(f"  {'─'*50}")
    
    m, n = eps.shape
    
    # 2D FFT
    eps_fft = np.fft.fft2(eps.astype(np.float64))
    magnitude = np.abs(np.fft.fftshift(eps_fft))
    
    # Power spectrum
    total_power = np.sum(magnitude ** 2)
    dc_power = magnitude[m//2, n//2] ** 2
    print(f"    DC power / total: {dc_power/total_power:.4f}")
    
    # Radial power spectrum
    cy, cx = m//2, n//2
    Y, X = np.ogrid[:m, :n]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    max_r = min(cy, cx)
    radial_power = np.zeros(max_r)
    for r in range(max_r):
        mask = (R == r)
        if mask.any():
            radial_power[r] = np.mean(magnitude[mask] ** 2)
    
    # Is it flat (white noise) or structured?
    rp_norm = radial_power / (radial_power.sum() + 1e-30)
    spectral_entropy = -np.sum(rp_norm * np.log2(rp_norm + 1e-30))
    max_spectral_entropy = np.log2(max_r)
    flatness = spectral_entropy / max_spectral_entropy
    print(f"    Spectral flatness: {flatness:.4f} (1.0=white noise)")
    
    # Row-wise FFT: average power spectrum
    row_fft = np.abs(np.fft.rfft(eps.astype(np.float64), axis=1))
    avg_row_power = np.mean(row_fft ** 2, axis=0)
    rp = avg_row_power / avg_row_power.sum()
    row_flatness = np.exp(np.mean(np.log(rp + 1e-30))) / np.mean(rp)
    print(f"    Row spectral flatness: {row_flatness:.4f}")
    
    # Peak frequencies in row spectrum
    freq = np.fft.rfftfreq(n)
    top_k = np.argsort(-avg_row_power)[:10]
    print(f"    Top-5 row frequencies:")
    for i in range(5):
        k = top_k[i]
        print(f"      f={freq[k]:.4f} (period={1/(freq[k]+1e-10):.1f}): power={avg_row_power[k]:.2f}")
    
    # Column-wise FFT
    col_fft = np.abs(np.fft.rfft(eps.astype(np.float64), axis=0))
    avg_col_power = np.mean(col_fft ** 2, axis=1)
    
    return flatness, avg_row_power, avg_col_power


# ============================================================================
# PHASE 3: FRACTAL DEPTH PROBE (MGOP)
# ============================================================================

def phase3_fractal_depth(eps, name=""):
    """Multi-scale analysis of residual structure."""
    print(f"\n  PHASE 3: FRACTAL DEPTH PROBE ({name})")
    print(f"  {'─'*50}")
    
    m, n = eps.shape
    
    # Multi-scale: variance at different block sizes
    print(f"    Block-scale variance:")
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        if bs > min(m, n) // 2: break
        # Average over blocks
        mb, nb = m // bs, n // bs
        blocked = eps[:mb*bs, :nb*bs].reshape(mb, bs, nb, bs)
        block_means = blocked.mean(axis=(1, 3))
        block_vars = blocked.var(axis=(1, 3)).mean()
        inter_var = block_means.var()
        total_var = eps[:mb*bs, :nb*bs].var()
        print(f"      bs={bs:>4d}: intra={block_vars:.4f}, inter={inter_var:.4f}, "
              f"ratio={inter_var/(total_var+1e-30):.4f}")
    
    # Row-level structure: how much variance is between rows vs within?
    row_means = eps.mean(axis=1)
    row_var = np.var(row_means)
    total_var = np.var(eps)
    print(f"\n    Row structure: between_rows={row_var:.4f}, total={total_var:.4f}, "
          f"ratio={row_var/total_var:.4f}")
    
    # Column-level structure
    col_means = eps.mean(axis=0)
    col_var = np.var(col_means)
    print(f"    Col structure: between_cols={col_var:.4f}, total={total_var:.4f}, "
          f"ratio={col_var/total_var:.4f}")
    
    # How much of ε is explained by row_mean + col_mean?
    additive = row_means[:, None] + col_means[None, :] - eps.mean()
    eps_residual2 = eps - additive
    explained = 1 - np.var(eps_residual2) / np.var(eps)
    print(f"    Additive (row+col) model explains: {explained:.4f}")
    
    # SVD of residual
    U, s, Vt = np.linalg.svd(eps.astype(np.float64), full_matrices=False)
    cum_energy = np.cumsum(s**2) / np.sum(s**2)
    print(f"\n    Residual SVD:")
    print(f"      σ₁/σ₂={s[0]/s[1]:.3f}")
    for pct in [0.5, 0.9, 0.95, 0.99]:
        rank = np.searchsorted(cum_energy, pct) + 1
        print(f"      rank{int(pct*100):>2d}={rank}")
    
    return explained, s


# ============================================================================
# PHASE 4: ZETA/φ RESONANCE (MGOP + EDP)
# ============================================================================

def phase4_phi_resonance(eps, unique_vals, counts, name=""):
    """Search for φ-patterns, prime resonance, number-theoretic structure."""
    print(f"\n  PHASE 4: φ/ZETA RESONANCE ({name})")
    print(f"  {'─'*50}")
    
    LOG_PHI = np.log(PHI)
    
    # Are the unique values related to φ?
    print(f"    Unique ε values in φ-space:")
    sorted_idx = np.argsort(-counts)
    for i in range(min(20, len(unique_vals))):
        idx = sorted_idx[i]
        val = unique_vals[idx]
        freq = counts[idx]
        if val != 0:
            phi_level = np.log(abs(val)) / LOG_PHI if abs(val) > 0 else 0
            print(f"      ε={val:>4.0f}: freq={freq:>7,} ({freq/eps.size:.1%}), "
                  f"log_φ(|ε|)={phi_level:.3f}")
        else:
            print(f"      ε={val:>4.0f}: freq={freq:>7,} ({freq/eps.size:.1%})")
    
    # Distribution of |ε| values
    abs_eps = np.abs(eps.flatten())
    print(f"\n    |ε| distribution:")
    for threshold in [0, 1, 2, 3, 4, 5, 10]:
        frac = np.mean(abs_eps <= threshold)
        print(f"      |ε| ≤ {threshold}: {frac:.1%}")
    
    # Check: are ε values mostly odd or even?
    flat = eps.flatten().astype(int)
    n_even = np.sum(flat % 2 == 0)
    n_odd = np.sum(flat % 2 == 1)
    print(f"\n    Parity: even={n_even/flat.size:.1%}, odd={n_odd/flat.size:.1%}")
    
    # Check: do ε values cluster around φ-lattice points?
    # φ^k for small k
    print(f"\n    Distance to nearest φ^k (k=-5..5):")
    phi_lattice = np.array([PHI**k for k in range(-5, 6)])
    for i in range(min(10, len(unique_vals))):
        idx = sorted_idx[i]
        val = abs(unique_vals[idx])
        if val > 0:
            dists = np.abs(val - phi_lattice)
            nearest_k = np.argmin(dists) - 5
            nearest_d = dists.min()
            print(f"      |ε|={val:.0f}: nearest φ^{nearest_k}={phi_lattice[nearest_k+5]:.3f}, "
                  f"dist={nearest_d:.3f}")
    
    # Check: are ε spacings related to primes?
    if len(unique_vals) > 5:
        spacings = np.diff(np.sort(unique_vals))
        spacing_counts = Counter(spacings.astype(int))
        print(f"\n    ε value spacings (most common):")
        for val, cnt in spacing_counts.most_common(10):
            print(f"      Δε={val}: {cnt} times")
    
    # Row-level: does the residual for each row have structure?
    print(f"\n    Row-level ε patterns (sample of 5 rows):")
    for j in [0, 500, 1000, 2000, 3000]:
        if j >= eps.shape[0]: break
        row = eps[j]
        row_unique = len(np.unique(row))
        row_range = row.max() - row.min()
        row_mean = row.mean()
        # Run length encoding
        changes = np.sum(row[1:] != row[:-1])
        avg_run = eps.shape[1] / (changes + 1)
        print(f"      Row {j:>5d}: unique={row_unique}, range={row_range:.0f}, "
              f"mean={row_mean:.2f}, avg_run={avg_run:.2f}")


# ============================================================================
# PHASE 5: ERROR-AS-SIGNAL (EDP)
# ============================================================================

def phase5_error_as_signal(eps, u, v, lvl, name=""):
    """The residual IS signal. What equation governs it?"""
    print(f"\n  PHASE 5: ERROR-AS-SIGNAL ({name})")
    print(f"  {'─'*50}")
    
    m, n = eps.shape
    
    # ε = lvl_true - u⊗v
    # Is ε = f(u[j]) + g(v[i]) + noise?  (additive separable)
    row_means = eps.mean(axis=1)
    col_means = eps.mean(axis=0)
    additive = row_means[:, None] + col_means[None, :] - eps.mean()
    eps2 = eps - additive
    
    var_orig = np.var(eps)
    var_after = np.var(eps2)
    print(f"    Additive model: explains {1-var_after/var_orig:.1%} of ε variance")
    
    # If additive works, the row_means and col_means ARE the correction
    # This would mean: lvl ≈ u⊗v + f(row) + g(col) = (u+f)⊗(v+g) approximately?
    # No, it's u⊗v + f + g, which is rank-1 + rank-1 + rank-1 = rank-3 at most
    # Let's test this!
    lvl_r1 = np.outer(u, v)
    lvl_corrected = lvl_r1 + row_means[:, None] + col_means[None, :] - np.mean(lvl_r1)
    
    err_before = np.mean(np.abs(lvl.astype(np.float64) - lvl_r1))
    err_after = np.mean(np.abs(lvl.astype(np.float64) - lvl_corrected))
    print(f"    Level MAE: rank-1={err_before:.3f}, +additive={err_after:.3f}")
    
    # Test matmul with additive correction
    S = np.sign(lvl).astype(np.float64)  # approximate (not exact signs)
    # Actually get the true signs
    
    # Is ε correlated with u or v?
    corr_u = np.corrcoef(row_means, u)[0, 1]
    corr_v = np.corrcoef(col_means, v)[0, 1]
    print(f"    corr(ε_row, u)={corr_u:.4f}")
    print(f"    corr(ε_col, v)={corr_v:.4f}")
    
    # ε as a function of the rank-1 prediction
    lvl_r1_flat = lvl_r1.flatten()
    eps_flat = eps.flatten().astype(np.float64)
    
    # Bin by rank-1 level value
    print(f"\n    ε vs rank-1 level (binned):")
    for lo, hi in [(-10, -5), (-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5), (5, 10)]:
        mask = (lvl_r1_flat >= lo) & (lvl_r1_flat < hi)
        if mask.sum() > 0:
            print(f"      rank1∈[{lo:>3},{hi:>3}): ε_mean={eps_flat[mask].mean():>+.3f}, "
                  f"ε_std={eps_flat[mask].std():.3f}, n={mask.sum():,}")
    
    # The key question: can we predict ε from (u[j], v[i])?
    # If ε = f(u[j]) × g(v[i]) then it's rank-1 in a DIFFERENT basis
    # Let's check the SVD of ε itself
    U_e, s_e, Vt_e = np.linalg.svd(eps.astype(np.float64), full_matrices=False)
    
    print(f"\n    ε SVD:")
    print(f"      σ₁/σ₂={s_e[0]/s_e[1]:.3f}")
    print(f"      Top-5 σ: {s_e[:5].round(2)}")
    cum = np.cumsum(s_e**2) / np.sum(s_e**2)
    for k in [1, 2, 5, 10, 20]:
        print(f"      rank-{k}: explains {cum[k-1]:.4f}")
    
    return row_means, col_means, eps2, s_e


# ============================================================================
# PHASE 6: PROJECTION SYNTHESIS + RECONSTRUCTION (MGOP)
# ============================================================================

def phase6_reconstruction(W, W_dec, S, u, v, lvl, eps, row_means, col_means, s_e, name=""):
    """Try additive correction and low-rank ε for reconstruction."""
    print(f"\n  PHASE 6: PROJECTION SYNTHESIS + RECONSTRUCTION ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    # Baseline
    c_sign = matmul_corr(S, W_dec)
    W_r1 = S * (PHI ** np.outer(u, v)).astype(np.float32)
    c_r1 = matmul_corr(W_r1, W_dec)
    print(f"    Sign-only:         corr={c_sign:.4f}")
    print(f"    Sign + rank-1:     corr={c_r1:.4f}")
    
    # Additive correction: lvl ≈ u⊗v + row_bias + col_bias
    lvl_add = np.outer(u, v) + row_means[:, None] + col_means[None, :] - np.mean(np.outer(u, v))
    W_add = (S.astype(np.float64) * (PHI ** lvl_add)).astype(np.float32)
    c_add = matmul_corr(W_add, W_dec)
    print(f"    + additive bias:   corr={c_add:.4f}  (rank-1 + row/col bias)")
    
    # Additive + round to integer
    lvl_add_int = np.round(lvl_add).astype(np.float64)
    W_add_int = (S.astype(np.float64) * (PHI ** lvl_add_int)).astype(np.float32)
    c_add_int = matmul_corr(W_add_int, W_dec)
    exact_add = np.mean(lvl_add_int == lvl.astype(np.float64))
    print(f"    + additive + int:  corr={c_add_int:.4f}  (exact match={exact_add:.1%})")
    
    # Low-rank ε: use top-K SVD of ε as correction
    U_e, s_e_full, Vt_e = np.linalg.svd(eps.astype(np.float64), full_matrices=False)
    
    print(f"\n    Low-rank ε correction:")
    for K in [1, 2, 5, 10, 20, 50, 100]:
        eps_K = (U_e[:, :K] * s_e_full[:K]) @ Vt_e[:K, :]
        lvl_corrected = np.outer(u, v) + eps_K
        lvl_int = np.round(lvl_corrected).astype(np.float64)
        W_corr = (S.astype(np.float64) * (PHI ** lvl_int)).astype(np.float32)
        c_corr = matmul_corr(W_corr, W_dec)
        exact = np.mean(lvl_int == lvl.astype(np.float64))
        print(f"      ε rank-{K:>3d} + int: corr={c_corr:.4f}  (exact={exact:.1%})")
    
    # Full ε (oracle — the target)
    W_exact = (S.astype(np.float64) * (PHI ** lvl.astype(np.float64))).astype(np.float32)
    c_exact = matmul_corr(W_exact, W_dec)
    print(f"\n    Oracle (exact lvl): corr={c_exact:.4f}")


def run():
    print("=" * 70)
    print("  GOP/MGOP/EDP PROTOCOL: RESIDUAL ANALYSIS")
    print("  'Error IS signal. The residual has structure. Find it.'")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    for wname in ['q_proj', 'gate_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        S = W.signs.astype(np.float32)
        u, v, sigma, lvl = extract_rank1(W)
        m, n = W.shape
        
        # Compute residual
        lvl_r1 = np.outer(u, v)
        eps = lvl.astype(np.float64) - lvl_r1
        
        print(f"\n{'='*70}")
        print(f"  {wname} ({m}×{n})")
        print(f"  Level residual: ε = lvl_true - u⊗v")
        print(f"{'='*70}")
        
        # Phase 1: Fractal Peel
        entropy, resfrac, unique_vals, counts = phase1_fractal_peel(eps, wname)
        
        # Phase 2: Holographic Scan
        flatness, row_power, col_power = phase2_holographic_scan(eps, wname)
        
        # Phase 3: Fractal Depth
        explained, svd_s = phase3_fractal_depth(eps, wname)
        
        # Phase 4: φ Resonance
        phase4_phi_resonance(eps, unique_vals, counts, wname)
        
        # Phase 5: Error-as-Signal
        row_means, col_means, eps2, s_e = phase5_error_as_signal(eps, u, v, lvl, wname)
        
        # Phase 6: Reconstruction
        phase6_reconstruction(W, W_dec, S, u, v, lvl, eps, row_means, col_means, s_e, wname)
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  PROTOCOL COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
