#!/usr/bin/env python3
"""
GOP/MGOP/EDP Protocol v2 — INTEGER Residual Analysis

v1 BUG: computed ε = lvl_true - u⊗v (continuous), giving 6M meaningless fractional values.
v2 FIX: compute ε_int = lvl_true - round(u⊗v) — the ACTUAL integer correction alphabet.

ALSO: gate_proj showed corr(ε_row, u) = -0.989! The residual is a function of the
rank-1 vectors we already have. This must be exploitable.
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


def analyze_integer_residual(wname, W, W_dec, S, u, v, sigma, lvl):
    m, n = W.shape
    
    # The CORRECT integer residual
    lvl_r1 = np.outer(u, v)
    lvl_r1_int = np.round(lvl_r1).astype(np.int32)
    lvl_true = lvl.astype(np.int32)
    eps_int = lvl_true - lvl_r1_int  # INTEGER residual
    
    print(f"\n{'='*70}")
    print(f"  {wname} ({m}×{n}) — INTEGER RESIDUAL ANALYSIS")
    print(f"{'='*70}")
    
    # ================================================================
    # PHASE 1: FRACTAL PEEL on integer residual
    # ================================================================
    print(f"\n  PHASE 1: FRACTAL PEEL (integer ε)")
    print(f"  {'─'*50}")
    
    flat = eps_int.flatten()
    unique_vals, counts = np.unique(flat, return_counts=True)
    n_unique = len(unique_vals)
    
    print(f"    Shape: {m}×{n} = {flat.size:,}")
    print(f"    Range: [{eps_int.min()}, {eps_int.max()}]")
    print(f"    Mean: {eps_int.mean():.4f}, Std: {eps_int.std():.4f}")
    print(f"    UNIQUE INTEGER VALUES: {n_unique}")
    
    # Top values
    sorted_idx = np.argsort(-counts)
    cum = 0
    print(f"    Top-15 values:")
    for i in range(min(15, n_unique)):
        idx = sorted_idx[i]
        cum += counts[idx]
        print(f"      ε={unique_vals[idx]:>4d}: {counts[idx]:>8,} ({counts[idx]/flat.size:.1%})  cum={cum/flat.size:.1%}")
    
    # Entropy of integer alphabet
    probs = counts / flat.size
    entropy = -np.sum(probs * np.log2(probs + 1e-30))
    print(f"    Entropy: {entropy:.3f} bits (ceiling={np.log2(n_unique):.1f})")
    
    # |ε| distribution
    abs_eps = np.abs(flat)
    print(f"\n    |ε_int| distribution:")
    for t in [0, 1, 2, 3, 4, 5, 10]:
        print(f"      |ε| ≤ {t}: {np.mean(abs_eps <= t):.1%}")
    
    # Autocorrelation on integer residual
    print(f"\n    Row autocorrelation:")
    for lag in [1, 2, 3, 5, 10]:
        ac = np.mean([np.corrcoef(eps_int[j, :-lag].astype(float), 
                                   eps_int[j, lag:].astype(float))[0,1] 
                       for j in range(min(200, m))])
        print(f"      lag={lag}: {ac:.4f}")
    
    # Resfrac
    pred = np.zeros_like(eps_int, dtype=np.float64)
    pred[:, 1:] += eps_int[:, :-1]; pred[:, :-1] += eps_int[:, 1:]
    pred[1:, :] += eps_int[:-1, :]; pred[:-1, :] += eps_int[1:, :]
    cnt = np.full_like(eps_int, 4, dtype=np.float64)
    cnt[0, :] -= 1; cnt[-1, :] -= 1; cnt[:, 0] -= 1; cnt[:, -1] -= 1
    pred /= cnt
    resfrac = np.abs(eps_int - pred).mean() / (eps_int.std() + 1e-30)
    print(f"    Resfrac: ρ={resfrac:.4f}")
    
    # ================================================================
    # PHASE 2: ROW/COL STRUCTURE of integer residual
    # ================================================================
    print(f"\n  PHASE 2: ROW/COL STRUCTURE")
    print(f"  {'─'*50}")
    
    eps_f = eps_int.astype(np.float64)
    row_means = eps_f.mean(axis=1)
    col_means = eps_f.mean(axis=0)
    total_var = np.var(eps_f)
    
    print(f"    Row means: range=[{row_means.min():.3f}, {row_means.max():.3f}], var={np.var(row_means):.6f}")
    print(f"    Col means: range=[{col_means.min():.3f}, {col_means.max():.3f}], var={np.var(col_means):.6f}")
    print(f"    Row var / total: {np.var(row_means)/total_var:.6f}")
    print(f"    Col var / total: {np.var(col_means)/total_var:.6f}")
    
    # Correlation with u, v
    corr_row_u = np.corrcoef(row_means, u)[0, 1]
    corr_col_v = np.corrcoef(col_means, v)[0, 1]
    print(f"    corr(ε_row_mean, u) = {corr_row_u:.4f}")
    print(f"    corr(ε_col_mean, v) = {corr_col_v:.4f}")
    
    # What about row VARIANCE (not mean)?
    row_stds = eps_f.std(axis=1)
    col_stds = eps_f.std(axis=0)
    corr_rowstd_u = np.corrcoef(row_stds, np.abs(u))[0, 1]
    corr_colstd_v = np.corrcoef(col_stds, np.abs(v))[0, 1]
    print(f"    corr(ε_row_std, |u|) = {corr_rowstd_u:.4f}")
    print(f"    corr(ε_col_std, |v|) = {corr_colstd_v:.4f}")
    
    # What about the FRACTIONAL part of u⊗v that we rounded away?
    frac_part = lvl_r1 - lvl_r1_int.astype(np.float64)  # what round() discarded
    corr_frac_eps = np.corrcoef(frac_part.flatten()[:100000], 
                                 eps_f.flatten()[:100000])[0, 1]
    print(f"    corr(frac(u⊗v), ε_int) = {corr_frac_eps:.4f}")
    
    # ================================================================
    # PHASE 3: SVD of integer residual
    # ================================================================
    print(f"\n  PHASE 3: ε_int SVD")
    print(f"  {'─'*50}")
    
    U_e, s_e, Vt_e = np.linalg.svd(eps_f, full_matrices=False)
    cum_e = np.cumsum(s_e**2) / np.sum(s_e**2)
    
    print(f"    σ₁/σ₂ = {s_e[0]/s_e[1]:.3f}")
    print(f"    Top-10 σ: {s_e[:10].round(1)}")
    for k in [1, 2, 5, 10, 20, 50]:
        print(f"    rank-{k:>3d}: explains {cum_e[k-1]:.4f}")
    
    # Correlation of SVD components with u, v
    print(f"\n    SVD component correlations with u, v:")
    for k in range(min(5, len(s_e))):
        cu = np.corrcoef(U_e[:, k], u)[0, 1]
        cv = np.corrcoef(Vt_e[k, :], v)[0, 1]
        print(f"      Component {k}: corr(u_k, u)={cu:.4f}, corr(v_k, v)={cv:.4f}, σ={s_e[k]:.1f}")
    
    # ================================================================
    # PHASE 4: EDP — φ-patterns in the integer alphabet
    # ================================================================
    print(f"\n  PHASE 4: φ-PATTERN SEARCH (EDP)")
    print(f"  {'─'*50}")
    
    LOG_PHI = np.log(PHI)
    
    # The unique integer values — are they φ-structured?
    print(f"    Unique ε values and their φ-levels:")
    for i in range(min(20, n_unique)):
        idx = sorted_idx[i]
        val = unique_vals[idx]
        freq = counts[idx]
        if val != 0:
            phi_lvl = np.log(abs(val)) / LOG_PHI
            nearest_int_phi = round(phi_lvl)
            phi_err = abs(phi_lvl - nearest_int_phi)
            print(f"      ε={val:>4d}: freq={freq:>8,} ({freq/flat.size:.1%}), "
                  f"log_φ|ε|={phi_lvl:>+.3f}, nearest_int={nearest_int_phi}, err={phi_err:.3f}")
        else:
            print(f"      ε={val:>4d}: freq={freq:>8,} ({freq/flat.size:.1%})")
    
    # Distribution shape: is it Laplacian? Gaussian?
    from scipy import stats
    _, norm_p = stats.normaltest(flat[:100000].astype(float))
    print(f"\n    Normality test (D'Agostino): p={norm_p:.2e}")
    kurtosis = stats.kurtosis(flat.astype(float))
    skewness = stats.skew(flat.astype(float))
    print(f"    Kurtosis: {kurtosis:.3f} (Gaussian=0, Laplace=3)")
    print(f"    Skewness: {skewness:.4f}")
    
    # ================================================================
    # PHASE 5: RECONSTRUCTION with integer corrections
    # ================================================================
    print(f"\n  PHASE 5: RECONSTRUCTION")
    print(f"  {'─'*50}")
    
    # Baselines
    c_sign = matmul_corr(S, W_dec)
    W_r1 = S * (PHI ** np.outer(u, v)).astype(np.float32)
    c_r1 = matmul_corr(W_r1, W_dec)
    W_r1_int = (S.astype(np.float64) * (PHI ** lvl_r1_int.astype(np.float64))).astype(np.float32)
    c_r1_int = matmul_corr(W_r1_int, W_dec)
    
    print(f"    Sign-only:           corr={c_sign:.4f}")
    print(f"    Sign + rank-1:       corr={c_r1:.4f}")
    print(f"    Sign + round(rank-1): corr={c_r1_int:.4f}")
    
    # Low-rank ε_int correction + round
    print(f"\n    Low-rank ε_int SVD correction:")
    for K in [1, 2, 3, 5, 10, 20, 50, 100]:
        if K > min(m, n): break
        eps_K = (U_e[:, :K] * s_e[:K]) @ Vt_e[:K, :]
        lvl_corrected = lvl_r1_int.astype(np.float64) + np.round(eps_K)
        W_corr = (S.astype(np.float64) * (PHI ** lvl_corrected)).astype(np.float32)
        c = matmul_corr(W_corr, W_dec)
        exact = np.mean(lvl_corrected == lvl_true.astype(np.float64))
        print(f"      + ε_int rank-{K:>3d}: corr={c:.4f}  (exact={exact:.1%})")
    
    # What if we use u² and v² as predictors for ε?
    # ε_row_mean ≈ α·u → ε ≈ α·u⊗1 + β·1⊗v?
    if abs(corr_row_u) > 0.5 or abs(corr_col_v) > 0.5:
        print(f"\n    Exploiting u/v correlation (corr_u={corr_row_u:.3f}, corr_v={corr_col_v:.3f}):")
        
        # Linear regression: row_means ≈ α·u + β
        from numpy.polynomial import polynomial as P
        alpha_u = np.polyfit(u, row_means, 1)
        pred_row = np.polyval(alpha_u, u)
        
        alpha_v = np.polyfit(v, col_means, 1)
        pred_col = np.polyval(alpha_v, v)
        
        # Correction: ε ≈ pred_row + pred_col
        eps_pred = pred_row[:, None] + pred_col[None, :] - np.mean(row_means)
        lvl_corrected = lvl_r1 + eps_pred
        lvl_corrected_int = np.round(lvl_corrected).astype(np.int32)
        
        W_uv = (S.astype(np.float64) * (PHI ** lvl_corrected_int.astype(np.float64))).astype(np.float32)
        c_uv = matmul_corr(W_uv, W_dec)
        exact_uv = np.mean(lvl_corrected_int == lvl_true)
        print(f"      Linear(u,v) pred: corr={c_uv:.4f}  (exact={exact_uv:.1%})")
        
        # Quadratic: row_means ≈ α·u² + β·u + γ
        alpha_u2 = np.polyfit(u, row_means, 2)
        pred_row2 = np.polyval(alpha_u2, u)
        alpha_v2 = np.polyfit(v, col_means, 2)
        pred_col2 = np.polyval(alpha_v2, v)
        
        eps_pred2 = pred_row2[:, None] + pred_col2[None, :] - np.mean(row_means)
        lvl_corrected2 = lvl_r1 + eps_pred2
        lvl_int2 = np.round(lvl_corrected2).astype(np.int32)
        
        W_uv2 = (S.astype(np.float64) * (PHI ** lvl_int2.astype(np.float64))).astype(np.float32)
        c_uv2 = matmul_corr(W_uv2, W_dec)
        exact_uv2 = np.mean(lvl_int2 == lvl_true)
        print(f"      Quad(u,v) pred:   corr={c_uv2:.4f}  (exact={exact_uv2:.1%})")
    
    # Rank-2 of the FULL level matrix (not ε)
    print(f"\n    Full level SVD (rank-K):")
    U_l, s_l, Vt_l = np.linalg.svd(lvl.astype(np.float64), full_matrices=False)
    for K in [1, 2, 3, 5, 10]:
        lvl_K = (U_l[:, :K] * s_l[:K]) @ Vt_l[:K, :]
        lvl_K_int = np.round(lvl_K).astype(np.int32)
        W_K = (S.astype(np.float64) * (PHI ** lvl_K_int.astype(np.float64))).astype(np.float32)
        c_K = matmul_corr(W_K, W_dec)
        exact_K = np.mean(lvl_K_int == lvl_true)
        print(f"      Level rank-{K} + int: corr={c_K:.4f}  (exact={exact_K:.1%})")
    
    # Oracle
    W_exact = (S.astype(np.float64) * (PHI ** lvl_true.astype(np.float64))).astype(np.float32)
    c_exact = matmul_corr(W_exact, W_dec)
    print(f"\n    Oracle (exact int levels): corr={c_exact:.4f}")
    
    return eps_int, unique_vals, counts, entropy


def run():
    print("=" * 70)
    print("  GOP/EDP PROTOCOL v2: INTEGER RESIDUAL")
    print("  ε_int = lvl_true - round(u⊗v)")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    for wname in ['q_proj', 'gate_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        S = W.signs.astype(np.float32)
        u, v, sigma, lvl = extract_rank1(W)
        
        analyze_integer_residual(wname, W, W_dec, S, u, v, sigma, lvl)
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  PROTOCOL v2 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
