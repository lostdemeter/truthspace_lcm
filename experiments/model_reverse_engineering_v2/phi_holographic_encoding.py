#!/usr/bin/env python3
"""
Holographic Encoding — How the weight matrix stores knowledge.
Sign = holographic plate. Input x = reference beam. sign@x = reconstruction.
"""

import os, sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'temp', 'outside_projects', 'holographersworkbench'))

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID
from workbench.processors.holographic import phase_retrieve_hilbert, holographic_refinement

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')

def levels(W): return W.exponents.astype(np.int32) // PHI_GRID

def extract_rank1(W):
    lvl = levels(W).astype(np.float32)
    U, s, Vt = np.linalg.svd(lvl, full_matrices=False)
    return U[:, 0] * s[0], Vt[0, :], s, lvl


def stereo_decomposition(W, x, name=""):
    """y = pos_sum - neg_sum = stereo disparity."""
    W_dec = W.decode_cached()
    sgn = W.signs.astype(np.float32)
    mag = np.abs(W_dec)
    xv = x[0]
    y = W_dec @ xv

    pos_sum = np.where(sgn > 0, mag, 0) @ xv
    neg_sum = np.where(sgn < 0, mag, 0) @ xv
    baseline = (pos_sum + neg_sum) / 2
    disparity = pos_sum - neg_sum

    print(f"\n  STEREO DECOMPOSITION ({name}):")
    print(f"    y = pos - neg: corr={np.corrcoef(y, disparity)[0,1]:.6f}")
    print(f"    |pos| mean={np.mean(np.abs(pos_sum)):.6f}")
    print(f"    |neg| mean={np.mean(np.abs(neg_sum)):.6f}")
    print(f"    disp/base ratio={np.mean(np.abs(disparity))/np.mean(np.abs(baseline)):.4f}")
    print(f"    corr(pos, neg)={np.corrcoef(pos_sum, neg_sum)[0,1]:.4f}")
    print(f"    corr(baseline, y)={np.corrcoef(baseline, y)[0,1]:.4f}")
    print(f"    α=0.5 is EXACT: I_L=base-0.5*disp=neg, I_R=base+0.5*disp=pos")


def phase_retrieval_rows(W, name=""):
    """Hilbert transform on weight rows: envelope + phase."""
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    sgn = W.signs.astype(np.float32)

    print(f"\n  PHASE RETRIEVAL ({name}):")
    for r in [0, 500, 1000, 2000]:
        if r >= out_f: break
        env, pv = phase_retrieve_hilbert(W_dec[r])
        phase = np.angle(np.fft.ifft(np.fft.fft(W_dec[r]) * 
            np.concatenate([[1], 2*np.ones(in_f//2-1), [1], np.zeros(in_f//2-1)])))
        pred_sign = np.sign(np.cos(phase))
        agree = np.mean(pred_sign == sgn[r])
        print(f"    Row {r:>5d}: phase_var={pv:.4f}, sign_from_phase={agree:.1%}")


def column_correlation(W, name=""):
    """Column correlation structure = angular correlation of hologram."""
    sgn = W.signs.astype(np.float32)
    out_f, in_f = sgn.shape

    print(f"\n  COLUMN CORRELATION ({name}):")
    n_s = min(200, in_f)
    cols = np.sort(np.random.choice(in_f, n_s, replace=False))
    S = sgn[:, cols]
    C = (S.T @ S) / out_f

    _, sc, _ = np.linalg.svd(C)
    e = np.cumsum(sc**2) / np.sum(sc**2)
    print(f"    Top-5 σ: {sc[:5].round(3)}")
    print(f"    σ₁/σ₂={sc[0]/sc[1]:.2f}")
    print(f"    rank50={np.searchsorted(e,.5)+1}, rank90={np.searchsorted(e,.9)+1}/{n_s}")

    np.fill_diagonal(C, 0)
    print(f"    Off-diag |corr|: mean={np.mean(np.abs(C)):.4f}, max={np.max(np.abs(C)):.4f}")

    print(f"\n    Correlation vs column distance:")
    for di in [1, 2, 5, 10, 50, 100]:
        if di >= n_s: break
        pairs = [(i, i+di) for i in range(n_s - di)]
        mc = np.mean([np.abs(C[i, j]) for i, j in pairs])
        print(f"      Δ={di:>4d}: mean|corr|={mc:.4f}")


def sign_svd_matmul(W, name=""):
    """How many sign SVD components needed for matmul accuracy?"""
    sgn = W.signs.astype(np.float32)
    out_f, in_f = sgn.shape
    W_dec = W.decode_cached()

    U_s, s_s, Vt_s = np.linalg.svd(sgn, full_matrices=False)
    print(f"\n  SIGN SVD ({name}):")
    print(f"    Top-10 σ: {s_s[:10].round(2)}")
    print(f"    σ₁/σ₂={s_s[0]/s_s[1]:.3f}")

    np.random.seed(42)
    X = np.random.randn(50, in_f).astype(np.float32) * 0.02
    Y_full = X @ W_dec.T

    for K in [1, 5, 10, 25, 50, 100, 200, 500]:
        if K > min(out_f, in_f): break
        sign_K = (U_s[:, :K] * s_s[:K]) @ Vt_s[:K, :]
        Y_K = X @ sign_K.T
        c = np.corrcoef(Y_full.flatten(), Y_K.flatten())[0, 1]
        print(f"    K={K:>4d}: corr={c:.4f}")


def holographic_matmul(W, x, name=""):
    """Apply holographic_refinement to improve sign-only matmul."""
    W_dec = W.decode_cached()
    sgn = W.signs.astype(np.float32)
    u, v, _, _ = extract_rank1(W)
    xv = x[0]

    y_full = W_dec @ xv
    y_sign = sgn @ xv
    mag_r1 = PHI ** np.outer(u, v).astype(np.float64)
    y_r1 = (sgn.astype(np.float64) * mag_r1 @ xv.astype(np.float64)).astype(np.float32)

    print(f"\n  HOLOGRAPHIC MATMUL ({name}):")
    print(f"    Sign-only: {np.corrcoef(y_full, y_sign)[0,1]:.4f}")
    print(f"    Rank-1:    {np.corrcoef(y_full, y_r1)[0,1]:.4f}")

    for rn, ref in [("abs(y_r1)", np.abs(y_r1)), ("uniform", np.ones(len(y_sign)))]:
        try:
            yr = holographic_refinement(y_sign, ref, method="hilbert", blend_ratio=0.6)
            print(f"    Holo-refined (ref={rn}): {np.corrcoef(y_full, yr)[0,1]:.4f}")
        except: pass


def run():
    print("=" * 70)
    print("  HOLOGRAPHIC ENCODING ANALYSIS")
    print("=" * 70)
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)

    for wn in ['q_proj', 'gate_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wn}.npz'))
        print(f"\n{'='*70}\n  {wn} ({W.shape[0]}×{W.shape[1]})\n{'='*70}")
        x = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
        stereo_decomposition(W, x, wn)
        phase_retrieval_rows(W, wn)
        column_correlation(W, wn)
        sign_svd_matmul(W, wn)
        holographic_matmul(W, x, wn)
        W.clear_cache()

    print(f"\n{'='*70}\n  SYNTHESIS\n{'='*70}")

if __name__ == '__main__':
    run()
