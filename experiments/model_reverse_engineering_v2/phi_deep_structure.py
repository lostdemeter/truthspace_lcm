#!/usr/bin/env python3
"""
Deep Structure: How sign + rank-1 level work together.
W[j,i] = sign[j,i] × φ^(u[j]×v[i]) × φ^(residual[j,i])
"""

import os, sys, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
N_HEADS, HEAD_DIM = 28, 128


def levels(W):
    return W.exponents.astype(np.int32) // PHI_GRID

def extract_rank1(W):
    lvl = levels(W).astype(np.float32)
    U, sigma, Vt = np.linalg.svd(lvl, full_matrices=False)
    u = U[:, 0] * sigma[0]
    v = Vt[0, :]
    return u, v, sigma, lvl


def analyze_residual(W, name=""):
    u, v, sigma, lvl = extract_rank1(W)
    out_f, in_f = lvl.shape
    residual = lvl - np.outer(u, v)

    print(f"\n  LEVEL RESIDUAL ({name}):")
    print(f"    Std: {np.std(residual):.4f} (level std: {np.std(lvl):.4f})")
    print(f"    Ratio: {np.std(residual)/np.std(lvl):.3f}")

    U_r, s_r, Vt_r = np.linalg.svd(residual, full_matrices=False)
    e = np.cumsum(s_r**2) / np.sum(s_r**2)
    print(f"    Residual σ₁/σ₂: {s_r[0]/s_r[1]:.2f}")
    print(f"    rank50={np.searchsorted(e,0.5)+1}, rank90={np.searchsorted(e,0.9)+1}")

    for thresh in [0.5, 1.0, 2.0, 5.0]:
        print(f"    |res| > {thresh}: {np.mean(np.abs(residual)>thresh):.1%}")

    # Row/col bias
    rm = np.mean(residual, axis=1)
    cm = np.mean(residual, axis=0)
    debiased = residual - rm[:, None] - cm[None, :] + np.mean(residual)
    reduction = 1 - np.std(debiased) / np.std(residual)
    print(f"    Row+Col bias explains: {reduction:.1%}")

    # Correction factor
    correction = PHI ** residual.astype(np.float64)
    print(f"    φ^residual: mean={np.mean(correction):.4f}, std={np.std(correction):.4f}")

    return residual


def analyze_sign(W, name=""):
    sgn = W.signs.astype(np.float32)
    out_f, in_f = sgn.shape

    print(f"\n  SIGN MATRIX ({name}):")
    print(f"    Balance: {np.mean(sgn>0):.4%} positive")
    print(f"    Row balance std: {np.std(np.mean(sgn>0, axis=1)):.4f}")
    print(f"    Col balance std: {np.std(np.mean(sgn>0, axis=0)):.4f}")

    # Row-row agreement
    sample = np.random.choice(out_f, min(100, out_f), replace=False)
    agrees = []
    for i in range(len(sample)):
        for j in range(i+1, min(len(sample), i+3)):
            agrees.append(np.mean(sgn[sample[i]] == sgn[sample[j]]))
    print(f"    Row-row agreement: {np.mean(agrees):.4f} (random=0.5)")

    if out_f == 3584:
        sgn3d = sgn.reshape(N_HEADS, HEAD_DIM, in_f)
        head_means = np.array([np.mean(sgn3d[h], axis=0) for h in range(N_HEADS)])
        hc = np.corrcoef(head_means)
        off = hc[np.triu_indices(N_HEADS, k=1)]
        print(f"    Head fingerprint corr: mean={np.mean(off):.4f}, "
              f"range=[{np.min(off):.4f}, {np.max(off):.4f}]")


def test_4_components(W, x_float, name=""):
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    full = x_float @ W_dec.T
    u, v, sigma, lvl = extract_rank1(W)
    sgn = W.signs.astype(np.float32)
    U_l, s_l, Vt_l = np.linalg.svd(lvl, full_matrices=False)

    def corr(a, b):
        return np.corrcoef(a.flatten(), b.flatten())[0, 1]

    # Stage 1: sign only
    r_sign = x_float @ sgn.T
    # Stage 2: sign × col_scale
    r_sv = (x_float * PHI**v.astype(np.float64)).astype(np.float32) @ sgn.T
    # Stage 3: sign × rank-1 level
    W_r1 = sgn.astype(np.float64) * PHI**np.outer(u, v).astype(np.float64)
    r_r1 = (x_float.astype(np.float64) @ W_r1.T).astype(np.float32)
    # Stage 3b: rank-2
    lvl2 = (U_l[:,:2]*s_l[:2]) @ Vt_l[:2,:]
    W_r2 = sgn.astype(np.float64) * PHI**lvl2.astype(np.float64)
    r_r2 = (x_float.astype(np.float64) @ W_r2.T).astype(np.float32)
    # Stage 3c: rank-5
    lvl5 = (U_l[:,:5]*s_l[:5]) @ Vt_l[:5,:]
    W_r5 = sgn.astype(np.float64) * PHI**lvl5.astype(np.float64)
    r_r5 = (x_float.astype(np.float64) @ W_r5.T).astype(np.float32)
    # Stage 4: exact tetromino
    W_tet = sgn.astype(np.float64) * PHI**lvl.astype(np.float64)
    r_tet = (x_float.astype(np.float64) @ W_tet.T).astype(np.float32)

    cs = corr(full, r_sign)
    csv = corr(full, r_sv)
    cr1 = corr(full, r_r1)
    cr2 = corr(full, r_r2)
    cr5 = corr(full, r_r5)
    ct = corr(full, r_tet)

    print(f"\n  4-COMPONENT DECOMPOSITION ({name}):")
    print(f"    sign only:           {cs:.6f}")
    print(f"    + col_scale (v):     {csv:.6f}  (+{csv-cs:.4f})")
    print(f"    + row_scale (rank1): {cr1:.6f}  (+{cr1-csv:.4f})")
    print(f"    + rank-2 level:      {cr2:.6f}  (+{cr2-cr1:.4f})")
    print(f"    + rank-5 level:      {cr5:.6f}  (+{cr5-cr2:.4f})")
    print(f"    + exact level (tet): {ct:.6f}  (+{ct-cr5:.4f})")
    print(f"    exact float:         1.000000  (+{1-ct:.4f})")

    # Storage
    b = lambda x: f"{x/8/1024:.0f}KB"
    print(f"\n    Storage: sign={b(out_f*in_f)}, +v={b(in_f*32)}, "
          f"+u={b(out_f*32)}, full_tet={b(out_f*in_f*16+out_f*in_f)}")

    return {'sign': cs, 'r1': cr1, 'r2': cr2, 'r5': cr5, 'tet': ct}


def sign_vs_full(W, name=""):
    """Multi-sample sign correlation per head."""
    W_dec = W.decode_cached()
    sgn = W.signs.astype(np.float32)
    out_f, in_f = W.shape

    X = np.random.randn(200, in_f).astype(np.float32) * 0.02
    Y_full = X @ W_dec.T
    Y_sign = X @ sgn.T

    print(f"\n  SIGN-ONLY COMPUTATION ({name}, 200 samples):")
    per_dim = np.array([np.corrcoef(Y_full[:,j], Y_sign[:,j])[0,1]
                        for j in range(out_f)])
    print(f"    Per-dim corr: mean={np.nanmean(per_dim):.4f}, "
          f"median={np.nanmedian(per_dim):.4f}")

    if out_f == 3584:
        print(f"    Per-head:")
        for h in range(N_HEADS):
            s = h * HEAD_DIM
            hc = np.nanmean(per_dim[s:s+HEAD_DIM])
            bar = "█" * int(abs(hc) * 50)
            print(f"      Head {h:2d}: {hc:>6.3f} {bar}")


def analyze_interaction(W, name=""):
    u, v, sigma, lvl = extract_rank1(W)
    residual = lvl - np.outer(u, v)
    sgn = W.signs.astype(np.float32)
    W_dec = W.decode_cached()

    print(f"\n  SIGN × RESIDUAL INTERACTION ({name}):")
    c = np.corrcoef(sgn.flatten(), residual.flatten())[0, 1]
    print(f"    corr(sign, residual): {c:.6f}")

    pos = sgn > 0
    print(f"    |res| where sign=+1: {np.mean(np.abs(residual[pos])):.4f}")
    print(f"    |res| where sign=-1: {np.mean(np.abs(residual[~pos])):.4f}")

    # Error by magnitude quintile
    mag_r1 = PHI ** np.outer(u, v).astype(np.float64)
    W_r1 = sgn.astype(np.float64) * mag_r1
    error = np.abs(W_dec - W_r1).flatten()
    w_abs = np.abs(W_dec).flatten()
    idx = np.argsort(w_abs)
    n = len(idx)
    print(f"    Error by |W| quintile:")
    for q in range(5):
        s, e = q*n//5, (q+1)*n//5 if q<4 else n
        qi = idx[s:e]
        print(f"      Q{q+1} (|W|={np.mean(w_abs[qi]):.5f}): "
              f"|err|={np.mean(error[qi]):.6f}")


def run():
    print("=" * 70)
    print("  DEEP STRUCTURE: How sign + magnitude work together")
    print("=" * 70)

    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)

    for wname in ['q_proj', 'gate_proj', 'down_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        print(f"\n{'='*70}\n  {wname} ({W.shape[0]}×{W.shape[1]})\n{'='*70}")

        analyze_residual(W, wname)
        analyze_sign(W, wname)
        analyze_interaction(W, wname)

        x = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
        test_4_components(W, x, wname)
        sign_vs_full(W, wname)
        W.clear_cache()

    print(f"\n{'='*70}\n  SYNTHESIS\n{'='*70}")


if __name__ == '__main__':
    run()
