#!/usr/bin/env python3
"""
The Hidden Dimension — weight matrix is NOT 2D, it's a slice of 3D+.

q_proj (3584,3584) = (28 heads × 128 head_dim, 3584 input) — 3D object.
Test consistency in the TRUE basis, not the flat projection.
"""

import os, sys, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
N_HEADS, N_KV_HEADS, HEAD_DIM = 28, 4, 128
HIDDEN_DIM, INTERMEDIATE_DIM = 3584, 18944


def tet_ids(W):
    e = W.exponents.astype(np.int32) // PHI_GRID
    return e.astype(np.int16) * 2 + (W.signs > 0).astype(np.int16)

def levels(W):
    return W.exponents.astype(np.int32) // PHI_GRID


def test_per_head(W, n_heads, name=""):
    """Column consistency WITHIN each head (the 3D view)."""
    tid = tet_ids(W)
    out_f, in_f = tid.shape
    hd = out_f // n_heads
    t3d = tid.reshape(n_heads, hd, in_f)

    print(f"\n  Per-head column consistency ({name}, {n_heads} heads × {hd}):")
    head_cons = []
    for h in range(n_heads):
        fracs = np.zeros(in_f)
        for i in range(in_f):
            vals, counts = np.unique(t3d[h, :, i], return_counts=True)
            fracs[i] = np.max(counts) / hd
        head_cons.append(np.mean(fracs))

    print(f"    Mean within-head consistency: {np.mean(head_cons):.1%}")
    print(f"    Range: [{min(head_cons):.1%}, {max(head_cons):.1%}]")
    for h in [0, 7, 14, 21, 27]:
        if h < n_heads:
            print(f"      Head {h:2d}: {head_cons[h]:.1%}")
    return head_cons, t3d


def test_cross_head(t3d, name=""):
    """Do different heads agree on column modes?"""
    n_heads, hd, in_f = t3d.shape
    modes = np.zeros((n_heads, in_f), dtype=np.int16)
    for h in range(n_heads):
        for i in range(in_f):
            vals, counts = np.unique(t3d[h, :, i], return_counts=True)
            modes[h, i] = vals[np.argmax(counts)]

    print(f"\n  Cross-head agreement ({name}):")
    agrees = []
    for h1 in range(n_heads):
        for h2 in range(h1+1, n_heads):
            agrees.append(np.mean(modes[h1] == modes[h2]))
    print(f"    Mean cross-head mode agreement: {np.mean(agrees):.1%}")
    print(f"    Random baseline: {1/67:.1%}")
    return modes, agrees


def test_head_levels(W, n_heads, name=""):
    """Does each head operate at a characteristic φ-level?"""
    lvl = levels(W)
    out_f, in_f = lvl.shape
    hd = out_f // n_heads
    l3d = lvl.reshape(n_heads, hd, in_f)

    print(f"\n  Per-head level characteristics ({name}):")
    means, stds = [], []
    for h in range(n_heads):
        m = np.mean(l3d[h])
        s = np.std(l3d[h])
        means.append(m)
        stds.append(s)
        print(f"    Head {h:2d}: mean_level={m:>6.2f} std={s:.2f}")

    cross_std = np.std(means)
    within_std = np.mean(stds)
    print(f"\n    Cross-head std of means: {cross_std:.3f}")
    print(f"    Mean within-head std: {within_std:.3f}")
    print(f"    Ratio: {cross_std/within_std:.3f}")
    if cross_std > within_std * 0.1:
        print(f"    → Heads DO operate at different scales")
    return means, stds


def test_block_vs_heads(W, n_heads, name=""):
    """Do block mode transitions align with head boundaries?"""
    tid = tet_ids(W)
    out_f, in_f = tid.shape
    hd = out_f // n_heads
    bs = 32
    nr, nc = out_f // bs, in_f // bs

    bm = np.zeros((nr, nc), dtype=np.int16)
    for bi in range(nr):
        for bj in range(nc):
            blk = tid[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
            vals, counts = np.unique(blk, return_counts=True)
            bm[bi, bj] = vals[np.argmax(counts)]

    bph = hd // bs  # blocks per head
    within, boundary = 0, 0
    wt, bt = 0, 0
    for bj in range(nc):
        for bi in range(nr - 1):
            changed = bm[bi, bj] != bm[bi+1, bj]
            if (bi + 1) % bph == 0:
                boundary += changed; bt += 1
            else:
                within += changed; wt += 1

    wr = within / max(wt, 1)
    br = boundary / max(bt, 1)
    print(f"\n  Block transitions vs head boundaries ({name}):")
    print(f"    Within-head transition rate: {wr:.1%}")
    print(f"    At head boundary rate: {br:.1%}")
    print(f"    Ratio: {br/max(wr,1e-10):.2f}×")
    return wr, br


def test_3d_factored_matmul(W, x_float, n_heads, name=""):
    """Factor matmul using per-head column levels."""
    W_dec = W.decode_cached()
    out_f, in_f = W.shape
    hd = out_f // n_heads
    full = x_float @ W_dec.T

    lvl = levels(W).reshape(n_heads, hd, in_f)
    sgn = W.signs.reshape(n_heads, hd, in_f)

    # Per-head column level modes
    hcl = np.zeros((n_heads, in_f), dtype=np.int16)
    cons = np.zeros(n_heads)
    for h in range(n_heads):
        fracs = np.zeros(in_f)
        for i in range(in_f):
            vals, counts = np.unique(lvl[h, :, i], return_counts=True)
            hcl[h, i] = vals[np.argmax(counts)]
            fracs[i] = np.max(counts) / hd
        cons[h] = np.mean(fracs)

    # Factored: y[h,d] = sign[h,d,:] · (PHI^head_level[h,:] * x)
    result = np.zeros((1, out_f), dtype=np.float32)
    for h in range(n_heads):
        mag = PHI ** hcl[h].astype(np.float64)
        xs = (x_float[0] * mag).astype(np.float32)
        hr = sgn[h].astype(np.float32) @ xs
        result[0, h*hd:(h+1)*hd] = hr

    corr = np.corrcoef(full.flatten(), result.flatten())[0, 1]
    rel_err = np.linalg.norm(full - result) / np.linalg.norm(full)

    print(f"\n  3D-factored matmul ({name}):")
    print(f"    Per-head column level consistency: {np.mean(cons):.1%}")
    print(f"    3D-factored correlation: {corr:.6f}")
    print(f"    Relative error: {rel_err:.4f}")
    print(f"    (Flat 2D factored was: 0.671)")
    if corr > 0.75:
        print(f"    → IMPROVEMENT from seeing the hidden dimension!")
    return corr, cons


def test_sign_rank_per_head(W, n_heads, name=""):
    """SVD of sign matrix per head — is the binary core low-rank?"""
    sgn = W.signs.reshape(n_heads, W.shape[0]//n_heads, W.shape[1])
    print(f"\n  Sign matrix rank per head ({name}):")
    for h in [0, 7, 14, 21, 27]:
        if h >= n_heads: continue
        s = sgn[h].astype(np.float32)
        _, sigma, _ = np.linalg.svd(s, full_matrices=False)
        energy = np.cumsum(sigma**2) / np.sum(sigma**2)
        r90 = np.searchsorted(energy, 0.90) + 1
        r95 = np.searchsorted(energy, 0.95) + 1
        r99 = np.searchsorted(energy, 0.99) + 1
        print(f"    Head {h:2d}: rank90={r90}, rank95={r95}, rank99={r99}/{len(sigma)}")


def run():
    print("=" * 70)
    print("  THE HIDDEN DIMENSION")
    print("  Weight matrix is a 3D object projected to 2D")
    print("=" * 70)

    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)

    # q_proj: (3584, 3584) = (28 heads × 128, 3584)
    W = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
    print(f"\n{'='*70}\n  q_proj: (28×128, 3584) — the 3D object\n{'='*70}")

    hc, t3d = test_per_head(W, N_HEADS, "q_proj")
    test_cross_head(t3d, "q_proj")
    test_head_levels(W, N_HEADS, "q_proj")
    test_block_vs_heads(W, N_HEADS, "q_proj")
    test_sign_rank_per_head(W, N_HEADS, "q_proj")

    x = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
    test_3d_factored_matmul(W, x, N_HEADS, "q_proj")
    W.clear_cache()

    # o_proj: (3584, 3584) = (3584, 28×128) — transposed head structure
    W = PhiEncoded.load(os.path.join(layer_dir, 'o_proj.npz'))
    print(f"\n{'='*70}\n  o_proj: (3584, 28×128) — heads in INPUT dim\n{'='*70}")
    # For o_proj, heads are in the INPUT dimension
    tid = tet_ids(W)
    tid_t = tid.T  # (3584, 3584) → transpose so heads are in "output"
    # Reshape: (28, 128, 3584)
    t3d_o = tid_t.reshape(N_HEADS, HEAD_DIM, W.shape[0])
    print(f"\n  Per-head consistency (o_proj, heads in cols):")
    for h in [0, 14, 27]:
        fracs = []
        for i in range(W.shape[0]):
            vals, counts = np.unique(t3d_o[h, :, i], return_counts=True)
            fracs.append(np.max(counts) / HEAD_DIM)
        print(f"    Head {h:2d}: mean consistency={np.mean(fracs):.1%}")
    W.clear_cache()

    # k_proj: (512, 3584) = (4 kv_heads × 128, 3584)
    W = PhiEncoded.load(os.path.join(layer_dir, 'k_proj.npz'))
    print(f"\n{'='*70}\n  k_proj: (4×128, 3584) — KV heads\n{'='*70}")
    hc_k, t3d_k = test_per_head(W, N_KV_HEADS, "k_proj")
    x_k = np.random.randn(1, W.shape[1]).astype(np.float32) * 0.02
    test_3d_factored_matmul(W, x_k, N_KV_HEADS, "k_proj")
    W.clear_cache()

    print(f"\n{'='*70}\n  CONCLUSIONS\n{'='*70}")


if __name__ == '__main__':
    run()
