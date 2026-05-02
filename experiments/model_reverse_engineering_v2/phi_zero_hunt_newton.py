#!/usr/bin/env python3
"""
Zero-Hunting with Newton Refinement

The first-order zero-hunt (phi_zero_hunt_gate.py) achieved 50% hit rate.
The miss comes from float32 accumulation differences between:
  - Analytical: h_j(δ) = h_j(0) + c_j * (φ^δ - 1)
  - Verification: x @ gate_W_shifted.T  (different summation order)

Fix: float64 + Newton iteration on actual gate output.

rhzeros pipeline:
  1. Lambert W estimate → analytical δ₀ (already have this)
  2. Newton on actual ζ(1/2+it) → Newton on actual h_j(δ)
  3. Snap to machine precision → exact gate flip

Also tests: bisection fallback for stubborn zeros.
"""

import os, sys, time, gc
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from phi_geometric.inference.phi_types import PhiEncoded, PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phi_model')
LN_PHI = np.log(PHI)

def levels(W): return W.exponents.astype(np.int32) // PHI_GRID

def extract_rank1(W):
    lvl = levels(W).astype(np.float32)
    U, s, Vt = np.linalg.svd(lvl, full_matrices=False)
    return U[:, 0] * s[0], Vt[0, :], lvl

def rms_norm(x, weight, eps=1e-6):
    return (x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)) * weight

def load_gate_only(layer_idx):
    """Load only gate_proj weights + norms (memory efficient)."""
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    W = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz'))
    W_dec = W.decode_cached()
    u, v, lvl = extract_rank1(W)
    lvl_r1 = np.round(np.outer(u, v)).astype(np.int32)
    eps_int = lvl.astype(np.int32) - lvl_r1
    W.clear_cache(); del W
    norms = np.load(os.path.join(layer_dir, 'norms.npz'))
    norm_w = norms['post_attention_layernorm'].astype(np.float32)
    return W_dec, eps_int, norm_w


def get_top_eps(eps_int):
    unique, counts = np.unique(eps_int, return_counts=True)
    return int(unique[np.argmax(counts)])


def compute_gate_at_delta(h0_64, c_64, delta):
    """Analytical gate pre-activation at shift δ. Exact per-row in float64."""
    return h0_64 + c_64 * (PHI ** delta - 1.0)


def newton_refine(h0_j, c_j, delta_est, max_iter=10, tol=1e-14):
    """Newton iteration to find exact δ where h_j(δ) = 0.

    f(δ) = h0_j + c_j * (φ^δ - 1) = 0
    f'(δ) = c_j * φ^δ * ln(φ)

    Returns (refined_δ, n_iterations, final_residual).
    """
    d = delta_est
    for i in range(max_iter):
        phi_d = PHI ** d
        f = h0_j + c_j * (phi_d - 1.0)
        if abs(f) < tol:
            return d, i + 1, abs(f)
        fp = c_j * phi_d * LN_PHI
        if abs(fp) < 1e-30:
            break
        d = d - f / fp
    return d, max_iter, abs(h0_j + c_j * (PHI ** d - 1.0))


def bisect_refine(h0_j, c_j, delta_est, max_iter=60, tol=1e-14):
    """Bisection to find exact δ where h_j(δ) = 0.

    Brackets the zero between δ=0 and 2*δ_est, then bisects.
    """
    f0 = h0_j  # f(0) = h0_j
    f_est = h0_j + c_j * (PHI ** delta_est - 1.0)

    # Find bracket: f(lo) and f(hi) should have opposite signs
    lo, hi = 0.0, delta_est
    if f0 * f_est > 0:
        # Same sign — expand bracket
        hi = 2.0 * delta_est
        f_hi = h0_j + c_j * (PHI ** hi - 1.0)
        if f0 * f_hi > 0:
            # Still same sign — try other direction
            lo, hi = delta_est, 0.0
            if h0_j * (h0_j + c_j * (PHI ** lo - 1.0)) > 0:
                return delta_est, 0, abs(f_est)  # Can't bracket

    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = h0_j + c_j * (PHI ** mid - 1.0)
        if abs(f_mid) < tol:
            return mid, i + 1, abs(f_mid)
        f_lo = h0_j + c_j * (PHI ** lo - 1.0)
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid

    mid = (lo + hi) / 2.0
    return mid, max_iter, abs(h0_j + c_j * (PHI ** mid - 1.0))


def run():
    print("=" * 70)
    print("  ZERO-HUNTING WITH NEWTON REFINEMENT")
    print("  (rhzeros Stage 2: Ramanujan → Newton snap)")
    print("=" * 70)
    sys.stdout.flush()

    HD = 3584
    N_INPUTS = 5
    N_TEST = 100  # Test top 100 zeros

    np.random.seed(42)
    inputs = [np.random.randn(1, HD).astype(np.float32) * 0.1 for _ in range(N_INPUTS)]

    for layer_idx in [0, 3]:
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")
        sys.stdout.flush()

        gate_W, gate_eps, norm_w = load_gate_only(layer_idx)
        top_eps = get_top_eps(gate_eps)
        mask_eps = (gate_eps == top_eps)

        print(f"  Loaded in {time.time()-t0:.1f}s, top_ε={top_eps}")
        print(f"  Gate shape: {gate_W.shape}")
        sys.stdout.flush()

        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Compute h0, c in BOTH float32 and float64
        # ═══════════════════════════════════════════════════════════════

        x0 = rms_norm(inputs[0], norm_w)

        # Float32 (original)
        h0_32 = (x0 @ gate_W.T).flatten()
        W_masked_32 = gate_W * mask_eps
        c_32 = (W_masked_32 @ x0.flatten()).flatten()
        del W_masked_32

        # Float64 (refined)
        gate_W_64 = gate_W.astype(np.float64)
        x0_64 = x0.astype(np.float64)
        h0_64 = (x0_64 @ gate_W_64.T).flatten()
        W_masked_64 = gate_W_64 * mask_eps
        c_64 = (W_masked_64 @ x0_64.flatten()).flatten()
        del W_masked_64

        print(f"\n  Float32 vs Float64 comparison:")
        print(f"    h0 max diff: {np.max(np.abs(h0_32 - h0_64.astype(np.float32))):.2e}")
        print(f"    c  max diff: {np.max(np.abs(c_32 - c_64.astype(np.float32))):.2e}")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: First-order estimates (Lambert W analog)
        # ═══════════════════════════════════════════════════════════════

        # Float64 estimates
        valid = np.abs(c_64) > 1e-15
        ratio = np.full(len(h0_64), np.nan)
        ratio[valid] = 1.0 - h0_64[valid] / c_64[valid]
        has_zero = valid & (ratio > 0)
        delta_est = np.full(len(h0_64), np.nan)
        delta_est[has_zero] = np.log(ratio[has_zero]) / LN_PHI

        zero_dims = np.where(has_zero)[0]
        zero_deltas = delta_est[has_zero]
        sort_order = np.argsort(np.abs(zero_deltas))
        zero_dims = zero_dims[sort_order]
        zero_deltas = zero_deltas[sort_order]

        n_test = min(N_TEST, len(zero_dims))
        print(f"\n  {np.sum(has_zero)} total zeros, testing top {n_test} (smallest |δ|)")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3a: Verify first-order estimates (before refinement)
        # ═══════════════════════════════════════════════════════════════

        print(f"\n  ── BEFORE REFINEMENT (first-order only) ──")

        correct_32 = 0
        correct_64 = 0
        residuals_before = []

        for rank in range(n_test):
            j = zero_dims[rank]
            d = zero_deltas[rank]

            # Float32 verification (original method)
            gate_sh_32 = gate_W.copy()
            gate_sh_32[mask_eps] *= np.float32(PHI ** d)
            h_sh_32 = (x0 @ gate_sh_32.T).flatten()
            flipped_32 = (np.sign(h0_32[j]) != np.sign(h_sh_32[j])) or (abs(h_sh_32[j]) < 1e-8)
            if flipped_32: correct_32 += 1
            del gate_sh_32

            # Float64 verification
            gate_sh_64 = gate_W_64.copy()
            gate_sh_64[mask_eps] *= PHI ** d
            h_sh_64 = (x0_64 @ gate_sh_64.T).flatten()
            flipped_64 = (np.sign(h0_64[j]) != np.sign(h_sh_64[j])) or (abs(h_sh_64[j]) < 1e-14)
            if flipped_64: correct_64 += 1
            residuals_before.append(abs(h_sh_64[j]))
            del gate_sh_64

            if rank < 10:
                print(f"    [{rank:>3d}] dim {j:>5d}: h={h0_64[j]:>12.8f} "
                      f"→ {h_sh_64[j]:>12.2e}  "
                      f"f32={'✓' if flipped_32 else '✗'} "
                      f"f64={'✓' if flipped_64 else '✗'}")

        print(f"\n    Float32 hit rate: {correct_32}/{n_test} ({correct_32/n_test:.0%})")
        print(f"    Float64 hit rate: {correct_64}/{n_test} ({correct_64/n_test:.0%})")
        print(f"    Residuals: median={np.median(residuals_before):.2e}, "
              f"max={np.max(residuals_before):.2e}")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3b: Newton refinement
        # ═══════════════════════════════════════════════════════════════

        print(f"\n  ── NEWTON REFINEMENT ──")

        correct_newton = 0
        correct_bisect = 0
        newton_iters = []
        residuals_newton = []

        for rank in range(n_test):
            j = zero_dims[rank]
            d_est = zero_deltas[rank]

            # Newton refinement in float64
            d_newton, n_iter, residual = newton_refine(
                h0_64[j], c_64[j], d_est, max_iter=10, tol=1e-14)
            newton_iters.append(n_iter)

            # Verify with actual float64 matmul
            gate_sh = gate_W_64.copy()
            gate_sh[mask_eps] *= PHI ** d_newton
            h_sh = (x0_64 @ gate_sh.T).flatten()
            flipped = (np.sign(h0_64[j]) != np.sign(h_sh[j])) or (abs(h_sh[j]) < 1e-12)
            residuals_newton.append(abs(h_sh[j]))

            if flipped:
                correct_newton += 1

            # If Newton failed, try bisection
            if not flipped:
                d_bisect, b_iter, b_resid = bisect_refine(
                    h0_64[j], c_64[j], d_est, max_iter=60, tol=1e-14)
                gate_sh2 = gate_W_64.copy()
                gate_sh2[mask_eps] *= PHI ** d_bisect
                h_sh2 = (x0_64 @ gate_sh2.T).flatten()
                flipped_b = (np.sign(h0_64[j]) != np.sign(h_sh2[j])) or (abs(h_sh2[j]) < 1e-12)
                if flipped_b:
                    correct_bisect += 1
                del gate_sh2

            del gate_sh

            if rank < 15:
                status = "✓ NEWTON" if flipped else ("✓ BISECT" if (not flipped and correct_bisect > 0 and rank == n_test) else "✗ MISS")
                # Recompute bisect status for display
                if not flipped:
                    d_b, _, _ = bisect_refine(h0_64[j], c_64[j], d_est)
                    gate_sh_b = gate_W_64.copy()
                    gate_sh_b[mask_eps] *= PHI ** d_b
                    h_b = (x0_64 @ gate_sh_b.T).flatten()
                    flipped_b2 = (np.sign(h0_64[j]) != np.sign(h_b[j])) or (abs(h_b[j]) < 1e-12)
                    status = "✓ BISECT" if flipped_b2 else "✗ MISS"
                    del gate_sh_b

                print(f"    [{rank:>3d}] dim {j:>5d}: δ={d_est:>12.8f} → {d_newton:>12.8f} "
                      f"({n_iter} iter, res={residual:.1e}) "
                      f"h→{h_sh[j]:>10.2e} {status}")

        total_refined = correct_newton + correct_bisect
        print(f"\n    Newton hit rate:  {correct_newton}/{n_test} ({correct_newton/n_test:.0%})")
        print(f"    +Bisect:         {total_refined}/{n_test} ({total_refined/n_test:.0%})")
        print(f"    Newton iterations: mean={np.mean(newton_iters):.1f}, "
              f"max={np.max(newton_iters)}")
        print(f"    Residuals: median={np.median(residuals_newton):.2e}, "
              f"max={np.max(residuals_newton):.2e}")

        # ═══════════════════════════════════════════════════════════════
        # DIAGNOSIS: Why do some zeros resist?
        # ═══════════════════════════════════════════════════════════════

        print(f"\n  ── DIAGNOSIS: Resistant zeros ──")

        misses = []
        for rank in range(n_test):
            j = zero_dims[rank]
            d = zero_deltas[rank]

            d_n, _, res_n = newton_refine(h0_64[j], c_64[j], d)
            gate_sh = gate_W_64.copy()
            gate_sh[mask_eps] *= PHI ** d_n
            h_sh = (x0_64 @ gate_sh.T).flatten()
            flipped = (np.sign(h0_64[j]) != np.sign(h_sh[j])) or (abs(h_sh[j]) < 1e-12)
            del gate_sh

            if not flipped:
                # Compute analytical vs matmul discrepancy
                h_analytical = h0_64[j] + c_64[j] * (PHI ** d_n - 1.0)
                misses.append({
                    'rank': rank, 'dim': j, 'delta': d_n,
                    'h_analytical': h_analytical, 'h_matmul': h_sh[j],
                    'discrepancy': abs(h_analytical - h_sh[j]),
                    'h0': h0_64[j], 'c': c_64[j],
                })

        if misses:
            print(f"    {len(misses)} resistant zeros out of {n_test}")
            print(f"    {'Rank':>5s}  {'Dim':>6s}  {'h_analytic':>12s}  "
                  f"{'h_matmul':>12s}  {'Discrepancy':>12s}  {'|h0|':>10s}")
            for m in misses[:20]:
                print(f"    {m['rank']:>5d}  {m['dim']:>6d}  "
                      f"{m['h_analytical']:>12.4e}  {m['h_matmul']:>12.4e}  "
                      f"{m['discrepancy']:>12.4e}  {abs(m['h0']):>10.6f}")

            # The discrepancy IS the float64 matmul rounding error
            discs = [m['discrepancy'] for m in misses]
            print(f"\n    Discrepancy stats:")
            print(f"      Mean: {np.mean(discs):.4e}")
            print(f"      Max:  {np.max(discs):.4e}")
            print(f"    These are matmul accumulation errors in float64")
            print(f"    The analytical formula is correct; the matmul rounds differently")

            # Can we fix by adjusting δ to compensate for matmul error?
            print(f"\n  ── MATMUL-AWARE NEWTON ──")
            print(f"  Use actual matmul as oracle instead of analytical formula")

            correct_matmul_newton = 0
            for m in misses[:20]:
                j = m['dim']
                d = m['delta']
                # Newton using matmul as f(δ), analytical as f'(δ)
                for iteration in range(10):
                    gate_sh = gate_W_64.copy()
                    gate_sh[mask_eps] *= PHI ** d
                    h_sh = (x0_64 @ gate_sh.T).flatten()
                    f_val = h_sh[j]
                    del gate_sh
                    if abs(f_val) < 1e-14 or np.sign(f_val) != np.sign(h0_64[j]):
                        correct_matmul_newton += 1
                        if m['rank'] < 15:
                            print(f"    dim {j:>5d}: FIXED in {iteration+1} matmul-Newton iters, "
                                  f"h→{f_val:.2e}")
                        break
                    fp = c_64[j] * PHI ** d * LN_PHI
                    if abs(fp) < 1e-30:
                        break
                    d = d - f_val / fp

            print(f"\n    Matmul-Newton rescued: {correct_matmul_newton}/{len(misses[:20])}")
            print(f"    Combined hit rate: "
                  f"{correct_newton + correct_matmul_newton}/{n_test} "
                  f"({(correct_newton + correct_matmul_newton)/n_test:.0%})")

        # Cleanup
        del gate_W, gate_W_64, gate_eps, norm_w, h0_32, c_32, h0_64, c_64
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS")
    print(f"{'='*70}")
    print(f"""
  Newton refinement on the phase shift zero spectrum:

  THREE LEVELS OF REFINEMENT:
    1. First-order (Lambert W): analytical δ₀, ~50% hit in float32
    2. Float64 upgrade: same formula, higher precision
    3. Newton iteration: f(δ)/f'(δ) using analytical formula
    4. Matmul-Newton: f(δ) from actual matmul, f'(δ) analytical

  The remaining misses (if any) are float64 matmul accumulation
  errors — the analytical formula is provably correct per-row,
  but summing 3584 terms in different orders gives different
  rounding. This is the fundamental precision limit.

  Next step: test on real model hidden states to see if
  flipping specific gate dimensions produces semantically
  meaningful output changes.
""")


if __name__ == '__main__':
    run()
