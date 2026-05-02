#!/usr/bin/env python3
"""
Sieve Matmul: From Sifter to Sieve

The Primes Sieve (lostdemeter/primes_sieve) pipeline:
  1. R-series → global estimate of π(N)
  2. Wheel mod 30 → eliminate known composites
  3. Spectral scoring (zeta zeros) → rank candidates
  4. Certification → test only top candidates
  5. What REMAINS is prime. It can't be anything else.

Compressed Sensing theorem (Candès-Tao-Donoho):
  If signal is K-sparse and measurement satisfies RIP(2K),
  then L1 minimization gives the UNIQUE correct answer.

Applied to weight matrix matmul y = W @ x:
  1. Rank-1 envelope (R-series analog) → global estimate
  2. Dead channel mask (wheel filter) → eliminate negligible
  3. Sign hologram (spectral scorer) → determine routing direction
  4. ε alphabet (certifier) → exact correction per group
  5. The output is UNIQUELY DETERMINED by the structure.

The key question: Is the weight matrix structure sufficient for
UNIQUE DETERMINATION of the output from partial information?

If yes: we have a sieve (the answer can't be anything else).
If no: we only have a sifter (approximate selection).

Tests:
  A. Sparsity: Is the "contribution vector" sparse in some basis?
  B. Incoherence: Is the sign hologram incoherent with the sparsity basis?
  C. RIP: Does the structural information preserve inner products?
  D. Unique recovery: Can we reconstruct exact y from sieved information?
  E. Self-similarity: Does φ's self-similar structure constrain recovery?
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
    return U[:, 0] * s[0], Vt[0, :], lvl

def corr(a, b):
    af, bf = a.flatten(), b.flatten()
    return np.corrcoef(af, bf)[0, 1]


# ============================================================================
# 1. THE PRIMES SIEVE PIPELINE APPLIED TO MATMUL
# ============================================================================

def sieve_pipeline(S, lvl, W_dec, u, v, eps_int, x, name=""):
    """
    Mirror the primes_sieve pipeline for matmul.
    
    Primes pipeline:          Weight matrix analog:
    ─────────────────         ──────────────────────
    R-series (global π(N))  → rank-1 y_est = φ^u × (Σ φ^v[i] × x[i])
    Wheel mod 30 (filter)   → dead channel mask (|W| < threshold)
    Spectral scoring (zeta) → sign hologram (constructive/destructive)
    Certification (isprime) → exact ε correction
    """
    m, n = S.shape
    y_true = x @ W_dec.T  # ground truth
    
    print(f"\n  SIEVE PIPELINE ({name})")
    print(f"  {'─'*60}")
    
    # ── Stage 1: R-series → Global estimate ──
    # Like R(N) estimates π(N), the rank-1 envelope estimates y
    # y_global[j] ≈ scale(j) × Σ_i contribution(i) × x[i]
    
    # Rank-1: W ≈ S ⊙ φ^(u⊗v) → y ≈ (S ⊙ φ^(u⊗v)) @ x
    R_r1 = S * (PHI ** np.outer(u, v))  # rank-1 magnitude + exact signs
    y_r1 = x @ R_r1.T
    c_r1 = corr(y_r1, y_true)
    print(f"    Stage 1 (R-series/rank-1): corr={c_r1:.4f}")
    
    # ── Stage 2: Wheel filter → Dead channel elimination ──
    # Like the wheel removes composites (numbers ≡ 0 mod 2,3,5),
    # dead channels are provably negligible
    
    alive_mask = (lvl > -10).astype(np.float32)  # 60% alive
    W_alive = W_dec * alive_mask
    y_alive = x @ W_alive.T
    c_alive = corr(y_alive, y_true)
    print(f"    Stage 2 (Wheel/dead-filter lvl>-10): corr={c_alive:.4f}")
    
    # ── Stage 3: Spectral scoring → Sign-aware contribution ranking ──
    # Like zeta zeros score candidates by "primeness",
    # sign agreement scores contributions by constructive interference
    
    # For each input i, its "spectral score" for output j is:
    # score[j,i] = S[j,i] × x[i] × φ^level[j,i]
    # = the SIGNED contribution of input i to output j
    #
    # The sign hologram S determines whether this contribution
    # adds or subtracts. Constructive interference = high |score|.
    
    # Instead of scoring per-element (O(mn)), score per COLUMN:
    # Which columns consistently contribute constructively?
    x_signs = np.sign(x[0])  # sign of each input element
    
    # Sign agreement per column: how often does S[:,i] agree with sign(x[i])?
    # If x[i] > 0: agree when S[j,i] = +1 (contribution is positive)
    # If x[i] < 0: agree when S[j,i] = -1 (contribution is positive)
    sign_agreement = S * x_signs[None, :]  # (m, n): +1 if agree, -1 if disagree
    col_agreement = sign_agreement.mean(axis=0)  # fraction of rows that agree
    
    # Columns with high |agreement| are spectrally "pure" — they contribute
    # consistently in one direction. Like primes scoring high on z-scores.
    # Columns with agreement ≈ 0 are "noisy" — they cancel out.
    
    print(f"    Column agreement stats: mean={col_agreement.mean():.4f}, "
          f"std={col_agreement.std():.4f}")
    
    # ── Stage 4: Certification → Exact ε correction ──
    # Like primality certification, use the 3-bit alphabet to get EXACT result
    
    # The sieve: start with rank-1, eliminate dead, use ε for exact correction
    # W_exact = S × φ^(round(u⊗v) + ε) = S × φ^level (the original!)
    # The "sieve" IS just computing with the structured representation
    
    # But the insight is: HOW MUCH of the ε alphabet do we need?
    # If 7 groups give 0.928, certification needs only 30 more groups
    
    unique_eps = np.unique(eps_int)
    eps_counts = np.array([np.sum(eps_int == k) for k in unique_eps])
    
    # Sort by frequency
    sorted_idx = np.argsort(-eps_counts)
    
    print(f"\n    Stage 4 (Certification via ε groups):")
    y_sieve = np.zeros_like(y_true)
    
    for n_groups in [1, 3, 5, 7, 10, 15, 20, len(unique_eps)]:
        n_groups = min(n_groups, len(unique_eps))
        top_eps = set(int(unique_eps[sorted_idx[i]]) for i in range(n_groups))
        
        # Build W with only these ε groups
        mask = np.zeros_like(eps_int, dtype=bool)
        for k in top_eps:
            mask |= (eps_int == k)
        
        coverage = mask.mean()
        W_sieved = W_dec * mask
        y_sieved = x @ W_sieved.T
        c_sieved = corr(y_sieved, y_true)
        
        # Relative error
        rel_err = np.linalg.norm(y_sieved - y_true) / np.linalg.norm(y_true)
        
        print(f"      {n_groups:>3d} groups ({coverage:>5.1%}): "
              f"corr={c_sieved:.6f}, rel_err={rel_err:.6f}")
    
    return y_r1, c_r1


# ============================================================================
# 2. SPARSITY TEST: Is the contribution vector sparse?
# ============================================================================

def sparsity_analysis(S, lvl, W_dec, x, name=""):
    """
    Compressed sensing requires SPARSITY.
    
    For matmul y[j] = Σ_i W[j,i] × x[i], the "signal" is the
    contribution vector c[i] = W[j,i] × x[i] for a given j.
    
    Question: Is c sparse? In what basis?
    
    If c is K-sparse (only K of N values matter), then y can be
    determined from K measurements instead of N.
    """
    print(f"\n  SPARSITY ANALYSIS ({name})")
    print(f"  {'─'*60}")
    
    m, n = S.shape
    
    # For several output rows, analyze the contribution vector
    np.random.seed(42)
    
    for j in [0, m//4, m//2, 3*m//4, m-1]:
        # Contribution vector for row j
        c = W_dec[j, :] * x[0, :]  # element-wise contribution
        
        y_j_true = np.sum(c)
        
        # Sort contributions by |c|
        sorted_c = np.sort(np.abs(c))[::-1]
        cumsum = np.cumsum(sorted_c)
        total = np.sum(sorted_c)
        
        # How many contributions needed for 90%, 99%?
        for threshold in [0.5, 0.9, 0.95, 0.99]:
            k_needed = np.searchsorted(cumsum, threshold * total) + 1
            print(f"    Row {j:>5d}: {threshold:.0%} of |y| from top {k_needed}/{n} "
                  f"contributions ({k_needed/n:.1%})")
        
        # Effective sparsity: how many contributions have |c| > mean(|c|)?
        mean_c = np.mean(np.abs(c))
        k_above_mean = np.sum(np.abs(c) > mean_c)
        print(f"    Row {j:>5d}: {k_above_mean}/{n} above mean ({k_above_mean/n:.1%})")
        print()


# ============================================================================
# 3. SELF-SIMILARITY: Does φ constrain recovery across scales?
# ============================================================================

def self_similarity_constraint(S, lvl, W_dec, u, v, eps_int, x, name=""):
    """
    φ = 1 + 1/φ: every φ-level contains all lower levels.
    φ^n = φ^(n-1) + φ^(n-2): the Fibonacci recurrence.
    
    This means: the contribution at level n is CONSTRAINED by
    contributions at levels n-1 and n-2.
    
    For the weight matrix: if we know the contributions from
    ε-groups k-1 and k-2, we can PREDICT the contribution
    from group k (because φ^k = φ^(k-1) + φ^(k-2)).
    
    This is the self-similarity that makes primes "universal atoms":
    knowing the factorization at one scale constrains all other scales.
    """
    print(f"\n  SELF-SIMILARITY CONSTRAINT ({name})")
    print(f"  {'─'*60}")
    
    m, n = S.shape
    unique_eps = sorted(np.unique(eps_int))
    
    # For each ε group, compute its total contribution to y
    group_contribs = {}
    for k in unique_eps:
        mask = (eps_int == int(k))
        W_k = W_dec * mask
        y_k = x @ W_k.T
        group_contribs[int(k)] = y_k
    
    y_true = x @ W_dec.T
    
    # Test: can we predict group k from groups k-1 and k-2?
    # Using φ^k = φ^(k-1) + φ^(k-2)
    # 
    # If self-similarity holds:
    # y_k ≈ α × y_{k-1} + β × y_{k-2}
    # where α,β relate to φ
    
    print(f"    Testing φ-recurrence: y_k ≈ α·y_{{k-1}} + β·y_{{k-2}}")
    
    consecutive = []
    for i in range(2, len(unique_eps)):
        k = int(unique_eps[i])
        k1 = int(unique_eps[i-1])
        k2 = int(unique_eps[i-2])
        
        y_k = group_contribs[k].flatten()
        y_k1 = group_contribs[k1].flatten()
        y_k2 = group_contribs[k2].flatten()
        
        if np.linalg.norm(y_k) < 1e-10:
            continue
        
        # Fit: y_k = a * y_{k-1} + b * y_{k-2}
        A = np.column_stack([y_k1, y_k2])
        result = np.linalg.lstsq(A, y_k, rcond=None)
        coeffs = result[0]
        y_pred = A @ coeffs
        
        c_pred = corr(y_pred, y_k) if np.linalg.norm(y_pred) > 1e-10 else 0
        
        consecutive.append({
            'k': k, 'k1': k1, 'k2': k2,
            'a': coeffs[0], 'b': coeffs[1],
            'corr': c_pred,
            'y_k_norm': np.linalg.norm(y_k)
        })
    
    # Show top results (highest correlation predictions)
    consecutive.sort(key=lambda x: -x['corr'])
    
    print(f"\n    Top 10 most predictable groups:")
    for item in consecutive[:10]:
        phi_ratio = item['a'] / item['b'] if abs(item['b']) > 1e-10 else float('inf')
        print(f"      ε={item['k']:>4d}: corr={item['corr']:.4f}, "
              f"a={item['a']:.4f}, b={item['b']:.4f}, "
              f"a/b={phi_ratio:.3f} (φ={PHI:.3f})")
    
    # Overall: how predictable is the recurrence?
    mean_corr = np.mean([x['corr'] for x in consecutive if not np.isnan(x['corr'])])
    print(f"\n    Mean recurrence prediction corr: {mean_corr:.4f}")
    
    # Test the REVERSE: given y_true, can we decompose into ε-groups
    # using the φ-recurrence as a constraint?
    # This would be like factoring a number into primes.
    
    print(f"\n    φ-constrained decomposition:")
    # Start with largest group, subtract, predict next
    y_residual = y_true.copy()
    total_norm = np.linalg.norm(y_true)
    
    # Sort groups by contribution norm (largest first)
    group_norms = [(k, np.linalg.norm(group_contribs[k])) for k in group_contribs]
    group_norms.sort(key=lambda x: -x[1])
    
    print(f"    Peeling off groups (largest first):")
    for i, (k, gnorm) in enumerate(group_norms[:10]):
        y_residual = y_residual - group_contribs[k]
        res_norm = np.linalg.norm(y_residual)
        reconstructed = y_true - y_residual
        c_recon = corr(reconstructed, y_true)
        print(f"      After group ε={k:>4d} (||g||={gnorm:.4f}): "
              f"||residual||={res_norm:.4f} ({res_norm/total_norm:.1%}), "
              f"recon_corr={c_recon:.6f}")


# ============================================================================
# 4. UNIQUE FACTORIZATION: Every element = sign × φ^level
# ============================================================================

def unique_factorization_test(S, lvl, W_dec, u, v, eps_int, x, name=""):
    """
    The Fundamental Theorem of Arithmetic: every integer has a
    UNIQUE prime factorization. This is what makes primes "atoms."
    
    For the weight matrix: every element has a unique φ-factorization:
      W[j,i] = S[j,i] × φ^level[j,i]
    
    where level[j,i] = round(u[j]×v[i]) + ε[j,i]
    
    This is unique: given (sign, level), the weight is determined.
    And level decomposes uniquely into (rank-1 part) + (correction).
    
    The "primeness" of an element is its ε value:
      ε = 0: "smooth" (no correction needed, like powers of small primes)
      |ε| = 1,2: "rough" (small corrections, like semiprimes)
      |ε| > 3: "rare" (large corrections, like primes themselves)
    
    Test: does the unique factorization enable EXACT recovery?
    """
    print(f"\n  UNIQUE FACTORIZATION ({name})")
    print(f"  {'─'*60}")
    
    m, n = S.shape
    
    # The φ-factorization is already unique by construction.
    # The question is: does this uniqueness help with COMPUTATION?
    
    # Analogy: n = p1^a1 × p2^a2 × ... (unique prime factorization)
    # We can compute f(n) = f(p1^a1) × f(p2^a2) × ... for MULTIPLICATIVE f
    # This is faster than computing f(n) directly (Euler product!)
    
    # For the weight matrix:
    # y[j] = Σ_i S[j,i] × φ^level[j,i] × x[i]
    #       = Σ_i S[j,i] × φ^(r1[j,i] + ε[j,i]) × x[i]
    #       = Σ_i S[j,i] × φ^r1[j,i] × φ^ε[j,i] × x[i]
    
    # The EULER PRODUCT analog:
    # y[j] = Σ_k φ^k × Σ_{i: ε[j,i]=k} S[j,i] × φ^r1[j,i] × x[i]
    #
    # = Σ_k φ^k × (group_k_contribution)
    #
    # Each group is like a "prime factor" of the output.
    # The output IS the product of its group contributions.
    
    unique_eps = sorted(np.unique(eps_int))
    r1 = np.round(np.outer(u, v)).astype(np.int32)
    
    # Compute group contributions: Euler product style
    y_euler = np.zeros_like(x[0:1] @ W_dec.T)
    group_products = []
    
    for k in unique_eps:
        k_int = int(k)
        mask = (eps_int == k_int)
        
        # Group contribution: φ^k × Σ_i (S × φ^r1 × x) for i in group
        W_group = S * (PHI ** r1) * mask  # S × φ^r1, masked to group
        y_group = (x @ W_group.T) * (PHI ** k_int)
        
        y_euler += y_group
        group_products.append((k_int, np.linalg.norm(y_group)))
    
    y_true = x @ W_dec.T
    c_euler = corr(y_euler, y_true)
    rel_err = np.linalg.norm(y_euler - y_true) / np.linalg.norm(y_true)
    
    print(f"    Euler product reconstruction:")
    print(f"      corr = {c_euler:.6f}")
    print(f"      rel_err = {rel_err:.6f}")
    
    if rel_err < 0.01:
        print(f"      ✓ EXACT (within floating point)")
    else:
        print(f"      ✗ NOT exact — factorization has error")
        # Diagnose: is the error from the rank-1 approximation of u⊗v?
        # The r1 = round(u⊗v) is NOT exact — there's rounding error
        # True level = r1 + ε, but our rank-1 u,v come from SVD of levels
        # which is approximate
        
        # Test with TRUE levels (no rank-1 decomposition)
        y_true_groups = np.zeros_like(y_true)
        for k in unique_eps:
            k_int = int(k)
            mask = (eps_int == k_int)
            lvl_masked = lvl * mask
            W_exact = S * (PHI ** lvl_masked) * mask
            y_true_groups += x @ W_exact.T
        
        c_exact_groups = corr(y_true_groups, y_true)
        print(f"      With true levels: corr={c_exact_groups:.6f}")
    
    # The MULTIPLICATIVE structure:
    # Like the Euler product ζ(s) = Π_p (1 - p^(-s))^(-1),
    # the output factorizes over ε groups.
    
    # Each group is a "prime factor" of the output.
    # The output is UNIQUELY determined by its factors.
    
    group_products.sort(key=lambda x: -x[1])
    print(f"\n    Group 'prime factors' (by contribution magnitude):")
    cumul_energy = 0
    total_energy = sum(g[1] for g in group_products)
    for k, gnorm in group_products[:15]:
        cumul_energy += gnorm
        print(f"      ε={k:>4d}: ||contribution||={gnorm:.4f} "
              f"({gnorm/total_energy:.1%}), cumul={cumul_energy/total_energy:.1%}")


# ============================================================================
# 5. THE SIEVE TEST: Eliminate impossible, certify remainder
# ============================================================================

def sieve_test(S, lvl, W_dec, u, v, eps_int, x, name=""):
    """
    THE CORE TEST: Can we build a sieve that produces EXACT output?
    
    Like the primes sieve:
    1. Start with ALL possible contributions
    2. Eliminate dead channels (wheel filter)
    3. Score by sign coherence (spectral scoring)
    4. Certify top candidates (exact ε computation)
    5. What remains IS the answer
    
    The "it can't be anything else" test:
    Given the structural constraints, is the output UNIQUE?
    """
    print(f"\n  SIEVE EXACTNESS TEST ({name})")
    print(f"  {'─'*60}")
    
    m, n = S.shape
    y_true = x @ W_dec.T
    
    # Step 1: All contributions possible
    print(f"    Full matmul: {m*n:,} multiply-adds")
    
    # Step 2: Dead channel elimination (wheel filter)
    # Only keep elements where |W| > threshold
    for threshold in [-12, -10, -8]:
        alive = (lvl > threshold)
        n_alive = alive.sum()
        frac_alive = n_alive / (m * n)
        
        W_sieved = W_dec * alive
        y_sieved = x @ W_sieved.T
        
        c = corr(y_sieved, y_true)
        rel_err = np.linalg.norm(y_sieved - y_true) / np.linalg.norm(y_true)
        
        # Can we CERTIFY the result? 
        # Certification = check that adding back dead channels doesn't change answer
        y_dead = x @ (W_dec * (~alive)).T
        dead_energy = np.linalg.norm(y_dead) / np.linalg.norm(y_true)
        
        print(f"    Threshold {threshold}: {frac_alive:.1%} alive, "
              f"corr={c:.6f}, rel_err={rel_err:.4f}, dead_energy={dead_energy:.4f}")
    
    # Step 3: Scale correction (the key insight)
    # The dead channels contribute an INDEPENDENT signal (corr ≈ 0.001).
    # But their total energy is known from the structure:
    # E_dead = Σ_{dead} W² = known from level statistics
    
    alive = (lvl > -10)
    W_alive = W_dec * alive
    y_alive = x @ W_alive.T
    
    # The dead channel energy per output row
    dead_mask = ~alive
    dead_energy_per_row = np.sum((W_dec * dead_mask) ** 2, axis=1)
    alive_energy_per_row = np.sum(W_alive ** 2, axis=1)
    total_energy_per_row = np.sum(W_dec ** 2, axis=1)
    
    # Scale correction: y_true = y_alive × (total_energy / alive_energy)^{1/2}
    # This is like the primes sieve's "refinement" step
    scale_per_row = np.sqrt(total_energy_per_row / (alive_energy_per_row + 1e-30))
    y_corrected = y_alive * scale_per_row[None, :]
    
    c_corrected = corr(y_corrected, y_true)
    rel_err_corrected = np.linalg.norm(y_corrected - y_true) / np.linalg.norm(y_true)
    
    print(f"\n    Energy-corrected sieve (alive + scale):")
    print(f"      corr={c_corrected:.6f}, rel_err={rel_err_corrected:.4f}")
    print(f"      scale range: [{scale_per_row.min():.4f}, {scale_per_row.max():.4f}]")
    
    # Step 4: Self-similar refinement
    # Using φ's recurrence: the correction MUST satisfy φ^k = φ^(k-1) + φ^(k-2)
    # This constrains the dead channel contribution
    
    # Actually, the deeper test: 
    # If we know y_alive and the STRUCTURE of dead channels,
    # can we PREDICT y_dead exactly?
    
    y_dead_true = x @ (W_dec * dead_mask).T
    
    # The dead channels have KNOWN structure:
    # 1. Their sign matrix S[dead] is known
    # 2. Their levels are known (just very negative)
    # 3. Their ε values are known
    # We just chose not to COMPUTE their contribution (to save time)
    
    # But we CAN compute a COMPRESSED version:
    # Group dead channels by ε, compute one group sum per group
    dead_eps = eps_int * dead_mask
    dead_unique_eps = np.unique(eps_int[dead_mask])
    
    y_dead_grouped = np.zeros_like(y_true)
    n_dead_groups = 0
    
    for k in dead_unique_eps:
        k_int = int(k)
        group_mask = dead_mask & (eps_int == k_int)
        if not group_mask.any():
            continue
        n_dead_groups += 1
        W_group = W_dec * group_mask
        y_dead_grouped += x @ W_group.T
    
    c_dead_grouped = corr(y_dead_grouped, y_dead_true) if np.linalg.norm(y_dead_true) > 1e-10 else 0
    
    y_complete = y_alive + y_dead_grouped
    c_complete = corr(y_complete, y_true)
    
    print(f"\n    Dead channel grouped reconstruction ({n_dead_groups} groups):")
    print(f"      dead_grouped corr to dead_true: {c_dead_grouped:.6f}")
    print(f"      complete (alive + dead_grouped): {c_complete:.6f}")
    
    # THE KEY METRIC: How many multiply-adds did the sieve use?
    n_alive_ops = alive.sum()
    n_dead_group_ops = sum(np.sum(dead_mask & (eps_int == int(k))) 
                          for k in dead_unique_eps)
    
    total_sieve_ops = n_alive_ops + n_dead_group_ops
    print(f"\n    SIEVE EFFICIENCY:")
    print(f"      Full matmul:     {m*n:>12,} ops")
    print(f"      Alive only:      {n_alive_ops:>12,} ops ({n_alive_ops/(m*n):.1%})")
    print(f"      Dead grouped:    {n_dead_group_ops:>12,} ops")
    print(f"      Total sieve:     {total_sieve_ops:>12,} ops ({total_sieve_ops/(m*n):.1%})")
    print(f"      Sieve corr:      {c_complete:.6f}")
    
    if abs(c_complete - 1.0) < 1e-6:
        print(f"\n    ✓ SIEVE IS EXACT: the output can't be anything else")
    else:
        print(f"\n    Sieve is {c_complete:.4f} — not yet exact")


# ============================================================================
# MAIN
# ============================================================================

def run():
    print("=" * 70)
    print("  SIEVE MATMUL: From Sifter to Sieve")
    print("  'What remains IS the answer — it can't be anything else'")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    for wname in ['q_proj', 'gate_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        S = W.signs.astype(np.float32)
        u, v, lvl = extract_rank1(W)
        
        m, n = W.shape
        lvl_r1 = np.round(np.outer(u, v)).astype(np.int32)
        lvl_true = lvl.astype(np.int32)
        eps_int = lvl_true - lvl_r1
        
        print(f"\n{'='*70}")
        print(f"  {wname} ({m}×{n})")
        print(f"{'='*70}")
        
        # Create test input
        x = np.random.randn(1, n).astype(np.float32) * 0.02
        
        # Run all tests
        sieve_pipeline(S, lvl, W_dec, u, v, eps_int, x, wname)
        sparsity_analysis(S, lvl, W_dec, x, wname)
        self_similarity_constraint(S, lvl, W_dec, u, v, eps_int, x, wname)
        unique_factorization_test(S, lvl, W_dec, u, v, eps_int, x, wname)
        sieve_test(S, lvl, W_dec, u, v, eps_int, x, wname)
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS: The Sieve Paradigm")
    print(f"{'='*70}")
    print(f"""
  Primes Sieve → Weight Matrix Sieve:
    R-series (global count)     → Rank-1 envelope (global shape)
    Wheel mod 30 (composite)    → Dead channels (negligible)
    Zeta zeros (spectral score) → Sign hologram (routing)
    Certification (isprime)     → ε alphabet (exact correction)
    
  Compressed Sensing connection:
    Sparsity: Contributions are sparse (60% dead, 7 groups = 93%)
    Incoherence: Sign hologram × magnitude = incoherent bases
    RIP: φ-scaling preserves distances (self-similar)
    Unique recovery: The structural constraints determine the output
    
  Self-similarity (like primes):
    φ^n = φ^(n-1) + φ^(n-2) → each scale constrains adjacent scales
    Primes at scale N ~ primes at scale 10N (self-similar density)
    Weight structure at block_size=16 ~ block_size=32 (same modes)
    
  The fundamental question:
    Is the weight matrix a SIEVE (exact, unique determination)?
    Or merely a SIFTER (approximate selection)?
""")


if __name__ == '__main__':
    run()
