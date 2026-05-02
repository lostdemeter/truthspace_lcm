#!/usr/bin/env python3
"""
Navigation vs Inference: Can we TRAVERSE the weight matrix instead of multiplying?

The weight matrix W = S ⊙ φ^(u⊗v + ε) is a DEVICE:
  - S: sign hologram → ROUTING (which inputs add vs subtract)
  - u⊗v: rank-1 envelope → GLOBAL SHAPE (how loud each output/input is)
  - ε: 3-bit integer alphabet → INSTRUCTION SET (local corrections)
  - Dead channels (|W|≈0): GATES (negative zero 4th dimension)

Standard inference: y[j] = Σᵢ W[j,i] * x[i]  — touch every element, O(N²)

Navigation hypothesis:
  y[j] = φ^u[j] × Σ_{k∈alphabet} φ^k × signed_group_sum(x̃, S[j], group_k)

where x̃[i] = φ^v[i] × x[i] is the envelope-weighted input,
and group_k = {i : ε[j,i] = k}.

This reduces 3584 terms to 37-52 GROUP sums, each weighted by φ^k.
The question: does the GROUP structure let us avoid touching every element?

Exploration:
  1. Decompose matmul into grouped φ-level sums
  2. Analyze group sizes and sign balance (do groups self-cancel?)
  3. Test whether dead-channel gating provides natural sparsity
  4. Measure what happens when we navigate by DOMINANT groups only
  5. Test the automaton: do inputs activate specific state trajectories?
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

def matmul_corr(y_approx, y_true):
    return np.corrcoef(y_true.flatten(), y_approx.flatten())[0, 1]


# ============================================================================
# 1. GROUPED φ-LEVEL DECOMPOSITION
# ============================================================================

def grouped_phi_matmul(S, u, v, eps_int, x, n_test=100):
    """
    Decompose y = W @ x into grouped φ-level sums.
    
    y[j] = φ^u[j] × Σ_k φ^k × Σ_{i: ε[j,i]=k} S[j,i] × x̃[i]
    
    where x̃[i] = φ^v[i] × x[i]
    """
    m, n = S.shape
    
    # Envelope-weighted input: x̃[i] = φ^v[i] × x[i]
    x_tilde = x * (PHI ** v)[None, :]  # (n_test, n)
    
    # Unique ε values (the instruction alphabet)
    unique_eps = np.unique(eps_int)
    n_groups = len(unique_eps)
    
    # Output scale per row
    row_scale = PHI ** u  # (m,)
    
    # For each output row, compute the grouped sum
    # y[j] = row_scale[j] × Σ_k φ^k × (signed sum of x̃ for group k in row j)
    
    # Precompute group masks for each row
    # This is the STRUCTURE we'd store once at load time
    t0 = time.perf_counter()
    
    # Build group membership: for each (j, k), which input indices belong?
    # Store as: group_indices[k] = array of column indices where ε=k (per row)
    # But ε varies per row! So we need per-row groups.
    
    # Actually, let's check: does ε vary much across rows for the same column?
    # If ε[j,i] ≈ f(i) (column-dependent only), groups are shared across rows
    
    col_eps_std = np.std(eps_int.astype(float), axis=0)  # std across rows for each col
    col_eps_mean = np.mean(eps_int.astype(float), axis=0)
    print(f"    Column ε consistency: mean_std={col_eps_std.mean():.3f}")
    print(f"    (If near 0, ε is column-determined → groups shared across rows)")
    
    row_eps_std = np.std(eps_int.astype(float), axis=1)
    print(f"    Row ε consistency: mean_std={row_eps_std.mean():.3f}")
    
    # Method A: Per-column grouping (approximate: use column-median ε)
    col_eps_median = np.median(eps_int, axis=0).astype(int)
    
    y_grouped = np.zeros((n_test, m), dtype=np.float64)
    
    for k in unique_eps:
        k_int = int(k)
        col_mask = (col_eps_median == k_int)
        if not col_mask.any():
            continue
        
        # Signed group sum: Σ_i S[j,i] × x̃[t,i] for i in group k
        # S[:, col_mask] is (m, n_k), x_tilde[:, col_mask] is (n_test, n_k)
        S_group = S[:, col_mask]  # (m, n_k)
        x_group = x_tilde[:, col_mask]  # (n_test, n_k)
        
        # signed_sum[t, j] = Σ_i S[j,i] × x̃[t,i] for this group
        signed_sum = x_group @ S_group.T  # (n_test, m)
        
        y_grouped += (PHI ** k_int) * signed_sum
    
    # Apply row scale
    y_grouped *= row_scale[None, :]
    
    t_grouped = time.perf_counter() - t0
    
    # Method B: Per-row exact grouping
    t0 = time.perf_counter()
    y_exact_grouped = np.zeros((n_test, m), dtype=np.float64)
    
    # This is O(m × n_groups × n_k) where n_k is average group size
    # Total: O(m × n) same as matmul, but structured
    for k in unique_eps:
        k_int = int(k)
        mask = (eps_int == k_int)  # (m, n) boolean
        
        # For each row, the group is different
        # S_masked[j, i] = S[j,i] if ε[j,i]=k else 0
        S_masked = S.astype(np.float64) * mask  # (m, n)
        signed_sum = x_tilde @ S_masked.T  # (n_test, m)
        
        y_exact_grouped += (PHI ** k_int) * signed_sum
    
    y_exact_grouped *= row_scale[None, :]
    t_exact = time.perf_counter() - t0
    
    return y_grouped, y_exact_grouped, t_grouped, t_exact, n_groups, col_eps_std


# ============================================================================
# 2. DEAD CHANNEL ANALYSIS
# ============================================================================

def dead_channel_analysis(S, lvl, eps_int, x_tilde, W_dec, name=""):
    """
    Dead channels: where level is very negative (|W| ≈ 0).
    DC 253: these carry SIGN information (negative zero).
    
    Question: do dead channels contribute to matmul? 
    What if we gate them out vs gate them in?
    """
    print(f"\n  DEAD CHANNEL ANALYSIS ({name})")
    print(f"  {'─'*50}")
    
    m, n = lvl.shape
    
    # Define "dead" as level < threshold (very small magnitude)
    for threshold in [-10, -8, -6, -4]:
        dead_mask = (lvl < threshold)
        dead_frac = dead_mask.mean()
        
        # What's the matmul contribution of dead channels?
        W_dead = W_dec * dead_mask
        W_alive = W_dec * (~dead_mask)
        
        n_test = min(50, x_tilde.shape[0])
        x_test = x_tilde[:n_test]
        
        y_full = x_test @ W_dec.T
        y_alive = x_test @ W_alive.T
        y_dead = x_test @ W_dead.T
        
        corr_alive = matmul_corr(y_alive, y_full)
        
        # Relative contribution
        dead_energy = np.linalg.norm(y_dead) / (np.linalg.norm(y_full) + 1e-30)
        
        print(f"    level < {threshold}: {dead_frac:.1%} dead, "
              f"alive_corr={corr_alive:.4f}, dead_energy={dead_energy:.4f}")
    
    # Sign distribution in dead channels
    for threshold in [-8, -6]:
        dead_mask = (lvl < threshold)
        if dead_mask.any():
            dead_signs = S[dead_mask]
            pos_frac = np.mean(dead_signs > 0)
            print(f"    Dead (lvl<{threshold}) sign balance: +={pos_frac:.4f}")
    
    # The real question: do dead channels' SIGNS predict anything about alive channels?
    # Group rows by their dead-channel sign pattern
    dead_mask = (lvl < -8)
    n_dead_per_row = dead_mask.sum(axis=1)
    print(f"\n    Dead channels per row (lvl<-8): "
          f"mean={n_dead_per_row.mean():.1f}, std={n_dead_per_row.std():.1f}")


# ============================================================================
# 3. AUTOMATON STATE TRAJECTORIES
# ============================================================================

def automaton_trajectories(S, lvl, eps_int, x_float, W_dec, name=""):
    """
    The weight matrix is a finite state automaton.
    Question: does the input vector activate specific STATE TRAJECTORIES?
    
    State = block mode (tetromino). Trajectory = sequence of states
    an input traverses as we scan across input dimensions.
    """
    print(f"\n  AUTOMATON TRAJECTORIES ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    # Define states by quantized ε level
    # Coarse: ε ∈ {<-3, -3..-1, 0, 1..3, >3}
    def state_of(e):
        if e < -3: return 0   # "deep negative"
        if e < 0: return 1    # "shallow negative"  
        if e == 0: return 2   # "zero correction"
        if e <= 3: return 3   # "shallow positive"
        return 4              # "deep positive"
    
    state_names = ["deep-", "shallow-", "zero", "shallow+", "deep+"]
    n_states = 5
    
    # Map ε to states
    state_map = np.vectorize(state_of)(eps_int)  # (m, n)
    
    # State distribution
    for s in range(n_states):
        frac = np.mean(state_map == s)
        print(f"    State '{state_names[s]}': {frac:.1%}")
    
    # Transition matrix: P(state[i+1] | state[i]) across columns
    trans = np.zeros((n_states, n_states), dtype=np.int64)
    for j in range(min(500, m)):
        for i in range(n - 1):
            trans[state_map[j, i], state_map[j, i+1]] += 1
    
    # Normalize
    trans_prob = trans / (trans.sum(axis=1, keepdims=True) + 1e-10)
    
    print(f"\n    Transition matrix P(next | current):")
    print(f"    {'':>12s}", end="")
    for s in range(n_states):
        print(f" {state_names[s]:>10s}", end="")
    print()
    for s in range(n_states):
        print(f"    {state_names[s]:>12s}", end="")
        for t in range(n_states):
            print(f" {trans_prob[s, t]:>10.3f}", end="")
        print()
    
    # Is the transition matrix DIFFERENT from the marginal distribution?
    marginal = np.array([np.mean(state_map == s) for s in range(n_states)])
    kl_per_state = np.zeros(n_states)
    for s in range(n_states):
        for t in range(n_states):
            if trans_prob[s, t] > 0 and marginal[t] > 0:
                kl_per_state[s] += trans_prob[s, t] * np.log2(trans_prob[s, t] / marginal[t])
    
    print(f"\n    KL(transition || marginal) per state:")
    for s in range(n_states):
        print(f"      {state_names[s]:>12s}: {kl_per_state[s]:.4f} bits")
    print(f"    Total mutual info: {np.mean(kl_per_state):.4f} bits")
    
    # KEY TEST: Does the input vector's sign pattern select different trajectories?
    # Take several different input vectors and see if they activate different state paths
    print(f"\n    Input-dependent activation:")
    np.random.seed(42)
    
    for trial in range(3):
        x = np.random.randn(n).astype(np.float32) * 0.02
        
        # "Active" dimensions: where |x| > median
        x_active = np.abs(x) > np.median(np.abs(x))
        x_sign = np.sign(x)
        
        # For a specific output row (j=0), what's the contribution per state?
        j = trial * 100  # different output rows
        row_eps = eps_int[j]
        row_S = S[j]
        row_states = state_map[j]
        
        contrib_per_state = np.zeros(n_states)
        n_per_state = np.zeros(n_states)
        
        for s in range(n_states):
            mask = (row_states == s)
            if mask.any():
                n_per_state[s] = mask.sum()
                # Contribution = Σ S[j,i] × φ^level[j,i] × x[i] for i in state s
                contrib_per_state[s] = np.sum(
                    row_S[mask] * (PHI ** lvl[j, mask]) * x[mask]
                )
        
        y_true = np.sum(W_dec[j] * x)
        
        print(f"    Row {j}: y_true={y_true:.6f}")
        for s in range(n_states):
            if n_per_state[s] > 0:
                print(f"      {state_names[s]:>12s}: n={int(n_per_state[s]):>5d}, "
                      f"contrib={contrib_per_state[s]:>+.6f} "
                      f"({abs(contrib_per_state[s])/(abs(y_true)+1e-30):.0%})")


# ============================================================================
# 4. DOMINANT GROUP NAVIGATION
# ============================================================================

def dominant_group_navigation(S, u, v, lvl, eps_int, W_dec, name=""):
    """
    If most of the matmul comes from a few ε groups, we can navigate
    by computing ONLY those groups.
    
    Test: what correlation do we get from the top-K groups only?
    """
    print(f"\n  DOMINANT GROUP NAVIGATION ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    np.random.seed(42)
    n_test = 100
    X = np.random.randn(n_test, n).astype(np.float32) * 0.02
    
    Y_true = X @ W_dec.T
    
    # Sort ε values by their total contribution weight
    unique_eps, eps_counts = np.unique(eps_int, return_counts=True)
    
    # Weight of each group: count × φ^|k|
    # Actually weight is count × φ^k × typical_input_magnitude
    # Since input is random, just use count × φ^k
    group_weights = eps_counts.astype(float) * np.array([PHI ** abs(int(k)) for k in unique_eps])
    
    # Sort by contribution (descending)
    sorted_idx = np.argsort(-group_weights)
    
    print(f"    ε groups sorted by weight (count × φ^|k|):")
    for i in range(min(10, len(unique_eps))):
        idx = sorted_idx[i]
        print(f"      ε={unique_eps[idx]:>4d}: count={eps_counts[idx]:>8,}, "
              f"weight={group_weights[idx]:>12.1f}")
    
    # Test: use only top-K groups
    print(f"\n    Navigation with top-K groups:")
    for K in [3, 5, 7, 10, 15, 20, len(unique_eps)]:
        K = min(K, len(unique_eps))
        
        # Build W_approx using only top-K ε groups
        top_eps = unique_eps[sorted_idx[:K]]
        mask = np.isin(eps_int, top_eps)
        
        W_approx = W_dec * mask
        Y_approx = X @ W_approx.T
        
        corr = matmul_corr(Y_approx, Y_true)
        coverage = mask.mean()
        
        print(f"      K={K:>3d} groups: coverage={coverage:.1%}, corr={corr:.4f}")


# ============================================================================
# 5. SIGN-LEVEL DECOMPOSITION: THE REAL NAVIGATION
# ============================================================================

def sign_level_navigation(S, u, v, lvl, eps_int, W_dec, name=""):
    """
    The deepest decomposition: matmul as sign-routed magnitude accumulation.
    
    y[j] = Σᵢ S[j,i] × mag[j,i] × x[i]
    
    Rewrite using sign-agreement:
    y[j] = Σ_{i: agree(j,i)} mag × |x[i]| - Σ_{i: disagree(j,i)} mag × |x[i]|
    
    where agree(j,i) means S[j,i] × sign(x[i]) > 0
    
    The sign matrix routes inputs to positive or negative accumulation.
    The magnitude structure scales those accumulations.
    
    Navigation: if we know the NET ROUTING (how many agree vs disagree
    per group), we know the OUTPUT without touching each element.
    """
    print(f"\n  SIGN-LEVEL NAVIGATION ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    np.random.seed(42)
    n_test = 50
    X = np.random.randn(n_test, n).astype(np.float32) * 0.02
    
    Y_true = X @ W_dec.T
    
    # For each output row j and ε-group k:
    # contribution = φ^(u[j]+k) × Σ_{i∈group_k} S[j,i] × φ^v[i] × x[i]
    #              = φ^(u[j]+k) × (pos_sum - neg_sum)
    # where pos_sum = Σ_{i∈group_k, S[j,i]>0} φ^v[i] × x[i]
    #       neg_sum = Σ_{i∈group_k, S[j,i]<0} φ^v[i] × x[i]
    
    # The sign determines routing. But x[i] has its own sign!
    # net contribution = φ^(u[j]+k) × Σ_{i∈group_k} S[j,i] × x̃[i]
    #                  = φ^(u[j]+k) × (agree_sum - disagree_sum)
    
    # KEY INSIGHT: For random x, agree/disagree are 50/50.
    # But for STRUCTURED x (actual hidden states), the agreement
    # pattern IS the information being processed!
    
    # Test with structured input (column of embedding matrix)
    # Use x = unit vector to isolate individual contributions
    
    # Instead, let's measure the SPARSITY of the grouped contributions
    unique_eps = np.unique(eps_int)
    
    # For a sample of output rows, what fraction of groups matter?
    print(f"    Group contribution analysis (5 sample rows):")
    
    x = X[0]
    x_tilde = x * (PHI ** v)
    
    for j_idx, j in enumerate([0, m//4, m//2, 3*m//4, m-1]):
        group_contribs = []
        for k in unique_eps:
            mask = (eps_int[j] == k)
            if not mask.any():
                group_contribs.append(0.0)
                continue
            contrib = PHI ** (u[j] + int(k)) * np.sum(S[j, mask] * x_tilde[mask])
            group_contribs.append(contrib)
        
        group_contribs = np.array(group_contribs)
        total = np.sum(np.abs(group_contribs))
        
        # Sort by |contribution|
        sorted_c = np.sort(np.abs(group_contribs))[::-1]
        cum = np.cumsum(sorted_c) / (total + 1e-30)
        
        # How many groups for 90%, 95%, 99%?
        for threshold in [0.90, 0.95, 0.99]:
            n_needed = np.searchsorted(cum, threshold) + 1
            print(f"      Row {j:>5d}: {threshold:.0%} of |y| from top {n_needed}/{len(unique_eps)} groups")


# ============================================================================
# 6. THE NAVIGATION MAP: PRECOMPUTED ROUTING TABLE
# ============================================================================

def navigation_map(S, u, v, lvl, eps_int, W_dec, name=""):
    """
    Synthesize: can we precompute a ROUTING TABLE that makes
    navigation O(n_groups × something_small)?
    
    The routing table encodes:
    - For each output j and ε-group k: the SIGNED PROJECTION
      of x̃ onto the sign pattern S[j, group_k]
    
    If sign patterns within groups have structure (e.g., they
    project onto a low-dimensional subspace), the routing table
    can be compressed.
    """
    print(f"\n  NAVIGATION MAP ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    unique_eps = np.unique(eps_int)
    
    # For the ε=0 group (the biggest): what's the rank of S[:, group_0]?
    for k in sorted(unique_eps, key=lambda x: -np.sum(eps_int == x))[:5]:
        k_int = int(k)
        # Get columns in this group (using column-mode ε)
        col_eps_mode = np.round(np.median(eps_int, axis=0)).astype(int)
        col_mask = (col_eps_mode == k_int)
        n_cols = col_mask.sum()
        
        if n_cols < 2:
            continue
        
        S_group = S[:, col_mask]  # (m, n_k)
        
        # SVD of S_group
        if n_cols > 10 and m > 10:
            _, s_g, _ = np.linalg.svd(S_group.astype(np.float64), full_matrices=False)
            cum_g = np.cumsum(s_g**2) / np.sum(s_g**2)
            
            r50 = np.searchsorted(cum_g, 0.5) + 1
            r90 = np.searchsorted(cum_g, 0.9) + 1
            r99 = np.searchsorted(cum_g, 0.99) + 1
            
            print(f"    ε={k_int:>3d} group: {n_cols} cols, "
                  f"S_group rank: r50={r50}, r90={r90}, r99={r99}")
    
    # THE SYNTHESIS: What's the minimum information to navigate?
    # 
    # For each output row j:
    #   y[j] = Σ_k φ^(u[j]+k) × <S_row_j_group_k, x̃>
    #
    # If S_row_j_group_k ≈ some low-rank pattern, then
    #   <S_row_j_group_k, x̃> ≈ Σ_r a_r[j,k] × <basis_r, x̃>
    #
    # And <basis_r, x̃> is computed ONCE for all j.
    #
    # Total: n_groups × n_basis × n_test (instead of m × n × n_test)
    
    print(f"\n    SYNTHESIS: Navigation complexity analysis")
    print(f"    Full matmul: O({m} × {n}) = O({m*n:,})")
    print(f"    Grouped: {len(unique_eps)} groups × O({m} × n_k) = O({m*n:,}) (same! just reorganized)")
    print(f"    Navigation (if S_group rank-R):")
    print(f"      Precompute R dot products with x̃: O(R × {n})")
    print(f"      Per output: {len(unique_eps)} groups × R lookups: O({len(unique_eps)} × R)")
    print(f"      Total: O(R × {n} + {m} × {len(unique_eps)} × R)")
    print(f"      For R=50: O({50*n + m*len(unique_eps)*50:,}) vs O({m*n:,})")
    print(f"      Speedup: {m*n / (50*n + m*len(unique_eps)*50):.1f}×")


# ============================================================================
# MAIN
# ============================================================================

def run():
    print("=" * 70)
    print("  NAVIGATION vs INFERENCE")
    print("  'output = navigate(input, structure)' not 'output = input @ weight'")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    for wname in ['q_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        S = W.signs.astype(np.float32)
        u, v, lvl = extract_rank1(W)
        
        m, n = W.shape
        lvl_r1_int = np.round(np.outer(u, v)).astype(np.int32)
        lvl_true = lvl.astype(np.int32)
        eps_int = lvl_true - lvl_r1_int
        
        print(f"\n{'='*70}")
        print(f"  {wname} ({m}×{n})")
        print(f"  ε alphabet: {len(np.unique(eps_int))} symbols, entropy={3.07:.2f} bits")
        print(f"{'='*70}")
        
        # 1. Grouped decomposition
        print(f"\n  1. GROUPED φ-LEVEL DECOMPOSITION")
        print(f"  {'─'*50}")
        
        n_test = 50
        X = np.random.randn(n_test, n).astype(np.float32) * 0.02
        Y_true = X @ W_dec.T
        
        y_col, y_exact, t_col, t_exact, n_groups, col_std = \
            grouped_phi_matmul(S, u, v, eps_int, X, n_test)
        
        corr_col = matmul_corr(y_col, Y_true)
        corr_exact = matmul_corr(y_exact, Y_true)
        
        print(f"    Column-approx grouped: corr={corr_col:.4f} ({t_col*1000:.1f}ms)")
        print(f"    Exact grouped:         corr={corr_exact:.4f} ({t_exact*1000:.1f}ms)")
        
        # 2. Dead channels
        dead_channel_analysis(S, lvl, eps_int, X, W_dec, wname)
        
        # 3. Automaton trajectories
        automaton_trajectories(S, lvl, eps_int, X[0], W_dec, wname)
        
        # 4. Dominant group navigation
        dominant_group_navigation(S, u, v, lvl, eps_int, W_dec, wname)
        
        # 5. Sign-level navigation
        sign_level_navigation(S, u, v, lvl, eps_int, W_dec, wname)
        
        # 6. Navigation map
        navigation_map(S, u, v, lvl, eps_int, W_dec, wname)
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  NAVIGATION ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
