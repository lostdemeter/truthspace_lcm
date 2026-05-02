#!/usr/bin/env python3
"""
Parametric Holographic Readout

The weight matrix IS a volume hologram:
  - Active channels = bright fringes (constructive interference)
  - Dead channels = dark fringes (destructive interference, Bragg selectivity)
  - Together = the interference pattern that IS the information
  
In optical holography, readout is parametric:
  - You don't read every pixel
  - You illuminate with a reference beam at angle θ
  - The hologram reconstructs the stored pattern for that angle
  - Parameters: beam angle, wavelength, polarization — NOT pixel coordinates

For the weight matrix:
  - The "reference beam" is the rank-1 envelope (u, v)
  - The input vector x is the illumination
  - The output y is the reconstructed pattern
  - The dead/alive pattern provides BRAGG SELECTIVITY

Key test: can we express y = W @ x as a function of a SMALL number
of holographic parameters of x, rather than 3584 element-wise products?

Holographic parameters of x:
  1. Alignment with reference beam v: α = <x, v> / ||x||
  2. Alignment with dead-channel pattern: how x distributes across dead/alive
  3. Sign agreement: how sign(x) aligns with sign columns of S
  4. Angular decomposition: x projected onto hologram's Bragg planes

If the hologram has K stored patterns (like K multiplexed images),
then K parameters should suffice: {<x, pattern_k>} for k=1..K.
"""

import os, sys, time
import numpy as np

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
    return np.corrcoef(a.flatten(), b.flatten())[0, 1]


# ============================================================================
# 1. HOLOGRAPHIC FRINGE STRUCTURE
# ============================================================================

def fringe_structure(S, lvl, W_dec, name=""):
    """
    Map the holographic fringe pattern: where are bright/dark fringes?
    
    Bright fringe: high |W| = constructive interference
    Dark fringe: low |W| = destructive interference
    
    The fringes form the interference pattern. Their SPACING and 
    ORIENTATION encode the stored patterns.
    """
    print(f"\n  FRINGE STRUCTURE ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    # Define fringe intensity by level
    # Bright: level > threshold, Dark: level < threshold
    for threshold in [-6, -8, -10, -12]:
        bright = (lvl > threshold)
        bright_frac = bright.mean()
        
        # Fringe spacing: average distance between bright cells in each row
        spacings = []
        for j in range(min(500, m)):
            bright_positions = np.where(bright[j])[0]
            if len(bright_positions) > 1:
                diffs = np.diff(bright_positions)
                spacings.extend(diffs.tolist())
        
        if spacings:
            spacings = np.array(spacings)
            mean_spacing = spacings.mean()
            # Is the spacing related to φ?
            LOG_PHI = np.log(PHI)
            phi_spacing = mean_spacing / PHI
            print(f"    Threshold {threshold}: bright={bright_frac:.1%}, "
                  f"mean_spacing={mean_spacing:.2f}, spacing/φ={phi_spacing:.2f}")
    
    # Column-wise bright fraction (is it uniform or structured?)
    bright_8 = (lvl > -8)
    col_bright = bright_8.mean(axis=0)
    print(f"\n    Column bright fraction (lvl>-8):")
    print(f"      min={col_bright.min():.3f}, max={col_bright.max():.3f}, "
          f"std={col_bright.std():.4f}")
    
    # Row-wise bright fraction
    row_bright = bright_8.mean(axis=1)
    print(f"    Row bright fraction (lvl>-8):")
    print(f"      min={row_bright.min():.3f}, max={row_bright.max():.3f}, "
          f"std={row_bright.std():.4f}")
    
    return bright_8


# ============================================================================
# 2. BRAGG SELECTIVITY: Dead Channel Pattern × Input
# ============================================================================

def bragg_selectivity(S, lvl, W_dec, u, v, name=""):
    """
    In volume holography, Bragg selectivity means: only the right
    reference beam angle reconstructs the stored pattern.
    
    Test: does the dead/alive pattern ACT as angular selectivity?
    When x aligns with certain directions, the dead channels
    selectively pass or block information.
    
    Key metric: For different input ANGLES (directions), how does
    the output change? If the hologram has angular selectivity,
    small angle changes should produce large output changes at
    specific angles (resonance) and small changes elsewhere.
    """
    print(f"\n  BRAGG SELECTIVITY ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    # Compute full W SVD (right singular vectors = input directions)
    U_w, s_w, Vt_w = np.linalg.svd(W_dec, full_matrices=False)
    
    np.random.seed(42)
    
    # Test: illuminate with individual singular vectors
    print(f"    Illumination with singular vectors of W:")
    y_norms = []
    for k in range(min(20, n)):
        x_k = Vt_w[k, :].astype(np.float32)
        y_k = x_k @ W_dec.T
        y_norm = np.linalg.norm(y_k)
        y_norms.append(y_norm)
        if k < 10:
            print(f"      v_{k}: ||y|| = {y_norm:.4f} (σ_{k} = {s_w[k]:.4f})")
    
    # The ratio ||y_k||/σ_k should be ~1 if W is well-conditioned
    # But with dead channels, some directions might be amplified/suppressed
    
    # Now test: illuminate with reference beam direction v
    x_ref = v / np.linalg.norm(v)
    y_ref = x_ref.astype(np.float32) @ W_dec.T
    y_ref_norm = np.linalg.norm(y_ref)
    print(f"\n    Reference beam (v direction): ||y|| = {y_ref_norm:.4f}")
    
    # Perturb the reference direction and measure angular sensitivity
    print(f"\n    Angular sensitivity (perturb reference by δ):")
    for delta_deg in [0.1, 0.5, 1, 5, 10, 30, 45, 90]:
        delta_rad = np.radians(delta_deg)
        # Random perpendicular direction
        perp = np.random.randn(n).astype(np.float32)
        perp -= perp.dot(x_ref) * x_ref
        perp /= np.linalg.norm(perp)
        
        x_perturbed = np.cos(delta_rad) * x_ref + np.sin(delta_rad) * perp
        y_perturbed = x_perturbed @ W_dec.T
        
        # How different is the output?
        output_corr = corr(y_ref, y_perturbed)
        norm_ratio = np.linalg.norm(y_perturbed) / y_ref_norm
        
        print(f"      δ={delta_deg:>5.1f}°: output_corr={output_corr:.4f}, "
              f"||y_δ||/||y_ref||={norm_ratio:.4f}")
    
    # Dead channel selectivity: how does the output change when
    # x is aligned vs anti-aligned with the dead channel pattern?
    dead_mask = (lvl < -8).astype(np.float32)
    dead_pattern = dead_mask.mean(axis=0)  # fraction dead per column
    alive_pattern = 1.0 - dead_pattern
    
    # Normalize to unit vector
    dp = dead_pattern / (np.linalg.norm(dead_pattern) + 1e-10)
    ap = alive_pattern / (np.linalg.norm(alive_pattern) + 1e-10)
    
    x_dead_aligned = dp.astype(np.float32)
    x_alive_aligned = ap.astype(np.float32)
    
    y_dead = x_dead_aligned @ W_dec.T
    y_alive = x_alive_aligned @ W_dec.T
    
    print(f"\n    Dead/alive channel selectivity:")
    print(f"      x aligned with dead pattern:  ||y|| = {np.linalg.norm(y_dead):.4f}")
    print(f"      x aligned with alive pattern: ||y|| = {np.linalg.norm(y_alive):.4f}")
    print(f"      Selectivity ratio: {np.linalg.norm(y_alive)/np.linalg.norm(y_dead):.2f}×")
    
    return s_w, Vt_w


# ============================================================================
# 3. MULTIPLEXED PATTERN READOUT
# ============================================================================

def multiplexed_readout(S, lvl, W_dec, u, v, name=""):
    """
    A multiplexed hologram stores multiple images at different angles.
    
    If W stores K patterns, then y = W @ x should be decomposable as:
    y ≈ Σ_{k=1}^{K} α_k(x) × pattern_k
    
    where α_k(x) = <x, direction_k> are the holographic parameters.
    
    The patterns are the LEFT singular vectors of W.
    The directions are the RIGHT singular vectors of W.
    This IS just the SVD: y = Σ σ_k (v_k · x) u_k
    
    But the HOLOGRAPHIC insight is: the dead channels modify which
    patterns can be reconstructed. The dead/alive pattern acts as a
    FILTER on the stored patterns.
    
    Test: how many holographic parameters K are needed to reconstruct
    y to within 99% correlation?
    """
    print(f"\n  MULTIPLEXED PATTERN READOUT ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    np.random.seed(42)
    n_test = 200
    X = np.random.randn(n_test, n).astype(np.float32) * 0.02
    Y_true = X @ W_dec.T  # (n_test, m)
    
    # SVD of W
    U_w, s_w, Vt_w = np.linalg.svd(W_dec, full_matrices=False)
    
    # Holographic parameters: α_k = Vt_w @ x for each test input
    # Y_approx_K = Σ_{k=1}^{K} σ_k × (Vt_w[k] · x) × U_w[:, k]
    #            = X @ Vt_w[:K].T @ diag(s[:K]) @ U_w[:, :K].T
    
    print(f"    Standard SVD truncation:")
    for K in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        if K > min(m, n): break
        Y_K = (X @ Vt_w[:K].T) * s_w[:K][None, :] @ U_w[:, :K].T
        c = corr(Y_K, Y_true)
        print(f"      K={K:>4d}: corr={c:.4f}")
    
    # Now THE KEY TEST: holographic readout using DEAD/ALIVE structure
    # 
    # Hypothesis: the dead channels provide selectivity.
    # Split W into W_alive and W_dead:
    #   W = W_alive + W_dead
    #   y = W_alive @ x + W_dead @ x
    #
    # W_alive has low effective rank (fewer fringe patterns)
    # W_dead has the complementary rank (fills in)
    
    dead_mask = (lvl < -8)  # 80% dead
    alive_mask = ~dead_mask  # 20% alive
    
    W_alive = W_dec * alive_mask
    W_dead = W_dec * dead_mask
    
    # SVD of W_alive (the bright fringes only)
    U_a, s_a, Vt_a = np.linalg.svd(W_alive, full_matrices=False)
    
    print(f"\n    SVD of W_alive (bright fringes, {alive_mask.mean():.0%} of matrix):")
    cum_a = np.cumsum(s_a**2) / np.sum(s_a**2)
    for K in [1, 2, 5, 10, 20, 50, 100, 200]:
        if K > len(s_a): break
        Y_K = (X @ Vt_a[:K].T) * s_a[:K][None, :] @ U_a[:, :K].T
        c_alive = corr(Y_K, X @ W_alive.T)
        c_full = corr(Y_K, Y_true)
        print(f"      K={K:>4d}: alive_corr={c_alive:.4f}, full_corr={c_full:.4f}, "
              f"alive_energy={cum_a[K-1]:.4f}")
    
    # KEY: How does W_alive's rank compare to W's rank?
    cum_w = np.cumsum(s_w**2) / np.sum(s_w**2)
    
    print(f"\n    Rank comparison (50/90/99% energy):")
    for pct in [0.5, 0.9, 0.95, 0.99]:
        r_w = np.searchsorted(cum_w, pct) + 1
        r_a = np.searchsorted(cum_a, pct) + 1
        print(f"      {pct:.0%}: W rank={r_w}, W_alive rank={r_a}, "
              f"ratio={r_a/r_w:.2f}")
    
    # Test: W_alive SVD + dead channel correction
    # Can we reconstruct the dead channel contribution from the alive SVD?
    print(f"\n    Alive SVD + dead correction:")
    Y_alive = X @ W_alive.T
    Y_dead = X @ W_dead.T
    
    # Correlation between alive and dead outputs
    alive_dead_corr = corr(Y_alive, Y_dead)
    print(f"      corr(y_alive, y_dead) = {alive_dead_corr:.4f}")
    
    # Can y_dead be predicted from y_alive?
    # If the hologram is coherent, the dead contribution should be
    # a function of the alive contribution (they're from the same pattern)
    
    # Linear prediction: y_dead ≈ A @ y_alive
    # Use least squares: A = Y_dead.T @ Y_alive @ pinv(Y_alive.T @ Y_alive)
    # But that's m × m — too big. Use projection instead.
    
    # Project y_dead onto the space of y_alive
    # y_dead ≈ Y_dead @ Y_alive.T @ pinv(Y_alive @ Y_alive.T) @ Y_alive
    YaYa = Y_alive @ Y_alive.T  # (n_test, n_test)
    YdYa = Y_dead @ Y_alive.T   # (n_test, n_test)
    
    # Simple scalar: β = <y_dead, y_alive> / <y_alive, y_alive>
    beta = np.sum(Y_dead * Y_alive) / (np.sum(Y_alive * Y_alive) + 1e-30)
    Y_dead_pred = beta * Y_alive
    Y_total_pred = Y_alive + Y_dead_pred
    
    c_scalar = corr(Y_total_pred, Y_true)
    print(f"      Scalar prediction (β={beta:.4f}): corr={c_scalar:.4f}")
    
    # Per-column beta (each output dim has its own scale)
    beta_col = np.sum(Y_dead * Y_alive, axis=0) / (np.sum(Y_alive * Y_alive, axis=0) + 1e-30)
    Y_dead_pred2 = Y_alive * beta_col[None, :]
    Y_total_pred2 = Y_alive + Y_dead_pred2
    
    c_percol = corr(Y_total_pred2, Y_true)
    print(f"      Per-output β: corr={c_percol:.4f}")
    
    # What's the optimal scale factor?
    alpha_opt = np.sum(Y_true * Y_alive) / (np.sum(Y_alive * Y_alive) + 1e-30)
    Y_rescaled = alpha_opt * Y_alive
    c_rescaled = corr(Y_rescaled, Y_true)
    print(f"      Optimal rescale (α={alpha_opt:.4f}): corr={c_rescaled:.4f}")


# ============================================================================
# 4. INTERFERENCE PATTERN DECOMPOSITION
# ============================================================================

def interference_decomposition(S, lvl, W_dec, u, v, name=""):
    """
    Decompose the weight matrix into reference + object beams.
    
    In holography: I = |R + O|² = |R|² + |O|² + R*O + RO*
    The cross terms R*O and RO* are the holographic signal.
    
    For our weight matrix:
    W = S ⊙ φ^(u⊗v + ε)
    
    Reference beam: R = φ^(u⊗v) = rank-1 magnitude envelope
    Object beam: O = the actual weight pattern
    Sign matrix: S = the phase (0 or π)
    
    The interference is: W[j,i] = S[j,i] × R[j,i] × φ^ε[j,i]
    
    The "holographic signal" is S × φ^ε — the DEVIATION from the
    reference beam. This is what encodes the actual information.
    
    Test: can the holographic signal be parametrically reconstructed
    from a small number of "beam angles"?
    """
    print(f"\n  INTERFERENCE DECOMPOSITION ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    # Reference beam magnitude
    R = (PHI ** np.outer(u, v)).astype(np.float64)
    
    # Full weight (decoded)
    W_f64 = W_dec.astype(np.float64)
    
    # Holographic signal: H = W / R = S × φ^ε
    # Avoid division by zero
    H = np.where(R > 1e-30, W_f64 / R, 0.0)
    
    # H should be close to S × φ^ε
    print(f"    Reference beam R = φ^(u⊗v):")
    print(f"      ||R||_F = {np.linalg.norm(R):.2f}")
    print(f"      ||W||_F = {np.linalg.norm(W_f64):.2f}")
    print(f"      ||H||_F = {np.linalg.norm(H):.2f}")
    
    # H is the object beam × sign. What's its structure?
    # If ε is small (3-bit, mostly 0,1,2), then H ≈ S × φ^(small_int)
    # H has only 37-52 × 2 possible values (ε values × signs)
    
    H_unique = len(np.unique(np.round(H * 1000)))
    print(f"      Unique H values (rounded): {H_unique}")
    
    # SVD of H
    U_h, s_h, Vt_h = np.linalg.svd(H, full_matrices=False)
    cum_h = np.cumsum(s_h**2) / np.sum(s_h**2)
    
    print(f"\n    Holographic signal H = W/R SVD:")
    for pct in [0.5, 0.9, 0.95, 0.99]:
        rank = np.searchsorted(cum_h, pct) + 1
        print(f"      {pct:.0%}: rank={rank}")
    
    # Compare: sign-only = S has what rank?
    U_s, s_s, Vt_s = np.linalg.svd(S.astype(np.float64), full_matrices=False)
    cum_s = np.cumsum(s_s**2) / np.sum(s_s**2)
    
    print(f"\n    Sign matrix S SVD:")
    for pct in [0.5, 0.9, 0.95, 0.99]:
        rank = np.searchsorted(cum_s, pct) + 1
        print(f"      {pct:.0%}: rank={rank}")
    
    # KEY: H = S × φ^ε. If ε is mostly 0 or 1, then H ≈ S or H ≈ φ×S
    # The φ^ε factor is a SMALL PERTURBATION of S.
    # So H should have similar rank structure to S.
    
    # Test: does H have LOWER effective rank than S?
    # (because ε constrains H to a smaller set of values)
    
    r90_H = np.searchsorted(cum_h, 0.9) + 1
    r90_S = np.searchsorted(cum_s, 0.9) + 1
    print(f"\n    r90 comparison: H={r90_H}, S={r90_S}")
    print(f"    H is {'more' if r90_H < r90_S else 'less'} compressible than S alone")
    
    # PARAMETRIC TEST: 
    # Can we reconstruct H from K "beam angles" (SVD components of H)?
    # Then: W ≈ R ⊙ (Σ_k σ_k u_k v_k^T) = R ⊙ H_K
    # And: y = (R ⊙ H_K) @ x
    
    np.random.seed(42)
    n_test = 200
    X = np.random.randn(n_test, n).astype(np.float32) * 0.02
    Y_true = X @ W_dec.T
    
    print(f"\n    Parametric holographic readout (H truncated to rank K):")
    for K in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        if K > min(m, n): break
        H_K = (U_h[:, :K] * s_h[:K]) @ Vt_h[:K, :]
        W_K = (R * H_K).astype(np.float32)
        Y_K = X @ W_K.T
        c = corr(Y_K, Y_true)
        print(f"      K={K:>4d}: corr={c:.4f}")
    
    # COMPARE with direct W SVD truncation
    U_w, s_w, Vt_w = np.linalg.svd(W_dec, full_matrices=False)
    print(f"\n    Direct W SVD truncation (for comparison):")
    for K in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        if K > min(m, n): break
        Y_K = (X @ Vt_w[:K].T) * s_w[:K][None, :] @ U_w[:, :K].T
        c = corr(Y_K, Y_true)
        print(f"      K={K:>4d}: corr={c:.4f}")


# ============================================================================
# 5. HOLOGRAPHIC PARAMETER COUNT
# ============================================================================

def holographic_parameter_count(S, lvl, W_dec, u, v, name=""):
    """
    THE SYNTHESIS: How many parameters does the hologram ACTUALLY need?
    
    Full matmul: m × n parameters (element-wise)
    SVD rank-K: K × (m + n + 1) parameters
    
    Holographic readout:
      - Reference beam: 2 vectors (u, v) → m + n parameters
      - Sign matrix: 1 bit per element → m × n bits
      - ε residual: 3 bits per element → 3 × m × n bits
      
    But the HOLOGRAPHIC insight is: the sign + ε together form
    the holographic signal H, and H might have low effective rank
    when measured in the RIGHT basis.
    
    What if we decompose:
      W = R ⊙ H
    where R = φ^(u⊗v) is rank-1, and H = S × φ^ε.
    
    Then: W @ x = (R ⊙ H) @ x = diag(R[:,0:n] ... ) × H × x
    
    Wait, element-wise product doesn't distribute over matmul like that.
    
    BUT: y[j] = Σ_i R[j,i] × H[j,i] × x[i]
              = Σ_i R[j,i] × h[j,i] × x[i]
    
    If we define x̃[i] = R_col[i] × x[i] (column-weighted input):
    WAIT — R[j,i] = φ^(u[j]×v[i]) varies with j too!
    
    So: y[j] = φ^(u[j]×...) × ... — this doesn't separate.
    
    UNLESS we work in log-magnitude space.
    
    Let's just measure: for the holographic signal H, what's the
    minimum parametric description?
    """
    print(f"\n  HOLOGRAPHIC PARAMETER COUNT ({name})")
    print(f"  {'─'*50}")
    
    m, n = S.shape
    
    # Current: full W = m × n × 16 bits = 12.8M × 16 = 205M bits
    full_bits = m * n * 16
    
    # φ-encoded: sign (1 bit) + level (needs 3.07 bits for ε after rank-1)
    # + 2 vectors for rank-1 (m + n floats)
    phi_bits = m * n * (1 + 3.07) + (m + n) * 16
    
    # SVD rank-K: K × (m + n + 1) × 16 bits
    U_w, s_w, Vt_w = np.linalg.svd(W_dec, full_matrices=False)
    cum_w = np.cumsum(s_w**2) / np.sum(s_w**2)
    
    np.random.seed(42)
    n_test = 200
    X = np.random.randn(n_test, n).astype(np.float32) * 0.02
    Y_true = X @ W_dec.T
    
    print(f"    Full matrix: {full_bits/8/1e6:.1f} MB ({m}×{n} × 16 bits)")
    print(f"    φ-encoded:   {phi_bits/8/1e6:.1f} MB (1 bit sign + 3 bits ε + rank-1)")
    
    print(f"\n    SVD rank-K storage vs quality:")
    for K in [10, 20, 50, 100, 200, 500, 1000]:
        if K > min(m, n): break
        svd_bits = K * (m + n + 1) * 16
        Y_K = (X @ Vt_w[:K].T) * s_w[:K][None, :] @ U_w[:, :K].T
        c = corr(Y_K, Y_true)
        speedup = (m * n) / (K * (m + n))
        print(f"      K={K:>4d}: {svd_bits/8/1e6:>5.1f} MB, corr={c:.4f}, "
              f"matmul speedup={speedup:.1f}×")
    
    # The φ-holographic approach:
    # Store: sign (1 bit/elem) + rank-1 (m+n) + ε (3 bits/elem)
    # = 4 bits/element + negligible
    # For inference: need to decode each element = no speedup
    
    # BUT: if we store H = S × φ^ε as low-rank SVD + correction...
    R = (PHI ** np.outer(u, v)).astype(np.float64)
    H = np.where(R > 1e-30, W_dec.astype(np.float64) / R, 0.0)
    
    U_h, s_h, Vt_h = np.linalg.svd(H, full_matrices=False)
    cum_h = np.cumsum(s_h**2) / np.sum(s_h**2)
    
    print(f"\n    Holographic signal H = W/R, SVD rank-K:")
    for K in [10, 20, 50, 100, 200, 500, 1000]:
        if K > len(s_h): break
        h_bits = K * (m + n + 1) * 16 + (m + n) * 16  # SVD of H + rank-1 R
        H_K = (U_h[:, :K] * s_h[:K]) @ Vt_h[:K, :]
        W_K = (R * H_K).astype(np.float32)
        Y_K = X @ W_K.T
        c = corr(Y_K, Y_true)
        print(f"      K={K:>4d}: {h_bits/8/1e6:>5.1f} MB, corr={c:.4f}")
    
    # THE REAL QUESTION: Does factoring through H give better rank?
    r90_W = np.searchsorted(cum_w, 0.9) + 1
    r90_H = np.searchsorted(cum_h, 0.9) + 1
    r99_W = np.searchsorted(cum_w, 0.99) + 1
    r99_H = np.searchsorted(cum_h, 0.99) + 1
    
    print(f"\n    COMPARISON — Does holographic factoring help?")
    print(f"      r90: W={r90_W}, H={r90_H} → {'YES' if r90_H < r90_W else 'NO'} "
          f"({'%.0f%%' % ((1-r90_H/r90_W)*100)} {'better' if r90_H < r90_W else 'worse'})")
    print(f"      r99: W={r99_W}, H={r99_H} → {'YES' if r99_H < r99_W else 'NO'} "
          f"({'%.0f%%' % ((1-r99_H/r99_W)*100)} {'better' if r99_H < r99_W else 'worse'})")


# ============================================================================
# MAIN
# ============================================================================

def run():
    print("=" * 70)
    print("  PARAMETRIC HOLOGRAPHIC READOUT")
    print("  Can we illuminate instead of multiply?")
    print("=" * 70)
    
    layer_dir = os.path.join(MODEL_DIR, 'layer_00')
    np.random.seed(42)
    
    for wname in ['q_proj']:
        W = PhiEncoded.load(os.path.join(layer_dir, f'{wname}.npz'))
        W_dec = W.decode_cached()
        S = W.signs.astype(np.float32)
        u, v, lvl = extract_rank1(W)
        m, n = W.shape
        
        print(f"\n{'='*70}")
        print(f"  {wname} ({m}×{n})")
        print(f"{'='*70}")
        
        # 1. Fringe structure
        bright = fringe_structure(S, lvl, W_dec, wname)
        
        # 2. Bragg selectivity
        s_w, Vt_w = bragg_selectivity(S, lvl, W_dec, u, v, wname)
        
        # 3. Multiplexed readout
        multiplexed_readout(S, lvl, W_dec, u, v, wname)
        
        # 4. Interference decomposition
        interference_decomposition(S, lvl, W_dec, u, v, wname)
        
        # 5. Parameter count
        holographic_parameter_count(S, lvl, W_dec, u, v, wname)
        
        W.clear_cache()
    
    print(f"\n{'='*70}")
    print(f"  HOLOGRAPHIC READOUT ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    run()
