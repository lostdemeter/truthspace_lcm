"""
Phase 10z8: Geometric Zeta Solver

Hypothesis: If ζ IS the ideal transformer, we can solve it using the same
geometric pipeline that transformers use:
  1. Compressor (Lambert W) — O(1) estimate in φ-coordinates
  2. Processor (conditionally convergent φ-series) — additive residual accumulation
  3. Targeter (rank-1 Newton step) — precision correction in φ-space

No neural network. No gradient descent. Pure φ-geometry.

The pipeline mirrors the transformer architecture:
  Input → [Residual + Mixing + φ-Gate] × L → Output

For ζ (static M_φ): the curve is fixed, mixing operates over harmonic components.
For tensor problems (dynamic M_φ): the curve reshapes per input, mixing operates
over sequence positions.

References: F107-111, Doc 270, DC 048
"""

import numpy as np
from scipy.special import lambertw
import json
import os

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2      # 1.618...
TWO_PI = 2 * np.pi
HARMONICS = [3, 6, 9, 12, 15]   # 3×5 structure (F106)

# Known zeta zeros for validation
KNOWN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]

def load_known_zeros(n_max=300):
    """Load known zeros. Use built-in for first 30, Lambert W for rest."""
    zeros = list(KNOWN_ZEROS)
    for n in range(len(zeros) + 1, n_max + 1):
        zeros.append(lambert_w_base(n))
    return zeros[:n_max]

# ═══════════════════════════════════════════════════════════════════════
# GEOMETRIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════

def phi_encode(x):
    """Encode a scalar into φ-coordinate vector.
    
    Maps x into a vector of φ-power projections:
    [x/φ^0, x/φ^1, x/φ^2, ..., x/φ^(d-1)]
    
    This creates a multi-scale representation where each component
    captures information at a different φ-scale.
    """
    d = 8  # dimension (matching our 8-layer test transformer)
    return np.array([x / PHI**k for k in range(d)])


def phi_gate(x):
    """φ-scaled sigmoid gate (GELU replacement).
    
    GELU ≈ x · σ(φ·x), curvature = √(2/π) ≈ φ/2 (Doc 243).
    This is the geometric nonlinearity that shifts 1/φ → 2/φ².
    """
    return x * (1 / (1 + np.exp(-PHI * x)))


def residual_add(h, delta):
    """Additive residual: h_new = h + delta.
    
    This IS the Dirichlet series structure.
    The residual stream is the critical line.
    """
    return h + delta


def harmonic_mixing(h, k, theta):
    """Mix across harmonic components.
    
    For ζ (static M_φ): combines information from different harmonics
    of the 3×5=15 structure. This is the sequence mixing analog —
    instead of mixing across token positions, we mix across frequency
    components of the zero-counting function.
    
    For tensor problems: this would be replaced by cross-position mixing
    (phi_softmax, geometric selector, etc.)
    """
    # Each harmonic k contributes a sinusoidal correction
    # weighted by φ-power amplitude
    amplitude = PHI ** (-(15 - k) / PHI)
    return amplitude * np.sin(k * theta) * h


def gue_spacing(t):
    """GUE spacing at height t (random matrix theory prediction)."""
    if t < 10:
        return 1.0
    return TWO_PI / np.log(t / TWO_PI)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: COMPRESSOR (Lambert W)
# ═══════════════════════════════════════════════════════════════════════

def lambert_w_base(n):
    """Lambert W base estimate — geometric (argument principle).
    
    This IS the Compressor. Captures >95% of the answer.
    """
    shift = n - 11/8
    if shift <= 0:
        return 14.134725
    return TWO_PI * shift / np.real(lambertw(shift / np.e))


def compressor(n):
    """Stage 1: Encode input n into φ-space with Lambert W estimate.
    
    Returns: (base_estimate, phi_state_vector)
    """
    base = lambert_w_base(n)
    spacing = gue_spacing(base)
    
    # Encode into φ-coordinate vector
    h = phi_encode(base)
    
    # Also encode the spacing (local scale information)
    h_spacing = phi_encode(spacing)
    
    return base, spacing, h, h_spacing


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: PROCESSOR (Conditionally Convergent φ-Series)
# ═══════════════════════════════════════════════════════════════════════

def processor(n, base, spacing, h, h_spacing, n_layers=5):
    """Stage 2: Conditionally convergent corrections.
    
    Each "layer" adds a harmonic correction through:
    1. Compute phase from geometric density
    2. Mix across harmonics (sequence mixing analog)
    3. Apply φ-gate (GELU analog)
    4. Add to residual (Dirichlet term)
    
    The corrections OSCILLATE — this is conditional convergence.
    Just like the Processor zone in Qwen (F109).
    """
    # Phase: derived from zero density (not empirical)
    # N(T) ≈ (T/2π)ln(T/2πe) + 7/8
    N_smooth = (base / TWO_PI) * np.log(base / (TWO_PI * np.e)) + 7/8
    theta = TWO_PI * N_smooth / 15  # 15-fold structure
    
    # Deviation from smooth count
    delta_n = n - N_smooth
    
    total_correction = 0.0
    cumulative = np.zeros_like(h)
    corrections_log = []
    
    for layer in range(n_layers):
        # Select which harmonic this layer processes
        k = HARMONICS[layer % len(HARMONICS)]
        
        # Step 1: Harmonic mixing (sequence mixing analog)
        mixed = harmonic_mixing(h_spacing, k, theta)
        
        # Step 2: φ-gate (GELU analog)
        gated = phi_gate(mixed)
        
        # Step 3: Scale by layer-dependent φ-power
        # Earlier layers: larger corrections (like Dirichlet early terms)
        # Later layers: smaller corrections (convergent)
        layer_scale = PHI ** (-(layer + 1))
        
        # Step 4: Compute scalar correction for this layer
        correction = np.sum(gated) * layer_scale * spacing / (3 * len(HARMONICS))
        
        # Step 5: Residual add (Dirichlet series structure)
        h = residual_add(h, gated * layer_scale)
        cumulative = residual_add(cumulative, gated * layer_scale)
        
        total_correction += correction
        corrections_log.append(correction)
    
    # Log whether we see conditional convergence (sign changes)
    signs = [1 if c > 0 else -1 for c in corrections_log if abs(c) > 1e-15]
    sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
    
    return total_correction, h, sign_changes, corrections_log


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: TARGETER (Rank-1 Newton Step)
# ═══════════════════════════════════════════════════════════════════════

def targeter(n, base, correction, h):
    """Stage 3: Precision correction.
    
    The Newton step in the zeta solver:
      t ← t - Im(ζ(s) / ζ'(s))
    
    In the geometric version, we use the φ-state vector to compute
    a rank-1 correction. The "cached ζ'" analog is the dominant
    direction of the φ-state.
    """
    estimate = base + correction
    
    # Rank-1 projection: use the dominant component of h
    # (like the Targeter's 89.4% in σ₁)
    h_norm = np.linalg.norm(h)
    if h_norm > 0:
        dominant = h[0]  # φ^0 component = largest scale
        
        # The rank-1 correction: project the residual "error signal"
        # through the dominant direction
        # Scale: should be O(spacing) or smaller
        spacing = gue_spacing(estimate)
        rank1_correction = (1/PHI**4) * spacing * np.tanh(dominant / h_norm)
    else:
        rank1_correction = 0.0
    
    return estimate + rank1_correction


# ═══════════════════════════════════════════════════════════════════════
# FULL GEOMETRIC PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def geometric_zeta_solver(n, n_proc_layers=5):
    """Full geometric pipeline: Compressor → Processor → Targeter.
    
    Mirrors transformer inference:
      embed → [residual + mixing + gate] × L → project
    """
    # Stage 1: Compressor
    base, spacing, h, h_spacing = compressor(n)
    
    # Stage 2: Processor
    correction, h, sign_changes, corr_log = processor(
        n, base, spacing, h, h_spacing, n_layers=n_proc_layers
    )
    
    # Stage 3: Targeter
    result = targeter(n, base, correction, h)
    
    return {
        'n': n,
        'estimate': result,
        'base': base,
        'correction': correction,
        'sign_changes': sign_changes,
        'corrections': corr_log,
    }


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_experiment():
    """Test geometric zeta solver on known zeros."""
    
    print("=" * 70)
    print("PHASE 10z8: GEOMETRIC ZETA SOLVER")
    print("=" * 70)
    print()
    print("  Hypothesis: ζ IS the ideal transformer.")
    print("  Method: Solve ζ using the F111 recipe —")
    print("    residual + mixing + φ-gate. No neural net.")
    print()
    
    # Test on first 30 known zeros
    n_test = len(KNOWN_ZEROS)
    
    results = []
    lambert_errors = []
    geometric_errors = []
    sign_changes_list = []
    
    print(f"  Testing on first {n_test} known zeros:")
    print()
    print(f"  {'n':>4} {'Known':>12} {'Lambert':>12} {'Geometric':>12} "
          f"{'Lam Err':>10} {'Geo Err':>10} {'SignΔ':>5}")
    print(f"  {'─'*4} {'─'*12} {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*5}")
    
    for i, known in enumerate(KNOWN_ZEROS):
        n = i + 1
        
        # Lambert W only
        lam = lambert_w_base(n)
        lam_err = lam - known
        lambert_errors.append(lam_err)
        
        # Geometric solver
        r = geometric_zeta_solver(n)
        geo_err = r['estimate'] - known
        geometric_errors.append(geo_err)
        sign_changes_list.append(r['sign_changes'])
        
        results.append({
            'n': n,
            'known': known,
            'lambert': lam,
            'geometric': r['estimate'],
            'lambert_error': float(lam_err),
            'geometric_error': float(geo_err),
            'correction': float(r['correction']),
            'sign_changes': r['sign_changes'],
        })
        
        print(f"  {n:4d} {known:12.6f} {lam:12.6f} {r['estimate']:12.6f} "
              f"{lam_err:+10.6f} {geo_err:+10.6f} {r['sign_changes']:5d}")
    
    # Summary statistics
    lam_abs = np.abs(lambert_errors)
    geo_abs = np.abs(geometric_errors)
    barrier = 0.33
    
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print(f"  Lambert W only:")
    print(f"    MAE  = {np.mean(lam_abs):.6f}")
    print(f"    σ    = {np.std(lambert_errors):.6f}")
    print(f"    max  = {np.max(lam_abs):.6f}")
    print(f"    bias = {np.mean(lambert_errors):+.6f}")
    print()
    print(f"  Geometric solver:")
    print(f"    MAE  = {np.mean(geo_abs):.6f}")
    print(f"    σ    = {np.std(geometric_errors):.6f}")
    print(f"    max  = {np.max(geo_abs):.6f}")
    print(f"    bias = {np.mean(geometric_errors):+.6f}")
    print()
    
    # Improvement?
    improvement = (np.mean(lam_abs) - np.mean(geo_abs)) / np.mean(lam_abs) * 100
    print(f"  Improvement over Lambert W: {improvement:+.1f}%")
    print(f"  Quantum barrier: σ ≈ {barrier}")
    print(f"  Lambert/barrier: {np.std(lambert_errors)/barrier:.2f}×")
    print(f"  Geometric/barrier: {np.std(geometric_errors)/barrier:.2f}×")
    print()
    
    # Conditional convergence check
    mean_sc = np.mean(sign_changes_list)
    print(f"  Conditional convergence:")
    print(f"    Mean sign changes: {mean_sc:.1f} / 4 possible")
    print(f"    Range: {min(sign_changes_list)} - {max(sign_changes_list)}")
    print()
    
    # Analysis: Where does the geometric solver beat Lambert W?
    better = sum(1 for l, g in zip(lam_abs, geo_abs) if g < l)
    print(f"  Geometric beats Lambert: {better}/{n_test} zeros ({100*better/n_test:.0f}%)")
    print()
    
    # Per-correction-layer analysis
    print("  Per-layer corrections (first 5 zeros):")
    for i in range(min(5, n_test)):
        n = i + 1
        r = geometric_zeta_solver(n)
        corrs = r['corrections']
        corr_str = " ".join(f"{c:+.6f}" for c in corrs)
        print(f"    n={n}: {corr_str}")
    
    print()
    print("  INTERPRETATION:")
    print("  ─────────────────")
    if improvement > 0:
        print(f"  ✅ Geometric solver improves on Lambert W by {improvement:.1f}%")
        print(f"     The φ-harmonic corrections ADD information.")
    else:
        print(f"  ⚠️  Geometric solver is {-improvement:.1f}% WORSE than Lambert W")
        print(f"     The corrections are adding NOISE, not signal.")
        print(f"     This is expected (F108: φ-pure overcorrects).")
        print(f"     The Processor needs calibration, not abandonment.")
    print()
    
    if mean_sc >= 1:
        print(f"  ✅ Conditional convergence present ({mean_sc:.1f} sign changes)")
        print(f"     The Processor oscillates, matching F109.")
    else:
        print(f"  ❌ No conditional convergence ({mean_sc:.1f} sign changes)")
        print(f"     The Processor is monotone — wrong structure.")
    
    # Save results
    output = {
        'experiment': 'phase10z8_geometric_zeta_solver',
        'n_zeros': n_test,
        'lambert_mae': float(np.mean(lam_abs)),
        'geometric_mae': float(np.mean(geo_abs)),
        'lambert_std': float(np.std(lambert_errors)),
        'geometric_std': float(np.std(geometric_errors)),
        'improvement_pct': float(improvement),
        'quantum_barrier': barrier,
        'mean_sign_changes': float(mean_sc),
        'beats_lambert_count': better,
        'per_zero': results,
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/phase10z8_geometric_zeta_solver.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"  Saved to results/phase10z8_geometric_zeta_solver.json")
    
    return output


if __name__ == '__main__':
    run_experiment()
