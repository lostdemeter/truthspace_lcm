"""
Phase 10z9b: Curved Operations on the φ-Manifold

Key insight from v1: applying FLAT rotation to a CURVED manifold fails (1.4%).
You can't walk a straight line on a sphere.

New approach: work ON the curve.

The three stages (Compressor, Processor, Targeter) perform iterative refinement
on the φ-warped manifold:

  1. Compressor (Global): Estimate position from global curve shape
  2. Processor (Local): Refine using local curvature (φ-scaled steps)
  3. Targeter (Precision): Snap to exact position

The operation: "φ-addition" = angle addition on the manifold + inverse lookup.
  θ_target = (θ_a + θ_b) mod 2π
  answer = find position k where θ_k ≈ θ_target

This is the NATURAL operation on the curved manifold. It IS modular addition
when the manifold is flat. On a curved manifold, it gives something different —
and that difference IS the deformation the transformer must learn.

The hierarchy:
  ζ-manifold (ideal, static) → φ-manifold (computable approximation)
  → problem-specific manifold (reached by deformation)
  → deformation = learned weights

References: F107-111, Doc 270, DC 048
"""

import numpy as np
import json
import os

PHI = (1 + np.sqrt(5)) / 2
TWO_PI = 2 * np.pi


# ═══════════════════════════════════════════════════════════════════════
# THE MANIFOLD
# ═══════════════════════════════════════════════════════════════════════

class PhiManifold:
    """A φ-warped cycle with N positions.
    
    Each position k has an angle θ_k on [0, 2π) determined by
    the φ-warped counting function (analog of ζ's N(T)).
    
    The warping parameter controls how much curvature:
      warp=0 → flat (uniform spacing)
      warp=1 → full φ-warping (ζ-like)
    """
    
    def __init__(self, N, warp=1.0):
        self.N = N
        self.warp = warp
        self.angles = self._build_angles()
        self.spacings = np.diff(np.append(self.angles, self.angles[0] + TWO_PI))
    
    def _build_angles(self):
        """Build φ-warped angles.
        
        Uses logarithmic warping inspired by ζ's counting function:
          N(T) ≈ (T/2π)ln(T/2πe) + 7/8
        
        The discrete analog: position k maps to angle via the
        inverse counting function, interpolating between flat and φ-warped.
        """
        angles = np.zeros(self.N)
        for k in range(self.N):
            flat_angle = TWO_PI * k / self.N
            
            if k > 0 and self.warp > 0:
                # φ-warped angle: logarithmic compression near k=0,
                # expansion near k=N
                warped = TWO_PI * np.log(1 + k / PHI) / np.log(1 + self.N / PHI)
                angles[k] = (1 - self.warp) * flat_angle + self.warp * warped
            else:
                angles[k] = flat_angle
        
        return angles
    
    def angle_to_position(self, theta):
        """Map an angle back to the nearest position (inverse lookup).
        
        This is the key operation — the "zero-finding" step.
        On a flat manifold: trivial (multiply by N/2π).
        On a curved manifold: requires the three-stage process.
        """
        theta = theta % TWO_PI
        
        # Find the position whose angle is closest
        diffs = np.abs(self.angles - theta)
        # Handle wraparound
        diffs = np.minimum(diffs, TWO_PI - diffs)
        
        return np.argmin(diffs)
    
    def angle_to_position_threestage(self, theta):
        """Three-stage inverse lookup (Compressor → Processor → Targeter).
        
        This mirrors the transformer's three-zone architecture:
          Compressor: O(1) global estimate
          Processor: iterative local refinement
          Targeter: precision snap
        """
        theta = theta % TWO_PI
        
        # ── STAGE 1: COMPRESSOR (Global Estimate) ──
        # Use the global shape to make an O(1) estimate.
        # On a flat manifold: k_est = round(theta * N / 2π)
        # On a curved manifold: invert the smooth counting function
        #
        # For our log-warped manifold:
        #   θ(k) = 2π × ln(1 + k/φ) / ln(1 + N/φ)
        #   k(θ) = φ × (exp(θ × ln(1+N/φ)/2π) - 1)
        
        if self.warp > 0:
            L = np.log(1 + self.N / PHI)
            k_curved = PHI * (np.exp(theta * L / TWO_PI) - 1)
            k_flat = theta * self.N / TWO_PI
            k_est = (1 - self.warp) * k_flat + self.warp * k_curved
        else:
            k_est = theta * self.N / TWO_PI
        
        k_est = int(round(k_est)) % self.N
        compressor_est = k_est
        
        # ── STAGE 2: PROCESSOR (Local Refinement) ──
        # Iteratively correct using local curvature.
        # Each "layer" computes the angular error and adjusts
        # using the local spacing (like Dirichlet series corrections).
        
        processor_corrections = []
        for iteration in range(5):  # 5 "layers"
            # Angular error at current estimate
            theta_est = self.angles[k_est]
            delta_theta = theta - theta_est
            
            # Handle wraparound
            if delta_theta > np.pi:
                delta_theta -= TWO_PI
            elif delta_theta < -np.pi:
                delta_theta += TWO_PI
            
            if abs(delta_theta) < 1e-12:
                break
            
            # Local spacing at current position
            local_spacing = self.spacings[k_est]
            
            # How many positions to shift (φ-scaled correction)
            dk = delta_theta / local_spacing
            processor_corrections.append(dk)
            
            # Apply correction
            k_est = int(round(k_est + dk)) % self.N
        
        processor_est = k_est
        
        # ── STAGE 3: TARGETER (Precision Snap) ──
        # Check the estimate and its immediate neighbors
        # (like rank-1 Newton correction)
        
        candidates = [(k_est - 1) % self.N, k_est, (k_est + 1) % self.N]
        best_k = k_est
        best_dist = float('inf')
        
        for c in candidates:
            d = abs(self.angles[c] - theta)
            d = min(d, TWO_PI - d)
            if d < best_dist:
                best_dist = d
                best_k = c
        
        return best_k, {
            'compressor': compressor_est,
            'processor': processor_est,
            'targeter': best_k,
            'corrections': processor_corrections,
        }


# ═══════════════════════════════════════════════════════════════════════
# CURVED OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

def phi_addition(manifold, a, b):
    """Addition on the φ-warped manifold.
    
    The geometric operation:
      θ_result = (θ_a + θ_b) mod 2π
      answer = inverse_lookup(θ_result)
    
    On a flat manifold: this IS (a + b) mod N.
    On a curved manifold: this gives the φ-deformed answer.
    The DIFFERENCE between φ-addition and modular addition
    measures the deformation the transformer must learn.
    """
    theta_a = manifold.angles[a]
    theta_b = manifold.angles[b]
    theta_target = (theta_a + theta_b) % TWO_PI
    
    result, stages = manifold.angle_to_position_threestage(theta_target)
    
    return result, theta_target, stages


def phi_addition_walksteps(manifold, a, b):
    """Alternative: walk b steps along the curve from position a.
    
    This preserves the METRIC structure of the manifold:
    each step advances by exactly one position, regardless of
    the angular distance. This IS modular addition on any manifold.
    
    The difference from phi_addition: walking steps always gives
    (a+b) mod N, while angle addition gives the φ-deformed answer.
    """
    return (a + b) % manifold.N


# ═══════════════════════════════════════════════════════════════════════
# THE EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(N=97, n_test=500, seed=42):
    np.random.seed(seed)
    
    print("=" * 70)
    print("PHASE 10z9b: CURVED OPERATIONS ON THE φ-MANIFOLD")
    print("=" * 70)
    print()
    
    # ── Test across warping levels ──
    warp_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    
    test_pairs = [(np.random.randint(0, N), np.random.randint(0, N))
                  for _ in range(n_test)]
    
    print(f"  Test: φ-addition on {N}-point manifold, {n_test} pairs")
    print(f"  Question: does the three-stage process find the geometric answer?")
    print(f"  And: how does the geometric answer differ from (a+b) mod {N}?")
    print()
    
    print(f"  {'Warp':>6} {'3-Stage':>8} {'Direct':>8} "
          f"{'Agree':>8} {'Deformation':>12} {'Max Δk':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*8}")
    
    all_results = {}
    
    for warp in warp_levels:
        manifold = PhiManifold(N, warp=warp)
        
        threestage_correct = 0  # Does 3-stage match direct lookup?
        mod_agreement = 0       # Does φ-addition match (a+b) mod N?
        deformation_sizes = []  # |φ-answer - mod-answer| circular
        
        for a, b in test_pairs:
            mod_answer = (a + b) % N
            
            # φ-addition via three-stage process
            phi_answer, theta_target, stages = phi_addition(manifold, a, b)
            
            # φ-addition via direct lookup (ground truth for geometric answer)
            direct_answer = manifold.angle_to_position(theta_target)
            
            # Three-stage matches direct?
            if phi_answer == direct_answer:
                threestage_correct += 1
            
            # Geometric answer matches modular answer?
            if phi_answer == mod_answer:
                mod_agreement += 1
            
            # Deformation size (circular distance)
            deform = min(abs(phi_answer - mod_answer),
                        N - abs(phi_answer - mod_answer))
            deformation_sizes.append(deform)
        
        mean_deform = np.mean(deformation_sizes)
        max_deform = max(deformation_sizes)
        
        print(f"  {warp:6.2f} {threestage_correct:7d}/{n_test:1d} "
              f"{n_test:7d}/{n_test:1d} "
              f"{mod_agreement:7d}/{n_test:1d} "
              f"{mean_deform:11.2f} {max_deform:8d}")
        
        all_results[f'warp_{warp}'] = {
            'warp': warp,
            'threestage_accuracy': threestage_correct / n_test,
            'mod_agreement': mod_agreement / n_test,
            'mean_deformation': float(mean_deform),
            'max_deformation': int(max_deform),
        }
    
    print()
    
    # ── Detailed analysis at full warping ──
    print("  DETAILED ANALYSIS (warp=1.0):")
    print("  ──────────────────────────────")
    
    manifold = PhiManifold(N, warp=1.0)
    
    # Analyze the three-stage process
    stage_stats = {'compressor_exact': 0, 'processor_exact': 0, 'targeter_exact': 0}
    processor_iterations = []
    
    for a, b in test_pairs[:100]:
        phi_answer, theta_target, stages = phi_addition(manifold, a, b)
        direct = manifold.angle_to_position(theta_target)
        
        if stages['compressor'] == direct:
            stage_stats['compressor_exact'] += 1
        if stages['processor'] == direct:
            stage_stats['processor_exact'] += 1
        if stages['targeter'] == direct:
            stage_stats['targeter_exact'] += 1
        processor_iterations.append(len(stages['corrections']))
    
    n_detail = 100
    print(f"    Compressor alone:  {stage_stats['compressor_exact']}/{n_detail} "
          f"({100*stage_stats['compressor_exact']/n_detail:.0f}%)")
    print(f"    + Processor:       {stage_stats['processor_exact']}/{n_detail} "
          f"({100*stage_stats['processor_exact']/n_detail:.0f}%)")
    print(f"    + Targeter:        {stage_stats['targeter_exact']}/{n_detail} "
          f"({100*stage_stats['targeter_exact']/n_detail:.0f}%)")
    print(f"    Processor iterations: mean={np.mean(processor_iterations):.1f}, "
          f"max={max(processor_iterations)}")
    print()
    
    # ── The deformation map ──
    print("  THE DEFORMATION MAP:")
    print("  ─────────────────────")
    print()
    print("  φ-addition(a,b) vs (a+b) mod 97 — first 10 examples:")
    print(f"  {'a':>4} {'b':>4} {'a+b%97':>7} {'φ-add':>6} {'Δ':>4}")
    print(f"  {'─'*4} {'─'*4} {'─'*7} {'─'*6} {'─'*4}")
    
    deformation_map = {}
    manifold = PhiManifold(N, warp=1.0)
    
    for i, (a, b) in enumerate(test_pairs[:10]):
        mod_ans = (a + b) % N
        phi_ans, _, _ = phi_addition(manifold, a, b)
        delta = phi_ans - mod_ans
        if delta > N // 2:
            delta -= N
        elif delta < -N // 2:
            delta += N
        print(f"  {a:4d} {b:4d} {mod_ans:7d} {phi_ans:6d} {delta:+4d}")
    
    # Full deformation histogram
    deltas = []
    for a, b in test_pairs:
        mod_ans = (a + b) % N
        phi_ans, _, _ = phi_addition(manifold, a, b)
        delta = phi_ans - mod_ans
        if delta > N // 2:
            delta -= N
        elif delta < -N // 2:
            delta += N
        deltas.append(delta)
    
    print()
    print(f"  Deformation statistics (Δ = φ_answer - mod_answer):")
    print(f"    Mean: {np.mean(deltas):+.2f}")
    print(f"    Std:  {np.std(deltas):.2f}")
    print(f"    Exact agreement (Δ=0): {sum(1 for d in deltas if d==0)}/{n_test} "
          f"({100*sum(1 for d in deltas if d==0)/n_test:.1f}%)")
    print()
    
    # Is the deformation φ-structured?
    abs_deltas = [abs(d) for d in deltas if d != 0]
    if abs_deltas:
        # Check if deformation sizes cluster near φ-powers
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        print(f"  φ-structure in deformation sizes:")
        for f in fib:
            count = sum(1 for d in abs_deltas if d == f)
            pct = 100 * count / len(abs_deltas) if abs_deltas else 0
            if count > 0:
                print(f"    |Δ|={f}: {count} ({pct:.1f}%)")
    
    print()
    
    # ── Key insight ──
    print("  INTERPRETATION:")
    print("  ─────────────────")
    print()
    print("  The φ-warped manifold defines its OWN addition operation.")
    print("  This 'φ-addition' ≠ modular addition (unless warp=0).")
    print()
    print("  The DEFORMATION between them:")
    print(f"    D(a,b) = φ_add(a,b) - mod_add(a,b)")
    print(f"    This is what the transformer learns during training.")
    print(f"    It maps the NATURAL geometry (φ-curved) to the TARGET")
    print(f"    computation (flat modular arithmetic).")
    print()
    print("  For the ζ-manifold (ideal reference):")
    print("    - ζ-addition IS the natural operation")
    print("    - No deformation needed for ζ's own zeros")
    print("    - Deformation to OTHER problems = learned weights")
    print()
    print("  NEXT: Can we express D(a,b) as a φ-geometric function?")
    print("  If so: D IS the transformer's computation, expressed")
    print("  without attention, without weights, pure geometry.")
    
    # Save results
    output = {
        'experiment': 'phase10z9b_curved_operations',
        'N': N,
        'n_test': n_test,
        'warp_results': all_results,
        'deformation_mean': float(np.mean(deltas)),
        'deformation_std': float(np.std(deltas)),
        'stage_accuracy': {k: v/n_detail for k, v in stage_stats.items()},
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/phase10z9b_curved_operations.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"  Saved to results/phase10z9b_curved_operations.json")
    
    return output


if __name__ == '__main__':
    run_experiment()
