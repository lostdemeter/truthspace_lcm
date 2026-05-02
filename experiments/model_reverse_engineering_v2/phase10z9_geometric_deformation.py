"""
Phase 10z9: Geometric Deformation Model

Core idea: ζ IS the reference geometry. Computation = deformation of that reference.

Framework:
  1. Reference manifold M_φ: defined by ζ-structure (global + local curve)
  2. Deformation D(x): input x warps the manifold
  3. Zero-finding: output = where deformed manifold crosses zero

For ζ (static):  zeros are fixed. Finding them = the whole problem.
For tensors (dynamic): zeros MOVE with input. Predicting the movement = replacing attention.

Test problem: (a + b) mod 97
  - Reference: 97-point cycle on M_φ
  - Deformation: input rotates/warps the manifold
  - Output: where the deformed manifold zeros out

If this works: the same framework should extend to any problem a transformer solves,
because the transformer IS computing the deformation of M_φ.

References: F107-111, Doc 270, DC 048
"""

import numpy as np
import json
import os

PHI = (1 + np.sqrt(5)) / 2
TWO_PI = 2 * np.pi

# ═══════════════════════════════════════════════════════════════════════
# PART 1: THE REFERENCE MANIFOLD
# ═══════════════════════════════════════════════════════════════════════
#
# The ζ function defines a reference geometry. For a discrete problem
# with N outcomes, the reference manifold is an N-point cycle embedded
# in φ-space.
#
# Why φ-space? Because:
#   - ζ zeros have φ-structured spacing (F106-107)
#   - Transformer residual streams organize in φ-power laws (F111)
#   - The geometry IS the information (our core hypothesis)
#
# The reference manifold has:
#   GLOBAL structure: the overall shape (smooth counting function)
#   LOCAL structure: the fine corrections (harmonic / GUE)

class ReferenceManifold:
    """The ζ-aligned reference manifold for N-element computation.
    
    This is the STATIC version — the "ideal transformer" at rest.
    No input has been applied. All N positions exist on a φ-curved cycle.
    """
    
    def __init__(self, N):
        self.N = N
        
        # Global curve: embed N points on a circle using φ-spacing
        # ζ zeros aren't equally spaced — they follow the smooth counting
        # function N(T) ≈ (T/2π)ln(T/2πe). For our discrete case,
        # the analog is: position k has angle θ_k on a φ-warped circle.
        #
        # Equal spacing would be θ_k = 2πk/N (flat geometry).
        # φ-warped spacing accounts for "curvature" of the number line.
        self.angles = self._build_global_curve()
        
        # Local curve: each position has a local curvature
        # (how much the manifold bends at that point)
        # For ζ, this is related to ζ'/ζ (the logarithmic derivative)
        # For our discrete case, it's the φ-scaled local spacing
        self.curvatures = self._build_local_curve()
        
        # The φ-coordinate embedding of each position
        self.coordinates = self._build_coordinates()
    
    def _build_global_curve(self):
        """Build the global structure: N points on a φ-warped circle.
        
        The key insight: the ζ counting function N(T) is not linear in T.
        It grows as (T/2π)ln(T/2πe). Our discrete analog uses the
        same logarithmic warping.
        """
        angles = np.zeros(self.N)
        for k in range(self.N):
            # Base angle (equal spacing = flat geometry)
            theta_flat = TWO_PI * k / self.N
            
            # φ-warping: positions near zero are "closer together"
            # (like ζ zeros being denser at low height)
            # This uses the smooth N(T) analog
            if k > 0:
                # Logarithmic warping inspired by N(T) = (T/2π)ln(T/2πe)
                warp = np.log(1 + k / PHI) / np.log(1 + self.N / PHI)
                theta_warped = TWO_PI * warp
            else:
                theta_warped = 0.0
            
            angles[k] = theta_warped
        
        return angles
    
    def _build_local_curve(self):
        """Build the local curvature at each position.
        
        For ζ, local curvature is related to the density of zeros.
        Higher density = more curvature = the manifold bends more.
        
        For discrete problems: local curvature follows φ-power scaling.
        """
        curvatures = np.zeros(self.N)
        for k in range(self.N):
            # GUE-inspired local curvature
            # Near the "origin" (k=0), curvature is highest
            # It decays as 1/φ^(distance from origin)
            dist = min(k, self.N - k)  # circular distance
            curvatures[k] = 1.0 / PHI ** (dist / (self.N / TWO_PI))
        
        return curvatures
    
    def _build_coordinates(self):
        """Embed each position as a φ-coordinate vector.
        
        Each position k gets a vector:
          [cos(θ_k), sin(θ_k), cos(2θ_k), sin(2θ_k), ...]
        
        This is like Fourier embedding, but on the φ-warped angles.
        The number of harmonics = the local "bandwidth" at that position.
        """
        d = 8  # embedding dimension (4 harmonics × 2 for sin/cos)
        coords = np.zeros((self.N, d))
        
        for k in range(self.N):
            theta = self.angles[k]
            for h in range(d // 2):
                # Each harmonic weighted by φ-power
                weight = PHI ** (-h)
                coords[k, 2*h] = weight * np.cos((h + 1) * theta)
                coords[k, 2*h + 1] = weight * np.sin((h + 1) * theta)
        
        return coords


# ═══════════════════════════════════════════════════════════════════════
# PART 2: DEFORMATION OPERATORS
# ═══════════════════════════════════════════════════════════════════════
#
# An input DEFORMS the reference manifold. This is the key operation
# that attention currently computes.
#
# For the ζ analog: computing ζ(1/2 + it) for different t gives
# different "views" of the same static manifold. The value of t
# selects which zero you're near. This is the "Compressor" stage.
#
# For the tensor analog: each input token warps the manifold differently.
# The warping is the "dynamic curvature" from DC 048.
#
# For modular arithmetic: the deformation is ROTATION. Adding a to
# the reference cycle rotates it by a positions. This is the simplest
# possible deformation — and it IS exactly what ζ's argument does
# (winding around the origin).

class Deformation:
    """A geometric deformation of the reference manifold.
    
    Represents how an input value changes the shape of M_φ.
    """
    
    def __init__(self, manifold):
        self.manifold = manifold
        self.N = manifold.N
    
    def rotate(self, value):
        """Apply a rotation deformation (the simplest case).
        
        For modular arithmetic: adding 'value' rotates the cycle.
        For ζ: changing t rotates the argument of ζ(1/2+it).
        
        Returns: deformed angle array
        """
        rotation_angle = TWO_PI * value / self.N
        return self.manifold.angles + rotation_angle
    
    def warp(self, value):
        """Apply a curvature-changing deformation.
        
        This is the INTERESTING case — not just rotation, but
        actual shape change. Like how different inputs to a
        transformer change the local geometry of the residual stream.
        
        The warping follows φ-power laws (self-similar).
        """
        # The deformation amplitude scales with φ
        amplitude = (value / self.N) * (1 / PHI)
        
        # The deformation pattern: each position's curvature changes
        # based on its φ-distance from the "input position"
        warped_curvatures = np.copy(self.manifold.curvatures)
        
        for k in range(self.N):
            # Circular distance from value to position k
            dist = min(abs(k - value), self.N - abs(k - value))
            
            # φ-scaled influence: nearby positions deform more
            influence = PHI ** (-dist / (self.N / (TWO_PI * PHI)))
            
            # The deformation: curvature increases near the input
            warped_curvatures[k] += amplitude * influence
        
        return warped_curvatures
    
    def compose(self, values):
        """Compose multiple deformations (residual accumulation).
        
        This IS the Dirichlet series structure:
          M'_φ = M_φ + D(x₁) + D(x₂) + ...
        
        Each input adds its deformation to the residual.
        The corrections are ADDITIVE — exactly like the residual stream.
        """
        # Start with reference angles
        total_rotation = 0.0
        total_curvature = np.copy(self.manifold.curvatures)
        
        for v in values:
            # Rotation (global deformation)
            total_rotation += TWO_PI * v / self.N
            
            # Curvature change (local deformation)
            amplitude = (v / self.N) * (1 / PHI)
            for k in range(self.N):
                dist = min(abs(k - v), self.N - abs(k - v))
                influence = PHI ** (-dist / (self.N / (TWO_PI * PHI)))
                total_curvature[k] += amplitude * influence
        
        deformed_angles = self.manifold.angles + total_rotation
        return deformed_angles, total_curvature


# ═══════════════════════════════════════════════════════════════════════
# PART 3: ZERO-FINDING
# ═══════════════════════════════════════════════════════════════════════
#
# After deformation, the output = where the deformed manifold "zeros out."
#
# For ζ: this literally means ζ(1/2 + it) = 0.
# For modular arithmetic: this means "which position on the deformed
# cycle is closest to the reference zero position."
#
# The zero-finding has three stages (matching our three curves):
#   1. Global estimate: which region of the cycle? (Compressor)
#   2. Local refinement: which specific position? (Processor)
#   3. Precision: snap to exact answer (Targeter)

class ZeroFinder:
    """Find zeros of the deformed manifold.
    
    The output of the computation = the zero of the deformed M'_φ.
    """
    
    def __init__(self, manifold):
        self.manifold = manifold
        self.N = manifold.N
    
    def find_zero_rotation(self, deformed_angles):
        """For pure rotation: which position is now at angle 0?
        
        This is the EXACT solution for modular arithmetic:
        after rotating by (a+b), position (N - (a+b) mod N) mod N
        is at angle 0.
        
        But we find it GEOMETRICALLY — by searching for the position
        whose deformed angle is closest to 0 (mod 2π).
        """
        # Normalize angles to [0, 2π)
        normalized = deformed_angles % TWO_PI
        
        # Find which position is closest to angle 0 (or 2π)
        # This is "zero hunting" on the deformed manifold
        distances = np.minimum(normalized, TWO_PI - normalized)
        
        return np.argmin(distances)
    
    def find_zero_curvature(self, deformed_angles, deformed_curvatures):
        """For curvature deformation: find the zero using both
        rotation AND curvature information.
        
        Stage 1 (Global): Rotation gives the rough estimate
        Stage 2 (Local): Curvature refinement
        Stage 3 (Precision): Snap to nearest integer position
        """
        # Stage 1: Global estimate from rotation
        normalized = deformed_angles % TWO_PI
        distances = np.minimum(normalized, TWO_PI - normalized)
        
        # Stage 2: Weight by local curvature
        # Higher curvature positions are "more certain" — the manifold
        # bends sharply there, making the zero easier to locate
        curvature_weight = deformed_curvatures / np.sum(deformed_curvatures)
        weighted_distances = distances * (1 - curvature_weight * self.N)
        
        # Stage 3: Find minimum (the zero)
        return np.argmin(np.abs(weighted_distances))
    
    def find_zero_full(self, a, b):
        """Full three-stage zero finding for (a+b) mod N.
        
        Stage 1 (Compressor): Global rotation estimate
        Stage 2 (Processor): Curvature-weighted refinement
        Stage 3 (Targeter): Snap to exact position
        """
        deformer = Deformation(self.manifold)
        
        # Compose deformations for both inputs
        deformed_angles, deformed_curvatures = deformer.compose([a, b])
        
        # Find the zero
        # The "zero" is the position that was originally at the TARGET
        # After rotating by (a+b), position 0 has moved to angle 2π(a+b)/N
        # So the position that is NOW at 0 was originally at N-(a+b)%N
        # Which means the ANSWER is at position (a+b)%N
        
        # But we find this geometrically:
        result = self.find_zero_rotation(deformed_angles)
        
        # The zero-finding gives us which position is at angle 0
        # The ANSWER is actually the position label, not which is at 0
        # After rotating by angle 2π(a+b)/N:
        #   Position k is now at angle θ_k + 2π(a+b)/N
        #   Position k is at angle 0 when θ_k ≈ -2π(a+b)/N (mod 2π)
        #   i.e., k ≈ N - (a+b) mod N
        # So the answer (a+b) mod N = N - result if result > 0, else 0
        # ... but this depends on the warping!
        
        # Actually, let's think about this differently.
        # The reference manifold has position k at angle θ_k.
        # For flat (unwarped) geometry: θ_k = 2πk/N
        # After deforming by a and b: angles shift by 2π(a+b)/N
        # The position whose NEW angle ≈ 0 is:
        #   k where θ_k + 2π(a+b)/N ≈ 0 (mod 2π)
        #   → θ_k ≈ -2π(a+b)/N (mod 2π)
        #   → k ≈ N - (a+b) mod N
        # So if result = N - (a+b)%N, the answer = (N - result) % N
        answer = (self.N - result) % self.N
        
        return answer, result, deformed_angles, deformed_curvatures


# ═══════════════════════════════════════════════════════════════════════
# PART 4: THE EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def test_modular_arithmetic(N=97, n_test=500):
    """Test the geometric deformation model on (a+b) mod N.
    
    This is the simplest possible test:
    - Deformation = rotation
    - Zero-finding = find which position is at angle 0
    - Should get 100% if the geometry is correct
    
    If it DOESN'T get 100%, the warping is wrong — the φ-curved
    angles don't perfectly map to uniform modular positions.
    That's information about the gap between ζ-geometry and flat geometry.
    """
    
    print("=" * 70)
    print("PHASE 10z9: GEOMETRIC DEFORMATION MODEL")
    print("=" * 70)
    print()
    print(f"  Reference manifold: {N}-point φ-warped cycle")
    print(f"  Test problem: (a + b) mod {N}")
    print(f"  Test cases: {n_test}")
    print()
    
    # Build the reference manifold
    manifold = ReferenceManifold(N)
    finder = ZeroFinder(manifold)
    
    # Analyze the reference manifold structure
    print("  REFERENCE MANIFOLD ANALYSIS:")
    print("  ─────────────────────────────")
    
    # Check angle spacing (is it uniform or φ-warped?)
    spacings = np.diff(manifold.angles)
    print(f"    Angle spacings: mean={np.mean(spacings):.6f}, "
          f"std={np.std(spacings):.6f}, "
          f"min={np.min(spacings):.6f}, max={np.max(spacings):.6f}")
    print(f"    Uniform spacing would be: {TWO_PI/N:.6f}")
    print(f"    Warping ratio (max/min): {np.max(spacings)/np.min(spacings):.3f}")
    print()
    
    # Check curvature profile
    print(f"    Curvature: mean={np.mean(manifold.curvatures):.6f}, "
          f"max={np.max(manifold.curvatures):.6f}, "
          f"min={np.min(manifold.curvatures):.6f}")
    print()
    
    # Test with flat geometry first (sanity check)
    print("  TEST 1: FLAT GEOMETRY (uniform angles)")
    print("  ─────────────────────────────────────────")
    
    flat_manifold = ReferenceManifold(N)
    # Override with flat angles
    flat_manifold.angles = np.array([TWO_PI * k / N for k in range(N)])
    flat_finder = ZeroFinder(flat_manifold)
    
    flat_correct = 0
    np.random.seed(42)
    test_pairs = [(np.random.randint(0, N), np.random.randint(0, N)) 
                  for _ in range(n_test)]
    
    for a, b in test_pairs:
        expected = (a + b) % N
        predicted, _, _, _ = flat_finder.find_zero_full(a, b)
        if predicted == expected:
            flat_correct += 1
    
    print(f"    Accuracy: {flat_correct}/{n_test} ({100*flat_correct/n_test:.1f}%)")
    print()
    
    # Test with φ-warped geometry
    print("  TEST 2: φ-WARPED GEOMETRY")
    print("  ──────────────────────────")
    
    warped_correct = 0
    warped_errors = []
    
    for a, b in test_pairs:
        expected = (a + b) % N
        predicted, raw_pos, def_angles, def_curvatures = finder.find_zero_full(a, b)
        if predicted == expected:
            warped_correct += 1
        else:
            err = min(abs(predicted - expected), N - abs(predicted - expected))
            warped_errors.append({
                'a': int(a), 'b': int(b),
                'expected': int(expected), 'predicted': int(predicted),
                'circular_error': int(err),
            })
    
    print(f"    Accuracy: {warped_correct}/{n_test} ({100*warped_correct/n_test:.1f}%)")
    
    if warped_errors:
        errs = [e['circular_error'] for e in warped_errors]
        print(f"    Error count: {len(warped_errors)}")
        print(f"    Mean circular error: {np.mean(errs):.2f}")
        print(f"    Max circular error: {max(errs)}")
        
        # Are errors φ-structured?
        unique_errs = sorted(set(errs))
        print(f"    Unique error values: {unique_errs[:10]}")
        
        # Check if errors cluster near φ-powers
        phi_powers = [1, 2, 3, 5, 8, 13, 21, 34, 55]  # Fibonacci ≈ φ-powers
        for p in phi_powers[:5]:
            count = sum(1 for e in errs if e == p)
            if count > 0:
                print(f"    Errors at ±{p}: {count} ({100*count/len(errs):.0f}%)")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════
    # PART 5: THE DEFORMATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    print("  DEFORMATION STRUCTURE ANALYSIS:")
    print("  ────────────────────────────────")
    
    # How does the deformation change with different inputs?
    deformer = Deformation(manifold)
    
    # Measure deformation magnitude for different input values
    deformation_norms = []
    for v in range(N):
        _, curvs = deformer.compose([v])
        delta_curv = curvs - manifold.curvatures
        deformation_norms.append(np.linalg.norm(delta_curv))
    
    deformation_norms = np.array(deformation_norms)
    print(f"    Single-input deformation norms:")
    print(f"      Range: [{deformation_norms.min():.6f}, {deformation_norms.max():.6f}]")
    print(f"      Mean: {deformation_norms.mean():.6f}")
    print(f"      Ratio max/min: {deformation_norms.max()/max(deformation_norms.min(), 1e-10):.3f}")
    print()
    
    # Does deformation compose linearly? (residual structure)
    print("    Composition test (is D(a+b) ≈ D(a) + D(b)?):")
    n_comp_test = 50
    linearity_errors = []
    for _ in range(n_comp_test):
        a, b = np.random.randint(0, N, 2)
        
        # D(a) + D(b) separately
        _, curv_a = deformer.compose([a])
        _, curv_b = deformer.compose([b])
        delta_a = curv_a - manifold.curvatures
        delta_b = curv_b - manifold.curvatures
        composed_separate = manifold.curvatures + delta_a + delta_b
        
        # D(a,b) together
        _, curv_ab = deformer.compose([a, b])
        
        # How close?
        diff = np.linalg.norm(composed_separate - curv_ab)
        linearity_errors.append(diff)
    
    mean_lin_err = np.mean(linearity_errors)
    print(f"      Mean ||D(a)+D(b) - D(a,b)||: {mean_lin_err:.10f}")
    if mean_lin_err < 1e-10:
        print(f"      ✅ Perfectly linear! Deformations compose additively.")
    else:
        print(f"      ⚠️  Non-zero: deformations have nonlinear interaction.")
    print()
    
    # ═══════════════════════════════════════════════════════════════════
    # PART 6: WHAT THE WARPING ERROR TELLS US
    # ═══════════════════════════════════════════════════════════════════
    
    if warped_correct < n_test:
        print("  THE φ-WARPING GAP:")
        print("  ───────────────────")
        print(f"    Flat geometry:   {flat_correct}/{n_test} ({100*flat_correct/n_test:.1f}%)")
        print(f"    φ-warped:        {warped_correct}/{n_test} ({100*warped_correct/n_test:.1f}%)")
        print()
        print("    The gap = the cost of φ-curvature for THIS problem.")
        print("    Modular arithmetic lives on a FLAT cycle (uniform spacing).")
        print("    φ-warping introduces systematic errors because the")
        print("    reference manifold has non-uniform spacing.")
        print()
        print("    This is EXACTLY the point:")
        print("    - ζ geometry is NON-UNIFORM (φ-warped)")
        print("    - Modular arithmetic is UNIFORM")
        print("    - The DEFORMATION from ζ-reference to flat = the computation")
        print("    - A transformer LEARNS this deformation during training")
        print()
        print("    NEXT STEP: Instead of starting with φ-warped reference and")
        print("    trying to make it flat, start with ζ-reference and ask:")
        print("    'What deformation makes this problem solvable?'")
        print("    That deformation IS the 'learned weights' of the transformer.")
    else:
        print("  PERFECT ACCURACY!")
        print("  The φ-warped reference + rotation deformation solves mod arithmetic.")
    
    print()
    
    # Save results
    output = {
        'experiment': 'phase10z9_geometric_deformation',
        'N': N,
        'n_test': n_test,
        'flat_accuracy': flat_correct / n_test,
        'warped_accuracy': warped_correct / n_test,
        'warped_errors': warped_errors[:20],  # first 20 errors
        'deformation_norm_mean': float(deformation_norms.mean()),
        'deformation_norm_range': [float(deformation_norms.min()), 
                                    float(deformation_norms.max())],
        'linearity_error': float(mean_lin_err),
        'angle_spacing_std': float(np.std(spacings)),
        'warping_ratio': float(np.max(spacings)/np.min(spacings)),
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/phase10z9_geometric_deformation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved to results/phase10z9_geometric_deformation.json")
    
    return output


if __name__ == '__main__':
    test_modular_arithmetic()
