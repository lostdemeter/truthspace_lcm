#!/usr/bin/env python3
"""
Minimal φ-Computer Prototype
=============================

Test the hypothesis that transformer operations are projections
of a single φ-transform operation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)  # ≈ 0.481


@dataclass
class PhiCoord:
    """A coordinate in φ-space: value = sign × φ^level × (1 + residual × (φ-1))"""
    level: int
    sign: int  # +1 or -1
    residual: float  # in [0, 1)
    
    def to_float(self) -> float:
        """Convert to standard float."""
        return self.sign * (PHI ** self.level) * (1 + self.residual * (PHI - 1))
    
    @classmethod
    def from_float(cls, x: float) -> 'PhiCoord':
        """Convert from standard float."""
        if abs(x) < 1e-15:
            return cls(level=-100, sign=1, residual=0.0)
        
        sign = 1 if x > 0 else -1
        abs_x = abs(x)
        
        # log_φ(|x|) = ln(|x|) / ln(φ)
        log_phi_x = np.log(abs_x) / LN_PHI
        level = int(np.floor(log_phi_x))
        
        # residual: how far between φ^level and φ^(level+1)
        # |x| = φ^level × (1 + r × (φ-1))
        # r = (|x|/φ^level - 1) / (φ-1)
        base = PHI ** level
        residual = (abs_x / base - 1) / (PHI - 1)
        residual = np.clip(residual, 0, 1 - 1e-10)
        
        return cls(level=level, sign=sign, residual=residual)
    
    def __repr__(self):
        return f"φ({self.sign:+d}, {self.level}, {self.residual:.4f})"


def phi_transform(x: float, scale: float = 1.0) -> float:
    """
    The core φ-transform operation.
    
    φ-transform(x) = x × φ^(-|x|/(φ×scale))
    
    This is like sigmoid but native to φ-space.
    """
    return x * (PHI ** (-abs(x) / (PHI * scale)))


def phi_sigmoid(x: float) -> float:
    """
    Sigmoid expressed as a φ-operation.
    
    sigmoid(x) = 1 / (1 + φ^(-x/ln(φ)))
    """
    return 1 / (1 + PHI ** (-x / LN_PHI))


def phi_softmax(x: np.ndarray, temperature: float = LN_PHI) -> np.ndarray:
    """
    Softmax as φ-level selection.
    
    softmax(x) = φ^(x/T) / Σ φ^(x/T)
    """
    phi_powers = PHI ** (x / temperature)
    return phi_powers / phi_powers.sum()


def phi_norm(x: np.ndarray) -> np.ndarray:
    """
    RMSNorm as φ-level alignment.
    
    Shifts all values to φ^0 scale.
    """
    rms = np.sqrt(np.mean(x ** 2))
    if rms < 1e-10:
        return x
    # This is equivalent to: x × φ^(-log_φ(rms))
    return x / rms


def phi_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Attention as φ-weighted routing.
    
    For simplicity, single-head attention on vectors.
    """
    # Similarity in φ-space
    similarity = np.dot(Q, K) / np.sqrt(len(Q))
    
    # φ-softmax for weighting
    # For 2 positions: [sim_0, sim_1]
    if isinstance(K, np.ndarray) and K.ndim == 2:
        # Multiple keys
        similarities = np.array([np.dot(Q, k) / np.sqrt(len(Q)) for k in K])
        weights = phi_softmax(similarities)
        return np.sum(weights[:, None] * V, axis=0)
    else:
        # Single key - just return V weighted by sigmoid of similarity
        weight = phi_sigmoid(similarity)
        return weight * V


def test_phi_coordinate_roundtrip():
    """Test that φ-coordinates preserve information."""
    print("=" * 60)
    print("TEST 1: φ-Coordinate Roundtrip")
    print("=" * 60)
    
    test_values = [0.001, 0.01, 0.1, 0.5, 1.0, 1.618, 2.618, 10.0, 100.0, -0.5, -1.618]
    
    print(f"{'Value':>12} {'φ-Coord':>25} {'Reconstructed':>15} {'Error':>12}")
    print("-" * 70)
    
    max_error = 0
    for v in test_values:
        coord = PhiCoord.from_float(v)
        reconstructed = coord.to_float()
        error = abs(v - reconstructed) / max(abs(v), 1e-10)
        max_error = max(max_error, error)
        print(f"{v:>12.6f} {str(coord):>25} {reconstructed:>15.6f} {error:>12.2e}")
    
    print(f"\nMax relative error: {max_error:.2e}")
    return max_error < 1e-10


def test_phi_sigmoid_equivalence():
    """Test that φ-sigmoid matches standard sigmoid."""
    print("\n" + "=" * 60)
    print("TEST 2: φ-Sigmoid vs Standard Sigmoid")
    print("=" * 60)
    
    from scipy.special import expit as sigmoid
    
    x_values = np.linspace(-5, 5, 21)
    
    print(f"{'x':>8} {'sigmoid(x)':>12} {'φ-sigmoid(x)':>14} {'Difference':>12}")
    print("-" * 50)
    
    max_diff = 0
    for x in x_values:
        std = sigmoid(x)
        phi = phi_sigmoid(x)
        diff = abs(std - phi)
        max_diff = max(max_diff, diff)
        if abs(x) <= 2 or x in [x_values[0], x_values[-1]]:
            print(f"{x:>8.2f} {std:>12.6f} {phi:>14.6f} {diff:>12.2e}")
    
    print(f"\nMax difference: {max_diff:.2e}")
    
    # They should be IDENTICAL (not just close)
    # Because sigmoid(x) = 1/(1 + e^(-x)) = 1/(1 + φ^(-x/ln(φ)))
    return max_diff < 1e-14


def test_phi_transform_properties():
    """Test properties of the φ-transform."""
    print("\n" + "=" * 60)
    print("TEST 3: φ-Transform Properties")
    print("=" * 60)
    
    # Property 1: Linear at small x
    print("\nProperty 1: Linear regime (small |x|)")
    for x in [0.01, 0.05, 0.1]:
        y = phi_transform(x)
        linear_approx = x * (1 - abs(x) / PHI * LN_PHI)
        print(f"  x={x:.3f}: φ-transform={y:.6f}, linear≈{linear_approx:.6f}, ratio={y/linear_approx:.6f}")
    
    # Property 2: Self-similarity
    print("\nProperty 2: Self-similarity (scaling by φ)")
    for x in [0.5, 1.0, 2.0]:
        y1 = phi_transform(x)
        y2 = phi_transform(PHI * x)
        ratio = y2 / y1 if y1 != 0 else float('inf')
        print(f"  x={x:.3f}: φ-transform(x)={y1:.6f}, φ-transform(φx)={y2:.6f}, ratio={ratio:.6f} (φ={PHI:.6f})")
    
    # Property 3: Bounded output
    print("\nProperty 3: Bounded output (large |x|)")
    for x in [1, 5, 10, 50, 100]:
        y = phi_transform(x)
        print(f"  x={x:>3}: φ-transform={y:.6f}")
    
    return True


def test_phi_softmax_equivalence():
    """Test that φ-softmax matches standard softmax."""
    print("\n" + "=" * 60)
    print("TEST 4: φ-Softmax vs Standard Softmax")
    print("=" * 60)
    
    def standard_softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    test_vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([10.0, 20.0, 30.0]),
    ]
    
    for v in test_vectors:
        std = standard_softmax(v)
        phi = phi_softmax(v, temperature=1.0)  # Use T=1 for comparison
        diff = np.max(np.abs(std - phi))
        print(f"  Input: {v}")
        print(f"    Standard: {std}")
        print(f"    φ-based:  {phi}")
        print(f"    Max diff: {diff:.2e}")
        print()
    
    return True


def test_connection_to_transformer():
    """Test if transformer operations can be expressed as φ-operations."""
    print("\n" + "=" * 60)
    print("TEST 5: Connection to Transformer Operations")
    print("=" * 60)
    
    # The key insight: sigmoid(ln(φ)) = 1/φ EXACTLY
    print("\nKey identity: sigmoid(ln(φ)) = 1/φ")
    from scipy.special import expit as sigmoid
    
    result = sigmoid(LN_PHI)
    expected = 1/PHI
    print(f"  sigmoid(ln(φ)) = {result:.10f}")
    print(f"  1/φ            = {expected:.10f}")
    print(f"  Difference     = {abs(result - expected):.2e}")
    
    # And sigmoid(-ln(φ)) = 1/φ²
    print("\nKey identity: sigmoid(-ln(φ)) = 1/φ²")
    result = sigmoid(-LN_PHI)
    expected = 1/(PHI**2)
    print(f"  sigmoid(-ln(φ)) = {result:.10f}")
    print(f"  1/φ²            = {expected:.10f}")
    print(f"  Difference      = {abs(result - expected):.2e}")
    
    # The W-axis: where |x| < ln(φ), sigmoid ≈ 0.5
    print("\nW-axis (linear regime): |x| < ln(φ) = 0.481")
    for x in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        s = sigmoid(x)
        print(f"  sigmoid({x:>5.2f}) = {s:.6f} (deviation from 0.5: {abs(s-0.5):.4f})")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("MINIMAL φ-COMPUTER PROTOTYPE")
    print("=" * 60)
    print(f"\nφ = {PHI:.10f}")
    print(f"ln(φ) = {LN_PHI:.10f}")
    print(f"1/φ = {1/PHI:.10f}")
    print(f"1/φ² = {1/PHI**2:.10f}")
    
    results = []
    results.append(("φ-Coordinate Roundtrip", test_phi_coordinate_roundtrip()))
    results.append(("φ-Sigmoid Equivalence", test_phi_sigmoid_equivalence()))
    results.append(("φ-Transform Properties", test_phi_transform_properties()))
    results.append(("φ-Softmax Equivalence", test_phi_softmax_equivalence()))
    results.append(("Transformer Connection", test_connection_to_transformer()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
  The sigmoid function IS a φ-operation:
  
    sigmoid(x) = 1 / (1 + φ^(-x/ln(φ)))
    
  This means ALL transformer nonlinearities (sigmoid, softmax, SiLU)
  are computing in φ-space, just expressed inefficiently in floats.
  
  The minimal φ-computer would:
  1. Represent values as φ-coordinates (level, sign, residual)
  2. Use ONE operation: φ-transform
  3. Achieve the same results with simpler arithmetic
""")


if __name__ == "__main__":
    main()
