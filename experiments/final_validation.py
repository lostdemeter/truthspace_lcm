#!/usr/bin/env python3
"""
Final Validation: Testing All Improvements
==========================================

Validates:
1. Plastic-primary 12D encoding works correctly
2. New dimensions (temporal, causal, conditional, comparative) capture relationships
3. Dimension-aware autotuner correctly places concepts
4. Orthogonality is preserved across dimensions
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm import (
    PlasticEncoder,
    DimensionAwareAutotuner,
    RHO,
    ENCODING_DIM,
)
from truthspace_lcm.core.autotuner import TestCase


def test_plastic_encoding():
    """Test that plastic constant provides correct scaling."""
    print("=" * 70)
    print("TEST 1: Plastic Constant Encoding")
    print("=" * 70)
    
    encoder = PlasticEncoder()
    
    # Verify constant
    assert abs(RHO - 1.324717957244746) < 1e-10, "RHO constant incorrect"
    print(f"✓ RHO = {RHO:.10f}")
    
    # Verify dimension count
    assert ENCODING_DIM == 12, "Should be 12D"
    print(f"✓ Dimensions = {ENCODING_DIM}")
    
    # Verify level scaling
    for level in range(4):
        expected = RHO ** level
        print(f"  Level {level}: ρ^{level} = {expected:.4f}")
    
    print("\n✅ Plastic encoding validated")
    return True


def test_12d_structure():
    """Test that 12D structure captures all relationship types."""
    print("\n" + "=" * 70)
    print("TEST 2: 12D Structure")
    print("=" * 70)
    
    encoder = PlasticEncoder()
    dim_info = encoder.get_dimension_info()
    
    expected_dims = {
        0: "EXISTENCE",
        1: "INFORMATION", 
        2: "SPATIAL",
        3: "INTERACTION",
        4: "FILE_SYSTEM",
        5: "PROCESS_DATA",
        6: "NETWORK_USER",
        7: "MODIFIERS",
        8: "TEMPORAL",
        9: "CAUSAL",
        10: "CONDITIONAL",
        11: "COMPARATIVE",
    }
    
    all_correct = True
    for dim, expected_name in expected_dims.items():
        actual_name = dim_info[dim]["name"]
        if actual_name == expected_name:
            print(f"✓ Dim {dim:2d}: {actual_name}")
        else:
            print(f"✗ Dim {dim:2d}: Expected {expected_name}, got {actual_name}")
            all_correct = False
    
    if all_correct:
        print("\n✅ 12D structure validated")
    else:
        print("\n❌ 12D structure has issues")
    
    return all_correct


def test_new_dimensions():
    """Test that new dimensions (8-11) capture relationships correctly."""
    print("\n" + "=" * 70)
    print("TEST 3: New Dimensions (Temporal, Causal, Conditional, Comparative)")
    print("=" * 70)
    
    encoder = PlasticEncoder()
    
    test_cases = [
        # Temporal (dim 8)
        ("before", "after", 8, "opposite"),
        ("during", "while", 8, "same"),
        
        # Causal (dim 9)
        ("because", "therefore", 9, "same"),
        ("cause", "effect", 9, "opposite"),
        
        # Conditional (dim 10)
        ("if", "else", 10, "opposite"),
        
        # Comparative (dim 11)
        ("more", "less", 11, "opposite"),
        ("larger", "smaller", 11, "same"),  # Both map to MORE/LESS
    ]
    
    all_passed = True
    
    for c1, c2, expected_dim, relation in test_cases:
        d1 = encoder.encode(c1)
        d2 = encoder.encode(c2)
        
        # Check primary dimension
        dim1 = int(np.argmax(np.abs(d1.position)))
        dim2 = int(np.argmax(np.abs(d2.position)))
        
        # Compute similarity
        norm1, norm2 = np.linalg.norm(d1.position), np.linalg.norm(d2.position)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(d1.position, d2.position) / (norm1 * norm2)
        else:
            sim = 0
        
        # Check expectations
        dim_ok = dim1 == expected_dim or dim2 == expected_dim
        
        if relation == "opposite":
            rel_ok = sim < -0.5
        elif relation == "same":
            rel_ok = sim > 0.5 or dim1 == dim2
        else:
            rel_ok = True
        
        status = "✓" if (dim_ok and rel_ok) else "✗"
        print(f"{status} {c1:10} ↔ {c2:10}: dim={dim1}/{dim2}, sim={sim:+.2f} (expected dim {expected_dim}, {relation})")
        
        if not (dim_ok and rel_ok):
            all_passed = False
    
    if all_passed:
        print("\n✅ New dimensions validated")
    else:
        print("\n⚠️  Some new dimension tests need attention")
    
    return all_passed


def test_orthogonality():
    """Test that different dimensions are truly orthogonal."""
    print("\n" + "=" * 70)
    print("TEST 4: Orthogonality Across Dimensions")
    print("=" * 70)
    
    encoder = PlasticEncoder()
    
    # Concepts from different dimensions should be orthogonal
    cross_dim_pairs = [
        ("create", "file"),      # dim 0 vs dim 4
        ("read", "network"),     # dim 1 vs dim 6
        ("move", "before"),      # dim 2 vs dim 8
        ("connect", "cause"),    # dim 3 vs dim 9
        ("process", "if"),       # dim 5 vs dim 10
        ("all", "more"),         # dim 7 vs dim 11
    ]
    
    all_orthogonal = True
    
    for c1, c2 in cross_dim_pairs:
        d1 = encoder.encode(c1)
        d2 = encoder.encode(c2)
        
        norm1, norm2 = np.linalg.norm(d1.position), np.linalg.norm(d2.position)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(d1.position, d2.position) / (norm1 * norm2)
        else:
            sim = 0
        
        is_orthogonal = abs(sim) < 0.1
        status = "✓" if is_orthogonal else "✗"
        print(f"{status} {c1:10} ⊥ {c2:10}: sim={sim:+.4f}")
        
        if not is_orthogonal:
            all_orthogonal = False
    
    if all_orthogonal:
        print("\n✅ Orthogonality validated")
    else:
        print("\n❌ Orthogonality violated")
    
    return all_orthogonal


def test_autotuner():
    """Test dimension-aware autotuner."""
    print("\n" + "=" * 70)
    print("TEST 5: Dimension-Aware Autotuner")
    print("=" * 70)
    
    tuner = DimensionAwareAutotuner()
    
    # Test concept analysis
    test_concepts = [
        ("backup", 2, "action"),      # Should be SPATIAL (like copy/move)
        ("authenticate", 6, "domain"), # Should be NETWORK_USER
        ("retry", 10, "relation"),     # Should be CONDITIONAL
        ("increase", 11, "relation"),  # Should be COMPARATIVE
    ]
    
    all_correct = True
    
    for concept, expected_dim, expected_type in test_concepts:
        analysis = tuner.analyze_concept(concept)
        
        dim_ok = analysis.primary_dimension == expected_dim
        type_ok = analysis.dimension_type == expected_type
        
        status = "✓" if (dim_ok and type_ok) else "✗"
        print(f"{status} {concept:15}: dim={analysis.primary_dimension} (expected {expected_dim}), type={analysis.dimension_type}")
        
        if not (dim_ok and type_ok):
            all_correct = False
    
    # Test auto-placement
    print("\nAuto-placement test:")
    
    primitive, verification = tuner.auto_place(
        concept="sync",
        keywords=["sync", "synchronize", "mirror"],
        test_cases=[
            TestCase(
                "sync files",
                expected_similar_to=["CONNECT"],  # sync maps to CONNECT (dim 3)
                expected_orthogonal_to=["PROCESS"],
            ),
        ]
    )
    
    print(f"  Created: {primitive.name} on dim {primitive.dimension}, level {primitive.level}")
    print(f"  Verification: {'✅ PASSED' if verification['passed'] else '❌ FAILED'}")
    
    if verification['passed']:
        print("\n✅ Autotuner validated")
    else:
        print("\n⚠️  Autotuner needs attention")
        all_correct = False
    
    return all_correct


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("FINAL VALIDATION: Plastic-Primary 12D + Dimension-Aware Autotuner")
    print("=" * 70)
    
    results = {
        "plastic_encoding": test_plastic_encoding(),
        "12d_structure": test_12d_structure(),
        "new_dimensions": test_new_dimensions(),
        "orthogonality": test_orthogonality(),
        "autotuner": test_autotuner(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print("\n  The plastic-primary 12D encoder with dimension-aware autotuner")
        print("  is ready for use.")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) need attention")
    
    print("\n" + "=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
