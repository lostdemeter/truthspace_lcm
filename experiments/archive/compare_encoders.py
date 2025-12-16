#!/usr/bin/env python3
"""
Compare v1 (φ-8D) vs v2 (ρ-12D) Encoders
=========================================

Test whether plastic-primary 12D encoding provides better semantic separation.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.encoder import PhiEncoder
from truthspace_lcm.core.encoder_v2 import PlasticEncoder


# Test pairs
RELATED_PAIRS = [
    ("file", "directory"),
    ("process", "system"),
    ("read", "write"),
    ("create", "destroy"),
    ("copy", "move"),
    ("compress", "archive"),
    ("search", "find"),
    ("list", "show"),
    ("before", "after"),
    ("if", "then"),
    ("more", "less"),
    ("cause", "effect"),
]

UNRELATED_PAIRS = [
    ("file", "network"),
    ("compress", "ssh"),
    ("create", "search"),
    ("process", "directory"),
    ("read", "compress"),
    ("list", "connect"),
    ("move", "execute"),
    ("write", "find"),
    ("before", "file"),
    ("if", "copy"),
    ("more", "network"),
    ("cause", "delete"),
]


def compute_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 > 0 and norm2 > 0:
        return float(np.dot(v1, v2) / (norm1 * norm2))
    return 0.0


def test_encoder(encoder, name: str, related_pairs, unrelated_pairs):
    """Test an encoder's semantic separation."""
    print(f"\n{'=' * 70}")
    print(f"TESTING: {name}")
    print(f"{'=' * 70}")
    
    related_sims = []
    unrelated_sims = []
    
    print("\nRelated pairs:")
    for c1, c2 in related_pairs:
        try:
            v1 = encoder.encode(c1).position
            v2 = encoder.encode(c2).position
            sim = compute_similarity(v1, v2)
            related_sims.append(sim)
            marker = "✓" if sim > 0.3 or sim < -0.3 else "?"
            print(f"  {marker} {c1:12} ↔ {c2:12}: {sim:+.4f}")
        except Exception as e:
            print(f"  ✗ {c1:12} ↔ {c2:12}: ERROR - {e}")
    
    print("\nUnrelated pairs:")
    for c1, c2 in unrelated_pairs:
        try:
            v1 = encoder.encode(c1).position
            v2 = encoder.encode(c2).position
            sim = compute_similarity(v1, v2)
            unrelated_sims.append(sim)
            marker = "✓" if abs(sim) < 0.3 else "?"
            print(f"  {marker} {c1:12} ↔ {c2:12}: {sim:+.4f}")
        except Exception as e:
            print(f"  ✗ {c1:12} ↔ {c2:12}: ERROR - {e}")
    
    # Statistics
    related_sims = np.array(related_sims)
    unrelated_sims = np.array(unrelated_sims)
    
    print(f"\n{'-' * 70}")
    print("STATISTICS")
    print(f"{'-' * 70}")
    
    print(f"\nRelated pairs:")
    print(f"  Mean |similarity|: {np.mean(np.abs(related_sims)):.4f}")
    print(f"  Std:               {np.std(related_sims):.4f}")
    
    print(f"\nUnrelated pairs:")
    print(f"  Mean |similarity|: {np.mean(np.abs(unrelated_sims)):.4f}")
    print(f"  Std:               {np.std(unrelated_sims):.4f}")
    
    # Separation score: related should have high |sim|, unrelated should have low |sim|
    separation = np.mean(np.abs(related_sims)) - np.mean(np.abs(unrelated_sims))
    
    print(f"\n  SEPARATION SCORE: {separation:.4f}")
    print(f"  (Higher = better discrimination)")
    
    return {
        "related_mean": np.mean(np.abs(related_sims)),
        "unrelated_mean": np.mean(np.abs(unrelated_sims)),
        "separation": separation,
    }


def main():
    print("=" * 70)
    print("ENCODER COMPARISON: φ-8D vs ρ-12D")
    print("=" * 70)
    print("\nHypothesis: Plastic-primary 12D should provide better separation")
    print("because:")
    print("  1. Plastic constant grows slower → finer discrimination")
    print("  2. 12D captures temporal/causal/conditional/comparative relations")
    
    # Initialize encoders
    v1 = PhiEncoder()
    v2 = PlasticEncoder()
    
    # Test with pairs that both can handle (8D primitives)
    basic_related = RELATED_PAIRS[:8]  # First 8 pairs (basic concepts)
    basic_unrelated = UNRELATED_PAIRS[:8]
    
    print("\n" + "=" * 70)
    print("TEST 1: Basic concepts (both encoders can handle)")
    print("=" * 70)
    
    r1 = test_encoder(v1, "v1: φ-primary 8D", basic_related, basic_unrelated)
    r2 = test_encoder(v2, "v2: ρ-primary 12D", basic_related, basic_unrelated)
    
    # Test with extended pairs (v2 only)
    print("\n" + "=" * 70)
    print("TEST 2: Extended concepts (v2 only - new dimensions)")
    print("=" * 70)
    
    extended_related = RELATED_PAIRS[8:]  # Last 4 pairs (temporal/causal/etc)
    extended_unrelated = UNRELATED_PAIRS[8:]
    
    r2_ext = test_encoder(v2, "v2: Extended relations", extended_related, extended_unrelated)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nBasic concepts separation:")
    print(f"  v1 (φ-8D):  {r1['separation']:.4f}")
    print(f"  v2 (ρ-12D): {r2['separation']:.4f}")
    
    improvement = (r2['separation'] - r1['separation']) / (abs(r1['separation']) + 0.001) * 100
    print(f"\n  Improvement: {improvement:+.1f}%")
    
    if r2['separation'] > r1['separation']:
        print("\n  ✅ v2 (ρ-12D) shows BETTER separation")
    else:
        print("\n  ⚠️  v1 (φ-8D) shows better separation (unexpected)")
    
    print(f"\nExtended relations (v2 only):")
    print(f"  Separation: {r2_ext['separation']:.4f}")
    
    if r2_ext['separation'] > 0.3:
        print("  ✅ New dimensions capture meaningful relationships")
    else:
        print("  ⚠️  New dimensions need tuning")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
