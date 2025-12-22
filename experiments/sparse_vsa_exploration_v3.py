#!/usr/bin/env python3
"""
Sparse VSA Exploration v3

Key insight from v2: Linear transformation achieves 100% on direct mapping,
but analogies fail because relations aren't invariant across pairs.

The problem: paris - france ≠ berlin - germany (random vectors)

Solution approaches:
1. STRUCTURED relations: Make the relation explicit and shared
2. LEARNED positions: Let positions emerge from co-occurrence (attractor dynamics)
3. HOLOGRAPHIC encoding: Use phase to encode relation type

This experiment tests whether we can make relations invariant.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# APPROACH 1: Explicit Relation Vectors
# =============================================================================

def experiment_explicit_relations():
    """
    Make the relation an explicit, shared vector.
    
    Instead of computing relation = capital - country,
    define relation as a fixed vector that transforms countries to capitals.
    """
    
    print("=" * 70)
    print("APPROACH 1: Explicit Relation Vectors")
    print("=" * 70)
    print()
    
    dim = 256
    k = 16
    
    pairs = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
    ]
    
    # Create a FIXED relation vector for "capital_of"
    # This is the key insight: the relation is DEFINED, not computed
    capital_of = create_sparse_vector("__CAPITAL_OF__", dim, k)
    
    # Countries get random positions
    countries = {c: create_sparse_vector(c, dim, k) for c, _ in pairs}
    
    # Capitals are DEFINED as: capital = country + capital_of
    # This ensures the relation is invariant!
    capitals = {}
    for country, capital_name in pairs:
        capital_vec = countries[country] + capital_of
        capital_vec = capital_vec / np.linalg.norm(capital_vec)
        capitals[capital_name] = capital_vec
    
    print("Relation: capital = country + capital_of (fixed vector)")
    print()
    
    # Test: Can we recover capitals?
    print("Direct recovery test:")
    correct = 0
    for country, expected in pairs:
        predicted_vec = countries[country] + capital_of
        predicted_vec = predicted_vec / np.linalg.norm(predicted_vec)
        
        best = max(capitals.keys(), key=lambda c: np.dot(predicted_vec, capitals[c]))
        match = "✓" if best == expected else "✗"
        print(f"  {match} {country} + capital_of → {best}")
        if best == expected:
            correct += 1
    
    print(f"\nDirect accuracy: {correct}/{len(pairs)} = {correct/len(pairs):.1%}")
    print()
    
    # Test analogies
    print("Analogy test:")
    print("Since relation is fixed, france:paris :: germany:? should work")
    print()
    
    correct = 0
    total = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Extract relation from pair 1
            # relation = capital1 - country1
            # But we KNOW it should be capital_of!
            extracted_relation = capitals[capital1] - countries[country1]
            
            # How similar is extracted to the true relation?
            sim_to_true = np.dot(extracted_relation / np.linalg.norm(extracted_relation), 
                                capital_of)
            
            # Apply extracted relation to country2
            predicted = countries[country2] + extracted_relation
            predicted = predicted / np.linalg.norm(predicted)
            
            best = max(capitals.keys(), key=lambda c: np.dot(predicted, capitals[c]))
            
            if best == capital2:
                correct += 1
            total += 1
    
    print(f"Analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    print()
    
    # Check: Is the extracted relation consistent?
    print("Relation consistency check:")
    relations = []
    for country, capital in pairs:
        rel = capitals[capital] - countries[country]
        rel = rel / np.linalg.norm(rel)
        relations.append(rel)
        sim = np.dot(rel, capital_of)
        print(f"  {country}→{capital} relation · capital_of = {sim:.3f}")
    
    # Pairwise similarity of extracted relations
    print("\nPairwise relation similarities:")
    for i in range(min(3, len(relations))):
        for j in range(i+1, min(4, len(relations))):
            sim = np.dot(relations[i], relations[j])
            print(f"  rel_{i} · rel_{j} = {sim:.3f}")
    
    return correct / total


def create_sparse_vector(name: str, dim: int, k: int) -> np.ndarray:
    """Create sparse vector with k active dimensions."""
    seed = hash(name) % (2**32)
    rng = np.random.default_rng(seed)
    
    active = rng.choice(dim, size=k, replace=False)
    values = rng.choice([-1.0, 1.0], size=k)
    
    vec = np.zeros(dim)
    vec[active] = values
    return vec / np.linalg.norm(vec)


# =============================================================================
# APPROACH 2: Holographic Phase Encoding
# =============================================================================

def experiment_holographic_relations():
    """
    Use complex numbers (magnitude + phase) to encode relations.
    
    From prior work: Phase encodes WHAT KIND of concept.
    
    Idea: 
    - Entities have magnitude (identity)
    - Relations have phase (type of connection)
    - capital = country * e^(i*θ_capital_of)
    """
    
    print()
    print("=" * 70)
    print("APPROACH 2: Holographic Phase Encoding")
    print("=" * 70)
    print()
    
    dim = 64  # Smaller for complex (2x info per dim)
    k = 8
    
    pairs = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
    ]
    
    # Relation phases
    CAPITAL_OF_PHASE = np.pi / 4  # 45 degrees
    
    def create_complex_sparse(name: str) -> np.ndarray:
        """Create sparse complex vector."""
        seed = hash(name) % (2**32)
        rng = np.random.default_rng(seed)
        
        active = rng.choice(dim, size=k, replace=False)
        magnitudes = rng.uniform(0.5, 1.0, size=k)
        phases = rng.uniform(0, 2*np.pi, size=k)
        
        vec = np.zeros(dim, dtype=complex)
        vec[active] = magnitudes * np.exp(1j * phases)
        return vec / np.linalg.norm(vec)
    
    # Countries as complex vectors
    countries = {c: create_complex_sparse(c) for c, _ in pairs}
    
    # Capitals = countries rotated by capital_of phase
    capitals = {}
    for country, capital_name in pairs:
        capital_vec = countries[country] * np.exp(1j * CAPITAL_OF_PHASE)
        capitals[capital_name] = capital_vec
    
    print(f"Relation: capital = country × e^(i×{CAPITAL_OF_PHASE:.3f})")
    print()
    
    def complex_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Complex inner product, take real part."""
        return np.real(np.vdot(a, b))
    
    # Test direct recovery
    print("Direct recovery test:")
    correct = 0
    for country, expected in pairs:
        predicted = countries[country] * np.exp(1j * CAPITAL_OF_PHASE)
        
        best = max(capitals.keys(), key=lambda c: complex_similarity(predicted, capitals[c]))
        match = "✓" if best == expected else "✗"
        print(f"  {match} {country} × e^(iθ) → {best}")
        if best == expected:
            correct += 1
    
    print(f"\nDirect accuracy: {correct}/{len(pairs)} = {correct/len(pairs):.1%}")
    print()
    
    # Test analogies
    print("Analogy test:")
    correct = 0
    total = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Extract phase difference (the relation)
            # capital1 / country1 = e^(i*θ)
            # We need to find θ
            
            # For each active dimension, compute phase difference
            c1 = countries[country1]
            cap1 = capitals[capital1]
            
            # Phase difference where both are active
            active = (np.abs(c1) > 1e-10) & (np.abs(cap1) > 1e-10)
            if np.sum(active) > 0:
                phase_diffs = np.angle(cap1[active]) - np.angle(c1[active])
                avg_phase = np.mean(phase_diffs)
            else:
                avg_phase = CAPITAL_OF_PHASE  # Fallback
            
            # Apply to country2
            predicted = countries[country2] * np.exp(1j * avg_phase)
            
            best = max(capitals.keys(), key=lambda c: complex_similarity(predicted, capitals[c]))
            
            if best == capital2:
                correct += 1
            total += 1
    
    print(f"Analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    
    return correct / total


# =============================================================================
# APPROACH 3: Structured Sparse with Relation Slots
# =============================================================================

def experiment_relation_slots():
    """
    Dedicate specific dimensions to relations.
    
    Structure:
    - Dims 0-127: Entity identity
    - Dims 128-191: Relation type (which relation applies)
    - Dims 192-255: Relation value (the connected entity)
    
    For "paris is capital of france":
    - paris identity in dims 0-127
    - "capital_of" marker in dims 128-191
    - france reference in dims 192-255
    """
    
    print()
    print("=" * 70)
    print("APPROACH 3: Relation Slots")
    print("=" * 70)
    print()
    
    dim = 256
    k = 8  # Per section
    
    # Dimension ranges
    IDENTITY = (0, 128)
    RELATION_TYPE = (128, 192)
    RELATION_VALUE = (192, 256)
    
    pairs = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
    ]
    
    def create_identity(name: str) -> np.ndarray:
        """Create identity vector (dims 0-127)."""
        seed = hash(name) % (2**32)
        rng = np.random.default_rng(seed)
        
        vec = np.zeros(dim)
        start, end = IDENTITY
        active = rng.choice(end - start, size=k, replace=False) + start
        vec[active] = rng.choice([-1.0, 1.0], size=k)
        return vec
    
    def create_relation_marker(relation_name: str) -> np.ndarray:
        """Create relation type marker (dims 128-191)."""
        seed = hash(relation_name) % (2**32)
        rng = np.random.default_rng(seed)
        
        vec = np.zeros(dim)
        start, end = RELATION_TYPE
        active = rng.choice(end - start, size=k, replace=False) + start
        vec[active] = rng.choice([-1.0, 1.0], size=k)
        return vec
    
    def create_relation_value(entity_name: str) -> np.ndarray:
        """Create relation value reference (dims 192-255)."""
        seed = hash(entity_name + "_ref") % (2**32)
        rng = np.random.default_rng(seed)
        
        vec = np.zeros(dim)
        start, end = RELATION_VALUE
        active = rng.choice(end - start, size=k, replace=False) + start
        vec[active] = rng.choice([-1.0, 1.0], size=k)
        return vec
    
    # Create entities
    # Countries: just identity
    countries = {c: create_identity(c) for c, _ in pairs}
    
    # Capitals: identity + relation_type(capital_of) + relation_value(country)
    capital_of_marker = create_relation_marker("capital_of")
    
    capitals = {}
    for country, capital_name in pairs:
        capital_vec = (create_identity(capital_name) + 
                      capital_of_marker + 
                      create_relation_value(country))
        capital_vec = capital_vec / np.linalg.norm(capital_vec)
        capitals[capital_name] = capital_vec
    
    print("Structure:")
    print("  Country: [identity | 0 | 0]")
    print("  Capital: [identity | capital_of | country_ref]")
    print()
    
    # For analogies, we need to:
    # 1. Recognize that capital1 has "capital_of" relation to country1
    # 2. Find entity with "capital_of" relation to country2
    
    print("Testing relation-based lookup:")
    print()
    
    correct = 0
    total = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Query: Find capital with relation to country2
            # Build query vector: [? | capital_of | country2_ref]
            query = capital_of_marker + create_relation_value(country2)
            query = query / np.linalg.norm(query)
            
            # Find best matching capital
            best = max(capitals.keys(), key=lambda c: np.dot(query, capitals[c]))
            
            if best == capital2:
                correct += 1
            total += 1
    
    print(f"Relation-based analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    
    # Simpler test: direct lookup
    print("\nDirect lookup test:")
    correct_direct = 0
    for country, expected_capital in pairs:
        query = capital_of_marker + create_relation_value(country)
        query = query / np.linalg.norm(query)
        
        best = max(capitals.keys(), key=lambda c: np.dot(query, capitals[c]))
        match = "✓" if best == expected_capital else "✗"
        print(f"  {match} capital_of({country}) → {best}")
        if best == expected_capital:
            correct_direct += 1
    
    print(f"\nDirect accuracy: {correct_direct}/{len(pairs)} = {correct_direct/len(pairs):.1%}")
    
    return correct / total


# =============================================================================
# APPROACH 4: Attractor-Based Positions
# =============================================================================

def experiment_attractor_positions():
    """
    Let positions emerge from co-occurrence structure.
    
    Key insight from prior work: Words that appear together attract.
    
    For country-capital pairs:
    - france and paris should be NEAR each other
    - The "capital_of" relation is the DIRECTION from country to capital
    - If all pairs have similar direction, analogies work
    """
    
    print()
    print("=" * 70)
    print("APPROACH 4: Attractor-Based Positions")
    print("=" * 70)
    print()
    
    dim = 256
    
    pairs = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
    ]
    
    # Initialize random positions
    all_entities = list(set([c for c, _ in pairs] + [cap for _, cap in pairs]))
    positions = {e: np.random.randn(dim) for e in all_entities}
    for e in positions:
        positions[e] = positions[e] / np.linalg.norm(positions[e])
    
    print("Attractor dynamics:")
    print("  - Country-capital pairs attract (co-occur)")
    print("  - All pairs should have similar offset direction")
    print()
    
    # Define the target relation direction
    # All capital-country offsets should align with this
    relation_direction = np.random.randn(dim)
    relation_direction = relation_direction / np.linalg.norm(relation_direction)
    
    # Iteratively adjust positions
    n_iterations = 100
    learning_rate = 0.1
    
    for iteration in range(n_iterations):
        # For each pair, adjust so that capital - country aligns with relation_direction
        for country, capital in pairs:
            current_offset = positions[capital] - positions[country]
            current_offset_norm = current_offset / (np.linalg.norm(current_offset) + 1e-10)
            
            # Move capital to align offset with relation_direction
            target_offset = relation_direction * np.linalg.norm(current_offset)
            
            # Gradient: move capital toward target
            positions[capital] = positions[capital] + learning_rate * (
                positions[country] + target_offset - positions[capital]
            )
            positions[capital] = positions[capital] / np.linalg.norm(positions[capital])
    
    # Check alignment
    print("After attractor dynamics:")
    print()
    
    offsets = []
    for country, capital in pairs:
        offset = positions[capital] - positions[country]
        offset = offset / np.linalg.norm(offset)
        offsets.append(offset)
        
        alignment = np.dot(offset, relation_direction)
        print(f"  {country}→{capital} alignment with relation: {alignment:.3f}")
    
    # Pairwise offset similarity
    print("\nOffset similarities (should be high):")
    for i in range(min(3, len(offsets))):
        for j in range(i+1, min(4, len(offsets))):
            sim = np.dot(offsets[i], offsets[j])
            print(f"  offset_{i} · offset_{j} = {sim:.3f}")
    
    # Test analogies
    print("\nAnalogy test:")
    correct = 0
    total = 0
    
    countries = {c: positions[c] for c, _ in pairs}
    capitals = {cap: positions[cap] for _, cap in pairs}
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Extract relation from pair 1
            relation = positions[capital1] - positions[country1]
            
            # Apply to country2
            predicted = positions[country2] + relation
            predicted = predicted / np.linalg.norm(predicted)
            
            best = max(capitals.keys(), key=lambda c: np.dot(predicted, capitals[c]))
            
            if best == capital2:
                correct += 1
            total += 1
    
    print(f"Analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    
    return correct / total


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("SPARSE VSA EXPLORATION v3")
    print("=" * 70)
    print()
    print("Testing approaches to make relations INVARIANT across pairs.")
    print()
    
    acc1 = experiment_explicit_relations()
    acc2 = experiment_holographic_relations()
    acc3 = experiment_relation_slots()
    acc4 = experiment_attractor_positions()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Approach accuracies:")
    print(f"  1. Explicit relations:    {acc1:.1%}")
    print(f"  2. Holographic phase:     {acc2:.1%}")
    print(f"  3. Relation slots:        {acc3:.1%}")
    print(f"  4. Attractor positions:   {acc4:.1%}")
    print()
    
    best = max([
        ("Explicit relations", acc1),
        ("Holographic phase", acc2),
        ("Relation slots", acc3),
        ("Attractor positions", acc4),
    ], key=lambda x: x[1])
    
    print(f"Best approach: {best[0]} ({best[1]:.1%})")
    print()
    
    if best[1] > 0.5:
        print("SUCCESS: Found an approach that works!")
        print("Key insight: Relations must be DEFINED, not computed from random vectors.")
    else:
        print("All approaches struggle with analogies.")
        print("This suggests the problem is fundamental to the random-hash approach.")
        print()
        print("Next steps:")
        print("  - Use learned positions (from data, not random)")
        print("  - Or: Accept that analogies require explicit relation encoding")


if __name__ == "__main__":
    main()
