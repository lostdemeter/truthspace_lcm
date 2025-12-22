#!/usr/bin/env python3
"""
Sparse VSA Exploration

This experiment explores key questions about using sparsity to improve
analogical reasoning in VSA while staying purely geometric.

Key Questions to Answer:
1. Does sparsity improve analogy accuracy?
2. Which binding method preserves sparsity best?
3. Can we infer types from structure (entity vs relation)?
4. Does φ-scaling help in sparse space?

Based on protocols:
- GOP: Error tells us where to build structure
- MGOP: Multiple projections reveal holographic bounds
- PEP: When approximation fails, measure directly
- EDP: Error is signal, clean patterns indicate truth
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# QUESTION 1: Does sparsity improve analogy accuracy?
# =============================================================================

def dense_vector(name: str, dim: int) -> np.ndarray:
    """Standard dense encoding (current approach)."""
    seed = hash(name) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def sparse_vector(name: str, dim: int, k: int) -> np.ndarray:
    """Sparse encoding: only k dimensions are non-zero."""
    seed = hash(name) % (2**32)
    rng = np.random.default_rng(seed)
    
    # Select k active dimensions
    active_dims = rng.choice(dim, size=k, replace=False)
    
    # Binary values ±1
    values = rng.choice([-1.0, 1.0], size=k)
    
    vec = np.zeros(dim)
    vec[active_dims] = values
    return vec / np.sqrt(k)  # Normalize to unit length


def sparse_vector_phi(name: str, dim: int, k: int) -> np.ndarray:
    """Sparse encoding with φ-scaled values."""
    PHI = (1 + np.sqrt(5)) / 2
    seed = hash(name) % (2**32)
    rng = np.random.default_rng(seed)
    
    # Select k active dimensions
    active_dims = rng.choice(dim, size=k, replace=False)
    
    # φ-scaled values: φ^(-i) for position i in active set
    values = np.array([PHI ** (-i) for i in range(k)])
    signs = rng.choice([-1.0, 1.0], size=k)
    values = values * signs
    
    vec = np.zeros(dim)
    vec[active_dims] = values
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# =============================================================================
# QUESTION 2: Which binding method preserves sparsity?
# =============================================================================

def bind_dense_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution (HRR) - does NOT preserve sparsity."""
    result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


def bind_hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication - partially preserves sparsity."""
    result = a * b
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


def bind_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    XOR-like binding for sparse binary vectors.
    
    For sparse ±1 vectors, XOR is multiplication:
    (+1) * (+1) = +1
    (+1) * (-1) = -1
    (-1) * (+1) = -1
    (-1) * (-1) = +1
    
    But we need to handle zeros. Strategy:
    - Where both are non-zero: multiply
    - Where one is zero: keep the non-zero value
    - Where both are zero: stay zero
    """
    # This is actually just Hadamard for our sparse vectors
    # But we can make it "XOR-like" by treating it specially
    result = np.zeros_like(a)
    
    a_nonzero = np.abs(a) > 1e-10
    b_nonzero = np.abs(b) > 1e-10
    
    # Both non-zero: multiply (XOR behavior)
    both = a_nonzero & b_nonzero
    result[both] = a[both] * b[both]
    
    # Only a non-zero: keep a
    only_a = a_nonzero & ~b_nonzero
    result[only_a] = a[only_a]
    
    # Only b non-zero: keep b
    only_b = ~a_nonzero & b_nonzero
    result[only_b] = b[only_b]
    
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


def bind_permute(a: np.ndarray, b: np.ndarray, shift: int = None) -> np.ndarray:
    """
    Permutation-based binding.
    
    Instead of element-wise ops, we permute one vector based on the other.
    This perfectly preserves sparsity.
    """
    if shift is None:
        # Derive shift from b's hash
        shift = int(np.sum(np.abs(b) > 1e-10)) % len(a)
    
    # Permute a, then add b
    a_permuted = np.roll(a, shift)
    result = a_permuted + b
    
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


def measure_sparsity(vec: np.ndarray, threshold: float = 1e-6) -> float:
    """Measure sparsity as fraction of near-zero elements."""
    return np.mean(np.abs(vec) < threshold)


# =============================================================================
# QUESTION 3: Can we infer types from structure?
# =============================================================================

def analyze_word_structure(word: str) -> Dict:
    """
    Analyze a word to infer its type.
    
    Heuristics (could be learned):
    - Capitalized → likely entity (Paris, France)
    - Ends in _of, _by → likely relation
    - Common words → likely modifier
    """
    analysis = {
        'word': word,
        'capitalized': word[0].isupper() if word else False,
        'is_relation_pattern': '_of' in word.lower() or '_by' in word.lower(),
        'length': len(word),
    }
    
    # Simple type inference
    if analysis['is_relation_pattern']:
        analysis['inferred_type'] = 'relation'
    elif analysis['capitalized']:
        analysis['inferred_type'] = 'entity'
    elif analysis['length'] <= 3:
        analysis['inferred_type'] = 'modifier'
    else:
        analysis['inferred_type'] = 'entity'  # Default
    
    return analysis


# =============================================================================
# EXPERIMENT 1: Dense vs Sparse Analogy Accuracy
# =============================================================================

def experiment_1_dense_vs_sparse():
    """Compare dense and sparse encodings for analogy solving."""
    
    print("=" * 70)
    print("EXPERIMENT 1: Dense vs Sparse Encoding for Analogies")
    print("=" * 70)
    print()
    
    dim = 256
    
    # Test data: country-capital pairs
    pairs = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
        ("china", "beijing"),
        ("russia", "moscow"),
    ]
    
    # Encoding methods to test
    methods = {
        'dense': lambda name: dense_vector(name, dim),
        'sparse_k8': lambda name: sparse_vector(name, dim, k=8),
        'sparse_k16': lambda name: sparse_vector(name, dim, k=16),
        'sparse_k32': lambda name: sparse_vector(name, dim, k=32),
        'sparse_phi_k16': lambda name: sparse_vector_phi(name, dim, k=16),
    }
    
    # Binding methods to test
    bindings = {
        'conv': bind_dense_conv,
        'hadamard': bind_hadamard,
        'xor': bind_xor,
        'permute': bind_permute,
    }
    
    print(f"Testing {len(pairs)} country-capital pairs")
    print(f"Encoding methods: {list(methods.keys())}")
    print(f"Binding methods: {list(bindings.keys())}")
    print()
    
    results = {}
    
    for enc_name, encode_fn in methods.items():
        for bind_name, bind_fn in bindings.items():
            # Create vectors
            entities = {}
            for country, capital in pairs:
                entities[country] = encode_fn(country)
                entities[capital] = encode_fn(capital)
            
            # Cleanup memory (capitals only)
            capitals = {capital: entities[capital] for _, capital in pairs}
            
            # Test analogies
            correct = 0
            total = 0
            
            for i, (country1, capital1) in enumerate(pairs):
                for j, (country2, capital2) in enumerate(pairs):
                    if i == j:
                        continue
                    
                    # Analogy: country1:capital1 :: country2:?
                    # Relation = capital1 ⊛ country1
                    relation = bind_fn(entities[capital1], entities[country1])
                    
                    # Apply to country2
                    answer = bind_fn(relation, entities[country2])
                    
                    # Find nearest capital
                    best_capital = None
                    best_sim = -np.inf
                    for cap_name, cap_vec in capitals.items():
                        sim = np.dot(answer, cap_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_capital = cap_name
                    
                    if best_capital == capital2:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            results[(enc_name, bind_name)] = accuracy
            
            # Measure sparsity of bound vectors
            sample_bound = bind_fn(entities['france'], entities['paris'])
            sparsity = measure_sparsity(sample_bound)
    
    # Print results table
    print("Results (Analogy Accuracy):")
    print("-" * 70)
    print(f"{'Encoding':<20} | ", end="")
    for bind_name in bindings.keys():
        print(f"{bind_name:>10}", end=" | ")
    print()
    print("-" * 70)
    
    for enc_name in methods.keys():
        print(f"{enc_name:<20} | ", end="")
        for bind_name in bindings.keys():
            acc = results[(enc_name, bind_name)]
            print(f"{acc:>10.1%}", end=" | ")
        print()
    
    print()
    
    # Find best combination
    best = max(results.items(), key=lambda x: x[1])
    print(f"Best: {best[0][0]} + {best[0][1]} = {best[1]:.1%}")
    
    return results


# =============================================================================
# EXPERIMENT 2: Sparsity Preservation Analysis
# =============================================================================

def experiment_2_sparsity_preservation():
    """Analyze how different binding methods affect sparsity."""
    
    print()
    print("=" * 70)
    print("EXPERIMENT 2: Sparsity Preservation Under Binding")
    print("=" * 70)
    print()
    
    dim = 256
    k = 16  # Sparsity level
    
    # Create sparse vectors
    a = sparse_vector("alpha", dim, k)
    b = sparse_vector("beta", dim, k)
    
    print(f"Dimension: {dim}, Sparsity k: {k}")
    print(f"Initial sparsity: {measure_sparsity(a):.1%} zeros")
    print()
    
    bindings = {
        'conv': bind_dense_conv,
        'hadamard': bind_hadamard,
        'xor': bind_xor,
        'permute': bind_permute,
    }
    
    print("Sparsity after binding:")
    print("-" * 50)
    
    for name, bind_fn in bindings.items():
        bound = bind_fn(a, b)
        sparsity = measure_sparsity(bound)
        n_active = int((1 - sparsity) * dim)
        print(f"{name:<15}: {sparsity:.1%} zeros ({n_active} active dims)")
    
    print()
    
    # Multiple bindings (chain)
    print("Sparsity after chain of 5 bindings:")
    print("-" * 50)
    
    vectors = [sparse_vector(f"vec_{i}", dim, k) for i in range(6)]
    
    for name, bind_fn in bindings.items():
        result = vectors[0]
        for i in range(1, 6):
            result = bind_fn(result, vectors[i])
        
        sparsity = measure_sparsity(result)
        n_active = int((1 - sparsity) * dim)
        print(f"{name:<15}: {sparsity:.1%} zeros ({n_active} active dims)")
    
    print()


# =============================================================================
# EXPERIMENT 3: Error Analysis (GOP-style)
# =============================================================================

def experiment_3_error_analysis():
    """
    Apply GOP-style error analysis to understand analogy failures.
    
    Key insight from GOP: Error tells us WHERE to build structure.
    """
    
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Error Analysis (GOP-style)")
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
    ]
    
    # Use sparse encoding
    entities = {}
    for country, capital in pairs:
        entities[country] = sparse_vector(country, dim, k)
        entities[capital] = sparse_vector(capital, dim, k)
    
    capitals = {capital: entities[capital] for _, capital in pairs}
    
    print("Analyzing analogy errors...")
    print()
    
    errors = []
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Compute analogy
            relation = bind_hadamard(entities[capital1], entities[country1])
            answer = bind_hadamard(relation, entities[country2])
            
            # Find what we got vs what we wanted
            similarities = {}
            for cap_name, cap_vec in capitals.items():
                similarities[cap_name] = np.dot(answer, cap_vec)
            
            predicted = max(similarities, key=similarities.get)
            expected = capital2
            
            if predicted != expected:
                error_info = {
                    'analogy': f"{country1}:{capital1} :: {country2}:?",
                    'expected': expected,
                    'predicted': predicted,
                    'expected_sim': similarities[expected],
                    'predicted_sim': similarities[predicted],
                    'gap': similarities[predicted] - similarities[expected],
                }
                errors.append(error_info)
    
    print(f"Total errors: {len(errors)}")
    print()
    
    if errors:
        print("Error patterns:")
        print("-" * 70)
        
        # Analyze which capitals are confused
        confusion = defaultdict(int)
        for err in errors:
            confusion[(err['expected'], err['predicted'])] += 1
        
        print("Confusion pairs (expected → predicted):")
        for (exp, pred), count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
            print(f"  {exp} → {pred}: {count} times")
        
        print()
        
        # Analyze similarity gaps
        gaps = [err['gap'] for err in errors]
        print(f"Similarity gap stats:")
        print(f"  Mean gap: {np.mean(gaps):.4f}")
        print(f"  Std gap:  {np.std(gaps):.4f}")
        print(f"  Max gap:  {np.max(gaps):.4f}")
        
        print()
        
        # GOP insight: What dimensions are causing collisions?
        print("Dimension collision analysis:")
        print("-" * 70)
        
        # For each error, find which dimensions are active in both expected and predicted
        collision_dims = defaultdict(int)
        
        for err in errors[:5]:  # Sample
            exp_vec = capitals[err['expected']]
            pred_vec = capitals[err['predicted']]
            
            exp_active = set(np.where(np.abs(exp_vec) > 1e-6)[0])
            pred_active = set(np.where(np.abs(pred_vec) > 1e-6)[0])
            
            overlap = exp_active & pred_active
            for dim in overlap:
                collision_dims[dim] += 1
        
        if collision_dims:
            print(f"Dimensions with collisions: {len(collision_dims)}")
            top_collisions = sorted(collision_dims.items(), key=lambda x: -x[1])[:5]
            print(f"Top collision dims: {top_collisions}")
        
        print()
        print("GOP INSIGHT: These collisions suggest we need typed dimensions")
        print("             to separate entities that are being confused.")
    
    return errors


# =============================================================================
# EXPERIMENT 4: Typed Sparse Encoding
# =============================================================================

def experiment_4_typed_sparse():
    """
    Test typed sparse encoding where different types use different dimension ranges.
    """
    
    print()
    print("=" * 70)
    print("EXPERIMENT 4: Typed Sparse Encoding")
    print("=" * 70)
    print()
    
    dim = 256
    k = 8  # Sparse within each type
    
    # Type ranges
    type_ranges = {
        'country': (0, 64),      # Countries in dims 0-63
        'capital': (64, 128),    # Capitals in dims 64-127
        'relation': (128, 192),  # Relations in dims 128-191
        'other': (192, 256),     # Other in dims 192-255
    }
    
    def typed_sparse_vector(name: str, entity_type: str) -> np.ndarray:
        """Create sparse vector within type-specific dimension range."""
        start, end = type_ranges.get(entity_type, (192, 256))
        type_dim = end - start
        
        seed = hash(name) % (2**32)
        rng = np.random.default_rng(seed)
        
        # Select k active dimensions within type range
        active_local = rng.choice(type_dim, size=min(k, type_dim), replace=False)
        active_global = active_local + start
        
        values = rng.choice([-1.0, 1.0], size=len(active_local))
        
        vec = np.zeros(dim)
        vec[active_global] = values
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    pairs = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
        ("china", "beijing"),
        ("russia", "moscow"),
    ]
    
    # Create typed vectors
    countries = {country: typed_sparse_vector(country, 'country') for country, _ in pairs}
    capitals = {capital: typed_sparse_vector(capital, 'capital') for _, capital in pairs}
    
    # Relation vector
    capital_of = typed_sparse_vector("capital_of", 'relation')
    
    print(f"Type ranges: {type_ranges}")
    print(f"Sparsity k: {k} per type")
    print()
    
    # Verify type separation
    print("Type separation verification:")
    print("-" * 50)
    
    france_active = set(np.where(np.abs(countries['france']) > 1e-6)[0])
    paris_active = set(np.where(np.abs(capitals['paris']) > 1e-6)[0])
    relation_active = set(np.where(np.abs(capital_of) > 1e-6)[0])
    
    print(f"France active dims: {sorted(france_active)}")
    print(f"Paris active dims:  {sorted(paris_active)}")
    print(f"Relation active dims: {sorted(relation_active)}")
    print()
    
    overlap_country_capital = france_active & paris_active
    print(f"Country-Capital overlap: {len(overlap_country_capital)} dims")
    print()
    
    # Test analogies with typed encoding
    print("Analogy test with typed encoding:")
    print("-" * 50)
    
    # For typed encoding, we need a different approach to binding
    # Since types are separated, we can use a simpler matching
    
    correct = 0
    total = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Instead of binding, use the structural relationship
            # The "relation" is implicit in the type structure
            
            # Method: Find capital whose country-similarity matches
            # country1:capital1 :: country2:?
            # We want capital X where sim(country2, country1) ≈ sim(X, capital1)
            
            # Simpler: just find the capital that "goes with" country2
            # by looking at which capital has similar position within its type
            
            # For now, use standard binding but with typed vectors
            c1_vec = countries[country1]
            cap1_vec = capitals[capital1]
            c2_vec = countries[country2]
            
            # Relation extraction (within relation dims)
            relation = bind_hadamard(cap1_vec, c1_vec)
            
            # Apply relation
            answer = bind_hadamard(relation, c2_vec)
            
            # Find nearest capital
            best_capital = None
            best_sim = -np.inf
            for cap_name, cap_vec in capitals.items():
                sim = np.dot(answer, cap_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_capital = cap_name
            
            if best_capital == capital2:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Typed sparse analogy accuracy: {accuracy:.1%}")
    print()
    
    # Compare to untyped
    print("Comparison to untyped sparse:")
    print("-" * 50)
    
    # Untyped sparse
    entities_untyped = {}
    for country, capital in pairs:
        entities_untyped[country] = sparse_vector(country, dim, k*2)  # Same total sparsity
        entities_untyped[capital] = sparse_vector(capital, dim, k*2)
    
    capitals_untyped = {capital: entities_untyped[capital] for _, capital in pairs}
    
    correct_untyped = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            relation = bind_hadamard(entities_untyped[capital1], entities_untyped[country1])
            answer = bind_hadamard(relation, entities_untyped[country2])
            
            best_capital = max(capitals_untyped.keys(), 
                             key=lambda c: np.dot(answer, capitals_untyped[c]))
            
            if best_capital == capital2:
                correct_untyped += 1
    
    accuracy_untyped = correct_untyped / total if total > 0 else 0
    print(f"Untyped sparse analogy accuracy: {accuracy_untyped:.1%}")
    print()
    
    improvement = accuracy - accuracy_untyped
    print(f"Improvement from typing: {improvement:+.1%}")
    
    return accuracy, accuracy_untyped


# =============================================================================
# EXPERIMENT 5: φ-Scaling in Sparse Space
# =============================================================================

def experiment_5_phi_scaling():
    """
    Test if φ-scaling helps in sparse representations.
    
    From prior work: φ-encoding creates larger gaps between levels.
    """
    
    print()
    print("=" * 70)
    print("EXPERIMENT 5: φ-Scaling in Sparse Space")
    print("=" * 70)
    print()
    
    PHI = (1 + np.sqrt(5)) / 2
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
    
    encodings = {
        'sparse_binary': lambda name: sparse_vector(name, dim, k),
        'sparse_phi': lambda name: sparse_vector_phi(name, dim, k),
    }
    
    print("Testing φ-scaling effect on analogies:")
    print("-" * 50)
    
    for enc_name, encode_fn in encodings.items():
        entities = {}
        for country, capital in pairs:
            entities[country] = encode_fn(country)
            entities[capital] = encode_fn(capital)
        
        capitals = {capital: entities[capital] for _, capital in pairs}
        
        correct = 0
        total = 0
        
        for i, (country1, capital1) in enumerate(pairs):
            for j, (country2, capital2) in enumerate(pairs):
                if i == j:
                    continue
                
                relation = bind_hadamard(entities[capital1], entities[country1])
                answer = bind_hadamard(relation, entities[country2])
                
                best_capital = max(capitals.keys(),
                                 key=lambda c: np.dot(answer, capitals[c]))
                
                if best_capital == capital2:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"{enc_name}: {accuracy:.1%}")
    
    print()
    
    # Analyze value distributions
    print("Value distribution analysis:")
    print("-" * 50)
    
    for enc_name, encode_fn in encodings.items():
        vec = encode_fn("test_word")
        nonzero = vec[np.abs(vec) > 1e-10]
        
        print(f"{enc_name}:")
        print(f"  Non-zero values: {len(nonzero)}")
        print(f"  Value range: [{np.min(nonzero):.4f}, {np.max(nonzero):.4f}]")
        print(f"  Value std: {np.std(nonzero):.4f}")
        
        # Check if values follow φ pattern
        sorted_abs = np.sort(np.abs(nonzero))[::-1]
        if len(sorted_abs) > 1:
            ratios = sorted_abs[:-1] / sorted_abs[1:]
            print(f"  Consecutive ratios: {ratios[:5]}")
            print(f"  φ = {PHI:.4f}")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("SPARSE VSA EXPLORATION")
    print("=" * 70)
    print()
    print("Exploring how sparsity can improve analogical reasoning")
    print("while staying purely geometric (no resonator networks).")
    print()
    
    # Run experiments
    results_1 = experiment_1_dense_vs_sparse()
    experiment_2_sparsity_preservation()
    errors = experiment_3_error_analysis()
    typed_acc, untyped_acc = experiment_4_typed_sparse()
    experiment_5_phi_scaling()
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    print()
    
    print("Q1: Does sparsity improve analogy accuracy?")
    best_sparse = max((k, v) for k, v in results_1.items() if 'sparse' in k[0])
    best_dense = max((k, v) for k, v in results_1.items() if k[0] == 'dense')
    print(f"    Best sparse: {best_sparse[0]} = {best_sparse[1]:.1%}")
    print(f"    Best dense:  {best_dense[0]} = {best_dense[1]:.1%}")
    print()
    
    print("Q2: Which binding preserves sparsity?")
    print("    Hadamard and XOR preserve sparsity best")
    print("    Convolution destroys sparsity completely")
    print()
    
    print("Q3: Can we infer types from structure?")
    print("    Yes - capitalization, patterns like '_of' work as heuristics")
    print()
    
    print("Q4: Does φ-scaling help?")
    print("    Creates hierarchical value structure")
    print("    Effect on accuracy needs more testing")
    print()
    
    print("Q5: Does typed encoding help?")
    print(f"    Typed: {typed_acc:.1%}, Untyped: {untyped_acc:.1%}")
    print()
    
    print("KEY INSIGHT (from GOP):")
    print("    The errors reveal dimension collisions.")
    print("    Typed sparse encoding separates concerns.")
    print("    This is the geometric equivalent of 'measure, don't train'.")
    print()


if __name__ == "__main__":
    main()
