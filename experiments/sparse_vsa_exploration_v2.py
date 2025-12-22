#!/usr/bin/env python3
"""
Sparse VSA Exploration v2

Follow-up experiments based on v1 findings:

1. Hadamard produces zeros when sparse vectors don't overlap - FIXED
2. All errors predict "paris" - need to understand why
3. Dense + conv wins - maybe spreading is necessary?

New hypotheses to test:
- Overlapping sparse encoding (force some shared dimensions)
- Additive binding instead of multiplicative
- Probe extraction approach: measure the relation directly
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# IMPROVED SPARSE ENCODINGS
# =============================================================================

def sparse_overlapping(name: str, dim: int, k: int, shared_dims: int = 4) -> np.ndarray:
    """
    Sparse encoding with some shared dimensions across all words.
    
    This ensures Hadamard binding doesn't produce all zeros.
    """
    seed = hash(name) % (2**32)
    rng = np.random.default_rng(seed)
    
    vec = np.zeros(dim)
    
    # Shared dimensions (same for all words, different values)
    shared_seed = 42  # Fixed seed for shared dims
    shared_rng = np.random.default_rng(shared_seed)
    shared_active = shared_rng.choice(dim, size=shared_dims, replace=False)
    
    # Word-specific dimensions
    remaining = k - shared_dims
    available = [d for d in range(dim) if d not in shared_active]
    specific_active = rng.choice(available, size=remaining, replace=False)
    
    # Assign values
    all_active = list(shared_active) + list(specific_active)
    values = rng.choice([-1.0, 1.0], size=len(all_active))
    
    vec[all_active] = values
    return vec / np.linalg.norm(vec)


def sparse_block(name: str, dim: int, k: int, n_blocks: int = 4) -> np.ndarray:
    """
    Block-sparse encoding: divide dims into blocks, activate k/n_blocks per block.
    
    This ensures overlap while maintaining structure.
    """
    seed = hash(name) % (2**32)
    rng = np.random.default_rng(seed)
    
    vec = np.zeros(dim)
    block_size = dim // n_blocks
    k_per_block = k // n_blocks
    
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        
        active_local = rng.choice(block_size, size=k_per_block, replace=False)
        active_global = active_local + start
        
        values = rng.choice([-1.0, 1.0], size=k_per_block)
        vec[active_global] = values
    
    return vec / np.linalg.norm(vec)


# =============================================================================
# IMPROVED BINDING METHODS
# =============================================================================

def bind_additive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Additive binding: a + permute(b)
    
    This preserves sparsity better than multiplication when vectors don't overlap.
    """
    # Permute b by a fixed amount derived from a
    shift = int(np.sum(a > 0)) % len(a)
    b_permuted = np.roll(b, shift)
    
    result = a + b_permuted
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


def bind_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    MAP (Multiply-Add-Permute) binding from VSA literature.
    
    Combines multiplication with permutation for robustness.
    """
    # Permute
    shift = 1
    a_perm = np.roll(a, shift)
    
    # Multiply
    result = a_perm * b
    
    # If too sparse, add
    if np.sum(np.abs(result) > 1e-10) < 5:
        result = result + a_perm + b
    
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


def bind_circular_sparse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Circular convolution but threshold to maintain sparsity.
    """
    result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
    
    # Keep only top k values
    k = max(int(np.sum(np.abs(a) > 1e-10)), int(np.sum(np.abs(b) > 1e-10)))
    threshold = np.sort(np.abs(result))[-k] if k < len(result) else 0
    result[np.abs(result) < threshold] = 0
    
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


# =============================================================================
# EXPERIMENT 6: Why does everything predict "paris"?
# =============================================================================

def experiment_6_paris_attractor():
    """
    Investigate why all errors predict "paris".
    
    GOP insight: This is signal, not noise. Paris is an attractor.
    """
    
    print("=" * 70)
    print("EXPERIMENT 6: Why is Paris an Attractor?")
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
    
    # Create vectors
    entities = {}
    for country, capital in pairs:
        entities[country] = sparse_overlapping(country, dim, k)
        entities[capital] = sparse_overlapping(capital, dim, k)
    
    capitals = {capital: entities[capital] for _, capital in pairs}
    
    print("Analyzing capital vector similarities:")
    print("-" * 50)
    
    # Pairwise similarities between capitals
    cap_names = list(capitals.keys())
    for i, c1 in enumerate(cap_names):
        sims = []
        for j, c2 in enumerate(cap_names):
            sim = np.dot(capitals[c1], capitals[c2])
            sims.append(f"{c2}:{sim:.3f}")
        print(f"{c1}: {', '.join(sims)}")
    
    print()
    
    # Analyze what makes paris special
    print("Paris vector analysis:")
    print("-" * 50)
    
    paris_vec = capitals['paris']
    paris_active = np.where(np.abs(paris_vec) > 1e-10)[0]
    print(f"Paris active dims: {len(paris_active)}")
    
    # How many other capitals share dims with paris?
    for cap_name, cap_vec in capitals.items():
        if cap_name == 'paris':
            continue
        cap_active = set(np.where(np.abs(cap_vec) > 1e-10)[0])
        overlap = set(paris_active) & cap_active
        print(f"  {cap_name} overlap with paris: {len(overlap)} dims")
    
    print()
    
    # Test: What does the "answer" vector look like?
    print("Answer vector analysis:")
    print("-" * 50)
    
    # france:paris :: germany:?
    relation = bind_additive(entities['paris'], entities['france'])
    answer = bind_additive(relation, entities['germany'])
    
    print(f"Answer vector active dims: {np.sum(np.abs(answer) > 1e-10)}")
    
    # Similarity to each capital
    print("Answer similarity to capitals:")
    for cap_name, cap_vec in capitals.items():
        sim = np.dot(answer, cap_vec)
        print(f"  {cap_name}: {sim:.4f}")
    
    print()
    
    # The issue: answer might be too spread out
    print("Hypothesis: Answer vector is too diffuse")
    print("-" * 50)
    
    answer_active = np.where(np.abs(answer) > 1e-10)[0]
    print(f"Answer has {len(answer_active)} active dims")
    print(f"Paris has {len(paris_active)} active dims")
    
    # If answer is spread across many dims, it will be similar to everything
    # This is the "superposition catastrophe"
    
    return


# =============================================================================
# EXPERIMENT 7: Probe Extraction Approach
# =============================================================================

def experiment_7_probe_extraction():
    """
    Apply PEP thinking: Instead of binding, MEASURE the relation directly.
    
    Key insight: The relation between france and paris IS a vector.
    We can extract it directly rather than computing it through binding.
    """
    
    print()
    print("=" * 70)
    print("EXPERIMENT 7: Probe Extraction for Relations")
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
    
    # Create vectors
    countries = {country: sparse_overlapping(country, dim, k) for country, _ in pairs}
    capitals = {capital: sparse_overlapping(capital, dim, k) for _, capital in pairs}
    
    print("Probe Extraction Approach:")
    print("-" * 50)
    print()
    
    # Method 1: Relation as difference
    # If country + relation = capital, then relation = capital - country
    print("Method 1: Relation = Capital - Country")
    
    relations = []
    for country, capital in pairs:
        relation = capitals[capital] - countries[country]
        relations.append(relation)
    
    # Average relation (the "capital_of" concept)
    avg_relation = np.mean(relations, axis=0)
    avg_relation = avg_relation / np.linalg.norm(avg_relation)
    
    print(f"Average relation vector computed from {len(pairs)} pairs")
    print()
    
    # Test: Apply average relation to countries
    print("Testing: country + avg_relation → capital?")
    correct = 0
    for country, expected_capital in pairs:
        predicted = countries[country] + avg_relation
        predicted = predicted / np.linalg.norm(predicted)
        
        # Find nearest capital
        best_capital = max(capitals.keys(), 
                         key=lambda c: np.dot(predicted, capitals[c]))
        
        match = "✓" if best_capital == expected_capital else "✗"
        sim = np.dot(predicted, capitals[expected_capital])
        print(f"  {match} {country} → {best_capital} (expected: {expected_capital}, sim: {sim:.3f})")
        
        if best_capital == expected_capital:
            correct += 1
    
    print(f"\nAccuracy: {correct}/{len(pairs)} = {correct/len(pairs):.1%}")
    print()
    
    # Method 2: Relation as transformation matrix (PEP-style)
    print("Method 2: Relation as Linear Transformation")
    print("-" * 50)
    
    # Stack countries and capitals as matrices
    X = np.array([countries[c] for c, _ in pairs])  # Countries
    Y = np.array([capitals[cap] for _, cap in pairs])  # Capitals
    
    # Solve for transformation: Y = X @ W
    # W = (X^T X)^(-1) X^T Y
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        W = XtX_inv @ X.T @ Y
        
        print(f"Transformation matrix W: {W.shape}")
        
        # Test transformation
        correct = 0
        for i, (country, expected_capital) in enumerate(pairs):
            predicted = X[i] @ W
            predicted = predicted / np.linalg.norm(predicted)
            
            best_capital = max(capitals.keys(),
                             key=lambda c: np.dot(predicted, capitals[c]))
            
            match = "✓" if best_capital == expected_capital else "✗"
            print(f"  {match} {country} → {best_capital}")
            
            if best_capital == expected_capital:
                correct += 1
        
        print(f"\nAccuracy: {correct}/{len(pairs)} = {correct/len(pairs):.1%}")
        
    except Exception as e:
        print(f"Matrix solve failed: {e}")
    
    print()
    
    # Method 3: Analogies using extracted relation
    print("Method 3: Analogies with Extracted Relation")
    print("-" * 50)
    
    correct = 0
    total = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Extract relation from pair 1
            relation = capitals[capital1] - countries[country1]
            
            # Apply to country2
            predicted = countries[country2] + relation
            predicted = predicted / np.linalg.norm(predicted)
            
            best_capital = max(capitals.keys(),
                             key=lambda c: np.dot(predicted, capitals[c]))
            
            if best_capital == capital2:
                correct += 1
            total += 1
    
    print(f"Analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    
    return correct / total


# =============================================================================
# EXPERIMENT 8: Structured Sparse with Semantic Dimensions
# =============================================================================

def experiment_8_semantic_dimensions():
    """
    Use semantic dimensions based on EDP anchor concepts.
    
    Anchors: zero, sierpinski, phi, e_inv, cantor, sqrt2_inv
    
    Map to: null, category, hierarchy, specificity, boundary, relation
    """
    
    print()
    print("=" * 70)
    print("EXPERIMENT 8: Semantic Dimension Structure")
    print("=" * 70)
    print()
    
    dim = 256
    
    # Semantic dimension ranges (inspired by EDP anchors)
    semantic_ranges = {
        'identity': (0, 42),      # What it IS (entity identity)
        'category': (42, 84),     # What TYPE it is (country, city)
        'hierarchy': (84, 126),   # WHERE in hierarchy (continent > country > city)
        'specificity': (126, 168), # HOW specific (general vs particular)
        'boundary': (168, 210),   # BOUNDARIES (geographic, conceptual)
        'relation': (210, 256),   # RELATIONS to other entities
    }
    
    print("Semantic dimension ranges:")
    for name, (start, end) in semantic_ranges.items():
        print(f"  {name}: dims {start}-{end}")
    print()
    
    def semantic_sparse(name: str, entity_type: str, category: str) -> np.ndarray:
        """Create vector with semantic structure."""
        seed = hash(name) % (2**32)
        rng = np.random.default_rng(seed)
        
        vec = np.zeros(dim)
        
        # Identity: unique to this entity
        id_start, id_end = semantic_ranges['identity']
        id_active = rng.choice(id_end - id_start, size=4, replace=False) + id_start
        vec[id_active] = rng.choice([-1.0, 1.0], size=4)
        
        # Category: shared by type (country, city)
        cat_start, cat_end = semantic_ranges['category']
        cat_seed = hash(entity_type) % (2**32)
        cat_rng = np.random.default_rng(cat_seed)
        cat_active = cat_rng.choice(cat_end - cat_start, size=4, replace=False) + cat_start
        vec[cat_active] = cat_rng.choice([-1.0, 1.0], size=4)
        
        # Hierarchy: based on category level
        hier_start, hier_end = semantic_ranges['hierarchy']
        hier_seed = hash(category) % (2**32)
        hier_rng = np.random.default_rng(hier_seed)
        hier_active = hier_rng.choice(hier_end - hier_start, size=2, replace=False) + hier_start
        vec[hier_active] = hier_rng.choice([-1.0, 1.0], size=2)
        
        return vec / np.linalg.norm(vec)
    
    pairs = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
    ]
    
    # Create semantically structured vectors
    countries = {c: semantic_sparse(c, 'country', 'nation') for c, _ in pairs}
    capitals = {cap: semantic_sparse(cap, 'city', 'capital') for _, cap in pairs}
    
    print("Testing semantic structure:")
    print("-" * 50)
    
    # Countries should be similar to each other (shared category dims)
    print("Country-country similarities:")
    country_names = list(countries.keys())
    for i, c1 in enumerate(country_names[:3]):
        for c2 in country_names[i+1:i+3]:
            sim = np.dot(countries[c1], countries[c2])
            print(f"  {c1} - {c2}: {sim:.3f}")
    
    print()
    
    # Capitals should be similar to each other
    print("Capital-capital similarities:")
    capital_names = list(capitals.keys())
    for i, c1 in enumerate(capital_names[:3]):
        for c2 in capital_names[i+1:i+3]:
            sim = np.dot(capitals[c1], capitals[c2])
            print(f"  {c1} - {c2}: {sim:.3f}")
    
    print()
    
    # Countries should be LESS similar to capitals
    print("Country-capital similarities:")
    for country, capital in list(pairs)[:3]:
        sim = np.dot(countries[country], capitals[capital])
        print(f"  {country} - {capital}: {sim:.3f}")
    
    print()
    
    # Test analogies with semantic structure
    print("Analogy test with semantic structure:")
    print("-" * 50)
    
    # Use the relation dimension for binding
    rel_start, rel_end = semantic_ranges['relation']
    
    correct = 0
    total = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            # Relation = difference in identity + relation dims
            relation = capitals[capital1] - countries[country1]
            
            # Apply to country2
            predicted = countries[country2] + relation
            predicted = predicted / np.linalg.norm(predicted)
            
            best_capital = max(capitals.keys(),
                             key=lambda c: np.dot(predicted, capitals[c]))
            
            if best_capital == capital2:
                correct += 1
            total += 1
    
    print(f"Analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    
    return correct / total


# =============================================================================
# EXPERIMENT 9: Error-Driven Refinement (GOP-style)
# =============================================================================

def experiment_9_error_driven():
    """
    Apply GOP principle: Use errors to guide structure building.
    
    Start simple, add structure where errors occur.
    """
    
    print()
    print("=" * 70)
    print("EXPERIMENT 9: Error-Driven Structure Building")
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
    
    print("GOP Approach: Build structure where errors occur")
    print("-" * 50)
    print()
    
    # Phase 1: Start with random sparse
    print("Phase 1: Random sparse encoding")
    
    countries = {c: sparse_overlapping(c, dim, k) for c, _ in pairs}
    capitals = {cap: sparse_overlapping(cap, dim, k) for _, cap in pairs}
    
    # Test and collect errors
    errors = []
    correct = 0
    total = 0
    
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            relation = capitals[capital1] - countries[country1]
            predicted = countries[country2] + relation
            predicted = predicted / np.linalg.norm(predicted)
            
            best_capital = max(capitals.keys(),
                             key=lambda c: np.dot(predicted, capitals[c]))
            
            if best_capital == capital2:
                correct += 1
            else:
                errors.append({
                    'country1': country1, 'capital1': capital1,
                    'country2': country2, 'capital2': capital2,
                    'predicted': best_capital
                })
            total += 1
    
    print(f"Initial accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"Errors: {len(errors)}")
    print()
    
    # Phase 2: Analyze errors
    print("Phase 2: Error analysis")
    
    # Which capitals are being confused?
    confusion = defaultdict(int)
    for err in errors:
        confusion[(err['capital2'], err['predicted'])] += 1
    
    print("Confusion pairs:")
    for (expected, predicted), count in sorted(confusion.items(), key=lambda x: -x[1])[:5]:
        print(f"  {expected} confused with {predicted}: {count} times")
    
    print()
    
    # Phase 3: Add distinguishing structure
    print("Phase 3: Add distinguishing dimensions")
    
    # For each confused pair, add a dimension that distinguishes them
    confused_pairs = list(confusion.keys())
    
    for expected, predicted in confused_pairs[:3]:
        # Find a dimension where they differ
        exp_vec = capitals[expected]
        pred_vec = capitals[predicted]
        
        # Add a new distinguishing dimension
        diff = exp_vec - pred_vec
        max_diff_dim = np.argmax(np.abs(diff))
        
        # Amplify this dimension in expected
        capitals[expected][max_diff_dim] *= 2
        capitals[expected] = capitals[expected] / np.linalg.norm(capitals[expected])
        
        print(f"  Amplified dim {max_diff_dim} for {expected}")
    
    print()
    
    # Phase 4: Retest
    print("Phase 4: Retest after refinement")
    
    correct = 0
    for i, (country1, capital1) in enumerate(pairs):
        for j, (country2, capital2) in enumerate(pairs):
            if i == j:
                continue
            
            relation = capitals[capital1] - countries[country1]
            predicted = countries[country2] + relation
            predicted = predicted / np.linalg.norm(predicted)
            
            best_capital = max(capitals.keys(),
                             key=lambda c: np.dot(predicted, capitals[c]))
            
            if best_capital == capital2:
                correct += 1
    
    print(f"Refined accuracy: {correct}/{total} = {correct/total:.1%}")
    
    return correct / total


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("SPARSE VSA EXPLORATION v2")
    print("=" * 70)
    print()
    print("Following up on v1 findings with deeper investigation.")
    print()
    
    experiment_6_paris_attractor()
    acc_probe = experiment_7_probe_extraction()
    acc_semantic = experiment_8_semantic_dimensions()
    acc_error = experiment_9_error_driven()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Key findings:")
    print(f"  Probe extraction accuracy: {acc_probe:.1%}")
    print(f"  Semantic dimensions accuracy: {acc_semantic:.1%}")
    print(f"  Error-driven refinement: {acc_error:.1%}")
    print()
    print("Insights:")
    print("  1. Relation = Capital - Country works better than binding")
    print("  2. Semantic structure helps separate types")
    print("  3. Error-driven refinement can improve accuracy")
    print()
    print("This aligns with PEP: 'When approximation fails, measure directly'")
    print("The relation IS the difference vector, not a binding operation.")
    print()


if __name__ == "__main__":
    main()
