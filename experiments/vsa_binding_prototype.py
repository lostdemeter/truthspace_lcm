#!/usr/bin/env python3
"""
VSA Binding Prototype for TruthSpace LCM

This demonstrates how to add binding operations to make TruthSpace a full
Vector Symbolic Architecture (VSA) / Hyperdimensional Computing (HDC) system.

Binding enables:
1. Representing relations (capital_of ⊛ paris + france)
2. Analogical reasoning (France:Paris :: Germany:?)
3. Sequence encoding (position-tagged elements)
4. Predicate logic (subject ⊛ predicate ⊛ object)

All operations remain purely geometric - no training, no probabilities.

Binding Methods Implemented:
- Element-wise multiplication (Hadamard product) - simplest, self-inverse
- Circular convolution (HRR) - better orthogonality preservation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.vocabulary import Vocabulary, cosine_similarity, word_position


# =============================================================================
# BINDING OPERATIONS
# =============================================================================

def bind_hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind two vectors using element-wise multiplication (Hadamard product).
    
    Properties:
    - Commutative: a ⊛ b = b ⊛ a
    - Self-inverse for normalized vectors: a ⊛ a ≈ identity-like
    - Result is dissimilar to both inputs (key VSA property)
    - O(D) complexity
    
    For unit vectors, the result should be renormalized.
    """
    result = a * b
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


def unbind_hadamard(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Unbind using Hadamard product (self-inverse property).
    
    If bound = a ⊛ b, then:
        unbind(bound, a) ≈ b (recovers b)
        unbind(bound, b) ≈ a (recovers a)
    
    For real-valued vectors, this is approximate recovery via similarity search.
    """
    return bind_hadamard(bound, key)


def bind_circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind two vectors using circular convolution (HRR - Holographic Reduced Representations).
    
    Formula: x ⊛ y = ifft(fft(x) · fft(y))
    
    Properties:
    - Better orthogonality preservation than Hadamard
    - Exact inverse via correlation
    - Result is dissimilar to both inputs
    """
    result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


def unbind_circular_conv(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Unbind using circular correlation (inverse of convolution).
    
    Formula: unbind(bound, key) = ifft(fft(bound) · conj(fft(key)))
    """
    result = np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(key))).real
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


# =============================================================================
# BUNDLING (Superposition) - Already in TruthSpace, but explicit here
# =============================================================================

def bundle(*vectors: np.ndarray) -> np.ndarray:
    """
    Bundle multiple vectors via addition (superposition).
    
    This is the VSA equivalent of set union / multiset.
    The result is similar to all inputs (unlike binding).
    """
    result = np.sum(vectors, axis=0)
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


# =============================================================================
# PERMUTATION - For sequence encoding
# =============================================================================

def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Permute vector by cyclic shift.
    
    Used for encoding position/order in sequences.
    P^n(v) encodes element v at position n.
    """
    return np.roll(v, shift)


def inverse_permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Inverse permutation."""
    return np.roll(v, -shift)


# =============================================================================
# VSA MEMORY - Clean-up memory for recovering noisy results
# =============================================================================

class VSAMemory:
    """
    Clean-up memory for VSA operations.
    
    After unbinding, results are noisy. The clean-up memory finds the
    nearest known vector to recover the exact symbol.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.symbols: Dict[str, np.ndarray] = {}
    
    def add(self, name: str, vector: np.ndarray):
        """Add a symbol to memory."""
        self.symbols[name] = vector
    
    def cleanup(self, noisy: np.ndarray, threshold: float = 0.0) -> Tuple[str, float]:
        """
        Find nearest symbol to noisy vector.
        
        Returns (symbol_name, similarity) or (None, 0) if below threshold.
        """
        best_name = None
        best_sim = -1
        
        for name, vec in self.symbols.items():
            sim = cosine_similarity(noisy, vec)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        
        if best_sim < threshold:
            return None, 0.0
        return best_name, best_sim
    
    def cleanup_top_k(self, noisy: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k nearest symbols."""
        results = []
        for name, vec in self.symbols.items():
            sim = cosine_similarity(noisy, vec)
            results.append((name, sim))
        return sorted(results, key=lambda x: -x[1])[:k]


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_relational_binding():
    """
    Demonstrate binding for relational knowledge.
    
    Example: Encoding "Paris is the capital of France"
    """
    print("=" * 70)
    print("DEMO 1: Relational Binding")
    print("=" * 70)
    print()
    
    dim = 64
    memory = VSAMemory(dim)
    
    # Create role vectors (deterministic from hash)
    capital_of = word_position("capital_of", dim)
    located_in = word_position("located_in", dim)
    
    # Create entity vectors
    paris = word_position("paris", dim)
    france = word_position("france", dim)
    berlin = word_position("berlin", dim)
    germany = word_position("germany", dim)
    tokyo = word_position("tokyo", dim)
    japan = word_position("japan", dim)
    
    # Add entities to clean-up memory
    for name, vec in [("paris", paris), ("france", france), 
                      ("berlin", berlin), ("germany", germany),
                      ("tokyo", tokyo), ("japan", japan)]:
        memory.add(name, vec)
    
    print("Encoding facts as bound pairs:")
    print("-" * 70)
    
    # Encode facts: capital_of ⊛ city + country
    # This means: "the capital_of relation applied to city, bundled with country"
    fact_france = bundle(bind_hadamard(capital_of, paris), france)
    fact_germany = bundle(bind_hadamard(capital_of, berlin), germany)
    fact_japan = bundle(bind_hadamard(capital_of, tokyo), japan)
    
    print("fact_france = capital_of ⊛ paris + france")
    print("fact_germany = capital_of ⊛ berlin + germany")
    print("fact_japan = capital_of ⊛ tokyo + japan")
    print()
    
    # Bundle all facts into a single memory vector
    knowledge = bundle(fact_france, fact_germany, fact_japan)
    print("knowledge = bundle(fact_france, fact_germany, fact_japan)")
    print()
    
    # Query: "What is the capital of France?"
    # Unbind capital_of from knowledge, then check similarity to france
    print("Query: What is the capital of France?")
    print("-" * 70)
    
    # Method 1: Probe with country to get capital
    # We want to find X where capital_of ⊛ X + france is in knowledge
    # Approach: unbind(knowledge, capital_of) should be similar to cities
    
    probe = unbind_hadamard(fact_france, capital_of)
    result, sim = memory.cleanup(probe)
    print(f"unbind(fact_france, capital_of) → {result} (sim={sim:.3f})")
    
    # Method 2: Direct similarity - which fact matches query pattern?
    print()
    print("Direct matching: Which city goes with France?")
    for city_name, city_vec in [("paris", paris), ("berlin", berlin), ("tokyo", tokyo)]:
        # Create query pattern: capital_of ⊛ city + france
        query = bundle(bind_hadamard(capital_of, city_vec), france)
        sim = cosine_similarity(query, fact_france)
        print(f"  capital_of ⊛ {city_name} + france → sim={sim:.3f}")
    
    print()
    return True


def demo_analogical_reasoning():
    """
    Demonstrate analogical reasoning.
    
    France:Paris :: Germany:?
    """
    print("=" * 70)
    print("DEMO 2: Analogical Reasoning")
    print("=" * 70)
    print()
    
    dim = 64
    memory = VSAMemory(dim)
    
    # Entities
    paris = word_position("paris", dim)
    france = word_position("france", dim)
    berlin = word_position("berlin", dim)
    germany = word_position("germany", dim)
    tokyo = word_position("tokyo", dim)
    japan = word_position("japan", dim)
    rome = word_position("rome", dim)
    italy = word_position("italy", dim)
    
    for name, vec in [("paris", paris), ("berlin", berlin), 
                      ("tokyo", tokyo), ("rome", rome)]:
        memory.add(name, vec)
    
    print("Analogy: France:Paris :: Germany:?")
    print("-" * 70)
    
    # The relation between France and Paris
    # relation = paris ⊛ france^(-1) ≈ paris ⊛ france (for normalized vectors)
    relation = bind_hadamard(paris, france)
    print(f"relation = paris ⊛ france (captures 'capital-of' implicitly)")
    
    # Apply same relation to Germany
    answer = bind_hadamard(relation, germany)
    print(f"answer = relation ⊛ germany")
    
    # Clean up to find nearest city
    results = memory.cleanup_top_k(answer, k=4)
    print()
    print("Top matches:")
    for name, sim in results:
        marker = "✓" if name == "berlin" else " "
        print(f"  {marker} {name}: {sim:.3f}")
    
    print()
    
    # Test more analogies
    print("More analogies:")
    print("-" * 70)
    
    analogies = [
        ("Japan", "Tokyo", "Italy", "rome", japan, tokyo, italy),
        ("Germany", "Berlin", "France", "paris", germany, berlin, france),
    ]
    
    correct = 0
    for country1, city1, country2, expected, c1_vec, ci1_vec, c2_vec in analogies:
        relation = bind_hadamard(ci1_vec, c1_vec)
        answer = bind_hadamard(relation, c2_vec)
        result, sim = memory.cleanup(answer)
        match = result == expected
        if match:
            correct += 1
        marker = "✓" if match else "✗"
        print(f"  {marker} {country1}:{city1} :: {country2}:? → {result} (expected: {expected})")
    
    print()
    return True


def demo_sequence_encoding():
    """
    Demonstrate sequence encoding using permutation.
    """
    print("=" * 70)
    print("DEMO 3: Sequence Encoding")
    print("=" * 70)
    print()
    
    dim = 64
    memory = VSAMemory(dim)
    
    # Words
    the = word_position("the", dim)
    quick = word_position("quick", dim)
    brown = word_position("brown", dim)
    fox = word_position("fox", dim)
    
    for name, vec in [("the", the), ("quick", quick), ("brown", brown), ("fox", fox)]:
        memory.add(name, vec)
    
    print("Encoding: 'the quick brown fox'")
    print("-" * 70)
    
    # Encode sequence with position tags
    # seq = P^0 ⊛ the + P^1 ⊛ quick + P^2 ⊛ brown + P^3 ⊛ fox
    words = [the, quick, brown, fox]
    word_names = ["the", "quick", "brown", "fox"]
    
    # Create position vectors
    P = word_position("POSITION", dim)  # Base permutation vector
    
    seq = np.zeros(dim)
    for i, (w, name) in enumerate(zip(words, word_names)):
        # Position encoding: bind word with permuted position marker
        pos_marker = permute(P, i)
        bound = bind_hadamard(pos_marker, w)
        seq = seq + bound
        print(f"  Position {i}: P^{i} ⊛ {name}")
    
    # Normalize
    seq = seq / np.linalg.norm(seq)
    print()
    
    # Query: "What's at position 2?"
    print("Query: What's at position 2?")
    pos2_marker = permute(P, 2)
    recovered = unbind_hadamard(seq, pos2_marker)
    result, sim = memory.cleanup(recovered)
    print(f"  unbind(seq, P^2) → {result} (sim={sim:.3f})")
    print(f"  Expected: brown")
    
    print()
    
    # Query all positions
    print("Recovering all positions:")
    for i in range(4):
        pos_marker = permute(P, i)
        recovered = unbind_hadamard(seq, pos_marker)
        result, sim = memory.cleanup(recovered)
        expected = word_names[i]
        marker = "✓" if result == expected else "✗"
        print(f"  {marker} Position {i}: {result} (expected: {expected}, sim={sim:.3f})")
    
    print()
    return True


def demo_circular_convolution():
    """
    Compare Hadamard vs Circular Convolution binding.
    """
    print("=" * 70)
    print("DEMO 4: Hadamard vs Circular Convolution")
    print("=" * 70)
    print()
    
    dim = 64
    
    a = word_position("alpha", dim)
    b = word_position("beta", dim)
    
    print("Binding alpha ⊛ beta:")
    print("-" * 70)
    
    # Hadamard
    bound_h = bind_hadamard(a, b)
    recovered_h = unbind_hadamard(bound_h, a)
    sim_h = cosine_similarity(recovered_h, b)
    
    # Circular convolution
    bound_c = bind_circular_conv(a, b)
    recovered_c = unbind_circular_conv(bound_c, a)
    sim_c = cosine_similarity(recovered_c, b)
    
    print(f"Hadamard:     unbind(a⊛b, a) · b = {sim_h:.4f}")
    print(f"Convolution:  unbind(a⊛b, a) · b = {sim_c:.4f}")
    print()
    
    # Key property: bound is dissimilar to inputs
    print("Dissimilarity property (bound should be unlike inputs):")
    print(f"  Hadamard:     bound · a = {cosine_similarity(bound_h, a):.4f}, bound · b = {cosine_similarity(bound_h, b):.4f}")
    print(f"  Convolution:  bound · a = {cosine_similarity(bound_c, a):.4f}, bound · b = {cosine_similarity(bound_c, b):.4f}")
    print()
    
    return True


def demo_integration_with_truthspace():
    """
    Show how binding integrates with existing TruthSpace vocabulary.
    """
    print("=" * 70)
    print("DEMO 5: Integration with TruthSpace Vocabulary")
    print("=" * 70)
    print()
    
    # Use existing TruthSpace vocabulary
    vocab = Vocabulary(dim=64)
    
    # Add some text to build IDF weights
    texts = [
        "Paris is the capital of France",
        "Berlin is the capital of Germany", 
        "The Eiffel Tower is in Paris",
        "The Brandenburg Gate is in Berlin",
    ]
    for text in texts:
        vocab.add_text(text)
    
    print("Knowledge base:")
    for text in texts:
        print(f"  - {text}")
    print()
    
    # Get word vectors from vocabulary
    paris = vocab.get_position("paris")
    france = vocab.get_position("france")
    berlin = vocab.get_position("berlin")
    germany = vocab.get_position("germany")
    capital = vocab.get_position("capital")
    
    # Create clean-up memory
    memory = VSAMemory(vocab.dim)
    memory.add("paris", paris)
    memory.add("france", france)
    memory.add("berlin", berlin)
    memory.add("germany", germany)
    
    print("Using TruthSpace vocabulary vectors for binding:")
    print("-" * 70)
    
    # Encode relational facts
    # "capital" acts as the role vector
    fact_paris = bundle(bind_hadamard(capital, paris), france)
    fact_berlin = bundle(bind_hadamard(capital, berlin), germany)
    
    print("fact_paris = capital ⊛ paris + france")
    print("fact_berlin = capital ⊛ berlin + germany")
    print()
    
    # Query: capital of France?
    print("Query: What is the capital of France?")
    
    # Check which city, when bound with capital and bundled with france, matches
    for city_name in ["paris", "berlin"]:
        city_vec = vocab.get_position(city_name)
        query = bundle(bind_hadamard(capital, city_vec), france)
        sim = cosine_similarity(query, fact_paris)
        print(f"  capital ⊛ {city_name} + france → sim to fact_paris: {sim:.3f}")
    
    print()
    
    # Analogy using TruthSpace vectors
    print("Analogy: France:Paris :: Germany:?")
    relation = bind_hadamard(paris, france)
    answer = bind_hadamard(relation, germany)
    result, sim = memory.cleanup(answer)
    print(f"  Result: {result} (sim={sim:.3f})")
    
    print()
    return True


if __name__ == "__main__":
    print()
    print("VSA BINDING PROTOTYPE FOR TRUTHSPACE LCM")
    print("=" * 70)
    print()
    print("This demonstrates how binding operations extend TruthSpace")
    print("from bag-of-words semantics to full symbolic reasoning.")
    print()
    print("All operations are purely geometric - no training required.")
    print()
    
    demo_relational_binding()
    demo_analogical_reasoning()
    demo_sequence_encoding()
    demo_circular_convolution()
    demo_integration_with_truthspace()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Binding adds these capabilities to TruthSpace:")
    print("  1. Relational knowledge (capital_of ⊛ paris + france)")
    print("  2. Analogical reasoning (France:Paris :: Germany:?)")
    print("  3. Sequence encoding (ordered elements)")
    print("  4. Predicate logic (subject ⊛ predicate ⊛ object)")
    print()
    print("The system remains:")
    print("  - Fully geometric (linear algebra only)")
    print("  - Deterministic (hash-based, no training)")
    print("  - Interpretable (all operations are explicit)")
    print()
    print("Next steps:")
    print("  - Add binding.py to truthspace_lcm/core/")
    print("  - Integrate with Q&A as unbinding queries")
    print("  - Use for style transfer as role-filler binding")
    print()
