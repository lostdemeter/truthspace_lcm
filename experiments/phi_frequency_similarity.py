#!/usr/bin/env python3
"""
φ-Weighted Frequency Similarity Experiment

This experiment explores using φ-weighted frequency as the similarity metric
instead of word overlap or co-occurrence.

Key insight from Design 039 (φ and Zipf Duality):
- φ^(-log(freq)) ≡ Zipf for ranking
- High-frequency words = structural scaffolding
- Low-frequency words = meaningful content
- The ratio follows power law → autobalancing

The geometric principle:
- Similarity is based on SHARED FREQUENCY PATTERNS
- Words with similar φ-weights are geometrically close
- This is Zipf/Pareto under the hood

Author: TruthSpace LCM
License: GPLv3
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import Counter
import re


# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
CRITICAL_LINE = 0.5


class PhiFrequencySpace:
    """
    Geometric space where similarity is based on φ-weighted frequency.
    
    Key insight: We don't need word overlap or co-occurrence.
    The frequency distribution itself encodes semantic structure.
    
    From Design 039:
    - φ^(-log(freq)) produces identical rankings to Zipf
    - This is geometric because it's based on power-law structure
    - The structure IS the navigation
    """
    
    def __init__(self, dims: int = 8):
        self.dims = dims
        self.concepts: List['Concept'] = []
        self.word_freq: Counter = Counter()
        self.total_words: int = 0
        self._eigenvectors: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize to lowercase words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def phi_weight(self, word: str) -> float:
        """
        Compute φ-weight for a word based on its frequency.
        
        φ^(-log(1 + freq)) = (1/freq)^ln(φ) ≈ power law
        
        Rare words get HIGH weight (specific, meaningful)
        Common words get LOW weight (structural, noise)
        """
        freq = self.word_freq.get(word, 0)
        if freq == 0:
            return 0.0  # Unknown word
        
        # φ^(-log(1 + freq))
        return PHI ** (-np.log1p(freq))
    
    def text_phi_vector(self, text: str) -> np.ndarray:
        """
        Convert text to φ-weighted vector.
        
        Each word contributes its φ-weight to the vector.
        The vector represents the "frequency signature" of the text.
        """
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dims)
        
        # Get φ-weights for each word
        weights = [self.phi_weight(w) for w in words]
        
        # Create a signature based on weight distribution
        # Use histogram of weights as the vector
        if not any(w > 0 for w in weights):
            return np.zeros(self.dims)
        
        # Bin weights into dims buckets
        weights = [w for w in weights if w > 0]
        if not weights:
            return np.zeros(self.dims)
        
        # Create vector from weight statistics
        vec = np.zeros(self.dims)
        
        # Dim 0: Mean weight (overall specificity)
        vec[0] = np.mean(weights)
        
        # Dim 1: Max weight (most specific word)
        vec[1] = np.max(weights)
        
        # Dim 2: Std weight (diversity of specificity)
        vec[2] = np.std(weights) if len(weights) > 1 else 0
        
        # Dim 3: Number of high-weight words (> median)
        median = np.median(weights)
        vec[3] = sum(1 for w in weights if w > median) / len(weights)
        
        # Dim 4-7: Quartile values
        quartiles = np.percentile(weights, [25, 50, 75, 100]) if len(weights) >= 4 else [0, 0, 0, 0]
        for i, q in enumerate(quartiles[:4]):
            if i + 4 < self.dims:
                vec[i + 4] = q
        
        return vec
    
    def add_concept(self, text: str) -> 'Concept':
        """Add a concept and update frequency counts."""
        words = self._tokenize(text)
        self.word_freq.update(words)
        self.total_words += len(words)
        
        concept = Concept(text=text, words=set(words))
        self.concepts.append(concept)
        return concept
    
    def reproject(self):
        """
        Reproject using φ-weighted frequency similarity.
        
        Similarity between concepts is based on their φ-weight signatures.
        """
        n = len(self.concepts)
        if n == 0:
            return
        
        # Compute φ-vectors for all concepts
        vectors = np.array([self.text_phi_vector(c.text) for c in self.concepts])
        
        # Build similarity matrix from vector dot products
        # Normalize vectors first
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = vectors / norms
        
        S = normalized @ normalized.T
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        num_dims = min(n, self.dims)
        idx = np.argsort(eigenvalues)[::-1][:num_dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        
        self._eigenvectors = eigenvectors[:, idx]
        self._eigenvalues = valid_eigenvalues
        
        # Compute positions
        positions = self._eigenvectors * np.sqrt(valid_eigenvalues)
        
        for i, concept in enumerate(self.concepts):
            pos = positions[i]
            full_pos = np.zeros(self.dims)
            full_pos[:len(pos)] = pos
            norm = np.linalg.norm(full_pos)
            if norm > 1e-10:
                full_pos = full_pos / norm * CRITICAL_LINE
            concept.position = full_pos
    
    def project_query(self, text: str) -> np.ndarray:
        """Project a query using φ-weighted similarity."""
        if self._eigenvectors is None:
            return np.zeros(self.dims)
        
        # Get query's φ-vector
        query_vec = self.text_phi_vector(text)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        
        # Compute similarity to each concept
        n = len(self.concepts)
        similarities = np.zeros(n)
        for i, concept in enumerate(self.concepts):
            concept_vec = self.text_phi_vector(concept.text)
            concept_norm = np.linalg.norm(concept_vec)
            if concept_norm > 0:
                concept_vec = concept_vec / concept_norm
            similarities[i] = np.dot(query_vec, concept_vec)
        
        # Project into eigenspace
        pos = similarities @ self._eigenvectors
        
        full_pos = np.zeros(self.dims)
        full_pos[:len(pos)] = pos
        
        norm = np.linalg.norm(full_pos)
        if norm > 1e-10:
            full_pos = full_pos / norm * CRITICAL_LINE
        
        return full_pos
    
    def query(self, text: str, top_k: int = 5) -> List[Tuple['Concept', float]]:
        """Query the space."""
        query_pos = self.project_query(text)
        
        results = []
        for concept in self.concepts:
            similarity = np.dot(query_pos, concept.position)
            results.append((concept, similarity))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


@dataclass
class Concept:
    """A concept with text and geometric position."""
    text: str
    words: Set[str]
    position: np.ndarray = field(default_factory=lambda: np.zeros(8))


def experiment_phi_frequency():
    """Test φ-weighted frequency similarity."""
    print("=" * 70)
    print("φ-WEIGHTED FREQUENCY SIMILARITY EXPERIMENT")
    print("=" * 70)
    
    space = PhiFrequencySpace(dims=8)
    
    # Add concepts
    concepts = [
        # Holmes domain - specific vocabulary
        "Sherlock Holmes is a detective who solves crimes using deduction.",
        "Holmes examines clues at the crime scene with Watson.",
        "The detective uses forensic science to find criminals.",
        
        # Python domain - technical vocabulary
        "Python is a programming language with dynamic typing.",
        "Python uses indentation for code blocks instead of braces.",
        "Developers write scripts and applications in Python.",
        
        # Physics domain - scientific vocabulary
        "Physics studies matter and energy in the universe.",
        "Quantum mechanics describes subatomic particle behavior.",
        "Einstein developed the theory of relativity.",
    ]
    
    for text in concepts:
        space.add_concept(text)
    
    space.reproject()
    
    print(f"\nAdded {len(concepts)} concepts")
    print(f"Total words: {space.total_words}")
    print(f"Unique words: {len(space.word_freq)}")
    
    # Show φ-weights for key words
    print("\n" + "-" * 70)
    print("φ-WEIGHTS FOR KEY WORDS")
    print("-" * 70)
    
    key_words = [
        # Domain-specific (should be high weight)
        "holmes", "detective", "python", "programming", "physics", "quantum",
        # Common (should be low weight)
        "is", "the", "a", "and", "with", "in",
    ]
    
    print(f"\n{'Word':15} {'Freq':>6} {'φ-weight':>10}")
    print("-" * 35)
    for word in key_words:
        freq = space.word_freq.get(word, 0)
        weight = space.phi_weight(word)
        print(f"{word:15} {freq:>6} {weight:>10.4f}")
    
    # Test queries
    print("\n" + "-" * 70)
    print("QUERY TESTS")
    print("-" * 70)
    
    queries = [
        "Who is Holmes?",
        "What is Python?",
        "Tell me about physics",
        "detective investigation",
        "coding scripts",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = space.query(query, top_k=3)
        for concept, score in results:
            print(f"  [{score:.3f}] {concept.text[:55]}...")


def experiment_frequency_vs_overlap():
    """Compare φ-frequency similarity to word overlap."""
    print("\n" + "=" * 70)
    print("φ-FREQUENCY vs WORD OVERLAP COMPARISON")
    print("=" * 70)
    
    space = PhiFrequencySpace(dims=8)
    
    # Add concepts with varying vocabulary
    concepts = [
        # Formal/specific
        "Sherlock Holmes employs deductive reasoning and forensic analysis.",
        "Python emphasizes code readability with significant whitespace.",
        "Quantum mechanics describes nature at atomic scales.",
        
        # Casual/general
        "Holmes is a detective.",
        "Python is a language.",
        "Physics is science.",
    ]
    
    for text in concepts:
        space.add_concept(text)
    
    space.reproject()
    
    print(f"\nAdded {len(concepts)} concepts")
    
    # Compare φ-vectors
    print("\n" + "-" * 70)
    print("φ-VECTOR COMPARISON")
    print("-" * 70)
    
    for i, concept in enumerate(space.concepts):
        vec = space.text_phi_vector(concept.text)
        print(f"\n{i+1}. {concept.text[:50]}...")
        print(f"   φ-vector: mean={vec[0]:.3f}, max={vec[1]:.3f}, std={vec[2]:.3f}")
    
    # The formal concepts should have higher mean φ-weight
    # because they use more specific vocabulary
    
    print("\n" + "-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print("""
Formal/specific concepts have HIGHER mean φ-weight because:
- They use rare, domain-specific words
- Rare words get φ^(-log(freq)) → higher weight

Casual/general concepts have LOWER mean φ-weight because:
- They use common, universal words
- Common words get φ^(-log(freq)) → lower weight

This is Zipf/Pareto: the frequency distribution IS the semantic structure.
""")


def experiment_unknown_words():
    """Test handling of unknown words."""
    print("\n" + "=" * 70)
    print("UNKNOWN WORD HANDLING")
    print("=" * 70)
    
    space = PhiFrequencySpace(dims=8)
    
    # Add concepts
    concepts = [
        "Holmes is a detective who solves crimes.",
        "Python is a programming language.",
        "Physics studies matter and energy.",
    ]
    
    for text in concepts:
        space.add_concept(text)
    
    space.reproject()
    
    # Query with unknown word
    query = "sleuth"  # Not in corpus
    print(f"\nQuery: '{query}' (unknown word)")
    
    weight = space.phi_weight("sleuth")
    print(f"φ-weight of 'sleuth': {weight}")
    
    vec = space.text_phi_vector(query)
    print(f"φ-vector: {vec}")
    
    results = space.query(query, top_k=3)
    print("\nResults:")
    for concept, score in results:
        print(f"  [{score:.3f}] {concept.text[:55]}...")
    
    print("\n" + "-" * 70)
    print("OBSERVATION")
    print("-" * 70)
    print("""
Unknown words get φ-weight = 0 (no frequency data).
This means queries with only unknown words have zero similarity.

This is a fundamental limitation of frequency-based approaches.
The solution is either:
1. Bootstrap with external vocabulary (synonyms, embeddings)
2. Learn from use (add successful query-response pairs)
3. Use subword matching (not purely geometric)

For a pure geometric system, option 2 is most aligned with our philosophy:
- The system learns from its own behavior
- Successful matches add to frequency counts
- The structure improves through use
""")


if __name__ == "__main__":
    experiment_phi_frequency()
    experiment_frequency_vs_overlap()
    experiment_unknown_words()
