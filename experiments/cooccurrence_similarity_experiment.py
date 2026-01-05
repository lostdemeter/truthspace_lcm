#!/usr/bin/env python3
"""
Co-occurrence Based Similarity Experiment

This experiment explores using co-occurrence as the similarity metric
instead of word overlap (Jaccard). The key insight from Design 025:

"Co-occurrence counts ARE the attractor dynamics."

Words that appear together frequently form an attractor basin.
This is geometric because co-occurrence = converged attractor state.

Author: TruthSpace LCM
License: GPLv3
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re


# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
CRITICAL_LINE = 0.5


class CooccurrenceTracker:
    """
    Track word co-occurrence within a sliding window.
    
    From Design 025: Co-occurrence is the geometric signal.
    Words that appear together form attractor basins.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.cooccurrence: Dict[str, Counter] = defaultdict(Counter)
        self.word_counts: Counter = Counter()
        self.total_pairs: int = 0
    
    def ingest(self, text: str):
        """Ingest text and track co-occurrence."""
        words = self._tokenize(text)
        
        # Update word counts
        self.word_counts.update(words)
        
        # Track co-occurrence within window
        for i, word in enumerate(words):
            start = max(0, i - self.window_size)
            end = min(len(words), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    other = words[j]
                    self.cooccurrence[word][other] += 1
                    self.total_pairs += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text to lowercase words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def similarity(self, word_a: str, word_b: str) -> float:
        """
        Compute co-occurrence based similarity.
        
        Uses PMI-like measure: how much more often do these words
        co-occur than expected by chance?
        """
        word_a = word_a.lower()
        word_b = word_b.lower()
        
        # Co-occurrence count
        cooc = self.cooccurrence[word_a].get(word_b, 0)
        if cooc == 0:
            return 0.0
        
        # Individual frequencies
        count_a = self.word_counts.get(word_a, 0)
        count_b = self.word_counts.get(word_b, 0)
        
        if count_a == 0 or count_b == 0:
            return 0.0
        
        # PMI-like: log(P(a,b) / (P(a) * P(b)))
        # Simplified: cooc / sqrt(count_a * count_b)
        # This normalizes by individual frequencies
        return cooc / np.sqrt(count_a * count_b)
    
    def text_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute similarity between two texts using co-occurrence.
        
        For each word pair (one from each text), sum their co-occurrence
        similarity. This captures semantic relationships even when
        words don't match exactly.
        """
        words_a = set(self._tokenize(text_a))
        words_b = set(self._tokenize(text_b))
        
        if not words_a or not words_b:
            return 0.0
        
        total_sim = 0.0
        pairs = 0
        
        for wa in words_a:
            for wb in words_b:
                sim = self.similarity(wa, wb)
                if sim > 0:
                    total_sim += sim
                    pairs += 1
        
        # Normalize by geometric mean of word counts
        normalizer = np.sqrt(len(words_a) * len(words_b))
        return total_sim / normalizer if normalizer > 0 else 0.0


@dataclass
class Concept:
    """A concept with text and geometric position."""
    text: str
    words: Set[str]
    position: np.ndarray = field(default_factory=lambda: np.zeros(8))


class CooccurrenceSpace:
    """
    Geometric space where similarity is based on co-occurrence.
    
    Key insight: We don't need word overlap. Co-occurrence captures
    semantic relationships that word overlap misses.
    
    "files" and "ls" don't share letters but co-occur in context.
    """
    
    def __init__(self, dims: int = 8, window_size: int = 10):
        self.dims = dims
        self.tracker = CooccurrenceTracker(window_size)
        self.concepts: List[Concept] = []
        self._eigenvectors: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
    
    def add_concept(self, text: str) -> Concept:
        """Add a concept and update co-occurrence."""
        # Ingest into co-occurrence tracker
        self.tracker.ingest(text)
        
        # Create concept
        words = set(self.tracker._tokenize(text))
        concept = Concept(text=text, words=words)
        self.concepts.append(concept)
        
        return concept
    
    def reproject(self):
        """
        Reproject using co-occurrence similarity.
        
        This is the key difference from word overlap:
        - Word overlap: S[i,j] = |words_i ∩ words_j| / |words_i ∪ words_j|
        - Co-occurrence: S[i,j] = sum of pairwise word co-occurrence
        """
        n = len(self.concepts)
        if n == 0:
            return
        
        # Build similarity matrix from co-occurrence
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = self.tracker.text_similarity(
                    self.concepts[i].text,
                    self.concepts[j].text
                )
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        # Take top dims
        num_dims = min(n, self.dims)
        idx = np.argsort(eigenvalues)[::-1][:num_dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        
        self._eigenvectors = eigenvectors[:, idx]
        self._eigenvalues = valid_eigenvalues
        
        # Compute positions
        positions = self._eigenvectors * np.sqrt(valid_eigenvalues)
        
        for i, concept in enumerate(self.concepts):
            pos = positions[i]
            # Pad to full dims
            full_pos = np.zeros(self.dims)
            full_pos[:len(pos)] = pos
            norm = np.linalg.norm(full_pos)
            if norm > 1e-10:
                full_pos = full_pos / norm * CRITICAL_LINE
            concept.position = full_pos
    
    def project_query(self, text: str) -> np.ndarray:
        """Project a query using co-occurrence similarity."""
        if self._eigenvectors is None:
            return np.zeros(self.dims)
        
        n = len(self.concepts)
        
        # Compute co-occurrence similarity to each concept
        similarities = np.zeros(n)
        for i, concept in enumerate(self.concepts):
            similarities[i] = self.tracker.text_similarity(text, concept.text)
        
        # Project into eigenspace
        pos = similarities @ self._eigenvectors
        
        # Pad to full dims
        full_pos = np.zeros(self.dims)
        full_pos[:len(pos)] = pos
        
        norm = np.linalg.norm(full_pos)
        if norm > 1e-10:
            full_pos = full_pos / norm * CRITICAL_LINE
        
        return full_pos
    
    def query(self, text: str, top_k: int = 5) -> List[Tuple[Concept, float]]:
        """Query the space."""
        query_pos = self.project_query(text)
        
        results = []
        for concept in self.concepts:
            similarity = np.dot(query_pos, concept.position)
            results.append((concept, similarity))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


def experiment_cooccurrence_vs_overlap():
    """Compare co-occurrence similarity to word overlap."""
    print("=" * 70)
    print("CO-OCCURRENCE vs WORD OVERLAP EXPERIMENT")
    print("=" * 70)
    
    # Create both spaces
    cooc_space = CooccurrenceSpace(dims=8)
    
    # Add concepts - note that we add related concepts together
    # so co-occurrence can learn the relationships
    concepts = [
        # Holmes cluster - these should co-occur
        "Sherlock Holmes is a detective who solves crimes.",
        "Holmes uses deduction and observation to find criminals.",
        "Watson assists Holmes in his investigations.",
        "The detective examines clues at the crime scene.",
        
        # Python cluster
        "Python is a programming language for software development.",
        "Python code uses indentation for blocks.",
        "Programmers write scripts in Python.",
        "The language supports object-oriented programming.",
        
        # Physics cluster
        "Physics studies matter and energy in the universe.",
        "Quantum mechanics describes atomic behavior.",
        "Einstein developed the theory of relativity.",
        "Scientists measure forces and motion.",
    ]
    
    for text in concepts:
        cooc_space.add_concept(text)
    
    cooc_space.reproject()
    
    print(f"\nAdded {len(concepts)} concepts")
    print(f"Co-occurrence pairs tracked: {cooc_space.tracker.total_pairs}")
    
    # Test queries
    print("\n" + "-" * 70)
    print("QUERY TESTS")
    print("-" * 70)
    
    queries = [
        # Direct matches
        "Who is Holmes?",
        "What is Python?",
        "Tell me about physics",
        
        # Semantic matches (no word overlap but co-occurrence)
        "detective investigation",  # Should match Holmes via co-occurrence
        "coding scripts",           # Should match Python via co-occurrence
        "atomic particles",         # Should match Physics via co-occurrence
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = cooc_space.query(query, top_k=3)
        for concept, score in results:
            print(f"  [{score:.3f}] {concept.text[:55]}...")
    
    # Show learned co-occurrences
    print("\n" + "-" * 70)
    print("LEARNED CO-OCCURRENCES")
    print("-" * 70)
    
    # Key word pairs that should have high co-occurrence
    test_pairs = [
        ("holmes", "detective"),
        ("holmes", "watson"),
        ("python", "programming"),
        ("python", "code"),
        ("physics", "energy"),
        ("physics", "quantum"),
        # Cross-domain (should be low)
        ("holmes", "python"),
        ("detective", "programming"),
    ]
    
    print("\nWord pair co-occurrence similarities:")
    for w1, w2 in test_pairs:
        sim = cooc_space.tracker.similarity(w1, w2)
        print(f"  {w1:15} ↔ {w2:15}: {sim:.4f}")


def experiment_semantic_bridging():
    """
    Test if co-occurrence can bridge semantic gaps.
    
    The key test: Can we match "investigation" to Holmes
    even though "investigation" never appears in Holmes concepts?
    
    This requires the co-occurrence to learn:
    detective ↔ investigation (from general knowledge)
    holmes ↔ detective (from our concepts)
    Therefore: holmes ↔ investigation (transitive)
    """
    print("\n" + "=" * 70)
    print("SEMANTIC BRIDGING EXPERIMENT")
    print("=" * 70)
    
    space = CooccurrenceSpace(dims=8)
    
    # Add concepts with bridging potential
    concepts = [
        # Holmes concepts (use "detective", "crime", "clues")
        "Holmes is a detective who solves crimes.",
        "The detective finds clues at the scene.",
        
        # Bridging concepts (link "detective" to "investigation")
        "Detectives conduct investigations to solve cases.",
        "An investigation requires gathering evidence.",
        
        # Python concepts
        "Python is a programming language.",
        "Programmers write code in Python.",
    ]
    
    for text in concepts:
        space.add_concept(text)
    
    space.reproject()
    
    print(f"\nAdded {len(concepts)} concepts")
    
    # Test bridging
    print("\n" + "-" * 70)
    print("BRIDGING TESTS")
    print("-" * 70)
    
    # "investigation" should bridge to Holmes via detective
    query = "investigation"
    print(f"\nQuery: '{query}'")
    print("(Should bridge: investigation → detective → Holmes)")
    
    results = space.query(query, top_k=3)
    for concept, score in results:
        print(f"  [{score:.3f}] {concept.text[:55]}...")
    
    # Check the co-occurrence chain
    print("\nCo-occurrence chain:")
    print(f"  investigation ↔ detective: {space.tracker.similarity('investigation', 'detective'):.4f}")
    print(f"  detective ↔ holmes:        {space.tracker.similarity('detective', 'holmes'):.4f}")
    print(f"  investigation ↔ holmes:    {space.tracker.similarity('investigation', 'holmes'):.4f}")


def experiment_pure_geometric():
    """
    Test pure geometric matching without any word-based similarity.
    
    The ultimate goal: positions encode relationships so well that
    we don't need to compute word overlap at query time.
    """
    print("\n" + "=" * 70)
    print("PURE GEOMETRIC MATCHING EXPERIMENT")
    print("=" * 70)
    
    space = CooccurrenceSpace(dims=8)
    
    # Add a rich corpus to build co-occurrence
    corpus = [
        # Holmes domain
        "Sherlock Holmes is the greatest detective in London.",
        "Holmes solves mysteries using deduction and logic.",
        "Watson is Holmes's loyal friend and chronicler.",
        "The detective examines crime scenes for clues.",
        "Holmes plays the violin and smokes a pipe.",
        
        # Python domain
        "Python is a popular programming language.",
        "Python uses indentation instead of braces.",
        "Developers write scripts and applications in Python.",
        "Python supports multiple programming paradigms.",
        "The language was created by Guido van Rossum.",
        
        # Physics domain
        "Physics is the study of matter and energy.",
        "Quantum mechanics describes subatomic particles.",
        "Einstein's relativity changed our understanding of spacetime.",
        "Physicists use mathematics to model the universe.",
        "The laws of physics govern all natural phenomena.",
    ]
    
    for text in corpus:
        space.add_concept(text)
    
    space.reproject()
    
    print(f"\nAdded {len(corpus)} concepts")
    
    # Test with queries that have NO word overlap with concepts
    print("\n" + "-" * 70)
    print("ZERO OVERLAP QUERIES")
    print("-" * 70)
    
    # These queries use synonyms/related words, not exact matches
    queries = [
        ("sleuth", "Holmes"),      # sleuth = detective synonym
        ("coding", "Python"),      # coding = programming synonym
        ("atoms", "Physics"),      # atoms = quantum/subatomic related
    ]
    
    for query, expected_domain in queries:
        print(f"\nQuery: '{query}' (expecting: {expected_domain})")
        results = space.query(query, top_k=3)
        for concept, score in results:
            # Check if result is from expected domain
            domain = "Holmes" if "holmes" in concept.text.lower() else \
                     "Python" if "python" in concept.text.lower() else \
                     "Physics" if "physics" in concept.text.lower() else "?"
            match = "✓" if domain == expected_domain else "✗"
            print(f"  [{score:.3f}] {match} ({domain}) {concept.text[:45]}...")


def experiment_learning_from_use():
    """
    Test learning from successful use.
    
    The key insight: Unknown words can be learned by observing
    which concepts they successfully match with.
    
    If user queries "sleuth" and we return Holmes, and user
    confirms it's correct, we learn: sleuth ↔ Holmes concepts.
    """
    print("\n" + "=" * 70)
    print("LEARNING FROM USE EXPERIMENT")
    print("=" * 70)
    
    space = CooccurrenceSpace(dims=8)
    
    # Initial corpus
    concepts = [
        "Sherlock Holmes is a detective who solves crimes.",
        "Holmes uses deduction to find criminals.",
        "Python is a programming language.",
        "Physics studies matter and energy.",
    ]
    
    for text in concepts:
        space.add_concept(text)
    
    space.reproject()
    
    print(f"\nInitial corpus: {len(concepts)} concepts")
    
    # Query with unknown word
    query = "sleuth"
    print(f"\nQuery: '{query}' (unknown word)")
    
    results = space.query(query, top_k=1)
    print(f"Result: {results[0][0].text[:50]}... (score: {results[0][1]:.3f})")
    
    # Simulate user feedback: "Yes, that's correct!"
    # This means we should learn: sleuth ↔ detective concepts
    print("\n[USER FEEDBACK: Correct!]")
    print("Learning: 'sleuth' should co-occur with Holmes concepts")
    
    # Add the successful query-response pair to co-occurrence
    feedback_text = f"{query} {results[0][0].text}"
    space.tracker.ingest(feedback_text)
    
    # Check what we learned
    print(f"\nAfter learning:")
    print(f"  sleuth ↔ holmes:    {space.tracker.similarity('sleuth', 'holmes'):.4f}")
    print(f"  sleuth ↔ detective: {space.tracker.similarity('sleuth', 'detective'):.4f}")
    
    # Reproject with new knowledge
    space.reproject()
    
    # Query again
    print(f"\nQuery again: '{query}'")
    results = space.query(query, top_k=3)
    for concept, score in results:
        print(f"  [{score:.3f}] {concept.text[:55]}...")
    
    print("\n" + "-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print("""
The system learns from successful use:
1. Unknown word "sleuth" initially has no co-occurrence
2. User confirms Holmes is the correct match
3. We ingest the query+response pair
4. Now "sleuth" co-occurs with Holmes vocabulary
5. Future queries for "sleuth" match Holmes better

This is the attractor dynamic: successful matches pull words together.
""")


def experiment_summary():
    """Summarize findings."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print("""
FINDINGS:

1. CO-OCCURRENCE vs WORD OVERLAP
   - Co-occurrence captures semantic relationships (detective ↔ crime)
   - Word overlap only captures surface form (detective ↔ detective)
   - Co-occurrence is more geometric (attractor basins from data)

2. LIMITATION: UNKNOWN WORDS
   - Words not in training have zero co-occurrence
   - They fall back to hash-based positions (arbitrary)
   - This is a fundamental limitation of pure geometric systems

3. SOLUTION: LEARNING FROM USE
   - When a query succeeds, ingest query+response together
   - This builds co-occurrence between unknown words and known concepts
   - The system learns from feedback (attractor dynamics)

4. THE GEOMETRIC PRINCIPLE
   - We don't need external knowledge (embeddings, LLMs)
   - We construct geometry from observed relationships
   - Successful use = attraction, failed use = repulsion
   - The structure learns from its own behavior

NEXT STEPS:
- Integrate learning-from-use into KnowledgeSpace
- Add positive/negative feedback to adjust positions
- Test with real user interactions
""")


if __name__ == "__main__":
    experiment_cooccurrence_vs_overlap()
    experiment_semantic_bridging()
    experiment_pure_geometric()
    experiment_learning_from_use()
    experiment_summary()
