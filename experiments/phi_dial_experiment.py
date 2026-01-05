#!/usr/bin/env python3
"""
φ-Dial Dimensional Control Experiment

This experiment explores how the φ-dial can control dimensional navigation
in a geometric knowledge space.

Key concepts from design docs:
- φ^(-log(f)) ≡ Zipf for ranking (Design 039)
- Inward (φ^-n) for specific, Outward (φ^+n) for universal (Design 040)
- Single dial controls multiple dimensions simultaneously (Design 041)

The hypothesis:
- dial = -1: Inward navigation → specific, formal, rare
- dial = 0: Balanced navigation → neutral
- dial = +1: Outward navigation → universal, casual, common

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
import re


# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
CRITICAL_LINE = 0.5


@dataclass
class Concept:
    """A concept with text and geometric position."""
    text: str
    words: Set[str]
    position: np.ndarray
    frequency: int = 1  # How often this concept is used
    
    @property
    def magnitude(self) -> float:
        return np.linalg.norm(self.position)


class PhiDialSpace:
    """
    Geometric space with φ-dial dimensional control.
    
    The φ-dial controls navigation direction:
    - dial = -1: Inward (specific, rare, formal)
    - dial = 0: Balanced (neutral)
    - dial = +1: Outward (universal, common, casual)
    
    The key formula: weight = φ^(dial × log(value))
    """
    
    def __init__(self, dims: int = 8):
        self.dims = dims
        self.concepts: List[Concept] = []
        self._similarity_matrix: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        
        # Word frequency tracking
        self._word_counts: Dict[str, int] = {}
        self._total_concepts: int = 0
    
    def extract_words(self, text: str) -> Set[str]:
        """Extract content words from text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter short words (< 2 chars)
        return {w for w in words if len(w) >= 2}
    
    def word_overlap(self, words_a: Set[str], words_b: Set[str]) -> float:
        """Jaccard similarity between word sets."""
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0
    
    def add_concept(self, text: str) -> Concept:
        """Add a concept to the space."""
        words = self.extract_words(text)
        
        # Update word counts
        for word in words:
            self._word_counts[word] = self._word_counts.get(word, 0) + 1
        self._total_concepts += 1
        
        # Initial position (will be reprojected)
        position = np.zeros(self.dims)
        
        concept = Concept(text=text, words=words, position=position)
        self.concepts.append(concept)
        return concept
    
    def reproject(self):
        """
        Reproject all concepts using holographic projection.
        
        Builds similarity matrix from word overlap, then uses SVD
        to construct positions where dot(P[i], P[j]) ≈ S[i,j].
        """
        n = len(self.concepts)
        if n == 0:
            return
        
        # Build similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = self.word_overlap(
                    self.concepts[i].words,
                    self.concepts[j].words
                )
        
        self._similarity_matrix = S
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        # Take top dims eigenvectors
        num_dims = min(n, self.dims)
        idx = np.argsort(eigenvalues)[::-1][:num_dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        
        self._eigenvectors = eigenvectors[:, idx]
        self._eigenvalues = valid_eigenvalues
        
        # Compute positions: P = V @ sqrt(Λ)
        positions = self._eigenvectors * np.sqrt(valid_eigenvalues)
        
        # Update concept positions
        for i, concept in enumerate(self.concepts):
            pos = positions[i]
            if len(pos) < self.dims:
                pos = np.pad(pos, (0, self.dims - len(pos)))
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            concept.position = pos
    
    def project_query(self, text: str) -> np.ndarray:
        """Project a query into the geometric space."""
        if self._eigenvectors is None:
            return np.zeros(self.dims)
        
        query_words = self.extract_words(text)
        n = len(self.concepts)
        
        # Compute similarity to each concept
        similarities = np.zeros(n)
        for i, concept in enumerate(self.concepts):
            similarities[i] = self.word_overlap(query_words, concept.words)
        
        # Project into eigenspace
        pos = similarities @ self._eigenvectors
        
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        
        return pos
    
    def phi_weight(self, value: float, dial: float) -> float:
        """
        Apply φ-dial weighting.
        
        weight = φ^(dial × log(1 + value))
        
        - dial < 0: Inward (rare/specific values weighted higher)
        - dial = 0: Neutral (weight = 1)
        - dial > 0: Outward (common/universal values weighted higher)
        """
        if value <= 0:
            return 1.0
        log_val = np.log1p(value)
        return PHI ** (dial * log_val)
    
    def query(self, text: str, dial: float = 0.0, top_k: int = 5) -> List[Tuple[Concept, float]]:
        """
        Query the space with φ-dial control.
        
        Args:
            text: Query text
            dial: φ-dial setting (-1 to +1)
            top_k: Number of results
            
        Returns:
            List of (concept, weighted_similarity) tuples
        """
        query_pos = self.project_query(text)
        
        results = []
        for concept in self.concepts:
            # Base similarity from position
            similarity = np.dot(query_pos, concept.position)
            
            # Apply φ-dial weighting based on concept frequency
            weight = self.phi_weight(concept.frequency, dial)
            
            # Also weight by word overlap (content gate)
            query_words = self.extract_words(text)
            overlap = self.word_overlap(query_words, concept.words)
            
            # Combined score
            score = similarity * weight * (1 + overlap)
            
            results.append((concept, score))
        
        # Sort by score descending
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def analyze_dimensions(self) -> List[Dict]:
        """
        Analyze what each dimension represents.
        
        For each dimension, find words with highest/lowest loadings.
        """
        if self._eigenvectors is None:
            return []
        
        analyses = []
        n_dims = min(self._eigenvectors.shape[1], 4)  # Analyze top 4 dims
        
        for dim in range(n_dims):
            # Get loadings for this dimension
            loadings = self._eigenvectors[:, dim]
            
            # Find concepts at each pole
            sorted_idx = np.argsort(loadings)
            negative_concepts = [self.concepts[i] for i in sorted_idx[:3]]
            positive_concepts = [self.concepts[i] for i in sorted_idx[-3:][::-1]]
            
            # Collect words at each pole
            negative_words = set()
            for c in negative_concepts:
                negative_words.update(c.words)
            
            positive_words = set()
            for c in positive_concepts:
                positive_words.update(c.words)
            
            # Words unique to each pole
            unique_negative = negative_words - positive_words
            unique_positive = positive_words - negative_words
            
            analyses.append({
                'dimension': dim,
                'eigenvalue': self._eigenvalues[dim] if dim < len(self._eigenvalues) else 0,
                'negative_pole': [c.text[:50] for c in negative_concepts],
                'positive_pole': [c.text[:50] for c in positive_concepts],
                'negative_words': list(unique_negative)[:10],
                'positive_words': list(unique_positive)[:10],
            })
        
        return analyses


def experiment_quaternion_dial():
    """
    Experiment with quaternion φ-dial (4D control).
    
    From Design 044:
    - x: Style (formal ↔ casual)
    - y: Perspective (subjective ↔ meta)
    - z: Depth (terse ↔ elaborate)
    - w: Certainty (definitive ↔ hedged)
    """
    print("\n" + "=" * 70)
    print("QUATERNION φ-DIAL EXPERIMENT")
    print("=" * 70)
    
    # Create space
    space = PhiDialSpace(dims=8)
    
    # Add concepts with different properties
    # Format: (text, style, perspective, depth, certainty)
    # style: -1=formal, +1=casual
    # perspective: -1=subjective, +1=meta
    # depth: -1=terse, +1=elaborate
    # certainty: -1=definitive, +1=hedged
    
    concepts_with_properties = [
        # Formal, objective, terse, definitive
        ("Holmes is a detective.", -1, 0, -1, -1),
        ("Python is a programming language.", -1, 0, -1, -1),
        ("Physics studies matter and energy.", -1, 0, -1, -1),
        
        # Casual, subjective, elaborate, hedged
        ("I think Holmes is probably the smartest detective ever, you know?", 1, -1, 1, 1),
        ("Python seems like it might be a pretty good language for beginners.", 1, -1, 1, 1),
        ("Physics is kind of about how everything in the universe works, I guess.", 1, -1, 1, 1),
        
        # Formal, meta, elaborate, definitive
        ("Holmes represents the archetype of the rational detective in Victorian literature.", -1, 1, 1, -1),
        ("Python exemplifies the design philosophy of code readability and simplicity.", -1, 1, 1, -1),
        ("Physics embodies humanity's quest to understand the fundamental nature of reality.", -1, 1, 1, -1),
        
        # Casual, objective, terse, hedged
        ("Holmes solves crimes, maybe.", 1, 0, -1, 1),
        ("Python's a coding thing.", 1, 0, -1, 1),
        ("Physics is science stuff.", 1, 0, -1, 1),
    ]
    
    # Add concepts and store properties
    for text, style, perspective, depth, certainty in concepts_with_properties:
        c = space.add_concept(text)
        c.style = style
        c.perspective = perspective
        c.depth = depth
        c.certainty = certainty
    
    space.reproject()
    
    print(f"\nAdded {len(space.concepts)} concepts with quaternion properties")
    
    # Test if dimensions correlate with properties
    print("\n" + "-" * 70)
    print("DIMENSION-PROPERTY CORRELATION")
    print("-" * 70)
    
    if space._eigenvectors is not None:
        for dim in range(min(4, space._eigenvectors.shape[1])):
            loadings = space._eigenvectors[:, dim]
            
            # Correlate with each property
            styles = [c.style for c in space.concepts]
            perspectives = [c.perspective for c in space.concepts]
            depths = [c.depth for c in space.concepts]
            certainties = [c.certainty for c in space.concepts]
            
            corr_style = np.corrcoef(loadings, styles)[0, 1]
            corr_persp = np.corrcoef(loadings, perspectives)[0, 1]
            corr_depth = np.corrcoef(loadings, depths)[0, 1]
            corr_cert = np.corrcoef(loadings, certainties)[0, 1]
            
            print(f"\nDimension {dim}:")
            print(f"  Style correlation:       {corr_style:+.3f}")
            print(f"  Perspective correlation: {corr_persp:+.3f}")
            print(f"  Depth correlation:       {corr_depth:+.3f}")
            print(f"  Certainty correlation:   {corr_cert:+.3f}")
            
            # Find strongest correlation
            correlations = [
                (abs(corr_style), 'Style', corr_style),
                (abs(corr_persp), 'Perspective', corr_persp),
                (abs(corr_depth), 'Depth', corr_depth),
                (abs(corr_cert), 'Certainty', corr_cert),
            ]
            best = max(correlations, key=lambda x: x[0])
            print(f"  → Best match: {best[1]} ({best[2]:+.3f})")


def main():
    """Run φ-dial experiment."""
    print("=" * 70)
    print("φ-DIAL DIMENSIONAL CONTROL EXPERIMENT")
    print("=" * 70)
    
    # Create space
    space = PhiDialSpace(dims=8)
    
    # Add concepts with varying specificity/formality
    concepts = [
        # Formal/specific about Holmes
        "Sherlock Holmes is a fictional detective created by Sir Arthur Conan Doyle.",
        "Holmes employs deductive reasoning and forensic science to solve cases.",
        "The character first appeared in A Study in Scarlet published in 1887.",
        
        # Casual/universal about Holmes
        "Holmes is a detective who solves mysteries.",
        "He's really smart and figures things out.",
        "The guy with the pipe and hat.",
        
        # Formal/specific about Python
        "Python is a high-level, interpreted programming language with dynamic semantics.",
        "Python emphasizes code readability with significant whitespace indentation.",
        "Guido van Rossum created Python, first released in 1991.",
        
        # Casual/universal about Python
        "Python is a programming language.",
        "It's easy to learn and write.",
        "Good for beginners.",
        
        # Formal/specific about physics
        "Physics is the natural science studying matter, energy, and their interactions.",
        "Quantum mechanics describes nature at atomic and subatomic scales.",
        "Einstein's theory of relativity revolutionized our understanding of spacetime.",
        
        # Casual/universal about physics
        "Physics is about how things work.",
        "It explains why stuff moves and falls.",
        "Science about the universe.",
    ]
    
    # Add concepts and track which are formal vs casual
    formal_indices = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    casual_indices = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    
    for i, text in enumerate(concepts):
        c = space.add_concept(text)
        # Formal concepts are "rarer" (lower frequency)
        if i in formal_indices:
            c.frequency = 1
        else:
            c.frequency = 10  # Casual concepts are more "common"
    
    # Reproject
    space.reproject()
    
    print(f"\nAdded {len(space.concepts)} concepts")
    print(f"Formal concepts (freq=1): {len(formal_indices)}")
    print(f"Casual concepts (freq=10): {len(casual_indices)}")
    
    # Analyze dimensions
    print("\n" + "=" * 70)
    print("DIMENSION ANALYSIS")
    print("=" * 70)
    
    analyses = space.analyze_dimensions()
    for a in analyses:
        print(f"\nDimension {a['dimension']} (eigenvalue: {a['eigenvalue']:.3f})")
        print(f"  Negative pole: {a['negative_words'][:5]}")
        print(f"  Positive pole: {a['positive_words'][:5]}")
    
    # Test queries with different dial settings
    print("\n" + "=" * 70)
    print("φ-DIAL QUERY TESTS")
    print("=" * 70)
    
    queries = [
        "Who is Holmes?",
        "What is Python?",
        "Tell me about physics",
    ]
    
    dial_settings = [-1.0, 0.0, 1.0]
    dial_names = ["INWARD (specific)", "BALANCED", "OUTWARD (universal)"]
    
    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"Query: {query}")
        print(f"{'─' * 70}")
        
        for dial, name in zip(dial_settings, dial_names):
            results = space.query(query, dial=dial, top_k=3)
            print(f"\n  {name} (dial={dial:+.1f}):")
            for concept, score in results:
                freq_label = "formal" if concept.frequency == 1 else "casual"
                print(f"    [{score:.3f}] ({freq_label}) {concept.text[:55]}...")
    
    # Test φ-weighting directly
    print("\n" + "=" * 70)
    print("φ-WEIGHTING ANALYSIS")
    print("=" * 70)
    
    print("\nφ-weight for different frequencies and dial settings:")
    print(f"{'Freq':>6} | {'dial=-1':>10} | {'dial=0':>10} | {'dial=+1':>10}")
    print("-" * 45)
    
    for freq in [1, 2, 5, 10, 50, 100]:
        w_neg = space.phi_weight(freq, -1.0)
        w_zero = space.phi_weight(freq, 0.0)
        w_pos = space.phi_weight(freq, 1.0)
        print(f"{freq:>6} | {w_neg:>10.4f} | {w_zero:>10.4f} | {w_pos:>10.4f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    print("""
Key observations:
1. dial=-1 (INWARD): Should favor formal/specific concepts (freq=1)
2. dial=0 (BALANCED): Neutral weighting
3. dial=+1 (OUTWARD): Should favor casual/universal concepts (freq=10)

The φ-dial provides unified control over navigation direction.
""")


def experiment_dimension_specific_dial():
    """
    Experiment with dimension-specific φ-dial control.
    
    Instead of a single global dial, we have a dial per dimension.
    This allows controlling style independently from depth, etc.
    """
    print("\n" + "=" * 70)
    print("DIMENSION-SPECIFIC φ-DIAL EXPERIMENT")
    print("=" * 70)
    
    # Create space with concepts that vary along clear dimensions
    space = PhiDialSpace(dims=8)
    
    # Add concepts that vary primarily in ONE dimension each
    # This should create clearer dimensional separation
    
    # TOPIC dimension (Holmes vs Python vs Physics)
    space.add_concept("Holmes is a detective who solves crimes.")
    space.add_concept("Holmes uses deduction to find criminals.")
    space.add_concept("Holmes works with Watson in London.")
    
    space.add_concept("Python is a programming language.")
    space.add_concept("Python uses indentation for code blocks.")
    space.add_concept("Python was created by Guido van Rossum.")
    
    space.add_concept("Physics studies matter and energy.")
    space.add_concept("Physics explains how the universe works.")
    space.add_concept("Physics includes quantum mechanics.")
    
    space.reproject()
    
    print(f"\nAdded {len(space.concepts)} concepts")
    
    # Analyze dimensions
    print("\n" + "-" * 70)
    print("DIMENSION ANALYSIS (Topic Separation)")
    print("-" * 70)
    
    if space._eigenvectors is not None:
        # Check if topics separate along dimensions
        holmes_indices = [0, 1, 2]
        python_indices = [3, 4, 5]
        physics_indices = [6, 7, 8]
        
        for dim in range(min(3, space._eigenvectors.shape[1])):
            loadings = space._eigenvectors[:, dim]
            
            holmes_mean = np.mean([loadings[i] for i in holmes_indices])
            python_mean = np.mean([loadings[i] for i in python_indices])
            physics_mean = np.mean([loadings[i] for i in physics_indices])
            
            print(f"\nDimension {dim}:")
            print(f"  Holmes mean loading:  {holmes_mean:+.3f}")
            print(f"  Python mean loading:  {python_mean:+.3f}")
            print(f"  Physics mean loading: {physics_mean:+.3f}")
            
            # Which topic is most separated?
            spread = max(holmes_mean, python_mean, physics_mean) - min(holmes_mean, python_mean, physics_mean)
            print(f"  Topic spread: {spread:.3f}")
    
    # Test dimension-specific dial
    print("\n" + "-" * 70)
    print("DIMENSION-SPECIFIC DIAL TEST")
    print("-" * 70)
    
    query = "detective"
    query_pos = space.project_query(query)
    
    print(f"\nQuery: '{query}'")
    print(f"Query position: {query_pos[:4]}")
    
    # Apply dial to specific dimensions
    for target_dim in range(min(3, len(query_pos))):
        print(f"\n  Boosting dimension {target_dim}:")
        
        for dial in [-1.0, 0.0, 1.0]:
            # Create modified position
            modified_pos = query_pos.copy()
            
            # Apply φ-dial to just this dimension
            dim_value = abs(modified_pos[target_dim])
            weight = PHI ** (dial * np.log1p(dim_value * 10))  # Scale for visibility
            modified_pos[target_dim] *= weight
            
            # Find nearest concept
            best_concept = None
            best_score = -float('inf')
            for concept in space.concepts:
                score = np.dot(modified_pos, concept.position)
                if score > best_score:
                    best_score = score
                    best_concept = concept
            
            dial_name = "INWARD" if dial < 0 else ("OUTWARD" if dial > 0 else "NEUTRAL")
            print(f"    dial={dial:+.1f} ({dial_name}): {best_concept.text[:50]}...")


def experiment_emergent_dimensions():
    """
    Experiment with emergent dimension discovery.
    
    From Design 080: Dimensions emerge from behavioral patterns.
    We don't predefine them - we discover what the data tells us.
    """
    print("\n" + "=" * 70)
    print("EMERGENT DIMENSION DISCOVERY EXPERIMENT")
    print("=" * 70)
    
    space = PhiDialSpace(dims=8)
    
    # Add concepts with implicit dimensional variation
    # The dimensions should EMERGE from the word patterns
    
    concepts = [
        # Varying formality (implicit)
        "The detective investigates the crime scene.",
        "The guy checks out where it happened.",
        
        # Varying certainty (implicit)
        "Holmes is definitely the greatest detective.",
        "Holmes might be a good detective, perhaps.",
        
        # Varying tense (implicit)
        "Holmes solved the mystery yesterday.",
        "Holmes solves mysteries regularly.",
        "Holmes will solve the next mystery.",
        
        # Varying perspective (implicit)
        "I think Holmes is brilliant.",
        "Holmes is a detective.",
        "Holmes represents the archetype of reason.",
    ]
    
    for text in concepts:
        space.add_concept(text)
    
    space.reproject()
    
    print(f"\nAdded {len(space.concepts)} concepts")
    
    # Analyze what dimensions emerged
    print("\n" + "-" * 70)
    print("EMERGENT DIMENSIONS")
    print("-" * 70)
    
    analyses = space.analyze_dimensions()
    for a in analyses[:3]:
        print(f"\nDimension {a['dimension']} (eigenvalue: {a['eigenvalue']:.3f})")
        print(f"  Negative pole words: {a['negative_words'][:5]}")
        print(f"  Positive pole words: {a['positive_words'][:5]}")
        print(f"  Negative concepts:")
        for c in a['negative_pole'][:2]:
            print(f"    - {c}")
        print(f"  Positive concepts:")
        for c in a['positive_pole'][:2]:
            print(f"    - {c}")
    
    # The key question: Can we CONTROL these emergent dimensions?
    print("\n" + "-" * 70)
    print("CONTROLLING EMERGENT DIMENSIONS")
    print("-" * 70)
    
    query = "Holmes"
    print(f"\nQuery: '{query}'")
    
    # For each dimension, show what happens when we dial it
    for dim in range(min(3, space.dims)):
        print(f"\n  Dimension {dim}:")
        
        query_pos = space.project_query(query)
        
        for dial in [-1.0, 1.0]:
            # Shift query position along this dimension
            shifted_pos = query_pos.copy()
            shift_amount = 0.2 * dial  # Small shift
            shifted_pos[dim] += shift_amount
            
            # Normalize
            norm = np.linalg.norm(shifted_pos)
            if norm > 1e-10:
                shifted_pos = shifted_pos / norm * CRITICAL_LINE
            
            # Find nearest
            best_concept = None
            best_score = -float('inf')
            for concept in space.concepts:
                score = np.dot(shifted_pos, concept.position)
                if score > best_score:
                    best_score = score
                    best_concept = concept
            
            direction = "negative" if dial < 0 else "positive"
            print(f"    Shift {direction}: {best_concept.text[:50]}...")


if __name__ == "__main__":
    main()
    experiment_quaternion_dial()
    experiment_dimension_specific_dial()
    experiment_emergent_dimensions()
