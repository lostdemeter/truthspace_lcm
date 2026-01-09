"""
Holographic Transformer - Pure Geometric Sentence Transformation

Uses HolographicPatternSpace for content-aware encoding:
1. ENCODE: Text → Position (via similarity-based eigendecomposition)
2. TRANSFORM: Position + Delta → New Position  
3. DECODE: New Position → Text (via nearest neighbor in corpus)

Key insight from Design 104 and holographic_pattern_space.py:
- Positions are CONSTRUCTED from similarity (word overlap)
- The similarity matrix IS the structure
- Eigendecomposition gives positions where dot(P[i], P[j]) ≈ similarity(i, j)

No regex patterns. No LLM fallback. Pure geometry.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TransformationExample:
    """A source-target transformation pair."""
    source: str
    target: str
    dimension: str
    source_value: str
    target_value: str
    source_words: Set[str] = field(default_factory=set)
    target_words: Set[str] = field(default_factory=set)
    source_position: Optional[np.ndarray] = None
    target_position: Optional[np.ndarray] = None


@dataclass
class HolographicTransformResult:
    """Result of a holographic transformation."""
    original: str
    transformed: str
    dimension: str
    target_value: str
    confidence: float
    method: str  # "geometric", "no_match"
    nearest_source: str = ""
    nearest_target: str = ""


# =============================================================================
# HOLOGRAPHIC TRANSFORMER
# =============================================================================

# Golden ratio for φ-Zipf duality (Design 039)
PHI = (1 + np.sqrt(5)) / 2

# Bootstrap filler words - used only until corpus is loaded
# After load_corpus(), filler words are derived via φ-Zipf duality
BOOTSTRAP_FILLER = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and', 'or', 'in'}

# φ-weight threshold: words with φ^(-rank) below this are filler
# This replaces statistical document frequency with geometric φ-based weighting
# High frequency words have LOW rank (rank 1 = most frequent) → HIGH φ weight
# So we mark words with HIGH φ-weight as filler (they're the most frequent)
# φ^(-1) = 0.618, φ^(-5) = 0.09, φ^(-10) = 0.008
# Threshold 0.1 means top ~5 most frequent words are filler
PHI_FILLER_WEIGHT_THRESHOLD = 0.1


class HolographicTransformer:
    """
    Pure geometric sentence transformer using holographic projection.
    
    Positions are constructed from word overlap similarity.
    Transformations are vector additions.
    Decoding is nearest neighbor search.
    
    Filler words are derived via φ-Zipf duality (Design 039):
    - Words are ranked by frequency
    - φ^(-rank) gives geometric weight
    - Low weight words (high frequency) are filler
    - This follows the principle: Structure IS information
    - The geometry IS the weighting (not statistics)
    """
    
    def __init__(self, dims: int = 32):
        self.dims = dims
        
        # All sentences (source and target) with positions
        self._sentences: Dict[str, np.ndarray] = {}  # text -> position
        self._sentence_words: Dict[str, Set[str]] = {}  # text -> words
        
        # Transformation examples grouped by (dimension, target_value)
        self._examples: Dict[Tuple[str, str], List[TransformationExample]] = {}
        
        # Canonical deltas per transformation type
        self._deltas: Dict[Tuple[str, str], np.ndarray] = {}
        
        # Similarity matrix and positions (recomputed on load)
        self._similarity_matrix: Optional[np.ndarray] = None
        self._positions: Optional[np.ndarray] = None
        self._sentence_list: List[str] = []  # Ordered list for indexing
        
        # Emergent filler words - derived via φ-Zipf duality
        self._filler_words: Set[str] = BOOTSTRAP_FILLER.copy()
        self._word_phi_weight: Dict[str, float] = {}  # word -> φ^(-rank) weight
        self._word_frequency: Dict[str, int] = {}  # word -> raw frequency
    
    def _extract_all_words(self, text: str) -> Set[str]:
        """Extract ALL words from text (no filtering)."""
        return set(re.findall(r'\b[a-z]+\b', text.lower()))
    
    def extract_words(self, text: str) -> Set[str]:
        """Extract content words from text (filler removed)."""
        words = self._extract_all_words(text)
        return words - self._filler_words
    
    @staticmethod
    def word_overlap(words1: Set[str], words2: Set[str]) -> float:
        """Jaccard similarity between word sets."""
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def _derive_filler_words(self, sentences: List[str]) -> Set[str]:
        """
        Derive filler words via φ-Zipf duality (Design 039).
        
        Instead of statistical document frequency, we use geometric φ-weighting:
        1. Count word frequencies across corpus
        2. Rank words by frequency (most frequent = rank 1)
        3. Compute φ^(-rank) weight for each word
        4. Words with weight below threshold are filler
        
        This follows the principle: The geometry IS the weighting.
        φ^n for encoding (outward), φ^(-n) for weighting (inward).
        Same fractal, opposite directions.
        """
        from collections import Counter
        
        if not sentences:
            return BOOTSTRAP_FILLER.copy()
        
        # Count total word occurrences (not document frequency)
        word_freq: Counter = Counter()
        for sentence in sentences:
            words = self._extract_all_words(sentence)
            word_freq.update(words)
        
        # Store raw frequencies
        self._word_frequency = dict(word_freq)
        
        # Rank words by frequency (most frequent = rank 1)
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        
        # Compute φ^(-rank) weight and identify filler
        filler = set()
        for rank, (word, freq) in enumerate(sorted_words, 1):
            # φ-Zipf duality: weight = φ^(-rank)
            # Rank 1 (most frequent) → φ^(-1) = 0.618 (high weight)
            # Rank 100 (rare) → φ^(-100) ≈ 0 (low weight)
            phi_weight = PHI ** (-rank)
            self._word_phi_weight[word] = phi_weight
            
            # HIGH weight = HIGH frequency = filler
            # Words with φ-weight >= threshold are the most frequent → filler
            if phi_weight >= PHI_FILLER_WEIGHT_THRESHOLD:
                filler.add(word)
        
        # Always include bootstrap filler (structural words)
        filler.update(BOOTSTRAP_FILLER)
        
        return filler
    
    def load_corpus(self, path: Path) -> int:
        """
        Load transformation corpus and construct geometric space.
        
        Filler words are derived emergently from corpus document frequency.
        
        Returns number of transformations loaded.
        """
        if isinstance(path, str):
            path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        transformations = data.get("transformations", [])
        
        # First pass: collect all sentences to derive filler words
        all_sentences = []
        for t in transformations:
            all_sentences.append(t["source"])
            all_sentences.append(t["target"])
        
        # Derive filler words from corpus statistics (emergent, not hard-coded)
        self._filler_words = self._derive_filler_words(all_sentences)
        
        # Second pass: collect sentences with filler-filtered words
        for t in transformations:
            source = t["source"]
            target = t["target"]
            
            self._sentence_words[source] = self.extract_words(source)
            self._sentence_words[target] = self.extract_words(target)
            
            # Store transformation examples
            dimension_delta = t.get("dimension_delta", {})
            for dim, (src_val, tgt_val) in dimension_delta.items():
                key = (dim, tgt_val)
                if key not in self._examples:
                    self._examples[key] = []
                
                example = TransformationExample(
                    source=source,
                    target=target,
                    dimension=dim,
                    source_value=src_val,
                    target_value=tgt_val,
                    source_words=self._sentence_words[source],
                    target_words=self._sentence_words[target],
                )
                self._examples[key].append(example)
        
        # Build sentence list for matrix indexing
        self._sentence_list = list(self._sentence_words.keys())
        
        # Construct positions from similarity
        self._construct_positions()
        
        # Compute canonical deltas
        self._compute_deltas()
        
        return len(transformations)
    
    def _construct_positions(self):
        """
        Construct positions from similarity matrix using eigendecomposition.
        
        This is the key geometric insight:
        - Define similarity as word overlap
        - Eigendecompose to get positions
        - Now: dot(P[i], P[j]) ≈ similarity(i, j) by construction
        """
        n = len(self._sentence_list)
        if n == 0:
            return
        
        # Build similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    S[i, j] = 1.0
                else:
                    S[i, j] = self.word_overlap(
                        self._sentence_words[self._sentence_list[i]],
                        self._sentence_words[self._sentence_list[j]]
                    )
        
        self._similarity_matrix = S
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top dims dimensions
        k = min(self.dims, n)
        eigenvalues_k = np.maximum(eigenvalues[:k], 0)  # Clamp negatives
        
        self._positions = eigenvectors[:, :k] * np.sqrt(eigenvalues_k)
        
        # Pad to full dims if needed
        if k < self.dims:
            padding = np.zeros((n, self.dims - k))
            self._positions = np.hstack([self._positions, padding])
        
        # Store positions in dictionary
        for i, sent in enumerate(self._sentence_list):
            self._sentences[sent] = self._positions[i]
    
    def _compute_deltas(self):
        """Compute canonical delta for each transformation type."""
        for key, examples in self._examples.items():
            deltas = []
            for ex in examples:
                if ex.source in self._sentences and ex.target in self._sentences:
                    src_pos = self._sentences[ex.source]
                    tgt_pos = self._sentences[ex.target]
                    delta = tgt_pos - src_pos
                    deltas.append(delta)
            
            if deltas:
                self._deltas[key] = np.mean(deltas, axis=0)
    
    def project_query(self, query_text: str) -> np.ndarray:
        """
        Project a query into the space based on similarity to known sentences.
        """
        query_words = self.extract_words(query_text)
        
        if not self._sentence_list or self._positions is None:
            return np.zeros(self.dims)
        
        # Compute similarity to each known sentence
        similarities = np.array([
            self.word_overlap(query_words, self._sentence_words[sent])
            for sent in self._sentence_list
        ])
        
        # Weighted average of positions
        if np.sum(similarities) > 0:
            query_pos = similarities @ self._positions / (np.sum(similarities) + 1e-10)
        else:
            query_pos = np.zeros(self.dims)
        
        return query_pos
    
    def find_nearest(self, position: np.ndarray, k: int = 1) -> List[Tuple[str, float]]:
        """Find nearest sentences to a position."""
        if self._positions is None:
            return []
        
        distances = np.linalg.norm(self._positions - position, axis=1)
        indices = np.argsort(distances)[:k]
        
        return [(self._sentence_list[i], distances[i]) for i in indices]
    
    def transform(self, text: str, dimension: str, target_value: str) -> HolographicTransformResult:
        """
        Transform text along a dimension using pure geometry.
        
        1. Project query to get position
        2. Find nearest source sentence
        3. Add delta to source position
        4. Find nearest target sentence
        """
        key = (dimension, target_value)
        
        if key not in self._deltas:
            return HolographicTransformResult(
                original=text,
                transformed=text,
                dimension=dimension,
                target_value=target_value,
                confidence=0.0,
                method="no_delta",
            )
        
        delta = self._deltas[key]
        
        # Project query into space
        query_pos = self.project_query(text)
        
        # Find nearest source sentence (for reference)
        nearest_sources = self.find_nearest(query_pos, k=1)
        nearest_source = nearest_sources[0][0] if nearest_sources else ""
        source_dist = nearest_sources[0][1] if nearest_sources else float('inf')
        
        # Apply transformation
        transformed_pos = query_pos + delta
        
        # Find nearest sentence to transformed position
        nearest_targets = self.find_nearest(transformed_pos, k=1)
        nearest_target = nearest_targets[0][0] if nearest_targets else text
        target_dist = nearest_targets[0][1] if nearest_targets else float('inf')
        
        # Confidence based on how close we are to known sentences
        # Lower distance = higher confidence
        confidence = 1.0 / (1.0 + target_dist)
        
        return HolographicTransformResult(
            original=text,
            transformed=nearest_target,
            dimension=dimension,
            target_value=target_value,
            confidence=confidence,
            method="geometric",
            nearest_source=nearest_source,
            nearest_target=nearest_target,
        )
    
    def stats(self) -> Dict:
        """Get statistics about the transformer."""
        return {
            "sentences": len(self._sentences),
            "dimensions": self.dims,
            "transformations": list(self._deltas.keys()),
            "examples_per_transform": {
                f"{d}={v}": len(ex) for (d, v), ex in self._examples.items()
            },
            "filler_words": len(self._filler_words),
            "filler_words_emergent": len(self._filler_words - BOOTSTRAP_FILLER),
        }
    
    def get_filler_words(self) -> Dict[str, any]:
        """
        Get information about filler words (for debugging/inspection).
        
        Returns dict with:
        - bootstrap: words from bootstrap set
        - emergent: words derived via φ-Zipf duality
        - all: combined set
        - phi_weights: φ^(-rank) weight for each word
        """
        emergent = self._filler_words - BOOTSTRAP_FILLER
        return {
            "bootstrap": sorted(BOOTSTRAP_FILLER),
            "emergent": sorted(emergent),
            "all": sorted(self._filler_words),
            "total": len(self._filler_words),
            "phi_weight_threshold": PHI_FILLER_WEIGHT_THRESHOLD,
            "phi_weights": sorted(
                [(w, self._word_phi_weight.get(w, 0), self._word_frequency.get(w, 0)) 
                 for w in self._filler_words if w in self._word_phi_weight],
                key=lambda x: -x[2]  # Sort by frequency
            )[:20],  # Top 20 filler words by frequency
            "low_weight_words": sorted(
                [(w, weight, self._word_frequency.get(w, 0)) 
                 for w, weight in self._word_phi_weight.items() 
                 if weight < PHI_FILLER_WEIGHT_THRESHOLD],
                key=lambda x: -x[1]  # Sort by φ-weight (highest first)
            )[:10],  # Top 10 content words (low φ-weight = rare = meaningful)
        }
    
    def test_accuracy(self) -> Dict:
        """
        Test transformation accuracy on the corpus itself.
        
        For each source sentence with a known target, check if
        transform(source) finds the correct target.
        """
        results = {"total": 0, "correct": 0, "by_dimension": {}}
        
        for key, examples in self._examples.items():
            dim, tgt_val = key
            dim_key = f"{dim}={tgt_val}"
            
            correct = 0
            total = 0
            
            for ex in examples:
                result = self.transform(ex.source, dim, tgt_val)
                total += 1
                
                if result.transformed == ex.target:
                    correct += 1
            
            results["by_dimension"][dim_key] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else 0.0,
            }
            results["total"] += total
            results["correct"] += correct
        
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_holographic_transformer(corpus_path: Path = None) -> HolographicTransformer:
    """Load holographic transformer with default corpus."""
    if corpus_path is None:
        corpus_path = Path(__file__).parent.parent / "corpus" / "transformation_corpus.json"
    
    transformer = HolographicTransformer()
    
    if corpus_path.exists():
        transformer.load_corpus(corpus_path)
    
    return transformer
