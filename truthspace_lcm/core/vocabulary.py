"""
Vocabulary System for Geometric Chat System

Implements hash-based word positions with IDF weighting as specified in the SDS.

Core formulas:
- Word Position: pos(w) = hash(w) → ℝ^dim (deterministic)
- IDF Weight: w = 1 / log(1 + count)
- Text Encoding: enc(t) = Σᵢ wᵢ·pos(wordᵢ) / Σᵢ wᵢ
"""

import numpy as np
import re
import math
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
from dataclasses import dataclass, field


# Default dimensionality of semantic space
DEFAULT_DIM = 64


def tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r'\w+', text.lower())


def word_position(word: str, dim: int) -> np.ndarray:
    """
    Assign deterministic position to word.
    
    Uses hash as random seed for reproducibility.
    Position is on unit hypersphere.
    """
    seed = hash(word) % (2**32)
    rng = np.random.default_rng(seed)
    position = rng.standard_normal(dim)
    
    # Normalize to unit sphere
    norm = np.linalg.norm(position)
    if norm > 0:
        position = position / norm
    
    return position


@dataclass
class Vocabulary:
    """
    Word embedding vocabulary.
    
    Stores word positions and counts for IDF weighting.
    """
    dim: int = DEFAULT_DIM
    word_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    word_counts: Counter = field(default_factory=Counter)
    total_docs: int = 0
    
    def get_position(self, word: str) -> np.ndarray:
        """Get or create position for word."""
        word = word.lower()
        if word not in self.word_positions:
            self.word_positions[word] = word_position(word, self.dim)
        return self.word_positions[word]
    
    def add_text(self, text: str, style_bias: np.ndarray = None, bias_strength: float = 0.15):
        """Add text to vocabulary, optionally with style bias."""
        words = tokenize(text)
        for word in words:
            pos = self.get_position(word)
            self.word_counts[word] += 1
            if style_bias is not None:
                # Bias word position toward style centroid
                self.word_positions[word] = (1 - bias_strength) * pos + bias_strength * style_bias
        self.total_docs += 1
    
    def idf_weight(self, word: str) -> float:
        """
        Compute IDF-like weight for word.
        
        Rare words get higher weight.
        Formula: w = 1 / log(1 + count)
        """
        count = self.word_counts.get(word.lower(), 1)
        return 1.0 / math.log(1 + count)
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text as IDF-weighted average of word positions.
        
        Formula:
            v = Σᵢ wᵢ · pos(wordᵢ) / Σᵢ wᵢ
            where wᵢ = 1 / log(1 + count(wordᵢ))
        """
        words = tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        positions = []
        weights = []
        
        for word in words:
            pos = self.get_position(word)
            positions.append(pos)
            weights.append(self.idf_weight(word))
        
        positions = np.array(positions)
        weights = np.array(weights)
        
        # Normalize weights
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        
        # Weighted average
        return np.average(positions, axis=0, weights=weights)
    
    def nearest_words(self, vector: np.ndarray, k: int = 10, 
                      exclude: Set[str] = None) -> List[Tuple[str, float]]:
        """Find k nearest words to vector."""
        if exclude is None:
            exclude = set()
        
        results = []
        vec_norm = np.linalg.norm(vector)
        if vec_norm < 1e-8:
            return []
        
        for word, pos in self.word_positions.items():
            if word in exclude:
                continue
            pos_norm = np.linalg.norm(pos)
            if pos_norm < 1e-8:
                continue
            sim = np.dot(vector, pos) / (vec_norm * pos_norm)
            results.append((word, float(sim)))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        v1 = self.encode(text1)
        v2 = self.encode(text2)
        return cosine_similarity(v1, v2)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Formula:
        cos(θ) = (v1 · v2) / (‖v1‖ · ‖v2‖)
    
    Range: [-1, 1]
        1  = identical direction
        0  = orthogonal (unrelated)
        -1 = opposite direction
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    return float(np.dot(v1, v2) / (norm1 * norm2))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Formula:
        d = ‖v1 - v2‖ = √(Σᵢ (v1ᵢ - v2ᵢ)²)
    """
    return float(np.linalg.norm(v1 - v2))
