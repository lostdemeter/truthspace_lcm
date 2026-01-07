"""
Geometric Transformer - Pure Geometric Sentence Transformation

Transforms sentences using only geometric operations:
1. ENCODE: Text → Position (via HolographicPatternSpace - similarity-based)
2. TRANSFORM: Position + Delta → New Position
3. DECODE: New Position → Text (via nearest neighbor in corpus)

No regex patterns. No LLM fallback. Pure geometry.

Key insight: Positions are CONSTRUCTED from similarity (word overlap).
The similarity matrix IS the structure. Eigendecomposition gives positions.

If this works, we validate the hypothesis.
If it fails, we learn where the hypothesis breaks down.

Author: Lesley Gushurst
License: GPLv3
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .quaternion_encoder import QuaternionEncoder, QuaternionPosition


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GeometricTransformResult:
    """Result of a geometric transformation."""
    original: str
    transformed: str
    dimension: str
    target_value: str
    words_transformed: List[Tuple[str, str]]  # (original, transformed) pairs
    success: bool
    failure_reason: str = ""


@dataclass 
class WordPosition:
    """A word and its position in geometric space."""
    word: str
    position: np.ndarray
    dimension: str = ""  # Which dimension this word is associated with
    value: str = ""      # Which value (e.g., "past", "future")


# =============================================================================
# GEOMETRIC TRANSFORMER
# =============================================================================

class GeometricTransformer:
    """
    Pure geometric sentence transformer.
    
    Learns transformation deltas from corpus examples.
    Applies transformations via vector addition.
    Decodes via nearest neighbor in vocabulary.
    """
    
    def __init__(self, encoder: QuaternionEncoder = None):
        if encoder is None:
            encoder = QuaternionEncoder()
        
        self.encoder = encoder
        
        # Vocabulary: word -> position (for decoding)
        self._vocabulary: Dict[str, np.ndarray] = {}
        
        # Dimension deltas: (dimension, target_value) -> delta vector
        self._deltas: Dict[Tuple[str, str], np.ndarray] = {}
        
        # Word associations: dimension -> value -> set of words
        self._word_associations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Statistics
        self._corpus_size = 0
    
    def load_corpus(self, path: Path) -> int:
        """
        Load transformation corpus and learn geometric structure.
        
        Returns number of transformations loaded.
        """
        if isinstance(path, str):
            path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        transformations = data.get("transformations", [])
        
        # First pass: collect word pairs from transformations
        # Words that transform together define the geometric structure
        word_pairs: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        # word_pairs[dimension][source_word] = {target_words}
        
        for t in transformations:
            source_words = set(self._tokenize(t["source"]))
            target_words = set(self._tokenize(t["target"]))
            dimension_delta = t.get("dimension_delta", {})
            
            # Words that disappeared -> source value
            # Words that appeared -> target value
            removed = source_words - target_words
            added = target_words - source_words
            
            for dim, (src_val, tgt_val) in dimension_delta.items():
                # Pair removed words with added words (they're transformations)
                for src_word in removed:
                    for tgt_word in added:
                        word_pairs[dim][src_word.lower()].add(tgt_word.lower())
        
        # Build vocabulary with positions derived from transformation relationships
        # Key insight: words that transform to each other should have positions
        # such that position(target) = position(source) + delta
        self._build_geometric_vocabulary(word_pairs, transformations)
        
        # Second pass: compute deltas for each dimension
        delta_accumulator: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
        
        for t in transformations:
            source = t["source"]
            target = t["target"]
            dimension_delta = t.get("dimension_delta", {})
            
            # Encode full sentences
            source_pos = self.encoder.encode(source).to_flat()
            target_pos = self.encoder.encode(target).to_flat()
            delta = target_pos - source_pos
            
            # Store delta for each dimension changed
            for dim, (src_val, tgt_val) in dimension_delta.items():
                key = (dim, tgt_val)
                delta_accumulator[key].append(delta)
                
                # Track word associations
                source_words = set(self._tokenize(source))
                target_words = set(self._tokenize(target))
                
                # Words that disappeared are associated with source value
                for word in source_words - target_words:
                    self._word_associations[dim][src_val].add(word.lower())
                
                # Words that appeared are associated with target value
                for word in target_words - source_words:
                    self._word_associations[dim][tgt_val].add(word.lower())
        
        # Compute canonical delta for each dimension (average)
        for key, deltas in delta_accumulator.items():
            self._deltas[key] = np.mean(deltas, axis=0)
        
        self._corpus_size = len(transformations)
        return self._corpus_size
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        import re
        return re.findall(r'\b[\w\']+\b', text.lower())
    
    def _build_geometric_vocabulary(self, word_pairs: Dict, transformations: List) -> None:
        """
        Build vocabulary with positions derived from transformation relationships.
        
        Key insight from attractor/repeller dynamics:
        - Words that transform together should have related positions
        - position(target) ≈ position(source) + delta
        
        We assign positions such that transformation pairs are geometrically consistent.
        """
        # Collect all unique words
        all_words = set()
        for t in transformations:
            all_words.update(self._tokenize(t["source"]))
            all_words.update(self._tokenize(t["target"]))
        
        # Initialize with encoder positions (dimension-level structure)
        for word in all_words:
            pos = self.encoder.encode(word)
            self._vocabulary[word.lower()] = pos.to_flat()
        
        # Now refine positions based on transformation pairs
        # Words that transform to each other should have consistent delta
        
        # For each dimension, identify source and target word clusters
        for dim, src_to_tgt in word_pairs.items():
            # Get all source words and their targets
            source_words = set(src_to_tgt.keys())
            target_words = set()
            for targets in src_to_tgt.values():
                target_words.update(targets)
            
            # Assign distinct positions within each cluster
            # Source words get base positions
            # Target words get base + canonical_delta positions
            
            # Use φ-based spacing for distinctness
            phi = 1.618033988749895
            
            # Assign source words distinct positions
            for i, word in enumerate(sorted(source_words)):
                if word in self._vocabulary:
                    # Add small offset to make words distinct
                    # Use different dimensions for different words
                    offset = np.zeros(len(self._vocabulary[word]))
                    # Spread across dimensions using φ
                    dim_idx = i % len(offset)
                    offset[dim_idx] = phi ** (-(i // len(offset) + 1))
                    self._vocabulary[word] = self._vocabulary[word] + offset
                    
                    # Track this word as associated with source value
                    self._word_associations[dim]["source"].add(word)
            
            # Assign target words positions that are delta away from sources
            for i, word in enumerate(sorted(target_words)):
                if word in self._vocabulary:
                    offset = np.zeros(len(self._vocabulary[word]))
                    dim_idx = i % len(offset)
                    offset[dim_idx] = phi ** (-(i // len(offset) + 1))
                    self._vocabulary[word] = self._vocabulary[word] + offset
                    
                    # Track this word as associated with target value
                    self._word_associations[dim]["target"].add(word)
    
    def transform(self, text: str, dimension: str, target_value: str) -> GeometricTransformResult:
        """
        Transform text along a dimension using pure geometry.
        
        1. Tokenize into words
        2. For each word that should transform:
           a. Get word position
           b. Add delta
           c. Find nearest word in vocabulary
        3. Reconstruct sentence
        """
        key = (dimension, target_value)
        
        if key not in self._deltas:
            return GeometricTransformResult(
                original=text,
                transformed=text,
                dimension=dimension,
                target_value=target_value,
                words_transformed=[],
                success=False,
                failure_reason=f"No delta learned for {dimension}={target_value}"
            )
        
        delta = self._deltas[key]
        words = self._tokenize(text)
        
        # Identify which words should transform
        # A word should transform if it's associated with a different value of this dimension
        transformable_values = set(self._word_associations.get(dimension, {}).keys())
        transformable_values.discard(target_value)  # Don't transform words already at target
        
        result_words = []
        words_transformed = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if this word is associated with a source value
            should_transform = False
            for val in transformable_values:
                if word_lower in self._word_associations[dimension][val]:
                    should_transform = True
                    break
            
            if should_transform and word_lower in self._vocabulary:
                # Transform geometrically
                word_pos = self._vocabulary[word_lower]
                new_pos = word_pos + delta
                
                # Find nearest word in vocabulary
                nearest_word = self._nearest_neighbor(new_pos)
                
                if nearest_word and nearest_word != word_lower:
                    result_words.append(nearest_word)
                    words_transformed.append((word, nearest_word))
                else:
                    result_words.append(word)
            else:
                result_words.append(word)
        
        transformed = " ".join(result_words)
        
        return GeometricTransformResult(
            original=text,
            transformed=transformed,
            dimension=dimension,
            target_value=target_value,
            words_transformed=words_transformed,
            success=len(words_transformed) > 0 or text.lower() == transformed.lower(),
        )
    
    def _nearest_neighbor(self, position: np.ndarray) -> Optional[str]:
        """Find nearest word in vocabulary to given position."""
        best_word = None
        best_dist = float('inf')
        
        for word, word_pos in self._vocabulary.items():
            dist = np.linalg.norm(position - word_pos)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        
        return best_word
    
    def get_delta(self, dimension: str, target_value: str) -> Optional[np.ndarray]:
        """Get the canonical delta for a dimension transformation."""
        return self._deltas.get((dimension, target_value))
    
    def stats(self) -> Dict:
        """Get statistics about learned structure."""
        return {
            "corpus_size": self._corpus_size,
            "vocabulary_size": len(self._vocabulary),
            "deltas_learned": list(self._deltas.keys()),
            "dimensions": list(set(d for d, _ in self._deltas.keys())),
        }
    
    def test_self_similarity(self, dimension: str, target_value: str) -> Dict:
        """
        Test if the delta for a dimension is self-similar.
        
        Returns statistics about delta consistency across word pairs.
        """
        key = (dimension, target_value)
        if key not in self._deltas:
            return {"error": f"No delta for {dimension}={target_value}"}
        
        canonical_delta = self._deltas[key]
        
        # Find word pairs for this transformation
        source_words = self._word_associations.get(dimension, {})
        
        # Get all source values (not target)
        source_values = [v for v in source_words.keys() if v != target_value]
        
        if not source_values:
            return {"error": "No source values found"}
        
        # For each source word, find if there's a corresponding target word
        # and check if the delta is consistent
        results = []
        
        for src_val in source_values:
            for src_word in source_words[src_val]:
                if src_word not in self._vocabulary:
                    continue
                
                src_pos = self._vocabulary[src_word]
                expected_pos = src_pos + canonical_delta
                
                # Find nearest target word
                nearest = self._nearest_neighbor(expected_pos)
                if nearest:
                    nearest_pos = self._vocabulary[nearest]
                    actual_delta = nearest_pos - src_pos
                    deviation = np.linalg.norm(actual_delta - canonical_delta)
                    
                    results.append({
                        "source": src_word,
                        "target": nearest,
                        "deviation": deviation,
                    })
        
        if not results:
            return {"error": "No word pairs found"}
        
        deviations = [r["deviation"] for r in results]
        
        return {
            "dimension": dimension,
            "target_value": target_value,
            "pairs_tested": len(results),
            "mean_deviation": np.mean(deviations),
            "max_deviation": np.max(deviations),
            "min_deviation": np.min(deviations),
            "is_self_similar": np.max(deviations) < 0.1,  # Threshold for "same"
            "sample_pairs": results[:5],
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_geometric_transformer(corpus_path: Path = None) -> GeometricTransformer:
    """Load geometric transformer with default corpus."""
    if corpus_path is None:
        corpus_path = Path(__file__).parent.parent / "corpus" / "transformation_corpus.json"
    
    transformer = GeometricTransformer()
    
    if corpus_path.exists():
        transformer.load_corpus(corpus_path)
    
    return transformer
