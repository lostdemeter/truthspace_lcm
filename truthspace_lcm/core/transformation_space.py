"""
Geometric Transformation Space (Design 112 - Music Box Principle)

Transforms text using geometric vocabulary lookup instead of hard-coded patterns.
The transformation emerges from find_nearest(position + delta).

No word->word mappings. The music emerges from the geometry.

Author: Lesley Gushurst
License: GPLv3
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict

import numpy as np

from .geometric_vocabulary import (
    GeometricVocabulary,
    get_default_vocabulary,
    TENSE_DELTAS,
    FORMALITY_DELTAS,
    TENSE,
    FORMALITY,
    DOMAIN,
    INTENSITY,
)
from .quaternion_encoder import QuaternionEncoder


@dataclass
class TransformationResult:
    """Result of a transformation."""
    original: str
    transformed: str
    dimension: str
    target_value: str
    confidence: float
    method: str = "geometric"
    word_changes: List[Tuple[str, str]] = None
    needs_llm: bool = False
    expected_changes: int = 0
    coverage: float = 1.0
    
    def __post_init__(self):
        if self.word_changes is None:
            self.word_changes = []


# Dimension-specific deltas
# These map (dimension, target_value) -> delta vector
TRANSFORMATION_DELTAS = {
    # Tense transformations
    ("tense", "future"): np.array([2, 0, 0, 0]),   # past->future or present->future
    ("tense", "present"): np.array([1, 0, 0, 0]),  # past->present
    ("tense", "past"): np.array([-1, 0, 0, 0]),    # present->past
    
    # Formality transformations
    ("formality", "formal"): np.array([0, 1, 0, 0]),
    ("formality", "casual"): np.array([0, -1, 0, 0]),
    ("formality", "archaic"): np.array([0, 2, 0, 0]),
    
    # Regality (formality + domain shift)
    ("regality", "royal"): np.array([0, 2, 1, 0]),
    ("regality", "noble"): np.array([0, 1, 0, 0]),
    ("regality", "common"): np.array([0, -1, 0, 0]),
    
    # Domain transformations
    ("domain", "technical"): np.array([0, 0, 1, 0]),
    ("domain", "sacred"): np.array([0, 1, 2, 0]),
    ("domain", "mundane"): np.array([0, -1, -1, 0]),
    
    # Intensity transformations
    ("intensity", "strong"): np.array([0, 0, 0, 1]),
    ("intensity", "weak"): np.array([0, 0, 0, -1]),
}


class GeometricTransformationSpace:
    """
    Transformation space using geometric vocabulary (Design 112).
    
    Instead of pattern-based word->word mappings, transformations
    emerge from find_nearest(position + delta).
    
    The vocabulary is the DRUM.
    The find_nearest is the COMB.
    The output is the MUSIC.
    """
    
    def __init__(self, vocab: Optional[GeometricVocabulary] = None):
        self.vocab = vocab or get_default_vocabulary()
        self._dimensions_learned = set(TRANSFORMATION_DELTAS.keys())
        
        # For compatibility with existing code
        self._corpus_size = 0
        self._deltas = defaultdict(dict)
        self._vocabulary = defaultdict(lambda: defaultdict(dict))
    
    def transform(self, text: str, dimension: str, target_value: str) -> TransformationResult:
        """
        Transform text along a dimension using geometric lookup.
        
        Args:
            text: Input sentence
            dimension: Dimension to transform (e.g., "tense", "formality")
            target_value: Target value (e.g., "future", "formal")
            
        Returns:
            TransformationResult with transformed text
        """
        # Get the delta for this transformation
        delta = TRANSFORMATION_DELTAS.get((dimension, target_value))
        
        if delta is None:
            # Unknown transformation - return unchanged
            return TransformationResult(
                original=text,
                transformed=text,
                dimension=dimension,
                target_value=target_value,
                confidence=0.0,
                method="unknown",
                needs_llm=True,
            )
        
        # Transform each word using geometric lookup
        transformed, word_changes = self._transform_text(text, delta)
        
        # Calculate confidence based on how many words changed
        words = re.findall(r'\b[\w\'-]+\b', text)
        vocab_words = sum(1 for w in words if self.vocab.has_word(w))
        
        if vocab_words > 0:
            coverage = len(word_changes) / vocab_words
        else:
            coverage = 1.0 if len(word_changes) == 0 else 0.5
        
        return TransformationResult(
            original=text,
            transformed=transformed,
            dimension=dimension,
            target_value=target_value,
            confidence=min(1.0, coverage + 0.5),  # Boost confidence since geometric is reliable
            method="geometric",
            word_changes=word_changes,
            needs_llm=False,
            expected_changes=vocab_words,
            coverage=coverage,
        )
    
    def _transform_text(self, text: str, delta: np.ndarray) -> Tuple[str, List[Tuple[str, str]]]:
        """Transform text word by word using geometric lookup."""
        # Split into tokens preserving punctuation and whitespace
        tokens = re.findall(r'\b[\w\'-]+\b|[^\w\s]+|\s+', text)
        
        result = []
        word_changes = []
        
        for token in tokens:
            # Skip whitespace and punctuation
            if not token.strip() or not token[0].isalnum():
                result.append(token)
                continue
            
            # Try to transform the word
            transformed = self.vocab.transform(token, delta)
            
            if transformed and transformed.lower() != token.lower():
                # Preserve original capitalization
                if token.isupper():
                    transformed = transformed.upper()
                elif token[0].isupper():
                    transformed = transformed.capitalize()
                
                result.append(transformed)
                word_changes.append((token, transformed))
            else:
                result.append(token)
        
        return ''.join(result), word_changes
    
    def transform_multi(self, text: str, 
                        transformations: List[Tuple[str, str]]) -> TransformationResult:
        """Apply multiple transformations in sequence."""
        current = text
        all_changes = []
        total_coverage = 0.0
        
        for dim, target_val in transformations:
            result = self.transform(current, dim, target_val)
            current = result.transformed
            all_changes.extend(result.word_changes)
            total_coverage += result.coverage
        
        avg_coverage = total_coverage / len(transformations) if transformations else 1.0
        
        return TransformationResult(
            original=text,
            transformed=current,
            dimension="+".join(d for d, _ in transformations),
            target_value="+".join(v for _, v in transformations),
            confidence=min(1.0, avg_coverage + 0.3),
            method="geometric_multi",
            word_changes=all_changes,
            needs_llm=False,
            expected_changes=len(all_changes),
            coverage=avg_coverage,
        )
    
    def available_dimensions(self) -> List[str]:
        """Get list of available dimensions."""
        dims = set()
        for (dim, _) in TRANSFORMATION_DELTAS.keys():
            dims.add(dim)
        return sorted(dims)
    
    def available_values(self, dimension: str) -> List[str]:
        """Get available target values for a dimension."""
        values = []
        for (dim, val) in TRANSFORMATION_DELTAS.keys():
            if dim == dimension:
                values.append(val)
        return sorted(values)
    
    def stats(self) -> Dict[str, Any]:
        """Get transformation space statistics."""
        return {
            'corpus_size': self._corpus_size,
            'dimensions_learned': list(self.available_dimensions()),
            'vocabulary_size': self.vocab.stats()['total_words'],
            'total_deltas': len(TRANSFORMATION_DELTAS),
            'method': 'geometric',
        }
    
    def get_missing_words(self, text: str, dimension: str) -> Set[str]:
        """Get words that aren't in the vocabulary."""
        words = set(re.findall(r'\b[\w\'-]+\b', text.lower()))
        missing = set()
        for word in words:
            if not self.vocab.has_word(word):
                missing.add(word)
        return missing
    
    # Compatibility methods for existing code
    def load_corpus(self, path) -> int:
        """Compatibility: corpus is now the geometric vocabulary."""
        return self.vocab.stats()['total_words']
    
    def learn_from_llm_result(self, source: str, target: str,
                               dimension: str, target_value: str) -> int:
        """Learn new words from LLM result by adding to vocabulary."""
        # Extract word mappings
        src_words = set(re.findall(r'\b[\w\'-]+\b', source.lower()))
        tgt_words = set(re.findall(r'\b[\w\'-]+\b', target.lower()))
        
        # Find words that changed
        removed = src_words - tgt_words
        added = tgt_words - src_words
        
        # Get the delta for this transformation
        delta = TRANSFORMATION_DELTAS.get((dimension, target_value))
        if delta is None:
            return 0
        
        learned = 0
        # For each removed word, if we know its position, add the new word
        for src_word in removed:
            src_pos = self.vocab.get_position(src_word)
            if src_pos is not None:
                # Find the most likely target word (closest to expected position)
                expected_pos = src_pos + delta
                best_tgt = None
                best_dist = float('inf')
                
                for tgt_word in added:
                    if not self.vocab.has_word(tgt_word):
                        # This is a new word - estimate its position
                        dist = 0  # Would need more context to estimate
                        if dist < best_dist:
                            best_dist = dist
                            best_tgt = tgt_word
                
                if best_tgt:
                    # Add the new word at the expected position
                    concept = self.vocab.get_concept(src_word)
                    self.vocab.add_word(best_tgt, expected_pos, concept)
                    learned += 1
        
        return learned


def load_transformation_space() -> GeometricTransformationSpace:
    """Load the geometric transformation space."""
    return GeometricTransformationSpace()


# Backward compatibility alias
TransformationSpace = GeometricTransformationSpace
