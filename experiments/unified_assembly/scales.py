#!/usr/bin/env python3
"""
Multi-Scale Dimension Architecture

This module provides scale-aware transformations and the fractal dimension
space that exhibits self-similarity across scales.

The key insight: The SAME dimension types appear at multiple scales.
"Formality" exists at word, phrase, sentence, and document levels.
This is true fractal structure.

Classes:
- ScaleAwareTransformer: Apply transforms at the appropriate scale
- FractalDimensionSpace: Cross-scale dimension management
- ScaleDetector: Automatic scale detection from text

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set
import re

from experiments.unified_assembly.core import (
    UnifiedCorpus,
    Scale,
    DimensionType,
    ScaledDimension,
    SCALE_DIMENSIONS,
    PHI,
)


# =============================================================================
# SCALE-AWARE TRANSFORMER
# =============================================================================

class ScaleAwareTransformer:
    """
    Applies transformations at the appropriate scale.
    
    Key insight: A transformation at one scale may need to be
    applied differently depending on the actual text structure.
    """
    
    def __init__(self, corpus: UnifiedCorpus):
        self.corpus = corpus
        
        # Transform functions by (scale, dimension)
        self._transforms: Dict[Tuple[Scale, str], Callable[[str, float], str]] = {}
    
    def register_transform(self, scale: Scale, dimension: str,
                          transform_fn: Callable[[str, float], str]):
        """
        Register a transformation function.
        
        Args:
            scale: Scale at which this transform operates
            dimension: Dimension name
            transform_fn: Function(text, amount) -> transformed_text
                         amount is in range [-1, 1] for negative to positive pole
        """
        self._transforms[(scale, dimension)] = transform_fn
    
    def transform(self, text: str, dimension: str, 
                  amount: float = 1.0, scale: Scale = None) -> str:
        """
        Apply a transformation to text.
        
        Args:
            text: Input text
            dimension: Dimension to transform along
            amount: How much to transform (-1 to 1)
            scale: Scale to operate at (auto-detected if None)
        """
        # Auto-detect scale if not provided
        if scale is None:
            scale = self._detect_scale(text, dimension)
        
        # Get transform function
        key = (scale, dimension)
        if key in self._transforms:
            return self._transforms[key](text, amount)
        
        # Try adjacent scales
        for adj_scale in [scale.parent_scale, scale.child_scale]:
            if adj_scale and (adj_scale, dimension) in self._transforms:
                return self._transforms[(adj_scale, dimension)](text, amount)
        
        return text
    
    def _detect_scale(self, text: str, dimension: str) -> Scale:
        """Detect the appropriate scale for a transformation."""
        # Check if dimension has explicit scale
        scale = self.corpus.get_dimension_scale(dimension)
        if scale:
            return scale
        
        # Heuristic based on text length
        text_len = len(text)
        for scale in Scale:
            if text_len <= scale.typical_length * 2:
                return scale
        
        return Scale.DOCUMENT
    
    def transform_cascade(self, text: str, dimension: str,
                          amount: float = 1.0) -> str:
        """
        Apply a transformation cascading through all relevant scales.
        
        Higher scales are applied first, then lower scales.
        """
        result = text
        
        # Detect relevant scales
        relevant = self.corpus.analyze_text_scales(text)
        
        # Apply from highest to lowest
        for scale in reversed(list(Scale)):
            if scale in relevant:
                key = (scale, dimension)
                if key in self._transforms:
                    result = self._transforms[key](result, amount)
        
        return result


# =============================================================================
# FRACTAL DIMENSION SPACE
# =============================================================================

class FractalDimensionSpace:
    """
    A dimension space that exhibits self-similarity across scales.
    
    Key insight: The SAME dimension types appear at multiple scales:
    
    - "formality" at WORD scale: guy → gentleman
    - "formality" at PHRASE scale: "what's up" → "how do you do"
    - "formality" at SENTENCE scale: casual → formal register
    - "formality" at DOCUMENT scale: blog post → academic paper
    
    This is FRACTAL - the same pattern repeats at every scale.
    """
    
    def __init__(self):
        self.corpus = UnifiedCorpus()
        
        # Track cross-scale dimensions
        self._cross_scale: Dict[str, Dict[Scale, Tuple[str, str]]] = {}
    
    def add_cross_scale_dimension(self, name: str,
                                   scale_examples: Dict[Scale, Tuple[str, str]],
                                   dim_type: DimensionType = DimensionType.PATTERN):
        """
        Add a dimension that exists at multiple scales.
        
        Args:
            name: Dimension name (e.g., "formality")
            scale_examples: {Scale: (negative_pole, positive_pole)}
            dim_type: Type of dimension
        """
        self._cross_scale[name] = scale_examples
        
        for scale, (neg, pos) in scale_examples.items():
            self.corpus._add_typed_pair(neg, pos, name, dim_type, scale)
    
    def get_dimension_at_scale(self, dimension: str, scale: Scale) -> Optional[str]:
        """Get the full dimension name at a specific scale."""
        if dimension in self._cross_scale and scale in self._cross_scale[dimension]:
            return f"{scale.name.lower()}:{dimension}"
        return None
    
    def get_poles_at_scale(self, dimension: str, scale: Scale) -> Optional[Tuple[str, str]]:
        """Get the poles of a dimension at a specific scale."""
        if dimension in self._cross_scale:
            return self._cross_scale[dimension].get(scale)
        return None
    
    def traverse_cross_scale(self, concept: str, dimension: str,
                             from_scale: Scale, to_scale: Scale) -> Optional[str]:
        """
        Traverse a dimension while changing scales.
        
        Example: Move from word-level "guy" to document-level "blog"
        along the formality dimension.
        """
        # Get poles at both scales
        from_poles = self.get_poles_at_scale(dimension, from_scale)
        to_poles = self.get_poles_at_scale(dimension, to_scale)
        
        if not from_poles or not to_poles:
            return None
        
        # Determine which pole the concept is at
        if concept == from_poles[0]:  # Negative pole
            return to_poles[0]
        elif concept == from_poles[1]:  # Positive pole
            return to_poles[1]
        
        return None
    
    def list_cross_scale_dimensions(self) -> List[str]:
        """List all dimensions that exist at multiple scales."""
        return list(self._cross_scale.keys())
    
    def get_scales_for_dimension(self, dimension: str) -> List[Scale]:
        """Get all scales at which a dimension exists."""
        if dimension in self._cross_scale:
            return list(self._cross_scale[dimension].keys())
        return []


# =============================================================================
# SCALE DETECTOR
# =============================================================================

class ScaleDetector:
    """
    Automatic scale detection from text structure.
    """
    
    @staticmethod
    def detect_primary_scale(text: str) -> Scale:
        """Detect the primary scale of a piece of text."""
        text_len = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        paragraph_count = text.count('\n\n') + 1
        
        # Document scale
        if text_len > 1000 or paragraph_count >= 5:
            return Scale.DOCUMENT
        
        # Section scale
        if paragraph_count >= 3 or text_len > 500:
            return Scale.SECTION
        
        # Paragraph scale
        if sentence_count >= 3 or paragraph_count >= 2:
            return Scale.PARAGRAPH
        
        # Sentence scale
        if sentence_count >= 1 or word_count >= 5:
            return Scale.SENTENCE
        
        # Phrase scale
        if word_count >= 2:
            return Scale.PHRASE
        
        # Word scale
        if word_count == 1:
            return Scale.WORD
        
        # Character scale
        return Scale.CHARACTER
    
    @staticmethod
    def detect_all_scales(text: str) -> List[Scale]:
        """Detect all scales present in text."""
        scales = [Scale.CHARACTER]  # Always present
        
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        paragraph_count = text.count('\n\n') + 1
        
        if word_count >= 1:
            scales.append(Scale.WORD)
        if word_count >= 2:
            scales.append(Scale.PHRASE)
        if sentence_count >= 1 or word_count >= 5:
            scales.append(Scale.SENTENCE)
        if sentence_count >= 2:
            scales.append(Scale.PARAGRAPH)
        if paragraph_count >= 2:
            scales.append(Scale.SECTION)
        if len(text) > 500 or paragraph_count >= 3:
            scales.append(Scale.DOCUMENT)
        
        return scales
    
    @staticmethod
    def segment_by_scale(text: str, scale: Scale) -> List[str]:
        """Segment text into units at a given scale."""
        if scale == Scale.CHARACTER:
            return list(text)
        elif scale == Scale.WORD:
            return text.split()
        elif scale == Scale.PHRASE:
            # Simple phrase detection (comma/semicolon separated)
            return re.split(r'[,;]', text)
        elif scale == Scale.SENTENCE:
            return re.split(r'[.!?]+', text)
        elif scale == Scale.PARAGRAPH:
            return text.split('\n\n')
        elif scale == Scale.SECTION:
            # Look for section markers
            return re.split(r'\n#{1,3}\s', text)
        elif scale == Scale.DOCUMENT:
            return [text]
        
        return [text]


# =============================================================================
# DEMO
# =============================================================================

def demo_fractal_dimensions():
    """Demonstrate fractal dimensions across scales."""
    print("=" * 60)
    print("DEMO: Fractal Dimensions")
    print("=" * 60)
    print()
    
    space = FractalDimensionSpace()
    
    # Add cross-scale dimensions
    space.add_cross_scale_dimension("formality", {
        Scale.WORD: ("guy", "gentleman"),
        Scale.PHRASE: ("what's up", "how do you do"),
        Scale.SENTENCE: ("casual", "formal"),
        Scale.DOCUMENT: ("blog", "paper"),
    })
    
    space.add_cross_scale_dimension("complexity", {
        Scale.WORD: ("simple", "complex"),
        Scale.SENTENCE: ("short", "elaborate"),
        Scale.PARAGRAPH: ("sparse", "dense"),
        Scale.DOCUMENT: ("brief", "comprehensive"),
    })
    
    space.corpus.recompute()
    
    print("Cross-scale dimensions:")
    for dim in space.list_cross_scale_dimensions():
        scales = space.get_scales_for_dimension(dim)
        print(f"  {dim}: {[s.name for s in scales]}")
    print()
    
    print("Formality at each scale:")
    for scale in [Scale.WORD, Scale.PHRASE, Scale.SENTENCE, Scale.DOCUMENT]:
        poles = space.get_poles_at_scale("formality", scale)
        if poles:
            print(f"  {scale.name}: {poles[0]} ↔ {poles[1]}")
    print()
    
    print("Cross-scale traversal:")
    result = space.traverse_cross_scale("guy", "formality", Scale.WORD, Scale.DOCUMENT)
    print(f"  'guy' (WORD) → '{result}' (DOCUMENT) along formality")
    
    result = space.traverse_cross_scale("gentleman", "formality", Scale.WORD, Scale.SENTENCE)
    print(f"  'gentleman' (WORD) → '{result}' (SENTENCE) along formality")
    print()
    
    return space


def demo_scale_detection():
    """Demonstrate automatic scale detection."""
    print("=" * 60)
    print("DEMO: Scale Detection")
    print("=" * 60)
    print()
    
    test_texts = [
        "hello",
        "The king ruled wisely.",
        "The king ruled wisely. His subjects loved him dearly.",
        """The king ruled wisely over his vast kingdom.

His subjects loved him dearly, for he was just and fair.

Under his reign, the land prospered greatly.""",
    ]
    
    for text in test_texts:
        primary = ScaleDetector.detect_primary_scale(text)
        all_scales = ScaleDetector.detect_all_scales(text)
        
        preview = text[:40] + "..." if len(text) > 40 else text
        preview = preview.replace('\n', ' ')
        
        print(f"Text: '{preview}'")
        print(f"  Primary scale: {primary.name}")
        print(f"  All scales: {[s.name for s in all_scales]}")
        print()


if __name__ == "__main__":
    demo_fractal_dimensions()
    demo_scale_detection()
