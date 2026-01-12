#!/usr/bin/env python3
"""
Core classes for the Unified Self-Assembly System.

This module provides the foundational classes that support the Universal
Dimension Principle: ANY transformation at ANY scale can be a dimension.

Classes:
- Scale: Enum for the scale hierarchy (CHARACTER through DOCUMENT)
- DimensionType: Enum for dimension categories (CONTENT, PATTERN, STYLIZATION)
- ScaledDimension: A dimension tagged with its operating scale
- UnifiedCorpus: Corpus that handles all dimension types at all scales

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from enum import Enum, auto

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.self_assembling_corpus import (
    SelfAssemblingCorpus,
    TransformationPair,
    Dimension,
    ConceptType,
    PHI,
)


# =============================================================================
# ENUMS
# =============================================================================

class Scale(Enum):
    """
    Scales at which dimensions can operate.
    
    Each scale is roughly φ² ≈ 2.6x larger than the previous.
    This creates a natural hierarchy where:
    - Lower scales are more granular (characters, words)
    - Higher scales are more abstract (sections, documents)
    - Transformations at higher scales constrain lower scales
    """
    CHARACTER = 0    # Individual glyphs: a, b, c
    WORD = 1         # Lexical units: king, queen, dog
    PHRASE = 2       # Multi-word units: "kick the bucket"
    SENTENCE = 3     # Complete thoughts: "The king ruled wisely."
    PARAGRAPH = 4    # Coherent blocks: introduction, argument
    SECTION = 5      # Major divisions: chapter, act
    DOCUMENT = 6     # Complete works: paper, book, article
    
    @property
    def typical_length(self) -> int:
        """Approximate character count at this scale."""
        base = 1
        return int(base * (PHI ** (2 * self.value)))
    
    @property
    def parent_scale(self) -> Optional['Scale']:
        """The next larger scale."""
        if self.value < Scale.DOCUMENT.value:
            return Scale(self.value + 1)
        return None
    
    @property
    def child_scale(self) -> Optional['Scale']:
        """The next smaller scale."""
        if self.value > Scale.CHARACTER.value:
            return Scale(self.value - 1)
        return None


class DimensionType(Enum):
    """
    Categories of dimensions.
    
    While all dimensions use the same φ-geometry, categorizing them
    helps with organization and discovery.
    """
    CONTENT = auto()      # What is being discussed (king→queen, small→large)
    PATTERN = auto()      # How it's expressed (formal→casual, prose→verse)
    STYLIZATION = auto()  # Visual presentation (plain→vaporwave, lower→upper)
    UNKNOWN = auto()      # Not yet categorized


# Dimension types typically associated with each scale
SCALE_DIMENSION_TYPES: Dict[Scale, List[DimensionType]] = {
    Scale.CHARACTER: [DimensionType.STYLIZATION],
    Scale.WORD: [DimensionType.CONTENT],
    Scale.PHRASE: [DimensionType.CONTENT, DimensionType.PATTERN],
    Scale.SENTENCE: [DimensionType.PATTERN],
    Scale.PARAGRAPH: [DimensionType.PATTERN],
    Scale.SECTION: [DimensionType.PATTERN],
    Scale.DOCUMENT: [DimensionType.PATTERN],
}

# Dimension names by scale
SCALE_DIMENSIONS: Dict[Scale, List[str]] = {
    Scale.CHARACTER: ['spacing', 'case', 'substitution', 'decoration'],
    Scale.WORD: ['gender', 'size', 'age', 'formality', 'specificity', 'sentiment'],
    Scale.PHRASE: ['idiom', 'register', 'collocation', 'verbosity'],
    Scale.SENTENCE: ['tone', 'meter', 'speech_act', 'complexity', 'voice', 'certainty'],
    Scale.PARAGRAPH: ['structure', 'coherence', 'topic_flow', 'density'],
    Scale.SECTION: ['argument', 'narrative', 'emphasis', 'pacing'],
    Scale.DOCUMENT: ['genre', 'audience', 'purpose', 'style'],
}


# =============================================================================
# SCALED DIMENSION
# =============================================================================

@dataclass
class ScaledDimension:
    """
    A dimension that operates at a specific scale.
    
    The key insight: A dimension is defined by:
    1. Its name (what transformation it represents)
    2. Its scale (what level of text it operates on)
    3. Its type (content, pattern, or stylization)
    4. Its poles (the endpoints of the transformation)
    """
    name: str
    scale: Scale
    dim_type: DimensionType
    negative_pole: str
    positive_pole: str
    
    @property
    def full_name(self) -> str:
        """Unique identifier including scale."""
        return f"{self.scale.name.lower()}:{self.name}"
    
    def applies_to(self, text_length: int) -> bool:
        """Check if this dimension applies to text of given length."""
        min_len = self.scale.typical_length // 2
        max_len = self.scale.typical_length * 5
        return min_len <= text_length <= max_len


# =============================================================================
# UNIFIED CORPUS
# =============================================================================

class UnifiedCorpus(SelfAssemblingCorpus):
    """
    Corpus that handles all dimension types at all scales.
    
    This is the core class that implements the Universal Dimension Principle:
    ANY transformation at ANY scale can be a dimension.
    
    Features:
    1. Dimensions tagged with scale and type
    2. Cross-scale composition with φ-weighting
    3. Automatic scale detection
    4. Unified self-assembly for all dimension types
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Track dimensions by scale and type
        self._scaled_dimensions: Dict[str, ScaledDimension] = {}
        self._dimension_types: Dict[str, DimensionType] = {}
        
        # Track active scales
        self._active_scales: Set[Scale] = set()
    
    # -------------------------------------------------------------------------
    # Adding Pairs by Type
    # -------------------------------------------------------------------------
    
    def add_content_pair(self, source: str, target: str, 
                         dimension: str, scale: Scale = Scale.WORD) -> bool:
        """Add a content transformation pair."""
        return self._add_typed_pair(source, target, dimension, 
                                    DimensionType.CONTENT, scale)
    
    def add_pattern_pair(self, source: str, target: str,
                         dimension: str, scale: Scale = Scale.SENTENCE) -> bool:
        """Add a pattern transformation pair."""
        return self._add_typed_pair(source, target, dimension,
                                    DimensionType.PATTERN, scale)
    
    def add_stylization_pair(self, source: str, target: str,
                             dimension: str, scale: Scale = Scale.CHARACTER) -> bool:
        """Add a stylization transformation pair."""
        return self._add_typed_pair(source, target, dimension,
                                    DimensionType.STYLIZATION, scale)
    
    def _add_typed_pair(self, source: str, target: str, dimension: str,
                        dim_type: DimensionType, scale: Scale) -> bool:
        """Add a transformation pair with type and scale."""
        # Create full dimension name
        full_dim = f"{scale.name.lower()}:{dimension}"
        
        # Track dimension metadata
        if full_dim not in self._scaled_dimensions:
            self._scaled_dimensions[full_dim] = ScaledDimension(
                name=dimension,
                scale=scale,
                dim_type=dim_type,
                negative_pole=source,
                positive_pole=target
            )
        
        self._dimension_types[full_dim] = dim_type
        self._active_scales.add(scale)
        
        # Add to base corpus
        return self.add_pair(source, target, full_dim)
    
    # -------------------------------------------------------------------------
    # Querying Dimensions
    # -------------------------------------------------------------------------
    
    def get_dimension_type(self, dimension: str) -> DimensionType:
        """Get the type of a dimension."""
        # Check full name first
        if dimension in self._dimension_types:
            return self._dimension_types[dimension]
        
        # Check short name
        for full_name, dim_type in self._dimension_types.items():
            if full_name.endswith(f":{dimension}"):
                return dim_type
        
        return DimensionType.UNKNOWN
    
    def get_dimension_scale(self, dimension: str) -> Optional[Scale]:
        """Get the scale of a dimension."""
        if dimension in self._scaled_dimensions:
            return self._scaled_dimensions[dimension].scale
        
        # Check short name
        for full_name, scaled_dim in self._scaled_dimensions.items():
            if scaled_dim.name == dimension:
                return scaled_dim.scale
        
        return None
    
    def get_dimensions_at_scale(self, scale: Scale) -> List[str]:
        """Get all dimensions that operate at a given scale."""
        return [
            dim.name for dim in self._scaled_dimensions.values()
            if dim.scale == scale
        ]
    
    def get_dimensions_by_type(self, dim_type: DimensionType) -> List[str]:
        """Get all dimensions of a given type."""
        return [
            dim.name for dim in self._scaled_dimensions.values()
            if dim.dim_type == dim_type
        ]
    
    def get_concept_type(self, concept: str) -> DimensionType:
        """Determine if a concept is primarily content, pattern, or stylization."""
        # Check which dimension types this concept participates in
        type_counts = {t: 0 for t in DimensionType}
        
        for pair in self.pairs:
            if pair.source == concept or pair.target == concept:
                dim_type = self.get_dimension_type(pair.relationship)
                type_counts[dim_type] += 1
        
        # Return the most common type
        max_type = max(type_counts, key=type_counts.get)
        return max_type if type_counts[max_type] > 0 else DimensionType.UNKNOWN
    
    # -------------------------------------------------------------------------
    # Multi-Scale Composition
    # -------------------------------------------------------------------------
    
    def compose(self, *concepts: str) -> Optional[np.ndarray]:
        """
        Compose multiple concepts using φ-Zipf weighting.
        
        Concepts are weighted by their scale (higher scales dominate).
        """
        if not concepts:
            return None
        
        # Get positions and scales
        items = []
        for concept in concepts:
            pos = self.get_position(concept)
            if pos is not None:
                scale = self._infer_concept_scale(concept)
                items.append((pos, scale))
        
        if not items:
            return None
        
        # Sort by scale (highest first)
        items.sort(key=lambda x: x[1].value if x[1] else -1, reverse=True)
        
        # Compose with φ-weighting
        result = None
        for i, (pos, scale) in enumerate(items):
            weight = PHI ** (-i)
            
            if result is None:
                result = pos * weight
            else:
                # Pad to same length
                max_len = max(len(result), len(pos))
                result = np.pad(result, (0, max_len - len(result)))
                pos = np.pad(pos, (0, max_len - len(pos)))
                result = result + pos * weight
        
        return result
    
    def compose_with_scales(self, *concepts_with_scales: Tuple[str, Scale]) -> np.ndarray:
        """
        Compose concepts with explicit scale specification.
        
        Args:
            concepts_with_scales: Tuples of (concept, scale)
        """
        if not concepts_with_scales:
            return np.zeros(len(self.dimensions))
        
        # Sort by scale (highest first)
        sorted_items = sorted(concepts_with_scales,
                             key=lambda x: x[1].value,
                             reverse=True)
        
        result = None
        for i, (concept, scale) in enumerate(sorted_items):
            pos = self.get_position(concept)
            if pos is None:
                continue
            
            weight = PHI ** (-i)
            
            if result is None:
                result = pos * weight
            else:
                max_len = max(len(result), len(pos))
                result = np.pad(result, (0, max_len - len(result)))
                pos = np.pad(pos, (0, max_len - len(pos)))
                result = result + pos * weight
        
        return result if result is not None else np.zeros(len(self.dimensions))
    
    def decompose(self, position: np.ndarray) -> Tuple[List[str], List[str], List[str]]:
        """
        Decompose a position into content, pattern, and stylization components.
        
        Returns:
            Tuple of (content_concepts, pattern_concepts, stylization_concepts)
        """
        content = []
        pattern = []
        stylization = []
        
        # Find nearest concepts
        nearest = self.find_nearest(position, n=10)
        
        for concept, distance in nearest:
            concept_type = self.get_concept_type(concept)
            
            if concept_type == DimensionType.CONTENT:
                content.append(concept)
            elif concept_type == DimensionType.PATTERN:
                pattern.append(concept)
            elif concept_type == DimensionType.STYLIZATION:
                stylization.append(concept)
        
        return content[:3], pattern[:2], stylization[:2]
    
    def _infer_concept_scale(self, concept: str) -> Optional[Scale]:
        """Infer the scale of a concept from its dimension participation."""
        scales = []
        
        for pair in self.pairs:
            if pair.source == concept or pair.target == concept:
                dim_scale = self.get_dimension_scale(pair.relationship)
                if dim_scale:
                    scales.append(dim_scale)
        
        if scales:
            # Return the most common scale
            return max(set(scales), key=scales.count)
        
        return None
    
    # -------------------------------------------------------------------------
    # Scale Detection
    # -------------------------------------------------------------------------
    
    def analyze_text_scales(self, text: str) -> Dict[Scale, List[str]]:
        """
        Analyze which scales are relevant for a piece of text.
        
        Returns dimensions that might apply at each scale.
        """
        text_len = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        paragraph_count = text.count('\n\n') + 1
        
        relevant = {}
        
        # Character scale (always relevant)
        relevant[Scale.CHARACTER] = SCALE_DIMENSIONS[Scale.CHARACTER]
        
        # Word scale
        if word_count > 0:
            relevant[Scale.WORD] = SCALE_DIMENSIONS[Scale.WORD]
        
        # Phrase scale
        if word_count >= 2:
            relevant[Scale.PHRASE] = SCALE_DIMENSIONS[Scale.PHRASE]
        
        # Sentence scale
        if sentence_count > 0 or word_count >= 3:
            relevant[Scale.SENTENCE] = SCALE_DIMENSIONS[Scale.SENTENCE]
        
        # Paragraph scale
        if sentence_count >= 2:
            relevant[Scale.PARAGRAPH] = SCALE_DIMENSIONS[Scale.PARAGRAPH]
        
        # Section scale
        if paragraph_count >= 2:
            relevant[Scale.SECTION] = SCALE_DIMENSIONS[Scale.SECTION]
        
        # Document scale
        if text_len > 500 or paragraph_count >= 3:
            relevant[Scale.DOCUMENT] = SCALE_DIMENSIONS[Scale.DOCUMENT]
        
        return relevant
    
    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict:
        """Get current status of the unified corpus."""
        self.recompute()
        
        content_dims = len(self.get_dimensions_by_type(DimensionType.CONTENT))
        pattern_dims = len(self.get_dimensions_by_type(DimensionType.PATTERN))
        style_dims = len(self.get_dimensions_by_type(DimensionType.STYLIZATION))
        
        return {
            "total_pairs": len(self.pairs),
            "total_dimensions": len(self.dimensions),
            "content_dimensions": content_dims,
            "pattern_dimensions": pattern_dims,
            "stylization_dimensions": style_dims,
            "total_concepts": len(self.concepts),
            "total_ideals": len(self.ideals),
            "active_scales": [s.name for s in self._active_scales],
        }
