"""
Quaternion Encoder - Dynamic Quaternion Layers for Text Encoding

Implements the quaternion layer architecture (Design 104-105):

Q = w + xi + yj + zk

Where:
- w (4D): Core Semantic (domain, specificity, intent, formality)
- x (4D): Grammatical (tense, aspect, mood, voice)
- y (4D): Contextual (register, evidentiality, politeness, emphasis)
- z (ND): Dynamic/Emergent (gender, regality, tempo, etc.)

The z-layer is dynamic and can grow as new dimensions are discovered.
This allows for 128+ dimensions while maintaining predictable navigation.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum

from .dynamic_dimensions import DynamicDimensionRegistry, PHI


# =============================================================================
# STRUCTURED LAYER DEFINITIONS
# =============================================================================

class SemanticDim(IntEnum):
    """w-layer: Core semantic dimensions."""
    DOMAIN = 0       # general ↔ specialized
    SPECIFICITY = 1  # abstract ↔ concrete
    INTENT = 2       # inform ↔ persuade
    FORMALITY = 3    # casual ↔ formal


class GrammaticalDim(IntEnum):
    """x-layer: Grammatical dimensions."""
    TENSE = 0        # past ↔ future
    ASPECT = 1       # perfective ↔ progressive
    MOOD = 2         # indicative ↔ subjunctive
    VOICE = 3        # passive ↔ active


class ContextualDim(IntEnum):
    """y-layer: Contextual dimensions."""
    REGISTER = 0     # informal ↔ formal
    EVIDENTIALITY = 1  # hearsay ↔ direct
    POLITENESS = 2   # blunt ↔ polite
    EMPHASIS = 3     # neutral ↔ emphatic


# Semantic meanings for each level
SEMANTIC_MEANINGS = {
    SemanticDim.DOMAIN: {0: 'general', 1: 'broad', 2: 'focused', 3: 'specialized'},
    SemanticDim.SPECIFICITY: {0: 'abstract', 1: 'conceptual', 2: 'concrete', 3: 'specific'},
    SemanticDim.INTENT: {0: 'inform', 1: 'explain', 2: 'guide', 3: 'persuade'},
    SemanticDim.FORMALITY: {0: 'casual', 1: 'neutral', 2: 'professional', 3: 'formal'},
}

GRAMMATICAL_MEANINGS = {
    GrammaticalDim.TENSE: {0: 'past', 1: 'recent', 2: 'present', 3: 'future'},
    GrammaticalDim.ASPECT: {0: 'perfective', 1: 'completed', 2: 'ongoing', 3: 'progressive'},
    GrammaticalDim.MOOD: {0: 'indicative', 1: 'conditional', 2: 'potential', 3: 'subjunctive'},
    GrammaticalDim.VOICE: {0: 'passive', 1: 'middle', 2: 'reflexive', 3: 'active'},
}

CONTEXTUAL_MEANINGS = {
    ContextualDim.REGISTER: {0: 'informal', 1: 'conversational', 2: 'standard', 3: 'formal'},
    ContextualDim.EVIDENTIALITY: {0: 'hearsay', 1: 'inferred', 2: 'reported', 3: 'direct'},
    ContextualDim.POLITENESS: {0: 'blunt', 1: 'direct', 2: 'polite', 3: 'deferential'},
    ContextualDim.EMPHASIS: {0: 'neutral', 1: 'mild', 2: 'moderate', 3: 'emphatic'},
}


# =============================================================================
# QUATERNION POSITION
# =============================================================================

@dataclass
class QuaternionPosition:
    """
    A position in quaternion space with dynamic z-layer.
    
    Q = w + xi + yj + zk
    
    w, x, y are fixed 4D layers (structured)
    z is variable-length (dynamic/emergent)
    """
    w: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Semantic
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Grammatical
    y: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Contextual
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))  # Dynamic
    z_labels: Dict[str, int] = field(default_factory=dict)      # z dimension names
    
    @classmethod
    def from_levels(cls, w_levels: List[int] = None, 
                    x_levels: List[int] = None,
                    y_levels: List[int] = None,
                    z_values: Dict[str, float] = None) -> 'QuaternionPosition':
        """Create position from φ-levels and z-values."""
        def levels_to_coords(levels: List[int]) -> np.ndarray:
            if levels is None:
                return np.ones(4)
            coords = np.ones(4)
            for i, level in enumerate(levels[:4]):
                coords[i] = PHI ** level
            return coords
        
        pos = cls(
            w=levels_to_coords(w_levels),
            x=levels_to_coords(x_levels),
            y=levels_to_coords(y_levels),
        )
        
        if z_values:
            pos.z_labels = {name: i for i, name in enumerate(z_values.keys())}
            pos.z = np.array(list(z_values.values()))
        
        return pos
    
    def to_flat(self) -> np.ndarray:
        """Flatten to single vector."""
        return np.concatenate([self.w, self.x, self.y, self.z])
    
    @classmethod
    def from_flat(cls, flat: np.ndarray, z_labels: Dict[str, int] = None) -> 'QuaternionPosition':
        """Create from flat vector."""
        pos = cls(
            w=flat[:4].copy(),
            x=flat[4:8].copy(),
            y=flat[8:12].copy(),
            z=flat[12:].copy() if len(flat) > 12 else np.zeros(0),
        )
        if z_labels:
            pos.z_labels = z_labels
        return pos
    
    def distance(self, other: 'QuaternionPosition', 
                 layer_weights: Tuple[float, float, float, float] = None) -> float:
        """
        Weighted distance between positions.
        
        Default weights: (1.0, 0.5, 0.25, 0.125) - semantic most important
        """
        if layer_weights is None:
            layer_weights = (1.0, 0.5, 0.25, 0.125)
        
        w_dist = np.linalg.norm(self.w - other.w)
        x_dist = np.linalg.norm(self.x - other.x)
        y_dist = np.linalg.norm(self.y - other.y)
        
        # Handle variable-length z
        z_dist = 0.0
        if len(self.z) > 0 or len(other.z) > 0:
            max_len = max(len(self.z), len(other.z))
            z1 = np.zeros(max_len)
            z2 = np.zeros(max_len)
            z1[:len(self.z)] = self.z
            z2[:len(other.z)] = other.z
            z_dist = np.linalg.norm(z1 - z2)
        
        return float(
            layer_weights[0] * w_dist +
            layer_weights[1] * x_dist +
            layer_weights[2] * y_dist +
            layer_weights[3] * z_dist
        )
    
    def similarity(self, other: 'QuaternionPosition',
                   layer_weights: Tuple[float, float, float, float] = None) -> float:
        """Similarity (inverse of distance, normalized)."""
        dist = self.distance(other, layer_weights)
        return 1.0 / (1.0 + dist)
    
    def describe(self) -> Dict[str, Any]:
        """Get human-readable description."""
        def describe_layer(coords: np.ndarray, meanings: Dict, dim_enum) -> Dict[str, str]:
            result = {}
            for dim in dim_enum:
                level = int(np.clip(np.log(coords[dim]) / np.log(PHI), 0, 3))
                result[dim.name.lower()] = meanings[dim].get(level, f"level_{level}")
            return result
        
        desc = {
            'semantic': describe_layer(self.w, SEMANTIC_MEANINGS, SemanticDim),
            'grammatical': describe_layer(self.x, GRAMMATICAL_MEANINGS, GrammaticalDim),
            'contextual': describe_layer(self.y, CONTEXTUAL_MEANINGS, ContextualDim),
        }
        
        if len(self.z) > 0:
            z_desc = {}
            for name, idx in self.z_labels.items():
                if idx < len(self.z):
                    z_desc[name] = float(self.z[idx])
            desc['dynamic'] = z_desc
        
        return desc
    
    def __repr__(self) -> str:
        z_info = f", z=[{len(self.z)} dims]" if len(self.z) > 0 else ""
        return f"Q(w={self.w.tolist()}, x={self.x.tolist()}, y={self.y.tolist()}{z_info})"


# =============================================================================
# QUATERNION ENCODER
# =============================================================================

class QuaternionEncoder:
    """
    Encodes text into quaternion positions with dynamic z-layer.
    
    Integrates:
    - Structured layers (w, x, y) from φ-lattice primitives
    - Dynamic layer (z) from DynamicDimensionRegistry
    """
    
    def __init__(self, dimension_registry: DynamicDimensionRegistry = None):
        if dimension_registry is None:
            dimension_registry = DynamicDimensionRegistry()
        
        self.dim_registry = dimension_registry
        
        # Structured layer anchors (simplified for now)
        self._w_anchors = self._build_semantic_anchors()
        self._x_anchors = self._build_grammatical_anchors()
        self._y_anchors = self._build_contextual_anchors()
    
    def _build_semantic_anchors(self) -> Dict[str, Tuple[int, int]]:
        """Build semantic dimension anchors."""
        return {
            # Domain
            'general': (SemanticDim.DOMAIN, 0),
            'specific': (SemanticDim.DOMAIN, 3),
            'technical': (SemanticDim.DOMAIN, 3),
            'specialized': (SemanticDim.DOMAIN, 3),
            
            # Specificity
            'abstract': (SemanticDim.SPECIFICITY, 0),
            'concrete': (SemanticDim.SPECIFICITY, 3),
            'example': (SemanticDim.SPECIFICITY, 3),
            'instance': (SemanticDim.SPECIFICITY, 3),
            
            # Intent
            'inform': (SemanticDim.INTENT, 0),
            'explain': (SemanticDim.INTENT, 1),
            'guide': (SemanticDim.INTENT, 2),
            'persuade': (SemanticDim.INTENT, 3),
            'convince': (SemanticDim.INTENT, 3),
            
            # Formality
            'casual': (SemanticDim.FORMALITY, 0),
            'informal': (SemanticDim.FORMALITY, 0),
            'formal': (SemanticDim.FORMALITY, 3),
            'professional': (SemanticDim.FORMALITY, 2),
        }
    
    def _build_grammatical_anchors(self) -> Dict[str, Tuple[int, int]]:
        """Build grammatical dimension anchors."""
        return {
            # Tense
            'was': (GrammaticalDim.TENSE, 0),
            'were': (GrammaticalDim.TENSE, 0),
            'had': (GrammaticalDim.TENSE, 0),
            'is': (GrammaticalDim.TENSE, 2),
            'are': (GrammaticalDim.TENSE, 2),
            'will': (GrammaticalDim.TENSE, 3),
            'shall': (GrammaticalDim.TENSE, 3),
            
            # Aspect
            'completed': (GrammaticalDim.ASPECT, 0),
            'finished': (GrammaticalDim.ASPECT, 0),
            'ongoing': (GrammaticalDim.ASPECT, 2),
            'continuing': (GrammaticalDim.ASPECT, 3),
            
            # Voice
            'passive': (GrammaticalDim.VOICE, 0),
            'active': (GrammaticalDim.VOICE, 3),
        }
    
    def _build_contextual_anchors(self) -> Dict[str, Tuple[int, int]]:
        """Build contextual dimension anchors."""
        return {
            # Register
            'hey': (ContextualDim.REGISTER, 0),
            'yo': (ContextualDim.REGISTER, 0),
            'dear': (ContextualDim.REGISTER, 3),
            'respectfully': (ContextualDim.REGISTER, 3),
            
            # Politeness
            'please': (ContextualDim.POLITENESS, 2),
            'kindly': (ContextualDim.POLITENESS, 3),
            'thanks': (ContextualDim.POLITENESS, 2),
            
            # Emphasis
            'very': (ContextualDim.EMPHASIS, 2),
            'extremely': (ContextualDim.EMPHASIS, 3),
            'absolutely': (ContextualDim.EMPHASIS, 3),
        }
    
    def encode(self, text: str) -> QuaternionPosition:
        """
        Encode text to quaternion position.
        
        Uses MAX aggregation for each dimension.
        """
        tokens = self.dim_registry.tokenize(text)
        
        # Initialize with base levels
        w_levels = [1, 1, 1, 1]
        x_levels = [1, 1, 1, 1]
        y_levels = [1, 1, 1, 1]
        
        # Process structured layers
        for token in tokens:
            if token in self._w_anchors:
                dim, level = self._w_anchors[token]
                w_levels[dim] = max(w_levels[dim], level)
            
            if token in self._x_anchors:
                dim, level = self._x_anchors[token]
                x_levels[dim] = max(x_levels[dim], level)
            
            if token in self._y_anchors:
                dim, level = self._y_anchors[token]
                y_levels[dim] = max(y_levels[dim], level)
        
        # Get dynamic z-layer from dimension registry
        z_vector = self.dim_registry.encode_text(text)
        z_labels = {name: idx for name, idx in self.dim_registry._dimensions.items()}
        
        # Build position
        pos = QuaternionPosition.from_levels(w_levels, x_levels, y_levels)
        pos.z = z_vector
        pos.z_labels = z_labels
        
        return pos
    
    def encode_with_description(self, text: str) -> Tuple[QuaternionPosition, Dict[str, Any]]:
        """Encode text and return position with description."""
        pos = self.encode(text)
        desc = pos.describe()
        
        # Add active z-dimensions
        z_active = self.dim_registry.describe_vector(pos.z)
        desc['z_active'] = z_active
        
        return pos, desc
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        pos1 = self.encode(text1)
        pos2 = self.encode(text2)
        return pos1.similarity(pos2)
    
    def distance(self, text1: str, text2: str) -> float:
        """Compute distance between two texts."""
        pos1 = self.encode(text1)
        pos2 = self.encode(text2)
        return pos1.distance(pos2)
    
    def ingest_corpus(self, text: str):
        """Ingest a corpus to build dimension registry."""
        self.dim_registry.ingest_text(text)
    
    def discover_entities(self) -> List[Tuple[str, float, float]]:
        """Discover entities in ingested text."""
        return self.dim_registry.discover_entities()
    
    @property
    def num_dimensions(self) -> int:
        """Total number of dimensions (12 structured + z dynamic)."""
        return 12 + self.dim_registry.num_dims
    
    def summary(self) -> Dict[str, Any]:
        """Get encoder summary."""
        return {
            'structured_dims': 12,
            'dynamic_dims': self.dim_registry.num_dims,
            'total_dims': self.num_dimensions,
            'dimension_names': self.dim_registry.dimension_names,
        }
