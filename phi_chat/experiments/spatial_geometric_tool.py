#!/usr/bin/env python3
"""
Spatial Geometric Tool - The "Comb" of the Music Box

This implements the Music Box Principle for spatial processing:
- Drum: Data positioned in φ-space (memories, embeddings, features)
- Comb: Geometric operations that read positions and produce outputs
- Music: The emergent result of drum + comb interaction

The key insight: We don't store transformations, we compute them geometrically.

From DA2 reverse engineering, we learned:
1. Dimensions encode specific features (depth, position, luminance)
2. φ-scaled weights can decode these features
3. In φ-basis, multiplication becomes addition

This tool demonstrates:
1. Spatial encoding of data into φ-space
2. Geometric operations (projection, transformation, nearest-neighbor)
3. Emergent outputs from structure, not lookup

Author: TruthSpace LCM Project
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
LOG_PHI = np.log(PHI)


@dataclass
class SpatialPoint:
    """A point in φ-space with associated data."""
    position: np.ndarray  # Position in φ-space
    data: any  # Associated data (word, embedding, feature)
    phi_level: float = 0.0  # φ-level of this point
    
    def __post_init__(self):
        self.phi_level = np.log(np.linalg.norm(self.position) + 1e-10) / LOG_PHI


class GeometricComb:
    """
    The "Comb" - a geometric processor that reads spatial structure.
    
    Unlike lookup tables, the comb computes transformations geometrically.
    The same comb can produce different outputs for different drums.
    """
    
    def __init__(self, n_dims: int = 4):
        self.n_dims = n_dims
        self.points: List[SpatialPoint] = []
        
        # Dimension semantics (like DA2's dimension mapping)
        self.dim_names = ['intensity', 'formality', 'domain', 'temporality'][:n_dims]
    
    def add_point(self, data: any, position: np.ndarray) -> SpatialPoint:
        """Add a point to the spatial structure (the drum)."""
        if len(position) != self.n_dims:
            raise ValueError(f"Position must have {self.n_dims} dimensions")
        
        point = SpatialPoint(position=np.array(position), data=data)
        self.points.append(point)
        return point
    
    def find_nearest(self, position: np.ndarray, k: int = 1) -> List[Tuple[SpatialPoint, float]]:
        """
        The core comb operation: find nearest points to a position.
        
        This is how the comb "reads" the drum - by finding what's near
        a given position in φ-space.
        """
        distances = []
        for point in self.points:
            dist = np.linalg.norm(position - point.position)
            distances.append((point, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def transform(self, data: any, delta: np.ndarray) -> any:
        """
        Transform data by applying a delta vector.
        
        This is the Music Box Principle in action:
        1. Find current position of data
        2. Apply delta to get new position
        3. Find nearest point at new position
        4. Return that point's data
        
        No lookup table - the transformation emerges from geometry.
        """
        # Find current position
        current_point = None
        for point in self.points:
            if point.data == data:
                current_point = point
                break
        
        if current_point is None:
            return data  # Data not in space, return unchanged
        
        # Apply delta
        new_position = current_point.position + delta
        
        # Find nearest at new position
        nearest = self.find_nearest(new_position, k=1)
        if nearest:
            return nearest[0][0].data
        
        return data
    
    def project(self, position: np.ndarray, weights: np.ndarray) -> float:
        """
        Project a position using φ-scaled weights.
        
        This is like DA2's depth decoding:
        output = Σ weight_i × position_i
        
        With φ-scaling, weights become φ^exponent.
        """
        if len(weights) != self.n_dims:
            raise ValueError(f"Weights must have {self.n_dims} dimensions")
        
        return np.dot(position, weights)
    
    def phi_project(self, position: np.ndarray, exponents: np.ndarray) -> float:
        """
        Project using φ-exponent weights.
        
        weight_i = φ^exponent_i
        output = Σ φ^exponent_i × position_i
        
        This is the φ-basis transformation from DA2.
        """
        weights = np.array([PHI ** e for e in exponents])
        return self.project(position, weights)
    
    def interpolate(self, pos1: np.ndarray, pos2: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate between two positions in φ-space.
        
        Uses φ-weighted interpolation for smooth transitions.
        """
        # φ-weighted interpolation
        phi_t = t ** INV_PHI  # Non-linear φ-scaling
        return pos1 * (1 - phi_t) + pos2 * phi_t
    
    def get_phi_level_distribution(self) -> Dict[str, float]:
        """Get statistics about φ-levels in the space."""
        if not self.points:
            return {}
        
        levels = [p.phi_level for p in self.points]
        return {
            'min': min(levels),
            'max': max(levels),
            'mean': np.mean(levels),
            'std': np.std(levels),
        }


class SpatialVocabulary(GeometricComb):
    """
    A vocabulary organized in φ-space.
    
    Words have positions based on semantic dimensions.
    Transformations are geometric, not lookup-based.
    """
    
    def __init__(self):
        super().__init__(n_dims=4)
        self.dim_names = ['tense', 'formality', 'domain', 'intensity']
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build a vocabulary with semantic positions."""
        # Format: (word, [tense, formality, domain, intensity])
        # tense: -1=past, 0=present, 1=future
        # formality: -1=casual, 0=neutral, 1=formal, 2=archaic
        # domain: -1=mundane, 0=neutral, 1=technical, 2=sacred
        # intensity: -1=weak, 0=neutral, 1=strong
        
        vocab = [
            # Basic words
            ("code", [0, 0, 1, 0]),
            ("program", [0, 0, 1, 0]),
            ("computer", [0, 0, 1, 0]),
            ("data", [0, 0, 1, 0]),
            ("error", [0, 0, 1, -0.5]),
            ("programmer", [0, 0, 1, 0]),
            
            # Warhammer 40K equivalents
            ("holy scripture", [0, 2, 2, 1]),
            ("sacred rite", [0, 2, 2, 1]),
            ("cogitator", [0, 2, 2, 0]),
            ("sacred data-hymns", [0, 2, 2, 1]),
            ("machine spirit's displeasure", [0, 2, 2, 0.5]),
            ("code-priest", [0, 2, 2, 1]),
            
            # Pirate equivalents
            ("treasure map", [0, -1, -1, 0]),
            ("sea shanty", [0, -1, -1, 0.5]),
            ("magic box", [0, -1, -1, 0]),
            ("booty", [0, -1, -1, 0]),
            ("bug", [0, -1, -1, -0.5]),
            ("coder", [0, -1, 0, 0]),
            
            # Verbs - past
            ("went", [-1, 0, 0, 0]),
            ("sat", [-1, 0, 0, 0]),
            ("walked", [-1, 0, 0, 0]),
            ("said", [-1, 0, 0, 0]),
            ("knew", [-1, 0, 0, 0]),
            ("made", [-1, 0, 0, 0]),
            
            # Verbs - future
            ("will go", [1, 0, 0, 0]),
            ("will sit", [1, 0, 0, 0]),
            ("will walk", [1, 0, 0, 0]),
            ("will say", [1, 0, 0, 0]),
            ("will know", [1, 0, 0, 0]),
            ("will make", [1, 0, 0, 0]),
            
            # Verbs - archaic
            ("did proceed", [-0.5, 2, 0, 0.5]),
            ("was seated", [-0.5, 2, 0, 0]),
            ("strode", [-0.5, 2, 0, 0.5]),
            ("spoke", [-0.5, 1, 0, 0.5]),
            ("understood", [-0.5, 1, 0, 0]),
            ("crafted", [-0.5, 1, 0, 0.5]),
            ("wrought", [-0.5, 2, 0, 1]),
            ("intoned", [-0.5, 2, 0, 0.5]),
        ]
        
        for word, pos in vocab:
            self.add_point(word, np.array(pos, dtype=np.float32))
    
    def get_word_position(self, word: str) -> Optional[np.ndarray]:
        """Get the position of a word in φ-space."""
        for point in self.points:
            if point.data == word:
                return point.position
        return None


class PerspectiveTransformer:
    """
    Transform text using perspective delta vectors.
    
    This is the Music Box Principle applied to style transfer:
    - No word→word mappings stored
    - Transformations computed geometrically
    - Output emerges from structure
    """
    
    # Perspective deltas (not word mappings!)
    PERSPECTIVES = {
        'warhammer_40k': np.array([0, 2, 2, 0.5]),  # archaic + sacred + intensity
        'pirate': np.array([0, -1, -1, 0]),  # casual + mundane
        'future': np.array([2, 0, 0, 0]),  # past→future
        'archaic': np.array([0.5, 2, 0, 0.5]),  # more formal + archaic
        'casual': np.array([0, -1, 0, 0]),  # less formal
    }
    
    def __init__(self):
        self.vocab = SpatialVocabulary()
    
    def transform_word(self, word: str, perspective: str) -> str:
        """Transform a single word using a perspective delta."""
        if perspective not in self.PERSPECTIVES:
            return word
        
        delta = self.PERSPECTIVES[perspective]
        return self.vocab.transform(word, delta)
    
    def transform_sentence(self, sentence: str, perspective: str) -> str:
        """Transform a sentence word by word."""
        words = sentence.lower().split()
        transformed = []
        
        for word in words:
            # Strip punctuation
            punct = ''
            if word and word[-1] in '.,!?':
                punct = word[-1]
                word = word[:-1]
            
            new_word = self.transform_word(word, perspective)
            transformed.append(new_word + punct)
        
        return ' '.join(transformed)


class SpatialFeatureProcessor:
    """
    Process spatial features using geometric operations.
    
    Like DA2's depth decoder, but generalized:
    - Input: Feature vector in high-dimensional space
    - Operation: φ-scaled projection
    - Output: Decoded value
    """
    
    def __init__(self, n_dims: int, feature_names: List[str] = None):
        self.n_dims = n_dims
        self.feature_names = feature_names or [f'dim_{i}' for i in range(n_dims)]
        
        # Dimension-to-feature correlations (like DA2's dimension mapping)
        self.dim_correlations: Dict[str, np.ndarray] = {}
        
        # φ-exponents for each feature (optimized like DA2)
        self.phi_exponents: Dict[str, np.ndarray] = {}
    
    def set_dimension_mapping(self, feature: str, correlations: np.ndarray, exponents: np.ndarray = None):
        """
        Set the dimension mapping for a feature.
        
        correlations: How each dimension correlates with this feature
        exponents: φ-exponents for decoding (optional, defaults to correlation-based)
        """
        self.dim_correlations[feature] = correlations
        
        if exponents is None:
            # Default: exponent based on correlation magnitude
            exponents = np.array([
                np.sign(c) * np.log(abs(c) + 0.1) / LOG_PHI
                for c in correlations
            ])
        
        self.phi_exponents[feature] = exponents
    
    def decode(self, features: np.ndarray, target: str) -> float:
        """
        Decode a feature vector to a target value.
        
        Uses φ-scaled weights like DA2's optimized decoder:
        output = Σ sign(corr_i) × φ^exponent_i × feature_i
        """
        if target not in self.phi_exponents:
            raise ValueError(f"No mapping for target: {target}")
        
        exponents = self.phi_exponents[target]
        correlations = self.dim_correlations[target]
        
        # φ-scaled weights
        weights = np.array([
            np.sign(correlations[i]) * (PHI ** exponents[i])
            for i in range(self.n_dims)
        ])
        
        # Normalize
        weights = weights / np.abs(weights).sum()
        
        # Decode
        return np.dot(features, weights)
    
    def encode(self, value: float, target: str, base_features: np.ndarray = None) -> np.ndarray:
        """
        Encode a value back into feature space.
        
        This is the inverse operation - given a target value,
        adjust features to produce that value.
        
        Uses the φ-basis insight: in φ-space, encoding and decoding
        are the same operation in opposite directions.
        """
        if target not in self.phi_exponents:
            raise ValueError(f"No mapping for target: {target}")
        
        if base_features is None:
            base_features = np.zeros(self.n_dims)
        
        exponents = self.phi_exponents[target]
        correlations = self.dim_correlations[target]
        
        # Current decoded value
        current_value = self.decode(base_features, target)
        
        # Difference to achieve
        diff = value - current_value
        
        # Distribute difference across dimensions based on weights
        weights = np.array([
            np.sign(correlations[i]) * (PHI ** exponents[i])
            for i in range(self.n_dims)
        ])
        weights = weights / np.abs(weights).sum()
        
        # Adjust features
        new_features = base_features + diff * weights
        
        return new_features


def demo_music_box_principle():
    """Demonstrate the Music Box Principle."""
    print("=" * 70)
    print("MUSIC BOX PRINCIPLE DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. PERSPECTIVE TRANSFORMATION")
    print("-" * 50)
    
    transformer = PerspectiveTransformer()
    
    test_sentences = [
        "The programmer made code and said it worked",
        "The computer went and sat",
    ]
    
    perspectives = ['warhammer_40k', 'pirate']
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        for perspective in perspectives:
            transformed = transformer.transform_sentence(sentence, perspective)
            print(f"  {perspective}: {transformed}")
    
    print("\n2. WORD TRANSFORMATIONS")
    print("-" * 50)
    
    words = ['code', 'computer', 'data', 'programmer', 'error']
    
    print("\n  Word → Warhammer 40K → Pirate")
    for word in words:
        w40k = transformer.transform_word(word, 'warhammer_40k')
        pirate = transformer.transform_word(word, 'pirate')
        print(f"  {word:15s} → {w40k:30s} → {pirate}")
    
    print("\n3. VERB TENSE TRANSFORMATIONS")
    print("-" * 50)
    
    verbs = ['went', 'sat', 'walked', 'said', 'knew', 'made']
    
    print("\n  Past → Future → Archaic")
    for verb in verbs:
        future = transformer.transform_word(verb, 'future')
        archaic = transformer.transform_word(verb, 'archaic')
        print(f"  {verb:10s} → {future:15s} → {archaic}")
    
    print("\n4. φ-SPACE STATISTICS")
    print("-" * 50)
    
    stats = transformer.vocab.get_phi_level_distribution()
    print(f"\n  φ-level range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  φ-level mean: {stats['mean']:.3f}")
    print(f"  φ-level std: {stats['std']:.3f}")
    
    print("\n5. THE KEY INSIGHT")
    print("-" * 50)
    print("""
  No word→word mappings were stored!
  
  The transformations emerged from:
  1. Words positioned in 4D semantic space
  2. Perspectives as delta vectors
  3. Nearest-neighbor lookup at new positions
  
  This is the Music Box Principle:
  - Drum: Words with positions
  - Comb: find_nearest(position + delta)
  - Music: Transformed text
  
  The comb doesn't contain the music.
  The music emerges from drum + comb interaction.
""")


def demo_spatial_feature_processor():
    """Demonstrate spatial feature processing like DA2."""
    print("\n" + "=" * 70)
    print("SPATIAL FEATURE PROCESSOR (DA2-STYLE)")
    print("=" * 70)
    
    # Create a processor with 8 dimensions (like DA2's head features)
    processor = SpatialFeatureProcessor(n_dims=8, feature_names=[
        'depth', 'y_pos', 'x_pos', 'luminance',
        'edge', 'contrast', 'saturation', 'texture'
    ])
    
    # Set dimension mappings (simulating DA2's discovered correlations)
    # Depth is encoded primarily in dims 0, 2, 5
    depth_corrs = np.array([0.66, 0.1, 0.45, 0.2, 0.3, 0.55, 0.1, 0.15])
    processor.set_dimension_mapping('depth', depth_corrs)
    
    # Y-position is encoded primarily in dim 1
    y_corrs = np.array([0.1, 0.96, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05])
    processor.set_dimension_mapping('y_position', y_corrs)
    
    print("\n1. DECODING FEATURES")
    print("-" * 50)
    
    # Simulate some feature vectors
    test_features = [
        np.array([0.8, 0.2, 0.6, 0.3, 0.4, 0.7, 0.2, 0.3]),  # High depth
        np.array([0.2, 0.8, 0.3, 0.5, 0.2, 0.2, 0.4, 0.3]),  # High y-position
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # Neutral
    ]
    
    for i, features in enumerate(test_features):
        depth = processor.decode(features, 'depth')
        y_pos = processor.decode(features, 'y_position')
        print(f"\n  Feature set {i+1}:")
        print(f"    Decoded depth: {depth:.4f}")
        print(f"    Decoded y_pos: {y_pos:.4f}")
    
    print("\n2. ENCODING VALUES")
    print("-" * 50)
    
    # Encode a target depth value
    target_depth = 0.8
    encoded = processor.encode(target_depth, 'depth')
    decoded_back = processor.decode(encoded, 'depth')
    
    print(f"\n  Target depth: {target_depth}")
    print(f"  Encoded features: {encoded}")
    print(f"  Decoded back: {decoded_back:.4f}")
    
    print("\n3. THE φ-BASIS INSIGHT")
    print("-" * 50)
    print("""
  Like DA2, we use φ-scaled weights:
  
    weight_i = sign(corr_i) × φ^exponent_i
    output = Σ weight_i × feature_i
  
  This is the "comb" reading the "drum":
  - Drum: Feature vectors (positions in high-D space)
  - Comb: φ-scaled projection
  - Music: Decoded values (depth, position, etc.)
  
  The same comb can decode different features
  by using different correlation/exponent mappings.
""")


if __name__ == "__main__":
    demo_music_box_principle()
    demo_spatial_feature_processor()
    
    print("\n" + "=" * 70)
    print("SPATIAL GEOMETRIC TOOL COMPLETE")
    print("=" * 70)
    print("""
  We have demonstrated two types of spatial "combs":
  
  1. SpatialVocabulary + PerspectiveTransformer
     - Words positioned in semantic space
     - Perspectives as delta vectors
     - Transformations via nearest-neighbor lookup
  
  2. SpatialFeatureProcessor
     - Features in high-dimensional space
     - φ-scaled projection weights
     - Encode/decode via geometric operations
  
  Both follow the Music Box Principle:
  - Structure (drum) contains the information
  - Processor (comb) reads it geometrically
  - Output (music) emerges from the interaction
  
  No lookup tables. No stored transformations.
  Just geometry.
""")
