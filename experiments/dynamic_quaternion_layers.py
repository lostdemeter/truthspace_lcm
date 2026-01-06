"""
Dynamic Quaternion Layers Experiment (Design 105)

Explores hybrid approach to dimensional scalability:
- Structured layers (w, x, y): Named, predictable dimensions
- Dynamic layer (z): High-dimensional overflow for emergent/unknown dimensions

The insight: We can't predefine every dimension in natural language.
"Regality", "intimacy", "urgency" - these exist but aren't in grammar books.

Architecture:
  Q = w + xi + yj + zk
  
  w = Core Semantic (4D, named)
  x = Grammatical (4D, named)  
  y = Contextual (4D, named)
  z = Dynamic (ND, emergent) - projects to 4D for quaternion math

The z-layer is where the magic happens:
- High-dimensional space (e.g., 64D or 128D)
- Dimensions discovered from data
- Projected to 4D for quaternion operations
- Can grow as new dimensions emerge

Example:
  "she put out the table ware for guests"
  "he put out the finery for company"
  
  Same semantic (w): setting table for visitors
  Same grammatical (x): past, perfective, active
  Different dynamic (z): gender, regality, formality of vocabulary

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import IntEnum

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# STRUCTURED LAYERS (same as before)
# =============================================================================

class SemanticDim(IntEnum):
    DOMAIN = 0
    SPECIFICITY = 1
    INTENT = 2
    FORMALITY = 3

class GrammaticalDim(IntEnum):
    TENSE = 0
    ASPECT = 1
    MOOD = 2
    VOICE = 3

class ContextualDim(IntEnum):
    REGISTER = 0
    EVIDENTIALITY = 1
    POLITENESS = 2
    EMPHASIS = 3


# =============================================================================
# DYNAMIC DIMENSION REGISTRY
# =============================================================================

class DynamicDimensionRegistry:
    """
    Registry for emergent/dynamic dimensions.
    
    Dimensions are discovered from data, not predefined.
    Each dimension has:
    - name: human-readable identifier (may be auto-generated)
    - index: position in the high-dimensional z-space
    - anchors: example values that define the dimension's meaning
    
    Example dimensions that might emerge:
    - gender: male ↔ female
    - regality: common ↔ regal
    - urgency: relaxed ↔ urgent
    - intimacy: distant ↔ intimate
    - age: young ↔ old
    - certainty: uncertain ↔ certain
    """
    
    def __init__(self, max_dims: int = 128):
        self.max_dims = max_dims
        self._dimensions: Dict[str, int] = {}  # name → index
        self._index_to_name: Dict[int, str] = {}
        self._anchors: Dict[str, Dict[str, float]] = {}  # name → {anchor_word: level}
        self._next_index = 0
    
    def register(self, name: str, anchors: Dict[str, float] = None) -> int:
        """
        Register a new dynamic dimension.
        
        Args:
            name: Dimension name (e.g., 'gender', 'regality')
            anchors: Optional dict of anchor words to levels
                     e.g., {'male': 1.0, 'female': -1.0, 'neutral': 0.0}
        
        Returns:
            Index of the dimension in z-space
        """
        if name in self._dimensions:
            return self._dimensions[name]
        
        if self._next_index >= self.max_dims:
            raise ValueError(f"Maximum dimensions ({self.max_dims}) reached")
        
        idx = self._next_index
        self._dimensions[name] = idx
        self._index_to_name[idx] = name
        self._anchors[name] = anchors or {}
        self._next_index += 1
        
        return idx
    
    def get_index(self, name: str) -> Optional[int]:
        return self._dimensions.get(name)
    
    def get_name(self, index: int) -> Optional[str]:
        return self._index_to_name.get(index)
    
    def add_anchor(self, dim_name: str, word: str, level: float):
        """Add an anchor word to a dimension."""
        if dim_name not in self._anchors:
            self._anchors[dim_name] = {}
        self._anchors[dim_name][word] = level
    
    def get_level_for_word(self, word: str) -> Dict[str, float]:
        """Get all dimension levels activated by a word."""
        levels = {}
        for dim_name, anchors in self._anchors.items():
            if word in anchors:
                levels[dim_name] = anchors[word]
        return levels
    
    @property
    def num_dims(self) -> int:
        return self._next_index
    
    def __repr__(self):
        return f"DynamicDimensionRegistry({self.num_dims} dims: {list(self._dimensions.keys())})"


# =============================================================================
# DYNAMIC QUATERNION POSITION
# =============================================================================

@dataclass
class DynamicQuaternionPosition:
    """
    A position in quaternion space with dynamic z-layer.
    
    w, x, y: Fixed 4D vectors (structured layers)
    z: Variable-length vector (dynamic layer)
    
    For quaternion operations, z is projected to 4D.
    """
    w: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Semantic
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Grammatical
    y: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Contextual
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))  # Dynamic (variable size)
    
    @classmethod
    def from_levels(cls, 
                    w_levels: List[int] = None,
                    x_levels: List[int] = None,
                    y_levels: List[int] = None,
                    z_values: Dict[str, float] = None,
                    registry: DynamicDimensionRegistry = None) -> 'DynamicQuaternionPosition':
        """
        Create position from levels.
        
        z_values is a dict of dimension_name → level value.
        """
        def levels_to_pos(levels):
            if levels is None:
                return np.array([PHI ** 0] * 4)
            return np.array([PHI ** k for k in levels])
        
        # Build z vector from named dimensions
        z = np.zeros(0)
        if z_values and registry:
            z = np.zeros(registry.max_dims)
            for dim_name, value in z_values.items():
                idx = registry.get_index(dim_name)
                if idx is not None:
                    z[idx] = value
        
        return cls(
            w=levels_to_pos(w_levels),
            x=levels_to_pos(x_levels),
            y=levels_to_pos(y_levels),
            z=z
        )
    
    def z_projected(self, target_dim: int = 4) -> np.ndarray:
        """
        Project z to fixed dimensionality for quaternion operations.
        
        Uses PCA-like projection: take first N principal components.
        For now, simple truncation/padding.
        """
        if len(self.z) == 0:
            return np.zeros(target_dim)
        
        if len(self.z) <= target_dim:
            result = np.zeros(target_dim)
            result[:len(self.z)] = self.z
            return result
        
        # For high-dim z, use magnitude-weighted selection
        # (In production, would use learned projection matrix)
        indices = np.argsort(np.abs(self.z))[-target_dim:]
        return self.z[indices]
    
    def distance(self, other: 'DynamicQuaternionPosition',
                 layer_weights: Tuple[float, float, float, float] = None) -> float:
        """
        Distance between positions.
        
        Uses projected z for the dynamic layer.
        """
        if layer_weights is None:
            layer_weights = (PHI**2, PHI, 1.0, PHI**-1)
        
        w_dist = np.linalg.norm(self.w - other.w)
        x_dist = np.linalg.norm(self.x - other.x)
        y_dist = np.linalg.norm(self.y - other.y)
        
        # For z, compare in full space if same size, else project
        if len(self.z) == len(other.z) and len(self.z) > 0:
            z_dist = np.linalg.norm(self.z - other.z)
        else:
            z_dist = np.linalg.norm(self.z_projected() - other.z_projected())
        
        return float(np.sqrt(
            layer_weights[0] * w_dist**2 +
            layer_weights[1] * x_dist**2 +
            layer_weights[2] * y_dist**2 +
            layer_weights[3] * z_dist**2
        ))
    
    def similarity(self, other: 'DynamicQuaternionPosition',
                   layer_weights: Tuple[float, float, float, float] = None) -> float:
        dist = self.distance(other, layer_weights)
        return 1.0 / (1.0 + dist)
    
    def z_distance(self, other: 'DynamicQuaternionPosition') -> float:
        """Distance in dynamic layer only."""
        if len(self.z) == len(other.z) and len(self.z) > 0:
            return float(np.linalg.norm(self.z - other.z))
        return float(np.linalg.norm(self.z_projected() - other.z_projected()))
    
    def describe_z(self, registry: DynamicDimensionRegistry) -> Dict[str, float]:
        """Get human-readable description of z dimensions."""
        desc = {}
        for i, val in enumerate(self.z):
            if abs(val) > 1e-6:
                name = registry.get_name(i) or f"dim_{i}"
                desc[name] = float(val)
        return desc
    
    def __repr__(self):
        z_nonzero = np.sum(np.abs(self.z) > 1e-6) if len(self.z) > 0 else 0
        return f"DQ(w={self.w.tolist()}, x={self.x.tolist()}, y={self.y.tolist()}, z=[{z_nonzero} active dims])"


# =============================================================================
# DEMONSTRATION: REGALITY EXAMPLE
# =============================================================================

def demo_regality():
    """
    Demonstrate the regality dimension.
    
    "she put out the table ware for guests"
    "he put out the finery for company"
    
    Same action, different style dimensions.
    """
    print("=" * 70)
    print("REGALITY DIMENSION DEMO")
    print("=" * 70)
    print()
    
    # Create registry with emergent dimensions
    registry = DynamicDimensionRegistry(max_dims=128)
    
    # Register dimensions we discover
    registry.register('gender', {
        'he': 1.0, 'him': 1.0, 'his': 1.0,
        'she': -1.0, 'her': -1.0, 'hers': -1.0,
        'they': 0.0, 'them': 0.0, 'their': 0.0,
    })
    
    registry.register('regality', {
        'finery': 2.0, 'silverware': 1.5, 'china': 1.5,
        'table ware': 0.0, 'dishes': -0.5, 'plates': -0.5,
        'company': 1.5, 'guests': 0.0, 'visitors': 0.5, 'folks': -1.0,
    })
    
    registry.register('vocabulary_level', {
        'put out': 0.0, 'set out': 0.5, 'arranged': 1.0, 'laid out': 1.5,
        'prepared': 1.0, 'readied': 1.5,
    })
    
    print(f"Registry: {registry}")
    print()
    
    # Sentence 1: "she put out the table ware for guests"
    # Neutral regality, female gender
    pos1 = DynamicQuaternionPosition.from_levels(
        w_levels=[1, 2, 1, 0],  # general, specific, inform, casual
        x_levels=[0, 0, 0, 1],  # past, perfective, indicative, active
        y_levels=[0, 2, 0, 0],  # neutral register, direct evidentiality
        z_values={'gender': -1.0, 'regality': 0.0, 'vocabulary_level': 0.0},
        registry=registry
    )
    
    # Sentence 2: "he put out the finery for company"
    # High regality, male gender
    pos2 = DynamicQuaternionPosition.from_levels(
        w_levels=[1, 2, 1, 0],  # SAME semantic!
        x_levels=[0, 0, 0, 1],  # SAME grammatical!
        y_levels=[0, 2, 0, 0],  # SAME contextual!
        z_values={'gender': 1.0, 'regality': 2.0, 'vocabulary_level': 0.0},
        registry=registry
    )
    
    # Sentence 3: "they laid out the china for the visitors"
    # Medium regality, neutral gender, elevated vocabulary
    pos3 = DynamicQuaternionPosition.from_levels(
        w_levels=[1, 2, 1, 0],
        x_levels=[0, 0, 0, 1],
        y_levels=[0, 2, 0, 0],
        z_values={'gender': 0.0, 'regality': 1.5, 'vocabulary_level': 1.5},
        registry=registry
    )
    
    print("Sentence 1: 'she put out the table ware for guests'")
    print(f"  Position: {pos1}")
    print(f"  Dynamic dims: {pos1.describe_z(registry)}")
    print()
    
    print("Sentence 2: 'he put out the finery for company'")
    print(f"  Position: {pos2}")
    print(f"  Dynamic dims: {pos2.describe_z(registry)}")
    print()
    
    print("Sentence 3: 'they laid out the china for the visitors'")
    print(f"  Position: {pos3}")
    print(f"  Dynamic dims: {pos3.describe_z(registry)}")
    print()
    
    # Distances
    print("DISTANCES:")
    print(f"  pos1 ↔ pos2 (full):     {pos1.distance(pos2):.3f}")
    print(f"  pos1 ↔ pos2 (z only):   {pos1.z_distance(pos2):.3f}")
    print(f"  pos1 ↔ pos3 (full):     {pos1.distance(pos3):.3f}")
    print(f"  pos2 ↔ pos3 (full):     {pos2.distance(pos3):.3f}")
    print()
    
    # Key insight
    print("KEY INSIGHT:")
    print("  All three sentences have IDENTICAL w, x, y positions!")
    print("  The ONLY difference is in the dynamic z-layer:")
    print("    - gender: she(-1) vs he(+1) vs they(0)")
    print("    - regality: table ware(0) vs finery(+2) vs china(+1.5)")
    print("    - vocabulary: put out(0) vs laid out(+1.5)")
    print()
    print("  This is exactly the kind of nuance that can't be")
    print("  captured in fixed grammatical dimensions!")
    print()


def demo_dimension_discovery():
    """
    Demonstrate how dimensions can be discovered from data.
    
    We don't need to predefine everything - dimensions emerge
    as we encounter new contrasts in the data.
    """
    print("=" * 70)
    print("DIMENSION DISCOVERY DEMO")
    print("=" * 70)
    print()
    
    registry = DynamicDimensionRegistry(max_dims=128)
    
    # Simulate discovering dimensions from contrasting pairs
    contrasts = [
        # (word1, word2, dimension_name)
        ('he', 'she', 'gender'),
        ('king', 'queen', 'gender'),
        ('boy', 'girl', 'gender'),
        
        ('finery', 'dishes', 'regality'),
        ('palace', 'house', 'regality'),
        ('monarch', 'person', 'regality'),
        
        ('quickly', 'slowly', 'tempo'),
        ('rushed', 'leisurely', 'tempo'),
        
        ('whispered', 'shouted', 'volume'),
        ('murmured', 'bellowed', 'volume'),
        
        ('ancient', 'modern', 'temporal_distance'),
        ('old', 'new', 'temporal_distance'),
        
        ('tiny', 'enormous', 'scale'),
        ('microscopic', 'cosmic', 'scale'),
    ]
    
    print("Discovering dimensions from contrasting pairs...")
    print()
    
    for word1, word2, dim_name in contrasts:
        idx = registry.register(dim_name)
        registry.add_anchor(dim_name, word1, 1.0)
        registry.add_anchor(dim_name, word2, -1.0)
    
    print(f"Discovered {registry.num_dims} dimensions:")
    for name, idx in registry._dimensions.items():
        anchors = registry._anchors[name]
        pos_anchors = [w for w, v in anchors.items() if v > 0]
        neg_anchors = [w for w, v in anchors.items() if v < 0]
        print(f"  [{idx}] {name}: {pos_anchors} ↔ {neg_anchors}")
    print()
    
    # Show how a word activates multiple dimensions
    test_words = ['king', 'queen', 'palace', 'whispered', 'ancient']
    print("Word → Dimension activations:")
    for word in test_words:
        activations = registry.get_level_for_word(word)
        if activations:
            print(f"  '{word}' → {activations}")
        else:
            print(f"  '{word}' → (no activations)")
    print()
    
    print("KEY INSIGHT:")
    print("  Dimensions EMERGE from contrasting pairs in data.")
    print("  We don't need to predefine 'regality' or 'tempo' -")
    print("  they appear when we see words that contrast along those axes.")
    print()


def demo_high_dimensional_navigation():
    """
    Demonstrate navigation in high-dimensional z-space.
    
    Even with 128 dimensions, we can navigate predictably
    by transforming along specific axes.
    """
    print("=" * 70)
    print("HIGH-DIMENSIONAL NAVIGATION DEMO")
    print("=" * 70)
    print()
    
    registry = DynamicDimensionRegistry(max_dims=128)
    
    # Register many dimensions
    dimensions = [
        'gender', 'regality', 'tempo', 'volume', 'scale',
        'formality', 'intimacy', 'urgency', 'certainty', 'politeness',
        'age', 'status', 'energy', 'warmth', 'complexity',
        'abstraction', 'emotionality', 'technicality', 'locality', 'temporality',
    ]
    
    for dim in dimensions:
        registry.register(dim)
    
    print(f"Registered {registry.num_dims} dimensions")
    print()
    
    # Create a base position
    base_z = {dim: 0.0 for dim in dimensions}
    base_z['gender'] = -1.0  # female
    base_z['regality'] = 0.0  # neutral
    base_z['formality'] = 0.0  # neutral
    
    base = DynamicQuaternionPosition.from_levels(
        w_levels=[1, 2, 1, 0],
        x_levels=[0, 0, 0, 1],
        y_levels=[0, 2, 0, 0],
        z_values=base_z,
        registry=registry
    )
    
    print("Base position: 'she put out the table ware for guests'")
    print(f"  Active z-dims: {base.describe_z(registry)}")
    print()
    
    # Define transformations
    def transform_z(pos: DynamicQuaternionPosition, 
                    changes: Dict[str, float],
                    registry: DynamicDimensionRegistry) -> DynamicQuaternionPosition:
        """Apply changes to z dimensions."""
        new_z = pos.z.copy()
        for dim_name, delta in changes.items():
            idx = registry.get_index(dim_name)
            if idx is not None and idx < len(new_z):
                new_z[idx] += delta
        return DynamicQuaternionPosition(
            w=pos.w.copy(), x=pos.x.copy(), y=pos.y.copy(), z=new_z
        )
    
    # Apply transformations
    print("TRANSFORMATIONS:")
    print()
    
    # Gender flip
    gender_flipped = transform_z(base, {'gender': 2.0}, registry)  # -1 + 2 = +1
    print("  gender_flip (+2.0):")
    print(f"    → {gender_flipped.describe_z(registry)}")
    print("    'he put out the table ware for guests'")
    print()
    
    # Increase regality
    regal = transform_z(base, {'regality': 2.0}, registry)
    print("  regality_increase (+2.0):")
    print(f"    → {regal.describe_z(registry)}")
    print("    'she put out the finery for company'")
    print()
    
    # Compose: gender flip + regality
    composed = transform_z(base, {'gender': 2.0, 'regality': 2.0}, registry)
    print("  gender_flip + regality_increase:")
    print(f"    → {composed.describe_z(registry)}")
    print("    'he put out the finery for company'")
    print()
    
    # Add urgency and formality
    complex_transform = transform_z(base, {
        'gender': 2.0,
        'regality': 2.0,
        'urgency': 1.5,
        'formality': 1.0,
    }, registry)
    print("  gender + regality + urgency + formality:")
    print(f"    → {complex_transform.describe_z(registry)}")
    print("    'He hastily arranged the fine china for the arriving dignitaries'")
    print()
    
    # Distances
    print("DISTANCES FROM BASE:")
    print(f"  → gender_flipped:    {base.distance(gender_flipped):.3f}")
    print(f"  → regal:             {base.distance(regal):.3f}")
    print(f"  → composed:          {base.distance(composed):.3f}")
    print(f"  → complex_transform: {base.distance(complex_transform):.3f}")
    print()
    
    print("KEY INSIGHT:")
    print(f"  With {registry.num_dims} dimensions, we can still navigate predictably.")
    print("  Each transformation is a vector addition in z-space.")
    print("  The structure remains navigable even at high dimensionality.")
    print()


def demo_generation_from_position():
    """
    Demonstrate text generation from a target position.
    
    Given a position in the full quaternion space (including z),
    we can describe what text should exist there.
    """
    print("=" * 70)
    print("GENERATION FROM POSITION DEMO")
    print("=" * 70)
    print()
    
    registry = DynamicDimensionRegistry(max_dims=128)
    
    # Register dimensions
    for dim in ['gender', 'regality', 'tempo', 'formality', 'intimacy', 'urgency']:
        registry.register(dim)
    
    # Define a target position
    target = DynamicQuaternionPosition.from_levels(
        w_levels=[1, 2, 1, 1],  # general, specific, inform, professional
        x_levels=[2, 2, 0, 1],  # future, prospective, indicative, active
        y_levels=[1, 2, 1, 0],  # formal register, direct, polite
        z_values={
            'gender': 1.0,      # male
            'regality': 1.5,    # elevated
            'tempo': 0.5,       # slightly quick
            'formality': 1.0,   # formal
            'intimacy': -0.5,   # somewhat distant
            'urgency': 0.0,     # neutral
        },
        registry=registry
    )
    
    print("TARGET POSITION:")
    print(f"  w (semantic):    general, specific, inform, professional")
    print(f"  x (grammatical): future, prospective, indicative, active")
    print(f"  y (contextual):  formal, direct, polite, neutral")
    print(f"  z (dynamic):     {target.describe_z(registry)}")
    print()
    
    print("PREDICTED TEXT CHARACTERISTICS:")
    print("  - Domain: general (not specialized)")
    print("  - Specificity: specific action")
    print("  - Intent: informative")
    print("  - Formality: professional tone")
    print("  - Tense: future")
    print("  - Aspect: prospective (about to happen)")
    print("  - Voice: active")
    print("  - Register: formal")
    print("  - Gender: male subject")
    print("  - Regality: elevated vocabulary")
    print("  - Tempo: slightly brisk")
    print()
    
    print("EXAMPLE TEXT THAT FITS THIS POSITION:")
    print('  "The gentleman will shortly arrange the fine settings')
    print('   for the distinguished guests."')
    print()
    
    print("CONTRAST WITH TRANSFORMED POSITION:")
    
    # Transform to casual, female, low regality
    casual = DynamicQuaternionPosition.from_levels(
        w_levels=[1, 2, 1, -1],  # informal
        x_levels=[2, 2, 0, 1],   # same tense
        y_levels=[-1, 2, 0, 0],  # casual register
        z_values={
            'gender': -1.0,     # female
            'regality': -1.0,   # common
            'tempo': 0.0,
            'formality': -1.0,  # informal
            'intimacy': 1.0,    # intimate
            'urgency': 0.0,
        },
        registry=registry
    )
    
    print()
    print("  Transformed position (casual, female, common):")
    print(f"    z: {casual.describe_z(registry)}")
    print()
    print('  "She\'s gonna set out the plates for the folks coming over."')
    print()
    
    print(f"  Distance between positions: {target.distance(casual):.3f}")
    print()
    
    print("KEY INSIGHT:")
    print("  The SAME core action (setting table for visitors)")
    print("  produces COMPLETELY different text based on z-dimensions.")
    print("  This is the power of dynamic layers - capturing nuance")
    print("  that can't be expressed in fixed grammatical categories.")
    print()


if __name__ == "__main__":
    demo_regality()
    print("\n")
    demo_dimension_discovery()
    print("\n")
    demo_high_dimensional_navigation()
    print("\n")
    demo_generation_from_position()
