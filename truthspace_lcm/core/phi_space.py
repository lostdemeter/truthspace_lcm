"""
φ-Space: The Simplest Possible API for Geometric Transformations

This is the "batteries included" version of GeometricSpace.
Maximum ease of use, minimum boilerplate.

Usage:
    from truthspace_lcm.core.phi_space import PhiSpace
    
    # Create a space and teach it transformations
    space = PhiSpace()
    space.learn("went", "will go", "tense")
    space.learn("sat", "will sit", "tense")
    
    # Transform!
    result = space("went", "tense")  # Returns "will go"
    
    # Or with explicit values
    result = space.transform("went", "tense", "past", "future")

The key insight: transformation pairs define concept identity.
If A transforms to B, they are the SAME concept in different states.

Author: Lesley Gushurst
License: GPLv3
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# The universal constant
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class Result:
    """Simple result container."""
    value: Any
    success: bool
    confidence: float = 1.0
    error: str = ""
    
    def __bool__(self):
        return self.success
    
    def __str__(self):
        return str(self.value) if self.success else f"Error: {self.error}"


class PhiSpace:
    """
    The simplest possible geometric transformation space.
    
    Core idea: Learn pairs, transform items.
    
    Examples:
        # Text transformations
        space = PhiSpace()
        space.learn("went", "will go", "tense")
        space.learn("sat", "will sit", "tense")
        print(space("went", "tense"))  # "will go"
        
        # Color transformations
        colors = PhiSpace()
        colors.learn("navy", "blue", "brightness")
        colors.learn("blue", "sky blue", "brightness")
        print(colors("navy", "brightness"))  # "blue"
        
        # Music transformations
        music = PhiSpace()
        music.learn("Am", "A", "mode")
        print(music("Am", "mode"))  # "A"
    """
    
    def __init__(self, 
                 normalize: Callable[[Any], str] = None,
                 denormalize: Callable[[str], Any] = None):
        """
        Create a new φ-space.
        
        Args:
            normalize: Function to convert items to keys (default: str.lower)
            denormalize: Function to convert keys back (default: identity)
        """
        self._normalize = normalize or (lambda x: str(x).lower().strip())
        self._denormalize = denormalize or (lambda x: x)
        
        # Core data structures
        self._concepts: Dict[str, int] = {}  # key -> concept_id
        self._concept_counter = 0
        
        # Pairs: dimension -> [(source_key, target_key, src_val, tgt_val)]
        self._pairs: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
        
        # Dimension levels: dimension -> {value: level}
        self._levels: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Positions: (key, dimension, value) -> position
        self._positions: Dict[Tuple[str, str, str], np.ndarray] = {}
        
        # Deltas: (dimension, src_val, tgt_val) -> delta
        self._deltas: Dict[Tuple[str, str, str], np.ndarray] = {}
        
        # Original items (for denormalization)
        self._originals: Dict[str, Any] = {}
    
    def learn(self, 
              source: Any, 
              target: Any, 
              dimension: str,
              source_value: str = "a",
              target_value: str = "b") -> 'PhiSpace':
        """
        Learn a transformation pair.
        
        Args:
            source: Source item
            target: Target item
            dimension: Name of the dimension (e.g., "tense", "brightness")
            source_value: Source state (default: "a")
            target_value: Target state (default: "b")
            
        Returns:
            self (for chaining)
        """
        src_key = self._normalize(source)
        tgt_key = self._normalize(target)
        
        # Store originals
        self._originals[src_key] = source
        self._originals[tgt_key] = target
        
        # Assign to same concept
        self._assign_concept(src_key, tgt_key)
        
        # Auto-assign levels
        if source_value not in self._levels[dimension]:
            self._levels[dimension][source_value] = len(self._levels[dimension])
        if target_value not in self._levels[dimension]:
            self._levels[dimension][target_value] = len(self._levels[dimension])
        
        # Record pair
        self._pairs[dimension].append((src_key, tgt_key, source_value, target_value))
        
        # Compute positions
        self._update_positions(src_key, dimension, source_value)
        self._update_positions(tgt_key, dimension, target_value)
        
        # Recompute deltas
        self._compute_deltas()
        
        return self
    
    def _assign_concept(self, key1: str, key2: str) -> int:
        """Assign both keys to the same concept."""
        if key1 not in self._concepts and key2 not in self._concepts:
            cid = self._concept_counter
            self._concepts[key1] = cid
            self._concepts[key2] = cid
            self._concept_counter += 1
            return cid
        elif key1 in self._concepts and key2 not in self._concepts:
            self._concepts[key2] = self._concepts[key1]
            return self._concepts[key1]
        elif key2 in self._concepts and key1 not in self._concepts:
            self._concepts[key1] = self._concepts[key2]
            return self._concepts[key2]
        else:
            # Merge concepts
            old = self._concepts[key2]
            new = self._concepts[key1]
            if old != new:
                for k, c in list(self._concepts.items()):
                    if c == old:
                        self._concepts[k] = new
            return new
    
    def _update_positions(self, key: str, dimension: str, value: str):
        """Update position for a key in a dimension state."""
        concept_id = self._concepts.get(key, 0)
        level = self._levels[dimension].get(value, 0)
        
        # Position: [content, dim_value]
        # Content = concept_id × φ
        # Dim value = φ^level
        ndims = 1 + len(self._levels)
        pos = np.ones(ndims) * PHI
        pos[0] = concept_id * PHI
        
        dim_idx = list(self._levels.keys()).index(dimension) + 1
        pos[dim_idx] = PHI ** level
        
        self._positions[(key, dimension, value)] = pos
    
    def _compute_deltas(self):
        """Compute canonical deltas."""
        for dim, pairs in self._pairs.items():
            grouped = defaultdict(list)
            for src_key, tgt_key, src_val, tgt_val in pairs:
                grouped[(src_val, tgt_val)].append((src_key, tgt_key))
            
            for (src_val, tgt_val), key_pairs in grouped.items():
                deltas = []
                for src_key, tgt_key in key_pairs:
                    src_pos = self._positions.get((src_key, dim, src_val))
                    tgt_pos = self._positions.get((tgt_key, dim, tgt_val))
                    if src_pos is not None and tgt_pos is not None:
                        deltas.append(tgt_pos - src_pos)
                
                if deltas:
                    self._deltas[(dim, src_val, tgt_val)] = np.mean(deltas, axis=0)
    
    def transform(self,
                  item: Any,
                  dimension: str,
                  source_value: str = None,
                  target_value: str = None) -> Result:
        """
        Transform an item along a dimension.
        
        Args:
            item: Item to transform
            dimension: Dimension to transform along
            source_value: Source state (auto-detected if None)
            target_value: Target state (auto-detected if None)
            
        Returns:
            Result with transformed item
        """
        key = self._normalize(item)
        
        if key not in self._concepts:
            return Result(None, False, 0.0, f"Unknown item: {item}")
        
        # Auto-detect source/target values if not provided
        if source_value is None or target_value is None:
            # Find the pair that includes this key
            for src_key, tgt_key, src_val, tgt_val in self._pairs.get(dimension, []):
                if src_key == key:
                    source_value = source_value or src_val
                    target_value = target_value or tgt_val
                    break
                elif tgt_key == key:
                    source_value = source_value or tgt_val
                    target_value = target_value or src_val
                    break
        
        if source_value is None or target_value is None:
            return Result(None, False, 0.0, f"Cannot determine transformation direction")
        
        # Get delta
        delta_key = (dimension, source_value, target_value)
        if delta_key not in self._deltas:
            return Result(None, False, 0.0, f"No delta for {dimension}: {source_value} → {target_value}")
        
        delta = self._deltas[delta_key]
        
        # Get source position
        pos_key = (key, dimension, source_value)
        if pos_key not in self._positions:
            self._update_positions(key, dimension, source_value)
        
        source_pos = self._positions[pos_key]
        target_pos = source_pos + delta
        
        # Find nearest
        best_key = None
        best_dist = float('inf')
        
        for (k, d, v), pos in self._positions.items():
            if d == dimension and v == target_value:
                dist = np.linalg.norm(pos - target_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_key = k
        
        if best_key is None:
            return Result(None, False, 0.0, "No target found")
        
        # Return original form if available
        result = self._originals.get(best_key, self._denormalize(best_key))
        confidence = 1.0 / (1.0 + best_dist)
        
        return Result(result, True, confidence)
    
    def __call__(self, item: Any, dimension: str, 
                 source_value: str = None, target_value: str = None) -> Any:
        """
        Shorthand for transform.
        
        Returns the transformed item directly, or None if failed.
        """
        result = self.transform(item, dimension, source_value, target_value)
        return result.value if result.success else None
    
    def __len__(self) -> int:
        """Number of items in the space."""
        return len(self._concepts)
    
    def __contains__(self, item: Any) -> bool:
        """Check if an item is in the space."""
        return self._normalize(item) in self._concepts
    
    def dimensions(self) -> List[str]:
        """List of dimensions."""
        return list(self._levels.keys())
    
    def values(self, dimension: str) -> List[str]:
        """List of values for a dimension."""
        return list(self._levels.get(dimension, {}).keys())
    
    def items(self) -> List[Any]:
        """List of all items."""
        return [self._originals.get(k, k) for k in self._concepts.keys()]
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'items': len(self._concepts),
            'concepts': self._concept_counter,
            'dimensions': self.dimensions(),
            'pairs': sum(len(p) for p in self._pairs.values()),
        }
    
    def __repr__(self) -> str:
        dims = ", ".join(self.dimensions()) if self.dimensions() else "none"
        return f"PhiSpace({len(self)} items, dimensions: {dims})"


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

def tense_space() -> PhiSpace:
    """Pre-configured space for English tense transformations."""
    space = PhiSpace()
    
    pairs = [
        ("went", "go", "goes", "will go"),
        ("sat", "sit", "sits", "will sit"),
        ("stood", "stand", "stands", "will stand"),
        ("walked", "walk", "walks", "will walk"),
        ("ran", "run", "runs", "will run"),
        ("came", "come", "comes", "will come"),
        ("was", "is", "is", "will be"),
        ("were", "are", "are", "will be"),
        ("had", "have", "has", "will have"),
        ("did", "do", "does", "will do"),
    ]
    
    for past, present, third, future in pairs:
        space.learn(past, present, "tense", "past", "present")
        space.learn(past, future, "tense", "past", "future")
        space.learn(present, future, "tense", "present", "future")
    
    return space


def color_space() -> PhiSpace:
    """Pre-configured space for color brightness transformations."""
    space = PhiSpace()
    
    pairs = [
        ("navy", "blue", "sky blue"),
        ("maroon", "red", "pink"),
        ("forest green", "green", "lime"),
        ("charcoal", "gray", "silver"),
        ("chocolate", "brown", "tan"),
        ("indigo", "purple", "lavender"),
    ]
    
    for dark, medium, light in pairs:
        space.learn(dark, medium, "brightness", "dark", "medium")
        space.learn(medium, light, "brightness", "medium", "light")
    
    return space


def music_space() -> PhiSpace:
    """Pre-configured space for music chord transformations."""
    space = PhiSpace()
    
    # Minor to major
    for note in ["A", "B", "C", "D", "E", "F", "G"]:
        space.learn(f"{note}m", note, "mode", "minor", "major")
    
    # Seventh chords
    for note in ["A", "B", "C", "D", "E", "F", "G"]:
        space.learn(note, f"{note}7", "extension", "triad", "seventh")
        space.learn(f"{note}m", f"{note}m7", "extension", "triad", "seventh")
    
    return space
