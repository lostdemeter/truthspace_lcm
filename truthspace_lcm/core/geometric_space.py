"""
Geometric Space - Domain-Agnostic Transformation System

A generalized version of ConceptTransformer that works with ANY dimension,
including:
- Named dimensions (tense, voice, regality)
- Unnamed dimensions (discovered from data patterns)
- Non-linguistic domains (colors, music, code, etc.)

The core insight: transformation pairs define concept identity.
If A transforms to B along some dimension, they share the same concept.

Position = [content, dim1, dim2, dim3, ...]
- Content: concept_id × φ (unique per concept)
- Each dimension: φ^level (differs between states)
- Delta: exactly φ^(target_level - source_level)

This is PURE GEOMETRY - no domain-specific knowledge required.

Author: Lesley Gushurst
License: GPLv3
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Golden ratio - the universal constant
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Dimension:
    """
    A dimension in the geometric space.
    
    Dimensions can be:
    - Named with explicit levels (tense: past=0, present=1, future=2)
    - Discovered from data (unnamed, levels inferred)
    - Continuous (no discrete levels, just positions)
    """
    name: str
    levels: Dict[str, int] = field(default_factory=dict)
    is_discovered: bool = False  # True if discovered from data, not predefined
    
    def level_for(self, value: str) -> int:
        """Get level for a value, auto-assigning if needed."""
        if value not in self.levels:
            # Auto-assign next level
            self.levels[value] = len(self.levels)
        return self.levels[value]
    
    def position_for(self, value: str) -> float:
        """Get φ-based position for a value."""
        return PHI ** self.level_for(value)


@dataclass
class TransformResult:
    """Result of a geometric transformation."""
    original: Any
    transformed: Any
    source_item: Any
    target_item: Any
    dimension: str
    source_value: str
    target_value: str
    confidence: float
    success: bool
    failure_reason: str = ""
    was_injected: bool = False


# =============================================================================
# GEOMETRIC SPACE
# =============================================================================

class GeometricSpace:
    """
    Domain-agnostic geometric transformation space.
    
    Works with ANY type of items that can be:
    1. Grouped into concepts (items that transform to each other)
    2. Positioned in a multi-dimensional space
    3. Transformed by adding deltas
    
    The space discovers its own structure from transformation pairs.
    """
    
    def __init__(self, 
                 item_to_key: Callable[[Any], str] = None,
                 key_to_item: Callable[[str], Any] = None):
        """
        Initialize a geometric space.
        
        Args:
            item_to_key: Function to convert items to string keys (default: str)
            key_to_item: Function to convert keys back to items (default: identity)
        """
        self.item_to_key = item_to_key or str
        self.key_to_item = key_to_item or (lambda x: x)
        
        # Dimensions (discovered or predefined)
        self._dimensions: Dict[str, Dimension] = {}
        
        # Concept assignments (key -> concept_id)
        self._key_to_concept: Dict[str, int] = {}
        self._concept_counter = 0
        
        # Positions: (key, dimension, value) -> position vector
        self._positions: Dict[Tuple[str, str, str], np.ndarray] = {}
        
        # Transformation pairs per dimension
        self._pairs: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
        
        # Canonical deltas: (dimension, source_value, target_value) -> delta vector
        self._deltas: Dict[Tuple[str, str, str], np.ndarray] = {}
        
        # Temporary injections
        self._temporary_concepts: Set[int] = set()
        self._temporary_keys: Set[str] = set()
    
    # =========================================================================
    # DIMENSION MANAGEMENT
    # =========================================================================
    
    def add_dimension(self, name: str, levels: Dict[str, int] = None) -> Dimension:
        """
        Add a dimension to the space.
        
        Args:
            name: Dimension name
            levels: Optional mapping of value names to levels
            
        Returns:
            The created Dimension
        """
        dim = Dimension(
            name=name,
            levels=levels or {},
            is_discovered=levels is None
        )
        self._dimensions[name] = dim
        return dim
    
    def get_dimension(self, name: str) -> Optional[Dimension]:
        """Get a dimension by name, creating if needed."""
        if name not in self._dimensions:
            self._dimensions[name] = Dimension(name=name, is_discovered=True)
        return self._dimensions[name]
    
    @property
    def ndims(self) -> int:
        """Number of dimensions (content + transformation dims)."""
        return 1 + len(self._dimensions)
    
    # =========================================================================
    # CONCEPT MANAGEMENT
    # =========================================================================
    
    def _assign_concept(self, key1: str, key2: str) -> int:
        """Assign both keys to the same concept, returning the concept ID."""
        if key1 not in self._key_to_concept and key2 not in self._key_to_concept:
            # Neither has a concept - create new one
            concept_id = self._concept_counter
            self._key_to_concept[key1] = concept_id
            self._key_to_concept[key2] = concept_id
            self._concept_counter += 1
            return concept_id
        elif key1 in self._key_to_concept and key2 not in self._key_to_concept:
            concept_id = self._key_to_concept[key1]
            self._key_to_concept[key2] = concept_id
            return concept_id
        elif key2 in self._key_to_concept and key1 not in self._key_to_concept:
            concept_id = self._key_to_concept[key2]
            self._key_to_concept[key1] = concept_id
            return concept_id
        else:
            # Both have concepts - merge (use key1's)
            old_concept = self._key_to_concept[key2]
            new_concept = self._key_to_concept[key1]
            if old_concept != new_concept:
                for k, c in list(self._key_to_concept.items()):
                    if c == old_concept:
                        self._key_to_concept[k] = new_concept
            return new_concept
    
    def get_concept(self, key: str) -> Optional[int]:
        """Get concept ID for a key."""
        return self._key_to_concept.get(key)
    
    # =========================================================================
    # POSITION CALCULATION
    # =========================================================================
    
    def _get_position(self, key: str, dimension: str, value: str) -> np.ndarray:
        """
        Calculate position for a key in a specific dimension state.
        
        Position = [content, dim1, dim2, ...]
        - Content: concept_id × φ
        - Each dimension: φ^level
        """
        concept_id = self._key_to_concept.get(key, 0)
        dim_names = list(self._dimensions.keys())
        
        # Initialize with neutral (φ^1) for all dimensions
        pos = np.ones(self.ndims) * PHI
        
        # Content dimension: concept_id × φ
        pos[0] = concept_id * PHI
        
        # Set the specific dimension
        if dimension in self._dimensions:
            dim_idx = dim_names.index(dimension) + 1
            pos[dim_idx] = self._dimensions[dimension].position_for(value)
        
        return pos
    
    # =========================================================================
    # LEARNING FROM PAIRS
    # =========================================================================
    
    def learn_pair(self, 
                   source: Any, 
                   target: Any, 
                   dimension: str,
                   source_value: str,
                   target_value: str) -> None:
        """
        Learn a transformation pair.
        
        This is the core learning operation:
        1. Assign source and target to the same concept
        2. Compute their positions
        3. Record the pair for delta computation
        
        Args:
            source: Source item
            target: Target item  
            dimension: Dimension of transformation
            source_value: Source state in dimension
            target_value: Target state in dimension
        """
        # Ensure dimension exists
        self.get_dimension(dimension)
        
        # Convert to keys
        src_key = self.item_to_key(source)
        tgt_key = self.item_to_key(target)
        
        # Assign to same concept
        self._assign_concept(src_key, tgt_key)
        
        # Record pair
        self._pairs[dimension].append((src_key, tgt_key, source_value, target_value))
        
        # Compute positions
        key_src = (src_key, dimension, source_value)
        key_tgt = (tgt_key, dimension, target_value)
        
        if key_src not in self._positions:
            self._positions[key_src] = self._get_position(src_key, dimension, source_value)
        if key_tgt not in self._positions:
            self._positions[key_tgt] = self._get_position(tgt_key, dimension, target_value)
    
    def compute_deltas(self) -> Dict[Tuple[str, str, str], np.ndarray]:
        """
        Compute canonical deltas from learned pairs.
        
        Returns mapping of (dimension, source_value, target_value) -> delta vector.
        """
        for dim, pairs in self._pairs.items():
            # Group by (source_value, target_value)
            grouped = defaultdict(list)
            for src_key, tgt_key, src_val, tgt_val in pairs:
                grouped[(src_val, tgt_val)].append((src_key, tgt_key))
            
            for (src_val, tgt_val), key_pairs in grouped.items():
                deltas = []
                for src_key, tgt_key in key_pairs:
                    key_src = (src_key, dim, src_val)
                    key_tgt = (tgt_key, dim, tgt_val)
                    
                    if key_src in self._positions and key_tgt in self._positions:
                        delta = self._positions[key_tgt] - self._positions[key_src]
                        deltas.append(delta)
                
                if deltas:
                    # Use mean delta (should be identical for φ-based positions)
                    self._deltas[(dim, src_val, tgt_val)] = np.mean(deltas, axis=0)
        
        return self._deltas
    
    # =========================================================================
    # TRANSFORMATION
    # =========================================================================
    
    def transform(self,
                  item: Any,
                  dimension: str,
                  source_value: str,
                  target_value: str,
                  allow_injection: bool = False) -> TransformResult:
        """
        Transform an item along a dimension.
        
        Args:
            item: Item to transform
            dimension: Dimension to transform along
            source_value: Current state in dimension
            target_value: Target state in dimension
            allow_injection: If True, inject unknown items as temporary
            
        Returns:
            TransformResult with transformed item
        """
        key = self.item_to_key(item)
        
        # Check if we have this item
        if key not in self._key_to_concept:
            if allow_injection:
                return self._inject_and_transform(
                    item, key, dimension, source_value, target_value
                )
            return TransformResult(
                original=item,
                transformed=None,
                source_item=item,
                target_item=None,
                dimension=dimension,
                source_value=source_value,
                target_value=target_value,
                confidence=0.0,
                success=False,
                failure_reason=f"Unknown item: {key}"
            )
        
        # Get delta
        delta_key = (dimension, source_value, target_value)
        if delta_key not in self._deltas:
            return TransformResult(
                original=item,
                transformed=None,
                source_item=item,
                target_item=None,
                dimension=dimension,
                source_value=source_value,
                target_value=target_value,
                confidence=0.0,
                success=False,
                failure_reason=f"No delta for {dimension}: {source_value} → {target_value}"
            )
        
        delta = self._deltas[delta_key]
        
        # Get source position
        pos_key = (key, dimension, source_value)
        if pos_key not in self._positions:
            # Compute position on the fly
            self._positions[pos_key] = self._get_position(key, dimension, source_value)
        
        source_pos = self._positions[pos_key]
        target_pos = source_pos + delta
        
        # Find nearest item at target position
        target_item, confidence = self._find_nearest(
            target_pos, dimension, target_value
        )
        
        if target_item is None:
            return TransformResult(
                original=item,
                transformed=None,
                source_item=item,
                target_item=None,
                dimension=dimension,
                source_value=source_value,
                target_value=target_value,
                confidence=0.0,
                success=False,
                failure_reason="No item found at target position"
            )
        
        return TransformResult(
            original=item,
            transformed=self.key_to_item(target_item),
            source_item=item,
            target_item=self.key_to_item(target_item),
            dimension=dimension,
            source_value=source_value,
            target_value=target_value,
            confidence=confidence,
            success=True
        )
    
    def _find_nearest(self, 
                      target_pos: np.ndarray,
                      dimension: str,
                      target_value: str) -> Tuple[Optional[str], float]:
        """Find the nearest item to a target position."""
        best_key = None
        best_dist = float('inf')
        
        for (key, dim, val), pos in self._positions.items():
            if dim == dimension and val == target_value:
                dist = np.linalg.norm(pos - target_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_key = key
        
        if best_key is None:
            return None, 0.0
        
        # Convert distance to confidence (closer = higher confidence)
        confidence = 1.0 / (1.0 + best_dist)
        return best_key, confidence
    
    # =========================================================================
    # TEMPORARY INJECTION (Design 085)
    # =========================================================================
    
    def _inject_and_transform(self,
                              item: Any,
                              key: str,
                              dimension: str,
                              source_value: str,
                              target_value: str) -> TransformResult:
        """Inject an unknown item as temporary and attempt transformation."""
        # Create new concept for this item
        concept_id = self._concept_counter
        self._key_to_concept[key] = concept_id
        self._concept_counter += 1
        
        # Mark as temporary
        self._temporary_concepts.add(concept_id)
        self._temporary_keys.add(key)
        
        # Compute position
        pos_key = (key, dimension, source_value)
        self._positions[pos_key] = self._get_position(key, dimension, source_value)
        
        # Now try to transform
        result = self.transform(item, dimension, source_value, target_value, allow_injection=False)
        result.was_injected = True
        
        return result
    
    def promote_temporary(self,
                          source_key: str,
                          target_key: str,
                          dimension: str,
                          source_value: str,
                          target_value: str) -> bool:
        """
        Promote a temporary item to permanent by linking it with a target.
        
        This is called when an external system (like an LLM) provides the
        correct transformation, allowing us to learn from it.
        """
        if source_key not in self._temporary_keys:
            return False
        
        # Assign to same concept
        self._assign_concept(source_key, target_key)
        
        # Record pair
        self._pairs[dimension].append((source_key, target_key, source_value, target_value))
        
        # Compute target position
        tgt_pos_key = (target_key, dimension, target_value)
        if tgt_pos_key not in self._positions:
            self._positions[tgt_pos_key] = self._get_position(target_key, dimension, target_value)
        
        # Remove from temporary
        concept_id = self._key_to_concept.get(source_key)
        if concept_id in self._temporary_concepts:
            self._temporary_concepts.remove(concept_id)
        self._temporary_keys.discard(source_key)
        
        # Recompute deltas
        self.compute_deltas()
        
        return True
    
    def remove_temporary(self, key: str) -> bool:
        """Remove a temporary item (transformation failed)."""
        if key not in self._temporary_keys:
            return False
        
        concept_id = self._key_to_concept.get(key)
        
        # Remove from positions
        keys_to_remove = [k for k in self._positions if k[0] == key]
        for k in keys_to_remove:
            del self._positions[k]
        
        # Remove from concept mapping
        if key in self._key_to_concept:
            del self._key_to_concept[key]
        
        # Remove from temporary sets
        if concept_id in self._temporary_concepts:
            self._temporary_concepts.remove(concept_id)
        self._temporary_keys.discard(key)
        
        return True
    
    def clear_temporary(self) -> int:
        """Clear all temporary items. Returns count removed."""
        count = len(self._temporary_keys)
        for key in list(self._temporary_keys):
            self.remove_temporary(key)
        return count
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the space."""
        return {
            'concepts': self._concept_counter,
            'items': len(self._key_to_concept),
            'positions': len(self._positions),
            'dimensions': list(self._dimensions.keys()),
            'dimension_count': len(self._dimensions),
            'deltas': len(self._deltas),
            'pairs_per_dimension': {
                dim: len(pairs) for dim, pairs in self._pairs.items()
            },
            'temporary_items': len(self._temporary_keys),
        }
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the space to a dictionary."""
        return {
            'dimensions': {
                name: {'levels': dim.levels, 'is_discovered': dim.is_discovered}
                for name, dim in self._dimensions.items()
            },
            'concepts': self._key_to_concept,
            'concept_counter': self._concept_counter,
            'positions': {
                f"{k[0]}|{k[1]}|{k[2]}": pos.tolist()
                for k, pos in self._positions.items()
            },
            'pairs': {dim: pairs for dim, pairs in self._pairs.items()},
            'deltas': {
                f"{k[0]}|{k[1]}|{k[2]}": delta.tolist()
                for k, delta in self._deltas.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> 'GeometricSpace':
        """Deserialize a space from a dictionary."""
        space = cls(**kwargs)
        
        # Restore dimensions
        for name, dim_data in data.get('dimensions', {}).items():
            space._dimensions[name] = Dimension(
                name=name,
                levels=dim_data['levels'],
                is_discovered=dim_data.get('is_discovered', False)
            )
        
        # Restore concepts
        space._key_to_concept = data.get('concepts', {})
        space._concept_counter = data.get('concept_counter', 0)
        
        # Restore positions
        for key_str, pos_list in data.get('positions', {}).items():
            parts = key_str.split('|')
            if len(parts) == 3:
                key = (parts[0], parts[1], parts[2])
                space._positions[key] = np.array(pos_list)
        
        # Restore pairs
        for dim, pairs in data.get('pairs', {}).items():
            space._pairs[dim] = [tuple(p) for p in pairs]
        
        # Restore deltas
        for key_str, delta_list in data.get('deltas', {}).items():
            parts = key_str.split('|')
            if len(parts) == 3:
                key = (parts[0], parts[1], parts[2])
                space._deltas[key] = np.array(delta_list)
        
        return space
    
    def save(self, path: Path) -> None:
        """Save the space to a JSON file."""
        if isinstance(path, str):
            path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path, **kwargs) -> 'GeometricSpace':
        """Load a space from a JSON file."""
        if isinstance(path, str):
            path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, **kwargs)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_text_space() -> GeometricSpace:
    """
    Create a GeometricSpace configured for text/phrase transformation.
    
    Uses lowercase normalization for keys.
    """
    return GeometricSpace(
        item_to_key=lambda x: x.lower().strip(),
        key_to_item=lambda x: x
    )


def create_numeric_space() -> GeometricSpace:
    """
    Create a GeometricSpace for numeric transformations.
    
    Useful for things like unit conversions, scale transformations, etc.
    """
    return GeometricSpace(
        item_to_key=lambda x: str(x),
        key_to_item=lambda x: float(x) if '.' in x else int(x)
    )
