#!/usr/bin/env python3
"""
PhiSpace - A Geometric Data Structure

A standard computer science data structure based on the Music Box Principle:
- Drum: Data positioned in φ-space
- Comb: Geometric operations (project, transform, query)
- Music: Emergent outputs from structure

This is like a dictionary, but spatial:
- Keys are positions in n-dimensional φ-space
- Values are any data
- Operations are geometric (nearest-neighbor, projection, transformation)

Usage:
    space = PhiSpace(dims=4)
    space.add("hello", [0, 0, 0, 0])
    space.add("goodbye", [1, 0, 0, 0])
    
    nearest = space.query([0.1, 0, 0, 0])  # Returns "hello"
    transformed = space.transform("hello", delta=[1, 0, 0, 0])  # Returns "goodbye"

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
import json
import pickle
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path

try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
LOG_PHI = np.log(PHI)


@dataclass
class PhiPoint:
    """A point in φ-space."""
    position: np.ndarray
    data: Any
    phi_level: float = field(init=False)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.phi_level = np.log(np.linalg.norm(self.position) + 1e-10) / LOG_PHI
    
    def distance_to(self, other: Union['PhiPoint', np.ndarray]) -> float:
        """Euclidean distance to another point or position."""
        if isinstance(other, PhiPoint):
            return np.linalg.norm(self.position - other.position)
        return np.linalg.norm(self.position - np.asarray(other))
    
    def phi_distance_to(self, other: Union['PhiPoint', np.ndarray]) -> float:
        """Distance in φ-level (logarithmic scale)."""
        if isinstance(other, PhiPoint):
            other_level = other.phi_level
        else:
            other_level = np.log(np.linalg.norm(other) + 1e-10) / LOG_PHI
        return abs(self.phi_level - other_level)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'position': self.position.tolist(),
            'data': self.data if isinstance(self.data, (str, int, float, bool, list, dict)) else str(self.data),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PhiPoint':
        """Deserialize from dictionary."""
        point = cls(position=d['position'], data=d['data'])
        point.metadata = d.get('metadata', {})
        return point


class PhiSpace:
    """
    A geometric data structure for storing and querying data in φ-space.
    
    Like a dictionary, but spatial:
    - add(data, position) - Store data at a position
    - query(position, k=1) - Find k nearest items
    - transform(data, delta) - Move data and find what's there
    - project(position, weights) - Compute weighted sum (decode)
    
    The key insight: Structure IS information. The positions encode
    relationships, and operations are geometric, not lookup-based.
    """
    
    def __init__(self, dims: int = 4, dim_names: List[str] = None):
        """
        Initialize a PhiSpace.
        
        Args:
            dims: Number of dimensions
            dim_names: Optional names for each dimension (for semantics)
        """
        self.dims = dims
        self.dim_names = dim_names or [f'dim_{i}' for i in range(dims)]
        self.points: List[PhiPoint] = []
        self._index: Dict[Any, int] = {}  # data -> point index for O(1) lookup
        
        # Comb configurations (different ways to read the drum)
        self.combs: Dict[str, np.ndarray] = {}
        
        # Spatial index for fast queries (built lazily)
        self._kdtree: Optional['cKDTree'] = None
        self._kdtree_dirty: bool = True
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __contains__(self, data: Any) -> bool:
        return data in self._index
    
    def __iter__(self):
        return iter(self.points)
    
    def __getitem__(self, data: Any) -> PhiPoint:
        """Get point by data value."""
        if data not in self._index:
            raise KeyError(f"Data not in space: {data}")
        return self.points[self._index[data]]
    
    # ==================== DRUM OPERATIONS (Data Storage) ====================
    
    def add(self, data: Any, position: Union[List, np.ndarray], 
            metadata: Dict = None) -> PhiPoint:
        """
        Add data at a position in φ-space.
        
        Args:
            data: The data to store (like a dictionary value)
            position: The position in φ-space (like a spatial key)
            metadata: Optional metadata dictionary
        
        Returns:
            The created PhiPoint
        """
        position = np.asarray(position, dtype=np.float32)
        if len(position) != self.dims:
            raise ValueError(f"Position must have {self.dims} dimensions, got {len(position)}")
        
        point = PhiPoint(position=position, data=data, metadata=metadata or {})
        
        if data in self._index:
            # Update existing
            self.points[self._index[data]] = point
        else:
            # Add new
            self._index[data] = len(self.points)
            self.points.append(point)
        
        # Mark spatial index as dirty
        self._kdtree_dirty = True
        
        return point
    
    def remove(self, data: Any) -> bool:
        """
        Remove data from the space.
        
        Args:
            data: The data to remove
        
        Returns:
            True if removed, False if not found
        """
        if data not in self._index:
            return False
        
        idx = self._index[data]
        del self.points[idx]
        del self._index[data]
        
        # Rebuild index for items after the removed one
        for d, i in list(self._index.items()):
            if i > idx:
                self._index[d] = i - 1
        
        # Mark spatial index as dirty
        self._kdtree_dirty = True
        
        return True
    
    def update_position(self, data: Any, new_position: Union[List, np.ndarray]) -> PhiPoint:
        """
        Update the position of existing data.
        
        Args:
            data: The data to move
            new_position: The new position
        
        Returns:
            The updated PhiPoint
        """
        if data not in self._index:
            raise KeyError(f"Data not in space: {data}")
        
        return self.add(data, new_position, self.points[self._index[data]].metadata)
    
    def get_position(self, data: Any) -> Optional[np.ndarray]:
        """Get the position of data, or None if not found."""
        if data not in self._index:
            return None
        return self.points[self._index[data]].position.copy()
    
    # ==================== COMB OPERATIONS (Geometric Processing) ====================
    
    def _build_kdtree(self):
        """Build or rebuild the KD-tree spatial index."""
        if not HAS_KDTREE or len(self.points) == 0:
            self._kdtree = None
            return
        
        positions = np.array([p.position for p in self.points], dtype=np.float32)
        self._kdtree = cKDTree(positions)
        self._kdtree_dirty = False
    
    def build_index(self):
        """Explicitly build the spatial index. Call after bulk insertions."""
        self._build_kdtree()
    
    def query(self, position: Union[List, np.ndarray], k: int = 1) -> List[Tuple[Any, float]]:
        """
        Find k nearest items to a position.
        
        This is the core "comb" operation - reading the drum.
        Uses KD-tree for O(log n) queries when available.
        
        Args:
            position: The query position
            k: Number of nearest neighbors to return
        
        Returns:
            List of (data, distance) tuples, sorted by distance
        """
        position = np.asarray(position, dtype=np.float32)
        
        if len(self.points) == 0:
            return []
        
        # Use KD-tree if available and we have enough points
        if HAS_KDTREE and len(self.points) > 100:
            if self._kdtree_dirty:
                self._build_kdtree()
            
            if self._kdtree is not None:
                k_actual = min(k, len(self.points))
                distances, indices = self._kdtree.query(position, k=k_actual)
                
                # Handle single result case
                if k_actual == 1:
                    distances = [distances]
                    indices = [indices]
                
                return [(self.points[idx].data, float(dist)) 
                        for dist, idx in zip(distances, indices)]
        
        # Fallback to brute force for small spaces
        distances = []
        for point in self.points:
            dist = point.distance_to(position)
            distances.append((point.data, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def query_radius(self, position: Union[List, np.ndarray], radius: float) -> List[Tuple[Any, float]]:
        """
        Find all items within a radius of a position.
        
        Args:
            position: The query position
            radius: The search radius
        
        Returns:
            List of (data, distance) tuples within radius
        """
        position = np.asarray(position, dtype=np.float32)
        
        results = []
        for point in self.points:
            dist = point.distance_to(position)
            if dist <= radius:
                results.append((point.data, dist))
        
        results.sort(key=lambda x: x[1])
        return results
    
    def transform(self, data: Any, delta: Union[List, np.ndarray]) -> Any:
        """
        Transform data by applying a delta vector and finding nearest.
        
        This is the Music Box Principle in action:
        1. Find current position
        2. Apply delta
        3. Find nearest at new position
        
        Args:
            data: The data to transform
            delta: The transformation vector
        
        Returns:
            The nearest data at the new position
        """
        if data not in self._index:
            return data  # Not in space, return unchanged
        
        current_pos = self.points[self._index[data]].position
        new_pos = current_pos + np.asarray(delta, dtype=np.float32)
        
        nearest = self.query(new_pos, k=1)
        if nearest:
            return nearest[0][0]
        return data
    
    def project(self, position: Union[List, np.ndarray], weights: Union[List, np.ndarray]) -> float:
        """
        Project a position using weights (like DA2's depth decoding).
        
        output = Σ weight_i × position_i
        
        Args:
            position: The position to project
            weights: The projection weights
        
        Returns:
            The projected scalar value
        """
        position = np.asarray(position, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        return float(np.dot(position, weights))
    
    def phi_project(self, position: Union[List, np.ndarray], exponents: Union[List, np.ndarray]) -> float:
        """
        Project using φ-exponent weights.
        
        weight_i = φ^exponent_i
        output = Σ φ^exponent_i × position_i
        
        Args:
            position: The position to project
            exponents: The φ-exponents for each dimension
        
        Returns:
            The projected scalar value
        """
        weights = np.array([PHI ** e for e in exponents], dtype=np.float32)
        return self.project(position, weights)
    
    # ==================== COMB CONFIGURATION ====================
    
    def add_comb(self, name: str, weights: Union[List, np.ndarray]):
        """
        Add a named comb (projection configuration).
        
        Args:
            name: Name of the comb
            weights: The projection weights
        """
        self.combs[name] = np.asarray(weights, dtype=np.float32)
    
    def add_phi_comb(self, name: str, exponents: Union[List, np.ndarray]):
        """
        Add a named comb using φ-exponents.
        
        Args:
            name: Name of the comb
            exponents: The φ-exponents
        """
        weights = np.array([PHI ** e for e in exponents], dtype=np.float32)
        self.combs[name] = weights
    
    def decode(self, position: Union[List, np.ndarray], comb: str) -> float:
        """
        Decode a position using a named comb.
        
        Args:
            position: The position to decode
            comb: Name of the comb to use
        
        Returns:
            The decoded value
        """
        if comb not in self.combs:
            raise KeyError(f"Comb not found: {comb}")
        return self.project(position, self.combs[comb])
    
    # ==================== BULK OPERATIONS ====================
    
    def add_many(self, items: List[Tuple[Any, Union[List, np.ndarray]]]):
        """
        Add multiple items at once.
        
        Args:
            items: List of (data, position) tuples
        """
        for data, position in items:
            self.add(data, position)
    
    def all_positions(self) -> np.ndarray:
        """Get all positions as a numpy array."""
        if not self.points:
            return np.array([]).reshape(0, self.dims)
        return np.array([p.position for p in self.points])
    
    def all_data(self) -> List[Any]:
        """Get all data values."""
        return [p.data for p in self.points]
    
    # ==================== STATISTICS ====================
    
    def stats(self) -> Dict:
        """Get statistics about the space."""
        if not self.points:
            return {'count': 0}
        
        positions = self.all_positions()
        phi_levels = [p.phi_level for p in self.points]
        
        return {
            'count': len(self.points),
            'dims': self.dims,
            'dim_names': self.dim_names,
            'position_mean': positions.mean(axis=0).tolist(),
            'position_std': positions.std(axis=0).tolist(),
            'position_min': positions.min(axis=0).tolist(),
            'position_max': positions.max(axis=0).tolist(),
            'phi_level_mean': float(np.mean(phi_levels)),
            'phi_level_std': float(np.std(phi_levels)),
            'phi_level_range': [float(min(phi_levels)), float(max(phi_levels))],
            'combs': list(self.combs.keys()),
        }
    
    # ==================== SERIALIZATION ====================
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'dims': self.dims,
            'dim_names': self.dim_names,
            'points': [p.to_dict() for p in self.points],
            'combs': {k: v.tolist() for k, v in self.combs.items()},
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PhiSpace':
        """Deserialize from dictionary."""
        space = cls(dims=d['dims'], dim_names=d.get('dim_names'))
        for pd in d['points']:
            point = PhiPoint.from_dict(pd)
            space._index[point.data] = len(space.points)
            space.points.append(point)
        for name, weights in d.get('combs', {}).items():
            space.combs[name] = np.array(weights, dtype=np.float32)
        return space
    
    def save(self, path: Union[str, Path], format: str = 'json'):
        """
        Save to file.
        
        Args:
            path: File path
            format: 'json' or 'pickle'
        """
        path = Path(path)
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path], format: str = 'json') -> 'PhiSpace':
        """
        Load from file.
        
        Args:
            path: File path
            format: 'json' or 'pickle'
        
        Returns:
            Loaded PhiSpace
        """
        path = Path(path)
        if format == 'json':
            with open(path, 'r') as f:
                return cls.from_dict(json.load(f))
        elif format == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    # ==================== CONVENIENCE METHODS ====================
    
    def nearest(self, position: Union[List, np.ndarray]) -> Any:
        """Get the single nearest data item to a position."""
        result = self.query(position, k=1)
        return result[0][0] if result else None
    
    def interpolate(self, pos1: Union[List, np.ndarray], pos2: Union[List, np.ndarray], 
                    t: float, phi_weighted: bool = True) -> np.ndarray:
        """
        Interpolate between two positions.
        
        Args:
            pos1: Start position
            pos2: End position
            t: Interpolation parameter (0 to 1)
            phi_weighted: Use φ-weighted interpolation
        
        Returns:
            Interpolated position
        """
        pos1 = np.asarray(pos1, dtype=np.float32)
        pos2 = np.asarray(pos2, dtype=np.float32)
        
        if phi_weighted:
            t = t ** INV_PHI  # Non-linear φ-scaling
        
        return pos1 * (1 - t) + pos2 * t


# ==================== CONVENIENCE FUNCTIONS ====================

def create_semantic_space(dim_names: List[str] = None) -> PhiSpace:
    """
    Create a PhiSpace for semantic/vocabulary use.
    
    Default dimensions: [tense, formality, domain, intensity]
    """
    dim_names = dim_names or ['tense', 'formality', 'domain', 'intensity']
    return PhiSpace(dims=len(dim_names), dim_names=dim_names)


def create_feature_space(n_dims: int, feature_name: str = 'feature') -> PhiSpace:
    """
    Create a PhiSpace for feature vectors (like DA2).
    
    Args:
        n_dims: Number of dimensions
        feature_name: Name prefix for dimensions
    """
    dim_names = [f'{feature_name}_{i}' for i in range(n_dims)]
    return PhiSpace(dims=n_dims, dim_names=dim_names)


# ==================== DEMO ====================

if __name__ == "__main__":
    print("=" * 60)
    print("PhiSpace - A Geometric Data Structure")
    print("=" * 60)
    
    # Create a semantic space
    space = create_semantic_space()
    
    # Add vocabulary (the "drum")
    vocab = [
        ("code", [0, 0, 1, 0]),
        ("holy scripture", [0, 2, 2, 1]),
        ("treasure map", [0, -1, -1, 0]),
        ("went", [-1, 0, 0, 0]),
        ("will go", [1, 0, 0, 0]),
        ("did proceed", [-0.5, 2, 0, 0.5]),
    ]
    
    for word, pos in vocab:
        space.add(word, pos)
    
    print(f"\n1. Created space with {len(space)} items")
    print(f"   Dimensions: {space.dim_names}")
    
    # Query (the "comb" reading the "drum")
    print("\n2. Query operations:")
    query_pos = [0, 1.5, 1.5, 0.5]
    nearest = space.query(query_pos, k=2)
    print(f"   Nearest to {query_pos}:")
    for data, dist in nearest:
        print(f"     {data}: distance={dist:.3f}")
    
    # Transform
    print("\n3. Transform operations:")
    warhammer_delta = [0, 2, 2, 0.5]
    result = space.transform("code", warhammer_delta)
    print(f"   'code' + Warhammer delta = '{result}'")
    
    future_delta = [2, 0, 0, 0]
    result = space.transform("went", future_delta)
    print(f"   'went' + future delta = '{result}'")
    
    # Add a comb for projection
    print("\n4. Projection with combs:")
    space.add_phi_comb('intensity_decoder', [0, 0, 0, 1])  # φ^1 weight on intensity
    
    for word in ["code", "holy scripture", "treasure map"]:
        pos = space.get_position(word)
        intensity = space.decode(pos, 'intensity_decoder')
        print(f"   {word}: intensity = {intensity:.3f}")
    
    # Statistics
    print("\n5. Space statistics:")
    stats = space.stats()
    print(f"   Count: {stats['count']}")
    print(f"   φ-level range: {stats['phi_level_range']}")
    
    # Serialization
    print("\n6. Serialization:")
    space.save("/tmp/test_phi_space.json")
    loaded = PhiSpace.load("/tmp/test_phi_space.json")
    print(f"   Saved and loaded: {len(loaded)} items")
    
    print("\n" + "=" * 60)
    print("PhiSpace is ready for use!")
    print("=" * 60)
    print("""
Usage:
    from phi_space import PhiSpace
    
    space = PhiSpace(dims=4)
    space.add("hello", [0, 0, 0, 0])
    space.add("goodbye", [1, 0, 0, 0])
    
    nearest = space.query([0.1, 0, 0, 0])  # [("hello", 0.1)]
    result = space.transform("hello", [1, 0, 0, 0])  # "goodbye"
    
    space.save("my_space.json")
    loaded = PhiSpace.load("my_space.json")
""")
