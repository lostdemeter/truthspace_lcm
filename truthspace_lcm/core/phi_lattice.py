"""
φ-Lattice Coordinate System

Provides absolute coordinates based on φ^k levels with semantic dimensions.
Replaces relative eigenspace coordinates for knowledge matching.

Design Principles (from Design 099):
- Positions at φ^k for integer k (absolute, verifiable)
- Semantic dimensions with clear meaning
- No DC component - positions aren't derived from similarity
- Full dynamic range: φ^(-10) to φ^(+10)

The key insight: Similarity matrices point at absolute positions,
but eigenspace gives us relative coordinates. The φ-lattice provides
the absolute coordinate system that similarity is pointing to.

Connection to existing work:
- Zeta Line Method: Values naturally cluster at φ^(-k) levels
- Kerr Truth Space: Event horizon at σ = 1/(2φ), natural curve exists
- Old TruthSpace: Used φ^level on semantic dimensions

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Golden ratio - the fundamental constant
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895

# Precomputed φ-levels for efficiency
PHI_LEVELS = {k: PHI ** k for k in range(-15, 16)}


@dataclass
class SemanticDimension:
    """
    Definition of a semantic dimension in the φ-lattice.
    
    Each dimension has:
    - index: Position in the coordinate vector
    - name: Human-readable identifier
    - description: What this dimension represents
    - level_meanings: Semantic meaning of each φ-level
    - weight: Importance weight for distance calculations
    """
    index: int
    name: str
    description: str
    level_meanings: Dict[int, str] = field(default_factory=dict)
    weight: float = 1.0
    
    def level_to_value(self, level: int) -> float:
        """Convert φ-level index to actual value."""
        return PHI ** level
    
    def value_to_level(self, value: float) -> int:
        """Convert value to nearest φ-level index."""
        if abs(value) < 1e-10:
            return -15  # Minimum level for near-zero
        level = round(np.log(abs(value)) / np.log(PHI))
        return max(-15, min(15, level))
    
    def get_meaning(self, level: int) -> str:
        """Get semantic meaning for a level, or 'unknown' if not defined."""
        return self.level_meanings.get(level, f"level_{level}")


class PhiLattice:
    """
    φ-Lattice coordinate system with semantic dimensions.
    
    Positions are at φ^k for integer k on each dimension.
    Each dimension has semantic meaning defined at initialization.
    
    This provides ABSOLUTE coordinates (unlike eigenspace which is relative):
    - Positions are mathematically verifiable
    - No DC component problem
    - Full dynamic range: φ^(-15) to φ^(+15) ≈ [0.0006, 1364]
    
    Usage:
        lattice = PhiLattice(dimensions)
        
        # Convert levels to position
        pos = lattice.levels_to_position([2, 1, 0, -1])
        
        # Snap arbitrary position to lattice
        snapped = lattice.snap_to_lattice(noisy_position)
        
        # Check validity
        is_valid = lattice.is_valid_position(pos)
        
        # Compute distance
        dist = lattice.distance(pos_a, pos_b)
    """
    
    def __init__(self, dimensions: List[SemanticDimension]):
        """
        Initialize φ-lattice with semantic dimensions.
        
        Args:
            dimensions: List of SemanticDimension objects defining the space
        """
        self.dimensions = {d.index: d for d in dimensions}
        self.ndim = len(dimensions)
        
        # Build weights array from dimensions
        self._weights = np.array([
            self.dimensions.get(i, SemanticDimension(i, f"dim_{i}", "")).weight
            for i in range(self.ndim)
        ])
        
        # Dimension name lookup
        self._name_to_index = {d.name: d.index for d in dimensions}
    
    def levels_to_position(self, levels: List[int]) -> np.ndarray:
        """
        Convert φ-level indices to position vector.
        
        Args:
            levels: List of integer levels, one per dimension
            
        Returns:
            Position vector with values at φ^k for each level k
        """
        if len(levels) != self.ndim:
            raise ValueError(f"Expected {self.ndim} levels, got {len(levels)}")
        return np.array([PHI ** k for k in levels])
    
    def position_to_levels(self, position: np.ndarray) -> List[int]:
        """
        Convert position to nearest φ-level indices.
        
        Args:
            position: Position vector
            
        Returns:
            List of integer levels (nearest φ^k for each dimension)
        """
        levels = []
        for v in position:
            if abs(v) < 1e-10:
                levels.append(-15)
            else:
                k = round(np.log(abs(v)) / np.log(PHI))
                k = max(-15, min(15, k))
                levels.append(k)
        return levels
    
    def snap_to_lattice(self, position: np.ndarray) -> np.ndarray:
        """
        Snap position to nearest valid φ-lattice point.
        
        Args:
            position: Arbitrary position vector
            
        Returns:
            Position snapped to nearest lattice point
        """
        levels = self.position_to_levels(position)
        return self.levels_to_position(levels)
    
    def is_valid_position(self, position: np.ndarray, 
                          tolerance: float = 0.01) -> bool:
        """
        Check if position is on the φ-lattice.
        
        Args:
            position: Position to check
            tolerance: Maximum deviation from lattice point
            
        Returns:
            True if position is within tolerance of a lattice point
        """
        snapped = self.snap_to_lattice(position)
        return np.allclose(position, snapped, atol=tolerance, rtol=tolerance)
    
    def distance(self, a: np.ndarray, b: np.ndarray, 
                 weights: Optional[np.ndarray] = None) -> float:
        """
        Weighted Euclidean distance in φ-space.
        
        Args:
            a: First position
            b: Second position
            weights: Optional weight vector (uses dimension weights if None)
            
        Returns:
            Weighted Euclidean distance
        """
        if weights is None:
            weights = self._weights
        diff = (a - b) * np.sqrt(weights)
        return float(np.linalg.norm(diff))
    
    def similarity(self, a: np.ndarray, b: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> float:
        """
        Convert distance to similarity score.
        
        Args:
            a: First position
            b: Second position
            weights: Optional weight vector
            
        Returns:
            Similarity in range (0, 1], where 1 = identical
        """
        dist = self.distance(a, b, weights)
        return 1.0 / (1.0 + dist)
    
    def get_dimension(self, name_or_index) -> Optional[SemanticDimension]:
        """Get dimension by name or index."""
        if isinstance(name_or_index, str):
            idx = self._name_to_index.get(name_or_index)
            if idx is not None:
                return self.dimensions.get(idx)
            return None
        return self.dimensions.get(name_or_index)
    
    def describe_position(self, position: np.ndarray) -> Dict[str, str]:
        """
        Get semantic description of a position.
        
        Args:
            position: Position vector
            
        Returns:
            Dict mapping dimension name to semantic meaning
        """
        levels = self.position_to_levels(position)
        description = {}
        for i, level in enumerate(levels):
            dim = self.dimensions.get(i)
            if dim:
                description[dim.name] = dim.get_meaning(level)
        return description
    
    def __repr__(self) -> str:
        dim_names = [self.dimensions[i].name for i in range(self.ndim)]
        return f"PhiLattice(dims={self.ndim}, dimensions={dim_names})"


# Utility functions

def phi_level(k: int) -> float:
    """Get φ^k value for a level."""
    return PHI_LEVELS.get(k, PHI ** k)


def nearest_phi_level(value: float) -> Tuple[int, float]:
    """
    Find the nearest φ-level to a value.
    
    Returns:
        (level_index, level_value)
    """
    if abs(value) < 1e-10:
        return -15, PHI ** -15
    
    k = round(np.log(abs(value)) / np.log(PHI))
    k = max(-15, min(15, k))
    return k, PHI ** k


def is_phi_level(value: float, tolerance: float = 0.01) -> bool:
    """Check if a value is close to a φ-level."""
    _, nearest = nearest_phi_level(value)
    return abs(value - nearest) < tolerance * nearest
