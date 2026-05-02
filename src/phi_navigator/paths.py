"""
Paths: Concept-Specific Transformations
=======================================

A Path is the specific transformation from one concept to another.
Unlike the universal φ-coordinates, paths are concept-specific.

hot→cold is a different path than tall→short.
Both are "opposites", but they traverse different dimensions.

Paths are stored, not computed. This gives us:
- 100% accuracy (we use the model's own knowledge)
- O(1) lookup (just retrieve the stored path)
"""

import torch
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import hashlib

try:
    from .coordinates import PhiPoint, PhiCoordinates
except ImportError:
    from coordinates import PhiPoint, PhiCoordinates


@dataclass
class RelationshipPath:
    """
    A specific path from source to target.
    
    Stores the exact transformation in φ-space.
    """
    source: str
    target: str
    relationship: str
    
    # The transformation in φ-space
    level_delta: Optional[List[float]] = None  # Per-dimension level change
    flip_dims: Optional[List[int]] = None      # Dimensions that flip sign
    
    # Metadata
    validated: bool = False
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'target': self.target,
            'relationship': self.relationship,
            'level_delta': self.level_delta,
            'flip_dims': self.flip_dims,
            'validated': self.validated,
            'confidence': self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'RelationshipPath':
        return cls(**d)
    
    def apply(self, point: PhiPoint) -> PhiPoint:
        """Apply this path transformation to a point."""
        new_point = point.clone()
        
        if self.flip_dims:
            flip_tensor = torch.tensor(self.flip_dims, dtype=torch.long)
            new_point = new_point.flip_dims(flip_tensor)
        
        if self.level_delta:
            delta_tensor = torch.tensor(self.level_delta, dtype=torch.float32)
            new_point = new_point.shift_levels(delta_tensor)
        
        return new_point


class PathStore:
    """
    Storage for relationship paths.
    
    Provides O(1) lookup for (source, relationship) → target.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Primary index: (source, relationship) → Path
        self._paths: Dict[Tuple[str, str], RelationshipPath] = {}
        
        # Reverse index: (target, relationship) → source (for symmetric relationships)
        self._reverse: Dict[Tuple[str, str], str] = {}
        
        # Relationship index: relationship → list of sources
        self._by_relationship: Dict[str, List[str]] = {}
        
        if self.storage_path and self.storage_path.exists():
            self.load()
    
    def add(self, path: RelationshipPath, symmetric: bool = True) -> None:
        """Add a path to the store."""
        key = (path.source, path.relationship)
        self._paths[key] = path
        
        # Update relationship index
        if path.relationship not in self._by_relationship:
            self._by_relationship[path.relationship] = []
        if path.source not in self._by_relationship[path.relationship]:
            self._by_relationship[path.relationship].append(path.source)
        
        # Add reverse mapping for symmetric relationships
        if symmetric:
            reverse_key = (path.target, path.relationship)
            self._reverse[reverse_key] = path.source
    
    def get(self, source: str, relationship: str) -> Optional[RelationshipPath]:
        """Get a path by source and relationship."""
        key = (source, relationship)
        return self._paths.get(key)
    
    def get_target(self, source: str, relationship: str) -> Optional[str]:
        """Get just the target for a source and relationship."""
        path = self.get(source, relationship)
        if path:
            return path.target
        
        # Check reverse index for symmetric relationships
        reverse_key = (source, relationship)
        if reverse_key in self._reverse:
            return self._reverse[reverse_key]
        
        return None
    
    def has(self, source: str, relationship: str) -> bool:
        """Check if a path exists."""
        return (source, relationship) in self._paths or (source, relationship) in self._reverse
    
    def list_relationships(self) -> List[str]:
        """List all relationship types in the store."""
        return list(self._by_relationship.keys())
    
    def list_sources(self, relationship: str) -> List[str]:
        """List all sources for a relationship."""
        return self._by_relationship.get(relationship, [])
    
    def count(self, relationship: Optional[str] = None) -> int:
        """Count paths, optionally filtered by relationship."""
        if relationship:
            return len(self._by_relationship.get(relationship, []))
        return len(self._paths)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the store to disk."""
        save_path = Path(path) if path else self.storage_path
        if not save_path:
            raise ValueError("No storage path specified")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'paths': [p.to_dict() for p in self._paths.values()],
            'version': '1.0',
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Optional[str] = None) -> None:
        """Load the store from disk."""
        load_path = Path(path) if path else self.storage_path
        if not load_path or not load_path.exists():
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        for path_dict in data.get('paths', []):
            path = RelationshipPath.from_dict(path_dict)
            self.add(path)
    
    def stats(self) -> dict:
        """Get statistics about the store."""
        return {
            'total_paths': len(self._paths),
            'relationships': len(self._by_relationship),
            'by_relationship': {
                rel: len(sources) 
                for rel, sources in self._by_relationship.items()
            },
        }
