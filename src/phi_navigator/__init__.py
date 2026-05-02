"""
φ-Navigator: Semantic Navigation via φ-Lattice Coordinates
===========================================================

Three ways of representing the same thing:

1. REPRESENTATION (φ-coordinates) - Universal encoding
   - Any value → (sign, level) in φ-space
   - Lossless, 99.9988% correlation

2. PATHS (relationships) - Concept-specific transformations
   - Each relationship is a path through φ-space
   - hot→cold is different from tall→short
   - Paths are stored, not computed

3. LOOKUP (navigation) - O(1) access
   - Given (concept, relationship) → answer
   - Instant, 100% accurate

The φ-lattice is the coordinate system.
Relationships are paths in that system.
Navigation is following the paths.
"""

from .coordinates import PhiCoordinates
from .paths import RelationshipPath, PathStore
from .navigator import PhiNavigator
from .relationships import Relationship, OppositeRelationship, GenderRelationship

__all__ = [
    'PhiCoordinates',
    'RelationshipPath',
    'PathStore', 
    'PhiNavigator',
    'Relationship',
    'OppositeRelationship',
    'GenderRelationship',
]
