"""
Knowledge Module - Geometric Knowledge Persistence

This module provides persistent storage for geometric knowledge in TruthSpace LCM.
The core insight: geometry IS the knowledge. We persist geometry directly, not text.

Key Components:
- Concept: The atomic unit of knowledge (position + words + metadata)
- GeometricKnowledgeStore: The main store for concepts

Design Principles (from Design 088 and 089):
1. Persist geometry directly (similarity matrices, positions)
2. Hierarchical granularity (facts → clusters → topics)
3. Two-tier persistence (temporary → permanent via promotion)
4. ENCODE = DECODE: The space is conformally symmetric

Author: Lesley Gushurst
License: GPLv3
"""

from .concept import Concept, ConceptLevel
from .geometric_store import GeometricKnowledgeStore

__all__ = [
    'Concept',
    'ConceptLevel', 
    'GeometricKnowledgeStore',
]
