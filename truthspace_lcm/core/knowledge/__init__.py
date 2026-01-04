"""
Knowledge Module - Geometric Knowledge Persistence

This module provides persistent storage for geometric knowledge in TruthSpace LCM.
The core insight: geometry IS the knowledge. We persist geometry directly, not text.

Key Components:
- Concept: The atomic unit of knowledge (position + words)
- GeometricKnowledgeStore: The main store for concepts
- CRITICAL_LINE: The information horizon (σ = 0.5)

Design Principles (from Design 091 - Position Is Everything):
1. POSITION IS IDENTITY - a concept IS its position
2. MOVEMENT IS LEARNING - success/failure moves concepts
3. THE CRITICAL LINE IS THE HORIZON - concepts past 0.5 persist

Author: Lesley Gushurst
License: GPLv3
"""

from .concept import Concept, CRITICAL_LINE
from .geometric_store import GeometricKnowledgeStore

__all__ = [
    'Concept',
    'CRITICAL_LINE',
    'GeometricKnowledgeStore',
]
