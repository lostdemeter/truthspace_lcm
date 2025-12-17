"""
TruthSpace LCM Core Module

Hypergeometric knowledge resolution using φ-MAX encoding.

This is a showcase of how pure geometry can replace trained LLM/LCM
functionality with mathematically-derived semantic resolution.

Components:
- TruthSpace: Geometric knowledge storage and resolution
- Primitive: Semantic anchors in 12D truth space
- KnowledgeEntry: Knowledge with geometric position

Key concepts:
- φ-MAX encoding: φ^level with MAX per dimension
- Sierpinski property: Overlapping activations don't stack
- φ-weighted distance: Actions > Domains > Relations
"""

from truthspace_lcm.core.truthspace import (
    TruthSpace,
    KnowledgeEntry,
    KnowledgeGapError,
    Primitive,
    PRIMITIVES,
    PHI,
    DIM,
    PHI_BLOCK_WEIGHTS,
)

__all__ = [
    "TruthSpace",
    "KnowledgeEntry",
    "KnowledgeGapError",
    "Primitive",
    "PRIMITIVES",
    "PHI",
    "DIM",
    "PHI_BLOCK_WEIGHTS",
]
