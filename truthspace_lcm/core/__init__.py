"""
TruthSpace LCM Core Module

Hypergeometric knowledge resolution using φ-MAX encoding and stacked
geometric embeddings.

Primary Components:
- StackedLCM: 128D hierarchical embedding system (primary driver)
- TruthSpace: Original 12D geometric knowledge storage

Architecture:
- StackedLCM uses 7 layers to encode text at multiple scales
- No training required - structure emerges from geometric operations
- Interpretable - each dimension has semantic meaning

Usage:
    from truthspace_lcm.core import StackedLCM
    
    lcm = StackedLCM()
    lcm.ingest("chop onions", "cooking food preparation")
    content, similarity, cluster = lcm.resolve("cut vegetables")
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

from truthspace_lcm.core.stacked_lcm import (
    StackedLCM,
    MorphologicalLayer,
    LexicalLayer,
    SyntacticLayer,
    CompositionalLayer,
    DisambiguationLayer,
    ContextualLayer,
    GlobalLayer,
)

__all__ = [
    # Primary - Stacked LCM
    "StackedLCM",
    "MorphologicalLayer",
    "LexicalLayer", 
    "SyntacticLayer",
    "CompositionalLayer",
    "DisambiguationLayer",
    "ContextualLayer",
    "GlobalLayer",
    # Legacy - TruthSpace
    "TruthSpace",
    "KnowledgeEntry",
    "KnowledgeGapError",
    "Primitive",
    "PRIMITIVES",
    "PHI",
    "DIM",
    "PHI_BLOCK_WEIGHTS",
]
