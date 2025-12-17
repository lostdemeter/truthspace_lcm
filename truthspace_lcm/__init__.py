"""
TruthSpace LCM - Hypergeometric Language-Code Model

A natural language to code system that uses φ-MAX geometric encoding
to translate human requests into executable bash commands.

No training. No keywords. Pure hypergeometry.

Example:
    from truthspace_lcm import TruthSpace
    
    ts = TruthSpace()
    output, entry, similarity = ts.resolve("list files in directory")
    print(output)  # ls
"""

__version__ = "0.3.0"
__author__ = "TruthSpace Team"

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
