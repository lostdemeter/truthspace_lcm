"""
Unified Self-Assembly System

This package consolidates the experiments that demonstrate the Universal
Dimension Principle: ANY transformation at ANY scale can be a dimension.

Components:
- core: Base classes for unified corpus and dimensions
- scales: Multi-scale dimension architecture
- stylization: Character-level stylization transforms
- patterns: Speech patterns and discourse dimensions
- loop: The unified self-assembly loop

The key insight: Content, patterns, stylization, and scale are all just
dimensions in the same φ-based geometry. The self-assembly mechanism
works identically for all of them.

Author: TruthSpace LCM Project
License: GPLv3
"""

from experiments.unified_assembly.core import (
    UnifiedCorpus,
    DimensionType,
    ScaledDimension,
    Scale,
)

from experiments.unified_assembly.loop import (
    UnifiedSelfAssemblyLoop,
    UnifiedAssemblyState,
)

__all__ = [
    'UnifiedCorpus',
    'DimensionType',
    'ScaledDimension',
    'Scale',
    'UnifiedSelfAssemblyLoop',
    'UnifiedAssemblyState',
]
