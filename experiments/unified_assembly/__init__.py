"""
Unified Self-Assembly System

This package consolidates the experiments that demonstrate the Universal
Dimension Principle: ANY transformation at ANY scale can be a dimension.

Components:
- core: Base classes for unified corpus and dimensions (text-focused)
- scales: Multi-scale dimension architecture
- stylization: Character-level stylization transforms
- loop: The unified self-assembly loop
- bidirectional: ENCODE = DECODE traversal in both directions
- modality: Modality-agnostic core (works for ANY modality)
- image_adapter: Image-specific adapter with transforms

The key insight: Content, patterns, stylization, and scale are all just
dimensions in the same φ-based geometry. The self-assembly mechanism
works identically for all of them AND for all modalities.

ENCODE = DECODE:
- Forward: dimensions → output (generation)
- Reverse: output → dimensions (analysis)

MODALITY AGNOSTIC:
- Text: king → queen, formal → casual
- Image: color → grayscale, sharp → blurred
- Audio: (future) loud → quiet, fast → slow
- The φ-geometry is the same for all modalities

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

from experiments.unified_assembly.bidirectional import (
    TextAnalyzer,
    DimensionalAnalysis,
    BidirectionalTraversal,
)

from experiments.unified_assembly.modality import (
    Modality,
    Artifact,
    Transform,
    UniversalDimension,
    UniversalCorpus,
    ModalityAdapter,
)

from experiments.unified_assembly.image_adapter import (
    ImageScale,
    ImageAdapter,
    ImageCorpus,
    IMAGE_TRANSFORMS,
)

__all__ = [
    # Text-focused (original)
    'UnifiedCorpus',
    'DimensionType',
    'ScaledDimension',
    'Scale',
    'UnifiedSelfAssemblyLoop',
    'UnifiedAssemblyState',
    'TextAnalyzer',
    'DimensionalAnalysis',
    'BidirectionalTraversal',
    # Modality-agnostic
    'Modality',
    'Artifact',
    'Transform',
    'UniversalDimension',
    'UniversalCorpus',
    'ModalityAdapter',
    # Image-specific
    'ImageScale',
    'ImageAdapter',
    'ImageCorpus',
    'IMAGE_TRANSFORMS',
]
