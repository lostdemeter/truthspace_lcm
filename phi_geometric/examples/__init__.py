"""
Pattern Examples

This module contains example implementations of each pattern
from the φ-Geometric Pattern Taxonomy.

Patterns:
    - Funnel: Convergent (many → one) - Classification, regression
    - Spiral: Self-referential - Language modeling
    - Web: Cross-connected - Colorization, segmentation
    - Tree: Divergent (one → many) - Multi-task learning
    - Braid: Intertwined - Multi-modal fusion
    - Hourglass: Compress/expand - Autoencoders, generation

Each example demonstrates:
    1. Problem specification
    2. Pattern selection
    3. Weight projection
    4. Knowledge injection
    5. Inference without training

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

from .funnel_example import FunnelClassifier
from .spiral_example import SpiralLanguageModel
from .web_example import WebColorizer
from .tree_example import TreeMultiTask
from .braid_example import BraidMultiModal
from .hourglass_example import HourglassAutoencoder
from .archetypes import (
    get_archetype, discover_archetype, list_archetypes,
    ARCHETYPES, ARCHETYPE_DESCRIPTIONS,
)

__all__ = [
    "FunnelClassifier",
    "SpiralLanguageModel",
    "WebColorizer",
    "TreeMultiTask",
    "BraidMultiModal",
    "HourglassAutoencoder",
    "get_archetype",
    "discover_archetype",
    "list_archetypes",
    "ARCHETYPES",
    "ARCHETYPE_DESCRIPTIONS",
]
