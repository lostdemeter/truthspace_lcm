"""
φ-Geometric Transformation Engine

Automatic discovery and execution of sequence transformation pipelines.
Feed example (input, output) pairs — get an executable, interpretable,
deterministic transformation pipeline. No training, no GPU, no torch.

Quick Start:
    from phi_geometric import PhaseDiscovery

    pd = PhaseDiscovery()
    pd.add_pair(['s', 'h', 'i', 'p'], ['ʃ', 'ɪ', 'p'])
    pd.add_pair(['c', 'a', 't'],      ['k', 'æ', 't'])

    result = pd.discover()
    nav = result.to_navigator()

    trace = nav.execute(['s', 'h', 'o', 'p'])
    print(trace.output_elements)  # ['ʃ', 'ɒ', 'p']

Features:
    - Inconsistency-driven phase discovery (8 proven archetypes)
    - Collapse (N→M), expand (1→N), context-dependent, and map phases
    - Geometric context windows (φ-decay, covers distance 1-12)
    - Full traceability — every rule, every decision, inspectable
    - Zero configuration — just add_pair() and discover()

Author: TruthSpace LCM Project
Date: February 2026
Version: 3.0.0-rc1
"""

__version__ = "3.0.0-rc1"
__author__ = "TruthSpace LCM Project"

# =========================================================================
# PUBLIC API — the proven engine, no torch required
# =========================================================================

from .core.phase_discovery import (
    PhaseDiscovery,
    PhaseDiscoveryResult,
    PhaseCandidate,
    MultiTokenPattern,
    ExpandPattern,
)
from .core.cascade_navigator import (
    CascadeNavigator,
    Phase,
    CascadeTrace,
    ElementTrace,
    PhaseTrace,
    geometric_context_extractor,
)
from .core.discovery import (
    StructureDiscovery,
    TransformRule,
    DiscoveryResult,
    Observation,
)
from .core.serialization import (
    save_pipeline,
    load_pipeline,
)
from .core.generation import ReverseEngine

__all__ = [
    # Primary API
    "PhaseDiscovery",
    "PhaseDiscoveryResult",
    "CascadeNavigator",
    # Supporting classes
    "PhaseCandidate",
    "MultiTokenPattern",
    "ExpandPattern",
    "Phase",
    "CascadeTrace",
    "ElementTrace",
    "PhaseTrace",
    "geometric_context_extractor",
    # Serialization
    "save_pipeline",
    "load_pipeline",
    # Generation
    "ReverseEngine",
    # Low-level discovery
    "StructureDiscovery",
    "TransformRule",
    "DiscoveryResult",
    "Observation",
]
