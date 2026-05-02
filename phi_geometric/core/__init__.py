"""
Core components of the φ-Geometric Framework.

Proven engine (no torch required):
    - discovery: Information-gain based rule discovery
    - cascade_navigator: Ordered phase pipeline execution
    - phase_discovery: Automatic phase structure discovery

Experimental (requires torch):
    - encoder, patterns, projector, navigator, memory,
      injector, filter, geometric_ai, knowledge_base
"""

# =========================================================================
# PROVEN ENGINE — always available, no torch dependency
# =========================================================================

from .discovery import (
    StructureDiscovery, TransformRule, DiscoveryResult,
    Observation, discover_gears, discover_selector,
    detect_inconsistencies
)
from .cascade_navigator import (
    CascadeNavigator, Phase, CascadeTrace, ElementTrace, PhaseTrace,
    geometric_context_extractor,
)
from .phase_discovery import (
    PhaseDiscovery, PhaseDiscoveryResult, PhaseCandidate,
    MultiTokenPattern, ExpandPattern,
)

__all__ = [
    # Discovery
    "StructureDiscovery", "TransformRule", "DiscoveryResult",
    "Observation", "discover_gears", "discover_selector",
    "detect_inconsistencies",
    # Navigator
    "CascadeNavigator", "Phase", "CascadeTrace", "ElementTrace", "PhaseTrace",
    "geometric_context_extractor",
    # Phase Discovery
    "PhaseDiscovery", "PhaseDiscoveryResult", "PhaseCandidate",
    "MultiTokenPattern", "ExpandPattern",
]

# =========================================================================
# EXPERIMENTAL — lazy-loaded, requires torch
# =========================================================================

def __getattr__(name):
    """Lazy-load torch-dependent modules on first access."""
    _experimental = {
        "PhiEncoder", "PHI", "LN_PHI",
        "Pattern", "Topology", "SelfReference",
        "Funnel", "Spiral", "Web", "Tree", "Braid", "Hourglass", "Ring",
        "Constellation", "Fractal", "Mirror", "Cascade", "compose",
        "ShapeProjector", "ProblemSpec", "IOSpec", "DataType",
        "Navigator", "SignatureMemory", "KnowledgeInjector",
        "BottleneckFilter", "GeometricAI",
        "KnowledgeBase", "KnowledgeAtom", "KnowledgeMolecule",
        "KnowledgeReaction", "RelationType", "ReactionTrigger",
        "AtomProperty", "create_color_knowledge_base",
    }
    if name in _experimental:
        # Import on demand
        if name in ("PhiEncoder", "PHI", "LN_PHI"):
            from .encoder import PhiEncoder, PHI, LN_PHI
            return locals()[name]
        elif name in ("Pattern", "Topology", "SelfReference", "Funnel",
                      "Spiral", "Web", "Tree", "Braid", "Hourglass",
                      "Ring", "Constellation", "Fractal", "Mirror",
                      "Cascade", "compose"):
            import importlib
            mod = importlib.import_module(".patterns", __package__)
            return getattr(mod, name)
        elif name in ("ShapeProjector", "ProblemSpec", "IOSpec", "DataType"):
            import importlib
            mod = importlib.import_module(".projector", __package__)
            return getattr(mod, name)
        elif name == "Navigator":
            from .navigator import Navigator
            return Navigator
        elif name == "SignatureMemory":
            from .memory import SignatureMemory
            return SignatureMemory
        elif name == "KnowledgeInjector":
            from .injector import KnowledgeInjector
            return KnowledgeInjector
        elif name == "BottleneckFilter":
            from .filter import BottleneckFilter
            return BottleneckFilter
        elif name == "GeometricAI":
            from .geometric_ai import GeometricAI
            return GeometricAI
        else:
            import importlib
            mod = importlib.import_module(".knowledge_base", __package__)
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
