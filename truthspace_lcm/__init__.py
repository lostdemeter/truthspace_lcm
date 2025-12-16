"""
TruthSpace LCM - Language-Code Model

A natural language to code system that uses φ-based geometric knowledge encoding
to translate human requests into executable Python and Bash code.

Minimal architecture with maximum knowledge-space usage.

Example:
    from truthspace_lcm import TruthSpace, Resolver
    
    ts = TruthSpace()
    resolver = Resolver(ts)
    result = resolver.resolve("list files in directory")
    print(result.output)  # ls -la
"""

__version__ = "0.2.0"
__author__ = "TruthSpace Team"

from truthspace_lcm.core.truthspace import (
    TruthSpace,
    KnowledgeEntry,
    KnowledgeDomain,
    EntryType,
    KnowledgeGapError,
)
from truthspace_lcm.core.resolver import (
    Resolver,
    Resolution,
    OutputType,
)
from truthspace_lcm.core.ingestor import (
    Ingestor,
    SourceType,
    IngestionError,
)
from truthspace_lcm.core.encoder import (
    PlasticEncoder,
    SemanticDecomposition,
    RHO,
    ENCODING_DIM,
    BOOTSTRAP_PRIMITIVES_12D,
    PrimitiveType,
)
from truthspace_lcm.core.executor import (
    CodeExecutor,
    ExecutionResult,
    ExecutionStatus,
)
from truthspace_lcm.core.clock import (
    ClockOracle,
    CLOCK_RATIOS_12D,
    CLOCK_RATIOS_6D,
    phase_at,
    phase_vector,
    phase_similarity,
)
from truthspace_lcm.core.autotuner import (
    DimensionAwareAutotuner,
    DimensionAnalysis,
    CollisionReport,
    PlacementRecommendation,
    TestCase,
)

__all__ = [
    # TruthSpace
    "TruthSpace",
    "KnowledgeEntry",
    "KnowledgeDomain",
    "EntryType",
    "KnowledgeGapError",
    # Resolver
    "Resolver",
    "Resolution",
    "OutputType",
    # Ingestor
    "Ingestor",
    "SourceType",
    "IngestionError",
    # Encoder
    "PhiEncoder",
    "SemanticDecomposition",
    # Executor
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    # Clock
    "ClockOracle",
    "CLOCK_RATIOS_12D",
    "CLOCK_RATIOS_6D",
    "phase_at",
    "phase_vector",
    "phase_similarity",
    # Encoder (Plastic-primary 12D)
    "PlasticEncoder",
    "SemanticDecomposition",
    "RHO",
    "ENCODING_DIM",
    "BOOTSTRAP_PRIMITIVES_12D",
    "PrimitiveType",
    # Autotuner (Dimension-aware)
    "DimensionAwareAutotuner",
    "DimensionAnalysis",
    "CollisionReport",
    "PlacementRecommendation",
    "TestCase",
]
