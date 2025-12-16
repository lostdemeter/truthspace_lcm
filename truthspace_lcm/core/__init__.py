"""
TruthSpace LCM Core Module

Minimal architecture with maximum knowledge-space usage.

Components:
- TruthSpace: Unified knowledge storage and query interface
- Resolver: Thin NL → Knowledge → Output resolver
- Ingestor: Knowledge acquisition from various sources
- PlasticEncoder: ρ-based 12D semantic encoding
- CodeExecutor: Safe code execution
- DimensionAwareAutotuner: Dimension-aware knowledge tuning
"""

from truthspace_lcm.core.truthspace import (
    TruthSpace,
    KnowledgeEntry,
    KnowledgeDomain,
    EntryType,
    KnowledgeGapError,
    QueryResult,
)
from truthspace_lcm.core.resolver import (
    Resolver,
    Resolution,
    ExecutionResult as ResolverExecutionResult,
    OutputType,
)
from truthspace_lcm.core.ingestor import (
    Ingestor,
    SourceType,
    ParsedKnowledge,
    IngestionError,
)
from truthspace_lcm.core.encoder import (
    PlasticEncoder,
    SemanticDecomposition,
    Primitive,
    PrimitiveType,
    RHO,
    ENCODING_DIM,
)
from truthspace_lcm.core.executor import (
    CodeExecutor,
    ExecutionResult,
    ExecutionStatus,
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
    "QueryResult",
    # Resolver
    "Resolver",
    "Resolution",
    "OutputType",
    # Ingestor
    "Ingestor",
    "SourceType",
    "ParsedKnowledge",
    "IngestionError",
    # Encoder
    "PlasticEncoder",
    "SemanticDecomposition",
    "Primitive",
    "PrimitiveType",
    "RHO",
    "ENCODING_DIM",
    # Executor
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    # Autotuner
    "DimensionAwareAutotuner",
    "DimensionAnalysis",
    "CollisionReport",
    "PlacementRecommendation",
    "TestCase",
]
