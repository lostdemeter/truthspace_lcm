"""
TruthSpace LCM Core Module

Minimal architecture with maximum knowledge-space usage.

Components:
- TruthSpace: Unified knowledge storage and query interface
- Resolver: Thin NL → Knowledge → Output resolver
- Ingestor: Knowledge acquisition from various sources
- PhiEncoder: φ-based semantic encoding
- CodeExecutor: Safe code execution
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
    PhiEncoder,
    SemanticDecomposition,
    Primitive,
    PrimitiveType,
)
from truthspace_lcm.core.executor import (
    CodeExecutor,
    ExecutionResult,
    ExecutionStatus,
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
    "PhiEncoder",
    "SemanticDecomposition",
    "Primitive",
    "PrimitiveType",
    # Executor
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
]
