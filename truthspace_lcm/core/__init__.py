"""
TruthSpace LCM Core Module

Geometric Chat System (GCS) implementation based on the SDS specification.

Core Principle: All semantic operations are geometric operations in vector space.

Primary Components:
- Vocabulary: Hash-based word positions with IDF weighting
- KnowledgeBase: Facts, triples, Q&A pairs with semantic search
- StyleEngine: Style extraction, classification, and transfer

Core Formulas:
- Word Position: pos(w) = hash(w) → ℝ^dim (deterministic)
- Text Encoding: enc(t) = Σᵢ wᵢ·pos(wordᵢ) / Σᵢ wᵢ (IDF-weighted)
- Style Centroid: c = (1/n) Σᵢ enc(exemplarᵢ)
- Style Transfer: styled = (1-α)·content + α·centroid
- Similarity: cos(θ) = (a·b) / (‖a‖·‖b‖)

Usage:
    from truthspace_lcm.core import Vocabulary, KnowledgeBase, StyleEngine
    
    # Create vocabulary and knowledge base
    vocab = Vocabulary(dim=64)
    kb = KnowledgeBase(vocab)
    
    # Ingest knowledge
    kb.ingest_text("Captain Ahab is the captain of the Pequod.")
    
    # Query
    results = kb.search_qa("Who is Captain Ahab?")
    
    # Style operations
    style_engine = StyleEngine(vocab)
    style = style_engine.extract_from_text(hemingway_text, "Hemingway")
    classification = style_engine.classify("The man sat alone.")
"""

from .vocabulary import (
    Vocabulary,
    tokenize,
    word_position,
    cosine_similarity,
    euclidean_distance,
    DEFAULT_DIM,
)

from .knowledge import (
    KnowledgeBase,
    Fact,
    Triple,
    QAPair,
    detect_question_type,
    extract_triples,
    generate_qa_from_triple,
    QUESTION_PATTERNS,
)

from .style import (
    StyleEngine,
    Style,
)

from .binding import (
    bind,
    unbind,
    bundle,
    permute,
    inverse_permute,
    similarity,
    BindingMethod,
    CleanupMemory,
    RelationalStore,
    SequenceEncoder,
)

from .geometric_lcm import (
    GeometricLCM,
    GeoEntity,
    GeoRelation,
    GeoFact,
    FactParser,
)

__all__ = [
    # Vocabulary
    "Vocabulary",
    "tokenize",
    "word_position",
    "cosine_similarity",
    "euclidean_distance",
    "DEFAULT_DIM",
    # Knowledge Base
    "KnowledgeBase",
    "Fact",
    "Triple",
    "QAPair",
    "detect_question_type",
    "extract_triples",
    "generate_qa_from_triple",
    "QUESTION_PATTERNS",
    # Style Engine
    "StyleEngine",
    "Style",
    # Binding (VSA)
    "bind",
    "unbind",
    "bundle",
    "permute",
    "inverse_permute",
    "similarity",
    "BindingMethod",
    "CleanupMemory",
    "RelationalStore",
    "SequenceEncoder",
    # Geometric LCM
    "GeometricLCM",
    "GeoEntity",
    "GeoRelation",
    "GeoFact",
    "FactParser",
]
