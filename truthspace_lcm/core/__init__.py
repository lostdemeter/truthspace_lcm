"""
TruthSpace LCM Core Module

Concept Language approach to knowledge extraction and Q&A.

Core Principle: All semantic operations are geometric operations in concept space.

Architecture:
    Surface Text (any language)
            ↓
    Language-Specific Parser
            ↓
    CONCEPT FRAME (order-free)
    {AGENT: X, ACTION: Y, PATIENT: Z, LOCATION: W}
            ↓
    Vector Representation (language-agnostic)
            ↓
    Storage / Query / Holographic Projection

Primary Components:
- Vocabulary: Hash-based word positions with IDF weighting
- ConceptLanguage: Order-free semantic frames with universal primitives
- ConceptKnowledge: Language-agnostic knowledge storage and query
- HolographicProjector: Resolves queries by filling the "gap" in questions

Core Formulas:
- Word Position: pos(w) = hash(w) → ℝ^dim (deterministic)
- Frame Vector: vec(frame) = Σ hash(ROLE:value) (order-independent)
- Similarity: cos(θ) = (a·b) / (‖a‖·‖b‖)

Holographic Principle:
    Question = Content - Gap    (has missing information)
    Answer   = Content + Fill   (provides missing information)

Usage:
    from truthspace_lcm.core import ConceptQA
    
    # Load knowledge corpus
    qa = ConceptQA()
    qa.load_corpus('concept_corpus.json')
    
    # Ask questions
    answer = qa.ask("Who is Darcy?")
    # "Darcy is a character from Pride and Prejudice..."
"""

from .vocabulary import (
    Vocabulary,
    tokenize,
    word_position,
    cosine_similarity,
    euclidean_distance,
    DEFAULT_DIM,
)

from .concept_language import (
    ConceptFrame,
    ConceptExtractor,
    ConceptStore,
    ACTION_PRIMITIVES,
    SEMANTIC_ROLES,
)

from .concept_knowledge import (
    ConceptKnowledge,
    HolographicProjector,
    ConceptQA,
    QUESTION_AXES,
)

from .answer_patterns import (
    reverse_tune,
    DIAL_SIGNATURES,
)

from .learnable_structure import (
    LearnableStructure,
    EntityProfile,
    train_from_examples,
    ROLE_VOCABULARY,
    QUALITY_VOCABULARY,
    ACTION_VOCABULARY,
)

from .conversation_memory import (
    ConversationMemory,
    ConversationTurn,
)

from .reasoning_engine import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningPath,
)

from .holographic_generator import (
    HolographicGenerator,
    InterferencePattern,
)

__all__ = [
    # Vocabulary (foundation)
    "Vocabulary",
    "tokenize",
    "word_position",
    "cosine_similarity",
    "euclidean_distance",
    "DEFAULT_DIM",
    # Concept Language
    "ConceptFrame",
    "ConceptExtractor",
    "ConceptStore",
    "ACTION_PRIMITIVES",
    "SEMANTIC_ROLES",
    # Concept Knowledge & Holographic Q&A
    "ConceptKnowledge",
    "HolographicProjector",
    "ConceptQA",
    "QUESTION_AXES",
    # Reverse Tuning (Phase Conjugation)
    "reverse_tune",
    "DIAL_SIGNATURES",
    # Learnable Structure (Gradient-Free Learning)
    "LearnableStructure",
    "EntityProfile",
    "train_from_examples",
    "ROLE_VOCABULARY",
    "QUALITY_VOCABULARY",
    "ACTION_VOCABULARY",
    # Conversation Memory (Multi-Turn Dialogue)
    "ConversationMemory",
    "ConversationTurn",
    # Reasoning Engine (Multi-Hop)
    "ReasoningEngine",
    "ReasoningStep",
    "ReasoningPath",
    # Holographic Generator
    "HolographicGenerator",
    "InterferencePattern",
]
