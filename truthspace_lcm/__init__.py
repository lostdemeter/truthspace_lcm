"""
TruthSpace LCM - Geometric Language Concept Model

A conversational AI system using holographic concept resolution.
All semantic operations are geometric operations in concept space.

No training. No neural networks. Pure geometry.

Architecture:
    Surface Text (any language)
            ↓
    Concept Frame (order-free, language-agnostic)
            ↓
    Holographic Projection (fill the gap)
            ↓
    Answer

Example:
    from truthspace_lcm.core import ConceptQA
    
    qa = ConceptQA()
    qa.load_corpus('concept_corpus.json')
    
    answer = qa.ask("Who is Darcy?")
    # "Darcy is a character from Pride and Prejudice..."
"""

__version__ = "0.5.0"
__author__ = "TruthSpace Team"

from truthspace_lcm.core import (
    Vocabulary,
    tokenize,
    word_position,
    cosine_similarity,
    ConceptFrame,
    ConceptExtractor,
    ConceptStore,
    ConceptKnowledge,
    HolographicProjector,
    ConceptQA,
    ACTION_PRIMITIVES,
    SEMANTIC_ROLES,
    QUESTION_AXES,
)

__all__ = [
    # Vocabulary
    "Vocabulary",
    "tokenize",
    "word_position",
    "cosine_similarity",
    # Concept Language
    "ConceptFrame",
    "ConceptExtractor",
    "ConceptStore",
    "ACTION_PRIMITIVES",
    "SEMANTIC_ROLES",
    # Concept Knowledge & Q&A
    "ConceptKnowledge",
    "HolographicProjector",
    "ConceptQA",
    "QUESTION_AXES",
]
