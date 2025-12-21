"""
TruthSpace LCM - Geometric Language Concept Model

A conversational AI system using pure geometric operations.
All semantic operations are geometric operations in vector space.

No training. No neural networks. Pure geometry.

Example:
    from truthspace_lcm.core import Vocabulary, KnowledgeBase, StyleEngine
    
    vocab = Vocabulary(dim=64)
    kb = KnowledgeBase(vocab)
    kb.add_qa_pair("What is TruthSpace?", "A geometric approach to language.")
    
    results = kb.search_qa("Tell me about TruthSpace")
"""

__version__ = "0.4.0"
__author__ = "TruthSpace Team"

from truthspace_lcm.core import (
    Vocabulary,
    KnowledgeBase,
    StyleEngine,
    Style,
    Fact,
    Triple,
    QAPair,
    cosine_similarity,
    tokenize,
    detect_question_type,
)

__all__ = [
    "Vocabulary",
    "KnowledgeBase",
    "StyleEngine",
    "Style",
    "Fact",
    "Triple",
    "QAPair",
    "cosine_similarity",
    "tokenize",
    "detect_question_type",
]
