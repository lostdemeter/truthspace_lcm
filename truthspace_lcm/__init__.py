"""
TruthSpace LCM - Geometric Language Concept Model

A conversational AI system using fully geometric language understanding.
All semantic operations are geometric operations in concept space.

No training. No neural networks. Pure geometry.

Architecture:
    Surface Text (any language)
            ↓
    Position-Based Frame Extraction
            ↓
    Holographic Template Projection + Semantic Quaternion
            ↓
    φ-Dial Styled Response

Example:
    from truthspace_lcm.core import HolographicGeometricQA
    
    qa = HolographicGeometricQA()
    qa.load_corpus('concept_corpus.json')
    
    answer = qa.ask("Who is Darcy?")
    # "Darcy is a character from Pride and Prejudice..."
    
    # Complete analogies with 100% accuracy
    results = qa.complete_analogy("king", "queen", "man")  # -> woman
"""

__version__ = "1.0.0"
__author__ = "TruthSpace Team"

from truthspace_lcm.core import (
    # Primary Geometric Components
    PHI,
    GeometricConcept,
    Frame,
    GeometricMorphology,
    GeometricConjugation,
    GeometricKnowledge,
    GeometricQA,
    HolographicGeometricQA,
    
    # Holographic Templates
    HolographicTemplateProjector,
    HolographicResponseSynthesizer,
    HolographicConceptNavigator,
    
    # Semantic Quaternions
    SemanticQuaternion,
    SemanticQuaternionNavigator,
    
    # Supporting Components
    ConversationMemory,
    ReasoningEngine,
    CodeGenerator,
    Planner,
)

__all__ = [
    # Primary Geometric Components
    "PHI",
    "GeometricConcept",
    "Frame",
    "GeometricMorphology",
    "GeometricConjugation",
    "GeometricKnowledge",
    "GeometricQA",
    "HolographicGeometricQA",
    
    # Holographic Templates
    "HolographicTemplateProjector",
    "HolographicResponseSynthesizer",
    "HolographicConceptNavigator",
    
    # Semantic Quaternions
    "SemanticQuaternion",
    "SemanticQuaternionNavigator",
    
    # Supporting Components
    "ConversationMemory",
    "ReasoningEngine",
    "CodeGenerator",
    "Planner",
]
