"""
TruthSpace LCM Core Module

Fully Geometric Language Understanding with Holographic Templates and Semantic Quaternions.

Core Principle: All semantic operations are geometric operations in concept space.

Architecture:
    Surface Text (any language)
            ↓
    Position-Based Frame Extraction
            ↓
    GEOMETRIC FRAME (position bands)
    {INITIATOR: [0, 0.33), MEDIATOR: [0.33, 0.66), RECEIVER: [0.66, 1]}
            ↓
    φ-Space Representation (language-agnostic)
            ↓
    Holographic Template Projection + Semantic Quaternion
            ↓
    φ-Dial Styled Response

Primary Components:
- GeometricKnowledge: Position-based frame extraction, geometric stop words
- GeometricMorphology: Verb equivalence learned from parallel structures
- GeometricConjugation: Output generation learned from parallel structures
- GeometricQA: Question answering using geometric principles
- HolographicGeometricQA: Enhanced QA with holographic templates + quaternions
- HolographicTemplateProjector: Dynamic templates via interference
- SemanticQuaternionNavigator: 4D concept encoding for analogies (100% accuracy)

Two Quaternions:
- φ-Dial (OUTPUT): Style, Perspective, Depth, Certainty
- Semantic (ENCODING): Gender, Age, Agency (φ-direction), Animacy

Core Formulas (Geometric):
- Position: p(w) = normalized position in sentence [0, 1]
- φ-direction: (initiator_count - receiver_count) / total_roles
- Stop word: no semantic role OR short+frequent
- Morphology: learned from parallel structures ("I love. I loved.")
- Phase: φ-direction × π (geometric encoding, not hash)
- Magnitude: role_strength (how strongly typed)

Usage:
    from truthspace_lcm.core import HolographicGeometricQA
    
    # Create enhanced QA system
    qa = HolographicGeometricQA()
    qa.load_corpus('concept_corpus.json')
    
    # Ask questions (uses holographic templates)
    answer = qa.ask("Who is Darcy?")
    
    # Complete analogies (uses semantic quaternions)
    results = qa.complete_analogy("king", "queen", "man")  # -> woman
"""

# =============================================================================
# PRIMARY GEOMETRIC COMPONENTS
# =============================================================================

from .geometric import (
    PHI,
    MORPHOLOGY_BOOTSTRAP,
    GeometricConcept,
    Frame,
    VerbCluster,
    GeometricMorphology,
    GeometricConjugation,
    GeometricKnowledge,
    GeometricQA,
    HolographicGeometricQA,
)

from .holographic_templates import (
    HolographicTemplateProjector,
    HolographicResponseSynthesizer,
    HolographicConceptNavigator,
    HolographicSummarizer,
    HolographicParaphraser,
    ProjectedTemplate,
    QAPair,
)

from .semantic_quaternion import (
    SemanticQuaternion,
    SemanticQuaternionNavigator,
    SemanticFeatureLearner,
)

# =============================================================================
# SUPPORTING COMPONENTS
# =============================================================================

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

from .code_generator import (
    CodeGenerator,
    CodeFrame,
)

from .planner import (
    Planner,
    PlanStep,
    ExecutionPlan,
    Sandbox,
)

__all__ = [
    # Primary Geometric Components
    "PHI",
    "MORPHOLOGY_BOOTSTRAP",
    "GeometricConcept",
    "Frame",
    "VerbCluster",
    "GeometricMorphology",
    "GeometricConjugation",
    "GeometricKnowledge",
    "GeometricQA",
    "HolographicGeometricQA",
    
    # Holographic Templates
    "HolographicTemplateProjector",
    "HolographicResponseSynthesizer",
    "HolographicConceptNavigator",
    "HolographicSummarizer",
    "HolographicParaphraser",
    "ProjectedTemplate",
    "QAPair",
    
    # Semantic Quaternions
    "SemanticQuaternion",
    "SemanticQuaternionNavigator",
    "SemanticFeatureLearner",
    
    # Conversation Memory
    "ConversationMemory",
    "ConversationTurn",
    
    # Reasoning Engine
    "ReasoningEngine",
    "ReasoningStep",
    "ReasoningPath",
    
    # Holographic Generator
    "HolographicGenerator",
    "InterferencePattern",
    
    # Code Generator
    "CodeGenerator",
    "CodeFrame",
    
    # Planner
    "Planner",
    "PlanStep",
    "ExecutionPlan",
    "Sandbox",
]
