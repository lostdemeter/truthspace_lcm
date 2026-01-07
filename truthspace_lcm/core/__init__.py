"""
TruthSpace LCM Core

The core module for TruthSpace LCM, providing both the new HyperMapping-based
architecture and legacy gear-based classes for backwards compatibility.

NEW ARCHITECTURE (HyperMapping-based):
- ChatPipeline: Main chat interface with intent detection, knowledge, and code
- KnowledgeSpace: Geometric knowledge storage (replaces GeometricKnowledgeStore)
- CodeSpace: Code generation via geometric pattern matching
- IntentSpace: Intent detection via bootstrap + geometric matching

LEGACY ARCHITECTURE (Gear-based):
- Located in legacy_gears/ for backwards compatibility
- Gear, GearState, GearChain, Quaternion
- ConversationalChain, ChatGearChain, etc.

Usage (New):
    from truthspace_lcm.core import ChatPipeline, KnowledgeSpace, CodeSpace
    
    pipeline = ChatPipeline()
    pipeline.add_knowledge("Python is a programming language")
    response = pipeline.chat("What is Python?")

Usage (Legacy - for backwards compatibility):
    from truthspace_lcm.core import Gear, GearState, GearChain
    from truthspace_lcm.core import ConversationalChain
"""

# =============================================================================
# NEW HYPERMAPPING-BASED ARCHITECTURE
# =============================================================================

from .chat_pipeline import ChatPipeline, ChatConfig, Intent, IntentResult, IntentSpace
from .knowledge_space import KnowledgeSpace
from .code_space import CodeSpace, CodeResult, CodeVerifier
from .plot_space import PlotSpace, PlotResult, PlotPattern
from .ollama_space import OllamaSpace, KnowledgeResult

# Dynamic Quaternion Layers (Design 104-105)
from .dynamic_dimensions import (
    DynamicDimensionRegistry, 
    PhiZipfWeighting,
    TachyonNavigator,
    DimensionHypothesis,
    BOOTSTRAP_DIMENSIONS,
)
from .quaternion_encoder import (
    QuaternionEncoder,
    QuaternionPosition,
    SemanticDim,
    GrammaticalDim,
    ContextualDim,
)
from .learned_knowledge import LearnedKnowledge, LearnedFact, extract_llm_response

# =============================================================================
# LEGACY GEAR-BASED ARCHITECTURE (for backwards compatibility)
# =============================================================================

# Base classes (legacy)
from .legacy_gears.gear import Gear, GearState, GearChain, Quaternion

# Protocol (legacy)
from .legacy_gears.protocol import (
    GearMessage, GearProtocol, MessageIntent, MessageAwareGear,
    EmergentIntentSpace, get_intent_space,
    normalize_input, normalize_output,
    adapt_to_gear_state, adapt_from_gear_state,
)

# Chains (legacy)
from .legacy_gears.chains.base_chain import EmergentDimensionChain, DimensionInfo, DataItem
from .legacy_gears.chains.semantic_chain import SemanticChain
from .legacy_gears.chains.linguistic_chain import LinguisticChain
from .legacy_gears.chains.conversational_chain import ConversationalChain, KnowledgeItem, ConversationTurn

# Utils
from .utils.folding_deficiency import (
    FoldingStructure, FoldingDeficiencyDetector,
    ShapeDeficiency, ShapeDeficiencyType,
)
from .utils.gear_improvement_loop import (
    GearImprovementLoop, GearTestHarness, ShapeBasedTestHarness,
    DeficiencyType, Deficiency, TestCase, TestResult,
)

# Gears (legacy)
from .legacy_gears.gears.chat_improvement_gear import (
    ChatImprovementGear, ResponseTemplate, ImprovementResult,
)
from .legacy_gears.gears.corpus_builder_gear import (
    SelfBuildingCorpusGear, CorpusItem, CorpusCategory,
)

__all__ = [
    # ==========================================================================
    # NEW HYPERMAPPING-BASED ARCHITECTURE
    # ==========================================================================
    
    # Chat pipeline
    'ChatPipeline',
    'ChatConfig',
    'Intent',
    'IntentResult',
    'IntentSpace',
    
    # Knowledge space
    'KnowledgeSpace',
    
    # Code space
    'CodeSpace',
    'CodeResult',
    'CodeVerifier',
    
    # Plot space
    'PlotSpace',
    'PlotResult',
    'PlotPattern',
    
    # Ollama space
    'OllamaSpace',
    'KnowledgeResult',
    
    # Dynamic Quaternion Layers (Design 104-105)
    'DynamicDimensionRegistry',
    'PhiZipfWeighting',
    'TachyonNavigator',
    'DimensionHypothesis',
    'BOOTSTRAP_DIMENSIONS',
    'QuaternionEncoder',
    'QuaternionPosition',
    'SemanticDim',
    'GrammaticalDim',
    'ContextualDim',
    
    # Learned Knowledge
    'LearnedKnowledge',
    'LearnedFact',
    'extract_llm_response',
    
    # ==========================================================================
    # LEGACY GEAR-BASED ARCHITECTURE (backwards compatibility)
    # ==========================================================================
    
    # Base classes
    'Gear',
    'GearState',
    'GearChain',
    'Quaternion',
    
    # Gear message protocol
    'GearMessage',
    'GearProtocol',
    'MessageAwareGear',
    'MessageIntent',
    'EmergentIntentSpace',
    'get_intent_space',
    'normalize_input',
    'normalize_output',
    'adapt_to_gear_state',
    'adapt_from_gear_state',
    
    # Emergent dimension chains
    'EmergentDimensionChain',
    'DimensionInfo',
    'DataItem',
    'SemanticChain',
    'LinguisticChain',
    
    # Conversational chain
    'ConversationalChain',
    'KnowledgeItem',
    'ConversationTurn',
    
    # Folding deficiency detection
    'FoldingStructure',
    'FoldingDeficiencyDetector',
    'ShapeDeficiency',
    'ShapeDeficiencyType',
    
    # Improvement loop
    'GearImprovementLoop',
    'GearTestHarness',
    'ShapeBasedTestHarness',
    'DeficiencyType',
    'Deficiency',
    'TestCase',
    'TestResult',
    
    # Chat improvement
    'ChatImprovementGear',
    'ResponseTemplate',
    'ImprovementResult',
    
    # Self-building corpus
    'SelfBuildingCorpusGear',
    'CorpusItem',
    'CorpusCategory',
]
