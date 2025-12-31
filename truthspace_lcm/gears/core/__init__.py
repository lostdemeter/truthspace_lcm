"""
Gear Chain Core

The foundational classes for building gear-based transformation pipelines.
These are domain-agnostic and can be used for any transformation task.

Core Components:
- Quaternion: 4D rotation encoding for gear parameters
- GearState: Base state object that flows through the chain
- Gear: Abstract base class for all transformation gears
- GearChain: Container for composing gears into pipelines

Emergent Components:
- EmergentDimensionChain: Base class for self-discovering dimension chains
- SemanticChain: Understanding chain (discovers dimensions from agent behavior)
- LinguisticChain: Output chain (discovers dimensions from sentence structure)

Usage:
    from truthspace_lcm.gears.core import Gear, GearState, GearChain, Quaternion
    from truthspace_lcm.gears.core import SemanticChain, LinguisticChain
    
    # Traditional gear chain
    class MyGear(Gear):
        def forward(self, state: GearState) -> GearState:
            return state
    
    chain = GearChain("MyPipeline")
    chain.add(MyGear())
    result = chain.process(initial_state)
    
    # Emergent dimension chain
    semantic = SemanticChain()
    semantic.ingest_corpus("corpus.json")
    semantic.learn_dimensions()
    similar = semantic.find_similar("holmes")
"""

from .base import Gear, GearState, GearChain, Quaternion
from .error_correction import ErrorCorrectionGear
from .emergent_chain import EmergentDimensionChain, DimensionInfo, DataItem
from .semantic_chain import SemanticChain
from .linguistic_chain import LinguisticChain
from .conversational_chain import ConversationalChain, KnowledgeItem, ConversationTurn
from .gear_message import (
    GearMessage, GearProtocol, MessageIntent, MessageAwareGear,
    EmergentIntentSpace, get_intent_space,
    normalize_input, normalize_output,
    adapt_to_gear_state, adapt_from_gear_state,
)
from .folding_deficiency import (
    FoldingStructure, FoldingDeficiencyDetector,
    ShapeDeficiency, ShapeDeficiencyType,
)
from .gear_improvement_loop import (
    GearImprovementLoop, GearTestHarness, ShapeBasedTestHarness,
    DeficiencyType, Deficiency, TestCase, TestResult,
)

__all__ = [
    # Base classes
    'Gear',
    'GearState',
    'GearChain',
    'Quaternion',
    'ErrorCorrectionGear',
    # Gear message protocol
    'GearMessage',
    'GearProtocol',
    'MessageAwareGear',  # Alias for GearProtocol
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
    # Folding deficiency detection (shape-based)
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
]
