"""
Gear Chain Core

The foundational classes for building gear-based transformation pipelines.
These are domain-agnostic and can be used for any transformation task.

Structure:
- gear.py: Base classes (Gear, GearState, GearChain, Quaternion)
- protocol.py: Message protocol (GearProtocol, GearMessage, MessageIntent)
- gears/: Individual gear implementations
- chains/: Chain implementations (ConversationalChain, etc.)
- orchestrators/: Multi-gear orchestration (CodeOrchestrator, GearOrchestrator)
- classifiers/: Intent classification
- utils/: Utility classes (holographic space, templates, etc.)

Usage:
    from truthspace_lcm.core import Gear, GearState, GearChain, Quaternion
    from truthspace_lcm.core import ConversationalChain
    
    # Traditional gear chain
    class MyGear(Gear):
        def forward(self, state: GearState) -> GearState:
            return state
    
    chain = GearChain("MyPipeline")
    chain.add(MyGear())
    result = chain.process(initial_state)
"""

# Base classes
from .gear import Gear, GearState, GearChain, Quaternion

# Protocol
from .protocol import (
    GearMessage, GearProtocol, MessageIntent, MessageAwareGear,
    EmergentIntentSpace, get_intent_space,
    normalize_input, normalize_output,
    adapt_to_gear_state, adapt_from_gear_state,
)

# Chains
from .chains.base_chain import EmergentDimensionChain, DimensionInfo, DataItem
from .chains.semantic_chain import SemanticChain
from .chains.linguistic_chain import LinguisticChain
from .chains.conversational_chain import ConversationalChain, KnowledgeItem, ConversationTurn

# Utils
from .utils.folding_deficiency import (
    FoldingStructure, FoldingDeficiencyDetector,
    ShapeDeficiency, ShapeDeficiencyType,
)
from .utils.gear_improvement_loop import (
    GearImprovementLoop, GearTestHarness, ShapeBasedTestHarness,
    DeficiencyType, Deficiency, TestCase, TestResult,
)

# Gears
from .gears.chat_improvement_gear import (
    ChatImprovementGear, ResponseTemplate, ImprovementResult,
)
from .gears.corpus_builder_gear import (
    SelfBuildingCorpusGear, CorpusItem, CorpusCategory,
)

__all__ = [
    # Base classes
    'Gear',
    'GearState',
    'GearChain',
    'Quaternion',
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
    # Chat improvement
    'ChatImprovementGear',
    'ResponseTemplate',
    'ImprovementResult',
    # Self-building corpus
    'SelfBuildingCorpusGear',
    'CorpusItem',
    'CorpusCategory',
]
