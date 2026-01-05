"""
Legacy Gears Module

This module contains the original gear-based architecture that has been
superseded by the HyperMapping-based approach.

The new architecture uses:
- ChatPipeline instead of ChatGearChain
- KnowledgeSpace instead of GeometricKnowledgeStore
- CodeSpace instead of PythonCodeGear
- IntentSpace instead of IntentDetectorGear
- HyperPipeline instead of GearChain

This legacy code is preserved for backwards compatibility and reference.
New code should use the HyperMapping-based classes in:
- truthspace_lcm.core.chat_pipeline
- truthspace_lcm.core.knowledge_space
- truthspace_lcm.core.code_space

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path

# Add this directory to path so internal imports work
# This allows legacy code to use "from truthspace_lcm.core.legacy_gears.gear import ..."
# by creating module aliases
_legacy_dir = Path(__file__).parent

# Create module aliases for backwards compatibility
# This maps old import paths to new locations
import importlib.util

def _create_module_alias(old_path: str, module):
    """Create a module alias for backwards compatibility."""
    sys.modules[old_path] = module

# Import and alias the core modules
from . import gear as _gear
from . import protocol as _protocol

_create_module_alias('truthspace_lcm.core.gear', _gear)
_create_module_alias('truthspace_lcm.core.protocol', _protocol)

# Import and alias submodules
from .gears import chat_gear_chain as _chat_gear_chain
from .gears import intent_detector_gear as _intent_detector_gear
from .gears import python_code_gear as _python_code_gear
from .gears import corpus_builder_gear as _corpus_builder_gear
from .gears import emergent_classifier_gear as _emergent_classifier_gear
from .gears import emergent_gear as _emergent_gear
from .gears import bootstrap_gear as _bootstrap_gear
from .gears import factory_gear as _factory_gear
from .gears import chat_improvement_gear as _chat_improvement_gear

_create_module_alias('truthspace_lcm.core.gears.chat_gear_chain', _chat_gear_chain)
_create_module_alias('truthspace_lcm.core.gears.intent_detector_gear', _intent_detector_gear)
_create_module_alias('truthspace_lcm.core.gears.python_code_gear', _python_code_gear)
_create_module_alias('truthspace_lcm.core.gears.corpus_builder_gear', _corpus_builder_gear)
_create_module_alias('truthspace_lcm.core.gears.emergent_classifier_gear', _emergent_classifier_gear)
_create_module_alias('truthspace_lcm.core.gears.emergent_gear', _emergent_gear)
_create_module_alias('truthspace_lcm.core.gears.bootstrap_gear', _bootstrap_gear)
_create_module_alias('truthspace_lcm.core.gears.factory_gear', _factory_gear)
_create_module_alias('truthspace_lcm.core.gears.chat_improvement_gear', _chat_improvement_gear)

from .chains import conversational_chain as _conversational_chain
from .chains import semantic_chain as _semantic_chain
from .chains import linguistic_chain as _linguistic_chain
from .chains import base_chain as _base_chain

_create_module_alias('truthspace_lcm.core.chains.conversational_chain', _conversational_chain)
_create_module_alias('truthspace_lcm.core.chains.semantic_chain', _semantic_chain)
_create_module_alias('truthspace_lcm.core.chains.linguistic_chain', _linguistic_chain)
_create_module_alias('truthspace_lcm.core.chains.base_chain', _base_chain)

from .orchestrators import gear_orchestrator as _gear_orchestrator
from .orchestrators import code_orchestrator as _code_orchestrator

_create_module_alias('truthspace_lcm.core.orchestrators.gear_orchestrator', _gear_orchestrator)
_create_module_alias('truthspace_lcm.core.orchestrators.code_orchestrator', _code_orchestrator)

# Re-export legacy classes for backwards compatibility
from .gear import Gear, GearState, GearChain, Quaternion

# Legacy chains
from .chains.conversational_chain import ConversationalChain
from .chains.semantic_chain import SemanticChain
from .chains.linguistic_chain import LinguisticChain
from .chains.base_chain import EmergentDimensionChain

# Legacy gears
from .gears.chat_gear_chain import ChatGearChain, KnowledgeLearningGear
from .gears.intent_detector_gear import IntentDetectorGear
from .gears.python_code_gear import PythonCodeGear, PythonCodeCorpus
from .gears.corpus_builder_gear import SelfBuildingCorpusGear
from .gears.emergent_classifier_gear import EmergentClassifierGear
from .gears.emergent_gear import EmergentGear
from .gears.bootstrap_gear import BootstrapGear
from .gears.factory_gear import GearFactoryGear

# Legacy orchestrators
from .orchestrators.gear_orchestrator import GearOrchestrator
from .orchestrators.code_orchestrator import CodeOrchestrator

# Protocol
from .protocol import GearProtocol, GearMessage, MessageIntent

__all__ = [
    # Core gear classes
    'Gear',
    'GearState', 
    'GearChain',
    'Quaternion',
    
    # Chains
    'ConversationalChain',
    'SemanticChain',
    'LinguisticChain',
    'BaseChain',
    
    # Gears
    'ChatGearChain',
    'KnowledgeLearningGear',
    'IntentDetectorGear',
    'PythonCodeGear',
    'PythonCodeCorpus',
    'CorpusBuilderGear',
    'EmergentClassifierGear',
    'EmergentGear',
    'BootstrapGear',
    'FactoryGear',
    
    # Orchestrators
    'GearOrchestrator',
    'CodeOrchestrator',
    
    # Protocol
    'GearProtocol',
    'GearMessage',
    'MessageIntent',
]
