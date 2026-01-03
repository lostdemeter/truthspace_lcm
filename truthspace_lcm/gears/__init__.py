"""
TruthSpace Gear System

A modular, extensible gear chain architecture for transformation pipelines.

The gear system is organized into:
- core/: Base classes (Gear, GearState, GearChain, Quaternion)
- corpus/: Corpus management utilities
- tools/: Corpus pruning, correction, and reinforcement tools
- practical_applications/: Domain-specific implementations
  - nlp/: NLP gears + chat/API applications
  - data/: Data transformation gears + pipeline demo

Usage (NLP):
    from truthspace_lcm.gears.core import GearChain, GearState
    from truthspace_lcm.gears.practical_applications.nlp import RoleGear, ActionGear, TenseGear, OutputGear
    
    chain = GearChain("NLPPipeline")
    chain.add(RoleGear())
    chain.add(ActionGear())
    chain.add(TenseGear(tense='past'))
    chain.add(OutputGear())
    
    result = chain.process(initial_state)

Usage (Data):
    from truthspace_lcm.gears.core import GearChain
    from truthspace_lcm.gears.practical_applications.data import ValidationGear, NormalizationGear, FormatGear, DataState
    
    chain = GearChain("DataPipeline")
    chain.add(ValidationGear())
    chain.add(NormalizationGear())
    chain.add(FormatGear(format='json'))
    
    state = DataState()
    state.add_records(data)
    result = chain.process(state)

Corpus management:
    from truthspace_lcm.gears.corpus import load_corpus, save_corpus
    
Tools:
    from truthspace_lcm.gears.tools import CorpusPruner, CorpusCorrector, CorpusReinforcer
"""

# Re-export core classes for convenience
from .core import Gear, GearState, GearChain, Quaternion

# Re-export NLP gears for convenience
from .practical_applications.nlp import (
    RoleGear, ActionGear, TenseGear,
    SignalGear, DomainGear, StructureGear, OutputGear,
)

__all__ = [
    # Core classes
    'Gear',
    'GearState', 
    'GearChain',
    'Quaternion',
    # NLP Gears (for backward compatibility)
    'RoleGear',
    'ActionGear',
    'TenseGear',
    'SignalGear',
    'DomainGear',
    'StructureGear',
    'OutputGear',
]
