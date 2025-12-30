"""
NLP Gears for Natural Language Processing

Domain-specific gears for text transformation, chat, and language understanding.

Gears:
- RoleGear: Transforms roles based on concept type
- ActionGear: Converts verbs to gerunds
- TenseGear: Transforms verb tenses
- SignalGear: Applies style patterns
- DomainGear: Domain-specific transformations
- StructureGear: Sentence structure decisions
- OutputGear: Assembles final text output

Usage:
    from truthspace_lcm.gears.nlp import RoleGear, ActionGear, TenseGear, OutputGear
    from truthspace_lcm.gears.core import GearChain
    
    chain = GearChain("NLPPipeline")
    chain.add(RoleGear())
    chain.add(ActionGear())
    chain.add(TenseGear(tense='past'))
    chain.add(OutputGear())
"""

from .role import RoleGear
from .action import ActionGear
from .tense import TenseGear
from .signal_gear import SignalGear
from .domain import DomainGear
from .structure import StructureGear
from .output import OutputGear

__all__ = [
    'RoleGear',
    'ActionGear',
    'TenseGear',
    'SignalGear',
    'DomainGear',
    'StructureGear',
    'OutputGear',
]
