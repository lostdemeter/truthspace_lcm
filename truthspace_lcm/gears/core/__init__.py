"""
Gear Chain Core

The foundational classes for building gear-based transformation pipelines.
These are domain-agnostic and can be used for any transformation task.

Core Components:
- Quaternion: 4D rotation encoding for gear parameters
- GearState: Base state object that flows through the chain
- Gear: Abstract base class for all transformation gears
- GearChain: Container for composing gears into pipelines

Usage:
    from truthspace_lcm.gears.core import Gear, GearState, GearChain, Quaternion
    
    class MyGear(Gear):
        def forward(self, state: GearState) -> GearState:
            # Transform state
            return state
    
    chain = GearChain("MyPipeline")
    chain.add(MyGear())
    result = chain.process(initial_state)
"""

from .base import Gear, GearState, GearChain, Quaternion
from .error_correction import ErrorCorrectionGear

__all__ = [
    'Gear',
    'GearState',
    'GearChain',
    'Quaternion',
    'ErrorCorrectionGear',
]
