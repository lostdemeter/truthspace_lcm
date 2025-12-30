"""
Structure Gear

Determines final sentence structure based on accumulated state.

Author: Lesley Gushurst
License: GPLv3
"""

from truthspace_lcm.gears.core import Gear, GearState


class StructureGear(Gear):
    """
    Determines final sentence structure.
    
    Uses the accumulated quaternion and other state to make
    final structural decisions before output.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("StructureGear", ratio)
    
    def forward(self, state: GearState) -> GearState:
        q = state.accumulated_q.normalize()
        
        # Use quaternion components for structure decisions
        # w: formality
        # x: variation
        # y: precision
        # z: flow
        
        # Prefix decision based on formality
        if not state.signal_prefix and q.w < 0.5:
            state.use_prefix = True
        
        return state
