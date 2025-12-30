"""
Domain Gear

Applies domain-specific transformations based on detected domain.

Author: Lesley Gushurst
License: GPLv3
"""

from typing import List

from truthspace_lcm.gears.core import Gear, GearState


class DomainGear(Gear):
    """
    Applies domain-specific transformations.
    
    Detects domain from entity and targets, then adjusts style accordingly.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("DomainGear", ratio)
        
        self.science_markers: List[str] = [
            'physics', 'chemistry', 'biology', 'mathematics', 'quantum', 
            'molecular', 'atomic', 'genetic', 'evolution', 'neuroscience'
        ]
        
        self.narrative_markers: List[str] = [
            'holmes', 'watson', 'mystery', 'detective', 'crime', 
            'story', 'character', 'plot', 'narrative'
        ]
        
        self.technical_markers: List[str] = [
            'algorithm', 'system', 'process', 'function', 'method',
            'computation', 'data', 'network', 'protocol'
        ]
    
    def add_domain_marker(self, domain: str, marker: str) -> 'DomainGear':
        """Add a marker for a domain."""
        if domain == 'science':
            self.science_markers.append(marker.lower())
        elif domain == 'narrative':
            self.narrative_markers.append(marker.lower())
        elif domain == 'technical':
            self.technical_markers.append(marker.lower())
        return self
    
    def forward(self, state: GearState) -> GearState:
        entity_lower = state.entity.lower()
        targets_lower = ' '.join(state.targets).lower() if state.targets else ''
        combined = entity_lower + ' ' + targets_lower
        
        # Detect domain
        if any(m in combined for m in self.science_markers):
            state.signal_style = 'technical'
            if state.connector == 'who':
                state.connector = 'that'
        
        elif any(m in combined for m in self.narrative_markers):
            state.signal_style = 'narrative'
            if state.role in ['detective', 'doctor', 'character']:
                state.connector = 'who'
                state.use_gerunds = False
        
        elif any(m in combined for m in self.technical_markers):
            state.signal_style = 'technical'
        
        return state
