"""
Action Gear

Transforms verbs to gerunds or other forms.

Author: Lesley Gushurst
License: GPLv3
"""

from typing import Dict

from truthspace_lcm.gears.core import Gear, GearState


class ActionGear(Gear):
    """
    Transforms action verbs.
    
    When ratio > 0.5, converts verbs to gerunds.
    When ratio <= 0.5, keeps base forms.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("ActionGear", ratio)
        
        # Known gerund mappings for common verbs
        self.to_gerund: Dict[str, str] = {
            'investigates': 'investigating', 'studies': 'studying',
            'examines': 'examining', 'explores': 'exploring',
            'analyzes': 'analyzing', 'solves': 'solving',
            'processes': 'processing', 'involves': 'involving',
            'formalizes': 'formalizing', 'confirms': 'confirming',
            'articulates': 'articulating', 'presents': 'presenting',
            'observes': 'observing', 'monitors': 'monitoring',
            'facilitates': 'facilitating', 'influences': 'influencing',
            'emphasizes': 'emphasizing', 'structures': 'structuring',
            'illuminates': 'illuminating', 'experiences': 'experiencing',
            'perceives': 'perceiving', 'pressures': 'pressuring',
            'marks': 'marking', 'changes': 'changing',
            'develops': 'developing', 'adapts': 'adapting',
            'focuses': 'focusing', 'supports': 'supporting',
            'assists': 'assisting', 'documents': 'documenting',
            'deduces': 'deducing', 'highlights': 'highlighting',
            'includes': 'including', 'prompts': 'prompting',
            'provides': 'providing', 'expands': 'expanding',
            'collides': 'colliding', 'collisions': 'colliding',
        }
    
    def add_gerund_mapping(self, verb: str, gerund: str) -> 'ActionGear':
        """Add a verb to gerund mapping."""
        self.to_gerund[verb.lower()] = gerund.lower()
        return self
    
    def forward(self, state: GearState) -> GearState:
        if self.ratio > 0.5:
            state.use_gerunds = True
            state.actions = [self._to_gerund(a) for a in state.actions]
        else:
            state.use_gerunds = False
        return state
    
    def _to_gerund(self, verb: str) -> str:
        """Convert verb to gerund form."""
        verb_lower = verb.lower().strip()
        
        # Check known mappings
        if verb_lower in self.to_gerund:
            return self.to_gerund[verb_lower]
        
        # Already a gerund
        if verb_lower.endswith('ing'):
            return verb_lower
        
        # Get base form first (remove 's' endings)
        base = verb_lower
        if base.endswith('ies'):
            base = base[:-3] + 'y'
        elif base.endswith('es') and len(base) > 3:
            base = base[:-2]
            if not base.endswith('e'):
                base = base + 'e'  # processes -> process -> processe -> process
        elif base.endswith('s') and not base.endswith('ss') and len(base) > 2:
            base = base[:-1]
        
        # Apply gerund rules to base
        if base.endswith('e') and not base.endswith('ee'):
            return base[:-1] + 'ing'
        elif base.endswith('ie'):
            return base[:-2] + 'ying'
        
        # Don't double consonants for most verbs
        # Only double for short verbs with CVC pattern (run -> running)
        return base + 'ing'
