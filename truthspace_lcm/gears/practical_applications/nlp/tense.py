"""
Tense Gear

Transforms verb tenses (present, past, future, perfect).

Author: Lesley Gushurst
License: GPLv3
"""

from truthspace_lcm.gears.core import Gear, GearState


class TenseGear(Gear):
    """
    Transforms verb tenses.
    
    Supports:
    - present: gerunds (investigating)
    - past: past tense (investigated)
    - future: base form with 'will' connector
    - perfect: past participle with 'has' connector
    
    The gear ratio can control tense selection in 'auto' mode:
    - 0.0-0.25: past
    - 0.25-0.5: present
    - 0.5-0.75: future
    - 0.75-1.0: perfect
    """
    
    def __init__(self, ratio: float = 0.4, tense: str = 'present'):
        super().__init__("TenseGear", ratio)
        self.tense = tense
    
    def set_tense(self, tense: str) -> 'TenseGear':
        """Set the target tense."""
        self.tense = tense
        return self
    
    def forward(self, state: GearState) -> GearState:
        # Determine tense
        if self.tense == 'auto':
            if self.ratio < 0.25:
                tense = 'past'
            elif self.ratio < 0.5:
                tense = 'present'
            elif self.ratio < 0.75:
                tense = 'future'
            else:
                tense = 'perfect'
        else:
            tense = self.tense
        
        state.tense = tense
        
        # Transform actions
        state.actions = [self._to_tense(a, tense) for a in state.actions]
        
        # Set connector based on tense
        if tense == 'past':
            state.connector = "that"
            state.use_gerunds = False
        elif tense == 'future':
            state.connector = "that will"
            state.use_gerunds = False
        elif tense == 'perfect':
            state.connector = "that has"
            state.use_gerunds = False
        elif tense == 'present':
            state.connector = "that involves"
            state.use_gerunds = True
        
        return state
    
    def _to_tense(self, verb: str, tense: str) -> str:
        """Convert verb to specified tense."""
        base = self._to_base(verb)
        
        if tense == 'present':
            return self._to_gerund(base) if verb.endswith('ing') else self._to_gerund(base)
        elif tense == 'past':
            return self._to_past(base)
        elif tense == 'future':
            return base
        elif tense == 'perfect':
            return self._to_past(base)  # Past participle same as past for regular verbs
        
        return verb
    
    def _to_base(self, verb: str) -> str:
        """Convert verb to base form."""
        verb = verb.lower().strip()
        
        if verb.endswith('ing'):
            base = verb[:-3]
            # Add back 'e' for verbs like 'make' -> 'making'
            if base.endswith('at') or base.endswith('it') or base.endswith('ut'):
                return base + 'e'
            if base.endswith('ak') or base.endswith('iv') or base.endswith('ov'):
                return base + 'e'
            return base
        
        if verb.endswith('ies'):
            return verb[:-3] + 'y'
        if verb.endswith('es') and len(verb) > 3:
            return verb[:-1]
        if verb.endswith('s') and not verb.endswith('ss'):
            return verb[:-1]
        
        return verb
    
    def _to_gerund(self, base: str) -> str:
        """Convert base form to gerund."""
        if base.endswith('e') and not base.endswith('ee'):
            return base[:-1] + 'ing'
        elif base.endswith('ie'):
            return base[:-2] + 'ying'
        # Don't double consonants - just add 'ing'
        return base + 'ing'
    
    def _to_base(self, verb: str) -> str:
        """Convert verb to base form."""
        verb = verb.lower().strip()
        
        # Already base form check
        if not any(verb.endswith(s) for s in ['ing', 'ed', 's', 'es', 'ies']):
            return verb
        
        if verb.endswith('ing'):
            base = verb[:-3]
            # Add back 'e' for verbs like 'make' -> 'making' -> 'make'
            # Check common patterns
            if len(base) > 1 and base[-1] in 'vtdkc' and base[-2] not in 'aeiou':
                return base + 'e'
            return base
        
        if verb.endswith('ied'):
            return verb[:-3] + 'y'
        
        if verb.endswith('ed'):
            base = verb[:-2]
            # Check if we need to add 'e'
            if len(base) > 1 and base[-1] in 'vtdkc':
                return base + 'e'
            return base
        
        if verb.endswith('ies'):
            return verb[:-3] + 'y'
        
        if verb.endswith('es') and len(verb) > 3:
            base = verb[:-2]
            # 'processes' -> 'process', 'changes' -> 'change'
            if base.endswith('ss') or base.endswith('ch') or base.endswith('sh'):
                return base
            return base + 'e'
        
        if verb.endswith('s') and not verb.endswith('ss') and len(verb) > 2:
            return verb[:-1]
        
        return verb
    
    def _to_past(self, base: str) -> str:
        """Convert base form to past tense."""
        if base.endswith('e'):
            return base + 'd'
        elif base.endswith('y') and len(base) > 2 and base[-2] not in 'aeiou':
            return base[:-1] + 'ied'
        return base + 'ed'
