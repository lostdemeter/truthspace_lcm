"""
Role Gear

Transforms roles based on concept type (person, abstract, plural, etc.)

Author: Lesley Gushurst
License: GPLv3
"""

from typing import Set

from truthspace_lcm.gears.core import Gear, GearState


class RoleGear(Gear):
    """
    Transforms roles based on concept type.
    
    Detects whether a concept is:
    - A person (keeps as 'character')
    - An abstract concept (changes to 'concept')
    - A plural noun (changes to 'concept')
    - Other (uses default mapping)
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("RoleGear", ratio)
        
        # Known person names
        self.person_names: Set[str] = {
            'holmes', 'watson', 'moriarty', 'lestrade', 'mycroft', 'irene',
            'darwin', 'einstein', 'newton', 'galileo', 'aristotle', 'plato',
            'socrates', 'descartes', 'kant', 'hegel', 'nietzsche', 'marx',
        }
        
        # Abstract concept suffixes
        self.abstract_suffixes = [
            'ology', 'ics', 'istry', 'tion', 'ment', 'ness', 
            'ism', 'ure', 'ance', 'ence', 'ity', 'ty'
        ]
        
        # Role mappings
        self.role_map = {
            'character': 'concept',
            'someone': 'entity',
            'protagonist': 'concept',
        }
    
    def add_person(self, name: str) -> 'RoleGear':
        """Add a person name to the known list."""
        self.person_names.add(name.lower())
        return self
    
    def add_abstract_suffix(self, suffix: str) -> 'RoleGear':
        """Add an abstract concept suffix."""
        self.abstract_suffixes.append(suffix.lower())
        return self
    
    def forward(self, state: GearState) -> GearState:
        entity_lower = state.entity.lower()
        
        # Detect concept type
        is_person = any(name in entity_lower for name in self.person_names)
        is_abstract = any(entity_lower.endswith(s) for s in self.abstract_suffixes)
        is_plural = (entity_lower.endswith('s') and 
                    not entity_lower.endswith('ss') and 
                    len(entity_lower) > 3 and
                    entity_lower not in self.person_names)
        
        # Transform role
        if state.role in ['character', 'someone', 'protagonist']:
            if is_person:
                pass  # Keep as character
            elif is_abstract:
                state.role = 'concept'
            elif is_plural:
                state.role = 'concept'
            else:
                state.role = self.role_map.get(state.role, 'concept')
        
        return state
