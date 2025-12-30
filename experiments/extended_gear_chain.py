#!/usr/bin/env python3
"""
Extended Gear Chain with Signal Gear

Instead of bypassing the gear chain for signal frames, we make signal
a gear in the chain. This allows:

1. Signal patterns to be applied consistently
2. Other gears to still transform the output
3. Unlimited gear composition

The chain becomes:
    Truth → [RoleGear] → [ActionGear] → [SignalGear] → [StructureGear] → [OutputGear] → Final

SignalGear learns patterns from the signal corpus and applies them to
transform the state, rather than replacing the output entirely.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


@dataclass
class Quaternion:
    """Quaternion for encoding gear parameters."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
        )
    
    def normalize(self) -> 'Quaternion':
        n = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if n < 1e-10:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)


@dataclass
class GearState:
    """State passed between gears."""
    entity: str = ""
    role: str = "entity"
    actions: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    
    # Accumulated quaternion from gear chain
    accumulated_q: Quaternion = field(default_factory=Quaternion)
    
    # Style flags
    use_prefix: bool = False
    use_gerunds: bool = True
    connector: str = "that involves"
    target_connector: str = "particularly"
    
    # Signal-specific additions
    signal_prefix: str = ""
    signal_suffix: str = ""
    signal_style: str = "default"  # 'formal', 'casual', 'technical', etc.


class Gear:
    """Base class for gears."""
    
    def __init__(self, name: str, ratio: float = 1.0):
        self.name = name
        self.ratio = ratio
        self.quaternion = Quaternion(1, 0, 0, 0)
    
    def forward(self, state: GearState) -> GearState:
        """Transform state forward."""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.name}(ratio={self.ratio})"


class RoleGear(Gear):
    """Transforms roles based on concept type."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("RoleGear", ratio)
        self.person_names = {'holmes', 'watson', 'moriarty', 'lestrade', 'mycroft', 'irene'}
        self.abstract_suffixes = ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism', 'ure', 'ance', 'ence']
    
    def forward(self, state: GearState) -> GearState:
        entity_lower = state.entity.lower()
        
        is_person = any(name in entity_lower for name in self.person_names)
        is_abstract = any(entity_lower.endswith(s) for s in self.abstract_suffixes)
        is_plural = (entity_lower.endswith('s') and not entity_lower.endswith('ss') and 
                    len(entity_lower) > 3 and entity_lower not in self.person_names)
        
        if state.role in ['character', 'someone', 'protagonist']:
            if is_person:
                pass  # Keep as character
            elif is_abstract or is_plural:
                state.role = 'concept'
            else:
                state.role = 'concept'  # Default transform
        
        return state


class ActionGear(Gear):
    """Transforms verbs to gerunds."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("ActionGear", ratio)
        
        self.to_gerund = {
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
            'deduces': 'deducing', 'collisions': 'colliding',
            'collides': 'colliding', 'highlights': 'highlighting',
            'includes': 'including', 'prompts': 'prompting',
            'provides': 'providing', 'expands': 'expanding',
        }
    
    def forward(self, state: GearState) -> GearState:
        if self.ratio > 0.5:
            state.use_gerunds = True
            state.actions = [self._to_gerund(a) for a in state.actions]
        else:
            state.use_gerunds = False
        return state
    
    def _to_gerund(self, verb: str) -> str:
        verb = verb.lower().strip()
        if verb in self.to_gerund:
            return self.to_gerund[verb]
        if verb.endswith('ing'):
            return verb
        elif verb.endswith('e') and not verb.endswith('ee'):
            return verb[:-1] + 'ing'
        elif verb.endswith('s') and not verb.endswith('ss'):
            base = verb[:-1]
            if base.endswith('e'):
                return base[:-1] + 'ing'
            return base + 'ing'
        return verb + 'ing'


class SignalGear(Gear):
    """
    Applies signal corpus patterns to the state.
    
    Instead of replacing output with signal frames, this gear:
    1. Learns style patterns from signal corpus
    2. Applies those patterns to transform the state
    3. Passes the transformed state to the next gear
    
    This allows signal influence while maintaining gear chain integrity.
    """
    
    def __init__(self, signal_corpus_path: str, ratio: float = 1.0):
        super().__init__("SignalGear", ratio)
        
        # Learn patterns from signal corpus
        self.patterns = self._learn_patterns(signal_corpus_path)
        self.style_quaternions = self._compute_style_quaternions()
        
        print(f"SignalGear learned {len(self.patterns)} patterns")
    
    def _learn_patterns(self, corpus_path: str) -> Dict[str, Any]:
        """Learn style patterns from signal corpus."""
        patterns = {
            'prefixes': Counter(),
            'connectors': Counter(),
            'target_connectors': Counter(),
            'suffixes': Counter(),
            'role_phrases': defaultdict(Counter),
        }
        
        if not os.path.exists(corpus_path):
            return patterns
        
        with open(corpus_path, 'r') as f:
            data = json.load(f)
        
        for frame in data.get('frames', []):
            text = frame.get('text', '')
            text_lower = text.lower()
            
            # Learn prefixes
            if text.startswith('It seems'):
                patterns['prefixes']['it_seems'] += 1
            elif text.startswith('It appears'):
                patterns['prefixes']['it_appears'] += 1
            else:
                patterns['prefixes']['direct'] += 1
            
            # Learn connectors
            if 'that involves' in text_lower:
                patterns['connectors']['that_involves'] += 1
            elif 'who' in text_lower and 'is a' in text_lower:
                patterns['connectors']['who'] += 1
            elif 'that' in text_lower:
                patterns['connectors']['that'] += 1
            
            # Learn target connectors
            if 'particularly' in text_lower:
                patterns['target_connectors']['particularly'] += 1
            elif 'relating to' in text_lower:
                patterns['target_connectors']['relating_to'] += 1
            elif 'focusing on' in text_lower:
                patterns['target_connectors']['focusing_on'] += 1
            
            # Learn role phrases
            match = re.search(r'is a[n]? (\w+)', text_lower)
            if match:
                role = match.group(1)
                # Find what comes after the role
                after_role = text_lower.split(f'is a {role}')[1] if f'is a {role}' in text_lower else ''
                if after_role:
                    first_word = after_role.strip().split()[0] if after_role.strip() else ''
                    patterns['role_phrases'][role][first_word] += 1
        
        return patterns
    
    def _compute_style_quaternions(self) -> Dict[str, Quaternion]:
        """Compute quaternions representing different styles."""
        styles = {}
        
        # Formal style: high w (confidence), low x (variation)
        styles['formal'] = Quaternion(0.9, 0.1, 0.0, 0.0)
        
        # Casual style: medium w, higher x
        styles['casual'] = Quaternion(0.7, 0.3, 0.1, 0.0)
        
        # Technical style: high w, high y (precision)
        styles['technical'] = Quaternion(0.8, 0.1, 0.5, 0.0)
        
        # Narrative style: medium w, high z (flow)
        styles['narrative'] = Quaternion(0.6, 0.2, 0.1, 0.5)
        
        return styles
    
    def forward(self, state: GearState) -> GearState:
        """Apply signal patterns to state."""
        
        # Determine style based on role and entity
        if state.role in ['detective', 'doctor', 'character']:
            state.signal_style = 'narrative'
        elif state.role in ['concept', 'field', 'science']:
            state.signal_style = 'technical'
        else:
            state.signal_style = 'formal'
        
        # Apply style quaternion
        if state.signal_style in self.style_quaternions:
            style_q = self.style_quaternions[state.signal_style]
            state.accumulated_q = state.accumulated_q * style_q
        
        # Determine prefix based on patterns
        if self.patterns['prefixes']:
            total = sum(self.patterns['prefixes'].values())
            it_seems_ratio = self.patterns['prefixes'].get('it_seems', 0) / total if total > 0 else 0
            
            # Use prefix if signal corpus often uses it AND ratio allows
            if it_seems_ratio > 0.1 and self.ratio > 0.5:
                state.use_prefix = True
                state.signal_prefix = "It seems that"
        
        # Determine connector based on patterns and gerund usage
        # BUT preserve special tense connectors (that will, that has)
        if state.connector in ["that will", "that has"]:
            pass  # Preserve tense-specific connector
        elif state.use_gerunds:
            state.connector = "that involves"
        elif self.patterns['connectors']:
            # Use most common connector from signal
            most_common = self.patterns['connectors'].most_common(1)
            if most_common:
                conn = most_common[0][0]
                if conn == 'who' and state.role in ['detective', 'doctor', 'character']:
                    state.connector = "who"
                elif conn == 'that_involves':
                    state.connector = "that involves"
                else:
                    state.connector = "that"
        
        # Determine target connector
        if self.patterns['target_connectors']:
            most_common = self.patterns['target_connectors'].most_common(1)
            if most_common:
                tc = most_common[0][0]
                if tc == 'particularly':
                    state.target_connector = "particularly"
                elif tc == 'focusing_on':
                    state.target_connector = "focusing on"
                else:
                    state.target_connector = "relating to"
        
        return state


class StructureGear(Gear):
    """Determines final sentence structure."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("StructureGear", ratio)
    
    def forward(self, state: GearState) -> GearState:
        q = state.accumulated_q.normalize()
        
        # Use quaternion to make final structure decisions
        # w component: formality
        # x component: variation
        # y component: precision
        # z component: flow
        
        # Prefix decision
        if not state.signal_prefix and q.w < 0.5:
            state.use_prefix = True
        
        return state


class OutputGear(Gear):
    """Assembles final output string."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("OutputGear", ratio)
    
    def forward(self, state: GearState) -> str:
        # Build prefix
        if state.use_prefix:
            if state.signal_prefix:
                prefix = f"{state.signal_prefix} {state.entity} is"
            else:
                prefix = f"It appears that {state.entity} is"
        else:
            prefix = f"{state.entity} is"
        
        # Article
        article = "an" if state.role[0].lower() in 'aeiou' else "a"
        
        # Build action string
        if state.actions:
            if len(state.actions) == 1:
                action_str = state.actions[0]
            elif len(state.actions) == 2:
                action_str = f"{state.actions[0]} and {state.actions[1]}"
            else:
                action_str = f"{state.actions[0]}, {state.actions[1]}, and {state.actions[2]}"
        else:
            action_str = ""
        
        # Build target string
        target_str = ' and '.join(state.targets[:2]) if state.targets else ""
        
        # Assemble - handle special connectors
        connector = state.connector
        
        # For "that will" and "that has", we need different structure
        if connector in ["that will", "that has"]:
            if action_str and target_str:
                return f"{prefix} {article} {state.role} {connector} {action_str}, {state.target_connector} {target_str}."
            elif action_str:
                return f"{prefix} {article} {state.role} {connector} {action_str}."
            elif target_str:
                return f"{prefix} {article} {state.role} {state.target_connector} {target_str}."
            else:
                return f"{prefix} {article} {state.role}."
        else:
            if action_str and target_str:
                return f"{prefix} {article} {state.role} {connector} {action_str}, {state.target_connector} {target_str}."
            elif action_str:
                return f"{prefix} {article} {state.role} {connector} {action_str}."
            elif target_str:
                return f"{prefix} {article} {state.role} {state.target_connector} {target_str}."
            else:
                return f"{prefix} {article} {state.role}."


class TenseGear(Gear):
    """
    Transforms verb tenses.
    
    Supports:
    - present: "investigates", "investigating"
    - past: "investigated"
    - future: "will investigate"
    - perfect: "has investigated"
    
    The gear ratio controls tense selection:
    - 0.0-0.25: past
    - 0.25-0.5: present
    - 0.5-0.75: future
    - 0.75-1.0: perfect
    
    Or set tense explicitly via set_tense().
    """
    
    def __init__(self, ratio: float = 0.4, tense: str = 'present'):
        super().__init__("TenseGear", ratio)
        self.tense = tense  # 'past', 'present', 'future', 'perfect'
        
        # Irregular verb forms
        self.irregulars = {
            # base: (past, past_participle)
            'investigate': ('investigated', 'investigated'),
            'study': ('studied', 'studied'),
            'examine': ('examined', 'examined'),
            'explore': ('explored', 'explored'),
            'analyze': ('analyzed', 'analyzed'),
            'solve': ('solved', 'solved'),
            'process': ('processed', 'processed'),
            'involve': ('involved', 'involved'),
            'formalize': ('formalized', 'formalized'),
            'confirm': ('confirmed', 'confirmed'),
            'articulate': ('articulated', 'articulated'),
            'present': ('presented', 'presented'),
            'observe': ('observed', 'observed'),
            'monitor': ('monitored', 'monitored'),
            'facilitate': ('facilitated', 'facilitated'),
            'influence': ('influenced', 'influenced'),
            'emphasize': ('emphasized', 'emphasized'),
            'structure': ('structured', 'structured'),
            'illuminate': ('illuminated', 'illuminated'),
            'experience': ('experienced', 'experienced'),
            'perceive': ('perceived', 'perceived'),
            'pressure': ('pressured', 'pressured'),
            'mark': ('marked', 'marked'),
            'change': ('changed', 'changed'),
            'develop': ('developed', 'developed'),
            'adapt': ('adapted', 'adapted'),
            'focus': ('focused', 'focused'),
            'support': ('supported', 'supported'),
            'assist': ('assisted', 'assisted'),
            'document': ('documented', 'documented'),
            'deduce': ('deduced', 'deduced'),
            'collide': ('collided', 'collided'),
            'highlight': ('highlighted', 'highlighted'),
            'include': ('included', 'included'),
            'prompt': ('prompted', 'prompted'),
            'provide': ('provided', 'provided'),
            'expand': ('expanded', 'expanded'),
            'combine': ('combined', 'combined'),
            'integrate': ('integrated', 'integrated'),
            'bring': ('brought', 'brought'),
            'underscore': ('underscored', 'underscored'),
            'divide': ('divided', 'divided'),
            'quantify': ('quantified', 'quantified'),
            'ignore': ('ignored', 'ignored'),
            'shape': ('shaped', 'shaped'),
            'rigorize': ('rigorized', 'rigorized'),
        }
    
    def set_tense(self, tense: str) -> 'TenseGear':
        """Set the target tense."""
        self.tense = tense
        return self
    
    def forward(self, state: GearState) -> GearState:
        """Transform verbs to target tense."""
        # Determine tense from ratio if not explicitly set
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
        
        # Transform actions to target tense
        transformed = []
        for action in state.actions:
            transformed.append(self._to_tense(action, tense))
        
        state.actions = transformed
        
        # Adjust connector based on tense
        if tense == 'past':
            state.connector = "that"  # "concept that investigated"
            state.use_gerunds = False
        elif tense == 'future':
            state.connector = "that will"
            state.use_gerunds = False
        elif tense == 'perfect':
            state.connector = "that has"
            state.use_gerunds = False
        elif tense == 'present':
            # Keep gerunds for present tense
            state.connector = "that involves"
            state.use_gerunds = True
        
        return state
    
    def _to_tense(self, verb: str, tense: str) -> str:
        """Convert verb to specified tense."""
        # First get base form
        base = self._to_base(verb)
        
        if tense == 'present':
            # Return gerund or 3rd person based on state
            if verb.endswith('ing'):
                return verb  # Already gerund
            return self._to_gerund(base)
        
        elif tense == 'past':
            if base in self.irregulars:
                return self.irregulars[base][0]
            # Regular past tense
            if base.endswith('e'):
                return base + 'd'
            elif base.endswith('y') and len(base) > 2 and base[-2] not in 'aeiou':
                return base[:-1] + 'ied'
            else:
                return base + 'ed'
        
        elif tense == 'future':
            return base  # "will" is added by connector
        
        elif tense == 'perfect':
            if base in self.irregulars:
                return self.irregulars[base][1]
            # Regular past participle (same as past for regular verbs)
            if base.endswith('e'):
                return base + 'd'
            elif base.endswith('y') and len(base) > 2 and base[-2] not in 'aeiou':
                return base[:-1] + 'ied'
            else:
                return base + 'ed'
        
        return verb
    
    def _to_base(self, verb: str) -> str:
        """Convert verb to base form."""
        verb = verb.lower().strip()
        
        # Remove gerund ending
        if verb.endswith('ing'):
            base = verb[:-3]
            # Check if we need to add back 'e'
            if base + 'e' in self.irregulars:
                return base + 'e'
            if base.endswith('at') or base.endswith('it') or base.endswith('ut'):
                return base + 'e'
            return base
        
        # Remove 3rd person 's'
        if verb.endswith('ies'):
            return verb[:-3] + 'y'
        if verb.endswith('es') and len(verb) > 3:
            base = verb[:-2]
            if base + 'e' in self.irregulars:
                return base + 'e'
            return verb[:-1]  # Just remove 's'
        if verb.endswith('s') and not verb.endswith('ss'):
            return verb[:-1]
        
        return verb
    
    def _to_gerund(self, base: str) -> str:
        """Convert base form to gerund."""
        if base.endswith('e') and not base.endswith('ee'):
            return base[:-1] + 'ing'
        elif base.endswith('ie'):
            return base[:-2] + 'ying'
        else:
            return base + 'ing'


class DomainGear(Gear):
    """
    Applies domain-specific transformations.
    
    Different domains (science, narrative, technical) have different
    conventions for how concepts are described.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("DomainGear", ratio)
        
        # Domain detection patterns
        self.science_markers = ['physics', 'chemistry', 'biology', 'mathematics', 'quantum', 'molecular']
        self.narrative_markers = ['holmes', 'watson', 'mystery', 'detective', 'crime']
        self.technical_markers = ['algorithm', 'system', 'process', 'function', 'method']
    
    def forward(self, state: GearState) -> GearState:
        entity_lower = state.entity.lower()
        targets_lower = ' '.join(state.targets).lower() if state.targets else ''
        
        # Detect domain
        if any(m in entity_lower or m in targets_lower for m in self.science_markers):
            state.signal_style = 'technical'
            # Science prefers precise language
            if state.connector == 'who':
                state.connector = 'that'
        
        elif any(m in entity_lower or m in targets_lower for m in self.narrative_markers):
            state.signal_style = 'narrative'
            # Narrative prefers "who" for characters
            if state.role in ['detective', 'doctor', 'character']:
                state.connector = 'who'
                state.use_gerunds = False
        
        elif any(m in entity_lower or m in targets_lower for m in self.technical_markers):
            state.signal_style = 'technical'
        
        return state


class ExtendedGearChain:
    """
    Extended gear chain with signal gear.
    
    Chain: Truth → [Role] → [Action] → [Signal] → [Domain] → [Structure] → [Output]
    
    Each gear transforms the state, and the final output reflects
    all transformations applied.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Build extended gear chain
        # TenseGear comes after ActionGear to transform the gerunds/verbs
        self.gears = [
            RoleGear(ratio=1.0),
            ActionGear(ratio=1.0),
            TenseGear(ratio=0.4, tense='present'),  # Default to present tense
            SignalGear(signal_corpus_path, ratio=1.0),
            DomainGear(ratio=1.0),
            StructureGear(ratio=1.0),
            OutputGear(ratio=1.0),
        ]
        
        print(f"Extended gear chain: {' → '.join(g.name for g in self.gears)}")
    
    def project(self, concept: str) -> str:
        """Project concept through the full gear chain."""
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # Parse truth into initial state
        state = self._parse_to_state(truth, concept)
        
        # Run through ALL gears (including signal)
        for gear in self.gears[:-1]:
            state = gear.forward(state)
        
        # Output gear returns string
        return self.gears[-1].forward(state)
    
    def _parse_to_state(self, truth: str, concept: str) -> GearState:
        """Parse truth into gear state."""
        truth_lower = truth.lower()
        
        state = GearState()
        state.entity = concept.title()
        
        # Role
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            state.role = match.group(1)
        
        # Actions
        match = re.search(r'who\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            state.actions = [a for a in match.groups() if a]
        else:
            match = re.search(r'is a \w+ that\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
            if match:
                state.actions = [a for a in match.groups() if a]
        
        # Targets
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            state.targets = [t for t in match.groups() if t]
        
        return state
    
    def add_gear(self, gear: Gear, position: int = -1) -> 'ExtendedGearChain':
        """Add a gear to the chain at the specified position."""
        if position == -1:
            # Insert before OutputGear
            self.gears.insert(-1, gear)
        else:
            self.gears.insert(position, gear)
        
        print(f"Added {gear.name}. Chain: {' → '.join(g.name for g in self.gears)}")
        return self
    
    def remove_gear(self, name: str) -> 'ExtendedGearChain':
        """Remove a gear by name."""
        self.gears = [g for g in self.gears if g.name != name]
        print(f"Removed {name}. Chain: {' → '.join(g.name for g in self.gears)}")
        return self
    
    def set_gear_ratio(self, name: str, ratio: float) -> 'ExtendedGearChain':
        """Set the ratio for a specific gear."""
        for gear in self.gears:
            if gear.name == name:
                gear.ratio = ratio
                print(f"Set {name} ratio to {ratio}")
                break
        return self


def demo():
    """Demo the extended gear chain."""
    print("=" * 70)
    print("EXTENDED GEAR CHAIN WITH SIGNAL GEAR")
    print("Signal frames are now a gear, not a bypass")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    chain = ExtendedGearChain(truth_path, signal_path)
    
    # Test concepts
    test_concepts = [
        'missions', 'reforms', 'events', 'development',
        'evolution', 'biochemistry', 'analysis',
        'holmes', 'watson', 'physics'
    ]
    
    print("\n" + "=" * 70)
    print("Testing extended gear chain:")
    print("=" * 70)
    
    for concept in test_concepts:
        output = chain.project(concept)
        print(f"\n{concept.upper()}: {output[:90]}..." if len(output) > 90 else f"\n{concept.upper()}: {output}")
    
    # Demo gear manipulation
    print("\n" + "=" * 70)
    print("Gear manipulation demo:")
    print("=" * 70)
    
    # Test different tenses
    concept = 'evolution'
    
    # Find TenseGear
    tense_gear = None
    for g in chain.gears:
        if g.name == "TenseGear":
            tense_gear = g
            break
    
    if tense_gear:
        print(f"\nTense transformations for '{concept}':")
        
        for tense in ['present', 'past', 'future', 'perfect']:
            tense_gear.set_tense(tense)
            output = chain.project(concept)
            print(f"  {tense.upper():8} → {output}")
        
        # Reset to present
        tense_gear.set_tense('present')
    
    # Also demo action gear ratio
    print(f"\nAction gear ratio demo:")
    chain.set_gear_ratio("ActionGear", 0.3)
    print(f"  ActionGear ratio=0.3: {chain.project('analysis')}")
    chain.set_gear_ratio("ActionGear", 1.0)
    print(f"  ActionGear ratio=1.0: {chain.project('analysis')}")


if __name__ == "__main__":
    demo()
