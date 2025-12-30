#!/usr/bin/env python3
"""
Gear Chain Projection

Instead of learned patterns, we chain multiple gears together:

    Truth → [Role Gear] → [Action Gear] → [Structure Gear] → [Output Gear] → Signal

Each gear handles ONE transformation:
- Role Gear: Transforms role (character → concept, someone → entity)
- Action Gear: Transforms verbs (investigates → investigating)
- Structure Gear: Determines sentence structure (prefix, connector, etc.)
- Output Gear: Assembles final output

Gear ratios control how much each transformation applies.
Quaternions encode the transformation parameters.

This is pure geometric composition - no patterns, no templates.

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
    
    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Quaternion':
        n = self.norm()
        if n < 1e-10:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def scale(self, factor: float) -> 'Quaternion':
        """Scale quaternion (for gear ratio)."""
        return Quaternion(self.w * factor, self.x * factor, self.y * factor, self.z * factor)


@dataclass
class GearState:
    """State passed between gears in the chain."""
    entity: str = ""
    role: str = "entity"
    actions: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    
    # Accumulated quaternion from gear chain
    accumulated_q: Quaternion = field(default_factory=Quaternion)
    
    # Style flags set by gears
    use_prefix: bool = False
    use_gerunds: bool = True
    connector: str = "that involves"
    target_connector: str = "particularly"


class Gear:
    """Base class for a gear in the chain."""
    
    def __init__(self, name: str, ratio: float = 1.0):
        self.name = name
        self.ratio = ratio
        self.quaternion = Quaternion(1, 0, 0, 0)
    
    def mesh(self, state: GearState) -> GearState:
        """Transform state through this gear."""
        raise NotImplementedError
    
    def set_quaternion(self, q: Quaternion):
        """Set gear's quaternion parameters."""
        self.quaternion = q


class RoleGear(Gear):
    """
    Transforms roles based on concept type.
    
    Quaternion encoding:
    - w: role confidence (high = keep original, low = transform)
    - x: character-ness (high = person, low = abstract)
    - y: scientific-ness (high = scientific term)
    - z: reserved
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("RoleGear", ratio)
        
        # Role transformation rules (learned from signal corpus)
        self.role_map = {
            'character': 'concept',  # Default transform
            'someone': 'entity',
            'protagonist': 'concept',
            'entity': 'entity',
            'concept': 'concept',
            'detective': 'detective',
            'doctor': 'doctor',
            'science': 'science',
            'study': 'field',
            'field': 'field',
        }
        
        # Person names that should keep "character" role
        self.person_names = {'holmes', 'watson', 'moriarty', 'lestrade', 'mycroft', 'irene'}
    
    def mesh(self, state: GearState) -> GearState:
        """Transform role through this gear."""
        entity_lower = state.entity.lower()
        role = state.role
        
        # Compute role quaternion based on entity
        q = Quaternion(1, 0, 0, 0)
        
        # Check if this is a person
        is_person = any(name in entity_lower for name in self.person_names)
        q.x = 1.0 if is_person else -1.0
        
        # Check if scientific term
        scientific_suffixes = ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism', 'ons', 'ure']
        is_scientific = any(entity_lower.endswith(s) for s in scientific_suffixes)
        q.y = 1.0 if is_scientific else 0.0
        
        # Apply gear ratio
        q = q.scale(self.ratio)
        
        # Transform role based on quaternion
        if role in ['character', 'someone', 'protagonist']:
            if q.x > 0:  # Person
                state.role = 'character' if role == 'character' else state.role
            elif q.y > 0:  # Scientific
                state.role = 'concept'
            else:
                state.role = self.role_map.get(role, 'concept')
        else:
            state.role = self.role_map.get(role, role)
        
        # Accumulate quaternion
        state.accumulated_q = state.accumulated_q * q.normalize()
        
        return state


class ActionGear(Gear):
    """
    Transforms actions (verbs).
    
    Quaternion encoding:
    - w: transformation strength
    - x: gerund preference (high = use -ing forms)
    - y: formality (high = formal verbs)
    - z: reserved
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("ActionGear", ratio)
        
        # Verb transformations
        self.verb_map = {
            'investigates': 'investigating', 'studies': 'studying',
            'examines': 'examining', 'explores': 'exploring',
            'analyzes': 'analyzing', 'solves': 'solving',
            'deduces': 'deducing', 'assists': 'assisting',
            'supports': 'supporting', 'documents': 'documenting',
            'changes': 'changing', 'develops': 'developing',
            'adapts': 'adapting', 'transforms': 'transforming',
            'processes': 'processing', 'involves': 'involving',
            'encompasses': 'encompassing', 'illuminates': 'illuminating',
            'experiences': 'experiencing', 'perceives': 'perceiving',
            'powers': 'powering', 'focuses': 'focusing',
            'calculates': 'calculating', 'proves': 'proving',
            'confirms': 'confirming', 'articulates': 'articulating',
            'presents': 'presenting', 'observes': 'observing',
            'monitors': 'monitoring', 'formalizes': 'formalizing',
            'pressures': 'pressuring', 'marks': 'marking',
            'causes': 'causing', 'stems': 'stemming',
            'emphasizes': 'emphasizing', 'overlaps': 'overlapping',
            'describes': 'describing', 'ejects': 'ejecting',
            'consists': 'consisting', 'rigorizes': 'rigorizing',
        }
    
    def mesh(self, state: GearState) -> GearState:
        """Transform actions through this gear."""
        # Gear quaternion controls transformation
        q = Quaternion(1, self.ratio, 0, 0)  # x controls gerund preference
        
        # Transform actions based on gear ratio
        if self.ratio > 0.5:
            # High ratio = use gerunds
            state.use_gerunds = True
            state.actions = [self._to_gerund(a) for a in state.actions]
        else:
            # Low ratio = keep original forms
            state.use_gerunds = False
        
        # Accumulate quaternion
        state.accumulated_q = state.accumulated_q * q.normalize()
        
        return state
    
    def _to_gerund(self, verb: str) -> str:
        """Convert verb to gerund."""
        verb = verb.lower().strip()
        if verb in self.verb_map:
            return self.verb_map[verb]
        if verb.endswith('ing'):
            return verb
        elif verb.endswith('e') and not verb.endswith('ee'):
            return verb[:-1] + 'ing'
        elif verb.endswith('s') and not verb.endswith('ss'):
            base = verb[:-1]
            if base.endswith('e') and not base.endswith('ee'):
                return base[:-1] + 'ing'
            return base + 'ing'
        else:
            return verb + 'ing'


class StructureGear(Gear):
    """
    Determines sentence structure.
    
    Quaternion encoding:
    - w: formality (high = "X is a Y", low = "It seems X is a Y")
    - x: connector type (high = "who", low = "that involves")
    - y: target connector (high = "particularly", low = "relating to")
    - z: verbosity (high = more elaborate)
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("StructureGear", ratio)
    
    def mesh(self, state: GearState) -> GearState:
        """Determine structure through this gear."""
        # Use accumulated quaternion to make decisions
        q = state.accumulated_q.normalize()
        
        # Apply gear ratio to decisions
        threshold = 0.5 * self.ratio
        
        # Prefix decision (w component)
        state.use_prefix = q.w < threshold
        
        # Connector decision (based on gerund usage and role)
        if state.use_gerunds:
            state.connector = "that involves"
        elif state.role in ['detective', 'doctor', 'character']:
            # Person-like roles use "who"
            state.connector = "who"
        else:
            # Abstract concepts use "that"
            state.connector = "that"
        
        # Target connector (y component)
        state.target_connector = "particularly" if q.y > 0 else "relating to"
        
        return state


class OutputGear(Gear):
    """
    Assembles final output from state.
    
    This is the final gear that produces the output string.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("OutputGear", ratio)
    
    def mesh(self, state: GearState) -> str:
        """Assemble output from state."""
        # Build prefix
        if state.use_prefix:
            prefix = f"It seems that {state.entity} is"
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
        
        # Assemble
        if action_str and target_str:
            return f"{prefix} {article} {state.role} {state.connector} {action_str}, {state.target_connector} {target_str}."
        elif action_str:
            return f"{prefix} {article} {state.role} {state.connector} {action_str}."
        elif target_str:
            return f"{prefix} {article} {state.role} {state.target_connector} {target_str}."
        else:
            return f"{prefix} {article} {state.role}."


class GearChain:
    """
    A chain of gears that transforms truth to signal.
    
    Each gear meshes with the next, passing state through the chain.
    """
    
    def __init__(self):
        self.gears: List[Gear] = []
    
    def add_gear(self, gear: Gear) -> 'GearChain':
        """Add a gear to the chain."""
        self.gears.append(gear)
        return self
    
    def run(self, initial_state: GearState) -> str:
        """Run state through the gear chain."""
        state = initial_state
        
        for gear in self.gears[:-1]:  # All but output gear
            state = gear.mesh(state)
        
        # Output gear returns string
        if self.gears:
            return self.gears[-1].mesh(state)
        
        return ""


class GearChainProjector:
    """
    Projects using a chain of gears.
    
    No patterns. No templates. Just gears.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal corpus for direct matches
        self.signal_frames = {}
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
        
        # Build gear chain
        self.chain = self._build_chain()
        
        print(f"Gear chain: {' → '.join(g.name for g in self.chain.gears)}")
    
    def _build_chain(self, role_ratio: float = 1.0, action_ratio: float = 1.0, 
                     structure_ratio: float = 1.0) -> GearChain:
        """Build the gear chain with specified ratios."""
        chain = GearChain()
        chain.add_gear(RoleGear(ratio=role_ratio))
        chain.add_gear(ActionGear(ratio=action_ratio))
        chain.add_gear(StructureGear(ratio=structure_ratio))
        chain.add_gear(OutputGear())
        return chain
    
    def project(self, concept: str, role_ratio: float = 1.0, 
                action_ratio: float = 1.0, structure_ratio: float = 1.0) -> str:
        """
        Project truth to signal using gear chain.
        
        Gear ratios control each transformation:
        - role_ratio: How much to transform roles
        - action_ratio: How much to transform actions (0 = base, 1 = gerund)
        - structure_ratio: How much to apply signal structure
        """
        concept_lower = concept.lower()
        
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Parse truth into initial state
        state = self._parse_to_state(truth, concept)
        
        # Rebuild chain with specified ratios
        self.chain = self._build_chain(role_ratio, action_ratio, structure_ratio)
        
        # Run through gear chain
        return self.chain.run(state)
    
    def _parse_to_state(self, truth: str, concept: str) -> GearState:
        """Parse truth into initial gear state."""
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


def demo():
    """Demo the gear chain projector."""
    print("=" * 70)
    print("GEAR CHAIN PROJECTION")
    print("Truth → [Role] → [Action] → [Structure] → [Output] → Signal")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    projector = GearChainProjector(truth_path, signal_path)
    
    # Find test concepts
    test_concepts = []
    for concept in projector.truth_qa.knowledge.concepts:
        if concept not in projector.signal_frames:
            c = projector.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 6:
            break
    
    print("\n" + "=" * 70)
    print("Testing with default ratios (all 1.0):")
    print("=" * 70)
    
    for concept in test_concepts:
        truth = projector.truth_qa.ask(f"What is {concept}?")
        result = projector.project(concept)
        
        print(f"\n{concept.upper()}")
        print(f"  TRUTH:  {truth}")
        print(f"  OUTPUT: {result}")
    
    print("\n" + "=" * 70)
    print("Testing with different gear ratios:")
    print("=" * 70)
    
    concept = test_concepts[0] if test_concepts else "evolution"
    truth = projector.truth_qa.ask(f"What is {concept}?")
    print(f"\n{concept.upper()}: {truth}")
    
    for action_ratio in [0.0, 0.5, 1.0]:
        result = projector.project(concept, action_ratio=action_ratio)
        print(f"  action_ratio={action_ratio}: {result}")


if __name__ == "__main__":
    demo()
