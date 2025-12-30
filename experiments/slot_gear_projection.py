#!/usr/bin/env python3
"""
Slot-Based Gear Projection

The gear metaphor applied to semantic slots:

Truth gear teeth:     [Entity] [Role] [Action1] [Action2] [Action3] [Target1] [Target2]
Signal gear teeth:    [Prefix] [Entity] [Copula] [Article] [Role] [Connector] [Actions] [TargetConn] [Targets]

Gear ratio = len(signal_teeth) / len(truth_teeth)

Meshing rules:
- Truth[Entity] → Signal[Entity]
- Truth[Role] → Signal[Role]  
- Truth[Actions] → Signal[Actions] (with transformation)
- Truth[Targets] → Signal[Targets]

The quaternion for each slot encodes:
- w: importance (how often this slot is filled in signal)
- x: position (where in sentence)
- y: transformation type (none, gerund, etc.)
- z: optional vs required

This is TRUE gear meshing - discrete teeth engaging discrete teeth.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import math
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


class SlotType(Enum):
    PREFIX = "prefix"
    ENTITY = "entity"
    COPULA = "copula"
    ARTICLE = "article"
    ROLE = "role"
    CONNECTOR = "connector"
    ACTIONS = "actions"
    TARGET_CONN = "target_conn"
    TARGETS = "targets"
    SUFFIX = "suffix"


@dataclass
class Slot:
    """A semantic slot with quaternion encoding."""
    slot_type: SlotType
    value: str
    q_w: float  # Importance
    q_x: float  # Position
    q_y: float  # Transform type
    q_z: float  # Optional flag
    
    def quaternion_tuple(self) -> Tuple[float, float, float, float]:
        return (self.q_w, self.q_x, self.q_y, self.q_z)


class SlotGear:
    """
    A gear represented as a sequence of slots (teeth).
    
    Each tooth is a semantic slot with a quaternion encoding.
    """
    
    def __init__(self, slots: List[Slot]):
        self.slots = slots
        self.tooth_count = len(slots)
    
    def get_slot(self, slot_type: SlotType) -> Optional[Slot]:
        """Get slot by type."""
        for s in self.slots:
            if s.slot_type == slot_type:
                return s
        return None
    
    def mesh_with(self, other: 'SlotGear', ratio: float = 1.0) -> 'SlotGear':
        """
        Mesh this gear with another gear.
        
        ratio > 1: other gear has more teeth (more verbose output)
        ratio < 1: other gear has fewer teeth (more concise output)
        ratio = 1: direct mapping
        """
        # For each slot in self, find corresponding slot in other
        meshed_slots = []
        
        for self_slot in self.slots:
            other_slot = other.get_slot(self_slot.slot_type)
            
            if other_slot:
                # Blend quaternions based on ratio
                blend = min(1.0, ratio)
                meshed_slots.append(Slot(
                    slot_type=self_slot.slot_type,
                    value=self_slot.value,  # Keep self's value
                    q_w=self_slot.q_w * (1 - blend) + other_slot.q_w * blend,
                    q_x=self_slot.q_x * (1 - blend) + other_slot.q_x * blend,
                    q_y=self_slot.q_y * (1 - blend) + other_slot.q_y * blend,
                    q_z=self_slot.q_z * (1 - blend) + other_slot.q_z * blend,
                ))
            else:
                # No corresponding slot - keep self's slot
                meshed_slots.append(self_slot)
        
        return SlotGear(meshed_slots)


class SlotGearProjector:
    """
    Projects using slot-based gear meshing.
    
    1. Parse truth into slots (truth gear)
    2. Learn signal slot patterns (signal gear template)
    3. Mesh truth gear with signal gear
    4. Generate output from meshed gear
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal corpus
        self.signal_frames = {}
        signal_texts = []
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
                        signal_texts.append(text)
        
        # Learn signal slot patterns
        self.signal_gear_template = self._learn_signal_gear(signal_texts)
        
        # Learn verb transformations
        self.verb_map = self._learn_verbs()
        
        print(f"Signal gear has {self.signal_gear_template.tooth_count} teeth")
        for slot in self.signal_gear_template.slots:
            print(f"  {slot.slot_type.value}: q=({slot.q_w:.2f}, {slot.q_x:.2f}, {slot.q_y:.2f}, {slot.q_z:.2f})")
    
    def _learn_signal_gear(self, texts: List[str]) -> SlotGear:
        """
        Learn the signal gear template from corpus.
        
        Analyze how often each slot appears and in what position.
        """
        slot_stats = {st: {'count': 0, 'positions': [], 'transforms': []} for st in SlotType}
        total = len(texts)
        
        for text in texts:
            text_lower = text.lower()
            words = text.split()
            n_words = len(words)
            
            # PREFIX: "It appears", "It seems", etc.
            if text_lower.startswith('it appears') or text_lower.startswith('it seems'):
                slot_stats[SlotType.PREFIX]['count'] += 1
                slot_stats[SlotType.PREFIX]['positions'].append(0.0)
            
            # ENTITY: Usually first capitalized word or after "that"
            match = re.search(r'^(?:It (?:appears|seems) that\s+)?(\w+)', text)
            if match:
                slot_stats[SlotType.ENTITY]['count'] += 1
                pos = text.find(match.group(1)) / len(text) if text else 0
                slot_stats[SlotType.ENTITY]['positions'].append(pos)
            
            # COPULA: "is", "seems to be"
            if ' is ' in text_lower:
                slot_stats[SlotType.COPULA]['count'] += 1
                pos = text_lower.find(' is ') / len(text)
                slot_stats[SlotType.COPULA]['positions'].append(pos)
            
            # ARTICLE: "a", "an"
            if ' a ' in text_lower or ' an ' in text_lower:
                slot_stats[SlotType.ARTICLE]['count'] += 1
            
            # ROLE: word after "is a"
            match = re.search(r'is a[n]? (\w+)', text_lower)
            if match:
                slot_stats[SlotType.ROLE]['count'] += 1
                pos = text_lower.find(match.group(1)) / len(text)
                slot_stats[SlotType.ROLE]['positions'].append(pos)
            
            # CONNECTOR: "that", "who", "that involves"
            if 'that involves' in text_lower:
                slot_stats[SlotType.CONNECTOR]['count'] += 1
                slot_stats[SlotType.CONNECTOR]['transforms'].append('that_involves')
            elif ' who ' in text_lower:
                slot_stats[SlotType.CONNECTOR]['count'] += 1
                slot_stats[SlotType.CONNECTOR]['transforms'].append('who')
            elif ' that ' in text_lower:
                slot_stats[SlotType.CONNECTOR]['count'] += 1
                slot_stats[SlotType.CONNECTOR]['transforms'].append('that')
            
            # ACTIONS: gerunds or verbs
            gerunds = re.findall(r'\b(\w+ing)\b', text_lower)
            if gerunds:
                slot_stats[SlotType.ACTIONS]['count'] += 1
                slot_stats[SlotType.ACTIONS]['transforms'].append('gerund')
            
            # TARGET_CONN: "particularly", "relating to"
            if 'particularly' in text_lower:
                slot_stats[SlotType.TARGET_CONN]['count'] += 1
                slot_stats[SlotType.TARGET_CONN]['transforms'].append('particularly')
            elif 'relating to' in text_lower:
                slot_stats[SlotType.TARGET_CONN]['count'] += 1
                slot_stats[SlotType.TARGET_CONN]['transforms'].append('relating_to')
        
        # Build gear template
        slots = []
        for st in SlotType:
            stats = slot_stats[st]
            
            # q_w: importance (frequency)
            q_w = stats['count'] / total if total > 0 else 0
            
            # q_x: average position
            q_x = sum(stats['positions']) / len(stats['positions']) if stats['positions'] else 0.5
            
            # q_y: transform type (1 = gerund, 0 = none)
            q_y = 1.0 if 'gerund' in stats['transforms'] else 0.0
            
            # q_z: optional (low frequency = optional)
            q_z = 0.0 if q_w > 0.5 else 1.0
            
            # Determine default value based on most common transform
            if stats['transforms']:
                transform_counts = Counter(stats['transforms'])
                default_value = transform_counts.most_common(1)[0][0]
            else:
                default_value = ""
            
            slots.append(Slot(st, default_value, q_w, q_x, q_y, q_z))
        
        return SlotGear(slots)
    
    def _learn_verbs(self) -> Dict[str, str]:
        """Learn verb transformations."""
        return {
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
    
    def _parse_truth_to_gear(self, truth: str, concept: str) -> SlotGear:
        """Parse truth into a slot gear."""
        truth_lower = truth.lower()
        slots = []
        
        # ENTITY
        slots.append(Slot(SlotType.ENTITY, concept.title(), 1.0, 0.0, 0.0, 0.0))
        
        # ROLE - detect and correct inappropriate roles
        role = "entity"
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            role = match.group(1)
        
        # Fix inappropriate "character" role for non-character concepts
        if role == "character" or role == "someone":
            # Detect what kind of concept this actually is
            concept_lower = concept.lower()
            
            # Scientific/academic terms
            if any(suffix in concept_lower for suffix in ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism']):
                role = "concept"
            # Plural scientific terms (neutrons, electrons, etc.)
            elif concept_lower.endswith('s') and concept_lower not in ['holmes', 'watson']:
                role = "concept"
            # Known abstract concepts
            elif concept_lower in ['evolution', 'consciousness', 'energy', 'time', 'space', 'matter']:
                role = "concept"
            # Default to "concept" for non-person entities
            elif not any(name in concept_lower for name in ['holmes', 'watson', 'moriarty', 'lestrade']):
                role = "concept"
        
        slots.append(Slot(SlotType.ROLE, role, 1.0, 0.2, 0.0, 0.0))
        
        # ACTIONS
        actions = []
        match = re.search(r'who\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            actions = [a for a in match.groups() if a]
        else:
            match = re.search(r'is a \w+ that\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
            if match:
                actions = [a for a in match.groups() if a]
        
        if actions:
            slots.append(Slot(SlotType.ACTIONS, ','.join(actions), 1.0, 0.4, 0.0, 0.0))
        
        # TARGETS
        targets = []
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            targets = [t for t in match.groups() if t]
        
        if targets:
            slots.append(Slot(SlotType.TARGETS, ','.join(targets), 1.0, 0.8, 0.0, 0.0))
        
        return SlotGear(slots)
    
    def project(self, concept: str, gear_ratio: float = 1.0) -> str:
        """
        Project truth to signal using gear meshing.
        
        gear_ratio: controls how much signal style influences output
        - 0.0 = pure truth style
        - 1.0 = full signal style
        - >1.0 = exaggerated signal style
        """
        concept_lower = concept.lower()
        
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Parse truth into gear
        truth_gear = self._parse_truth_to_gear(truth, concept)
        
        # Mesh with signal gear
        meshed_gear = truth_gear.mesh_with(self.signal_gear_template, gear_ratio)
        
        # Generate output from meshed gear
        return self._generate_from_gear(meshed_gear)
    
    def _generate_from_gear(self, gear: SlotGear) -> str:
        """Generate output from a meshed gear."""
        # Extract slot values
        entity_slot = gear.get_slot(SlotType.ENTITY)
        role_slot = gear.get_slot(SlotType.ROLE)
        actions_slot = gear.get_slot(SlotType.ACTIONS)
        targets_slot = gear.get_slot(SlotType.TARGETS)
        
        entity = entity_slot.value if entity_slot else "It"
        role = role_slot.value if role_slot else "entity"
        
        # Get signal gear preferences
        sig_prefix = self.signal_gear_template.get_slot(SlotType.PREFIX)
        sig_connector = self.signal_gear_template.get_slot(SlotType.CONNECTOR)
        sig_actions = self.signal_gear_template.get_slot(SlotType.ACTIONS)
        sig_target_conn = self.signal_gear_template.get_slot(SlotType.TARGET_CONN)
        
        # Decide prefix based on signal gear importance
        if sig_prefix and sig_prefix.q_w > 0.3:
            prefix = f"It seems that {entity} is"
        else:
            prefix = f"{entity} is"
        
        # Article
        article = "an" if role[0].lower() in 'aeiou' else "a"
        
        # Connector based on signal preference
        # When using gerunds, we need "that involves" for grammar
        use_gerunds = sig_actions and sig_actions.q_y > 0.5
        
        if sig_connector:
            if use_gerunds:
                connector = "that involves"  # Gerunds need "involves"
            elif sig_connector.value == 'who':
                connector = "who"
            else:
                connector = "that"
        else:
            connector = "that involves" if use_gerunds else "that"
        
        # Transform actions based on signal gear's q_y (gerund preference)
        actions = []
        if actions_slot:
            raw_actions = actions_slot.value.split(',')
            if sig_actions and sig_actions.q_y > 0.5:
                # Signal prefers gerunds
                actions = [self._to_gerund(a) for a in raw_actions]
            else:
                actions = raw_actions
        
        # Build action string
        if actions:
            if len(actions) == 1:
                action_str = actions[0]
            elif len(actions) == 2:
                action_str = f"{actions[0]} and {actions[1]}"
            else:
                action_str = f"{actions[0]}, {actions[1]}, and {actions[2]}"
        else:
            action_str = ""
        
        # Target connector
        if sig_target_conn and sig_target_conn.value == 'particularly':
            target_conn = "particularly"
        else:
            target_conn = "relating to"
        
        # Targets
        targets = targets_slot.value.split(',') if targets_slot else []
        target_str = ' and '.join(targets) if targets else ""
        
        # Construct output
        if action_str and target_str:
            return f"{prefix} {article} {role} {connector} {action_str}, {target_conn} {target_str}."
        elif action_str:
            return f"{prefix} {article} {role} {connector} {action_str}."
        elif target_str:
            return f"{prefix} {article} {role} {target_conn} {target_str}."
        else:
            return f"{prefix} {article} {role}."
    
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


def demo():
    """Demo the slot gear projector."""
    print("=" * 70)
    print("SLOT-BASED GEAR PROJECTION")
    print("Semantic slots as gear teeth")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    projector = SlotGearProjector(truth_path, signal_path)
    
    # Find test concepts
    test_concepts = []
    for concept in projector.truth_qa.knowledge.concepts:
        if concept not in projector.signal_frames:
            c = projector.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 8:
            break
    
    print("\n" + "=" * 70)
    print("Testing with different gear ratios:")
    print("=" * 70)
    
    for concept in test_concepts:
        truth = projector.truth_qa.ask(f"What is {concept}?")
        result = projector.project(concept, gear_ratio=1.0)
        
        print(f"\n{concept.upper()}")
        print(f"  TRUTH:     {truth}")
        print(f"  PROJECTED: {result}")


if __name__ == "__main__":
    demo()
