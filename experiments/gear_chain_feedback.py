#!/usr/bin/env python3
"""
Bidirectional Gear Chain with Feedback

Extends the gear chain to support:
1. Forward: Truth → Signal (projection)
2. Backward: Signal → Truth (correction propagation)
3. Corpus modification: Save corrections back to knowledge base

When you correct an output like:
  "Analysis is an entity..." → "Analysis is a concept..."

The system:
1. Parses the correction to identify what changed (role: entity → concept)
2. Propagates the change backward through the gear chain
3. Updates the underlying knowledge corpus
4. Saves the changes

This enables iterative refinement of the model through output correction.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

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
    
    def conjugate(self) -> 'Quaternion':
        """Inverse rotation."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self) -> float:
        return (self.w**2 + self.x**2 + self.y**2 + self.z**2) ** 0.5
    
    def normalize(self) -> 'Quaternion':
        n = self.norm()
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
    
    accumulated_q: Quaternion = field(default_factory=Quaternion)
    
    use_prefix: bool = False
    use_gerunds: bool = True
    connector: str = "that involves"
    target_connector: str = "particularly"
    
    # Track original values for diff
    original_role: str = ""
    original_actions: List[str] = field(default_factory=list)


@dataclass
class Correction:
    """Represents a correction to be propagated."""
    field: str  # 'role', 'actions', 'targets', etc.
    old_value: Any
    new_value: Any
    concept: str


class BidirectionalGear:
    """Base class for bidirectional gears."""
    
    def __init__(self, name: str, ratio: float = 1.0):
        self.name = name
        self.ratio = ratio
    
    def forward(self, state: GearState) -> GearState:
        """Forward transformation (truth → signal)."""
        raise NotImplementedError
    
    def backward(self, correction: Correction, state: GearState) -> Correction:
        """Backward transformation (signal → truth)."""
        raise NotImplementedError


class RoleGear(BidirectionalGear):
    """Bidirectional role transformation."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("RoleGear", ratio)
        
        # Forward mapping
        self.forward_map = {
            'character': 'concept',
            'someone': 'entity',
            'protagonist': 'concept',
        }
        
        # Reverse mapping (for backward propagation)
        self.backward_map = {
            'concept': 'concept',  # Keep as concept in corpus
            'entity': 'entity',
            'detective': 'detective',
            'doctor': 'doctor',
        }
        
        self.person_names = {'holmes', 'watson', 'moriarty', 'lestrade', 'mycroft'}
    
    def forward(self, state: GearState) -> GearState:
        """Transform role forward."""
        state.original_role = state.role
        entity_lower = state.entity.lower()
        
        # Check if person
        is_person = any(name in entity_lower for name in self.person_names)
        
        # Check if scientific/abstract
        scientific_suffixes = ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism', 'ons', 'ure', 'ance', 'ence']
        is_scientific = any(entity_lower.endswith(s) for s in scientific_suffixes)
        
        # Check if plural (likely not a character)
        is_plural = (entity_lower.endswith('s') and 
                    not entity_lower.endswith('ss') and 
                    len(entity_lower) > 3 and
                    entity_lower not in self.person_names)
        
        if state.role in ['character', 'someone', 'protagonist']:
            if is_person:
                pass  # Keep role
            elif is_scientific:
                state.role = 'concept'
            elif is_plural:
                state.role = 'concept'  # Plurals are usually concepts, not characters
            else:
                state.role = self.forward_map.get(state.role, 'concept')
        
        return state
    
    def backward(self, correction: Correction, state: GearState) -> Correction:
        """Propagate role correction backward."""
        if correction.field != 'role':
            return correction
        
        # The correction says the output role should be X
        # We need to update the corpus to reflect this
        new_role = correction.new_value
        
        # Map back to corpus role
        corpus_role = self.backward_map.get(new_role, new_role)
        
        return Correction(
            field='role',
            old_value=state.original_role,
            new_value=corpus_role,
            concept=correction.concept
        )


class ActionGear(BidirectionalGear):
    """Bidirectional action transformation."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("ActionGear", ratio)
        
        # Gerund mappings - comprehensive list
        self.to_gerund = {
            # Common verbs
            'investigates': 'investigating', 'studies': 'studying',
            'examines': 'examining', 'explores': 'exploring',
            'analyzes': 'analyzing', 'solves': 'solving',
            'processes': 'processing', 'involves': 'involving',
            'formalizes': 'formalizing', 'rigorizes': 'rigorizing',
            'confirms': 'confirming', 'articulates': 'articulating',
            'presents': 'presenting', 'observes': 'observing',
            'monitors': 'monitoring', 'monitores': 'monitoring',
            'facilitates': 'facilitating', 'facilitats': 'facilitating',
            'influences': 'influencing', 'emphasizes': 'emphasizing',
            'structures': 'structuring', 'illuminates': 'illuminating',
            'experiences': 'experiencing', 'perceives': 'perceiving',
            'pressures': 'pressuring', 'marks': 'marking',
            'changes': 'changing', 'develops': 'developing',
            'adapts': 'adapting', 'focuses': 'focusing',
            'collides': 'colliding', 'supports': 'supporting',
            'assists': 'assisting', 'documents': 'documenting',
            'deduces': 'deducing', 'lights': 'lighting',
            # Nouns that shouldn't be converted - map to better verbs
            'collisions': 'colliding', 'collision': 'colliding',
            'formalizations': 'formalizing', 'formalization': 'formalizing',
            'emphasis': 'emphasizing', 'proces': 'processing',
            'ongoes': 'ongoing', 'ongoing': 'ongoing',
            'michels': 'influencing',  # Likely typo for "influences"
        }
        
        # Words that should be skipped (not verbs)
        self.skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        # Reverse mapping
        self.from_gerund = {v: k for k, v in self.to_gerund.items()}
    
    def forward(self, state: GearState) -> GearState:
        """Transform actions forward."""
        state.original_actions = state.actions.copy()
        
        if self.ratio > 0.5:
            state.use_gerunds = True
            state.actions = [self._to_gerund(a) for a in state.actions]
        else:
            state.use_gerunds = False
        
        return state
    
    def backward(self, correction: Correction, state: GearState) -> Correction:
        """Propagate action correction backward."""
        if correction.field != 'actions':
            return correction
        
        # Convert gerunds back to base forms for corpus
        new_actions = correction.new_value
        corpus_actions = [self._from_gerund(a) for a in new_actions]
        
        return Correction(
            field='actions',
            old_value=state.original_actions,
            new_value=corpus_actions,
            concept=correction.concept
        )
    
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
    
    def _from_gerund(self, verb: str) -> str:
        verb = verb.lower().strip()
        if verb in self.from_gerund:
            return self.from_gerund[verb]
        # Try to reverse gerund
        if verb.endswith('ing'):
            base = verb[:-3]
            # Check common patterns
            if base + 'e' + 's' in self.to_gerund:
                return base + 'e' + 's'
            if base + 's' in self.to_gerund:
                return base + 's'
            return base + 's'  # Default to 3rd person
        return verb


class StructureGear(BidirectionalGear):
    """Bidirectional structure transformation."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("StructureGear", ratio)
    
    def forward(self, state: GearState) -> GearState:
        """Determine structure."""
        q = state.accumulated_q.normalize()
        threshold = 0.5 * self.ratio
        
        state.use_prefix = q.w < threshold
        
        if state.use_gerunds:
            state.connector = "that involves"
        elif state.role in ['detective', 'doctor', 'character']:
            state.connector = "who"
        else:
            state.connector = "that"
        
        state.target_connector = "particularly" if q.y > 0 else "relating to"
        
        return state
    
    def backward(self, correction: Correction, state: GearState) -> Correction:
        """Structure changes don't propagate to corpus."""
        return correction


class OutputGear(BidirectionalGear):
    """Bidirectional output assembly/parsing."""
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("OutputGear", ratio)
    
    def forward(self, state: GearState) -> str:
        """Assemble output."""
        if state.use_prefix:
            prefix = f"It seems that {state.entity} is"
        else:
            prefix = f"{state.entity} is"
        
        article = "an" if state.role[0].lower() in 'aeiou' else "a"
        
        if state.actions:
            if len(state.actions) == 1:
                action_str = state.actions[0]
            elif len(state.actions) == 2:
                action_str = f"{state.actions[0]} and {state.actions[1]}"
            else:
                action_str = f"{state.actions[0]}, {state.actions[1]}, and {state.actions[2]}"
        else:
            action_str = ""
        
        target_str = ' and '.join(state.targets[:2]) if state.targets else ""
        
        if action_str and target_str:
            return f"{prefix} {article} {state.role} {state.connector} {action_str}, {state.target_connector} {target_str}."
        elif action_str:
            return f"{prefix} {article} {state.role} {state.connector} {action_str}."
        elif target_str:
            return f"{prefix} {article} {state.role} {state.target_connector} {target_str}."
        else:
            return f"{prefix} {article} {state.role}."
    
    def backward(self, corrected_output: str, original_state: GearState) -> List[Correction]:
        """Parse corrected output to identify changes."""
        corrections = []
        
        # Parse the corrected output
        parsed = self._parse_output(corrected_output)
        
        # Compare with original state
        if parsed.get('role') and parsed['role'] != original_state.role:
            corrections.append(Correction(
                field='role',
                old_value=original_state.role,
                new_value=parsed['role'],
                concept=original_state.entity.lower()
            ))
        
        if parsed.get('actions') and parsed['actions'] != original_state.actions:
            corrections.append(Correction(
                field='actions',
                old_value=original_state.actions,
                new_value=parsed['actions'],
                concept=original_state.entity.lower()
            ))
        
        if parsed.get('targets') and parsed['targets'] != original_state.targets:
            corrections.append(Correction(
                field='targets',
                old_value=original_state.targets,
                new_value=parsed['targets'],
                concept=original_state.entity.lower()
            ))
        
        return corrections
    
    def _parse_output(self, text: str) -> Dict[str, Any]:
        """Parse output text into components."""
        result = {}
        text_lower = text.lower()
        
        # Extract role
        match = re.search(r'is a[n]? (\w+)', text_lower)
        if match:
            result['role'] = match.group(1)
        
        # Extract actions
        match = re.search(r'(?:that involves|who|that)\s+(.+?)(?:,\s*(?:particularly|relating to)|\.)', text_lower)
        if match:
            action_str = match.group(1)
            # Parse action list
            actions = re.split(r',\s*(?:and\s+)?|\s+and\s+', action_str)
            result['actions'] = [a.strip() for a in actions if a.strip()]
        
        # Extract targets
        match = re.search(r'(?:particularly|relating to)\s+(.+?)\.', text_lower)
        if match:
            target_str = match.group(1)
            targets = re.split(r'\s+and\s+', target_str)
            result['targets'] = [t.strip() for t in targets if t.strip()]
        
        return result


class FeedbackGearChain:
    """
    Bidirectional gear chain with feedback capability.
    
    Supports:
    - Forward projection (truth → signal)
    - Backward correction (signal → truth)
    - Corpus modification
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        self.truth_corpus_path = truth_corpus_path
        
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
        self.gears = [
            RoleGear(ratio=1.0),
            ActionGear(ratio=1.0),
            StructureGear(ratio=1.0),
            OutputGear(ratio=1.0),
        ]
        
        # Track last projection state for corrections
        self.last_state: Optional[GearState] = None
        self.last_concept: str = ""
        
        # Pending corrections
        self.pending_corrections: List[Correction] = []
        
        # Load saved corrections
        self.corrections = self._load_corrections()
        
        print(f"Feedback gear chain: {' → '.join(g.name for g in self.gears)}")
        if self.corrections:
            print(f"Loaded {len(self.corrections)} concept corrections")
    
    def _load_corrections(self) -> Dict[str, Dict]:
        """Load saved corrections from file."""
        corrections_path = self.truth_corpus_path.replace('.json', '_corrections.json')
        if os.path.exists(corrections_path):
            with open(corrections_path, 'r') as f:
                return json.load(f)
        return {}
    
    def project(self, concept: str) -> str:
        """Forward projection."""
        concept_lower = concept.lower()
        self.last_concept = concept
        
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # Check signal frames, but skip if they have wrong roles
        if concept_lower in self.signal_frames:
            signal = self.signal_frames[concept_lower]
            # Don't use signal frame if it has "character" for non-person concepts
            if 'is a character' in signal.lower() or 'is someone' in signal.lower():
                # Check if this should actually be a concept
                if self._should_be_concept(concept_lower):
                    pass  # Skip signal frame, use gear chain
                else:
                    return signal
            else:
                return signal
        
        # Parse truth into initial state
        state = self._parse_to_state(truth, concept)
        
        # Run through gears (except output)
        for gear in self.gears[:-1]:
            state = gear.forward(state)
        
        # Save state for potential correction
        self.last_state = state
        
        # Output gear returns string
        return self.gears[-1].forward(state)
    
    def correct(self, corrected_output: str) -> List[Correction]:
        """
        Apply a correction and propagate backward.
        
        Returns list of corrections that will be applied to corpus.
        """
        if self.last_state is None:
            print("No previous projection to correct.")
            return []
        
        # Parse correction using output gear
        output_gear = self.gears[-1]
        corrections = output_gear.backward(corrected_output, self.last_state)
        
        if not corrections:
            print("No changes detected in correction.")
            return []
        
        # Propagate each correction backward through gears
        propagated = []
        for correction in corrections:
            # Go backward through gears (skip output gear)
            for gear in reversed(self.gears[:-1]):
                correction = gear.backward(correction, self.last_state)
            propagated.append(correction)
        
        self.pending_corrections.extend(propagated)
        
        return propagated
    
    def apply_corrections(self, save: bool = False) -> Dict[str, Any]:
        """
        Apply pending corrections to the knowledge corpus.
        
        Returns summary of changes made.
        """
        if not self.pending_corrections:
            print("No pending corrections.")
            return {}
        
        changes = defaultdict(list)
        
        for correction in self.pending_corrections:
            concept = correction.concept
            field = correction.field
            new_value = correction.new_value
            
            # Find concept in knowledge base
            if concept in self.truth_qa.knowledge.concepts:
                c = self.truth_qa.knowledge.concepts[concept]
                
                if field == 'role':
                    # Update role in concept
                    old_role = getattr(c, 'role', None)
                    # The role is stored in the concept's relationships
                    # We need to update the frames that define this concept
                    changes[concept].append({
                        'field': 'role',
                        'old': correction.old_value,
                        'new': new_value
                    })
                    
                elif field == 'actions':
                    # Update actions - actions is a Counter, get list of keys
                    old_actions = list(c.actions.keys()) if c.actions else []
                    changes[concept].append({
                        'field': 'actions',
                        'old': old_actions,
                        'new': new_value
                    })
                
                elif field == 'targets':
                    # Update targets - targets is a Counter, get list of keys
                    old_targets = list(c.targets.keys()) if c.targets else []
                    changes[concept].append({
                        'field': 'targets',
                        'old': old_targets,
                        'new': new_value
                    })
        
        # Clear pending corrections
        self.pending_corrections = []
        
        # Save if requested
        if save and changes:
            self._save_corpus_changes(changes)
        
        return dict(changes)
    
    def _save_corpus_changes(self, changes: Dict[str, List], strength: int = 10):
        """
        Apply changes by adding reinforcement frames to the corpus.
        
        This follows the geometric reinforcement learning approach from design 073:
        - Don't modify existing frames
        - Add new frames that reinforce the corrections
        - Repetition (strength) ensures the correction has enough weight
        """
        frames_added = 0
        
        for concept, change_list in changes.items():
            entity = concept.title()
            
            for change in change_list:
                field = change['field']
                new_value = change['new']
                
                if field == 'role':
                    # Add frames that reinforce the new role
                    # Boost role counts in the concept
                    if concept in self.truth_qa.knowledge.concepts:
                        c = self.truth_qa.knowledge.concepts[concept]
                        boost = strength * 2
                        if new_value == 'concept':
                            c.mediator_count += boost
                        elif new_value in ['entity', 'protagonist', 'character']:
                            c.initiator_count += boost
                        print(f"  Boosted {concept} role to {new_value}")
                
                elif field == 'actions':
                    # Add frames with the new actions
                    actions = new_value if isinstance(new_value, list) else [new_value]
                    for action in actions:
                        # Clean up action
                        action = action.strip()
                        if not action or len(action) < 3:
                            continue
                        
                        # Normalize verb form
                        action = self._normalize_verb(action)
                        
                        # Skip nouns misidentified as verbs
                        skip_words = {'more', 'various', 'diverse', 'crucial', 'fundamental',
                                     'integral', 'aspect', 'role', 'concept', 'entity'}
                        if action.lower() in skip_words:
                            continue
                        
                        # Add reinforcement frames
                        for _ in range(strength):
                            frame_text = f"{entity} {action}."
                            self.truth_qa.knowledge.learn(frame_text, source="reinforcement")
                            frames_added += 1
                
                elif field == 'targets':
                    # Add frames with the new targets
                    targets = new_value if isinstance(new_value, list) else [new_value]
                    
                    # Get a common action for this entity
                    if concept in self.truth_qa.knowledge.concepts:
                        c = self.truth_qa.knowledge.concepts[concept]
                        if hasattr(c, 'actions') and c.actions:
                            action = c.actions.most_common(1)[0][0]
                        else:
                            action = "involves"
                    else:
                        action = "involves"
                    
                    for target in targets:
                        target = target.strip()
                        if not target or len(target) < 3:
                            continue
                        
                        for _ in range(strength):
                            frame_text = f"{entity} {action} {target}."
                            self.truth_qa.knowledge.learn(frame_text, source="reinforcement")
                            frames_added += 1
        
        # Save the updated corpus
        if frames_added > 0:
            self._save_corpus_to_file()
            print(f"Added {frames_added} reinforcement frames to corpus")
    
    def _normalize_verb(self, verb: str) -> str:
        """Normalize verb to base form."""
        verb = verb.lower().strip()
        
        # Remove common suffixes to get base form
        if verb.endswith('ves') and len(verb) > 4:
            return verb[:-1]  # solves -> solve
        elif verb.endswith('ies') and len(verb) > 4:
            return verb[:-3] + 'y'  # studies -> study
        elif verb.endswith('es') and len(verb) > 4:
            return verb[:-1]  # provides -> provide
        elif verb.endswith('s') and len(verb) > 3 and not verb.endswith('ss'):
            return verb[:-1]  # assists -> assist
        
        return verb
    
    def _save_corpus_to_file(self):
        """Save the modified corpus to file."""
        # Convert knowledge to JSON format
        data = {
            'frames': [
                {
                    'text': f.text,
                    'source': getattr(f, 'source', 'unknown'),
                    'agent': getattr(f, 'initiator', ''),
                }
                for f in self.truth_qa.knowledge.frames
            ]
        }
        
        # Backup existing file
        backup_path = self.truth_corpus_path + '.backup'
        if os.path.exists(self.truth_corpus_path):
            import shutil
            shutil.copy(self.truth_corpus_path, backup_path)
        
        # Save new corpus
        with open(self.truth_corpus_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Corpus saved ({len(data['frames'])} frames). Backup at: {backup_path}")
    
    def _should_be_concept(self, concept: str) -> bool:
        """Check if a concept should be labeled as 'concept' not 'character'."""
        concept_lower = concept.lower()
        
        # Person names should stay as character
        person_names = {'holmes', 'watson', 'moriarty', 'lestrade', 'mycroft', 'irene'}
        if concept_lower in person_names:
            return False
        
        # Abstract suffixes
        abstract_suffixes = ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism', 'ure', 'ance', 'ence']
        if any(concept_lower.endswith(s) for s in abstract_suffixes):
            return True
        
        # Plurals (likely not characters)
        if concept_lower.endswith('s') and not concept_lower.endswith('ss') and len(concept_lower) > 3:
            return True
        
        return False
    
    def _parse_to_state(self, truth: str, concept: str) -> GearState:
        """Parse truth into gear state."""
        truth_lower = truth.lower()
        
        state = GearState()
        state.entity = concept.title()
        
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            state.role = match.group(1)
        
        match = re.search(r'who\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            state.actions = [a for a in match.groups() if a]
        else:
            match = re.search(r'is a \w+ that\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
            if match:
                state.actions = [a for a in match.groups() if a]
        
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            state.targets = [t for t in match.groups() if t]
        
        return state
    
    def interactive_correct(self, concept: str):
        """Interactive correction session."""
        # Project
        output = self.project(concept)
        truth = self.truth_qa.ask(f"What is {concept}?")
        
        print(f"\nConcept: {concept.upper()}")
        print(f"Truth:   {truth}")
        print(f"Output:  {output}")
        print()
        
        # Get correction
        print("Enter corrected output (or press Enter to skip):")
        corrected = input("> ").strip()
        
        if not corrected:
            print("No correction.")
            return
        
        # Apply correction
        corrections = self.correct(corrected)
        
        if corrections:
            print("\nDetected corrections:")
            for c in corrections:
                print(f"  {c.field}: '{c.old_value}' → '{c.new_value}'")
            
            print("\nApply to corpus? (y/n)")
            if input("> ").strip().lower() == 'y':
                changes = self.apply_corrections(save=True)
                print(f"Applied {len(changes)} changes.")
            else:
                self.pending_corrections = []
                print("Corrections discarded.")


def demo():
    """Demo the feedback gear chain."""
    print("=" * 70)
    print("FEEDBACK GEAR CHAIN")
    print("Bidirectional projection with corpus correction")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    chain = FeedbackGearChain(truth_path, signal_path)
    
    # Test projection
    print("\n--- Forward Projection ---")
    for concept in ['analysis', 'neutrons', 'nomenclature']:
        truth = chain.truth_qa.ask(f"What is {concept}?")
        output = chain.project(concept)
        print(f"\n{concept.upper()}")
        print(f"  TRUTH:  {truth}")
        print(f"  OUTPUT: {output}")
    
    # Test correction
    print("\n" + "=" * 70)
    print("--- Correction Test ---")
    print("=" * 70)
    
    # Project analysis
    output = chain.project('analysis')
    print(f"\nOriginal output: {output}")
    
    # Apply correction
    corrected = "Analysis is a concept that involves involving, rigorizing, and formalizing, relating to involve and intricate."
    print(f"Corrected output: {corrected}")
    
    corrections = chain.correct(corrected)
    print(f"\nDetected corrections:")
    for c in corrections:
        print(f"  {c.field}: '{c.old_value}' → '{c.new_value}'")
    
    # Show what would be applied (don't actually save)
    print("\nChanges that would be applied to corpus:")
    changes = chain.apply_corrections(save=False)
    for concept, change_list in changes.items():
        for change in change_list:
            print(f"  {concept}.{change['field']}: {change['old']} → {change['new']}")


if __name__ == "__main__":
    demo()
