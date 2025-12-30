#!/usr/bin/env python3
"""
Holographic Projection Polisher

Instead of filtering (destructive interference), we PROJECT:
- Signal sentences define a STRUCTURE BASIS
- Truth content is PROJECTED onto this basis
- The projection fills structure with content

This is like holography:
- Reference beam (signal) = the structure/pattern
- Object beam (truth) = the content
- Interference pattern = how to reconstruct

Mathematical model:
- Each signal sentence is a basis vector in "phrasing space"
- Truth content is a vector in "content space"  
- Projection maps content → phrasing

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


@dataclass
class StructureBasis:
    """A sentence structure extracted from signal corpus."""
    pattern: str           # e.g., "{entity} seems to be a {role} known for {action}"
    slots: List[str]       # ['entity', 'role', 'action']
    example: str           # Original sentence
    slot_positions: Dict[str, int]  # Where each slot appears (for phase)


@dataclass  
class ContentVector:
    """Content extracted from truth beam."""
    entity: str
    role: str
    actions: List[str]
    targets: List[str]
    raw_text: str


class HolographicProjector:
    """
    Projects truth content onto signal structure basis.
    
    The signal corpus defines "how to phrase things".
    The truth corpus defines "what to say".
    Projection combines them geometrically.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.knowledge = self.truth_qa.knowledge
        
        # Load and parse signal corpus into structure bases
        self.bases = self._extract_bases(signal_corpus_path)
        
        # Build role → basis mapping for fast lookup
        self.role_bases = defaultdict(list)
        for basis in self.bases:
            # Infer role from the example
            role = self._infer_role(basis.example)
            self.role_bases[role].append(basis)
    
    def _extract_bases(self, signal_path: str) -> List[StructureBasis]:
        """Extract structure bases from signal corpus."""
        bases = []
        
        if not os.path.exists(signal_path):
            return bases
        
        with open(signal_path, 'r') as f:
            data = json.load(f)
        
        for frame in data.get('frames', []):
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            
            if not text or not agent:
                continue
            
            # Extract pattern by replacing content with slots
            pattern, slots, positions = self._text_to_pattern(text, agent)
            
            if pattern and slots:
                bases.append(StructureBasis(
                    pattern=pattern,
                    slots=slots,
                    example=text,
                    slot_positions=positions,
                ))
        
        return bases
    
    def _text_to_pattern(self, text: str, agent: str) -> Tuple[str, List[str], Dict[str, int]]:
        """Convert a sentence to a pattern with slots."""
        pattern = text
        slots = []
        positions = {}
        
        # Replace agent with {entity}
        agent_pattern = re.compile(rf'\b{re.escape(agent)}\b', re.IGNORECASE)
        match = agent_pattern.search(pattern)
        if match:
            positions['entity'] = match.start()
            pattern = agent_pattern.sub('{entity}', pattern)
            slots.append('entity')
        
        # Look for role words and replace with {role}
        role_words = ['detective', 'doctor', 'scientist', 'science', 'field', 
                     'study', 'concept', 'character', 'figure', 'process',
                     'phenomenon', 'discipline', 'person', 'entity']
        
        for role in role_words:
            role_pattern = re.compile(rf'\ba {role}\b', re.IGNORECASE)
            match = role_pattern.search(pattern)
            if match:
                positions['role'] = match.start()
                pattern = role_pattern.sub('a {role}', pattern, count=1)
                if 'role' not in slots:
                    slots.append('role')
                break
        
        # Look for action verbs (gerunds often indicate actions in signal)
        action_patterns = [
            (r'\bknown for (\w+ing)', 'action'),
            (r'\bwho (\w+s)\b', 'action'),
            (r'\bthat (\w+s)\b', 'action'),
        ]
        
        for action_re, slot_name in action_patterns:
            match = re.search(action_re, pattern, re.IGNORECASE)
            if match:
                positions[slot_name] = match.start(1)
                # Don't replace - keep the structure
                if slot_name not in slots:
                    slots.append(slot_name)
                break
        
        return pattern, slots, positions
    
    def _infer_role(self, text: str) -> str:
        """Infer the role category from a sentence."""
        text_lower = text.lower()
        
        if 'detective' in text_lower:
            return 'detective'
        elif 'science' in text_lower or 'scientific' in text_lower:
            return 'science'
        elif 'study' in text_lower or 'studies' in text_lower:
            return 'study'
        elif 'character' in text_lower:
            return 'character'
        elif 'concept' in text_lower:
            return 'concept'
        else:
            return 'general'
    
    def _extract_content(self, concept: str) -> ContentVector:
        """Extract content vector from truth corpus."""
        concept_lower = concept.lower()
        
        # Get raw answer
        raw = self.truth_qa.ask(f"What is {concept}?")
        
        # Extract components from knowledge graph
        entity = concept
        role = 'concept'
        actions = []
        targets = []
        
        if concept_lower in self.knowledge.concepts:
            c = self.knowledge.concepts[concept_lower]
            
            # Get role
            role_words = {'detective', 'doctor', 'scientist', 'science', 'field',
                         'study', 'concept', 'character', 'process', 'phenomenon'}
            if c.targets:
                for target, count in c.targets.most_common(10):
                    if target in role_words and count >= 2:
                        role = target
                        break
            
            # Get actions (good verbs only)
            good_verbs = {
                'investigates', 'studies', 'examines', 'explores', 'analyzes',
                'solves', 'deduces', 'helps', 'assists', 'discovers',
                'explains', 'describes', 'creates', 'develops', 'transforms',
            }
            if c.actions:
                for action, _ in c.actions.most_common(10):
                    if action.lower() in good_verbs:
                        actions.append(action)
            
            # Get targets
            if c.targets:
                for target, _ in c.targets.most_common(10):
                    if target in self.knowledge.concepts and len(target) > 3:
                        tc = self.knowledge.concepts[target]
                        if tc.is_content_word and target not in role_words:
                            targets.append(target)
        
        return ContentVector(
            entity=entity,
            role=role,
            actions=actions[:3],
            targets=targets[:3],
            raw_text=raw,
        )
    
    def project(self, concept: str) -> str:
        """
        Project truth content onto signal structure.
        
        1. Extract content from truth beam
        2. Find best matching structure basis
        3. Project content onto basis (fill slots)
        """
        content = self._extract_content(concept)
        
        # Find best basis for this content's role
        role = content.role
        candidates = self.role_bases.get(role, [])
        
        if not candidates:
            # Fall back to general bases
            candidates = self.role_bases.get('general', [])
        
        if not candidates:
            # No basis found - use simple construction
            return self._simple_construct(content)
        
        # Score and select best basis
        best_basis = None
        best_score = -1
        
        for basis in candidates:
            score = self._score_basis(basis, content)
            if score > best_score:
                best_score = score
                best_basis = basis
        
        if best_basis:
            return self._project_onto_basis(content, best_basis)
        else:
            return self._simple_construct(content)
    
    def _score_basis(self, basis: StructureBasis, content: ContentVector) -> float:
        """Score how well a basis fits the content."""
        score = 0.0
        example_lower = basis.example.lower()
        
        # Bonus for having entity slot
        if 'entity' in basis.slots:
            score += 2.0
        
        # Bonus for having role slot
        if 'role' in basis.slots:
            score += 1.0
        
        # Bonus for action-related patterns
        if content.actions and 'action' in basis.slots:
            score += 1.0
        
        # Bonus for matching role in example
        if content.role in example_lower:
            score += 3.0
        
        # BONUS for bases that contain ACTION VERBS (not just "includes")
        good_action_words = ['studies', 'examines', 'investigates', 'explores', 'analyzes',
                            'describes', 'involves', 'encompasses', 'changes', 'develops',
                            'adapts', 'transforms', 'creates', 'provides', 'supports']
        action_count = sum(1 for w in good_action_words if w in example_lower)
        score += action_count * 1.5
        
        # PENALTY for weak/vague bases
        weak_patterns = ['that includes', 'seems to be about', 'plays a role']
        if any(p in example_lower for p in weak_patterns):
            score -= 3.0
        
        # PENALTY for overly long/complex bases (prefer simple structures)
        words = len(basis.example.split())
        if words > 25:
            score -= 2.0
        elif words > 15:
            score -= 0.5
        elif words < 12:
            score += 1.0  # Bonus for concise
        
        # PENALTY for bases with weird/bad content
        bad_words = ['interstellar', 'biochemistry', 'stubbornly', 'resists', 'dynamics',
                     'demolishes', 'poisonous', 'confines', 'collisions', 'gravity',
                     'neutrons', 'correlates', 'pressures', 'formalizing', 'groundbreaking']
        if any(w in example_lower for w in bad_words):
            score -= 10.0
        
        # PENALTY for nonsensical patterns
        nonsense_patterns = ['concept that demolishes', 'concept that stubbornly',
                            'concept that confines', 'concept that correlates',
                            'seems to be about', 'plays a role in shaping']
        if any(p in example_lower for p in nonsense_patterns):
            score -= 20.0
        
        # Bonus for clean patterns with "seems to be" or "is a"
        if 'seems to be a' in basis.pattern.lower():
            score += 1.5
        if 'is a {role}' in basis.pattern.lower():
            score += 1.0
        
        # BONUS for bases that match content's actions (VERY strong bonus)
        if content.actions:
            action_matches = 0
            for action in content.actions:
                if action.lower() in example_lower or self._to_gerund(action) in example_lower:
                    action_matches += 1
            score += action_matches * 5.0  # Very strong bonus for action matches
            
            # If NO actions match, this is probably a bad basis for this content
            if action_matches == 0:
                score -= 3.0
        
        # PENALTY for generic/vague bases
        generic_patterns = ['involves integration', 'involves permitting', 'oversees, coordinates',
                           'supports or maintains', 'light and study', 'economic systems',
                           'permitting and positioning', 'academic discipline that explores']
        if any(p in example_lower for p in generic_patterns):
            score -= 15.0
        
        return score
    
    def _project_onto_basis(self, content: ContentVector, basis: StructureBasis) -> str:
        """Project content onto a structure basis."""
        result = basis.pattern
        
        # Fill entity slot
        result = result.replace('{entity}', content.entity.title())
        
        # Fill role slot
        result = result.replace('{role}', content.role)
        
        # For action slots, we need to adapt the verbs
        # The basis might have "known for investigating" - we keep that structure
        # but ensure our content's actions are represented
        
        # If the basis has specific action words, try to substitute
        if content.actions:
            # Build action string from content
            gerunds = [self._to_gerund(a) for a in content.actions[:3]]
            if len(gerunds) == 1:
                action_str = gerunds[0]
            elif len(gerunds) == 2:
                action_str = f"{gerunds[0]} and {gerunds[1]}"
            else:
                action_str = f"{gerunds[0]}, {gerunds[1]}, and {gerunds[2]}"
            
            # Replace action patterns with our action string
            result = re.sub(
                r'\b(investigating|studying|examining|solving|exploring|deducing|analyzing)(,?\s*(and\s*)?(investigating|studying|examining|solving|exploring|deducing|analyzing))*\b',
                action_str,
                result,
                count=1,
                flags=re.IGNORECASE
            )
        
        # Add targets if the basis doesn't have them
        if content.targets and '{target}' not in result:
            # Check if we should append targets
            if not any(t in result.lower() for t in content.targets):
                target_str = ' and '.join(content.targets[:2])
                # Append to end if it makes sense
                if result.endswith('.'):
                    result = result[:-1] + f", particularly {target_str}."
                else:
                    result += f", particularly {target_str}."
        
        return result
    
    def _to_gerund(self, verb: str) -> str:
        """Convert verb to gerund form."""
        verb = verb.lower()
        if verb.endswith('ing'):
            return verb
        elif verb.endswith('e'):
            return verb[:-1] + 'ing'
        elif verb.endswith('s'):
            base = verb[:-1]
            if base.endswith('e'):
                return base[:-1] + 'ing'
            return base + 'ing'
        else:
            return verb + 'ing'
    
    def _simple_construct(self, content: ContentVector) -> str:
        """
        Simple fallback construction using clean templates.
        
        This is used when no good basis matches - produces reliable output.
        """
        name = content.entity.title()
        role = content.role
        actions = content.actions
        targets = content.targets
        
        # Clean up actions (convert to gerunds for natural phrasing)
        if actions:
            gerunds = [self._to_gerund(a) for a in actions[:3]]
            if len(gerunds) == 1:
                action_str = gerunds[0]
            elif len(gerunds) == 2:
                action_str = f"{gerunds[0]} and {gerunds[1]}"
            else:
                action_str = f"{gerunds[0]}, {gerunds[1]}, and {gerunds[2]}"
            
            # Role-specific templates
            if role == 'detective':
                if targets:
                    return f"{name} is a skilled {role} known for {action_str}, particularly in matters of {targets[0]}."
                return f"{name} is a skilled {role} known for {action_str}."
            elif role == 'doctor':
                if targets:
                    return f"{name} is a {role} who excels at {action_str}, especially regarding {targets[0]}."
                return f"{name} is a {role} who excels at {action_str}."
            elif role in ('science', 'study', 'field', 'discipline'):
                if targets:
                    return f"{name} is a {role} that involves {action_str}, focusing on {targets[0]} and related phenomena."
                return f"{name} is a {role} that involves {action_str}."
            elif role == 'concept':
                if targets:
                    return f"{name} is a {role} characterized by {action_str}, relating to {targets[0]}."
                return f"{name} is a {role} characterized by {action_str}."
            else:
                if targets:
                    return f"{name} is a {role} that involves {action_str}, particularly {targets[0]}."
                return f"{name} is a {role} that involves {action_str}."
        else:
            return f"{name} is a {role}."
    
    def compare(self, concept: str) -> Dict[str, str]:
        """Compare raw truth vs projected output."""
        content = self._extract_content(concept)
        projected = self.project(concept)
        
        return {
            'truth': content.raw_text,
            'projected': projected,
            'role': content.role,
            'actions': content.actions,
            'targets': content.targets,
        }


def demo():
    """Demo the holographic projector."""
    print("=" * 70)
    print("HOLOGRAPHIC PROJECTION POLISHER")
    print("Project truth content onto signal structure")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal.json"
    
    if not os.path.exists(signal_path):
        print("\nSignal corpus not found. Please create it first.")
        return
    
    projector = HolographicProjector(truth_path, signal_path)
    
    print(f"\nExtracted {len(projector.bases)} structure bases")
    print(f"Role categories: {list(projector.role_bases.keys())}")
    
    print("\n" + "=" * 70)
    print("PROJECTION RESULTS")
    print("=" * 70)
    
    for concept in ['holmes', 'physics', 'evolution', 'biology', 'consciousness', 'watson']:
        result = projector.compare(concept)
        print(f"\n{concept.upper()} (role: {result['role']})")
        print(f"  Actions: {result['actions']}")
        print(f"  Targets: {result['targets']}")
        print(f"  TRUTH:     {result['truth']}")
        print(f"  PROJECTED: {result['projected']}")


if __name__ == "__main__":
    demo()
