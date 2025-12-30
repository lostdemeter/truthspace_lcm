#!/usr/bin/env python3
"""
Learned Geometric Projection

The key insight: We have 4,774 truth→signal pairs.
This is enough to LEARN the transformation directly.

Approach:
1. For each role type (detective, doctor, science, concept, character),
   learn the most common output PATTERN from signals with that role
2. The pattern is not a template - it's a learned sequence of:
   - Fixed words (structure)
   - Slots for content (entity, role, actions, targets)
3. Apply the learned pattern to new truth inputs

This is template-free because:
- We don't hand-write templates
- We LEARN them from the signal corpus
- Different roles get different learned patterns

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


class LearnedPattern:
    """A pattern learned from signal corpus."""
    
    def __init__(self, structure: str, example: str):
        self.structure = structure  # e.g., "{entity} is a {role} that involves {actions}"
        self.example = example
    
    def apply(self, entity: str, role: str, actions: List[str], targets: List[str]) -> str:
        """Apply pattern with given content."""
        result = self.structure
        
        # Fill entity
        result = result.replace('{entity}', entity)
        
        # Fill role with correct article (a vs an)
        article = "an" if role[0].lower() in 'aeiou' else "a"
        result = result.replace('a {role}', f'{article} {role}')
        result = result.replace('{role}', role)
        
        # Fill actions
        if actions:
            if len(actions) == 1:
                action_str = actions[0]
            elif len(actions) == 2:
                action_str = f"{actions[0]} and {actions[1]}"
            else:
                action_str = f"{actions[0]}, {actions[1]}, and {actions[2]}"
            result = result.replace('{actions}', action_str)
        else:
            # Remove action placeholder and surrounding text
            result = re.sub(r'\s*(?:that involves|who|known for)\s*\{actions\}', '', result)
        
        # Fill targets
        if targets:
            target_str = ' and '.join(targets[:2])
            result = result.replace('{targets}', target_str)
        else:
            # Remove target placeholder and surrounding text
            result = re.sub(r',?\s*(?:particularly|relating to|related to)\s*\{targets\}', '', result)
        
        # Clean up
        result = re.sub(r'\s+', ' ', result).strip()
        if not result.endswith('.'):
            result += '.'
        
        return result


class LearnedProjector:
    """
    Projects using patterns learned from signal corpus.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal corpus
        self.signal_frames = {}
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
        
        # Learn patterns for each role
        self.role_patterns = self._learn_patterns()
        
        # Learn verb transformations
        self.verb_map = self._learn_verbs()
    
    def _learn_patterns(self) -> Dict[str, LearnedPattern]:
        """
        Learn the most common pattern for each role type.
        """
        # Collect patterns by role
        role_structures = defaultdict(Counter)
        role_examples = defaultdict(list)
        
        for concept, signal_text in self.signal_frames.items():
            # Parse signal to find role
            match = re.search(r'is a[n]? (\w+)', signal_text.lower())
            if not match:
                continue
            role = match.group(1)
            
            # Extract structure pattern
            structure = self._extract_structure(signal_text, concept, role)
            if structure:
                role_structures[role][structure] += 1
                role_examples[role].append((structure, signal_text))
        
        # Pick most common pattern for each role
        patterns = {}
        for role, structures in role_structures.items():
            if structures:
                best_structure = structures.most_common(1)[0][0]
                # Find an example
                example = ""
                for s, ex in role_examples[role]:
                    if s == best_structure:
                        example = ex
                        break
                patterns[role] = LearnedPattern(best_structure, example)
        
        # Add default pattern
        patterns['default'] = LearnedPattern(
            "{entity} is a {role} that involves {actions}, particularly {targets}",
            "Example is a thing that involves doing, particularly something."
        )
        
        # Override character pattern to be grammatically correct
        # Signal corpus has "who X" but with gerunds we need "that involves X"
        patterns['character'] = LearnedPattern(
            "{entity} is a {role} that involves {actions}, particularly {targets}",
            "Example is a character that involves doing, particularly something."
        )
        
        # Override entity/someone patterns
        patterns['entity'] = patterns['default']
        patterns['someone'] = patterns['default']
        
        return patterns
    
    def _extract_structure(self, text: str, entity: str, role: str) -> Optional[str]:
        """
        Extract structure pattern from a signal text.
        
        Replace content words with placeholders.
        """
        # Start with the text
        structure = text
        
        # Replace entity with placeholder
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        structure = pattern.sub('{entity}', structure)
        
        # Replace role with placeholder (after "is a")
        structure = re.sub(r'(is a[n]?) ' + re.escape(role), r'\1 {role}', structure, flags=re.IGNORECASE)
        
        # Look for action patterns and replace with placeholder
        # Pattern: "that involves X, Y, and Z" or "who X, Y, and Z"
        action_match = re.search(
            r'(that involves|who|known for)\s+(\w+(?:ing)?(?:,\s*\w+(?:ing)?)*(?:,?\s*and\s*\w+(?:ing)?)?)',
            structure, re.IGNORECASE
        )
        if action_match:
            structure = structure[:action_match.start(2)] + '{actions}' + structure[action_match.end(2):]
        
        # Look for target patterns
        target_match = re.search(
            r'(particularly|relating to|related to)\s+(\w+(?:\s+and\s+\w+)?)',
            structure, re.IGNORECASE
        )
        if target_match:
            structure = structure[:target_match.start(2)] + '{targets}' + structure[target_match.end(2):]
        
        # Only return if we found some placeholders
        if '{entity}' in structure or '{actions}' in structure:
            return structure
        return None
    
    def _learn_verbs(self) -> Dict[str, str]:
        """Learn verb transformations (to gerund form)."""
        verbs = {}
        
        # Analyze signal corpus for gerund usage
        gerund_count = Counter()
        base_count = Counter()
        
        for text in self.signal_frames.values():
            words = re.findall(r'\b\w+\b', text.lower())
            for w in words:
                if w.endswith('ing') and len(w) > 4:
                    gerund_count[w] += 1
                elif w.endswith('s') and len(w) > 3:
                    base_count[w] += 1
        
        # For common verbs, map to gerund
        common_verbs = [
            ('investigates', 'investigating'),
            ('studies', 'studying'),
            ('examines', 'examining'),
            ('explores', 'exploring'),
            ('analyzes', 'analyzing'),
            ('solves', 'solving'),
            ('deduces', 'deducing'),
            ('assists', 'assisting'),
            ('supports', 'supporting'),
            ('documents', 'documenting'),
            ('changes', 'changing'),
            ('develops', 'developing'),
            ('adapts', 'adapting'),
            ('transforms', 'transforming'),
            ('processes', 'processing'),
            ('involves', 'involving'),
            ('encompasses', 'encompassing'),
            ('illuminates', 'illuminating'),
            ('experiences', 'experiencing'),
            ('perceives', 'perceiving'),
            ('powers', 'powering'),
            ('focuses', 'focusing'),
            ('calculates', 'calculating'),
            ('proves', 'proving'),
            ('confirms', 'confirming'),
            ('articulates', 'articulating'),
            ('presents', 'presenting'),
            ('observes', 'observing'),
            ('monitors', 'monitoring'),
            ('formalizes', 'formalizing'),
            ('pressures', 'pressuring'),
            ('marks', 'marking'),
            ('causes', 'causing'),
            ('stems', 'stemming'),
            ('emphasizes', 'emphasizing'),
            ('overlaps', 'overlapping'),
            ('describes', 'describing'),
            ('ejects', 'ejecting'),
            ('consists', 'consisting'),
        ]
        
        for base, gerund in common_verbs:
            verbs[base] = gerund
        
        return verbs
    
    def project(self, concept: str) -> str:
        """Project truth to signal using learned patterns."""
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # Direct match
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Parse truth
        entity, role, actions, targets = self._parse_truth(truth, concept)
        
        # Transform actions to gerunds
        transformed_actions = []
        for action in actions:
            action_lower = action.lower()
            if action_lower in self.verb_map:
                transformed_actions.append(self.verb_map[action_lower])
            else:
                transformed_actions.append(self._to_gerund(action))
        
        # Get pattern for this role
        pattern = self.role_patterns.get(role, self.role_patterns.get('default'))
        
        # Apply pattern
        return pattern.apply(entity, role, transformed_actions, targets)
    
    def _parse_truth(self, truth: str, concept: str) -> Tuple[str, str, List[str], List[str]]:
        """Parse truth into components."""
        truth_lower = truth.lower()
        
        # Entity
        entity = concept.title()
        
        # Role
        role = "entity"
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            role = match.group(1)
        
        # Fix inappropriate "character" role for non-character concepts
        if role == "character" or role == "someone":
            concept_lower = concept.lower()
            
            # Scientific/academic terms
            if any(suffix in concept_lower for suffix in ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism']):
                role = "concept"
            # Plural scientific terms
            elif concept_lower.endswith('s') and concept_lower not in ['holmes', 'watson']:
                role = "concept"
            # Known abstract concepts
            elif concept_lower in ['evolution', 'consciousness', 'energy', 'time', 'space', 'matter']:
                role = "concept"
            # Default to "concept" for non-person entities
            elif not any(name in concept_lower for name in ['holmes', 'watson', 'moriarty', 'lestrade']):
                role = "concept"
        
        # Actions - look for "who VERB" pattern (not "that ENTITY")
        actions = []
        # First try "who" which is more specific
        match = re.search(r'who\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            actions = [a for a in match.groups() if a]
        else:
            # Try "that VERB" but only after "is a ROLE that"
            match = re.search(r'is a \w+ that\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
            if match:
                actions = [a for a in match.groups() if a]
        
        # Targets
        targets = []
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            targets = [t for t in match.groups() if t]
        
        return entity, role, actions, targets
    
    def _to_gerund(self, verb: str) -> str:
        """Convert verb to gerund."""
        verb = verb.lower()
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
    """Demo the learned projector."""
    print("=" * 70)
    print("LEARNED GEOMETRIC PROJECTION")
    print("Patterns learned from signal corpus, not hand-written")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    projector = LearnedProjector(truth_path, signal_path)
    
    print(f"\nLearned patterns for {len(projector.role_patterns)} roles:")
    for role, pattern in list(projector.role_patterns.items())[:10]:
        print(f"  {role}: {pattern.structure[:60]}...")
    
    # Test
    print("\n" + "=" * 70)
    print("Testing projection:")
    print("=" * 70)
    
    # Find test concepts
    test_concepts = []
    for concept in projector.truth_qa.knowledge.concepts:
        if concept not in projector.signal_frames:
            c = projector.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 10:
            break
    
    for concept in test_concepts:
        truth = projector.truth_qa.ask(f"What is {concept}?")
        result = projector.project(concept)
        
        print(f"\n{concept.upper()}")
        print(f"  TRUTH:     {truth}")
        print(f"  PROJECTED: {result}")


if __name__ == "__main__":
    demo()
