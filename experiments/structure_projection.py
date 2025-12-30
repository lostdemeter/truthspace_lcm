#!/usr/bin/env python3
"""
Structure-Only Geometric Projection

The key insight: Language has two components:
1. STRUCTURE words (a, is, that, who, and) - appear in >10% of sentences
2. CONTENT words (holmes, investigates, crimes) - appear in <1% of sentences

True geometric projection should:
- PRESERVE content words (they carry the meaning)
- TRANSFORM structure words (they carry the style)

This is scalable because:
- Structure words are finite (~50 words)
- Content words are infinite but don't need transformation
- The transformation is a simple mapping learned from signal corpus

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


class StructureProjector:
    """
    Projects by transforming structure while preserving content.
    
    The signal corpus teaches us:
    1. Which words are structure (high frequency)
    2. How structure words transform (truth → signal mappings)
    3. What structure patterns are preferred
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal corpus
        self.signal_frames = {}
        self.signal_texts = []
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
                        self.signal_texts.append(text)
        
        # Learn structure vs content
        self.structure_words = self._identify_structure_words()
        
        # Learn structure transformations
        self.structure_transforms = self._learn_structure_transforms()
        
        # Learn verb transformations (special case)
        self.verb_transforms = self._learn_verb_transforms()
    
    def _identify_structure_words(self, threshold: float = 0.05) -> Set[str]:
        """
        Identify structure words from signal corpus.
        
        Structure words appear in >threshold of all sentences.
        """
        word_doc_freq = Counter()
        total_docs = len(self.signal_texts)
        
        for text in self.signal_texts:
            words = set(re.findall(r'\b\w+\b', text.lower()))
            word_doc_freq.update(words)
        
        structure = set()
        for word, count in word_doc_freq.items():
            if count / total_docs > threshold:
                structure.add(word)
        
        return structure
    
    def _learn_structure_transforms(self) -> Dict[str, Dict[str, float]]:
        """
        Learn how structure words transform from truth to signal.
        
        For each truth structure word, what signal structure words replace it?
        """
        transforms = defaultdict(Counter)
        
        for concept, signal_text in self.signal_frames.items():
            truth_text = self.truth_qa.ask(f"What is {concept}?")
            if "don't know" in truth_text.lower():
                continue
            
            truth_struct = [w.lower() for w in re.findall(r'\b\w+\b', truth_text) 
                          if w.lower() in self.structure_words]
            signal_struct = [w.lower() for w in re.findall(r'\b\w+\b', signal_text)
                           if w.lower() in self.structure_words]
            
            # Align by position (simple approach)
            for i, tw in enumerate(truth_struct):
                if i < len(signal_struct):
                    transforms[tw][signal_struct[i]] += 1
        
        # Normalize to probabilities
        result = {}
        for tw, counter in transforms.items():
            total = sum(counter.values())
            result[tw] = {sw: c/total for sw, c in counter.items()}
        
        return result
    
    def _learn_verb_transforms(self) -> Dict[str, str]:
        """
        Learn verb transformations (conjugation changes).
        
        Signal corpus often uses gerunds (-ing) instead of 3rd person (-s).
        """
        # Common verb transformations
        transforms = {}
        
        # Analyze signal corpus for verb patterns
        verb_forms = defaultdict(Counter)
        
        for text in self.signal_texts:
            words = re.findall(r'\b\w+\b', text.lower())
            for w in words:
                if w.endswith('ing') and len(w) > 4:
                    # This is a gerund
                    base = w[:-3]
                    verb_forms[base]['ing'] += 1
                elif w.endswith('s') and len(w) > 3:
                    base = w[:-1]
                    verb_forms[base]['s'] += 1
        
        # For verbs that appear more often as gerunds, transform
        for base, forms in verb_forms.items():
            if forms['ing'] > forms['s'] * 2:
                # Gerund is preferred
                if base.endswith('e'):
                    transforms[base + 's'] = base[:-1] + 'ing' if base[-2] not in 'aeiou' else base + 'ing'
                else:
                    transforms[base + 's'] = base + 'ing'
        
        # Add common manual transforms
        transforms.update({
            'investigates': 'investigating',
            'studies': 'studying', 
            'examines': 'examining',
            'explores': 'exploring',
            'analyzes': 'analyzing',
            'solves': 'solving',
            'deduces': 'deducing',
            'assists': 'assisting',
            'supports': 'supporting',
            'documents': 'documenting',
            'changes': 'changing',
            'develops': 'developing',
            'adapts': 'adapting',
            'transforms': 'transforming',
            'processes': 'processing',
            'involves': 'involving',
            'relates': 'relating',
            'encompasses': 'encompassing',
            'illuminates': 'illuminating',
            'experiences': 'experiencing',
            'perceives': 'perceiving',
            'powers': 'powering',
            'focuses': 'focusing',
            'calculates': 'calculating',
            'proves': 'proving',
        })
        
        return transforms
    
    def project(self, concept: str) -> str:
        """
        Project truth to signal by transforming structure only.
        """
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # If we have direct signal, return it
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Transform structure while preserving content
        return self._transform(truth)
    
    def _transform(self, truth: str) -> str:
        """
        Transform truth text using learned patterns.
        
        Instead of word-by-word transformation, we:
        1. Parse truth into semantic slots (entity, role, actions, targets)
        2. Apply signal-learned patterns to each slot
        3. Reconstruct using signal-preferred structure
        """
        # Parse truth into components
        entity, role, actions, targets = self._parse_components(truth)
        
        # Transform actions to gerunds (signal prefers -ing forms)
        transformed_actions = []
        for action in actions:
            action_lower = action.lower()
            if action_lower in self.verb_transforms:
                transformed_actions.append(self.verb_transforms[action_lower])
            elif action_lower.endswith('s') and not action_lower.endswith('ss'):
                # Convert 3rd person to gerund
                base = action_lower[:-1]
                if base.endswith('e'):
                    transformed_actions.append(base[:-1] + 'ing')
                else:
                    transformed_actions.append(base + 'ing')
            else:
                transformed_actions.append(action)
        
        # Build output using signal-preferred structure
        # Signal corpus shows: "{Entity} is a {role} that involves {actions}, particularly {targets}."
        
        # Build action phrase
        if transformed_actions:
            if len(transformed_actions) == 1:
                action_phrase = transformed_actions[0]
            elif len(transformed_actions) == 2:
                action_phrase = f"{transformed_actions[0]} and {transformed_actions[1]}"
            else:
                action_phrase = f"{transformed_actions[0]}, {transformed_actions[1]}, and {transformed_actions[2]}"
        else:
            action_phrase = ""
        
        # Build target phrase
        if targets:
            target_phrase = ' and '.join(targets[:2])
        else:
            target_phrase = ""
        
        # Construct output
        if action_phrase and target_phrase:
            return f"{entity} is a {role} that involves {action_phrase}, particularly {target_phrase}."
        elif action_phrase:
            return f"{entity} is a {role} that involves {action_phrase}."
        elif target_phrase:
            return f"{entity} is a {role} related to {target_phrase}."
        else:
            return f"{entity} is a {role}."
    
    def _parse_components(self, truth: str) -> Tuple[str, str, List[str], List[str]]:
        """Parse truth into semantic components."""
        truth_lower = truth.lower()
        
        # Default values
        entity = "It"
        role = "entity"
        actions = []
        targets = []
        
        # Extract entity (first capitalized word or word before "is")
        match = re.match(r'^(?:It appears that\s+)?(\w+)', truth)
        if match:
            entity = match.group(1)
        
        # Extract role
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            role = match.group(1)
        
        # Extract actions (after "who" or "that")
        match = re.search(r'(?:who|that)\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            actions = [a for a in match.groups() if a]
        
        # Extract targets (after "relates to" or "involving")
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            targets = [t for t in match.groups() if t]
        
        return entity, role, actions, targets


def demo():
    """Demo the structure projector."""
    print("=" * 70)
    print("STRUCTURE-ONLY GEOMETRIC PROJECTION")
    print("Preserve content, transform structure")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    projector = StructureProjector(truth_path, signal_path)
    
    print(f"\nIdentified {len(projector.structure_words)} structure words")
    print(f"Top structure words: {sorted(projector.structure_words)[:20]}")
    print(f"\nLearned {len(projector.verb_transforms)} verb transformations")
    
    # Test on concepts IN signal corpus
    print("\n" + "=" * 70)
    print("Concepts WITH direct signal (should return signal):")
    print("=" * 70)
    
    for concept in ['holmes', 'watson', 'physics']:
        truth = projector.truth_qa.ask(f"What is {concept}?")
        result = projector.project(concept)
        print(f"\n{concept.upper()}")
        print(f"  TRUTH:     {truth}")
        print(f"  PROJECTED: {result}")
    
    # Test on concepts NOT in signal corpus
    print("\n" + "=" * 70)
    print("Concepts WITHOUT direct signal (pure projection):")
    print("=" * 70)
    
    test_concepts = []
    for concept in projector.truth_qa.knowledge.concepts:
        if concept not in projector.signal_frames:
            c = projector.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 8:
            break
    
    for concept in test_concepts:
        truth = projector.truth_qa.ask(f"What is {concept}?")
        result = projector.project(concept)
        print(f"\n{concept.upper()}")
        print(f"  TRUTH:     {truth}")
        print(f"  PROJECTED: {result}")


if __name__ == "__main__":
    demo()
