#!/usr/bin/env python3
"""
Nearest Signal Projection

The simplest truly template-free approach:
1. Encode truth as a content vector (entity, role, actions, targets)
2. Find the NEAREST signal in content space
3. Substitute content words while keeping signal structure

This is pure geometric projection:
- Signal corpus defines a MANIFOLD in content space
- Projection = finding nearest point on manifold
- Output = that signal with content substituted

No templates. No rules. Just nearest neighbor + substitution.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


class ContentVector:
    """Represents the content of a sentence as a vector."""
    
    def __init__(self, entity: str, role: str, actions: List[str], targets: List[str]):
        self.entity = entity
        self.role = role
        self.actions = actions
        self.targets = targets
    
    def to_set(self) -> Set[str]:
        """Convert to set of content words for comparison."""
        words = {self.entity.lower(), self.role.lower()}
        words.update(a.lower() for a in self.actions)
        words.update(t.lower() for t in self.targets)
        return words
    
    def similarity(self, other: 'ContentVector') -> float:
        """Compute similarity to another content vector."""
        # Jaccard similarity on content words
        self_set = self.to_set()
        other_set = other.to_set()
        
        if not self_set or not other_set:
            return 0.0
        
        intersection = len(self_set & other_set)
        union = len(self_set | other_set)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Bonus for matching role (important for structure)
        role_bonus = 0.3 if self.role.lower() == other.role.lower() else 0.0
        
        # Bonus for matching action count (similar complexity)
        action_diff = abs(len(self.actions) - len(other.actions))
        action_bonus = 0.2 * max(0, 1 - action_diff / 3)
        
        return jaccard + role_bonus + action_bonus


class NearestSignalProjector:
    """
    Projects by finding nearest signal and substituting content.
    
    This is truly template-free:
    - We don't construct output from rules
    - We find the closest existing signal
    - We substitute content words
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load and index signal corpus
        self.signals = []  # List of (content_vector, signal_text, agent)
        
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                
            for frame in data.get('frames', []):
                agent = frame.get('agent', '').lower()
                text = frame.get('text', '')
                
                if agent and text:
                    # Parse signal into content vector
                    content = self._parse_text(text, agent)
                    self.signals.append((content, text, agent))
        
        print(f"Indexed {len(self.signals)} signals")
    
    def _parse_text(self, text: str, entity_hint: str = "") -> ContentVector:
        """Parse text into content vector."""
        text_lower = text.lower()
        
        # Entity
        entity = entity_hint if entity_hint else "it"
        
        # Role
        role = "entity"
        match = re.search(r'is a[n]? (\w+)', text_lower)
        if match:
            role = match.group(1)
        
        # Actions (verbs, especially gerunds)
        actions = []
        # Look for gerunds
        gerunds = re.findall(r'\b(\w+ing)\b', text_lower)
        actions.extend(g for g in gerunds if len(g) > 4 and g not in ['relating', 'involving', 'being'])
        
        # Look for verbs after "who" or "that"
        match = re.search(r'(?:who|that)\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', text_lower)
        if match:
            for v in match.groups():
                if v and v not in actions:
                    actions.append(v)
        
        # Targets (nouns after "particularly", "involving", "related to")
        targets = []
        match = re.search(r'(?:particularly|involving|related to|relating to)\s+(\w+)(?:\s+and\s+(\w+))?', text_lower)
        if match:
            targets = [t for t in match.groups() if t]
        
        return ContentVector(entity, role, actions[:3], targets[:2])
    
    def project(self, concept: str) -> str:
        """
        Project by finding nearest signal and substituting content.
        """
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # Check for direct match
        for content, signal_text, agent in self.signals:
            if agent == concept_lower:
                return signal_text
        
        # Parse truth into content vector
        truth_content = self._parse_text(truth, concept)
        
        # Find nearest signal
        best_signal = None
        best_similarity = -1
        best_content = None
        
        for signal_content, signal_text, agent in self.signals:
            sim = truth_content.similarity(signal_content)
            if sim > best_similarity:
                best_similarity = sim
                best_signal = signal_text
                best_content = signal_content
        
        if best_signal is None:
            return truth  # Fallback to truth if no signals
        
        # Substitute content words
        return self._substitute(best_signal, best_content, truth_content)
    
    def _substitute(self, signal: str, signal_content: ContentVector, 
                   truth_content: ContentVector) -> str:
        """
        Substitute truth content into signal structure.
        
        Replace:
        - signal entity → truth entity
        - signal role → truth role (if different)
        - signal actions → truth actions
        - signal targets → truth targets
        """
        result = signal
        
        # Substitute entity (case-insensitive)
        if signal_content.entity and truth_content.entity:
            # Replace entity name
            pattern = re.compile(re.escape(signal_content.entity), re.IGNORECASE)
            result = pattern.sub(truth_content.entity.title(), result)
        
        # Substitute role if different
        if signal_content.role != truth_content.role:
            pattern = re.compile(r'\b' + re.escape(signal_content.role) + r'\b', re.IGNORECASE)
            result = pattern.sub(truth_content.role, result)
        
        # Substitute actions
        if signal_content.actions and truth_content.actions:
            # Map signal actions to truth actions
            for i, sig_action in enumerate(signal_content.actions):
                if i < len(truth_content.actions):
                    truth_action = truth_content.actions[i]
                    # Convert to gerund if signal uses gerund
                    if sig_action.endswith('ing'):
                        truth_gerund = self._to_gerund(truth_action)
                        pattern = re.compile(r'\b' + re.escape(sig_action) + r'\b', re.IGNORECASE)
                        result = pattern.sub(truth_gerund, result)
                    else:
                        pattern = re.compile(r'\b' + re.escape(sig_action) + r'\b', re.IGNORECASE)
                        result = pattern.sub(truth_action, result)
        
        # Substitute targets
        if signal_content.targets and truth_content.targets:
            for i, sig_target in enumerate(signal_content.targets):
                if i < len(truth_content.targets):
                    pattern = re.compile(r'\b' + re.escape(sig_target) + r'\b', re.IGNORECASE)
                    result = pattern.sub(truth_content.targets[i], result)
        
        return result
    
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
    """Demo the nearest signal projector."""
    print("=" * 70)
    print("NEAREST SIGNAL PROJECTION")
    print("Find closest signal, substitute content")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    projector = NearestSignalProjector(truth_path, signal_path)
    
    # Test on concepts NOT in signal
    print("\n" + "=" * 70)
    print("Testing projection (concepts NOT in signal corpus):")
    print("=" * 70)
    
    test_concepts = []
    for concept in projector.truth_qa.knowledge.concepts:
        found = False
        for _, _, agent in projector.signals:
            if agent == concept:
                found = True
                break
        if not found:
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
