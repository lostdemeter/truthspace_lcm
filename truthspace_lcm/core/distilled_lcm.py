#!/usr/bin/env python3
"""
Distilled Large Concept Model

A pure concept model that operates without text storage.
Uses only geometric relationships for inference.

Key insight: Once we've extracted the structure from text,
we don't need the text for inference - only for citation.

This is the "distilled" form of the GeometricLCM:
- 10x smaller than frame-based corpus
- Same inference capabilities
- No text storage overhead

Author: Lesley Gushurst
License: GPLv3
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass
import math


PHI = 1.618034


@dataclass
class Concept:
    """A concept with geometric properties."""
    word: str
    phi_direction: float
    frequency: int
    initiator_count: int
    mediator_count: int
    receiver_count: int
    actions: List[Tuple[str, int]]  # [(action, count), ...]
    targets: List[Tuple[str, int]]  # [(target, count), ...]
    
    @property
    def is_initiator(self) -> bool:
        return self.phi_direction > 0.2
    
    @property
    def is_receiver(self) -> bool:
        return self.phi_direction < -0.2
    
    @property
    def is_mediator(self) -> bool:
        return abs(self.phi_direction) <= 0.2
    
    @property
    def top_action(self) -> Optional[str]:
        return self.actions[0][0] if self.actions else None
    
    @property
    def top_target(self) -> Optional[str]:
        return self.targets[0][0] if self.targets else None


class DistilledLCM:
    """
    Large Concept Model using distilled geometric knowledge.
    
    No text storage - pure concept relationships.
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.morphology: Dict[str, Set[str]] = {}
        self.relationships: Dict[str, List[Tuple[str, int]]] = {}
        self.metadata: Dict = {}
    
    def load(self, path: str):
        """Load a distilled concept model."""
        with open(path) as f:
            data = json.load(f)
        
        self.metadata = {
            'version': data.get('version'),
            'source_frames': data.get('source_frames'),
            'statistics': data.get('statistics', {}),
        }
        
        # Load concepts
        # Schema: [phi_dir, freq, i_count, m_count, r_count, actions, targets]
        for word, arr in data.get('concepts', {}).items():
            self.concepts[word] = Concept(
                word=word,
                phi_direction=arr[0],
                frequency=arr[1],
                initiator_count=arr[2],
                mediator_count=arr[3],
                receiver_count=arr[4],
                actions=[(a, c) for a, c in arr[5]],
                targets=[(t, c) for t, c in arr[6]],
            )
        
        # Load morphology
        for canonical, equivalents in data.get('morphology', {}).items():
            self.morphology[canonical] = set(equivalents)
        
        # Load relationships
        for word, edges in data.get('relationships', {}).items():
            self.relationships[word] = [(e, w) for e, w in edges]
        
        print(f"Loaded distilled LCM: {len(self.concepts)} concepts")
    
    def get_concept(self, word: str) -> Optional[Concept]:
        """Get a concept by word."""
        w = word.lower()
        if w in self.concepts:
            return self.concepts[w]
        
        # Check morphology equivalents
        for canonical, equivalents in self.morphology.items():
            if w in equivalents and canonical in self.concepts:
                return self.concepts[canonical]
        
        return None
    
    def describe(self, word: str) -> str:
        """Generate a description of a concept using only geometric properties."""
        concept = self.get_concept(word)
        if not concept:
            return f"Unknown concept: {word}"
        
        # Determine role
        if concept.is_initiator:
            role = "an active entity"
        elif concept.is_receiver:
            role = "a passive entity"
        else:
            role = "a mediating concept"
        
        # Build description from actions and targets
        parts = [f"{concept.word} is {role}"]
        
        if concept.actions:
            action_words = [a for a, _ in concept.actions[:3]]
            parts.append(f"that {', '.join(action_words)}")
        
        if concept.targets:
            target_words = [t for t, _ in concept.targets[:3]]
            parts.append(f"involving {', '.join(target_words)}")
        
        return ' '.join(parts) + '.'
    
    def find_related(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find concepts related to the given word."""
        w = word.lower()
        
        if w not in self.relationships:
            # Fallback: use actions and targets
            concept = self.get_concept(w)
            if not concept:
                return []
            
            related = {}
            for action, count in concept.actions:
                if action in self.concepts:
                    related[action] = related.get(action, 0) + count
            for target, count in concept.targets:
                if target in self.concepts:
                    related[target] = related.get(target, 0) + count
            
            sorted_related = sorted(related.items(), key=lambda x: -x[1])
            return sorted_related[:top_k]
        
        return self.relationships[w][:top_k]
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Compute similarity between two concepts using φ-direction and relationships.
        """
        c1 = self.get_concept(word1)
        c2 = self.get_concept(word2)
        
        if not c1 or not c2:
            return 0.0
        
        # φ-direction similarity (same agency = similar)
        phi_sim = 1.0 - abs(c1.phi_direction - c2.phi_direction) / 2.0
        
        # Action overlap
        actions1 = set(a for a, _ in c1.actions)
        actions2 = set(a for a, _ in c2.actions)
        action_overlap = len(actions1 & actions2) / max(len(actions1 | actions2), 1)
        
        # Target overlap
        targets1 = set(t for t, _ in c1.targets)
        targets2 = set(t for t, _ in c2.targets)
        target_overlap = len(targets1 & targets2) / max(len(targets1 | targets2), 1)
        
        # Combined similarity
        return 0.4 * phi_sim + 0.3 * action_overlap + 0.3 * target_overlap
    
    def complete_analogy(self, a: str, b: str, c: str) -> List[Tuple[str, float]]:
        """
        Complete analogy: a is to b as c is to ?
        
        Uses φ-direction difference as the transformation.
        """
        ca = self.get_concept(a)
        cb = self.get_concept(b)
        cc = self.get_concept(c)
        
        if not ca or not cb or not cc:
            return []
        
        # Compute transformation: what changes from a to b?
        delta_phi = cb.phi_direction - ca.phi_direction
        
        # Target φ-direction for answer
        target_phi = cc.phi_direction + delta_phi
        
        # Find concepts with similar φ-direction
        candidates = []
        for word, concept in self.concepts.items():
            if word in {a, b, c}:
                continue
            
            phi_diff = abs(concept.phi_direction - target_phi)
            if phi_diff < 0.5:  # Within range
                # Bonus for sharing relationships with c
                rel_bonus = self.similarity(word, c) * 0.5
                score = 1.0 - phi_diff + rel_bonus
                candidates.append((word, score))
        
        # Sort by score
        candidates.sort(key=lambda x: -x[1])
        return candidates[:5]
    
    def ask(self, question: str) -> str:
        """
        Answer a question using only concept relationships.
        
        This is a simplified version that works without text.
        """
        # Skip question words and function words
        skip_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which',
                      'is', 'are', 'was', 'were', 'do', 'does', 'did',
                      'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for'}
        
        words = question.lower().replace('?', '').split()
        
        # Find the main concept being asked about (skip question/function words)
        main_concept = None
        for word in words:
            if word in skip_words:
                continue
            if word in self.concepts:
                c = self.concepts[word]
                if c.frequency > 3:  # Skip rare words
                    main_concept = c
                    break
        
        if not main_concept:
            return "I don't have information about that concept."
        
        return self.describe(main_concept.word)
    
    def navigate(self, start: str, steps: int = 3) -> List[str]:
        """
        Navigate through concept space starting from a word.
        
        Returns a path of related concepts.
        """
        path = [start]
        current = start
        visited = {start}
        
        for _ in range(steps):
            related = self.find_related(current)
            # Find first unvisited related concept
            for word, _ in related:
                if word not in visited:
                    path.append(word)
                    visited.add(word)
                    current = word
                    break
            else:
                break  # No more unvisited concepts
        
        return path


def demo():
    """Demonstrate the distilled LCM."""
    lcm = DistilledLCM()
    lcm.load('truthspace_lcm/concepts_distilled.json')
    
    print("\n" + "=" * 60)
    print("DISTILLED LCM DEMO")
    print("=" * 60)
    
    # Describe concepts
    print("\n--- Concept Descriptions ---")
    for word in ['holmes', 'watson', 'physics', 'science', 'philosophy']:
        print(f"  {lcm.describe(word)}")
    
    # Find related concepts
    print("\n--- Related Concepts ---")
    for word in ['physics', 'holmes']:
        related = lcm.find_related(word)
        print(f"  {word}: {[w for w, _ in related]}")
    
    # Similarity
    print("\n--- Concept Similarity ---")
    pairs = [('physics', 'science'), ('physics', 'holmes'), ('watson', 'holmes')]
    for w1, w2 in pairs:
        sim = lcm.similarity(w1, w2)
        print(f"  {w1} <-> {w2}: {sim:.2f}")
    
    # Navigation
    print("\n--- Concept Navigation ---")
    for start in ['physics', 'holmes']:
        path = lcm.navigate(start, steps=4)
        print(f"  {' -> '.join(path)}")
    
    # Ask questions
    print("\n--- Questions ---")
    questions = [
        "What is physics?",
        "Who is Holmes?",
        "What is science?",
    ]
    for q in questions:
        print(f"  Q: {q}")
        print(f"  A: {lcm.ask(q)}")
    
    print("\n" + "=" * 60)
    print("All inference done with ZERO text storage!")
    print("=" * 60)


if __name__ == '__main__':
    demo()
