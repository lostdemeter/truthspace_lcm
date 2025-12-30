#!/usr/bin/env python3
"""
Concept-Level Correction: Changing Entire Ideas

The problem with keyword injection:
- Adding "consults" to Holmes doesn't change WHO Holmes IS
- It just adds one more action to a list of actions
- The concept's identity remains unchanged

The solution: Concept-level correction that changes:
1. IDENTITY: What category/type is this concept?
2. ACTIONS: What are the PRIMARY actions (not just adding one more)?
3. RELATIONS: How does this concept relate to others?

Key insight: In concept space, an "idea" is defined by:
- Its φ-direction (agency)
- Its TOP actions (not all actions, the dominant ones)
- Its TOP targets (what it primarily acts upon)
- Its CATEGORY (what type of thing it is)

To change "Holmes is a teacher" → "Holmes is a consulting detective":
- We need to DEMOTE "teacher" relationships
- We need to PROMOTE "detective" relationships
- We need to establish "detective" as Holmes's CATEGORY
- We need to make "consult", "deduce", "solve" the PRIMARY actions

Author: Lesley Gushurst
License: GPLv3
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truthspace_lcm.core.geometric import GeometricKnowledge, HolographicGeometricQA


@dataclass
class ConceptIdentity:
    """Defines what a concept IS."""
    word: str
    category: str  # e.g., "detective", "doctor", "science"
    primary_actions: List[str]  # Top 3-5 characteristic actions
    primary_targets: List[str]  # Top 3-5 characteristic targets
    related_concepts: List[str]  # Strongly associated concepts
    
    def to_frames(self, weight: int = 5) -> List[Dict]:
        """
        Convert identity to weighted frames.
        
        Higher weight = more frames = stronger signal.
        """
        frames = []
        
        # Category frame: "X is a Y"
        for _ in range(weight * 2):  # Double weight for category
            frames.append({
                'initiator': self.word,
                'mediator': 'be',
                'receiver': self.category,
                'source': 'ConceptCorrection',
                'text': f"[IDENTITY] {self.word} is a {self.category}"
            })
        
        # Action frames: "X does Y"
        for action in self.primary_actions:
            for _ in range(weight):
                # Use category as receiver to reinforce identity
                frames.append({
                    'initiator': self.word,
                    'mediator': action,
                    'receiver': self.category,
                    'source': 'ConceptCorrection',
                    'text': f"[ACTION] {self.word} {action}s as a {self.category}"
                })
        
        # Target frames: "X acts on Y"
        for target in self.primary_targets:
            for _ in range(weight):
                frames.append({
                    'initiator': self.word,
                    'mediator': self.primary_actions[0] if self.primary_actions else 'involve',
                    'receiver': target,
                    'source': 'ConceptCorrection',
                    'text': f"[TARGET] {self.word} involves {target}"
                })
        
        # Relation frames: "X relates to Y"
        for related in self.related_concepts:
            for _ in range(weight):
                frames.append({
                    'initiator': self.word,
                    'mediator': 'associate',
                    'receiver': related,
                    'source': 'ConceptCorrection',
                    'text': f"[RELATION] {self.word} associates with {related}"
                })
        
        return frames


class ConceptCorrector:
    """
    Correct concepts at the identity level, not just keywords.
    
    Usage:
        corrector = ConceptCorrector()
        corrector.load_corpus('truthspace_lcm/corpus_self_improved.json')
        
        # Define Holmes's true identity
        corrector.define_identity(ConceptIdentity(
            word='holmes',
            category='detective',
            primary_actions=['deduce', 'investigate', 'solve', 'observe', 'consult'],
            primary_targets=['mystery', 'crime', 'case', 'evidence', 'clue'],
            related_concepts=['watson', 'london', 'baker street', 'moriarty']
        ))
        
        corrector.apply_corrections()
        corrector.save_corpus()
    """
    
    def __init__(self):
        self.corpus_data: Dict = {}
        self.corpus_path: Optional[str] = None
        self.qa: Optional[HolographicGeometricQA] = None
        self.identities: Dict[str, ConceptIdentity] = {}
        self.demotions: Dict[str, Set[str]] = {}  # concept -> words to demote
    
    def load_corpus(self, path: str):
        """Load corpus."""
        self.corpus_path = path
        with open(path) as f:
            self.corpus_data = json.load(f)
        
        self.qa = HolographicGeometricQA()
        self.qa.load_corpus(path)
        print(f"Loaded corpus: {len(self.corpus_data.get('frames', []))} frames")
    
    def define_identity(self, identity: ConceptIdentity):
        """Define or update a concept's identity."""
        self.identities[identity.word.lower()] = identity
        print(f"Defined identity for '{identity.word}': {identity.category}")
    
    def demote_association(self, concept: str, wrong_associations: List[str]):
        """
        Mark associations to demote (reduce weight of).
        
        This doesn't delete frames, but we can filter them or add counter-frames.
        """
        concept = concept.lower()
        if concept not in self.demotions:
            self.demotions[concept] = set()
        self.demotions[concept].update(w.lower() for w in wrong_associations)
        print(f"Will demote for '{concept}': {wrong_associations}")
    
    def apply_corrections(self, weight: int = 5):
        """
        Apply all defined identity corrections.
        
        Args:
            weight: How many frames to add per relationship (higher = stronger)
        """
        frames = self.corpus_data.get('frames', [])
        original_count = len(frames)
        
        # Add identity frames
        for word, identity in self.identities.items():
            new_frames = identity.to_frames(weight)
            frames.extend(new_frames)
            print(f"  Added {len(new_frames)} frames for '{word}'")
        
        # Handle demotions by adding counter-frames or filtering
        # For now, we rely on the weight of new frames to outweigh old ones
        # A more aggressive approach would filter out demoted frames
        
        self.corpus_data['frames'] = frames
        print(f"\nTotal frames: {original_count} → {len(frames)} (+{len(frames) - original_count})")
    
    def save_corpus(self, path: Optional[str] = None):
        """Save updated corpus."""
        path = path or self.corpus_path
        with open(path, 'w') as f:
            json.dump(self.corpus_data, f, indent=2)
        print(f"Saved to {path}")
    
    def test_answer(self, question: str) -> str:
        """Test current answer."""
        if not self.qa:
            return "[QA not loaded]"
        return self.qa.ask(question)
    
    def reload_and_test(self, questions: List[str]):
        """Reload QA and test questions."""
        if self.corpus_path:
            self.qa = HolographicGeometricQA()
            self.qa.load_corpus(self.corpus_path)
        
        print("\n--- Answers After Correction ---")
        for q in questions:
            answer = self.test_answer(q)
            print(f"  Q: {q}")
            print(f"  A: {answer[:100]}...")
            print()


# Predefined identities for common concepts
KNOWN_IDENTITIES = {
    'holmes': ConceptIdentity(
        word='holmes',
        category='detective',
        primary_actions=['deduce', 'investigate', 'solve', 'observe', 'consult'],
        primary_targets=['mystery', 'crime', 'case', 'evidence', 'clue'],
        related_concepts=['watson', 'london', 'moriarty', 'criminal']
    ),
    'watson': ConceptIdentity(
        word='watson',
        category='doctor',
        primary_actions=['assist', 'accompany', 'chronicle', 'observe', 'heal'],
        primary_targets=['holmes', 'patient', 'case', 'adventure', 'story'],
        related_concepts=['holmes', 'medicine', 'army', 'narrator']
    ),
    'physics': ConceptIdentity(
        word='physics',
        category='science',
        primary_actions=['study', 'explain', 'describe', 'predict', 'measure'],
        primary_targets=['matter', 'energy', 'force', 'motion', 'universe'],
        related_concepts=['mathematics', 'chemistry', 'quantum', 'relativity']
    ),
}


def demo():
    """Demonstrate concept-level correction."""
    import shutil
    
    corpus_path = 'truthspace_lcm/corpus_self_improved.json'
    backup_path = 'truthspace_lcm/corpus_self_improved.backup.json'
    
    # Backup
    print("Backing up corpus...")
    shutil.copy(corpus_path, backup_path)
    
    corrector = ConceptCorrector()
    corrector.load_corpus(corpus_path)
    
    print("\n" + "=" * 70)
    print("CONCEPT-LEVEL CORRECTION DEMO")
    print("=" * 70)
    
    # Test questions
    questions = [
        "Who is Holmes?",
        "What does Holmes do?",
        "Who is Watson?",
    ]
    
    print("\n--- Answers BEFORE Correction ---")
    for q in questions:
        answer = corrector.test_answer(q)
        print(f"  Q: {q}")
        print(f"  A: {answer[:100]}...")
        print()
    
    # Apply known identities
    print("\n--- Applying Concept Identities ---")
    corrector.define_identity(KNOWN_IDENTITIES['holmes'])
    corrector.define_identity(KNOWN_IDENTITIES['watson'])
    
    corrector.apply_corrections(weight=10)  # Strong weight
    corrector.save_corpus()
    
    # Reload and test
    corrector.reload_and_test(questions)
    
    print("=" * 70)
    print("To restore: python3 -c \"import shutil; shutil.copy('truthspace_lcm/corpus_self_improved.backup.json', 'truthspace_lcm/corpus_self_improved.json')\"")
    print("=" * 70)


if __name__ == '__main__':
    demo()
