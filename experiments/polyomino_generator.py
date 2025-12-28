#!/usr/bin/env python3
"""
Polyomino Text Generator

Generate text by fitting concepts together like polyomino pieces.

The insight: Concepts that "fit" have OPPOSITE φ-directions.
- Entities: +1 (outward, φ^+n)
- Actions: -1 (inward, φ^-n)

Generation algorithm:
1. Start with a seed concept
2. Find concepts with OPPOSITE direction that fit
3. Build frames by fitting: entity (+) → action (-) → entity (+)
4. Project to natural language

This is text generation as PUZZLE SOLVING.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.symmetry_encoder import SymmetryEncoder

PHI = 1.618034


@dataclass
class Concept:
    """A concept with its φ-direction and co-occurrence patterns."""
    word: str
    direction: float  # +1 = entity, -1 = action, 0 = mixed
    role: str  # 'entity', 'action', 'mixed'
    connections: Set[str]  # Words this concept fits with


@dataclass 
class Frame:
    """A generated frame (actor-action-target)."""
    actor: str
    action: str
    target: Optional[str]
    fit_score: float  # How well the pieces fit together


class PolyominoGenerator:
    """
    Generate text by fitting concepts together like polyomino pieces.
    
    The key insight: valid frames have OPPOSITE φ-directions between
    adjacent concepts. This is the "fitting" constraint.
    """
    
    def __init__(self):
        self.encoder = SymmetryEncoder()
        self.concepts: Dict[str, Concept] = {}
        self.entities: List[str] = []  # Words with direction > 0
        self.actions: List[str] = []   # Words with direction < 0
        
        # Learned patterns
        self.actor_actions: Dict[str, Counter] = defaultdict(Counter)
        self.action_targets: Dict[str, Counter] = defaultdict(Counter)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _is_content_word(self, word: str) -> bool:
        function_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be',
                         'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                         'he', 'she', 'it', 'they', 'his', 'her', 'its',
                         'that', 'this', 'from', 'not', 'did', 'do', 'does',
                         'and', 'or', 'but', 'if', 'then', 'so', 'as',
                         'very', 'more', 'down', 'up', 'out', 'about'}
        return word not in function_words and len(word) > 2
    
    def learn_from_text(self, text: str):
        """Learn concepts and their φ-directions from text."""
        sentences = re.split(r'[.!?]+', text)
        
        # Track role counts for each word
        role_counts = defaultdict(lambda: {'actor': 0, 'action': 0, 'target': 0})
        
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            content = [t for t in tokens if self._is_content_word(t)]
            
            if len(content) >= 2:
                actor = content[0]
                action = content[1]
                target = content[2] if len(content) > 2 else None
                
                role_counts[actor]['actor'] += 1
                role_counts[action]['action'] += 1
                if target:
                    role_counts[target]['target'] += 1
                
                # Learn co-occurrence patterns
                self.actor_actions[actor][action] += 1
                if target:
                    self.action_targets[action][target] += 1
        
        # Compute φ-direction for each word
        for word, counts in role_counts.items():
            entity_count = counts['actor'] + counts['target']
            action_count = counts['action']
            total = entity_count + action_count
            
            if total > 0:
                direction = (entity_count - action_count) / total
            else:
                direction = 0.0
            
            # Determine role
            if direction > 0.3:
                role = 'entity'
                self.entities.append(word)
            elif direction < -0.3:
                role = 'action'
                self.actions.append(word)
            else:
                role = 'mixed'
            
            # Find connections (words that co-occur)
            connections = set()
            if word in self.actor_actions:
                connections.update(self.actor_actions[word].keys())
            if word in self.action_targets:
                connections.update(self.action_targets[word].keys())
            for actor, actions in self.actor_actions.items():
                if word in actions:
                    connections.add(actor)
            for action, targets in self.action_targets.items():
                if word in targets:
                    connections.add(action)
            
            self.concepts[word] = Concept(
                word=word,
                direction=direction,
                role=role,
                connections=connections,
            )
    
    def _fits(self, word1: str, word2: str) -> bool:
        """Check if two concepts fit together (opposite directions)."""
        if word1 not in self.concepts or word2 not in self.concepts:
            return False
        
        dir1 = self.concepts[word1].direction
        dir2 = self.concepts[word2].direction
        
        # Opposite directions = fitting
        return dir1 * dir2 < 0
    
    def _fit_score(self, word1: str, word2: str) -> float:
        """Compute how well two concepts fit together."""
        if word1 not in self.concepts or word2 not in self.concepts:
            return 0.0
        
        dir1 = self.concepts[word1].direction
        dir2 = self.concepts[word2].direction
        
        # Perfect fit: directions multiply to -1
        # Score = how close to -1
        product = dir1 * dir2
        if product >= 0:
            return 0.0  # Same direction = no fit
        
        return abs(product)  # Closer to -1 = better fit
    
    def _find_fitting_action(self, actor: str) -> Optional[str]:
        """Find an action that fits with the given actor."""
        if actor not in self.concepts:
            return random.choice(self.actions) if self.actions else None
        
        # Prefer actions we've seen with this actor
        if actor in self.actor_actions:
            candidates = list(self.actor_actions[actor].keys())
            if candidates:
                return random.choice(candidates)
        
        # Otherwise, find any fitting action
        fitting = [a for a in self.actions if self._fits(actor, a)]
        if fitting:
            return random.choice(fitting)
        
        return random.choice(self.actions) if self.actions else None
    
    def _find_fitting_target(self, action: str) -> Optional[str]:
        """Find a target that fits with the given action."""
        # Filter out non-entity targets
        bad_targets = {'tall', 'small', 'confused', 'scared', 'angrily', 'intently',
                      'gracefully', 'sweetly', 'wildly', 'slowly', 'quickly',
                      'carefully', 'methodically', 'proudly', 'completely',
                      'mysteriously', 'going', 'through', 'where', 'unusual',
                      'near', 'understand', 'elementary', 'immediately', 'love'}
        
        good_entities = [e for e in self.entities if e not in bad_targets]
        
        if action not in self.concepts:
            return random.choice(good_entities) if good_entities else None
        
        # Prefer targets we've seen with this action
        if action in self.action_targets:
            candidates = [t for t in self.action_targets[action].keys() if t not in bad_targets]
            if candidates:
                return random.choice(candidates)
        
        # Otherwise, find any fitting entity
        fitting = [e for e in good_entities if self._fits(action, e)]
        if fitting:
            return random.choice(fitting)
        
        return random.choice(good_entities) if good_entities else None
    
    def generate_frame(self, seed: Optional[str] = None) -> Frame:
        """
        Generate a frame by fitting pieces together.
        
        The polyomino constraint: each adjacent pair must have opposite directions.
        """
        # Start with seed or random entity
        if seed and seed in self.concepts:
            if self.concepts[seed].direction > 0:
                actor = seed
            else:
                # Seed is an action, find a fitting actor
                fitting_actors = [e for e in self.entities if self._fits(e, seed)]
                actor = random.choice(fitting_actors) if fitting_actors else random.choice(self.entities)
        else:
            actor = random.choice(self.entities) if self.entities else "someone"
        
        # Find fitting action
        action = self._find_fitting_action(actor)
        if not action:
            action = "acts"
        
        # Find fitting target
        target = self._find_fitting_target(action)
        
        # Compute overall fit score
        score1 = self._fit_score(actor, action)
        score2 = self._fit_score(action, target) if target else 0.5
        fit_score = (score1 + score2) / 2
        
        return Frame(
            actor=actor,
            action=action,
            target=target,
            fit_score=fit_score,
        )
    
    def frame_to_sentence(self, frame: Frame) -> str:
        """Convert a frame to a natural language sentence."""
        actor = frame.actor.title()
        action = frame.action
        
        # Handle verb form - convert to third person present
        def verb_third_person(v):
            # Already past tense
            if v.endswith('ed'):
                return v
            # Irregular
            irregulars = {'fell': 'falls', 'grew': 'grows', 'said': 'says', 
                         'wrote': 'writes', 'read': 'reads'}
            if v in irregulars:
                return irregulars[v]
            # Standard rules
            if v.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return v + 'es'
            elif v.endswith('y') and len(v) > 1 and v[-2] not in 'aeiou':
                return v[:-1] + 'ies'
            else:
                return v + 's'
        
        verb = verb_third_person(action)
        
        # Filter target - only use if it's a real entity
        target = frame.target
        if target:
            # Skip adverbs and non-entities
            if target.endswith('ly') or target in {'tall', 'small', 'confused', 'scared',
                                                    'angrily', 'intently', 'gracefully',
                                                    'sweetly', 'wildly', 'slowly', 'quickly',
                                                    'carefully', 'methodically', 'proudly',
                                                    'completely', 'mysteriously', 'going',
                                                    'through', 'where'}:
                target = None
        
        if target:
            # Check if target needs an article
            if target in {'evidence', 'room', 'journal', 'newspaper', 'garden',
                         'building', 'window', 'scene', 'tea', 'hole'}:
                return f"{actor} {verb} the {target}."
            else:
                return f"{actor} {verb} {target.title()}."
        else:
            return f"{actor} {verb}."
    
    def generate_sentence(self, seed: Optional[str] = None) -> Tuple[str, Frame]:
        """Generate a sentence by fitting polyomino pieces."""
        frame = self.generate_frame(seed)
        sentence = self.frame_to_sentence(frame)
        return sentence, frame
    
    def generate_paragraph(self, seed: Optional[str] = None, num_sentences: int = 3) -> str:
        """Generate a paragraph by chaining fitted frames."""
        sentences = []
        current_seed = seed
        
        for _ in range(num_sentences):
            sentence, frame = self.generate_sentence(current_seed)
            sentences.append(sentence)
            
            # Chain: use target as next seed (if entity) or actor
            if frame.target and frame.target in self.concepts:
                if self.concepts[frame.target].direction > 0:
                    current_seed = frame.target
                else:
                    current_seed = frame.actor
            else:
                current_seed = frame.actor
        
        return " ".join(sentences)


def run_experiment():
    """Test polyomino-based text generation."""
    print("=" * 70)
    print("POLYOMINO TEXT GENERATOR")
    print("=" * 70)
    print()
    print("Generating text by fitting concepts like polyomino pieces.")
    print("Constraint: Adjacent concepts must have OPPOSITE φ-directions.")
    print()
    
    # Training corpus
    corpus = """
    Holmes examined the evidence carefully. Watson watched from the doorway.
    The detective studied the footprints. He noticed something unusual.
    Holmes said to Watson that the case was elementary.
    Watson replied that he did not understand.
    The inspector arrived at the scene. Lestrade questioned the witnesses.
    Holmes observed the room methodically. He found a clue near the window.
    Watson wrote in his journal. The doctor recorded every detail.
    Holmes deduced the killer identity. He explained his reasoning.
    The criminal fled through the garden. Holmes pursued him quickly.
    Watson called for help. The police surrounded the building.
    Holmes captured the villain. Justice was served.
    Alice fell down the rabbit hole. She wondered where she was going.
    The Queen shouted angrily. Alice felt confused and scared.
    The Cheshire Cat smiled mysteriously. He disappeared slowly.
    Alice grew very tall. She shrank very small.
    The Mad Hatter laughed wildly. He poured more tea.
    Darcy looked at Elizabeth proudly. She ignored him completely.
    Elizabeth danced gracefully. Darcy watched her intently.
    Mr Bennet read his newspaper. Mrs Bennet worried about her daughters.
    Jane smiled sweetly. Bingley fell in love immediately.
    """
    
    # Learn from corpus
    generator = PolyominoGenerator()
    generator.learn_from_text(corpus)
    
    print(f"Learned {len(generator.concepts)} concepts")
    print(f"  Entities (φ^+n): {len(generator.entities)}")
    print(f"  Actions (φ^-n): {len(generator.actions)}")
    print()
    
    # Show concept directions
    print("Sample concept directions:")
    for word in ['holmes', 'watson', 'alice', 'examined', 'watched', 'smiled']:
        if word in generator.concepts:
            c = generator.concepts[word]
            dir_symbol = "+" if c.direction > 0 else "-" if c.direction < 0 else "○"
            print(f"  {word:12} dir={c.direction:+.2f} ({dir_symbol}) role={c.role}")
    print()
    
    # Generate frames
    print("=" * 70)
    print("GENERATED FRAMES (polyomino fitting)")
    print("=" * 70)
    print()
    
    for seed in ['holmes', 'watson', 'alice', 'darcy', None]:
        frame = generator.generate_frame(seed)
        fit_symbol = "✓" if frame.fit_score > 0.5 else "○"
        print(f"Seed: {seed or 'random'}")
        print(f"  Frame: {frame.actor} → {frame.action} → {frame.target}")
        print(f"  Fit score: {frame.fit_score:.2f} {fit_symbol}")
        print()
    
    # Generate sentences
    print("=" * 70)
    print("GENERATED SENTENCES")
    print("=" * 70)
    print()
    
    for seed in ['holmes', 'alice', 'elizabeth', None]:
        sentence, frame = generator.generate_sentence(seed)
        print(f"Seed: {seed or 'random'}")
        print(f"  {sentence}")
        print()
    
    # Generate paragraphs
    print("=" * 70)
    print("GENERATED PARAGRAPHS")
    print("=" * 70)
    print()
    
    for seed in ['holmes', 'alice']:
        print(f"Seed: {seed}")
        paragraph = generator.generate_paragraph(seed, num_sentences=3)
        print(f"  {paragraph}")
        print()
    
    # Evaluate: are generated frames valid?
    print("=" * 70)
    print("EVALUATION: Frame Validity")
    print("=" * 70)
    print()
    
    valid_count = 0
    total = 20
    
    for _ in range(total):
        frame = generator.generate_frame()
        # A frame is valid if actor-action and action-target both fit
        actor_action_fits = generator._fits(frame.actor, frame.action)
        action_target_fits = frame.target is None or generator._fits(frame.action, frame.target)
        
        if actor_action_fits and action_target_fits:
            valid_count += 1
    
    print(f"Generated {total} random frames")
    print(f"Valid frames (all pieces fit): {valid_count}/{total} ({100*valid_count/total:.0f}%)")
    print()
    
    if valid_count / total > 0.7:
        print("✅ POLYOMINO GENERATION WORKS!")
        print("   The fitting constraint produces valid frames.")
    else:
        print("⚠️  Generation needs refinement.")
    
    return generator


if __name__ == "__main__":
    generator = run_experiment()
