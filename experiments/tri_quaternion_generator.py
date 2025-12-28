#!/usr/bin/env python3
"""
Tri-Quaternion Text Generator

Unified pipeline using three quaternions:
  Q1 (Concept):  What fits together (polyomino fitting)
  Q2 (Output):   How to express it (style, certainty)
  Q3 (Morpho):   How words transform (conjugation)

The pipeline:
  1. Q1 generates valid frames by fitting concepts
  2. Q3 transforms verbs to correct conjugation
  3. Q2 applies style and certainty polish

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.polyomino_generator import PolyominoGenerator, Frame
from experiments.morphological_quaternion import MorphoQuaternion, MorphologicalTransformer

PHI = 1.618034


@dataclass
class OutputQuaternion:
    """
    Q2: Output space quaternion (style/certainty).
    
    X: Style      (-1=literary, 0=neutral, +1=hemingway)
    Y: Perspective (-1=actor, 0=neutral, +1=narrator)
    Z: Depth      (-1=terse, 0=moderate, +1=elaborate)
    W: Certainty  (-1=definitive, 0=neutral, +1=hedged)
    """
    x: float = 0.0  # Style
    y: float = 0.0  # Perspective
    z: float = 0.0  # Depth
    w: float = 0.0  # Certainty (tachyon axis)
    
    @property
    def style(self) -> str:
        if self.x < -0.3:
            return 'literary'
        elif self.x > 0.3:
            return 'hemingway'
        return 'neutral'
    
    @property
    def certainty(self) -> str:
        if self.w < -0.3:
            return 'definitive'
        elif self.w > 0.3:
            return 'hedged'
        return 'neutral'
    
    @property
    def depth(self) -> str:
        if self.z < -0.3:
            return 'terse'
        elif self.z > 0.3:
            return 'elaborate'
        return 'moderate'


class TriQuaternionGenerator:
    """
    Unified text generator using three quaternions.
    
    Q1 (Concept):  Polyomino fitting - generates valid frames
    Q2 (Output):   Style/certainty - controls expression
    Q3 (Morpho):   Conjugation - transforms word forms
    """
    
    def __init__(self):
        self.q1_generator = PolyominoGenerator()  # Concept space
        self.q3_transformer = MorphologicalTransformer()  # Morphological
        
        # Default Q2 and Q3 settings
        self.q2 = OutputQuaternion()
        self.q3 = MorphoQuaternion(x=1, y=-1, z=0, w=-1)  # 3rd sing present simple
    
    def learn(self, text: str):
        """Learn concept patterns from text."""
        self.q1_generator.learn_from_text(text)
    
    def set_output_style(self, style: str = 'neutral', certainty: str = 'neutral', 
                         depth: str = 'moderate'):
        """Set Q2 (output) parameters."""
        style_map = {'literary': -1, 'neutral': 0, 'hemingway': 1}
        certainty_map = {'definitive': -1, 'neutral': 0, 'hedged': 1}
        depth_map = {'terse': -1, 'moderate': 0, 'elaborate': 1}
        
        self.q2 = OutputQuaternion(
            x=style_map.get(style, 0),
            y=0,
            z=depth_map.get(depth, 0),
            w=certainty_map.get(certainty, 0),
        )
    
    def set_morphology(self, person: str = '3rd', number: str = 'singular',
                       tense: str = 'present', aspect: str = 'simple'):
        """Set Q3 (morphological) parameters."""
        person_map = {'1st': -1, '2nd': 0, '3rd': 1}
        number_map = {'singular': -1, 'plural': 1}
        tense_map = {'past': -1, 'present': 0, 'future': 1}
        aspect_map = {'simple': -1, 'perfect': 0, 'progressive': 1}
        
        self.q3 = MorphoQuaternion(
            x=person_map.get(person, 1),
            y=number_map.get(number, -1),
            z=tense_map.get(tense, 0),
            w=aspect_map.get(aspect, -1),
        )
    
    def _get_certainty_opener(self) -> str:
        """Get opener based on Q2 certainty."""
        if self.q2.certainty == 'definitive':
            return random.choice(['Certainly,', 'Without question,', 'Undoubtedly,']) + ' '
        elif self.q2.certainty == 'hedged':
            return random.choice(['Perhaps', 'It seems that', 'Arguably,']) + ' '
        return ''
    
    def _get_certainty_copula(self) -> str:
        """Get copula based on Q2 certainty."""
        if self.q2.certainty == 'definitive':
            return 'is undoubtedly'
        elif self.q2.certainty == 'hedged':
            return 'appears to be'
        return 'is'
    
    def _apply_style(self, actor: str, verb: str, target: Optional[str]) -> str:
        """Apply Q2 style to generate sentence."""
        actor_cap = actor.title()
        
        if self.q2.style == 'hemingway':
            # Terse, direct
            if target:
                return f"{actor_cap} {verb} {target}."
            return f"{actor_cap} {verb}."
        
        elif self.q2.style == 'literary':
            # Elaborate, formal
            opener = self._get_certainty_opener()
            if target:
                return f"{opener}{actor_cap}, with characteristic focus, {verb} {target}."
            return f"{opener}{actor_cap} {verb}, as is typical of the character."
        
        else:  # neutral
            opener = self._get_certainty_opener()
            if target:
                return f"{opener}{actor_cap} {verb} {target}."
            return f"{opener}{actor_cap} {verb}."
    
    def generate_sentence(self, seed: Optional[str] = None) -> str:
        """
        Generate a sentence using all three quaternions.
        
        Q1: Generate valid frame (polyomino fitting)
        Q3: Transform verb (morphological)
        Q2: Apply style (output)
        """
        # Q1: Generate frame with fitting constraint
        frame = self.q1_generator.generate_frame(seed)
        
        # Q3: Transform verb to correct conjugation
        # First get the base form of the action
        base_action = self.q3_transformer._get_base(frame.action)
        verb = self.q3_transformer.transform(base_action, self.q3)
        
        # Filter target
        target = frame.target
        if target:
            bad_targets = {'tall', 'small', 'confused', 'scared', 'angrily', 'intently',
                          'gracefully', 'sweetly', 'wildly', 'slowly', 'quickly',
                          'carefully', 'methodically', 'proudly', 'completely',
                          'mysteriously', 'going', 'through', 'where', 'unusual',
                          'near', 'understand', 'elementary', 'immediately', 'love',
                          'him', 'her', 'them'}
            if target in bad_targets or target.endswith('ly'):
                target = None
            elif target not in {'evidence', 'room', 'journal', 'newspaper', 'garden',
                               'building', 'window', 'scene', 'tea', 'hole', 'footprints',
                               'witnesses', 'villain', 'rabbit', 'doorway'}:
                target = target.title()  # Proper noun
            else:
                target = 'the ' + target  # Common noun
        
        # Q2: Apply style
        sentence = self._apply_style(frame.actor, verb, target)
        
        return sentence
    
    def generate_paragraph(self, seed: Optional[str] = None, 
                          num_sentences: int = 3) -> str:
        """Generate a paragraph by chaining sentences."""
        sentences = []
        current_seed = seed
        
        for i in range(num_sentences):
            sentence = self.generate_sentence(current_seed)
            sentences.append(sentence)
            
            # Get next seed from the frame
            frame = self.q1_generator.generate_frame(current_seed)
            if frame.target and frame.target in self.q1_generator.concepts:
                if self.q1_generator.concepts[frame.target].direction > 0:
                    current_seed = frame.target
            else:
                current_seed = frame.actor
        
        return " ".join(sentences)
    
    def generate_about(self, entity: str) -> str:
        """Generate a description about an entity."""
        if entity not in self.q1_generator.concepts:
            return f"I don't have information about {entity}."
        
        # Generate multiple sentences about the entity
        sentences = []
        
        # Intro sentence
        self.set_morphology(tense='present')
        sentences.append(self.generate_sentence(entity))
        
        # Action sentence (past tense)
        if self.q2.depth != 'terse':
            self.set_morphology(tense='past')
            sentences.append(self.generate_sentence(entity))
        
        # Elaboration (if elaborate)
        if self.q2.depth == 'elaborate':
            self.set_morphology(tense='present', aspect='progressive')
            sentences.append(self.generate_sentence(entity))
        
        return " ".join(sentences)


def run_experiment():
    """Test the tri-quaternion generator."""
    print("=" * 70)
    print("TRI-QUATERNION TEXT GENERATOR")
    print("=" * 70)
    print()
    print("Three quaternions working together:")
    print("  Q1 (Concept):  Polyomino fitting - valid frames")
    print("  Q2 (Output):   Style/certainty - expression control")
    print("  Q3 (Morpho):   Conjugation - word transformation")
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
    
    # Create generator and learn
    gen = TriQuaternionGenerator()
    gen.learn(corpus)
    
    print(f"Learned {len(gen.q1_generator.concepts)} concepts")
    print()
    
    # Test different Q2 settings (style/certainty)
    print("=" * 70)
    print("Q2 AXIS TEST: Style × Certainty")
    print("=" * 70)
    print()
    
    test_settings = [
        ('hemingway', 'definitive', 'terse'),
        ('hemingway', 'hedged', 'terse'),
        ('neutral', 'neutral', 'moderate'),
        ('literary', 'definitive', 'elaborate'),
        ('literary', 'hedged', 'elaborate'),
    ]
    
    for style, certainty, depth in test_settings:
        gen.set_output_style(style=style, certainty=certainty, depth=depth)
        gen.set_morphology(tense='present')
        
        print(f"Style={style}, Certainty={certainty}, Depth={depth}")
        sentence = gen.generate_sentence('holmes')
        print(f"  {sentence}")
        print()
    
    # Test different Q3 settings (morphology)
    print("=" * 70)
    print("Q3 AXIS TEST: Tense × Aspect")
    print("=" * 70)
    print()
    
    gen.set_output_style(style='neutral', certainty='neutral', depth='moderate')
    
    morpho_settings = [
        ('3rd', 'singular', 'present', 'simple'),
        ('3rd', 'singular', 'past', 'simple'),
        ('3rd', 'singular', 'present', 'progressive'),
        ('3rd', 'singular', 'present', 'perfect'),
        ('3rd', 'singular', 'future', 'simple'),
    ]
    
    for person, number, tense, aspect in morpho_settings:
        gen.set_morphology(person=person, number=number, tense=tense, aspect=aspect)
        
        print(f"Person={person}, Number={number}, Tense={tense}, Aspect={aspect}")
        sentence = gen.generate_sentence('holmes')
        print(f"  {sentence}")
        print()
    
    # Generate about entities
    print("=" * 70)
    print("ENTITY DESCRIPTIONS")
    print("=" * 70)
    print()
    
    for entity in ['holmes', 'watson', 'alice', 'darcy']:
        print(f"Who is {entity.title()}?")
        
        # Hemingway style
        gen.set_output_style(style='hemingway', certainty='definitive', depth='terse')
        print(f"  [Hemingway] {gen.generate_about(entity)}")
        
        # Literary style
        gen.set_output_style(style='literary', certainty='neutral', depth='elaborate')
        print(f"  [Literary]  {gen.generate_about(entity)}")
        
        print()
    
    # Generate paragraphs
    print("=" * 70)
    print("GENERATED PARAGRAPHS")
    print("=" * 70)
    print()
    
    gen.set_output_style(style='neutral', certainty='neutral', depth='moderate')
    gen.set_morphology(tense='past')
    
    for seed in ['holmes', 'alice']:
        print(f"Seed: {seed}")
        paragraph = gen.generate_paragraph(seed, num_sentences=3)
        print(f"  {paragraph}")
        print()
    
    # Full quaternion control demo
    print("=" * 70)
    print("FULL TRI-QUATERNION CONTROL")
    print("=" * 70)
    print()
    
    print("Same content, different quaternion settings:")
    print()
    
    configs = [
        # (style, certainty, depth, tense, aspect, label)
        ('hemingway', 'definitive', 'terse', 'past', 'simple', 
         "Hemingway + Definitive + Past"),
        ('hemingway', 'hedged', 'terse', 'present', 'progressive',
         "Hemingway + Hedged + Progressive"),
        ('literary', 'definitive', 'elaborate', 'present', 'perfect',
         "Literary + Definitive + Perfect"),
        ('literary', 'hedged', 'elaborate', 'past', 'simple',
         "Literary + Hedged + Past"),
    ]
    
    for style, certainty, depth, tense, aspect, label in configs:
        gen.set_output_style(style=style, certainty=certainty, depth=depth)
        gen.set_morphology(tense=tense, aspect=aspect)
        
        print(f"{label}:")
        print(f"  {gen.generate_sentence('holmes')}")
        print()
    
    return gen


if __name__ == "__main__":
    gen = run_experiment()
