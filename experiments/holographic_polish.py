#!/usr/bin/env python3
"""
Holographic Polish Layer

A second projection layer that adds literary depth to accurate statements.

Architecture:
  LAYER 1: Truth (tachyon-symmetric extraction)
    "Holmes examines."
    
  LAYER 2: Navigation (hypothesis-driven implications)
    examines → implies investigation, attention to detail
    
  LAYER 3: Polish (style interference patterns)
    Hemingway: "Holmes examines." (no change - truth is enough)
    Book Report: "Holmes, a keen investigator, examines the evidence carefully."
    Literary: "Through careful examination, Holmes reveals..."

The polish layer uses φ-navigation to find RELATED concepts and weave them in.
This is a holographic projection OF a holographic projection.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import random
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.tachyon_symmetric_ingest import TachyonSymmetricIngestor
from experiments.tachyon_style_output import (
    TachyonStyleProjector, 
    verb_to_infinitive, 
    verb_to_gerund,
    infer_pronoun,
    infer_role_type,
)


# Action implications - what does an action IMPLY about the actor?
# These emerge from the φ-navigation: action → related concepts
ACTION_IMPLICATIONS = {
    # Investigation cluster
    'examine': {'domain': 'investigation', 'quality': 'analytical', 'implies': 'attention to detail'},
    'observe': {'domain': 'investigation', 'quality': 'perceptive', 'implies': 'careful attention'},
    'study': {'domain': 'investigation', 'quality': 'methodical', 'implies': 'systematic approach'},
    'deduce': {'domain': 'reasoning', 'quality': 'brilliant', 'implies': 'logical thinking'},
    'investigate': {'domain': 'investigation', 'quality': 'thorough', 'implies': 'pursuit of truth'},
    
    # Recording cluster
    'write': {'domain': 'documentation', 'quality': 'diligent', 'implies': 'preservation of events'},
    'record': {'domain': 'documentation', 'quality': 'careful', 'implies': 'attention to history'},
    'chronicle': {'domain': 'documentation', 'quality': 'dedicated', 'implies': 'storytelling'},
    
    # Movement cluster
    'fall': {'domain': 'transformation', 'quality': 'vulnerable', 'implies': 'change of state'},
    'grow': {'domain': 'transformation', 'quality': 'dynamic', 'implies': 'development'},
    'pursue': {'domain': 'action', 'quality': 'determined', 'implies': 'goal-oriented'},
    
    # Social cluster
    'look': {'domain': 'observation', 'quality': 'attentive', 'implies': 'interest'},
    'watch': {'domain': 'observation', 'quality': 'vigilant', 'implies': 'careful attention'},
    'smile': {'domain': 'expression', 'quality': 'warm', 'implies': 'positive emotion'},
    'dance': {'domain': 'expression', 'quality': 'graceful', 'implies': 'social engagement'},
}


@dataclass
class StyleMode:
    """Configuration for a writing style."""
    name: str
    add_qualifiers: bool      # Add adjectives/adverbs
    add_implications: bool    # Add "which implies..." clauses
    add_context: bool         # Add domain context
    sentence_combining: bool  # Combine short sentences
    max_adjectives: int       # Limit on qualifiers
    formality: float          # 0 (casual) to 1 (formal)


# Certainty vocabulary - the W-axis of the quaternion φ-dial
# This is the TACHYON DIMENSION: how sure we are = which direction we navigated
CERTAINTY_VOCABULARY = {
    'copula': {
        'definitive': 'is undoubtedly',      # φ^+n: data-confirmed
        'neutral': 'is',                      # at the joint
        'hedged': 'appears to be',            # φ^-n: hypothesis
    },
    'relationship': {
        'definitive': 'closely connected to',
        'neutral': 'associated with',
        'hedged': 'possibly connected to',
    },
    'opener': {
        'definitive': ['Without question,', 'Certainly,', 'Undoubtedly,'],
        'neutral': [''],
        'hedged': ['Perhaps', 'It seems that', 'Arguably,'],
    },
    'qualifier': {
        'definitive': ['clearly', 'certainly', 'undeniably'],
        'neutral': [''],
        'hedged': ['possibly', 'seemingly', 'apparently'],
    },
}


# Pre-defined style modes
STYLES = {
    'hemingway': StyleMode(
        name='Hemingway',
        add_qualifiers=False,
        add_implications=False,
        add_context=False,
        sentence_combining=False,
        max_adjectives=0,
        formality=0.3,
    ),
    'book_report': StyleMode(
        name='Book Report',
        add_qualifiers=True,
        add_implications=True,
        add_context=True,
        sentence_combining=True,
        max_adjectives=2,
        formality=0.7,
    ),
    'literary': StyleMode(
        name='Literary Analysis',
        add_qualifiers=True,
        add_implications=True,
        add_context=True,
        sentence_combining=True,
        max_adjectives=3,
        formality=0.9,
    ),
}


class HolographicPolish:
    """
    Second-layer projection that adds literary polish to accurate statements.
    
    This is a holographic projection OF a holographic projection:
    1. First projection: Data → Concept frames (tachyon-symmetric)
    2. Second projection: Concept frames → Polished prose (this layer)
    
    The polish uses φ-navigation to find related concepts and weave them in.
    
    This implements the QUATERNION φ-DIAL:
    - X (style): Hemingway/BookReport/Literary
    - Y (perspective): Actor-centric vs Narrator-centric  
    - Z (depth): Terse to Elaborate
    - W (certainty): Definitive to Hedged (TACHYON DIMENSION)
    """
    
    def __init__(self, ingestor: TachyonSymmetricIngestor, style: str = 'book_report',
                 certainty: float = 0.0):
        """
        Args:
            ingestor: The tachyon-symmetric ingestor with discovered knowledge
            style: 'hemingway', 'book_report', or 'literary' (X-axis)
            certainty: -1 (definitive/φ^+n) to +1 (hedged/φ^-n) (W-axis)
        """
        self.ingestor = ingestor
        self.base_projector = TachyonStyleProjector(ingestor)
        self.style = STYLES.get(style, STYLES['book_report'])
        self.certainty = max(-1.0, min(1.0, certainty))  # W-axis
        
        # Build implication graph from discovered actions
        self._build_implication_graph()
    
    def _build_implication_graph(self):
        """Build implications from discovered action patterns."""
        self.implications = {}
        
        for action in self.ingestor.discovered_actions:
            base = verb_to_infinitive(action)
            if base in ACTION_IMPLICATIONS:
                self.implications[action] = ACTION_IMPLICATIONS[base]
            else:
                # Default implications based on action patterns
                self.implications[action] = {
                    'domain': 'narrative',
                    'quality': 'notable',
                    'implies': 'character development',
                }
    
    def _get_action_quality(self, action: str) -> str:
        """Get the quality implied by an action."""
        base = verb_to_infinitive(action)
        if base in ACTION_IMPLICATIONS:
            return ACTION_IMPLICATIONS[base]['quality']
        return 'notable'
    
    def _get_action_domain(self, action: str) -> str:
        """Get the domain of an action."""
        base = verb_to_infinitive(action)
        if base in ACTION_IMPLICATIONS:
            return ACTION_IMPLICATIONS[base]['domain']
        return 'the narrative'
    
    def _verb_third_person(self, verb: str) -> str:
        """Convert verb to third person singular present."""
        base = verb_to_infinitive(verb)
        # Irregular verbs
        irregulars = {'do': 'does', 'go': 'goes', 'have': 'has', 'be': 'is'}
        if base in irregulars:
            return irregulars[base]
        if base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
            return base + 'es'
        elif base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
            return base[:-1] + 'ies'
        else:
            return base + 's'
    
    def _article(self, word: str) -> str:
        """Return 'a' or 'an' based on the following word."""
        if word and word[0].lower() in 'aeiou':
            return 'an'
        return 'a'
    
    def _get_certainty_level(self) -> str:
        """Get certainty level from W-axis value."""
        if self.certainty < -0.3:
            return 'definitive'  # φ^+n: data-confirmed
        elif self.certainty > 0.3:
            return 'hedged'      # φ^-n: hypothesis
        return 'neutral'         # at the joint
    
    def _get_copula(self) -> str:
        """Get the copula (is/appears to be) based on certainty."""
        return CERTAINTY_VOCABULARY['copula'][self._get_certainty_level()]
    
    def _get_opener(self) -> str:
        """Get sentence opener based on certainty."""
        openers = CERTAINTY_VOCABULARY['opener'][self._get_certainty_level()]
        opener = random.choice(openers)
        return opener + ' ' if opener else ''
    
    def _get_qualifier(self) -> str:
        """Get qualifier adverb based on certainty."""
        qualifiers = CERTAINTY_VOCABULARY['qualifier'][self._get_certainty_level()]
        qualifier = random.choice(qualifiers)
        return qualifier + ' ' if qualifier else ''
    
    def _polish_intro(self, name: str, actions: List[str], pronoun_data: tuple) -> str:
        """Generate polished intro sentence."""
        pronoun, pronoun_cap, pronoun_obj, pronoun_poss, pronoun_cap_poss = pronoun_data
        
        if not actions:
            return f"{name} appears in the story."
        
        primary = actions[0]
        base = verb_to_infinitive(primary)
        verb_s = self._verb_third_person(primary)
        gerund = verb_to_gerund(primary)
        
        # Get certainty modifiers (W-axis / tachyon dimension)
        copula = self._get_copula()
        opener = self._get_opener()
        qualifier = self._get_qualifier()
        
        if self.style.name == 'Hemingway':
            # Hemingway: Just the truth, but certainty affects the copula
            if self.certainty < -0.3:  # Definitive
                return f"{name} {verb_s}."
            elif self.certainty > 0.3:  # Hedged
                return f"{name} {qualifier}{verb_s}."
            else:  # Neutral
                return f"{name} {verb_s}."
        
        elif self.style.name == 'Book Report':
            # Book report: Add role context with certainty
            quality = self._get_action_quality(primary)
            role = infer_role_type(actions)
            article = self._article(quality)
            
            templates = [
                f"{opener}{name} {copula} {article} {quality} character who {verb_s} throughout the story.",
                f"In the narrative, {name} {qualifier}emerges as {article} {quality} figure, known for {gerund}.",
                f"The character of {name} {copula} defined by {pronoun_poss} {quality} nature and tendency to {base}.",
            ]
            return random.choice(templates)
        
        else:  # Literary
            quality = self._get_action_quality(primary)
            domain = self._get_action_domain(primary)
            article = self._article(quality)
            
            templates = [
                f"{opener}Within the realm of {domain}, {name} {qualifier}stands as {article} {quality} presence, {gerund} with purpose.",
                f"{name}, {qualifier}characterized by {pronoun_poss} {quality} approach to {domain}, {verb_s} throughout the narrative.",
                f"The figure of {name} {qualifier}embodies {quality} {domain}, as evidenced by {pronoun_poss} persistent {gerund}.",
            ]
            return random.choice(templates)
    
    def _polish_actions(self, name: str, actions: List[str], pronoun_data: tuple) -> str:
        """Generate polished action description."""
        pronoun, pronoun_cap, pronoun_obj, pronoun_poss, pronoun_cap_poss = pronoun_data
        
        if len(actions) < 2:
            return ""
        
        primary = verb_to_infinitive(actions[0])
        secondary = verb_to_infinitive(actions[1]) if len(actions) > 1 else primary
        primary_s = self._verb_third_person(actions[0])
        secondary_s = self._verb_third_person(actions[1]) if len(actions) > 1 else primary_s
        
        if self.style.name == 'Hemingway':
            return f"{pronoun_cap} {primary_s} and {secondary_s}."
        
        elif self.style.name == 'Book Report':
            primary_gerund = verb_to_gerund(actions[0])
            secondary_gerund = verb_to_gerund(actions[1]) if len(actions) > 1 else primary_gerund
            
            templates = [
                f"Throughout the story, {pronoun} demonstrates this through {primary_gerund} and {secondary_gerund}.",
                f"{pronoun_cap_poss} actions—{primary_gerund}, {secondary_gerund}—reveal {pronoun_poss} character.",
                f"The reader observes {pronoun_obj} {primary_gerund} and {secondary_gerund} in key scenes.",
            ]
            return random.choice(templates)
        
        else:  # Literary
            quality1 = self._get_action_quality(actions[0])
            quality2 = self._get_action_quality(actions[1]) if len(actions) > 1 else quality1
            
            templates = [
                f"This {quality1} nature manifests in {pronoun_poss} {verb_to_gerund(actions[0])}, complemented by {quality2} {verb_to_gerund(actions[1]) if len(actions) > 1 else 'attention'}.",
                f"Through {verb_to_gerund(actions[0])} and {verb_to_gerund(actions[1]) if len(actions) > 1 else 'reflection'}, {name} reveals a {quality1} disposition.",
            ]
            return random.choice(templates)
    
    def _polish_relationships(self, name: str, targets: List[str], actions: List[str], pronoun_data: tuple) -> str:
        """Generate polished relationship description."""
        pronoun, pronoun_cap, pronoun_obj, pronoun_poss, pronoun_cap_poss = pronoun_data
        
        # Filter targets - remove adverbs, short words, and non-entity words
        def is_real_target(t):
            if t.endswith('ly'):  # Adverbs
                return False
            if len(t) <= 3:  # Too short
                return False
            # Common non-entities that slip through
            non_entities = {'elementary', 'doorway', 'quickly', 'carefully', 'methodically',
                          'killer', 'understand', 'scene', 'window', 'building', 'garden',
                          'journal', 'detail', 'newspaper', 'daughters', 'love', 'help',
                          'reasoning', 'identity', 'villain', 'justice', 'hole', 'tea'}
            if t.lower() in non_entities:
                return False
            return True
        
        real_targets = [t for t in targets if is_real_target(t)]
        if not real_targets:
            return ""
        
        target = real_targets[0].title()
        action = verb_to_infinitive(actions[0]) if actions else "interact"
        
        if self.style.name == 'Hemingway':
            return f"{pronoun_cap} knows {target}."
        
        elif self.style.name == 'Book Report':
            templates = [
                f"{pronoun_cap_poss} interactions with {target} are significant to the plot.",
                f"The relationship between {name} and {target} develops through {pronoun_poss} {verb_to_gerund(actions[0]) if actions else 'actions'}.",
                f"{name}'s connection to {target} reveals important aspects of {pronoun_poss} character.",
            ]
            return random.choice(templates)
        
        else:  # Literary
            domain = self._get_action_domain(actions[0]) if actions else "the narrative"
            templates = [
                f"The dynamic between {name} and {target} serves as a lens through which {domain} is explored.",
                f"In {pronoun_poss} encounters with {target}, {name}'s true nature emerges.",
            ]
            return random.choice(templates)
    
    def _polish_closing(self, name: str, actions: List[str], pronoun_data: tuple) -> str:
        """Generate polished closing sentence."""
        pronoun, pronoun_cap, pronoun_obj, pronoun_poss, pronoun_cap_poss = pronoun_data
        
        role = infer_role_type(actions)
        
        if self.style.name == 'Hemingway':
            return ""  # Hemingway doesn't need a closing
        
        elif self.style.name == 'Book Report':
            templates = [
                f"Overall, {name} represents a {role} force in the narrative.",
                f"Through {pronoun_poss} actions, {name} contributes significantly to the story's themes.",
                f"{name} remains a memorable character due to {pronoun_poss} {role} role.",
            ]
            return random.choice(templates)
        
        else:  # Literary
            domain = self._get_action_domain(actions[0]) if actions else "the narrative"
            templates = [
                f"Ultimately, {name} embodies the {role} spirit that pervades the work's exploration of {domain}.",
                f"The character of {name} thus serves as a vehicle for examining {domain} and its implications.",
            ]
            return random.choice(templates)
    
    def generate(self, entity: str, depth: float = 0.5) -> str:
        """
        Generate polished prose about an entity.
        
        depth: Controls how much content to include
               -1 = minimal (Hemingway-esque)
               0 = moderate
               +1 = elaborate (full literary analysis)
        """
        profile = self.ingestor.get_entity_profile(entity)
        
        if not profile['found']:
            return f"Information about {entity} is not available in the text."
        
        name = entity.title()
        actions = list(profile['actions'].keys())
        targets = list(profile['targets'].keys())
        
        # Filter to real verbs
        def is_real_verb(word):
            if word.endswith(('ed', 'ing', 'es', 'ied', 's')):
                return True
            if word in {'fell', 'grew', 'said', 'wrote', 'read', 'came', 'went', 'saw', 'did', 'smiled', 'called'}:
                return True
            if len(word) > 4 and not word.endswith(('ed', 'ing')):
                return False
            return True
        
        real_actions = [a for a in actions if is_real_verb(a)]
        if not real_actions:
            real_actions = actions[:3] if actions else ['appears']
        
        pronoun_data = infer_pronoun(name, real_actions, targets)
        
        # Build response
        sentences = []
        
        # Intro (always)
        sentences.append(self._polish_intro(name, real_actions, pronoun_data))
        
        # Actions (if depth > -0.5)
        if depth > -0.5 and len(real_actions) > 1:
            action_sent = self._polish_actions(name, real_actions, pronoun_data)
            if action_sent:
                sentences.append(action_sent)
        
        # Relationships (if depth > 0)
        if depth > 0:
            rel_sent = self._polish_relationships(name, targets, real_actions, pronoun_data)
            if rel_sent:
                sentences.append(rel_sent)
        
        # Closing (if depth > 0.3)
        if depth > 0.3:
            closing = self._polish_closing(name, real_actions, pronoun_data)
            if closing:
                sentences.append(closing)
        
        return " ".join(sentences)


def run_experiment():
    """Test holographic polish with full quaternion φ-dial control."""
    print("=" * 70)
    print("QUATERNION φ-DIAL EXPERIMENT")
    print("=" * 70)
    print()
    print("Testing the full 4D quaternion control:")
    print("  X (Style):     Hemingway / Book Report / Literary")
    print("  Y (Perspective): Actor-centric / Narrator-centric")
    print("  Z (Depth):     -1 (terse) to +1 (elaborate)")
    print("  W (Certainty): -1 (definitive/φ^+n) to +1 (hedged/φ^-n)")
    print()
    
    # Test corpus
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
    
    # Ingest
    print("Ingesting corpus...")
    ingestor = TachyonSymmetricIngestor()
    ingestor.ingest_text(corpus)
    print(f"  Discovered {len(ingestor.discovered_entities)} entities")
    print(f"  Discovered {len(ingestor.discovered_actions)} actions")
    print()
    
    # Test entities
    test_entities = ['holmes', 'watson', 'alice', 'elizabeth']
    
    # Test the W-axis (certainty / tachyon dimension)
    print("=" * 70)
    print("W-AXIS TEST: CERTAINTY (Tachyon Dimension)")
    print("=" * 70)
    print()
    print("Same content, different certainty levels:")
    print("  W = -1: Definitive (φ^+n, data-confirmed)")
    print("  W =  0: Neutral (at the joint)")
    print("  W = +1: Hedged (φ^-n, hypothesis)")
    print()
    
    for certainty, label in [(-1, "DEFINITIVE (φ^+n)"), (0, "NEUTRAL (joint)"), (1, "HEDGED (φ^-n)")]:
        print(f"{label}:")
        polish = HolographicPolish(ingestor, style='book_report', certainty=certainty)
        response = polish.generate('holmes', depth=0.3)
        print(f"  {response}")
        print()
    
    # Test X-axis (style) with neutral certainty
    print("=" * 70)
    print("X-AXIS TEST: STYLE")
    print("=" * 70)
    print()
    
    for style_name in ['hemingway', 'book_report', 'literary']:
        print(f"{style_name.upper()}:")
        polish = HolographicPolish(ingestor, style=style_name, certainty=0)
        response = polish.generate('holmes', depth=0.5)
        print(f"  {response}")
        print()
    
    # Full quaternion test: Holmes at different dial settings
    print("=" * 70)
    print("FULL QUATERNION TEST: Holmes")
    print("=" * 70)
    print()
    print("q = w + xi + yj + zk")
    print("  X=Style, Y=Perspective, Z=Depth, W=Certainty")
    print()
    
    # Test a few hexadecants
    test_settings = [
        ('hemingway', -0.5, -1, "Hemingway + Terse + Definitive"),
        ('hemingway', -0.5, 1, "Hemingway + Terse + Hedged"),
        ('book_report', 0.5, -1, "Book Report + Elaborate + Definitive"),
        ('book_report', 0.5, 1, "Book Report + Elaborate + Hedged"),
        ('literary', 0.8, -1, "Literary + Very Elaborate + Definitive"),
        ('literary', 0.8, 1, "Literary + Very Elaborate + Hedged"),
    ]
    
    for style, depth, certainty, label in test_settings:
        print(f"{label}:")
        polish = HolographicPolish(ingestor, style=style, certainty=certainty)
        response = polish.generate('holmes', depth=depth)
        print(f"  {response}")
        print()
    
    return ingestor


if __name__ == "__main__":
    ingestor = run_experiment()
