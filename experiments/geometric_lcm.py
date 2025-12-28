#!/usr/bin/env python3
"""
GeometricLCM: Quad-Quaternion Language Model

A geometric approach to language modeling using four quaternions:
  Q1 (Concept):  What fits together (polyomino fitting)
  Q2 (Output):   How to express it (style, certainty)
  Q3 (Morpho):   How words transform (conjugation)
  Q4 (Error):    What's wrong and how to fix it

This model can:
  - Ingest text and learn concept patterns
  - Generate text using polyomino fitting
  - Answer questions about ingested content
  - Control output style via quaternion dials

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.polyomino_generator import PolyominoGenerator, Frame
from experiments.morphological_quaternion import MorphoQuaternion, MorphologicalTransformer
from experiments.error_quaternion import ErrorQuaternion, ErrorDetector

PHI = 1.618034


@dataclass
class QuaternionSettings:
    """Settings for all four quaternions."""
    # Q2: Output
    style: str = 'neutral'        # literary, neutral, hemingway
    certainty: str = 'neutral'    # definitive, neutral, hedged
    depth: str = 'moderate'       # terse, moderate, elaborate
    
    # Q3: Morphology
    person: str = '3rd'           # 1st, 2nd, 3rd
    number: str = 'singular'      # singular, plural
    tense: str = 'present'        # past, present, future
    aspect: str = 'simple'        # simple, perfect, progressive


@dataclass
class EntityProfile:
    """Profile of an entity learned from text."""
    name: str
    actions: Counter = field(default_factory=Counter)
    targets: Counter = field(default_factory=Counter)
    acted_upon_by: Counter = field(default_factory=Counter)
    co_occurring: Set[str] = field(default_factory=set)
    direction: float = 0.0  # φ-direction (+1 entity, -1 action)


class GeometricLCM:
    """
    Geometric Language Concept Model using quad-quaternion architecture.
    
    The model learns from text by extracting:
    - Entity profiles (who does what to whom)
    - Action patterns (what actions occur)
    - Relationship patterns (who relates to whom)
    
    Generation uses polyomino fitting (Q1), morphological transformation (Q3),
    style projection (Q2), and error correction (Q4).
    """
    
    def __init__(self):
        # Core components
        self.concept_gen = PolyominoGenerator()  # Q1
        self.morpho = MorphologicalTransformer()  # Q3
        self.error_detector: Optional[ErrorDetector] = None  # Q4
        
        # Learned knowledge
        self.entities: Dict[str, EntityProfile] = {}
        self.actions: Set[str] = set()
        self.frames: List[Dict] = []
        
        # Current settings
        self.settings = QuaternionSettings()
        
        # Question patterns
        self.question_patterns = {
            'who': self._answer_who,
            'what': self._answer_what,
            'does': self._answer_does,
            'is': self._answer_is,
            'describe': self._answer_describe,
        }
    
    def ingest(self, text: str):
        """
        Ingest text and learn patterns.
        
        This populates Q1 (concept space) with learned patterns.
        """
        # Learn via polyomino generator
        self.concept_gen.learn_from_text(text)
        
        # Extract frames and build entity profiles
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            content = [t for t in tokens if self._is_content_word(t)]
            
            if len(content) >= 2:
                actor = content[0]
                action = content[1]
                target = content[2] if len(content) > 2 else None
                
                # Store frame
                self.frames.append({
                    'actor': actor,
                    'action': action,
                    'target': target,
                    'sentence': sentence.strip(),
                })
                
                # Update entity profiles
                self._update_entity(actor, action=action, target=target)
                if target:
                    self._update_entity(target, acted_upon_by=actor, action_received=action)
                
                # Track actions
                self.actions.add(action)
        
        # Compute φ-directions for entities
        self._compute_directions()
        
        # Initialize error detector
        self.error_detector = ErrorDetector(self.concept_gen)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _is_content_word(self, word: str) -> bool:
        function_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be',
                         'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                         'he', 'she', 'it', 'they', 'his', 'her', 'its',
                         'that', 'this', 'from', 'not', 'did', 'do', 'does',
                         'and', 'or', 'but', 'if', 'then', 'so', 'as',
                         'very', 'more', 'down', 'up', 'out', 'about',
                         'who', 'what', 'where', 'when', 'why', 'how'}
        return word not in function_words and len(word) > 2
    
    def _update_entity(self, name: str, action: str = None, target: str = None,
                       acted_upon_by: str = None, action_received: str = None):
        """Update or create entity profile."""
        if name not in self.entities:
            self.entities[name] = EntityProfile(name=name)
        
        profile = self.entities[name]
        
        if action:
            profile.actions[action] += 1
        if target:
            profile.targets[target] += 1
            profile.co_occurring.add(target)
        if acted_upon_by:
            profile.acted_upon_by[acted_upon_by] += 1
            profile.co_occurring.add(acted_upon_by)
    
    def _compute_directions(self):
        """Compute φ-directions for all entities."""
        for name, profile in self.entities.items():
            # Entities that DO things have positive direction
            # Entities that have things DONE TO them have negative direction
            actor_count = sum(profile.actions.values())
            patient_count = sum(profile.acted_upon_by.values())
            total = actor_count + patient_count
            
            if total > 0:
                profile.direction = (actor_count - patient_count) / total
    
    def set_style(self, style: str = None, certainty: str = None, 
                  depth: str = None):
        """Set Q2 (output) parameters."""
        if style:
            self.settings.style = style
        if certainty:
            self.settings.certainty = certainty
        if depth:
            self.settings.depth = depth
    
    def set_morphology(self, person: str = None, number: str = None,
                       tense: str = None, aspect: str = None):
        """Set Q3 (morphology) parameters."""
        if person:
            self.settings.person = person
        if number:
            self.settings.number = number
        if tense:
            self.settings.tense = tense
        if aspect:
            self.settings.aspect = aspect
    
    def _get_morpho_quaternion(self) -> MorphoQuaternion:
        """Convert settings to Q3 quaternion."""
        person_map = {'1st': -1, '2nd': 0, '3rd': 1}
        number_map = {'singular': -1, 'plural': 1}
        tense_map = {'past': -1, 'present': 0, 'future': 1}
        aspect_map = {'simple': -1, 'perfect': 0, 'progressive': 1}
        
        return MorphoQuaternion(
            x=person_map.get(self.settings.person, 1),
            y=number_map.get(self.settings.number, -1),
            z=tense_map.get(self.settings.tense, 0),
            w=aspect_map.get(self.settings.aspect, -1),
        )
    
    def _get_certainty_opener(self) -> str:
        """Get opener based on Q2 certainty."""
        if self.settings.certainty == 'definitive':
            return random.choice(['Certainly,', 'Without question,', 'Undoubtedly,']) + ' '
        elif self.settings.certainty == 'hedged':
            return random.choice(['Perhaps', 'It seems that', 'Arguably,']) + ' '
        return ''
    
    def _conjugate(self, verb: str) -> str:
        """Conjugate verb using Q3."""
        base = self.morpho._get_base(verb)
        q3 = self._get_morpho_quaternion()
        return self.morpho.transform(base, q3)
    
    def _format_sentence(self, actor: str, verb: str, target: str = None) -> str:
        """Format sentence using Q2 style settings."""
        actor_cap = actor.title()
        opener = self._get_certainty_opener()
        
        # Filter bad targets
        if target:
            bad_targets = {'tall', 'small', 'confused', 'scared', 'angrily', 
                          'intently', 'gracefully', 'wildly', 'slowly', 'quickly',
                          'him', 'her', 'them', 'it', 'carefully', 'methodically'}
            if target in bad_targets or target.endswith('ly'):
                target = None
        
        # Format target
        if target:
            common_nouns = {'evidence', 'room', 'journal', 'newspaper', 'garden',
                          'building', 'window', 'scene', 'tea', 'hole', 'footprints',
                          'witnesses', 'doorway', 'rabbit', 'villain'}
            if target in common_nouns:
                target_str = f"the {target}"
            else:
                target_str = target.title()
        else:
            target_str = None
        
        # Apply style
        if self.settings.style == 'hemingway':
            if target_str:
                return f"{opener}{actor_cap} {verb} {target_str}.".strip()
            return f"{opener}{actor_cap} {verb}.".strip()
        
        elif self.settings.style == 'literary':
            if target_str:
                return f"{opener}{actor_cap}, with characteristic focus, {verb} {target_str}."
            return f"{opener}{actor_cap} {verb}, as is typical of the character."
        
        else:  # neutral
            if target_str:
                return f"{opener}{actor_cap} {verb} {target_str}.".strip()
            return f"{opener}{actor_cap} {verb}.".strip()
    
    def generate(self, seed: str = None, num_sentences: int = 1) -> str:
        """
        Generate text using quad-quaternion pipeline.
        
        Q1: Find fitting concepts
        Q3: Transform to correct form
        Q2: Apply style
        Q4: Validate and correct if needed
        """
        sentences = []
        current_seed = seed.lower() if seed else None
        
        for _ in range(num_sentences):
            # Q1: Generate frame with fitting constraint
            frame = self.concept_gen.generate_frame(current_seed)
            
            # Q3: Conjugate verb
            verb = self._conjugate(frame.action)
            
            # Q2: Format with style
            sentence = self._format_sentence(frame.actor, verb, frame.target)
            
            # Q4: Check for errors (optional correction)
            if self.error_detector:
                q4 = self.error_detector.analyze(frame, sentence)
                if q4.needs_correction and q4.total_error > 0.5:
                    # Try regenerating with different seed
                    if frame.target and frame.target in self.concept_gen.concepts:
                        frame = self.concept_gen.generate_frame(frame.target)
                        verb = self._conjugate(frame.action)
                        sentence = self._format_sentence(frame.actor, verb, frame.target)
            
            sentences.append(sentence)
            
            # Chain to next seed
            if frame.target and frame.target in self.entities:
                current_seed = frame.target
            else:
                current_seed = frame.actor
        
        return " ".join(sentences)
    
    def ask(self, question: str) -> str:
        """
        Answer a question about ingested content.
        
        Uses Q1 to find relevant concepts and Q2 to format response.
        """
        question_lower = question.lower().strip().rstrip('?')
        
        # Detect question type
        for pattern, handler in self.question_patterns.items():
            if question_lower.startswith(pattern):
                return handler(question_lower)
        
        # Default: try to extract entity and describe
        words = self._tokenize(question_lower)
        for word in words:
            if word in self.entities:
                return self._answer_describe(f"describe {word}")
        
        return "I don't have enough information to answer that question."
    
    def _answer_who(self, question: str) -> str:
        """Answer 'Who is X?' questions."""
        # Extract entity name
        match = re.search(r'who\s+is\s+(\w+)', question)
        if not match:
            return "I'm not sure who you're asking about."
        
        entity = match.group(1).lower()
        
        if entity not in self.entities:
            return f"I don't have information about {entity.title()}."
        
        profile = self.entities[entity]
        
        # Build response using Q2 style
        opener = self._get_certainty_opener()
        
        # Get top actions
        top_actions = profile.actions.most_common(2)
        
        if self.settings.style == 'hemingway':
            if top_actions:
                action = top_actions[0][0]
                verb = self._conjugate(action)
                return f"{opener}{entity.title()} {verb}."
            return f"{opener}{entity.title()} exists in the story."
        
        elif self.settings.style == 'literary':
            if top_actions:
                actions_str = " and ".join([self._conjugate(a[0]) for a in top_actions])
                return f"{opener}{entity.title()} is a character who {actions_str} throughout the narrative."
            return f"{opener}{entity.title()} appears in the narrative."
        
        else:  # neutral
            if top_actions:
                action = top_actions[0][0]
                verb = self._conjugate(action)
                targets = list(profile.targets.keys())[:2]
                if targets:
                    target_str = " and ".join([t.title() for t in targets])
                    return f"{opener}{entity.title()} {verb}. They interact with {target_str}."
                return f"{opener}{entity.title()} {verb}."
            return f"{opener}{entity.title()} is mentioned in the text."
    
    def _answer_what(self, question: str) -> str:
        """Answer 'What does X do?' questions."""
        match = re.search(r'what\s+does\s+(\w+)\s+do', question)
        if not match:
            return "I'm not sure what you're asking about."
        
        entity = match.group(1).lower()
        
        if entity not in self.entities:
            return f"I don't have information about {entity.title()}."
        
        profile = self.entities[entity]
        top_actions = profile.actions.most_common(3)
        
        if not top_actions:
            return f"{entity.title()} doesn't have recorded actions."
        
        opener = self._get_certainty_opener()
        actions_str = ", ".join([self._conjugate(a[0]) for a in top_actions])
        
        return f"{opener}{entity.title()} {actions_str}."
    
    def _answer_does(self, question: str) -> str:
        """Answer 'Does X do Y?' questions."""
        match = re.search(r'does\s+(\w+)\s+(\w+)', question)
        if not match:
            return "I'm not sure what you're asking."
        
        entity = match.group(1).lower()
        action = match.group(2).lower()
        
        if entity not in self.entities:
            return f"I don't have information about {entity.title()}."
        
        profile = self.entities[entity]
        base_action = self.morpho._get_base(action)
        
        # Check if entity does this action
        for known_action in profile.actions:
            if self.morpho._get_base(known_action) == base_action:
                opener = self._get_certainty_opener()
                verb = self._conjugate(known_action)
                return f"{opener}Yes, {entity.title()} {verb}."
        
        if self.settings.certainty == 'hedged':
            return f"It's unclear whether {entity.title()} does that."
        return f"No, {entity.title()} doesn't appear to do that."
    
    def _answer_is(self, question: str) -> str:
        """Answer 'Is X related to Y?' or 'Is X a Y?' questions."""
        # Check for relationship question
        match = re.search(r'is\s+(\w+)\s+related\s+to\s+(\w+)', question)
        if match:
            entity1 = match.group(1).lower()
            entity2 = match.group(2).lower()
            
            if entity1 in self.entities:
                profile = self.entities[entity1]
                if entity2 in profile.co_occurring:
                    opener = self._get_certainty_opener()
                    return f"{opener}Yes, {entity1.title()} and {entity2.title()} are connected in the narrative."
            
            return f"I don't see a direct connection between {entity1.title()} and {entity2.title()}."
        
        return "I'm not sure how to answer that question."
    
    def _answer_describe(self, question: str) -> str:
        """Describe an entity."""
        match = re.search(r'describe\s+(\w+)', question)
        if not match:
            return "I'm not sure what to describe."
        
        entity = match.group(1).lower()
        
        if entity not in self.entities:
            return f"I don't have information about {entity.title()}."
        
        profile = self.entities[entity]
        
        # Generate multi-sentence description
        sentences = []
        
        # Opening
        opener = self._get_certainty_opener()
        top_actions = profile.actions.most_common(2)
        
        if top_actions:
            action = top_actions[0][0]
            verb = self._conjugate(action)
            sentences.append(f"{opener}{entity.title()} {verb}.")
        
        # Relationships
        if profile.co_occurring:
            related = list(profile.co_occurring)[:2]
            related_str = " and ".join([r.title() for r in related])
            sentences.append(f"They are connected to {related_str}.")
        
        # Additional action (if elaborate)
        if self.settings.depth == 'elaborate' and len(top_actions) > 1:
            action2 = top_actions[1][0]
            verb2 = self._conjugate(action2)
            sentences.append(f"They also {verb2}.")
        
        return " ".join(sentences) if sentences else f"{entity.title()} appears in the text."


def run_demo():
    """Demonstrate the GeometricLCM model."""
    print("=" * 70)
    print("GEOMETRIC LCM: Quad-Quaternion Language Model")
    print("=" * 70)
    print()
    
    # Create model
    model = GeometricLCM()
    
    # Ingest corpus
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
    
    print("Ingesting corpus...")
    model.ingest(corpus)
    print(f"Learned {len(model.entities)} entities and {len(model.actions)} actions")
    print()
    
    # Show entity profiles
    print("=" * 70)
    print("ENTITY PROFILES")
    print("=" * 70)
    print()
    
    for name in ['holmes', 'watson', 'alice', 'darcy']:
        if name in model.entities:
            profile = model.entities[name]
            actions = list(profile.actions.keys())[:3]
            print(f"{name.title()}:")
            print(f"  Direction: {profile.direction:+.2f} ({'entity' if profile.direction > 0 else 'patient'})")
            print(f"  Actions: {', '.join(actions)}")
            print(f"  Related to: {', '.join(list(profile.co_occurring)[:3])}")
            print()
    
    # Test generation with different Q2 settings
    print("=" * 70)
    print("TEXT GENERATION (Q2 Style Control)")
    print("=" * 70)
    print()
    
    styles = [
        ('hemingway', 'definitive', 'terse'),
        ('neutral', 'neutral', 'moderate'),
        ('literary', 'hedged', 'elaborate'),
    ]
    
    for style, certainty, depth in styles:
        model.set_style(style=style, certainty=certainty, depth=depth)
        model.set_morphology(tense='present')
        
        print(f"Style={style}, Certainty={certainty}, Depth={depth}")
        text = model.generate('holmes', num_sentences=2)
        print(f"  {text}")
        print()
    
    # Test generation with different Q3 settings
    print("=" * 70)
    print("TEXT GENERATION (Q3 Morphology Control)")
    print("=" * 70)
    print()
    
    model.set_style(style='neutral', certainty='neutral', depth='moderate')
    
    morphologies = [
        ('3rd', 'singular', 'present', 'simple'),
        ('3rd', 'singular', 'past', 'simple'),
        ('3rd', 'singular', 'present', 'progressive'),
        ('3rd', 'singular', 'present', 'perfect'),
    ]
    
    for person, number, tense, aspect in morphologies:
        model.set_morphology(person=person, number=number, tense=tense, aspect=aspect)
        
        print(f"Tense={tense}, Aspect={aspect}")
        text = model.generate('holmes', num_sentences=1)
        print(f"  {text}")
        print()
    
    # Test question answering
    print("=" * 70)
    print("QUESTION ANSWERING")
    print("=" * 70)
    print()
    
    model.set_style(style='neutral', certainty='neutral', depth='moderate')
    model.set_morphology(tense='present')
    
    questions = [
        "Who is Holmes?",
        "Who is Watson?",
        "Who is Alice?",
        "What does Holmes do?",
        "What does Watson do?",
        "Does Holmes examine?",
        "Does Watson write?",
        "Does Alice fly?",
        "Is Holmes related to Watson?",
        "Describe Darcy",
    ]
    
    for question in questions:
        print(f"Q: {question}")
        answer = model.ask(question)
        print(f"A: {answer}")
        print()
    
    # Test with different certainty levels
    print("=" * 70)
    print("CERTAINTY LEVELS (Q2 W-axis)")
    print("=" * 70)
    print()
    
    question = "Who is Holmes?"
    
    for certainty in ['definitive', 'neutral', 'hedged']:
        model.set_style(certainty=certainty)
        print(f"Certainty={certainty}")
        print(f"  Q: {question}")
        print(f"  A: {model.ask(question)}")
        print()
    
    return model


if __name__ == "__main__":
    model = run_demo()
