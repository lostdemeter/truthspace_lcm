#!/usr/bin/env python3
"""
GeometricLCM V2: Quad-Quaternion Language Model with Symmetric Ingestion

Improvements over V1:
1. Symmetric ingestion with φ-direction from role AND structure
2. Polyomino validation during frame extraction
3. Tachyon confidence tracking (forward vs backward evidence)
4. Expanded corpus (5 literary works)
5. Better relationship extraction

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

from experiments.morphological_quaternion import MorphoQuaternion, MorphologicalTransformer
from experiments.symmetric_ingest_v2 import SymmetricIngester, SymmetricConcept, SymmetricFrame, EXPANDED_CORPUS

PHI = 1.618034


@dataclass
class QuaternionSettings:
    """Settings for all four quaternions."""
    # Q2: Output
    style: str = 'neutral'
    certainty: str = 'neutral'
    depth: str = 'moderate'
    
    # Q3: Morphology
    person: str = '3rd'
    number: str = 'singular'
    tense: str = 'present'
    aspect: str = 'simple'


class GeometricLCMv2:
    """
    Geometric Language Concept Model V2 with symmetric ingestion.
    
    Uses quad-quaternion architecture:
    - Q1: Concept space (symmetric ingestion with φ-direction)
    - Q2: Output space (style, certainty, depth)
    - Q3: Morphological space (conjugation)
    - Q4: Error space (validation)
    """
    
    def __init__(self):
        # Core components
        self.ingester = SymmetricIngester()  # Q1 with symmetric understanding
        self.morpho = MorphologicalTransformer()  # Q3
        
        # Settings
        self.settings = QuaternionSettings()
        
        # Question patterns
        self.question_handlers = {
            'who': self._answer_who,
            'what': self._answer_what,
            'does': self._answer_does,
            'is': self._answer_is,
            'describe': self._answer_describe,
            'tell': self._answer_describe,
            'how': self._answer_how,
            'why': self._answer_why,
        }
    
    def ingest(self, text: str):
        """Ingest text using symmetric understanding."""
        self.ingester.ingest(text)
    
    def set_style(self, style: str = None, certainty: str = None, depth: str = None):
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
    
    def _conjugate(self, verb: str) -> str:
        """Conjugate verb using Q3."""
        base = self.morpho._get_base(verb)
        q3 = self._get_morpho_quaternion()
        return self.morpho.transform(base, q3)
    
    def _get_certainty_opener(self) -> str:
        """Get opener based on Q2 certainty (tachyon axis)."""
        if self.settings.certainty == 'definitive':
            return random.choice(['Certainly,', 'Without question,', 'Undoubtedly,']) + ' '
        elif self.settings.certainty == 'hedged':
            return random.choice(['Perhaps', 'It seems that', 'Arguably,']) + ' '
        return ''
    
    def _format_target(self, target: str) -> Optional[str]:
        """Format target with appropriate article."""
        if not target:
            return None
        
        # Skip bad targets
        bad_targets = {'tall', 'small', 'confused', 'scared', 'angrily', 'intently',
                      'gracefully', 'wildly', 'slowly', 'quickly', 'carefully',
                      'methodically', 'proudly', 'completely', 'mysteriously',
                      'him', 'her', 'them', 'it', 'against', 'through', 'afar',
                      'immediately', 'eventually', 'finally', 'suddenly', 'briefly',
                      'constantly', 'desperately', 'triumphantly', 'secretly',
                      'passionately', 'devotedly', 'treacherously', 'foolishly',
                      'victoriously', 'longingly', 'curiously', 'thoughtfully',
                      'tragically', 'eternally', 'deeply', 'humbly', 'joyfully'}
        
        if target in bad_targets or target.endswith('ly'):
            return None
        
        # Common nouns get articles
        common_nouns = {'evidence', 'room', 'journal', 'newspaper', 'garden',
                       'building', 'window', 'scene', 'tea', 'hole', 'footprints',
                       'witnesses', 'doorway', 'rabbit', 'villain', 'ball', 'party',
                       'table', 'pool', 'funeral', 'service', 'light', 'road',
                       'accident', 'curtain', 'duel', 'wine', 'throne', 'hookah',
                       'watch', 'stories', 'butter', 'croquet', 'flamingos',
                       'scandal', 'family', 'reputation', 'mistake', 'character',
                       'ghost', 'father', 'killer', 'identity', 'reasoning',
                       'case', 'clue', 'detail', 'audience', 'night', 'help',
                       'police', 'justice', 'adventures', 'knowledge', 'pattern',
                       'mud', 'shirts', 'money', 'connections', 'car', 'green'}
        
        if target in common_nouns:
            return f"the {target}"
        
        # Proper nouns get capitalized
        return target.title()
    
    def _format_sentence(self, actor: str, verb: str, target: str = None,
                         adverb: str = None) -> str:
        """Format sentence using Q2 style settings."""
        actor_cap = actor.title()
        opener = self._get_certainty_opener()
        target_str = self._format_target(target)
        
        # Include adverb if present and style allows
        adverb_str = ""
        if adverb and self.settings.depth != 'terse':
            adverb_str = f" {adverb}"
        
        if self.settings.style == 'hemingway':
            if target_str:
                return f"{opener}{actor_cap} {verb} {target_str}{adverb_str}.".strip()
            return f"{opener}{actor_cap} {verb}{adverb_str}.".strip()
        
        elif self.settings.style == 'literary':
            if target_str:
                return f"{opener}{actor_cap}, with characteristic focus, {verb} {target_str}{adverb_str}."
            return f"{opener}{actor_cap} {verb}{adverb_str}, as is typical of the character."
        
        else:  # neutral
            if target_str:
                return f"{opener}{actor_cap} {verb} {target_str}{adverb_str}.".strip()
            return f"{opener}{actor_cap} {verb}{adverb_str}.".strip()
    
    def _find_fitting_action(self, actor: str) -> Optional[str]:
        """Find an action that fits with the actor (opposite φ-direction)."""
        actor_lower = actor.lower()
        
        # First try actions the actor has performed
        if actor_lower in self.ingester.concepts:
            concept = self.ingester.concepts[actor_lower]
            if concept.actions_performed:
                return concept.actions_performed.most_common(1)[0][0]
        
        # Otherwise find fitting action by direction
        fitting = self.ingester.find_fitting_concepts(actor, n=10)
        for name, score in fitting:
            concept = self.ingester.concepts[name]
            if concept.action_count > 0:
                return name
        
        return None
    
    def _find_fitting_target(self, action: str) -> Optional[str]:
        """Find a target that fits with the action."""
        action_lower = action.lower()
        
        # Find concepts with opposite direction to action
        fitting = self.ingester.find_fitting_concepts(action, n=10)
        for name, score in fitting:
            concept = self.ingester.concepts[name]
            if concept.target_count > 0 or concept.actor_count > 0:
                return name
        
        return None
    
    def generate(self, seed: str = None, num_sentences: int = 1) -> str:
        """
        Generate text using quad-quaternion pipeline.
        
        Q1: Find fitting concepts (symmetric)
        Q3: Transform to correct form
        Q2: Apply style
        Q4: Validate (implicit in fitting)
        """
        sentences = []
        current_seed = seed.lower() if seed else None
        
        # If no seed, pick a random entity
        if not current_seed:
            entities = [n for n, c in self.ingester.concepts.items() 
                       if c.phi_direction > 0.3 and c.actor_count > 0]
            if entities:
                current_seed = random.choice(entities)
        
        for _ in range(num_sentences):
            if not current_seed or current_seed not in self.ingester.concepts:
                break
            
            # Q1: Find fitting action
            action = self._find_fitting_action(current_seed)
            if not action:
                break
            
            # Q1: Find fitting target
            target = self._find_fitting_target(action)
            
            # Q3: Conjugate verb
            verb = self._conjugate(action)
            
            # Q2: Format with style
            sentence = self._format_sentence(current_seed, verb, target)
            sentences.append(sentence)
            
            # Chain to next seed
            if target and target in self.ingester.concepts:
                concept = self.ingester.concepts[target]
                if concept.actor_count > 0:
                    current_seed = target
                else:
                    current_seed = None
            else:
                current_seed = None
        
        return " ".join(sentences) if sentences else "I don't have enough information to generate text."
    
    def ask(self, question: str) -> str:
        """Answer a question about ingested content."""
        question_lower = question.lower().strip().rstrip('?')
        
        # Detect question type
        for pattern, handler in self.question_handlers.items():
            if question_lower.startswith(pattern):
                return handler(question_lower)
        
        # Try to extract entity and describe
        words = re.findall(r'\b\w+\b', question_lower)
        for word in words:
            if word in self.ingester.concepts:
                concept = self.ingester.concepts[word]
                if concept.actor_count > 0:
                    return self._answer_describe(f"describe {word}")
        
        return "I don't have enough information to answer that question."
    
    def _answer_who(self, question: str) -> str:
        """Answer 'Who is X?' or 'Who does X?' questions."""
        # Who is X?
        match = re.search(r'who\s+is\s+(\w+)', question)
        if match:
            entity = match.group(1).lower()
            return self._describe_entity(entity)
        
        # Who does X?
        match = re.search(r'who\s+(\w+)', question)
        if match:
            action = match.group(1).lower()
            base_action = self.morpho._get_base(action)
            
            # Find entities that perform this action
            performers = []
            for name, concept in self.ingester.concepts.items():
                for act in concept.actions_performed:
                    if self.morpho._get_base(act) == base_action:
                        performers.append((name, concept.actions_performed[act]))
            
            if performers:
                performers.sort(key=lambda x: x[1], reverse=True)
                opener = self._get_certainty_opener()
                names = [p[0].title() for p in performers[:3]]
                return f"{opener}{', '.join(names)} {self._conjugate(action)}."
        
        return "I'm not sure who you're asking about."
    
    def _answer_what(self, question: str) -> str:
        """Answer 'What does X do?' questions."""
        match = re.search(r'what\s+does\s+(\w+)\s+do', question)
        if match:
            entity = match.group(1).lower()
            if entity in self.ingester.concepts:
                concept = self.ingester.concepts[entity]
                actions = list(concept.actions_performed.keys())[:3]
                if actions:
                    opener = self._get_certainty_opener()
                    verbs = [self._conjugate(a) for a in actions]
                    return f"{opener}{entity.title()} {', '.join(verbs)}."
        
        # What happened to X?
        match = re.search(r'what\s+happened\s+to\s+(\w+)', question)
        if match:
            entity = match.group(1).lower()
            if entity in self.ingester.concepts:
                concept = self.ingester.concepts[entity]
                received = list(concept.actions_received.keys())[:2]
                if received:
                    opener = self._get_certainty_opener()
                    self.set_morphology(tense='past')
                    verbs = [self._conjugate(a) for a in received]
                    self.set_morphology(tense='present')
                    return f"{opener}{entity.title()} was {', '.join(verbs)}."
        
        return "I'm not sure what you're asking about."
    
    def _answer_does(self, question: str) -> str:
        """Answer 'Does X do Y?' questions."""
        match = re.search(r'does\s+(\w+)\s+(\w+)', question)
        if not match:
            return "I'm not sure what you're asking."
        
        entity = match.group(1).lower()
        action = match.group(2).lower()
        
        if entity not in self.ingester.concepts:
            return f"I don't have information about {entity.title()}."
        
        concept = self.ingester.concepts[entity]
        base_action = self.morpho._get_base(action)
        
        for known_action in concept.actions_performed:
            if self.morpho._get_base(known_action) == base_action:
                opener = self._get_certainty_opener()
                verb = self._conjugate(known_action)
                return f"{opener}Yes, {entity.title()} {verb}."
        
        if self.settings.certainty == 'hedged':
            return f"It's unclear whether {entity.title()} does that."
        return f"No, {entity.title()} doesn't appear to do that."
    
    def _answer_is(self, question: str) -> str:
        """Answer 'Is X related to Y?' questions."""
        match = re.search(r'is\s+(\w+)\s+related\s+to\s+(\w+)', question)
        if match:
            entity1 = match.group(1).lower()
            entity2 = match.group(2).lower()
            
            if entity1 in self.ingester.concepts:
                concept = self.ingester.concepts[entity1]
                if entity2 in concept.co_targets or entity2 in concept.co_actors:
                    opener = self._get_certainty_opener()
                    return f"{opener}Yes, {entity1.title()} and {entity2.title()} are connected."
            
            return f"I don't see a direct connection between {entity1.title()} and {entity2.title()}."
        
        return "I'm not sure how to answer that question."
    
    def _answer_how(self, question: str) -> str:
        """Answer 'How does X do Y?' questions."""
        match = re.search(r'how\s+does\s+(\w+)\s+(\w+)', question)
        if match:
            entity = match.group(1).lower()
            action = match.group(2).lower()
            
            # Find frames with this actor and action
            for frame in self.ingester.frames:
                if frame.actor == entity and self.morpho._get_base(frame.action) == self.morpho._get_base(action):
                    if frame.adverb:
                        opener = self._get_certainty_opener()
                        return f"{opener}{entity.title()} {self._conjugate(action)} {frame.adverb}."
            
            return f"I don't have details about how {entity.title()} does that."
        
        return "I'm not sure how to answer that question."
    
    def _answer_why(self, question: str) -> str:
        """Answer 'Why does X do Y?' questions - limited capability."""
        opener = self._get_certainty_opener()
        return f"{opener}The text doesn't explicitly explain the motivation."
    
    def _answer_describe(self, question: str) -> str:
        """Describe an entity."""
        match = re.search(r'(?:describe|tell\s+me\s+about)\s+(\w+)', question)
        if not match:
            return "I'm not sure what to describe."
        
        entity = match.group(1).lower()
        return self._describe_entity(entity)
    
    def _describe_entity(self, entity: str) -> str:
        """Generate a description of an entity."""
        if entity not in self.ingester.concepts:
            return f"I don't have information about {entity.title()}."
        
        concept = self.ingester.concepts[entity]
        sentences = []
        opener = self._get_certainty_opener()
        
        # Main action
        if concept.actions_performed:
            action = concept.actions_performed.most_common(1)[0][0]
            verb = self._conjugate(action)
            
            # Find a target for this action
            target = None
            if concept.co_targets:
                target = concept.co_targets.most_common(1)[0][0]
            
            target_str = self._format_target(target)
            if target_str:
                sentences.append(f"{opener}{entity.title()} {verb} {target_str}.")
            else:
                sentences.append(f"{opener}{entity.title()} {verb}.")
        
        # Relationships (if moderate or elaborate)
        if self.settings.depth != 'terse':
            related = list(concept.co_targets.keys())[:2]
            if related:
                related_str = " and ".join([r.title() for r in related])
                sentences.append(f"They are connected to {related_str}.")
        
        # Additional action (if elaborate)
        if self.settings.depth == 'elaborate' and len(concept.actions_performed) > 1:
            actions = list(concept.actions_performed.keys())
            if len(actions) > 1:
                verb2 = self._conjugate(actions[1])
                sentences.append(f"They also {verb2}.")
        
        # Confidence info (if elaborate)
        if self.settings.depth == 'elaborate':
            conf = concept.confidence
            if conf > 0.5:
                sentences.append("This is well-documented in the text.")
            elif conf < 0:
                sentences.append("This is inferred from limited evidence.")
        
        return " ".join(sentences) if sentences else f"{entity.title()} appears in the text."
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        stats = self.ingester.get_statistics()
        stats['settings'] = {
            'style': self.settings.style,
            'certainty': self.settings.certainty,
            'depth': self.settings.depth,
            'tense': self.settings.tense,
            'aspect': self.settings.aspect,
        }
        return stats


def run_demo():
    """Demonstrate GeometricLCM V2."""
    print("=" * 70)
    print("GEOMETRIC LCM V2: Quad-Quaternion with Symmetric Ingestion")
    print("=" * 70)
    print()
    
    # Create model
    model = GeometricLCMv2()
    
    # Ingest expanded corpus
    print("Ingesting expanded corpus (5 literary works)...")
    model.ingest(EXPANDED_CORPUS)
    
    stats = model.get_stats()
    print(f"Learned {stats['total_concepts']} concepts")
    print(f"  Entities: {stats['entities']}")
    print(f"  Actions: {stats['actions']}")
    print(f"  Frames: {stats['total_frames']} (fit ratio: {stats['fit_ratio']:.1%})")
    print()
    
    # Test generation with different styles
    print("=" * 70)
    print("TEXT GENERATION")
    print("=" * 70)
    print()
    
    test_seeds = ['holmes', 'alice', 'darcy', 'gatsby', 'hamlet']
    
    for seed in test_seeds:
        print(f"Seed: {seed}")
        
        model.set_style(style='hemingway', certainty='definitive', depth='terse')
        model.set_morphology(tense='present')
        print(f"  [Hemingway] {model.generate(seed, 1)}")
        
        model.set_style(style='literary', certainty='hedged', depth='elaborate')
        print(f"  [Literary]  {model.generate(seed, 1)}")
        
        print()
    
    # Test morphology control
    print("=" * 70)
    print("MORPHOLOGY CONTROL (Q3)")
    print("=" * 70)
    print()
    
    model.set_style(style='neutral', certainty='neutral', depth='moderate')
    
    for tense in ['present', 'past', 'future']:
        model.set_morphology(tense=tense)
        print(f"Tense={tense}: {model.generate('holmes', 1)}")
    
    print()
    
    for aspect in ['simple', 'progressive', 'perfect']:
        model.set_morphology(tense='present', aspect=aspect)
        print(f"Aspect={aspect}: {model.generate('holmes', 1)}")
    
    print()
    
    # Test question answering
    print("=" * 70)
    print("QUESTION ANSWERING")
    print("=" * 70)
    print()
    
    model.set_style(style='neutral', certainty='neutral', depth='moderate')
    model.set_morphology(tense='present', aspect='simple')
    
    questions = [
        "Who is Holmes?",
        "Who is Gatsby?",
        "Who is Hamlet?",
        "What does Holmes do?",
        "What does Alice do?",
        "What does Darcy do?",
        "Does Holmes examine?",
        "Does Watson write?",
        "Does Alice fly?",
        "Does Hamlet kill?",
        "Is Holmes related to Watson?",
        "Is Darcy related to Elizabeth?",
        "Who examines?",
        "Who killed?",
        "Describe Moriarty",
        "Tell me about Ophelia",
    ]
    
    for question in questions:
        print(f"Q: {question}")
        answer = model.ask(question)
        print(f"A: {answer}")
        print()
    
    # Test certainty levels
    print("=" * 70)
    print("CERTAINTY LEVELS (Q2 W-axis / Tachyon)")
    print("=" * 70)
    print()
    
    question = "Who is Gatsby?"
    
    for certainty in ['definitive', 'neutral', 'hedged']:
        model.set_style(certainty=certainty)
        print(f"Certainty={certainty}")
        print(f"  Q: {question}")
        print(f"  A: {model.ask(question)}")
        print()
    
    return model


if __name__ == "__main__":
    model = run_demo()
