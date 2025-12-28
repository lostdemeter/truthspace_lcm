#!/usr/bin/env python3
"""
Symmetric Ingestion V2

Improved text ingestion using our symmetric understanding:
1. φ-Direction detection from word structure and role
2. Polyomino validation (co-occurring concepts should have opposite directions)
3. Tachyon confidence (forward = confirmed, backward = hypothesis)
4. Bidirectional frame extraction

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = 1.618034

# Expanded corpus with more variety
EXPANDED_CORPUS = """
# Sherlock Holmes
Holmes examined the evidence carefully. Watson watched from the doorway.
The detective studied the footprints in the mud. He noticed something unusual about the pattern.
Holmes said to Watson that the case was elementary. Watson replied that he did not understand.
The inspector arrived at the scene promptly. Lestrade questioned the witnesses thoroughly.
Holmes observed the room methodically. He found a crucial clue near the window.
Watson wrote in his journal diligently. The doctor recorded every detail with precision.
Holmes deduced the killer's identity brilliantly. He explained his reasoning to the amazed audience.
The criminal fled through the garden desperately. Holmes pursued him quickly through the night.
Watson called for help immediately. The police surrounded the building completely.
Holmes captured the villain triumphantly. Justice was served at last.
Moriarty plotted against Holmes secretly. The professor was a criminal mastermind.
Mrs Hudson prepared tea for the gentlemen. She worried about their dangerous adventures.
Mycroft advised his brother occasionally. The government official had vast knowledge.

# Alice in Wonderland
Alice fell down the rabbit hole unexpectedly. She wondered where she was going.
The Queen shouted angrily at everyone. Alice felt confused and scared.
The Cheshire Cat smiled mysteriously at Alice. He disappeared slowly into thin air.
Alice grew very tall suddenly. She shrank very small moments later.
The Mad Hatter laughed wildly at the party. He poured more tea endlessly.
The White Rabbit hurried past anxiously. He checked his watch constantly.
The Caterpillar smoked his hookah thoughtfully. He asked Alice strange questions.
The Dormouse slept peacefully at the table. He woke briefly to tell stories.
The March Hare acted quite mad indeed. He threw butter at the Dormouse.
Alice played croquet with the Queen. The flamingos served as mallets.

# Pride and Prejudice
Darcy looked at Elizabeth proudly. She ignored him completely at first.
Elizabeth danced gracefully at the ball. Darcy watched her intently from afar.
Mr Bennet read his newspaper quietly. Mrs Bennet worried about her daughters constantly.
Jane smiled sweetly at everyone. Bingley fell in love immediately with her.
Wickham deceived Elizabeth cunningly. He told lies about Darcy's character.
Lady Catherine visited Longbourn unexpectedly. She demanded Elizabeth refuse Darcy.
Lydia eloped with Wickham foolishly. The scandal threatened the family's reputation.
Darcy saved the Bennet family secretly. He paid Wickham to marry Lydia.
Elizabeth realized her mistake gradually. She began to appreciate Darcy's true character.
Darcy proposed to Elizabeth again humbly. She accepted him joyfully this time.

# The Great Gatsby
Gatsby watched the green light longingly. He dreamed of Daisy constantly.
Nick observed the wealthy parties curiously. He narrated the events thoughtfully.
Daisy cried over Gatsby's beautiful shirts. She had married Tom for money.
Tom confronted Gatsby aggressively. He revealed Gatsby's criminal connections.
Myrtle died in the accident tragically. Gatsby's car struck her on the road.
Wilson shot Gatsby in the pool. He believed Gatsby had killed Myrtle.
Nick arranged Gatsby's funeral alone. Nobody else attended the service.
The green light symbolized hope eternally. Gatsby reached for it desperately.

# Hamlet
Hamlet pondered existence deeply. He questioned whether to live or die.
The ghost appeared to Hamlet mysteriously. He revealed Claudius murdered his father.
Claudius poisoned King Hamlet treacherously. He married Gertrude immediately after.
Ophelia loved Hamlet devotedly. She went mad from grief eventually.
Polonius spied on Hamlet foolishly. Hamlet killed him behind the curtain.
Laertes sought revenge passionately. He challenged Hamlet to a duel.
Gertrude drank the poisoned wine accidentally. She died before Hamlet's eyes.
Hamlet killed Claudius finally. He avenged his father at last.
Fortinbras arrived at Elsinore victoriously. He claimed the Danish throne.
"""


@dataclass
class SymmetricConcept:
    """A concept with symmetric properties."""
    word: str
    
    # Role-based direction (from frame analysis)
    actor_count: int = 0
    action_count: int = 0
    target_count: int = 0
    
    # Structure-based signals
    vowel_ratio: float = 0.0
    consonant_clusters: int = 0
    syllable_estimate: int = 1
    
    # Computed φ-direction
    phi_direction: float = 0.0
    
    # Confidence (tachyon axis)
    forward_evidence: int = 0   # Directly observed
    backward_evidence: int = 0  # Inferred/hypothesized
    
    # Relationships
    co_actors: Counter = field(default_factory=Counter)
    co_targets: Counter = field(default_factory=Counter)
    actions_performed: Counter = field(default_factory=Counter)
    actions_received: Counter = field(default_factory=Counter)
    
    @property
    def confidence(self) -> float:
        """Tachyon confidence: positive = confirmed, negative = hypothesis."""
        total = self.forward_evidence + self.backward_evidence
        if total == 0:
            return 0.0
        return (self.forward_evidence - self.backward_evidence) / total
    
    @property
    def role_direction(self) -> float:
        """Direction based on grammatical role."""
        entity_count = self.actor_count + self.target_count
        total = entity_count + self.action_count
        if total == 0:
            return 0.0
        return (entity_count - self.action_count) / total
    
    def compute_phi_direction(self):
        """Compute φ-direction from multiple signals."""
        # Primary: role-based direction
        role_signal = self.role_direction
        
        # Secondary: structural signals (verbs tend to have certain patterns)
        # Verbs often end in -ed, -ing, -s (conjugated forms)
        # Nouns often have more consonant clusters
        struct_signal = 0.0
        if self.word.endswith(('ed', 'ing', 'es', 'ies')):
            struct_signal = -0.3  # Likely verb form
        elif self.consonant_clusters > 1:
            struct_signal = 0.2  # Likely noun
        
        # Combine signals (role is primary)
        self.phi_direction = 0.7 * role_signal + 0.3 * struct_signal


@dataclass
class SymmetricFrame:
    """A frame with symmetric validation."""
    actor: str
    action: str
    target: Optional[str]
    adverb: Optional[str] = None
    
    # Symmetric properties
    fit_score: float = 0.0      # How well pieces fit (opposite directions)
    confidence: float = 0.0     # Tachyon confidence
    source_sentence: str = ""


class SymmetricIngester:
    """
    Ingest text using symmetric understanding.
    
    Key improvements:
    1. Detect φ-direction from both role and structure
    2. Validate frames using polyomino fitting
    3. Track confidence (tachyon axis)
    4. Extract richer relationships
    """
    
    def __init__(self):
        self.concepts: Dict[str, SymmetricConcept] = {}
        self.frames: List[SymmetricFrame] = []
        self.sentences: List[str] = []
        
        # Pattern detection
        self.verb_patterns = re.compile(r'\b\w+(ed|ing|es|ies|s)\b')
        self.adverb_pattern = re.compile(r'\b\w+ly\b')
        
        # Function words to skip
        self.function_words = {
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
            'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their',
            'that', 'this', 'these', 'those', 'which', 'who', 'whom',
            'and', 'or', 'but', 'if', 'then', 'so', 'as', 'than',
            'very', 'more', 'most', 'down', 'up', 'out', 'about',
            'had', 'has', 'have', 'did', 'do', 'does', 'would', 'could', 'should',
            'not', 'no', 'yes', 'all', 'some', 'any', 'each', 'every',
        }
    
    def _analyze_structure(self, word: str) -> Tuple[float, int, int]:
        """Analyze word structure for φ-direction signals."""
        vowels = set('aeiou')
        
        # Vowel ratio
        vowel_count = sum(1 for c in word.lower() if c in vowels)
        vowel_ratio = vowel_count / len(word) if word else 0
        
        # Consonant clusters (2+ consonants in a row)
        clusters = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{2,}', word.lower()))
        
        # Syllable estimate (rough: count vowel groups)
        syllables = len(re.findall(r'[aeiou]+', word.lower()))
        syllables = max(1, syllables)
        
        return vowel_ratio, clusters, syllables
    
    def _get_or_create_concept(self, word: str) -> SymmetricConcept:
        """Get or create a concept with structural analysis."""
        word_lower = word.lower()
        
        if word_lower not in self.concepts:
            vowel_ratio, clusters, syllables = self._analyze_structure(word_lower)
            self.concepts[word_lower] = SymmetricConcept(
                word=word_lower,
                vowel_ratio=vowel_ratio,
                consonant_clusters=clusters,
                syllable_estimate=syllables,
            )
        
        return self.concepts[word_lower]
    
    def _extract_frame(self, sentence: str) -> Optional[SymmetricFrame]:
        """Extract actor-action-target frame from sentence."""
        # Tokenize
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        
        # Filter to content words
        content = []
        adverb = None
        
        for token in tokens:
            if token in self.function_words:
                continue
            if token.endswith('ly') and len(token) > 3:
                adverb = token
                continue
            if len(token) > 2:
                content.append(token)
        
        if len(content) < 2:
            return None
        
        # Heuristic: first content word is actor, second is action
        actor = content[0]
        action = content[1]
        target = content[2] if len(content) > 2 else None
        
        # Filter bad targets
        if target and target.endswith('ly'):
            target = content[3] if len(content) > 3 else None
        
        return SymmetricFrame(
            actor=actor,
            action=action,
            target=target,
            adverb=adverb,
            source_sentence=sentence.strip(),
        )
    
    def _compute_fit_score(self, frame: SymmetricFrame) -> float:
        """Compute polyomino fit score for a frame."""
        actor_concept = self.concepts.get(frame.actor)
        action_concept = self.concepts.get(frame.action)
        
        if not actor_concept or not action_concept:
            return 0.0
        
        # Opposite directions = good fit
        dir_product = actor_concept.phi_direction * action_concept.phi_direction
        
        if dir_product < 0:
            # Opposite directions - good fit
            return 1.0 - abs(dir_product)  # Closer to -1 = better fit
        elif dir_product == 0:
            return 0.5  # Neutral
        else:
            return 0.0  # Same direction - bad fit
    
    def _update_relationships(self, frame: SymmetricFrame):
        """Update concept relationships from frame."""
        actor = self._get_or_create_concept(frame.actor)
        action = self._get_or_create_concept(frame.action)
        
        # Update role counts
        actor.actor_count += 1
        action.action_count += 1
        
        # Update relationships
        actor.actions_performed[frame.action] += 1
        
        # Forward evidence (directly observed)
        actor.forward_evidence += 1
        action.forward_evidence += 1
        
        if frame.target:
            target = self._get_or_create_concept(frame.target)
            target.target_count += 1
            target.forward_evidence += 1
            
            # Co-occurrence
            actor.co_targets[frame.target] += 1
            target.actions_received[frame.action] += 1
            
            # Infer reverse relationship (backward evidence)
            target.backward_evidence += 1
    
    def ingest(self, text: str):
        """
        Ingest text with symmetric understanding.
        
        Process:
        1. Split into sentences
        2. Extract frames
        3. Update concept roles and relationships
        4. Compute φ-directions
        5. Validate frames with polyomino fitting
        """
        # Clean text (remove comments)
        lines = [l for l in text.split('\n') if not l.strip().startswith('#')]
        text = ' '.join(lines)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # First pass: extract frames and update roles
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            self.sentences.append(sentence)
            
            frame = self._extract_frame(sentence)
            if frame:
                self._update_relationships(frame)
                self.frames.append(frame)
        
        # Second pass: compute φ-directions
        for concept in self.concepts.values():
            concept.compute_phi_direction()
        
        # Third pass: compute fit scores
        for frame in self.frames:
            frame.fit_score = self._compute_fit_score(frame)
            
            # Confidence based on fit
            if frame.fit_score > 0.5:
                frame.confidence = frame.fit_score
            else:
                frame.confidence = -frame.fit_score
    
    def get_entity_profile(self, name: str) -> Optional[Dict]:
        """Get detailed profile for an entity."""
        name_lower = name.lower()
        if name_lower not in self.concepts:
            return None
        
        concept = self.concepts[name_lower]
        
        return {
            'name': name,
            'phi_direction': concept.phi_direction,
            'role_direction': concept.role_direction,
            'confidence': concept.confidence,
            'actor_count': concept.actor_count,
            'action_count': concept.action_count,
            'target_count': concept.target_count,
            'top_actions': concept.actions_performed.most_common(5),
            'top_targets': concept.co_targets.most_common(5),
            'actions_received': concept.actions_received.most_common(5),
        }
    
    def find_fitting_concepts(self, seed: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find concepts that fit with the seed (opposite φ-direction)."""
        seed_lower = seed.lower()
        if seed_lower not in self.concepts:
            return []
        
        seed_dir = self.concepts[seed_lower].phi_direction
        
        # Find concepts with opposite direction
        candidates = []
        for name, concept in self.concepts.items():
            if name == seed_lower:
                continue
            
            # Opposite direction = good fit
            fit = -seed_dir * concept.phi_direction
            if fit > 0:
                candidates.append((name, fit))
        
        # Sort by fit score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]
    
    def validate_frame(self, actor: str, action: str, target: str = None) -> Dict:
        """Validate a proposed frame using symmetric constraints."""
        result = {
            'valid': True,
            'fit_score': 0.0,
            'issues': [],
        }
        
        actor_c = self.concepts.get(actor.lower())
        action_c = self.concepts.get(action.lower())
        
        if not actor_c:
            result['issues'].append(f"Unknown actor: {actor}")
            result['valid'] = False
        
        if not action_c:
            result['issues'].append(f"Unknown action: {action}")
            result['valid'] = False
        
        if actor_c and action_c:
            # Check direction compatibility
            dir_product = actor_c.phi_direction * action_c.phi_direction
            
            if dir_product > 0.3:
                result['issues'].append(
                    f"Direction mismatch: {actor} ({actor_c.phi_direction:+.2f}) "
                    f"and {action} ({action_c.phi_direction:+.2f}) have same direction"
                )
                result['valid'] = False
            
            result['fit_score'] = 1.0 if dir_product < 0 else 0.0
        
        if target:
            target_c = self.concepts.get(target.lower())
            if not target_c:
                result['issues'].append(f"Unknown target: {target}")
            elif action_c:
                # Action should have opposite direction from target too
                if action_c.phi_direction * target_c.phi_direction > 0.3:
                    result['issues'].append(
                        f"Action-target direction mismatch"
                    )
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get ingestion statistics."""
        # Count by direction
        entities = [c for c in self.concepts.values() if c.phi_direction > 0.3]
        actions = [c for c in self.concepts.values() if c.phi_direction < -0.3]
        neutral = [c for c in self.concepts.values() if abs(c.phi_direction) <= 0.3]
        
        # Frame fit statistics
        good_fits = [f for f in self.frames if f.fit_score > 0.5]
        
        return {
            'total_concepts': len(self.concepts),
            'entities': len(entities),
            'actions': len(actions),
            'neutral': len(neutral),
            'total_frames': len(self.frames),
            'good_fit_frames': len(good_fits),
            'fit_ratio': len(good_fits) / len(self.frames) if self.frames else 0,
            'total_sentences': len(self.sentences),
        }


def run_demo():
    """Demonstrate symmetric ingestion."""
    print("=" * 70)
    print("SYMMETRIC INGESTION V2")
    print("=" * 70)
    print()
    print("Using expanded corpus with multiple literary works:")
    print("  - Sherlock Holmes")
    print("  - Alice in Wonderland")
    print("  - Pride and Prejudice")
    print("  - The Great Gatsby")
    print("  - Hamlet")
    print()
    
    # Create ingester and process corpus
    ingester = SymmetricIngester()
    ingester.ingest(EXPANDED_CORPUS)
    
    # Show statistics
    stats = ingester.get_statistics()
    print("=" * 70)
    print("INGESTION STATISTICS")
    print("=" * 70)
    print()
    print(f"Total concepts: {stats['total_concepts']}")
    print(f"  Entities (φ > 0.3): {stats['entities']}")
    print(f"  Actions (φ < -0.3): {stats['actions']}")
    print(f"  Neutral: {stats['neutral']}")
    print()
    print(f"Total frames: {stats['total_frames']}")
    print(f"  Good fits (score > 0.5): {stats['good_fit_frames']}")
    print(f"  Fit ratio: {stats['fit_ratio']:.1%}")
    print()
    
    # Show entity profiles
    print("=" * 70)
    print("ENTITY PROFILES")
    print("=" * 70)
    print()
    
    test_entities = ['holmes', 'watson', 'alice', 'darcy', 'gatsby', 'hamlet', 'moriarty']
    
    for name in test_entities:
        profile = ingester.get_entity_profile(name)
        if profile:
            print(f"{name.title()}:")
            print(f"  φ-direction: {profile['phi_direction']:+.2f} "
                  f"({'entity' if profile['phi_direction'] > 0 else 'action-like'})")
            print(f"  Confidence: {profile['confidence']:+.2f} "
                  f"({'confirmed' if profile['confidence'] > 0 else 'hypothesized'})")
            print(f"  Roles: actor={profile['actor_count']}, "
                  f"action={profile['action_count']}, target={profile['target_count']}")
            if profile['top_actions']:
                actions = [a[0] for a in profile['top_actions'][:3]]
                print(f"  Top actions: {', '.join(actions)}")
            if profile['top_targets']:
                targets = [t[0] for t in profile['top_targets'][:3]]
                print(f"  Top targets: {', '.join(targets)}")
            print()
    
    # Show fitting concepts
    print("=" * 70)
    print("POLYOMINO FITTING")
    print("=" * 70)
    print()
    
    for seed in ['holmes', 'examined', 'alice', 'loved']:
        fitting = ingester.find_fitting_concepts(seed, n=5)
        if fitting:
            seed_dir = ingester.concepts[seed].phi_direction
            print(f"{seed} (φ={seed_dir:+.2f}) fits with:")
            for name, score in fitting:
                concept = ingester.concepts[name]
                print(f"  {name:15} (φ={concept.phi_direction:+.2f}, fit={score:.2f})")
            print()
    
    # Validate frames
    print("=" * 70)
    print("FRAME VALIDATION")
    print("=" * 70)
    print()
    
    test_frames = [
        ('holmes', 'examined', 'evidence'),
        ('evidence', 'examined', 'holmes'),  # Wrong direction
        ('alice', 'fell', 'rabbit'),
        ('darcy', 'loved', 'elizabeth'),
        ('hamlet', 'killed', 'claudius'),
    ]
    
    for actor, action, target in test_frames:
        result = ingester.validate_frame(actor, action, target)
        status = "✓" if result['valid'] else "✗"
        print(f"{status} {actor} → {action} → {target}")
        print(f"    Fit score: {result['fit_score']:.2f}")
        if result['issues']:
            for issue in result['issues']:
                print(f"    ⚠️  {issue}")
        print()
    
    return ingester


if __name__ == "__main__":
    ingester = run_demo()
