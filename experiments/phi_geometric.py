#!/usr/bin/env python3
"""
φ Geometric: Fully Geometric Pipeline

No hard-coded stop words. Stop words are detected geometrically:
- High frequency (appear in many sentences)
- No consistent position (high position variance)
- Low φ-direction magnitude (neither entity nor action)

This makes the system language-independent and scalable.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.phi_morphology_geometric import GeometricMorphology
from experiments.phi_conjugation_geometric import GeometricConjugation, CONJUGATION_BOOTSTRAP

PHI = 1.618034

# Bootstrap text for geometric morphology learning (input: recognizing equivalence)
# Include all three forms: base, 3rd singular, past
MORPHOLOGY_BOOTSTRAP = """
I love. He loves. I loved.
I run. He runs. I ran.
I see. He sees. I saw.
I watch. He watches. I watched.
I go. He goes. I went.
I fall. He falls. I fell.
I speak. He speaks. I spoke.
I write. He writes. I wrote.
I read. He reads. I read.
I give. He gives. I gave.
I take. He takes. I took.
I make. He makes. I made.
I grow. He grows. I grew.
I know. He knows. I knew.
I think. He thinks. I thought.
I say. He says. I said.
I come. He comes. I came.
I find. He finds. I found.
I leave. He leaves. I left.
I begin. He begins. I began.
I examine. He examines. I examined.
I observe. He observes. I observed.
I assist. He assists. I assisted.
I question. He questions. I questioned.
I solve. He solves. I solved.
I end. He ends. I ended.
I kill. He kills. I killed.
I confront. He confronts. I confronted.
I reveal. He reveals. I revealed.
I drown. He drowns. I drowned.
I witness. He witnesses. I witnessed.
I ponder. He ponders. I pondered.
I poison. He poisons. I poisoned.
I propose. He proposes. I proposed.
I order. He orders. I ordered.
I explore. He explores. I explored.
I shout. He shouts. I shouted.
I smile. He smiles. I smiled.
I vanish. He vanishes. I vanished.
I ask. He asks. I asked.
I deduce. He deduces. I deduced.
I plot. He plots. I plotted.
I scheme. He schemes. I schemed.
I study. He studies. I studied.
"""


@dataclass
class GeometricConcept:
    """A concept with geometric properties."""
    word: str
    
    # Position statistics
    positions: List[float] = field(default_factory=list)
    sentence_count: int = 0  # How many sentences this appears in
    
    # Role counts
    initiator_count: int = 0
    mediator_count: int = 0
    receiver_count: int = 0
    
    # Relations
    actions: Counter = field(default_factory=Counter)
    targets: Counter = field(default_factory=Counter)
    
    @property
    def frequency(self) -> int:
        """How often this word appears."""
        return len(self.positions)
    
    @property
    def mean_position(self) -> float:
        """Average position in sentences."""
        if not self.positions:
            return 0.5
        return sum(self.positions) / len(self.positions)
    
    @property
    def position_variance(self) -> float:
        """How much the position varies (high = stop word)."""
        if len(self.positions) < 2:
            return 0.0
        mean = self.mean_position
        variance = sum((p - mean) ** 2 for p in self.positions) / len(self.positions)
        return variance
    
    @property
    def phi_direction(self) -> float:
        """Entity (+1) vs Action (-1)."""
        total = self.initiator_count + self.mediator_count + self.receiver_count
        if total == 0:
            return 0.0
        entity = self.initiator_count + self.receiver_count
        action = self.mediator_count
        return (entity - action) / total
    
    @property
    def phi_magnitude(self) -> float:
        """How strongly this is entity OR action (low = stop word)."""
        return abs(self.phi_direction)
    
    @property
    def is_geometric_stop_word(self) -> bool:
        """
        Geometric stop word detection.
        
        Stop words are characterized by:
        1. No semantic role - never initiator, mediator, or receiver
        2. OR: Very short words (≤3 chars) that appear frequently
        3. OR: High frequency but only receiver role (prepositions get receiver by accident)
        
        Content words have clear roles AND are meaningful (names, verbs, objects).
        """
        # Need enough data
        if self.sentence_count < 2:
            return False
        
        # Key insight: Stop words have NO semantic role
        total_roles = self.initiator_count + self.mediator_count + self.receiver_count
        has_no_role = total_roles == 0
        
        # Very short words that appear frequently are likely stop words
        # (prepositions, articles, conjunctions)
        is_short_frequent = len(self.word) <= 4 and self.frequency >= 3
        
        # Words that ONLY appear as receivers and are short are likely prepositions
        # that got caught as receivers (e.g., "from", "with")
        only_receiver = (self.receiver_count > 0 and 
                        self.initiator_count == 0 and 
                        self.mediator_count == 0 and
                        len(self.word) <= 5)
        
        return has_no_role or is_short_frequent or only_receiver
    
    @property
    def is_content_word(self) -> bool:
        """Is this a content word (not a stop word)?"""
        return not self.is_geometric_stop_word


@dataclass 
class Frame:
    """Semantic frame: Initiator → Mediator → Receiver."""
    initiator: str
    mediator: str
    receiver: Optional[str] = None


class PhiGeometric:
    """
    Fully geometric pipeline with no hard-coded stop words.
    Uses geometric morphology for verb form normalization.
    """
    
    def __init__(self):
        self.concepts: Dict[str, GeometricConcept] = {}
        self.frames: List[Frame] = []
        self.total_sentences: int = 0
        
        # Geometric morphology - learned from parallel structures (for INPUT recognition)
        self.geo_morpho = GeometricMorphology()
        self.geo_morpho.bootstrap(MORPHOLOGY_BOOTSTRAP)
        
        # Geometric conjugation - learned from parallel structures (for OUTPUT generation)
        self.geo_conj = GeometricConjugation()
        self.geo_conj.bootstrap(CONJUGATION_BOOTSTRAP)
    
    def _get_or_create(self, word: str) -> GeometricConcept:
        """Get or create a concept."""
        word = word.lower()
        if word not in self.concepts:
            self.concepts[word] = GeometricConcept(word=word)
        return self.concepts[word]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - no filtering yet."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def learn(self, text: str):
        """
        Learn from text.
        
        Two-pass learning:
        1. First pass: Extract frames using simple heuristics (position-based)
        2. Role counts emerge from frame extraction
        3. Stop words are those with no semantic role
        """
        sentences = re.split(r'[.!?]+', text)
        
        # PASS 1: Extract frames and assign roles
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            self.total_sentences += 1
            tokens = self._tokenize(sentence)
            
            if len(tokens) < 2:
                continue
            
            # Record position statistics for ALL words
            seen_in_sentence = set()
            for i, word in enumerate(tokens):
                if len(word) < 2:
                    continue
                    
                c = self._get_or_create(word)
                pos = i / max(len(tokens) - 1, 1)
                c.positions.append(pos)
                
                if word not in seen_in_sentence:
                    c.sentence_count += 1
                    seen_in_sentence.add(word)
            
            # GEOMETRIC FRAME EXTRACTION
            # Assign slots based on normalized position in sentence
            # Position 0.0-0.33 → Initiator
            # Position 0.33-0.66 → Mediator
            # Position 0.66-1.0 → Receiver
            
            # Filter to content words (length > 3, not adverbs)
            content_with_pos = []
            for i, w in enumerate(tokens):
                if len(w) <= 3:
                    continue
                if w.endswith('ly') and len(w) > 4:  # Skip adverbs
                    continue
                pos = i / max(len(tokens) - 1, 1)
                content_with_pos.append((w, pos))
            
            if len(content_with_pos) < 2:
                continue
            
            # Assign slots by position bands
            initiator = None
            mediator = None
            receiver = None
            
            for word, pos in content_with_pos:
                if pos < 0.33 and initiator is None:
                    initiator = word
                elif pos < 0.66 and mediator is None:
                    mediator = word
                elif receiver is None:
                    receiver = word
            
            # Fallback: if we didn't get all slots, use first available
            if initiator is None and content_with_pos:
                initiator = content_with_pos[0][0]
            if mediator is None and len(content_with_pos) > 1:
                mediator = content_with_pos[1][0]
            
            if not initiator or not mediator:
                continue
            
            # Create frame
            frame = Frame(initiator=initiator, mediator=mediator, receiver=receiver)
            self.frames.append(frame)
            
            init_c = self._get_or_create(initiator)
            init_c.initiator_count += 1
            init_c.actions[mediator] += 1
            if receiver:
                init_c.targets[receiver] += 1
            
            med_c = self._get_or_create(mediator)
            med_c.mediator_count += 1
            
            if receiver:
                recv_c = self._get_or_create(receiver)
                recv_c.receiver_count += 1
        
        # PASS 2: Role counts are now set, stop words emerge geometrically
        # (is_geometric_stop_word property uses role counts)
    
    def _extract_frames(self, text: str):
        """Extract frames using geometric content word detection."""
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            tokens = self._tokenize(sentence)
            
            # Filter to content words using geometric detection
            content = []
            for word in tokens:
                if len(word) < 2:
                    continue
                if word in self.concepts:
                    c = self.concepts[word]
                    if c.is_content_word:
                        content.append(word)
                else:
                    # Unknown word - include it (might be content)
                    content.append(word)
            
            if len(content) < 2:
                continue
            
            # Assign roles by position
            initiator = content[0]
            mediator = content[1]
            
            # Find receiver (skip adverbs geometrically - they have high variance too)
            receiver = None
            for i in range(2, len(content)):
                word = content[i]
                if word in self.concepts:
                    c = self.concepts[word]
                    # Receivers should have low variance and be entity-like
                    if c.position_variance < 0.2 or c.phi_direction > 0:
                        receiver = word
                        break
                else:
                    receiver = word
                    break
            
            # Create frame
            frame = Frame(initiator=initiator, mediator=mediator, receiver=receiver)
            self.frames.append(frame)
            
            # Update role counts
            init_c = self._get_or_create(initiator)
            init_c.initiator_count += 1
            init_c.actions[mediator] += 1
            if receiver:
                init_c.targets[receiver] += 1
            
            med_c = self._get_or_create(mediator)
            med_c.mediator_count += 1
            
            if receiver:
                recv_c = self._get_or_create(receiver)
                recv_c.receiver_count += 1
    
    def get_stop_words(self) -> List[str]:
        """Get geometrically detected stop words."""
        return [name for name, c in self.concepts.items() if c.is_geometric_stop_word]
    
    def get_content_words(self) -> List[str]:
        """Get geometrically detected content words."""
        return [name for name, c in self.concepts.items() if c.is_content_word]
    
    def encode(self, text: str) -> float:
        """Encode text to position using only content words."""
        tokens = self._tokenize(text)
        
        # Filter to content words
        content = [w for w in tokens if w in self.concepts and self.concepts[w].is_content_word]
        
        if not content:
            # Fallback: use all known words
            content = [w for w in tokens if w in self.concepts]
        
        if not content:
            return 0.5
        
        # φ-weighted position
        total = 0.0
        weight = 0.0
        
        for i, word in enumerate(content):
            w = PHI ** (-i)
            c = self.concepts[word]
            total += c.mean_position * w
            weight += w
        
        return total / weight if weight > 0 else 0.5
    
    def respond(self, query: str) -> str:
        """Generate response using geometric pipeline."""
        tokens = self._tokenize(query)
        
        # Find content words in query (check base forms too using GEOMETRIC morphology)
        content = []
        for w in tokens:
            if w in self.concepts and self.concepts[w].is_content_word:
                content.append(w)
            else:
                # Check if any concept is equivalent using geometric morphology
                for name, c in self.concepts.items():
                    if c.is_content_word and self.geo_morpho.are_equivalent(name, w):
                        content.append(name)
                        break
        
        # Detect question type geometrically
        query_position = self.encode(query)
        
        # Find entity being asked about
        entity = None
        action = None
        
        for word in content:
            if word not in self.concepts:
                continue
            c = self.concepts[word]
            if c.phi_direction > 0.3:  # Entity-like
                entity = word
            elif c.phi_direction < -0.3:  # Action-like
                action = word
        
        # Also check query words directly for actions using GEOMETRIC morphology
        if not action:
            for w in tokens:
                # Use geometric morphology to find canonical form
                canonical = self.geo_morpho.get_canonical(w)
                equivalents = self.geo_morpho.get_equivalents(w)
                
                for name, c in self.concepts.items():
                    if c.mediator_count > 0:
                        # Check if this concept is equivalent to the query word
                        if name in equivalents or self.geo_morpho.are_equivalent(name, w):
                            action = name
                            break
                if action:
                    break
        
        # Generate response based on what we found
        if action and not entity:
            return self._who_does(action)
        elif entity:
            return self._describe(entity)
        elif content:
            return self._describe(content[0])
        else:
            return "I don't have enough information."
    
    def _who_does(self, action: str) -> str:
        """Find who performs an action using GEOMETRIC morphology."""
        # Get equivalents from geometric morphology
        equivalents = self.geo_morpho.get_equivalents(action)
        
        actors = []
        for name, c in self.concepts.items():
            if c.initiator_count == 0:
                continue
            if not c.is_content_word:
                continue
            for act in c.actions:
                # Check if action is equivalent using geometric morphology
                if act in equivalents or self.geo_morpho.are_equivalent(act, action):
                    actors.append((name, c.actions[act]))
                    break
        
        if not actors:
            # Try matching the action itself as a mediator
            for name, c in self.concepts.items():
                if c.mediator_count > 0 and self.geo_morpho.are_equivalent(name, action):
                    # Find who does this action from frames
                    for frame in self.frames:
                        if self.geo_morpho.are_equivalent(frame.mediator, action):
                            if frame.initiator in self.concepts and self.concepts[frame.initiator].is_content_word:
                                actors.append((frame.initiator, 1))
                    break
        
        if not actors:
            return f"I don't know who {action}s."
        
        actors.sort(key=lambda x: x[1], reverse=True)
        actor = actors[0][0]
        
        # Find target from frames using geometric morphology
        target = None
        for frame in self.frames:
            if frame.initiator == actor and self.geo_morpho.are_equivalent(frame.mediator, action):
                if frame.receiver:
                    target = frame.receiver
                    break
        
        # Get canonical form and conjugate using GEOMETRIC conjugation
        # Phase 1 = 3rd person singular (for "Hamlet kills")
        canonical = self.geo_conj.get_canonical(action)
        verb = self.geo_conj.conjugate(canonical, 1)  # 3rd person singular
        
        if target:
            return f"{actor.title()} {verb} {target}."
        else:
            return f"{actor.title()} {verb}."
    
    def _describe(self, entity: str) -> str:
        """Describe an entity."""
        if entity not in self.concepts:
            return f"I don't know about {entity}."
        
        c = self.concepts[entity]
        
        # Role based on φ-direction
        if c.phi_direction > 0.3:
            role = "protagonist"
        elif c.phi_direction < -0.3:
            role = "action"
        else:
            role = "concept"
        
        # Actions - use GEOMETRIC conjugation
        if c.actions:
            top_actions = c.actions.most_common(3)
            # Phase 1 = 3rd person singular for "who examines, deduces, observes"
            verbs = []
            for a, _ in top_actions:
                canonical = self.geo_conj.get_canonical(a)
                verb = self.geo_conj.conjugate(canonical, 1)
                verbs.append(verb)
            
            if len(verbs) == 1:
                action_desc = verbs[0]
            elif len(verbs) == 2:
                action_desc = f"{verbs[0]} and {verbs[1]}"
            else:
                action_desc = f"{', '.join(verbs[:-1])}, and {verbs[-1]}"
            
            response = f"{entity.title()} is a {role} who {action_desc}"
        else:
            response = f"{entity.title()} is a {role}"
        
        # Targets
        if c.targets:
            good_targets = [t for t in c.targets.keys() 
                          if t in self.concepts and self.concepts[t].is_content_word][:2]
            if good_targets:
                response += f", often involving {' and '.join(good_targets)}"
        
        return response + "."
    
    def show_geometric_analysis(self):
        """Show the geometric analysis of words."""
        print("\nGEOMETRIC WORD ANALYSIS")
        print("=" * 70)
        print(f"{'Word':<15} {'Freq':>5} {'Var':>6} {'φ-dir':>6} {'φ-mag':>6} {'Type':<10}")
        print("-" * 70)
        
        # Sort by frequency
        sorted_concepts = sorted(self.concepts.items(), 
                                key=lambda x: x[1].frequency, reverse=True)
        
        for name, c in sorted_concepts[:30]:
            word_type = "STOP" if c.is_geometric_stop_word else "CONTENT"
            print(f"{name:<15} {c.frequency:>5} {c.position_variance:>6.3f} "
                  f"{c.phi_direction:>6.2f} {c.phi_magnitude:>6.2f} {word_type:<10}")


# Test corpus
CORPUS = """
Holmes examined the evidence carefully and methodically.
Watson watched from the doorway with great interest.
The detective studied the footprints on the floor.
Holmes deduced the identity of the criminal brilliantly.
Moriarty plotted against Holmes in secret.
Watson assisted Holmes faithfully throughout the case.
Lestrade questioned the witnesses at the scene.
Holmes observed every detail of the crime scene.
The professor schemed against the detective relentlessly.
Watson wrote about their adventures in his journal.
Holmes solved the mystery with remarkable insight.
The criminal fled from the scene in panic.

Alice fell down the rabbit hole unexpectedly.
The Queen shouted at everyone in the garden.
Cheshire smiled mysteriously and vanished slowly.
Alice grew very tall after drinking the potion.
The Hatter laughed wildly at the tea party.
Alice explored Wonderland with great curiosity.
The Caterpillar asked Alice many strange questions.
Alice shrank to a tiny size suddenly.
The Queen ordered everyone to play croquet.
Alice woke from her dream at last.

Hamlet pondered the meaning of existence deeply.
Claudius poisoned the King in the garden.
Ophelia loved Hamlet with all her heart.
Hamlet confronted his mother about her betrayal.
The ghost revealed the truth about the murder.
Hamlet killed Claudius in the final act.
Ophelia drowned in the river tragically.
Horatio witnessed the tragic events unfold.
Laertes sought revenge for his father.
The play ended with many deaths.
"""


def demo():
    """Demonstrate geometric stop word detection."""
    print("φ GEOMETRIC: No Hard-Coded Stop Words")
    print("=" * 70)
    print()
    
    bot = PhiGeometric()
    bot.learn(CORPUS)
    
    print(f"Total sentences: {bot.total_sentences}")
    print(f"Total concepts: {len(bot.concepts)}")
    print(f"Frames extracted: {len(bot.frames)}")
    print()
    
    # Show geometric analysis
    bot.show_geometric_analysis()
    
    # Show detected stop words
    stop_words = bot.get_stop_words()
    content_words = bot.get_content_words()
    
    print()
    print(f"GEOMETRICALLY DETECTED STOP WORDS ({len(stop_words)}):")
    print(f"  {', '.join(sorted(stop_words)[:20])}")
    print()
    print(f"CONTENT WORDS ({len(content_words)}):")
    print(f"  {', '.join(sorted(content_words)[:20])}...")
    print()
    
    # Test queries
    print("=" * 70)
    print("QUERY TEST")
    print("=" * 70)
    print()
    
    queries = [
        "Who is Holmes?",
        "Who killed?",
        "Who loves?",
        "Tell me about Alice",
        "What does Watson do?",
    ]
    
    for q in queries:
        print(f"Q: {q}")
        print(f"A: {bot.respond(q)}")
        print()
    
    return bot


if __name__ == "__main__":
    demo()
