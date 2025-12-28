#!/usr/bin/env python3
"""
φ Quality: Combined Encoder-Decoder with Grammar

Combines:
- φ encoding (position in concept space)
- Relations graph (what connects to what)
- Morphological Q3 (verb conjugation)
- Frame structure (Initiator → Mediator → Receiver)

The goal: Generate grammatical, meaningful responses.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.morphological_quaternion import MorphologicalTransformer, MorphoQuaternion

PHI = 1.618034


@dataclass
class Concept:
    """A concept with position and role information."""
    word: str
    position: float = 0.5
    
    # Role counts
    initiator_count: int = 0
    mediator_count: int = 0
    receiver_count: int = 0
    
    # What this concept does/receives
    actions: Counter = field(default_factory=Counter)
    targets: Counter = field(default_factory=Counter)
    
    # Co-occurrence
    co_occurs: Counter = field(default_factory=Counter)
    
    @property
    def is_entity(self) -> bool:
        """Is this primarily an entity (actor/target)?"""
        return (self.initiator_count + self.receiver_count) > self.mediator_count
    
    @property
    def is_action(self) -> bool:
        """Is this primarily an action?"""
        return self.mediator_count > (self.initiator_count + self.receiver_count)
    
    @property
    def primary_role(self) -> str:
        """What role does this concept primarily play?"""
        if self.initiator_count >= self.mediator_count and self.initiator_count >= self.receiver_count:
            return "initiator"
        elif self.mediator_count >= self.receiver_count:
            return "mediator"
        else:
            return "receiver"


@dataclass
class Frame:
    """A semantic frame: Initiator → Mediator → Receiver."""
    initiator: str
    mediator: str
    receiver: Optional[str] = None
    
    def to_sentence(self, morpho: MorphologicalTransformer, tense: str = "present") -> str:
        """Convert frame to a grammatical sentence."""
        # Get the right quaternion for tense
        if tense == "present":
            q = MorphoQuaternion(x=1, y=-1, z=0, w=-1)  # 3rd singular present
        elif tense == "past":
            q = MorphoQuaternion(x=-1, y=-1, z=0, w=-1)  # past
        else:
            q = MorphoQuaternion(x=1, y=-1, z=0, w=-1)
        
        # Conjugate the verb
        verb = morpho.transform(morpho._get_base(self.mediator), q)
        
        # Build sentence
        if self.receiver:
            return f"{self.initiator.title()} {verb} {self.receiver}."
        else:
            return f"{self.initiator.title()} {verb}."


class PhiQuality:
    """
    Quality chatbot using φ encoding + frames + morphology.
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.frames: List[Frame] = []
        self.morpho = MorphologicalTransformer()
        
        # Question patterns
        self.question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'does', 'did', 'is', 'are', 'was', 'were'}
        
        # Stop words
        self.stop_words = {
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
            'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their',
            'that', 'this', 'these', 'those', 'which', 'who', 'whom',
            'and', 'or', 'but', 'if', 'then', 'so', 'as', 'than',
            'very', 'more', 'most', 'down', 'up', 'out', 'about',
            'had', 'has', 'have', 'did', 'do', 'does', 'would', 'could', 'should',
            'not', 'no', 'yes', 'all', 'some', 'any', 'each', 'every',
            'me', 'tell', 'about', 'please', 'can', 'you',
            'several', 'many', 'few', 'other', 'another', 'such',
            'after', 'before', 'during', 'through', 'into', 'onto',
            'against', 'between', 'among', 'within', 'without',
        }
    
    def _get_or_create(self, word: str) -> Concept:
        """Get or create a concept."""
        word = word.lower()
        if word not in self.concepts:
            self.concepts[word] = Concept(word=word)
        return self.concepts[word]
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract content words from text."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]
    
    def learn(self, text: str):
        """Learn from text by extracting frames and building concept space."""
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            words = self._extract_words(sentence)
            if len(words) < 2:
                continue
            
            # Extract frame: first content word = initiator, second = mediator, third = receiver
            initiator = words[0]
            mediator = words[1]
            
            # Find receiver (skip adverbs)
            receiver = None
            for i in range(2, len(words)):
                w = words[i]
                if not w.endswith('ly') and len(w) > 3:
                    receiver = w
                    break
            
            # Create frame
            frame = Frame(initiator=initiator, mediator=mediator, receiver=receiver)
            self.frames.append(frame)
            
            # Update concepts
            init_c = self._get_or_create(initiator)
            init_c.initiator_count += 1
            init_c.position = 0.0  # Initiators at position 0
            init_c.actions[mediator] += 1
            if receiver:
                init_c.targets[receiver] += 1
            
            med_c = self._get_or_create(mediator)
            med_c.mediator_count += 1
            med_c.position = 0.5  # Mediators at position 0.5
            
            if receiver:
                recv_c = self._get_or_create(receiver)
                recv_c.receiver_count += 1
                recv_c.position = 1.0  # Receivers at position 1
            
            # Track co-occurrence
            for i, w1 in enumerate(words):
                c1 = self._get_or_create(w1)
                for w2 in words[i+1:]:
                    c1.co_occurs[w2] += 1
                    c2 = self._get_or_create(w2)
                    c2.co_occurs[w1] += 1
    
    def encode(self, text: str) -> float:
        """Encode text to a position in φ-space."""
        words = self._extract_words(text)
        if not words:
            return 0.5
        
        total = 0.0
        weight = 0.0
        
        for i, word in enumerate(words):
            w = PHI ** (-i)
            if word in self.concepts:
                total += self.concepts[word].position * w
            else:
                total += 0.5 * w  # Unknown words at center
            weight += w
        
        return total / weight if weight > 0 else 0.5
    
    def find_entity(self, query_words: List[str]) -> Optional[str]:
        """Find the entity being asked about."""
        for word in query_words:
            if word in self.concepts:
                c = self.concepts[word]
                if c.is_entity:
                    return word
        return None
    
    def find_action(self, query_words: List[str]) -> Optional[str]:
        """Find the action being asked about."""
        for word in query_words:
            base = self.morpho._get_base(word)
            # Check if this word or its base is a known action
            for name, c in self.concepts.items():
                if c.is_action:
                    if name == word or name == base or self.morpho._get_base(name) == base:
                        return name
        return None
    
    def respond(self, query: str) -> str:
        """Generate a response to a query."""
        query_lower = query.lower().strip().rstrip('?')
        words = self._extract_words(query_lower)
        all_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Detect question type
        is_who = 'who' in all_words
        is_what = 'what' in all_words
        is_does = 'does' in all_words or 'do' in all_words or 'did' in all_words
        is_describe = 'describe' in all_words or 'tell' in all_words or 'about' in all_words
        
        # Find subject and action in query
        entity = self.find_entity(words)
        action = self.find_action(words)
        
        # Generate response based on question type
        if is_who and action:
            return self._who_does(action)
        elif is_what and is_does and entity:
            return self._what_does(entity)
        elif is_describe or (entity and not is_who and not is_what):
            if entity:
                return self._describe(entity)
            # Try to find any mentioned entity
            for word in words:
                if word in self.concepts:
                    return self._describe(word)
        elif is_who:
            return self._list_entities()
        
        # Fallback: find most relevant concept and describe it
        best_match = None
        best_score = 0
        
        for word in words:
            if word in self.concepts:
                c = self.concepts[word]
                score = c.initiator_count + c.mediator_count + c.receiver_count
                if score > best_score:
                    best_score = score
                    best_match = word
        
        if best_match:
            return self._describe(best_match)
        
        return "I don't have enough information to answer that."
    
    def _is_good_target(self, word: str) -> bool:
        """Check if a word is a good target (not noise)."""
        if not word or len(word) <= 3:
            return False
        if word.endswith('ly'):
            return False
        if word in self.stop_words:
            return False
        # Check if it's an entity in our concepts
        if word in self.concepts:
            c = self.concepts[word]
            return c.is_entity or c.receiver_count > 0
        return True
    
    def _who_does(self, action: str) -> str:
        """Answer 'Who [action]s?'"""
        base_action = self.morpho._get_base(action)
        
        # Find all entities that perform this action
        actors = []
        for name, c in self.concepts.items():
            if not c.is_entity:
                continue
            for act in c.actions:
                if self.morpho._get_base(act) == base_action:
                    actors.append((name, c.actions[act]))
                    break
        
        if not actors:
            return f"I don't know who {action}s."
        
        # Sort by frequency
        actors.sort(key=lambda x: x[1], reverse=True)
        
        if len(actors) == 1:
            actor = actors[0][0]
            # Find what they do it to - look in frames for this specific action
            c = self.concepts[actor]
            
            # Find targets specifically for this action from frames
            action_targets = []
            for frame in self.frames:
                if frame.initiator == actor:
                    frame_base = self.morpho._get_base(frame.mediator)
                    if frame_base == base_action and frame.receiver:
                        if self._is_good_target(frame.receiver):
                            action_targets.append(frame.receiver)
            
            # Fallback to general targets if no specific ones found
            if not action_targets:
                action_targets = [t for t in c.targets.keys() if self._is_good_target(t)][:2]
            
            verb = self.morpho.transform(base_action, MorphoQuaternion(x=1, y=-1, z=0, w=-1))
            
            if action_targets:
                return f"{actor.title()} {verb} {', '.join(action_targets[:2])}."
            else:
                return f"{actor.title()} {verb}."
        else:
            # Multiple actors
            actor_names = [a[0].title() for a in actors[:3]]
            verb = action  # Use base form for plural
            
            if len(actor_names) == 2:
                return f"{actor_names[0]} and {actor_names[1]} both {verb}."
            else:
                return f"{', '.join(actor_names[:-1])}, and {actor_names[-1]} all {verb}."
    
    def _what_does(self, entity: str) -> str:
        """Answer 'What does [entity] do?'"""
        if entity not in self.concepts:
            return f"I don't know about {entity}."
        
        c = self.concepts[entity]
        
        if not c.actions:
            return f"{entity.title()} doesn't seem to do anything notable."
        
        # Get top actions
        top_actions = c.actions.most_common(3)
        
        # Build response with proper conjugation
        q = MorphoQuaternion(x=1, y=-1, z=0, w=-1)  # 3rd singular present
        
        verbs = []
        for action, count in top_actions:
            base = self.morpho._get_base(action)
            verb = self.morpho.transform(base, q)
            verbs.append(verb)
        
        # Get targets - use filter
        good_targets = [t for t in c.targets.keys() if self._is_good_target(t)][:2]
        
        if len(verbs) == 1:
            response = f"{entity.title()} {verbs[0]}"
        elif len(verbs) == 2:
            response = f"{entity.title()} {verbs[0]} and {verbs[1]}"
        else:
            response = f"{entity.title()} {', '.join(verbs[:-1])}, and {verbs[-1]}"
        
        if good_targets:
            response += f", often involving {' and '.join(good_targets)}"
        
        return response + "."
    
    def _describe(self, entity: str) -> str:
        """Describe an entity."""
        if entity not in self.concepts:
            return f"I don't know about {entity}."
        
        c = self.concepts[entity]
        
        # Determine role
        role = c.primary_role
        if role == "initiator":
            role_desc = "a protagonist who"
        elif role == "mediator":
            role_desc = "an action that"
        else:
            role_desc = "something that"
        
        # Get actions
        if c.actions:
            top_actions = c.actions.most_common(3)
            q = MorphoQuaternion(x=1, y=-1, z=0, w=-1)
            verbs = [self.morpho.transform(self.morpho._get_base(a), q) for a, _ in top_actions]
            
            if len(verbs) == 1:
                action_desc = verbs[0]
            elif len(verbs) == 2:
                action_desc = f"{verbs[0]} and {verbs[1]}"
            else:
                action_desc = f"{', '.join(verbs[:-1])}, and {verbs[-1]}"
            
            response = f"{entity.title()} is {role_desc} {action_desc}"
        else:
            response = f"{entity.title()} is {role_desc} appears in the narrative"
        
        # Add targets - use filter
        good_targets = [t for t in c.targets.keys() if self._is_good_target(t)][:2]
        
        if good_targets:
            response += f", often involving {' and '.join(good_targets)}"
        
        return response + "."
    
    def _list_entities(self) -> str:
        """List main entities."""
        entities = [(name, c) for name, c in self.concepts.items() if c.is_entity]
        
        if not entities:
            return "I don't know of any characters."
        
        # Sort by initiator count
        entities.sort(key=lambda x: x[1].initiator_count, reverse=True)
        
        top = [name.title() for name, _ in entities[:5]]
        
        if len(top) == 1:
            return f"The main character is {top[0]}."
        elif len(top) == 2:
            return f"The main characters are {top[0]} and {top[1]}."
        else:
            return f"The main characters are {', '.join(top[:-1])}, and {top[-1]}."
    
    def chat(self):
        """Interactive chat loop."""
        print("φ Quality Chat")
        print("=" * 50)
        print(f"Loaded {len(self.concepts)} concepts, {len(self.frames)} frames")
        print("Type 'quit' to exit")
        print()
        
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not user:
                continue
            if user.lower() == 'quit':
                print("Goodbye!")
                break
            
            response = self.respond(user)
            print(f"φ: {response}")
            print()


# Rich corpus for quality testing
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

Darcy watched Elizabeth at the ball intently.
Elizabeth danced gracefully with several partners.
Bingley fell in love with Jane immediately.
Jane smiled sweetly at everyone she met.
Wickham deceived everyone with his charm.
Darcy proposed to Elizabeth awkwardly.
Elizabeth rejected Darcy's proposal firmly.
Darcy wrote Elizabeth a long letter.
Elizabeth reconsidered her opinion of Darcy.
Darcy and Elizabeth married happily.
"""


def demo():
    """Demonstrate quality chatbot."""
    print("φ Quality Chatbot Demo")
    print("=" * 60)
    print()
    
    bot = PhiQuality()
    bot.learn(CORPUS)
    
    print(f"Learned {len(bot.concepts)} concepts from {len(bot.frames)} frames")
    print()
    
    # Test queries
    queries = [
        "Who is Holmes?",
        "What does Watson do?",
        "Who examines?",
        "Tell me about Moriarty",
        "What does Alice do?",
        "Who loves?",
        "Describe Hamlet",
        "What does Darcy do?",
        "Who smiled?",
        "Tell me about the Queen",
    ]
    
    print("=" * 60)
    print("QUALITY TEST")
    print("=" * 60)
    print()
    
    for q in queries:
        print(f"Q: {q}")
        print(f"A: {bot.respond(q)}")
        print()
    
    return bot


def main():
    """Main entry point."""
    import sys
    
    bot = PhiQuality()
    bot.learn(CORPUS)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        bot.chat()


if __name__ == "__main__":
    main()
