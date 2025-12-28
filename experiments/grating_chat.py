#!/usr/bin/env python3
"""
Grating Chat: Two-Source Diffraction for Meaningful Answers

Key insight: If the process is the same everywhere, we just need two sources:
1. Knowledge source - WHAT to say (content)
2. Style source - HOW to say it (form)

The interference pattern between them produces meaningful, styled answers.

This is like a diffraction grating:
- Source 1 creates wavefronts (knowledge concepts)
- Source 2 creates wavefronts (style patterns)
- Where they align = constructive interference = the answer

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import math
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.morphological_quaternion import MorphologicalTransformer, MorphoQuaternion

PHI = 1.618034


@dataclass
class GratingConcept:
    """A concept with position in the grating."""
    word: str
    count: int = 0
    actor_count: int = 0
    action_count: int = 0
    target_count: int = 0
    
    # Relationships
    actions: Counter = field(default_factory=Counter)
    targets: Counter = field(default_factory=Counter)
    modifiers: Counter = field(default_factory=Counter)
    
    @property
    def phi_direction(self) -> float:
        """Entity (+) vs Action (-)"""
        total = self.actor_count + self.action_count + self.target_count
        if total == 0:
            return 0.0
        entity = self.actor_count + self.target_count
        return (entity - self.action_count) / total
    
    @property
    def narrative_position(self) -> float:
        """0=actor, 0.5=action, 1=target"""
        total = self.actor_count + self.action_count + self.target_count
        if total == 0:
            return 0.5
        return (self.actor_count * 0.0 + self.action_count * 0.5 + self.target_count * 1.0) / total


class GratingSource:
    """
    A source for the diffraction grating.
    
    Can be used for knowledge (what to say) or style (how to say it).
    """
    
    def __init__(self, name: str):
        self.name = name
        self.concepts: Dict[str, GratingConcept] = {}
        self.frames: List[Tuple[str, str, Optional[str]]] = []  # (actor, action, target)
        self.patterns: List[str] = []  # Sentence patterns
        self.morpho = MorphologicalTransformer()
        
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
    
    def _get_or_create(self, word: str) -> GratingConcept:
        word_lower = word.lower()
        if word_lower not in self.concepts:
            self.concepts[word_lower] = GratingConcept(word=word_lower)
        return self.concepts[word_lower]
    
    def ingest(self, text: str):
        """Ingest text to build the source."""
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Store pattern
            self.patterns.append(sentence)
            
            # Extract frame
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            content = [t for t in tokens if t not in self.function_words and len(t) > 2]
            
            if len(content) < 2:
                continue
            
            actor = content[0]
            action = content[1]
            target = content[2] if len(content) > 2 else None
            
            # Skip adverbs as targets
            if target and (target.endswith('ly') or len(target) <= 3):
                target = content[3] if len(content) > 3 else None
            
            self.frames.append((actor, action, target))
            
            # Update concepts
            actor_c = self._get_or_create(actor)
            actor_c.count += 1
            actor_c.actor_count += 1
            actor_c.actions[action] += 1
            if target:
                actor_c.targets[target] += 1
            
            action_c = self._get_or_create(action)
            action_c.count += 1
            action_c.action_count += 1
            
            if target:
                target_c = self._get_or_create(target)
                target_c.count += 1
                target_c.target_count += 1
    
    def get_actors(self) -> List[str]:
        """Get all actors in this source."""
        return [n for n, c in self.concepts.items() if c.actor_count > 0]
    
    def get_actions_for(self, actor: str) -> List[str]:
        """Get actions for an actor."""
        if actor not in self.concepts:
            return []
        return list(self.concepts[actor].actions.keys())
    
    def get_targets_for(self, actor: str) -> List[str]:
        """Get targets for an actor."""
        if actor not in self.concepts:
            return []
        return list(self.concepts[actor].targets.keys())
    
    def get_pattern(self) -> str:
        """Get a random sentence pattern."""
        if self.patterns:
            return random.choice(self.patterns)
        return "{actor} {action} {target}."


class GratingChat:
    """
    Chat interface using two-source diffraction.
    
    Source 1: Knowledge (what to say)
    Source 2: Style (how to say it)
    
    The interference between them produces meaningful, styled answers.
    """
    
    def __init__(self, knowledge_source: GratingSource, style_source: GratingSource):
        self.knowledge = knowledge_source
        self.style = style_source
        self.morpho = MorphologicalTransformer()
        self.q3_present = MorphoQuaternion(x=1, y=-1, z=0, w=-1)  # 3rd singular present
    
    def _find_matching_concept(self, query_word: str, source: GratingSource) -> Optional[str]:
        """Find a concept in source that matches the query word."""
        base = self.morpho._get_base(query_word.lower())
        
        # Direct match
        if base in source.concepts:
            return base
        if query_word.lower() in source.concepts:
            return query_word.lower()
        
        # Fuzzy match by action
        for name, concept in source.concepts.items():
            if base in [self.morpho._get_base(a) for a in concept.actions]:
                return name
        
        return None
    
    def _interference(self, knowledge_concept: GratingConcept, style_concept: GratingConcept) -> float:
        """
        Compute interference between knowledge and style concepts.
        
        Constructive when:
        - Same φ-direction (both entities or both actions)
        - Complementary narrative positions (actor-target fit)
        """
        # φ-direction alignment (same type = constructive)
        phi_alignment = knowledge_concept.phi_direction * style_concept.phi_direction
        
        # Narrative complementarity (different positions = constructive)
        narrative_diff = abs(knowledge_concept.narrative_position - style_concept.narrative_position)
        narrative_alignment = narrative_diff  # Different = good
        
        return 0.6 * phi_alignment + 0.4 * narrative_alignment
    
    def _find_best_style_match(self, knowledge_actor: str) -> Optional[str]:
        """Find the best style actor that interferes constructively with knowledge actor."""
        if knowledge_actor not in self.knowledge.concepts:
            return None
        
        k_concept = self.knowledge.concepts[knowledge_actor]
        
        best_match = None
        best_score = -float('inf')
        
        for name, s_concept in self.style.concepts.items():
            if s_concept.actor_count == 0:
                continue
            
            score = self._interference(k_concept, s_concept)
            if score > best_score:
                best_score = score
                best_match = name
        
        return best_match
    
    def ask(self, question: str) -> str:
        """
        Answer a question using two-source diffraction.
        
        1. Parse question to find what's being asked
        2. Find relevant knowledge
        3. Apply style interference
        4. Generate styled answer
        """
        question_lower = question.lower().strip().rstrip('?')
        words = re.findall(r'\b\w+\b', question_lower)
        
        # Detect question type
        is_who = 'who' in words
        is_what = 'what' in words
        is_does = 'does' in words or 'do' in words
        is_describe = 'describe' in words or 'tell' in words
        
        # Find the subject of the question
        subject = None
        query_action = None
        
        for word in words:
            if word in {'who', 'what', 'does', 'do', 'is', 'are', 'tell', 'me', 'about', 'describe'}:
                continue
            
            # Check if it's an entity in knowledge
            if word in self.knowledge.concepts:
                c = self.knowledge.concepts[word]
                if c.actor_count > 0:
                    subject = word
                    break
            
            # Check if it's an action
            base = self.morpho._get_base(word)
            for name, c in self.knowledge.concepts.items():
                if base in [self.morpho._get_base(a) for a in c.actions]:
                    query_action = base
                    break
        
        # Generate answer based on question type
        if is_who and query_action:
            # "Who [action]s?" - find actors who do this action
            return self._answer_who_does(query_action)
        
        elif is_what and is_does and subject:
            # "What does [subject] do?" - find actions
            return self._answer_what_does(subject)
        
        elif is_describe or (subject and not is_who and not is_what):
            # "Describe [subject]" or "[subject]?"
            if not subject:
                for word in words:
                    if word in self.knowledge.concepts:
                        subject = word
                        break
            if subject:
                return self._describe_entity(subject)
        
        elif is_who:
            # Generic "who" - list main actors
            return self._list_actors()
        
        return self._styled_response("I don't have enough information to answer that.")
    
    def _answer_who_does(self, action: str) -> str:
        """Answer 'Who [action]s?'"""
        actors = []
        base_action = self.morpho._get_base(action)
        
        for name, concept in self.knowledge.concepts.items():
            if concept.actor_count == 0:
                continue
            for act in concept.actions:
                if self.morpho._get_base(act) == base_action:
                    actors.append(name)
                    break
        
        if not actors:
            return self._styled_response(f"No one {action}s in the knowledge I have.")
        
        # Apply style
        verb = self.morpho.transform(base_action, self.q3_present)
        
        if len(actors) == 1:
            return self._styled_response(f"{actors[0].title()} {verb}.")
        else:
            actor_list = ', '.join([a.title() for a in actors[:-1]]) + f" and {actors[-1].title()}"
            return self._styled_response(f"{actor_list} all {action}.")
    
    def _answer_what_does(self, subject: str) -> str:
        """Answer 'What does [subject] do?'"""
        if subject not in self.knowledge.concepts:
            return self._styled_response(f"I don't have information about {subject}.")
        
        concept = self.knowledge.concepts[subject]
        actions = list(concept.actions.keys())[:3]
        
        if not actions:
            return self._styled_response(f"{subject.title()} doesn't seem to do much.")
        
        # Conjugate actions
        verbs = [self.morpho.transform(self.morpho._get_base(a), self.q3_present) for a in actions]
        
        # Get targets
        targets = list(concept.targets.keys())[:2]
        
        if targets:
            good_targets = [t for t in targets if not t.endswith('ly') and len(t) > 3]
            if good_targets:
                return self._styled_response(
                    f"{subject.title()} {', '.join(verbs)}. "
                    f"Often involving {', '.join(good_targets)}."
                )
        
        return self._styled_response(f"{subject.title()} {', '.join(verbs)}.")
    
    def _describe_entity(self, entity: str) -> str:
        """Describe an entity with styled output."""
        if entity not in self.knowledge.concepts:
            return self._styled_response(f"I don't have information about {entity}.")
        
        concept = self.knowledge.concepts[entity]
        
        # Role
        if concept.actor_count > concept.target_count:
            role = "protagonist"
        elif concept.target_count > 0:
            role = "object of action"
        else:
            role = "concept"
        
        # Actions
        actions = list(concept.actions.keys())[:3]
        verbs = [self.morpho.transform(self.morpho._get_base(a), self.q3_present) for a in actions] if actions else []
        
        # Targets
        targets = list(concept.targets.keys())[:2]
        good_targets = [t for t in targets if not t.endswith('ly') and len(t) > 3]
        
        # Build description
        parts = [f"{entity.title()} is a {role}"]
        
        if verbs:
            parts.append(f"who {', '.join(verbs)}")
        
        if good_targets:
            parts.append(f"(often involving {', '.join(good_targets)})")
        
        return self._styled_response('. '.join([' '.join(parts)]) + '.')
    
    def _list_actors(self) -> str:
        """List main actors."""
        actors = self.knowledge.get_actors()
        if not actors:
            return self._styled_response("I don't know of any actors.")
        
        # Sort by count
        sorted_actors = sorted(actors, 
                               key=lambda a: self.knowledge.concepts[a].actor_count, 
                               reverse=True)[:5]
        
        return self._styled_response(
            f"The main characters are: {', '.join([a.title() for a in sorted_actors])}."
        )
    
    def _styled_response(self, content: str) -> str:
        """
        Apply style interference to the response.
        
        This is where the two sources interfere:
        - Content comes from knowledge
        - Style comes from style source
        
        The interference pattern determines:
        - Sentence structure (from style patterns)
        - Word choice modifiers (from style concepts)
        - Overall tone (from style name)
        """
        # Style-specific transformations
        if self.style.name == "formal":
            # Formal: Add hedging, use passive voice hints
            content = content.replace(" is ", " appears to be ")
            content = content.replace("The main", "The principal")
            prefix = "Upon examination, "
            suffix = ""
        
        elif self.style.name == "casual":
            # Casual: Simplify, add filler
            content = content.replace("protagonist", "main character")
            content = content.replace("involving", "with")
            content = content.replace(".", ", you know.")
            prefix = "So like, "
            suffix = " Pretty cool, right?"
        
        elif self.style.name == "literary":
            # Literary: Add flourish, metaphor hints
            content = content.replace(" is ", " emerges as ")
            content = content.replace("The main", "Among the dramatis personae,")
            content = content.replace("protagonist", "central figure")
            prefix = "In the tapestry of this narrative, "
            suffix = ""
        
        elif self.style.name == "scientific":
            # Scientific: Add precision, hedging
            content = content.replace(" is ", " can be classified as ")
            content = content.replace("The main", "Primary subjects include")
            content = content.replace("protagonist", "primary agent")
            prefix = "Data analysis reveals: "
            suffix = " Further investigation recommended."
        
        elif self.style.name == "pirate":
            # Fun: Pirate speak
            content = content.replace(" is ", " be ")
            content = content.replace("The main", "The scurvy")
            content = content.replace("protagonist", "captain")
            content = content.replace(".", ", arrr!")
            prefix = "Ahoy! "
            suffix = " Shiver me timbers!"
        
        elif self.style.name == "noir":
            # Noir detective style
            content = content.replace(" is ", " was ")
            content = content.replace("The main", "The usual")
            content = content.replace("protagonist", "player in this game")
            prefix = "The rain fell hard that night. "
            suffix = " But that's just how it goes in this town."
        
        else:
            prefix = ""
            suffix = ""
        
        return prefix + content + suffix
    
    def chat(self):
        """Interactive chat loop."""
        print(f"\n{'='*60}")
        print(f"GRATING CHAT")
        print(f"{'='*60}")
        print(f"Knowledge source: {self.knowledge.name}")
        print(f"  - {len(self.knowledge.concepts)} concepts")
        print(f"  - {len(self.knowledge.frames)} frames")
        print(f"Style source: {self.style.name}")
        print(f"  - {len(self.style.concepts)} concepts")
        print(f"  - {len(self.style.patterns)} patterns")
        print(f"{'='*60}")
        print("Type 'quit' to exit, 'style <name>' to change style")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('style '):
                style_name = user_input[6:].strip()
                print(f"[Style changed to: {style_name}]")
                self.style.name = style_name
                continue
            
            response = self.ask(user_input)
            print(f"Bot: {response}")
            print()


# Sample knowledge sources
SHERLOCK_KNOWLEDGE = """
Holmes examined the evidence carefully. Watson watched from the doorway.
The detective studied the footprints methodically. He noticed something unusual.
Holmes said to Watson that the case was elementary. Watson replied thoughtfully.
Lestrade questioned the witnesses thoroughly. Holmes observed the room.
Watson wrote in his journal diligently. Holmes deduced the identity brilliantly.
Moriarty plotted against Holmes secretly. The professor was cunning.
Holmes solved the mystery. Watson assisted Holmes faithfully.
The detective found the clue. Holmes confronted the criminal.
"""

ALICE_KNOWLEDGE = """
Alice fell down the rabbit hole unexpectedly. She wondered where she was going.
The Queen shouted angrily at everyone. Alice felt confused and scared.
The Cheshire Cat smiled mysteriously. He disappeared slowly into thin air.
Alice grew very tall suddenly. She shrank very small moments later.
The Mad Hatter laughed wildly at the party. He poured more tea endlessly.
Alice explored Wonderland curiously. She met strange creatures everywhere.
The Caterpillar asked Alice questions. Alice answered uncertainly.
"""

# Sample style sources
FORMAL_STYLE = """
One observes that the situation presents itself clearly.
It would appear that the evidence suggests a conclusion.
The analysis indicates a pattern of significance.
Furthermore, the data supports this interpretation.
In conclusion, the findings are most illuminating.
"""

CASUAL_STYLE = """
So basically this thing happened and it was pretty cool.
Like, you know, stuff just kind of works out sometimes.
Anyway, the whole deal is actually not that complicated.
Pretty much everyone agrees that this makes sense.
Yeah, so that's the gist of it really.
"""

LITERARY_STYLE = """
The shadows of meaning dance upon the page, revealing truths.
One might observe the delicate interplay of fate and choice.
In the tapestry of narrative, threads weave together purposefully.
The protagonist stands at the crossroads of destiny.
Through the mist of uncertainty, clarity emerges triumphantly.
"""

SCIENTIFIC_STYLE = """
Analysis of the data reveals significant patterns.
The hypothesis is supported by empirical observation.
Results indicate a correlation between variables.
Further investigation is warranted to confirm findings.
The methodology demonstrates reproducible outcomes.
"""


def create_source(name: str, text: str) -> GratingSource:
    """Create a grating source from text."""
    source = GratingSource(name)
    source.ingest(text)
    return source


def demo():
    """Demonstrate the grating chat."""
    print("="*70)
    print("GRATING CHAT DEMO")
    print("="*70)
    print()
    print("Two-source diffraction for meaningful answers:")
    print("  Source 1 (Knowledge): WHAT to say")
    print("  Source 2 (Style): HOW to say it")
    print()
    
    # Create sources
    sherlock = create_source("sherlock", SHERLOCK_KNOWLEDGE)
    alice = create_source("alice", ALICE_KNOWLEDGE)
    
    formal = create_source("formal", FORMAL_STYLE)
    casual = create_source("casual", CASUAL_STYLE)
    literary = create_source("literary", LITERARY_STYLE)
    scientific = create_source("scientific", SCIENTIFIC_STYLE)
    
    # Demo with different combinations
    print("="*70)
    print("SHERLOCK + FORMAL STYLE")
    print("="*70)
    chat1 = GratingChat(sherlock, formal)
    
    questions = [
        "Who is Holmes?",
        "What does Watson do?",
        "Who examines?",
        "Describe Moriarty",
    ]
    
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {chat1.ask(q)}")
        print()
    
    print("="*70)
    print("SHERLOCK + CASUAL STYLE")
    print("="*70)
    chat2 = GratingChat(sherlock, casual)
    
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {chat2.ask(q)}")
        print()
    
    print("="*70)
    print("ALICE + LITERARY STYLE")
    print("="*70)
    chat3 = GratingChat(alice, literary)
    
    alice_questions = [
        "Who is Alice?",
        "What does the Queen do?",
        "Describe the Cheshire Cat",
        "Who smiles?",
    ]
    
    for q in alice_questions:
        print(f"Q: {q}")
        print(f"A: {chat3.ask(q)}")
        print()
    
    print("="*70)
    print("ALICE + SCIENTIFIC STYLE")
    print("="*70)
    chat4 = GratingChat(alice, scientific)
    
    for q in alice_questions:
        print(f"Q: {q}")
        print(f"A: {chat4.ask(q)}")
        print()
    
    # Interactive mode
    print("="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print()
    print("Choose knowledge source:")
    print("  1. Sherlock Holmes")
    print("  2. Alice in Wonderland")
    print()
    
    try:
        choice = input("Enter 1 or 2 (or press Enter for Sherlock): ").strip()
        knowledge = alice if choice == '2' else sherlock
        
        print()
        print("Choose style:")
        print("  1. Formal")
        print("  2. Casual")
        print("  3. Literary")
        print("  4. Scientific")
        print()
        
        style_choice = input("Enter 1-4 (or press Enter for Formal): ").strip()
        styles = {'1': formal, '2': casual, '3': literary, '4': scientific}
        style = styles.get(style_choice, formal)
        
        chat = GratingChat(knowledge, style)
        chat.chat()
    except (EOFError, KeyboardInterrupt):
        print("\nDemo complete!")


if __name__ == "__main__":
    demo()
