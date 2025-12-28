#!/usr/bin/env python3
"""
Unified Grating: One Process for Knowledge, Style, and Projection

Key insight: If the process is the same everywhere, then knowledge,
style, and projection are all just "sources" that can be ingested
with the same symmetric approach.

The only difference is WHAT we extract:
- Knowledge: actor-action-target frames (WHAT happens)
- Style: pattern-tone-transform rules (HOW to say it)
- Projection: modifier-emphasis-structure (HOW MUCH to say)

But the PROCESS is identical:
1. Ingest text symmetrically
2. Extract φ-direction (entity vs action)
3. Build interference patterns
4. Apply to output

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import math
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.morphological_quaternion import MorphologicalTransformer, MorphoQuaternion

PHI = 1.618034


@dataclass
class UnifiedConcept:
    """
    A concept that works for knowledge, style, or projection.
    
    The same structure captures:
    - Knowledge: "Holmes" (actor), "examines" (action), "evidence" (target)
    - Style: "indeed" (modifier), "appears" (hedge), "upon examination" (opener)
    - Projection: "brilliantly" (emphasis), "with care" (manner), "deeply" (intensity)
    """
    word: str
    
    # Role counts (same for all source types)
    initiator_count: int = 0   # Actor / Opener / Intensifier
    mediator_count: int = 0    # Action / Hedge / Manner
    receiver_count: int = 0    # Target / Closer / Object
    
    # Relationships (same structure, different semantics)
    performs: Counter = field(default_factory=Counter)  # What this initiates
    receives: Counter = field(default_factory=Counter)  # What this receives
    modifies: Counter = field(default_factory=Counter)  # What this modifies
    
    # Co-occurrence for interference
    co_occurs: Counter = field(default_factory=Counter)
    
    @property
    def phi_direction(self) -> float:
        """
        φ-direction: +1 for entities/nouns, -1 for actions/verbs.
        
        This is the SAME calculation regardless of source type:
        - Knowledge: actor/target (+) vs action (-)
        - Style: opener/closer (+) vs hedge (-)
        - Projection: intensifier/object (+) vs manner (-)
        """
        total = self.initiator_count + self.mediator_count + self.receiver_count
        if total == 0:
            return 0.0
        entity_like = self.initiator_count + self.receiver_count
        action_like = self.mediator_count
        return (entity_like - action_like) / total
    
    @property
    def position(self) -> float:
        """
        Position in the flow: 0=initiator, 0.5=mediator, 1=receiver.
        
        Same calculation for all source types.
        """
        total = self.initiator_count + self.mediator_count + self.receiver_count
        if total == 0:
            return 0.5
        return (
            self.initiator_count * 0.0 +
            self.mediator_count * 0.5 +
            self.receiver_count * 1.0
        ) / total
    
    @property
    def magnitude(self) -> float:
        """Importance based on total usage."""
        return math.log1p(self.initiator_count + self.mediator_count + self.receiver_count)


@dataclass
class UnifiedFrame:
    """
    A frame that works for knowledge, style, or projection.
    
    Structure: Initiator → Mediator → Receiver
    
    - Knowledge: Actor → Action → Target
    - Style: Opener → Hedge → Closer
    - Projection: Intensifier → Manner → Object
    """
    initiator: str
    mediator: str
    receiver: Optional[str] = None
    source_type: str = "knowledge"  # knowledge, style, projection


class UnifiedSource:
    """
    A unified source that handles knowledge, style, and projection
    with the SAME ingestion process.
    
    The only difference is how we INTERPRET the extracted structure,
    not how we EXTRACT it.
    """
    
    def __init__(self, name: str, source_type: str = "knowledge"):
        """
        Create a unified source.
        
        source_type: "knowledge", "style", or "projection"
        """
        self.name = name
        self.source_type = source_type
        self.concepts: Dict[str, UnifiedConcept] = {}
        self.frames: List[UnifiedFrame] = []
        self.patterns: List[str] = []
        self.morpho = MorphologicalTransformer()
        
        # Function words to skip (same for all types)
        self.function_words = {
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
            'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their',
            'that', 'this', 'these', 'those', 'which', 'who', 'whom',
            'and', 'or', 'but', 'if', 'then', 'so', 'as', 'than',
        }
    
    def _get_or_create(self, word: str) -> UnifiedConcept:
        """Get or create a concept."""
        word_lower = word.lower()
        if word_lower not in self.concepts:
            self.concepts[word_lower] = UnifiedConcept(word=word_lower)
        return self.concepts[word_lower]
    
    def ingest(self, text: str):
        """
        Ingest text using SYMMETRIC extraction.
        
        The same process works for knowledge, style, and projection:
        1. Split into sentences
        2. Extract content words
        3. Assign roles based on position (initiator, mediator, receiver)
        4. Build frames and update concepts
        
        The INTERPRETATION differs by source_type, but the PROCESS is identical.
        """
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Store pattern
            self.patterns.append(sentence)
            
            # Extract content words
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            content = [t for t in tokens if t not in self.function_words and len(t) > 2]
            
            if len(content) < 2:
                continue
            
            # SYMMETRIC ROLE ASSIGNMENT
            # Position 0 = Initiator (actor/opener/intensifier)
            # Position 1 = Mediator (action/hedge/manner)
            # Position 2+ = Receiver (target/closer/object)
            
            initiator = content[0]
            mediator = content[1]
            receiver = None
            
            # Find receiver (skip adverbs)
            for i in range(2, len(content)):
                word = content[i]
                if not word.endswith('ly') and len(word) > 3:
                    receiver = word
                    break
            
            # Create frame
            frame = UnifiedFrame(
                initiator=initiator,
                mediator=mediator,
                receiver=receiver,
                source_type=self.source_type
            )
            self.frames.append(frame)
            
            # Update concepts (SAME process for all types)
            self._update_concepts(frame, content)
    
    def _update_concepts(self, frame: UnifiedFrame, all_words: List[str]):
        """Update concepts from a frame."""
        # Initiator
        init_c = self._get_or_create(frame.initiator)
        init_c.initiator_count += 1
        init_c.performs[frame.mediator] += 1
        if frame.receiver:
            init_c.receives[frame.receiver] += 1
        
        # Mediator
        med_c = self._get_or_create(frame.mediator)
        med_c.mediator_count += 1
        
        # Receiver
        if frame.receiver:
            recv_c = self._get_or_create(frame.receiver)
            recv_c.receiver_count += 1
        
        # Co-occurrence (for interference)
        for i, w1 in enumerate(all_words):
            c1 = self._get_or_create(w1)
            for w2 in all_words[i+1:]:
                c1.co_occurs[w2] += 1
                c2 = self._get_or_create(w2)
                c2.co_occurs[w1] += 1
    
    def get_initiators(self) -> List[str]:
        """Get all initiators (actors/openers/intensifiers)."""
        return [n for n, c in self.concepts.items() if c.initiator_count > 0]
    
    def get_mediators(self) -> List[str]:
        """Get all mediators (actions/hedges/manners)."""
        return [n for n, c in self.concepts.items() if c.mediator_count > 0]
    
    def get_receivers(self) -> List[str]:
        """Get all receivers (targets/closers/objects)."""
        return [n for n, c in self.concepts.items() if c.receiver_count > 0]
    
    def interference(self, other: 'UnifiedSource') -> Dict[str, float]:
        """
        Compute interference with another source.
        
        Returns a mapping of concept → interference score.
        Positive = constructive (concepts align)
        Negative = destructive (concepts conflict)
        """
        scores = {}
        
        for name, c1 in self.concepts.items():
            if name in other.concepts:
                c2 = other.concepts[name]
                
                # φ-direction alignment (same direction = constructive)
                phi_align = c1.phi_direction * c2.phi_direction
                
                # Position complementarity (different positions = constructive)
                pos_diff = abs(c1.position - c2.position)
                
                # Combined score
                scores[name] = 0.6 * phi_align + 0.4 * pos_diff
        
        return scores


class UnifiedGrating:
    """
    A unified grating that combines multiple sources.
    
    The grating can have any number of sources:
    - 1 source: Simple retrieval
    - 2 sources: Knowledge + Style (our chat demo)
    - 3 sources: Knowledge + Style + Projection
    - N sources: Multi-dimensional interference
    
    The PROCESS is always the same: interference between sources.
    """
    
    def __init__(self):
        self.sources: Dict[str, UnifiedSource] = {}
        self.morpho = MorphologicalTransformer()
        self.q3_present = MorphoQuaternion(x=1, y=-1, z=0, w=-1)
    
    def add_source(self, name: str, text: str, source_type: str = "knowledge"):
        """Add a source to the grating."""
        source = UnifiedSource(name, source_type)
        source.ingest(text)
        self.sources[name] = source
        return source
    
    def query(self, question: str) -> str:
        """
        Query the grating using all sources.
        
        The answer emerges from the interference of all sources.
        """
        if not self.sources:
            return "No sources loaded."
        
        question_lower = question.lower()
        
        # Find knowledge source
        knowledge = None
        for name, source in self.sources.items():
            if source.source_type == "knowledge":
                knowledge = source
                break
        
        if not knowledge:
            knowledge = list(self.sources.values())[0]
        
        # Parse question
        words = re.findall(r'\b\w+\b', question_lower)
        is_who = 'who' in words
        is_what = 'what' in words
        is_describe = 'describe' in words or 'tell' in words
        
        # Find subject
        subject = None
        for word in words:
            if word in {'who', 'what', 'does', 'do', 'is', 'are', 'tell', 'me', 'about', 'describe'}:
                continue
            if word in knowledge.concepts:
                c = knowledge.concepts[word]
                if c.initiator_count > 0:
                    subject = word
                    break
        
        # Generate base answer from knowledge
        if is_describe and subject:
            answer = self._describe(knowledge, subject)
        elif is_who:
            answer = self._who(knowledge)
        elif is_what and subject:
            answer = self._what_does(knowledge, subject)
        else:
            answer = self._general(knowledge, question_lower)
        
        # Apply style and projection interference
        for name, source in self.sources.items():
            if source.source_type == "style":
                answer = self._apply_style(answer, source)
            elif source.source_type == "projection":
                answer = self._apply_projection(answer, source)
        
        return answer
    
    def _describe(self, source: UnifiedSource, entity: str) -> str:
        """Describe an entity."""
        if entity not in source.concepts:
            return f"I don't have information about {entity}."
        
        c = source.concepts[entity]
        
        # Role
        if c.initiator_count > c.receiver_count:
            role = "protagonist"
        elif c.receiver_count > 0:
            role = "object"
        else:
            role = "concept"
        
        # Actions
        actions = list(c.performs.keys())[:3]
        verbs = []
        for a in actions:
            base = self.morpho._get_base(a)
            verb = self.morpho.transform(base, self.q3_present)
            verbs.append(verb)
        
        # Targets
        targets = [t for t in list(c.receives.keys())[:2] 
                   if not t.endswith('ly') and len(t) > 3]
        
        parts = [f"{entity.title()} is a {role}"]
        if verbs:
            parts.append(f"who {', '.join(verbs)}")
        if targets:
            parts.append(f"(involving {', '.join(targets)})")
        
        return ' '.join(parts) + '.'
    
    def _who(self, source: UnifiedSource) -> str:
        """List main initiators."""
        initiators = source.get_initiators()
        if not initiators:
            return "I don't know of any characters."
        
        sorted_init = sorted(
            initiators,
            key=lambda x: source.concepts[x].initiator_count,
            reverse=True
        )[:5]
        
        return f"The main characters are: {', '.join([i.title() for i in sorted_init])}."
    
    def _what_does(self, source: UnifiedSource, subject: str) -> str:
        """What does subject do?"""
        if subject not in source.concepts:
            return f"I don't have information about {subject}."
        
        c = source.concepts[subject]
        actions = list(c.performs.keys())[:3]
        
        if not actions:
            return f"{subject.title()} doesn't seem to do much."
        
        verbs = [self.morpho.transform(self.morpho._get_base(a), self.q3_present) 
                 for a in actions]
        
        return f"{subject.title()} {', '.join(verbs)}."
    
    def _general(self, source: UnifiedSource, query: str) -> str:
        """General query."""
        # Find best matching concept
        words = set(re.findall(r'\b\w+\b', query))
        
        best = None
        best_score = 0
        
        for name, c in source.concepts.items():
            score = 0
            if name in words:
                score += 2
            for w in words:
                if w in c.co_occurs:
                    score += 1
            if score > best_score:
                best_score = score
                best = name
        
        if best:
            return self._describe(source, best)
        
        return "I don't have enough information to answer that."
    
    def _apply_style(self, content: str, style: UnifiedSource) -> str:
        """
        Apply style interference to content.
        
        The style source's patterns transform the content.
        """
        # Get style characteristics from the source itself
        openers = style.get_initiators()
        hedges = style.get_mediators()
        
        # Find dominant opener
        opener = ""
        if openers:
            best_opener = max(openers, 
                             key=lambda x: style.concepts[x].initiator_count)
            # Use the pattern that contains this opener
            for pattern in style.patterns:
                if best_opener in pattern.lower():
                    # Extract opener phrase
                    words = pattern.split()
                    if len(words) >= 3:
                        opener = ' '.join(words[:3]).rstrip('.,') + ', '
                        break
        
        # Apply hedging from mediators
        if hedges:
            for hedge in hedges[:2]:
                if hedge in ['appears', 'seems', 'might', 'could', 'would']:
                    content = content.replace(' is ', f' {hedge} to be ')
                    break
        
        return opener + content
    
    def _apply_projection(self, content: str, projection: UnifiedSource) -> str:
        """
        Apply projection interference to content.
        
        The projection source's intensity modifies the content.
        """
        # Get projection characteristics
        intensifiers = projection.get_initiators()
        manners = projection.get_mediators()
        
        # Add intensity
        if intensifiers:
            best = max(intensifiers,
                      key=lambda x: projection.concepts[x].initiator_count)
            if best not in content.lower():
                content = content.replace('.', f', {best}.')
        
        return content


# Sample texts for testing
SHERLOCK_KNOWLEDGE = """
Holmes examined the evidence carefully. Watson watched from the doorway.
The detective studied the footprints methodically. He noticed something unusual.
Holmes said to Watson that the case was elementary. Watson replied thoughtfully.
Lestrade questioned the witnesses thoroughly. Holmes observed the room.
Watson wrote in his journal diligently. Holmes deduced the identity brilliantly.
Moriarty plotted against Holmes secretly. The professor was cunning.
Holmes solved the mystery. Watson assisted Holmes faithfully.
"""

FORMAL_STYLE = """
One observes that the situation presents itself clearly.
It would appear that the evidence suggests a conclusion.
The analysis indicates a pattern of significance.
Furthermore, the data supports this interpretation.
In conclusion, the findings are most illuminating.
Upon examination, the matter becomes apparent.
"""

LITERARY_PROJECTION = """
Brilliantly the truth emerges from shadow.
Deeply the meaning resonates within.
Profoundly the insight transforms understanding.
Elegantly the solution presents itself.
Magnificently the pattern reveals its nature.
"""

CASUAL_STYLE = """
So basically this thing happened and it was pretty cool.
Like, you know, stuff just kind of works out sometimes.
Anyway, the whole deal is actually not that complicated.
Pretty much everyone agrees that this makes sense.
Yeah, so that's the gist of it really.
"""

INTENSE_PROJECTION = """
Absolutely the evidence proves everything.
Completely the case demonstrates the truth.
Utterly the facts speak for themselves.
Totally the conclusion is undeniable.
Definitely the answer is clear now.
"""


def demo():
    """Demonstrate unified grating."""
    print("=" * 70)
    print("UNIFIED GRATING DEMO")
    print("=" * 70)
    print()
    print("One process for knowledge, style, and projection.")
    print("The same symmetric ingestion handles all three.")
    print()
    
    # Create grating
    grating = UnifiedGrating()
    
    # Add sources
    print("Loading sources...")
    k = grating.add_source("sherlock", SHERLOCK_KNOWLEDGE, "knowledge")
    print(f"  Knowledge: {len(k.concepts)} concepts, {len(k.frames)} frames")
    
    s = grating.add_source("formal", FORMAL_STYLE, "style")
    print(f"  Style: {len(s.concepts)} concepts, {len(s.frames)} frames")
    
    p = grating.add_source("literary", LITERARY_PROJECTION, "projection")
    print(f"  Projection: {len(p.concepts)} concepts, {len(p.frames)} frames")
    print()
    
    # Show unified structure
    print("=" * 70)
    print("UNIFIED STRUCTURE")
    print("=" * 70)
    print()
    print("All sources use the same Initiator → Mediator → Receiver structure:")
    print()
    
    print("KNOWLEDGE (Actor → Action → Target):")
    for frame in k.frames[:3]:
        print(f"  {frame.initiator} → {frame.mediator} → {frame.receiver or '∅'}")
    print()
    
    print("STYLE (Opener → Hedge → Closer):")
    for frame in s.frames[:3]:
        print(f"  {frame.initiator} → {frame.mediator} → {frame.receiver or '∅'}")
    print()
    
    print("PROJECTION (Intensifier → Manner → Object):")
    for frame in p.frames[:3]:
        print(f"  {frame.initiator} → {frame.mediator} → {frame.receiver or '∅'}")
    print()
    
    # Show φ-directions
    print("=" * 70)
    print("φ-DIRECTIONS (Same calculation for all types)")
    print("=" * 70)
    print()
    
    print(f"{'Source':<12} {'Concept':<15} {'φ-dir':>8} {'Position':>10}")
    print("-" * 50)
    
    for source_name, source in grating.sources.items():
        for name, c in list(source.concepts.items())[:3]:
            print(f"{source_name:<12} {name:<15} {c.phi_direction:>8.2f} {c.position:>10.2f}")
        print()
    
    # Query with all sources
    print("=" * 70)
    print("QUERIES (Interference of all sources)")
    print("=" * 70)
    print()
    
    queries = [
        "Who is Holmes?",
        "What does Watson do?",
        "Describe Moriarty",
    ]
    
    for q in queries:
        print(f"Q: {q}")
        print(f"A: {grating.query(q)}")
        print()
    
    # Compare with different style
    print("=" * 70)
    print("DIFFERENT STYLE SOURCE")
    print("=" * 70)
    print()
    
    grating2 = UnifiedGrating()
    grating2.add_source("sherlock", SHERLOCK_KNOWLEDGE, "knowledge")
    grating2.add_source("casual", CASUAL_STYLE, "style")
    grating2.add_source("intense", INTENSE_PROJECTION, "projection")
    
    for q in queries:
        print(f"Q: {q}")
        print(f"A: {grating2.query(q)}")
        print()
    
    return grating


if __name__ == "__main__":
    grating = demo()
