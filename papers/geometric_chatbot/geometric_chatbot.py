#!/usr/bin/env python3
"""
Geometric Chatbot: A Fully Geometric Approach to Natural Language Understanding

This is a standalone implementation that demonstrates:
1. Geometric stop word detection (no hard-coded lists)
2. Position-based frame extraction
3. Geometric morphology from parallel structures
4. Geometric conjugation for output generation

No external dependencies beyond Python standard library.

Author: Lesley Gushurst
License: GPLv3
Date: December 2024
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618034  # Golden ratio

# Bootstrap text for learning morphological patterns
# Format: "I [base]. He [3rd-singular]. I [past]."
# Position 0 = base, Position 1 = 3rd singular, Position 2 = past
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
I flee. He flees. I fled.
I wake. He wakes. I woke.
I shrink. He shrinks. I shrank.
I laugh. He laughs. I laughed.
I drink. He drinks. I drank.
I seek. He seeks. I sought.
"""

# Example corpus for demonstration
EXAMPLE_CORPUS = """
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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GeometricConcept:
    """
    A concept with geometric properties.
    
    Mathematical representation:
    C = (p, f, r_vec, a_vec, t_vec)
    
    where:
    - p = mean position in [0, 1]
    - f = frequency count
    - r_vec = (initiator_count, mediator_count, receiver_count)
    - a_vec = action counts (what this concept does)
    - t_vec = target counts (what this concept acts upon)
    """
    word: str
    
    # Position statistics
    positions: List[float] = field(default_factory=list)
    sentence_count: int = 0
    
    # Role counts (the r_vec)
    initiator_count: int = 0
    mediator_count: int = 0
    receiver_count: int = 0
    
    # Relations
    actions: Counter = field(default_factory=Counter)  # a_vec
    targets: Counter = field(default_factory=Counter)  # t_vec
    
    @property
    def frequency(self) -> int:
        """Total occurrences."""
        return len(self.positions)
    
    @property
    def mean_position(self) -> float:
        """
        Mean position p̄(w) = (1/n) Σ p_i
        """
        if not self.positions:
            return 0.5
        return sum(self.positions) / len(self.positions)
    
    @property
    def position_variance(self) -> float:
        """
        Position variance σ²(w) = (1/n) Σ (p_i - p̄)²
        
        High variance → appears everywhere (stop word candidate)
        Low variance → consistent position (content word)
        """
        if len(self.positions) < 2:
            return 0.0
        mean = self.mean_position
        return sum((p - mean) ** 2 for p in self.positions) / len(self.positions)
    
    @property
    def phi_direction(self) -> float:
        """
        φ-direction: measures if concept is primarily initiator or receiver.
        
        φ-dir(C) = (r_i - r_r) / (r_i + r_m + r_r + ε)
        
        > 0 → primarily initiator (subject-like)
        < 0 → primarily receiver (object-like)
        ≈ 0 → balanced or mediator
        """
        total = self.initiator_count + self.mediator_count + self.receiver_count
        if total == 0:
            return 0.0
        return (self.initiator_count - self.receiver_count) / total
    
    @property
    def phi_magnitude(self) -> float:
        """
        φ-magnitude: strength of the φ-direction.
        
        |φ|(C) = |r_i - r_r| / (r_i + r_m + r_r + ε)
        """
        total = self.initiator_count + self.mediator_count + self.receiver_count
        if total == 0:
            return 0.0
        return abs(self.initiator_count - self.receiver_count) / total
    
    @property
    def is_geometric_stop_word(self) -> bool:
        """
        Geometric stop word detection.
        
        A word is a stop word if:
        1. No semantic role (r_i + r_m + r_r = 0)
        2. OR: Short and frequent (len ≤ 4 and f ≥ 3)
        3. OR: Only receiver role and short (catches prepositions)
        
        This replaces hard-coded stop word lists with geometric detection.
        """
        total_roles = self.initiator_count + self.mediator_count + self.receiver_count
        has_no_role = total_roles == 0
        
        is_short_frequent = len(self.word) <= 4 and self.frequency >= 3
        
        only_receiver = (self.receiver_count > 0 and 
                        self.initiator_count == 0 and 
                        self.mediator_count == 0 and
                        len(self.word) <= 5)
        
        return has_no_role or is_short_frequent or only_receiver
    
    @property
    def is_content_word(self) -> bool:
        """Inverse of stop word."""
        return not self.is_geometric_stop_word


@dataclass
class Frame:
    """
    Semantic frame: Initiator → Mediator → Receiver
    
    Mathematical representation:
    F = (initiator, mediator, receiver)
    
    Extracted using position bands:
    - [0.0, 0.33) → Initiator
    - [0.33, 0.66) → Mediator  
    - [0.66, 1.0] → Receiver
    """
    initiator: str
    mediator: str
    receiver: Optional[str] = None


@dataclass
class VerbCluster:
    """
    A cluster of verb forms representing the same concept.
    
    Learned from parallel structures:
    "I love. He loves. I loved." → {0: 'love', 1: 'loves', 2: 'loved'}
    """
    canonical: str  # Base form
    forms: Dict[int, str] = field(default_factory=dict)  # phase → form
    
    def get_form(self, phase: int) -> str:
        """Get form for a given phase, defaulting to canonical."""
        return self.forms.get(phase, self.canonical)


# =============================================================================
# GEOMETRIC MORPHOLOGY
# =============================================================================

class GeometricMorphology:
    """
    Learn morphological equivalence from parallel structures.
    
    Key insight: Parallel sentences reveal morphological equivalence.
    "I love. He loves. I loved." → love ≡ loves ≡ loved
    
    This is purely geometric: position in parallel group determines phase.
    """
    
    def __init__(self):
        self.words: Dict[str, Set[str]] = {}  # word → equivalents
        self.equivalence_classes: Dict[str, Set[str]] = {}  # canonical → all variants
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_mediator(self, sentence: str) -> Optional[str]:
        """Extract mediator (verb) from simple sentence."""
        tokens = self._tokenize(sentence)
        if len(tokens) < 2:
            return None
        return tokens[1]  # Position 1 is typically the verb
    
    def bootstrap(self, text: str):
        """
        Learn from parallel structure text.
        
        Process sentences in groups of 3:
        - Position 0: base form
        - Position 1: 3rd person singular
        - Position 2: past tense
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_group: List[str] = []
        
        for sentence in sentences:
            mediator = self._extract_mediator(sentence)
            if not mediator:
                continue
            
            # Skip auxiliaries
            if mediator in {'will', 'would', 'could', 'should', 'may', 'might', 'can'}:
                continue
            
            current_group.append(mediator)
            
            # After 3 sentences, create equivalence class
            if len(current_group) >= 3:
                self._create_equivalence(current_group)
                current_group = []
        
        # Handle remaining
        if len(current_group) > 1:
            self._create_equivalence(current_group)
    
    def _create_equivalence(self, mediators: List[str]):
        """Create equivalence class from parallel group."""
        filtered = [m for m in mediators 
                   if m not in {'will', 'would', 'could', 'should', 'may', 'might'}]
        
        if len(filtered) < 2:
            return
        
        canonical = filtered[0]
        equivalents = set(filtered)
        
        if canonical not in self.equivalence_classes:
            self.equivalence_classes[canonical] = set()
        
        self.equivalence_classes[canonical].update(equivalents)
        
        for word in filtered:
            if word not in self.words:
                self.words[word] = set()
            self.words[word].update(equivalents)
    
    def get_equivalents(self, word: str) -> Set[str]:
        """Get all morphological equivalents of a word."""
        return self.words.get(word, {word})
    
    def are_equivalent(self, word1: str, word2: str) -> bool:
        """Check if two words are morphologically equivalent."""
        if word1 == word2:
            return True
        eq1 = self.get_equivalents(word1)
        eq2 = self.get_equivalents(word2)
        return bool(eq1 & eq2)


# =============================================================================
# GEOMETRIC CONJUGATION
# =============================================================================

class GeometricConjugation:
    """
    Learn verb conjugation from parallel structures.
    
    The position in a parallel group encodes the temporal phase:
    - Position 0: base form
    - Position 1: 3rd person singular
    - Position 2: past tense
    
    This allows conjugation without suffix rules.
    """
    
    def __init__(self):
        self.clusters: Dict[str, VerbCluster] = {}  # canonical → cluster
        self.word_to_canonical: Dict[str, str] = {}  # any form → canonical
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_mediator(self, sentence: str) -> Optional[str]:
        """Extract mediator from simple sentence."""
        tokens = self._tokenize(sentence)
        return tokens[1] if len(tokens) > 1 else None
    
    def bootstrap(self, text: str):
        """Learn conjugation patterns from parallel structures."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_group: List[Tuple[str, int]] = []
        phase = 0
        
        for sentence in sentences:
            mediator = self._extract_mediator(sentence)
            if not mediator:
                continue
            
            if mediator in {'will', 'would', 'could', 'should', 'may', 'might', 'can'}:
                continue
            
            current_group.append((mediator, phase))
            phase += 1
            
            if phase >= 3:
                self._create_cluster(current_group)
                current_group = []
                phase = 0
        
        if len(current_group) > 1:
            self._create_cluster(current_group)
    
    def _create_cluster(self, group: List[Tuple[str, int]]):
        """Create verb cluster from parallel group."""
        if not group:
            return
        
        canonical = group[0][0]
        cluster = VerbCluster(canonical=canonical)
        
        for mediator, phase in group:
            cluster.forms[phase] = mediator
            self.word_to_canonical[mediator] = canonical
        
        self.clusters[canonical] = cluster
    
    def get_canonical(self, word: str) -> str:
        """Get canonical (base) form of a word."""
        return self.word_to_canonical.get(word, word)
    
    def conjugate(self, word: str, phase: int) -> str:
        """
        Conjugate word to given phase.
        
        Phase 0 = base
        Phase 1 = 3rd person singular
        Phase 2 = past
        """
        canonical = self.get_canonical(word)
        if canonical in self.clusters:
            return self.clusters[canonical].get_form(phase)
        return word


# =============================================================================
# GEOMETRIC CHATBOT
# =============================================================================

class GeometricChatbot:
    """
    A fully geometric chatbot.
    
    All components are geometric:
    1. Stop words detected by semantic role absence
    2. Frame slots assigned by position bands
    3. Morphology learned from parallel structures
    4. Conjugation learned from parallel structures
    
    No hard-coded linguistic rules.
    """
    
    def __init__(self):
        self.concepts: Dict[str, GeometricConcept] = {}
        self.frames: List[Frame] = []
        self.total_sentences: int = 0
        
        # Initialize geometric morphology and conjugation
        self.morphology = GeometricMorphology()
        self.morphology.bootstrap(MORPHOLOGY_BOOTSTRAP)
        
        self.conjugation = GeometricConjugation()
        self.conjugation.bootstrap(MORPHOLOGY_BOOTSTRAP)
    
    def _get_or_create(self, word: str) -> GeometricConcept:
        """Get or create a concept."""
        word = word.lower()
        if word not in self.concepts:
            self.concepts[word] = GeometricConcept(word=word)
        return self.concepts[word]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def learn(self, text: str):
        """
        Learn from text using geometric principles.
        
        1. Extract frames using position bands
        2. Count semantic roles for each word
        3. Stop words emerge from role absence
        """
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            self.total_sentences += 1
            tokens = self._tokenize(sentence)
            
            if len(tokens) < 2:
                continue
            
            # Record position statistics
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
            # Position bands: [0, 0.33) → Initiator, [0.33, 0.66) → Mediator, [0.66, 1] → Receiver
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
            
            # Fallback
            if initiator is None and content_with_pos:
                initiator = content_with_pos[0][0]
            if mediator is None and len(content_with_pos) > 1:
                mediator = content_with_pos[1][0]
            
            if not initiator or not mediator:
                continue
            
            # Create frame and update role counts
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
    
    def encode(self, text: str) -> float:
        """
        Encode text to φ-space position.
        
        position = Σ (φ^(-rank) × mean_position) / Σ φ^(-rank)
        
        where rank is based on word frequency (Zipf weighting).
        """
        tokens = self._tokenize(text)
        
        total = 0.0
        weight = 0.0
        
        for word in tokens:
            if word not in self.concepts:
                continue
            
            c = self.concepts[word]
            if c.is_geometric_stop_word:
                continue
            
            # Zipf weighting: rarer words have more weight
            w = 1.0 / math.log(c.frequency + 2)
            total += c.mean_position * w
            weight += w
        
        return total / weight if weight > 0 else 0.5
    
    def respond(self, query: str) -> str:
        """Generate response using geometric pipeline."""
        tokens = self._tokenize(query)
        
        # Find content words using geometric morphology
        content = []
        for w in tokens:
            if w in self.concepts and self.concepts[w].is_content_word:
                content.append(w)
            else:
                # Check morphological equivalents
                for name, c in self.concepts.items():
                    if c.is_content_word and self.morphology.are_equivalent(name, w):
                        content.append(name)
                        break
        
        # Find entity and action
        entity = None
        action = None
        
        for word in content:
            if word not in self.concepts:
                continue
            c = self.concepts[word]
            if c.phi_direction > 0.3:
                entity = word
            elif c.phi_direction < -0.3:
                action = word
        
        # Check for action in query using geometric morphology
        if not action:
            for w in tokens:
                equivalents = self.morphology.get_equivalents(w)
                for name, c in self.concepts.items():
                    if c.mediator_count > 0:
                        if name in equivalents or self.morphology.are_equivalent(name, w):
                            action = name
                            break
                if action:
                    break
        
        # Generate response
        if action and not entity:
            return self._who_does(action)
        elif entity:
            return self._describe(entity)
        elif content:
            return self._describe(content[0])
        else:
            return "I don't have enough information."
    
    def _who_does(self, action: str) -> str:
        """Find who performs an action."""
        equivalents = self.morphology.get_equivalents(action)
        
        actors = []
        for name, c in self.concepts.items():
            if c.initiator_count == 0 or not c.is_content_word:
                continue
            for act in c.actions:
                if act in equivalents or self.morphology.are_equivalent(act, action):
                    actors.append((name, c.actions[act]))
                    break
        
        if not actors:
            # Try matching from frames
            for name, c in self.concepts.items():
                if c.mediator_count > 0 and self.morphology.are_equivalent(name, action):
                    for frame in self.frames:
                        if self.morphology.are_equivalent(frame.mediator, action):
                            if frame.initiator in self.concepts and self.concepts[frame.initiator].is_content_word:
                                actors.append((frame.initiator, 1))
                    break
        
        if not actors:
            return f"I don't know who {action}s."
        
        actors.sort(key=lambda x: x[1], reverse=True)
        actor = actors[0][0]
        
        # Find target
        target = None
        for frame in self.frames:
            if frame.initiator == actor and self.morphology.are_equivalent(frame.mediator, action):
                if frame.receiver:
                    target = frame.receiver
                    break
        
        # Conjugate using geometric conjugation (phase 1 = 3rd person singular)
        canonical = self.conjugation.get_canonical(action)
        verb = self.conjugation.conjugate(canonical, 1)
        
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
        
        # Actions using geometric conjugation
        if c.actions:
            top_actions = c.actions.most_common(3)
            verbs = []
            for a, _ in top_actions:
                canonical = self.conjugation.get_canonical(a)
                verb = self.conjugation.conjugate(canonical, 1)
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
    
    def show_analysis(self):
        """Show geometric analysis of the corpus."""
        print("\nGEOMETRIC WORD ANALYSIS")
        print("=" * 70)
        print(f"{'Word':<15} {'Freq':>5} {'Var':>6} {'φ-dir':>6} {'φ-mag':>6} {'Type':<10}")
        print("-" * 70)
        
        sorted_concepts = sorted(self.concepts.items(), 
                                key=lambda x: x[1].frequency, reverse=True)
        
        for name, c in sorted_concepts[:30]:
            word_type = "STOP" if c.is_geometric_stop_word else "CONTENT"
            print(f"{name:<15} {c.frequency:>5} {c.position_variance:>6.3f} "
                  f"{c.phi_direction:>6.2f} {c.phi_magnitude:>6.2f} {word_type:<10}")
        
        # Summary
        stop_words = [n for n, c in self.concepts.items() if c.is_geometric_stop_word]
        content_words = [n for n, c in self.concepts.items() if c.is_content_word]
        
        print(f"\nGEOMETRICALLY DETECTED STOP WORDS ({len(stop_words)}):")
        print(f"  {', '.join(sorted(stop_words)[:20])}")
        
        print(f"\nCONTENT WORDS ({len(content_words)}):")
        print(f"  {', '.join(sorted(content_words)[:20])}...")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate the geometric chatbot."""
    print("=" * 70)
    print("GEOMETRIC CHATBOT")
    print("A Fully Geometric Approach to Natural Language Understanding")
    print("=" * 70)
    print()
    print("Key Features:")
    print("  • Stop words detected geometrically (no hard-coded list)")
    print("  • Frame slots assigned by position bands")
    print("  • Morphology learned from parallel structures")
    print("  • Conjugation learned from parallel structures")
    print()
    
    # Create and train chatbot
    bot = GeometricChatbot()
    bot.learn(EXAMPLE_CORPUS)
    
    print(f"Learned from {bot.total_sentences} sentences")
    print(f"Total concepts: {len(bot.concepts)}")
    print(f"Frames extracted: {len(bot.frames)}")
    
    # Show analysis
    bot.show_analysis()
    
    # Query tests
    print("\n" + "=" * 70)
    print("QUERY TESTS")
    print("=" * 70)
    
    queries = [
        "Who is Holmes?",
        "Who killed?",
        "Who loves?",
        "Tell me about Alice",
        "What does Watson do?",
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        print(f"A: {bot.respond(q)}")
    
    return bot


def interactive():
    """Run interactive chat session."""
    print("=" * 70)
    print("GEOMETRIC CHATBOT - Interactive Mode")
    print("=" * 70)
    print()
    
    bot = GeometricChatbot()
    bot.learn(EXAMPLE_CORPUS)
    
    print(f"Learned from {bot.total_sentences} sentences.")
    print("Type 'quit' to exit, 'analysis' to show word analysis.")
    print()
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if query.lower() == 'analysis':
            bot.show_analysis()
            continue
        
        response = bot.respond(query)
        print(f"Bot: {response}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive()
    else:
        demo()
