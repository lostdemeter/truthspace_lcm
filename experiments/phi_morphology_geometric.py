#!/usr/bin/env python3
"""
φ Morphology Geometric: Learning Morphology from Parallel Structures

The bootstrap approach: Parallel sentences reveal morphological equivalence.

"I love. I loved." → love ≡ loved (same concept, different phase)
"He runs. He ran." → runs ≡ ran (same concept, different phase)

Words in the same frame slot across parallel structures are the same concept
at different temporal phases.

This is purely geometric:
- Position = concept identity
- Phase = temporal aspect

No suffix patterns, no word length heuristics.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

PHI = 1.618034


@dataclass
class ConceptWord:
    """A word with concept space properties."""
    word: str
    
    # Position in concept space (from frame role)
    positions: List[float] = field(default_factory=list)
    
    # Morphological equivalents (same concept, different phase)
    equivalents: Set[str] = field(default_factory=set)
    
    # The canonical form (first seen or most frequent)
    canonical: Optional[str] = None
    
    @property
    def mean_position(self) -> float:
        if not self.positions:
            return 0.5
        return sum(self.positions) / len(self.positions)


class GeometricMorphology:
    """
    Learn morphology from parallel structures in concept space.
    
    Key insight: Parallel sentences with the same structure but different
    words in the same slot reveal morphological equivalence.
    
    "I love. I loved." → The mediator slot contains "love" and "loved"
    These are the same concept at different phases.
    """
    
    def __init__(self):
        self.words: Dict[str, ConceptWord] = {}
        self.equivalence_classes: Dict[str, Set[str]] = {}  # canonical -> all variants
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_or_create(self, word: str) -> ConceptWord:
        if word not in self.words:
            self.words[word] = ConceptWord(word=word)
        return self.words[word]
    
    def _extract_frame(self, sentence: str) -> Optional[Tuple[str, str, Optional[str]]]:
        """Extract (initiator, mediator, receiver) from a sentence."""
        tokens = self._tokenize(sentence)
        
        if len(tokens) < 2:
            return None
        
        # For bootstrap sentences, we want ALL words including pronouns
        # The geometric structure is: position 0 = initiator, position 1 = mediator
        content = [(i, w) for i, w in enumerate(tokens)]
        
        # Assign by normalized position
        n = len(tokens) - 1
        if n == 0:
            n = 1
        
        initiator = None
        mediator = None
        receiver = None
        
        for idx, word in content:
            pos = idx / n
            if pos < 0.33 and initiator is None:
                initiator = word
            elif pos < 0.66 and mediator is None:
                mediator = word
            elif receiver is None:
                receiver = word
        
        # Fallback
        if initiator is None and content:
            initiator = content[0][1]
        if mediator is None and len(content) > 1:
            mediator = content[1][1]
        
        if initiator and mediator:
            return (initiator, mediator, receiver)
        return None
    
    def learn_parallel(self, sentences: List[str]):
        """
        Learn morphological equivalence from parallel sentences.
        
        Process sentences in groups of 3 (base, 3rd-singular, past).
        The bootstrap format is: "I love. He loves. I loved."
        
        All three mediators in a group are equivalent.
        """
        # Process sentences in groups of 3
        current_group: List[str] = []  # Mediators in current group
        
        for sentence in sentences:
            frame = self._extract_frame(sentence)
            if not frame:
                continue
            
            initiator, mediator, receiver = frame
            
            # Skip auxiliary verbs
            if mediator in {'will', 'would', 'could', 'should', 'may', 'might', 'can'}:
                continue
            
            # Record position
            w = self._get_or_create(mediator)
            w.positions.append(0.5)
            
            # Add to current group
            current_group.append(mediator)
            
            # After 3 sentences, create equivalence and reset
            if len(current_group) >= 3:
                self._create_equivalence(current_group)
                current_group = []
        
        # Handle any remaining group
        if len(current_group) > 1:
            self._create_equivalence(current_group)
    
    def _create_equivalence(self, mediators: List[str]):
        """Create an equivalence class from a list of mediators."""
        # Filter out "will" and other auxiliaries that appear in future tense
        # These are structural, not content
        filtered = [m for m in mediators if m not in {'will', 'would', 'could', 'should', 'may', 'might'}]
        
        if len(filtered) < 2:
            return
        
        canonical = filtered[0]
        
        if canonical not in self.equivalence_classes:
            self.equivalence_classes[canonical] = set()
        
        for med in filtered:
            self.equivalence_classes[canonical].add(med)
            w = self._get_or_create(med)
            w.equivalents.update(filtered)
            w.canonical = canonical
    
    def bootstrap(self, bootstrap_text: str):
        """
        Bootstrap morphology from parallel structure text.
        
        The bootstrap text should contain parallel sentences that
        reveal morphological relationships.
        """
        sentences = re.split(r'[.!?]+', bootstrap_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        self.learn_parallel(sentences)
    
    def get_canonical(self, word: str) -> str:
        """Get the canonical form of a word."""
        if word in self.words:
            w = self.words[word]
            if w.canonical:
                return w.canonical
        return word
    
    def get_equivalents(self, word: str) -> Set[str]:
        """Get all morphological equivalents of a word."""
        if word in self.words:
            return self.words[word].equivalents
        return {word}
    
    def are_equivalent(self, word1: str, word2: str) -> bool:
        """Check if two words are morphologically equivalent."""
        if word1 == word2:
            return True
        
        eq1 = self.get_equivalents(word1)
        eq2 = self.get_equivalents(word2)
        
        return bool(eq1 & eq2)
    
    def show_equivalences(self):
        """Show discovered morphological equivalences."""
        print("\nMORPHOLOGICAL EQUIVALENCES (Geometric Bootstrap)")
        print("=" * 60)
        
        shown = set()
        for canonical, variants in self.equivalence_classes.items():
            if canonical in shown:
                continue
            if len(variants) > 1:
                print(f"  {canonical} ≡ {', '.join(sorted(variants - {canonical}))}")
                shown.update(variants)


# Bootstrap text: parallel structures that reveal morphology
BOOTSTRAP = """
I love. I loved. I will love.
He runs. He ran. He will run.
She sees. She saw. She will see.
They watch. They watched. They will watch.
We go. We went. We will go.
It falls. It fell. It will fall.
You speak. You spoke. You will speak.
I write. I wrote. I will write.
He reads. He read. He will read.
She gives. She gave. She will give.
They take. They took. They will take.
We make. We made. We will make.
It grows. It grew. It will grow.
You know. You knew. You will know.
I think. I thought. I will think.
He says. He said. He will say.
She comes. She came. She will come.
They find. They found. They will find.
We leave. We left. We will leave.
It begins. It began. It will begin.
"""


def demo():
    """Demonstrate geometric morphology learning."""
    print("φ MORPHOLOGY: Geometric Bootstrap")
    print("=" * 60)
    print()
    print("Learning morphology from parallel structures...")
    print("No suffix patterns. No word length. Pure geometry.")
    print()
    
    morph = GeometricMorphology()
    morph.bootstrap(BOOTSTRAP)
    
    print(f"Words learned: {len(morph.words)}")
    print(f"Equivalence classes: {len(morph.equivalence_classes)}")
    
    morph.show_equivalences()
    
    # Test equivalence
    print("\n" + "=" * 60)
    print("EQUIVALENCE TESTS")
    print("=" * 60)
    
    test_pairs = [
        ("love", "loved"),
        ("runs", "ran"),
        ("watches", "watched"),
        ("sees", "saw"),
        ("go", "went"),
        ("write", "wrote"),
    ]
    
    for w1, w2 in test_pairs:
        eq = morph.are_equivalent(w1, w2)
        canonical = morph.get_canonical(w1)
        print(f"  {w1} ≡ {w2}? {eq} (canonical: {canonical})")
    
    # Test with corpus
    print("\n" + "=" * 60)
    print("APPLYING TO CORPUS")
    print("=" * 60)
    
    corpus_sentences = [
        "Holmes examined the evidence.",
        "Watson watched from the doorway.",
        "Alice fell down the rabbit hole.",
        "Ophelia loved Hamlet deeply.",
    ]
    
    for sentence in corpus_sentences:
        frame = morph._extract_frame(sentence)
        if frame:
            init, med, recv = frame
            canonical_med = morph.get_canonical(med)
            print(f"  '{sentence}'")
            print(f"    Frame: {init} -> {med} -> {recv or '∅'}")
            print(f"    Canonical mediator: {canonical_med}")
            print()
    
    return morph


if __name__ == "__main__":
    demo()
