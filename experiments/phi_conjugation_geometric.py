#!/usr/bin/env python3
"""
φ Conjugation Geometric: Learning Verb Conjugation from Parallel Structures

The bootstrap approach extended: Learn not just equivalence, but DIRECTION.

"I love. I loved." → 
  - love and loved are equivalent
  - love is at position 0 (present/base)
  - loved is at position 1 (past)

The position in the parallel group encodes the temporal phase.
We can then use this to conjugate: given a base form and target phase,
look up the form at that position.

Author: Lesley Gushurst
License: GPLv3
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

PHI = 1.618034


@dataclass
class VerbForm:
    """A verb form with its phase position."""
    word: str
    phase: int  # 0 = present/base, 1 = past, 2 = future (if present)
    
    
@dataclass 
class VerbCluster:
    """A cluster of verb forms representing the same concept."""
    canonical: str  # The base/present form
    forms: Dict[int, str] = field(default_factory=dict)  # phase -> form
    
    def get_form(self, phase: int) -> str:
        """Get the form for a given phase, defaulting to canonical."""
        return self.forms.get(phase, self.canonical)


class GeometricConjugation:
    """
    Learn verb conjugation from parallel structures.
    
    Key insight: The POSITION in a parallel group encodes the PHASE.
    
    "I love. I loved." →
      - Position 0: love (present)
      - Position 1: loved (past)
    
    This is purely geometric: position determines phase.
    """
    
    def __init__(self):
        self.clusters: Dict[str, VerbCluster] = {}  # canonical -> cluster
        self.word_to_canonical: Dict[str, str] = {}  # any form -> canonical
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_mediator(self, sentence: str) -> Optional[str]:
        """Extract the mediator (verb) from a sentence."""
        tokens = self._tokenize(sentence)
        if len(tokens) < 2:
            return None
        
        # For simple "Subject Verb" sentences, verb is at position 1
        # (position 0 is the subject like "I", "He", "She")
        return tokens[1] if len(tokens) > 1 else None
    
    def bootstrap(self, text: str):
        """
        Learn conjugation patterns from parallel structure text.
        
        The text should contain parallel sentences that reveal
        conjugation relationships. The ORDER of sentences matters:
        - First sentence = present/base form
        - Second sentence = past form
        - Third sentence = future form (if present)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Process sentences in groups of 3 (base, 3rd-singular, past)
        # The bootstrap format is: "I love. He loves. I loved."
        # We detect groups by looking for the pattern: I/He/I or similar
        
        current_group: List[Tuple[str, int]] = []  # (mediator, phase)
        phase = 0
        
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            if len(tokens) < 2:
                continue
            
            subject = tokens[0]
            mediator = self._extract_mediator(sentence)
            
            if not mediator:
                continue
            
            # Skip auxiliary verbs
            if mediator in {'will', 'would', 'could', 'should', 'may', 'might', 'can'}:
                continue
            
            # Add to current group
            current_group.append((mediator, phase))
            phase += 1
            
            # After 3 sentences, create cluster and reset
            # (base, 3rd-singular, past)
            if phase >= 3:
                self._create_cluster(current_group)
                current_group = []
                phase = 0
        
        # Handle any remaining group
        if len(current_group) > 1:
            self._create_cluster(current_group)
    
    def _create_cluster(self, group: List[Tuple[str, int]]):
        """Create a verb cluster from a parallel group."""
        if not group:
            return
        
        # First form (position 0) is the canonical/present form
        canonical = group[0][0]
        
        cluster = VerbCluster(canonical=canonical)
        
        for mediator, phase in group:
            cluster.forms[phase] = mediator
            self.word_to_canonical[mediator] = canonical
        
        self.clusters[canonical] = cluster
    
    def get_canonical(self, word: str) -> str:
        """Get the canonical (base/present) form of a word."""
        return self.word_to_canonical.get(word, word)
    
    def conjugate(self, word: str, phase: int) -> str:
        """
        Conjugate a word to a given phase.
        
        Phase 0 = present/base
        Phase 1 = past
        Phase 2 = future (if learned)
        """
        canonical = self.get_canonical(word)
        
        if canonical in self.clusters:
            return self.clusters[canonical].get_form(phase)
        
        # Unknown verb - return as-is
        return word
    
    def show_clusters(self):
        """Show learned conjugation clusters."""
        print("\nCONJUGATION CLUSTERS (Geometric)")
        print("=" * 60)
        
        for canonical, cluster in sorted(self.clusters.items()):
            forms = [f"phase {p}: {f}" for p, f in sorted(cluster.forms.items())]
            print(f"  {canonical}: {', '.join(forms)}")


# Bootstrap text with explicit phase ordering
# Position 0 = base/infinitive, Position 1 = 3rd person singular, Position 2 = past
# This teaches THREE forms: base, present-3rd, past
CONJUGATION_BOOTSTRAP = """
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


def demo():
    """Demonstrate geometric conjugation learning."""
    print("φ CONJUGATION: Geometric Learning")
    print("=" * 60)
    print()
    print("Learning conjugation from parallel structures...")
    print("Position 0 = present, Position 1 = past")
    print()
    
    conj = GeometricConjugation()
    conj.bootstrap(CONJUGATION_BOOTSTRAP)
    
    print(f"Verb clusters learned: {len(conj.clusters)}")
    
    conj.show_clusters()
    
    # Test conjugation
    print("\n" + "=" * 60)
    print("CONJUGATION TESTS")
    print("=" * 60)
    
    # Phase 0 = base, Phase 1 = 3rd singular, Phase 2 = past
    test_cases = [
        ("love", 0, "base"),
        ("love", 1, "3rd singular"),
        ("love", 2, "past"),
        ("loved", 0, "base (from past form)"),
        ("loved", 2, "past (from past form)"),
        ("go", 0, "base"),
        ("go", 1, "3rd singular"),
        ("go", 2, "past"),
        ("went", 0, "base (from past form)"),
        ("see", 2, "past"),
        ("think", 2, "past"),
        ("watch", 2, "past"),
    ]
    
    for word, phase, description in test_cases:
        result = conj.conjugate(word, phase)
        print(f"  {word} → phase {phase} ({description}): {result}")
    
    return conj


if __name__ == "__main__":
    demo()
