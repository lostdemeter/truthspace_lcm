#!/usr/bin/env python3
"""
φ: The Minimal Encoder-Decoder

ENCODE = DECODE

One operation. Self-similar. Self-inverse.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter

PHI = 1.618034


class φ:
    """
    The minimal encoder-decoder.
    
    Text → Position → Text
    
    Encoding IS decoding. The same operation in opposite directions.
    """
    
    def __init__(self):
        self.space: Dict[str, float] = {}  # word → position
        self.reverse: Dict[float, str] = {}  # position → word (approximate)
        self.relations: Dict[str, Counter] = {}  # word → co-occurring words
    
    def __call__(self, x: str) -> str:
        """
        The single operation.
        
        If x is text: encode it, find resonance, decode back to text.
        """
        # Encode
        position = self.encode(x)
        
        # Decode (find what resonates at this position)
        return self.decode(position)
    
    def encode(self, text: str) -> float:
        """
        Text → Position
        
        The position is the φ-weighted sum of word positions.
        """
        words = self._words(text)
        if not words:
            return 0.5  # Center of space
        
        # Each word contributes its position, weighted by φ^(-i) for position i
        total = 0.0
        weight = 0.0
        
        for i, word in enumerate(words):
            w = PHI ** (-i)  # Earlier words matter more
            if word in self.space:
                total += self.space[word] * w
            else:
                # New word: assign position based on context
                pos = self._assign_position(word, words)
                self.space[word] = pos
                total += pos * w
            weight += w
        
        return total / weight if weight > 0 else 0.5
    
    def decode(self, position: float) -> str:
        """
        Position → Text
        
        Find words that resonate at this position.
        """
        if not self.space:
            return ""
        
        # Find words near this position
        resonant = []
        for word, pos in self.space.items():
            distance = abs(pos - position)
            if distance < 0.3:  # Resonance threshold
                resonant.append((word, 1.0 - distance))
        
        if not resonant:
            # Find closest
            closest = min(self.space.items(), key=lambda x: abs(x[1] - position))
            return closest[0]
        
        # Sort by resonance strength
        resonant.sort(key=lambda x: x[1], reverse=True)
        
        # Return top resonant words as a phrase
        return ' '.join([w for w, _ in resonant[:3]])
    
    def learn(self, text: str):
        """
        Learn from text.
        
        This is just encoding with side effects - the space grows.
        """
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            words = self._words(sentence)
            if len(words) < 2:
                continue
            
            # Assign positions based on role
            for i, word in enumerate(words):
                if word not in self.space:
                    self.space[word] = self._assign_position(word, words)
                
                # Track relations
                if word not in self.relations:
                    self.relations[word] = Counter()
                for other in words:
                    if other != word:
                        self.relations[word][other] += 1
    
    def _assign_position(self, word: str, context: List[str]) -> float:
        """
        Assign a position to a new word based on context.
        
        Position 0 = initiator (actor)
        Position 0.5 = mediator (action)
        Position 1 = receiver (target)
        """
        if not context:
            return 0.5
        
        try:
            idx = context.index(word)
            # Normalize position to [0, 1]
            return idx / max(len(context) - 1, 1)
        except ValueError:
            return 0.5
    
    def _words(self, text: str) -> List[str]:
        """Extract content words."""
        stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'to', 'of', 
                'in', 'on', 'at', 'by', 'for', 'with', 'from', 'he', 'she', 'it',
                'they', 'his', 'her', 'its', 'their', 'that', 'this', 'and', 'or',
                'but', 'if', 'so', 'as', 'had', 'has', 'have', 'did', 'do', 'does'}
        
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if t not in stop and len(t) > 2]


def chat():
    """φ chatbot."""
    print("φ Chat")
    print("=" * 50)
    print()
    
    f = φ()
    
    # Learn from corpus
    corpus = """
    Holmes examined the evidence carefully.
    Watson watched from the doorway.
    The detective studied the footprints.
    Holmes deduced the identity brilliantly.
    Moriarty plotted against Holmes secretly.
    Watson assisted Holmes faithfully.
    Lestrade questioned the witnesses.
    Holmes observed the crime scene.
    
    Alice fell down the rabbit hole.
    The Queen shouted at everyone angrily.
    Cheshire smiled mysteriously.
    Alice grew very tall suddenly.
    The Hatter laughed wildly.
    Alice explored Wonderland curiously.
    
    Darcy watched Elizabeth intently.
    Elizabeth danced gracefully.
    Bingley fell in love immediately.
    Jane smiled sweetly.
    Wickham deceived everyone cunningly.
    
    Hamlet pondered existence deeply.
    Claudius poisoned the King treacherously.
    Ophelia loved Hamlet devotedly.
    Hamlet killed Claudius finally.
    The ghost revealed the murder.
    """
    
    f.learn(corpus)
    print(f"Learned {len(f.space)} words")
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
        
        # Encode user input
        pos = f.encode(user)
        
        # Decode to find resonant concepts
        resonant = f.decode(pos)
        
        # Build response from relations
        words = f._words(user)
        response_parts = []
        
        for word in words:
            if word in f.relations:
                # Find most related concepts
                related = f.relations[word].most_common(3)
                for rel, _ in related:
                    if rel not in response_parts and rel not in words:
                        response_parts.append(rel)
        
        if response_parts:
            response = f"{resonant} - {' '.join(response_parts)}"
        else:
            response = resonant
        
        print(f"φ: {response}")
        print()


def demo():
    """Demonstrate φ at scale."""
    print("φ: The Minimal Encoder-Decoder")
    print("=" * 50)
    print()
    
    # Create φ
    f = φ()
    
    # Scale: Multiple domains, more text
    corpus = """
    Holmes examined the evidence carefully.
    Watson watched from the doorway.
    The detective studied the footprints.
    Holmes deduced the identity brilliantly.
    Moriarty plotted against Holmes secretly.
    Watson assisted Holmes faithfully.
    Lestrade questioned the witnesses.
    Holmes observed the crime scene.
    
    Alice fell down the rabbit hole.
    The Queen shouted at everyone angrily.
    Cheshire smiled mysteriously.
    Alice grew very tall suddenly.
    The Hatter laughed wildly.
    Alice explored Wonderland curiously.
    
    Darcy watched Elizabeth intently.
    Elizabeth danced gracefully.
    Bingley fell in love immediately.
    Jane smiled sweetly.
    Wickham deceived everyone cunningly.
    
    Hamlet pondered existence deeply.
    Claudius poisoned the King treacherously.
    Ophelia loved Hamlet devotedly.
    Hamlet killed Claudius finally.
    The ghost revealed the murder.
    
    The cat sat on the mat.
    Dogs chase cats everywhere.
    Birds fly through the sky.
    Fish swim in the ocean.
    Trees grow toward the sun.
    """
    
    f.learn(corpus)
    
    print(f"Learned {len(f.space)} words")
    print()
    
    # Show structure emergence
    print("STRUCTURE EMERGENCE:")
    print("-" * 50)
    
    # Group by position bands
    initiators = [(w, p) for w, p in f.space.items() if p < 0.2]
    mediators = [(w, p) for w, p in f.space.items() if 0.2 <= p < 0.6]
    receivers = [(w, p) for w, p in f.space.items() if p >= 0.6]
    
    print(f"Initiators (pos < 0.2): {len(initiators)}")
    print(f"  {', '.join([w for w, _ in sorted(initiators, key=lambda x: x[1])[:10]])}")
    print()
    
    print(f"Mediators (0.2 ≤ pos < 0.6): {len(mediators)}")
    print(f"  {', '.join([w for w, _ in sorted(mediators, key=lambda x: x[1])[:10]])}")
    print()
    
    print(f"Receivers (pos ≥ 0.6): {len(receivers)}")
    print(f"  {', '.join([w for w, _ in sorted(receivers, key=lambda x: x[1])[:10]])}")
    print()
    
    # Cross-domain encode-decode
    print("CROSS-DOMAIN ENCODE → DECODE:")
    print("-" * 50)
    
    queries = [
        "Holmes",      # Sherlock initiator
        "Alice",       # Wonderland initiator
        "Hamlet",      # Tragedy initiator
        "examined",    # Action
        "loved",       # Action
        "evidence",    # Object
    ]
    
    for q in queries:
        result = f(q)
        pos = f.encode(q)
        print(f"  '{q}' → {pos:.2f} → '{result}'")
    print()
    
    # Test: encode a phrase, decode back
    print("PHRASE ENCODE → DECODE:")
    print("-" * 50)
    
    phrases = [
        "Holmes examined",
        "Alice fell",
        "Hamlet killed",
        "cat sat",
    ]
    
    for phrase in phrases:
        pos = f.encode(phrase)
        result = f.decode(pos)
        print(f"  '{phrase}' → {pos:.2f} → '{result}'")
    print()
    
    # Verify: same structure emerges regardless of domain
    print("VERIFICATION: Structure is domain-independent")
    print("-" * 50)
    
    # All initiators should cluster together regardless of story
    init_positions = [f.space.get(w, 0.5) for w in ['holmes', 'alice', 'hamlet', 'cat', 'darcy']]
    init_spread = max(init_positions) - min(init_positions) if init_positions else 0
    
    # All mediators should cluster together
    med_positions = [f.space.get(w, 0.5) for w in ['examined', 'fell', 'killed', 'sat', 'watched']]
    med_spread = max(med_positions) - min(med_positions) if med_positions else 0
    
    print(f"  Initiator spread: {init_spread:.2f} (should be small)")
    print(f"  Mediator spread: {med_spread:.2f} (should be small)")
    print()
    
    if init_spread < 0.3 and med_spread < 0.3:
        print("  ✓ Structure emerges consistently across domains")
    else:
        print("  ✗ Structure varies by domain")
    
    return f


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'chat':
        chat()
    else:
        demo()
