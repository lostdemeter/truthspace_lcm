#!/usr/bin/env python3
"""
Experiment: Pure Geometric Vocabulary Transformation

The Music Box Principle:
- DRUM (structure): Words have positions in φ-space based on semantic dimensions
- COMB (decoder): Find nearest word at a given position
- MUSIC (output): Emerges from structure + decoder, NOT hard-coded

The key insight: We don't store "went -> will go" as a mapping.
Instead:
- "went" is at position [tense=-1, ...]  (past)
- "will go" is at position [tense=+1, ...]  (future)
- Transformation "past->future" is delta [+2, 0, 0, ...]
- Apply delta to "went" position, find nearest word = "will go"

The music emerges from the geometry. The comb doesn't contain the music.

For the Emperor. The Omnissiah provides.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# THE DRUM: Vocabulary with Geometric Positions
# =============================================================================

@dataclass
class WordPosition:
    """A word's position in semantic space."""
    word: str
    position: np.ndarray  # [tense, formality, domain, intensity, ...]
    
    def distance_to(self, other: np.ndarray) -> float:
        """Euclidean distance to another position."""
        return np.linalg.norm(self.position - other)


class GeometricVocabulary:
    """
    Vocabulary organized by geometric position.
    
    Each word has a position in multi-dimensional semantic space.
    Dimensions might include:
    - tense: past (-1) / present (0) / future (+1)
    - formality: casual (-1) / neutral (0) / formal (+1) / archaic (+2)
    - domain: mundane (-1) / neutral (0) / technical (+1) / sacred (+2)
    - intensity: weak (-1) / normal (0) / strong (+1)
    
    The vocabulary is the DRUM - it contains the structure.
    """
    
    DIMENSIONS = ['tense', 'formality', 'domain', 'intensity']
    
    def __init__(self):
        self._words: Dict[str, WordPosition] = {}
        self._by_concept: Dict[str, List[str]] = defaultdict(list)  # concept -> [words]
    
    def add_word(self, word: str, position: np.ndarray, concept: Optional[str] = None):
        """Add a word at a specific position."""
        self._words[word.lower()] = WordPosition(word=word, position=position)
        if concept:
            self._by_concept[concept].append(word.lower())
    
    def get_position(self, word: str) -> Optional[np.ndarray]:
        """Get a word's position."""
        wp = self._words.get(word.lower())
        return wp.position if wp else None
    
    def find_nearest(self, position: np.ndarray, exclude: Optional[Set[str]] = None) -> Optional[str]:
        """
        Find the nearest word to a position.
        
        This is the COMB - it reads the structure and produces output.
        The comb doesn't know what word it will produce until it reads the position.
        """
        exclude = exclude or set()
        
        best_word = None
        best_distance = float('inf')
        
        for word, wp in self._words.items():
            if word in exclude:
                continue
            dist = wp.distance_to(position)
            if dist < best_distance:
                best_distance = dist
                best_word = word
        
        return best_word
    
    def find_nearest_in_concept(self, position: np.ndarray, concept: str) -> Optional[str]:
        """Find nearest word within a concept group."""
        concept_words = self._by_concept.get(concept, [])
        if not concept_words:
            return self.find_nearest(position)
        
        best_word = None
        best_distance = float('inf')
        
        for word in concept_words:
            wp = self._words.get(word)
            if wp:
                dist = wp.distance_to(position)
                if dist < best_distance:
                    best_distance = dist
                    best_word = word
        
        return best_word
    
    def transform(self, word: str, delta: np.ndarray, concept: Optional[str] = None) -> Optional[str]:
        """
        Transform a word by applying a delta vector.
        
        This is the MUSIC emerging:
        1. Get word's current position (read the drum)
        2. Apply delta (the rotation of the drum)
        3. Find nearest word at new position (comb produces sound)
        
        No lookup table. The output emerges from geometry.
        
        If concept is provided, only search within that concept group.
        This preserves semantic meaning while changing style.
        """
        current_pos = self.get_position(word)
        if current_pos is None:
            return None
        
        # Apply transformation
        new_pos = current_pos + delta
        
        # Find what word lives at the new position
        # If we know the concept, stay within it
        if concept:
            result = self.find_nearest_in_concept(new_pos, concept)
        else:
            # Try to find the concept this word belongs to
            for c, words in self._by_concept.items():
                if word.lower() in words:
                    result = self.find_nearest_in_concept(new_pos, c)
                    if result and result != word.lower():
                        return result
            # Fallback to global search
            result = self.find_nearest(new_pos, exclude={word.lower()})
        
        return result
    
    def stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'total_words': len(self._words),
            'concepts': len(self._by_concept),
            'dimensions': len(self.DIMENSIONS),
        }


# =============================================================================
# BOOTSTRAP: Build the Drum from Semantic Relationships
# =============================================================================

def build_verb_vocabulary() -> GeometricVocabulary:
    """
    Build a vocabulary of verbs organized by tense.
    
    Position dimensions: [tense, formality, domain, intensity]
    - tense: -1=past, 0=present, +1=future
    - formality: -1=casual, 0=neutral, +1=formal, +2=archaic
    - domain: -1=mundane, 0=neutral, +1=technical, +2=sacred
    - intensity: -1=weak, 0=normal, +1=strong
    """
    vocab = GeometricVocabulary()
    
    # GO concept - different tenses and formalities
    # [tense, formality, domain, intensity]
    vocab.add_word("went", np.array([-1, 0, 0, 0]), concept="GO")           # past, neutral
    vocab.add_word("go", np.array([0, 0, 0, 0]), concept="GO")              # present, neutral
    vocab.add_word("goes", np.array([0, 0, 0, 0]), concept="GO")            # present 3rd person
    vocab.add_word("will go", np.array([1, 0, 0, 0]), concept="GO")         # future, neutral
    vocab.add_word("shall go", np.array([1, 1, 0, 0]), concept="GO")        # future, formal
    vocab.add_word("did proceed", np.array([-1, 2, 0, 0]), concept="GO")    # past, archaic
    vocab.add_word("doth proceed", np.array([0, 2, 0, 0]), concept="GO")    # present, archaic
    vocab.add_word("shall proceed", np.array([1, 2, 0, 0]), concept="GO")   # future, archaic
    
    # SIT concept
    vocab.add_word("sat", np.array([-1, 0, 0, 0]), concept="SIT")
    vocab.add_word("sit", np.array([0, 0, 0, 0]), concept="SIT")
    vocab.add_word("sits", np.array([0, 0, 0, 0]), concept="SIT")
    vocab.add_word("will sit", np.array([1, 0, 0, 0]), concept="SIT")
    vocab.add_word("shall sit", np.array([1, 1, 0, 0]), concept="SIT")
    vocab.add_word("was seated", np.array([-1, 1, 0, 0]), concept="SIT")
    vocab.add_word("shall be seated", np.array([1, 2, 0, 0]), concept="SIT")
    
    # WALK concept
    vocab.add_word("walked", np.array([-1, 0, 0, 0]), concept="WALK")
    vocab.add_word("walk", np.array([0, 0, 0, 0]), concept="WALK")
    vocab.add_word("walks", np.array([0, 0, 0, 0]), concept="WALK")
    vocab.add_word("will walk", np.array([1, 0, 0, 0]), concept="WALK")
    vocab.add_word("strode", np.array([-1, 1, 0, 1]), concept="WALK")       # past, formal, strong
    vocab.add_word("shall stride", np.array([1, 1, 0, 1]), concept="WALK")  # future, formal, strong
    
    # SAY concept
    vocab.add_word("said", np.array([-1, 0, 0, 0]), concept="SAY")
    vocab.add_word("say", np.array([0, 0, 0, 0]), concept="SAY")
    vocab.add_word("says", np.array([0, 0, 0, 0]), concept="SAY")
    vocab.add_word("will say", np.array([1, 0, 0, 0]), concept="SAY")
    vocab.add_word("spoke", np.array([-1, 1, 0, 0]), concept="SAY")         # past, formal
    vocab.add_word("declared", np.array([-1, 1, 0, 1]), concept="SAY")      # past, formal, strong
    vocab.add_word("shall declare", np.array([1, 2, 0, 1]), concept="SAY")  # future, archaic, strong
    vocab.add_word("intoned", np.array([-1, 2, 2, 0]), concept="SAY")       # past, archaic, sacred
    vocab.add_word("shall intone", np.array([1, 2, 2, 0]), concept="SAY")   # future, archaic, sacred
    
    # KNOW concept
    vocab.add_word("knew", np.array([-1, 0, 0, 0]), concept="KNOW")
    vocab.add_word("know", np.array([0, 0, 0, 0]), concept="KNOW")
    vocab.add_word("knows", np.array([0, 0, 0, 0]), concept="KNOW")
    vocab.add_word("will know", np.array([1, 0, 0, 0]), concept="KNOW")
    vocab.add_word("understood", np.array([-1, 1, 0, 0]), concept="KNOW")
    vocab.add_word("comprehended", np.array([-1, 1, 1, 0]), concept="KNOW") # past, formal, technical
    vocab.add_word("divined", np.array([-1, 2, 2, 0]), concept="KNOW")      # past, archaic, sacred
    
    # MAKE concept
    vocab.add_word("made", np.array([-1, 0, 0, 0]), concept="MAKE")
    vocab.add_word("make", np.array([0, 0, 0, 0]), concept="MAKE")
    vocab.add_word("makes", np.array([0, 0, 0, 0]), concept="MAKE")
    vocab.add_word("will make", np.array([1, 0, 0, 0]), concept="MAKE")
    vocab.add_word("crafted", np.array([-1, 1, 0, 0]), concept="MAKE")
    vocab.add_word("forged", np.array([-1, 1, 0, 1]), concept="MAKE")       # past, formal, strong
    vocab.add_word("wrought", np.array([-1, 2, 0, 1]), concept="MAKE")      # past, archaic, strong
    vocab.add_word("shall forge", np.array([1, 1, 0, 1]), concept="MAKE")
    
    return vocab


def build_noun_vocabulary() -> GeometricVocabulary:
    """
    Build a vocabulary of nouns organized by domain/formality.
    
    This demonstrates how "code" -> "holy scripture" can emerge geometrically.
    """
    vocab = GeometricVocabulary()
    
    # CODE concept - different domains
    # [tense, formality, domain, intensity]
    vocab.add_word("code", np.array([0, 0, 1, 0]), concept="CODE")              # technical
    vocab.add_word("program", np.array([0, 0, 1, 0]), concept="CODE")           # technical
    vocab.add_word("script", np.array([0, 0, 1, 0]), concept="CODE")            # technical
    vocab.add_word("scripture", np.array([0, 1, 2, 0]), concept="CODE")         # formal, sacred
    vocab.add_word("holy scripture", np.array([0, 2, 2, 1]), concept="CODE")    # archaic, sacred, strong
    vocab.add_word("sacred text", np.array([0, 2, 2, 0]), concept="CODE")       # archaic, sacred
    vocab.add_word("treasure map", np.array([0, -1, -1, 0]), concept="CODE")    # casual, mundane (pirate)
    
    # COMPUTER concept
    vocab.add_word("computer", np.array([0, 0, 1, 0]), concept="COMPUTER")
    vocab.add_word("machine", np.array([0, 0, 1, 0]), concept="COMPUTER")
    vocab.add_word("cogitator", np.array([0, 2, 2, 0]), concept="COMPUTER")     # archaic, sacred (40k)
    vocab.add_word("thinking engine", np.array([0, 1, 1, 0]), concept="COMPUTER")
    vocab.add_word("magic box", np.array([0, -1, -1, 0]), concept="COMPUTER")   # casual, mundane (pirate)
    
    # DATA concept
    vocab.add_word("data", np.array([0, 0, 1, 0]), concept="DATA")
    vocab.add_word("information", np.array([0, 1, 1, 0]), concept="DATA")
    vocab.add_word("sacred data-hymns", np.array([0, 2, 2, 1]), concept="DATA") # archaic, sacred (40k)
    vocab.add_word("booty", np.array([0, -1, -1, 0]), concept="DATA")           # casual, mundane (pirate)
    vocab.add_word("lore", np.array([0, 1, 2, 0]), concept="DATA")              # formal, sacred
    
    # PROGRAMMER concept
    vocab.add_word("programmer", np.array([0, 0, 1, 0]), concept="PROGRAMMER")
    vocab.add_word("developer", np.array([0, 0, 1, 0]), concept="PROGRAMMER")
    vocab.add_word("coder", np.array([0, -1, 1, 0]), concept="PROGRAMMER")      # casual, technical
    vocab.add_word("tech-adept", np.array([0, 1, 2, 0]), concept="PROGRAMMER")  # formal, sacred (40k)
    vocab.add_word("code-priest", np.array([0, 2, 2, 1]), concept="PROGRAMMER") # archaic, sacred (40k)
    
    # ERROR concept
    vocab.add_word("error", np.array([0, 0, 1, 0]), concept="ERROR")
    vocab.add_word("bug", np.array([0, -1, 1, 0]), concept="ERROR")
    vocab.add_word("mistake", np.array([0, 0, 0, 0]), concept="ERROR")
    vocab.add_word("machine spirit's displeasure", np.array([0, 2, 2, 1]), concept="ERROR")  # 40k
    vocab.add_word("corruption", np.array([0, 1, 2, 0]), concept="ERROR")
    vocab.add_word("scurvy mistake", np.array([0, -1, -1, 0]), concept="ERROR") # pirate
    
    return vocab


# =============================================================================
# TRANSFORMATION DELTAS: The Rotation of the Drum
# =============================================================================

# These are the MOVEMENTS through the space, not word->word mappings
TRANSFORMATION_DELTAS = {
    # Tense transformations
    "past_to_future": np.array([2, 0, 0, 0]),      # tense: -1 -> +1
    "past_to_present": np.array([1, 0, 0, 0]),     # tense: -1 -> 0
    "present_to_future": np.array([1, 0, 0, 0]),   # tense: 0 -> +1
    "future_to_past": np.array([-2, 0, 0, 0]),     # tense: +1 -> -1
    
    # Formality transformations
    "casual_to_formal": np.array([0, 2, 0, 0]),    # formality: -1 -> +1
    "neutral_to_archaic": np.array([0, 2, 0, 0]),  # formality: 0 -> +2
    "formal_to_casual": np.array([0, -2, 0, 0]),   # formality: +1 -> -1
    
    # Domain transformations (perspective shifts!)
    "technical_to_sacred": np.array([0, 1, 1, 0]), # domain: +1 -> +2, formality boost
    "neutral_to_sacred": np.array([0, 2, 2, 0]),   # full grimdark shift
    "neutral_to_mundane": np.array([0, -1, -1, 0]), # pirate shift
    
    # Combined: Warhammer 40k perspective
    "warhammer40k": np.array([0, 2, 2, 0.5]),      # archaic + sacred + slight intensity
    
    # Combined: Pirate perspective  
    "pirate": np.array([0, -1, -1, 0]),            # casual + mundane
    
    # Combined: Shakespeare perspective
    "shakespeare": np.array([0, 2, 0, 0]),         # archaic formality
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_verb_transformation():
    """Show how verb tense transformation emerges from geometry."""
    print("=" * 70)
    print("VERB TRANSFORMATION: The Music Emerges from the Drum")
    print("=" * 70)
    print()
    
    vocab = build_verb_vocabulary()
    print(f"Vocabulary: {vocab.stats()}")
    print()
    
    # Test tense transformations
    test_words = ["went", "sat", "walked", "said", "knew", "made"]
    
    print("PAST -> FUTURE transformation (delta = [+2, 0, 0, 0])")
    print("-" * 50)
    delta = TRANSFORMATION_DELTAS["past_to_future"]
    
    for word in test_words:
        result = vocab.transform(word, delta)
        pos_before = vocab.get_position(word)
        pos_after = pos_before + delta if pos_before is not None else None
        print(f"  {word:12} -> {result:15} (pos: {pos_before} -> {pos_after})")
    
    print()
    print("PAST -> ARCHAIC transformation (delta = [0, +2, 0, 0])")
    print("-" * 50)
    delta = TRANSFORMATION_DELTAS["neutral_to_archaic"]
    
    for word in test_words:
        result = vocab.transform(word, delta)
        print(f"  {word:12} -> {result}")
    
    print()


def demonstrate_noun_transformation():
    """Show how noun domain transformation emerges from geometry."""
    print("=" * 70)
    print("NOUN TRANSFORMATION: Perspective as Geometric Shift")
    print("=" * 70)
    print()
    
    vocab = build_noun_vocabulary()
    print(f"Vocabulary: {vocab.stats()}")
    print()
    
    test_words = ["code", "computer", "data", "programmer", "error"]
    
    print("WARHAMMER 40K perspective (delta = [0, +2, +2, +0.5])")
    print("-" * 50)
    delta = TRANSFORMATION_DELTAS["warhammer40k"]
    
    for word in test_words:
        result = vocab.transform(word, delta)
        print(f"  {word:12} -> {result}")
    
    print()
    print("PIRATE perspective (delta = [0, -1, -1, 0])")
    print("-" * 50)
    delta = TRANSFORMATION_DELTAS["pirate"]
    
    for word in test_words:
        result = vocab.transform(word, delta)
        print(f"  {word:12} -> {result}")
    
    print()


def demonstrate_music_box_principle():
    """Show the complete music box analogy."""
    print("=" * 70)
    print("THE MUSIC BOX PRINCIPLE")
    print("=" * 70)
    print("""
    DRUM (structure):   Words have positions in φ-space
    COMB (decoder):     find_nearest(position) -> word
    MUSIC (output):     Emerges from drum + comb
    
    The comb doesn't contain the music.
    The comb reads the drum and produces sound.
    
    We don't store: "code" -> "holy scripture"
    We store:
      - "code" at position [0, 0, 1, 0]
      - "holy scripture" at position [0, 2, 2, 1]
      - Warhammer40k delta = [0, 2, 2, 0.5]
    
    When we apply the delta:
      [0, 0, 1, 0] + [0, 2, 2, 0.5] = [0, 2, 3, 0.5]
      find_nearest([0, 2, 3, 0.5]) = "holy scripture"
    
    The transformation EMERGES. It's not stored.
    """)
    
    vocab = build_noun_vocabulary()
    
    print("DEMONSTRATION:")
    print("-" * 50)
    
    word = "code"
    pos = vocab.get_position(word)
    delta = TRANSFORMATION_DELTAS["warhammer40k"]
    new_pos = pos + delta
    result = vocab.transform(word, delta)
    
    print(f"  Word:        '{word}'")
    print(f"  Position:    {pos}")
    print(f"  Delta:       {delta} (warhammer40k)")
    print(f"  New pos:     {new_pos}")
    print(f"  Nearest:     '{result}'")
    print()
    print("  The music emerged from the geometry.")
    print("  No lookup table was consulted.")
    print()


def demonstrate_sentence_transformation():
    """Transform a complete sentence using geometric vocabulary."""
    print("=" * 70)
    print("SENTENCE TRANSFORMATION")
    print("=" * 70)
    print()
    
    verb_vocab = build_verb_vocabulary()
    noun_vocab = build_noun_vocabulary()
    
    sentence = "The programmer made code and said it worked"
    print(f"Original: {sentence}")
    print()
    
    # Apply Warhammer 40k perspective
    delta = TRANSFORMATION_DELTAS["warhammer40k"]
    print(f"Applying Warhammer 40k delta: {delta}")
    print()
    
    words = sentence.lower().split()
    transformed = []
    
    for word in words:
        # Try noun vocab first
        result = noun_vocab.transform(word, delta)
        if result and result != word:
            transformed.append(result)
            print(f"  {word:12} -> {result} (noun)")
        else:
            # Try verb vocab
            result = verb_vocab.transform(word, delta)
            if result and result != word:
                transformed.append(result)
                print(f"  {word:12} -> {result} (verb)")
            else:
                transformed.append(word)
                print(f"  {word:12} -> {word} (unchanged)")
    
    print()
    print(f"Transformed: {' '.join(transformed)}")
    print()


if __name__ == "__main__":
    demonstrate_music_box_principle()
    print()
    demonstrate_verb_transformation()
    print()
    demonstrate_noun_transformation()
    print()
    demonstrate_sentence_transformation()
    
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    The current perspective.py has:
        style_rules = {"code": "holy scripture", ...}
    
    This is MUSIC EMBEDDED IN THE COMB.
    
    The geometric approach has:
        vocab["code"] = position [0, 0, 1, 0]
        vocab["holy scripture"] = position [0, 2, 2, 1]
        perspective_delta = [0, 2, 2, 0.5]
    
    This is STRUCTURE (drum) + DECODER (comb) = EMERGENT MUSIC.
    
    The transformation is not stored. It emerges.
    
    The Omnissiah provides. The Machine God protects.
    """)
