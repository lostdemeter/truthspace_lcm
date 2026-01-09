"""
Geometric Vocabulary - The Drum of the Music Box (Design 112)

Words have positions in semantic space. Transformations are delta vectors.
Output emerges from find_nearest(position + delta).

No word->word mappings. The music emerges from the geometry.

Dimensions:
- tense: -1 (past) / 0 (present) / +1 (future)
- formality: -1 (casual) / 0 (neutral) / +1 (formal) / +2 (archaic)
- domain: -1 (mundane) / 0 (neutral) / +1 (technical) / +2 (sacred)
- intensity: -1 (weak) / 0 (normal) / +1 (strong)

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from pathlib import Path
import json


PHI = (1 + np.sqrt(5)) / 2

# Dimension indices
TENSE = 0
FORMALITY = 1
DOMAIN = 2
INTENSITY = 3

DIMENSION_NAMES = ['tense', 'formality', 'domain', 'intensity']


@dataclass
class WordPosition:
    """A word's position in semantic space."""
    word: str
    position: np.ndarray
    concept: Optional[str] = None
    
    def distance_to(self, other: np.ndarray) -> float:
        """Euclidean distance to another position."""
        return np.linalg.norm(self.position - other)


class GeometricVocabulary:
    """
    Vocabulary organized by geometric position (The Drum).
    
    Each word has a position in multi-dimensional semantic space.
    The vocabulary is the DRUM - it contains the structure.
    The find_nearest method is the COMB - it reads the structure.
    The output is the MUSIC - it emerges from drum + comb.
    """
    
    def __init__(self, dims: int = 4):
        self.dims = dims
        self._words: Dict[str, WordPosition] = {}
        self._by_concept: Dict[str, List[str]] = defaultdict(list)
    
    def add_word(self, word: str, position: np.ndarray, concept: Optional[str] = None):
        """Add a word at a specific position."""
        if len(position) != self.dims:
            position = np.zeros(self.dims)
            position[:len(position)] = position[:self.dims]
        
        word_lower = word.lower()
        self._words[word_lower] = WordPosition(word=word, position=position, concept=concept)
        if concept:
            self._by_concept[concept].append(word_lower)
    
    def get_position(self, word: str) -> Optional[np.ndarray]:
        """Get a word's position."""
        wp = self._words.get(word.lower())
        return wp.position.copy() if wp else None
    
    def get_concept(self, word: str) -> Optional[str]:
        """Get a word's concept group."""
        wp = self._words.get(word.lower())
        return wp.concept if wp else None
    
    def find_nearest(self, position: np.ndarray, exclude: Optional[Set[str]] = None) -> Optional[str]:
        """
        Find the nearest word to a position (The Comb).
        
        This reads the structure and produces output.
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
                best_word = wp.word  # Return original case
        
        return best_word
    
    def find_nearest_in_concept(self, position: np.ndarray, concept: str, 
                                 exclude: Optional[Set[str]] = None) -> Optional[str]:
        """Find nearest word within a concept group."""
        exclude = exclude or set()
        concept_words = self._by_concept.get(concept, [])
        
        if not concept_words:
            return self.find_nearest(position, exclude)
        
        best_word = None
        best_distance = float('inf')
        
        for word in concept_words:
            if word in exclude:
                continue
            wp = self._words.get(word)
            if wp:
                dist = wp.distance_to(position)
                if dist < best_distance:
                    best_distance = dist
                    best_word = wp.word
        
        return best_word
    
    def transform(self, word: str, delta: np.ndarray) -> Optional[str]:
        """
        Transform a word by applying a delta vector (The Music).
        
        1. Get word's current position (read the drum)
        2. Apply delta (rotation of the drum)
        3. Find nearest word at new position (comb produces sound)
        
        No lookup table. The output emerges from geometry.
        """
        current_pos = self.get_position(word)
        if current_pos is None:
            return None
        
        new_pos = current_pos + delta
        
        # Stay within concept if word has one
        concept = self.get_concept(word)
        if concept:
            result = self.find_nearest_in_concept(new_pos, concept, exclude={word.lower()})
            if result and result.lower() != word.lower():
                return result
        
        # Fallback to global search
        return self.find_nearest(new_pos, exclude={word.lower()})
    
    def has_word(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word.lower() in self._words
    
    def stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'total_words': len(self._words),
            'concepts': len(self._by_concept),
            'dimensions': self.dims,
        }
    
    def save(self, path: Path):
        """Save vocabulary to JSON."""
        data = {
            'dims': self.dims,
            'words': [
                {
                    'word': wp.word,
                    'position': wp.position.tolist(),
                    'concept': wp.concept
                }
                for wp in self._words.values()
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'GeometricVocabulary':
        """Load vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        vocab = cls(dims=data.get('dims', 4))
        for item in data.get('words', []):
            vocab.add_word(
                item['word'],
                np.array(item['position']),
                item.get('concept')
            )
        return vocab


# =============================================================================
# BOOTSTRAP VOCABULARY
# =============================================================================

def build_default_vocabulary() -> GeometricVocabulary:
    """
    Build the default vocabulary with words positioned by semantic dimensions.
    
    Position: [tense, formality, domain, intensity]
    """
    vocab = GeometricVocabulary(dims=4)
    
    # -----------------------------------------------------------------
    # VERBS - organized by tense and formality
    # -----------------------------------------------------------------
    
    # GO concept
    vocab.add_word("went", np.array([-1, 0, 0, 0]), "GO")
    vocab.add_word("go", np.array([0, 0, 0, 0]), "GO")
    vocab.add_word("goes", np.array([0, 0, 0, 0]), "GO")
    vocab.add_word("will go", np.array([1, 0, 0, 0]), "GO")
    vocab.add_word("shall go", np.array([1, 1, 0, 0]), "GO")
    vocab.add_word("did proceed", np.array([-1, 2, 0, 0]), "GO")
    vocab.add_word("doth proceed", np.array([0, 2, 0, 0]), "GO")
    vocab.add_word("shall proceed", np.array([1, 2, 0, 0]), "GO")
    
    # SIT concept
    vocab.add_word("sat", np.array([-1, 0, 0, 0]), "SIT")
    vocab.add_word("sit", np.array([0, 0, 0, 0]), "SIT")
    vocab.add_word("sits", np.array([0, 0, 0, 0]), "SIT")
    vocab.add_word("will sit", np.array([1, 0, 0, 0]), "SIT")
    vocab.add_word("shall sit", np.array([1, 1, 0, 0]), "SIT")
    vocab.add_word("was seated", np.array([-1, 1, 0, 0]), "SIT")
    vocab.add_word("shall be seated", np.array([1, 2, 0, 0]), "SIT")
    
    # WALK concept
    vocab.add_word("walked", np.array([-1, 0, 0, 0]), "WALK")
    vocab.add_word("walk", np.array([0, 0, 0, 0]), "WALK")
    vocab.add_word("walks", np.array([0, 0, 0, 0]), "WALK")
    vocab.add_word("will walk", np.array([1, 0, 0, 0]), "WALK")
    vocab.add_word("strode", np.array([-1, 1, 0, 1]), "WALK")
    vocab.add_word("shall stride", np.array([1, 1, 0, 1]), "WALK")
    
    # SAY concept
    vocab.add_word("said", np.array([-1, 0, 0, 0]), "SAY")
    vocab.add_word("say", np.array([0, 0, 0, 0]), "SAY")
    vocab.add_word("says", np.array([0, 0, 0, 0]), "SAY")
    vocab.add_word("will say", np.array([1, 0, 0, 0]), "SAY")
    vocab.add_word("spoke", np.array([-1, 1, 0, 0]), "SAY")
    vocab.add_word("declared", np.array([-1, 1, 0, 1]), "SAY")
    vocab.add_word("shall declare", np.array([1, 2, 0, 1]), "SAY")
    vocab.add_word("intoned", np.array([-1, 2, 2, 0]), "SAY")
    vocab.add_word("shall intone", np.array([1, 2, 2, 0]), "SAY")
    
    # KNOW concept
    vocab.add_word("knew", np.array([-1, 0, 0, 0]), "KNOW")
    vocab.add_word("know", np.array([0, 0, 0, 0]), "KNOW")
    vocab.add_word("knows", np.array([0, 0, 0, 0]), "KNOW")
    vocab.add_word("will know", np.array([1, 0, 0, 0]), "KNOW")
    vocab.add_word("understood", np.array([-1, 1, 0, 0]), "KNOW")
    vocab.add_word("comprehended", np.array([-1, 1, 1, 0]), "KNOW")
    vocab.add_word("divined", np.array([-1, 2, 2, 0]), "KNOW")
    
    # MAKE concept
    vocab.add_word("made", np.array([-1, 0, 0, 0]), "MAKE")
    vocab.add_word("make", np.array([0, 0, 0, 0]), "MAKE")
    vocab.add_word("makes", np.array([0, 0, 0, 0]), "MAKE")
    vocab.add_word("will make", np.array([1, 0, 0, 0]), "MAKE")
    vocab.add_word("crafted", np.array([-1, 1, 0, 0]), "MAKE")
    vocab.add_word("forged", np.array([-1, 1, 0, 1]), "MAKE")
    vocab.add_word("wrought", np.array([-1, 2, 0, 1]), "MAKE")
    vocab.add_word("shall forge", np.array([1, 1, 0, 1]), "MAKE")
    
    # WORK concept
    vocab.add_word("worked", np.array([-1, 0, 0, 0]), "WORK")
    vocab.add_word("work", np.array([0, 0, 0, 0]), "WORK")
    vocab.add_word("works", np.array([0, 0, 0, 0]), "WORK")
    vocab.add_word("will work", np.array([1, 0, 0, 0]), "WORK")
    vocab.add_word("functioned", np.array([-1, 1, 1, 0]), "WORK")
    vocab.add_word("operated", np.array([-1, 1, 1, 0]), "WORK")
    
    # -----------------------------------------------------------------
    # NOUNS - organized by domain and formality
    # -----------------------------------------------------------------
    
    # CODE concept
    vocab.add_word("code", np.array([0, 0, 1, 0]), "CODE")
    vocab.add_word("program", np.array([0, 0, 1, 0]), "CODE")
    vocab.add_word("script", np.array([0, 0, 1, 0]), "CODE")
    vocab.add_word("scripture", np.array([0, 1, 2, 0]), "CODE")
    vocab.add_word("holy scripture", np.array([0, 2, 2, 1]), "CODE")
    vocab.add_word("sacred text", np.array([0, 2, 2, 0]), "CODE")
    vocab.add_word("treasure map", np.array([0, -1, -1, 0]), "CODE")
    
    # COMPUTER concept
    vocab.add_word("computer", np.array([0, 0, 1, 0]), "COMPUTER")
    vocab.add_word("machine", np.array([0, 0, 1, 0]), "COMPUTER")
    vocab.add_word("cogitator", np.array([0, 2, 2, 0]), "COMPUTER")
    vocab.add_word("thinking engine", np.array([0, 1, 1, 0]), "COMPUTER")
    vocab.add_word("magic box", np.array([0, -1, -1, 0]), "COMPUTER")
    
    # DATA concept
    vocab.add_word("data", np.array([0, 0, 1, 0]), "DATA")
    vocab.add_word("information", np.array([0, 1, 1, 0]), "DATA")
    vocab.add_word("sacred data-hymns", np.array([0, 2, 2, 1]), "DATA")
    vocab.add_word("booty", np.array([0, -1, -1, 0]), "DATA")
    vocab.add_word("lore", np.array([0, 1, 2, 0]), "DATA")
    
    # PROGRAMMER concept
    vocab.add_word("programmer", np.array([0, 0, 1, 0]), "PROGRAMMER")
    vocab.add_word("developer", np.array([0, 0, 1, 0]), "PROGRAMMER")
    vocab.add_word("coder", np.array([0, -1, 1, 0]), "PROGRAMMER")
    vocab.add_word("tech-adept", np.array([0, 1, 2, 0]), "PROGRAMMER")
    vocab.add_word("code-priest", np.array([0, 2, 2, 1]), "PROGRAMMER")
    
    # ERROR concept
    vocab.add_word("error", np.array([0, 0, 1, 0]), "ERROR")
    vocab.add_word("bug", np.array([0, -1, 1, 0]), "ERROR")
    vocab.add_word("mistake", np.array([0, 0, 0, 0]), "ERROR")
    vocab.add_word("machine spirit's displeasure", np.array([0, 2, 2, 1]), "ERROR")
    vocab.add_word("corruption", np.array([0, 1, 2, 0]), "ERROR")
    vocab.add_word("scurvy mistake", np.array([0, -1, -1, 0]), "ERROR")
    
    # FUNCTION concept
    vocab.add_word("function", np.array([0, 0, 1, 0]), "FUNCTION")
    vocab.add_word("method", np.array([0, 0, 1, 0]), "FUNCTION")
    vocab.add_word("sacred ritual", np.array([0, 2, 2, 0]), "FUNCTION")
    vocab.add_word("rite", np.array([0, 2, 2, 0]), "FUNCTION")
    
    # ALGORITHM concept
    vocab.add_word("algorithm", np.array([0, 0, 1, 0]), "ALGORITHM")
    vocab.add_word("process", np.array([0, 0, 1, 0]), "ALGORITHM")
    vocab.add_word("divine computation", np.array([0, 2, 2, 1]), "ALGORITHM")
    vocab.add_word("sacred process", np.array([0, 2, 2, 0]), "ALGORITHM")
    
    # SYSTEM concept
    vocab.add_word("system", np.array([0, 0, 1, 0]), "SYSTEM")
    vocab.add_word("holy system", np.array([0, 2, 2, 0]), "SYSTEM")
    vocab.add_word("framework", np.array([0, 0, 1, 0]), "SYSTEM")
    vocab.add_word("blessed framework", np.array([0, 2, 2, 0]), "SYSTEM")
    
    # TOOL concept
    vocab.add_word("tool", np.array([0, 0, 1, 0]), "TOOL")
    vocab.add_word("sacred instrument", np.array([0, 2, 2, 0]), "TOOL")
    vocab.add_word("utility", np.array([0, 1, 1, 0]), "TOOL")
    
    # PROBLEM concept
    vocab.add_word("problem", np.array([0, 0, 0, 0]), "PROBLEM")
    vocab.add_word("issue", np.array([0, 0, 0, 0]), "PROBLEM")
    vocab.add_word("heretical obstruction", np.array([0, 2, 2, 1]), "PROBLEM")
    vocab.add_word("rough waters", np.array([0, -1, -1, 0]), "PROBLEM")
    vocab.add_word("challenge", np.array([0, 1, 0, 0]), "PROBLEM")
    vocab.add_word("trial", np.array([0, 2, 2, 0]), "PROBLEM")
    
    # SOLUTION concept
    vocab.add_word("solution", np.array([0, 0, 0, 0]), "SOLUTION")
    vocab.add_word("answer", np.array([0, 0, 0, 0]), "SOLUTION")
    vocab.add_word("blessed resolution", np.array([0, 2, 2, 0]), "SOLUTION")
    vocab.add_word("safe harbor", np.array([0, -1, -1, 0]), "SOLUTION")
    
    # HELP concept
    vocab.add_word("help", np.array([0, 0, 0, 0]), "HELP")
    vocab.add_word("assist", np.array([0, 1, 0, 0]), "HELP")
    vocab.add_word("serve", np.array([0, 1, 0, 0]), "HELP")
    vocab.add_word("lend a hand", np.array([0, -1, -1, 0]), "HELP")
    
    # GOOD concept
    vocab.add_word("good", np.array([0, 0, 0, 0]), "GOOD")
    vocab.add_word("fine", np.array([0, 0, 0, 0]), "GOOD")
    vocab.add_word("blessed", np.array([0, 2, 2, 0]), "GOOD")
    vocab.add_word("mighty fine", np.array([0, -1, -1, 1]), "GOOD")
    vocab.add_word("excellent", np.array([0, 1, 0, 1]), "GOOD")
    
    # GREAT concept
    vocab.add_word("great", np.array([0, 0, 0, 1]), "GREAT")
    vocab.add_word("glorious", np.array([0, 2, 2, 1]), "GREAT")
    vocab.add_word("wondrous", np.array([0, 2, 0, 1]), "GREAT")
    
    # UNDERSTAND concept
    vocab.add_word("understand", np.array([0, 0, 0, 0]), "UNDERSTAND")
    vocab.add_word("comprehend", np.array([0, 1, 0, 0]), "UNDERSTAND")
    vocab.add_word("savvy", np.array([0, -1, -1, 0]), "UNDERSTAND")
    vocab.add_word("comprehend the mysteries", np.array([0, 2, 2, 0]), "UNDERSTAND")
    
    # LEARN concept
    vocab.add_word("learn", np.array([0, 0, 0, 0]), "LEARN")
    vocab.add_word("study", np.array([0, 1, 0, 0]), "LEARN")
    vocab.add_word("receive the sacred knowledge", np.array([0, 2, 2, 0]), "LEARN")
    
    return vocab


# =============================================================================
# PERSPECTIVE DELTAS (The Rotation of the Drum)
# =============================================================================

# These are movements through the space, not word->word mappings
PERSPECTIVE_DELTAS = {
    # Default - no change
    "default": np.array([0, 0, 0, 0]),
    
    # Warhammer 40k - archaic + sacred + intensity
    "warhammer40k": np.array([0, 2, 2, 0.5]),
    "wh40k": np.array([0, 2, 2, 0.5]),
    "grimdark": np.array([0, 2, 2, 0.5]),
    
    # Pirate - casual + mundane
    "pirate": np.array([0, -1, -1, 0]),
    
    # Shakespeare - archaic formality
    "shakespeare": np.array([0, 2, 0, 0]),
    "bard": np.array([0, 2, 0, 0]),
}

# Tense transformation deltas
TENSE_DELTAS = {
    "past_to_future": np.array([2, 0, 0, 0]),
    "past_to_present": np.array([1, 0, 0, 0]),
    "present_to_future": np.array([1, 0, 0, 0]),
    "future_to_past": np.array([-2, 0, 0, 0]),
    "present_to_past": np.array([-1, 0, 0, 0]),
    "future_to_present": np.array([-1, 0, 0, 0]),
}

# Formality transformation deltas
FORMALITY_DELTAS = {
    "casual_to_formal": np.array([0, 2, 0, 0]),
    "neutral_to_archaic": np.array([0, 2, 0, 0]),
    "formal_to_casual": np.array([0, -2, 0, 0]),
    "neutral_to_formal": np.array([0, 1, 0, 0]),
}


# Global vocabulary instance (lazy loaded)
_default_vocabulary: Optional[GeometricVocabulary] = None


def get_default_vocabulary() -> GeometricVocabulary:
    """Get the default vocabulary (lazy loaded singleton)."""
    global _default_vocabulary
    if _default_vocabulary is None:
        _default_vocabulary = build_default_vocabulary()
    return _default_vocabulary


def get_perspective_delta(name: str) -> np.ndarray:
    """Get a perspective delta by name."""
    return PERSPECTIVE_DELTAS.get(name.lower(), PERSPECTIVE_DELTAS["default"]).copy()
