#!/usr/bin/env python3
"""
Style Vector Arithmetic Experiment

The hypothesis: Style transfer can be done with pure vector arithmetic.

Key operations:
  1. style_direction = encode(styled_text) - encode(neutral_text)
  2. styled_content = encode(content) + style_direction
  3. Find nearest neighbor in vocabulary to decode

This is analogous to word2vec's famous:
  king - man + woman = queen

We're testing:
  neutral_sentence + warhammer_direction = warhammer_sentence
  neutral_sentence + romance_direction = romance_sentence
"""

import sys
import os
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

PHI = (1 + math.sqrt(5)) / 2


# ============================================================================
# VOCABULARY - Words with their vector positions
# ============================================================================

class Vocabulary:
    """
    A vocabulary where each word has a position in semantic space.
    
    Positions are deterministic (hash-based) so the same word
    always gets the same position.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.word_positions: Dict[str, np.ndarray] = {}
        self.words: List[str] = []
    
    def _get_position(self, word: str) -> np.ndarray:
        """Get or create position for a word."""
        if word in self.word_positions:
            return self.word_positions[word]
        
        # Deterministic position from hash
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim)
        pos = pos / np.linalg.norm(pos)  # Normalize to unit sphere
        np.random.seed(None)
        
        self.word_positions[word] = pos
        self.words.append(word)
        return pos
    
    def add_words(self, words: List[str]):
        """Add words to vocabulary."""
        for word in words:
            self._get_position(word.lower())
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text as average of word positions."""
        words = re.findall(r'\w+', text.lower())
        if not words:
            return np.zeros(self.dim)
        
        positions = [self._get_position(w) for w in words]
        return np.mean(positions, axis=0)
    
    def nearest_words(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find k nearest words to a vector."""
        results = []
        for word, pos in self.word_positions.items():
            sim = np.dot(vector, pos) / (np.linalg.norm(vector) * np.linalg.norm(pos) + 1e-8)
            results.append((word, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    def analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve analogy: a is to b as c is to ?
        
        Formula: ? = b - a + c
        """
        vec_a = self._get_position(a.lower())
        vec_b = self._get_position(b.lower())
        vec_c = self._get_position(c.lower())
        
        # b - a + c
        result_vec = vec_b - vec_a + vec_c
        
        # Find nearest words (excluding a, b, c)
        results = []
        for word, pos in self.word_positions.items():
            if word in [a.lower(), b.lower(), c.lower()]:
                continue
            sim = np.dot(result_vec, pos) / (np.linalg.norm(result_vec) * np.linalg.norm(pos) + 1e-8)
            results.append((word, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]


# ============================================================================
# STYLE DIRECTIONS - Computed from contrastive pairs
# ============================================================================

class StyleDirection:
    """
    A style direction computed from contrastive pairs.
    
    Given pairs of (neutral, styled) text, the style direction is:
      direction = average(encode(styled) - encode(neutral))
    """
    
    def __init__(self, name: str, vocab: Vocabulary):
        self.name = name
        self.vocab = vocab
        self.direction: Optional[np.ndarray] = None
        self.pairs: List[Tuple[str, str]] = []
    
    def add_pair(self, neutral: str, styled: str):
        """Add a contrastive pair."""
        self.pairs.append((neutral, styled))
    
    def compute_direction(self):
        """Compute style direction from pairs."""
        if not self.pairs:
            raise ValueError("No pairs added")
        
        directions = []
        for neutral, styled in self.pairs:
            vec_neutral = self.vocab.encode_text(neutral)
            vec_styled = self.vocab.encode_text(styled)
            diff = vec_styled - vec_neutral
            directions.append(diff)
        
        # Average direction
        self.direction = np.mean(directions, axis=0)
        
        # Normalize
        norm = np.linalg.norm(self.direction)
        if norm > 1e-8:
            self.direction = self.direction / norm
        
        return self.direction
    
    def apply(self, text: str, strength: float = 1.0) -> np.ndarray:
        """
        Apply style to text.
        
        Returns: styled_vector = encode(text) + strength * direction
        """
        if self.direction is None:
            self.compute_direction()
        
        vec = self.vocab.encode_text(text)
        styled_vec = vec + strength * self.direction
        return styled_vec
    
    def measure(self, text: str) -> float:
        """
        Measure how much of this style is in the text.
        
        Returns: projection of text onto style direction
        """
        if self.direction is None:
            self.compute_direction()
        
        vec = self.vocab.encode_text(text)
        return np.dot(vec, self.direction)


# ============================================================================
# STYLE TRANSFER SYSTEM
# ============================================================================

class StyleTransferSystem:
    """
    System for style transfer via vector arithmetic.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab = Vocabulary(dim)
        self.styles: Dict[str, StyleDirection] = {}
        
        # Build vocabulary from all style exemplars
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build vocabulary from predefined word lists."""
        # Common words
        common = [
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him', 'her',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
            'in', 'on', 'at', 'to', 'from', 'by', 'with', 'about',
        ]
        
        # Neutral words
        neutral = [
            'person', 'man', 'woman', 'people', 'thing', 'place', 'time',
            'way', 'day', 'year', 'world', 'life', 'hand', 'part', 'child',
            'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government',
            'company', 'number', 'group', 'problem', 'fact',
            'said', 'went', 'came', 'made', 'found', 'gave', 'told',
            'asked', 'used', 'tried', 'left', 'called', 'kept', 'let',
            'began', 'seemed', 'helped', 'showed', 'heard', 'played',
            'captain', 'ship', 'whale', 'sea', 'ocean', 'hunt', 'chase',
            'obsession', 'revenge', 'crew', 'voyage', 'storm', 'death',
        ]
        
        # Warhammer 40k words
        warhammer = [
            'emperor', 'imperium', 'chaos', 'heretic', 'xenos', 'purge',
            'inquisitor', 'astartes', 'marine', 'bolter', 'crusade', 'zealot',
            'fury', 'wrath', 'doom', 'darkness', 'eternal', 'war', 'battle',
            'skull', 'blood', 'throne', 'holy', 'sacred', 'corrupt', 'taint',
            'daemon', 'warp', 'psyker', 'heresy', 'exterminatus', 'abomination',
            'servo', 'mechanicus', 'tech', 'priest', 'machine', 'spirit',
            'strike', 'cruiser', 'fleet', 'sector', 'planet', 'hive', 'forge',
        ]
        
        # Romance words
        romance = [
            'heart', 'love', 'passion', 'desire', 'longing', 'yearning',
            'eyes', 'gaze', 'touch', 'kiss', 'embrace', 'whisper', 'sigh',
            'beautiful', 'handsome', 'gorgeous', 'stunning', 'breathtaking',
            'tender', 'gentle', 'soft', 'warm', 'sweet', 'precious',
            'forbidden', 'secret', 'hidden', 'stolen', 'impossible',
            'ache', 'burn', 'tremble', 'shiver', 'flutter', 'race',
            'destiny', 'fate', 'soul', 'forever', 'always', 'never',
        ]
        
        # Technical words
        technical = [
            'function', 'method', 'class', 'object', 'variable', 'parameter',
            'return', 'value', 'type', 'string', 'integer', 'boolean', 'array',
            'list', 'dictionary', 'tuple', 'set', 'map', 'hash', 'key',
            'input', 'output', 'process', 'execute', 'run', 'call', 'invoke',
            'error', 'exception', 'handle', 'catch', 'throw', 'raise',
            'file', 'path', 'directory', 'read', 'write', 'open', 'close',
            'database', 'query', 'table', 'column', 'row', 'index', 'schema',
        ]
        
        # Noir words
        noir = [
            'rain', 'night', 'dark', 'shadow', 'smoke', 'cigarette', 'whiskey',
            'dame', 'broad', 'doll', 'trouble', 'danger', 'mystery', 'secret',
            'gun', 'bullet', 'dead', 'murder', 'crime', 'cop', 'detective',
            'city', 'street', 'alley', 'bar', 'office', 'hotel', 'apartment',
            'lie', 'truth', 'betrayal', 'trust', 'money', 'greed', 'revenge',
            'lonely', 'bitter', 'cynical', 'weary', 'broken', 'lost',
        ]
        
        self.vocab.add_words(common + neutral + warhammer + romance + technical + noir)
    
    def define_style(self, name: str, pairs: List[Tuple[str, str]]) -> StyleDirection:
        """
        Define a style from contrastive pairs.
        
        pairs: List of (neutral_text, styled_text)
        """
        style = StyleDirection(name, self.vocab)
        for neutral, styled in pairs:
            style.add_pair(neutral, styled)
        style.compute_direction()
        self.styles[name] = style
        return style
    
    def transfer(self, text: str, target_style: str, strength: float = 1.0) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Transfer text to target style.
        
        Returns: (styled_vector, nearest_words)
        """
        if target_style not in self.styles:
            raise ValueError(f"Unknown style: {target_style}")
        
        style = self.styles[target_style]
        styled_vec = style.apply(text, strength)
        nearest = self.vocab.nearest_words(styled_vec, k=10)
        
        return styled_vec, nearest
    
    def measure_style(self, text: str, style_name: str) -> float:
        """Measure how much of a style is in the text."""
        if style_name not in self.styles:
            raise ValueError(f"Unknown style: {style_name}")
        
        return self.styles[style_name].measure(text)
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify text by style (measure all styles)."""
        results = []
        for name, style in self.styles.items():
            score = style.measure(text)
            results.append((name, score))
        
        results.sort(key=lambda x: -x[1])
        return results


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("  STYLE VECTOR ARITHMETIC EXPERIMENT")
    print("  Testing: styled = neutral + style_direction")
    print("=" * 70)
    
    system = StyleTransferSystem(dim=64)
    
    # Define styles from contrastive pairs
    print("\n[1] Defining styles from contrastive pairs...")
    print("-" * 70)
    
    # Warhammer 40k style
    warhammer_pairs = [
        ("The captain commanded the ship", "The Inquisitor commanded the strike cruiser"),
        ("The soldier fought bravely", "The Astartes purged the heretics with holy fury"),
        ("The enemy attacked", "The xenos abomination assaulted with corrupt fury"),
        ("He was obsessed with revenge", "He burned with zealous wrath for vengeance"),
        ("The city was destroyed", "The hive world suffered Exterminatus"),
        ("The leader spoke", "The Chapter Master proclaimed with righteous fury"),
        ("They went to war", "They embarked on an eternal crusade"),
        ("The machine worked", "The machine spirit awakened"),
    ]
    
    wh_style = system.define_style("Warhammer", warhammer_pairs)
    print(f"  Warhammer style: {len(warhammer_pairs)} pairs")
    
    # Romance style
    romance_pairs = [
        ("He looked at her", "His eyes met hers, and time seemed to stop"),
        ("She felt something", "Her heart raced with forbidden longing"),
        ("They touched", "Their fingers brushed, sending shivers down her spine"),
        ("He was attractive", "He was devastatingly handsome, with eyes like storms"),
        ("She left", "She turned away, hiding the tears that threatened to fall"),
        ("They were together", "In his arms, she finally found home"),
        ("He spoke softly", "He whispered words that made her soul tremble"),
        ("She wanted him", "She ached for him with every fiber of her being"),
    ]
    
    rom_style = system.define_style("Romance", romance_pairs)
    print(f"  Romance style: {len(romance_pairs)} pairs")
    
    # Noir style
    noir_pairs = [
        ("It was raining", "The rain fell like tears from a broken heart"),
        ("She entered", "She walked in like trouble in high heels"),
        ("I waited", "I lit a cigarette and watched the smoke curl"),
        ("The city was dark", "The city was a shadow, hiding secrets in every alley"),
        ("He lied", "He lied like he breathed - naturally"),
        ("She was beautiful", "She was the kind of dame that spelled trouble"),
        ("I drank", "I poured three fingers of bourbon and waited"),
        ("It was over", "In the end, we're all just shadows chasing shadows"),
    ]
    
    noir_style = system.define_style("Noir", noir_pairs)
    print(f"  Noir style: {len(noir_pairs)} pairs")
    
    # Technical style
    tech_pairs = [
        ("It does something", "The function accepts parameters and returns a value"),
        ("Use it like this", "To invoke the method, call object.method(args)"),
        ("It broke", "An exception was raised due to invalid input"),
        ("Save the data", "Write the data to the file using the output stream"),
        ("Find the item", "Query the database with the specified index"),
        ("It's a list", "The data structure is an array of string elements"),
        ("Connect to it", "Establish a connection using the configuration parameters"),
        ("It failed", "The process terminated with error code -1"),
    ]
    
    tech_style = system.define_style("Technical", tech_pairs)
    print(f"  Technical style: {len(tech_pairs)} pairs")
    
    # Test style transfer
    print("\n\n[2] Testing style transfer...")
    print("-" * 70)
    
    test_sentences = [
        "The captain hunted the whale with obsession",
        "She looked at him across the room",
        "The rain fell on the dark streets",
    ]
    
    for sentence in test_sentences:
        print(f"\n  Original: \"{sentence}\"")
        
        for style_name in ["Warhammer", "Romance", "Noir", "Technical"]:
            styled_vec, nearest = system.transfer(sentence, style_name, strength=1.0)
            top_words = [w for w, s in nearest[:5]]
            print(f"    + {style_name:12s} → {', '.join(top_words)}")
    
    # Test style classification
    print("\n\n[3] Testing style classification...")
    print("-" * 70)
    
    test_texts = [
        "The Inquisitor purged the heretics with holy fury",
        "Her heart raced as their eyes met",
        "The function returns a boolean value",
        "The rain fell like tears on the city streets",
        "The captain commanded the ship to pursue the whale",
    ]
    
    for text in test_texts:
        print(f"\n  \"{text[:50]}...\"" if len(text) > 50 else f"\n  \"{text}\"")
        scores = system.classify(text)
        print(f"    Scores: {', '.join(f'{n}={s:+.3f}' for n, s in scores)}")
    
    # Test analogy
    print("\n\n[4] Testing word analogies...")
    print("-" * 70)
    
    analogies = [
        ("captain", "inquisitor", "ship"),      # captain:inquisitor :: ship:?
        ("soldier", "astartes", "fight"),       # soldier:astartes :: fight:?
        ("look", "gaze", "touch"),              # look:gaze :: touch:?
        ("city", "hive", "war"),                # city:hive :: war:?
    ]
    
    for a, b, c in analogies:
        results = system.vocab.analogy(a, b, c, k=5)
        print(f"\n  {a} : {b} :: {c} : ?")
        print(f"    Top answers: {', '.join(f'{w}({s:.2f})' for w, s in results[:3])}")
    
    # The key insight
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHT")
    print("=" * 70)
    print("""
  Style transfer via vector arithmetic:
  
    styled_vector = encode(text) + style_direction
    
  Where style_direction is computed from contrastive pairs:
  
    style_direction = average(encode(styled) - encode(neutral))
  
  The nearest words to styled_vector give us the "flavor" of the style.
  
  This is purely GEOMETRIC:
    - No neural network
    - No training
    - Just vector addition and nearest neighbor lookup
  
  The style IS the direction. Moving along that direction applies the style.
""")


if __name__ == "__main__":
    demo()
