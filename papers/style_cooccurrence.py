#!/usr/bin/env python3
"""
Style Vector Arithmetic with Co-occurrence Embeddings

The previous experiment used random (hash-based) word vectors.
This experiment uses CO-OCCURRENCE to build meaningful vectors.

Key insight: Words that appear together should have similar vectors.

Process:
1. Build co-occurrence matrix from style exemplars
2. Use SVD to get low-dimensional embeddings
3. Style direction = average(styled - neutral) in this space
4. Test style transfer and classification
"""

import sys
import os
import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

PHI = (1 + math.sqrt(5)) / 2


# ============================================================================
# CO-OCCURRENCE VOCABULARY
# ============================================================================

class CooccurrenceVocabulary:
    """
    Build word vectors from co-occurrence statistics.
    
    Words that appear together get similar vectors.
    """
    
    def __init__(self, dim: int = 32, window: int = 3):
        self.dim = dim
        self.window = window
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_vectors: Optional[np.ndarray] = None
        self.cooccurrence: Optional[np.ndarray] = None
        self.word_counts: Counter = Counter()
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def add_text(self, text: str):
        """Add text to build co-occurrence statistics."""
        words = self._tokenize(text)
        
        # Count words
        self.word_counts.update(words)
        
        # Add to vocabulary
        for word in words:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
    
    def build_cooccurrence(self, texts: List[str]):
        """Build co-occurrence matrix from texts."""
        # First pass: build vocabulary
        for text in texts:
            self.add_text(text)
        
        n_words = len(self.word_to_idx)
        self.cooccurrence = np.zeros((n_words, n_words))
        
        # Second pass: count co-occurrences
        for text in texts:
            words = self._tokenize(text)
            for i, word in enumerate(words):
                idx_i = self.word_to_idx[word]
                
                # Look at window around word
                start = max(0, i - self.window)
                end = min(len(words), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j:
                        other_word = words[j]
                        idx_j = self.word_to_idx[other_word]
                        # Weight by distance
                        weight = 1.0 / abs(i - j)
                        self.cooccurrence[idx_i, idx_j] += weight
        
        # Make symmetric
        self.cooccurrence = (self.cooccurrence + self.cooccurrence.T) / 2
        
        # Apply log transform (like GloVe)
        self.cooccurrence = np.log1p(self.cooccurrence)
    
    def build_vectors(self):
        """Build word vectors using SVD on co-occurrence matrix."""
        if self.cooccurrence is None:
            raise ValueError("Must build co-occurrence first")
        
        # SVD
        U, S, Vt = np.linalg.svd(self.cooccurrence, full_matrices=False)
        
        # Take top dimensions
        k = min(self.dim, len(S))
        
        # Word vectors = U * sqrt(S)
        self.word_vectors = U[:, :k] * np.sqrt(S[:k])
        
        # Normalize
        norms = np.linalg.norm(self.word_vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.word_vectors = self.word_vectors / norms
        
        print(f"Built {len(self.word_to_idx)} word vectors of dimension {k}")
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a word."""
        word = word.lower()
        if word not in self.word_to_idx:
            return None
        idx = self.word_to_idx[word]
        return self.word_vectors[idx]
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text as average of word vectors."""
        words = self._tokenize(text)
        vectors = []
        
        for word in words:
            vec = self.get_vector(word)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(self.word_vectors.shape[1])
        
        return np.mean(vectors, axis=0)
    
    def nearest_words(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest words to a vector."""
        # Normalize query
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return []
        vector = vector / norm
        
        # Compute similarities
        sims = self.word_vectors @ vector
        
        # Get top k
        top_indices = np.argsort(sims)[::-1][:k]
        
        results = []
        for idx in top_indices:
            word = self.idx_to_word[idx]
            sim = sims[idx]
            results.append((word, sim))
        
        return results
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute similarity between two words."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return np.dot(vec1, vec2)


# ============================================================================
# STYLE SYSTEM WITH CO-OCCURRENCE
# ============================================================================

class StyleSystem:
    """
    Style transfer system using co-occurrence embeddings.
    """
    
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.vocab = CooccurrenceVocabulary(dim=dim)
        self.styles: Dict[str, np.ndarray] = {}  # style_name -> direction vector
        self.style_pairs: Dict[str, List[Tuple[str, str]]] = {}
    
    def add_style_pairs(self, style_name: str, pairs: List[Tuple[str, str]]):
        """Add contrastive pairs for a style."""
        self.style_pairs[style_name] = pairs
    
    def build(self):
        """Build vocabulary and style directions from all pairs."""
        # Collect all texts
        all_texts = []
        for pairs in self.style_pairs.values():
            for neutral, styled in pairs:
                all_texts.append(neutral)
                all_texts.append(styled)
        
        # Build co-occurrence and vectors
        self.vocab.build_cooccurrence(all_texts)
        self.vocab.build_vectors()
        
        # Compute style directions
        for style_name, pairs in self.style_pairs.items():
            directions = []
            for neutral, styled in pairs:
                vec_neutral = self.vocab.encode_text(neutral)
                vec_styled = self.vocab.encode_text(styled)
                diff = vec_styled - vec_neutral
                directions.append(diff)
            
            # Average direction
            direction = np.mean(directions, axis=0)
            
            # Normalize
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            
            self.styles[style_name] = direction
            print(f"  {style_name}: direction computed from {len(pairs)} pairs")
    
    def apply_style(self, text: str, style_name: str, strength: float = 1.0) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """Apply style to text, return styled vector and nearest words."""
        if style_name not in self.styles:
            raise ValueError(f"Unknown style: {style_name}")
        
        vec = self.vocab.encode_text(text)
        direction = self.styles[style_name]
        
        styled_vec = vec + strength * direction
        nearest = self.vocab.nearest_words(styled_vec, k=10)
        
        return styled_vec, nearest
    
    def measure_style(self, text: str, style_name: str) -> float:
        """Measure how much of a style is in the text."""
        if style_name not in self.styles:
            raise ValueError(f"Unknown style: {style_name}")
        
        vec = self.vocab.encode_text(text)
        direction = self.styles[style_name]
        
        return np.dot(vec, direction)
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify text by measuring all styles."""
        results = []
        for style_name in self.styles:
            score = self.measure_style(text, style_name)
            results.append((style_name, score))
        
        results.sort(key=lambda x: -x[1])
        return results
    
    def transfer_between_styles(self, text: str, from_style: str, to_style: str, strength: float = 1.0) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """Transfer text from one style to another."""
        vec = self.vocab.encode_text(text)
        
        # Remove source style, add target style
        if from_style in self.styles:
            vec = vec - strength * self.styles[from_style]
        if to_style in self.styles:
            vec = vec + strength * self.styles[to_style]
        
        nearest = self.vocab.nearest_words(vec, k=10)
        return vec, nearest


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("  STYLE VECTOR ARITHMETIC WITH CO-OCCURRENCE")
    print("  Words that appear together get similar vectors")
    print("=" * 70)
    
    system = StyleSystem(dim=32)
    
    # Define styles with contrastive pairs
    print("\n[1] Adding style pairs...")
    print("-" * 70)
    
    # Warhammer 40k
    warhammer_pairs = [
        ("The captain commanded the ship", "The Inquisitor commanded the strike cruiser"),
        ("The soldier fought bravely", "The Space Marine purged the heretics with holy fury"),
        ("The enemy attacked", "The xenos abomination assaulted with corrupt fury"),
        ("He was obsessed with revenge", "He burned with zealous wrath for vengeance"),
        ("The city was destroyed", "The hive world suffered Exterminatus"),
        ("The leader spoke to the troops", "The Chapter Master proclaimed with righteous fury"),
        ("They went to war", "They embarked on an eternal crusade against chaos"),
        ("The machine worked", "The machine spirit awakened with sacred purpose"),
        ("The man prayed", "The zealot prayed to the God Emperor"),
        ("The weapon fired", "The bolter roared with holy vengeance"),
    ]
    system.add_style_pairs("Warhammer", warhammer_pairs)
    
    # Romance
    romance_pairs = [
        ("He looked at her", "His smoldering eyes met hers and time seemed to stop"),
        ("She felt something", "Her heart raced with forbidden longing"),
        ("They touched hands", "Their fingers brushed sending shivers down her spine"),
        ("He was attractive", "He was devastatingly handsome with eyes like storms"),
        ("She left the room", "She turned away hiding the tears that threatened to fall"),
        ("They were together", "In his strong arms she finally found home"),
        ("He spoke softly", "He whispered words that made her soul tremble"),
        ("She wanted him", "She ached for him with every fiber of her being"),
        ("They kissed", "Their lips met in a passionate embrace"),
        ("He held her", "He pulled her close his heart beating against hers"),
    ]
    system.add_style_pairs("Romance", romance_pairs)
    
    # Noir
    noir_pairs = [
        ("It was raining", "The rain fell like tears from a broken heart"),
        ("She entered the office", "She walked in like trouble in high heels"),
        ("I waited", "I lit a cigarette and watched the smoke curl toward the ceiling"),
        ("The city was dark", "The city was a shadow hiding secrets in every alley"),
        ("He lied to me", "He lied like he breathed naturally and without thinking"),
        ("She was beautiful", "She was the kind of dame that spelled trouble"),
        ("I had a drink", "I poured three fingers of bourbon and waited for the other shoe"),
        ("It was over", "In the end we are all just shadows chasing shadows"),
        ("He was dead", "He was as dead as my faith in humanity"),
        ("The night was long", "The night stretched out like a bad dream"),
    ]
    system.add_style_pairs("Noir", noir_pairs)
    
    # Technical
    tech_pairs = [
        ("It does something", "The function accepts parameters and returns a value"),
        ("Use it like this", "Invoke the method by calling object dot method with args"),
        ("It broke", "An exception was raised due to invalid input parameters"),
        ("Save the data", "Write the data to file using the output stream"),
        ("Find the item", "Query the database using the specified index key"),
        ("It is a list", "The data structure is an array of string elements"),
        ("Connect to it", "Establish a connection using the configuration object"),
        ("It failed", "The process terminated with error code negative one"),
        ("Check if true", "Evaluate the boolean condition and return the result"),
        ("Loop through items", "Iterate over the collection using a for loop"),
    ]
    system.add_style_pairs("Technical", tech_pairs)
    
    # Build the system
    print("\n[2] Building co-occurrence vectors...")
    print("-" * 70)
    system.build()
    
    # Test word similarities
    print("\n\n[3] Testing word similarities (co-occurrence based)...")
    print("-" * 70)
    
    test_pairs = [
        ("inquisitor", "emperor"),
        ("heart", "love"),
        ("rain", "tears"),
        ("function", "method"),
        ("ship", "cruiser"),
        ("fury", "wrath"),
    ]
    
    for w1, w2 in test_pairs:
        sim = system.vocab.similarity(w1, w2)
        print(f"  {w1} ~ {w2}: {sim:.3f}")
    
    # Test style classification
    print("\n\n[4] Testing style classification...")
    print("-" * 70)
    
    test_texts = [
        "The Inquisitor purged the heretics with holy fury",
        "Her heart raced as their eyes met across the room",
        "The function returns a boolean value",
        "The rain fell like tears on the dark city streets",
        "The captain commanded the ship to pursue the whale",
    ]
    
    for text in test_texts:
        print(f"\n  \"{text[:50]}...\"" if len(text) > 50 else f"\n  \"{text}\"")
        scores = system.classify(text)
        print(f"    {', '.join(f'{n}={s:+.3f}' for n, s in scores)}")
    
    # Test style transfer
    print("\n\n[5] Testing style transfer (nearest words after adding style)...")
    print("-" * 70)
    
    source = "The captain hunted the whale with obsession"
    print(f"\n  Source: \"{source}\"")
    
    for style_name in ["Warhammer", "Romance", "Noir", "Technical"]:
        _, nearest = system.apply_style(source, style_name, strength=0.5)
        words = [w for w, s in nearest[:7]]
        print(f"    + {style_name:12s} → {', '.join(words)}")
    
    # Test style-to-style transfer
    print("\n\n[6] Testing style-to-style transfer...")
    print("-" * 70)
    
    wh_text = "The Inquisitor purged the heretics with holy fury"
    print(f"\n  Warhammer text: \"{wh_text}\"")
    
    _, nearest = system.transfer_between_styles(wh_text, "Warhammer", "Romance", strength=0.5)
    words = [w for w, s in nearest[:7]]
    print(f"    → Romance: {', '.join(words)}")
    
    _, nearest = system.transfer_between_styles(wh_text, "Warhammer", "Noir", strength=0.5)
    words = [w for w, s in nearest[:7]]
    print(f"    → Noir: {', '.join(words)}")
    
    # Key insight
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHT")
    print("=" * 70)
    print("""
  Co-occurrence embeddings capture SEMANTIC relationships:
    - Words that appear together have similar vectors
    - Style direction = average(styled - neutral)
    - Style transfer = content + style_direction
  
  This is still purely GEOMETRIC:
    - Co-occurrence matrix from text
    - SVD for dimensionality reduction
    - Vector arithmetic for style transfer
  
  The style IS the direction from neutral to styled.
  Moving along that direction applies the style.
""")


if __name__ == "__main__":
    demo()
