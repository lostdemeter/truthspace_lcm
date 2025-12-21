#!/usr/bin/env python3
"""
Style Centroid Approach

Instead of computing style as direction (styled - neutral),
compute style as CENTROID of styled exemplars.

Then classification = distance to centroid.
Style transfer = move toward centroid.

This is simpler and may work better with limited data.
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
# SIMPLE EMBEDDING VOCABULARY
# ============================================================================

class SimpleVocabulary:
    """
    Simple vocabulary with hash-based positions that are 
    refined by co-occurrence within style exemplars.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.word_positions: Dict[str, np.ndarray] = {}
        self.word_counts: Counter = Counter()
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _base_position(self, word: str) -> np.ndarray:
        """Get base position for a word (deterministic from hash)."""
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim)
        np.random.seed(None)
        return pos
    
    def add_text(self, text: str, style_bias: Optional[np.ndarray] = None, bias_strength: float = 0.3):
        """
        Add text to vocabulary.
        
        If style_bias is provided, words get pulled toward that direction.
        """
        words = self._tokenize(text)
        self.word_counts.update(words)
        
        for word in words:
            if word not in self.word_positions:
                self.word_positions[word] = self._base_position(word)
            
            # Apply style bias if provided
            if style_bias is not None:
                self.word_positions[word] = (
                    (1 - bias_strength) * self.word_positions[word] + 
                    bias_strength * style_bias
                )
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text as weighted average of word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        # Weight by inverse frequency (rare words matter more)
        positions = []
        weights = []
        
        for word in words:
            if word in self.word_positions:
                positions.append(self.word_positions[word])
                # IDF-like weighting
                count = self.word_counts.get(word, 1)
                weight = 1.0 / math.log(1 + count)
                weights.append(weight)
        
        if not positions:
            return np.zeros(self.dim)
        
        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(positions, axis=0, weights=weights)
    
    def nearest_words(self, vector: np.ndarray, k: int = 5, exclude: Set[str] = None) -> List[Tuple[str, float]]:
        """Find nearest words to a vector."""
        if exclude is None:
            exclude = set()
        
        results = []
        vec_norm = np.linalg.norm(vector)
        if vec_norm < 1e-8:
            return []
        
        for word, pos in self.word_positions.items():
            if word in exclude:
                continue
            pos_norm = np.linalg.norm(pos)
            if pos_norm < 1e-8:
                continue
            sim = np.dot(vector, pos) / (vec_norm * pos_norm)
            results.append((word, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]


# ============================================================================
# STYLE CENTROID SYSTEM
# ============================================================================

class StyleCentroidSystem:
    """
    Style system based on centroids.
    
    Each style is defined by a centroid (average position of exemplars).
    Classification = which centroid is closest.
    Transfer = move toward target centroid.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab = SimpleVocabulary(dim)
        self.style_centroids: Dict[str, np.ndarray] = {}
        self.style_exemplars: Dict[str, List[str]] = {}
    
    def add_style(self, name: str, exemplars: List[str]):
        """Add a style defined by exemplars."""
        self.style_exemplars[name] = exemplars
        
        # First, compute rough centroid
        rough_positions = []
        for text in exemplars:
            words = re.findall(r'\w+', text.lower())
            for word in words:
                if word not in self.vocab.word_positions:
                    self.vocab.word_positions[word] = self.vocab._base_position(word)
            
            pos = self.vocab.encode_text(text)
            rough_positions.append(pos)
        
        rough_centroid = np.mean(rough_positions, axis=0)
        
        # Now add texts with style bias
        for text in exemplars:
            self.vocab.add_text(text, style_bias=rough_centroid, bias_strength=0.2)
        
        # Recompute centroid with biased positions
        positions = [self.vocab.encode_text(text) for text in exemplars]
        self.style_centroids[name] = np.mean(positions, axis=0)
        
        print(f"  {name}: centroid from {len(exemplars)} exemplars")
    
    def distance_to_style(self, text: str, style_name: str) -> float:
        """Euclidean distance from text to style centroid."""
        if style_name not in self.style_centroids:
            raise ValueError(f"Unknown style: {style_name}")
        
        vec = self.vocab.encode_text(text)
        centroid = self.style_centroids[style_name]
        
        return np.linalg.norm(vec - centroid)
    
    def similarity_to_style(self, text: str, style_name: str) -> float:
        """Cosine similarity to style centroid."""
        if style_name not in self.style_centroids:
            raise ValueError(f"Unknown style: {style_name}")
        
        vec = self.vocab.encode_text(text)
        centroid = self.style_centroids[style_name]
        
        vec_norm = np.linalg.norm(vec)
        cent_norm = np.linalg.norm(centroid)
        
        if vec_norm < 1e-8 or cent_norm < 1e-8:
            return 0.0
        
        return np.dot(vec, centroid) / (vec_norm * cent_norm)
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify text by similarity to each style centroid."""
        results = []
        for style_name in self.style_centroids:
            sim = self.similarity_to_style(text, style_name)
            results.append((style_name, sim))
        
        results.sort(key=lambda x: -x[1])
        return results
    
    def transfer_to_style(self, text: str, target_style: str, strength: float = 0.5) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Transfer text toward target style.
        
        new_vec = (1 - strength) * text_vec + strength * centroid
        """
        if target_style not in self.style_centroids:
            raise ValueError(f"Unknown style: {target_style}")
        
        vec = self.vocab.encode_text(text)
        centroid = self.style_centroids[target_style]
        
        # Interpolate toward centroid
        new_vec = (1 - strength) * vec + strength * centroid
        
        # Find nearest words
        text_words = set(re.findall(r'\w+', text.lower()))
        nearest = self.vocab.nearest_words(new_vec, k=10, exclude=text_words)
        
        return new_vec, nearest
    
    def style_direction(self, from_style: str, to_style: str) -> np.ndarray:
        """Get direction from one style to another."""
        if from_style not in self.style_centroids or to_style not in self.style_centroids:
            raise ValueError("Unknown style")
        
        return self.style_centroids[to_style] - self.style_centroids[from_style]


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("  STYLE CENTROID APPROACH")
    print("  Styles as centroids in semantic space")
    print("=" * 70)
    
    system = StyleCentroidSystem(dim=64)
    
    # Define styles with exemplars
    print("\n[1] Defining styles from exemplars...")
    print("-" * 70)
    
    # Warhammer 40k exemplars
    warhammer = [
        "The Inquisitor commanded the strike cruiser with righteous fury",
        "The Space Marine purged the heretics in the name of the Emperor",
        "The xenos abomination was destroyed by holy bolter fire",
        "He burned with zealous wrath seeking vengeance for the fallen",
        "The hive world suffered Exterminatus for its corruption",
        "The Chapter Master proclaimed the eternal crusade against chaos",
        "The machine spirit awakened with sacred purpose",
        "Blood for the Blood God skulls for the Skull Throne",
        "In the grim darkness of the far future there is only war",
        "The daemon was banished back to the warp by the psyker",
        "The Imperial Guard held the line against the ork horde",
        "The tech priest communed with the machine spirit",
        "Heresy grows from idleness purge the unclean",
        "The Astartes chapter defended the sector with honor",
        "Corruption spreads like a plague through the imperium",
    ]
    system.add_style("Warhammer", warhammer)
    
    # Romance exemplars
    romance = [
        "His smoldering eyes met hers and time seemed to stop",
        "Her heart raced with forbidden longing as he approached",
        "Their fingers brushed sending shivers down her spine",
        "He was devastatingly handsome with eyes like storms",
        "She turned away hiding the tears that threatened to fall",
        "In his strong arms she finally found home",
        "He whispered words that made her soul tremble with desire",
        "She ached for him with every fiber of her being",
        "Their lips met in a passionate embrace under the moonlight",
        "He pulled her close his heart beating against hers",
        "The tension between them was electric and undeniable",
        "She knew she should not love him but her heart had other plans",
        "His touch awakened feelings she had buried long ago",
        "Against all odds against all reason she loved him still",
        "Their love was forbidden but burned brighter for it",
    ]
    system.add_style("Romance", romance)
    
    # Noir exemplars
    noir = [
        "The rain fell like tears from a broken heart on the city",
        "She walked in like trouble in high heels and red lipstick",
        "I lit a cigarette and watched the smoke curl toward the ceiling",
        "The city was a shadow hiding secrets in every dark alley",
        "He lied like he breathed naturally and without thinking",
        "She was the kind of dame that spelled trouble in neon",
        "I poured three fingers of bourbon and waited for the shoe to drop",
        "In the end we are all just shadows chasing shadows",
        "He was as dead as my faith in humanity lying there",
        "The night stretched out like a bad dream that would not end",
        "Everyone in this city has an angle and most cut deep",
        "The truth was buried deeper than the bodies in the river",
        "I had seen a lot of dead men but this one had a story",
        "The dame had legs that went up and a story that went down",
        "In this city the rain washes away sins but not memories",
    ]
    system.add_style("Noir", noir)
    
    # Technical exemplars
    technical = [
        "The function accepts two parameters and returns a boolean value",
        "Invoke the method by calling object dot method with arguments",
        "An exception was raised due to invalid input parameters",
        "Write the data to the file using the output stream object",
        "Query the database using the specified index key value",
        "The data structure is an array of string elements",
        "Establish a connection using the configuration object",
        "The process terminated with error code negative one",
        "Evaluate the boolean condition and return the result",
        "Iterate over the collection using a for loop construct",
        "The class implements the interface and extends the base",
        "Initialize the variable with the default value null",
        "The algorithm has complexity of order n log n",
        "Parse the input string and validate the format",
        "The module exports the following public functions",
    ]
    system.add_style("Technical", technical)
    
    # Neutral exemplars (for comparison)
    neutral = [
        "The man walked down the street",
        "She looked at him across the room",
        "The ship sailed across the ocean",
        "He spoke to the group of people",
        "The building was tall and old",
        "They went to the place together",
        "The thing was on the table",
        "She said something to him",
        "The day was long and tiring",
        "He found what he was looking for",
    ]
    system.add_style("Neutral", neutral)
    
    # Test classification
    print("\n\n[2] Testing style classification...")
    print("-" * 70)
    
    test_texts = [
        "The Inquisitor purged the heretics with holy fury",
        "Her heart raced as their eyes met across the ballroom",
        "The function returns a boolean value based on input",
        "The rain fell like tears on the dark city streets",
        "The captain commanded the ship to pursue the whale",
        "He burned with zealous wrath for the Emperor",
        "She ached for him with every fiber of her being",
        "I lit a cigarette and watched the smoke curl upward",
        "Initialize the array with default values",
    ]
    
    for text in test_texts:
        print(f"\n  \"{text[:55]}...\"" if len(text) > 55 else f"\n  \"{text}\"")
        scores = system.classify(text)
        # Show top 3
        top3 = scores[:3]
        print(f"    {' > '.join(f'{n}({s:.3f})' for n, s in top3)}")
    
    # Test style transfer
    print("\n\n[3] Testing style transfer (words that appear after moving toward style)...")
    print("-" * 70)
    
    source = "The captain hunted the whale with obsession"
    print(f"\n  Source: \"{source}\"")
    
    for style_name in ["Warhammer", "Romance", "Noir", "Technical"]:
        _, nearest = system.transfer_to_style(source, style_name, strength=0.7)
        words = [w for w, s in nearest[:7]]
        print(f"    → {style_name:12s}: {', '.join(words)}")
    
    # Test style directions
    print("\n\n[4] Style directions (what changes between styles)...")
    print("-" * 70)
    
    pairs = [
        ("Neutral", "Warhammer"),
        ("Neutral", "Romance"),
        ("Neutral", "Noir"),
        ("Romance", "Warhammer"),
    ]
    
    for from_style, to_style in pairs:
        direction = system.style_direction(from_style, to_style)
        # Find words most aligned with this direction
        aligned = system.vocab.nearest_words(direction, k=5)
        anti_aligned = system.vocab.nearest_words(-direction, k=5)
        
        print(f"\n  {from_style} → {to_style}:")
        print(f"    Toward: {', '.join(w for w, s in aligned)}")
        print(f"    Away:   {', '.join(w for w, s in anti_aligned)}")
    
    # Key insight
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHT")
    print("=" * 70)
    print("""
  Style as CENTROID:
    - Each style is the average position of its exemplars
    - Classification = which centroid is closest (cosine similarity)
    - Transfer = interpolate toward target centroid
  
  Style DIRECTION:
    - direction = centroid_B - centroid_A
    - Words aligned with direction = what makes B different from A
    - This IS the style difference, geometrically
  
  Vector arithmetic:
    - new_position = (1-α) * content + α * style_centroid
    - This moves content toward the style
    - α controls strength of style transfer
  
  The style IS a position in semantic space.
  Style transfer IS movement toward that position.
""")


if __name__ == "__main__":
    demo()
