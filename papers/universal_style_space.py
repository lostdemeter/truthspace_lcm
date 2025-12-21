#!/usr/bin/env python3
"""
Universal Style Space

The key insight: There are UNIVERSAL semantic axes that all language exists in.
Styles are not separate systems - they are POSITIONS in this universal space.

Universal Axes (like X, Y, Z in physical space):
  1. CONCRETE ←→ ABSTRACT
  2. FORMAL ←→ CASUAL  
  3. POSITIVE ←→ NEGATIVE (valence)
  4. ACTIVE ←→ PASSIVE
  5. SPECIFIC ←→ GENERAL
  6. EMOTIONAL ←→ RATIONAL
  7. SIMPLE ←→ COMPLEX

A style is defined by its position on these axes:
  - Warhammer 40k: High ABSTRACT, High FORMAL, High NEGATIVE, High EMOTIONAL
  - Romance: High EMOTIONAL, Medium ABSTRACT, Variable VALENCE
  - Technical: High FORMAL, High SPECIFIC, High RATIONAL, High CONCRETE

Style transfer = moving content from one position to another along these axes.

This is purely GEOMETRIC - no neural network needed.
"""

import sys
import os
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import numpy as np

PHI = (1 + math.sqrt(5)) / 2


# ============================================================================
# UNIVERSAL SEMANTIC AXES
# ============================================================================

class UniversalAxis(NamedTuple):
    """A universal semantic axis defined by polar word pairs."""
    name: str
    negative_pole: str  # e.g., "concrete"
    positive_pole: str  # e.g., "abstract"
    negative_words: Set[str]  # Words that indicate negative pole
    positive_words: Set[str]  # Words that indicate positive pole


UNIVERSAL_AXES = [
    UniversalAxis(
        name="CONCRETE_ABSTRACT",
        negative_pole="concrete",
        positive_pole="abstract",
        negative_words={'thing', 'object', 'item', 'stuff', 'material', 'physical', 
                       'tangible', 'solid', 'real', 'actual', 'literal', 'specific',
                       'hand', 'foot', 'rock', 'tree', 'house', 'car', 'table'},
        positive_words={'concept', 'idea', 'notion', 'essence', 'spirit', 'soul',
                       'destiny', 'fate', 'cosmic', 'eternal', 'infinite', 'divine',
                       'transcendent', 'metaphysical', 'philosophical', 'abstract'},
    ),
    UniversalAxis(
        name="FORMAL_CASUAL",
        negative_pole="casual",
        positive_pole="formal",
        negative_words={'guy', 'stuff', 'thing', 'gonna', 'wanna', 'kinda', 'yeah',
                       'cool', 'awesome', 'dude', 'man', 'like', 'totally', 'super',
                       'ok', 'okay', 'hey', 'hi', 'yo', 'nope', 'yep'},
        positive_words={'individual', 'entity', 'therefore', 'hence', 'thus',
                       'consequently', 'furthermore', 'moreover', 'nevertheless',
                       'notwithstanding', 'whereas', 'hereby', 'therein', 'wherein',
                       'henceforth', 'aforementioned', 'subsequent', 'prior'},
    ),
    UniversalAxis(
        name="POSITIVE_NEGATIVE",
        negative_pole="negative",
        positive_pole="positive",
        negative_words={'death', 'destruction', 'doom', 'despair', 'darkness', 'evil',
                       'corrupt', 'decay', 'ruin', 'horror', 'terror', 'dread',
                       'suffering', 'pain', 'agony', 'torment', 'hate', 'war'},
        positive_words={'life', 'creation', 'hope', 'light', 'good', 'pure', 'growth',
                       'flourish', 'joy', 'peace', 'love', 'harmony', 'beauty',
                       'happiness', 'pleasure', 'delight', 'wonder', 'miracle'},
    ),
    UniversalAxis(
        name="ACTIVE_PASSIVE",
        negative_pole="passive",
        positive_pole="active",
        negative_words={'was', 'were', 'been', 'being', 'received', 'given',
                       'happened', 'occurred', 'experienced', 'underwent', 'suffered',
                       'endured', 'waited', 'remained', 'stayed', 'rested'},
        positive_words={'did', 'made', 'created', 'built', 'destroyed', 'conquered',
                       'seized', 'struck', 'charged', 'attacked', 'defended',
                       'fought', 'commanded', 'led', 'drove', 'pushed', 'pulled'},
    ),
    UniversalAxis(
        name="SPECIFIC_GENERAL",
        negative_pole="general",
        positive_pole="specific",
        negative_words={'some', 'any', 'many', 'few', 'several', 'various', 'certain',
                       'thing', 'stuff', 'something', 'someone', 'somewhere',
                       'often', 'sometimes', 'usually', 'generally', 'typically'},
        positive_words={'exactly', 'precisely', 'specifically', 'particularly',
                       'namely', 'especially', 'notably', 'distinctly', 'uniquely',
                       'the', 'this', 'that', 'these', 'those', 'said', 'named'},
    ),
    UniversalAxis(
        name="EMOTIONAL_RATIONAL",
        negative_pole="rational",
        positive_pole="emotional",
        negative_words={'logical', 'rational', 'reasonable', 'calculated', 'measured',
                       'analyzed', 'determined', 'concluded', 'deduced', 'inferred',
                       'therefore', 'thus', 'hence', 'because', 'since', 'given'},
        positive_words={'felt', 'feeling', 'emotion', 'passion', 'fury', 'rage',
                       'love', 'hate', 'fear', 'joy', 'sorrow', 'grief', 'ecstasy',
                       'burning', 'aching', 'yearning', 'longing', 'desperate'},
    ),
    UniversalAxis(
        name="SIMPLE_COMPLEX",
        negative_pole="simple",
        positive_pole="complex",
        negative_words={'simple', 'basic', 'plain', 'clear', 'easy', 'straightforward',
                       'direct', 'brief', 'short', 'quick', 'fast', 'small'},
        positive_words={'complex', 'intricate', 'elaborate', 'ornate', 'sophisticated',
                       'nuanced', 'multifaceted', 'layered', 'detailed', 'extensive',
                       'comprehensive', 'thorough', 'exhaustive', 'byzantine'},
    ),
]


# ============================================================================
# STYLE DEFINITIONS AS POSITIONS IN UNIVERSAL SPACE
# ============================================================================

class StylePosition(NamedTuple):
    """A style defined by its position on universal axes."""
    name: str
    positions: Dict[str, float]  # axis_name -> position (-1 to +1)
    description: str


# Predefined styles as positions in universal space
PREDEFINED_STYLES = {
    "Q&A": StylePosition(
        name="Q&A",
        positions={
            "CONCRETE_ABSTRACT": -0.3,    # Slightly concrete (factual)
            "FORMAL_CASUAL": 0.2,          # Slightly formal
            "POSITIVE_NEGATIVE": 0.0,      # Neutral valence
            "ACTIVE_PASSIVE": 0.3,         # Slightly active (questions probe)
            "SPECIFIC_GENERAL": 0.5,       # Specific (precise answers)
            "EMOTIONAL_RATIONAL": -0.5,    # Rational (informational)
            "SIMPLE_COMPLEX": -0.3,        # Simple (clear answers)
        },
        description="Question and answer style - factual, specific, clear"
    ),
    "Warhammer": StylePosition(
        name="Warhammer 40k",
        positions={
            "CONCRETE_ABSTRACT": 0.7,      # Abstract (cosmic, epic)
            "FORMAL_CASUAL": 0.8,          # Very formal (gothic)
            "POSITIVE_NEGATIVE": -0.9,     # Very negative (grimdark)
            "ACTIVE_PASSIVE": 0.6,         # Active (war, conflict)
            "SPECIFIC_GENERAL": 0.3,       # Somewhat specific
            "EMOTIONAL_RATIONAL": 0.8,     # Very emotional (zealotry)
            "SIMPLE_COMPLEX": 0.7,         # Complex (ornate)
        },
        description="Warhammer 40k style - grimdark, gothic, epic, zealous"
    ),
    "Romance": StylePosition(
        name="Romance",
        positions={
            "CONCRETE_ABSTRACT": 0.2,      # Slightly abstract (emotions)
            "FORMAL_CASUAL": 0.0,          # Neutral formality
            "POSITIVE_NEGATIVE": 0.3,      # Slightly positive (love)
            "ACTIVE_PASSIVE": 0.2,         # Slightly active
            "SPECIFIC_GENERAL": 0.4,       # Specific (sensory details)
            "EMOTIONAL_RATIONAL": 0.9,     # Very emotional
            "SIMPLE_COMPLEX": 0.3,         # Somewhat complex
        },
        description="Romance style - emotional, sensory, tension-filled"
    ),
    "Technical": StylePosition(
        name="Technical",
        positions={
            "CONCRETE_ABSTRACT": -0.6,     # Concrete (practical)
            "FORMAL_CASUAL": 0.5,          # Formal (professional)
            "POSITIVE_NEGATIVE": 0.0,      # Neutral
            "ACTIVE_PASSIVE": 0.4,         # Active (do this, run that)
            "SPECIFIC_GENERAL": 0.8,       # Very specific (precise)
            "EMOTIONAL_RATIONAL": -0.8,    # Very rational
            "SIMPLE_COMPLEX": 0.0,         # Neutral complexity
        },
        description="Technical documentation style - precise, practical, clear"
    ),
    "Noir": StylePosition(
        name="Noir",
        positions={
            "CONCRETE_ABSTRACT": 0.3,      # Somewhat abstract (metaphors)
            "FORMAL_CASUAL": 0.3,          # Somewhat formal
            "POSITIVE_NEGATIVE": -0.6,     # Negative (cynical)
            "ACTIVE_PASSIVE": 0.1,         # Slightly active
            "SPECIFIC_GENERAL": 0.5,       # Specific (details)
            "EMOTIONAL_RATIONAL": 0.4,     # Somewhat emotional
            "SIMPLE_COMPLEX": 0.4,         # Somewhat complex (metaphors)
        },
        description="Film noir style - cynical, atmospheric, metaphorical"
    ),
}


# ============================================================================
# UNIVERSAL STYLE SPACE
# ============================================================================

class UniversalStyleSpace:
    """
    A semantic space defined by universal axes.
    
    All content exists in this space.
    Styles are positions in this space.
    Style transfer = moving content toward a style position.
    """
    
    def __init__(self):
        self.axes = UNIVERSAL_AXES
        self.styles = PREDEFINED_STYLES
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def measure_position(self, text: str) -> Dict[str, float]:
        """
        Measure where text sits on each universal axis.
        
        Returns position from -1 (negative pole) to +1 (positive pole).
        """
        words = set(self._tokenize(text))
        positions = {}
        
        for axis in self.axes:
            neg_count = len(words & axis.negative_words)
            pos_count = len(words & axis.positive_words)
            total = neg_count + pos_count
            
            if total == 0:
                positions[axis.name] = 0.0  # Neutral if no indicator words
            else:
                # Position from -1 (all negative) to +1 (all positive)
                positions[axis.name] = (pos_count - neg_count) / total
        
        return positions
    
    def distance_to_style(self, text: str, style_name: str) -> float:
        """Euclidean distance from text position to style position."""
        if style_name not in self.styles:
            raise ValueError(f"Unknown style: {style_name}")
        
        text_pos = self.measure_position(text)
        style_pos = self.styles[style_name].positions
        
        dist_sq = 0.0
        for axis_name in style_pos:
            text_val = text_pos.get(axis_name, 0.0)
            style_val = style_pos[axis_name]
            dist_sq += (text_val - style_val) ** 2
        
        return math.sqrt(dist_sq)
    
    def style_similarity(self, text: str, style_name: str) -> float:
        """Similarity to style (0 to 1, higher = more similar)."""
        dist = self.distance_to_style(text, style_name)
        # Max possible distance is sqrt(7 * 4) ≈ 5.3 (7 axes, range 2 each)
        max_dist = math.sqrt(len(self.axes) * 4)
        return 1.0 - (dist / max_dist)
    
    def classify_style(self, text: str) -> Tuple[str, float]:
        """Find the best matching style for text."""
        best_style = None
        best_sim = -1
        
        for style_name in self.styles:
            sim = self.style_similarity(text, style_name)
            if sim > best_sim:
                best_sim = sim
                best_style = style_name
        
        return best_style, best_sim
    
    def style_vector(self, text: str, style_name: str) -> Dict[str, float]:
        """
        Compute the vector from text position to style position.
        
        This is the "direction to move" to apply the style.
        """
        if style_name not in self.styles:
            raise ValueError(f"Unknown style: {style_name}")
        
        text_pos = self.measure_position(text)
        style_pos = self.styles[style_name].positions
        
        vector = {}
        for axis_name in style_pos:
            text_val = text_pos.get(axis_name, 0.0)
            style_val = style_pos[axis_name]
            vector[axis_name] = style_val - text_val
        
        return vector
    
    def add_style(self, name: str, exemplars: List[str], description: str = ""):
        """
        Define a new style from exemplars.
        
        The style position is the average position of the exemplars.
        """
        if not exemplars:
            raise ValueError("Need at least one exemplar")
        
        # Measure position of each exemplar
        positions_list = [self.measure_position(e) for e in exemplars]
        
        # Average positions
        avg_positions = {}
        for axis in self.axes:
            values = [p.get(axis.name, 0.0) for p in positions_list]
            avg_positions[axis.name] = sum(values) / len(values)
        
        # Create style
        style = StylePosition(
            name=name,
            positions=avg_positions,
            description=description
        )
        self.styles[name] = style
        
        return style
    
    def visualize_position(self, text: str) -> str:
        """Create ASCII visualization of text position on axes."""
        positions = self.measure_position(text)
        
        lines = []
        for axis in self.axes:
            pos = positions.get(axis.name, 0.0)
            
            # Create bar from -1 to +1
            bar_width = 40
            center = bar_width // 2
            marker_pos = int((pos + 1) / 2 * bar_width)
            marker_pos = max(0, min(bar_width - 1, marker_pos))
            
            bar = ['-'] * bar_width
            bar[center] = '|'  # Center marker
            bar[marker_pos] = '*'  # Position marker
            
            neg_label = axis.negative_pole[:8].ljust(8)
            pos_label = axis.positive_pole[:8].rjust(8)
            
            lines.append(f"  {neg_label} [{''.join(bar)}] {pos_label}  ({pos:+.2f})")
        
        return '\n'.join(lines)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("  UNIVERSAL STYLE SPACE")
    print("  Styles are positions on universal semantic axes")
    print("=" * 70)
    
    space = UniversalStyleSpace()
    
    # Show predefined style positions
    print("\n[1] Predefined style positions on universal axes:")
    print("-" * 70)
    
    for style_name, style in space.styles.items():
        print(f"\n  {style_name}: {style.description}")
        for axis_name, pos in style.positions.items():
            axis_short = axis_name.replace("_", "→")
            bar = '█' * int(abs(pos) * 10)
            sign = '+' if pos >= 0 else '-'
            print(f"    {axis_short:25s} {sign}{bar:10s} ({pos:+.1f})")
    
    # Test sentences
    test_sentences = [
        "Who is the main character of the story?",
        "The Inquisitor's power sword crackled with holy fury as he purged the heretics.",
        "Her heart raced as their eyes met across the crowded ballroom.",
        "The function accepts two parameters and returns a boolean value.",
        "The rain fell like tears on the city streets as I lit another cigarette.",
        "Captain Ahab hunted the white whale with monomaniacal obsession.",
    ]
    
    print("\n\n[2] Classifying test sentences:")
    print("-" * 70)
    
    for sentence in test_sentences:
        print(f"\n  \"{sentence[:60]}...\"" if len(sentence) > 60 else f"\n  \"{sentence}\"")
        
        # Classify
        best_style, best_sim = space.classify_style(sentence)
        print(f"    Best match: {best_style} (similarity: {best_sim:.3f})")
        
        # Show all similarities
        sims = [(name, space.style_similarity(sentence, name)) for name in space.styles]
        sims.sort(key=lambda x: -x[1])
        print(f"    All: {', '.join(f'{n}={s:.2f}' for n, s in sims)}")
    
    # Visualize a sentence's position
    print("\n\n[3] Visualizing sentence position on universal axes:")
    print("-" * 70)
    
    sentence = "The Inquisitor's power sword crackled with holy fury as he purged the heretics."
    print(f"\n  \"{sentence}\"")
    print()
    print(space.visualize_position(sentence))
    
    # Show style transfer vector
    print("\n\n[4] Style transfer vectors (what needs to change):")
    print("-" * 70)
    
    source = "The captain commanded the ship to pursue the whale."
    print(f"\n  Source: \"{source}\"")
    
    for target_style in ["Warhammer", "Romance", "Technical"]:
        print(f"\n  → To {target_style}:")
        vector = space.style_vector(source, target_style)
        for axis_name, delta in sorted(vector.items(), key=lambda x: -abs(x[1])):
            if abs(delta) > 0.1:
                direction = "more" if delta > 0 else "less"
                axis_short = axis_name.split("_")[1].lower()
                print(f"      {direction} {axis_short}: {delta:+.2f}")
    
    # Add a custom style from exemplars
    print("\n\n[5] Defining a custom style from exemplars:")
    print("-" * 70)
    
    pirate_exemplars = [
        "Arrr, ye scurvy dog, hand over the treasure or walk the plank!",
        "Shiver me timbers, there be a storm brewin' on the horizon!",
        "Yo ho ho and a bottle of rum, we sail for Tortuga at dawn!",
        "Avast ye landlubbers, the Jolly Roger flies high today!",
        "Blimey, that be the finest booty I've seen in all me years at sea!",
    ]
    
    pirate_style = space.add_style("Pirate", pirate_exemplars, "Pirate speak - nautical, informal, exclamatory")
    print(f"\n  Added 'Pirate' style from {len(pirate_exemplars)} exemplars:")
    for axis_name, pos in pirate_style.positions.items():
        axis_short = axis_name.replace("_", "→")
        print(f"    {axis_short:25s} ({pos:+.2f})")
    
    # Test the new style
    print("\n  Testing pirate style detection:")
    test = "Arrr, the captain be huntin' the white whale!"
    best, sim = space.classify_style(test)
    print(f"    \"{test}\"")
    print(f"    Best match: {best} ({sim:.3f})")
    
    # Key insight
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHT")
    print("=" * 70)
    print("""
  Universal axes are FIXED - they're built into language:
    - CONCRETE ←→ ABSTRACT
    - FORMAL ←→ CASUAL
    - POSITIVE ←→ NEGATIVE
    - ACTIVE ←→ PASSIVE
    - SPECIFIC ←→ GENERAL
    - EMOTIONAL ←→ RATIONAL
    - SIMPLE ←→ COMPLEX
  
  Styles are POSITIONS in this universal space.
  
  To define a new style:
    1. Give exemplars
    2. Measure their average position on universal axes
    3. That position IS the style
  
  To apply a style:
    1. Measure content position
    2. Compute vector to target style
    3. Move content along that vector
  
  This solves the chicken-and-egg problem:
    - Axes are universal (no need to discover them)
    - Styles are just positions (easy to define from exemplars)
    - Style transfer is vector arithmetic (purely geometric)
""")


if __name__ == "__main__":
    demo()
