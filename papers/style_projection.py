#!/usr/bin/env python3
"""
Generalizable Style Projection System

The key insight: ANY style can be defined by axes that EMERGE from exemplars.

Q&A style has axes: WHO, WHAT, WHERE, WHY, HOW
Warhammer 40k style has axes: FACTION, GRIMDARK, SCALE, CONFLICT, GOTHIC
Romance style has axes: TENSION, EMOTION, SETTING, OBSTACLE

The chicken-and-egg solution:
  1. Given style exemplars, compute principal components
  2. These components ARE the style axes
  3. Project content onto these axes
  4. Reconstruct in the target style

This is true holographic projection:
  - Content = object beam
  - Style axes = reference beam  
  - Output = hologram (content viewed through style)
"""

import sys
import os
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, NamedTuple, Callable
import numpy as np

PHI = (1 + math.sqrt(5)) / 2


# ============================================================================
# STYLE AXIS - Emerges from exemplars via PCA
# ============================================================================

class StyleAxis(NamedTuple):
    """A style axis that emerged from exemplar analysis."""
    name: str
    direction: np.ndarray
    variance_explained: float
    exemplar_words: List[str]  # Words that load heavily on this axis


class Style:
    """
    A style defined by axes that emerge from exemplars.
    
    The axes are computed via PCA on the exemplar encodings.
    """
    
    def __init__(self, name: str, dim: int = 64):
        self.name = name
        self.dim = dim
        self.axes: List[StyleAxis] = []
        self.exemplars: List[str] = []
        self.word_positions: Dict[str, np.ndarray] = {}
        self.centroid: np.ndarray = np.zeros(dim)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_word_position(self, word: str) -> np.ndarray:
        if word in self.word_positions:
            return self.word_positions[word]
        
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.5
        np.random.seed(None)
        self.word_positions[word] = pos
        return pos
    
    def _encode_text(self, text: str) -> np.ndarray:
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        pos = np.zeros(self.dim)
        for word in words:
            pos += self._get_word_position(word)
        return pos / len(words)
    
    def learn_from_exemplars(self, exemplars: List[str], n_axes: int = 5):
        """
        Learn style axes from exemplars using PCA.
        
        The principal components of the exemplar encodings
        become the style axes.
        """
        self.exemplars = exemplars
        
        if len(exemplars) < 2:
            print(f"Warning: Need at least 2 exemplars, got {len(exemplars)}")
            return
        
        # Encode all exemplars
        encodings = np.array([self._encode_text(e) for e in exemplars])
        
        # Compute centroid (mean style)
        self.centroid = np.mean(encodings, axis=0)
        
        # Center the data
        centered = encodings - self.centroid
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Total variance
        total_var = np.sum(eigenvalues)
        
        # Create axes from top eigenvectors
        self.axes = []
        for i in range(min(n_axes, len(eigenvalues))):
            direction = eigenvectors[:, i]
            var_explained = eigenvalues[i] / total_var if total_var > 0 else 0
            
            # Find words that load heavily on this axis
            # (words whose positions align with this direction)
            word_loadings = []
            for word, pos in self.word_positions.items():
                loading = abs(np.dot(pos, direction))
                word_loadings.append((word, loading))
            
            word_loadings.sort(key=lambda x: -x[1])
            top_words = [w for w, _ in word_loadings[:10]]
            
            axis = StyleAxis(
                name=f"AXIS_{i+1}",
                direction=direction,
                variance_explained=var_explained,
                exemplar_words=top_words
            )
            self.axes.append(axis)
        
        print(f"Learned {len(self.axes)} axes from {len(exemplars)} exemplars")
        for ax in self.axes:
            print(f"  {ax.name}: {ax.variance_explained:.1%} variance, words: {ax.exemplar_words[:5]}")
    
    def project(self, text: str) -> Dict[str, float]:
        """Project text onto style axes. Returns projection magnitudes."""
        encoding = self._encode_text(text)
        centered = encoding - self.centroid
        
        projections = {}
        for axis in self.axes:
            mag = np.dot(centered, axis.direction)
            projections[axis.name] = mag
        
        return projections
    
    def distance_to_style(self, text: str) -> float:
        """How far is this text from the style centroid?"""
        encoding = self._encode_text(text)
        return np.linalg.norm(encoding - self.centroid)
    
    def style_similarity(self, text: str) -> float:
        """How similar is this text to the style? (0 to 1)"""
        dist = self.distance_to_style(text)
        # Convert distance to similarity using exponential decay
        return np.exp(-dist / 2)


# ============================================================================
# STYLE TRANSFER - Project content through style axes
# ============================================================================

class StyleProjector:
    """
    Projects content from one style to another.
    
    This is the holographic projection:
    - Source content is the "object"
    - Target style axes are the "reference beam"
    - Output is the "hologram"
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.styles: Dict[str, Style] = {}
        self.word_positions: Dict[str, np.ndarray] = {}
    
    def _get_word_position(self, word: str) -> np.ndarray:
        if word in self.word_positions:
            return self.word_positions[word]
        
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.5
        np.random.seed(None)
        self.word_positions[word] = pos
        return pos
    
    def define_style(self, name: str, exemplars: List[str], n_axes: int = 5) -> Style:
        """Define a style from exemplars."""
        style = Style(name, self.dim)
        style.word_positions = self.word_positions  # Share word positions
        style.learn_from_exemplars(exemplars, n_axes)
        self.styles[name] = style
        return style
    
    def analyze_content(self, content: str, style_name: str) -> Dict[str, float]:
        """Analyze how content projects onto a style's axes."""
        if style_name not in self.styles:
            raise ValueError(f"Unknown style: {style_name}")
        
        style = self.styles[style_name]
        return style.project(content)
    
    def find_best_style(self, content: str) -> Tuple[str, float]:
        """Find which defined style best matches the content."""
        best_style = None
        best_sim = -1
        
        for name, style in self.styles.items():
            sim = style.style_similarity(content)
            if sim > best_sim:
                best_sim = sim
                best_style = name
        
        return best_style, best_sim


# ============================================================================
# PREDEFINED STYLES WITH EXEMPLARS
# ============================================================================

QA_EXEMPLARS = [
    "Who is the main character?",
    "What is the story about?",
    "Where does the action take place?",
    "When did this happen?",
    "Why did the protagonist do that?",
    "How does the story end?",
    "What is the meaning of this word?",
    "Who wrote this book?",
    "Where is the setting?",
    "What happened next?",
]

WARHAMMER_EXEMPLARS = [
    "The Emperor protects, but having a loaded bolter never hurts.",
    "In the grim darkness of the far future, there is only war.",
    "The Inquisitor strode through the burning hab-blocks, his power sword crackling.",
    "Chaos Space Marines erupted from the warp rift, their armor dripping with corruption.",
    "The Astartes chapter had held the line for three days against the xenos tide.",
    "Servo-skulls drifted through the cathedral, their red eyes scanning for heresy.",
    "The Tech-Priest's mechadendrites whirred as he communed with the machine spirit.",
    "Blood for the Blood God! Skulls for the Skull Throne!",
    "The Imperial Guard regiment advanced through the ash wastes, lasguns ready.",
    "Exterminatus was the only solution for a world so corrupted.",
]

ROMANCE_EXEMPLARS = [
    "Their eyes met across the crowded ballroom, and time seemed to stop.",
    "She knew she shouldn't fall for him, but her heart had other plans.",
    "His touch sent shivers down her spine, awakening feelings she'd buried long ago.",
    "They were from different worlds, but love knows no boundaries.",
    "The tension between them was palpable, electric, undeniable.",
    "She turned away, hiding the tears that threatened to fall.",
    "Against all odds, against all reason, she loved him still.",
    "His confession left her breathless, her heart racing with hope and fear.",
    "The forbidden nature of their love only made it burn brighter.",
    "In his arms, she finally found the home she'd been searching for.",
]

TECHNICAL_EXEMPLARS = [
    "The function accepts two parameters: input_data and config.",
    "To install the package, run: pip install package-name",
    "The API returns a JSON object containing the following fields.",
    "Error handling is implemented using try-except blocks.",
    "The algorithm has O(n log n) time complexity.",
    "Configure the database connection in the settings.py file.",
    "The class inherits from BaseModel and implements the interface.",
    "Unit tests should cover edge cases and error conditions.",
    "The module provides utilities for data transformation.",
    "See the documentation for a complete list of parameters.",
]

NOIR_EXEMPLARS = [
    "The rain fell like tears from a broken heart, washing the sins off the city streets.",
    "She walked into my office like trouble in high heels.",
    "I lit a cigarette and watched the smoke curl toward the ceiling fan.",
    "In this city, everyone's got an angle, and most of them are sharp enough to cut.",
    "The dame had legs that went all the way up and a story that went all the way down.",
    "I'd seen a lot of dead men in my time, but this one had a story to tell.",
    "The night was dark, but the secrets were darker.",
    "She lied like she breathed - naturally and without thinking.",
    "I poured myself three fingers of bourbon and waited for the other shoe to drop.",
    "In the end, we're all just shadows chasing shadows in a city of broken dreams.",
]


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("  GENERALIZABLE STYLE PROJECTION SYSTEM")
    print("  Axes emerge from exemplars via PCA")
    print("=" * 70)
    
    projector = StyleProjector(dim=64)
    
    # Define styles from exemplars
    print("\n[1] Learning styles from exemplars...")
    print("-" * 70)
    
    print("\nQ&A Style:")
    qa_style = projector.define_style("Q&A", QA_EXEMPLARS, n_axes=4)
    
    print("\nWarhammer 40k Style:")
    wh_style = projector.define_style("Warhammer", WARHAMMER_EXEMPLARS, n_axes=4)
    
    print("\nRomance Style:")
    rom_style = projector.define_style("Romance", ROMANCE_EXEMPLARS, n_axes=4)
    
    print("\nTechnical Style:")
    tech_style = projector.define_style("Technical", TECHNICAL_EXEMPLARS, n_axes=4)
    
    print("\nNoir Style:")
    noir_style = projector.define_style("Noir", NOIR_EXEMPLARS, n_axes=4)
    
    # Test content
    test_sentences = [
        "Captain Ahab is the monomaniacal captain of the Pequod.",
        "Who is the white whale?",
        "The function returns a list of integers.",
        "She couldn't stop thinking about him.",
        "The Space Marine raised his bolter.",
        "I watched her walk away into the rain.",
    ]
    
    print("\n\n[2] Classifying test sentences by style...")
    print("-" * 70)
    
    for sentence in test_sentences:
        print(f"\n  \"{sentence}\"")
        
        # Find best matching style
        best_style, best_sim = projector.find_best_style(sentence)
        print(f"    Best match: {best_style} (similarity: {best_sim:.3f})")
        
        # Show all style similarities
        sims = []
        for name, style in projector.styles.items():
            sim = style.style_similarity(sentence)
            sims.append((name, sim))
        sims.sort(key=lambda x: -x[1])
        
        print(f"    All styles: {', '.join(f'{n}={s:.2f}' for n, s in sims)}")
    
    # Show projections onto style axes
    print("\n\n[3] Projecting content onto style axes...")
    print("-" * 70)
    
    content = "Captain Ahab hunts the white whale because it took his leg."
    print(f"\n  Content: \"{content}\"")
    
    for style_name in ["Q&A", "Warhammer", "Romance", "Noir"]:
        print(f"\n  Projected onto {style_name} axes:")
        projections = projector.analyze_content(content, style_name)
        for axis_name, mag in projections.items():
            print(f"    {axis_name}: {mag:+.3f}")
    
    # The key insight
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHT")
    print("=" * 70)
    print("""
  The axes EMERGE from exemplars - they are not predefined.
  
  Given ANY set of style exemplars:
  1. Encode them as vectors
  2. Compute principal components (PCA)
  3. These components ARE the style axes
  
  This solves the chicken-and-egg problem:
  - You don't need to know the axes in advance
  - You just need examples of the style
  - The axes emerge from the examples
  
  The same content can be projected onto ANY style's axes.
  This is true holographic projection.
""")


if __name__ == "__main__":
    demo()
