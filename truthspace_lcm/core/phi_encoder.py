"""
φ-Lattice Encoder

Encodes text to φ-lattice positions using primitives.
Replaces eigenspace-based encoding for knowledge matching.

Design Principles:
- Uses primitive detection to determine semantic dimensions
- MAX aggregation per dimension (Sierpinski property)
- Position decay by word order (optional)
- Returns absolute, verifiable positions

Author: Lesley Gushurst
License: GPLv3
"""

import re
import numpy as np
from typing import List, Set, Dict, Optional, Tuple

from .phi_lattice import PhiLattice, PHI
from .semantic_dimensions import DEFAULT_DIMENSIONS, DEFAULT_WEIGHTS
from .primitives import KEYWORD_MAP, Primitive


class PhiLatticeEncoder:
    """
    Encodes text to φ-lattice positions.
    
    Uses primitive detection to determine which semantic dimensions
    are activated and at what level.
    
    Encoding follows MAX aggregation (Sierpinski property):
    - Multiple words activating same dimension → take max level
    - This ensures the most specific/relevant primitive wins
    
    Usage:
        encoder = PhiLatticeEncoder()
        
        # Encode text to position
        pos = encoder.encode("what is physics?")
        
        # Get both position and levels
        pos, levels = encoder.encode_with_levels("hello there")
        
        # Compute distance
        dist = encoder.distance(pos_a, pos_b)
    """
    
    def __init__(self, lattice: Optional[PhiLattice] = None,
                 keyword_map: Optional[Dict[str, Primitive]] = None):
        """
        Initialize encoder.
        
        Args:
            lattice: PhiLattice to use (creates default if None)
            keyword_map: Keyword → Primitive mapping (uses default if None)
        """
        if lattice is None:
            lattice = PhiLattice(DEFAULT_DIMENSIONS)
        self.lattice = lattice
        
        if keyword_map is None:
            keyword_map = KEYWORD_MAP
        self.keyword_map = keyword_map
        
        # Weights for distance calculation
        self._weights = np.array(DEFAULT_WEIGHTS)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text to words.
        
        Simple word extraction - all lowercase.
        """
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to φ-lattice position.
        
        Uses MAX aggregation per dimension.
        
        Args:
            text: Input text to encode
            
        Returns:
            Position vector on φ-lattice
        """
        pos, _ = self.encode_with_levels(text)
        return pos
    
    def encode_with_levels(self, text: str) -> Tuple[np.ndarray, List[int]]:
        """
        Encode text and return both position and levels.
        
        Args:
            text: Input text to encode
            
        Returns:
            (position, levels) tuple
        """
        words = self.tokenize(text)
        
        # Default levels: 0 (neutral) for each dimension
        levels = [0] * self.lattice.ndim
        
        # Track which dimensions were activated
        activated = [False] * self.lattice.ndim
        
        for word in words:
            if word in self.keyword_map:
                prim = self.keyword_map[word]
                dim = prim.dimension
                
                if dim < self.lattice.ndim:
                    # MAX aggregation (Sierpinski property)
                    if not activated[dim] or prim.level > levels[dim]:
                        levels[dim] = prim.level
                        activated[dim] = True
        
        position = self.lattice.levels_to_position(levels)
        return position, levels
    
    def encode_to_levels(self, text: str) -> List[int]:
        """
        Encode text to φ-level indices only.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of level indices
        """
        _, levels = self.encode_with_levels(text)
        return levels
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Weighted distance in φ-space.
        
        Args:
            a: First position
            b: Second position
            
        Returns:
            Weighted Euclidean distance
        """
        return self.lattice.distance(a, b, self._weights)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Similarity score between two positions.
        
        Args:
            a: First position
            b: Second position
            
        Returns:
            Similarity in range (0, 1]
        """
        return self.lattice.similarity(a, b, self._weights)
    
    def describe(self, text: str) -> Dict[str, str]:
        """
        Get semantic description of encoded text.
        
        Args:
            text: Input text
            
        Returns:
            Dict mapping dimension name to semantic meaning
        """
        pos, levels = self.encode_with_levels(text)
        return self.lattice.describe_position(pos)
    
    def get_activated_primitives(self, text: str) -> List[Primitive]:
        """
        Get list of primitives activated by text.
        
        Args:
            text: Input text
            
        Returns:
            List of Primitive objects that were activated
        """
        words = self.tokenize(text)
        primitives = []
        seen = set()
        
        for word in words:
            if word in self.keyword_map:
                prim = self.keyword_map[word]
                if prim.name not in seen:
                    primitives.append(prim)
                    seen.add(prim.name)
        
        return primitives
    
    def explain_encoding(self, text: str) -> str:
        """
        Explain how text was encoded.
        
        Args:
            text: Input text
            
        Returns:
            Human-readable explanation
        """
        words = self.tokenize(text)
        pos, levels = self.encode_with_levels(text)
        
        lines = [
            f"Text: \"{text}\"",
            f"Words: {words}",
            "",
            "Activated primitives:"
        ]
        
        for word in words:
            if word in self.keyword_map:
                prim = self.keyword_map[word]
                dim_name = self.lattice.dimensions[prim.dimension].name if prim.dimension < self.lattice.ndim else f"dim_{prim.dimension}"
                lines.append(f"  '{word}' → {prim.name} (dim={dim_name}, level={prim.level})")
        
        lines.extend([
            "",
            f"Final levels: {levels}",
            f"Position: {pos}",
            "",
            "Semantic description:"
        ])
        
        desc = self.lattice.describe_position(pos)
        for dim_name, meaning in desc.items():
            lines.append(f"  {dim_name}: {meaning}")
        
        return "\n".join(lines)


def create_default_encoder() -> PhiLatticeEncoder:
    """Create encoder with default configuration."""
    return PhiLatticeEncoder()
