"""
Primitive Registry - Self-Assembling Keyword-to-Primitive Bridge

Transforms text into geometric positions through primitives.
Supports both single-word and multi-word primitives.

Design Principles (from Design 103):
- Transform everything to geometry, match nothing as text
- Bootstrap is acceptable, fallbacks are not
- Multi-word primitives are attractors in the space
- The structure self-assembles through attractor dynamics

Author: Lesley Gushurst
License: GPLv3
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from .phi_lattice import PhiLattice, PHI
from .semantic_dimensions import DEFAULT_DIMENSIONS


@dataclass
class RegisteredPrimitive:
    """
    A primitive registered in the system.
    
    Unlike static Primitives, RegisteredPrimitives can come from:
    - Bootstrap keywords (known positions)
    - Ingested data (discovered positions)
    - Self-assembly (emergent positions)
    """
    keyword: str
    levels: List[int]
    source: str  # "bootstrap", "ingested", "emergent"
    words: Tuple[str, ...]  # Tokenized form
    confidence: float = 1.0  # How certain we are about this primitive
    
    def __hash__(self):
        return hash(self.keyword)
    
    @property
    def is_multi_word(self) -> bool:
        return len(self.words) > 1


class PrimitiveRegistry:
    """
    Registry of primitives for geometric text encoding.
    
    Transforms keywords into primitives at registration time.
    At query time, encodes text to geometry through primitive activation.
    
    This replaces pattern matching with geometric transformation:
    - OLD: "what is physics" in query → boost similarity (pattern match)
    - NEW: "what is physics" → primitive → levels → position (geometry)
    
    Usage:
        registry = PrimitiveRegistry()
        
        # Register from bootstrap
        registry.register("what is physics", [3, 2, 1, 1], "bootstrap")
        registry.register("physics", [3, 0, 0, 0], "bootstrap")
        
        # Encode query
        position, levels = registry.encode("what is physics?")
        # Returns position at [3, 2, 1, 1] - exact geometric match
    """
    
    def __init__(self, lattice: Optional[PhiLattice] = None):
        """
        Initialize registry.
        
        Args:
            lattice: PhiLattice for position computation (creates default if None)
        """
        if lattice is None:
            lattice = PhiLattice(DEFAULT_DIMENSIONS)
        self.lattice = lattice
        
        # Single-word primitives: word → RegisteredPrimitive
        self._single_word: Dict[str, RegisteredPrimitive] = {}
        
        # Multi-word primitives: tuple(words) → RegisteredPrimitive
        self._multi_word: Dict[Tuple[str, ...], RegisteredPrimitive] = {}
        
        # Index for efficient phrase lookup: first_word → [phrases]
        self._phrase_index: Dict[str, List[Tuple[str, ...]]] = {}
        
        # Track sources for debugging
        self._by_source: Dict[str, List[RegisteredPrimitive]] = {}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to lowercase words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def register(self, keyword: str, levels: List[int], 
                 source: str = "manual", confidence: float = 1.0) -> RegisteredPrimitive:
        """
        Register a keyword as a primitive.
        
        The keyword is transformed to geometry immediately.
        It becomes an attractor in the space.
        
        Args:
            keyword: The keyword or phrase to register
            levels: φ-levels for each dimension [domain, specificity, intent, formality]
            source: Where this primitive came from
            confidence: How certain we are (1.0 = bootstrap, lower = inferred)
            
        Returns:
            The registered primitive
        """
        words = tuple(self.tokenize(keyword))
        if not words:
            return None
        
        prim = RegisteredPrimitive(
            keyword=keyword,
            levels=levels,
            source=source,
            words=words,
            confidence=confidence
        )
        
        if len(words) == 1:
            # Single-word primitive
            existing = self._single_word.get(words[0])
            # Keep higher confidence or higher level
            if existing is None or confidence > existing.confidence:
                self._single_word[words[0]] = prim
        else:
            # Multi-word primitive
            self._multi_word[words] = prim
            # Index by first word for efficient lookup
            if words[0] not in self._phrase_index:
                self._phrase_index[words[0]] = []
            if words not in self._phrase_index[words[0]]:
                self._phrase_index[words[0]].append(words)
                # Sort by length descending (longest match first)
                self._phrase_index[words[0]].sort(key=len, reverse=True)
        
        # Track by source
        if source not in self._by_source:
            self._by_source[source] = []
        self._by_source[source].append(prim)
        
        return prim
    
    def register_from_bootstrap(self, text: str, keywords: List[str], 
                                 phi_levels: List[int]) -> List[RegisteredPrimitive]:
        """
        Register primitives from a bootstrap knowledge item.
        
        Each keyword inherits the concept's phi_levels.
        
        Args:
            text: The concept text (for potential future use)
            keywords: List of keywords to register
            phi_levels: The concept's position in φ-lattice
            
        Returns:
            List of registered primitives
        """
        registered = []
        for kw in keywords:
            prim = self.register(kw, phi_levels, source="bootstrap", confidence=1.0)
            if prim:
                registered.append(prim)
        return registered
    
    def _matches_phrase(self, words: List[str], start: int, 
                        phrase: Tuple[str, ...]) -> bool:
        """Check if words starting at index match the phrase."""
        if start + len(phrase) > len(words):
            return False
        for i, word in enumerate(phrase):
            if words[start + i] != word:
                return False
        return True
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to φ-lattice position.
        
        Uses registered primitives for transformation.
        Multi-word primitives take precedence (more specific).
        
        Args:
            text: Input text to encode
            
        Returns:
            Position vector on φ-lattice
        """
        position, _ = self.encode_with_levels(text)
        return position
    
    def encode_with_levels(self, text: str) -> Tuple[np.ndarray, List[int]]:
        """
        Encode text and return both position and levels.
        
        Args:
            text: Input text to encode
            
        Returns:
            (position, levels) tuple
        """
        words = self.tokenize(text)
        if not words:
            levels = [0] * self.lattice.ndim
            return self.lattice.levels_to_position(levels), levels
        
        # Track levels and what activated them
        levels = [0] * self.lattice.ndim
        activated = [False] * self.lattice.ndim
        
        i = 0
        while i < len(words):
            matched_phrase = False
            
            # Try multi-word matches first (greedy, longest match)
            if words[i] in self._phrase_index:
                for phrase in self._phrase_index[words[i]]:
                    if self._matches_phrase(words, i, phrase):
                        prim = self._multi_word[phrase]
                        # Multi-word primitive sets ALL dimensions it specifies
                        for dim, level in enumerate(prim.levels):
                            if dim < self.lattice.ndim:
                                # Multi-word primitives override (they're more specific)
                                if not activated[dim] or level > levels[dim]:
                                    levels[dim] = level
                                    activated[dim] = True
                        i += len(phrase)
                        matched_phrase = True
                        break
            
            if not matched_phrase:
                # Try single-word primitive
                word = words[i]
                if word in self._single_word:
                    prim = self._single_word[word]
                    # Single-word primitives use MAX aggregation
                    # For intrinsic_functional (dim 4), use largest absolute value
                    for dim, level in enumerate(prim.levels):
                        if dim < self.lattice.ndim:
                            if not activated[dim]:
                                levels[dim] = level
                                activated[dim] = True
                            elif dim == 4:  # intrinsic_functional: use largest |value|
                                if abs(level) > abs(levels[dim]):
                                    levels[dim] = level
                            elif level > levels[dim]:
                                levels[dim] = level
                i += 1
        
        position = self.lattice.levels_to_position(levels)
        return position, levels
    
    def get_activated_primitives(self, text: str) -> List[RegisteredPrimitive]:
        """Get list of primitives activated by text."""
        words = self.tokenize(text)
        if not words:
            return []
        
        primitives = []
        i = 0
        while i < len(words):
            matched_phrase = False
            
            if words[i] in self._phrase_index:
                for phrase in self._phrase_index[words[i]]:
                    if self._matches_phrase(words, i, phrase):
                        primitives.append(self._multi_word[phrase])
                        i += len(phrase)
                        matched_phrase = True
                        break
            
            if not matched_phrase:
                word = words[i]
                if word in self._single_word:
                    primitives.append(self._single_word[word])
                i += 1
        
        return primitives
    
    def explain_encoding(self, text: str) -> str:
        """Explain how text was encoded."""
        words = self.tokenize(text)
        position, levels = self.encode_with_levels(text)
        primitives = self.get_activated_primitives(text)
        
        lines = [
            f"Text: \"{text}\"",
            f"Words: {words}",
            "",
            "Activated primitives:"
        ]
        
        for prim in primitives:
            if prim.is_multi_word:
                lines.append(f"  '{prim.keyword}' (multi-word) → levels {prim.levels}")
            else:
                lines.append(f"  '{prim.keyword}' → levels {prim.levels}")
        
        lines.extend([
            "",
            f"Final levels: {levels}",
            f"Position: {position}",
            "",
            "Semantic description:"
        ])
        
        desc = self.lattice.describe_position(position)
        for dim_name, meaning in desc.items():
            lines.append(f"  {dim_name}: {meaning}")
        
        return "\n".join(lines)
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between positions."""
        return self.lattice.distance(a, b)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between positions."""
        return self.lattice.similarity(a, b)
    
    @property
    def stats(self) -> Dict:
        """Get registry statistics."""
        return {
            "single_word_count": len(self._single_word),
            "multi_word_count": len(self._multi_word),
            "total_count": len(self._single_word) + len(self._multi_word),
            "by_source": {k: len(v) for k, v in self._by_source.items()}
        }
    
    def __repr__(self) -> str:
        stats = self.stats
        return f"PrimitiveRegistry(single={stats['single_word_count']}, multi={stats['multi_word_count']})"


def create_registry_from_primitives(lattice: Optional[PhiLattice] = None) -> PrimitiveRegistry:
    """
    Create a PrimitiveRegistry seeded from static primitives.
    
    Uses ALL_PRIMITIVES from primitives.py to populate the registry.
    Each primitive defines a (dimension, level) mapping for its keywords.
    
    Keywords that appear in multiple primitives get COMBINED levels
    (one keyword can activate multiple dimensions).
    
    Args:
        lattice: Optional PhiLattice instance
        
    Returns:
        Populated PrimitiveRegistry
    """
    from .primitives import ALL_PRIMITIVES
    from collections import defaultdict
    
    registry = PrimitiveRegistry(lattice)
    ndim = registry.lattice.ndim
    
    # First pass: collect all dimension activations per keyword
    keyword_levels: Dict[str, List[int]] = defaultdict(lambda: [0] * ndim)
    
    for prim in ALL_PRIMITIVES:
        if prim.dimension < ndim:
            for kw in prim.keywords:
                kw_lower = kw.lower()
                # Use MAX per dimension (Sierpinski property)
                current = keyword_levels[kw_lower][prim.dimension]
                # For intrinsic_functional (dim 4), negative levels are meaningful
                # Use the level with larger absolute value, preserving sign
                if prim.dimension == 4:  # intrinsic_functional
                    if abs(prim.level) > abs(current):
                        keyword_levels[kw_lower][prim.dimension] = prim.level
                else:
                    keyword_levels[kw_lower][prim.dimension] = max(current, prim.level)
    
    # Second pass: register combined levels
    for kw, levels in keyword_levels.items():
        registry.register(kw, levels, source="primitives", confidence=1.0)
    
    return registry


def create_registry_from_bootstrap(knowledge_items: List[dict], 
                                    lattice: Optional[PhiLattice] = None) -> PrimitiveRegistry:
    """
    Create a PrimitiveRegistry from bootstrap knowledge items.
    
    Each item should have:
    - "keywords": List of keywords to register
    - "phi_levels": Position in φ-lattice
    
    Args:
        knowledge_items: List of bootstrap knowledge dicts
        lattice: Optional PhiLattice instance
        
    Returns:
        Populated PrimitiveRegistry
    """
    registry = PrimitiveRegistry(lattice)
    
    for item in knowledge_items:
        keywords = item.get("keywords", [])
        phi_levels = item.get("phi_levels")
        
        if keywords and phi_levels:
            registry.register_from_bootstrap(
                text=item.get("text", ""),
                keywords=keywords,
                phi_levels=phi_levels
            )
    
    return registry
