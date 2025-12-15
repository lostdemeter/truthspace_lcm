"""
φ-Encoder: Semantic Encoding with Knowledge-Loaded Primitives

This is the refactored encoder that:
- Loads primitives from TruthSpace (not hardcoded)
- Provides pure mathematical transformation text → position
- Supports bidirectional mapping

Design principles:
- Primitives ARE knowledge entries (type=PRIMITIVE)
- The encoder is just math - no domain logic
- Bootstrap: if no primitives in DB, use minimal hardcoded set
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


# =============================================================================
# PRIMITIVE TYPES
# =============================================================================

class PrimitiveType(Enum):
    """Types of semantic primitives."""
    ACTION = "action"       # Verbs: what to do
    DOMAIN = "domain"       # Nouns: what domain
    MODIFIER = "modifier"   # Adjectives/adverbs: how


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Primitive:
    """A semantic primitive - an anchor point in TruthSpace."""
    name: str
    ptype: PrimitiveType
    keywords: Set[str]
    dimension: int = 0      # Which dimension this primitive occupies
    level: int = 0          # φ^level scaling within dimension
    opposite: Optional[str] = None
    
    def get_position(self, dim: int = 8) -> np.ndarray:
        """Compute the φ-scaled position for this primitive."""
        position = np.zeros(dim)
        
        # Base value at φ^level
        value = PHI ** self.level
        
        # Opposite primitives get negative values
        if self.opposite and self.level % 2 == 1:
            value = -value
        
        position[self.dimension] = value
        return position


@dataclass
class SemanticDecomposition:
    """Result of decomposing text into primitives."""
    primitives: List[Tuple[Primitive, float]]  # (primitive, relevance)
    residual_keywords: List[str]
    position: np.ndarray
    confidence: float


# =============================================================================
# BOOTSTRAP PRIMITIVES
# =============================================================================

# Minimal set of primitives for bootstrap (before DB is populated)
BOOTSTRAP_PRIMITIVES = [
    # Actions (dims 0-3)
    Primitive("CREATE", PrimitiveType.ACTION, {"create", "make", "new", "add", "build", "mkdir", "touch"}, 0, 0),
    Primitive("DESTROY", PrimitiveType.ACTION, {"destroy", "delete", "remove", "rm", "kill"}, 0, 1, opposite="CREATE"),
    Primitive("READ", PrimitiveType.ACTION, {"read", "show", "display", "view", "list", "cat", "get"}, 1, 0),
    Primitive("WRITE", PrimitiveType.ACTION, {"write", "modify", "edit", "save", "update"}, 1, 1, opposite="READ"),
    Primitive("MOVE", PrimitiveType.ACTION, {"move", "copy", "mv", "cp", "transfer"}, 2, 0),
    Primitive("CONNECT", PrimitiveType.ACTION, {"connect", "link", "ssh", "curl", "fetch"}, 3, 0),
    Primitive("SEARCH", PrimitiveType.ACTION, {"search", "find", "grep", "locate"}, 2, 1),
    Primitive("EXECUTE", PrimitiveType.ACTION, {"run", "execute", "start", "launch"}, 3, 1),
    
    # Domains (dims 4-6)
    Primitive("FILE", PrimitiveType.DOMAIN, {"file", "files", "directory", "folder", "path"}, 4, 0),
    Primitive("PROCESS", PrimitiveType.DOMAIN, {"process", "pid", "job", "task", "daemon"}, 5, 0),
    Primitive("NETWORK", PrimitiveType.DOMAIN, {"network", "interface", "ip", "port", "socket"}, 6, 0),
    Primitive("SYSTEM", PrimitiveType.DOMAIN, {"system", "kernel", "os", "memory", "cpu"}, 4, 1),
    Primitive("DATA", PrimitiveType.DOMAIN, {"data", "json", "text", "content", "output"}, 5, 1),
    
    # Modifiers (dim 7)
    Primitive("ALL", PrimitiveType.MODIFIER, {"all", "every", "entire", "complete"}, 7, 0),
    Primitive("RECURSIVE", PrimitiveType.MODIFIER, {"recursive", "-r", "deep", "nested"}, 7, 1),
    Primitive("FORCE", PrimitiveType.MODIFIER, {"force", "-f", "override"}, 7, 2),
]


# =============================================================================
# ENCODER
# =============================================================================

class PhiEncoder:
    """
    φ-based semantic encoder.
    
    Encodes natural language to geometric positions using φ-anchored primitives.
    Primitives can be loaded from TruthSpace or use bootstrap defaults.
    """
    
    def __init__(self, primitives: List[Primitive] = None):
        """
        Initialize encoder.
        
        Args:
            primitives: List of primitives to use. If None, uses bootstrap set.
        """
        self.dim = 8
        self.primitives: Dict[str, Primitive] = {}
        self.keyword_to_primitive: Dict[str, str] = {}
        
        # Load primitives
        if primitives:
            for p in primitives:
                self._register_primitive(p)
        else:
            for p in BOOTSTRAP_PRIMITIVES:
                self._register_primitive(p)
    
    def _register_primitive(self, primitive: Primitive):
        """Register a primitive and its keywords."""
        self.primitives[primitive.name] = primitive
        for keyword in primitive.keywords:
            self.keyword_to_primitive[keyword.lower()] = primitive.name
    
    @classmethod
    def from_truthspace(cls, truthspace) -> 'PhiEncoder':
        """
        Create encoder with primitives loaded from TruthSpace.
        
        Falls back to bootstrap primitives if none found in DB.
        """
        from truthspace_lcm.core.truthspace import EntryType
        
        primitives = []
        
        try:
            entries = truthspace.list_by_type(EntryType.PRIMITIVE)
            
            for entry in entries:
                meta = entry.metadata
                ptype = PrimitiveType(meta.get("primitive_type", "action"))
                keywords = set(entry.keywords)
                
                p = Primitive(
                    name=entry.name,
                    ptype=ptype,
                    keywords=keywords,
                    dimension=meta.get("dimension", 0),
                    level=meta.get("level", 0),
                    opposite=meta.get("opposite"),
                )
                primitives.append(p)
        except Exception:
            pass
        
        # Fall back to bootstrap if no primitives found
        if not primitives:
            primitives = BOOTSTRAP_PRIMITIVES
        
        return cls(primitives)
    
    # =========================================================================
    # ENCODING
    # =========================================================================
    
    def encode(self, text: str) -> SemanticDecomposition:
        """
        Encode text to semantic decomposition.
        
        Returns position in φ-space based on primitive composition.
        """
        words = self._tokenize(text)
        
        # Match words to primitives
        matched_primitives = []
        residual = []
        
        for word in words:
            word_lower = word.lower()
            
            if word_lower in self.keyword_to_primitive:
                prim_name = self.keyword_to_primitive[word_lower]
                primitive = self.primitives[prim_name]
                matched_primitives.append((primitive, 1.0))
            else:
                # Check partial matches
                found = False
                for keyword, prim_name in self.keyword_to_primitive.items():
                    if word_lower in keyword or keyword in word_lower:
                        primitive = self.primitives[prim_name]
                        matched_primitives.append((primitive, 0.5))
                        found = True
                        break
                
                if not found:
                    residual.append(word)
        
        # Compute position by summing primitive positions
        position = np.zeros(self.dim)
        
        for primitive, relevance in matched_primitives:
            prim_pos = primitive.get_position(self.dim)
            position += prim_pos * relevance
        
        # Add residual contribution
        if residual:
            residual_pos = self._encode_residual(residual)
            position += residual_pos * 0.1
        
        # Normalize
        norm = np.linalg.norm(position)
        if norm > 0:
            position = position / norm
        
        # Compute confidence
        total_words = len(words)
        matched_words = total_words - len(residual)
        confidence = matched_words / total_words if total_words > 0 else 0.0
        
        return SemanticDecomposition(
            primitives=matched_primitives,
            residual_keywords=residual,
            position=position,
            confidence=confidence,
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        import re
        # Keep hyphenated words and flags like -r, -f
        words = re.findall(r'-[a-zA-Z]|\b\w+\b', text.lower())
        return words
    
    def _encode_residual(self, words: List[str]) -> np.ndarray:
        """Encode residual words that didn't match primitives."""
        position = np.zeros(self.dim)
        
        for word in words:
            # Simple hash-based encoding for residuals
            hash_val = sum(ord(c) for c in word)
            dim = hash_val % self.dim
            position[dim] += (hash_val % 100) / 100.0
        
        return position
    
    # =========================================================================
    # DECODING (Reverse mapping)
    # =========================================================================
    
    def decode(self, position: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Decode position back to primitive names.
        
        Returns list of (primitive_name, similarity) tuples.
        """
        results = []
        
        for name, primitive in self.primitives.items():
            prim_pos = primitive.get_position(self.dim)
            
            # Compute similarity
            norm1 = np.linalg.norm(position)
            norm2 = np.linalg.norm(prim_pos)
            
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(position, prim_pos) / (norm1 * norm2))
            else:
                sim = 0.0
            
            if sim > 0.1:
                results.append((name, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    
    def get_primitive_position(self, name: str) -> Optional[np.ndarray]:
        """Get the position of a primitive by name."""
        if name in self.primitives:
            return self.primitives[name].get_position(self.dim)
        return None
    
    def list_primitives(self) -> Dict[str, Dict[str, Any]]:
        """List all primitives with their metadata."""
        result = {}
        for name, prim in self.primitives.items():
            result[name] = {
                "type": prim.ptype.value,
                "keywords": list(prim.keywords)[:5],
                "dimension": prim.dimension,
                "level": prim.level,
                "position": prim.get_position(self.dim).tolist(),
            }
        return result


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("φ-ENCODER - Semantic Encoding")
    print("=" * 70)
    
    encoder = PhiEncoder()
    
    # Show primitives
    print("\nPrimitives:")
    for ptype in PrimitiveType:
        print(f"\n  {ptype.value.upper()}:")
        for name, prim in encoder.primitives.items():
            if prim.ptype == ptype:
                pos = prim.get_position()
                non_zero = [(i, v) for i, v in enumerate(pos) if abs(v) > 0.01]
                pos_str = ", ".join([f"d{i}={v:.2f}" for i, v in non_zero])
                print(f"    {name:12}: {pos_str}")
    
    # Test encoding
    print("\n" + "=" * 70)
    print("ENCODING TEST")
    print("=" * 70)
    
    test_phrases = [
        "list files in directory",
        "create new folder",
        "delete all files recursively",
        "show network interfaces",
        "search for pattern in files",
    ]
    
    for phrase in test_phrases:
        result = encoder.encode(phrase)
        prims = [p.name for p, _ in result.primitives]
        print(f"\n'{phrase}'")
        print(f"  Primitives: {prims}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Residual: {result.residual_keywords}")
    
    print("\n" + "=" * 70)
    print("Encoder test complete!")
    print("=" * 70)
