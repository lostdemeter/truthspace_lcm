"""
φ-Encoder v2: Plastic-Primary 12D Semantic Encoding
====================================================

Improvements over v1:
1. PLASTIC-PRIMARY: Uses plastic constant (ρ ≈ 1.3247) as primary scaling
   - Slower growth = finer discrimination
   - Cubic recurrence = deeper hierarchy capture

2. 12D ENCODING: Expanded from 8D to 12D
   - Dims 0-3: Actions (existence, information, spatial, interaction)
   - Dims 4-7: Domains (file, process, network, system)
   - Dims 8-11: Relations (temporal, causal, conditional, comparative)

3. DUAL-CONSTANT: Uses both plastic and golden for different purposes
   - Plastic for level scaling (finer granularity)
   - Golden for cross-dimension interference (harmonic relationships)

Design principles:
- Primitives ARE knowledge entries (type=PRIMITIVE)
- The encoder is just math - no domain logic
- 12D aligns with the 12D clock for full phase coverage
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONSTANTS
# =============================================================================

# Primary constant: Plastic (slower growth, finer discrimination)
RHO = 1.324717957244746  # Plastic constant: x³ = x + 1
LOG_RHO = np.log(RHO)

# Secondary constant: Golden (harmonic relationships)
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio: x² = x + 1
LOG_PHI = np.log(PHI)

# Encoding dimension
ENCODING_DIM = 12


# =============================================================================
# PRIMITIVE TYPES
# =============================================================================

class PrimitiveType(Enum):
    """Types of semantic primitives."""
    ACTION = "action"       # Verbs: what to do
    DOMAIN = "domain"       # Nouns: what domain
    MODIFIER = "modifier"   # Adjectives/adverbs: how
    RELATION = "relation"   # Connectives: relationships


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
    level: int = 0          # ρ^level scaling within dimension
    opposite: Optional[str] = None
    
    def get_position(self, dim: int = ENCODING_DIM, use_plastic: bool = True) -> np.ndarray:
        """
        Compute the position for this primitive.
        
        Args:
            dim: Number of dimensions
            use_plastic: If True, use plastic constant; else golden ratio
        """
        position = np.zeros(dim)
        
        # Base value at constant^level
        constant = RHO if use_plastic else PHI
        value = constant ** self.level
        
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
# BOOTSTRAP PRIMITIVES (12D)
# =============================================================================

BOOTSTRAP_PRIMITIVES_12D = [
    # ==========================================================================
    # ACTIONS (dims 0-3): What to do
    # ==========================================================================
    
    # Dim 0: EXISTENCE AXIS (create ↔ destroy)
    Primitive("CREATE", PrimitiveType.ACTION, 
              {"create", "make", "new", "add", "build", "mkdir", "touch", "generate"}, 
              0, 0),
    Primitive("DESTROY", PrimitiveType.ACTION, 
              {"destroy", "delete", "remove", "rm", "kill", "erase", "drop"}, 
              0, 1, opposite="CREATE"),
    
    # Dim 1: INFORMATION FLOW AXIS (read ↔ write)
    Primitive("READ", PrimitiveType.ACTION, 
              {"read", "show", "display", "view", "list", "cat", "get", "print", "output"}, 
              1, 0),
    Primitive("WRITE", PrimitiveType.ACTION, 
              {"write", "modify", "edit", "save", "update", "set", "put", "store"}, 
              1, 1, opposite="READ"),
    
    # Dim 2: SPATIAL AXIS (move/copy, search/find)
    Primitive("MOVE", PrimitiveType.ACTION, 
              {"move", "copy", "mv", "cp", "transfer", "backup", "clone", "duplicate"}, 
              2, 0),
    Primitive("SEARCH", PrimitiveType.ACTION, 
              {"search", "find", "grep", "locate", "lookup", "query", "filter"}, 
              2, 1),
    Primitive("TRANSFORM", PrimitiveType.ACTION, 
              {"compress", "decompress", "zip", "unzip", "tar", "archive", "extract", "pack", "unpack", "encode", "decode"}, 
              2, 2),
    
    # Dim 3: INTERACTION AXIS (connect, execute)
    Primitive("CONNECT", PrimitiveType.ACTION, 
              {"connect", "link", "ssh", "curl", "fetch", "download", "upload", "sync"}, 
              3, 0),
    Primitive("EXECUTE", PrimitiveType.ACTION, 
              {"run", "execute", "start", "launch", "invoke", "call", "trigger"}, 
              3, 1),
    
    # ==========================================================================
    # DOMAINS (dims 4-7): What type of thing
    # ==========================================================================
    
    # Dim 4: FILE DOMAIN
    Primitive("FILE", PrimitiveType.DOMAIN, 
              {"file", "files", "directory", "folder", "path", "document", "archive"}, 
              4, 0),
    Primitive("SYSTEM", PrimitiveType.DOMAIN, 
              {"system", "kernel", "os", "memory", "cpu", "hardware", "device"}, 
              4, 1),
    
    # Dim 5: PROCESS DOMAIN
    Primitive("PROCESS", PrimitiveType.DOMAIN, 
              {"process", "pid", "job", "task", "daemon", "service", "thread"}, 
              5, 0),
    Primitive("DATA", PrimitiveType.DOMAIN, 
              {"data", "json", "text", "content", "output", "input", "stream"}, 
              5, 1),
    
    # Dim 6: NETWORK DOMAIN
    Primitive("NETWORK", PrimitiveType.DOMAIN, 
              {"network", "interface", "ip", "port", "socket", "connection", "host"}, 
              6, 0),
    Primitive("USER", PrimitiveType.DOMAIN, 
              {"user", "account", "permission", "owner", "group", "access", "auth", "authenticate", "login", "credential"}, 
              6, 1),
    
    # Dim 7: MODIFIERS
    Primitive("ALL", PrimitiveType.MODIFIER, 
              {"all", "every", "entire", "complete", "full", "whole"}, 
              7, 0),
    Primitive("RECURSIVE", PrimitiveType.MODIFIER, 
              {"recursive", "-r", "deep", "nested", "tree", "hierarchy"}, 
              7, 1),
    Primitive("FORCE", PrimitiveType.MODIFIER, 
              {"force", "-f", "override", "ignore", "skip"}, 
              7, 2),
    Primitive("VERBOSE", PrimitiveType.MODIFIER, 
              {"verbose", "-v", "detailed", "debug", "trace"}, 
              7, 3),
    
    # ==========================================================================
    # RELATIONS (dims 8-11): How things relate - NEW IN V2
    # ==========================================================================
    
    # Dim 8: TEMPORAL RELATIONS
    Primitive("BEFORE", PrimitiveType.RELATION, 
              {"before", "prior", "previous", "earlier", "first", "initially"}, 
              8, 0),
    Primitive("AFTER", PrimitiveType.RELATION, 
              {"after", "then", "next", "later", "finally", "subsequently"}, 
              8, 1, opposite="BEFORE"),
    Primitive("DURING", PrimitiveType.RELATION, 
              {"during", "while", "when", "as", "simultaneously"}, 
              8, 2),
    
    # Dim 9: CAUSAL RELATIONS
    Primitive("CAUSE", PrimitiveType.RELATION, 
              {"because", "since", "cause", "reason", "due", "therefore"}, 
              9, 0),
    Primitive("EFFECT", PrimitiveType.RELATION, 
              {"result", "effect", "outcome", "consequence", "leads", "produces"}, 
              9, 1, opposite="CAUSE"),
    
    # Dim 10: CONDITIONAL RELATIONS
    Primitive("IF", PrimitiveType.RELATION, 
              {"if", "when", "provided", "assuming", "given", "unless", "retry", "attempt", "try"}, 
              10, 0),
    Primitive("ELSE", PrimitiveType.RELATION, 
              {"else", "otherwise", "alternatively", "instead", "fallback", "fail"}, 
              10, 1, opposite="IF"),
    
    # Dim 11: COMPARATIVE RELATIONS
    Primitive("MORE", PrimitiveType.RELATION, 
              {"more", "greater", "larger", "higher", "increase", "above"}, 
              11, 0),
    Primitive("LESS", PrimitiveType.RELATION, 
              {"less", "fewer", "smaller", "lower", "decrease", "below"}, 
              11, 1, opposite="MORE"),
    Primitive("EQUAL", PrimitiveType.RELATION, 
              {"equal", "same", "identical", "match", "equivalent"}, 
              11, 2),
]


# =============================================================================
# ENCODER V2
# =============================================================================

class PlasticEncoder:
    """
    Plastic-primary 12D semantic encoder.
    
    Uses plastic constant (ρ ≈ 1.3247) for level scaling, providing
    finer discrimination than golden ratio. Expanded to 12D to capture
    temporal, causal, conditional, and comparative relationships.
    """
    
    def __init__(self, primitives: List[Primitive] = None, dim: int = ENCODING_DIM):
        """
        Initialize encoder.
        
        Args:
            primitives: List of primitives to use. If None, uses bootstrap set.
            dim: Number of dimensions (default: 12)
        """
        self.dim = dim
        self.primitives: Dict[str, Primitive] = {}
        self.keyword_to_primitive: Dict[str, str] = {}
        
        # Load primitives
        if primitives:
            for p in primitives:
                self._register_primitive(p)
        else:
            for p in BOOTSTRAP_PRIMITIVES_12D:
                self._register_primitive(p)
    
    def _register_primitive(self, primitive: Primitive):
        """Register a primitive and its keywords."""
        self.primitives[primitive.name] = primitive
        for keyword in primitive.keywords:
            self.keyword_to_primitive[keyword.lower()] = primitive.name
    
    @classmethod
    def from_truthspace(cls, truthspace) -> 'PlasticEncoder':
        """Create encoder with primitives loaded from TruthSpace."""
        from truthspace_lcm.core.truthspace import EntryType
        
        primitives = []
        
        try:
            entries = truthspace.list_by_type(EntryType.PRIMITIVE)
            
            for entry in entries:
                meta = entry.metadata
                ptype_str = meta.get("primitive_type", "action")
                
                # Handle new RELATION type
                try:
                    ptype = PrimitiveType(ptype_str)
                except ValueError:
                    ptype = PrimitiveType.ACTION
                
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
            primitives = BOOTSTRAP_PRIMITIVES_12D
        
        return cls(primitives)
    
    # =========================================================================
    # ENCODING
    # =========================================================================
    
    def encode(self, text: str) -> SemanticDecomposition:
        """
        Encode text to semantic decomposition using plastic-primary scaling.
        
        Returns position in 12D ρ-space based on primitive composition.
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
        
        # Compute position by summing primitive positions (using plastic constant)
        position = np.zeros(self.dim)
        
        for primitive, relevance in matched_primitives:
            prim_pos = primitive.get_position(self.dim, use_plastic=True)
            position += prim_pos * relevance
        
        # Add residual contribution with golden-ratio encoding
        # (different constant for unknown words creates interference pattern)
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
        words = re.findall(r'-[a-zA-Z]|\b\w+\b', text.lower())
        return words
    
    def _encode_residual(self, words: List[str]) -> np.ndarray:
        """
        Encode residual words using golden ratio (different from plastic).
        
        This creates an interference pattern that distinguishes known
        from unknown concepts.
        """
        position = np.zeros(self.dim)
        
        for word in words:
            # Hash-based encoding using golden ratio
            hash_val = sum(ord(c) * (i + 1) for i, c in enumerate(word))
            dim = hash_val % self.dim
            
            # Use golden ratio for residual scaling
            level = (hash_val // self.dim) % 4
            value = PHI ** level * 0.1
            
            position[dim] += value
        
        return position
    
    # =========================================================================
    # DECODING
    # =========================================================================
    
    def decode(self, position: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Decode position back to primitive names."""
        results = []
        
        for name, primitive in self.primitives.items():
            prim_pos = primitive.get_position(self.dim, use_plastic=True)
            
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
    # DIMENSION ANALYSIS
    # =========================================================================
    
    def get_dimension_info(self) -> Dict[int, Dict]:
        """Get information about what each dimension represents."""
        dim_info = {
            0: {"name": "EXISTENCE", "axis": "create ↔ destroy", "type": "action"},
            1: {"name": "INFORMATION", "axis": "read ↔ write", "type": "action"},
            2: {"name": "SPATIAL", "axis": "move ↔ search", "type": "action"},
            3: {"name": "INTERACTION", "axis": "connect ↔ execute", "type": "action"},
            4: {"name": "FILE_SYSTEM", "axis": "file ↔ system", "type": "domain"},
            5: {"name": "PROCESS_DATA", "axis": "process ↔ data", "type": "domain"},
            6: {"name": "NETWORK_USER", "axis": "network ↔ user", "type": "domain"},
            7: {"name": "MODIFIERS", "axis": "all/recursive/force/verbose", "type": "modifier"},
            8: {"name": "TEMPORAL", "axis": "before ↔ after", "type": "relation"},
            9: {"name": "CAUSAL", "axis": "cause ↔ effect", "type": "relation"},
            10: {"name": "CONDITIONAL", "axis": "if ↔ else", "type": "relation"},
            11: {"name": "COMPARATIVE", "axis": "more ↔ less", "type": "relation"},
        }
        
        # Add primitives on each dimension
        for dim in range(self.dim):
            dim_info[dim]["primitives"] = [
                p.name for p in self.primitives.values() 
                if p.dimension == dim
            ]
        
        return dim_info
    
    def find_dimension_for_concept(self, text: str) -> Tuple[int, float]:
        """
        Find which dimension a concept belongs to.
        
        Returns (dimension, confidence).
        """
        decomp = self.encode(text)
        
        # Find dimension with highest absolute value
        abs_pos = np.abs(decomp.position)
        best_dim = int(np.argmax(abs_pos))
        confidence = float(abs_pos[best_dim])
        
        return best_dim, confidence
    
    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    
    def get_primitive_position(self, name: str) -> Optional[np.ndarray]:
        """Get the position of a primitive by name."""
        if name in self.primitives:
            return self.primitives[name].get_position(self.dim, use_plastic=True)
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
                "opposite": prim.opposite,
                "position": prim.get_position(self.dim, use_plastic=True).tolist(),
            }
        return result
    
    def get_constants(self) -> Dict[str, float]:
        """Get the mathematical constants used."""
        return {
            "plastic (primary)": RHO,
            "golden (secondary)": PHI,
            "dimensions": self.dim,
        }


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_encoders():
    """Compare v1 (phi-primary 8D) vs v2 (plastic-primary 12D)."""
    from truthspace_lcm.core.encoder import PhiEncoder
    
    v1 = PhiEncoder()
    v2 = PlasticEncoder()
    
    test_phrases = [
        "create new file",
        "delete all files recursively",
        "copy files before backup",
        "if error then retry",
        "search for larger files",
    ]
    
    print("=" * 70)
    print("ENCODER COMPARISON: v1 (φ-8D) vs v2 (ρ-12D)")
    print("=" * 70)
    
    for phrase in test_phrases:
        print(f"\n'{phrase}'")
        
        r1 = v1.encode(phrase)
        r2 = v2.encode(phrase)
        
        print(f"  v1 (φ-8D):  {[p.name for p, _ in r1.primitives]}")
        print(f"  v2 (ρ-12D): {[p.name for p, _ in r2.primitives]}")
        
        # Show new dimensions captured in v2
        new_dims = [p.name for p, _ in r2.primitives if p.dimension >= 8]
        if new_dims:
            print(f"  NEW in v2:  {new_dims} (dims 8-11)")


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PLASTIC ENCODER v2 - 12D Semantic Encoding")
    print("=" * 70)
    
    encoder = PlasticEncoder()
    
    # Show constants
    print("\nConstants:")
    for name, value in encoder.get_constants().items():
        print(f"  {name}: {value}")
    
    # Show dimension structure
    print("\n" + "-" * 70)
    print("DIMENSION STRUCTURE")
    print("-" * 70)
    
    dim_info = encoder.get_dimension_info()
    for dim, info in dim_info.items():
        prims = ", ".join(info["primitives"][:3])
        if len(info["primitives"]) > 3:
            prims += "..."
        print(f"  Dim {dim:2d}: {info['name']:12} ({info['axis']}) [{prims}]")
    
    # Test encoding
    print("\n" + "-" * 70)
    print("ENCODING TEST")
    print("-" * 70)
    
    test_phrases = [
        "list files in directory",
        "create new folder",
        "delete all files recursively",
        "copy files before backup",
        "if error then retry",
        "search for larger files",
        "connect after authentication",
    ]
    
    for phrase in test_phrases:
        result = encoder.encode(phrase)
        prims = [p.name for p, _ in result.primitives]
        dim, conf = encoder.find_dimension_for_concept(phrase)
        
        print(f"\n'{phrase}'")
        print(f"  Primitives: {prims}")
        print(f"  Primary dimension: {dim} ({dim_info[dim]['name']})")
        print(f"  Confidence: {result.confidence:.2f}")
    
    # Compare with v1
    print("\n" + "-" * 70)
    compare_encoders()
    
    print("\n" + "=" * 70)
    print("Plastic Encoder v2 ready!")
    print("=" * 70)
