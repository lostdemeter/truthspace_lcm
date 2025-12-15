"""
φ-Based Semantic Encoder for TruthSpace

Replaces hash-based encoding with φ-anchored primitive composition.

Key principles:
1. Primitives are anchored at φ-scaled positions in orthogonal dimensions
2. Concepts are composed by summing relevant primitive anchors
3. Residuals capture what primitives don't explain
4. Encoding is BIDIRECTIONAL: text ↔ position ↔ code

The φ structure:
- Each primitive type (action, domain, modifier) gets dedicated dimensions
- Within a type, primitives are positioned at φ^k intervals
- This creates self-similar, well-separated anchor points

From dimensional navigation:
- DOWNCAST: text → primitive indices
- QUANTIZE: store residual
- RECONSTRUCT: indices → position → output
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re

# The golden ratio - our fundamental anchor constant
PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


class PrimitiveType(Enum):
    """Types of semantic primitives."""
    ACTION = "action"       # Verbs: what to do
    DOMAIN = "domain"       # Nouns: what domain
    MODIFIER = "modifier"   # Adjectives/adverbs: how


@dataclass
class Primitive:
    """A semantic primitive - an anchor point in TruthSpace."""
    name: str
    ptype: PrimitiveType
    keywords: Set[str]          # Words that map to this primitive
    opposite: Optional[str] = None  # Name of opposite primitive
    related: List[str] = field(default_factory=list)
    
    # Position is computed, not stored
    _dimension: int = -1        # Which dimension this primitive occupies
    _level: int = 0             # φ^level scaling within dimension


@dataclass 
class SemanticDecomposition:
    """Result of decomposing text into primitives."""
    primitives: List[Tuple[Primitive, float]]  # (primitive, relevance)
    residual_keywords: List[str]  # Keywords not matched to any primitive
    position: np.ndarray         # Computed position in TruthSpace
    confidence: float            # How well primitives cover the input


class PhiEncoder:
    """
    φ-based semantic encoder for TruthSpace.
    
    Encodes natural language to geometric positions using φ-anchored primitives.
    Supports bidirectional mapping: text ↔ position ↔ code.
    """
    
    def __init__(self):
        self.primitives: Dict[str, Primitive] = {}
        self.keyword_to_primitive: Dict[str, str] = {}  # keyword → primitive name
        
        # Dimension allocation
        # Dims 0-3: Actions (4 dimensions for action primitives)
        # Dims 4-6: Domains (3 dimensions for domain primitives)  
        # Dim 7: Modifiers (1 dimension, uses φ^k levels)
        self.dim = 8
        self.action_dims = (0, 4)    # Dimensions 0-3
        self.domain_dims = (4, 7)    # Dimensions 4-6
        self.modifier_dim = 7        # Dimension 7
        
        self._init_primitives()
        self._compute_positions()
    
    def _init_primitives(self):
        """Initialize the primitive vocabulary."""
        
        # =====================================================================
        # ACTION PRIMITIVES (dimensions 0-3)
        # Organized as opposite pairs on same dimension
        # =====================================================================
        
        action_primitives = [
            # Dimension 0: CREATE ↔ DESTROY
            Primitive(
                name="CREATE",
                ptype=PrimitiveType.ACTION,
                keywords={"create", "make", "new", "add", "generate", "init", "initialize", 
                         "build", "mkdir", "touch", "write"},
                opposite="DESTROY",
                related=["WRITE", "EXECUTE"],
            ),
            Primitive(
                name="DESTROY",
                ptype=PrimitiveType.ACTION,
                keywords={"destroy", "delete", "remove", "rm", "rmdir", "kill", "terminate",
                         "clear", "erase", "drop"},
                opposite="CREATE",
                related=["WRITE"],
            ),
            
            # Dimension 1: READ ↔ WRITE  
            Primitive(
                name="READ",
                ptype=PrimitiveType.ACTION,
                keywords={"read", "show", "display", "view", "print", "list", "get", "cat",
                         "head", "tail", "less", "more", "see", "look", "check", "inspect"},
                opposite="WRITE",
                related=["FILE", "DATA"],
            ),
            Primitive(
                name="WRITE",
                ptype=PrimitiveType.ACTION,
                keywords={"write", "modify", "edit", "change", "set", "update", "save",
                         "store", "put", "append", "insert"},
                opposite="READ",
                related=["FILE", "DATA"],
            ),
            
            # Dimension 2: MOVE ↔ STAY (STAY is implicit/zero)
            Primitive(
                name="MOVE",
                ptype=PrimitiveType.ACTION,
                keywords={"move", "copy", "mv", "cp", "transfer", "send", "migrate",
                         "relocate", "shift", "push", "pull"},
                opposite=None,
                related=["FILE"],
            ),
            
            # Dimension 3: CONNECT ↔ DISCONNECT
            Primitive(
                name="CONNECT",
                ptype=PrimitiveType.ACTION,
                keywords={"connect", "link", "join", "attach", "bind", "mount", "ssh",
                         "ping", "curl", "wget", "fetch", "request"},
                opposite="DISCONNECT",
                related=["NETWORK"],
            ),
            Primitive(
                name="DISCONNECT",
                ptype=PrimitiveType.ACTION,
                keywords={"disconnect", "unlink", "detach", "unbind", "umount", "unmount",
                         "close", "hangup"},
                opposite="CONNECT",
                related=["NETWORK"],
            ),
            
            # Additional actions (share dimensions via φ^k levels)
            Primitive(
                name="EXECUTE",
                ptype=PrimitiveType.ACTION,
                keywords={"run", "execute", "start", "launch", "invoke", "call", "spawn",
                         "trigger", "fire", "begin"},
                opposite="STOP",
                related=["PROCESS"],
            ),
            Primitive(
                name="STOP",
                ptype=PrimitiveType.ACTION,
                keywords={"stop", "halt", "pause", "suspend", "freeze", "abort", "cancel",
                         "interrupt", "break"},
                opposite="EXECUTE",
                related=["PROCESS"],
            ),
            Primitive(
                name="SEARCH",
                ptype=PrimitiveType.ACTION,
                keywords={"search", "find", "locate", "grep", "look", "seek", "query",
                         "filter", "match", "scan"},
                opposite=None,
                related=["FILE", "DATA"],
            ),
            Primitive(
                name="TRANSFORM",
                ptype=PrimitiveType.ACTION,
                keywords={"transform", "convert", "parse", "format", "encode", "decode",
                         "compress", "decompress", "zip", "unzip", "tar", "extract",
                         "serialize", "deserialize"},
                opposite=None,
                related=["DATA"],
            ),
            Primitive(
                name="ITERATE",
                ptype=PrimitiveType.ACTION,
                keywords={"iterate", "loop", "repeat", "foreach", "map", "each", "traverse",
                         "walk", "cycle"},
                opposite=None,
                related=["DATA"],
            ),
        ]
        
        # =====================================================================
        # DOMAIN PRIMITIVES (dimensions 4-6)
        # =====================================================================
        
        domain_primitives = [
            # Dimension 4: FILE
            Primitive(
                name="FILE",
                ptype=PrimitiveType.DOMAIN,
                keywords={"file", "files", "directory", "folder", "dir", "path", "disk",
                         "storage", "filesystem", "document", "archive"},
                related=["READ", "WRITE", "MOVE", "CREATE", "DESTROY"],
            ),
            
            # Dimension 5: PROCESS
            Primitive(
                name="PROCESS",
                ptype=PrimitiveType.DOMAIN,
                keywords={"process", "processes", "pid", "job", "task", "daemon", "service",
                         "thread", "worker", "program", "application", "app"},
                related=["EXECUTE", "STOP", "READ"],
            ),
            
            # Dimension 6: NETWORK
            Primitive(
                name="NETWORK",
                ptype=PrimitiveType.DOMAIN,
                keywords={"network", "interface", "ip", "ethernet", "wifi", "socket", "port",
                         "connection", "http", "https", "url", "api", "web", "internet",
                         "remote", "server", "client", "host"},
                related=["CONNECT", "DISCONNECT", "READ"],
            ),
            
            # Additional domains (share dimensions via φ^k levels)
            Primitive(
                name="SYSTEM",
                ptype=PrimitiveType.DOMAIN,
                keywords={"system", "kernel", "os", "boot", "hardware", "memory", "cpu",
                         "ram", "disk", "device", "driver", "module", "config",
                         "environment", "env", "variable", "setting"},
                related=["READ", "WRITE", "EXECUTE"],
            ),
            Primitive(
                name="USER",
                ptype=PrimitiveType.DOMAIN,
                keywords={"user", "users", "account", "permission", "group", "owner",
                         "root", "admin", "sudo", "login", "password", "auth",
                         "authentication", "authorization"},
                related=["READ", "WRITE", "CREATE", "DESTROY"],
            ),
            Primitive(
                name="DATA",
                ptype=PrimitiveType.DOMAIN,
                keywords={"data", "json", "xml", "csv", "text", "string", "content",
                         "output", "input", "result", "response", "payload", "body",
                         "message", "log", "record", "entry"},
                related=["READ", "WRITE", "TRANSFORM", "ITERATE"],
            ),
        ]
        
        # =====================================================================
        # MODIFIER PRIMITIVES (dimension 7)
        # =====================================================================
        
        modifier_primitives = [
            Primitive(
                name="ALL",
                ptype=PrimitiveType.MODIFIER,
                keywords={"all", "every", "each", "entire", "whole", "complete", "full",
                         "total", "global", "everywhere"},
            ),
            Primitive(
                name="RECURSIVE",
                ptype=PrimitiveType.MODIFIER,
                keywords={"recursive", "recursively", "-r", "-R", "deep", "nested",
                         "hierarchical", "tree"},
            ),
            Primitive(
                name="VERBOSE",
                ptype=PrimitiveType.MODIFIER,
                keywords={"verbose", "-v", "detailed", "debug", "trace", "full"},
            ),
            Primitive(
                name="FORCE",
                ptype=PrimitiveType.MODIFIER,
                keywords={"force", "-f", "override", "overwrite", "ignore", "skip"},
            ),
            Primitive(
                name="QUIET",
                ptype=PrimitiveType.MODIFIER,
                keywords={"quiet", "-q", "silent", "suppress", "hide", "mute"},
            ),
        ]
        
        # Register all primitives
        all_primitives = action_primitives + domain_primitives + modifier_primitives
        for prim in all_primitives:
            self.primitives[prim.name] = prim
            for kw in prim.keywords:
                self.keyword_to_primitive[kw.lower()] = prim.name
    
    def _compute_positions(self):
        """Compute φ-based positions for all primitives."""
        
        # Assign dimensions and levels to primitives
        action_assignments = {
            # Dimension 0: CREATE/DESTROY axis
            "CREATE": (0, 1),      # φ^1 in positive direction
            "DESTROY": (0, -1),    # φ^1 in negative direction
            
            # Dimension 1: READ/WRITE axis
            "READ": (1, 1),
            "WRITE": (1, -1),
            
            # Dimension 2: MOVE axis (and related)
            "MOVE": (2, 1),
            "SEARCH": (2, 0.5),    # φ^0.5 - between MOVE and origin
            
            # Dimension 3: CONNECT/DISCONNECT and EXECUTE/STOP
            "CONNECT": (3, 1),
            "DISCONNECT": (3, -1),
            "EXECUTE": (3, 0.618),  # 1/φ level
            "STOP": (3, -0.618),
            
            # TRANSFORM and ITERATE share dimension 2 at different levels
            "TRANSFORM": (2, -0.5),
            "ITERATE": (2, 0.382),  # 1/φ^2
        }
        
        domain_assignments = {
            "FILE": (4, 1),
            "PROCESS": (5, 1),
            "NETWORK": (6, 1),
            "SYSTEM": (4, 0.618),   # Share FILE dimension at 1/φ
            "USER": (5, 0.618),     # Share PROCESS dimension at 1/φ
            "DATA": (6, 0.618),     # Share NETWORK dimension at 1/φ
        }
        
        modifier_assignments = {
            "ALL": (7, 1),
            "RECURSIVE": (7, 0.618),
            "VERBOSE": (7, 0.382),
            "FORCE": (7, -0.618),
            "QUIET": (7, -1),
        }
        
        all_assignments = {**action_assignments, **domain_assignments, **modifier_assignments}
        
        for name, (dim, level) in all_assignments.items():
            if name in self.primitives:
                self.primitives[name]._dimension = dim
                self.primitives[name]._level = level
    
    def get_primitive_position(self, name: str) -> np.ndarray:
        """Get the φ-anchored position for a primitive."""
        if name not in self.primitives:
            return np.zeros(self.dim)
        
        prim = self.primitives[name]
        position = np.zeros(self.dim)
        
        if prim._dimension >= 0:
            # Position = φ^level in the assigned dimension
            if prim._level >= 0:
                position[prim._dimension] = PHI ** prim._level
            else:
                position[prim._dimension] = -(PHI ** abs(prim._level))
        
        return position
    
    def encode(self, text: str) -> SemanticDecomposition:
        """
        Encode natural language text to a position in TruthSpace.
        
        DOWNCAST: text → primitive indices → position
        """
        # Tokenize
        words = self._tokenize(text)
        
        # Match words to primitives
        matched_primitives: Dict[str, float] = {}  # primitive_name → relevance
        residual_keywords = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.keyword_to_primitive:
                prim_name = self.keyword_to_primitive[word_lower]
                matched_primitives[prim_name] = matched_primitives.get(prim_name, 0) + 1
            else:
                residual_keywords.append(word)
        
        # Compute position as weighted sum of primitive positions
        position = np.zeros(self.dim)
        primitives_list = []
        
        for prim_name, count in matched_primitives.items():
            prim = self.primitives[prim_name]
            prim_pos = self.get_primitive_position(prim_name)
            relevance = count / len(words) if words else 0
            position += prim_pos * relevance
            primitives_list.append((prim, relevance))
        
        # Normalize position to unit sphere
        norm = np.linalg.norm(position)
        if norm > 0:
            position = position / norm
        
        # Compute confidence (coverage)
        matched_count = sum(matched_primitives.values())
        confidence = matched_count / len(words) if words else 0
        
        return SemanticDecomposition(
            primitives=primitives_list,
            residual_keywords=residual_keywords,
            position=position,
            confidence=confidence
        )
    
    def decode(self, position: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Decode a position back to primitive names.
        
        RECONSTRUCT: position → primitive indices → names
        """
        results = []
        
        for name, prim in self.primitives.items():
            prim_pos = self.get_primitive_position(name)
            
            # Compute similarity
            prim_norm = np.linalg.norm(prim_pos)
            pos_norm = np.linalg.norm(position)
            
            if prim_norm > 0 and pos_norm > 0:
                similarity = np.dot(position, prim_pos) / (pos_norm * prim_norm)
                if similarity > 0.1:  # Threshold
                    results.append((name, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def position_to_description(self, position: np.ndarray) -> str:
        """
        Convert a position to a natural language description.
        
        This is the REVERSE mapping: position → text
        """
        components = self.decode(position, top_k=5)
        
        if not components:
            return "unknown concept"
        
        # Separate by type
        actions = []
        domains = []
        modifiers = []
        
        for name, sim in components:
            prim = self.primitives[name]
            if prim.ptype == PrimitiveType.ACTION:
                actions.append((name.lower(), sim))
            elif prim.ptype == PrimitiveType.DOMAIN:
                domains.append((name.lower(), sim))
            elif prim.ptype == PrimitiveType.MODIFIER:
                modifiers.append((name.lower(), sim))
        
        # Build description
        parts = []
        
        if modifiers:
            parts.append(modifiers[0][0])
        if actions:
            parts.append(actions[0][0])
        if domains:
            parts.append(domains[0][0])
        
        return " ".join(parts) if parts else "unknown"
    
    def similarity(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute cosine similarity between two positions."""
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(pos1, pos2) / (norm1 * norm2))
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_-]*\b', text.lower())
        
        # Filter stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'can', 'may', 'might', 'must',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'if', 'then', 'so', 'as',
            'this', 'that', 'these', 'those', 'it', 'its',
            'i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'their',
            'what', 'which', 'who', 'how', 'when', 'where', 'why',
            'please', 'want', 'need', 'like', 'using', 'use'
        }
        
        return [w for w in words if w not in stopwords and len(w) > 1]
    
    def add_primitive(self, name: str, ptype: PrimitiveType, keywords: Set[str],
                     dimension: int, level: float, opposite: str = None) -> Primitive:
        """
        Add a new primitive to the encoder.
        
        This allows the mesh to GROW as new concepts are discovered.
        """
        prim = Primitive(
            name=name,
            ptype=ptype,
            keywords=keywords,
            opposite=opposite,
        )
        prim._dimension = dimension
        prim._level = level
        
        self.primitives[name] = prim
        for kw in keywords:
            self.keyword_to_primitive[kw.lower()] = name
        
        return prim
    
    def get_all_primitives(self) -> Dict[str, Dict[str, Any]]:
        """Get all primitives with their positions."""
        result = {}
        for name, prim in self.primitives.items():
            pos = self.get_primitive_position(name)
            result[name] = {
                'type': prim.ptype.value,
                'keywords': list(prim.keywords)[:5],
                'dimension': prim._dimension,
                'level': prim._level,
                'position': pos.tolist(),
            }
        return result


def demonstrate():
    """Demonstrate the φ-encoder."""
    print("=" * 70)
    print("φ-BASED SEMANTIC ENCODER")
    print("=" * 70)
    
    encoder = PhiEncoder()
    
    # Show primitive positions
    print("\nPrimitive Anchors (φ-based positions):")
    print("-" * 60)
    
    for ptype in PrimitiveType:
        print(f"\n{ptype.value.upper()}S:")
        for name, prim in encoder.primitives.items():
            if prim.ptype == ptype:
                pos = encoder.get_primitive_position(name)
                non_zero = [(i, v) for i, v in enumerate(pos) if abs(v) > 0.01]
                pos_str = ", ".join([f"d{i}={v:.3f}" for i, v in non_zero])
                print(f"  {name:12}: {pos_str}")
    
    # Test encoding
    print("\n" + "=" * 70)
    print("ENCODING TEST: Natural Language → Position")
    print("=" * 70)
    
    test_phrases = [
        "show network interfaces",
        "create a new directory",
        "delete all files recursively",
        "read the json file",
        "connect to remote server",
        "list running processes",
        "copy file to backup",
        "search for text in files",
    ]
    
    for phrase in test_phrases:
        result = encoder.encode(phrase)
        decoded = encoder.decode(result.position, top_k=3)
        description = encoder.position_to_description(result.position)
        
        print(f"\n'{phrase}'")
        print(f"  Primitives: {[p.name for p, r in result.primitives]}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Decoded: {[f'{n}({s:.2f})' for n, s in decoded]}")
        print(f"  Description: {description}")
    
    # Test similarity
    print("\n" + "=" * 70)
    print("SIMILARITY TEST")
    print("=" * 70)
    
    pairs = [
        ("show files", "list directory"),           # Should be similar
        ("create folder", "make directory"),        # Should be very similar
        ("delete file", "remove file"),             # Should be very similar
        ("show files", "connect network"),          # Should be different
        ("create file", "destroy file"),            # Should be opposite
    ]
    
    for phrase1, phrase2 in pairs:
        pos1 = encoder.encode(phrase1).position
        pos2 = encoder.encode(phrase2).position
        sim = encoder.similarity(pos1, pos2)
        print(f"  '{phrase1}' ↔ '{phrase2}': {sim:.3f}")
    
    print("\n" + "=" * 70)
    print("φ-encoder demonstration complete!")


if __name__ == "__main__":
    demonstrate()
