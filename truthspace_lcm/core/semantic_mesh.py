"""
Semantic Mesh: Concept Anchors for TruthSpace Navigation

This is experimental code exploring how to create a semantic mesh
analogous to the φ-Lens mesh for numerical values.

Key ideas:
1. Define fundamental concept anchors (like φ^(-k) for numbers)
2. Position knowledge relative to these anchors
3. Use the mesh for O(1) semantic navigation

Philosophy:
- Mathematical constants anchor numerical truth space
- Semantic primitives anchor conceptual truth space
- The mesh IS the knowledge structure
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Mathematical constants (from knowledge_manager)
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e
SQRT2 = np.sqrt(2)


class SemanticPrimitive(Enum):
    """
    Fundamental semantic anchors - the "mathematical constants" of meaning.
    
    These are the primitives from which all other concepts are composed.
    Like φ^(-k) for weights, these serve as the mesh anchors.
    
    This is speculative - we're discovering what these should be.
    """
    # Actions (verbs)
    CREATE = "create"      # Bring into existence
    DESTROY = "destroy"    # Remove from existence
    READ = "read"          # Observe without changing
    WRITE = "write"        # Modify/change
    MOVE = "move"          # Change location
    CONNECT = "connect"    # Establish relationship
    EXECUTE = "execute"    # Run/perform
    
    # Domains (nouns)
    FILE = "file"          # Persistent data
    PROCESS = "process"    # Running computation
    NETWORK = "network"    # Communication
    SYSTEM = "system"      # The whole/environment
    USER = "user"          # Human actor
    DATA = "data"          # Information
    
    # Modifiers
    ALL = "all"            # Universal scope
    ONE = "one"            # Singular scope
    RECURSIVE = "recursive"  # Self-referential


@dataclass
class ConceptAnchor:
    """
    A concept anchor in the semantic mesh.
    
    Like φ^(-k) in the numerical mesh, these are fixed points
    that other concepts are positioned relative to.
    """
    primitive: SemanticPrimitive
    position: np.ndarray  # Position in TruthSpace
    
    # Relationships to other anchors
    related: List[SemanticPrimitive] = None
    opposite: SemanticPrimitive = None


class SemanticMesh:
    """
    The semantic mesh - precomputed structure for concept navigation.
    
    Analogous to the φ-LUT in dimensional navigation.
    """
    
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.anchors: Dict[SemanticPrimitive, ConceptAnchor] = {}
        self._build_mesh()
    
    def _build_mesh(self):
        """
        Build the semantic mesh.
        
        Key insight: We want related concepts to be geometrically close,
        and opposite concepts to be geometrically distant.
        
        Using mathematical constants to create well-distributed positions.
        """
        # Action primitives - arranged on a "verb axis"
        # CREATE and DESTROY are opposites
        # READ and WRITE are opposites
        action_anchors = {
            SemanticPrimitive.CREATE:  self._make_position([PHI, 0, 0, 0]),
            SemanticPrimitive.DESTROY: self._make_position([-PHI, 0, 0, 0]),  # Opposite
            SemanticPrimitive.READ:    self._make_position([0, PHI, 0, 0]),
            SemanticPrimitive.WRITE:   self._make_position([0, -PHI, 0, 0]),  # Opposite
            SemanticPrimitive.MOVE:    self._make_position([0, 0, PHI, 0]),
            SemanticPrimitive.CONNECT: self._make_position([0, 0, 0, PHI]),
            SemanticPrimitive.EXECUTE: self._make_position([PHI/2, PHI/2, 0, 0]),  # Between create and read
        }
        
        # Domain primitives - arranged on a "noun axis" (dimensions 4-7)
        domain_anchors = {
            SemanticPrimitive.FILE:    self._make_position([0, 0, 0, 0, PHI, 0, 0, 0]),
            SemanticPrimitive.PROCESS: self._make_position([0, 0, 0, 0, 0, PHI, 0, 0]),
            SemanticPrimitive.NETWORK: self._make_position([0, 0, 0, 0, 0, 0, PHI, 0]),
            SemanticPrimitive.SYSTEM:  self._make_position([0, 0, 0, 0, 0, 0, 0, PHI]),
            SemanticPrimitive.USER:    self._make_position([0, 0, 0, 0, PHI/2, 0, 0, PHI/2]),
            SemanticPrimitive.DATA:    self._make_position([0, 0, 0, 0, PHI/2, PHI/2, 0, 0]),
        }
        
        # Modifier primitives - scale factors
        modifier_anchors = {
            SemanticPrimitive.ALL:       self._make_position([0]*8, scale=PHI),
            SemanticPrimitive.ONE:       self._make_position([0]*8, scale=1/PHI),
            SemanticPrimitive.RECURSIVE: self._make_position([0]*8, scale=PHI**2),
        }
        
        # Build anchors with relationships
        for prim, pos in {**action_anchors, **domain_anchors, **modifier_anchors}.items():
            self.anchors[prim] = ConceptAnchor(
                primitive=prim,
                position=pos,
                related=self._get_related(prim),
                opposite=self._get_opposite(prim)
            )
    
    def _make_position(self, values: List[float], scale: float = 1.0) -> np.ndarray:
        """Create a position vector, padding to full dimensionality."""
        pos = np.zeros(self.dim)
        for i, v in enumerate(values[:self.dim]):
            pos[i] = v * scale
        return pos
    
    def _get_related(self, prim: SemanticPrimitive) -> List[SemanticPrimitive]:
        """Get semantically related primitives."""
        relations = {
            SemanticPrimitive.CREATE: [SemanticPrimitive.WRITE, SemanticPrimitive.EXECUTE],
            SemanticPrimitive.DESTROY: [SemanticPrimitive.WRITE],
            SemanticPrimitive.READ: [SemanticPrimitive.FILE, SemanticPrimitive.DATA],
            SemanticPrimitive.WRITE: [SemanticPrimitive.FILE, SemanticPrimitive.DATA],
            SemanticPrimitive.MOVE: [SemanticPrimitive.FILE],
            SemanticPrimitive.CONNECT: [SemanticPrimitive.NETWORK],
            SemanticPrimitive.EXECUTE: [SemanticPrimitive.PROCESS],
            SemanticPrimitive.FILE: [SemanticPrimitive.READ, SemanticPrimitive.WRITE, SemanticPrimitive.DATA],
            SemanticPrimitive.PROCESS: [SemanticPrimitive.EXECUTE, SemanticPrimitive.SYSTEM],
            SemanticPrimitive.NETWORK: [SemanticPrimitive.CONNECT, SemanticPrimitive.SYSTEM],
            SemanticPrimitive.SYSTEM: [SemanticPrimitive.PROCESS, SemanticPrimitive.NETWORK],
        }
        return relations.get(prim, [])
    
    def _get_opposite(self, prim: SemanticPrimitive) -> Optional[SemanticPrimitive]:
        """Get the semantic opposite."""
        opposites = {
            SemanticPrimitive.CREATE: SemanticPrimitive.DESTROY,
            SemanticPrimitive.DESTROY: SemanticPrimitive.CREATE,
            SemanticPrimitive.READ: SemanticPrimitive.WRITE,
            SemanticPrimitive.WRITE: SemanticPrimitive.READ,
        }
        return opposites.get(prim)
    
    def encode_concept(self, keywords: List[str]) -> np.ndarray:
        """
        Encode a concept as a position in TruthSpace.
        
        The position is computed as a weighted combination of
        the nearest semantic anchors - like φ-Lens but for meaning.
        """
        position = np.zeros(self.dim)
        
        # Map keywords to primitives
        keyword_to_primitive = {
            # Actions
            'create': SemanticPrimitive.CREATE, 'make': SemanticPrimitive.CREATE,
            'new': SemanticPrimitive.CREATE, 'add': SemanticPrimitive.CREATE,
            'delete': SemanticPrimitive.DESTROY, 'remove': SemanticPrimitive.DESTROY,
            'rm': SemanticPrimitive.DESTROY, 'destroy': SemanticPrimitive.DESTROY,
            'read': SemanticPrimitive.READ, 'show': SemanticPrimitive.READ,
            'display': SemanticPrimitive.READ, 'view': SemanticPrimitive.READ,
            'list': SemanticPrimitive.READ, 'cat': SemanticPrimitive.READ,
            'get': SemanticPrimitive.READ, 'print': SemanticPrimitive.READ,
            'write': SemanticPrimitive.WRITE, 'modify': SemanticPrimitive.WRITE,
            'edit': SemanticPrimitive.WRITE, 'change': SemanticPrimitive.WRITE,
            'set': SemanticPrimitive.WRITE, 'update': SemanticPrimitive.WRITE,
            'move': SemanticPrimitive.MOVE, 'copy': SemanticPrimitive.MOVE,
            'mv': SemanticPrimitive.MOVE, 'cp': SemanticPrimitive.MOVE,
            'connect': SemanticPrimitive.CONNECT, 'ping': SemanticPrimitive.CONNECT,
            'ssh': SemanticPrimitive.CONNECT, 'link': SemanticPrimitive.CONNECT,
            'run': SemanticPrimitive.EXECUTE, 'execute': SemanticPrimitive.EXECUTE,
            'start': SemanticPrimitive.EXECUTE, 'launch': SemanticPrimitive.EXECUTE,
            
            # Domains
            'file': SemanticPrimitive.FILE, 'files': SemanticPrimitive.FILE,
            'directory': SemanticPrimitive.FILE, 'folder': SemanticPrimitive.FILE,
            'path': SemanticPrimitive.FILE, 'disk': SemanticPrimitive.FILE,
            'process': SemanticPrimitive.PROCESS, 'pid': SemanticPrimitive.PROCESS,
            'job': SemanticPrimitive.PROCESS, 'task': SemanticPrimitive.PROCESS,
            'daemon': SemanticPrimitive.PROCESS, 'service': SemanticPrimitive.PROCESS,
            'network': SemanticPrimitive.NETWORK, 'interface': SemanticPrimitive.NETWORK,
            'ip': SemanticPrimitive.NETWORK, 'ethernet': SemanticPrimitive.NETWORK,
            'socket': SemanticPrimitive.NETWORK, 'port': SemanticPrimitive.NETWORK,
            'system': SemanticPrimitive.SYSTEM, 'kernel': SemanticPrimitive.SYSTEM,
            'os': SemanticPrimitive.SYSTEM, 'boot': SemanticPrimitive.SYSTEM,
            'memory': SemanticPrimitive.SYSTEM, 'cpu': SemanticPrimitive.SYSTEM,
            'user': SemanticPrimitive.USER, 'account': SemanticPrimitive.USER,
            'permission': SemanticPrimitive.USER, 'owner': SemanticPrimitive.USER,
            'data': SemanticPrimitive.DATA, 'json': SemanticPrimitive.DATA,
            'text': SemanticPrimitive.DATA, 'content': SemanticPrimitive.DATA,
            
            # Modifiers
            'all': SemanticPrimitive.ALL, 'every': SemanticPrimitive.ALL,
            'recursive': SemanticPrimitive.RECURSIVE, '-r': SemanticPrimitive.RECURSIVE,
        }
        
        # Accumulate positions from matched primitives
        matched = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in keyword_to_primitive:
                prim = keyword_to_primitive[kw_lower]
                anchor = self.anchors.get(prim)
                if anchor:
                    position += anchor.position
                    matched.append(prim)
        
        # Normalize
        norm = np.linalg.norm(position)
        if norm > 0:
            position = position / norm
        
        return position, matched
    
    def find_nearest_anchor(self, position: np.ndarray) -> Tuple[SemanticPrimitive, float]:
        """Find the nearest semantic anchor to a position."""
        best_sim = -1
        best_prim = None
        
        for prim, anchor in self.anchors.items():
            sim = np.dot(position, anchor.position) / (
                np.linalg.norm(position) * np.linalg.norm(anchor.position) + 1e-10
            )
            if sim > best_sim:
                best_sim = sim
                best_prim = prim
        
        return best_prim, best_sim
    
    def decompose(self, position: np.ndarray) -> List[Tuple[SemanticPrimitive, float]]:
        """
        Decompose a position into its primitive components.
        
        Like finding the φ^(-k) components of a weight.
        """
        components = []
        
        for prim, anchor in self.anchors.items():
            # Project position onto anchor
            anchor_norm = np.linalg.norm(anchor.position)
            if anchor_norm > 0:
                projection = np.dot(position, anchor.position) / (anchor_norm ** 2)
                if abs(projection) > 0.1:  # Threshold for significance
                    components.append((prim, projection))
        
        # Sort by absolute contribution
        components.sort(key=lambda x: -abs(x[1]))
        return components


def demonstrate():
    """Demonstrate the semantic mesh."""
    print("=" * 70)
    print("SEMANTIC MESH - Concept Anchors for TruthSpace")
    print("=" * 70)
    print()
    
    mesh = SemanticMesh()
    
    # Test encoding various concepts
    test_concepts = [
        ["show", "network", "interface"],      # ifconfig
        ["create", "directory"],               # mkdir
        ["delete", "file"],                    # rm
        ["show", "process"],                   # ps
        ["read", "file", "content"],           # cat
        ["connect", "network", "remote"],      # ssh
        ["show", "system", "kernel"],          # dmesg
    ]
    
    print("Concept Encoding:")
    print("-" * 60)
    
    for keywords in test_concepts:
        position, matched = mesh.encode_concept(keywords)
        components = mesh.decompose(position)
        
        print(f"\nKeywords: {keywords}")
        print(f"  Matched primitives: {[p.value for p in matched]}")
        print(f"  Top components: {[(p.value, f'{w:.2f}') for p, w in components[:3]]}")
    
    # Test similarity between related concepts
    print("\n" + "-" * 60)
    print("Semantic Similarity (should be high for related concepts):")
    print("-" * 60)
    
    pairs = [
        (["show", "network"], ["ping", "network"]),      # Both network-read
        (["show", "network"], ["create", "directory"]),  # Different domains
        (["create", "file"], ["delete", "file"]),        # Same domain, opposite action
        (["show", "file"], ["read", "data"]),            # Related
    ]
    
    for kw1, kw2 in pairs:
        pos1, _ = mesh.encode_concept(kw1)
        pos2, _ = mesh.encode_concept(kw2)
        
        sim = np.dot(pos1, pos2) / (np.linalg.norm(pos1) * np.linalg.norm(pos2) + 1e-10)
        print(f"  {kw1} <-> {kw2}: {sim:.3f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate()
