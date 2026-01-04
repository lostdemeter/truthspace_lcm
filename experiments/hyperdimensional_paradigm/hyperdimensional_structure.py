"""
HyperdimensionalStructure - The Data Structure

This is the pure data structure side of the hyperdimensional paradigm.
It manages positions in N-dimensional space with NO knowledge of what
those positions represent or how they'll be used.

Responsibilities:
- Store positions in N-dimensional space
- Add/remove/update positions
- Query by proximity (nearest neighbors)
- Maintain structural stability (reprojection)
- Serialize/deserialize the structure

NOT responsible for:
- What the positions mean
- How to use the positions
- Domain-specific logic (chat, images, etc.)

The structure is domain-agnostic. The same structure could be used for:
- Text/chat (words as positions)
- Images (pixels/features as positions)
- Audio (frequencies as positions)
- Any domain that can be mapped to positions

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from datetime import datetime


# The critical line - positions with magnitude >= this persist
CRITICAL_LINE = 0.5


@dataclass
class Node:
    """
    A node in the hyperdimensional structure.
    
    Contains only:
    - id: Unique identifier
    - position: N-dimensional coordinates
    - data: Opaque payload (the structure doesn't interpret this)
    - created/modified: Timestamps
    """
    id: str
    position: np.ndarray
    data: Any = None  # Opaque - structure doesn't care what this is
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def magnitude(self) -> float:
        """Distance from origin."""
        return float(np.linalg.norm(self.position))
    
    @property
    def persists(self) -> bool:
        """Whether this node is above the critical line."""
        return self.magnitude >= CRITICAL_LINE
    
    @property
    def normalized(self) -> np.ndarray:
        """Unit vector in the direction of position."""
        mag = self.magnitude
        if mag < 1e-10:
            return self.position
        return self.position / mag
    
    def move_toward(self, target: np.ndarray, strength: float = 0.1) -> None:
        """Move position toward target (learning/attraction)."""
        direction = target - self.position
        self.position = self.position + direction * strength
        self.modified = datetime.now().isoformat()
    
    def move_away(self, target: np.ndarray, strength: float = 0.05) -> None:
        """Move position away from target (unlearning/repulsion)."""
        direction = self.position - target
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            self.position = self.position + (direction / norm) * strength
            self.modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'data': self.data,
            'created': self.created,
            'modified': self.modified,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Node':
        return cls(
            id=d['id'],
            position=np.array(d['position']),
            data=d.get('data'),
            created=d.get('created', datetime.now().isoformat()),
            modified=d.get('modified', datetime.now().isoformat()),
        )


class HyperdimensionalStructure:
    """
    A domain-agnostic hyperdimensional data structure.
    
    This is the pure data structure - it knows nothing about what
    the positions represent. It only knows how to:
    
    1. Store nodes with positions
    2. Find nodes by proximity
    3. Update positions (learning)
    4. Maintain stability (reprojection)
    5. Serialize/deserialize
    
    The dimensionality is flexible - you can use any number of dimensions.
    More dimensions = more capacity for distinctions.
    """
    
    def __init__(self, dims: int = 4, name: str = "default"):
        self.dims = dims
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self._positions_cache: Optional[np.ndarray] = None
        self._needs_reproject = False
        self.created = datetime.now().isoformat()
        self.modified = datetime.now().isoformat()
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes
    
    def __iter__(self):
        return iter(self.nodes.values())
    
    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================
    
    def add(self, node_id: str, position: np.ndarray = None, data: Any = None) -> Node:
        """
        Add a node to the structure.
        
        Args:
            node_id: Unique identifier
            position: N-dimensional position (random if not provided)
            data: Opaque payload
            
        Returns:
            The created node
        """
        if position is None:
            # Random position on unit sphere
            position = np.random.randn(self.dims)
            position = position / np.linalg.norm(position) * CRITICAL_LINE
        
        position = np.array(position, dtype=np.float64)
        
        if len(position) != self.dims:
            raise ValueError(f"Position has {len(position)} dims, expected {self.dims}")
        
        node = Node(id=node_id, position=position, data=data)
        self.nodes[node_id] = node
        self._invalidate_cache()
        self.modified = datetime.now().isoformat()
        
        return node
    
    def remove(self, node_id: str) -> bool:
        """Remove a node by ID. Returns True if removed."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._invalidate_cache()
            self.modified = datetime.now().isoformat()
            return True
        return False
    
    def get(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def update_position(self, node_id: str, new_position: np.ndarray) -> bool:
        """Update a node's position directly."""
        node = self.nodes.get(node_id)
        if node:
            node.position = np.array(new_position, dtype=np.float64)
            node.modified = datetime.now().isoformat()
            self._invalidate_cache()
            self.modified = datetime.now().isoformat()
            return True
        return False
    
    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================
    
    def query_nearest(self, position: np.ndarray, k: int = 5) -> List[Tuple[Node, float]]:
        """
        Find k nearest nodes to a position.
        
        Args:
            position: Query position
            k: Number of results
            
        Returns:
            List of (node, similarity) tuples, sorted by similarity descending
        """
        if not self.nodes:
            return []
        
        position = np.array(position, dtype=np.float64)
        
        results = []
        for node in self.nodes.values():
            # Cosine similarity
            dot = np.dot(position, node.position)
            norm_q = np.linalg.norm(position)
            norm_n = np.linalg.norm(node.position)
            
            if norm_q > 1e-10 and norm_n > 1e-10:
                similarity = dot / (norm_q * norm_n)
            else:
                similarity = 0.0
            
            results.append((node, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def query_radius(self, position: np.ndarray, radius: float) -> List[Tuple[Node, float]]:
        """
        Find all nodes within radius of a position.
        
        Args:
            position: Query position
            radius: Maximum distance (Euclidean)
            
        Returns:
            List of (node, distance) tuples within radius
        """
        if not self.nodes:
            return []
        
        position = np.array(position, dtype=np.float64)
        
        results = []
        for node in self.nodes.values():
            distance = np.linalg.norm(position - node.position)
            if distance <= radius:
                results.append((node, distance))
        
        results.sort(key=lambda x: x[1])
        return results
    
    # =========================================================================
    # LEARNING OPERATIONS
    # =========================================================================
    
    def attract(self, node_id: str, target: np.ndarray, strength: float = 0.1) -> bool:
        """Move a node toward a target position."""
        node = self.nodes.get(node_id)
        if node:
            node.move_toward(np.array(target), strength)
            self._invalidate_cache()
            return True
        return False
    
    def repel(self, node_id: str, target: np.ndarray, strength: float = 0.05) -> bool:
        """Move a node away from a target position."""
        node = self.nodes.get(node_id)
        if node:
            node.move_away(np.array(target), strength)
            self._invalidate_cache()
            return True
        return False
    
    def feedback(self, node_id: str, query_position: np.ndarray, success: bool,
                 attract_strength: float = 0.1, repel_strength: float = 0.05) -> bool:
        """
        Provide feedback on a node's match to a query.
        
        If success: move node toward query (reinforce)
        If failure: move node away from query (correct)
        """
        if success:
            return self.attract(node_id, query_position, attract_strength)
        else:
            return self.repel(node_id, query_position, repel_strength)
    
    # =========================================================================
    # MAINTENANCE OPERATIONS
    # =========================================================================
    
    def prune(self, threshold: float = None) -> int:
        """
        Remove nodes below the critical line.
        
        Returns number of nodes removed.
        """
        threshold = threshold or CRITICAL_LINE
        
        to_remove = [
            node_id for node_id, node in self.nodes.items()
            if node.magnitude < threshold
        ]
        
        for node_id in to_remove:
            del self.nodes[node_id]
        
        if to_remove:
            self._invalidate_cache()
            self.modified = datetime.now().isoformat()
        
        return len(to_remove)
    
    def reproject(self, similarity_fn=None) -> None:
        """
        Reproject all positions based on similarity matrix.
        
        This maintains structural stability by ensuring positions
        reflect the actual similarities between nodes.
        
        Args:
            similarity_fn: Optional function(node1, node2) -> float
                          If not provided, uses current position similarity
        """
        if len(self.nodes) < 2:
            return
        
        node_list = list(self.nodes.values())
        n = len(node_list)
        
        # Build similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if similarity_fn:
                    S[i, j] = similarity_fn(node_list[i], node_list[j])
                else:
                    # Default: cosine similarity of current positions
                    pos_i = node_list[i].position
                    pos_j = node_list[j].position
                    dot = np.dot(pos_i, pos_j)
                    norm_i = np.linalg.norm(pos_i)
                    norm_j = np.linalg.norm(pos_j)
                    if norm_i > 1e-10 and norm_j > 1e-10:
                        S[i, j] = dot / (norm_i * norm_j)
                    else:
                        S[i, j] = 0.0
        
        self.similarity_matrix = S
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        # Take top dims eigenvectors
        idx = np.argsort(eigenvalues)[::-1][:self.dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        new_positions = eigenvectors[:, idx] * np.sqrt(valid_eigenvalues)
        
        # Update node positions
        for i, node in enumerate(node_list):
            node.position = new_positions[i]
            node.modified = datetime.now().isoformat()
        
        self._positions_cache = new_positions
        self._needs_reproject = False
        self.modified = datetime.now().isoformat()
    
    def _invalidate_cache(self):
        """Mark caches as invalid."""
        self._positions_cache = None
        self._needs_reproject = True
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the structure to a dictionary."""
        return {
            'type': 'HyperdimensionalStructure',
            'version': '1.0',
            'name': self.name,
            'dims': self.dims,
            'created': self.created,
            'modified': self.modified,
            'nodes': [node.to_dict() for node in self.nodes.values()],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HyperdimensionalStructure':
        """Deserialize from a dictionary."""
        structure = cls(
            dims=d.get('dims', 4),
            name=d.get('name', 'default'),
        )
        structure.created = d.get('created', structure.created)
        structure.modified = d.get('modified', structure.modified)
        
        for node_data in d.get('nodes', []):
            node = Node.from_dict(node_data)
            structure.nodes[node.id] = node
        
        return structure
    
    def save(self, path: str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'HyperdimensionalStructure':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Get structure statistics."""
        if not self.nodes:
            return {
                'name': self.name,
                'dims': self.dims,
                'node_count': 0,
                'persisting': 0,
                'temporary': 0,
            }
        
        magnitudes = [n.magnitude for n in self.nodes.values()]
        persisting = sum(1 for n in self.nodes.values() if n.persists)
        
        return {
            'name': self.name,
            'dims': self.dims,
            'node_count': len(self.nodes),
            'persisting': persisting,
            'temporary': len(self.nodes) - persisting,
            'mean_magnitude': float(np.mean(magnitudes)),
            'min_magnitude': float(np.min(magnitudes)),
            'max_magnitude': float(np.max(magnitudes)),
        }
    
    def resize(self, new_dims: int) -> None:
        """
        Resize the structure to a new dimensionality.
        
        If increasing: pads positions with zeros
        If decreasing: truncates positions (loses information)
        """
        if new_dims == self.dims:
            return
        
        for node in self.nodes.values():
            old_pos = node.position
            if new_dims > self.dims:
                # Pad with zeros
                node.position = np.concatenate([old_pos, np.zeros(new_dims - self.dims)])
            else:
                # Truncate
                node.position = old_pos[:new_dims]
        
        self.dims = new_dims
        self._invalidate_cache()
        self.modified = datetime.now().isoformat()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_structure(dims: int = 4, name: str = "default") -> HyperdimensionalStructure:
    """Create a new hyperdimensional structure."""
    return HyperdimensionalStructure(dims=dims, name=name)


if __name__ == "__main__":
    # Quick test
    print("=== HyperdimensionalStructure Test ===")
    
    structure = create_structure(dims=4, name="test")
    
    # Add some nodes
    structure.add("node1", data={"label": "first"})
    structure.add("node2", data={"label": "second"})
    structure.add("node3", data={"label": "third"})
    
    print(f"Created structure with {len(structure)} nodes")
    print(f"Stats: {structure.stats()}")
    
    # Query
    query_pos = structure.get("node1").position
    results = structure.query_nearest(query_pos, k=3)
    print(f"\nNearest to node1:")
    for node, sim in results:
        print(f"  {node.id}: {sim:.3f}")
    
    # Learning
    structure.feedback("node2", query_pos, success=True)
    print(f"\nAfter feedback, node2 magnitude: {structure.get('node2').magnitude:.3f}")
    
    # Serialize
    data = structure.to_dict()
    print(f"\nSerialized: {len(json.dumps(data))} bytes")
    
    # Deserialize
    restored = HyperdimensionalStructure.from_dict(data)
    print(f"Restored: {len(restored)} nodes")
    
    print("\n✓ HyperdimensionalStructure working!")
