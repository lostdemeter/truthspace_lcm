"""
HyperSpace - A Hyperdimensional Data Structure

A new kind of data structure for geometric computation.
Designed to be as simple to use as dict, list, or set.

Core Components:
1. HyperSpace - The container (like dict)
2. HyperCodec - Encoder/decoder (like json)
3. HyperPipeline - Chained spaces (like itertools.chain)

Usage:
    # Simple usage (like dict)
    space = HyperSpace(dims=8)
    space["hello"] = "world"
    result = space["hello"]  # Returns "world"
    
    # Query by similarity
    matches = space.query("hi", k=3)  # Find similar to "hi"
    
    # With custom codec
    space = HyperSpace(dims=8, codec=TextCodec())
    space.add("greeting", "Hello, how are you?")
    matches = space.query("Hi there!")
    
    # Chaining
    pipeline = space1 >> space2 >> space3
    result = pipeline.process(input)
    
    # Learning
    space.feedback("hello", success=True)

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Tuple, Set, Any, Optional, 
    Iterator, Callable, TypeVar, Generic, Union
)
from pathlib import Path
import hashlib

# Type variables for generic typing
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type
T = TypeVar('T')  # Generic type

# Critical line constant (from zeta function)
CRITICAL_LINE = 0.5


# =============================================================================
# HYPERNODE - A node in the space
# =============================================================================

@dataclass
class HyperNode(Generic[V]):
    """
    A node in the hyperdimensional space.
    
    Contains:
    - id: Unique identifier
    - position: N-dimensional coordinates
    - value: The stored value
    - metadata: Optional additional data
    """
    id: str
    position: np.ndarray
    value: V
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def magnitude(self) -> float:
        """Distance from origin."""
        return float(np.linalg.norm(self.position))
    
    @property
    def normalized(self) -> np.ndarray:
        """Unit vector in same direction."""
        mag = self.magnitude
        if mag > 1e-10:
            return self.position / mag
        return self.position
    
    def distance_to(self, other: Union['HyperNode', np.ndarray]) -> float:
        """Euclidean distance to another node or position."""
        if isinstance(other, HyperNode):
            return float(np.linalg.norm(self.position - other.position))
        return float(np.linalg.norm(self.position - other))
    
    def similarity_to(self, other: Union['HyperNode', np.ndarray]) -> float:
        """Cosine similarity to another node or position."""
        if isinstance(other, HyperNode):
            other_pos = other.position
        else:
            other_pos = other
        
        dot = np.dot(self.position, other_pos)
        norm1 = np.linalg.norm(self.position)
        norm2 = np.linalg.norm(other_pos)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            return float(dot / (norm1 * norm2))
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'value': self.value,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperNode':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            position=np.array(data['position']),
            value=data['value'],
            metadata=data.get('metadata', {}),
        )


# =============================================================================
# HYPERCODEC - Encoder/Decoder (Abstract Base)
# =============================================================================

class HyperCodec(ABC, Generic[T]):
    """
    Abstract base class for encoding/decoding values to/from positions.
    
    Like json.JSONEncoder/JSONDecoder, but for hyperdimensional space.
    
    Subclasses must implement:
    - encode(value) -> position
    - decode(position, candidates) -> value
    """
    
    def __init__(self, dims: int):
        self.dims = dims
    
    @abstractmethod
    def encode(self, value: T) -> np.ndarray:
        """Encode a value to a position vector."""
        pass
    
    @abstractmethod
    def decode(self, position: np.ndarray, 
               candidates: List[HyperNode]) -> Optional[T]:
        """Decode a position to a value using candidate nodes."""
        pass
    
    def similarity(self, value1: T, value2: T) -> float:
        """Compute similarity between two values (optional override)."""
        pos1 = self.encode(value1)
        pos2 = self.encode(value2)
        dot = np.dot(pos1, pos2)
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)
        if norm1 > 1e-10 and norm2 > 1e-10:
            return float(dot / (norm1 * norm2))
        return 0.0


# =============================================================================
# BUILT-IN CODECS
# =============================================================================

class IdentityCodec(HyperCodec[np.ndarray]):
    """
    Identity codec - values are already positions.
    
    Use when you want to work directly with position vectors.
    """
    
    def encode(self, value: np.ndarray) -> np.ndarray:
        if len(value) != self.dims:
            raise ValueError(f"Expected {self.dims} dims, got {len(value)}")
        return value
    
    def decode(self, position: np.ndarray, 
               candidates: List[HyperNode]) -> Optional[np.ndarray]:
        return position


class HashCodec(HyperCodec[str]):
    """
    Hash codec - deterministic positions from string hashes.
    
    Use for simple key-value storage where keys are strings.
    Similar words do NOT get similar positions.
    """
    
    def encode(self, value: str) -> np.ndarray:
        # Use hash to seed random generator for deterministic position
        seed = int(hashlib.md5(value.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        pos = np.random.randn(self.dims)
        pos = pos / np.linalg.norm(pos) * CRITICAL_LINE
        return pos
    
    def decode(self, position: np.ndarray,
               candidates: List[HyperNode]) -> Optional[str]:
        if not candidates:
            return None
        # Find nearest candidate
        best = min(candidates, key=lambda n: n.distance_to(position))
        return best.value


class TextCodec(HyperCodec[str]):
    """
    Text codec - positions from word co-occurrence.
    
    Similar text gets similar positions.
    Requires bootstrap data to learn word positions.
    """
    
    def __init__(self, dims: int):
        super().__init__(dims)
        self.word_positions: Dict[str, np.ndarray] = {}
        self.synonyms: List[Set[str]] = []
    
    def _extract_words(self, text: str) -> Set[str]:
        """Extract words from text."""
        words = text.lower().split()
        words = [''.join(c for c in w if c.isalnum()) for w in words]
        return {w for w in words if w}
    
    def learn_from_corpus(self, texts: List[str]) -> None:
        """Learn word positions from a corpus of texts."""
        # Build word co-occurrence
        word_to_texts: Dict[str, Set[int]] = {}
        all_words = set()
        
        for i, text in enumerate(texts):
            words = self._extract_words(text)
            all_words.update(words)
            for word in words:
                if word not in word_to_texts:
                    word_to_texts[word] = set()
                word_to_texts[word].add(i)
        
        # Expand with synonyms
        for group in self.synonyms:
            group_texts = set()
            for word in group:
                if word in word_to_texts:
                    group_texts.update(word_to_texts[word])
            for word in group:
                all_words.add(word)
                if word not in word_to_texts:
                    word_to_texts[word] = set()
                word_to_texts[word].update(group_texts)
        
        word_list = sorted(all_words)
        n = len(word_list)
        
        if n == 0:
            return
        
        # Build co-occurrence matrix
        cooccurrence = np.zeros((n, n))
        for i, w1 in enumerate(word_list):
            texts1 = word_to_texts.get(w1, set())
            for j, w2 in enumerate(word_list):
                texts2 = word_to_texts.get(w2, set())
                if texts1 or texts2:
                    intersection = len(texts1 & texts2)
                    union = len(texts1 | texts2)
                    cooccurrence[i, j] = intersection / union if union > 0 else 0
        
        # Holographic projection
        eigenvalues, eigenvectors = np.linalg.eigh(cooccurrence)
        idx = np.argsort(eigenvalues)[::-1][:self.dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        positions = eigenvectors[:, idx] * np.sqrt(valid_eigenvalues)
        
        # Store word positions
        for i, word in enumerate(word_list):
            pos = positions[i]
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            self.word_positions[word] = pos
    
    def add_synonyms(self, synonyms: List[List[str]]) -> None:
        """Add synonym groups."""
        self.synonyms = [set(group) for group in synonyms]
    
    def encode(self, value: str) -> np.ndarray:
        """Encode text to position (average of word positions)."""
        words = self._extract_words(value)
        
        positions = []
        for word in words:
            if word in self.word_positions:
                positions.append(self.word_positions[word])
            else:
                # Check synonyms
                for group in self.synonyms:
                    if word in group:
                        for syn in group:
                            if syn in self.word_positions:
                                positions.append(self.word_positions[syn])
                                break
                        break
        
        if not positions:
            # Fallback to hash for unknown words
            seed = int(hashlib.md5(value.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            pos = np.random.randn(self.dims)
            return pos / np.linalg.norm(pos) * CRITICAL_LINE * 0.3
        
        pos = np.mean(positions, axis=0)
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def decode(self, position: np.ndarray,
               candidates: List[HyperNode]) -> Optional[str]:
        if not candidates:
            return None
        # Find nearest candidate by cosine similarity
        best = max(candidates, key=lambda n: n.similarity_to(position))
        return best.value


# =============================================================================
# HYPERSPACE - The Main Data Structure
# =============================================================================

class HyperSpace(Generic[K, V]):
    """
    A hyperdimensional data structure.
    
    Like a dict, but with geometric operations:
    - Keys map to positions
    - Values are stored at those positions
    - Query finds similar keys
    - Learning moves positions
    
    Usage:
        # Simple (like dict)
        space = HyperSpace(dims=8)
        space["hello"] = "world"
        print(space["hello"])  # "world"
        
        # Query by similarity
        matches = space.query("hi", k=3)
        
        # With custom codec
        codec = TextCodec(dims=8)
        codec.learn_from_corpus(["hello world", "hi there"])
        space = HyperSpace(dims=8, codec=codec)
        
        # Iteration
        for key, value in space.items():
            print(key, value)
    """
    
    def __init__(self, dims: int = 8, 
                 codec: Optional[HyperCodec] = None,
                 name: str = "space"):
        self.dims = dims
        self.name = name
        self.codec = codec or HashCodec(dims)
        self._nodes: Dict[str, HyperNode] = {}
        self._key_to_id: Dict[Any, str] = {}
    
    # -------------------------------------------------------------------------
    # Dict-like interface
    # -------------------------------------------------------------------------
    
    def __setitem__(self, key: K, value: V) -> None:
        """Set item (like dict)."""
        self.add(key, value)
    
    def __getitem__(self, key: K) -> V:
        """Get item (like dict)."""
        node_id = self._key_to_id.get(key)
        if node_id is None:
            raise KeyError(key)
        return self._nodes[node_id].value
    
    def __delitem__(self, key: K) -> None:
        """Delete item (like dict)."""
        node_id = self._key_to_id.get(key)
        if node_id is None:
            raise KeyError(key)
        del self._nodes[node_id]
        del self._key_to_id[key]
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists (like dict)."""
        return key in self._key_to_id
    
    def __len__(self) -> int:
        """Number of items (like dict)."""
        return len(self._nodes)
    
    def __iter__(self) -> Iterator[K]:
        """Iterate over keys (like dict)."""
        return iter(self._key_to_id.keys())
    
    def keys(self) -> Iterator[K]:
        """Return keys (like dict)."""
        return iter(self._key_to_id.keys())
    
    def values(self) -> Iterator[V]:
        """Return values (like dict)."""
        for node in self._nodes.values():
            yield node.value
    
    def items(self) -> Iterator[Tuple[K, V]]:
        """Return key-value pairs (like dict)."""
        for key, node_id in self._key_to_id.items():
            yield key, self._nodes[node_id].value
    
    def get(self, key: K, default: V = None) -> V:
        """Get with default (like dict)."""
        try:
            return self[key]
        except KeyError:
            return default
    
    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------
    
    def add(self, key: K, value: V, 
            position: Optional[np.ndarray] = None,
            metadata: Optional[Dict[str, Any]] = None) -> HyperNode:
        """
        Add a key-value pair to the space.
        
        If position is not provided, it's computed from the key using the codec.
        """
        # Generate node ID
        node_id = str(key) if isinstance(key, (str, int)) else hashlib.md5(
            str(key).encode()).hexdigest()[:8]
        
        # Compute position if not provided
        if position is None:
            position = self.codec.encode(key)
        
        # Create node
        node = HyperNode(
            id=node_id,
            position=position,
            value=value,
            metadata=metadata or {}
        )
        
        # Store
        self._nodes[node_id] = node
        self._key_to_id[key] = node_id
        
        return node
    
    def query(self, key: K, k: int = 5) -> List[Tuple[K, V, float]]:
        """
        Query for similar keys.
        
        Returns list of (key, value, similarity) tuples, sorted by similarity.
        """
        position = self.codec.encode(key)
        return self.query_position(position, k)
    
    def query_position(self, position: np.ndarray, 
                       k: int = 5) -> List[Tuple[K, V, float]]:
        """
        Query by position vector.
        
        Returns list of (key, value, similarity) tuples.
        """
        results = []
        
        for key, node_id in self._key_to_id.items():
            node = self._nodes[node_id]
            similarity = node.similarity_to(position)
            results.append((key, node.value, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:k]
    
    def nearest(self, key: K) -> Optional[Tuple[K, V, float]]:
        """Find the single nearest match."""
        results = self.query(key, k=1)
        return results[0] if results else None
    
    # -------------------------------------------------------------------------
    # Learning operations
    # -------------------------------------------------------------------------
    
    def feedback(self, key: K, success: bool,
                 strength: float = 0.1) -> None:
        """
        Provide feedback on a key.
        
        Success: Move toward query position (reinforce)
        Failure: Move away from query position (correct)
        """
        node_id = self._key_to_id.get(key)
        if node_id is None:
            return
        
        node = self._nodes[node_id]
        query_pos = self.codec.encode(key)
        
        if success:
            # Attract toward query
            direction = query_pos - node.position
            node.position = node.position + strength * direction
        else:
            # Repel from query
            direction = node.position - query_pos
            node.position = node.position + (strength * 0.5) * direction
        
        # Renormalize
        norm = np.linalg.norm(node.position)
        if norm > 1e-10:
            node.position = node.position / norm * CRITICAL_LINE
    
    def attract(self, key1: K, key2: K, strength: float = 0.1) -> None:
        """Move two keys closer together."""
        id1 = self._key_to_id.get(key1)
        id2 = self._key_to_id.get(key2)
        if id1 is None or id2 is None:
            return
        
        node1 = self._nodes[id1]
        node2 = self._nodes[id2]
        
        direction = node2.position - node1.position
        node1.position = node1.position + strength * direction
        node2.position = node2.position - strength * direction
    
    def repel(self, key1: K, key2: K, strength: float = 0.05) -> None:
        """Move two keys further apart."""
        id1 = self._key_to_id.get(key1)
        id2 = self._key_to_id.get(key2)
        if id1 is None or id2 is None:
            return
        
        node1 = self._nodes[id1]
        node2 = self._nodes[id2]
        
        direction = node1.position - node2.position
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
            node1.position = node1.position + strength * direction
            node2.position = node2.position - strength * direction
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'HyperSpace',
            'version': '1.0',
            'name': self.name,
            'dims': self.dims,
            'nodes': {nid: node.to_dict() for nid, node in self._nodes.items()},
            'key_to_id': {str(k): v for k, v in self._key_to_id.items()},
        }
    
    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], 
                  codec: Optional[HyperCodec] = None) -> 'HyperSpace':
        """Deserialize from dictionary."""
        space = cls(
            dims=data.get('dims', 8),
            codec=codec,
            name=data.get('name', 'space')
        )
        
        for nid, node_data in data.get('nodes', {}).items():
            space._nodes[nid] = HyperNode.from_dict(node_data)
        
        space._key_to_id = {k: v for k, v in data.get('key_to_id', {}).items()}
        
        return space
    
    @classmethod
    def load(cls, path: str, 
             codec: Optional[HyperCodec] = None) -> 'HyperSpace':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, codec)
    
    # -------------------------------------------------------------------------
    # Chaining (pipe operator)
    # -------------------------------------------------------------------------
    
    def __rshift__(self, other: 'HyperSpace') -> 'HyperPipeline':
        """Chain spaces with >> operator."""
        return HyperPipeline([self, other])
    
    def __repr__(self) -> str:
        return f"HyperSpace(name='{self.name}', dims={self.dims}, len={len(self)})"


# =============================================================================
# HYPERPIPELINE - Chained Spaces
# =============================================================================

class HyperPipeline:
    """
    A pipeline of chained HyperSpaces.
    
    Usage:
        pipeline = space1 >> space2 >> space3
        result = pipeline.process(input)
    """
    
    def __init__(self, spaces: List[HyperSpace]):
        self.spaces = spaces
        self.router: Optional[Callable[[int, Any], int]] = None
    
    def __rshift__(self, other: HyperSpace) -> 'HyperPipeline':
        """Add another space to the pipeline."""
        return HyperPipeline(self.spaces + [other])
    
    def set_router(self, router: Callable[[int, Any], int]) -> None:
        """
        Set a routing function.
        
        router(current_index, result) -> next_index
        Return -1 to stop, or index of next space.
        """
        self.router = router
    
    def process(self, key: Any, k: int = 1) -> List[Tuple[Any, float]]:
        """
        Process input through the pipeline.
        
        Each space's output becomes the next space's input.
        """
        current_key = key
        current_idx = 0
        
        while current_idx < len(self.spaces):
            space = self.spaces[current_idx]
            
            # Query current space
            results = space.query(current_key, k=k)
            
            if not results:
                return []
            
            # Get best result
            best_key, best_value, confidence = results[0]
            
            # Determine next space
            if self.router:
                next_idx = self.router(current_idx, best_value)
                if next_idx < 0:
                    return [(best_value, confidence)]
                current_idx = next_idx
            else:
                current_idx += 1
            
            # Use result as next input
            current_key = best_value
        
        # Return final result
        return [(current_key, 1.0)]
    
    def __repr__(self) -> str:
        names = [s.name for s in self.spaces]
        return f"HyperPipeline({' >> '.join(names)})"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  HYPERSPACE - A Hyperdimensional Data Structure")
    print("=" * 60)
    print()
    
    # Test 1: Simple dict-like usage
    print("--- Test 1: Dict-like Usage ---")
    space = HyperSpace(dims=8, name="simple")
    
    space["hello"] = "world"
    space["foo"] = "bar"
    space["python"] = "programming"
    
    print(f"space['hello'] = {space['hello']}")
    print(f"space['foo'] = {space['foo']}")
    print(f"'hello' in space: {'hello' in space}")
    print(f"len(space): {len(space)}")
    print()
    
    # Test 2: Query by similarity
    print("--- Test 2: Query by Similarity ---")
    # With hash codec, similar strings don't get similar positions
    # But we can still query
    results = space.query("hello", k=3)
    for key, value, sim in results:
        print(f"  {key} → {value} (sim={sim:.3f})")
    print()
    
    # Test 3: Text codec with learned positions
    print("--- Test 3: Text Codec ---")
    codec = TextCodec(dims=8)
    
    # Learn from corpus
    corpus = [
        "list files",
        "show files",
        "display files",
        "delete file",
        "remove file",
        "kill process",
        "stop process",
    ]
    codec.learn_from_corpus(corpus)
    
    text_space = HyperSpace(dims=8, codec=codec, name="commands")
    text_space["list files"] = "ls"
    text_space["show files"] = "ls"
    text_space["delete file"] = "rm"
    text_space["kill process"] = "kill"
    
    # Query with similar text
    print("Query: 'display files'")
    results = text_space.query("display files", k=3)
    for key, value, sim in results:
        print(f"  {key} → {value} (sim={sim:.3f})")
    print()
    
    print("Query: 'remove file'")
    results = text_space.query("remove file", k=3)
    for key, value, sim in results:
        print(f"  {key} → {value} (sim={sim:.3f})")
    print()
    
    # Test 4: Learning
    print("--- Test 4: Learning ---")
    print("Before feedback:")
    results = text_space.query("erase file", k=1)
    print(f"  'erase file' → {results[0][1] if results else 'NO MATCH'}")
    
    # Provide feedback
    text_space.feedback("delete file", success=True)
    text_space.feedback("delete file", success=True)
    
    print("After positive feedback on 'delete file':")
    results = text_space.query("delete file", k=1)
    print(f"  'delete file' → {results[0][1]} (sim={results[0][2]:.3f})")
    print()
    
    # Test 5: Pipeline
    print("--- Test 5: Pipeline ---")
    intent_space = HyperSpace(dims=8, name="intent")
    intent_space["file"] = "file_ops"
    intent_space["process"] = "proc_ops"
    
    cmd_space = HyperSpace(dims=8, name="commands")
    cmd_space["file_ops"] = "ls"
    cmd_space["proc_ops"] = "ps"
    
    pipeline = intent_space >> cmd_space
    print(f"Pipeline: {pipeline}")
    
    result = pipeline.process("file")
    print(f"  'file' → {result}")
    print()
    
    # Test 6: Serialization
    print("--- Test 6: Serialization ---")
    text_space.save("/tmp/hyperspace_test.json")
    print("Saved to /tmp/hyperspace_test.json")
    
    loaded = HyperSpace.load("/tmp/hyperspace_test.json")
    print(f"Loaded: {loaded}")
    print(f"  loaded['list files'] = {loaded['list files']}")
    print()
    
    # Test 7: Iteration
    print("--- Test 7: Iteration ---")
    print("Keys:", list(text_space.keys()))
    print("Items:")
    for key, value in text_space.items():
        print(f"  {key} → {value}")
    print()
    
    print("=" * 60)
    print("  TESTS COMPLETE")
    print("=" * 60)
    print()
    print("HyperSpace provides:")
    print("  ✓ Dict-like interface ([], in, len, keys, values, items)")
    print("  ✓ Similarity queries (query, nearest)")
    print("  ✓ Learning (feedback, attract, repel)")
    print("  ✓ Pluggable codecs (HashCodec, TextCodec, custom)")
    print("  ✓ Chaining (>> operator)")
    print("  ✓ Serialization (save, load)")
