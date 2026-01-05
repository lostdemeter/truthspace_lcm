"""
HyperMapping - A Bidirectional Hyperdimensional Data Structure

A cleaner abstraction than key-value: bidirectional mappings in geometric space.

Core Insight:
- Not key → value, but input ↔ output
- Both sides get positions
- Query from either direction
- The RELATIONSHIP is what matters

Usage:
    # Create a mapping space
    space = HyperMapping(dims=8)
    
    # Add bidirectional mappings
    space.map("list files", "ls")
    space.map("show files", "ls")
    space.map("delete file", "rm")
    
    # Query forward (input → output)
    result = space.forward("display files")  # → "ls"
    
    # Query backward (output → input)
    result = space.backward("ls")  # → ["list files", "show files"]
    
    # Query (finds nearest mapping)
    result = space.query("enumerate files")  # → Mapping("list files", "ls", 0.95)
    
    # Learning
    space.feedback("list files", "ls", success=True)
    
    # Chaining
    pipeline = space1 | space2 | space3
    result = pipeline("input")

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

# Type variables
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

# Critical line constant
CRITICAL_LINE = 0.5


# =============================================================================
# MAPPING - A single input-output pair with position
# =============================================================================

@dataclass
class Mapping(Generic[I, O]):
    """
    A single mapping in the space.
    
    Contains:
    - input: The input value
    - output: The output value
    - position: N-dimensional coordinates (computed from both)
    - metadata: Optional additional data
    """
    input: I
    output: O
    position: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def magnitude(self) -> float:
        """Distance from origin."""
        return float(np.linalg.norm(self.position))
    
    def similarity_to(self, position: np.ndarray) -> float:
        """Cosine similarity to a position."""
        dot = np.dot(self.position, position)
        norm1 = np.linalg.norm(self.position)
        norm2 = np.linalg.norm(position)
        if norm1 > 1e-10 and norm2 > 1e-10:
            return float(dot / (norm1 * norm2))
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'input': self.input,
            'output': self.output,
            'position': self.position.tolist(),
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mapping':
        """Deserialize from dictionary."""
        return cls(
            input=data['input'],
            output=data['output'],
            position=np.array(data['position']),
            metadata=data.get('metadata', {}),
        )
    
    def __repr__(self) -> str:
        return f"Mapping({self.input!r} → {self.output!r})"


# =============================================================================
# MATCH RESULT - Result from a query
# =============================================================================

@dataclass
class MatchResult(Generic[I, O]):
    """Result from a query operation."""
    mapping: Mapping[I, O]
    similarity: float
    
    @property
    def input(self) -> I:
        return self.mapping.input
    
    @property
    def output(self) -> O:
        return self.mapping.output
    
    def __repr__(self) -> str:
        return f"Match({self.input!r} → {self.output!r}, sim={self.similarity:.3f})"


# =============================================================================
# ENCODER - How to compute positions
# =============================================================================

class Encoder(ABC):
    """
    Abstract encoder that computes positions from inputs/outputs.
    
    Unlike a codec that only encodes keys, an Encoder can use
    BOTH the input and output to compute a position.
    """
    
    def __init__(self, dims: int):
        self.dims = dims
    
    @abstractmethod
    def encode_input(self, input_val: Any) -> np.ndarray:
        """Encode an input value to a position."""
        pass
    
    @abstractmethod
    def encode_output(self, output_val: Any) -> np.ndarray:
        """Encode an output value to a position."""
        pass
    
    def encode_mapping(self, input_val: Any, output_val: Any) -> np.ndarray:
        """
        Encode a mapping to a position.
        
        Default: average of input and output positions.
        Override for custom behavior.
        """
        input_pos = self.encode_input(input_val)
        output_pos = self.encode_output(output_val)
        
        # Average position
        pos = (input_pos + output_pos) / 2
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos


# =============================================================================
# BUILT-IN ENCODERS
# =============================================================================

class HashEncoder(Encoder):
    """
    Hash-based encoder - deterministic positions from hashes.
    
    Simple but no semantic similarity.
    """
    
    def _hash_to_position(self, value: Any) -> np.ndarray:
        seed = int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        pos = np.random.randn(self.dims)
        return pos / np.linalg.norm(pos) * CRITICAL_LINE
    
    def encode_input(self, input_val: Any) -> np.ndarray:
        return self._hash_to_position(input_val)
    
    def encode_output(self, output_val: Any) -> np.ndarray:
        return self._hash_to_position(output_val)


class TextEncoder(Encoder):
    """
    Text encoder - positions from word co-occurrence.
    
    Similar text gets similar positions.
    """
    
    def __init__(self, dims: int):
        super().__init__(dims)
        self.word_positions: Dict[str, np.ndarray] = {}
        self.synonyms: List[Set[str]] = []
    
    def _extract_words(self, text: str) -> Set[str]:
        words = str(text).lower().split()
        words = [''.join(c for c in w if c.isalnum()) for w in words]
        return {w for w in words if w}
    
    def add_synonyms(self, synonym_groups: List[List[str]]) -> None:
        """Add synonym groups."""
        self.synonyms = [set(g) for g in synonym_groups]
    
    def learn(self, texts: List[str]) -> None:
        """Learn word positions from texts."""
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
        
        # Co-occurrence matrix
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
        
        for i, word in enumerate(word_list):
            pos = positions[i]
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            self.word_positions[word] = pos
    
    def _encode_text(self, text: str) -> np.ndarray:
        words = self._extract_words(text)
        
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
            # Fallback to hash
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            pos = np.random.randn(self.dims)
            return pos / np.linalg.norm(pos) * CRITICAL_LINE * 0.3
        
        pos = np.mean(positions, axis=0)
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_input(self, input_val: Any) -> np.ndarray:
        return self._encode_text(str(input_val))
    
    def encode_output(self, output_val: Any) -> np.ndarray:
        return self._encode_text(str(output_val))


# =============================================================================
# HYPERMAPPING - The Main Data Structure
# =============================================================================

class HyperMapping(Generic[I, O]):
    """
    A bidirectional hyperdimensional mapping.
    
    Maps inputs to outputs through geometric space.
    Both directions are queryable.
    
    Usage:
        space = HyperMapping(dims=8)
        
        # Add mappings
        space.map("list files", "ls")
        space.map("show files", "ls")
        
        # Query forward (input → output)
        result = space.forward("display files")
        print(result.output)  # "ls"
        
        # Query backward (output → input)
        results = space.backward("ls")
        for r in results:
            print(r.input)  # "list files", "show files"
        
        # General query
        results = space.query("enumerate files", k=3)
    """
    
    def __init__(self, dims: int = 8,
                 encoder: Optional[Encoder] = None,
                 name: str = "mapping"):
        self.dims = dims
        self.name = name
        self.encoder = encoder or HashEncoder(dims)
        self._mappings: List[Mapping[I, O]] = []
        
        # Indices for fast lookup
        self._input_index: Dict[Any, List[int]] = {}
        self._output_index: Dict[Any, List[int]] = {}
    
    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------
    
    def map(self, input_val: I, output_val: O,
            position: Optional[np.ndarray] = None,
            metadata: Optional[Dict[str, Any]] = None) -> Mapping[I, O]:
        """
        Add a mapping from input to output.
        
        If position is not provided, it's computed from both values.
        """
        if position is None:
            position = self.encoder.encode_mapping(input_val, output_val)
        
        mapping = Mapping(
            input=input_val,
            output=output_val,
            position=position,
            metadata=metadata or {}
        )
        
        idx = len(self._mappings)
        self._mappings.append(mapping)
        
        # Update indices
        if input_val not in self._input_index:
            self._input_index[input_val] = []
        self._input_index[input_val].append(idx)
        
        if output_val not in self._output_index:
            self._output_index[output_val] = []
        self._output_index[output_val].append(idx)
        
        return mapping
    
    def forward(self, input_val: I, k: int = 1) -> Optional[MatchResult[I, O]]:
        """
        Query forward: input → output.
        
        Returns the best matching output for the given input.
        """
        position = self.encoder.encode_input(input_val)
        results = self._query_by_position(position, k)
        return results[0] if results else None
    
    def backward(self, output_val: O, k: int = 5) -> List[MatchResult[I, O]]:
        """
        Query backward: output → inputs.
        
        Returns all inputs that map to similar outputs.
        """
        position = self.encoder.encode_output(output_val)
        return self._query_by_position(position, k)
    
    def query(self, value: Any, k: int = 5) -> List[MatchResult[I, O]]:
        """
        General query - finds nearest mappings.
        
        Encodes value as input and finds similar mappings.
        """
        position = self.encoder.encode_input(value)
        return self._query_by_position(position, k)
    
    def _query_by_position(self, position: np.ndarray,
                           k: int) -> List[MatchResult[I, O]]:
        """Internal query by position vector."""
        results = []
        
        for mapping in self._mappings:
            similarity = mapping.similarity_to(position)
            results.append(MatchResult(mapping, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        return results[:k]
    
    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------
    
    def feedback(self, input_val: I, output_val: O,
                 success: bool, strength: float = 0.1) -> None:
        """
        Provide feedback on a mapping.
        
        Success: Move mapping toward query position
        Failure: Move mapping away from query position
        """
        query_pos = self.encoder.encode_mapping(input_val, output_val)
        
        # Find matching mappings
        for mapping in self._mappings:
            if mapping.input == input_val and mapping.output == output_val:
                if success:
                    direction = query_pos - mapping.position
                    mapping.position = mapping.position + strength * direction
                else:
                    direction = mapping.position - query_pos
                    mapping.position = mapping.position + (strength * 0.5) * direction
                
                # Renormalize
                norm = np.linalg.norm(mapping.position)
                if norm > 1e-10:
                    mapping.position = mapping.position / norm * CRITICAL_LINE
    
    def attract(self, mapping1: Mapping, mapping2: Mapping,
                strength: float = 0.1) -> None:
        """Move two mappings closer together."""
        direction = mapping2.position - mapping1.position
        mapping1.position = mapping1.position + strength * direction
        mapping2.position = mapping2.position - strength * direction
    
    def repel(self, mapping1: Mapping, mapping2: Mapping,
              strength: float = 0.05) -> None:
        """Move two mappings further apart."""
        direction = mapping1.position - mapping2.position
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
            mapping1.position = mapping1.position + strength * direction
            mapping2.position = mapping2.position - strength * direction
    
    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------
    
    def __len__(self) -> int:
        return len(self._mappings)
    
    def __iter__(self) -> Iterator[Mapping[I, O]]:
        return iter(self._mappings)
    
    def inputs(self) -> Set[I]:
        """Return all unique inputs."""
        return set(self._input_index.keys())
    
    def outputs(self) -> Set[O]:
        """Return all unique outputs."""
        return set(self._output_index.keys())
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'HyperMapping',
            'version': '1.0',
            'name': self.name,
            'dims': self.dims,
            'mappings': [m.to_dict() for m in self._mappings],
        }
    
    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  encoder: Optional[Encoder] = None) -> 'HyperMapping':
        """Deserialize from dictionary."""
        space = cls(
            dims=data.get('dims', 8),
            encoder=encoder,
            name=data.get('name', 'mapping')
        )
        
        for m_data in data.get('mappings', []):
            mapping = Mapping.from_dict(m_data)
            idx = len(space._mappings)
            space._mappings.append(mapping)
            
            if mapping.input not in space._input_index:
                space._input_index[mapping.input] = []
            space._input_index[mapping.input].append(idx)
            
            if mapping.output not in space._output_index:
                space._output_index[mapping.output] = []
            space._output_index[mapping.output].append(idx)
        
        return space
    
    @classmethod
    def load(cls, path: str,
             encoder: Optional[Encoder] = None) -> 'HyperMapping':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, encoder)
    
    # -------------------------------------------------------------------------
    # Chaining (pipe operator)
    # -------------------------------------------------------------------------
    
    def __or__(self, other: 'HyperMapping') -> 'HyperPipeline':
        """Chain mappings with | operator."""
        return HyperPipeline([self, other])
    
    def __repr__(self) -> str:
        return f"HyperMapping(name='{self.name}', dims={self.dims}, len={len(self)})"


# =============================================================================
# HYPERPIPELINE - Chained Mappings
# =============================================================================

class HyperPipeline:
    """
    A pipeline of chained HyperMappings.
    
    Usage:
        pipeline = space1 | space2 | space3
        result = pipeline("input")
    """
    
    def __init__(self, mappings: List[HyperMapping]):
        self.mappings = mappings
    
    def __or__(self, other: HyperMapping) -> 'HyperPipeline':
        """Add another mapping to the pipeline."""
        return HyperPipeline(self.mappings + [other])
    
    def __call__(self, input_val: Any) -> Optional[Any]:
        """Process input through the pipeline."""
        current = input_val
        
        for mapping in self.mappings:
            result = mapping.forward(current)
            if result is None:
                return None
            current = result.output
        
        return current
    
    def process(self, input_val: Any, k: int = 1) -> List[MatchResult]:
        """Process with full results."""
        results = []
        current = input_val
        
        for mapping in self.mappings:
            result = mapping.forward(current, k=k)
            if result is None:
                return results
            results.append(result)
            current = result.output
        
        return results
    
    def __repr__(self) -> str:
        names = [m.name for m in self.mappings]
        return f"HyperPipeline({' | '.join(names)})"


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def from_pairs(pairs: List[Tuple[Any, Any]],
               dims: int = 8,
               encoder: Optional[Encoder] = None,
               name: str = "mapping") -> HyperMapping:
    """
    Create a HyperMapping from a list of (input, output) pairs.
    
    Usage:
        space = from_pairs([
            ("list files", "ls"),
            ("show files", "ls"),
            ("delete file", "rm"),
        ])
    """
    space = HyperMapping(dims=dims, encoder=encoder, name=name)
    for input_val, output_val in pairs:
        space.map(input_val, output_val)
    return space


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  HYPERMAPPING - Bidirectional Hyperdimensional Mapping")
    print("=" * 60)
    print()
    
    # Test 1: Basic mapping
    print("--- Test 1: Basic Mapping ---")
    space = HyperMapping(dims=8, name="commands")
    
    space.map("list files", "ls")
    space.map("show files", "ls")
    space.map("delete file", "rm")
    space.map("kill process", "kill")
    
    print(f"Created: {space}")
    print(f"Inputs: {space.inputs()}")
    print(f"Outputs: {space.outputs()}")
    print()
    
    # Test 2: Forward query
    print("--- Test 2: Forward Query (input → output) ---")
    result = space.forward("list files")
    print(f"  'list files' → {result}")
    
    result = space.forward("delete file")
    print(f"  'delete file' → {result}")
    print()
    
    # Test 3: Backward query
    print("--- Test 3: Backward Query (output → inputs) ---")
    results = space.backward("ls", k=5)
    print(f"  'ls' ← ")
    for r in results:
        print(f"    {r}")
    print()
    
    # Test 4: Text encoder with similarity
    print("--- Test 4: Text Encoder with Similarity ---")
    encoder = TextEncoder(dims=8)
    
    # Learn from corpus
    corpus = [
        "list files", "show files", "display files",
        "delete file", "remove file",
        "kill process", "stop process",
    ]
    encoder.learn(corpus)
    encoder.add_synonyms([
        ["list", "show", "display", "enumerate"],
        ["delete", "remove", "erase"],
        ["kill", "stop", "terminate"],
    ])
    
    text_space = HyperMapping(dims=8, encoder=encoder, name="text_commands")
    text_space.map("list files", "ls")
    text_space.map("show files", "ls")
    text_space.map("delete file", "rm")
    text_space.map("kill process", "kill")
    
    print("Query: 'display files'")
    result = text_space.forward("display files")
    print(f"  → {result}")
    
    print("Query: 'remove file'")
    result = text_space.forward("remove file")
    print(f"  → {result}")
    
    print("Query: 'terminate process'")
    result = text_space.forward("terminate process")
    print(f"  → {result}")
    print()
    
    # Test 5: Pipeline
    print("--- Test 5: Pipeline ---")
    intent_space = HyperMapping(dims=8, name="intent")
    intent_space.map("file", "file_ops")
    intent_space.map("process", "proc_ops")
    
    cmd_space = HyperMapping(dims=8, name="commands")
    cmd_space.map("file_ops", "ls")
    cmd_space.map("proc_ops", "ps")
    
    pipeline = intent_space | cmd_space
    print(f"Pipeline: {pipeline}")
    
    result = pipeline("file")
    print(f"  'file' → {result}")
    
    result = pipeline("process")
    print(f"  'process' → {result}")
    print()
    
    # Test 6: Convenience function
    print("--- Test 6: from_pairs() ---")
    quick_space = from_pairs([
        ("hello", "world"),
        ("foo", "bar"),
        ("python", "programming"),
    ], name="quick")
    print(f"Created: {quick_space}")
    print()
    
    # Test 7: Serialization
    print("--- Test 7: Serialization ---")
    text_space.save("/tmp/hypermapping_test.json")
    print("Saved to /tmp/hypermapping_test.json")
    
    loaded = HyperMapping.load("/tmp/hypermapping_test.json")
    print(f"Loaded: {loaded}")
    result = loaded.forward("list files")
    print(f"  'list files' → {result}")
    print()
    
    # Test 8: Iteration
    print("--- Test 8: Iteration ---")
    print("All mappings:")
    for mapping in text_space:
        print(f"  {mapping}")
    print()
    
    print("=" * 60)
    print("  TESTS COMPLETE")
    print("=" * 60)
    print()
    print("HyperMapping provides:")
    print("  ✓ Bidirectional mappings (input ↔ output)")
    print("  ✓ Forward query (input → output)")
    print("  ✓ Backward query (output → inputs)")
    print("  ✓ Similarity-based matching")
    print("  ✓ Pluggable encoders")
    print("  ✓ Learning (feedback, attract, repel)")
    print("  ✓ Chaining with | operator")
    print("  ✓ Serialization")
