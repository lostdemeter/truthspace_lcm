"""
Vector Symbolic Architecture (VSA) Binding Operations

This module extends TruthSpace LCM with binding operations, transforming it from
a bag-of-words semantic system into a full Vector Symbolic Architecture (VSA)
capable of representing relations, sequences, and structured knowledge.

Core Operations:
- bind(a, b): Reversibly associate two vectors (result dissimilar to both)
- unbind(bound, key): Recover the other vector from a binding
- bundle(*vectors): Superposition (result similar to all inputs)
- permute(v, n): Position encoding for sequences

All operations are purely geometric - no training, no probabilities.

References:
- Plate (1995): Holographic Reduced Representations
- Kanerva (2009): Hyperdimensional Computing
- Gayler (2003): Vector Symbolic Architectures
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum


class BindingMethod(Enum):
    """Available binding methods."""
    HADAMARD = "hadamard"           # Element-wise multiplication
    CIRCULAR_CONV = "circular_conv"  # Circular convolution (HRR)


# Default binding method - circular convolution has better properties
DEFAULT_METHOD = BindingMethod.CIRCULAR_CONV


# =============================================================================
# CORE BINDING OPERATIONS
# =============================================================================

def bind(a: np.ndarray, b: np.ndarray, 
         method: BindingMethod = DEFAULT_METHOD) -> np.ndarray:
    """
    Bind two vectors to create an association.
    
    The bound result is DISSIMILAR to both inputs - this is the key VSA property
    that enables storing multiple bindings in superposition without interference.
    
    Args:
        a: First vector (e.g., role vector like "capital_of")
        b: Second vector (e.g., filler vector like "paris")
        method: Binding method to use
        
    Returns:
        Bound vector representing the association a ⊛ b
        
    Example:
        >>> capital_of = get_position("capital_of")
        >>> paris = get_position("paris")
        >>> bound = bind(capital_of, paris)  # Represents "capital_of:paris"
    """
    if method == BindingMethod.HADAMARD:
        return _bind_hadamard(a, b)
    elif method == BindingMethod.CIRCULAR_CONV:
        return _bind_circular_conv(a, b)
    else:
        raise ValueError(f"Unknown binding method: {method}")


def unbind(bound: np.ndarray, key: np.ndarray,
           method: BindingMethod = DEFAULT_METHOD) -> np.ndarray:
    """
    Unbind to recover the associated vector.
    
    If bound = bind(a, b), then:
        unbind(bound, a) ≈ b
        unbind(bound, b) ≈ a
    
    The result is noisy and requires clean-up via similarity search.
    
    Args:
        bound: The bound vector
        key: The key to unbind with
        method: Binding method (must match the one used for binding)
        
    Returns:
        Noisy approximation of the other vector
        
    Example:
        >>> recovered = unbind(bound, capital_of)
        >>> # recovered ≈ paris (use similarity search to clean up)
    """
    if method == BindingMethod.HADAMARD:
        return _unbind_hadamard(bound, key)
    elif method == BindingMethod.CIRCULAR_CONV:
        return _unbind_circular_conv(bound, key)
    else:
        raise ValueError(f"Unknown binding method: {method}")


def bundle(*vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Bundle multiple vectors via superposition (addition).
    
    Unlike binding, the bundled result is SIMILAR to all inputs.
    This is the VSA equivalent of set union.
    
    Args:
        *vectors: Vectors to bundle together
        normalize: Whether to normalize the result to unit length
        
    Returns:
        Bundled vector representing the set/multiset of inputs
        
    Example:
        >>> fact1 = bind(capital_of, paris)
        >>> fact2 = bind(capital_of, berlin)
        >>> knowledge = bundle(fact1, fact2)  # Contains both facts
    """
    if len(vectors) == 0:
        raise ValueError("Cannot bundle zero vectors")
    
    result = np.sum(vectors, axis=0)
    
    if normalize:
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result = result / norm
    
    return result


def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Permute vector by cyclic shift for position encoding.
    
    Used to encode order in sequences:
        seq = bundle(permute(w1, 0), permute(w2, 1), permute(w3, 2), ...)
    
    Args:
        v: Vector to permute
        shift: Number of positions to shift (can be negative)
        
    Returns:
        Permuted vector
        
    Example:
        >>> seq = bundle(
        ...     bind(permute(P, 0), word1),
        ...     bind(permute(P, 1), word2),
        ...     bind(permute(P, 2), word3),
        ... )
    """
    return np.roll(v, shift)


def inverse_permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Inverse of permute operation."""
    return np.roll(v, -shift)


# =============================================================================
# BINDING METHOD IMPLEMENTATIONS
# =============================================================================

def _bind_hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind using element-wise multiplication (Hadamard product).
    
    Properties:
    - Commutative: a ⊛ b = b ⊛ a
    - Self-inverse: a ⊛ a ≈ 1 (for normalized vectors)
    - O(D) complexity
    - Simpler but noisier recovery than convolution
    """
    result = a * b
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


def _unbind_hadamard(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Unbind using Hadamard (self-inverse property)."""
    return _bind_hadamard(bound, key)


def _bind_circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind using circular convolution (Holographic Reduced Representations).
    
    Formula: a ⊛ b = ifft(fft(a) · fft(b))
    
    Properties:
    - Better orthogonality preservation than Hadamard
    - Exact inverse via correlation
    - O(D log D) complexity via FFT
    - Cleaner recovery
    """
    result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


def _unbind_circular_conv(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Unbind using circular correlation (inverse of convolution).
    
    Formula: unbind(bound, key) = ifft(fft(bound) · conj(fft(key)))
    """
    result = np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(key))).real
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


# =============================================================================
# CLEAN-UP MEMORY
# =============================================================================

class CleanupMemory:
    """
    Clean-up memory for recovering symbols from noisy unbinding results.
    
    After unbinding, results are noisy approximations. The clean-up memory
    finds the nearest known vector to recover the exact symbol.
    
    Example:
        >>> memory = CleanupMemory()
        >>> memory.add("paris", paris_vector)
        >>> memory.add("berlin", berlin_vector)
        >>> 
        >>> noisy = unbind(bound, capital_of)
        >>> symbol, similarity = memory.cleanup(noisy)
        >>> print(symbol)  # "paris"
    """
    
    def __init__(self):
        self.symbols: Dict[str, np.ndarray] = {}
    
    def add(self, name: str, vector: np.ndarray) -> None:
        """Add a symbol to the clean-up memory."""
        self.symbols[name] = vector.copy()
    
    def add_many(self, items: Dict[str, np.ndarray]) -> None:
        """Add multiple symbols at once."""
        for name, vector in items.items():
            self.add(name, vector)
    
    def cleanup(self, noisy: np.ndarray, 
                threshold: float = 0.0) -> Tuple[Optional[str], float]:
        """
        Find the nearest symbol to a noisy vector.
        
        Args:
            noisy: Noisy vector from unbinding
            threshold: Minimum similarity to return a match
            
        Returns:
            (symbol_name, similarity) or (None, 0.0) if below threshold
        """
        if len(self.symbols) == 0:
            return None, 0.0
        
        best_name = None
        best_sim = -np.inf
        
        noisy_norm = np.linalg.norm(noisy)
        if noisy_norm < 1e-8:
            return None, 0.0
        
        for name, vec in self.symbols.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-8:
                continue
            sim = np.dot(noisy, vec) / (noisy_norm * vec_norm)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        
        if best_sim < threshold:
            return None, 0.0
        
        return best_name, float(best_sim)
    
    def cleanup_top_k(self, noisy: np.ndarray, 
                      k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k nearest symbols with similarities."""
        if len(self.symbols) == 0:
            return []
        
        noisy_norm = np.linalg.norm(noisy)
        if noisy_norm < 1e-8:
            return []
        
        results = []
        for name, vec in self.symbols.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-8:
                continue
            sim = np.dot(noisy, vec) / (noisy_norm * vec_norm)
            results.append((name, float(sim)))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def __len__(self) -> int:
        return len(self.symbols)
    
    def __contains__(self, name: str) -> bool:
        return name in self.symbols


# =============================================================================
# RELATIONAL KNOWLEDGE STORE
# =============================================================================

class RelationalStore:
    """
    Store for relational knowledge using VSA binding.
    
    Stores facts as bound role-filler pairs, enabling:
    - Relational queries: "What is the capital of France?"
    - Analogical reasoning: "France:Paris :: Germany:?"
    
    Example:
        >>> store = RelationalStore(dim=256)
        >>> store.add_relation("capital_of", "france", "paris")
        >>> store.add_relation("capital_of", "germany", "berlin")
        >>> 
        >>> result = store.query("capital_of", "france")
        >>> print(result)  # [("paris", 0.85), ...]
    """
    
    def __init__(self, dim: int = 256, method: BindingMethod = DEFAULT_METHOD):
        self.dim = dim
        self.method = method
        self.cleanup = CleanupMemory()
        self._role_vectors: Dict[str, np.ndarray] = {}
        self._entity_vectors: Dict[str, np.ndarray] = {}
        self._facts: List[Tuple[str, str, str, np.ndarray]] = []  # (role, arg, value, vector)
    
    def _get_vector(self, name: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
        """Get or create a deterministic vector for a name."""
        if name not in cache:
            seed = hash(name) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim)
            vec = vec / np.linalg.norm(vec)
            cache[name] = vec
        return cache[name]
    
    def get_role(self, role: str) -> np.ndarray:
        """Get vector for a role (e.g., 'capital_of')."""
        return self._get_vector(role, self._role_vectors)
    
    def get_entity(self, entity: str) -> np.ndarray:
        """Get vector for an entity (e.g., 'paris')."""
        return self._get_vector(entity, self._entity_vectors)
    
    def add_relation(self, role: str, arg: str, value: str) -> None:
        """
        Add a relational fact: role(arg) = value
        
        Example: add_relation("capital_of", "france", "paris")
        Encodes: capital_of ⊛ france + paris
        """
        role_vec = self.get_role(role)
        arg_vec = self.get_entity(arg)
        value_vec = self.get_entity(value)
        
        # Encode as: role ⊛ arg bundled with value
        bound = bind(role_vec, arg_vec, self.method)
        fact_vec = bundle(bound, value_vec)
        
        self._facts.append((role, arg, value, fact_vec))
        self.cleanup.add(value, value_vec)
    
    def query(self, role: str, arg: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Query for value given role and argument.
        
        Example: query("capital_of", "france") → [("paris", 0.85), ...]
        """
        role_vec = self.get_role(role)
        arg_vec = self.get_entity(arg)
        
        # Create query pattern
        query_bound = bind(role_vec, arg_vec, self.method)
        
        # Find best matching facts
        results = []
        for fact_role, fact_arg, fact_value, fact_vec in self._facts:
            # Unbind to get candidate value
            candidate = unbind(fact_vec, query_bound, self.method)
            
            # Check similarity to known values
            matches = self.cleanup.cleanup_top_k(candidate, k=1)
            if matches:
                results.append(matches[0])
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for name, sim in sorted(results, key=lambda x: -x[1]):
            if name not in seen:
                seen.add(name)
                unique_results.append((name, sim))
        
        return unique_results[:k]
    
    def analogy(self, a1: str, b1: str, a2: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve analogy: a1:b1 :: a2:?
        
        Example: analogy("france", "paris", "germany") → [("berlin", 0.75), ...]
        """
        a1_vec = self.get_entity(a1)
        b1_vec = self.get_entity(b1)
        a2_vec = self.get_entity(a2)
        
        # Extract implicit relation: b1 ⊛ a1^(-1) ≈ b1 ⊛ a1
        relation = bind(b1_vec, a1_vec, self.method)
        
        # Apply to a2
        answer = bind(relation, a2_vec, self.method)
        
        return self.cleanup.cleanup_top_k(answer, k)


# =============================================================================
# SEQUENCE ENCODER
# =============================================================================

class SequenceEncoder:
    """
    Encode ordered sequences using VSA permutation binding.
    
    Sequences are encoded as:
        seq = bundle(bind(P^0, w1), bind(P^1, w2), bind(P^2, w3), ...)
    
    Where P is a position marker and P^n means permute(P, n).
    
    Example:
        >>> encoder = SequenceEncoder(dim=256)
        >>> seq_vec = encoder.encode(["the", "quick", "brown", "fox"])
        >>> word, sim = encoder.decode_position(seq_vec, 2)
        >>> print(word)  # "brown"
    """
    
    def __init__(self, dim: int = 256, method: BindingMethod = DEFAULT_METHOD):
        self.dim = dim
        self.method = method
        self.cleanup = CleanupMemory()
        self._word_vectors: Dict[str, np.ndarray] = {}
        
        # Position marker vector
        seed = hash("__POSITION_MARKER__") % (2**32)
        rng = np.random.default_rng(seed)
        self._position_marker = rng.standard_normal(dim)
        self._position_marker = self._position_marker / np.linalg.norm(self._position_marker)
    
    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get or create vector for a word."""
        word = word.lower()
        if word not in self._word_vectors:
            seed = hash(word) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim)
            vec = vec / np.linalg.norm(vec)
            self._word_vectors[word] = vec
            self.cleanup.add(word, vec)
        return self._word_vectors[word]
    
    def encode(self, words: List[str]) -> np.ndarray:
        """
        Encode a sequence of words into a single vector.
        
        The encoding preserves order - each position can be queried.
        """
        if not words:
            return np.zeros(self.dim)
        
        components = []
        for i, word in enumerate(words):
            word_vec = self._get_word_vector(word)
            pos_marker = permute(self._position_marker, i)
            bound = bind(pos_marker, word_vec, self.method)
            components.append(bound)
        
        return bundle(*components)
    
    def decode_position(self, seq_vec: np.ndarray, 
                        position: int) -> Tuple[Optional[str], float]:
        """
        Decode the word at a specific position.
        
        Returns (word, similarity) or (None, 0.0) if not found.
        """
        pos_marker = permute(self._position_marker, position)
        recovered = unbind(seq_vec, pos_marker, self.method)
        return self.cleanup.cleanup(recovered)
    
    def decode_all(self, seq_vec: np.ndarray, 
                   max_length: int = 20) -> List[Tuple[int, str, float]]:
        """
        Attempt to decode all positions in a sequence.
        
        Returns list of (position, word, similarity) for positions with
        similarity above a threshold.
        """
        results = []
        for i in range(max_length):
            word, sim = self.decode_position(seq_vec, i)
            if word is not None and sim > 0.1:  # Threshold for noise
                results.append((i, word, sim))
        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def is_dissimilar(bound: np.ndarray, a: np.ndarray, b: np.ndarray,
                  threshold: float = 0.3) -> bool:
    """
    Check if bound vector is dissimilar to both inputs.
    
    This is a key VSA property - binding should produce vectors
    that don't interfere with the inputs in superposition.
    """
    sim_a = abs(similarity(bound, a))
    sim_b = abs(similarity(bound, b))
    return sim_a < threshold and sim_b < threshold
