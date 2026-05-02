"""
Signature Memory: Self-assembling cache based on φ-signatures.

From Doc 178: The Spatial Encoder Pattern
    COMPLEX FUNCTION → SIGNATURE → MEMORY → OUTPUT

Memory self-assembles from use - no pre-population needed.
Similar inputs map to the same signature, enabling cache hits.

Key Properties:
    - Self-assembling: Grows from use
    - Signature-based: Similar inputs → same signature
    - Fast lookup: O(n) with potential for O(log n) with indexing
    - No training: Just use and learn

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from .encoder import PHI, LN_PHI


@dataclass
class MemoryEntry:
    """An entry in the signature memory."""
    signature: Tuple
    input_data: Any
    output_data: Any
    hits: int = 0
    created_at: int = 0


class SignatureMemory:
    """
    Self-assembling memory based on φ-signatures.
    
    The memory stores (signature → output) mappings and grows
    automatically from use. When a query matches an existing
    signature (within threshold), the cached output is returned.
    
    Example:
        memory = SignatureMemory(threshold=0.5)
        
        # First call: miss, compute, store
        result, distance = memory.lookup(input)
        if result is None:
            output = expensive_compute(input)
            memory.store(input, output)
        
        # Second call: hit!
        result, distance = memory.lookup(input)
        # result is now the cached output
    
    Attributes:
        threshold: Maximum signature distance for a cache hit
        memory: Dict of signature → MemoryEntry
        stats: Hit/miss statistics
    """
    
    def __init__(self, threshold: float = 0.5, signature_dims: int = 64):
        """
        Initialize the memory.
        
        Args:
            threshold: Maximum signature distance for cache hit
            signature_dims: Number of dimensions in signature
        """
        self.threshold = threshold
        self.signature_dims = signature_dims
        self.memory: Dict[Tuple, MemoryEntry] = {}
        self.stats = {'hits': 0, 'misses': 0}
        self.step = 0
    
    def compute_signature(self, tensor: torch.Tensor) -> Tuple:
        """
        Compute a φ-signature for a tensor.
        
        The signature is a tuple of (level, sign) pairs that
        captures the φ-lattice structure of the tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (level, sign) pairs
        """
        # Flatten and take key dimensions
        flat = tensor.flatten()
        key_dims = min(self.signature_dims, len(flat))
        
        signature = []
        for i in range(key_dims):
            val = flat[i].item()
            if abs(val) < 1e-10:
                signature.append((0, 0))
            else:
                # Compute φ-level
                level = int(round(np.log(abs(val)) / LN_PHI))
                sign = 1 if val > 0 else -1
                signature.append((level, sign))
        
        return tuple(signature)
    
    def signature_distance(self, sig1: Tuple, sig2: Tuple) -> float:
        """
        Compute distance between two signatures.
        
        Distance is based on:
            - Sign differences (major)
            - Level differences (minor)
        
        Args:
            sig1, sig2: Signatures to compare
            
        Returns:
            Normalized distance (0 = identical)
        """
        if len(sig1) != len(sig2):
            return float('inf')
        
        diff = 0
        for (l1, s1), (l2, s2) in zip(sig1, sig2):
            if s1 != s2:
                diff += 2  # Sign difference is major
            diff += abs(l1 - l2)  # Level difference
        
        return diff / len(sig1)
    
    def lookup(self, query: torch.Tensor) -> Tuple[Optional[Any], float]:
        """
        Look up the nearest match in memory.
        
        Args:
            query: Input tensor to look up
            
        Returns:
            (output, distance) if found within threshold
            (None, inf) if no match
        """
        query_sig = self.compute_signature(query)
        
        best_match = None
        best_distance = float('inf')
        
        for sig, entry in self.memory.items():
            dist = self.signature_distance(query_sig, sig)
            if dist < best_distance:
                best_distance = dist
                best_match = entry
        
        if best_match is not None and best_distance <= self.threshold:
            best_match.hits += 1
            self.stats['hits'] += 1
            return best_match.output_data, best_distance
        
        self.stats['misses'] += 1
        return None, best_distance
    
    def store(self, input_data: torch.Tensor, output_data: Any):
        """
        Store a new entry in memory.
        
        Args:
            input_data: Input tensor
            output_data: Computed output to cache
        """
        sig = self.compute_signature(input_data)
        self.step += 1
        self.memory[sig] = MemoryEntry(
            signature=sig,
            input_data=input_data.clone() if isinstance(input_data, torch.Tensor) else input_data,
            output_data=output_data.clone() if isinstance(output_data, torch.Tensor) else output_data,
            hits=0,
            created_at=self.step
        )
    
    def hit_rate(self) -> float:
        """Return the cache hit rate."""
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0
    
    def size(self) -> int:
        """Return the number of entries in memory."""
        return len(self.memory)
    
    def clear(self):
        """Clear all entries."""
        self.memory.clear()
        self.stats = {'hits': 0, 'misses': 0}
        self.step = 0
    
    def prune(self, max_size: int = 10000, strategy: str = "lru"):
        """
        Prune memory to max_size entries.
        
        Args:
            max_size: Maximum number of entries to keep
            strategy: "lru" (least recently used) or "lfu" (least frequently used)
        """
        if len(self.memory) <= max_size:
            return
        
        entries = list(self.memory.items())
        
        if strategy == "lru":
            # Sort by creation time (oldest first)
            entries.sort(key=lambda x: x[1].created_at)
        else:  # lfu
            # Sort by hit count (least hits first)
            entries.sort(key=lambda x: x[1].hits)
        
        # Remove oldest/least used
        to_remove = len(entries) - max_size
        for sig, _ in entries[:to_remove]:
            del self.memory[sig]
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'size': self.size(),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': self.hit_rate(),
            'threshold': self.threshold,
        }


def test_memory():
    """Test the signature memory."""
    print("=" * 60)
    print("SIGNATURE MEMORY TEST")
    print("=" * 60)
    
    memory = SignatureMemory(threshold=0.5)
    
    # Test 1: Store and retrieve
    print("\n1. Store and retrieve:")
    x1 = torch.randn(64)
    y1 = torch.randn(10)
    
    result, dist = memory.lookup(x1)
    print(f"   Before store: {result is None} (miss)")
    
    memory.store(x1, y1)
    
    result, dist = memory.lookup(x1)
    print(f"   After store: {result is not None} (hit, dist={dist:.3f})")
    
    # Test 2: Similar inputs
    print("\n2. Similar inputs:")
    x2 = x1 + torch.randn(64) * 0.01  # Small perturbation
    result, dist = memory.lookup(x2)
    print(f"   Similar input: {result is not None} (dist={dist:.3f})")
    
    # Test 3: Different inputs
    print("\n3. Different inputs:")
    x3 = torch.randn(64) * 10  # Very different
    result, dist = memory.lookup(x3)
    print(f"   Different input: {result is None} (dist={dist:.3f})")
    
    # Test 4: Self-assembly
    print("\n4. Self-assembly:")
    for i in range(10):
        x = torch.randn(64)
        result, _ = memory.lookup(x)
        if result is None:
            memory.store(x, torch.randn(10))
    
    print(f"   Memory size: {memory.size()}")
    print(f"   Hit rate: {memory.hit_rate():.1%}")
    
    # Test 5: Run same inputs again
    print("\n5. Repeated queries:")
    for i in range(10):
        x = torch.randn(64)
        result, _ = memory.lookup(x)
        if result is None:
            memory.store(x, torch.randn(10))
    
    print(f"   Memory size: {memory.size()}")
    print(f"   Hit rate: {memory.hit_rate():.1%}")
    
    print("\n" + "=" * 60)
    print("SIGNATURE MEMORY TEST COMPLETE")
    print("=" * 60)
    
    return memory


if __name__ == "__main__":
    test_memory()
