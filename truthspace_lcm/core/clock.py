"""
12D Clock Phase Oracle
======================

Ported from holographersworkbench/practical_applications/ribbon_demos/fast_clock_predictor.py

The 12D clock provides deterministic phase relationships that can:
1. Replace learned attention patterns
2. Provide different "viewing angles" into semantic space
3. Enable phase-shift probing of LLM geometry

The key insight: these ratios create equidistributed phases with
long-range correlations (golden), medium-range patterns (silver),
and interference patterns that mimic attention.

11 functional dimensions + 1 zero-aligned = 12D tensor
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from functools import lru_cache


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618


# =============================================================================
# CLOCK RATIOS
# =============================================================================

CLOCK_RATIOS_6D = {
    'golden': PHI,                          # φ - long-range correlations
    'silver': 1 + np.sqrt(2),               # δ_S ≈ 2.414 - medium-range
    'bronze': (3 + np.sqrt(13)) / 2,        # ≈ 3.303
    'plastic': 1.324717957244746,           # Plastic constant
    'tribonacci': 1.839286755214161,        # Tribonacci constant
    'supergolden': 1.465571231876768,       # Supergolden ratio
}

CLOCK_RATIOS_12D = {
    **CLOCK_RATIOS_6D,
    'narayana': 1.465571231876768,          # Narayana's cows constant
    'copper': PHI + 1,                       # ≈ 2.618
    'nickel': np.sqrt(3),                    # ≈ 1.732
    'aluminum': (1 + np.sqrt(2)) / 2,        # ≈ 1.207
    'titanium': 2 ** (1/3),                  # ≈ 1.260
    'chromium': (1 + np.sqrt(13)) / 2,       # ≈ 2.303
}


# =============================================================================
# RECURSIVE THETA (Core Algorithm)
# =============================================================================

def recursive_theta(n: int, ratio: float = PHI) -> float:
    """
    O(log n) recursive clock phase computation.
    
    θ(n) = θ(n//2) + δ ± arctan(tan(θ(n//2)))
    
    This creates equidistributed phases with structure determined by
    the ratio parameter. Different ratios create different correlation
    patterns.
    
    Args:
        n: Index (position in sequence)
        ratio: Clock ratio (default: golden ratio)
    
    Returns:
        Phase in radians
    """
    if n <= 0:
        return 0.0
    prev = recursive_theta(n // 2, ratio)
    bit = n % 2
    delta = 2 * np.pi * ratio
    tan_prev = np.tan(prev % np.pi - np.pi/2 + 1e-10)
    if bit:
        return prev + delta + np.arctan(tan_prev)
    else:
        return prev + delta - np.arctan(tan_prev)


# =============================================================================
# CLOCK ORACLE
# =============================================================================

class ClockOracle:
    """
    12D Clock Phase Oracle with memoization.
    
    Provides O(1) lookup for precomputed phases, enabling fast
    phase-shift operations for semantic space exploration.
    
    Usage:
        oracle = ClockOracle()
        
        # Get 12D phase vector at position n
        phases = oracle.get_12d_phase(1000)
        
        # Compute attention-like similarity
        similarity = np.dot(phases_a, phases_b)
    """
    
    def __init__(self, max_n: int = 10000, use_12d: bool = True):
        """
        Initialize the oracle with precomputed phases.
        
        Args:
            max_n: Maximum index to precompute (higher = more memory, faster lookup)
            use_12d: Use 12D tensor (True) or 6D tensor (False)
        """
        self.max_n = max_n
        self.use_12d = use_12d
        self._ratios = CLOCK_RATIOS_12D if use_12d else CLOCK_RATIOS_6D
        
        # Precompute phases for O(1) lookup
        self._memo: Dict[str, np.ndarray] = {}
        self._precompute()
    
    def _precompute(self):
        """Precompute phases for all ratios up to max_n."""
        for name, ratio in self._ratios.items():
            phases = np.zeros(self.max_n + 1)
            for n in range(1, self.max_n + 1):
                phases[n] = (recursive_theta(n, ratio) / (2 * np.pi)) % 1.0
            self._memo[name] = phases
    
    def get_phase(self, n: int, clock_name: str = 'golden') -> float:
        """
        Get the n-th phase for the specified clock.
        
        Args:
            n: Position index
            clock_name: Which clock ratio to use
        
        Returns:
            Phase in radians [0, 2π)
        """
        return self.get_fractional_phase(n, clock_name) * 2 * np.pi
    
    def get_fractional_phase(self, n: int, clock_name: str = 'golden') -> float:
        """
        Get fractional phase in [0, 1).
        
        Args:
            n: Position index
            clock_name: Which clock ratio to use
        
        Returns:
            Fractional phase [0, 1)
        """
        ratio = self._ratios.get(clock_name, PHI)
        
        if clock_name in self._memo and 0 <= n <= self.max_n:
            return self._memo[clock_name][n]
        else:
            # Fall back to computation for out-of-range
            return (recursive_theta(n, ratio) / (2 * np.pi)) % 1.0
    
    def get_6d_phase(self, n: int) -> np.ndarray:
        """Get 6D phase vector at position n."""
        return np.array([
            self.get_fractional_phase(n, name) 
            for name in CLOCK_RATIOS_6D
        ])
    
    def get_12d_phase(self, n: int) -> np.ndarray:
        """Get 12D phase vector at position n."""
        return np.array([
            self.get_fractional_phase(n, name) 
            for name in CLOCK_RATIOS_12D
        ])
    
    def get_phase_vector(self, n: int) -> np.ndarray:
        """Get phase vector (6D or 12D based on configuration)."""
        if self.use_12d:
            return self.get_12d_phase(n)
        else:
            return self.get_6d_phase(n)
    
    # =========================================================================
    # PHASE-SHIFT OPERATIONS
    # =========================================================================
    
    def phase_similarity(self, n1: int, n2: int) -> float:
        """
        Compute similarity between two positions using phase vectors.
        
        This is analogous to attention: how much should position n1
        attend to position n2?
        
        Args:
            n1: First position
            n2: Second position
        
        Returns:
            Similarity score (dot product of phase vectors)
        """
        v1 = self.get_phase_vector(n1)
        v2 = self.get_phase_vector(n2)
        return np.dot(v1, v2)
    
    def phase_shift_query(self, base_vector: np.ndarray, shift: int) -> np.ndarray:
        """
        Apply a phase shift to a vector.
        
        This provides different "viewing angles" into semantic space.
        
        Args:
            base_vector: Original vector
            shift: Phase shift amount (position offset)
        
        Returns:
            Phase-shifted vector
        """
        shift_vector = self.get_phase_vector(shift)
        # Modulate base vector by shift phases
        return base_vector * np.cos(2 * np.pi * shift_vector)
    
    def attention_weights(self, position: int, context_length: int) -> np.ndarray:
        """
        Compute attention-like weights over a context window.
        
        This replaces learned attention with deterministic phase-based
        attention patterns.
        
        Args:
            position: Current position
            context_length: Number of previous positions to attend to
        
        Returns:
            Attention weights (sums to 1)
        """
        if context_length == 0:
            return np.array([])
        
        weights = np.zeros(context_length)
        current_phase = self.get_phase_vector(position)
        
        for i in range(context_length):
            context_pos = position - context_length + i
            if context_pos > 0:
                context_phase = self.get_phase_vector(context_pos)
                # Similarity = dot product
                similarity = np.dot(current_phase, context_phase)
                # Distance decay
                dist = context_length - i
                decay = 1.0 / (1.0 + 0.1 * dist)
                weights[i] = similarity * decay
        
        # Softmax normalization
        weights = weights - weights.max()
        weights = np.exp(weights)
        weights = weights / (weights.sum() + 1e-10)
        
        return weights


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global oracle instance (lazy initialization)
_global_oracle: Optional[ClockOracle] = None

def get_oracle(max_n: int = 10000) -> ClockOracle:
    """Get or create global oracle instance."""
    global _global_oracle
    if _global_oracle is None or _global_oracle.max_n < max_n:
        _global_oracle = ClockOracle(max_n=max_n)
    return _global_oracle


def phase_at(n: int, clock: str = 'golden') -> float:
    """Quick access to phase at position n."""
    return get_oracle().get_fractional_phase(n, clock)


def phase_vector(n: int) -> np.ndarray:
    """Quick access to 12D phase vector at position n."""
    return get_oracle().get_12d_phase(n)


def phase_similarity(n1: int, n2: int) -> float:
    """Quick access to phase similarity."""
    return get_oracle().phase_similarity(n1, n2)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("12D CLOCK PHASE ORACLE")
    print("=" * 60)
    
    oracle = ClockOracle(max_n=1000)
    
    print("\nClock Ratios:")
    for name, ratio in CLOCK_RATIOS_12D.items():
        print(f"  {name:12}: {ratio:.6f}")
    
    print("\n" + "-" * 60)
    print("Phase Vectors at Different Positions")
    print("-" * 60)
    
    for n in [1, 10, 100, 500, 1000]:
        phases = oracle.get_12d_phase(n)
        print(f"\nn={n}:")
        print(f"  phases = [{', '.join(f'{p:.3f}' for p in phases[:6])}...]")
    
    print("\n" + "-" * 60)
    print("Phase Similarities")
    print("-" * 60)
    
    pairs = [(1, 2), (1, 10), (1, 100), (100, 101), (100, 200)]
    for n1, n2 in pairs:
        sim = oracle.phase_similarity(n1, n2)
        print(f"  sim({n1}, {n2}) = {sim:.4f}")
    
    print("\n" + "-" * 60)
    print("Attention Weights (position=100, context=10)")
    print("-" * 60)
    
    weights = oracle.attention_weights(100, 10)
    print(f"  weights = [{', '.join(f'{w:.3f}' for w in weights)}]")
    print(f"  sum = {weights.sum():.4f}")
    
    print("\n" + "=" * 60)
