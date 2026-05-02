"""
Bottleneck Filter: Validate outputs through the φ-bottleneck.

From Doc 204: Reverse Navigation for Novel Idea Generation
    The layer 27 bottleneck acts as a validity constraint.
    Only ideas that pass through φ-level ≈ 1.618 are valid.
    Invalid/impossible ideas cannot fit through the bottleneck.

Key Properties:
    - Geometric validity check
    - No external validation needed
    - Contradictory ideas automatically filtered
    - Based on singular value structure

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import Tuple, List, Optional

from .encoder import PHI


class BottleneckFilter:
    """
    Filter outputs through the φ-bottleneck for validity.
    
    The bottleneck checks if a tensor's geometric structure
    is close to φ = 1.618. Valid outputs have structure that
    converges to φ; invalid outputs diverge.
    
    Example:
        filter = BottleneckFilter(tolerance=0.3)
        
        is_valid, phi_level = filter.is_valid(output)
        if not is_valid:
            # Output may be invalid/contradictory
            pass
    
    From Doc 204:
        - Possible ideas: 0.101 distance from φ
        - Impossible ideas: 0.228 distance from φ
        - 2x difference in validity signal
    
    Attributes:
        target_phi: Target φ-level (default: 1.618)
        tolerance: Maximum distance from φ for validity
    """
    
    def __init__(self, target_phi: float = PHI, tolerance: float = 0.3):
        """
        Initialize the filter.
        
        Args:
            target_phi: Target φ-level for validity
            tolerance: Maximum distance from φ
        """
        self.target_phi = target_phi
        self.tolerance = tolerance
    
    def compute_phi_level(self, tensor: torch.Tensor) -> float:
        """
        Compute the φ-level of a tensor.
        
        The φ-level measures how close the tensor's structure
        is to the golden ratio. It's computed as the ratio of
        the first two singular values.
        
        Args:
            tensor: Input tensor
            
        Returns:
            φ-level (ideally ≈ 1.618 for valid outputs)
        """
        # Ensure 2D for SVD
        if tensor.dim() == 0:
            return 1.0
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        
        try:
            # Compute singular values
            S = torch.linalg.svdvals(tensor.float())
            
            if len(S) < 2:
                return 1.0
            
            # Ratio of first two singular values
            ratio = (S[0] / (S[1] + 1e-10)).item()
            
            # Clamp to reasonable range
            ratio = min(max(ratio, 0.5), 3.0)
            
            return ratio
        except:
            return 1.0
    
    def is_valid(self, tensor: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if a tensor passes the φ-bottleneck.
        
        Args:
            tensor: Output tensor to validate
            
        Returns:
            (is_valid, phi_level)
        """
        phi_level = self.compute_phi_level(tensor)
        distance = abs(phi_level - self.target_phi)
        is_valid = distance <= self.tolerance
        
        return is_valid, phi_level
    
    def validity_score(self, tensor: torch.Tensor) -> float:
        """
        Compute a validity score (0 to 1).
        
        Higher score = more valid (closer to φ).
        
        Args:
            tensor: Output tensor
            
        Returns:
            Validity score (1.0 = perfect, 0.0 = invalid)
        """
        phi_level = self.compute_phi_level(tensor)
        distance = abs(phi_level - self.target_phi)
        
        # Convert distance to score (exponential decay)
        score = max(0, 1 - distance / self.tolerance)
        
        return score
    
    def filter_candidates(
        self, 
        candidates: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Filter a list of candidates, returning only valid ones.
        
        Args:
            candidates: List of candidate tensors
            
        Returns:
            List of (tensor, phi_level, score) for valid candidates,
            sorted by closeness to φ
        """
        valid = []
        
        for candidate in candidates:
            is_valid, phi_level = self.is_valid(candidate)
            if is_valid:
                score = self.validity_score(candidate)
                valid.append((candidate, phi_level, score))
        
        # Sort by score (highest first)
        valid.sort(key=lambda x: x[2], reverse=True)
        
        return valid
    
    def adjust_for_validity(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adjust a tensor to improve its validity.
        
        This scales the tensor to bring its φ-level closer to
        the target. Note: This is a heuristic adjustment.
        
        Args:
            tensor: Tensor to adjust
            
        Returns:
            Adjusted tensor
        """
        phi_level = self.compute_phi_level(tensor)
        
        if abs(phi_level - self.target_phi) <= self.tolerance:
            return tensor  # Already valid
        
        # Scale to bring closer to φ
        scale = self.target_phi / (phi_level + 1e-10)
        scale = min(max(scale, 0.5), 2.0)  # Clamp
        
        return tensor * scale


def test_filter():
    """Test the bottleneck filter."""
    print("=" * 60)
    print("BOTTLENECK FILTER TEST")
    print("=" * 60)
    
    filter = BottleneckFilter(tolerance=0.3)
    
    # Test 1: Random tensors
    print("\n1. Random tensors:")
    for i in range(5):
        x = torch.randn(10, 20)
        is_valid, phi = filter.is_valid(x)
        score = filter.validity_score(x)
        print(f"   Tensor {i}: φ={phi:.3f}, valid={is_valid}, score={score:.3f}")
    
    # Test 2: Construct tensor with specific φ-level
    print("\n2. Constructed tensors:")
    
    # Create tensor with φ-like structure
    U = torch.randn(10, 10)
    V = torch.randn(20, 20)
    S = torch.zeros(10, 20)
    S[0, 0] = PHI
    S[1, 1] = 1.0
    for i in range(2, 10):
        S[i, i] = 1.0 / (PHI ** i)
    
    phi_tensor = U @ S @ V.T
    is_valid, phi = filter.is_valid(phi_tensor)
    print(f"   φ-structured: φ={phi:.3f}, valid={is_valid}")
    
    # Test 3: Filter candidates
    print("\n3. Filtering candidates:")
    candidates = [torch.randn(10, 20) for _ in range(10)]
    valid = filter.filter_candidates(candidates)
    print(f"   {len(valid)}/{len(candidates)} candidates passed")
    
    # Test 4: Adjust for validity
    print("\n4. Adjusting for validity:")
    x = torch.randn(10, 20)
    is_valid_before, phi_before = filter.is_valid(x)
    x_adjusted = filter.adjust_for_validity(x)
    is_valid_after, phi_after = filter.is_valid(x_adjusted)
    print(f"   Before: φ={phi_before:.3f}, valid={is_valid_before}")
    print(f"   After:  φ={phi_after:.3f}, valid={is_valid_after}")
    
    print("\n" + "=" * 60)
    print("BOTTLENECK FILTER TEST COMPLETE")
    print("=" * 60)
    
    return filter


if __name__ == "__main__":
    test_filter()
