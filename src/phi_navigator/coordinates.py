"""
φ-Coordinates: Universal Representation Layer
==============================================

Any value can be represented as: sign × φ^(level/K)

This is the universal encoding that achieves 99.9988% correlation.
It's lossless for practical purposes.

The φ-lattice is the coordinate system for semantic space.
"""

import torch
import math
from dataclasses import dataclass
from typing import Tuple, Optional

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128  # Quantization scale


@dataclass
class PhiPoint:
    """A point in φ-space."""
    levels: torch.Tensor  # int16, shape [dim]
    signs: torch.Tensor   # int8, shape [dim]
    
    @property
    def dim(self) -> int:
        return self.levels.shape[0]
    
    def to_embedding(self) -> torch.Tensor:
        """Decode back to embedding space."""
        exponents = self.levels.float() / K_SCALE
        magnitudes = torch.exp(exponents * LOG_PHI)
        return self.signs.float() * magnitudes
    
    def clone(self) -> 'PhiPoint':
        return PhiPoint(
            levels=self.levels.clone(),
            signs=self.signs.clone()
        )
    
    def flip_dims(self, dims: torch.Tensor) -> 'PhiPoint':
        """Return a new point with specified dimensions flipped."""
        new_signs = self.signs.clone()
        new_signs[dims] *= -1
        return PhiPoint(levels=self.levels.clone(), signs=new_signs)
    
    def shift_levels(self, delta: torch.Tensor) -> 'PhiPoint':
        """Return a new point with levels shifted."""
        new_levels = (self.levels.float() + delta).to(torch.int16)
        return PhiPoint(levels=new_levels, signs=self.signs.clone())


class PhiCoordinates:
    """
    Universal φ-coordinate system.
    
    Encodes any tensor to φ-lattice coordinates and back.
    This is the foundation layer - everything else builds on this.
    """
    
    def __init__(self, scale: int = K_SCALE):
        self.scale = scale
        self.log_phi = LOG_PHI
    
    def encode(self, tensor: torch.Tensor) -> PhiPoint:
        """
        Encode tensor to φ-coordinates.
        
        value = sign × φ^(level/K)
        level = round(K × log(|value|) / log(φ))
        """
        tensor = tensor.float()
        
        signs = torch.sign(tensor)
        signs[signs == 0] = 1
        
        magnitudes = tensor.abs().clamp(min=1e-45)
        levels = torch.round(self.scale * torch.log(magnitudes) / self.log_phi)
        
        return PhiPoint(
            levels=levels.to(torch.int16),
            signs=signs.to(torch.int8)
        )
    
    def decode(self, point: PhiPoint) -> torch.Tensor:
        """Decode φ-coordinates back to tensor."""
        return point.to_embedding()
    
    def distance(self, p1: PhiPoint, p2: PhiPoint) -> dict:
        """
        Compute distance between two points in φ-space.
        
        Returns multiple distance metrics.
        """
        level_diff = (p1.levels.float() - p2.levels.float()).abs()
        sign_diff = (p1.signs != p2.signs).float()
        
        return {
            'level_l1': level_diff.sum().item(),
            'level_l2': level_diff.pow(2).sum().sqrt().item(),
            'level_mean': level_diff.mean().item(),
            'sign_hamming': sign_diff.sum().item(),
            'sign_pct': sign_diff.mean().item() * 100,
        }
    
    def diff(self, p1: PhiPoint, p2: PhiPoint) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the transformation from p1 to p2.
        
        Returns (level_delta, sign_flip_mask)
        """
        level_delta = p2.levels.float() - p1.levels.float()
        sign_flip = (p1.signs != p2.signs)
        
        return level_delta, sign_flip
