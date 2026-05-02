"""
Hourglass Pattern Example: Autoencoder

The Hourglass pattern is symmetric (compress → expand).
Used for: Autoencoders, U-Net, generation, reconstruction.

Characteristics:
    - Symmetric encoder-decoder structure
    - Bottleneck in the middle
    - Skip connections across the waist

This example builds an autoencoder without training.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import Optional, Tuple

from ..core import (
    GeometricAI, ProblemSpec, IOSpec, DataType,
    PhiEncoder, Hourglass
)


class HourglassAutoencoder:
    """
    An autoencoder using the Hourglass pattern.
    
    The Hourglass pattern compresses input to a bottleneck
    then expands back to the original size. Skip connections
    help preserve details.
    
    Example:
        ae = HourglassAutoencoder(
            input_dim=256,
            bottleneck_dim=32
        )
        
        # Inject reconstruction knowledge
        ae.inject_knowledge("Preserve important features through bottleneck")
        
        # Encode and decode
        reconstructed = ae.forward(input)
        latent = ae.encode(input)
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        bottleneck_dim: int = 32,
        hidden_dims: Optional[list] = None
    ):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Input/output dimension
            bottleneck_dim: Dimension of the bottleneck
            hidden_dims: Intermediate dimensions (auto-computed if None)
        """
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Auto-compute hidden dimensions
        if hidden_dims is None:
            hidden_dims = []
            dim = input_dim
            while dim > bottleneck_dim * 2:
                dim = dim // 2
                hidden_dims.append(dim)
        
        self.hidden_dims = hidden_dims
        
        # Create problem specification
        self.problem = ProblemSpec(
            name="hourglass_autoencoder",
            inputs=[IOSpec("input", DataType.VECTOR, (input_dim,), "input data")],
            outputs=[IOSpec("output", DataType.VECTOR, (input_dim,), "reconstructed data")],
            symmetric=True
        )
        
        # Create GeometricAI
        self.ai = GeometricAI(self.problem)
        
        # Inject default knowledge
        self._inject_default_knowledge()
    
    def _inject_default_knowledge(self):
        """Inject default autoencoder knowledge."""
        self.ai.inject_knowledge("Compress to essential features")
        self.ai.inject_knowledge("Reconstruct from compressed representation")
        self.ai.inject_knowledge("Bottleneck forces abstraction")
    
    def inject_knowledge(self, fact: str):
        """Inject reconstruction knowledge."""
        self.ai.inject_knowledge(fact)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass (encode + decode).
        
        Args:
            x: Input tensor [input_dim] or [B, input_dim]
            
        Returns:
            Reconstructed tensor
        """
        return self.ai(x)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode to bottleneck representation.
        
        Note: This is a simplified version that runs full forward
        and extracts the bottleneck. In practice, would stop at bottleneck.
        
        Args:
            x: Input tensor
            
        Returns:
            Bottleneck representation
        """
        output = self.ai(x)
        # Return first bottleneck_dim dimensions as approximation
        return output[..., :self.bottleneck_dim]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from bottleneck representation.
        
        Note: Simplified - pads bottleneck to input size and runs forward.
        
        Args:
            z: Bottleneck representation
            
        Returns:
            Reconstructed tensor
        """
        # Pad to input dimension
        if z.shape[-1] < self.input_dim:
            padding = torch.zeros(*z.shape[:-1], self.input_dim - z.shape[-1])
            z = torch.cat([z, padding], dim=-1)
        
        return self.ai(z)
    
    def reconstruction_error(self, x: torch.Tensor) -> float:
        """
        Compute reconstruction error.
        
        Args:
            x: Input tensor
            
        Returns:
            Mean squared error
        """
        reconstructed = self.forward(x)
        
        # Handle dimension mismatch
        min_dim = min(x.shape[-1], reconstructed.shape[-1])
        x_trimmed = x[..., :min_dim]
        r_trimmed = reconstructed[..., :min_dim]
        
        mse = ((x_trimmed - r_trimmed) ** 2).mean().item()
        return mse
    
    def stats(self):
        """Get autoencoder statistics."""
        return self.ai.stats()


def demo_hourglass_autoencoder():
    """Demonstrate the Hourglass autoencoder."""
    print("=" * 70)
    print("HOURGLASS PATTERN EXAMPLE: Autoencoder")
    print("=" * 70)
    
    # Create autoencoder
    ae = HourglassAutoencoder(
        input_dim=64,
        bottleneck_dim=8
    )
    
    # Inject knowledge
    ae.inject_knowledge("Important features should survive compression")
    ae.inject_knowledge("Noise should be filtered out")
    
    print("\nAutoencoder created:")
    print(f"  Input dim: {ae.input_dim}")
    print(f"  Bottleneck dim: {ae.bottleneck_dim}")
    print(f"  Hidden dims: {ae.hidden_dims}")
    print(f"  Pattern: Hourglass (symmetric)")
    
    # Test forward
    print("\n--- Reconstruction ---")
    x = torch.randn(64)
    reconstructed = ae.forward(x)
    print(f"  Input: {x.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")
    
    # Reconstruction error
    error = ae.reconstruction_error(x)
    print(f"  Reconstruction error: {error:.4f}")
    
    # Encode
    print("\n--- Encoding ---")
    z = ae.encode(x)
    print(f"  Latent: {z.shape}")
    print(f"  Compression ratio: {ae.input_dim / ae.bottleneck_dim:.1f}x")
    
    # Decode
    print("\n--- Decoding ---")
    decoded = ae.decode(z)
    print(f"  Decoded: {decoded.shape}")
    
    # Stats
    print("\n--- Statistics ---")
    stats = ae.stats()
    print(f"  Pattern: {stats['pattern']}")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Memory hit rate: {stats['memory_hit_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("HOURGLASS EXAMPLE COMPLETE")
    print("=" * 70)
    
    return ae


if __name__ == "__main__":
    demo_hourglass_autoencoder()
