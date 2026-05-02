"""
Web Pattern Example: Colorization

The Web pattern is cross-connected (queries attend to features).
Used for: Colorization, segmentation, conditional generation.

Observed in: DDColor

Characteristics:
    - Learnable queries that attend to features
    - Cross-attention + self-attention
    - Multi-scale feature processing

This example builds a colorizer without training.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import Optional, Tuple

from ..core import (
    GeometricAI, ProblemSpec, IOSpec, DataType,
    PhiEncoder, Web
)


class WebColorizer:
    """
    A colorizer using the Web pattern.
    
    The Web pattern uses queries that attend to features
    at multiple scales, building up color information
    through cross-attention.
    
    Example:
        colorizer = WebColorizer(
            image_size=64,
            queries=100,
            layers=9
        )
        
        # Inject color knowledge
        colorizer.inject_knowledge("Sky is typically blue")
        colorizer.inject_knowledge("Grass is typically green")
        
        # Colorize
        ab_colors = colorizer.colorize(grayscale_features)
    """
    
    def __init__(
        self,
        image_size: int = 64,
        queries: int = 100,
        dim: int = 256,
        layers: int = 9,
        feature_scales: int = 3
    ):
        """
        Initialize the colorizer.
        
        Args:
            image_size: Size of input image (assumed square)
            queries: Number of color queries
            dim: Hidden dimension
            layers: Number of decoder layers
            feature_scales: Number of feature scales
        """
        self.image_size = image_size
        self.queries = queries
        self.dim = dim
        self.layers = layers
        self.feature_scales = feature_scales
        
        # Input: flattened grayscale, Output: ab channels
        input_dim = image_size * image_size
        output_dim = image_size * image_size * 2
        
        # Create problem specification
        self.problem = ProblemSpec(
            name="web_colorizer",
            inputs=[IOSpec("grayscale", DataType.IMAGE, (image_size, image_size, 1), "luminance")],
            outputs=[IOSpec("color", DataType.IMAGE, (image_size, image_size, 2), "ab channels")],
            hierarchical=True
        )
        
        # Create GeometricAI
        self.ai = GeometricAI(self.problem)
        
        # Inject default color knowledge
        self._inject_default_knowledge()
    
    def _inject_default_knowledge(self):
        """Inject default color knowledge."""
        self.ai.inject_knowledge("Dark regions tend to be neutral colors")
        self.ai.inject_knowledge("Bright regions can be any color")
        self.ai.inject_knowledge("Colors are spatially coherent")
    
    def inject_knowledge(self, fact: str):
        """Inject color knowledge."""
        self.ai.inject_knowledge(fact)
    
    def colorize(self, grayscale: torch.Tensor) -> torch.Tensor:
        """
        Colorize a grayscale image.
        
        Args:
            grayscale: Grayscale image [H, W] or [H, W, 1] or flattened
            
        Returns:
            ab color channels [H, W, 2] or flattened
        """
        # Flatten if needed
        if grayscale.dim() > 1:
            original_shape = grayscale.shape
            grayscale = grayscale.flatten()
        else:
            original_shape = None
        
        # Run through geometric AI
        output = self.ai(grayscale)
        
        # Reshape if needed
        if original_shape is not None and len(original_shape) >= 2:
            h, w = original_shape[0], original_shape[1]
            if output.numel() >= h * w * 2:
                output = output[:h * w * 2].reshape(h, w, 2)
        
        return output
    
    def compute_saturation(self, ab: torch.Tensor) -> float:
        """
        Compute average saturation.
        
        Args:
            ab: ab color channels
            
        Returns:
            Average saturation (0-1)
        """
        if ab.dim() < 2 or ab.shape[-1] != 2:
            # Reshape if flattened
            if ab.numel() % 2 == 0:
                ab = ab.reshape(-1, 2)
            else:
                return 0.0
        
        a = ab[..., 0]
        b = ab[..., 1]
        sat = torch.sqrt(a**2 + b**2) / 128
        return sat.mean().item()
    
    def colorize_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Colorize a batch of images.
        
        Args:
            images: Batch of grayscale images [B, H, W] or [B, H, W, 1]
            
        Returns:
            Batch of ab channels [B, H, W, 2]
        """
        results = []
        for img in images:
            result = self.colorize(img)
            results.append(result)
        return torch.stack(results)
    
    def stats(self):
        """Get colorizer statistics."""
        return self.ai.stats()


def demo_web_colorizer():
    """Demonstrate the Web colorizer."""
    print("=" * 70)
    print("WEB PATTERN EXAMPLE: Colorization")
    print("=" * 70)
    
    # Create colorizer
    colorizer = WebColorizer(
        image_size=8,  # Small for demo
        queries=10,
        dim=64,
        layers=3
    )
    
    # Inject color knowledge
    colorizer.inject_knowledge("Sky is typically blue (negative b)")
    colorizer.inject_knowledge("Grass is typically green (negative a)")
    colorizer.inject_knowledge("Skin tones are warm (positive a, positive b)")
    colorizer.inject_knowledge("Water reflects sky color")
    
    print("\nColorizer created:")
    print(f"  Image size: {colorizer.image_size}x{colorizer.image_size}")
    print(f"  Queries: {colorizer.queries}")
    print(f"  Layers: {colorizer.layers}")
    print(f"  Pattern: Web (cross-connected)")
    
    # Test colorization
    print("\n--- Colorization Tests ---")
    
    # Dark image
    dark = torch.zeros(8, 8) + 0.1
    dark_colors = colorizer.colorize(dark)
    print(f"  Dark image: output shape = {dark_colors.shape}")
    
    # Bright image
    bright = torch.zeros(8, 8) + 0.9
    bright_colors = colorizer.colorize(bright)
    print(f"  Bright image: output shape = {bright_colors.shape}")
    
    # Gradient image
    gradient = torch.linspace(0, 1, 64).reshape(8, 8)
    gradient_colors = colorizer.colorize(gradient)
    print(f"  Gradient image: output shape = {gradient_colors.shape}")
    
    # Saturation
    print("\n--- Saturation Analysis ---")
    for name, colors in [("Dark", dark_colors), ("Bright", bright_colors), ("Gradient", gradient_colors)]:
        sat = colorizer.compute_saturation(colors)
        print(f"  {name}: saturation = {sat:.4f}")
    
    # Stats
    print("\n--- Statistics ---")
    stats = colorizer.stats()
    print(f"  Pattern: {stats['pattern']}")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Facts: {stats['num_facts']}")
    print(f"  Memory hit rate: {stats['memory_hit_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("WEB EXAMPLE COMPLETE")
    print("=" * 70)
    
    return colorizer


if __name__ == "__main__":
    demo_web_colorizer()
