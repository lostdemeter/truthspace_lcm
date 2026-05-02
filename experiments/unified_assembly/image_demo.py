#!/usr/bin/env python3
"""
Visual Demo: Image Transforms as Dimensions

This script generates a visual grid showing the original image
and all transformed versions, saved as a PNG file.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.unified_assembly.image_adapter import (
    IMAGE_TRANSFORMS,
    ImageCorpus,
)


def create_test_image(size: int = 128) -> np.ndarray:
    """Create a colorful test image with gradients and patterns."""
    h, w = size, size
    img = np.zeros((h, w, 3))
    
    # Create a more interesting test pattern
    for i in range(h):
        for j in range(w):
            # Radial gradient
            cx, cy = w / 2, h / 2
            r = np.sqrt((j - cx)**2 + (i - cy)**2) / (size / 2)
            
            # Color based on position and radius
            img[i, j, 0] = 0.3 + 0.4 * np.sin(j / 10) + 0.3 * (1 - r)  # Red
            img[i, j, 1] = 0.3 + 0.4 * np.sin(i / 10) + 0.3 * r        # Green
            img[i, j, 2] = 0.5 + 0.3 * np.cos((i + j) / 15)            # Blue
    
    # Add some shapes
    # Circle in center
    for i in range(h):
        for j in range(w):
            cx, cy = w / 2, h / 2
            r = np.sqrt((j - cx)**2 + (i - cy)**2)
            if 20 < r < 30:
                img[i, j] = [0.9, 0.7, 0.2]  # Yellow ring
    
    # Rectangle
    img[10:30, 10:50] = [0.2, 0.6, 0.9]  # Blue rectangle
    
    # Triangle-ish
    for i in range(20):
        img[h-30+i, w-40:w-40+i*2] = [0.9, 0.3, 0.3]  # Red triangle
    
    return np.clip(img, 0, 1)


def generate_visual_demo():
    """Generate a visual grid of all transforms."""
    print("Generating visual demo of image transforms...")
    
    # Create test image
    original = create_test_image(128)
    
    # Get all transforms
    transform_names = list(IMAGE_TRANSFORMS.keys())
    n_transforms = len(transform_names)
    
    # Calculate grid size
    n_cols = 4
    n_rows = (n_transforms + 2) // n_cols + 1  # +1 for original
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    # Plot original
    axes[0].imshow(original)
    axes[0].set_title("ORIGINAL", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Plot each transform
    for idx, name in enumerate(transform_names):
        ax = axes[idx + 1]
        
        try:
            transformed = IMAGE_TRANSFORMS[name](original.copy())
            ax.imshow(np.clip(transformed, 0, 1))
            ax.set_title(name.upper(), fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{name} (ERROR)", fontsize=10, color='red')
        
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_transforms + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Image Transforms as Dimensions\n(All have Δ = φ = 1.618)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to file
    output_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/image_transforms_demo.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    # Also show
    plt.show()
    
    return output_path


if __name__ == "__main__":
    generate_visual_demo()
