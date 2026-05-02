#!/usr/bin/env python3
"""
Visualize Geometric Depth Prediction Results
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from experiments.unified_assembly.depth_geometric import (
    GeometricDepthSpace,
    COCO_VAL_PATH,
    DEPTH_CACHE_PATH,
)


def visualize_geometric(n_train: int = 400, n_show: int = 6):
    """Visualize geometric depth prediction."""
    print("Training geometric depth model...")
    
    # Get all images that have depth cache
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Initialize and train
    space = GeometricDepthSpace(patch_size=8, grid_size=8, n_dims=32)
    
    loaded = 0
    for img_id in available_ids[:n_train]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_cache = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_cache)
        if depth.max() > 1:
            depth = depth / 255.0
        
        space.add_training_pair(rgb, depth, img_id)
        loaded += 1
    
    print(f"  Loaded {loaded} images, {len(space.rgb_patches)} patches")
    space.learn()
    
    # Visualize - use remaining cached images for testing
    test_ids = available_ids[n_train:n_train + n_show]
    
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    
    for i, img_id in enumerate(test_ids):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        depth_cache = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        true_depth = np.load(depth_cache)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        pred_depth = space.predict(rgb)
        true_resized = space._resize_image(true_depth)
        
        mae = np.mean(np.abs(pred_depth - true_resized))
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB: {img_path.name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_resized, cmap='plasma')
        axes[i, 1].set_title("True Depth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_depth, cmap='plasma')
        axes[i, 2].set_title(f"Predicted (MAE: {mae:.3f})")
        axes[i, 2].axis('off')
    
    plt.suptitle(f"Geometric Depth: {n_train} training images\n(Holographic projection + probe extraction)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_geometric_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()
    return space


if __name__ == "__main__":
    visualize_geometric(n_train=400, n_show=6)
