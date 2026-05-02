#!/usr/bin/env python3
"""
Visualize φ-Holographic Depth Results
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from experiments.unified_assembly.depth_phi_holographic import (
    PhiHolographicDepth,
    COCO_VAL_PATH,
    DEPTH_CACHE_PATH,
)


def visualize_phi_holographic(n_train: int = 100, n_show: int = 6):
    """Visualize φ-holographic depth prediction."""
    print("Training φ-holographic depth model...")
    
    # Get available images
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    # Initialize and train
    estimator = PhiHolographicDepth(should_learn_weights=True)
    
    for img_id in available_ids[:n_train]:
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        estimator.add_training_pair(rgb, depth)
    
    print(f"  Loaded {len(estimator.training_pairs)} pairs")
    estimator.learn_weights(n_iterations=5)
    
    # Visualize
    test_ids = available_ids[n_train:n_train + n_show]
    
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    
    for i, img_id in enumerate(test_ids):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        true_depth = np.load(depth_path)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        pred_depth = estimator.predict(rgb)
        
        # Resize true depth to match
        h, w = pred_depth.shape
        true_resized = np.array(Image.fromarray(
            (true_depth * 255).astype(np.uint8)
        ).resize((w, h))).astype(np.float32) / 255.0
        
        mae = np.mean(np.abs(pred_depth - true_resized))
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB: {img_id}.jpg")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_resized, cmap='plasma')
        axes[i, 1].set_title("True Depth (Depth Anything V2)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_depth, cmap='plasma')
        axes[i, 2].set_title(f"φ-Holographic (MAE: {mae:.3f})")
        axes[i, 2].axis('off')
    
    # Get dimension contributions
    contributions = estimator.get_dimension_contributions(rgb)
    contrib_str = ", ".join([f"{k}: {v:.0%}" for k, v in 
                            sorted(contributions.items(), key=lambda x: -x[1])[:3]])
    
    plt.suptitle(f"φ-Holographic Depth: {n_train} training images\n"
                 f"Top dimensions: {contrib_str}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_phi_holographic_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    return estimator


if __name__ == "__main__":
    visualize_phi_holographic(n_train=100, n_show=6)
