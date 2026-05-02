#!/usr/bin/env python3
"""
Visualize Emergent Depth Prediction Results

Shows RGB, true depth, and predicted depth side by side.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from experiments.unified_assembly.depth_emergence import (
    EmergentDepthPredictor,
    COCO_VAL_PATH,
    DEPTH_CACHE_PATH,
)


def visualize_emergence(n_train: int = 400, n_show: int = 6):
    """Visualize emergent depth prediction."""
    print("Training model...")
    
    # Load training data
    image_files = sorted(COCO_VAL_PATH.glob("*.jpg"))[:n_train + n_show]
    
    train_rgb = []
    train_depth = []
    train_ids = []
    
    for img_path in image_files[:n_train]:
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        depth_cache = DEPTH_CACHE_PATH / f"{img_path.stem}_depth.npy"
        if not depth_cache.exists():
            continue
        
        depth = np.load(depth_cache)
        if depth.max() > 1:
            depth = depth / 255.0
        
        train_rgb.append(rgb)
        train_depth.append(depth)
        train_ids.append(img_path.stem)
    
    # Train with higher resolution (16x16 grid = 256 patches per image)
    predictor = EmergentDepthPredictor(patch_size=8, grid_size=16)
    predictor.train(train_rgb, train_depth, train_ids)
    
    # Visualize test samples
    test_files = image_files[n_train:n_train + n_show]
    
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    
    for i, img_path in enumerate(test_files):
        # Load RGB
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Load true depth
        depth_cache = DEPTH_CACHE_PATH / f"{img_path.stem}_depth.npy"
        if not depth_cache.exists():
            continue
        
        true_depth = np.load(depth_cache)
        if true_depth.max() > 1:
            true_depth = true_depth / 255.0
        
        # Predict
        pred_depth = predictor.predict(rgb)
        
        # Resize true depth to match
        true_resized = predictor.patch_extractor._resize(true_depth, pred_depth.shape[0])
        
        # Compute error
        mae = np.mean(np.abs(pred_depth - true_resized))
        
        # Plot
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB: {img_path.name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_resized, cmap='plasma')
        axes[i, 1].set_title("True Depth (Depth Anything V2)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_depth, cmap='plasma')
        axes[i, 2].set_title(f"Predicted Depth (MAE: {mae:.3f})")
        axes[i, 2].axis('off')
    
    plt.suptitle("Emergent Depth Dimensions: RGB → Depth\n(Dimensions discovered via self-assembly, not predefined)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_emergence_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()
    
    return output_path


if __name__ == "__main__":
    visualize_emergence()
