#!/usr/bin/env python3
"""
Visualize Depth Learning Results

Shows RGB images alongside their true depth maps and predicted depth features.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from experiments.unified_assembly.depth_learning import (
    DepthEstimator,
    FeatureExtractor,
    GeometricDepthLearner,
    COCO_VAL_PATH,
    DEPTH_CACHE_PATH,
)


def visualize_depth_learning(n_train: int = 50, n_show: int = 6):
    """Visualize depth learning results."""
    print("Loading and training model...")
    
    # Get images
    image_files = sorted(COCO_VAL_PATH.glob("*.jpg"))[:n_train + n_show]
    train_files = image_files[:n_train]
    test_files = image_files[n_train:n_train + n_show]
    
    # Initialize
    depth_estimator = DepthEstimator()
    learner = GeometricDepthLearner()
    
    # Train
    for img_path in train_files:
        rgb_image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        depth_cache_file = DEPTH_CACHE_PATH / f"{img_path.stem}_depth.npy"
        if depth_cache_file.exists():
            depth = np.load(depth_cache_file)
        else:
            depth = depth_estimator.estimate(img_path)
            if depth is not None:
                np.save(depth_cache_file, depth)
        
        if depth is None:
            continue
        
        if depth.shape[:2] != rgb_image.shape[:2]:
            depth_pil = Image.fromarray((depth * 255).astype(np.uint8))
            depth_pil = depth_pil.resize((rgb_image.shape[1], rgb_image.shape[0]))
            depth = np.array(depth_pil).astype(np.float32) / 255.0
        
        rgb_features = FeatureExtractor.extract_rgb_features(rgb_image)
        depth_features = FeatureExtractor.extract_depth_features(depth)
        learner.add_sample(rgb_features, depth_features)
    
    learner.learn()
    
    # Visualize test samples
    n_cols = 3
    n_rows = n_show
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    for i, img_path in enumerate(test_files):
        # Load RGB
        rgb_image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Load depth
        depth_cache_file = DEPTH_CACHE_PATH / f"{img_path.stem}_depth.npy"
        if depth_cache_file.exists():
            depth = np.load(depth_cache_file)
        else:
            depth = depth_estimator.estimate(img_path)
        
        if depth is None:
            continue
        
        if depth.shape[:2] != rgb_image.shape[:2]:
            depth_pil = Image.fromarray((depth * 255).astype(np.uint8))
            depth_pil = depth_pil.resize((rgb_image.shape[1], rgb_image.shape[0]))
            depth = np.array(depth_pil).astype(np.float32) / 255.0
        
        # Get features
        rgb_features = FeatureExtractor.extract_rgb_features(rgb_image)
        true_depth_features = FeatureExtractor.extract_depth_features(depth)
        predicted_depth_features = learner.predict(rgb_features)
        
        # Calculate error
        eval_result = learner.evaluate(rgb_features, true_depth_features)
        mae = eval_result['mae']
        
        # Plot RGB
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"RGB: {img_path.name}")
        axes[i, 0].axis('off')
        
        # Plot depth
        axes[i, 1].imshow(depth, cmap='plasma')
        axes[i, 1].set_title("Depth (Depth Anything V2)")
        axes[i, 1].axis('off')
        
        # Plot feature comparison
        feature_names = list(true_depth_features.keys())[:5]  # Top 5
        true_vals = [true_depth_features[n] for n in feature_names]
        pred_vals = [predicted_depth_features[n] for n in feature_names]
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        axes[i, 2].bar(x - width/2, true_vals, width, label='True', alpha=0.8)
        axes[i, 2].bar(x + width/2, pred_vals, width, label='Predicted', alpha=0.8)
        axes[i, 2].set_xticks(x)
        axes[i, 2].set_xticklabels([n.replace('_', '\n') for n in feature_names], fontsize=7)
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_title(f"Features (MAE: {mae:.4f})")
        axes[i, 2].set_ylim(0, 1.1)
    
    plt.suptitle("Geometric Depth Learning: RGB → Depth Features\n(Using probe extraction: W = Y @ X @ (X^T X)^(-1))", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_learning_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()
    
    return output_path


if __name__ == "__main__":
    visualize_depth_learning()
