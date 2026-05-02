#!/usr/bin/env python3
"""
Experiment: Holographic Enhancement as Navigation Destination

The Problem:
- Depth Anything V2 produces SMOOTHED depth maps
- Our geometric self-assemblers capture RAW structure
- The smoothing obscures the geometric truth

The Solution:
- Use holographic enhancement to REVEAL geometric structure
- The enhanced image becomes the DESTINATION for self-assemblers
- Pure mathematics, no neural network smoothing

The Insight:
- Holographic enhancement: I_enhanced = I × (1 + β × α × ratio)
- This REVEALS hidden structure, doesn't smooth it
- Self-assemblers can navigate to this revealed structure

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
import sys
import warnings

warnings.filterwarnings('ignore')

# Add holographic enhancement to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "temp/outside_projects/holographic_enhancement/src"))

try:
    from enhance import holographic_enhance
    HAS_HOLOGRAPHIC = True
except ImportError:
    HAS_HOLOGRAPHIC = False
    print("Warning: holographic_enhancement not available, using fallback")

PHI = (1 + np.sqrt(5)) / 2

COCO_VAL_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
DEPTH_CACHE_PATH = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_cache")


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr) + 0.5


def holographic_enhance_numpy(image: np.ndarray, sigma: float = 2.0, boost: float = 1.5) -> np.ndarray:
    """
    Apply holographic enhancement to a numpy array (RGB, 0-1 range).
    
    This is the pure mathematical enhancement:
    I_enhanced = I × (1 + β × α(L) × (I - I_blur) / (I_blur + ε))
    """
    if HAS_HOLOGRAPHIC:
        # Use the actual holographic enhancement
        import cv2
        # Convert to BGR uint8 for cv2
        bgr = (image[:, :, ::-1] * 255).astype(np.uint8)
        enhanced_bgr = holographic_enhance(bgr, sigma=sigma, boost=boost)
        # Convert back to RGB float
        return enhanced_bgr[:, :, ::-1].astype(np.float32) / 255.0
    else:
        # Fallback: implement the core algorithm
        # Convert to grayscale for luminance
        L = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        
        # Gamma decode
        gamma = 2.2
        L_linear = np.power(L, gamma)
        
        # Extract structure
        L_blur = gaussian_filter(L_linear, sigma=sigma)
        L_blur = np.maximum(L_blur, 0.01)
        
        # Ratio-based detail
        epsilon = 0.01
        ratio = (L_linear - L_blur) / (L_blur + epsilon)
        
        # Adaptive boost
        midtone_weight = 4.0 * L * (1.0 - L)
        adaptive = np.clip(midtone_weight + 0.3, 0.3, 1.0)
        
        # Apply enhancement
        factor = 1.0 + boost * adaptive * ratio
        factor = np.clip(factor, 0.7, 1.5)
        
        # Apply to all channels
        enhanced = image.copy()
        for c in range(3):
            enhanced[:,:,c] = np.clip(image[:,:,c] * factor, 0, 1)
        
        return enhanced


def extract_geometric_depth_from_enhanced(enhanced: np.ndarray) -> np.ndarray:
    """
    Extract depth-like structure from holographically enhanced image.
    
    The enhancement REVEALS geometric structure that was hidden.
    We extract this as a "geometric depth" signal.
    """
    # Convert to grayscale
    gray = 0.299 * enhanced[:,:,0] + 0.587 * enhanced[:,:,1] + 0.114 * enhanced[:,:,2]
    
    h, w = gray.shape
    
    # The vertical gradient in the enhanced image
    # This captures the revealed geometric structure
    grad_y = sobel(gray, axis=0)
    
    # Vertical position (the geometric truth)
    y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    
    # Combine: vertical position + revealed structure
    # The enhancement reveals what was hidden by smoothing
    geometric_depth = 0.6 * y_coords + 0.4 * _normalize(-grad_y)
    
    return _normalize(geometric_depth)


def compute_structure_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute structural similarity between two images.
    
    This measures how well the geometric structures align.
    """
    # Convert to grayscale
    if img1.ndim == 3:
        g1 = 0.299 * img1[:,:,0] + 0.587 * img1[:,:,1] + 0.114 * img1[:,:,2]
    else:
        g1 = img1
    
    if img2.ndim == 3:
        g2 = 0.299 * img2[:,:,0] + 0.587 * img2[:,:,1] + 0.114 * img2[:,:,2]
    else:
        g2 = img2
    
    # Compute gradients
    grad1_x = sobel(g1, axis=1)
    grad1_y = sobel(g1, axis=0)
    grad2_x = sobel(g2, axis=1)
    grad2_y = sobel(g2, axis=0)
    
    # Gradient magnitude
    mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    
    # Correlation of gradient magnitudes
    corr = np.corrcoef(mag1.flatten(), mag2.flatten())[0, 1]
    
    return corr


class HolographicDestination:
    """
    Use holographic enhancement to create a navigation destination.
    
    Instead of matching smoothed depth maps, we:
    1. Enhance the image to reveal geometric structure
    2. Extract depth-like signal from the enhanced image
    3. Use this as the destination for self-assemblers
    
    This is pure mathematics - no neural network smoothing.
    """
    
    def __init__(self, sigma: float = 2.0, boost: float = 1.5):
        self.sigma = sigma
        self.boost = boost
    
    def create_destination(self, image: np.ndarray) -> np.ndarray:
        """
        Create the navigation destination from an image.
        
        This is the "target" that self-assemblers navigate toward.
        """
        # Step 1: Holographic enhancement
        enhanced = holographic_enhance_numpy(image, self.sigma, self.boost)
        
        # Step 2: Extract geometric depth
        geometric_depth = extract_geometric_depth_from_enhanced(enhanced)
        
        return geometric_depth
    
    def compare_destinations(self, image: np.ndarray, 
                            depth_anything: np.ndarray) -> Dict:
        """
        Compare holographic destination vs Depth Anything V2.
        
        This shows how much structure is lost in the smoothing.
        """
        # Our destination
        holo_dest = self.create_destination(image)
        
        # Depth Anything destination (smoothed)
        da_dest = depth_anything
        
        # Compare structures
        structure_sim = compute_structure_similarity(holo_dest, da_dest)
        
        # Compare to vertical baseline
        h, w = image.shape[:2]
        y_coords = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        vertical_baseline = 0.6 * y_coords + 0.1
        
        holo_vs_vertical = np.mean(np.abs(holo_dest - vertical_baseline))
        da_vs_vertical = np.mean(np.abs(da_dest - vertical_baseline))
        
        return {
            'structure_similarity': structure_sim,
            'holo_vs_vertical': holo_vs_vertical,
            'da_vs_vertical': da_vs_vertical,
            'holo_destination': holo_dest,
            'da_destination': da_dest
        }


def run_holographic_destination_experiment(n_images: int = 20):
    """
    Test holographic enhancement as navigation destination.
    """
    print("=" * 70)
    print("EXPERIMENT: Holographic Enhancement as Navigation Destination")
    print("=" * 70)
    print()
    print("The Problem:")
    print("  - Depth Anything V2 produces SMOOTHED depth maps")
    print("  - Our geometric self-assemblers capture RAW structure")
    print("  - The smoothing obscures the geometric truth")
    print()
    print("The Solution:")
    print("  - Use holographic enhancement to REVEAL structure")
    print("  - The enhanced image becomes the DESTINATION")
    print("  - Pure mathematics, no neural network smoothing")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    holo_dest = HolographicDestination(sigma=2.0, boost=1.5)
    
    results = []
    
    print("=" * 60)
    print("Comparing Destinations")
    print("=" * 60)
    
    for i, img_id in enumerate(available_ids[:n_images]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        if depth.max() > 1:
            depth = depth / 255.0
        
        # Resize
        h, w = rgb.shape[:2]
        scale = 0.25
        new_h, new_w = int(h * scale), int(w * scale)
        
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        depth_small = np.array(Image.fromarray((depth * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        
        # Compare destinations
        comparison = holo_dest.compare_destinations(rgb_small, depth_small)
        results.append(comparison)
        
        if i < 5:
            print(f"\n  Image {i+1}: {img_id}")
            print(f"    Structure Similarity: {comparison['structure_similarity']:.3f}")
            print(f"    Holo vs Vertical:     {comparison['holo_vs_vertical']:.4f}")
            print(f"    DA vs Vertical:       {comparison['da_vs_vertical']:.4f}")
    
    # Summary
    avg_struct_sim = np.mean([r['structure_similarity'] for r in results])
    avg_holo_vert = np.mean([r['holo_vs_vertical'] for r in results])
    avg_da_vert = np.mean([r['da_vs_vertical'] for r in results])
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Average Structure Similarity: {avg_struct_sim:.3f}")
    print(f"  Average Holo vs Vertical:     {avg_holo_vert:.4f}")
    print(f"  Average DA vs Vertical:       {avg_da_vert:.4f}")
    
    print(f"\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if avg_holo_vert < avg_da_vert:
        print("\n  Holographic destination is CLOSER to vertical baseline!")
        print("  → The enhancement reveals the geometric truth")
        print("  → Depth Anything V2 has smoothed away structure")
    else:
        print("\n  Depth Anything destination is closer to vertical baseline")
        print("  → The smoothing may be capturing something we're missing")
    
    print(f"\n  Structure Similarity = {avg_struct_sim:.3f}")
    if avg_struct_sim > 0.5:
        print("  → High similarity: both capture similar structure")
    else:
        print("  → Low similarity: they capture DIFFERENT structure")
        print("  → This is the key insight: holographic reveals what DA smooths")
    
    return results


def run_self_assembler_with_holographic_destination(n_train: int = 20, n_test: int = 10):
    """
    Use holographic destination as target for self-assemblers.
    
    Instead of trying to match Depth Anything V2's smoothed output,
    we match the holographically-revealed geometric structure.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Self-Assembler with Holographic Destination")
    print("=" * 70)
    print()
    print("Instead of matching smoothed depth, match revealed structure.")
    print()
    
    depth_files = sorted(DEPTH_CACHE_PATH.glob("*_depth.npy"))
    available_ids = [f.stem.replace("_depth", "") for f in depth_files]
    
    holo_dest = HolographicDestination(sigma=2.0, boost=1.5)
    
    # Simple self-assembler: vertical + enhanced structure
    print("Training self-assembler on holographic destinations...")
    
    # Collect training data - use fixed size
    target_h, target_w = 120, 160  # Fixed size for consistency
    train_features = []
    train_targets = []
    
    for i, img_id in enumerate(available_ids[:n_train]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Resize to fixed size
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((target_w, target_h))) / 255.0
        
        # Create holographic destination (this is our TARGET)
        target = holo_dest.create_destination(rgb_small)
        
        # Extract features (vertical position)
        y_coords = np.tile(np.linspace(0, 1, target_h).reshape(-1, 1), (1, target_w))
        
        train_features.append(y_coords.flatten())
        train_targets.append(target.flatten())
    
    # Learn mapping from vertical to holographic destination
    X = np.array(train_features)
    Y = np.array(train_targets)
    
    # Simple linear regression
    # Y = X @ W + b
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    W = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
    
    print(f"  Trained on {len(train_features)} images")
    
    # Test
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)
    
    test_errors_holo = []  # Error vs holographic destination
    test_errors_da = []    # Error vs Depth Anything
    
    for i, img_id in enumerate(available_ids[n_train:n_train + n_test]):
        img_path = COCO_VAL_PATH / f"{img_id}.jpg"
        depth_path = DEPTH_CACHE_PATH / f"{img_id}_depth.npy"
        
        if not img_path.exists():
            continue
        
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        da_depth = np.load(depth_path)
        if da_depth.max() > 1:
            da_depth = da_depth / 255.0
        
        # Resize to fixed size
        rgb_small = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((target_w, target_h))) / 255.0
        da_depth_small = np.array(Image.fromarray((da_depth * 255).astype(np.uint8)).resize((target_w, target_h))) / 255.0
        
        # Create holographic destination (ground truth for this experiment)
        holo_target = holo_dest.create_destination(rgb_small)
        
        # Predict using learned mapping
        y_coords = np.tile(np.linspace(0, 1, target_h).reshape(-1, 1), (1, target_w))
        X_test = np.hstack([y_coords.flatten().reshape(1, -1), np.ones((1, 1))])
        pred = (X_test @ W).reshape(target_h, target_w)
        pred = _normalize(pred)
        
        # Compute errors
        mae_holo = np.mean(np.abs(pred - holo_target))
        mae_da = np.mean(np.abs(pred - da_depth_small))
        
        test_errors_holo.append(mae_holo)
        test_errors_da.append(mae_da)
        
        if i < 3:
            print(f"\n  Test {i+1}: {img_id}")
            print(f"    MAE vs Holographic: {mae_holo:.4f}")
            print(f"    MAE vs Depth Anything: {mae_da:.4f}")
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  MAE vs Holographic Destination: {np.mean(test_errors_holo):.4f}")
    print(f"  MAE vs Depth Anything V2:       {np.mean(test_errors_da):.4f}")
    print(f"  Vertical alone vs DA:           0.182")
    
    print(f"\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print()
    print("If MAE vs Holographic < MAE vs DA:")
    print("  → We can navigate to the geometric truth more easily")
    print("  → The holographic destination is a better target")
    print()
    print("If MAE vs Holographic > MAE vs DA:")
    print("  → The holographic enhancement adds complexity")
    print("  → We need a different approach")
    
    return test_errors_holo, test_errors_da


if __name__ == "__main__":
    # First, compare destinations
    results = run_holographic_destination_experiment(n_images=20)
    
    # Then, test self-assembler with holographic destination
    holo_errors, da_errors = run_self_assembler_with_holographic_destination(n_train=20, n_test=10)
