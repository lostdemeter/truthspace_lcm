#!/usr/bin/env python3
"""
Final Comparison: All Geometric Colorizer Versions vs DDColor

Summary of approaches:
- V1: Random geometric encoder + geometric decoder → Gray (averaging)
- V2: DDColor encoder + geometric decoder (random queries) → MSE ~8k
- V4: DDColor encoder + DDColor queries + geometric color grid → MSE ~14k
- V5: DDColor encoder + DDColor queries + extracted colors → MSE ~8k

Key insight: The bottleneck is NOT the color vocabulary, it's the 
attention→color mapping. DDColor uses einsum + refine_net, not simple averaging.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


def create_final_comparison(image_path: str, output_path: str):
    """Create a comprehensive comparison image."""
    from phi_geometric.models.geometric_colorizer import GeometricColorizer
    from phi_geometric.models.geometric_colorizer_v2 import HybridColorizer
    from phi_geometric.models.geometric_colorizer_v5 import V5Colorizer, DDColorReference
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Target size for comparison
    target_w = 300
    target_h = int(H * target_w / W)
    
    def resize(img):
        return cv2.resize(img, (target_w, target_h))
    
    results = {}
    
    # DDColor reference
    print("  DDColor...")
    ddcolor = DDColorReference()
    results['ddcolor'] = resize(ddcolor.colorize(img_bgr))
    
    # V1: Random geometric
    print("  V1...")
    v1 = GeometricColorizer()
    v1.eval()
    results['v1'] = resize(v1.colorize(gray_bgr))
    
    # V2: DDColor encoder + geometric decoder
    print("  V2...")
    v2 = HybridColorizer()
    results['v2'] = resize(v2.colorize(img_bgr))
    
    # V5: DDColor encoder + DDColor queries + extracted colors
    print("  V5...")
    v5 = V5Colorizer()
    results['v5'] = resize(v5.colorize(img_bgr))
    
    # Compute MSEs
    mse_v1 = np.mean((results['ddcolor'].astype(float) - results['v1'].astype(float))**2)
    mse_v2 = np.mean((results['ddcolor'].astype(float) - results['v2'].astype(float))**2)
    mse_v5 = np.mean((results['ddcolor'].astype(float) - results['v5'].astype(float))**2)
    
    # Create comparison grid
    # Row 1: Original | Grayscale | DDColor (reference)
    # Row 2: V1 (random) | V2 (hybrid) | V5 (extracted)
    
    orig_r = resize(img_bgr)
    gray_r = resize(gray_bgr)
    
    row1 = np.hstack([orig_r, gray_r, results['ddcolor']])
    row2 = np.hstack([results['v1'], results['v2'], results['v5']])
    
    comparison = np.vstack([row1, row2])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels_row1 = ["Original", "Grayscale", "DDColor (ref)"]
    labels_row2 = [f"V1 Random", f"V2 MSE:{mse_v2:.0f}", f"V5 MSE:{mse_v5:.0f}"]
    
    for i, label in enumerate(labels_row1):
        cv2.putText(comparison, label, (i*target_w + 5, 20), font, 0.5, (255, 255, 255), 1)
    for i, label in enumerate(labels_row2):
        cv2.putText(comparison, label, (i*target_w + 5, target_h + 20), font, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, comparison)
    print(f"  Saved: {output_path}")
    
    return {'mse_v1': mse_v1, 'mse_v2': mse_v2, 'mse_v5': mse_v5}


def main():
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    test_images = list(coco_path.glob("*.jpg"))[:5]
    
    all_results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"final_{img_path.stem}.jpg"
        try:
            r = create_final_comparison(str(img_path), str(output_path))
            all_results.append(r)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        
        avg_v1 = np.mean([r['mse_v1'] for r in all_results])
        avg_v2 = np.mean([r['mse_v2'] for r in all_results])
        avg_v5 = np.mean([r['mse_v5'] for r in all_results])
        
        print(f"\nAverage MSE vs DDColor:")
        print(f"  V1 (random geometric):     {avg_v1:.0f}")
        print(f"  V2 (DDColor enc + geo):    {avg_v2:.0f}")
        print(f"  V5 (extracted colors):     {avg_v5:.0f}")
        
        print(f"\n" + "-" * 60)
        print("ANALYSIS")
        print("-" * 60)
        print("""
V1 produces gray because random encoder features average out.

V2 and V5 have similar MSE (~8k) because the bottleneck is NOT
the color vocabulary - it's the attention→color mapping mechanism.

DDColor uses:
  1. einsum('bchw,nc->bnhw', img_feats, color_embed(query_feats))
  2. refine_net to project 103 channels → 2 ab channels

Our geometric approach uses:
  1. Simple softmax attention over queries
  2. Weighted average of query colors

The difference: DDColor's colors depend on BOTH query features AND
image features through the einsum. Our approach only uses queries.

CONCLUSION: To match DDColor, we need to replicate the einsum
interaction, not just the color vocabulary.
""")


if __name__ == "__main__":
    main()
