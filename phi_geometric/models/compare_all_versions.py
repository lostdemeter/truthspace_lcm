#!/usr/bin/env python3
"""
Compare all colorizer versions side by side.

Creates a single comparison image showing:
- Original
- Grayscale  
- DDColor (reference)
- V2 (DDColor enc + geo dec)
- V4 (DDColor enc + DDColor queries + geo colors)

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


def load_ddcolor():
    """Load DDColor reference."""
    from ddcolor import DDColor
    from ddcolor.pipeline import ColorizationPipeline
    from huggingface_hub import PyTorchModelHubMixin
    
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return ColorizationPipeline(model, input_size=512), model


def compare_single_image(img_path: str, output_path: str):
    """Create comparison for a single image."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {img_path}")
    
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Load DDColor
    print("  Loading DDColor...")
    pipeline, ddcolor_model = load_ddcolor()
    device = next(ddcolor_model.parameters()).device
    
    # Run DDColor
    print("  Running DDColor...")
    ddcolor_result = pipeline.process(img_bgr)
    ddcolor_result = cv2.resize(ddcolor_result, (W, H))
    
    # Run V2 (DDColor encoder + geometric decoder)
    print("  Running V2...")
    from phi_geometric.models.geometric_colorizer_v2 import HybridColorizer
    v2 = HybridColorizer()
    v2_result = cv2.resize(v2.colorize(img_bgr), (W, H))
    
    # Run V4 (DDColor encoder + DDColor queries + geometric colors)
    print("  Running V4...")
    from phi_geometric.models.geometric_colorizer_v4 import V4Colorizer
    v4 = V4Colorizer()
    v4_result = cv2.resize(v4.colorize(img_bgr), (W, H))
    
    # Compute metrics
    mse_v2 = np.mean((ddcolor_result.astype(float) - v2_result.astype(float))**2)
    mse_v4 = np.mean((ddcolor_result.astype(float) - v4_result.astype(float))**2)
    
    print(f"  MSE V2: {mse_v2:.1f}, MSE V4: {mse_v4:.1f}")
    
    # Create comparison image
    # Row 1: Original | Grayscale | DDColor
    # Row 2: V2 | V4 | Diff
    
    # Resize all to same size for comparison
    target_w = 400
    target_h = int(H * target_w / W)
    
    def resize(img):
        return cv2.resize(img, (target_w, target_h))
    
    orig_r = resize(img_bgr)
    gray_r = resize(gray_bgr)
    ddc_r = resize(ddcolor_result)
    v2_r = resize(v2_result)
    v4_r = resize(v4_result)
    
    # Diff images
    diff_v2 = np.clip(cv2.absdiff(ddc_r, v2_r) * 3, 0, 255).astype(np.uint8)
    diff_v4 = np.clip(cv2.absdiff(ddc_r, v4_r) * 3, 0, 255).astype(np.uint8)
    
    # Create rows
    row1 = np.hstack([orig_r, gray_r, ddc_r])
    row2 = np.hstack([v2_r, v4_r, diff_v4])
    
    # Stack rows
    comparison = np.vstack([row1, row2])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels_row1 = ["Original", "Grayscale", "DDColor"]
    labels_row2 = [f"V2 (MSE:{mse_v2:.0f})", f"V4 (MSE:{mse_v4:.0f})", "Diff V4 x3"]
    
    for i, label in enumerate(labels_row1):
        cv2.putText(comparison, label, (i*target_w + 10, 25), font, 0.6, (255, 255, 255), 2)
    for i, label in enumerate(labels_row2):
        cv2.putText(comparison, label, (i*target_w + 10, target_h + 25), font, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"  Saved: {output_path}")
    
    return {'mse_v2': mse_v2, 'mse_v4': mse_v4}


def main():
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    test_images = list(coco_path.glob("*.jpg"))[:5]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"full_comparison_{img_path.stem}.jpg"
        try:
            r = compare_single_image(str(img_path), str(output_path))
            results.append(r)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        avg_v2 = np.mean([r['mse_v2'] for r in results])
        avg_v4 = np.mean([r['mse_v4'] for r in results])
        print(f"Average MSE V2: {avg_v2:.1f}")
        print(f"Average MSE V4: {avg_v4:.1f}")
        print()
        print("V2 = DDColor encoder + geometric decoder (random queries)")
        print("V4 = DDColor encoder + DDColor queries + geometric color grid")


if __name__ == "__main__":
    main()
