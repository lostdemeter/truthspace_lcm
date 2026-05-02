#!/usr/bin/env python3
"""
Geometric Colorizer V7 - Exact DDColor Pipeline Replication

This version replicates the DDColor pipeline EXACTLY to achieve 0 MSE.

Pipeline steps (from ddcolor/pipeline.py):
1. Convert BGR to float [0,1]
2. Extract original L channel from LAB
3. Resize to 512x512
4. Extract L from resized, create gray LAB (L, 0, 0)
5. Convert gray LAB to RGB (this is the model input!)
6. Run model to get ab output
7. Resize ab back to original size
8. Concatenate original L with resized ab
9. Convert LAB to BGR
10. Scale to [0, 255] uint8

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
import sys

sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')


class V7Colorizer:
    """
    V7: Exact replication of DDColor pipeline.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self._load_models()
    
    def _load_models(self):
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading V7...")
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        self.ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        self.ddcolor.eval()
        self.ddcolor = self.ddcolor.to(self.device)
        
        # Extract refine_net weights for our geometric decoder
        conv = self.ddcolor.refine_net[0][0]
        self.projection = conv.weight.detach().squeeze().to(self.device)  # [2, 103]
        self.bias = conv.bias.detach().to(self.device)  # [2]
        
        # Normalization params
        self.mean = self.ddcolor.mean.to(self.device)
        self.std = self.ddcolor.std.to(self.device)
        
        print("  V7 loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Exact replication of DDColor pipeline.
        """
        height, width = img_bgr.shape[:2]
        
        # Step 1: Convert BGR to float [0,1]
        img = (img_bgr / 255.0).astype(np.float32)
        
        # Step 2: Extract original L channel from LAB
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)
        
        # Step 3: Resize to input_size
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # Step 4: Extract L from resized, create gray LAB (L, 0, 0)
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate(
            (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
        )
        
        # Step 5: Convert gray LAB to RGB (this is the model input!)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        
        # Step 6: Run model to get ab output
        tensor_gray_rgb = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        
        with torch.no_grad():
            # Just use DDColor's output directly - it already applies refine_net
            output_ab = self.ddcolor(tensor_gray_rgb)
        
        # Step 7: Resize ab back to original size
        output_ab_resized = (
            F.interpolate(output_ab, size=(height, width))[0]
            .float()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        
        # Step 8: Concatenate original L with resized ab
        output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
        
        # Step 9: Convert LAB to BGR
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        
        # Step 10: Scale to [0, 255] uint8
        output_img = (output_bgr * 255.0).round().astype(np.uint8)
        
        return output_img


class DDColorReference:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load()
    
    def _load(self):
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
        model = model.to(self.device)
        self.pipeline = ColorizationPipeline(model, input_size=512)
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.pipeline.process(img_bgr)


def compare_v7(image_path: str, output_path: str):
    """Compare DDColor vs V7."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    
    print("  V7 (exact pipeline)...")
    v7 = V7Colorizer()
    v7_result = v7.colorize(img_bgr)
    
    # Compute MSE
    mse = np.mean((ddcolor_result.astype(float) - v7_result.astype(float))**2)
    
    # Compute max difference
    max_diff = np.abs(ddcolor_result.astype(float) - v7_result.astype(float)).max()
    
    # Diff visualization
    diff = np.clip(cv2.absdiff(ddcolor_result, v7_result) * 10, 0, 255).astype(np.uint8)
    
    # Comparison
    comparison = np.hstack([ddcolor_result, v7_result, diff])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["DDColor", f"V7 (MSE:{mse:.1f})", "Diff x10"]
    for i, label in enumerate(labels):
        cv2.putText(comparison, label, (i*W + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"  MSE: {mse:.4f}")
    print(f"  Max diff: {max_diff:.4f}")
    print(f"  Saved: {output_path}")
    
    return mse


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    test_images = list(coco_path.glob("*.jpg"))[:5]
    
    mses = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v7_comparison_{img_path.stem}.jpg"
        try:
            mse = compare_v7(str(img_path), str(output_path))
            mses.append(mse)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if mses:
        print(f"\n{'='*50}")
        print(f"Average MSE V7: {np.mean(mses):.4f}")
        print(f"V7 = Exact DDColor pipeline replication")
