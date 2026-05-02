#!/usr/bin/env python3
"""
Geometric Colorizer V13 - DINOv2 + PEP Color Projection

V13 uses:
1. DINOv2 encoder (93-99% linear, φ-expressible per Doc 123)
2. PEP-extracted linear projection to ab colors (0.80-0.86 correlation)

This is the most "geometric" colorizer we can build using existing
reverse-engineered components.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2

from transformers import Dinov2Model

PHI = (1 + np.sqrt(5)) / 2


class PEPColorProjection(nn.Module):
    """
    PEP-extracted linear projection from DINOv2 features to ab colors.
    Achieves 0.80-0.86 correlation with DDColor output.
    """
    
    def __init__(self, projection_path: str):
        super().__init__()
        
        data = np.load(projection_path)
        W = data['W']  # [384, 2]
        b = data['b']  # [2]
        
        self.proj = nn.Linear(384, 2, bias=True)
        self.proj.weight.data = torch.from_numpy(W.T).float()
        self.proj.bias.data = torch.from_numpy(b).float()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, 384] DINOv2 patch features
        Returns:
            ab: [B, N, 2] color values
        """
        return self.proj(features)


class V13DINOv2PEPColorizer:
    """
    V13: DINOv2 encoder + PEP color projection.
    
    Components:
    - DINOv2: 93-99% linear, φ-expressible (Doc 123)
    - Color projection: PEP-extracted, 0.80-0.86 correlation
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 504  # 36 * 14 for clean patch grid
        self._load_models()
    
    def _load_models(self):
        print("Loading V13 (DINOv2 + PEP)...")
        
        # DINOv2 encoder
        self.encoder = Dinov2Model.from_pretrained('facebook/dinov2-small')
        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)
        
        # PEP color projection
        self.color_proj = PEPColorProjection(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/dinov2_to_ab.npz'
        )
        self.color_proj = self.color_proj.to(self.device)
        
        print("  V13 loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        
        img = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        # Resize to DINOv2 input size
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # Convert to grayscale RGB
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_gray_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
        
        # Normalize for DINOv2
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_gray_rgb - mean) / std
        
        tensor = torch.from_numpy(img_norm.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get DINOv2 features
            outputs = self.encoder(tensor)
            features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS
            
            # Project to colors
            ab = self.color_proj(features)  # [B, N, 2]
            
            # Reshape to spatial
            n_patches = features.shape[1]
            patch_dim = int(np.sqrt(n_patches))
            ab_spatial = ab.reshape(1, patch_dim, patch_dim, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
            
            # Upsample to input size
            ab_up = F.interpolate(ab_spatial, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        # Convert to numpy
        ab_np = ab_up[0].cpu().numpy().transpose(1, 2, 0)
        
        # Convert from normalized [-1, 1] back to LAB range
        # The PEP was trained on (LAB_ab / 255 * 2 - 1), so reverse: (ab + 1) / 2 * 255
        ab_lab = (ab_np + 1) / 2 * 255
        ab_lab = np.clip(ab_lab, 0, 255)  # Ensure valid range
        
        # Apply bilateral filter for smoothing
        ab_smooth = np.zeros_like(ab_lab)
        ab_smooth[:, :, 0] = cv2.bilateralFilter(ab_lab[:, :, 0].astype(np.float32), 9, 75, 75)
        ab_smooth[:, :, 1] = cv2.bilateralFilter(ab_lab[:, :, 1].astype(np.float32), 9, 75, 75)
        
        # Resize to original
        ab_resized = cv2.resize(ab_smooth, (width, height))
        
        # Combine with original L (convert L from [0,1] to [0,255])
        orig_l_255 = orig_l * 255
        output_lab = np.concatenate((orig_l_255, ab_resized), axis=-1).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        
        return output_bgr


class DDColorReference:
    def __init__(self):
        import sys
        sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')
        
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


def compare(image_path: str, output_path: str, v13: 'V13DINOv2PEPColorizer'):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    
    print("  V13 (DINOv2 + PEP)...")
    v13_result = v13.colorize(img_bgr)
    
    mse = np.mean((ddcolor_result.astype(float) - v13_result.astype(float))**2)
    
    def get_saturation(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean()
    
    ddcolor_sat = get_saturation(ddcolor_result)
    v13_sat = get_saturation(v13_result)
    
    top_row = np.hstack([img_bgr, gray_bgr])
    bottom_row = np.hstack([ddcolor_result, v13_result])
    comparison = np.vstack([top_row, bottom_row])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ("Original", (10, 30)),
        ("Grayscale", (W + 10, 30)),
        (f"DDColor (sat:{ddcolor_sat:.0f})", (10, H + 30)),
        (f"V13 DINOv2+PEP (sat:{v13_sat:.0f})", (W + 10, H + 30)),
    ]
    for label, pos in labels:
        cv2.putText(comparison, label, pos, font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, label, pos, font, 0.7, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, comparison)
    print(f"  MSE: {mse:.1f}, Sat - DD:{ddcolor_sat:.0f}, V13:{v13_sat:.0f}")
    
    return mse, v13_sat


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    v13 = V13DINOv2PEPColorizer()
    
    # Test on images NOT used for PEP training (skip first 30)
    test_images = list(coco_path.glob("*.jpg"))[30:35]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v13_dinov2_pep_{img_path.stem}.jpg"
        try:
            mse, sat = compare(str(img_path), str(output_path), v13)
            results.append((mse, sat))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        mses, sats = zip(*results)
        print(f"\n{'='*60}")
        print(f"V13 DINOv2 + PEP Results:")
        print(f"  Average MSE vs DDColor: {np.mean(mses):.1f}")
        print(f"  Average Saturation: {np.mean(sats):.0f}")
        print()
        print("Components (all reverse-engineered):")
        print("  - DINOv2: 93-99% linear, φ-expressible (Doc 123)")
        print("  - Color projection: PEP-extracted (0.80-0.86 correlation)")
        print()
        print("Path to fully geometric colorizer:")
        print("  1. Replace DINOv2 with φ-backbone (80x speedup, Doc 123)")
        print("  2. Color projection is already a single linear layer")
