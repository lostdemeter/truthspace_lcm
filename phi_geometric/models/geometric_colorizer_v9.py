#!/usr/bin/env python3
"""
Geometric Colorizer V9 - Fully Geometric Encoder

V9 replaces DDColor's ConvNeXt encoder with a geometric encoder
that uses classical CV features (Gabor, position encoding, local stats).

This is the first attempt at a fully geometric colorizer (no pretrained weights).

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

from phi_geometric.models.geometric_encoder import GeometricEncoder


class HookWrapper:
    """Wrapper to mimic DDColor's hook interface."""
    def __init__(self, device=None):
        self.feature = None
        self.device = device
    
    def set_feature(self, feat):
        """Set feature, ensuring it's on the correct device."""
        if self.device is not None:
            self.feature = feat.to(self.device)
        else:
            self.feature = feat


class GeometricEncoderWrapper(nn.Module):
    """
    Wrapper around GeometricEncoder that provides the same interface as DDColor's encoder.
    """
    
    def __init__(self, device=None):
        super().__init__()
        self.encoder = GeometricEncoder()
        self._device = device
        
        # Create hooks to store features (mimics DDColor interface)
        self.hooks = [HookWrapper(device) for _ in range(4)]
    
    def set_device(self, device):
        """Set device for all hooks."""
        self._device = device
        for hook in self.hooks:
            hook.device = device
    
    def forward(self, x: torch.Tensor):
        """Forward pass, storing features in hooks."""
        features = self.encoder(x)
        
        for i, feat in enumerate(features):
            self.hooks[i].set_feature(feat)
        
        return features[-1]  # Return last feature like DDColor


class V9Colorizer:
    """
    V9: Fully geometric colorizer with geometric encoder.
    
    Uses:
    - Geometric encoder (Gabor + position + local stats)
    - DDColor's decoder (for now - will replace later)
    - Probe-extracted projection
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self._load_models()
    
    def _load_models(self):
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading V9...")
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        # Load DDColor for decoder and refine_net
        self.ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        self.ddcolor.eval()
        self.ddcolor = self.ddcolor.to(self.device)
        
        # Replace encoder with geometric encoder
        self.geometric_encoder = GeometricEncoderWrapper(device=self.device)
        self.geometric_encoder = self.geometric_encoder.to(self.device)
        self.geometric_encoder.set_device(self.device)
        
        # Point decoder to use our geometric encoder's hooks
        # This is a bit hacky but allows us to reuse DDColor's decoder
        
        # Extract refine_net weights
        conv = self.ddcolor.refine_net[0][0]
        self.projection = conv.weight.detach().squeeze().to(self.device)
        self.bias = conv.bias.detach().to(self.device)
        
        # Normalization params
        self.mean = self.ddcolor.mean.to(self.device)
        self.std = self.ddcolor.std.to(self.device)
        
        print("  V9 loaded (geometric encoder)")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        
        # Prepare input (same as DDColor pipeline)
        img = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        
        tensor_gray_rgb = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        
        with torch.no_grad():
            # Normalize input
            normalized = (tensor_gray_rgb - self.mean) / self.std
            
            # Run geometric encoder
            self.geometric_encoder(normalized)
            
            # Temporarily replace DDColor encoder hooks with our geometric features
            original_hooks = self.ddcolor.encoder.hooks
            self.ddcolor.encoder.hooks = self.geometric_encoder.hooks
            
            # Run decoder (uses our geometric features via hooks)
            out_feat = self.ddcolor.decoder()
            
            # Restore original hooks
            self.ddcolor.encoder.hooks = original_hooks
            
            # Apply refine_net projection
            coarse_input = torch.cat([out_feat, normalized], dim=1)
            B, C, H, W = coarse_input.shape
            coarse_flat = coarse_input.permute(0, 2, 3, 1).reshape(B, H*W, C)
            ab_flat = torch.matmul(coarse_flat, self.projection.T) + self.bias
            output_ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        # Convert to output image
        output_ab_resized = (
            F.interpolate(output_ab, size=(height, width))[0]
            .float()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        
        output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
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


def compare_v9(image_path: str, output_path: str, v9: 'V9Colorizer'):
    """Compare DDColor vs V9 with visual output."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    
    # Get grayscale for reference
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    
    print("  V9 (geometric encoder)...")
    v9_result = v9.colorize(img_bgr)
    
    # Compute metrics
    mse = np.mean((ddcolor_result.astype(float) - v9_result.astype(float))**2)
    
    # Compute saturation for both
    def get_saturation(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean()
    
    ddcolor_sat = get_saturation(ddcolor_result)
    v9_sat = get_saturation(v9_result)
    
    # Create comparison image
    # Row 1: Original, Grayscale
    # Row 2: DDColor, V9
    top_row = np.hstack([img_bgr, gray_bgr])
    bottom_row = np.hstack([ddcolor_result, v9_result])
    
    # Resize to same width
    comparison = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ("Original", (10, 30)),
        ("Grayscale", (W + 10, 30)),
        (f"DDColor (sat:{ddcolor_sat:.0f})", (10, H + 30)),
        (f"V9 Geometric (sat:{v9_sat:.0f})", (W + 10, H + 30)),
    ]
    for label, pos in labels:
        cv2.putText(comparison, label, pos, font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, label, pos, font, 0.7, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, comparison)
    print(f"  MSE vs DDColor: {mse:.1f}")
    print(f"  Saturation - DDColor: {ddcolor_sat:.0f}, V9: {v9_sat:.0f}")
    print(f"  Saved: {output_path}")
    
    return mse, v9_sat


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    # Create V9 once
    v9 = V9Colorizer()
    
    # Test on several images
    test_images = list(coco_path.glob("*.jpg"))[:5]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v9_comparison_{img_path.stem}.jpg"
        try:
            mse, sat = compare_v9(str(img_path), str(output_path), v9)
            results.append((mse, sat))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        mses, sats = zip(*results)
        print(f"\n{'='*50}")
        print(f"V9 Results (Geometric Encoder):")
        print(f"  Average MSE vs DDColor: {np.mean(mses):.1f}")
        print(f"  Average Saturation: {np.mean(sats):.0f}")
