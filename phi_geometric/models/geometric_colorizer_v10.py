#!/usr/bin/env python3
"""
Geometric Colorizer V10 - Probe-Extracted Feature Mapping

V10 uses PEP to learn a mapping from geometric encoder features to DDColor's
encoder feature space. This allows us to use DDColor's decoder with our
geometric encoder.

Key insight: We can MEASURE the relationship between geometric features
and DDColor features, then use that mapping at inference time.

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


class FeatureMapper(nn.Module):
    """
    Maps geometric encoder features to DDColor encoder feature space.
    Uses probe-extracted weights.
    """
    
    def __init__(self, projection_path: str = None):
        super().__init__()
        
        # Projections for each level (will be loaded from file)
        self.proj0 = nn.Linear(96, 96, bias=True)
        self.proj1 = nn.Linear(192, 192, bias=True)
        self.proj2 = nn.Linear(384, 384, bias=True)
        self.proj3 = nn.Linear(768, 768, bias=True)
        
        if projection_path:
            self.load_projections(projection_path)
    
    def load_projections(self, path: str):
        """Load probe-extracted projections."""
        data = np.load(path)
        
        # For now, we only have level 0 projection
        W = torch.from_numpy(data['W']).float()
        b = torch.from_numpy(data['b']).float()
        
        self.proj0.weight.data = W
        self.proj0.bias.data = b
        
        # Initialize other levels with identity (will learn later)
        nn.init.eye_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.eye_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)
        nn.init.eye_(self.proj3.weight)
        nn.init.zeros_(self.proj3.bias)
    
    def forward(self, features: list) -> list:
        """Map geometric features to DDColor feature space."""
        f0, f1, f2, f3 = features
        
        # Apply projections (treating spatial dims as batch)
        B, C0, H0, W0 = f0.shape
        f0_flat = f0.permute(0, 2, 3, 1).reshape(-1, C0)
        f0_mapped = self.proj0(f0_flat).reshape(B, H0, W0, C0).permute(0, 3, 1, 2)
        
        B, C1, H1, W1 = f1.shape
        f1_flat = f1.permute(0, 2, 3, 1).reshape(-1, C1)
        f1_mapped = self.proj1(f1_flat).reshape(B, H1, W1, C1).permute(0, 3, 1, 2)
        
        B, C2, H2, W2 = f2.shape
        f2_flat = f2.permute(0, 2, 3, 1).reshape(-1, C2)
        f2_mapped = self.proj2(f2_flat).reshape(B, H2, W2, C2).permute(0, 3, 1, 2)
        
        B, C3, H3, W3 = f3.shape
        f3_flat = f3.permute(0, 2, 3, 1).reshape(-1, C3)
        f3_mapped = self.proj3(f3_flat).reshape(B, H3, W3, C3).permute(0, 3, 1, 2)
        
        return [f0_mapped, f1_mapped, f2_mapped, f3_mapped]


class HookWrapper:
    """Wrapper to mimic DDColor's hook interface."""
    def __init__(self):
        self.feature = None


class V10Colorizer:
    """
    V10: Geometric encoder with probe-extracted feature mapping.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self._load_models()
    
    def _load_models(self):
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading V10...")
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        # Load DDColor for decoder
        self.ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        self.ddcolor.eval()
        self.ddcolor = self.ddcolor.to(self.device)
        
        # Geometric encoder
        self.geo_encoder = GeometricEncoder()
        self.geo_encoder = self.geo_encoder.to(self.device)
        
        # Feature mapper (probe-extracted)
        self.mapper = FeatureMapper(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/geo_to_dd_projection.npz'
        )
        self.mapper = self.mapper.to(self.device)
        
        # Create hook wrappers for our mapped features
        self.hooks = [HookWrapper() for _ in range(4)]
        
        # Extract refine_net weights
        conv = self.ddcolor.refine_net[0][0]
        self.projection = conv.weight.detach().squeeze().to(self.device)
        self.bias = conv.bias.detach().to(self.device)
        
        # Normalization params
        self.mean = self.ddcolor.mean.to(self.device)
        self.std = self.ddcolor.std.to(self.device)
        
        print("  V10 loaded (geometric encoder + PEP mapping)")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        
        # Prepare input
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
            geo_features = self.geo_encoder(tensor_gray_rgb)
            
            # Map to DDColor feature space
            mapped_features = self.mapper(geo_features)
            
            # Store in hooks
            for i, feat in enumerate(mapped_features):
                self.hooks[i].feature = feat
            
            # Swap hooks temporarily
            original_hooks = self.ddcolor.encoder.hooks
            self.ddcolor.encoder.hooks = self.hooks
            
            # Run decoder
            out_feat = self.ddcolor.decoder()
            
            # Restore hooks
            self.ddcolor.encoder.hooks = original_hooks
            
            # Apply refine_net projection
            coarse_input = torch.cat([out_feat, normalized], dim=1)
            B, C, H, W = coarse_input.shape
            coarse_flat = coarse_input.permute(0, 2, 3, 1).reshape(B, H*W, C)
            ab_flat = torch.matmul(coarse_flat, self.projection.T) + self.bias
            output_ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        # Convert to output image
        output_ab_resized = (
            F.interpolate(output_ab, size=(height, width), mode='bilinear', align_corners=False)[0]
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


def compare_v10(image_path: str, output_path: str, v10: 'V10Colorizer'):
    """Compare DDColor vs V10 with visual output."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    
    print("  V10 (PEP mapping)...")
    v10_result = v10.colorize(img_bgr)
    
    mse = np.mean((ddcolor_result.astype(float) - v10_result.astype(float))**2)
    
    def get_saturation(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean()
    
    ddcolor_sat = get_saturation(ddcolor_result)
    v10_sat = get_saturation(v10_result)
    
    # Create comparison
    top_row = np.hstack([img_bgr, gray_bgr])
    bottom_row = np.hstack([ddcolor_result, v10_result])
    comparison = np.vstack([top_row, bottom_row])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ("Original", (10, 30)),
        ("Grayscale", (W + 10, 30)),
        (f"DDColor (sat:{ddcolor_sat:.0f})", (10, H + 30)),
        (f"V10 PEP (sat:{v10_sat:.0f})", (W + 10, H + 30)),
    ]
    for label, pos in labels:
        cv2.putText(comparison, label, pos, font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, label, pos, font, 0.7, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, comparison)
    print(f"  MSE vs DDColor: {mse:.1f}")
    print(f"  Saturation - DDColor: {ddcolor_sat:.0f}, V10: {v10_sat:.0f}")
    print(f"  Saved: {output_path}")
    
    return mse, v10_sat


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    v10 = V10Colorizer()
    
    # Test on images NOT used for PEP extraction (skip first 20)
    test_images = list(coco_path.glob("*.jpg"))[20:25]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v10_comparison_{img_path.stem}.jpg"
        try:
            mse, sat = compare_v10(str(img_path), str(output_path), v10)
            results.append((mse, sat))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        mses, sats = zip(*results)
        print(f"\n{'='*50}")
        print(f"V10 Results (Geometric Encoder + PEP Mapping):")
        print(f"  Average MSE vs DDColor: {np.mean(mses):.1f}")
        print(f"  Average Saturation: {np.mean(sats):.0f}")
