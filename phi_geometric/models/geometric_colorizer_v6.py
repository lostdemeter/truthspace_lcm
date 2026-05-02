#!/usr/bin/env python3
"""
Geometric Colorizer V6 - Geometric Einsum Replacement

The key insight: DDColor's refine_net is a learned linear projection
from 100 query scores to 2 ab values. We can replace this with:

1. Extract the refine_net weights (100 -> 2 projection)
2. Use these as our "geometric" color vocabulary
3. Apply the projection directly without softmax

This should match DDColor exactly since we're using its actual weights.

If this works, we can then try to DERIVE these weights geometrically
using the resfrac dual-space alignment approach.

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

PHI = (1 + np.sqrt(5)) / 2


class DirectProjectionDecoder(nn.Module):
    """
    Use DDColor's refine_net weights directly as a linear projection.
    
    The refine_net expects 103 channels: 100 query scores + 3 from input image.
    """
    
    def __init__(self, query_embed: torch.Tensor, refine_weight: torch.Tensor, refine_bias: torch.Tensor,
                 mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        
        self.n_queries = query_embed.shape[0]
        self.feature_dim = query_embed.shape[1]
        
        # DDColor's queries
        self.register_buffer('query_embed', query_embed)
        
        # Full refine_net weights: [2, 103]
        self.register_buffer('projection', refine_weight)  # [2, 103]
        self.register_buffer('bias', refine_bias)  # [2]
        
        # Normalization params
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        self.temperature = 0.07
    
    def forward(self, features: torch.Tensor, input_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] - decoder features
            input_img: [B, 3, H, W] - normalized input image
        Returns:
            ab: [B, 2, H, W]
        """
        B, C, H, W = features.shape
        
        # Reshape features: [B, H*W, C]
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Compute query scores: [B, H*W, 100]
        scores = torch.matmul(feat_flat, self.query_embed.T) / self.temperature
        
        # Reshape input image: [B, H*W, 3]
        img_flat = input_img.permute(0, 2, 3, 1).reshape(B, H*W, 3)
        
        # Concatenate: [B, H*W, 103]
        combined = torch.cat([scores, img_flat], dim=-1)
        
        # Apply projection: [B, H*W, 2]
        ab_flat = torch.matmul(combined, self.projection.T) + self.bias
        
        # Reshape: [B, 2, H, W]
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


class V6Colorizer:
    """
    V6: Use DDColor's refine_net weights as direct projection.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()
    
    def _load_models(self):
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading V6...")
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        self.ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        self.ddcolor.eval()
        self.ddcolor = self.ddcolor.to(self.device)
        
        # Extract components
        cd = self.ddcolor.decoder.color_decoder
        query_embed = cd.query_embed.weight.detach()
        
        # Extract refine_net weights
        conv = self.ddcolor.refine_net[0][0]
        refine_weight = conv.weight.detach().squeeze()  # [2, 103]
        refine_bias = conv.bias.detach()  # [2]
        
        # Normalization params
        mean = self.ddcolor.mean
        std = self.ddcolor.std
        
        # Create decoder
        self.decoder = DirectProjectionDecoder(query_embed, refine_weight, refine_bias, mean, std)
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()
        
        # Store projection weights directly for use in colorize
        self.projection = refine_weight.to(self.device)
        self.bias = refine_bias.to(self.device)
        
        print("  V6 loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = img_bgr.shape[:2]
        input_size = 512
        
        img_resized = cv2.resize(img_bgr, (input_size, input_size))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_3ch = np.stack([gray, gray, gray], axis=-1)
        
        tensor = torch.from_numpy(gray_3ch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Hook the decoder output (out_feat) - the 100-channel feature map
        captured = {}
        def hook_decoder(module, input, output):
            captured['out_feat'] = output.detach()
        
        hook = self.ddcolor.decoder.register_forward_hook(hook_decoder)
        
        with torch.no_grad():
            # Normalize input like DDColor does
            normalized = (tensor - self.decoder.mean) / self.decoder.std
            
            _ = self.ddcolor(tensor)
            out_feat = captured['out_feat']  # [B, 100, H, W]
            
            # Concatenate with normalized input: [B, 103, H, W]
            coarse_input = torch.cat([out_feat, normalized], dim=1)
            
            # Apply refine_net projection directly
            B, C, H, W = coarse_input.shape
            coarse_flat = coarse_input.permute(0, 2, 3, 1).reshape(B, H*W, C)
            ab_flat = torch.matmul(coarse_flat, self.projection.T) + self.bias
            ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        hook.remove()
        
        ab_np = ab[0].cpu().permute(1, 2, 0).numpy()
        
        # OpenCV LAB format
        L = gray.astype(np.float32)
        lab = np.zeros((input_size, input_size, 3), dtype=np.float32)
        lab[:, :, 0] = L
        lab[:, :, 1] = ab_np[:, :, 0] + 128
        lab[:, :, 2] = ab_np[:, :, 1] + 128
        
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        colorized = cv2.resize(colorized, (orig_w, orig_h))
        
        return colorized


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


def compare_v6(image_path: str, output_path: str):
    """Compare DDColor vs V6."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = cv2.resize(ddcolor.colorize(img_bgr), (W, H))
    
    print("  V6 (direct projection)...")
    v6 = V6Colorizer()
    v6_result = cv2.resize(v6.colorize(img_bgr), (W, H))
    
    # Compute MSE
    mse = np.mean((ddcolor_result.astype(float) - v6_result.astype(float))**2)
    
    # Diff
    diff = np.clip(cv2.absdiff(ddcolor_result, v6_result) * 5, 0, 255).astype(np.uint8)
    
    # Comparison
    comparison = np.hstack([img_bgr, gray_bgr, ddcolor_result, v6_result, diff])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["Original", "Grayscale", "DDColor", f"V6 (MSE:{mse:.0f})", "Diff x5"]
    for i, label in enumerate(labels):
        cv2.putText(comparison, label, (i*W + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"  MSE: {mse:.1f}")
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
        output_path = output_dir / f"v6_comparison_{img_path.stem}.jpg"
        try:
            mse = compare_v6(str(img_path), str(output_path))
            mses.append(mse)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if mses:
        print(f"\n{'='*50}")
        print(f"Average MSE V6: {np.mean(mses):.1f}")
        print(f"V6 = DDColor encoder + DDColor queries + refine_net projection")
