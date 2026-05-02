#!/usr/bin/env python3
"""
Geometric Colorizer V5 - Use Extracted Color Vocabulary

V5 uses:
1. DDColor's encoder (semantic features)
2. DDColor's queries (learned vocabulary)  
3. EXTRACTED color mapping (from observing DDColor outputs)

This should match DDColor closely since we're using its actual learned colors.

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


class ExtractedColorDecoder(nn.Module):
    """
    Decoder using DDColor's queries and extracted color mapping.
    """
    
    def __init__(self, query_embed: torch.Tensor, color_tensor: torch.Tensor):
        super().__init__()
        
        self.n_queries = query_embed.shape[0]
        self.feature_dim = query_embed.shape[1]
        
        # DDColor's learned queries
        self.register_buffer('query_embed', query_embed)
        
        # Extracted color mapping [100, 2]
        self.register_buffer('query_colors', color_tensor)
        
        self.temperature = 0.07  # Match DDColor
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
        Returns:
            ab: [B, 2, H, W]
        """
        B, C, H, W = features.shape
        
        # Reshape: [B, H*W, C]
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Attention over queries
        scores = torch.matmul(feat_flat, self.query_embed.T)
        scores = scores / self.temperature
        attention = F.softmax(scores, dim=-1)
        
        # Blend extracted colors
        ab_flat = torch.matmul(attention, self.query_colors)
        
        # Reshape: [B, 2, H, W]
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


class V5Colorizer:
    """V5: DDColor encoder + DDColor queries + extracted colors."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()
    
    def _load_models(self):
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading V5...")
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        self.ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        self.ddcolor.eval()
        self.ddcolor = self.ddcolor.to(self.device)
        
        # Get DDColor's queries
        query_embed = self.ddcolor.decoder.color_decoder.query_embed.weight.detach()
        
        # Load extracted color mapping
        color_path = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/query_color_tensor.pt")
        color_tensor = torch.load(color_path, weights_only=True)
        
        # Scale colors to match DDColor's output distribution
        # Target saturation ratio ~1.0
        # Previous: 4.0/2.5 gave ratio 1.36 (oversaturated)
        # Reduce by 1/1.36 = 0.74
        color_tensor[:, 0] = color_tensor[:, 0] * 3.0
        color_tensor[:, 1] = color_tensor[:, 1] * 1.85
        
        # Create decoder
        self.decoder = ExtractedColorDecoder(query_embed, color_tensor)
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()
        
        print("  V5 loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = img_bgr.shape[:2]
        input_size = 512
        
        img_resized = cv2.resize(img_bgr, (input_size, input_size))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_3ch = np.stack([gray, gray, gray], axis=-1)
        
        tensor = torch.from_numpy(gray_3ch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        
        captured = {}
        def hook_fn(module, input, output):
            captured['features'] = input[1]
        
        hook = self.ddcolor.decoder.color_decoder.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = self.ddcolor(tensor)
            features = captured['features']
            ab = self.decoder(features)
        
        hook.remove()
        
        ab_np = ab[0].cpu().permute(1, 2, 0).numpy()
        
        # OpenCV LAB: L is 0-255 (not 0-100), a and b are 0-255 centered at 128
        L = gray.astype(np.float32)  # Already 0-255
        lab = np.zeros((input_size, input_size, 3), dtype=np.float32)
        lab[:, :, 0] = L
        
        # Non-linear expansion: boost extreme colors more than moderate ones
        # DDColor can reach a=65, b=77 but our weighted avg maxes at a=34, b=67
        # Expansion: f(x) = x * (1 + k * |x|/threshold)
        a_val = ab_np[:, :, 0]
        b_val = ab_np[:, :, 1]
        
        # Tuned expansion to match DDColor's range
        a_expanded = a_val * (1 + 0.9 * np.abs(a_val) / 35)  # Reach ~65
        b_expanded = b_val * (1 + 0.15 * np.abs(b_val) / 70)  # Reach ~77
        
        lab[:, :, 1] = a_expanded + 128
        lab[:, :, 2] = b_expanded + 128
        
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


def compare_v5(image_path: str, output_path: str):
    """Compare DDColor vs V5."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = cv2.resize(ddcolor.colorize(img_bgr), (W, H))
    
    print("  V5 (extracted colors)...")
    v5 = V5Colorizer()
    v5_result = cv2.resize(v5.colorize(img_bgr), (W, H))
    
    # Compute metrics
    mse = np.mean((ddcolor_result.astype(float) - v5_result.astype(float))**2)
    
    # Diff
    diff = np.clip(cv2.absdiff(ddcolor_result, v5_result) * 3, 0, 255).astype(np.uint8)
    
    # Comparison
    comparison = np.hstack([img_bgr, gray_bgr, ddcolor_result, v5_result, diff])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["Original", "Grayscale", "DDColor", f"V5 (MSE:{mse:.0f})", "Diff x3"]
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
        output_path = output_dir / f"v5_comparison_{img_path.stem}.jpg"
        try:
            mse = compare_v5(str(img_path), str(output_path))
            mses.append(mse)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if mses:
        print(f"\n{'='*50}")
        print(f"Average MSE V5: {np.mean(mses):.1f}")
        print(f"V5 = DDColor encoder + DDColor queries + EXTRACTED colors")
