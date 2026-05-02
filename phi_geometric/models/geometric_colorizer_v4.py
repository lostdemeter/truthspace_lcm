#!/usr/bin/env python3
"""
Geometric Colorizer V4 - Extract DDColor's Actual Color Mapping

The key insight: DDColor's color_embed MLP maps query features to colors.
We can extract this mapping and use it directly.

V4 approach:
1. Use DDColor's encoder (semantic features)
2. Use DDColor's queries (learned vocabulary)
3. Use DDColor's color_embed (learned color mapping)
4. Replace only the attention mechanism with geometric MESH

This isolates: Can we replace attention with pre-computed MESH?

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


class MESHDecoder(nn.Module):
    """
    Use pre-computed MESH matrix instead of runtime attention.
    
    Simple approach: 
    1. Compute attention between features and queries
    2. Map attention weights directly to colors using a learned color table
    """
    
    def __init__(self, query_embed: torch.Tensor):
        super().__init__()
        
        self.n_queries = query_embed.shape[0]
        self.feature_dim = query_embed.shape[1]
        
        # Store query embeddings
        self.register_buffer('query_embed', query_embed)
        
        # Learn a color for each query: [100, 2] for ab channels
        # Initialize with a grid in LAB space
        colors = torch.zeros(self.n_queries, 2)
        a_vals = torch.linspace(-50, 50, 10)
        b_vals = torch.linspace(-50, 50, 10)
        for i in range(self.n_queries):
            colors[i, 0] = a_vals[i % 10]
            colors[i, 1] = b_vals[i // 10]
        self.query_colors = nn.Parameter(colors)
        
        self.temperature = 0.07
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] - image features from encoder
        Returns:
            ab: [B, 2, H, W] - predicted ab channels
        """
        B, C, H, W = features.shape
        
        # Reshape features: [B, H*W, C]
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Compute attention: features attend to queries
        scores = torch.matmul(feat_flat, self.query_embed.T)  # [B, H*W, 100]
        scores = scores / self.temperature
        attention = F.softmax(scores, dim=-1)
        
        # Blend colors based on attention: [B, H*W, 2]
        ab_flat = torch.matmul(attention, self.query_colors)
        
        # Reshape: [B, 2, H, W]
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


class V4Colorizer:
    """
    V4: Use DDColor's learned components with simplified attention.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()
    
    def _load_models(self):
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading DDColor for V4...")
        
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
        
        # Create MESH decoder
        self.decoder = MESHDecoder(query_embed)
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()
        
        print("  V4 loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = img_bgr.shape[:2]
        input_size = 512
        
        img_resized = cv2.resize(img_bgr, (input_size, input_size))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_3ch = np.stack([gray, gray, gray], axis=-1)
        
        tensor = torch.from_numpy(gray_3ch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Capture features
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
        
        L = gray.astype(np.float32) * 100 / 255
        lab = np.zeros((input_size, input_size, 3), dtype=np.float32)
        lab[:, :, 0] = L
        lab[:, :, 1] = ab_np[:, :, 0] * 110 + 128  # Scale to LAB range
        lab[:, :, 2] = ab_np[:, :, 1] * 110 + 128
        
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


def compare_v4(image_path: str, output_path: str):
    """Compare DDColor vs V4 (MESH-based)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = cv2.resize(ddcolor.colorize(img_bgr), (W, H))
    
    print("  V4 (MESH decoder)...")
    v4 = V4Colorizer()
    v4_result = cv2.resize(v4.colorize(img_bgr), (W, H))
    
    # Compute difference
    diff = cv2.absdiff(ddcolor_result, v4_result)
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    
    # Create comparison: Original | Gray | DDColor | V4 | Diff
    comparison = np.hstack([img_bgr, gray_bgr, ddcolor_result, v4_result, diff_amplified])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["Original", "Grayscale", "DDColor", "V4 MESH", "Diff x5"]
    for i, label in enumerate(labels):
        cv2.putText(comparison, label, (i*W + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"  Saved: {output_path}")
    
    # Compute metrics
    mse = np.mean((ddcolor_result.astype(float) - v4_result.astype(float))**2)
    print(f"  MSE vs DDColor: {mse:.2f}")


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    test_images = list(coco_path.glob("*.jpg"))[:3]
    
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v4_comparison_{img_path.stem}.jpg"
        try:
            compare_v4(str(img_path), str(output_path))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
