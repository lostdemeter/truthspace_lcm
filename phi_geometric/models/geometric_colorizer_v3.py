#!/usr/bin/env python3
"""
Geometric Colorizer V3 - Use DDColor's Learned Queries

V1: Random geometric encoder + geometric decoder → Gray (averaging)
V2: DDColor encoder + geometric decoder → ?
V3: DDColor encoder + DDColor queries + geometric color mapping → ?

This isolates: Is the problem in the color vocabulary mapping?

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


class GeometricDecoderV3(nn.Module):
    """
    Use DDColor's learned queries but map to geometric color vocabulary.
    """
    
    def __init__(self, ddcolor_queries: torch.Tensor, feature_dim: int = 256):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_concepts = ddcolor_queries.shape[0]
        
        # Use DDColor's learned queries
        self.query_embed = nn.Parameter(ddcolor_queries.clone())
        
        # Geometric color vocabulary
        self.register_buffer('color_ab', self._init_colors())
        
        # Learn a mapping from query to color (small network)
        self.color_map = nn.Linear(feature_dim, 2)
        self._init_color_map()
        
        self.temperature = 0.1
    
    def _init_colors(self) -> torch.Tensor:
        """Initialize color vocabulary in LAB ab space."""
        colors = torch.zeros(self.n_concepts, 2)
        
        # 10x10 grid in ab space
        a_vals = torch.linspace(-50, 50, 10)
        b_vals = torch.linspace(-50, 50, 10)
        
        for i in range(self.n_concepts):
            row = i // 10
            col = i % 10
            colors[i, 0] = a_vals[col]
            colors[i, 1] = b_vals[row]
        
        return colors
    
    def _init_color_map(self):
        """Initialize color mapping on φ-lattice."""
        scale = PHI ** -2
        nn.init.uniform_(self.color_map.weight, -scale, scale)
        nn.init.zeros_(self.color_map.bias)
    
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
        
        # Attention over queries: [B, H*W, 100]
        scores = torch.matmul(feat_flat, self.query_embed.T)
        scores = scores / (self.temperature * np.sqrt(self.feature_dim))
        attention = F.softmax(scores, dim=-1)
        
        # Weighted query features: [B, H*W, C]
        weighted_queries = torch.matmul(attention, self.query_embed)
        
        # Map to colors: [B, H*W, 2]
        ab_flat = self.color_map(weighted_queries)
        
        # Reshape: [B, 2, H, W]
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


class HybridColorizerV3:
    """
    V3: DDColor features + DDColor queries + learned color mapping.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()
    
    def _load_models(self):
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading DDColor...")
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        self.ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        self.ddcolor.eval()
        self.ddcolor = self.ddcolor.to(self.device)
        
        # Extract DDColor's queries
        queries = self.ddcolor.decoder.color_decoder.query_embed.weight.detach()
        
        # Create V3 decoder with DDColor's queries
        self.decoder = GeometricDecoderV3(queries, feature_dim=256)
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()
        
        print("  Models loaded")
    
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
        
        L = gray.astype(np.float32) * 100 / 255
        lab = np.zeros((input_size, input_size, 3), dtype=np.float32)
        lab[:, :, 0] = L
        lab[:, :, 1] = ab_np[:, :, 0] + 128
        lab[:, :, 2] = ab_np[:, :, 1] + 128
        
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        colorized = cv2.resize(colorized, (orig_w, orig_h))
        
        return colorized


class DDColorReference:
    """Reference DDColor for comparison."""
    
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


def compare_all(image_path: str, output_path: str):
    """Compare all versions."""
    from phi_geometric.models.geometric_colorizer import GeometricColorizer
    from phi_geometric.models.geometric_colorizer_v2 import HybridColorizer
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = cv2.resize(ddcolor.colorize(img_bgr), (W, H))
    
    print("  V1 (random geo)...")
    v1 = GeometricColorizer()
    v1.eval()
    v1_result = cv2.resize(v1.colorize(gray_bgr), (W, H))
    
    print("  V2 (DDColor enc + geo dec)...")
    v2 = HybridColorizer()
    v2_result = cv2.resize(v2.colorize(img_bgr), (W, H))
    
    print("  V3 (DDColor enc + DDColor queries + geo colors)...")
    v3 = HybridColorizerV3()
    v3_result = cv2.resize(v3.colorize(img_bgr), (W, H))
    
    # Create comparison: Original | DDColor | V1 | V2 | V3
    comparison = np.hstack([img_bgr, ddcolor_result, v1_result, v2_result, v3_result])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["Original", "DDColor", "V1 Geo", "V2 Hybrid", "V3 Queries"]
    for i, label in enumerate(labels):
        cv2.putText(comparison, label, (i*W + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    test_images = list(coco_path.glob("*.jpg"))[:3]
    
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v3_comparison_{img_path.stem}.jpg"
        try:
            compare_all(str(img_path), str(output_path))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
