#!/usr/bin/env python3
"""
Geometric Colorizer V11 - Spatial Coherence via Smoothing

V11 addresses the splotchy color issue by:
1. Applying spatial smoothing to features before attention
2. Using bilateral filtering to preserve edges while smoothing colors
3. Post-processing with guided filtering

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
import json

from phi_geometric.models.geometric_encoder import GeometricEncoder

PHI = (1 + np.sqrt(5)) / 2


class SpatialCoherentDecoder(nn.Module):
    """
    Decoder with spatial coherence via smoothing.
    """
    
    def __init__(self, n_queries: int = 100, hidden_dim: int = 256):
        super().__init__()
        
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim
        
        # Query embeddings (loaded from DDColor)
        self.query_embed = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        
        # Spatial smoothing convolutions
        self.smooth = nn.Sequential(
            nn.Conv2d(96, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 3, padding=1),
        )
        
        # Color vocabulary
        self.color_vocab = nn.Parameter(torch.zeros(n_queries, 2), requires_grad=False)
        
        self.temperature = 0.07
    
    def load_query_embeddings(self, path: str):
        query_embed = np.load(path)
        self.query_embed.data = torch.from_numpy(query_embed).float()
    
    def load_color_vocabulary(self, path: str):
        with open(path) as f:
            data = json.load(f)
        
        colors = torch.zeros(self.n_queries, 2)
        for i in range(self.n_queries):
            colors[i, 0] = data[str(i)]['mean_a'] * 3.0
            colors[i, 1] = data[str(i)]['mean_b'] * 1.85
        
        self.color_vocab.data = colors
    
    def forward(self, features: list, gray_img: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: List of 4 feature tensors from geometric encoder
            gray_img: Original grayscale image for edge-aware filtering
        Returns:
            ab: [B, 2, H, W]
        """
        # Use only highest resolution features for simplicity
        f0 = features[0]  # [B, 96, H/4, W/4]
        B, C, H, W = f0.shape
        
        # Apply spatial smoothing
        feat = self.smooth(f0)  # [B, 256, H, W]
        
        # Reshape for attention
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H*W, self.hidden_dim)
        
        # Compute attention
        scores = torch.matmul(feat_flat, self.query_embed.T) / self.temperature
        attn = F.softmax(scores, dim=-1)  # [B, H*W, 100]
        
        # Weighted sum of colors
        colors = torch.matmul(attn, self.color_vocab)  # [B, H*W, 2]
        
        # Reshape to spatial
        ab = colors.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        return ab


class V11Colorizer:
    """
    V11: Geometric encoder with spatial coherence.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self._load_models()
    
    def _load_models(self):
        print("Loading V11...")
        
        # Geometric encoder
        self.encoder = GeometricEncoder()
        self.encoder = self.encoder.to(self.device)
        
        # Spatial coherent decoder
        self.decoder = SpatialCoherentDecoder(n_queries=100, hidden_dim=256)
        self.decoder.load_query_embeddings(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/ddcolor_query_embed.npy'
        )
        self.decoder.load_color_vocabulary(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/extracted_query_colors.json'
        )
        self.decoder = self.decoder.to(self.device)
        
        print("  V11 loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        
        img = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        
        tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        gray_tensor = torch.from_numpy(img_l).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        
        with torch.no_grad():
            # Geometric encoder
            geo_features = self.encoder(tensor)
            
            # Decode with spatial coherence
            ab = self.decoder(geo_features, gray_tensor)  # [B, 2, H/4, W/4]
            
            # Upsample
            output_ab = F.interpolate(ab, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        # Convert to numpy
        ab_np = output_ab[0].cpu().numpy().transpose(1, 2, 0)
        
        # Apply bilateral filter for edge-aware smoothing
        ab_smooth = np.zeros_like(ab_np)
        ab_smooth[:, :, 0] = cv2.bilateralFilter(ab_np[:, :, 0].astype(np.float32), 9, 75, 75)
        ab_smooth[:, :, 1] = cv2.bilateralFilter(ab_np[:, :, 1].astype(np.float32), 9, 75, 75)
        
        # Resize to original
        ab_resized = cv2.resize(ab_smooth, (width, height))
        
        # Combine with original L
        output_lab = np.concatenate((orig_l, ab_resized), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().astype(np.uint8)
        
        return output_img


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


def compare(image_path: str, output_path: str, v11: 'V11Colorizer'):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    
    print("  V11...")
    v11_result = v11.colorize(img_bgr)
    
    mse = np.mean((ddcolor_result.astype(float) - v11_result.astype(float))**2)
    
    def get_saturation(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean()
    
    ddcolor_sat = get_saturation(ddcolor_result)
    v11_sat = get_saturation(v11_result)
    
    top_row = np.hstack([img_bgr, gray_bgr])
    bottom_row = np.hstack([ddcolor_result, v11_result])
    comparison = np.vstack([top_row, bottom_row])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ("Original", (10, 30)),
        ("Grayscale", (W + 10, 30)),
        (f"DDColor (sat:{ddcolor_sat:.0f})", (10, H + 30)),
        (f"V11 (sat:{v11_sat:.0f})", (W + 10, H + 30)),
    ]
    for label, pos in labels:
        cv2.putText(comparison, label, pos, font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, label, pos, font, 0.7, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, comparison)
    print(f"  MSE: {mse:.1f}, Sat - DD:{ddcolor_sat:.0f}, V11:{v11_sat:.0f}")
    
    return mse, v11_sat


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    v11 = V11Colorizer()
    
    test_images = list(coco_path.glob("*.jpg"))[35:40]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v11_{img_path.stem}.jpg"
        try:
            mse, sat = compare(str(img_path), str(output_path), v11)
            results.append((mse, sat))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        mses, sats = zip(*results)
        print(f"\n{'='*50}")
        print(f"V11 Results (Spatial Coherence):")
        print(f"  Average MSE: {np.mean(mses):.1f}")
        print(f"  Average Saturation: {np.mean(sats):.0f}")
