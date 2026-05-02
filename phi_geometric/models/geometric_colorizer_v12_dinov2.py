#!/usr/bin/env python3
"""
Geometric Colorizer V12 - DINOv2 Encoder

V12 uses DINOv2 as the encoder, which we've already reverse-engineered
(Doc 123: 93-99% linear, φ-expressible Q-K rotations).

This tests whether DINOv2's semantic features can drive colorization.

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

from transformers import Dinov2Model

PHI = (1 + np.sqrt(5)) / 2


class DINOv2ColorDecoder(nn.Module):
    """
    Decoder that maps DINOv2 features to ab colors.
    Uses the extracted color vocabulary from DDColor.
    """
    
    def __init__(self, hidden_dim: int = 384, n_queries: int = 100):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_queries = n_queries
        
        # Query embeddings for color attention
        self.query_embed = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        
        # Feature projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Color vocabulary (extracted from DDColor)
        self.color_vocab = nn.Parameter(torch.zeros(n_queries, 2), requires_grad=False)
        
        self.temperature = 0.07
    
    def load_color_vocabulary(self, path: str):
        """Load extracted color vocabulary."""
        with open(path) as f:
            data = json.load(f)
        
        colors = torch.zeros(self.n_queries, 2)
        for i in range(self.n_queries):
            colors[i, 0] = data[str(i)]['mean_a'] * 3.0
            colors[i, 1] = data[str(i)]['mean_b'] * 1.85
        
        self.color_vocab.data = colors
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, D] DINOv2 features (N = num_patches + 1 CLS)
        Returns:
            ab: [B, 2, H, W] at patch resolution
        """
        B, N, D = features.shape
        
        # Remove CLS token
        patch_features = features[:, 1:, :]  # [B, N-1, D]
        N_patches = N - 1
        H = W = int(np.sqrt(N_patches))
        
        # Project features
        feat = self.proj(patch_features)  # [B, N-1, D]
        
        # Compute attention to color queries
        scores = torch.matmul(feat, self.query_embed.T) / self.temperature  # [B, N-1, 100]
        attn = F.softmax(scores, dim=-1)
        
        # Weighted sum of colors
        colors = torch.matmul(attn, self.color_vocab)  # [B, N-1, 2]
        
        # Reshape to spatial
        ab = colors.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        return ab


class V12DINOv2Colorizer:
    """
    V12: DINOv2 encoder + color attention decoder.
    
    DINOv2 provides semantic features that we've shown are 93-99% linear
    and φ-expressible (Doc 123).
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 518  # DINOv2 expects multiples of 14
        self._load_models()
    
    def _load_models(self):
        print("Loading V12 (DINOv2)...")
        
        # DINOv2 encoder
        self.encoder = Dinov2Model.from_pretrained('facebook/dinov2-small')
        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)
        
        # Color decoder
        self.decoder = DINOv2ColorDecoder(hidden_dim=384, n_queries=100)
        self.decoder.load_color_vocabulary(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/extracted_query_colors.json'
        )
        self.decoder = self.decoder.to(self.device)
        
        print("  V12 loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        
        img = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        # Resize to DINOv2 input size (multiple of 14)
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # Convert to grayscale RGB (3 channels, same value)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_gray_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
        
        # Normalize for DINOv2 (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_gray_rgb - mean) / std
        
        tensor = torch.from_numpy(img_norm.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get DINOv2 features
            outputs = self.encoder(tensor)
            features = outputs.last_hidden_state  # [B, N, 384]
            
            # Decode to colors
            ab = self.decoder(features)  # [B, 2, 37, 37]
            
            # Upsample to input size
            ab_up = F.interpolate(ab, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        # Convert to numpy
        ab_np = ab_up[0].cpu().numpy().transpose(1, 2, 0)
        
        # Apply bilateral filter for smoothing
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


def compare(image_path: str, output_path: str, v12: 'V12DINOv2Colorizer'):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    
    print("  V12 (DINOv2)...")
    v12_result = v12.colorize(img_bgr)
    
    mse = np.mean((ddcolor_result.astype(float) - v12_result.astype(float))**2)
    
    def get_saturation(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean()
    
    ddcolor_sat = get_saturation(ddcolor_result)
    v12_sat = get_saturation(v12_result)
    
    top_row = np.hstack([img_bgr, gray_bgr])
    bottom_row = np.hstack([ddcolor_result, v12_result])
    comparison = np.vstack([top_row, bottom_row])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ("Original", (10, 30)),
        ("Grayscale", (W + 10, 30)),
        (f"DDColor (sat:{ddcolor_sat:.0f})", (10, H + 30)),
        (f"V12 DINOv2 (sat:{v12_sat:.0f})", (W + 10, H + 30)),
    ]
    for label, pos in labels:
        cv2.putText(comparison, label, pos, font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, label, pos, font, 0.7, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, comparison)
    print(f"  MSE: {mse:.1f}, Sat - DD:{ddcolor_sat:.0f}, V12:{v12_sat:.0f}")
    
    return mse, v12_sat


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    v12 = V12DINOv2Colorizer()
    
    test_images = list(coco_path.glob("*.jpg"))[40:45]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v12_dinov2_{img_path.stem}.jpg"
        try:
            mse, sat = compare(str(img_path), str(output_path), v12)
            results.append((mse, sat))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        mses, sats = zip(*results)
        print(f"\n{'='*50}")
        print(f"V12 DINOv2 Results:")
        print(f"  Average MSE: {np.mean(mses):.1f}")
        print(f"  Average Saturation: {np.mean(sats):.0f}")
        print()
        print("Note: DINOv2 is 93-99% linear and φ-expressible (Doc 123)")
        print("This provides a path to geometric colorization via:")
        print("  1. φ-backbone replacement (80x speedup)")
        print("  2. PEP extraction of color decoder")
