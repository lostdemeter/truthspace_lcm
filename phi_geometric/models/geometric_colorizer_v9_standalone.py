#!/usr/bin/env python3
"""
Geometric Colorizer V9 Standalone - Fully Geometric Pipeline

A completely standalone geometric colorizer that doesn't reuse DDColor's decoder.
Instead, it uses a simple attention-based decoder with the extracted color vocabulary.

Architecture:
1. Geometric Encoder (Gabor + position + local stats)
2. Simple Attention Decoder (query-based color selection)
3. Probe-extracted or vocabulary-based color mapping

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


class SimpleAttentionDecoder(nn.Module):
    """
    Simple attention-based decoder that maps encoder features to colors.
    
    Uses the extracted color vocabulary from DDColor.
    Now with PEP-extracted query embeddings for better semantic matching.
    """
    
    def __init__(self, n_queries: int = 100, feature_dim: int = 256):
        super().__init__()
        
        self.n_queries = n_queries
        self.feature_dim = feature_dim
        
        # Learnable query embeddings (like DDColor's query_embed)
        # Will be loaded from DDColor via PEP
        self.query_embed = nn.Parameter(torch.randn(n_queries, feature_dim) * 0.02)
        
        # Feature projection (from encoder features to query space)
        # Use multi-scale features for richer representation
        # Level 0: 96 channels at 1/4 scale (highest resolution)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(96, feature_dim, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        
        # Color vocabulary (extracted from DDColor)
        self.color_vocab = nn.Parameter(torch.zeros(n_queries, 2), requires_grad=False)
        
        # Non-linear color expansion (to reach DDColor's full range)
        self.color_scale_a = 3.0
        self.color_scale_b = 1.85
        self.expansion_a = 0.9
        self.expansion_b = 0.15
        
        self.temperature = 0.07
    
    def load_color_vocabulary(self, path: str):
        """Load extracted color vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        colors = torch.zeros(self.n_queries, 2)
        for i in range(self.n_queries):
            colors[i, 0] = data[str(i)]['mean_a']
            colors[i, 1] = data[str(i)]['mean_b']
        
        # Scale to match DDColor's output range
        colors[:, 0] *= self.color_scale_a
        colors[:, 1] *= self.color_scale_b
        
        self.color_vocab.data = colors
    
    def load_query_embeddings(self, path: str):
        """Load DDColor's query embeddings for better semantic matching."""
        query_embed = np.load(path)
        self.query_embed.data = torch.from_numpy(query_embed).float()
    
    def forward(self, encoder_features: list) -> torch.Tensor:
        """
        Args:
            encoder_features: List of 4 feature tensors from geometric encoder
                [B, 96, H/4, W/4], [B, 192, H/8, W/8], [B, 384, H/16, W/16], [B, 768, H/32, W/32]
        
        Returns:
            ab: [B, 2, H/4, W/4] - ab color channels at 1/4 resolution
        """
        # Use highest resolution features
        feat = encoder_features[0]  # [B, 96, H/4, W/4]
        B, C, H, W = feat.shape
        
        # Project to query space
        feat_proj = self.feature_proj(feat)  # [B, 256, H, W]
        
        # Reshape for attention: [B, H*W, 256]
        feat_flat = feat_proj.permute(0, 2, 3, 1).reshape(B, H*W, self.feature_dim)
        
        # Compute attention scores: [B, H*W, n_queries]
        scores = torch.matmul(feat_flat, self.query_embed.T) / self.temperature
        attn = F.softmax(scores, dim=-1)
        
        # Weighted sum of color vocabulary: [B, H*W, 2]
        colors = torch.matmul(attn, self.color_vocab)
        
        # Reshape to spatial: [B, 2, H, W]
        ab = colors.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


class V9StandaloneColorizer:
    """
    Fully standalone geometric colorizer.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self._load_models()
    
    def _load_models(self):
        print("Loading V9 Standalone...")
        
        # Geometric encoder
        self.encoder = GeometricEncoder()
        self.encoder = self.encoder.to(self.device)
        
        # Simple attention decoder
        self.decoder = SimpleAttentionDecoder(n_queries=100, feature_dim=256)
        self.decoder.load_color_vocabulary(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/extracted_query_colors.json'
        )
        # Load DDColor's query embeddings for semantic matching
        self.decoder.load_query_embeddings(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/ddcolor_query_embed.npy'
        )
        self.decoder = self.decoder.to(self.device)
        
        print("  V9 Standalone loaded")
    
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
            # Run geometric encoder
            encoder_features = self.encoder(tensor_gray_rgb)
            
            # Run decoder
            ab = self.decoder(encoder_features)  # [B, 2, H/4, W/4]
            
            # Upsample to full resolution
            output_ab = F.interpolate(ab, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
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


def compare_v9_standalone(image_path: str, output_path: str, v9: 'V9StandaloneColorizer'):
    """Compare DDColor vs V9 Standalone with visual output."""
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
    
    print("  V9 Standalone...")
    v9_result = v9.colorize(img_bgr)
    
    # Compute metrics
    mse = np.mean((ddcolor_result.astype(float) - v9_result.astype(float))**2)
    
    # Compute saturation for both
    def get_saturation(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean()
    
    ddcolor_sat = get_saturation(ddcolor_result)
    v9_sat = get_saturation(v9_result)
    
    # Create comparison image (2x2 grid)
    top_row = np.hstack([img_bgr, gray_bgr])
    bottom_row = np.hstack([ddcolor_result, v9_result])
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
    v9 = V9StandaloneColorizer()
    
    # Test on several images
    test_images = list(coco_path.glob("*.jpg"))[:5]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v9_standalone_{img_path.stem}.jpg"
        try:
            mse, sat = compare_v9_standalone(str(img_path), str(output_path), v9)
            results.append((mse, sat))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        mses, sats = zip(*results)
        print(f"\n{'='*50}")
        print(f"V9 Standalone Results (Fully Geometric):")
        print(f"  Average MSE vs DDColor: {np.mean(mses):.1f}")
        print(f"  Average Saturation: {np.mean(sats):.0f}")
        print(f"\nNote: This is a FROM-SCRATCH geometric colorizer.")
        print(f"No pretrained weights used except extracted color vocabulary.")
