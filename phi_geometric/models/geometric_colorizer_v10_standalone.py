#!/usr/bin/env python3
"""
Geometric Colorizer V10 Standalone - PEP Feature Mapping + Simple Decoder

V10 uses PEP-extracted projections to map geometric features to DDColor's
feature space, then uses a simple attention decoder with extracted colors.

This is fully standalone - no DDColor decoder reuse.

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


class PEPFeatureMapper(nn.Module):
    """
    Maps geometric encoder features to DDColor feature space using PEP-extracted projections.
    """
    
    def __init__(self, projection_path: str):
        super().__init__()
        
        # Load projections
        data = np.load(projection_path)
        
        # Create linear layers for each level
        self.proj0 = nn.Linear(96, 96, bias=True)
        self.proj1 = nn.Linear(192, 192, bias=True)
        self.proj2 = nn.Linear(384, 384, bias=True)
        self.proj3 = nn.Linear(768, 768, bias=True)
        
        # Load weights
        self.proj0.weight.data = torch.from_numpy(data['W0']).float()
        self.proj0.bias.data = torch.from_numpy(data['b0']).float()
        self.proj1.weight.data = torch.from_numpy(data['W1']).float()
        self.proj1.bias.data = torch.from_numpy(data['b1']).float()
        self.proj2.weight.data = torch.from_numpy(data['W2']).float()
        self.proj2.bias.data = torch.from_numpy(data['b2']).float()
        self.proj3.weight.data = torch.from_numpy(data['W3']).float()
        self.proj3.bias.data = torch.from_numpy(data['b3']).float()
    
    def forward(self, features: list) -> list:
        """Map geometric features to DDColor-like feature space."""
        f0, f1, f2, f3 = features
        
        # Apply projections (spatial dims as batch)
        def apply_proj(f, proj):
            B, C, H, W = f.shape
            f_flat = f.permute(0, 2, 3, 1).reshape(-1, C)
            f_mapped = proj(f_flat)
            return f_mapped.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        return [
            apply_proj(f0, self.proj0),
            apply_proj(f1, self.proj1),
            apply_proj(f2, self.proj2),
            apply_proj(f3, self.proj3),
        ]


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder that combines features from all levels.
    Uses DDColor's query embeddings and extracted color vocabulary.
    """
    
    def __init__(self, n_queries: int = 100, hidden_dim: int = 256):
        super().__init__()
        
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim
        
        # Query embeddings (loaded from DDColor)
        self.query_embed = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        
        # Feature fusion from multiple scales
        # Upsample all to 1/4 resolution and concatenate
        self.fusion = nn.Sequential(
            nn.Conv2d(96 + 192 + 384 + 768, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        
        # Color vocabulary
        self.color_vocab = nn.Parameter(torch.zeros(n_queries, 2), requires_grad=False)
        
        self.temperature = 0.07
    
    def load_query_embeddings(self, path: str):
        """Load DDColor's query embeddings."""
        query_embed = np.load(path)
        self.query_embed.data = torch.from_numpy(query_embed).float()
    
    def load_color_vocabulary(self, path: str):
        """Load extracted color vocabulary."""
        with open(path) as f:
            data = json.load(f)
        
        colors = torch.zeros(self.n_queries, 2)
        for i in range(self.n_queries):
            colors[i, 0] = data[str(i)]['mean_a'] * 3.0
            colors[i, 1] = data[str(i)]['mean_b'] * 1.85
        
        self.color_vocab.data = colors
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of 4 mapped feature tensors
        Returns:
            ab: [B, 2, H, W] at 1/4 resolution
        """
        f0, f1, f2, f3 = features
        B = f0.shape[0]
        target_size = f0.shape[2:]  # 1/4 resolution
        
        # Upsample all features to 1/4 resolution
        f1_up = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2_up = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate
        fused = torch.cat([f0, f1_up, f2_up, f3_up], dim=1)  # [B, 1440, H, W]
        
        # Apply fusion network
        feat = self.fusion(fused)  # [B, 256, H, W]
        
        H, W = feat.shape[2:]
        
        # Reshape for attention: [B, H*W, 256]
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H*W, self.hidden_dim)
        
        # Compute attention scores: [B, H*W, n_queries]
        scores = torch.matmul(feat_flat, self.query_embed.T) / self.temperature
        attn = F.softmax(scores, dim=-1)
        
        # Weighted sum of colors: [B, H*W, 2]
        colors = torch.matmul(attn, self.color_vocab)
        
        # Reshape: [B, 2, H, W]
        ab = colors.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


class V10StandaloneColorizer:
    """
    V10 Standalone: Geometric encoder + PEP mapping + multi-scale decoder.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self._load_models()
    
    def _load_models(self):
        print("Loading V10 Standalone...")
        
        # Geometric encoder
        self.encoder = GeometricEncoder()
        self.encoder = self.encoder.to(self.device)
        
        # PEP feature mapper
        self.mapper = PEPFeatureMapper(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/geo_to_dd_all_levels.npz'
        )
        self.mapper = self.mapper.to(self.device)
        
        # Multi-scale decoder
        self.decoder = MultiScaleDecoder(n_queries=100, hidden_dim=256)
        self.decoder.load_query_embeddings(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/ddcolor_query_embed.npy'
        )
        self.decoder.load_color_vocabulary(
            '/home/thorin/truthspace-lcm/phi_geometric/evaluations/extracted_query_colors.json'
        )
        self.decoder = self.decoder.to(self.device)
        
        print("  V10 Standalone loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        
        # Prepare input
        img = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        
        tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Geometric encoder
            geo_features = self.encoder(tensor)
            
            # Map to DDColor feature space
            mapped_features = self.mapper(geo_features)
            
            # Decode to colors
            ab = self.decoder(mapped_features)  # [B, 2, H/4, W/4]
            
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


def compare(image_path: str, output_path: str, v10: 'V10StandaloneColorizer'):
    """Compare DDColor vs V10 Standalone."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("  DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    
    print("  V10 Standalone...")
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
    
    v10 = V10StandaloneColorizer()
    
    # Test on images NOT used for PEP (skip first 30)
    test_images = list(coco_path.glob("*.jpg"))[30:35]
    
    results = []
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v10_standalone_{img_path.stem}.jpg"
        try:
            mse, sat = compare(str(img_path), str(output_path), v10)
            results.append((mse, sat))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        mses, sats = zip(*results)
        print(f"\n{'='*50}")
        print(f"V10 Standalone Results:")
        print(f"  Average MSE vs DDColor: {np.mean(mses):.1f}")
        print(f"  Average Saturation: {np.mean(sats):.0f}")
        print(f"\nComponents:")
        print(f"  - Geometric encoder (Gabor + position + local stats)")
        print(f"  - PEP-extracted feature mapping (0.82-0.97 correlation)")
        print(f"  - Multi-scale decoder with DDColor query embeddings")
        print(f"  - Extracted color vocabulary")
