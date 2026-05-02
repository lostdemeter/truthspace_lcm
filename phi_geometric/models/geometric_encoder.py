#!/usr/bin/env python3
"""
Geometric Encoder for Colorization

A geometric encoder that extracts semantic features without training,
using classical computer vision techniques organized geometrically.

The encoder needs to answer: "What TYPE of region is this pixel in?"
- Position (where in image)
- Texture (smooth vs textured)
- Intensity (bright vs dark)
- Context (what's around it)

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple

PHI = (1 + np.sqrt(5)) / 2


class GaborBank:
    """
    Bank of Gabor filters at multiple orientations and scales.
    Gabor filters detect edges and textures at specific orientations.
    """
    
    def __init__(self, n_orientations: int = 8, n_scales: int = 4):
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.filters = self._create_filters()
    
    def _create_filters(self) -> List[np.ndarray]:
        filters = []
        
        for scale_idx in range(self.n_scales):
            # Scale follows φ progression
            sigma = 2.0 * (PHI ** scale_idx)
            lambd = sigma * 2.0
            
            for orient_idx in range(self.n_orientations):
                theta = orient_idx * np.pi / self.n_orientations
                
                # Create Gabor kernel
                ksize = int(sigma * 6) | 1  # Ensure odd
                kernel = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma=sigma,
                    theta=theta,
                    lambd=lambd,
                    gamma=0.5,
                    psi=0
                )
                filters.append(kernel)
        
        return filters
    
    def apply(self, gray: np.ndarray) -> np.ndarray:
        """Apply all Gabor filters and return responses."""
        responses = []
        for kernel in self.filters:
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(response)
        return np.stack(responses, axis=0)  # [n_filters, H, W]


class PositionEncoder:
    """
    Encode spatial position using sinusoidal encoding.
    This tells the model WHERE in the image each pixel is.
    """
    
    def __init__(self, n_frequencies: int = 8):
        self.n_frequencies = n_frequencies
    
    def encode(self, height: int, width: int) -> np.ndarray:
        """Create position encoding for image of given size."""
        # Create coordinate grids
        y = np.linspace(-1, 1, height)
        x = np.linspace(-1, 1, width)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        encodings = []
        
        for freq in range(self.n_frequencies):
            # Frequency follows φ progression
            f = PHI ** freq
            
            # Sin and cos for both x and y
            encodings.append(np.sin(f * np.pi * xx))
            encodings.append(np.cos(f * np.pi * xx))
            encodings.append(np.sin(f * np.pi * yy))
            encodings.append(np.cos(f * np.pi * yy))
        
        return np.stack(encodings, axis=0)  # [4*n_freq, H, W]


class LocalStatistics:
    """
    Compute local statistics (mean, std, gradients) at multiple scales.
    """
    
    def __init__(self, scales: List[int] = [3, 7, 15, 31]):
        self.scales = scales
    
    def compute(self, gray: np.ndarray) -> np.ndarray:
        """Compute local statistics at multiple scales."""
        features = []
        
        for ksize in self.scales:
            # Local mean
            mean = cv2.blur(gray, (ksize, ksize))
            features.append(mean)
            
            # Local std (via variance)
            mean_sq = cv2.blur(gray ** 2, (ksize, ksize))
            var = mean_sq - mean ** 2
            std = np.sqrt(np.maximum(var, 0))
            features.append(std)
            
            # Deviation from local mean (contrast)
            contrast = gray - mean
            features.append(contrast)
        
        # Gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        
        features.extend([grad_x, grad_y, grad_mag, grad_dir])
        
        return np.stack(features, axis=0)  # [n_features, H, W]


class GeometricEncoder(nn.Module):
    """
    Geometric encoder that produces multi-scale features similar to ConvNeXt.
    
    Output shapes (to match DDColor decoder expectations):
    - Level 0: [B, 96, H/4, W/4]
    - Level 1: [B, 192, H/8, W/8]
    - Level 2: [B, 384, H/16, W/16]
    - Level 3: [B, 768, H/32, W/32]
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extractors
        self.gabor = GaborBank(n_orientations=8, n_scales=4)
        self.position = PositionEncoder(n_frequencies=8)
        self.local_stats = LocalStatistics()
        
        # Calculate feature dimensions
        self.n_gabor = 8 * 4  # 32 Gabor features
        self.n_position = 4 * 8  # 32 position features
        self.n_local = 4 * 3 + 4  # 16 local stat features
        self.n_raw = self.n_gabor + self.n_position + self.n_local + 1  # +1 for gray
        
        # Projection layers to match DDColor dimensions
        # These are simple linear projections (geometric)
        self.proj0 = nn.Conv2d(self.n_raw, 96, 1)
        self.proj1 = nn.Conv2d(self.n_raw, 192, 1)
        self.proj2 = nn.Conv2d(self.n_raw, 384, 1)
        self.proj3 = nn.Conv2d(self.n_raw, 768, 1)
        
        # Initialize with small random weights
        for proj in [self.proj0, self.proj1, self.proj2, self.proj3]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            nn.init.zeros_(proj.bias)
    
    def extract_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract all geometric features from grayscale image."""
        H, W = gray.shape
        
        # Gabor responses
        gabor_feat = self.gabor.apply(gray)  # [32, H, W]
        
        # Position encoding
        pos_feat = self.position.encode(H, W)  # [32, H, W]
        
        # Local statistics
        local_feat = self.local_stats.compute(gray)  # [16, H, W]
        
        # Raw intensity
        gray_feat = gray[np.newaxis, :, :]  # [1, H, W]
        
        # Concatenate all features
        features = np.concatenate([
            gabor_feat,
            pos_feat,
            local_feat,
            gray_feat
        ], axis=0)  # [n_raw, H, W]
        
        return features
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, 3, H, W] normalized RGB (grayscale)
        
        Returns:
            List of 4 feature tensors at different scales
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Convert to grayscale numpy for feature extraction
        # (geometric features are computed in numpy)
        features_list = []
        
        for b in range(B):
            # Get grayscale (average of RGB channels)
            gray = x[b].mean(dim=0).cpu().numpy()
            
            # Extract geometric features
            features = self.extract_features(gray)
            features_list.append(features)
        
        # Stack and convert to tensor (ensure on correct device)
        features = np.stack(features_list, axis=0)  # [B, n_raw, H, W]
        features = torch.from_numpy(features.copy()).float().to(device)
        
        # Normalize features
        features = (features - features.mean(dim=(2, 3), keepdim=True)) / (features.std(dim=(2, 3), keepdim=True) + 1e-6)
        
        # Multi-scale outputs
        # Level 0: 1/4 resolution
        f0 = F.avg_pool2d(features, 4)
        f0 = self.proj0(f0)
        
        # Level 1: 1/8 resolution
        f1 = F.avg_pool2d(features, 8)
        f1 = self.proj1(f1)
        
        # Level 2: 1/16 resolution
        f2 = F.avg_pool2d(features, 16)
        f2 = self.proj2(f2)
        
        # Level 3: 1/32 resolution
        f3 = F.avg_pool2d(features, 32)
        f3 = self.proj3(f3)
        
        return [f0, f1, f2, f3]


def test_encoder():
    """Test the geometric encoder."""
    import cv2
    
    # Load test image
    img = cv2.imread('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/000000127494.jpg')
    img = cv2.resize(img, (512, 512))
    
    # Convert to grayscale RGB tensor
    img_l = cv2.cvtColor((img/255.0).astype(np.float32), cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0)
    
    # Create encoder
    encoder = GeometricEncoder()
    
    # Forward pass
    outputs = encoder(tensor)
    
    print("Geometric Encoder Output:")
    for i, out in enumerate(outputs):
        print(f"  Level {i}: {out.shape}, range=[{out.min():.2f}, {out.max():.2f}]")


if __name__ == "__main__":
    test_encoder()
