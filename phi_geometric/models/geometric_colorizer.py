#!/usr/bin/env python3
"""
Geometric Colorizer - Built from Scratch Using φ-Lattice Framework

This is an attempt to build a competitive colorizer WITHOUT training,
using only:
1. Geometric principles (φ-lattice, MESH)
2. Semantic vocabulary (100 color concepts from DDColor analysis)
3. Pre-computed structure

The hypothesis: If LLMs are hyperdimensional transcoders, we should be
able to build the geometric structure directly without learning it.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


class PhiLattice:
    """φ-lattice for geometric positioning."""
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.phi = PHI
        
    def encode(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode values to φ-basis (sign, exponent)."""
        signs = torch.sign(values)
        magnitudes = torch.abs(values).clamp(min=1e-10)
        exponents = torch.log(magnitudes) / np.log(self.phi)
        return signs, exponents
    
    def decode(self, signs: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
        """Decode from φ-basis to values."""
        return signs * (self.phi ** exponents)
    
    def lattice_position(self, level: int) -> float:
        """Get the value at a specific φ-level."""
        return self.phi ** level


class ColorVocabulary:
    """
    The 100 color concepts - our semantic vocabulary.
    
    Each concept has:
    - A position in LAB color space (a, b values)
    - A spatial affinity (where in the image it tends to appear)
    - A semantic category
    """
    
    def __init__(self):
        # Initialize 100 color concepts on a φ-lattice grid
        # We arrange them in a 10x10 grid in LAB space
        self.n_concepts = 100
        self.concepts = self._initialize_concepts()
        
    def _initialize_concepts(self) -> Dict[int, Dict]:
        """Initialize color concepts geometrically."""
        concepts = {}
        
        # LAB color space ranges:
        # L: 0-100 (lightness)
        # a: -128 to 127 (green to red)
        # b: -128 to 127 (blue to yellow)
        
        # Create a 10x10 grid in ab space
        # Use φ-spacing for natural distribution
        a_range = np.linspace(-60, 60, 10)  # Green to red
        b_range = np.linspace(-60, 60, 10)  # Blue to yellow
        
        for i in range(100):
            row = i // 10
            col = i % 10
            
            a = a_range[col]
            b = b_range[row]
            
            # Determine color name from position
            color = self._ab_to_color_name(a, b)
            
            # Spatial affinity based on position
            # Top rows (low b) = sky (blue)
            # Bottom rows (high b) = ground (yellow/brown)
            if row < 3:
                spatial = "top"
            elif row < 7:
                spatial = "middle"
            else:
                spatial = "bottom"
            
            concepts[i] = {
                'a': float(a),
                'b': float(b),
                'color': color,
                'spatial': spatial,
                'row': row,
                'col': col,
            }
        
        return concepts
    
    def _ab_to_color_name(self, a: float, b: float) -> str:
        """Convert ab values to a color name."""
        if abs(a) < 15 and abs(b) < 15:
            return "gray"
        
        angle = np.arctan2(b, a) * 180 / np.pi
        
        if angle < -150 or angle >= 150:
            return "green"
        elif -150 <= angle < -90:
            return "cyan"
        elif -90 <= angle < -30:
            return "blue"
        elif -30 <= angle < 30:
            return "magenta" if a > 0 else "purple"
        elif 30 <= angle < 90:
            return "red" if a > 20 else "orange"
        elif 90 <= angle < 150:
            return "yellow"
        else:
            return "neutral"
    
    def get_color_embedding(self) -> torch.Tensor:
        """Get the color embeddings as a tensor [100, 2] for ab values."""
        ab = torch.zeros(100, 2)
        for i, c in self.concepts.items():
            ab[i, 0] = c['a']
            ab[i, 1] = c['b']
        return ab


class GeometricEncoder(nn.Module):
    """
    Encode grayscale image to feature space.
    
    Uses φ-lattice structured convolutions.
    No training - weights are initialized geometrically.
    """
    
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        
        # Simple encoder: grayscale -> features
        # Use φ-structured initialization
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, out_dim, 3, padding=1)
        
        # Initialize with φ-lattice structure
        self._phi_init()
        
    def _phi_init(self):
        """Initialize weights on φ-lattice."""
        phi = PHI
        
        for conv in [self.conv1, self.conv2, self.conv3]:
            # Initialize weights at φ^-2 level (typical for conv weights)
            scale = phi ** -2
            nn.init.uniform_(conv.weight, -scale, scale)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode grayscale image to features."""
        # x: [B, 1, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # [B, out_dim, H, W]
        return x


class GeometricColorizer(nn.Module):
    """
    Geometric Colorizer - No Training Required
    
    Architecture:
    1. Encode grayscale to features
    2. Compute attention between features and color vocabulary
    3. Blend colors based on attention
    
    The key insight: attention IS the semantic content.
    We pre-define the color vocabulary geometrically.
    """
    
    def __init__(self, feature_dim: int = 256, n_concepts: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_concepts = n_concepts
        
        # Components
        self.encoder = GeometricEncoder(feature_dim)
        self.vocabulary = ColorVocabulary()
        
        # Query embeddings for color concepts [100, feature_dim]
        # Initialize on φ-lattice
        self.query_embed = nn.Parameter(self._init_queries())
        
        # Color embeddings [100, 2] for ab output
        self.register_buffer('color_ab', self.vocabulary.get_color_embedding())
        
        # Temperature for attention
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def _init_queries(self) -> torch.Tensor:
        """Initialize query embeddings geometrically."""
        # Create orthogonal queries on φ-lattice
        queries = torch.zeros(self.n_concepts, self.feature_dim)
        
        # Use φ-structured initialization
        # Each query gets a unique direction in feature space
        for i in range(self.n_concepts):
            # Create a direction based on φ-encoding of the index
            row = i // 10
            col = i % 10
            
            # Use sinusoidal encoding at φ-frequencies
            for d in range(self.feature_dim):
                freq = PHI ** (d // 2 - self.feature_dim // 4)
                if d % 2 == 0:
                    queries[i, d] = np.sin(row * freq) * np.cos(col * freq)
                else:
                    queries[i, d] = np.cos(row * freq) * np.sin(col * freq)
        
        # Normalize
        queries = F.normalize(queries, dim=-1) * (PHI ** -1)
        
        return queries
    
    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        """
        Colorize a grayscale image.
        
        Args:
            gray: [B, 1, H, W] grayscale image (0-1)
        
        Returns:
            ab: [B, 2, H, W] predicted ab channels
        """
        B, _, H, W = gray.shape
        
        # Encode to features [B, D, H, W]
        features = self.encoder(gray)
        
        # Reshape for attention: [B, H*W, D]
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, self.feature_dim)
        
        # Compute attention: [B, H*W, 100]
        # attention[i,j,k] = how much pixel j attends to color concept k
        attention = torch.matmul(features_flat, self.query_embed.T)
        attention = attention / (self.temperature * np.sqrt(self.feature_dim))
        attention = F.softmax(attention, dim=-1)
        
        # Blend colors: [B, H*W, 2]
        ab_flat = torch.matmul(attention, self.color_ab)
        
        # Reshape to image: [B, 2, H, W]
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab
    
    def colorize(self, gray_bgr: np.ndarray) -> np.ndarray:
        """
        Convenience method to colorize a BGR grayscale image.
        
        Args:
            gray_bgr: [H, W, 3] BGR image (grayscale)
        
        Returns:
            colorized: [H, W, 3] BGR colorized image
        """
        import cv2
        
        # Convert to grayscale tensor
        gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
        gray_tensor = torch.from_numpy(gray).float() / 255.0
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Run model
        with torch.no_grad():
            ab = self(gray_tensor)  # [1, 2, H, W]
        
        # Convert to LAB image
        ab_np = ab[0].permute(1, 2, 0).numpy()  # [H, W, 2]
        
        # Create LAB image
        L = gray.astype(np.float32) * 100 / 255  # Scale to 0-100
        lab = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
        lab[:, :, 0] = L
        lab[:, :, 1] = ab_np[:, :, 0] + 128  # Center around 128
        lab[:, :, 2] = ab_np[:, :, 1] + 128
        
        # Convert to BGR
        lab_uint8 = lab.astype(np.uint8)
        colorized = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
        
        return colorized


class GeometricColorizerV2(nn.Module):
    """
    Version 2: Use DDColor's actual query structure.
    
    Instead of initializing queries geometrically, we extract
    the geometric structure from DDColor and use it directly.
    """
    
    def __init__(self, ddcolor_queries: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.feature_dim = 256
        self.n_concepts = 100
        
        # Simple encoder
        self.encoder = GeometricEncoder(self.feature_dim)
        
        # Use DDColor's queries if provided, else initialize geometrically
        if ddcolor_queries is not None:
            self.query_embed = nn.Parameter(ddcolor_queries.clone())
        else:
            self.query_embed = nn.Parameter(self._init_queries())
        
        # Color vocabulary
        self.vocabulary = ColorVocabulary()
        self.register_buffer('color_ab', self.vocabulary.get_color_embedding())
        
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def _init_queries(self) -> torch.Tensor:
        """Fallback geometric initialization."""
        queries = torch.randn(self.n_concepts, self.feature_dim)
        queries = F.normalize(queries, dim=-1) * (PHI ** -1)
        return queries
    
    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        """Same as V1."""
        B, _, H, W = gray.shape
        features = self.encoder(gray)
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, self.feature_dim)
        
        attention = torch.matmul(features_flat, self.query_embed.T)
        attention = attention / (self.temperature * np.sqrt(self.feature_dim))
        attention = F.softmax(attention, dim=-1)
        
        ab_flat = torch.matmul(attention, self.color_ab)
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


def compare_colorizers(image_path: str, output_dir: str = "comparisons"):
    """
    Compare our geometric colorizer with DDColor.
    
    Creates a side-by-side visualization.
    """
    import cv2
    import sys
    sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')
    
    from ddcolor import DDColor
    from ddcolor.pipeline import ColorizationPipeline
    from huggingface_hub import PyTorchModelHubMixin
    
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Load DDColor
    print("Loading DDColor...")
    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)
    
    ddcolor_model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
    ddcolor_model.eval()
    ddcolor_pipeline = ColorizationPipeline(ddcolor_model, input_size=512)
    
    # Run DDColor
    print("Running DDColor...")
    ddcolor_result = ddcolor_pipeline.process(img_bgr)
    
    # Load our geometric colorizer
    print("Running Geometric Colorizer...")
    geo_colorizer = GeometricColorizer()
    geo_colorizer.eval()
    geo_result = geo_colorizer.colorize(gray_bgr)
    
    # Resize all to same size
    H, W = img_bgr.shape[:2]
    ddcolor_result = cv2.resize(ddcolor_result, (W, H))
    geo_result = cv2.resize(geo_result, (W, H))
    
    # Create comparison image
    # Layout: Original | Grayscale | DDColor | Geometric
    comparison = np.hstack([img_bgr, gray_bgr, ddcolor_result, geo_result])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Grayscale", (W + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "DDColor", (2*W + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Geometric", (3*W + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_file = output_path / f"comparison_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(output_file), comparison)
    print(f"Saved comparison to: {output_file}")
    
    return comparison


if __name__ == "__main__":
    import sys
    
    # Test on a COCO image
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    test_images = list(coco_path.glob("*.jpg"))[:3]
    
    output_dir = "/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons"
    
    for img_path in test_images:
        print(f"\nProcessing: {img_path.name}")
        try:
            compare_colorizers(str(img_path), output_dir)
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nComparisons saved to: {output_dir}")
