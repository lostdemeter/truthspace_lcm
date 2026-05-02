#!/usr/bin/env python3
"""
Geometric Colorizer V2 - Use DDColor's Encoder

The problem with V1: Our encoder has no semantic understanding.
Solution: Use DDColor's trained encoder, but replace the decoder
with our geometric structure.

This tests whether the "intelligence" is in:
- The encoder (semantic feature extraction)
- The decoder (color vocabulary + attention)
- Both

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


class GeometricDecoder(nn.Module):
    """
    Geometric decoder using φ-lattice color vocabulary.
    
    Takes encoder features and produces ab colors using
    attention over a geometric color vocabulary.
    """
    
    def __init__(self, feature_dim: int = 256, n_concepts: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_concepts = n_concepts
        
        # Query embeddings - initialized geometrically
        self.query_embed = nn.Parameter(self._init_queries())
        
        # Color vocabulary - 10x10 grid in LAB ab space
        self.register_buffer('color_ab', self._init_colors())
        
        # Project features to query space
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        
        # Temperature
        self.temperature = 0.1
        
        self._init_weights()
    
    def _init_queries(self) -> torch.Tensor:
        """Initialize orthogonal queries on φ-lattice."""
        # Create near-orthogonal queries using φ-structured directions
        queries = torch.zeros(self.n_concepts, self.feature_dim)
        
        for i in range(self.n_concepts):
            row = i // 10
            col = i % 10
            
            # Use different frequency bands for different queries
            for d in range(self.feature_dim):
                freq_row = PHI ** ((d % 10) - 5)
                freq_col = PHI ** ((d // 10) - 12)
                
                phase_row = row * 2 * np.pi / 10
                phase_col = col * 2 * np.pi / 10
                
                queries[i, d] = np.sin(freq_row * phase_row + freq_col * phase_col)
        
        # Normalize to unit vectors
        queries = F.normalize(queries, dim=-1)
        return queries
    
    def _init_colors(self) -> torch.Tensor:
        """Initialize color vocabulary in LAB ab space."""
        colors = torch.zeros(self.n_concepts, 2)
        
        # Create a 10x10 grid covering useful ab range
        # a: -50 to 50 (green to red)
        # b: -50 to 50 (blue to yellow)
        a_vals = torch.linspace(-50, 50, 10)
        b_vals = torch.linspace(-50, 50, 10)
        
        for i in range(self.n_concepts):
            row = i // 10
            col = i % 10
            colors[i, 0] = a_vals[col]  # a channel
            colors[i, 1] = b_vals[row]  # b channel
        
        return colors
    
    def _init_weights(self):
        """Initialize projection weights on φ-lattice."""
        scale = PHI ** -2
        nn.init.uniform_(self.feature_proj.weight, -scale, scale)
        nn.init.zeros_(self.feature_proj.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to ab colors.
        
        Args:
            features: [B, C, H, W] encoder features
        
        Returns:
            ab: [B, 2, H, W] predicted ab channels
        """
        B, C, H, W = features.shape
        
        # Reshape: [B, H*W, C]
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Project features
        feat_proj = self.feature_proj(feat_flat)  # [B, H*W, D]
        
        # Compute attention scores: [B, H*W, 100]
        scores = torch.matmul(feat_proj, self.query_embed.T)
        scores = scores / (self.temperature * np.sqrt(self.feature_dim))
        attention = F.softmax(scores, dim=-1)
        
        # Blend colors: [B, H*W, 2]
        ab_flat = torch.matmul(attention, self.color_ab)
        
        # Reshape: [B, 2, H, W]
        ab = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        
        return ab


class HybridColorizer:
    """
    Hybrid colorizer: DDColor encoder + Geometric decoder.
    
    This tests whether we can replace the learned decoder
    with a geometric one while keeping semantic understanding.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = None
        self.decoder = None
        
        self._load_models()
    
    def _load_models(self):
        """Load DDColor encoder and create geometric decoder."""
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        print("Loading DDColor encoder...")
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        ddcolor.eval()
        ddcolor = ddcolor.to(self.device)
        
        # Extract encoder
        self.ddcolor = ddcolor
        self.encoder = ddcolor.encoder
        
        # Create geometric decoder
        # DDColor's encoder outputs 256-dim features
        self.decoder = GeometricDecoder(feature_dim=256, n_concepts=100)
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()
        
        print("  Models loaded")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Colorize an image using hybrid approach.
        
        Args:
            img_bgr: [H, W, 3] BGR input image
        
        Returns:
            colorized: [H, W, 3] BGR colorized image
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Preprocess: resize and convert to grayscale
        input_size = 512
        img_resized = cv2.resize(img_bgr, (input_size, input_size))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Create 3-channel grayscale for encoder
        gray_3ch = np.stack([gray, gray, gray], axis=-1)
        
        # To tensor: [1, 3, H, W]
        tensor = torch.from_numpy(gray_3ch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Hook to capture features before color decoder
        captured = {}
        def hook_fn(module, input, output):
            captured['features'] = input[1]  # img_features [1, 256, 512, 512]
        
        hook = self.ddcolor.decoder.color_decoder.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            # Run full model to get features
            _ = self.ddcolor(tensor)
            
            # Get the spatial features
            features = captured['features']  # [1, 256, H, W]
            
            # Run geometric decoder
            ab = self.decoder(features)  # [1, 2, H, W]
        
        hook.remove()
        
        # Convert to LAB image
        ab_np = ab[0].cpu().permute(1, 2, 0).numpy()  # [H, W, 2]
        
        # Create LAB image
        L = gray.astype(np.float32) * 100 / 255
        lab = np.zeros((input_size, input_size, 3), dtype=np.float32)
        lab[:, :, 0] = L
        lab[:, :, 1] = ab_np[:, :, 0] + 128
        lab[:, :, 2] = ab_np[:, :, 1] + 128
        
        # Clip and convert
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Resize back
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
    """Compare V1 geometric, V2 hybrid, and DDColor."""
    from phi_geometric.models.geometric_colorizer import GeometricColorizer
    
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("Running DDColor...")
    ddcolor = DDColorReference()
    ddcolor_result = ddcolor.colorize(img_bgr)
    ddcolor_result = cv2.resize(ddcolor_result, (W, H))
    
    print("Running V1 Geometric...")
    v1 = GeometricColorizer()
    v1.eval()
    v1_result = v1.colorize(gray_bgr)
    v1_result = cv2.resize(v1_result, (W, H))
    
    print("Running V2 Hybrid...")
    v2 = HybridColorizer()
    v2_result = v2.colorize(img_bgr)
    v2_result = cv2.resize(v2_result, (W, H))
    
    # Create comparison: Original | Gray | DDColor | V1 | V2
    comparison = np.hstack([img_bgr, gray_bgr, ddcolor_result, v1_result, v2_result])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["Original", "Grayscale", "DDColor", "V1 Geo", "V2 Hybrid"]
    for i, label in enumerate(labels):
        cv2.putText(comparison, label, (i*W + 10, 30), font, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"Saved: {output_path}")
    
    return comparison


if __name__ == "__main__":
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    output_dir = Path("/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    test_images = list(coco_path.glob("*.jpg"))[:3]
    
    for img_path in test_images:
        print(f"\n=== {img_path.name} ===")
        output_path = output_dir / f"v2_comparison_{img_path.stem}.jpg"
        try:
            compare_all(str(img_path), str(output_path))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
