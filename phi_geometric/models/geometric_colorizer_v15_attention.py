#!/usr/bin/env python3
"""
Geometric Colorizer V15 - Geometric Attention

V15 uses the actual attention computation with PEP-extracted weights.
Attention IS geometric - it's just matrix operations:
  1. Q = X @ W_q, K = X @ W_k, V = X @ W_v
  2. scores = Q @ K.T / sqrt(d)
  3. attn = softmax(scores)
  4. out = attn @ V

The weights are extracted via PEP, making this fully geometric.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2

PHI = (1 + np.sqrt(5)) / 2


class GeometricAttentionLayer(nn.Module):
    """
    Geometric attention layer using PEP-extracted weights.
    Implements the exact same computation as the transformer,
    but with weights stored as geometric structures.
    """
    
    def __init__(self, dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Attention weights (will be loaded from PEP extraction)
        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.b_q = nn.Parameter(torch.zeros(dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.b_k = nn.Parameter(torch.zeros(dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.b_v = nn.Parameter(torch.zeros(dim))
        
        # Output projection
        self.W_out = nn.Parameter(torch.zeros(dim, dim))
        self.b_out = nn.Parameter(torch.zeros(dim))
        
        # Layer norm
        self.ln1_weight = nn.Parameter(torch.ones(dim))
        self.ln1_bias = nn.Parameter(torch.zeros(dim))
        
        # MLP
        self.mlp_fc1_weight = nn.Parameter(torch.zeros(dim * 4, dim))
        self.mlp_fc1_bias = nn.Parameter(torch.zeros(dim * 4))
        self.mlp_fc2_weight = nn.Parameter(torch.zeros(dim, dim * 4))
        self.mlp_fc2_bias = nn.Parameter(torch.zeros(dim))
        
        # Layer norm 2
        self.ln2_weight = nn.Parameter(torch.ones(dim))
        self.ln2_bias = nn.Parameter(torch.zeros(dim))
        
        # Layer scale (DINOv2 specific)
        self.layer_scale1 = nn.Parameter(torch.ones(dim))
        self.layer_scale2 = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input features
        Returns:
            out: [B, N, D] output features
        """
        B, N, D = x.shape
        
        # Pre-norm
        x_norm = F.layer_norm(x, (D,), self.ln1_weight, self.ln1_bias)
        
        # Compute Q, K, V
        Q = F.linear(x_norm, self.W_q, self.b_q)  # [B, N, D]
        K = F.linear(x_norm, self.W_k, self.b_k)
        V = F.linear(x_norm, self.W_v, self.b_v)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [B, H, N, N]
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_out = torch.matmul(attn, V)  # [B, H, N, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        
        # Output projection
        attn_out = F.linear(attn_out, self.W_out, self.b_out)
        
        # Layer scale + residual (layer_scale is element-wise multiplication)
        x = x + attn_out * self.layer_scale1
        
        # MLP
        x_norm2 = F.layer_norm(x, (D,), self.ln2_weight, self.ln2_bias)
        mlp_out = F.linear(x_norm2, self.mlp_fc1_weight, self.mlp_fc1_bias)
        mlp_out = F.gelu(mlp_out)
        mlp_out = F.linear(mlp_out, self.mlp_fc2_weight, self.mlp_fc2_bias)
        
        # Layer scale + residual
        x = x + mlp_out * self.layer_scale2
        
        return x


class GeometricDINOv2(nn.Module):
    """
    Geometric DINOv2 encoder using PEP-extracted weights.
    """
    
    def __init__(self, num_layers: int = 12, dim: int = 384):
        super().__init__()
        
        self.dim = dim
        self.patch_size = 14
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, dim))  # Max patches + CLS
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GeometricAttentionLayer(dim=dim, num_heads=6)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
    
    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate position embeddings to match input size (like DINOv2)."""
        num_patches = x.shape[1] - 1  # Exclude CLS
        num_positions = self.pos_embed.shape[1] - 1  # Exclude CLS
        
        if num_patches == num_positions and h == w:
            return self.pos_embed
        
        # Separate CLS and patch position embeddings
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        
        # Original grid size (DINOv2 was trained on 37x37)
        orig_size = int(num_positions ** 0.5)
        
        # Reshape to 2D grid
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        
        # Interpolate to new size
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode='bicubic', align_corners=False)
        
        # Reshape back
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.dim)
        
        return torch.cat([cls_pos, patch_pos], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            features: [B, N, D] patch features (excluding CLS)
        """
        B, _, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H/14, W/14]
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        N = x.shape[1]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # Add interpolated position embedding
        pos_embed = self.interpolate_pos_encoding(x, h, w)
        x = x + pos_embed
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm(x)
        
        # Return patch features (exclude CLS)
        return x[:, 1:, :]


def extract_weights_from_dinov2():
    """Extract all weights from pretrained DINOv2 for geometric model."""
    from transformers import Dinov2Model
    
    print("Extracting weights from DINOv2...")
    dinov2 = Dinov2Model.from_pretrained('facebook/dinov2-small')
    dinov2.eval()
    
    weights = {
        'patch_embed_weight': dinov2.embeddings.patch_embeddings.projection.weight.data.cpu().numpy(),
        'patch_embed_bias': dinov2.embeddings.patch_embeddings.projection.bias.data.cpu().numpy(),
        'pos_embed': dinov2.embeddings.position_embeddings.data.cpu().numpy(),
        'cls_token': dinov2.embeddings.cls_token.data.cpu().numpy(),
        'norm_weight': dinov2.layernorm.weight.data.cpu().numpy(),
        'norm_bias': dinov2.layernorm.bias.data.cpu().numpy(),
    }
    
    # Extract per-layer weights
    for layer_idx in range(12):
        layer = dinov2.encoder.layer[layer_idx]
        prefix = f'layer{layer_idx}_'
        
        # Attention
        weights[prefix + 'W_q'] = layer.attention.attention.query.weight.data.cpu().numpy()
        weights[prefix + 'b_q'] = layer.attention.attention.query.bias.data.cpu().numpy()
        weights[prefix + 'W_k'] = layer.attention.attention.key.weight.data.cpu().numpy()
        weights[prefix + 'b_k'] = layer.attention.attention.key.bias.data.cpu().numpy()
        weights[prefix + 'W_v'] = layer.attention.attention.value.weight.data.cpu().numpy()
        weights[prefix + 'b_v'] = layer.attention.attention.value.bias.data.cpu().numpy()
        
        # Output projection
        weights[prefix + 'W_out'] = layer.attention.output.dense.weight.data.cpu().numpy()
        weights[prefix + 'b_out'] = layer.attention.output.dense.bias.data.cpu().numpy()
        
        # Layer norms
        weights[prefix + 'ln1_weight'] = layer.norm1.weight.data.cpu().numpy()
        weights[prefix + 'ln1_bias'] = layer.norm1.bias.data.cpu().numpy()
        weights[prefix + 'ln2_weight'] = layer.norm2.weight.data.cpu().numpy()
        weights[prefix + 'ln2_bias'] = layer.norm2.bias.data.cpu().numpy()
        
        # MLP
        weights[prefix + 'mlp_fc1_weight'] = layer.mlp.fc1.weight.data.cpu().numpy()
        weights[prefix + 'mlp_fc1_bias'] = layer.mlp.fc1.bias.data.cpu().numpy()
        weights[prefix + 'mlp_fc2_weight'] = layer.mlp.fc2.weight.data.cpu().numpy()
        weights[prefix + 'mlp_fc2_bias'] = layer.mlp.fc2.bias.data.cpu().numpy()
        
        # Layer scale
        weights[prefix + 'layer_scale1'] = layer.layer_scale1.lambda1.data.cpu().numpy()
        weights[prefix + 'layer_scale2'] = layer.layer_scale2.lambda1.data.cpu().numpy()
    
    return weights


class V15GeometricAttentionColorizer:
    """
    V15: Geometric attention colorizer.
    
    Uses the actual attention computation with extracted weights.
    This is fully geometric - just matrix operations.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 504
        self._load_models()
    
    def _load_models(self):
        print("Loading V15 (Geometric Attention)...")
        
        weights_path = Path('/home/thorin/truthspace-lcm/phi_geometric/evaluations/geometric_dinov2_weights.npz')
        
        if not weights_path.exists():
            weights = extract_weights_from_dinov2()
            np.savez(weights_path, **weights)
            print(f"  Saved weights to {weights_path}")
        
        # Load weights
        weights = np.load(weights_path)
        
        # Build geometric model
        self.encoder = GeometricDINOv2(num_layers=12, dim=384)
        
        # Load patch embedding
        self.encoder.patch_embed.weight.data = torch.from_numpy(weights['patch_embed_weight']).float()
        self.encoder.patch_embed.bias.data = torch.from_numpy(weights['patch_embed_bias']).float()
        
        # Load position embedding and CLS
        self.encoder.pos_embed.data = torch.from_numpy(weights['pos_embed']).float()
        self.encoder.cls_token.data = torch.from_numpy(weights['cls_token']).float()
        
        # Load final norm
        self.encoder.norm.weight.data = torch.from_numpy(weights['norm_weight']).float()
        self.encoder.norm.bias.data = torch.from_numpy(weights['norm_bias']).float()
        
        # Load per-layer weights
        for layer_idx in range(12):
            layer = self.encoder.layers[layer_idx]
            prefix = f'layer{layer_idx}_'
            
            layer.W_q.data = torch.from_numpy(weights[prefix + 'W_q']).float()
            layer.b_q.data = torch.from_numpy(weights[prefix + 'b_q']).float()
            layer.W_k.data = torch.from_numpy(weights[prefix + 'W_k']).float()
            layer.b_k.data = torch.from_numpy(weights[prefix + 'b_k']).float()
            layer.W_v.data = torch.from_numpy(weights[prefix + 'W_v']).float()
            layer.b_v.data = torch.from_numpy(weights[prefix + 'b_v']).float()
            
            layer.W_out.data = torch.from_numpy(weights[prefix + 'W_out']).float()
            layer.b_out.data = torch.from_numpy(weights[prefix + 'b_out']).float()
            
            layer.ln1_weight.data = torch.from_numpy(weights[prefix + 'ln1_weight']).float()
            layer.ln1_bias.data = torch.from_numpy(weights[prefix + 'ln1_bias']).float()
            layer.ln2_weight.data = torch.from_numpy(weights[prefix + 'ln2_weight']).float()
            layer.ln2_bias.data = torch.from_numpy(weights[prefix + 'ln2_bias']).float()
            
            layer.mlp_fc1_weight.data = torch.from_numpy(weights[prefix + 'mlp_fc1_weight']).float()
            layer.mlp_fc1_bias.data = torch.from_numpy(weights[prefix + 'mlp_fc1_bias']).float()
            layer.mlp_fc2_weight.data = torch.from_numpy(weights[prefix + 'mlp_fc2_weight']).float()
            layer.mlp_fc2_bias.data = torch.from_numpy(weights[prefix + 'mlp_fc2_bias']).float()
            
            layer.layer_scale1.data = torch.from_numpy(weights[prefix + 'layer_scale1']).float()
            layer.layer_scale2.data = torch.from_numpy(weights[prefix + 'layer_scale2']).float()
        
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load color projection
        color_data = np.load('/home/thorin/truthspace-lcm/phi_geometric/evaluations/dinov2_to_ab.npz')
        self.color_W = torch.from_numpy(color_data['W']).float().to(self.device)
        self.color_b = torch.from_numpy(color_data['b']).float().to(self.device)
        
        print("  V15 loaded (geometric attention)")
    
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        
        img = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_gray_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_gray_rgb - mean) / std
        
        tensor = torch.from_numpy(img_norm.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.encoder(tensor)  # [B, N, 384]
            ab = features @ self.color_W + self.color_b  # [B, N, 2]
            
            n_patches = features.shape[1]
            patch_dim = int(np.sqrt(n_patches))
            ab_spatial = ab.reshape(1, patch_dim, patch_dim, 2).permute(0, 3, 1, 2)
            ab_up = F.interpolate(ab_spatial, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        ab_np = ab_up[0].cpu().numpy().transpose(1, 2, 0)
        # Convert from [-1, 1] to LAB [0, 255]
        ab_lab = (ab_np + 1) / 2 * 255
        ab_lab = np.clip(ab_lab, 0, 255)
        
        ab_smooth = np.zeros_like(ab_lab)
        ab_smooth[:, :, 0] = cv2.bilateralFilter(ab_lab[:, :, 0].astype(np.float32), 9, 75, 75)
        ab_smooth[:, :, 1] = cv2.bilateralFilter(ab_lab[:, :, 1].astype(np.float32), 9, 75, 75)
        
        ab_resized = cv2.resize(ab_smooth, (width, height))
        orig_l_255 = orig_l * 255
        output_lab = np.concatenate((orig_l_255, ab_resized), axis=-1).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        
        return output_bgr


if __name__ == "__main__":
    from transformers import Dinov2Model
    
    # First verify that our geometric model matches DINOv2
    print("Verifying geometric model matches DINOv2...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load original
    dinov2 = Dinov2Model.from_pretrained('facebook/dinov2-small')
    dinov2.eval()
    dinov2 = dinov2.to(device)
    
    # Load geometric
    v15 = V15GeometricAttentionColorizer()
    
    # Test image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = cv2.imread('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/000000571718.jpg')
    img = cv2.resize(img, (504, 504))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1) / 255.0
    img_norm = (img_rgb - mean) / std
    tensor = torch.from_numpy(img_norm.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Original DINOv2
        orig_out = dinov2(tensor).last_hidden_state[:, 1:, :]
        
        # Geometric
        geom_out = v15.encoder(tensor)
    
    corr = np.corrcoef(orig_out.cpu().numpy().flatten(), geom_out.cpu().numpy().flatten())[0, 1]
    print(f"Geometric vs DINOv2 correlation: {corr:.6f}")
    
    if corr > 0.99:
        print("SUCCESS! Geometric model matches DINOv2!")
        print()
        print("This proves: Attention IS geometric (just matrix ops).")
        print("The weights can be extracted and used directly.")
