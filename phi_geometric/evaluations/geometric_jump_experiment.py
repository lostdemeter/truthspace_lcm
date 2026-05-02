#!/usr/bin/env python3
"""
Geometric Jump Experiment

Key insight from training observation:
- Structure (orthogonality, rank) converges FAST
- Scale (φ-level) is the main gap

Hypothesis: If structure matches, we can JUMP to the destination
by rescaling to the target φ-level.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI


class MinimalColorizer(nn.Module):
    """Minimal colorizer with learnable queries."""
    
    def __init__(self, n_queries: int = 100, dim: int = 256):
        super().__init__()
        
        self.n_queries = n_queries
        self.dim = dim
        
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.1)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, dim, 3, padding=1),
        )
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.color_head = nn.Linear(dim, 2)
    
    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        B, _, H, W = gray.shape
        
        features = self.encoder(gray)
        features_flat = features.flatten(2).permute(0, 2, 1)
        
        Q = self.query_proj(self.queries.unsqueeze(0).expand(B, -1, -1))
        K = self.key_proj(features_flat)
        V = self.value_proj(features_flat)
        
        attn = torch.softmax(Q @ K.transpose(-1, -2) / np.sqrt(self.dim), dim=-1)
        attended = attn @ V
        
        colors = self.color_head(attended)
        ab = torch.einsum('bqp,bqc->bpc', attn, colors)
        ab = ab.permute(0, 2, 1).view(B, 2, H, W)
        
        return ab
    
    def orthogonality_loss(self) -> torch.Tensor:
        q_norm = self.queries / (self.queries.norm(dim=1, keepdim=True) + 1e-8)
        sim = q_norm @ q_norm.T
        off_diag = sim - torch.eye(self.n_queries, device=self.queries.device)
        return off_diag.pow(2).mean()


def analyze_queries(queries, encoder, name=""):
    """Analyze query structure."""
    with torch.no_grad():
        signs, exps = encoder.encode(queries)
        levels = (exps.float() - encoder.bias) / encoder.K
        
        q_norm = queries / (queries.norm(dim=1, keepdim=True) + 1e-8)
        sim = q_norm @ q_norm.T
        off_diag = sim - torch.eye(queries.shape[0])
        
        U, S, Vt = torch.linalg.svd(queries, full_matrices=False)
        normalized_S = S / (S.sum() + 1e-8)
        entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
        
    return {
        'name': name,
        'phi_level': levels.mean().item(),
        'orthogonality': off_diag.abs().mean().item(),
        'effective_rank': torch.exp(entropy).item(),
        'coverage': (S[9] / (S[0] + 1e-8)).item(),
    }


def load_batch(batch_size=4, size=64):
    """Load real images from COCO."""
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    images = list(coco_path.glob("*.jpg"))[:100]
    
    batch_gray = []
    batch_ab = []
    
    for _ in range(batch_size):
        img_path = np.random.choice(images)
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (size, size))
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0:1] / 255.0
        ab = lab[:, :, 1:3] - 128
        
        batch_gray.append(torch.from_numpy(L).permute(2, 0, 1))
        batch_ab.append(torch.from_numpy(ab).permute(2, 0, 1))
    
    return torch.stack(batch_gray), torch.stack(batch_ab)


def train_briefly(model, device, n_steps=100, orth_weight=0.1):
    """Train briefly to establish structure."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for step in range(n_steps):
        gray, ab_target = load_batch()
        gray, ab_target = gray.to(device), ab_target.to(device)
        
        optimizer.zero_grad()
        ab_pred = model(gray)
        
        recon_loss = criterion(ab_pred, ab_target)
        orth_loss = model.orthogonality_loss()
        loss = recon_loss + orth_weight * orth_loss
        
        loss.backward()
        optimizer.step()
    
    return loss.item()


def jump_to_destination(model, target_phi_level, encoder):
    """
    Jump to destination by rescaling to target φ-level.
    
    The key insight: if structure matches, we just need to rescale.
    """
    with torch.no_grad():
        # Current φ-level
        signs, exps = encoder.encode(model.queries)
        current_levels = (exps.float() - encoder.bias) / encoder.K
        current_mean = current_levels.mean().item()
        
        # Compute scale factor
        level_shift = target_phi_level - current_mean
        scale_factor = PHI ** level_shift
        
        print(f"  Current φ-level: {current_mean:.2f}")
        print(f"  Target φ-level: {target_phi_level:.2f}")
        print(f"  Level shift: {level_shift:.2f}")
        print(f"  Scale factor: {scale_factor:.4f}")
        
        # Rescale queries
        model.queries.data = model.queries.data * scale_factor
        
        # Verify
        signs, exps = encoder.encode(model.queries)
        new_levels = (exps.float() - encoder.bias) / encoder.K
        print(f"  New φ-level: {new_levels.mean().item():.2f}")


def evaluate_colorization(model, device, n_samples=10):
    """Evaluate colorization quality."""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_samples):
            gray, ab_target = load_batch(batch_size=1)
            gray, ab_target = gray.to(device), ab_target.to(device)
            
            ab_pred = model(gray)
            loss = criterion(ab_pred, ab_target)
            total_loss += loss.item()
    
    model.train()
    return total_loss / n_samples


def run_jump_experiment():
    """Test if we can jump to destination after brief training."""
    print("=" * 70)
    print("GEOMETRIC JUMP EXPERIMENT")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = PhiEncoder(K=32)
    
    # Get DDColor target
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        ddcolor_queries = ddcolor.decoder.color_decoder.query_feat.weight.detach().cpu()
        ddcolor_stats = analyze_queries(ddcolor_queries, encoder, "DDColor")
        target_phi_level = ddcolor_stats['phi_level']
        
        print(f"\n## DDColor Target")
        print(f"  φ-level: {ddcolor_stats['phi_level']:.2f}")
        print(f"  Orthogonality: {ddcolor_stats['orthogonality']:.4f}")
        print(f"  Effective rank: {ddcolor_stats['effective_rank']:.1f}")
        
    except Exception as e:
        print(f"Could not load DDColor: {e}")
        target_phi_level = -1.6  # Use known value
    
    # Create model
    model = MinimalColorizer(n_queries=100, dim=256).to(device)
    
    print(f"\n## Initial State")
    initial_stats = analyze_queries(model.queries.detach().cpu(), encoder, "Initial")
    print(f"  φ-level: {initial_stats['phi_level']:.2f}")
    print(f"  Orthogonality: {initial_stats['orthogonality']:.4f}")
    print(f"  Effective rank: {initial_stats['effective_rank']:.1f}")
    
    initial_loss = evaluate_colorization(model, device)
    print(f"  Colorization loss: {initial_loss:.2f}")
    
    # Brief training to establish structure
    print(f"\n## Brief Training (50 steps)")
    train_briefly(model, device, n_steps=50, orth_weight=0.1)
    
    after_train_stats = analyze_queries(model.queries.detach().cpu(), encoder, "After Training")
    print(f"  φ-level: {after_train_stats['phi_level']:.2f}")
    print(f"  Orthogonality: {after_train_stats['orthogonality']:.4f}")
    print(f"  Effective rank: {after_train_stats['effective_rank']:.1f}")
    
    after_train_loss = evaluate_colorization(model, device)
    print(f"  Colorization loss: {after_train_loss:.2f}")
    
    # Jump to destination
    print(f"\n## Jumping to Destination")
    jump_to_destination(model, target_phi_level, encoder)
    
    after_jump_stats = analyze_queries(model.queries.detach().cpu(), encoder, "After Jump")
    print(f"  Orthogonality: {after_jump_stats['orthogonality']:.4f}")
    print(f"  Effective rank: {after_jump_stats['effective_rank']:.1f}")
    
    after_jump_loss = evaluate_colorization(model, device)
    print(f"  Colorization loss: {after_jump_loss:.2f}")
    
    # Continue training after jump
    print(f"\n## Continue Training After Jump (50 steps)")
    train_briefly(model, device, n_steps=50, orth_weight=0.1)
    
    final_stats = analyze_queries(model.queries.detach().cpu(), encoder, "Final")
    print(f"  φ-level: {final_stats['phi_level']:.2f}")
    print(f"  Orthogonality: {final_stats['orthogonality']:.4f}")
    print(f"  Effective rank: {final_stats['effective_rank']:.1f}")
    
    final_loss = evaluate_colorization(model, device)
    print(f"  Colorization loss: {final_loss:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Stage':<25} {'φ-level':>10} {'Orth':>10} {'Loss':>10}")
    print("-" * 55)
    print(f"{'Initial':<25} {initial_stats['phi_level']:>10.2f} {initial_stats['orthogonality']:>10.4f} {initial_loss:>10.2f}")
    print(f"{'After 50 steps':<25} {after_train_stats['phi_level']:>10.2f} {after_train_stats['orthogonality']:>10.4f} {after_train_loss:>10.2f}")
    print(f"{'After Jump':<25} {after_jump_stats['phi_level']:>10.2f} {after_jump_stats['orthogonality']:>10.4f} {after_jump_loss:>10.2f}")
    print(f"{'Final (50 more steps)':<25} {final_stats['phi_level']:>10.2f} {final_stats['orthogonality']:>10.4f} {final_loss:>10.2f}")
    
    # Did jumping help?
    print(f"\n## Analysis")
    if after_jump_loss < after_train_loss:
        print(f"  ✓ Jumping IMPROVED loss: {after_train_loss:.2f} → {after_jump_loss:.2f}")
    else:
        print(f"  ✗ Jumping did not improve loss: {after_train_loss:.2f} → {after_jump_loss:.2f}")
    
    if final_loss < after_train_loss:
        print(f"  ✓ Jump + training beat training alone: {final_loss:.2f} < {after_train_loss:.2f}")
    else:
        print(f"  ✗ Jump + training did not beat training alone")


if __name__ == "__main__":
    run_jump_experiment()
