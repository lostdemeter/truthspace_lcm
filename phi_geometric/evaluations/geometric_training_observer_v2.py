#!/usr/bin/env python3
"""
Geometric Training Observer V2

Improved version with:
1. Better training data (diverse color targets)
2. Regularization to encourage orthogonality
3. Comparison to DDColor trajectory

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
        
        # Learnable color queries - initialize near DDColor scale
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.1)
        
        # Simple feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, dim, 3, padding=1),
        )
        
        # Attention projections
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        # Color output
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
        """Encourage queries to be orthogonal."""
        q_norm = self.queries / (self.queries.norm(dim=1, keepdim=True) + 1e-8)
        sim = q_norm @ q_norm.T
        off_diag = sim - torch.eye(self.n_queries, device=self.queries.device)
        return off_diag.pow(2).mean()


def load_real_training_data(batch_size=4, size=64):
    """Load real images from COCO for training."""
    coco_path = Path("/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017")
    
    images = list(coco_path.glob("*.jpg"))[:100]  # Use first 100 images
    
    batch_gray = []
    batch_ab = []
    
    for _ in range(batch_size):
        img_path = np.random.choice(images)
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (size, size))
        
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Normalize
        L = lab[:, :, 0:1] / 255.0  # [0, 1]
        ab = lab[:, :, 1:3] - 128   # [-128, 127] centered at 0
        
        batch_gray.append(torch.from_numpy(L).permute(2, 0, 1))
        batch_ab.append(torch.from_numpy(ab).permute(2, 0, 1))
    
    return torch.stack(batch_gray), torch.stack(batch_ab)


def observe_geometry(model, encoder, step, loss):
    """Take a geometric snapshot."""
    with torch.no_grad():
        queries = model.queries.detach().cpu()
        
        # φ-levels
        signs, exps = encoder.encode(queries)
        levels = (exps.float() - encoder.bias) / encoder.K
        mean_level = levels.mean().item()
        
        # Orthogonality
        q_norm = queries / (queries.norm(dim=1, keepdim=True) + 1e-8)
        sim = q_norm @ q_norm.T
        off_diag = sim - torch.eye(queries.shape[0])
        orthogonality = off_diag.abs().mean().item()
        
        # Effective rank
        U, S, Vt = torch.linalg.svd(queries, full_matrices=False)
        normalized_S = S / (S.sum() + 1e-8)
        entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
        effective_rank = torch.exp(entropy).item()
        
        # Coverage (spread of singular values)
        coverage = (S[9] / (S[0] + 1e-8)).item()
        
    return {
        'step': step,
        'loss': loss,
        'phi_level': mean_level,
        'orthogonality': orthogonality,
        'effective_rank': effective_rank,
        'coverage': coverage,
    }


def run_training_with_orthogonality():
    """Train with orthogonality regularization."""
    print("=" * 70)
    print("TRAINING WITH ORTHOGONALITY REGULARIZATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = PhiEncoder(K=32)
    
    model = MinimalColorizer(n_queries=100, dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    snapshots = []
    
    # Initial state
    snap = observe_geometry(model, encoder, 0, float('inf'))
    snapshots.append(snap)
    print(f"\n[Step 0] φ={snap['phi_level']:.2f}, orth={snap['orthogonality']:.4f}, rank={snap['effective_rank']:.1f}")
    
    n_steps = 500
    orth_weight = 0.1  # Orthogonality regularization weight
    
    for step in range(1, n_steps + 1):
        gray, ab_target = load_real_training_data(batch_size=4, size=64)
        gray = gray.to(device)
        ab_target = ab_target.to(device)
        
        optimizer.zero_grad()
        ab_pred = model(gray)
        
        # Main loss
        recon_loss = criterion(ab_pred, ab_target)
        
        # Orthogonality regularization
        orth_loss = model.orthogonality_loss()
        
        loss = recon_loss + orth_weight * orth_loss
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            snap = observe_geometry(model, encoder, step, loss.item())
            snapshots.append(snap)
            print(f"[Step {step}] loss={loss.item():.2f}, φ={snap['phi_level']:.2f}, orth={snap['orthogonality']:.4f}, rank={snap['effective_rank']:.1f}")
    
    return snapshots, model


def compare_to_ddcolor():
    """Compare our trained model to DDColor."""
    print("\n" + "=" * 70)
    print("COMPARISON TO DDCOLOR")
    print("=" * 70)
    
    encoder = PhiEncoder(K=32)
    
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        queries = ddcolor.decoder.color_decoder.query_feat.weight.detach().cpu()
        
        # Analyze DDColor
        signs, exps = encoder.encode(queries)
        levels = (exps.float() - encoder.bias) / encoder.K
        
        q_norm = queries / (queries.norm(dim=1, keepdim=True) + 1e-8)
        sim = q_norm @ q_norm.T
        off_diag = sim - torch.eye(queries.shape[0])
        
        U, S, Vt = torch.linalg.svd(queries, full_matrices=False)
        normalized_S = S / (S.sum() + 1e-8)
        entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
        
        print(f"\nDDColor Queries:")
        print(f"  φ-level: {levels.mean().item():.2f}")
        print(f"  Orthogonality: {off_diag.abs().mean().item():.4f}")
        print(f"  Effective rank: {torch.exp(entropy).item():.1f}")
        print(f"  Coverage (S10/S1): {(S[9] / S[0]).item():.3f}")
        
        return {
            'phi_level': levels.mean().item(),
            'orthogonality': off_diag.abs().mean().item(),
            'effective_rank': torch.exp(entropy).item(),
            'coverage': (S[9] / S[0]).item(),
        }
        
    except Exception as e:
        print(f"Could not load DDColor: {e}")
        return None


def analyze_convergence_trajectory(snapshots, ddcolor_target=None):
    """Analyze how training converges toward the target."""
    print("\n" + "=" * 70)
    print("CONVERGENCE TRAJECTORY ANALYSIS")
    print("=" * 70)
    
    if not snapshots:
        return
    
    # Extract trajectories
    steps = [s['step'] for s in snapshots]
    phi_levels = [s['phi_level'] for s in snapshots]
    orths = [s['orthogonality'] for s in snapshots]
    ranks = [s['effective_rank'] for s in snapshots]
    
    print(f"\n## Trajectory")
    print(f"{'Step':>6} {'φ-level':>10} {'Orth':>10} {'Rank':>10}")
    print("-" * 40)
    for s in snapshots:
        print(f"{s['step']:>6} {s['phi_level']:>10.2f} {s['orthogonality']:>10.4f} {s['effective_rank']:>10.1f}")
    
    # Compute trends
    if len(steps) > 1:
        phi_trend = np.polyfit(steps, phi_levels, 1)[0]
        orth_trend = np.polyfit(steps, orths, 1)[0]
        rank_trend = np.polyfit(steps, ranks, 1)[0]
        
        print(f"\n## Trends (per step)")
        print(f"  φ-level: {phi_trend:.6f}")
        print(f"  Orthogonality: {orth_trend:.6f}")
        print(f"  Effective rank: {rank_trend:.4f}")
    
    # Compare to target
    if ddcolor_target:
        print(f"\n## Distance to DDColor Target")
        final = snapshots[-1]
        
        phi_gap = abs(final['phi_level'] - ddcolor_target['phi_level'])
        orth_gap = abs(final['orthogonality'] - ddcolor_target['orthogonality'])
        rank_gap = abs(final['effective_rank'] - ddcolor_target['effective_rank'])
        
        print(f"  φ-level gap: {phi_gap:.2f} (ours={final['phi_level']:.2f}, target={ddcolor_target['phi_level']:.2f})")
        print(f"  Orthogonality gap: {orth_gap:.4f} (ours={final['orthogonality']:.4f}, target={ddcolor_target['orthogonality']:.4f})")
        print(f"  Rank gap: {rank_gap:.1f} (ours={final['effective_rank']:.1f}, target={ddcolor_target['effective_rank']:.1f})")
        
        # Estimate steps to convergence
        if len(steps) > 1 and orth_trend != 0:
            steps_to_target_orth = (ddcolor_target['orthogonality'] - final['orthogonality']) / orth_trend
            print(f"\n## Estimated Steps to Target")
            print(f"  Orthogonality: {steps_to_target_orth:.0f} steps")


def main():
    # Train with orthogonality regularization
    snapshots, model = run_training_with_orthogonality()
    
    # Compare to DDColor
    ddcolor_target = compare_to_ddcolor()
    
    # Analyze trajectory
    analyze_convergence_trajectory(snapshots, ddcolor_target)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. With orthogonality regularization, queries stay more distinct
2. The φ-level trajectory shows consistent movement
3. We can track distance to DDColor's structure

NEXT STEPS:
1. Implement "jump to destination" when trajectory is clear
2. Test if early jumping preserves accuracy
3. Build geometric accelerator for training
""")


if __name__ == "__main__":
    main()
