#!/usr/bin/env python3
"""
Geometric Training Observer

Monitor traditional training through the lens of φ-geometry.
Track how weights move on the lattice during training.
Identify early geometric signals that predict convergence.

The goal: Detect WHERE training is going early, then jump there.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.core.encoder import PhiEncoder, PHI


@dataclass
class GeometricSnapshot:
    """Snapshot of geometric state at a training step."""
    step: int
    loss: float
    
    # φ-level statistics
    mean_phi_level: float
    std_phi_level: float
    mode_phi_level: float
    
    # Structural statistics
    orthogonality: float  # Mean off-diagonal similarity
    effective_rank: float  # Normalized effective rank
    coverage: float  # Singular value spread
    
    # Per-layer statistics
    layer_stats: Dict = field(default_factory=dict)


class GeometricObserver:
    """
    Observe training through geometric lens.
    
    Tracks:
    1. φ-level distribution over time
    2. Orthogonality of query-like layers
    3. Effective rank / coverage
    4. Trajectory toward known destinations
    """
    
    def __init__(self, model: nn.Module, encoder: Optional[PhiEncoder] = None):
        self.model = model
        self.encoder = encoder or PhiEncoder(K=32)
        self.snapshots: List[GeometricSnapshot] = []
        self.destinations: Dict[str, torch.Tensor] = {}
        
    def add_destination(self, name: str, weights: torch.Tensor):
        """Add a known destination for comparison."""
        self.destinations[name] = weights.detach().clone()
    
    def observe(self, step: int, loss: float) -> GeometricSnapshot:
        """Take a geometric snapshot of current model state."""
        all_levels = []
        layer_stats = {}
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.dim() < 2:
                    continue
                
                w = param.detach().cpu()
                
                # Get φ-levels
                signs, exps = self.encoder.encode(w)
                levels = (exps.float() - self.encoder.bias) / self.encoder.K
                all_levels.append(levels.flatten())
                
                # Per-layer stats
                stats = {
                    'mean_level': levels.mean().item(),
                    'std_level': levels.std().item(),
                }
                
                # Orthogonality (for query-like layers with 2D weights)
                if ('query' in name or 'embed' in name or w.shape[0] <= 256) and w.dim() == 2:
                    w_flat = w.view(w.shape[0], -1)  # Flatten to 2D
                    w_norm = w_flat / (w_flat.norm(dim=1, keepdim=True) + 1e-8)
                    sim = w_norm @ w_norm.T
                    off_diag = sim - torch.eye(w.shape[0])
                    stats['orthogonality'] = off_diag.abs().mean().item()
                    
                    # Effective rank
                    U, S, Vt = torch.linalg.svd(w, full_matrices=False)
                    normalized_S = S / (S.sum() + 1e-8)
                    entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
                    stats['effective_rank'] = torch.exp(entropy).item()
                    stats['coverage'] = (S[min(9, len(S)-1)] / (S[0] + 1e-8)).item()
                
                # Distance to destinations
                for dest_name, dest_weights in self.destinations.items():
                    if dest_weights.shape == w.shape:
                        dist = (w - dest_weights).norm().item()
                        stats[f'dist_to_{dest_name}'] = dist
                
                layer_stats[name] = stats
        
        # Global statistics
        all_levels = torch.cat(all_levels)
        
        # Compute global orthogonality/rank for first query-like layer
        global_orth = 0.0
        global_rank = 0.0
        global_coverage = 0.0
        
        for name, stats in layer_stats.items():
            if 'orthogonality' in stats:
                global_orth = stats['orthogonality']
                global_rank = stats['effective_rank']
                global_coverage = stats['coverage']
                break
        
        snapshot = GeometricSnapshot(
            step=step,
            loss=loss,
            mean_phi_level=all_levels.mean().item(),
            std_phi_level=all_levels.std().item(),
            mode_phi_level=all_levels.mode().values.item(),
            orthogonality=global_orth,
            effective_rank=global_rank,
            coverage=global_coverage,
            layer_stats=layer_stats,
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def print_snapshot(self, snapshot: GeometricSnapshot):
        """Print a snapshot summary."""
        print(f"\n[Step {snapshot.step}] Loss: {snapshot.loss:.4f}")
        print(f"  φ-levels: mean={snapshot.mean_phi_level:.2f}, std={snapshot.std_phi_level:.2f}")
        print(f"  Structure: orth={snapshot.orthogonality:.4f}, rank={snapshot.effective_rank:.1f}, cov={snapshot.coverage:.3f}")
    
    def analyze_trajectory(self) -> Dict:
        """Analyze the training trajectory."""
        if len(self.snapshots) < 2:
            return {}
        
        # Track how metrics evolve
        steps = [s.step for s in self.snapshots]
        losses = [s.loss for s in self.snapshots]
        phi_levels = [s.mean_phi_level for s in self.snapshots]
        orths = [s.orthogonality for s in self.snapshots]
        ranks = [s.effective_rank for s in self.snapshots]
        
        # Compute trends
        phi_trend = np.polyfit(steps, phi_levels, 1)[0]  # Slope
        orth_trend = np.polyfit(steps, orths, 1)[0]
        rank_trend = np.polyfit(steps, ranks, 1)[0]
        
        return {
            'phi_level_trend': phi_trend,
            'orthogonality_trend': orth_trend,
            'rank_trend': rank_trend,
            'final_phi_level': phi_levels[-1],
            'final_orthogonality': orths[-1],
            'final_rank': ranks[-1],
        }
    
    def predict_destination(self) -> Optional[str]:
        """Predict which destination we're heading toward."""
        if len(self.snapshots) < 5:
            return None
        
        trajectory = self.analyze_trajectory()
        
        # Check if we're approaching any known destination
        latest = self.snapshots[-1]
        
        for name, stats in latest.layer_stats.items():
            for dest_name in self.destinations:
                dist_key = f'dist_to_{dest_name}'
                if dist_key in stats:
                    # Check if distance is decreasing
                    distances = [
                        s.layer_stats.get(name, {}).get(dist_key, float('inf'))
                        for s in self.snapshots[-5:]
                    ]
                    if all(d1 > d2 for d1, d2 in zip(distances[:-1], distances[1:])):
                        return dest_name
        
        return None


class MinimalColorizer(nn.Module):
    """
    A minimal colorizer for training observation.
    
    Architecture:
    - 100 learnable color queries (like DDColor)
    - Simple attention to grayscale features
    - Output: 2 channels (ab)
    """
    
    def __init__(self, n_queries: int = 100, dim: int = 256):
        super().__init__()
        
        self.n_queries = n_queries
        self.dim = dim
        
        # Learnable color queries
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.01)
        
        # Feature encoder (simple conv)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, dim, 3, padding=1),
            nn.ReLU(),
        )
        
        # Cross-attention: queries attend to features
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        # Color output
        self.color_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # ab channels
        )
    
    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gray: [B, 1, H, W] grayscale image
        Returns:
            ab: [B, 2, H, W] color channels
        """
        B, _, H, W = gray.shape
        
        # Encode features
        features = self.encoder(gray)  # [B, dim, H, W]
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H*W, dim]
        
        # Cross-attention
        Q = self.query_proj(self.queries.unsqueeze(0).expand(B, -1, -1))  # [B, n_queries, dim]
        K = self.key_proj(features_flat)  # [B, H*W, dim]
        V = self.value_proj(features_flat)  # [B, H*W, dim]
        
        attn = torch.softmax(Q @ K.transpose(-1, -2) / np.sqrt(self.dim), dim=-1)  # [B, n_queries, H*W]
        attended = attn @ V  # [B, n_queries, dim]
        
        # Predict colors per query
        colors = self.color_head(attended)  # [B, n_queries, 2]
        
        # Distribute colors to pixels (simple: weighted sum by attention)
        ab = torch.einsum('bqp,bqc->bpc', attn, colors)  # [B, H*W, 2]
        ab = ab.permute(0, 2, 1).view(B, 2, H, W)
        
        return ab


def run_training_observation():
    """Run training and observe geometric changes."""
    print("=" * 70)
    print("GEOMETRIC TRAINING OBSERVATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MinimalColorizer(n_queries=100, dim=256).to(device)
    
    # Create observer
    observer = GeometricObserver(model)
    
    # Load DDColor destination for comparison
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
        observer.add_destination('ddcolor_queries', ddcolor_queries)
        print("Added DDColor queries as destination")
    except Exception as e:
        print(f"Could not load DDColor destination: {e}")
    
    # Create synthetic training data
    # Simple: grayscale patches with known colors
    def generate_batch(batch_size=8, size=64):
        # Random grayscale images
        gray = torch.rand(batch_size, 1, size, size, device=device)
        
        # Target: simple color mapping (just for observation)
        # Bright → warm (positive a, positive b)
        # Dark → cool (negative a, negative b)
        ab_target = torch.zeros(batch_size, 2, size, size, device=device)
        ab_target[:, 0] = (gray[:, 0] - 0.5) * 100  # a channel
        ab_target[:, 1] = (gray[:, 0] - 0.5) * 100  # b channel
        
        return gray, ab_target
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("\n## Initial State")
    snapshot = observer.observe(0, float('inf'))
    observer.print_snapshot(snapshot)
    
    print("\n## Training...")
    
    n_steps = 200
    observe_every = 20
    
    for step in range(1, n_steps + 1):
        gray, ab_target = generate_batch()
        
        optimizer.zero_grad()
        ab_pred = model(gray)
        loss = criterion(ab_pred, ab_target)
        loss.backward()
        optimizer.step()
        
        if step % observe_every == 0:
            snapshot = observer.observe(step, loss.item())
            observer.print_snapshot(snapshot)
            
            # Check if approaching destination
            dest = observer.predict_destination()
            if dest:
                print(f"  → Approaching destination: {dest}")
    
    # Analyze trajectory
    print("\n" + "=" * 70)
    print("TRAJECTORY ANALYSIS")
    print("=" * 70)
    
    trajectory = observer.analyze_trajectory()
    
    print(f"\n## Trends")
    print(f"  φ-level trend: {trajectory['phi_level_trend']:.4f} per step")
    print(f"  Orthogonality trend: {trajectory['orthogonality_trend']:.6f} per step")
    print(f"  Rank trend: {trajectory['rank_trend']:.4f} per step")
    
    print(f"\n## Final State")
    print(f"  φ-level: {trajectory['final_phi_level']:.2f}")
    print(f"  Orthogonality: {trajectory['final_orthogonality']:.4f}")
    print(f"  Effective rank: {trajectory['final_rank']:.1f}")
    
    # Compare to DDColor destination
    print(f"\n## Comparison to DDColor")
    print(f"  DDColor queries: orthogonality=0.05, rank=100, φ-level≈-1.6")
    print(f"  Our queries: orthogonality={trajectory['final_orthogonality']:.4f}, rank={trajectory['final_rank']:.1f}, φ-level={trajectory['final_phi_level']:.2f}")
    
    # Visualize trajectory
    print("\n## φ-Level Trajectory")
    for s in observer.snapshots:
        bar_len = int((s.mean_phi_level + 15) * 2)
        bar = "█" * max(0, bar_len)
        print(f"  Step {s.step:3d}: {s.mean_phi_level:6.2f} {bar}")
    
    return observer


def analyze_early_signals():
    """Analyze what early signals predict about convergence."""
    print("\n" + "=" * 70)
    print("EARLY SIGNAL ANALYSIS")
    print("=" * 70)
    
    print("""
## Key Questions

1. How early can we detect the destination?
   - After 10 steps? 50 steps? 100 steps?
   
2. What signals are most predictive?
   - φ-level trajectory?
   - Orthogonality trend?
   - Distance to known destinations?
   
3. Can we jump early?
   - If we detect destination at step 50, can we jump there?
   - What's the accuracy vs training to completion?

## Next Steps

1. Run multiple training runs with different seeds
2. Track when destination becomes predictable
3. Test early jumping and measure accuracy
4. Build a "geometric accelerator" that jumps when confident
""")


if __name__ == "__main__":
    observer = run_training_observation()
    analyze_early_signals()
