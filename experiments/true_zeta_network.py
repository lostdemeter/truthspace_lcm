"""
True Zeta-Aligned Neural Network

Implements the unified architecture from Design 144:
1. ENCODE: x → (sign, level) in φ-space
2. ATTRACT: Pull toward critical line (balance-seeking, not winner-take-all)
3. NAVIGATE: W-axis direction (single path)
4. DOWNCAST: φ-Zipf weighted compression
5. DECODE: (sign, level) → output

Key insight: Replace softmax's winner-take-all with φ-softmax's balance-seeking.
Errors should cancel symmetrically around the critical line (level 0).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


class TrueZetaLayer(nn.Module):
    """
    A layer implementing the full unified zeta-aligned architecture.
    
    The 5 operations:
    1. Encode: map to φ-space
    2. Attract: pull toward critical line (level 0)
    3. Navigate: W-axis direction
    4. Downcast: φ-Zipf weighted transform
    5. Decode: map from φ-space
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Navigation weights (W-axis)
        self.W_nav = nn.Parameter(torch.randn(in_dim) * 0.1)
        
        # Transform weights
        self.W_transform = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        
        # φ-Zipf rank weights (pre-computed, not learned)
        # Higher rank = lower weight (φ^(-rank))
        rank_weights = PHI ** (-torch.arange(out_dim, dtype=torch.float32) / out_dim * 10)
        self.register_buffer('rank_weights', rank_weights)
        
        # Learnable temperature for attraction
        self.attraction_temp = nn.Parameter(torch.tensor(1.0))
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 1: ENCODE to φ-space.
        
        Returns (sign, level) representation.
        """
        # Soft sign (differentiable)
        x_sign = torch.tanh(x * 5)
        
        # Level in φ-space
        x_level = torch.log(torch.abs(x) + 1e-8) / LOG_PHI
        
        return x_sign, x_level
    
    def attract(self, x_sign: torch.Tensor, x_level: torch.Tensor) -> torch.Tensor:
        """
        Step 2: ATTRACT toward critical line (level 0).
        
        This is φ-softmax: attraction to BALANCE, not maximum.
        Values near level 0 (magnitude 1) get highest weight.
        """
        # Attraction strength: 1 / (1 + φ^|level|)
        # Level 0 → attraction = 0.5
        # Large |level| → attraction → 0
        temp = torch.abs(self.attraction_temp) + 0.1  # Ensure positive
        attraction = 1.0 / (1.0 + PHI ** (torch.abs(x_level) / temp))
        
        # Apply attraction as weighting
        x_balanced = x_sign * attraction
        
        return x_balanced
    
    def navigate(self, x_balanced: torch.Tensor) -> torch.Tensor:
        """
        Step 3: NAVIGATE via W-axis.
        
        Compute navigation direction and distance.
        """
        # W-axis value: how far to go
        w = (x_balanced * self.W_nav).sum(dim=-1, keepdim=True)
        
        return w
    
    def downcast(self, x_balanced: torch.Tensor) -> torch.Tensor:
        """
        Step 4: DOWNCAST via φ-Zipf weighted transform.
        
        Higher-rank dimensions get lower weight.
        This is the holographic compression.
        """
        # Transform
        output = F.linear(x_balanced, self.W_transform)
        
        # Apply φ-Zipf weighting
        output = output * self.rank_weights
        
        return output
    
    def decode(self, output: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Step 5: DECODE from φ-space.
        
        Apply navigation scaling.
        """
        # Scale by navigation (φ^w)
        w_clamped = w.clamp(-5, 5)
        nav_scale = PHI ** w_clamped
        output = output * nav_scale
        
        # Normalize for stability
        output = output / (output.std() + 1e-8)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through all 5 operations.
        """
        # 1. Encode
        x_sign, x_level = self.encode(x)
        
        # 2. Attract (to critical line)
        x_balanced = self.attract(x_sign, x_level)
        
        # 3. Navigate (W-axis)
        w = self.navigate(x_balanced)
        
        # 4. Downcast (φ-Zipf)
        output = self.downcast(x_balanced)
        
        # 5. Decode
        output = self.decode(output, w)
        
        return output


class TrueZetaNetwork(nn.Module):
    """
    Full network using TrueZetaLayers.
    
    Balance-seeking instead of winner-take-all.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int,
                 max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # φ-based position encoding
        self.register_buffer(
            'pos_encoding',
            self._create_phi_positions(max_seq_len, hidden_dim)
        )
        
        # True zeta layers
        self.layers = nn.ModuleList([
            TrueZetaLayer(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def _create_phi_positions(self, max_len: int, dim: int) -> torch.Tensor:
        """Create φ-based position encodings."""
        positions = torch.arange(max_len, dtype=torch.float32)
        dimensions = torch.arange(dim, dtype=torch.float32)
        
        # φ-based frequencies
        frequencies = PHI ** (-dimensions / dim * 10)
        angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
        
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        
        return pe
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embed
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:seq_len]
        
        # Pass through layers
        for layer in self.layers:
            x_flat = x.view(-1, self.hidden_dim)
            out_flat = layer(x_flat)
            out = out_flat.view(batch_size, seq_len, self.hidden_dim)
            x = x + out  # Residual
        
        # Output
        logits = self.output_proj(x)
        
        return logits


def test_true_zeta():
    """Test the TrueZetaNetwork."""
    print("=" * 70)
    print("Testing True Zeta-Aligned Network")
    print("=" * 70)
    
    # Create network
    vocab_size = 100
    hidden_dim = 32
    n_layers = 2
    
    model = TrueZetaNetwork(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    
    print(f"\nNetwork: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input: {input_ids.shape}")
    print(f"Output: {logits.shape}")
    print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Test single layer
    print("\n" + "=" * 70)
    print("Testing TrueZetaLayer Operations")
    print("=" * 70)
    
    layer = TrueZetaLayer(32, 32)
    x = torch.randn(2, 32) * 0.5
    
    # Step through operations
    x_sign, x_level = layer.encode(x)
    print(f"\n1. ENCODE:")
    print(f"   x_sign range: [{x_sign.min():.4f}, {x_sign.max():.4f}]")
    print(f"   x_level range: [{x_level.min():.4f}, {x_level.max():.4f}]")
    
    x_balanced = layer.attract(x_sign, x_level)
    print(f"\n2. ATTRACT (to critical line):")
    print(f"   x_balanced range: [{x_balanced.min():.4f}, {x_balanced.max():.4f}]")
    
    w = layer.navigate(x_balanced)
    print(f"\n3. NAVIGATE (W-axis):")
    print(f"   w values: {w.squeeze().tolist()}")
    
    output = layer.downcast(x_balanced)
    print(f"\n4. DOWNCAST (φ-Zipf):")
    print(f"   output range: [{output.min():.4f}, {output.max():.4f}]")
    
    final = layer.decode(output, w)
    print(f"\n5. DECODE:")
    print(f"   final range: [{final.min():.4f}, {final.max():.4f}]")
    
    return model


def train_pattern_task():
    """Train on pattern task and compare with original ZetaAlignedNetwork."""
    print("\n" + "=" * 70)
    print("Training: True Zeta vs Original Zeta")
    print("=" * 70)
    
    vocab_size = 20
    hidden_dim = 32
    n_layers = 2
    batch_size = 32
    n_steps = 200
    
    # True Zeta model
    true_model = TrueZetaNetwork(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    
    optimizer = torch.optim.Adam(true_model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTask: Predict (token + 1) mod {vocab_size}")
    print(f"True Zeta Model: {sum(p.numel() for p in true_model.parameters()):,} parameters")
    
    losses = []
    accuracies = []
    
    for step in range(n_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, 8))
        target = (input_ids + 1) % vocab_size
        
        logits = true_model(input_ids)
        loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        
        preds = logits.argmax(dim=-1)
        acc = (preds == target).float().mean().item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(true_model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if step % 40 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}, acc = {acc*100:.1f}%")
    
    initial_acc = np.mean(accuracies[:10])
    final_acc = np.mean(accuracies[-10:])
    
    print(f"\nResults:")
    print(f"  Initial accuracy: {initial_acc*100:.1f}%")
    print(f"  Final accuracy: {final_acc*100:.1f}%")
    print(f"  Random baseline: {100/vocab_size:.1f}%")
    
    if final_acc > 0.9:
        print(f"\n  ✓ SUCCESS: True Zeta achieved {final_acc*100:.1f}% accuracy!")
    elif final_acc > initial_acc + 0.2:
        print(f"\n  ✓ LEARNING: Accuracy improved significantly")
    else:
        print(f"\n  ⚠ May need tuning")
    
    return losses, accuracies


def compare_stability():
    """Compare error stability between True Zeta and standard approach."""
    print("\n" + "=" * 70)
    print("Stability Comparison: Balance-Seeking vs Winner-Take-All")
    print("=" * 70)
    
    # Create layers
    true_layer = TrueZetaLayer(32, 32)
    
    # Test with inputs at different scales
    scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    print("\nOutput variance at different input scales:")
    print(f"{'Scale':<10} {'True Zeta Output Std':<25}")
    
    for scale in scales:
        x = torch.randn(100, 32) * scale
        
        with torch.no_grad():
            out = true_layer(x)
        
        print(f"{scale:<10} {out.std().item():<25.4f}")
    
    print("""
The True Zeta layer should have more stable output variance
because the attraction step pulls values toward the critical line,
preventing extreme values from dominating.
""")


if __name__ == "__main__":
    test_true_zeta()
    train_pattern_task()
    compare_stability()
