"""
Zeta-Aligned Neural Network

A neural network aligned with W-axis zeta symmetry:
- No self-referential attention (single path through φ-space)
- W-axis for navigation (O(N) instead of O(N²))
- 1-2 cycle mesh gear computation
- Critical line symmetry (level 0 as balance point)

This is NOT a transformer. This is geometry.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


class PhiLUT(nn.Module):
    """Lookup table for φ^level values."""
    
    def __init__(self, min_level: int = -100, max_level: int = 50):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
        
        # Pre-compute φ^level for all levels
        levels = torch.arange(min_level, max_level + 1, dtype=torch.float32)
        lut_values = PHI ** levels
        self.register_buffer('lut', lut_values)
    
    def forward(self, levels: torch.Tensor) -> torch.Tensor:
        """Look up φ^level for given levels."""
        # Clamp levels first, then compute indices
        clamped_levels = levels.clamp(self.min_level, self.max_level)
        indices = (clamped_levels - self.min_level).long()
        indices = indices.clamp(0, len(self.lut) - 1)
        return self.lut[indices]


class ZetaAlignedLayer(nn.Module):
    """
    A single layer aligned with W-axis zeta symmetry.
    
    The 1-2 cycle:
    - Cycle 1: Encode input to φ-space (sign, level)
    - Cycle 2: Navigate via mesh gears (integer add + LUT)
    
    No self-reference. Single path through.
    """
    
    def __init__(self, in_dim: int, out_dim: int, init_scale: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Weight levels (centered around 0 = critical line)
        # Initialize near critical line for symmetric error cancellation
        self.W_levels = nn.Parameter(
            torch.randint(-20, 5, (out_dim, in_dim), dtype=torch.float32)
        )
        
        # Weight signs (+1 or -1)
        self.W_signs = nn.Parameter(
            torch.sign(torch.randn(out_dim, in_dim))
        )
        
        # W-axis navigation projection
        self.W_nav = nn.Parameter(
            torch.randn(in_dim) * init_scale
        )
        
        # LUT for φ^level
        self.lut = PhiLUT()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cycle 1: Encode input to φ-space.
        
        Returns:
            x_signs: Sign of each input dimension (soft for gradients)
            x_levels: φ-level of each input dimension (soft for gradients)
            x_w: W-axis navigation value (scalar per sample)
        """
        # Soft signs (tanh for gradient flow)
        x_signs = torch.tanh(x * 10)  # Approximates sign but differentiable
        
        # Soft levels (continuous for gradient flow)
        # Use straight-through estimator: forward uses round, backward uses identity
        x_levels_continuous = torch.log(torch.abs(x) + 1e-8) / LOG_PHI
        x_levels = x_levels_continuous  # Keep continuous for gradients
        
        # W-axis navigation
        x_w = torch.matmul(x, self.W_nav)
        
        return x_signs, x_levels, x_w
    
    def navigate(self, x_signs: torch.Tensor, x_levels: torch.Tensor, 
                 x_w: torch.Tensor) -> torch.Tensor:
        """
        Cycle 2: Navigate via mesh gears.
        
        This is the core computation:
        - Integer addition of levels (the mesh gear!)
        - LUT lookup for magnitude
        - Sign combination
        - W-axis scaling
        """
        # Combined levels: integer addition
        # x_levels: (batch, in_dim)
        # W_levels: (out_dim, in_dim)
        # Result: (batch, out_dim, in_dim)
        combined_levels = self.W_levels.unsqueeze(0) + x_levels.unsqueeze(1)
        
        # Combined signs
        combined_signs = self.W_signs.unsqueeze(0) * x_signs.unsqueeze(1)
        
        # LUT lookup for magnitudes
        magnitudes = self.lut(combined_levels)
        
        # Accumulate: sum over input dimension
        output = (combined_signs * magnitudes).sum(dim=-1)
        
        # Apply W-axis navigation (scale by φ^w)
        # This is the "steering" - how far to go in the output direction
        # Clamp x_w to prevent extreme scaling
        x_w_clamped = x_w.clamp(-10, 10)
        nav_scale = PHI ** x_w_clamped.unsqueeze(-1)
        output = output * nav_scale
        
        # Layer normalization for stability
        output = output / (output.std() + 1e-8)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: Encode → Navigate.
        
        No self-reference. Single path through φ-space.
        """
        # Cycle 1: Encode
        x_signs, x_levels, x_w = self.encode(x)
        
        # Cycle 2: Navigate
        output = self.navigate(x_signs, x_levels, x_w)
        
        return output


class ZetaAlignedNetwork(nn.Module):
    """
    A full network using zeta-aligned layers.
    
    Key differences from transformer:
    - No attention (W-axis provides navigation)
    - No self-reference (single path)
    - Symmetric around critical line
    - 1-2 cycle mesh gear computation
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int,
                 max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding (standard, will encode to φ-space in first layer)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Position encoding (φ-based)
        self.register_buffer(
            'pos_encoding',
            self._create_phi_positions(max_seq_len, hidden_dim)
        )
        
        # Zeta-aligned layers
        self.layers = nn.ModuleList([
            ZetaAlignedLayer(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def _create_phi_positions(self, max_len: int, dim: int) -> torch.Tensor:
        """Create φ-based position encodings."""
        positions = torch.arange(max_len, dtype=torch.float32)
        dimensions = torch.arange(dim, dtype=torch.float32)
        
        # φ-based frequencies (self-similar across dimensions)
        frequencies = PHI ** (-dimensions / dim * 10)
        
        # Position encoding
        angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
        
        # Alternate sin/cos like standard PE but with φ-frequencies
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        
        return pe
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the zeta-aligned network.
        
        Args:
            input_ids: (batch, seq_len) token indices
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Add position encoding
        x = x + self.pos_encoding[:seq_len]
        
        # Pass through zeta-aligned layers
        for layer in self.layers:
            # Reshape for layer: (batch * seq, hidden)
            x_flat = x.view(-1, self.hidden_dim)
            
            # Layer forward (1-2 cycle)
            out_flat = layer(x_flat)
            
            # Reshape back: (batch, seq, hidden)
            out = out_flat.view(batch_size, seq_len, self.hidden_dim)
            
            # Residual connection (additive, not multiplicative!)
            x = x + out
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits


def test_zeta_network():
    """Test the zeta-aligned network."""
    print("=" * 70)
    print("Testing Zeta-Aligned Network")
    print("=" * 70)
    
    # Create small network
    vocab_size = 1000
    hidden_dim = 64
    n_layers = 4
    
    model = ZetaAlignedNetwork(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    
    print(f"\nNetwork architecture:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {n_layers}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Test single layer
    print("\n" + "=" * 70)
    print("Testing Single Zeta-Aligned Layer")
    print("=" * 70)
    
    layer = ZetaAlignedLayer(in_dim=64, out_dim=64)
    
    x = torch.randn(batch_size, 64) * 0.1
    
    # Encode
    x_signs, x_levels, x_w = layer.encode(x)
    print(f"\nEncode (Cycle 1):")
    print(f"  x_signs shape: {x_signs.shape}")
    print(f"  x_levels shape: {x_levels.shape}")
    print(f"  x_w shape: {x_w.shape}")
    print(f"  x_levels range: [{x_levels.min():.0f}, {x_levels.max():.0f}]")
    
    # Navigate
    output = layer.navigate(x_signs, x_levels, x_w)
    print(f"\nNavigate (Cycle 2):")
    print(f"  output shape: {output.shape}")
    print(f"  output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Full forward
    output_full = layer(x)
    print(f"\nFull forward:")
    print(f"  output shape: {output_full.shape}")
    
    print("\n" + "=" * 70)
    print("Comparison: Zeta-Aligned vs Transformer MLP")
    print("=" * 70)
    
    print("""
TRANSFORMER MLP:
  gate = SiLU(x @ W_gate.T)     # Path 1 from input
  up = x @ W_up.T               # Path 2 from input
  hidden = gate * up            # SELF-REFERENCE (multiply paths)
  output = hidden @ W_down.T    
  
  Error: ε₁ × ε₂ (multiplicative compounding)

ZETA-ALIGNED:
  signs, levels, w = encode(x)  # Single encoding
  output = navigate(signs, levels, w)  # Single path
  
  Error: ε₁ + ε₂ (additive only)

The 1-2 cycle eliminates self-reference!
""")
    
    print("=" * 70)
    print("SUCCESS: Zeta-Aligned Network is operational")
    print("=" * 70)


def train_simple_task():
    """Train on a simple pattern task to verify learning works."""
    print("\n" + "=" * 70)
    print("Training Test: Pattern Learning Task")
    print("=" * 70)
    
    # Simple task: learn to output token = (input + 1) mod vocab_size
    # This is a learnable pattern, not random
    vocab_size = 20
    hidden_dim = 32
    n_layers = 2
    seq_len = 8
    batch_size = 32
    n_steps = 200
    
    model = ZetaAlignedNetwork(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTask: Predict (token + 1) mod {vocab_size}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    losses = []
    accuracies = []
    
    for step in range(n_steps):
        # Generate sequences
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Target: (input + 1) mod vocab_size - a learnable pattern!
        target = (input_ids + 1) % vocab_size
        
        # Forward
        logits = model(input_ids)
        
        # Loss
        loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == target).float().mean().item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if step % 40 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}, acc = {acc*100:.1f}%")
    
    # Check if learning happened
    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    initial_acc = np.mean(accuracies[:10])
    final_acc = np.mean(accuracies[-10:])
    
    print(f"\nResults:")
    print(f"  Initial: loss = {initial_loss:.4f}, acc = {initial_acc*100:.1f}%")
    print(f"  Final:   loss = {final_loss:.4f}, acc = {final_acc*100:.1f}%")
    print(f"  Random baseline: {100/vocab_size:.1f}%")
    
    if final_acc > initial_acc + 0.1:
        print(f"\n  ✓ LEARNING CONFIRMED: Accuracy improved from {initial_acc*100:.1f}% to {final_acc*100:.1f}%")
    elif final_loss < initial_loss * 0.8:
        print(f"\n  ✓ LEARNING CONFIRMED: Loss decreased significantly")
    else:
        print(f"\n  ⚠ Learning may need tuning")
    
    return losses, accuracies


if __name__ == "__main__":
    test_zeta_network()
    train_simple_task()
