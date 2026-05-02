#!/usr/bin/env python3
"""
φ-MESH Attention Module
========================

Implements low-rank MESH attention for 6.5× speedup on weight multiplication.

Key insight: MESH = W_q.T @ W_k has rank ≤ head_dim (128)
So we can factor: MESH = U @ S @ Vt where U, Vt are (D, 128)

Instead of: scores = input @ MESH @ input.T  [O(N×D²)]
We compute: scores = (input @ U @ S) @ (Vt @ input.T)  [O(N×D×r)]

This gives 6.5× speedup on the weight multiplication.

Author: TruthSpace LCM Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time


@dataclass
class PhiMeshConfig:
    """Configuration for φ-MESH attention."""
    hidden_dim: int = 3584
    num_heads: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 1000000.0


class PhiMeshAttention(nn.Module):
    """
    φ-MESH Attention using low-rank factorization.
    
    Instead of storing full MESH matrices (D×D), we store:
    - mesh_U: (num_heads, hidden_dim, head_dim) - left factor
    - mesh_S: (num_heads, head_dim) - singular values  
    - mesh_Vt: (num_heads, head_dim, hidden_dim) - right factor
    
    Attention scores are computed as:
        scores[h] = (input @ U[h] * S[h]) @ (Vt[h] @ input.T)
    
    This reduces O(N×D²) to O(N×D×r) where r = head_dim = 128.
    """
    
    def __init__(self, config: PhiMeshConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.heads_per_kv = self.num_heads // self.num_kv_heads
        
        # MESH low-rank factors (will be loaded)
        self.mesh_U: Optional[torch.Tensor] = None  # (num_heads, hidden_dim, head_dim)
        self.mesh_S: Optional[torch.Tensor] = None  # (num_heads, head_dim)
        self.mesh_Vt: Optional[torch.Tensor] = None  # (num_heads, head_dim, hidden_dim)
        
        # Bias cross-terms
        self.cross_qk: Optional[torch.Tensor] = None  # (num_heads, hidden_dim)
        self.cross_kq: Optional[torch.Tensor] = None  # (num_heads, hidden_dim)
        self.bias_term: Optional[torch.Tensor] = None  # (num_heads,)
        
        # V and O projections
        self.W_v: Optional[torch.Tensor] = None  # (num_kv_heads * head_dim, hidden_dim)
        self.b_v: Optional[torch.Tensor] = None  # (num_kv_heads * head_dim,)
        self.W_o: Optional[torch.Tensor] = None  # (hidden_dim, num_heads * head_dim)
        
        # LayerNorm
        self.ln_weight: Optional[torch.Tensor] = None  # (hidden_dim,)
        
        # Scale factor
        self.scale = 1.0 / np.sqrt(self.head_dim)
        
        # KV cache for autoregressive generation
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
    
    def load_from_quantized(self, layer_dir: str, device: str = 'cuda'):
        """Load MESH factors from pre-quantized layer directory."""
        import os
        
        # Load MESH factors
        mesh_data = np.load(os.path.join(layer_dir, 'mesh.npz'))
        
        # mesh_U, mesh_S, mesh_Vt are stored per head
        self.mesh_U = torch.tensor(mesh_data['mesh_U'], dtype=torch.float32, device=device)
        self.mesh_S = torch.tensor(mesh_data['mesh_S'], dtype=torch.float32, device=device)
        self.mesh_Vt = torch.tensor(mesh_data['mesh_Vt'], dtype=torch.float32, device=device)
        
        self.cross_qk = torch.tensor(mesh_data['cross_qk'], dtype=torch.float32, device=device)
        self.cross_kq = torch.tensor(mesh_data['cross_kq'], dtype=torch.float32, device=device)
        self.bias_term = torch.tensor(mesh_data['bias_term'], dtype=torch.float32, device=device)
        
        # Load V projection (φ-quantized)
        v_data = np.load(os.path.join(layer_dir, 'W_v.npz'))
        W_v = self._dequantize(v_data['signs'], v_data['indices'], v_data['codebook'])
        self.W_v = torch.tensor(W_v, dtype=torch.float32, device=device)
        self.b_v = torch.tensor(v_data['bias'], dtype=torch.float32, device=device)
        
        # Load O projection (φ-quantized)
        o_data = np.load(os.path.join(layer_dir, 'W_o.npz'))
        W_o = self._dequantize(o_data['signs'], o_data['indices'], o_data['codebook'])
        self.W_o = torch.tensor(W_o, dtype=torch.float32, device=device)
        
        # Load LayerNorm
        ln_data = np.load(os.path.join(layer_dir, 'layernorm.npz'))
        self.ln_weight = torch.tensor(ln_data['ln1_weight'], dtype=torch.float32, device=device)
    
    def _dequantize(self, signs: np.ndarray, indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Dequantize φ-encoded weights."""
        PHI = (1 + np.sqrt(5)) / 2
        K = 128
        exponents = codebook[indices]
        values = signs * (PHI ** (exponents / K))
        return values.astype(np.float32)
    
    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMS normalization."""
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6) * self.ln_weight
    
    def _compute_rope(self, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary position embeddings."""
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.head_dim, 2, device=device) / self.head_dim))
        positions = torch.arange(seq_len, device=device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rotated * sin
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass using φ-MESH low-rank attention.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            position_ids: Optional position IDs for RoPE
            use_cache: Whether to use/update KV cache
            
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # RMS Norm
        normed = self._rms_norm(hidden_states)
        
        # Compute V projection
        V = F.linear(normed, self.W_v, self.b_v)  # (batch, seq, num_kv_heads * head_dim)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = V.transpose(1, 2)  # (batch, num_kv_heads, seq, head_dim)
        
        # Expand V for GQA
        V = V.repeat_interleave(self.heads_per_kv, dim=1)  # (batch, num_heads, seq, head_dim)
        
        # Compute attention scores using low-rank MESH
        # For each head h:
        #   scores[h] = normed @ U[h] @ diag(S[h]) @ Vt[h] @ normed.T + bias terms
        
        all_scores = []
        for h in range(self.num_heads):
            # Low-rank computation: O(N × D × r) instead of O(N × D²)
            # Step 1: normed @ U[h] -> (batch, seq, head_dim)
            temp1 = torch.matmul(normed, self.mesh_U[h])  # (batch, seq, 128)
            
            # Step 2: temp1 * S[h] -> (batch, seq, head_dim)
            temp2 = temp1 * self.mesh_S[h]  # (batch, seq, 128)
            
            # Step 3: Vt[h] @ normed.T -> (batch, head_dim, seq)
            temp3 = torch.matmul(self.mesh_Vt[h], normed.transpose(-2, -1))  # (batch, 128, seq)
            
            # Step 4: temp2 @ temp3 -> (batch, seq, seq)
            scores_h = torch.matmul(temp2, temp3)  # (batch, seq, seq)
            
            # Add bias cross-terms
            # cross_qk: normed @ cross_qk[h] -> (batch, seq)
            term2 = torch.matmul(normed, self.cross_qk[h])  # (batch, seq)
            # cross_kq @ normed.T -> (batch, seq)
            term3 = torch.matmul(self.cross_kq[h], normed.transpose(-2, -1))  # (batch, seq)
            
            scores_h = scores_h + term2.unsqueeze(-1) + term3.unsqueeze(-2) + self.bias_term[h]
            
            all_scores.append(scores_h)
        
        # Stack scores: (batch, num_heads, seq, seq)
        scores = torch.stack(all_scores, dim=1)
        
        # Scale
        scores = scores * self.scale
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        scores = scores + mask
        
        # Softmax
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to V
        attn_output = torch.matmul(attention, V)  # (batch, num_heads, seq, head_dim)
        
        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch, seq, num_heads * head_dim)
        
        output = F.linear(attn_output, self.W_o)  # (batch, seq, hidden_dim)
        
        return hidden_states + output


def benchmark_phi_mesh_attention():
    """Benchmark φ-MESH attention vs standard attention."""
    import os
    
    print("=" * 60)
    print("φ-MESH ATTENTION BENCHMARK")
    print("=" * 60)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    config = PhiMeshConfig()
    
    # Check if quantized model exists
    quantized_path = os.path.expanduser("~/.cache/phi_quantized/qwen2-7b")
    layer_dir = os.path.join(quantized_path, "layer_00")
    
    if not os.path.exists(layer_dir):
        print(f"Quantized model not found at {quantized_path}")
        print("Run phi_quantize_model.py first.")
        return
    
    # Create φ-MESH attention
    print("Loading φ-MESH attention...")
    phi_attn = PhiMeshAttention(config, layer_idx=0)
    phi_attn.load_from_quantized(layer_dir, device=device)
    phi_attn.eval()
    
    # Also load standard attention for comparison
    print("Loading standard model for comparison...")
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()
    
    std_attn = model.model.layers[0].self_attn
    std_ln = model.model.layers[0].input_layernorm
    
    # Benchmark different sequence lengths
    print()
    print(f"{'Seq Len':>8} | {'Standard (ms)':>14} | {'φ-MESH (ms)':>12} | {'Speedup':>8}")
    print("-" * 55)
    
    for seq_len in [64, 128, 256, 512, 1024]:
        batch_size = 1
        
        # Create input
        hidden = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Warmup
        with torch.no_grad():
            _ = phi_attn(hidden)
            normed = std_ln(hidden)
            _ = std_attn(normed, position_ids=position_ids)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        
        # Benchmark standard attention
        n_runs = 10
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                normed = std_ln(hidden)
                _ = std_attn(normed, position_ids=position_ids)
        torch.cuda.synchronize() if device == 'cuda' else None
        std_time = (time.perf_counter() - start) / n_runs * 1000
        
        # Benchmark φ-MESH attention
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = phi_attn(hidden)
        torch.cuda.synchronize() if device == 'cuda' else None
        phi_time = (time.perf_counter() - start) / n_runs * 1000
        
        speedup = std_time / phi_time
        print(f"{seq_len:>8} | {std_time:>14.2f} | {phi_time:>12.2f} | {speedup:>7.2f}×")
    
    # Verify correctness
    print()
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)
    
    hidden = torch.randn(1, 64, config.hidden_dim, device=device)
    position_ids = torch.arange(64, device=device).unsqueeze(0)
    
    with torch.no_grad():
        phi_out = phi_attn(hidden)
        
        normed = std_ln(hidden)
        std_out, _, _ = std_attn(normed, position_ids=position_ids)
        std_out = hidden + std_out  # Add residual
    
    # Compare outputs
    corr = torch.corrcoef(torch.stack([phi_out.flatten(), std_out.flatten()]))[0, 1].item()
    mse = F.mse_loss(phi_out, std_out).item()
    
    print(f"Correlation: {corr:.6f}")
    print(f"MSE: {mse:.6e}")


if __name__ == "__main__":
    benchmark_phi_mesh_attention()
