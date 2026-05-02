"""
φ-Attention: Multi-head attention with RoPE for Qwen2-7B.

Architecture:
  Q = normed @ W_q.T + b_q    → (batch, seq, num_heads * head_dim)
  K = normed @ W_k.T + b_k    → (batch, seq, num_kv_heads * head_dim)
  V = normed @ W_v.T + b_v    → (batch, seq, num_kv_heads * head_dim)

  Apply RoPE to Q, K
  Expand K, V for GQA (4 KV heads → 28 Q heads, 7× repeat)

  scores = (Q @ K.T) / sqrt(head_dim) + causal_mask
  attn = softmax(scores)
  output = attn @ V

  result = output @ W_o.T + residual

Phase 3 additions:
  KVCache — stores K/V tensors per layer for autoregressive generation
  Incremental decode — process single tokens against cached K/V
"""

import numpy as np
from typing import Optional, List
from .phi_types import PhiEncoded
from .phi_matmul import phi_linear
from .phi_components import rms_norm, phi_softmax


class RoPE:
    """
    Rotary Position Embedding.

    Qwen2-7B uses rope_theta=1_000_000 and head_dim=128.
    Pre-computes cos/sin tables for reuse across layers.
    """

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0,
                 max_seq_len: int = 4096):
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        # Inverse frequencies: 1 / (theta^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))

        # Position × frequency outer product
        positions = np.arange(max_seq_len, dtype=np.float64)
        freqs = np.outer(positions, inv_freq)  # (max_seq, head_dim/2)

        # Duplicate for full dimension
        emb = np.concatenate([freqs, freqs], axis=-1)  # (max_seq, head_dim)

        self.cos_cached = np.cos(emb).astype(np.float32)  # (max_seq, head_dim)
        self.sin_cached = np.sin(emb).astype(np.float32)

    def apply(self, x: np.ndarray, seq_offset: int = 0) -> np.ndarray:
        """
        Apply RoPE to x.

        Args:
            x: (batch, num_heads, seq_len, head_dim)
            seq_offset: position offset for KV cache continuation

        Returns:
            rotated x, same shape
        """
        seq_len = x.shape[2]
        cos = self.cos_cached[seq_offset:seq_offset + seq_len]  # (seq, head_dim)
        sin = self.sin_cached[seq_offset:seq_offset + seq_len]

        # Rotate: split into halves and swap with sign flip
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        x_rotated = np.concatenate([-x2, x1], axis=-1)

        # Broadcast cos/sin to (1, 1, seq, head_dim)
        cos = cos[np.newaxis, np.newaxis, :, :]
        sin = sin[np.newaxis, np.newaxis, :, :]

        return x * cos + x_rotated * sin


class KVCache:
    """
    Key-Value cache for autoregressive generation.

    Stores K and V tensors for each layer at the KV-head level
    (before GQA expansion) to minimize memory.

    Memory per layer: 2 × num_kv_heads × seq_len × head_dim × 4 bytes
    For Qwen2-7B at 1024 tokens: 2 × 4 × 1024 × 128 × 4 = 4 MB/layer
    28 layers × 4 MB = 112 MB total — very manageable.
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Per-layer caches: (batch=1, kv_heads, seq_so_far, head_dim)
        self.k_cache: List[Optional[np.ndarray]] = [None] * num_layers
        self.v_cache: List[Optional[np.ndarray]] = [None] * num_layers
        self.seq_len = 0

    def update(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        """
        Append new K/V to the cache for a given layer.

        Args:
            layer_idx: which layer
            new_k: (batch, kv_heads, new_seq, head_dim) — new keys
            new_v: (batch, kv_heads, new_seq, head_dim) — new values
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
        else:
            self.k_cache[layer_idx] = np.concatenate(
                [self.k_cache[layer_idx], new_k], axis=2)
            self.v_cache[layer_idx] = np.concatenate(
                [self.v_cache[layer_idx], new_v], axis=2)

        # Track total sequence length from layer 0
        if layer_idx == 0:
            self.seq_len = self.k_cache[0].shape[2]

    def get(self, layer_idx: int):
        """Return (k_cache, v_cache) for a layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        """Clear all cached state."""
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers
        self.seq_len = 0


class PhiAttention:
    """
    Multi-head attention block for one transformer layer.

    Loads Q/K/V/O projections from φ-encoded weights.
    Supports GQA (Grouped Query Attention): 28 Q heads, 4 KV heads.

    Two modes:
      - Prefill: full sequence, builds KV cache
      - Decode:  single token, reads from KV cache
    """

    def __init__(self, W_q: PhiEncoded, W_k: PhiEncoded, W_v: PhiEncoded,
                 W_o: PhiEncoded, b_q: np.ndarray, b_k: np.ndarray,
                 b_v: np.ndarray, norm_weight: np.ndarray, rope: RoPE,
                 num_heads: int = 28, num_kv_heads: int = 4,
                 head_dim: int = 128):
        self.W_q = W_q
        self.W_k = W_k
        self.W_v = W_v
        self.W_o = W_o
        self.b_q = b_q
        self.b_k = b_k
        self.b_v = b_v
        self.norm_weight = norm_weight
        self.rope = rope

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.heads_per_kv = num_heads // num_kv_heads
        self.scale = 1.0 / np.sqrt(head_dim)

    def __call__(self, hidden: np.ndarray, pure: bool = False,
                 kv_cache: Optional[KVCache] = None,
                 layer_idx: int = 0) -> np.ndarray:
        """
        Forward pass through attention.

        Args:
            hidden: (batch, seq_len, hidden_dim)
            pure: use pure φ-integer matmul
            kv_cache: if provided, use/update KV cache for generation
            layer_idx: which layer index (for KV cache addressing)

        Returns:
            output: hidden + attention_output, same shape
        """
        batch, seq_len, hidden_dim = hidden.shape

        # Pre-attention RMSNorm
        normed = rms_norm(hidden, self.norm_weight)

        # Q/K/V projections
        Q = phi_linear(self.W_q, normed, self.b_q, pure=pure)  # (batch, seq, num_heads*head_dim)
        K = phi_linear(self.W_k, normed, self.b_k, pure=pure)  # (batch, seq, num_kv_heads*head_dim)
        V = phi_linear(self.W_v, normed, self.b_v, pure=pure)  # (batch, seq, num_kv_heads*head_dim)

        # Reshape for multi-head
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        V = V.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq, dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Determine position offset for RoPE
        if kv_cache is not None and kv_cache.k_cache[layer_idx] is not None:
            seq_offset = kv_cache.k_cache[layer_idx].shape[2]
        else:
            seq_offset = 0

        # Apply RoPE to Q and K with correct positions
        Q = self.rope.apply(Q, seq_offset=seq_offset)
        K = self.rope.apply(K, seq_offset=seq_offset)

        # Update KV cache (store at KV-head level, before GQA expansion)
        if kv_cache is not None:
            kv_cache.update(layer_idx, K, V)
            # Use full cached K/V for attention
            K_full, V_full = kv_cache.get(layer_idx)
        else:
            K_full, V_full = K, V

        # Expand K, V for GQA: repeat each KV head for its Q head group
        K_expanded = np.repeat(K_full, self.heads_per_kv, axis=1)
        V_expanded = np.repeat(V_full, self.heads_per_kv, axis=1)

        # Attention scores: Q @ K.T / sqrt(d)
        # Q: (batch, num_heads, q_len, dim)
        # K_expanded: (batch, num_heads, kv_len, dim)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K_expanded) * self.scale

        # Causal mask: prevent attending to future tokens
        kv_len = K_expanded.shape[2]
        if kv_len > 1 and seq_len > 1:
            # Prefill: full causal mask
            causal_mask = np.triu(np.full((seq_len, kv_len), -1e9, dtype=np.float32), k=1)
            scores = scores + causal_mask
        # Decode (seq_len=1): no mask needed — single query attends to all past

        # Softmax (φ-form is exact equivalent of standard softmax)
        attn_weights = phi_softmax(scores, axis=-1)

        # Weighted sum of values
        attn_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, V_expanded)

        # Reshape back: (batch, seq, num_heads * head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # Output projection (no bias in Qwen2 o_proj)
        attn_output = phi_linear(self.W_o, attn_output, pure=pure)

        # Residual connection
        return hidden + attn_output
