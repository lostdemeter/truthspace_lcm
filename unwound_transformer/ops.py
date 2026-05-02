"""
Core Mathematical Operations for Unwound Transformer
=====================================================

All operations are implemented in float64 for numerical stability.
Each function is designed to be inspectable for geometric analysis.
"""

import numpy as np
from typing import Tuple


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    RMS Layer Normalization (no mean centering, no bias).
    
    Formula: RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
    
    Args:
        x: Input vector of shape (d,)
        weight: Scale weight γ of shape (d,)
        eps: Numerical stability constant
    
    Returns:
        Normalized vector of shape (d,)
    """
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def compute_rope_embeddings(seq_len: int, inv_freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute RoPE cos/sin for all positions.
    
    Formula: θ_i = 10000^(-2i/d_h), freqs[pos, i] = pos * θ_i
    
    Args:
        seq_len: Number of positions
        inv_freq: Inverse frequencies of shape (head_dim/2,)
    
    Returns:
        cos, sin: Both of shape (seq_len, head_dim)
    """
    positions = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(positions, inv_freq)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(freqs), np.sin(freqs)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply Rotary Position Embedding to a vector.
    
    Formula: RoPE(x) = x ⊙ cos(θ) + rotate_half(x) ⊙ sin(θ)
    Where rotate_half([a, b]) = [-b, a] applied to each pair.
    
    Args:
        x: Input vector of shape (head_dim,)
        cos: Cosine values of shape (head_dim,)
        sin: Sine values of shape (head_dim,)
    
    Returns:
        Rotated vector of shape (head_dim,)
    """
    half = len(x) // 2
    x1, x2 = x[:half], x[half:]
    x_rotated = np.concatenate([-x2, x1])
    return x * cos + x_rotated * sin


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Swish) activation function.
    
    Formula: SiLU(x) = x * σ(x) = x / (1 + e^(-x))
    
    Args:
        x: Input array of any shape
    
    Returns:
        Activated array of same shape
    """
    return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))


def softmax(scores: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    
    Formula: softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    
    Args:
        scores: Input scores of shape (n,)
    
    Returns:
        Probability distribution of shape (n,)
    """
    exp_scores = np.exp(scores - scores.max())
    return exp_scores / exp_scores.sum()


def attention_scores(q: np.ndarray, keys: list, head_dim: int) -> np.ndarray:
    """
    Compute scaled dot-product attention scores.
    
    Formula: score_i = (q · k_i) / √d_h
    
    Args:
        q: Query vector of shape (head_dim,)
        keys: List of key vectors, each of shape (head_dim,)
        head_dim: Dimension of each head (for scaling)
    
    Returns:
        Attention weights after softmax, shape (n_keys,)
    """
    scores = np.array([np.dot(q, k) for k in keys]) / np.sqrt(head_dim)
    return softmax(scores)


def gated_mlp(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, 
              W_down: np.ndarray) -> np.ndarray:
    """
    Gated MLP with SiLU activation.
    
    Formula: MLP(x) = W_down @ (SiLU(x @ W_gate.T) ⊙ (x @ W_up.T))
    
    Args:
        x: Input vector of shape (hidden_dim,)
        W_gate: Gate projection of shape (intermediate_dim, hidden_dim)
        W_up: Up projection of shape (intermediate_dim, hidden_dim)
        W_down: Down projection of shape (hidden_dim, intermediate_dim)
    
    Returns:
        Output vector of shape (hidden_dim,)
    """
    gate = silu(x @ W_gate.T)
    up = x @ W_up.T
    return (gate * up) @ W_down.T
