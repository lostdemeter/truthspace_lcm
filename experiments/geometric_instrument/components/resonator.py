"""
Module 5: Resonator — Geometric Fabry-Pérot Cavity
====================================================

Resonant amplification that creates a clean rank-1 score matrix
from the bias outer product, overwhelming the full-rank weight-weight
noise. Locks the instrument onto the selected entity.

Optical analog: Fabry-Pérot resonant cavity
Characteristic dimensionality: 1 (rank-1 outer product)

Specification:
    Amplification:     S[0] / S[1] > 100,000 (rank-1 dominance)
    Mechanism:         Bias outer product: b_q ⊗ b_k
    Bias/weight ratio: > 40× (bias overwhelms weights)
    Score matrix:      Effectively rank-1 after resonance
    Storage:           2 × head_dim bias values
                       Ideal: 0 (all-negative → formula)
"""

import numpy as np


class Resonator:
    """Resonant amplification. Creates rank-1 score matrix from biases."""

    def __init__(self, b_q, b_k, scale=1.0):
        """Initialize with bias vectors and attention scale.
        
        Args:
            b_q: Query bias vector [head_dim]
            b_k: Key bias vector [head_dim]
            scale: Attention scaling factor (1/sqrt(head_dim))
        """
        self.b_q = np.asarray(b_q, dtype=np.float32).copy()
        self.b_k = np.asarray(b_k, dtype=np.float32).copy()
        self.scale = float(scale)
        self.head_dim = self.b_q.shape[0]

    def score_matrix(self, seq_len):
        """Compute the rank-1 bias score matrix.
        
        The bias outer product b_q ⊗ b_k produces a rank-1 matrix
        that is constant across all positions (same score everywhere).
        This is the "resonance" — it amplifies the selected frequency.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            [seq_len, seq_len] rank-1 score matrix (before causal mask)
        """
        # b_q · b_k is a scalar — the resonance amplitude
        bias_dot = float(np.dot(self.b_q, self.b_k)) * self.scale
        # Rank-1: every entry is the same (uniform baseline)
        return np.full((seq_len, seq_len), bias_dot, dtype=np.float32)

    def resonate(self, Q_head, K_head):
        """Apply resonance to pre-computed Q and K projections.
        
        The full score = (Q @ K^T) * scale, where Q and K already
        include bias. The resonator's contribution is the bias-bias
        component which creates the rank-1 structure.
        
        For verification, we compute the full scores and show they
        are dominated by the rank-1 bias component.
        
        Args:
            Q_head: [seq_len, head_dim] query projections (with bias)
            K_head: [seq_len, head_dim] key projections (with bias)
            
        Returns:
            [seq_len, seq_len] attention scores
        """
        scores = (Q_head @ K_head.T) * self.scale
        return scores

    def attention_weights(self, Q_head, K_head, causal=True):
        """Compute attention weights with causal masking and softmax.
        
        Args:
            Q_head: [seq_len, head_dim]
            K_head: [seq_len, head_dim]
            causal: Apply causal mask
            
        Returns:
            [seq_len, seq_len] attention weight matrix
        """
        scores = self.resonate(Q_head, K_head)
        seq_len = scores.shape[0]
        if causal and seq_len > 1:
            mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            scores = scores + mask
        # Softmax along last axis
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return weights

    def measure_rank1(self, Q_head=None, K_head=None):
        """Measure the rank-1 dominance of the score matrix.
        
        If Q_head/K_head are provided, measures the full QK score matrix.
        Otherwise measures just the bias outer product.
        
        Returns:
            dict with s0, s1, ratio, is_rank1
        """
        if Q_head is not None and K_head is not None:
            M = (Q_head @ K_head.T) * self.scale
        else:
            # Pure bias outer product
            M = np.outer(self.b_q, self.b_k) * self.scale
        
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        s0 = float(S[0]) if len(S) > 0 else 0.0
        s1 = float(S[1]) if len(S) > 1 else 0.0
        ratio = s0 / s1 if s1 > 1e-12 else float('inf')
        
        return {
            's0': s0,
            's1': s1,
            'ratio': ratio,
            'is_rank1': ratio > 1000,
        }

    @classmethod
    def from_model(cls, engine, layer_idx, head_idx):
        """Extract Resonator biases from a real model.
        
        Args:
            engine: PhiQwen2Engine instance
            layer_idx: Layer index (e.g. 23)
            head_idx: Head index (e.g. 6)
            
        Returns:
            Resonator instance with biases from the model.
        """
        attn = engine.layers[layer_idx].attention
        hd = attn.head_dim
        nh = attn.num_heads
        nkv = attn.num_kv_heads
        kv = head_idx // (nh // nkv)
        
        b_q = attn.b_q[head_idx * hd:(head_idx + 1) * hd]
        b_k = attn.b_k[kv * hd:(kv + 1) * hd]
        
        return cls(b_q, b_k, scale=attn.scale)

    def spec(self):
        """Return specification measurements."""
        rank1 = self.measure_rank1()
        return {
            'head_dim': self.head_dim,
            'scale': self.scale,
            'b_q_norm': float(np.linalg.norm(self.b_q)),
            'b_k_norm': float(np.linalg.norm(self.b_k)),
            'bias_dot': float(np.dot(self.b_q, self.b_k)),
            'rank1_ratio': rank1['ratio'],
            'is_rank1': rank1['is_rank1'],
        }
