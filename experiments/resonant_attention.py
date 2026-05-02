#!/usr/bin/env python3
"""
Resonant Attention: Quantum-Inspired Attention via Resfrac

The hypothesis: Attention is a quantum interference problem.
Instead of computing O(N²) interactions, find the resonant modes.

This experiment tests:
1. Whether attention matrices have resonant structure (φ-Zipf SVD)
2. Whether we can identify "boom" positions without full attention
3. Whether resfrac-style optimization can find optimal attention patterns

Author: TruthSpace LCM Team
Date: 2026-01-29
"""

import numpy as np
import sys
from typing import List, Tuple, Optional
import time

# Add resfrac to path if available
sys.path.insert(0, '/home/thorin/resfrac')

PHI = 1.6180339887498949
INV_PHI = 1.0 / PHI


class AttentionGraph:
    """
    Map attention to a graph problem for resonant solving.
    
    Like TSP, but instead of finding optimal tour through cities,
    we find optimal attention weights through tokens.
    """
    type = 'attention'
    
    def __init__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray):
        """
        Args:
            Q: Query vectors (seq_len, head_dim) or (head_dim,) for single query
            K: Key vectors (seq_len, head_dim)
            V: Value vectors (seq_len, head_dim)
        """
        self.Q = Q if Q.ndim == 2 else Q.reshape(1, -1)
        self.K = K
        self.V = V
        self.seq_len = K.shape[0]
        self.head_dim = K.shape[1]
        
        # Precompute attention scores (the "distance matrix" analog)
        self.scores = self._compute_scores()
        
        # Precompute full attention for comparison
        self.full_attention = self._compute_full_attention()
    
    def _compute_scores(self) -> np.ndarray:
        """Compute Q·K^T / sqrt(d) scores."""
        scale = 1.0 / np.sqrt(self.head_dim)
        return (self.Q @ self.K.T) * scale  # (n_queries, seq_len)
    
    def _compute_full_attention(self) -> np.ndarray:
        """Compute full softmax attention."""
        # Softmax
        scores_max = np.max(self.scores, axis=-1, keepdims=True)
        exp_scores = np.exp(self.scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Weighted sum of V
        return weights @ self.V  # (n_queries, head_dim)
    
    def sparse_attention(self, positions: List[int]) -> np.ndarray:
        """
        Compute attention using only specified positions.
        
        This is the "resonant" attention - only attend to boom positions.
        """
        if len(positions) == 0:
            return np.zeros((self.Q.shape[0], self.head_dim))
        
        # Get scores for selected positions
        sparse_scores = self.scores[:, positions]
        
        # Softmax over selected positions
        scores_max = np.max(sparse_scores, axis=-1, keepdims=True)
        exp_scores = np.exp(sparse_scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Weighted sum of V at selected positions
        V_selected = self.V[positions]
        return weights @ V_selected


class ResonantAttentionSolver:
    """
    Solve attention using resonant optimization.
    
    Instead of computing all N² interactions:
    1. Find resonant positions (high attention weight)
    2. Compute attention only at those positions
    3. Use φ-guided search to find optimal positions
    """
    
    def __init__(self, max_iters: int = 50, boom_ratio: float = 0.2):
        """
        Args:
            max_iters: Maximum iterations for optimization
            boom_ratio: Target ratio of positions to keep (0.2 = 20%)
        """
        self.max_iters = max_iters
        self.boom_ratio = boom_ratio
        self.history = []
    
    def detect_booms_entropy(self, graph: AttentionGraph) -> List[int]:
        """
        Detect boom positions using entropy analysis.
        
        Positions with low entropy (concentrated attention) are booms.
        """
        # Compute per-position "importance" as attention weight magnitude
        weights = np.exp(graph.scores - np.max(graph.scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        
        # Average weight per position (across queries)
        position_importance = np.mean(weights, axis=0)
        
        # Select top positions by importance
        n_booms = max(1, int(graph.seq_len * self.boom_ratio))
        boom_positions = np.argsort(-position_importance)[:n_booms].tolist()
        
        return sorted(boom_positions)
    
    def detect_booms_phi_spectral(self, graph: AttentionGraph) -> List[int]:
        """
        Detect boom positions using φ-spectral analysis.
        
        Use SVD of attention scores to find resonant modes.
        """
        # SVD of scores
        U, S, Vt = np.linalg.svd(graph.scores, full_matrices=False)
        
        # φ-Zipf: S[i] ∝ 1/i^(1/φ)
        # Find positions that contribute most to top singular vectors
        
        # Top-k singular vectors (where k captures 90% variance)
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        k = np.searchsorted(cumvar, 0.9) + 1
        k = min(k, len(S))
        
        # Position importance from top-k right singular vectors
        Vt_top = Vt[:k, :]
        position_importance = np.sum(Vt_top**2, axis=0)
        
        # Select top positions
        n_booms = max(1, int(graph.seq_len * self.boom_ratio))
        boom_positions = np.argsort(-position_importance)[:n_booms].tolist()
        
        return sorted(boom_positions)
    
    def detect_booms_phi_greedy(self, graph: AttentionGraph) -> List[int]:
        """
        Detect boom positions using φ-biased greedy search.
        
        Like resfrac's TSP solver, use golden ratio to guide selection.
        """
        n_booms = max(1, int(graph.seq_len * self.boom_ratio))
        
        # Start with position 0 (always a boom)
        booms = [0]
        remaining = set(range(1, graph.seq_len))
        
        # Greedy selection with φ-bias
        while len(booms) < n_booms and remaining:
            last_boom = booms[-1]
            
            # Score each remaining position
            scores = []
            for pos in remaining:
                # Attention score contribution
                attn_score = np.mean(np.abs(graph.scores[:, pos]))
                
                # Distance from last boom (prefer φ-spaced positions)
                dist = abs(pos - last_boom)
                phi_dist = dist / PHI  # Normalize by φ
                
                # Combined score: high attention + φ-spacing
                combined = attn_score * (1 + 0.1 * np.sin(phi_dist * np.pi))
                scores.append((pos, combined))
            
            # Select best
            best_pos = max(scores, key=lambda x: x[1])[0]
            booms.append(best_pos)
            remaining.remove(best_pos)
        
        return sorted(booms)
    
    def solve(self, graph: AttentionGraph, method: str = 'entropy') -> Tuple[np.ndarray, List[int], float]:
        """
        Solve attention using resonant optimization.
        
        Returns:
            output: Attention output (n_queries, head_dim)
            boom_positions: Selected positions
            correlation: Correlation with full attention
        """
        # Detect boom positions
        if method == 'entropy':
            booms = self.detect_booms_entropy(graph)
        elif method == 'phi_spectral':
            booms = self.detect_booms_phi_spectral(graph)
        elif method == 'phi_greedy':
            booms = self.detect_booms_phi_greedy(graph)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute sparse attention
        output = graph.sparse_attention(booms)
        
        # Compute correlation with full attention
        full_output = graph.full_attention
        correlation = np.corrcoef(output.flatten(), full_output.flatten())[0, 1]
        
        return output, booms, correlation


def test_resonant_attention():
    """Test resonant attention on synthetic data."""
    print("=" * 60)
    print("Testing Resonant Attention")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test different sequence lengths
    for seq_len in [16, 64, 256]:
        print(f"\n--- Sequence length: {seq_len} ---")
        
        head_dim = 128
        
        # Random Q, K, V
        Q = np.random.randn(1, head_dim)  # Single query
        K = np.random.randn(seq_len, head_dim)
        V = np.random.randn(seq_len, head_dim)
        
        graph = AttentionGraph(Q, K, V)
        solver = ResonantAttentionSolver(boom_ratio=0.2)
        
        # Test different methods
        for method in ['entropy', 'phi_spectral', 'phi_greedy']:
            start = time.time()
            output, booms, corr = solver.solve(graph, method=method)
            elapsed = time.time() - start
            
            print(f"  {method:15s}: corr={corr:.4f}, booms={len(booms)}/{seq_len}, time={elapsed*1000:.1f}ms")


def test_with_qwen2_attention():
    """Test resonant attention with actual Qwen2 attention patterns."""
    print("\n" + "=" * 60)
    print("Testing with Qwen2 Attention")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Transformers not available, skipping Qwen2 test")
        return
    
    print("\nLoading Qwen2...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    
    # Test prompt
    prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    seq_len = input_ids.shape[1]
    
    print(f"\nPrompt: {prompt!r}")
    print(f"Sequence length: {seq_len}")
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[14][0].numpy()  # Layer 14, middle of model
    
    # Extract Q, K, V for head 0
    layer = model.model.layers[14]
    W_q = layer.self_attn.q_proj.weight.data.numpy()[:128, :]  # Head 0
    W_k = layer.self_attn.k_proj.weight.data.numpy()[:128, :]  # KV head 0
    W_v = layer.self_attn.v_proj.weight.data.numpy()[:128, :]
    b_q = layer.self_attn.q_proj.bias.data.numpy()[:128]
    b_k = layer.self_attn.k_proj.bias.data.numpy()[:128]
    b_v = layer.self_attn.v_proj.bias.data.numpy()[:128]
    
    # Compute Q, K, V
    Q = hidden @ W_q.T + b_q  # (seq_len, head_dim)
    K = hidden @ W_k.T + b_k
    V = hidden @ W_v.T + b_v
    
    # Use last position as query
    Q_last = Q[-1:]  # (1, head_dim)
    
    # Create attention graph
    graph = AttentionGraph(Q_last, K, V)
    
    print(f"\nFull attention output norm: {np.linalg.norm(graph.full_attention):.4f}")
    
    # Test resonant attention
    solver = ResonantAttentionSolver(boom_ratio=0.2)
    
    print("\nResonant attention results:")
    for method in ['entropy', 'phi_spectral', 'phi_greedy']:
        output, booms, corr = solver.solve(graph, method=method)
        print(f"  {method:15s}: corr={corr:.4f}, booms={len(booms)}/{seq_len}")
        print(f"    Boom positions: {booms}")
    
    # Analyze attention structure
    print("\n--- Attention Structure Analysis ---")
    
    # SVD of attention scores
    U, S, Vt = np.linalg.svd(graph.scores, full_matrices=False)
    
    # Check for φ-Zipf
    if len(S) > 1:
        # Fit power law: S[i] ∝ 1/i^α
        indices = np.arange(1, len(S) + 1)
        log_indices = np.log(indices)
        log_S = np.log(S + 1e-10)
        
        # Linear regression
        alpha = -np.polyfit(log_indices, log_S, 1)[0]
        
        print(f"Singular value decay exponent α: {alpha:.4f}")
        print(f"Target (1/φ): {INV_PHI:.4f}")
        print(f"Deviation: {abs(alpha - INV_PHI):.4f}")
        
        # Variance captured by top-k
        total_var = np.sum(S**2)
        for k in [1, 2, 4, 8]:
            if k <= len(S):
                var_k = np.sum(S[:k]**2) / total_var * 100
                print(f"Top-{k} singular values capture: {var_k:.1f}% variance")
    
    del model


def analyze_phi_structure():
    """Analyze the φ-structure of attention matrices."""
    print("\n" + "=" * 60)
    print("Analyzing φ-Structure of Attention")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create attention matrix with known φ-structure
    seq_len = 64
    head_dim = 128
    
    # φ-structured attention: S[i] = S[0] / i^(1/φ)
    k = min(seq_len, head_dim)
    S_phi = np.array([1.0 / (i+1)**(INV_PHI) for i in range(k)])
    
    # Random orthogonal U, V
    U, _ = np.linalg.qr(np.random.randn(seq_len, k))
    V, _ = np.linalg.qr(np.random.randn(head_dim, k))
    
    # Construct attention scores with φ-structure
    scores_phi = U @ np.diag(S_phi) @ V.T
    
    print(f"Created {seq_len}x{head_dim} attention matrix with φ-Zipf structure")
    
    # Verify structure
    _, S_recovered, _ = np.linalg.svd(scores_phi, full_matrices=False)
    
    # Fit power law
    indices = np.arange(1, len(S_recovered) + 1)
    alpha = -np.polyfit(np.log(indices), np.log(S_recovered + 1e-10), 1)[0]
    
    print(f"Recovered α: {alpha:.4f} (target: {INV_PHI:.4f})")
    
    # Test resonant attention on φ-structured matrix
    Q = np.random.randn(1, head_dim)
    K = np.random.randn(seq_len, head_dim)
    V = np.random.randn(seq_len, head_dim)
    
    graph = AttentionGraph(Q, K, V)
    solver = ResonantAttentionSolver(boom_ratio=0.2)
    
    print("\nResonant attention on φ-structured matrix:")
    for method in ['entropy', 'phi_spectral', 'phi_greedy']:
        output, booms, corr = solver.solve(graph, method=method)
        print(f"  {method:15s}: corr={corr:.4f}")


if __name__ == "__main__":
    test_resonant_attention()
    analyze_phi_structure()
    test_with_qwen2_attention()
