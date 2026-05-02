#!/usr/bin/env python3
"""
MLP SVD Compression Analysis

The MLP is the bottleneck (45x slower than attention).
Can we apply the same SVD compression approach?

MLP structure:
  gate = W_gate @ x  (18944, 3584)
  up = W_up @ x      (18944, 3584)
  hidden = silu(gate) * up
  out = W_down @ hidden  (3584, 18944)

The challenge: SiLU is nonlinear, so we can't directly merge matrices.

Options:
1. Linearize SiLU (works when gate ≈ 0.5, but fails in early layers)
2. Compress each matrix separately with SVD
3. Use low-rank approximation for the combined path

Author: TruthSpace LCM Team
Date: 2026-01-29
"""

import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949


def analyze_mlp_structure():
    """Analyze MLP weight structure for compression opportunities."""
    print("Loading Qwen2...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    
    print("\n" + "=" * 60)
    print("MLP Structure Analysis")
    print("=" * 60)
    
    # Analyze a few layers
    for layer_idx in [0, 7, 14, 21, 27]:
        layer = model.model.layers[layer_idx]
        
        W_gate = layer.mlp.gate_proj.weight.data.numpy()
        W_up = layer.mlp.up_proj.weight.data.numpy()
        W_down = layer.mlp.down_proj.weight.data.numpy()
        
        print(f"\nLayer {layer_idx}:")
        print(f"  W_gate: {W_gate.shape}, norm={np.linalg.norm(W_gate):.2f}")
        print(f"  W_up:   {W_up.shape}, norm={np.linalg.norm(W_up):.2f}")
        print(f"  W_down: {W_down.shape}, norm={np.linalg.norm(W_down):.2f}")
        
        # SVD analysis
        for name, W in [("gate", W_gate), ("up", W_up), ("down", W_down)]:
            _, S, _ = np.linalg.svd(W, full_matrices=False)
            
            # Find k for 90%, 95%, 99% variance
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var
            k90 = np.searchsorted(cumvar, 0.90) + 1
            k95 = np.searchsorted(cumvar, 0.95) + 1
            k99 = np.searchsorted(cumvar, 0.99) + 1
            
            print(f"    {name}: k90={k90}, k95={k95}, k99={k99} (max={len(S)})")
    
    del model


def test_mlp_svd_compression():
    """Test SVD compression on MLP."""
    print("\n" + "=" * 60)
    print("Testing MLP SVD Compression")
    print("=" * 60)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    
    # Get a test input
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        # Get hidden state before MLP in layer 14
        hidden = outputs.hidden_states[14][0, -1, :].numpy()
    
    layer = model.model.layers[14]
    W_gate = layer.mlp.gate_proj.weight.data.numpy()
    W_up = layer.mlp.up_proj.weight.data.numpy()
    W_down = layer.mlp.down_proj.weight.data.numpy()
    ln_weight = layer.post_attention_layernorm.weight.data.numpy()
    
    # Apply layer norm
    rms = np.sqrt(np.mean(hidden ** 2) + 1e-6)
    x_norm = (hidden / rms) * ln_weight
    
    # Exact MLP
    gate = W_gate @ x_norm
    up = W_up @ x_norm
    silu_gate = gate / (1 + np.exp(-gate))
    hidden_exact = silu_gate * up
    out_exact = W_down @ hidden_exact
    
    print(f"\nExact MLP output norm: {np.linalg.norm(out_exact):.4f}")
    
    # Test different k values for compression
    print("\nCompression results:")
    print(f"{'k':>6} {'Correlation':>12} {'Rel Error':>12} {'Speedup':>10}")
    print("-" * 44)
    
    for k in [64, 128, 256, 512, 1024, 2048]:
        # Compress each matrix
        U_gate, S_gate, Vt_gate = np.linalg.svd(W_gate, full_matrices=False)
        U_up, S_up, Vt_up = np.linalg.svd(W_up, full_matrices=False)
        U_down, S_down, Vt_down = np.linalg.svd(W_down, full_matrices=False)
        
        # Truncate
        W_gate_k = U_gate[:, :k] @ np.diag(S_gate[:k]) @ Vt_gate[:k, :]
        W_up_k = U_up[:, :k] @ np.diag(S_up[:k]) @ Vt_up[:k, :]
        W_down_k = U_down[:, :k] @ np.diag(S_down[:k]) @ Vt_down[:k, :]
        
        # Compressed MLP
        gate_k = W_gate_k @ x_norm
        up_k = W_up_k @ x_norm
        silu_gate_k = gate_k / (1 + np.exp(-gate_k))
        hidden_k = silu_gate_k * up_k
        out_k = W_down_k @ hidden_k
        
        # Metrics
        corr = np.corrcoef(out_exact.flatten(), out_k.flatten())[0, 1]
        rel_err = np.linalg.norm(out_exact - out_k) / np.linalg.norm(out_exact)
        
        # Theoretical speedup (ignoring overhead)
        # Original: 3 * (18944 * 3584) = 203M ops
        # Compressed: 3 * (k * 3584 + k * 18944 + k) ≈ 3 * k * 22528
        original_ops = 3 * 18944 * 3584
        compressed_ops = 3 * k * (3584 + 18944)
        speedup = original_ops / compressed_ops
        
        print(f"{k:>6} {corr:>12.6f} {rel_err:>12.6f} {speedup:>10.2f}x")
    
    del model


def test_mlp_factored_computation():
    """Test factored MLP computation for speed."""
    print("\n" + "=" * 60)
    print("Testing Factored MLP Computation")
    print("=" * 60)
    
    # Simulate MLP dimensions
    hidden_dim = 3584
    intermediate_dim = 18944
    k = 512  # Compression rank
    
    # Random weights (simulating compressed)
    np.random.seed(42)
    
    # Original matrices
    W_gate = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32)
    W_up = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32)
    W_down = np.random.randn(hidden_dim, intermediate_dim).astype(np.float32)
    
    # Factored matrices (U @ S @ Vt form, but stored as U @ (S @ Vt) for efficiency)
    U_gate = np.random.randn(intermediate_dim, k).astype(np.float32)
    SVt_gate = np.random.randn(k, hidden_dim).astype(np.float32)
    U_up = np.random.randn(intermediate_dim, k).astype(np.float32)
    SVt_up = np.random.randn(k, hidden_dim).astype(np.float32)
    U_down = np.random.randn(hidden_dim, k).astype(np.float32)
    SVt_down = np.random.randn(k, intermediate_dim).astype(np.float32)
    
    x = np.random.randn(hidden_dim).astype(np.float32)
    
    # Benchmark original
    n_iters = 10
    
    start = time.time()
    for _ in range(n_iters):
        gate = W_gate @ x
        up = W_up @ x
        silu = gate / (1 + np.exp(-gate))
        hidden = silu * up
        out = W_down @ hidden
    original_time = (time.time() - start) / n_iters
    
    # Benchmark factored
    start = time.time()
    for _ in range(n_iters):
        # Factored: U @ (SVt @ x)
        gate = U_gate @ (SVt_gate @ x)
        up = U_up @ (SVt_up @ x)
        silu = gate / (1 + np.exp(-gate))
        hidden = silu * up
        # For down, we need to be clever: out = U_down @ (SVt_down @ hidden)
        out = U_down @ (SVt_down @ hidden)
    factored_time = (time.time() - start) / n_iters
    
    print(f"\nOriginal MLP:  {original_time*1000:.1f}ms")
    print(f"Factored MLP (k={k}): {factored_time*1000:.1f}ms")
    print(f"Speedup: {original_time/factored_time:.2f}x")
    
    # The issue: factored down projection still needs intermediate_dim operations
    # Better approach: keep down as-is, only compress gate and up
    
    start = time.time()
    for _ in range(n_iters):
        gate = U_gate @ (SVt_gate @ x)
        up = U_up @ (SVt_up @ x)
        silu = gate / (1 + np.exp(-gate))
        hidden = silu * up
        out = W_down @ hidden  # Keep original
    hybrid_time = (time.time() - start) / n_iters
    
    print(f"Hybrid (compress gate/up only): {hybrid_time*1000:.1f}ms")
    print(f"Speedup: {original_time/hybrid_time:.2f}x")


if __name__ == "__main__":
    analyze_mlp_structure()
    test_mlp_svd_compression()
    test_mlp_factored_computation()
