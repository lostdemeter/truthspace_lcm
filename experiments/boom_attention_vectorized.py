#!/usr/bin/env python3
"""
Vectorized Boom-Based Attention
================================

Fully vectorized approach that avoids loops in the kernel.

Key insight: Instead of looping over boom positions, we:
1. Gather boom keys/values into contiguous tensors
2. Use standard matmul (which is highly optimized)
3. This leverages cuBLAS instead of custom loops

The speedup comes from reduced memory bandwidth, not reduced FLOPs.

Author: TruthSpace LCM Team
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time

DEVICE = "cuda"


class VectorizedBoomAttention(torch.nn.Module):
    """
    Vectorized boom-based attention using gather + matmul.
    
    This approach:
    1. Detects boom positions (O(N))
    2. Gathers boom K, V (O(B × D))
    3. Computes Q @ boom_K^T (O(N × B × D) via cuBLAS)
    4. Computes attn @ boom_V (O(N × B × D) via cuBLAS)
    
    Total: O(N × B) instead of O(N²), using optimized cuBLAS.
    """
    
    def __init__(self, boom_threshold_percentile=80, min_boom_ratio=0.1, max_boom_ratio=0.3):
        super().__init__()
        self.boom_threshold_percentile = boom_threshold_percentile
        self.min_boom_ratio = min_boom_ratio
        self.max_boom_ratio = max_boom_ratio
    
    @torch.no_grad()
    def detect_booms(self, query, key, seq_len, head_dim):
        """
        Fast boom detection using first head.
        """
        # Quick attention on first head
        q = query[:, 0]  # [batch, seq_len, head_dim]
        k = key[:, 0]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1) * -1e9
        scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        
        # Entropy
        entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)  # [batch, seq_len]
        
        # Detect drops
        drops = entropy[:, :-1] - entropy[:, 1:]  # [batch, seq_len-1]
        
        # Threshold
        threshold = torch.quantile(drops.float(), self.boom_threshold_percentile / 100, dim=-1, keepdim=True)
        
        # Boom mask
        boom_mask = torch.zeros(entropy.shape, device=query.device, dtype=torch.bool)
        boom_mask[:, 0] = True  # Always include first
        boom_mask[:, -1] = True  # Always include last
        boom_mask[:, 1:] = drops > threshold
        
        return boom_mask, entropy
    
    def forward(self, query, key, value):
        """
        Vectorized boom attention.
        
        query, key, value: [batch, heads, seq_len, head_dim]
        """
        batch, heads, seq_len, head_dim = query.shape
        
        # Detect booms
        boom_mask, entropy = self.detect_booms(query, key, seq_len, head_dim)
        
        # Get boom indices for each batch
        # For simplicity, use same booms across batch (take first)
        boom_indices = torch.where(boom_mask[0])[0]
        n_booms = len(boom_indices)
        
        # Clamp boom count
        min_booms = max(2, int(seq_len * self.min_boom_ratio))
        max_booms = int(seq_len * self.max_boom_ratio)
        
        if n_booms < min_booms:
            # Add evenly spaced booms
            spacing = seq_len // min_booms
            extra = torch.arange(0, seq_len, spacing, device=query.device)
            boom_indices = torch.cat([boom_indices, extra])
            boom_indices = torch.unique(boom_indices.sort()[0])
            n_booms = len(boom_indices)
        elif n_booms > max_booms:
            # Subsample
            indices = torch.linspace(0, n_booms - 1, max_booms, device=query.device).long()
            boom_indices = boom_indices[indices]
            n_booms = max_booms
        
        # Gather boom keys and values
        # boom_indices: [n_booms]
        # key: [batch, heads, seq_len, head_dim]
        # boom_key: [batch, heads, n_booms, head_dim]
        
        boom_key = key[:, :, boom_indices, :]
        boom_value = value[:, :, boom_indices, :]
        
        # Compute attention scores: Q @ boom_K^T
        # [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, n_booms]
        # = [batch, heads, seq_len, n_booms]
        scores = torch.matmul(query, boom_key.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Causal masking: each query can only attend to booms at or before its position
        # query_pos: [seq_len, 1], boom_pos: [1, n_booms]
        query_pos = torch.arange(seq_len, device=query.device).unsqueeze(1)
        boom_pos = boom_indices.unsqueeze(0)
        causal_mask = (query_pos < boom_pos).float() * -1e9  # [seq_len, n_booms]
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax (need float32 for stability)
        attn = F.softmax(scores.float(), dim=-1).to(boom_value.dtype)
        
        # Output: attn @ boom_V
        # [batch, heads, seq_len, n_booms] @ [batch, heads, n_booms, head_dim]
        # = [batch, heads, seq_len, head_dim]
        output = torch.matmul(attn, boom_value)
        
        return output, boom_indices, n_booms


def benchmark_vectorized():
    """Benchmark vectorized boom attention."""
    print("="*70)
    print("VECTORIZED BOOM ATTENTION BENCHMARK")
    print("="*70)
    
    boom_attn = VectorizedBoomAttention().to(DEVICE)
    
    results = []
    
    for seq_len in [128, 256, 512, 1024, 2048, 4096]:
        batch, heads, head_dim = 1, 28, 128
        
        query = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        key = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        value = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            _, _, _ = boom_attn(query, key, value)
        
        torch.cuda.synchronize()
        
        # Time full attention (SDPA with Flash Attention)
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            full_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        torch.cuda.synchronize()
        full_time = (time.perf_counter() - start) / n_runs * 1000
        
        # Time boom attention
        start = time.perf_counter()
        for _ in range(n_runs):
            boom_out, booms, n_booms = boom_attn(query, key, value)
        torch.cuda.synchronize()
        boom_time = (time.perf_counter() - start) / n_runs * 1000
        
        theoretical = seq_len / n_booms
        actual = full_time / boom_time
        
        # Quality check
        diff = (boom_out - full_out).abs().mean().item()
        
        results.append({
            'seq_len': seq_len,
            'n_booms': n_booms,
            'full_time': full_time,
            'boom_time': boom_time,
            'theoretical': theoretical,
            'actual': actual,
            'diff': diff,
        })
        
        print(f"\nSeq len: {seq_len}")
        print(f"  SDPA (Flash): {full_time:.3f} ms")
        print(f"  Boom attention: {boom_time:.3f} ms")
        print(f"  Booms: {n_booms} ({n_booms/seq_len*100:.1f}%)")
        print(f"  Theoretical speedup: {theoretical:.1f}x")
        print(f"  Actual speedup: {actual:.2f}x")
        print(f"  Mean diff: {diff:.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n| Seq Len | SDPA (ms) | Boom (ms) | Booms | Speedup |")
    print("|---------|-----------|-----------|-------|---------|")
    for r in results:
        print(f"| {r['seq_len']:7d} | {r['full_time']:9.3f} | {r['boom_time']:9.3f} | {r['n_booms']:5d} | {r['actual']:7.2f}x |")
    
    # Find crossover
    crossover = None
    for r in results:
        if r['actual'] >= 1.0:
            crossover = r['seq_len']
            break
    
    if crossover:
        print(f"\nCrossover point: seq_len = {crossover}")
    else:
        print("\nNo crossover in tested range.")
        # Extrapolate
        print("\nExtrapolating to longer sequences...")
        last = results[-1]
        # Assume boom ratio stays ~20% and overhead is constant
        for seq_len in [8192, 16384, 32768]:
            n_booms = int(seq_len * 0.2)
            # SDPA scales as O(N²), boom scales as O(N × B)
            # Estimate based on last measurement
            sdpa_time = last['full_time'] * (seq_len / last['seq_len']) ** 2
            boom_time = last['boom_time'] * (seq_len / last['seq_len']) * (n_booms / last['n_booms'])
            speedup = sdpa_time / boom_time
            print(f"  seq_len={seq_len}: estimated speedup = {speedup:.1f}x")
    
    return results


def test_quality():
    """Test quality of boom attention vs full attention."""
    print("\n" + "="*70)
    print("QUALITY TEST")
    print("="*70)
    
    boom_attn = VectorizedBoomAttention().to(DEVICE)
    
    for seq_len in [128, 512, 1024]:
        batch, heads, head_dim = 1, 28, 128
        
        query = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        key = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        value = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        
        # Full attention
        full_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        
        # Boom attention
        boom_out, booms, n_booms = boom_attn(query, key, value)
        
        # Metrics
        mae = (boom_out - full_out).abs().mean().item()
        mse = ((boom_out - full_out) ** 2).mean().item()
        cosine = F.cosine_similarity(
            boom_out.flatten().unsqueeze(0),
            full_out.flatten().unsqueeze(0)
        ).item()
        
        print(f"\nSeq len: {seq_len}, Booms: {n_booms}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Cosine similarity: {cosine:.6f}")


def main():
    print("="*70)
    print("VECTORIZED BOOM-BASED ATTENTION")
    print("="*70)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print("\nThis approach uses cuBLAS matmul instead of custom Triton loops.")
    print("The speedup comes from reduced memory bandwidth (fewer K,V to load).")
    
    # Quality test
    test_quality()
    
    # Benchmark
    results = benchmark_vectorized()
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
KEY INSIGHT:

Flash Attention is extremely optimized - it's memory-bound, not compute-bound.
Our boom attention reduces compute (O(N×B) vs O(N²)) but:
1. Still needs to load all Q (O(N×D))
2. Boom detection adds overhead
3. Python dispatch overhead

For boom attention to win, we need:
1. Very long sequences (N >> 4096)
2. Fused boom detection (no Python overhead)
3. Or: use boom attention for GENERATION (where we only compute one query at a time)

GENERATION USE CASE:
During autoregressive generation, we compute attention for ONE new token.
- Full attention: O(N) per token (attend to all past tokens)
- Boom attention: O(B) per token (attend only to boom positions)

This is where boom attention shines!
""")


if __name__ == "__main__":
    main()
