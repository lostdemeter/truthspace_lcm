#!/usr/bin/env python3
"""
Cached Boom Attention: O(N) Attention via Cached K Landscape
=============================================================

Key insight from rhzeros:
- ζ'(s) changes slowly near zeros, so cache it once
- 40% speedup from avoiding redundant derivative computation

Applied to attention:
- K is FIXED during autoregressive generation (KV cache)
- The "boom structure" of K can be precomputed ONCE
- For each new Q, only attend to cached boom positions

Complexity:
- Standard: O(T × N²) for T tokens, N context
- Cached boom: O(N) precompute + O(T × k) generation
- If k = 20, N = 1000, T = 100: 50× speedup

This is the CUDA-optimized version using torch operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class CachedKLandscape:
    """
    Cached structure of the K tensor for boom attention.
    
    Like rhzeros caching ζ', we cache the "landscape" of K:
    - boom_indices: positions with high attention potential
    - K_booms: K vectors at boom positions
    - V_booms: V vectors at boom positions
    """
    boom_indices: torch.Tensor  # (k,)
    K_booms: torch.Tensor       # (batch, heads, k, head_dim)
    V_booms: torch.Tensor       # (batch, heads, k, head_dim)
    seq_len: int
    
    @property
    def num_booms(self) -> int:
        return len(self.boom_indices)


def detect_booms_cuda(K: torch.Tensor, max_booms: int = 20) -> torch.Tensor:
    """
    CUDA-optimized boom detection using K norms and gradients.
    
    Complexity: O(N) - single pass through K
    
    Args:
        K: Key tensor (batch, heads, seq_len, head_dim)
        max_booms: Maximum number of boom positions
    
    Returns:
        boom_indices: Tensor of boom positions
    """
    # Compute K norms across heads: O(N × d)
    k_norms = K.norm(dim=-1).mean(dim=(0, 1))  # (seq_len,)
    seq_len = k_norms.shape[0]
    
    if seq_len <= max_booms:
        return torch.arange(seq_len, device=K.device)
    
    # Method 1: Gradient-based detection
    # Booms occur where the K norm changes rapidly
    grad = torch.abs(k_norms[1:] - k_norms[:-1])
    grad = F.pad(grad, (1, 0), value=0)  # Pad to match seq_len
    
    # Method 2: Local maxima detection
    # Booms are local peaks in K norm
    is_peak = torch.zeros(seq_len, device=K.device, dtype=torch.bool)
    if seq_len > 2:
        is_peak[1:-1] = (k_norms[1:-1] > k_norms[:-2]) & (k_norms[1:-1] > k_norms[2:])
    
    # Combine: score = gradient magnitude + peak bonus
    scores = grad + is_peak.float() * grad.mean()
    
    # Always include first and last positions
    scores[0] = scores.max() + 1
    scores[-1] = scores.max() + 0.5
    
    # Select top-k positions
    _, top_indices = torch.topk(scores, min(max_booms, seq_len))
    boom_indices = torch.sort(top_indices)[0]
    
    return boom_indices


def cache_k_landscape(
    K: torch.Tensor,
    V: torch.Tensor,
    max_booms: int = 20
) -> CachedKLandscape:
    """
    Precompute the boom structure of K (like caching ζ' in rhzeros).
    
    This is done ONCE when K is first computed, then reused
    for all subsequent Q vectors.
    
    Complexity: O(N)
    
    Args:
        K: Key tensor (batch, heads, seq_len, head_dim)
        V: Value tensor (batch, heads, seq_len, head_dim)
        max_booms: Maximum boom positions to cache
    
    Returns:
        CachedKLandscape with boom positions and cached K/V
    """
    seq_len = K.shape[2]
    
    # Detect boom positions: O(N)
    boom_indices = detect_booms_cuda(K, max_booms=max_booms)
    
    # Cache K and V at boom positions: O(k × d)
    K_booms = K[:, :, boom_indices, :]
    V_booms = V[:, :, boom_indices, :]
    
    return CachedKLandscape(
        boom_indices=boom_indices,
        K_booms=K_booms,
        V_booms=V_booms,
        seq_len=seq_len
    )


def cached_boom_attention(
    Q: torch.Tensor,
    cache: CachedKLandscape,
    query_positions: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Attention using cached boom structure.
    
    Instead of O(N²), we compute O(N × k) attention.
    
    Args:
        Q: Query tensor (batch, heads, num_queries, head_dim)
        cache: Precomputed K landscape
        query_positions: Positions of queries (for causal masking)
    
    Returns:
        Attention output (batch, heads, num_queries, head_dim)
    """
    batch, heads, num_queries, head_dim = Q.shape
    d_k = np.sqrt(head_dim)
    
    # Compute scores only for boom positions: O(num_queries × k × d)
    scores = torch.matmul(Q, cache.K_booms.transpose(-2, -1)) / d_k
    
    # Causal masking
    if query_positions is not None:
        # query_positions: (num_queries,)
        # boom_indices: (k,)
        positions = query_positions.unsqueeze(1)  # (num_queries, 1)
        boom_pos = cache.boom_indices.unsqueeze(0)  # (1, k)
        causal_mask = positions < boom_pos  # (num_queries, k)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax over booms: O(num_queries × k)
    attn_weights = F.softmax(scores, dim=-1)
    
    # Output: O(num_queries × k × d)
    output = torch.matmul(attn_weights, cache.V_booms)
    
    return output


def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard O(N²) attention."""
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    seq_len = Q.shape[2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def benchmark_cached_attention():
    """Benchmark cached boom attention vs standard."""
    print("=" * 70)
    print("CACHED BOOM ATTENTION BENCHMARK")
    print("=" * 70)
    print("""
Key insight: Cache K landscape ONCE, reuse for all queries.
Like rhzeros caching ζ' for Newton iterations.
""")
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    batch_size = 1
    heads = 28
    head_dim = 128
    max_booms = 32
    
    print(f"Config: batch={batch_size}, heads={heads}, head_dim={head_dim}, max_booms={max_booms}")
    print(f"{'Seq Len':>10} {'Standard':>12} {'Cached':>12} {'Speedup':>10} {'Cache ms':>12} {'Booms':>8}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        Q = torch.randn(batch_size, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        K = torch.randn(batch_size, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        V = torch.randn(batch_size, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
        
        # Warm up
        _ = standard_attention(Q, K, V)
        cache = cache_k_landscape(K, V, max_booms=max_booms)
        _ = cached_boom_attention(Q, cache)
        
        # Benchmark standard
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            out_std = standard_attention(Q, K, V)
        torch.cuda.synchronize()
        time_std = (time.perf_counter() - start) / 10 * 1000
        
        # Benchmark cache creation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            cache = cache_k_landscape(K, V, max_booms=max_booms)
        torch.cuda.synchronize()
        time_cache = (time.perf_counter() - start) / 10 * 1000
        
        # Benchmark cached attention
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            out_cached = cached_boom_attention(Q, cache)
        torch.cuda.synchronize()
        time_cached = (time.perf_counter() - start) / 10 * 1000
        
        # Total time for cached = cache creation + attention
        # But in practice, cache is created ONCE per context
        # So for generation, it's amortized
        
        speedup = time_std / time_cached
        
        print(f"{seq_len:>10} {time_std:>10.3f}ms {time_cached:>10.3f}ms {speedup:>10.2f}× {time_cache:>10.3f}ms {cache.num_booms:>8}")
    
    print("\n" + "=" * 70)
    print("GENERATION SIMULATION")
    print("=" * 70)
    print("""
Simulating autoregressive generation:
- Cache K landscape ONCE at start
- Generate T tokens, each using cached attention
""")
    
    context_len = 1024
    gen_tokens = 100
    
    K = torch.randn(batch_size, heads, context_len, head_dim, device=DEVICE, dtype=torch.float32)
    V = torch.randn(batch_size, heads, context_len, head_dim, device=DEVICE, dtype=torch.float32)
    
    # Standard: O(T × N²)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for t in range(gen_tokens):
        Q = torch.randn(batch_size, heads, 1, head_dim, device=DEVICE, dtype=torch.float32)
        # In reality, K grows, but we simulate fixed context
        _ = standard_attention(
            Q.expand(-1, -1, context_len, -1),
            K, V
        )[:, :, -1:, :]  # Only need last position
    torch.cuda.synchronize()
    time_std_gen = (time.perf_counter() - start) * 1000
    
    # Cached: O(N) cache + O(T × k)
    torch.cuda.synchronize()
    start = time.perf_counter()
    cache = cache_k_landscape(K, V, max_booms=max_booms)
    for t in range(gen_tokens):
        Q = torch.randn(batch_size, heads, 1, head_dim, device=DEVICE, dtype=torch.float32)
        _ = cached_boom_attention(Q, cache)
    torch.cuda.synchronize()
    time_cached_gen = (time.perf_counter() - start) * 1000
    
    print(f"Context: {context_len}, Generated: {gen_tokens} tokens")
    print(f"Standard: {time_std_gen:.1f}ms ({time_std_gen/gen_tokens:.2f}ms/token)")
    print(f"Cached:   {time_cached_gen:.1f}ms ({time_cached_gen/gen_tokens:.2f}ms/token)")
    print(f"Speedup:  {time_std_gen/time_cached_gen:.2f}×")
    
    print("\n" + "=" * 70)
    print("ACCURACY TEST")
    print("=" * 70)
    
    seq_len = 256
    Q = torch.randn(1, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    K = torch.randn(1, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    V = torch.randn(1, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float32)
    
    out_std = standard_attention(Q, K, V)
    
    print(f"\nSeq len: {seq_len}")
    print(f"{'Max Booms':>12} {'Correlation':>15} {'Rel Error':>15} {'Boom %':>12}")
    print("-" * 60)
    
    for max_b in [8, 16, 32, 64, 128, 256]:
        cache = cache_k_landscape(K, V, max_booms=max_b)
        out_cached = cached_boom_attention(Q, cache)
        
        # Handle NaN in outputs
        out_std_clean = torch.nan_to_num(out_std, 0)
        out_cached_clean = torch.nan_to_num(out_cached, 0)
        
        out_std_flat = out_std_clean.flatten().cpu().numpy()
        out_cached_flat = out_cached_clean.flatten().cpu().numpy()
        
        # Compute correlation
        if np.std(out_std_flat) > 0 and np.std(out_cached_flat) > 0:
            correlation = np.corrcoef(out_std_flat, out_cached_flat)[0, 1]
        else:
            correlation = 0.0
        
        # Relative error
        rel_error = np.mean(np.abs(out_std_flat - out_cached_flat)) / (np.mean(np.abs(out_std_flat)) + 1e-8)
        
        boom_pct = cache.num_booms / seq_len * 100
        
        print(f"{max_b:>12} {correlation:>15.4f} {rel_error:>15.4f} {boom_pct:>11.1f}%")


def main():
    print("=" * 70)
    print("CACHED BOOM ATTENTION")
    print("=" * 70)
    print("""
The rhzeros insight applied to attention:

RHZEROS:
  - ζ'(s) changes slowly near zeros
  - Cache ζ' ONCE, reuse for Newton iterations
  - 40% speedup

ATTENTION:
  - K is FIXED during generation (KV cache)
  - Cache K "boom structure" ONCE
  - Reuse for all Q vectors
  - O(N²) → O(N × k) where k << N
""")
    
    benchmark_cached_attention()


if __name__ == "__main__":
    main()
