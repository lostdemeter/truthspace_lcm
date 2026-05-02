#!/usr/bin/env python3
"""
Hybrid Spigot Attention: φ-Lattice + Local Window
==================================================

The pure spigot has low quality (~0.5 cosine) because it misses local context.
The hybrid approach combines:
1. φ-lattice booms (distant semantic anchors)
2. Local window (nearby tokens for context)

This is analogous to:
- BBP for distant digits (φ-lattice)
- Direct computation for nearby digits (local window)

Author: TruthSpace LCM Team
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = (1 + np.sqrt(5)) / 2


def hybrid_boom_positions(seq_len, local_window=8, target_ratio=0.2):
    """
    Generate hybrid boom positions: φ-lattice + local window.
    
    For each query position i:
    - Include positions [max(0, i-local_window), i] (local context)
    - Include φ-lattice positions before i (distant anchors)
    
    This gives O(local_window + n_booms) positions per query.
    """
    # φ-lattice positions
    target_spacing = int(1 / target_ratio)
    level = max(1, int(np.log(target_spacing) / np.log(PHI)))
    spacing = max(2, int(PHI ** level))
    
    lattice = set([0])
    for pos in range(0, seq_len, spacing):
        lattice.add(pos)
    lattice.add(seq_len - 1)
    
    return sorted(lattice), local_window, spacing


def hybrid_spigot_attention(query, key, value, lattice_positions, local_window, head_dim):
    """
    Hybrid attention: local window + φ-lattice.
    
    For each query position i:
    - Attend to local window [i-local_window, i]
    - Attend to lattice positions before i
    
    This captures both local context and distant anchors.
    """
    batch, heads, seq_len, _ = query.shape
    
    # Build attention mask for each position
    # For efficiency, we'll compute a sparse attention pattern
    
    outputs = []
    
    for i in range(seq_len):
        # Positions to attend to for query i
        # 1. Local window
        local_start = max(0, i - local_window + 1)
        local_positions = list(range(local_start, i + 1))
        
        # 2. Lattice positions before local window
        lattice_before = [p for p in lattice_positions if p < local_start]
        
        # Combine (no duplicates)
        attend_positions = sorted(set(lattice_before + local_positions))
        
        if len(attend_positions) == 0:
            attend_positions = [0]
        
        # Get Q, K, V for this query
        q_i = query[:, :, i:i+1, :]  # [batch, heads, 1, head_dim]
        k_attend = key[:, :, attend_positions, :]  # [batch, heads, n_attend, head_dim]
        v_attend = value[:, :, attend_positions, :]
        
        # Compute attention
        scores = torch.matmul(q_i, k_attend.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores.float(), dim=-1).to(value.dtype)
        out_i = torch.matmul(attn, v_attend)
        
        outputs.append(out_i)
    
    return torch.cat(outputs, dim=2)


def vectorized_hybrid_attention(query, key, value, lattice_positions, local_window, head_dim):
    """
    Vectorized hybrid attention for better performance.
    
    Instead of looping over positions, we:
    1. Compute full attention scores
    2. Mask out non-hybrid positions
    3. Apply softmax and compute output
    
    This is still O(N²) in memory but faster than the loop.
    """
    batch, heads, seq_len, _ = query.shape
    
    # Full attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Build hybrid mask
    # mask[i, j] = 1 if position j should be attended from position i
    mask = torch.zeros(seq_len, seq_len, device=query.device)
    
    # Local window: attend to [i-local_window+1, i]
    for i in range(seq_len):
        local_start = max(0, i - local_window + 1)
        mask[i, local_start:i+1] = 1
    
    # Lattice positions: attend to all lattice positions before local window
    for i in range(seq_len):
        local_start = max(0, i - local_window + 1)
        for p in lattice_positions:
            if p < local_start:
                mask[i, p] = 1
    
    # Apply mask (set non-attended positions to -inf)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax and output
    attn = F.softmax(scores.float(), dim=-1).to(value.dtype)
    output = torch.matmul(attn, value)
    
    return output, mask.squeeze()


def full_attention(query, key, value, head_dim):
    """Standard causal attention for comparison."""
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    
    seq_len = query.shape[2]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1) * float('-inf')
    scores = scores + causal_mask
    
    attn = F.softmax(scores.float(), dim=-1).to(value.dtype)
    output = torch.matmul(attn, value)
    
    return output


def benchmark_hybrid():
    """Benchmark hybrid spigot attention."""
    print("="*70)
    print("HYBRID SPIGOT ATTENTION BENCHMARK")
    print("="*70)
    print("\nHybrid = φ-lattice (distant anchors) + local window (nearby context)")
    
    results = []
    
    for seq_len in [128, 256, 512, 1024, 2048]:
        batch, heads, head_dim = 1, 28, 128
        local_window = 16  # Attend to 16 nearby tokens
        
        query = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        key = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        value = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        
        # Get hybrid positions
        lattice, _, spacing = hybrid_boom_positions(seq_len, local_window)
        n_lattice = len(lattice)
        
        # Average positions attended per query
        avg_attend = local_window + n_lattice * (1 - local_window / seq_len)
        
        # Warmup
        for _ in range(5):
            _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            _ = vectorized_hybrid_attention(query, key, value, lattice, local_window, head_dim)
        
        torch.cuda.synchronize()
        
        # Time full attention
        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            full_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        torch.cuda.synchronize()
        full_time = (time.perf_counter() - start) / n_runs * 1000
        
        # Time hybrid attention
        start = time.perf_counter()
        for _ in range(n_runs):
            hybrid_out, mask = vectorized_hybrid_attention(query, key, value, lattice, local_window, head_dim)
        torch.cuda.synchronize()
        hybrid_time = (time.perf_counter() - start) / n_runs * 1000
        
        # Quality
        full_out_check = full_attention(query, key, value, head_dim)
        cosine = F.cosine_similarity(
            hybrid_out.flatten().unsqueeze(0).float(),
            full_out_check.flatten().unsqueeze(0).float()
        ).item()
        
        # Sparsity
        sparsity = mask.sum().item() / (seq_len * seq_len)
        
        speedup = full_time / hybrid_time
        
        results.append({
            'seq_len': seq_len,
            'n_lattice': n_lattice,
            'local_window': local_window,
            'avg_attend': avg_attend,
            'sparsity': sparsity,
            'full_time': full_time,
            'hybrid_time': hybrid_time,
            'speedup': speedup,
            'cosine': cosine,
        })
        
        print(f"\nSeq len: {seq_len}")
        print(f"  Lattice: {n_lattice} positions (spacing={spacing})")
        print(f"  Local window: {local_window}")
        print(f"  Sparsity: {sparsity*100:.1f}% of full attention")
        print(f"  SDPA: {full_time:.3f} ms")
        print(f"  Hybrid: {hybrid_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Cosine similarity: {cosine:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n| Seq Len | Lattice | Window | Sparsity | SDPA (ms) | Hybrid (ms) | Speedup | Cosine |")
    print("|---------|---------|--------|----------|-----------|-------------|---------|--------|")
    for r in results:
        print(f"| {r['seq_len']:7d} | {r['n_lattice']:7d} | {r['local_window']:6d} | {r['sparsity']*100:7.1f}% | {r['full_time']:9.3f} | {r['hybrid_time']:11.3f} | {r['speedup']:7.2f}x | {r['cosine']:.4f} |")
    
    return results


def analyze_quality_vs_window():
    """Analyze how quality changes with local window size."""
    print("\n" + "="*70)
    print("QUALITY VS LOCAL WINDOW SIZE")
    print("="*70)
    
    seq_len = 512
    batch, heads, head_dim = 1, 28, 128
    
    query = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    value = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    
    full_out = full_attention(query, key, value, head_dim)
    
    print(f"\nSeq len: {seq_len}")
    print("\n| Window | Lattice | Sparsity | Cosine |")
    print("|--------|---------|----------|--------|")
    
    for window in [4, 8, 16, 32, 64, 128]:
        lattice, _, _ = hybrid_boom_positions(seq_len, window)
        hybrid_out, mask = vectorized_hybrid_attention(query, key, value, lattice, window, head_dim)
        
        cosine = F.cosine_similarity(
            hybrid_out.flatten().unsqueeze(0).float(),
            full_out.flatten().unsqueeze(0).float()
        ).item()
        
        sparsity = mask.sum().item() / (seq_len * seq_len)
        
        print(f"| {window:6d} | {len(lattice):7d} | {sparsity*100:7.1f}% | {cosine:.4f} |")


def main():
    print("="*70)
    print("HYBRID SPIGOT ATTENTION")
    print("="*70)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"φ = {PHI:.6f}")
    
    print("""
THE HYBRID INSIGHT:

Pure spigot (φ-lattice only) has low quality (~0.5 cosine) because
attention in transformers is dominated by LOCAL context.

The hybrid approach:
1. φ-lattice: Distant semantic anchors (like BBP for far digits)
2. Local window: Nearby tokens (like direct computation for near digits)

This is analogous to how BBP works:
- For digit k, you compute a sum over all terms
- But terms far from k contribute less (they're "rounded away")
- The local terms dominate

In attention:
- Nearby tokens dominate (local window)
- Distant anchors provide global context (φ-lattice)
""")
    
    # Quality vs window size
    analyze_quality_vs_window()
    
    # Full benchmark
    results = benchmark_hybrid()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    mean_cosine = np.mean([r['cosine'] for r in results])
    mean_speedup = np.mean([r['speedup'] for r in results])
    
    print(f"""
HYBRID SPIGOT RESULTS:

1. QUALITY IMPROVEMENT
   - Pure spigot: ~0.5 cosine
   - Hybrid (window=16): ~{mean_cosine:.2f} cosine
   - Local window is crucial for quality

2. SPEED
   - Mean speedup: {mean_speedup:.2f}x vs SDPA
   - Still slower due to Python overhead and non-sparse implementation

3. THE HOLOGRAPHIC CONNECTION
   - φ-lattice = reference beam (implicit, universal)
   - Local window = signal beam (content-dependent)
   - Together they reconstruct the full attention pattern

NEXT STEPS:
- Implement truly sparse attention (not masked dense)
- Fuse into Triton kernel
- Test on actual generation (where local window is naturally small)
""")


if __name__ == "__main__":
    main()
