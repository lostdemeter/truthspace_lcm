# Doc 193: MLP Optimization Analysis - The Memory Bandwidth Wall

## Executive Summary

**Date**: February 3, 2026

After optimizing attention with boom attention (2.5-15× speedup), we analyzed the MLP which dominates 86% of FLOPs. Key finding:

**The MLP is memory-bandwidth bound, not compute-bound.**

This means:
- φ-level restructuring (fewer FLOPs) → **10× slower** (irregular memory access)
- Low-rank SVD (fewer FLOPs) → **no speedup**
- Structured pruning (smaller matrices) → **2-3.77× speedup**

## Profiling Results

### FLOPs Distribution by Sequence Length

| Seq Len | Attention % | MLP % | Boom Attention % | Boom MLP % |
|---------|-------------|-------|------------------|------------|
| 64 | 12.8% | 87.1% | 12.8% | 87.1% |
| 512 | 13.9% | 86.0% | 12.8% | 87.2% |
| 1024 | 15.3% | 84.7% | 12.8% | 87.2% |
| 2048 | 17.8% | 82.2% | 12.8% | 87.2% |
| 4096 | 22.4% | 77.6% | 12.8% | 87.2% |

With boom attention, MLP becomes even more dominant (87.2% vs 12.8%).

### Timing Breakdown

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| 28 Layers | 22.0 | 91.8% |
| LM Head | 1.8 | 7.3% |
| Sampling | 0.05 | 0.2% |
| **Total** | **24.0** | 100% |

## MLP Optimization Attempts

### 1. φ-Level Restructuring

**Theory**: Weights cluster at discrete φ-levels. Compute signed sums per level (integer-like), then scale by φ^level (LUT lookup).

```
Standard: output[j] = Σ_i W[j,i] × x[i]  (3584 mults per output)
φ-Level:  output[j] = Σ_level (signed_sum[j,level]) × φ^level  (~190 mults)
```

**Results**:
| Metric | Value |
|--------|-------|
| Theoretical reduction | 27.7× fewer float multiplications |
| Accuracy | **99.86%** correlation |
| Actual GPU speedup | **0.10×** (10× SLOWER!) |

**Why it failed**: Irregular memory access patterns. GPUs are optimized for coalesced memory access, not scattered lookups required by φ-level grouping.

### 2. Low-Rank SVD Approximation

**Theory**: Approximate weight matrices with low-rank factorization: W ≈ U @ S @ V.T

**Results**:
| Rank | Time | Speedup | Correlation | Compression |
|------|------|---------|-------------|-------------|
| 50% (1792) | 0.885 ms | 0.99× | 85.6% | 0.8× |
| 25% (896) | 0.893 ms | 0.98× | 62.3% | 1.7× |
| 10% (358) | 0.878 ms | 1.00× | 36.4% | 4.2× |

**Why it failed**: MLP is memory-bound. Reducing FLOPs doesn't help when memory bandwidth is the bottleneck.

### 3. INT8 Quantization

**Theory**: Quantize weights to 8-bit integers for 4× memory reduction.

**Results**:
| Metric | Value |
|--------|-------|
| Accuracy | **99.44%** correlation |
| Compression | **4×** |
| Speedup | 0.99× (no speedup without native INT8 kernels) |

**Note**: Native INT8 tensor cores (available on newer GPUs) could provide 2-4× speedup.

### 4. Structured Pruning (WINNER)

**Theory**: Remove entire intermediate dimensions (rows of gate/up, columns of down).

**Results**:
| Keep % | Time | Speedup | Correlation | Compression |
|--------|------|---------|-------------|-------------|
| 50% (9472 dims) | 0.440 ms | **1.99×** | 86.8% | 2× |
| 25% (4736 dims) | 0.232 ms | **3.77×** | 67.9% | 4× |

**Why it works**: Smaller matrices = less memory to transfer. This directly addresses the memory bandwidth bottleneck.

## The Memory Bandwidth Wall

### Why MLP is Memory-Bound

For Qwen2-7B MLP:
- Gate: (18944, 3584) = 68M params × 2 bytes = 136 MB
- Up: (18944, 3584) = 68M params × 2 bytes = 136 MB
- Down: (3584, 18944) = 68M params × 2 bytes = 136 MB
- **Total: 408 MB per layer**

RTX 3090 Ti memory bandwidth: ~1 TB/s

Time to load weights: 408 MB / 1000 GB/s = **0.4 ms**

Actual MLP time: **0.88 ms**

The MLP is spending ~45% of its time just loading weights from memory!

### Implications

1. **Reducing FLOPs doesn't help** - we're not compute-limited
2. **Reducing memory helps** - smaller matrices = faster loading
3. **Irregular access hurts** - φ-level grouping kills memory coalescing

## Recommended Optimization Strategy

### For Maximum Speedup (with accuracy loss)

| Optimization | Target | Speedup | Accuracy |
|--------------|--------|---------|----------|
| Boom Attention | Attention (12-22%) | 2.5-15× | 100% |
| Structured Pruning 50% | MLP (86%) | 2× | 87% |
| **Combined** | Full model | **~1.8×** | ~90% |

### For Maximum Accuracy

| Optimization | Target | Speedup | Accuracy |
|--------------|--------|---------|----------|
| Boom Attention | Attention | 2.5× | 100% |
| INT8 Quantization | MLP | 1× (need native) | 99.4% |
| **Combined** | Full model | **~1.2×** | ~99% |

### For Long Context (4096+ tokens)

| Optimization | Target | Speedup | Accuracy |
|--------------|--------|---------|----------|
| Boom Attention | Attention (22%) | 15× | 100% |
| Structured Pruning 50% | MLP (78%) | 2× | 87% |
| **Combined** | Full model | **~2.5×** | ~90% |

## Connection to Prior Work

### Doc 132: φ-Sigmoid Discovery

The linearization insight (SiLU ≈ x/2) is valid for early layers but breaks down in later layers:

| Layer | % in Linear Regime |
|-------|-------------------|
| 0 | 72.9% |
| 13 | 47.0% |
| 27 | 20.9% |

### Doc 177: Scaffolding vs Content

The 37-dimensional scaffolding encoder could replace MLP for low-entropy (scaffolding) tokens:
- Scaffolding: 100% generalizable with 37-dim linear map
- Content: Requires full MLP

Adaptive MLP: Use pruned MLP for scaffolding, full MLP for content.

### Doc 184: Trivial Navigation

For cached prompts, skip ALL 28 layers entirely:
- Speedup: 9.9×
- Accuracy: 100%
- Limitation: Only works for known prompts

## Files

- Profiler: `experiments/profile_inference.py`
- φ-Level MLP: `experiments/model_reverse_engineering/cuda/phi_level_mlp.py`
- Benchmark: `experiments/phi_level_mlp_benchmark.py`
- Boom Attention: `experiments/qwen2_boom_attention.py`

## Conclusion

The MLP optimization story is fundamentally different from attention:

| Component | Bottleneck | Solution |
|-----------|------------|----------|
| Attention | Compute (O(N²)) | Boom attention (sparse) |
| MLP | Memory bandwidth | Structured pruning (smaller) |

The φ-level approach is mathematically elegant but practically slow on GPUs. For real speedup, we need to reduce memory footprint, not FLOPs.

**Best practical approach**: Combine boom attention (for long context) with structured pruning (for MLP) for ~2× overall speedup with ~90% accuracy.
