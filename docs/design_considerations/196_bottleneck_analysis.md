# Doc 196: Bottleneck Analysis - Memory Bandwidth is the Wall

## Date: February 3, 2026

## Status: Validated

---

## Executive Summary

We're at **41 tokens/sec** because we're **memory bandwidth limited**, not compute limited.

| Metric | Value |
|--------|-------|
| Model size | 15.23 GB |
| GPU bandwidth | 1008 GB/s (theoretical) |
| Time to read weights | 18.9 ms |
| **Theoretical max** | **53 tok/s** |
| **Actual** | **41 tok/s (79%)** |

The MLP is 87.4% of parameters but compute is essentially free - the GPU is waiting for memory.

---

## 1. The Memory Bandwidth Wall

For each token generated, we must:
1. Read all 15 GB of weights from GPU memory
2. Perform ~200M FLOPs of computation
3. Write results back

The RTX 3090 Ti has:
- **1008 GB/s** memory bandwidth
- **40 TFLOPS** compute (bfloat16)

Time to read weights: 15.23 GB / 806 GB/s = **18.9 ms**
Time to compute: 200M FLOPs / 40 TFLOPS = **0.005 ms**

**Compute is 3,800× faster than memory!**

---

## 2. The Bilinear Storage Reality

### Naive Approach: 331 TB per layer
```
vocab² × hidden = 152K² × 3584 × 4 bytes = 331 TB
```
Impossible.

### Factored Approach: 25 GB per layer
```
vocab × intermediate × 2 = 152K × 18944 × 4 × 2 = 25 GB
```
Still huge, and doesn't eliminate W_down computation.

### The Problem: W_down is Full-Rank
```
99% variance requires rank 3424 (out of 3584)
Low-rank approximation doesn't help
```

### The Surprise: Bilinear Product is NOT Sparse
At layer 27, the bilinear product gate × up has:
- **17,000+ active dims** (out of 18,944)
- Almost no sparsity to exploit

This is because hidden states grow through layers:
- Layer 0: std = 0.014
- Layer 27: std = 25.1 (1,800× larger!)

---

## 3. What Actually Matters

### Current Performance Breakdown

| Component | Time | % |
|-----------|------|---|
| Weight reads | 18.9 ms | 79% |
| KV cache | ~2 ms | 8% |
| Attention | ~1 ms | 4% |
| Overhead | ~2 ms | 8% |
| **Total** | **~24 ms** | 100% |

### The Only Ways to Go Faster

1. **Reduce model size** (quantization)
   - INT8: 7.5 GB → ~100 tok/s
   - INT4: 3.75 GB → ~200 tok/s

2. **Skip weight reads** (caching)
   - Trivial navigation: Cache final hidden state
   - Only read LM head (1 GB) → ~800 tok/s

3. **Amortize reads** (speculative decoding)
   - Generate multiple tokens per forward pass
   - Read weights once, use for N tokens

4. **More bandwidth** (tensor parallelism)
   - Split across GPUs
   - Aggregate bandwidth

---

## 4. The φ-Structure Opportunity

While the bilinear product isn't sparse, the **weights themselves** have structure:

| Property | Value |
|----------|-------|
| Unique φ-levels | 45 |
| 90% coverage | Top 10 levels |
| 99% coverage | Top 15 levels |

This means we can represent weights as:
```
W[i,j] = sign[i,j] × φ^level[i,j]
```

Storage: 1 bit (sign) + 4 bits (level) = **5 bits per weight**
Compression: 16 bits / 5 bits = **3.2×**

With 3.2× compression:
- Model size: 15.23 GB → 4.76 GB
- Theoretical max: 53 tok/s → **170 tok/s**

---

## 5. The Path Forward

### Short-term: Quantization
- Use existing INT4/INT8 quantization
- Get to 100-200 tok/s

### Medium-term: φ-Level Compression
- Represent weights as (sign, φ-level) pairs
- Custom CUDA kernels for φ-level matmul
- Target: 150+ tok/s with full accuracy

### Long-term: Trivial Navigation + Caching
- Cache final hidden states for common patterns
- Skip transformer entirely for cached queries
- Target: 800+ tok/s for cached queries

---

## 6. Key Insight

**The 40GB bilinear storage was a red herring.**

We don't need to precompute bilinear coefficients because:
1. The bilinear product isn't sparse at later layers
2. The real bottleneck is memory bandwidth, not compute
3. Reducing weight reads is more valuable than reducing FLOPs

**The model doesn't have 40GB of unique information** - it has ~15GB of weights that cluster at 15 φ-levels. The "information" is in the structure, not the raw bytes.

---

## 7. Experimental Results

### Quantization Benchmarks (RTX 3090 Ti)

| Method | Speed | Memory | Accuracy |
|--------|-------|--------|----------|
| bfloat16 (baseline) | 42 tok/s | 15.25 GB | 100% |
| bitsandbytes INT8 | 17 tok/s | 8.83 GB | ~99% |
| bitsandbytes INT4 | 46 tok/s | 7.07 GB | ~95% |
| vLLM (bfloat16) | 45 tok/s | ~15 GB | 100% |
| **φ-level (7-bit)** | TBD | 6.19 GB | **99.3%** |

### Key Observations

1. **INT8 is SLOWER** (17 tok/s vs 42 tok/s)
   - bitsandbytes dequantizes on-the-fly
   - Overhead exceeds bandwidth savings

2. **INT4 is marginally faster** (46 tok/s)
   - 4× compression helps
   - But dequantization overhead limits gains

3. **vLLM matches HuggingFace** (45 tok/s)
   - Confirms we're at the bandwidth wall
   - Optimized kernels don't help when bandwidth-limited

4. **φ-level compression achieves 99.3% accuracy**
   - 7 bits per weight (1 sign + 6 level)
   - 2.29× theoretical compression
   - Needs custom CUDA kernel for speedup

### The Real Path to 170 tok/s

To achieve 170 tok/s, we need:
1. **3× compression** (read 4.7 GB instead of 14 GB)
2. **Zero-overhead decompression** (in-register, not in-memory)
3. **Custom CUDA kernels** (like llama.cpp, exllama)

The φ-level representation is ideal because:
- Decompression is just `sign × φ^level` (one multiply)
- φ^level can be a 64-entry LUT (fits in shared memory)
- No complex dequantization logic

---

## 8. Custom CUDA Kernel Results

We implemented a custom CUDA kernel for φ-level matmul:

| Metric | cuBLAS (bfloat16) | φ-level Kernel |
|--------|-------------------|----------------|
| Time | 0.2 ms | 4.2 ms |
| Speedup | 1.0× | **0.05×** |
| Bandwidth | 700 GB/s | 16 GB/s |

**Why the custom kernel is slow:**
1. cuBLAS uses tensor cores (8×8 matrix ops in hardware)
2. cuBLAS has years of optimization for memory access patterns
3. Our kernel does element-by-element processing
4. Shared memory tiling helps but can't match tensor cores

**The reality:** You can't beat cuBLAS with a simple custom kernel.

---

## 9. The Practical Path Forward

### What φ-level compression gives us:
- **2× smaller model files** (7 GB instead of 14 GB)
- **99% accuracy** (correlation with original)
- **Same inference speed** (decompress once at load time)
- **~28 seconds** to decompress full model

### For actual inference speedup:

| Approach | Expected Speedup | Notes |
|----------|------------------|-------|
| GPTQ/AWQ 4-bit | 2-3× | Requires pre-quantized model or quantization |
| vLLM/TensorRT | 1.5-2× | Optimized serving, continuous batching |
| Speculative decoding | 2-3× | Draft model + verification |
| Trivial navigation | 10×+ | Only for cached queries |

### Recommendation:
1. **Short-term**: Use φ-level for model storage/distribution
2. **Medium-term**: Use GPTQ/AWQ for inference speedup
3. **Long-term**: Implement trivial navigation for common patterns

---

## 10. Key Insight

**The transformer is memory-bandwidth limited, not compute limited.**

At 42 tok/s, we're at 79% of the theoretical maximum (53 tok/s).
The remaining 21% is overhead from KV cache, attention, and CUDA launches.

To go faster, we must either:
1. **Read less data** (quantization, caching)
2. **Read faster** (more GPUs, faster memory)
3. **Skip reading** (trivial navigation for cached queries)

The φ-level representation validates that weights have structure (62 discrete levels),
but exploiting this structure for speedup requires custom hardware or highly optimized kernels
that we can't easily write.

---

*Document created: February 3, 2026*
*Updated: February 3, 2026 - Added CUDA kernel results*
*Related: 195 (Bilinear MLP), 146 (φ-Bandwidth Limit), 184 (Trivial Navigation)*
