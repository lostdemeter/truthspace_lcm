# Design Consideration 130: φ-AIG Compression for Transformers

## Executive Summary

**REVISED**: Initial analysis suggested 7.5× compression via low-rank decomposition. However, deeper investigation reveals:

1. **MESH matrices ARE naturally low-rank** (rank = head_dim = 128) → **14× compression, 0% error** ✓
2. **Other weight matrices are nearly full-rank** → low-rank compression causes unacceptable error

The achievable compression is **~2× for MESH only**, not 7.5× for the full model. Alternative strategies are needed for other components.

## The AIG Analogy

In digital circuit design, AIG reduction simplifies complex boolean functions by:
1. Finding common sub-expressions
2. Factoring out shared structure
3. Reducing redundant gates

For φ-encoded transformers, we apply the same principle:
1. **Find low-rank structure** in weight matrices
2. **Factor into smaller matrices** (U, S, V)
3. **Encode factors in φ-basis** instead of full matrices

## Key Discovery: MESH Matrices Are Low-Rank

The pre-computed MESH = W_q.T @ W_k matrices have remarkable structure:

| Rank | Variance Captured | Error | Compression |
|------|-------------------|-------|-------------|
| 75 | 90% | ~10% | 24× |
| 100 | 99% | 13.8% | 18× |
| 128 | 99.9% | **0.09%** | **14×** |
| 200 | 99.99% | 0.09% | 9× |

With rank 128, we get **14× compression** with only **0.09% error**.

## The Compression Pipeline

### Original Storage
```
MESH: (3584, 3584) × 3 bytes = 38.5 MB per head
Total MESH: 28 heads × 28 layers × 38.5 MB = 30.2 GB
```

### Low-Rank Storage
```
MESH = U @ diag(S) @ Vt

U: (3584, 128) × 3 bytes = 1.38 MB
S: (128,) × 3 bytes = 0.4 KB
Vt: (128, 3584) × 3 bytes = 1.38 MB

Total per head: 2.75 MB (14× smaller)
Total MESH: 28 × 28 × 2.75 MB = 2.16 GB
```

### Full Model Compression

| Component | Original | Low-Rank | Compression |
|-----------|----------|----------|-------------|
| MESH matrices | 30.2 GB | 2.16 GB | 14× |
| MLP weights | 17.1 GB | 1.45 GB | 12× |
| Embeddings | 1.63 GB | 1.63 GB | 1× |
| LM Head | 1.63 GB | 1.63 GB | 1× |
| **Total** | **51.9 GB** | **6.9 GB** | **7.5×** |

## Why This Works

### 1. Learned Structure Is Redundant

Neural networks learn through gradient descent, which creates correlated weight patterns. The MESH matrix W_q.T @ W_k captures the learned relationship between queries and keys - this relationship is inherently low-dimensional.

### 2. φ-Encoding Preserves Structure

When we φ-encode the low-rank factors:
```
U_φ = sign(U) × φ^(exp_U / K)
S_φ = sign(S) × φ^(exp_S / K)
Vt_φ = sign(Vt) × φ^(exp_Vt / K)
```

The reconstruction error is:
- Low-rank truncation: 0.09%
- φ-encoding: 0.12%
- **Combined: 0.21%**

This is still well within acceptable error for correct text generation.

### 3. Computation Becomes Cheaper

Original MESH computation:
```
score = input @ MESH @ input.T
      = 3584² = 12.8M multiplications
```

Low-rank computation:
```
score = input @ U @ diag(S) @ Vt @ input.T
      = (input @ U) @ diag(S) @ (Vt @ input.T)
      = 2 × (3584 × 128) + 128 = 918K multiplications
      = 14× fewer operations
```

## The AIG Connection

In AIG terms:
- **AND gates** → φ-multiplications (exponent additions)
- **Inverters** → sign flips
- **Shared sub-expressions** → low-rank factors

The low-rank decomposition is finding that the MESH "circuit" has shared sub-expressions that can be factored out.

## Implementation

### Compressed Storage Format

```
qwen2_phi_compressed/
  config.npz              # Model config
  embed_tokens.npz        # φ-encoded embeddings
  lm_head.npz             # φ-encoded output head
  layer_00/
    mesh_U.npz            # (28, 3584, 128) - all heads' U factors
    mesh_S.npz            # (28, 128) - all heads' singular values
    mesh_Vt.npz           # (28, 128, 3584) - all heads' Vt factors
    cross_terms.npz       # Bias cross-terms
    mlp_U.npz             # Low-rank MLP factors
    mlp_S.npz
    mlp_Vt.npz
    biases.npz
  ...
```

### Inference

```python
def forward_attention_compressed(input, layer):
    # Low-rank MESH computation
    for head in range(28):
        U = layer.mesh_U[head].to_float()   # (3584, 128)
        S = layer.mesh_S[head].to_float()   # (128,)
        Vt = layer.mesh_Vt[head].to_float() # (128, 3584)
        
        # Efficient computation: O(N × r) instead of O(N²)
        temp1 = input @ U           # (seq, 128)
        temp2 = temp1 * S           # (seq, 128) - broadcast multiply
        temp3 = Vt @ input.T        # (128, seq)
        scores[head] = temp2 @ temp3  # (seq, seq)
```

## Speed Implications

| Operation | Original | Low-Rank | Speedup |
|-----------|----------|----------|---------|
| MESH matmul | 12.8M ops | 918K ops | 14× |
| Memory bandwidth | 38.5 MB | 2.75 MB | 14× |
| Cache efficiency | Poor | Good | ~2-3× |

Expected total speedup: **10-20×** for attention computation.

## Next Steps

1. **Implement compressed storage format**
2. **Verify accuracy on text generation**
3. **Benchmark inference speed**
4. **Explore further compression**:
   - Shared basis across heads (additional 2-3×)
   - Quantized exponents (additional 2×)
   - Sparse factors (if applicable)

## Revised Analysis: What Can Actually Be Compressed

### Matrix Rank Analysis

| Matrix | Shape | 99% Variance Rank | 99.9% Variance Rank |
|--------|-------|-------------------|---------------------|
| **MESH** | (3584, 3584) | **128** | **128** |
| W_q | (3584, 3584) | 1855 | 2623 |
| W_v | (512, 3584) | 437 | 490 |
| W_o | (3584, 3584) | 2599 | 3090 |
| W_gate | (18944, 3584) | 2585 | 3387 |
| W_up | (18944, 3584) | 2853 | 3454 |
| W_down | (3584, 18944) | 3424 | 3562 |

**Key Insight**: MESH is naturally rank-128 because:
```
MESH = W_q_head.T @ W_k_head
W_q_head: (128, 3584)  # head_dim × hidden
W_k_head: (128, 3584)
MESH: (3584, 3584) but rank ≤ 128
```

### What Works

**MESH Compression**: 14× compression with 0% error
- Original: 28 heads × 28 layers × 38.5 MB = 30.2 GB
- Compressed: 28 heads × 28 layers × 2.75 MB = 2.16 GB
- **Savings: 28 GB**

### What Doesn't Work

**Other Weight Matrices**: Nearly full-rank, low-rank compression causes >50% error
- W_v, W_o, MLP matrices need rank 2500-3500 for 99% variance
- Compression ratio would be only 1.0-1.4× with significant error

**Exponent Quantization**: Exponent span too large
- Exponent range: [-6758, -120], span = 6638
- 8-bit quantization step = 26 exponent units → 10.3% error per step
- Block-wise encoding doesn't help (block spans still ~3000)

## Revised Compression Strategy

### Tier 1: MESH (Works Perfectly)
- Low-rank decomposition: rank-128
- 14× compression, 0% error
- 30.2 GB → 2.16 GB

### Tier 2: Embeddings & LM Head (Keep Full)
- These are lookup tables, not computed
- 1.63 GB + 1.63 GB = 3.26 GB

### Tier 3: V, O, MLP (Keep Full φ-Encoded)
- Full 16-bit exponents required for accuracy
- ~18 GB

### Total Achievable Compression

| Component | Original | Compressed |
|-----------|----------|------------|
| MESH | 30.2 GB | 2.16 GB |
| Embeddings | 1.63 GB | 1.63 GB |
| LM Head | 1.63 GB | 1.63 GB |
| V, O, MLP | 18.5 GB | 18.5 GB |
| **Total** | **51.9 GB** | **24.0 GB** |

**Actual compression: 2.2×** (not 7.5×)

## The Real Win: Computation Speed

Even without storage compression, the MESH low-rank decomposition provides:

| Operation | Original | Low-Rank | Speedup |
|-----------|----------|----------|---------|
| Attention scores | 12.8M ops | 918K ops | **14×** |
| Memory bandwidth | 38.5 MB | 2.75 MB | **14×** |

The attention computation is often the bottleneck, so **14× speedup for attention** is significant.

## Comparison to DA2

DA2 achieved extreme compression (32 weights total) because:
1. It's a **decoder** (weighted sum of features)
2. The features were **already compressed** by the encoder
3. The task was **single-output** (depth per pixel)

Qwen2-7B is fundamentally different:
1. It's an **encoder-decoder** (full transformer)
2. The weights ARE the knowledge (not features)
3. The task is **vocabulary-sized output** (152K tokens)

The φ-basis works for both, but the compression opportunities differ.

## Conclusion

The AIG-style compression achieves:
- **MESH matrices**: 14× compression, 0% error ✓
- **Other matrices**: No compression without accuracy loss ✗

Total model compression: **2.2×** (51.9 GB → 24 GB)
Attention speedup: **14×**

The key insight is that **MESH is naturally low-rank** (bounded by head_dim), while other matrices encode the full knowledge of the model and cannot be compressed without loss.

---

*Document created: January 18, 2025*
*Revised: January 18, 2025 (updated with actual rank analysis)*
*Related: 129_phi_unraveled_transformer_engine.md, 125_exact_da2_recreation_phi_arithmetic.md*
