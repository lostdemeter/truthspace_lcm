# Doc 197: Perspective-Invariant Analog Computation

## The Insight

**The tetrix looks different from different angles, but it's the SAME SHAPE.**

For transformers:
- The **weight matrix** is the invariant structure
- Different **inputs** are different perspectives
- Can we compute an "analog" that solves multiple problems at once?

## The Discovery

### Weight matrices have intrinsic structure

SVD decomposition of Qwen2-7B gate projection (18944 × 3584):

| Top-k SVs | Energy Captured | Error | Speedup |
|-----------|-----------------|-------|---------|
| 100 | 26.5% | 47.5% | **8.5×** |
| 500 | 58.9% | 31.9% | **4.1×** |
| 1000 | 81.7% | 19.2% | **2.0×** |
| 2000 | 97.1% | 8.1% | **1.3×** |
| 3000 | 99.6% | 3.5% | 0.95× |

### The Analog

The **perspective-invariant analog** is the SVD factorization:

```
W = U @ diag(S) @ Vh

Analog = (U_k, S_k, Vh_k)  for top-k singular values
```

Instead of computing `y = W @ h` directly:

1. **Project**: `p = Vh_k @ h` — O(k × d)
2. **Scale**: `s = S_k * p` — O(k)
3. **Expand**: `y = U_k @ s` — O(out × k)

**Total**: O(k × d + out × k) instead of O(out × d)

## Connection to Additive Error Stereoscopy

Just as additive error stereo computes ONE error field E and derives BOTH views:
- `LEFT = I - αE`
- `RIGHT = I + αE`

The SVD analog computes ONE factorization and handles ALL inputs:
- Project any input onto the shared subspace
- The "analog" (U_k, S_k, Vh_k) is perspective-invariant

## The Tradeoff

| k | Error | Speedup | Use Case |
|---|-------|---------|----------|
| 100 | 47% | 8.5× | Rough approximation, fast screening |
| 500 | 32% | 4.1× | Draft generation, speculative decoding |
| 1000 | 19% | 2.0× | Quality-speed balance |
| 2000 | 8% | 1.3× | High quality with modest speedup |
| 3000 | 3.5% | 1.0× | Near-lossless (no speedup) |

## Practical Applications

### 1. Speculative Decoding with Low-Rank Draft

Use k=500 (4× speedup, 32% error) as a draft model:
- Generate candidate tokens quickly
- Verify with full model
- Accept if match, reject if not

### 2. Adaptive Rank Selection

Different tokens need different precision:
- Common tokens (articles, prepositions): k=500
- Rare tokens (technical terms): k=2000
- Critical tokens (numbers, names): full rank

### 3. Hierarchical Computation

```
1. Compute with k=100 (fast, rough)
2. If confidence < threshold, refine with k=500
3. If still uncertain, use full rank
```

## The Deeper Principle

**The analog is the shared structure across all perspectives.**

Just as:
- The tetrix has the same fractal dimension from all angles
- Synthesis error E encodes depth gradients for all stereo views
- Top singular vectors capture the "core" of the weight matrix

The perspective-invariant analog lets us:
1. **Precompute** the shared structure once
2. **Apply** it to many inputs cheaply
3. **Refine** only when necessary

## Implementation Notes

```python
# Precompute analog (once at load time)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
U_k = U[:, :k]
S_k = S[:k]
Vh_k = Vh[:k, :]

# Fast inference (per token)
def fast_linear(h, U_k, S_k, Vh_k):
    p = Vh_k @ h      # Project
    s = S_k * p       # Scale
    return U_k @ s    # Expand
```

## Limitations

1. **Error accumulates**: Low-rank errors compound across layers
2. **Not all inputs equal**: Some inputs need full precision
3. **Memory overhead**: Must store U_k, S_k, Vh_k (but smaller than W)
4. **Precomputation cost**: SVD is expensive (one-time)

## Future Directions

1. **Learned rank selection**: Train a small network to predict optimal k per token
2. **Structured sparsity**: Combine low-rank with sparsity for more savings
3. **φ-level + low-rank**: Compress the factors themselves with φ-levels
4. **Adaptive refinement**: Start low-rank, refine based on output confidence

---

*Document created: February 3, 2026*
*Related: 196 (Bottleneck Analysis), DSS (Dimensional Shift Solver), Additive Error Stereo*
