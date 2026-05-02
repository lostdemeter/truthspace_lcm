# 171: Dislodging the Full-Rank Basin

## The Hypothesis

Full-rank attention is **not fundamental**. It represents a **training basin** that standard LLM training lands in.

Our job:
1. **Dislodge** the model from this full-rank basin
2. **Reorganize** the data/weights so geometric shortcuts become possible
3. **Prove** that low-rank attention can achieve equivalent performance

## Why We Believe This

### Evidence from Doc 170

Single-token attention in Qwen2-7B uses full rank (512 dims) for V→O. But:

1. **MESH (Q.T @ K) IS low-rank** - effective rank ~106 for 99% variance
2. **Attention heads specialize** - different heads have different entropy (0.1% to 92%)
3. **Later layers are more linear** - bilinear MLP works in layers 4+

The model has geometric structure - it just didn't organize V→O to be compressible.

### The Basin Metaphor

```
Loss Landscape:

        ╱╲
       ╱  ╲
      ╱    ╲
     ╱      ╲
    ╱   ●    ╲     ← Current basin (full-rank V→O)
   ╱  (here)  ╲
  ╱            ╲
 ╱              ╲
╱       ○        ╲   ← Target basin (low-rank V→O)
                      (same loss, geometric shortcuts)
```

Standard training finds A basin, not THE basin. The full-rank solution works, so gradient descent stops there. But there may be equivalent-loss solutions with low-rank structure.

## Approaches to Dislodge

### 1. Explicit Low-Rank Constraints

Force V and O projections to be low-rank during training:

```python
# Instead of:
V = nn.Linear(hidden_dim, kv_dim)
O = nn.Linear(qo_dim, hidden_dim)

# Use:
V = LowRankLinear(hidden_dim, kv_dim, rank=128)
O = LowRankLinear(qo_dim, hidden_dim, rank=128)

class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, out_dim, bias=True)
    
    def forward(self, x):
        return self.V(self.U(x))
```

**Pros**: Guaranteed low-rank
**Cons**: May limit expressiveness, need to find right rank

### 2. Rank Regularization

Add a penalty for high effective rank:

```python
def nuclear_norm_penalty(weight_matrix, lambda_reg=0.01):
    """Encourage low nuclear norm (sum of singular values)."""
    U, S, V = torch.linalg.svd(weight_matrix, full_matrices=False)
    return lambda_reg * S.sum()

# In training loop:
loss = cross_entropy_loss + nuclear_norm_penalty(model.v_proj.weight)
```

**Pros**: Soft constraint, model can use full rank if needed
**Cons**: May slow convergence, hyperparameter tuning

### 3. Progressive Rank Reduction

Start with full rank, gradually reduce:

```python
def progressive_rank_schedule(epoch, max_epochs, initial_rank, target_rank):
    """Linearly reduce allowed rank during training."""
    progress = epoch / max_epochs
    return int(initial_rank - progress * (initial_rank - target_rank))

# Apply SVD truncation after each epoch
def truncate_to_rank(weight, k):
    U, S, V = torch.linalg.svd(weight, full_matrices=False)
    return U[:, :k] @ torch.diag(S[:k]) @ V[:k, :]
```

**Pros**: Gradual transition, model adapts
**Cons**: May cause training instability

### 4. φ-Lattice Quantization

Force weights onto φ-lattice during training (Doc 128):

```python
def phi_quantize(weight, scale=8192):
    """Quantize weights to φ-lattice points."""
    PHI = 1.6180339887498949
    sign = torch.sign(weight)
    log_phi = torch.log(torch.abs(weight) + 1e-10) / math.log(PHI)
    quantized_exp = torch.round(log_phi * scale) / scale
    return sign * (PHI ** quantized_exp)
```

**Hypothesis**: φ-lattice structure might naturally encourage low-rank organization because related weights cluster on the same lattice levels.

### 5. Geometric Loss Term

Add a loss term that rewards geometric structure:

```python
def geometric_structure_loss(V_weight, O_weight, target_rank=128):
    """Reward low effective rank in V→O path."""
    # Compute combined V→O matrix
    A = O_weight @ expand_kv(V_weight)
    
    # SVD
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    
    # Penalize variance outside top-k
    total_var = (S ** 2).sum()
    top_k_var = (S[:target_rank] ** 2).sum()
    
    return (total_var - top_k_var) / total_var  # Fraction of variance outside top-k
```

## What "Reorganizing the Data" Means

The weights aren't random - they encode learned relationships. Reorganizing means:

### 1. Factorization

Find a factorization where the geometry is explicit:

```
Current: V→O is a 512-rank linear map
Target:  V→O = U @ Σ @ V.T where Σ has only k significant values
```

### 2. Basis Change

Transform to a basis where the structure is sparse:

```
Current: Dense 3584-dim operations
Target:  Sparse operations in φ-basis or discriminant basis
```

### 3. Head Reorganization

Group heads by function (Doc 135 found semantic specialization):

```
Current: 28 heads, each doing "everything"
Target:  Head groups with specific geometric roles
         - Gender heads: low-rank, specific transform
         - Syntax heads: sparse attention patterns
         - etc.
```

## Experimental Plan

### Phase 1: Analyze Existing Structure

1. Map which heads are already low-rank vs full-rank
2. Identify if certain token types produce lower-rank attention
3. Check if fine-tuning has different rank characteristics than pretraining

### Phase 2: Small-Scale Training Experiments

1. Train a small transformer (e.g., 125M params) with rank constraints
2. Compare loss curves: constrained vs unconstrained
3. Measure if low-rank solution achieves comparable perplexity

### Phase 3: Reorganization Without Retraining

1. Apply SVD to existing V→O matrices
2. Test if truncated version maintains key behaviors
3. Identify which dimensions are "essential" vs "redundant"

### Phase 4: Hybrid Architecture

1. Design architecture with explicit geometric structure
2. Some layers exact, some layers geometric
3. Train end-to-end with geometric constraints

## Success Criteria

We've successfully dislodged from the full-rank basin if:

1. **Low-rank V→O** achieves >95% of full-rank performance
2. **Geometric shortcuts** provide >2x speedup
3. **Structure is interpretable** - we can explain what each dimension does

## Open Questions

1. **What is the minimum rank for language modeling?**
   - Is 128 enough? 64? 32?
   - Does it vary by layer?

2. **Does low-rank hurt specific capabilities?**
   - Reasoning? Factual recall? Code generation?
   - Can we identify which capabilities need full rank?

3. **Is there a "natural" low-rank structure?**
   - Does the model want to be low-rank but training pushes it full?
   - Or is full-rank genuinely better for the task?

4. **Can we retrofit existing models?**
   - Post-training rank reduction?
   - Fine-tuning with rank constraints?

## Connection to Project Goals

This directly addresses the core hypothesis:

> "LLMs are hyperdimensional transcoders where the geometry IS the computation."

If we can dislodge from the full-rank basin and achieve equivalent performance with low-rank geometric structure, we prove that:

1. The geometry CAN be the computation
2. Current models just haven't found that solution
3. Spatial computing IS viable for LLMs

---

## Progress Log

### 2026-01-29: Initial Analysis

- Discovered V→O is full-rank (512) in Qwen2-7B
- MESH (Q.T @ K) is low-rank (~106)
- Bilinear MLP fails in early layers (gate values outside linear regime)
- Created this document to track dislodging efforts

### 2026-01-29: Per-Head Analysis - LOW-RANK HEADS EXIST!

**Key Finding: 24 out of 784 Q heads (3.1%) are already low-rank (k90 < 80)**

Distribution by layer:
- **Layer 0**: 7 low-rank heads (first layer - embedding processing)
- **Layer 27**: 7 low-rank heads (last layer - output preparation)
- **Middle layers**: 10 scattered heads

Distribution by KV head:
- **KV Head 1**: 11 low-rank Q heads (46% of all low-rank)
- KV Head 3: 8 low-rank Q heads
- KV Heads 0, 2: 5 combined

**Layer 27, KV Head 1 is the most compressible:**
- k90 = 53 (vs ~103 typical)
- Top singular value captures 23.5% of variance
- Top 32 dimensions capture 80.8%
- Dominant direction U[:, 0] is **98.9% sparse**

**Implication**: The model HAS found some low-rank basins naturally. The question is why only 3% of heads, and can we push more heads into low-rank basins?

**Hypothesis**: First and last layers are more compressible because:
- Layer 0: Processing raw embeddings (structured input)
- Layer 27: Preparing for output projection (structured output)
- Middle layers: Complex reasoning (needs full rank?)

### 2026-01-29: Why Layer 27 Truncates But Layer 0 Doesn't

**Critical Discovery: It's about BIAS CANCELLATION and INPUT ALIGNMENT**

| Layer | Linear Norm | Bias Norm | Total Norm | Bias/Linear |
|-------|-------------|-----------|------------|-------------|
| 0 | 7.55 | 7.14 | **0.48** | 0.95 |
| 27 | 449.34 | 32.68 | 438.87 | 0.07 |

**Layer 0's problem:**
- Linear and bias terms nearly cancel (total = 0.48 vs components ~7.5)
- Truncation error gets amplified by the cancellation
- Only 7.1% of input energy aligns with top-32 singular vectors

**Layer 27's success:**
- Linear term dominates (14x larger than bias)
- 43% of input energy aligns with top-32 singular vectors
- The model has learned to produce inputs that "fit" the low-rank basis

**Key insight**: The linear term ALONE truncates perfectly at BOTH layers:
- Layer 0, k=32: 99.89% correlation
- Layer 27, k=32: 99.86% correlation

**Implication**: The V→O matrices ARE compressible. The issue is:
1. Bias terms that cancel with linear terms (Layer 0)
2. Input distributions that don't align with low-rank basis (Layer 0)

**What this means for dislodging:**
- Don't just constrain the weights - also need to align the INPUT distribution
- The model at Layer 27 has naturally learned this alignment
- Training with rank constraints might force earlier layers to learn alignment too

### 2026-01-29: Augmented Matrix Approach SOLVES Bias Cancellation

**The fix**: Treat bias as part of the matrix by augmenting input with constant 1:

```python
# Standard: output = A @ x + bias (separate terms that cancel)
# Augmented: output = A_aug @ [x; 1] (combined, no cancellation)

A_aug = np.column_stack([A, bias_term])  # (3584, 3585)
x_aug = np.append(x_norm, 1.0)           # (3585,)
```

**Results for Layer 0, Q head 25:**

| k | Augmented Corr | Standard Corr | Improvement |
|---|----------------|---------------|-------------|
| 32 | **0.982** | 0.774 | +27% |
| 64 | **0.990** | 0.856 | +16% |

**Why it works**: The augmented SVD captures the bias direction as part of the low-rank structure. The first singular value (7.62) is 10x larger than the second (0.79), indicating the bias is the dominant component.

### Why Biases Exist (and why some models don't use them)

**Models WITH biases** (Qwen2, GPT-2, BERT):
- More expressive (affine vs linear transforms)
- Better optimization (gradient flow)
- Historical convention

**Models WITHOUT biases** (LLaMA, Mistral, Gemma):
- Simpler geometry (pure linear)
- Cleaner low-rank truncation (no cancellation!)
- Meta found no performance loss

**Implication**: For geometric analysis, bias-free models (LLaMA) are cleaner. For biased models (Qwen2), use the augmented matrix approach.

### Next Steps

1. [x] Analyze per-head rank in V→O ✓
2. [x] Test if low-rank heads can be truncated ✓ (Layer 27 yes, Layer 0 no due to bias)
3. [x] Investigate what makes Layer 0 and 27 special ✓ (bias cancellation + input alignment)
4. [x] Absorb bias into low-rank approximation ✓ (augmented matrix works!)
5. [ ] Test augmented approach on full model forward pass
6. [ ] Compare with LLaMA (bias-free) for cleaner geometry
7. [ ] Explore φ-lattice quantization effects on rank

---

## References

- Doc 112: Music Box Principle
- Doc 128: φ-Lattice Weight Structure
- Doc 135: Attention Head Semantic Specialization
- Doc 170: Full-Rank Attention Limit (current state)
- Spatial Computing Protocol: Step-by-step guide
