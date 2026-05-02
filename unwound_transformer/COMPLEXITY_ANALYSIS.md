# Qwen2-7B Complexity Analysis and Simplification Opportunities

**Goal:** Examine every component mathematically, determine algorithmic complexity, and identify simplification opportunities.

---

## 1. Component Inventory

### 1.1 Data Structures

| Component | Shape | Size (bytes) | Type |
|-----------|-------|--------------|------|
| Embeddings | (152064, 3584) | 2.18 GB | float16 |
| Per-layer weights | 28 × ~270M params | 15.1 GB | float16 |
| Final LN | (3584,) | 7 KB | float16 |
| LM Head | (152064, 3584) | 2.18 GB | float16 |
| **Total** | ~7B params | **~14 GB** | float16 |

### 1.2 Per-Layer Weights

| Weight | Shape | Params | Purpose |
|--------|-------|--------|---------|
| W_q | (3584, 3584) | 12.8M | Query projection |
| W_k | (512, 3584) | 1.8M | Key projection (GQA) |
| W_v | (512, 3584) | 1.8M | Value projection (GQA) |
| W_o | (3584, 3584) | 12.8M | Output projection |
| b_q | (3584,) | 3.5K | Query bias |
| b_k | (512,) | 0.5K | Key bias |
| b_v | (512,) | 0.5K | Value bias |
| ln_attn | (3584,) | 3.5K | Pre-attention norm |
| ln_mlp | (3584,) | 3.5K | Pre-MLP norm |
| W_gate | (18944, 3584) | 67.9M | MLP gate |
| W_up | (18944, 3584) | 67.9M | MLP up |
| W_down | (3584, 18944) | 67.9M | MLP down |
| **Per layer** | | **233M** | |

---

## 2. Mathematical Formulation

### 2.1 Full Forward Pass

For input tokens $(t_1, t_2, \ldots, t_n)$:

```
h⁰ = Embed(tokens)                    # (n, d)
for l = 0 to L-1:
    h^l = Layer_l(h^{l-1})            # (n, d)
h_final = RMSNorm(h^L)                # (n, d)
logits = h_final @ W_lm^T             # (n, V)
```

### 2.2 Single Layer

```
h_norm = RMSNorm(h, γ_attn)
attn_out = MultiHeadAttention(h_norm)
h' = h + attn_out                     # Residual 1

h'_norm = RMSNorm(h', γ_mlp)
mlp_out = MLP(h'_norm)
h'' = h' + mlp_out                    # Residual 2
```

### 2.3 Core Operations

#### RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

**Complexity:** O(d)

#### Attention (per head)
$$Q = xW_Q + b_Q, \quad K = xW_K + b_K, \quad V = xW_V + b_V$$
$$Q', K' = \text{RoPE}(Q, K)$$
$$\text{Attn} = \text{softmax}\left(\frac{Q'K'^T}{\sqrt{d_h}}\right) V$$

**Complexity:** O(n²d_h) per head, O(n²d) total

#### MLP
$$\text{MLP}(x) = W_{down} \cdot (\text{SiLU}(xW_{gate}^T) \odot xW_{up}^T)$$

**Complexity:** O(d × d_ff) = O(d × 5.3d) ≈ O(5.3d²)

---

## 3. Complexity Analysis

### 3.1 Per-Token Complexity

| Operation | FLOPs per token | Notes |
|-----------|-----------------|-------|
| Embedding lookup | O(1) | Just indexing |
| RMSNorm (×2 per layer) | O(d) | 2 × 3584 = 7K |
| Q projection | O(d²) | 3584² = 12.8M |
| K projection | O(d × d_kv) | 3584 × 512 = 1.8M |
| V projection | O(d × d_kv) | 3584 × 512 = 1.8M |
| RoPE | O(d) | 3584 |
| Attention scores | O(n × d) | n × 3584 |
| Softmax | O(n) | n |
| Value aggregation | O(n × d) | n × 3584 |
| Output projection | O(d²) | 12.8M |
| MLP gate | O(d × d_ff) | 67.9M |
| MLP up | O(d × d_ff) | 67.9M |
| MLP down | O(d × d_ff) | 67.9M |
| **Per layer** | **~233M** | |
| **28 layers** | **~6.5B** | |
| **+ LM head** | **+545M** | |
| **Total** | **~7B FLOPs/token** | |

### 3.2 Sequence Length Scaling

| Component | Complexity | Bottleneck at |
|-----------|------------|---------------|
| Attention | O(n²d) | n > 1000 |
| MLP | O(nd²) | Always significant |
| Memory (KV cache) | O(nLd_kv) | n > 10000 |

---

## 4. Data Representation Analysis

### 4.1 Embeddings (152064 × 3584)

**Current:** Dense float16 matrix

**Observations:**
- Vocabulary is 152K tokens
- Many tokens are rare (Zipf distribution)
- Embedding vectors are ~0.8 norm on average

**Potential improvements:**
1. **Sparse representation** - Rare tokens could share bases
2. **Factorized embeddings** - E = A @ B where A is (V, k), B is (k, d)
3. **Quantization** - 4-bit or 8-bit with minimal loss
4. **φ-lattice encoding** - Snap to φ^n levels (from Doc 128)

### 4.2 Attention Weights

**Current:** Separate W_q, W_k, W_v, W_o matrices

**Observations:**
- GQA already reduces K, V to 4 heads (7x compression)
- W_q and W_o are full rank (3584 × 3584)
- Effective rank of W_q is ~2000-3400 (from our analysis)

**Potential improvements:**
1. **Low-rank factorization** - W = UV where U is (d, r), V is (r, d)
2. **Shared projections** - Some layers have similar W_q (from similarity analysis)
3. **Structured matrices** - Toeplitz, circulant, or butterfly

### 4.3 MLP Weights

**Current:** Three dense matrices (gate, up, down)

**Observations:**
- MLP is 5.3× hidden dim (18944 vs 3584)
- From Doc 132: MLP operates in LINEAR regime (SiLU ≈ x/2)
- Linearized MLP has 99.99% correlation with full MLP

**Potential improvements:**
1. **Bilinear approximation** - Since SiLU ≈ x/2, MLP ≈ W_down @ ((gate/2) * up)
2. **Factorization** - Reduce intermediate dimension
3. **Pruning** - Many weights near zero

---

## 5. Simplification Opportunities

### 5.1 Attention Simplification

**Current complexity:** O(n²d)

**Opportunities:**
1. **Boom-based sparse attention** (Doc 159)
   - Only compute attention at "boom" positions
   - 80% of attention mass in 20% of positions
   - Potential: O(n × 0.2n × d) = O(0.2n²d)

2. **Linear attention**
   - Replace softmax(QK^T)V with φ(Q)φ(K)^T V
   - Complexity: O(nd²) instead of O(n²d)

3. **Fixed attention patterns**
   - Scaffolding tokens (Doc 177) have predictable attention
   - Pre-compute for common patterns

### 5.2 MLP Simplification

**Current complexity:** O(d × d_ff) = O(5.3d²)

**Opportunities:**
1. **Linearization** (Doc 132)
   - SiLU(gate) ≈ gate/2 in practice
   - MLP becomes bilinear: W_down @ ((W_gate @ x / 2) * (W_up @ x))
   - Same complexity but simpler gradient flow

2. **Rank reduction**
   - Effective rank ~1500 (from Doc 132)
   - Could use (d, 1500) × (1500, d_ff) factorization

3. **Scaffolding bypass**
   - For scaffolding tokens, use 37-dim linear map (Doc 177)
   - Skip full MLP entirely

### 5.3 Representation Simplification

**Current:** Dense float16

**Opportunities:**
1. **φ-lattice quantization** (Doc 128)
   - 97% of weights fit on φ^n lattice
   - Store as (level, sign) pairs
   - Decode: value = sign × φ^level

2. **Holographic encoding** (Doc 177)
   - 37 dimensions capture 90% variance
   - Project to 37-dim, compute, project back

3. **Bulge compression** (Doc 180)
   - Trajectories = geodesic + bulge
   - Store 10 coefficients instead of full trajectory
   - 2867× compression

---

## 6. Unified Geometric View

### 6.1 The Core Insight

From our discoveries:
- **ENCODE = DECODE** (Doc fundamental)
- **φ is everywhere** - weights, activations, attention
- **Self-similarity** - same patterns at all scales

### 6.2 Proposed Simplified Architecture

```
TOKEN → φ-COORDINATE → MANIFOLD TRAVERSAL → φ-COORDINATE → TOKEN
         (absolute)      (zeta-aligned)       (absolute)
```

Where:
1. **φ-coordinate** = position on φ-lattice (not learned embedding)
2. **Manifold traversal** = geodesic + boom-based attention
3. **Zeta-aligned** = phase transitions at 137/30 ratio

### 6.3 Complexity Comparison

| Component | Current | Proposed |
|-----------|---------|----------|
| Embedding | O(1) lookup | O(1) φ-decode |
| Attention | O(n²d) | O(n × booms × d) |
| MLP | O(d × d_ff) | O(d × rank) or skip |
| Memory | O(nLd) | O(booms × L × d) |

---

## 7. Measured Results

### 7.1 Effective Rank (threshold = 1% of max singular value)

| Layer | W_q | W_o | W_gate |
|-------|-----|-----|--------|
| 0 | 63% (2261/3584) | 92% | 82% |
| 7 | 95% | 97% | 100% |
| 14 | 95% | 95% | 100% |
| 21 | 96% | 95% | 100% |
| 27 | 87% | 89% | 100% |

**Key finding:** Layer 0 W_q has only 63% effective rank - significant compression possible.

### 7.2 MLP Linearization

| Metric | Random Inputs | Actual Inference |
|--------|---------------|------------------|
| Full vs Linear (x/2) correlation | 0.871 | 0.886 |
| Full vs Tanh approx correlation | - | **0.961** |
| Gate std | 0.83 | **2.12** |
| Gate range | - | [-6.25, 5.97] |
| % in |x| < 0.5 | 47% | 68% |

**Key finding:** Doc 132 claimed gate std = 0.014, but actual inference shows std = 2.12 (150× larger!). The linear approximation (x/2) gives 0.88 correlation, but a **tanh approximation gives 0.96 correlation**.

**Resolution:** Doc 132 may have measured a specific layer or normalized inputs. For actual inference, use:
```python
SiLU(x) ≈ x * (0.5 + 0.197 * tanh(0.797 * x))  # 0.96 correlation
```

### 7.3 φ-Lattice Alignment

| Component | % Aligned | Peak Level |
|-----------|-----------|------------|
| Embeddings | 20% | φ^-9 |
| Layer 0 W_q | 20% | φ^-9 |
| Layer 14 W_q | 20% | φ^-9 |
| Layer 27 W_q | 20% | φ^-9 |

**Key finding:** Only 20% φ-aligned (not 97% as hoped). Peak at φ^-9 ≈ 0.013 is consistent.

### 7.4 Embedding Structure

| Variance | Rank Needed |
|----------|-------------|
| 50% | 351 |
| 90% | 1425 |
| 95% | 2051 |
| 99% | 3218 |

**Key finding:** Embeddings are highly compressible - 351 dimensions capture 50% variance!

---

## 8. Prioritized Simplification Opportunities

Based on measurements:

### HIGH IMPACT
1. **Factorized Embeddings** - 351 dims for 50%, 1425 for 90%
   - Current: 152K × 3584 = 545M params
   - Factorized (90%): 152K × 1425 + 1425 × 3584 = 222M params (59% reduction)

2. **Layer 0 W_q Low-Rank** - Only 63% effective rank
   - Current: 3584 × 3584 = 12.8M params
   - Low-rank (63%): 3584 × 2261 + 2261 × 3584 = 16.2M (no savings, but faster matmul)

### MEDIUM IMPACT
3. **Boom-based Sparse Attention** - 80% mass in 20% positions
   - Potential 5× speedup for long sequences

4. **Scaffolding Token Bypass** - Use 37-dim linear map
   - Skip full computation for predictable tokens

### LOWER IMPACT (based on measurements)
5. **MLP Linearization** - 0.877 correlation (not high enough)
   - Need more investigation

6. **φ-Quantization** - Only 20% aligned
   - May need different quantization scheme

---

## 9. Next Steps

1. **Test factorized embeddings** - Measure accuracy with 1425-dim factorization
2. **Investigate MLP linearization discrepancy** - Why 0.877 not 0.999?
3. **Implement boom-based attention** - Test on long sequences
4. **Build minimal model** - Start with embeddings + layer 0

---

## 10. Key Questions

1. What is the **minimum rank** needed for each projection?
2. Can we **predict boom positions** without computing attention?
3. Is the **37-dim holographic bound** universal?
4. Can we **replace embeddings** with φ-coordinates?
5. What is the **irreducible core** of the transformer?

---

## 11. Summary Table

| Component | Current | Measured | Opportunity |
|-----------|---------|----------|-------------|
| Embeddings | 545M params | 90% var in 1425 dims | **59% reduction** |
| W_q (layer 0) | 12.8M | 63% rank | Low-rank factorization |
| W_q (other) | 12.8M × 27 | 87-96% rank | Minor compression |
| W_gate | 67.9M × 28 | 100% rank | No compression |
| MLP | SiLU nonlinear | 78% linear regime | 0.877 correlation (investigate) |
| Weights | float16 | 20% φ-aligned | Different quantization needed |
| Attention | O(n²) | - | Boom-based sparsity |

**Bottom line:** Embeddings are the biggest opportunity (59% reduction possible). MLP linearization needs more work. φ-quantization is not as promising as hoped.

---

## 12. Experimental Results (Feb 2, 2026)

### 12.1 Factorized Embeddings

| Rank k | Reconstruction Error | Accuracy | Param Savings |
|--------|---------------------|----------|---------------|
| 351 | 70.7% | 40% | 90% |
| 1425 | 31.7% | **80%** | **59%** |
| 2051 | 22.4% | 80% | 41% |

**Finding:** k=1425 gives 80% accuracy with 59% param savings. Accuracy plateaus - need to investigate why not 100%.

### 12.2 MLP Linearization

| Approximation | Correlation |
|---------------|-------------|
| Linear (x/2) | 0.88 |
| **Tanh approx** | **0.96** |

**Finding:** Doc 132's claim of 0.999 correlation was based on different inputs (gate std 0.014 vs actual 2.12). Tanh approximation is better:
```python
SiLU(x) ≈ x * (0.5 + 0.197 * tanh(0.797 * x))
```

### 12.3 Boom-Based Attention

| Sequence | Boom % | Attention Mass |
|----------|--------|----------------|
| 10 tokens | 20% | **79.7%** |
| 15 tokens | 20% | **76.1%** |
| 16 tokens | 19% | **76.5%** |
| 24 tokens | 17% | **73.6%** |

**Finding:** Top 20% of positions capture 73-80% of attention mass. Boom tokens are typically:
- Sentence starts ("The", "In", "Once")
- Punctuation (".", ",")
- Key structural words

**Challenge:** Need to identify boom positions WITHOUT computing full attention.

---

## 13. Revised Simplification Priorities

Based on experiments:

### PROVEN OPPORTUNITIES
1. **Factorized Embeddings (k=1425)** - 80% accuracy, 59% savings
2. **Boom-based Sparse Attention** - 73-80% mass in 20% positions
3. **Tanh MLP Approximation** - 0.96 correlation

### NEEDS MORE WORK
4. **Why factorized embeddings plateau at 80%** - Investigate
5. **Boom position prediction** - How to identify without full attention?
6. **φ-quantization** - Only 20% aligned, need different approach

### NOT PROMISING
7. **Linear MLP (x/2)** - Only 0.88 correlation
8. **Full-rank compression** - W_gate is 100% rank
