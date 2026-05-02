# Spatial Computing Protocol
## A Step-by-Step Guide for Geometric Problem Solving

---

## Overview

This protocol provides a systematic approach for solving problems using **spatial/geometric computing principles**. It addresses common pitfalls we've encountered and provides decision points for when to use geometric approximations vs. exact computation.

The core insight: **Not all problems have geometric shortcuts, but all problems have geometric structure.**

---

## The Five-Phase Protocol

```
┌─────────────────────────────────────────────────────────────┐
│              SPATIAL COMPUTING PROTOCOL                      │
│                                                             │
│  ┌──────────────┐                                          │
│  │ 1. STRUCTURE │  Identify the geometric structure        │
│  │    ANALYSIS  │  of the problem                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ 2. RANK      │  Determine effective dimensionality      │
│  │    ANALYSIS  │  and compressibility                     │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ 3. EXACT     │  Implement exact computation first       │
│  │    BASELINE  │  to establish ground truth               │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ 4. GEOMETRIC │  Test geometric approximations           │
│  │    APPROX    │  and measure correlation                 │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ 5. DECISION  │  Full-rank? → Exact computation          │
│  │    POINT     │  Low-rank? → Geometric shortcut          │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Structure Analysis

**Purpose**: Identify the geometric structure of the problem before attempting solutions.

**Key Questions**:
1. What are the inputs and outputs? (dimensions, types)
2. What transformations connect them? (linear, nonlinear, compositional)
3. What is the information flow? (sequential, parallel, recurrent)

**Tools**:
- Matrix shape analysis
- Computation graph tracing
- Weight distribution analysis (φ-level histogram)

**Process**:
```python
# Example: Analyzing a transformer layer
def analyze_structure(layer):
    print("=== Structure Analysis ===")
    
    # 1. Identify components
    components = {
        'attention': {
            'Q': layer.self_attn.q_proj.weight.shape,
            'K': layer.self_attn.k_proj.weight.shape,
            'V': layer.self_attn.v_proj.weight.shape,
            'O': layer.self_attn.o_proj.weight.shape,
        },
        'mlp': {
            'gate': layer.mlp.gate_proj.weight.shape,
            'up': layer.mlp.up_proj.weight.shape,
            'down': layer.mlp.down_proj.weight.shape,
        }
    }
    
    # 2. Check for biases (often forgotten!)
    biases = {
        'q_bias': layer.self_attn.q_proj.bias is not None,
        'k_bias': layer.self_attn.k_proj.bias is not None,
        'v_bias': layer.self_attn.v_proj.bias is not None,
    }
    
    # 3. Identify special structures (GQA, MQA, etc.)
    num_heads = layer.self_attn.num_heads
    num_kv_heads = layer.self_attn.num_key_value_heads
    is_gqa = num_kv_heads < num_heads
    
    return components, biases, is_gqa
```

**Output**:
- Component inventory with shapes
- Bias presence (critical for exact reproduction!)
- Special architectural features (GQA, RoPE, etc.)

**Common Pitfalls**:
- ❌ Forgetting biases (caused 0.926 → 1.000 correlation jump)
- ❌ Ignoring GQA head expansion
- ❌ Missing final layer norm

---

## Phase 2: Rank Analysis

**Purpose**: Determine if geometric compression is possible.

**Key Questions**:
1. What is the effective rank of each transformation?
2. How much variance is captured at different k values?
3. Is the structure inherently full-rank or compressible?

**Tools**:
- SVD analysis
- Cumulative variance plots
- Rank estimation with tolerance

**Process**:
```python
def analyze_rank(matrix, name="Matrix"):
    """Analyze the effective rank and compressibility of a matrix."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Effective rank at different thresholds
    total_var = np.sum(S**2)
    cumvar = np.cumsum(S**2) / total_var
    
    k90 = np.searchsorted(cumvar, 0.90) + 1
    k95 = np.searchsorted(cumvar, 0.95) + 1
    k99 = np.searchsorted(cumvar, 0.99) + 1
    
    print(f"=== Rank Analysis: {name} ===")
    print(f"Shape: {matrix.shape}")
    print(f"Full rank: {min(matrix.shape)}")
    print(f"k for 90% variance: {k90}")
    print(f"k for 95% variance: {k95}")
    print(f"k for 99% variance: {k99}")
    print(f"Top 5 singular values: {S[:5]}")
    
    # Decision: is this compressible?
    compression_ratio = k99 / min(matrix.shape)
    if compression_ratio < 0.5:
        print(f"✓ COMPRESSIBLE: {compression_ratio:.1%} of dims needed")
    else:
        print(f"✗ FULL-RANK: {compression_ratio:.1%} of dims needed")
    
    return {'k90': k90, 'k95': k95, 'k99': k99, 'S': S, 'compressible': compression_ratio < 0.5}
```

**Key Insight**: Different matrices have different compressibility:

| Matrix Type | Typical Rank | Compressible? |
|-------------|--------------|---------------|
| MESH (Q.T @ K) | ~106 / 3584 | ✓ Yes (3%) |
| V→O (single token) | 512 / 512 | ✗ No (100%) |
| MLP (gate, up, down) | ~3400 / 3584 | ✗ No (95%) |
| Embeddings | varies | Sometimes |

**Output**:
- Effective rank at 90%, 95%, 99% variance
- Compression feasibility assessment
- Singular value distribution (φ-Zipf check)

---

## Phase 3: Exact Baseline

**Purpose**: Implement exact computation to establish ground truth.

**Key Principle**: **Always get exact working first, then simplify.**

**Process**:
```python
def exact_forward(x, layer):
    """Exact layer computation - no approximations."""
    
    # 1. Input layer norm (RMSNorm)
    rms = np.sqrt(np.mean(x ** 2) + 1e-6)
    x_norm = (x / rms) * layer.input_layernorm.weight
    
    # 2. Attention (with biases!)
    q = layer.q_proj.weight @ x_norm + layer.q_proj.bias
    k = layer.k_proj.weight @ x_norm + layer.k_proj.bias
    v = layer.v_proj.weight @ x_norm + layer.v_proj.bias
    
    # 3. Handle GQA expansion
    v_expanded = expand_kv_heads(v, num_kv_heads=4, num_q_heads=28)
    
    # 4. For single token: attn_out = V @ O
    attn_out = layer.o_proj.weight @ v_expanded
    
    # 5. Residual
    x = x + attn_out
    
    # 6. Post-attention norm
    rms = np.sqrt(np.mean(x ** 2) + 1e-6)
    x_norm = (x / rms) * layer.post_attention_layernorm.weight
    
    # 7. MLP (exact SiLU, not bilinear!)
    gate = layer.gate_proj.weight @ x_norm
    up = layer.up_proj.weight @ x_norm
    silu_gate = gate / (1 + np.exp(-gate))  # Exact SiLU
    hidden = silu_gate * up
    mlp_out = layer.down_proj.weight @ hidden
    
    # 8. Residual
    x = x + mlp_out
    
    return x
```

**Verification**:
```python
# Compare with model output
model_output = model(input_ids, output_hidden_states=True).hidden_states[1]
our_output = exact_forward(embedding, layer)

correlation = np.corrcoef(our_output, model_output)[0, 1]
assert correlation > 0.9999, f"Exact baseline failed: {correlation}"
print(f"✓ Exact baseline verified: correlation = {correlation:.6f}")
```

**Output**:
- Working exact implementation
- Verified correlation with model (should be ~1.0)
- Baseline for comparison with approximations

---

## Phase 4: Geometric Approximation

**Purpose**: Test which geometric approximations work.

**Key Principle**: **Test one approximation at a time, measure impact.**

**Approximations to Test**:

### 4.1 Attention Approximation

```python
def test_attention_approximations(x_norm, layer, exact_attn):
    """Test different attention approximations."""
    
    results = {}
    
    # A. MESH SVD (for multi-token attention scores)
    MESH = layer.q_proj.weight.T @ layer.k_proj.weight
    U, S, Vt = np.linalg.svd(MESH)
    mesh_approx = U[:, :106] @ (S[:106] * (Vt[:106, :] @ x_norm))
    results['mesh_svd'] = np.corrcoef(mesh_approx, exact_attn)[0, 1]
    
    # B. V→O SVD (for single-token)
    A = layer.o_proj.weight @ expand_kv(layer.v_proj.weight)
    U, S, Vt = np.linalg.svd(A)
    for k in [128, 256, 384, 512]:
        approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :] @ x_norm
        results[f'vo_svd_k{k}'] = np.corrcoef(approx, exact_attn)[0, 1]
    
    return results
```

### 4.2 MLP Approximation

```python
def test_mlp_approximations(x_norm, layer, exact_mlp):
    """Test MLP approximations."""
    
    results = {}
    
    # A. Bilinear (SiLU ≈ x/2)
    gate = layer.gate_proj.weight @ x_norm
    up = layer.up_proj.weight @ x_norm
    bilinear = layer.down_proj.weight @ ((gate / 2) * up)
    results['bilinear'] = np.corrcoef(bilinear, exact_mlp)[0, 1]
    
    # B. Check gate range (is bilinear valid?)
    gate_range = (gate.min(), gate.max())
    pct_linear = np.mean(np.abs(gate) < 0.48) * 100
    results['gate_range'] = gate_range
    results['pct_in_linear'] = pct_linear
    
    return results
```

**Decision Matrix**:

| Approximation | When It Works | When It Fails |
|---------------|---------------|---------------|
| MESH SVD | Multi-token attention | Single-token (softmax=1) |
| V→O SVD | Never for Qwen2 | Always (full-rank) |
| Bilinear MLP | Later layers (4+) | Early layers (1-2) |
| φ-encoding | Storage | Computation |

---

## Phase 5: Decision Point

**Purpose**: Choose the right approach based on findings.

### Decision Tree

```
Is the transformation compressible? (k99 < 50% of full rank)
├── YES → Use geometric approximation
│   ├── Test correlation on held-out data
│   ├── Verify cumulative error doesn't explode
│   └── Document compression ratio achieved
│
└── NO → Use exact computation
    ├── Consider φ-encoding for storage
    ├── Look for sparsity patterns instead
    └── Accept that geometry is full-rank here
```

### When to Use Each Approach

**Use Exact Computation When**:
- Single-token generation (V→O is full-rank)
- Early layers (MLP is nonlinear)
- Accuracy is critical
- You're establishing a baseline

**Use Geometric Approximation When**:
- Multi-token context (attention scores matter)
- Later layers (MLP is more linear)
- Speed/memory is critical
- Compression ratio > 2x

**Use φ-Encoding When**:
- Storing weights (not computing)
- Transmitting model parameters
- Analyzing weight structure
- Integer arithmetic is available

---

## Common Pitfalls & Solutions

### Pitfall 1: Forgetting Biases
**Symptom**: Correlation stuck at 0.92-0.93
**Solution**: Check for biases on Q, K, V projections
```python
# Always check!
has_bias = layer.self_attn.v_proj.bias is not None
```

### Pitfall 2: Wrong Matrix for Approximation
**Symptom**: MESH SVD gives 0.01 correlation for single-token
**Solution**: MESH captures Q-K scores, not V-O output. Use the right matrix.
```python
# For attention scores: MESH = Q.T @ K
# For attention output: A = O @ expand(V)
```

### Pitfall 3: Assuming Linear Regime
**Symptom**: Bilinear MLP fails in early layers
**Solution**: Check gate value range before assuming SiLU ≈ x/2
```python
gate = gate_proj @ x_norm
if np.abs(gate).max() > 1.0:
    print("WARNING: Gate values outside linear regime!")
```

### Pitfall 4: Ignoring GQA
**Symptom**: Dimension mismatch errors
**Solution**: Expand KV heads to match Q heads
```python
def expand_kv_heads(v, num_kv_heads, num_q_heads):
    kv_per_q = num_q_heads // num_kv_heads
    v_heads = v.reshape(num_kv_heads, -1)
    return np.repeat(v_heads, kv_per_q, axis=0).reshape(-1)
```

### Pitfall 5: Missing Final Layer Norm
**Symptom**: Logits have wrong scale
**Solution**: Apply model.model.norm before LM head
```python
# Don't forget!
hidden = layer_norm(hidden, model.model.norm.weight)
logits = lm_head @ hidden
```

---

## Model-Specific Considerations

### Qwen2-7B-Instruct
- **GQA**: 4 KV heads, 28 Q heads (7:1 expansion)
- **Biases**: Q, K, V have biases; O does not
- **Early layers**: Gate values far outside linear regime
- **RoPE**: Position encoding (not needed for single-token)

### Depth Anything v2
- **Linear regime**: Gate values mostly in |x| < log(φ)
- **Bilinear MLP**: Works well (Doc 132)
- **Different architecture**: Findings may not transfer to LLMs

### General Principle
**Always verify assumptions on your specific model.** What works for one architecture may fail on another.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                 SPATIAL COMPUTING CHECKLIST                  │
├─────────────────────────────────────────────────────────────┤
│ □ Identify all components and their shapes                  │
│ □ Check for biases (Q, K, V, O, MLP)                       │
│ □ Check for special structures (GQA, MQA, RoPE)            │
│ □ Analyze rank of each transformation                       │
│ □ Implement exact baseline first                            │
│ □ Verify exact baseline matches model (corr > 0.999)       │
│ □ Test approximations ONE AT A TIME                         │
│ □ Check gate range before assuming bilinear MLP            │
│ □ Use correct matrix for approximation (MESH vs V→O)       │
│ □ Don't forget final layer norm before LM head             │
│ □ Document what works and what doesn't                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Theoretical Foundation

### Why Spatial Computing Works

1. **Structure IS Information**: Geometric relationships encode meaning
2. **Compression = Understanding**: Low-rank structure reveals patterns
3. **φ-Lattice**: Weights cluster on φ^n levels (Doc 128)
4. **Self-Similarity**: Same patterns at different scales

### Why Spatial Computing Has Limits

1. **Full-Rank Transformations**: Some operations are inherently full-rank
2. **Nonlinearity**: Activations can be essential, not just regularizers
3. **Model Training**: Models aren't trained for geometric efficiency
4. **Information Bottlenecks**: Compression has fundamental limits

### The Key Insight

> "Not all problems have geometric shortcuts, but all problems have geometric structure."

The goal isn't to force geometric approximations everywhere. It's to:
1. Understand the geometric structure
2. Use approximations where they work
3. Use exact computation where they don't
4. Learn from the structure even when shortcuts don't exist

---

## References

- Doc 112: Music Box Principle
- Doc 128: φ-Lattice Weight Structure
- Doc 132: φ-Sigmoid Discovery (model-specific!)
- Doc 135: Attention Head Semantic Specialization
- Doc 137: Integer φ-Encoding
- Doc 170: Full-Rank Attention Limit
- GOP: Gushurst Optimization Protocol

---

*"The geometry is always there. The question is whether it's compressible."*
— Spatial Computing Principle
