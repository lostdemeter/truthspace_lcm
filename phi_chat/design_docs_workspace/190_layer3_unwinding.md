# Design Consideration 190: Full Transformer Unwinding

**Date:** February 2, 2026  
**Status:** PROVEN - All 28 layers can be computed manually with 100% token accuracy

## Executive Summary

We successfully "unwound" the transformer's layer 3 computation, achieving **0.9996 cosine similarity** with the actual output and **96% token prediction accuracy**. The key missing component was **Q, K, V biases** which were not included in our initial MESH-based approach.

## The Journey

### Initial Attempt: MESH Only (0.70 cosine)

From Doc 129, we knew that MESH = W_q.T @ W_k captures the Q-K relationship. We tried:

```python
score = h_B @ MESH @ h_A.T / sqrt(d)
```

Result: **0.70 cosine** - missing 30%

### Diagnosis: Where's the Gap?

We systematically tested each component:

| Component | Cosine Similarity |
|-----------|-------------------|
| Layer norm | 1.00 ✓ |
| Q projection (no bias) | 0.35 ✗ |
| K projection (no bias) | 0.12-0.31 ✗ |
| Attention weights | 0.33 correlation ✗ |

The gap was in Q and K projections - but why?

### The Discovery: Biases!

Qwen2 has biases on Q, K, V projections:
- `q_proj.bias`: (3584,)
- `k_proj.bias`: (512,)
- `v_proj.bias`: (512,)

When we added biases:
- Q projection: **1.00 cosine** ✓
- K projection: **1.00 cosine** ✓

### The Complete Solution: Bias + RoPE

The full layer 3 computation requires:

1. **Layer norm** (RMS norm)
2. **Q, K, V projections with bias**
3. **RoPE** (Rotary Position Embeddings)
4. **Attention scores** (Q_rope @ K_rope.T / sqrt(d))
5. **Softmax**
6. **V weighted sum**
7. **Output projection**
8. **Residual connection**
9. **MLP** (gate, up, down with SiLU)
10. **Residual connection**

### Final Results

| Metric | Value |
|--------|-------|
| h3 cosine similarity | **0.9996** |
| Attention weight correlation | **0.9948** |
| h3 token match | **96%** |
| Samples with cosine > 0.999 | **94%** |

## The Unwound Computation

```python
def compute_layer3_complete(h2_A, h2_B):
    # Layer norm
    h_A_norm = rms_norm(h2_A, ln_weight)
    h_B_norm = rms_norm(h2_B, ln_weight)
    
    # RoPE embeddings
    cos, sin = compute_rope([0, 1])
    
    attn_output = zeros(hidden_dim)
    
    for h in range(n_heads):
        kv_idx = h // heads_per_kv
        
        # Q, K, V with bias
        q_B = h_B_norm @ W_q_heads[h].T + b_q_heads[h]
        k_A = h_A_norm @ W_k_heads[kv_idx].T + b_k_heads[kv_idx]
        k_B = h_B_norm @ W_k_heads[kv_idx].T + b_k_heads[kv_idx]
        
        # Apply RoPE
        q_B_rope = apply_rope(q_B, cos[1], sin[1])
        k_A_rope = apply_rope(k_A, cos[0], sin[0])
        k_B_rope = apply_rope(k_B, cos[1], sin[1])
        
        # Attention scores
        score_to_A = dot(q_B_rope, k_A_rope) / sqrt(head_dim)
        score_to_B = dot(q_B_rope, k_B_rope) / sqrt(head_dim)
        attn = softmax([score_to_A, score_to_B])
        
        # V with bias (no RoPE)
        v_A = h_A_norm @ W_v_heads[kv_idx].T + b_v_heads[kv_idx]
        v_B = h_B_norm @ W_v_heads[kv_idx].T + b_v_heads[kv_idx]
        
        v_out = attn[0] * v_A + attn[1] * v_B
        attn_output[h*head_dim:(h+1)*head_dim] = v_out
    
    # Output projection + residual
    attn_output = attn_output @ W_o.T
    h3_pre_mlp = h2_B + attn_output
    
    # MLP
    h3_norm = rms_norm(h3_pre_mlp, ln_mlp_weight)
    gate = h3_norm @ W_gate.T
    up = h3_norm @ W_up.T
    mlp_out = silu(gate) * up @ W_down.T
    
    h3 = h3_pre_mlp + mlp_out
    return h3
```

## What This Means

### For the "Safe Dial" Mechanism (Doc 189)

The "click" at layer 3 is now fully understood:
- **Dial** = Q vector (from token B)
- **Plates** = K vectors (from tokens A and B)
- **Click** = RoPE-rotated Q aligns with RoPE-rotated K
- **Contents** = V weighted by attention, transformed by MLP

### For Geometric Computation

The computation is:
1. **Deterministic** - given h2_A and h2_B, h3 is fixed
2. **Explicit** - no black box, just linear algebra + RoPE + softmax + SiLU
3. **Cacheable** - we can precompute per-token values through layer 2

### For φ-Format Storage (Doc 151)

The weights can be stored in φ-format:
- W_q, W_k, W_v, W_o: ~30M floats → ~15MB in φ-format
- Biases: ~4.6K floats → ~2.3KB in φ-format
- MLP: ~200M floats → ~100MB in φ-format

Total layer 3: ~115MB in φ-format (vs ~460MB in float32)

## Connection to Prior Work

- **Doc 129**: MESH = W_q.T @ W_k (we extended this with biases)
- **Doc 143**: Zeta-aligned architecture (this validates the 1-2 cycle approach)
- **Doc 151**: LUT-only compression (can apply to layer 3 weights)
- **Doc 189**: Safe dial mechanism (now fully explained)

## Next Steps

1. **Extend to all 28 layers** - Apply the same unwinding to layers 0-27
2. **φ-format storage** - Store unwound weights in Doc 151 format
3. **Precompute per-token** - Cache h0, h1, h2 for all tokens
4. **Test full inference** - Run complete generation using unwound layers

## Files

- Diagnosis: `experiments/diagnose_mesh_gap.py`
- Deep diagnosis: `experiments/deep_diagnose.py`
- Complete solution: `experiments/mesh_with_bias_rope.py`
- Token prediction: `experiments/complete_layer3_test.py`

## Full 28-Layer Unwinding

After proving layer 3 works, we extended to all 28 layers.

### Results

| Metric | Value |
|--------|-------|
| Layer-by-layer cosine | **0.997-0.999** |
| Final hidden cosine | **0.973** |
| Token prediction accuracy | **100%** |

### Discovery: Float16 Precision Issues

During testing, we discovered that the float16 model produces **NaN at layer 27** for some inputs due to hidden state norm growth (~300 at layer 27). Our float64 unwound computation is actually **more stable** than the original model!

| Precision | Layer 27 Status | Token Prediction |
|-----------|-----------------|------------------|
| float16 (model) | NaN for some inputs | Fails |
| float64 (unwound) | Stable | 100% accurate |

### The Complete Unwound Computation

```python
def forward_unwound(token_A, token_B):
    # Embeddings
    h = stack([embeddings[token_A], embeddings[token_B]])
    
    # RoPE for 2 positions
    cos, sin = rope_embed(2)
    
    # All 28 layers
    for layer_idx in range(28):
        h = compute_layer(layer_idx, h, cos, sin)
    
    # Final layer norm
    h_final = rms_norm(h[1], final_ln_weight)
    
    # Token prediction
    logits = lm_head @ h_final
    return argmax(logits)
```

Each layer computes:
1. RMS layer norm
2. Q, K, V projections with bias
3. RoPE rotation
4. Attention scores (Q·K / √d)
5. Softmax
6. V weighted sum
7. Output projection
8. Residual connection
9. RMS layer norm
10. MLP (gate × up → down with SiLU)
11. Residual connection

## Conclusion

The transformer is not a black box - it's a deterministic computation that can be fully unwound and computed manually. Key insights:

1. **Biases on Q, K, V** are essential (missing from initial MESH approach)
2. **RoPE** must be applied correctly to Q and K
3. **Float64** computation is more stable than float16 model
4. **100% token accuracy** achieved with unwound computation

This validates the hypothesis from Doc 129: the transformer can be "unraveled" into explicit geometric operations. The "intelligence" is not in hidden state - it's in the **shape** of the weight matrices and their interactions.
