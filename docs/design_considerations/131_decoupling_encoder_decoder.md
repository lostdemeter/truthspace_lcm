# Design Consideration 131: Decoupling Encoder-Decoder for φ-Basis Compression

## Executive Summary

The reason we can compress MESH (attention) but not MLP is **mathematical structure**:

- **MESH**: Bilinear form → natural low-rank (rank ≤ head_dim = 128)
- **MLP**: Element-wise product → full-rank preserved

To achieve DA2-level compression for transformers, we need to **decouple the encoder-decoder structure** that was coupled during training. The MoE architecture does this explicitly.

## The Chicken-and-Egg Problem

### Why Dense MLPs Are Full-Rank

During training, the dense MLP learns to do **everything at once**:
- Encode input features
- Transform through nonlinearity
- Decode to output features

The gradient descent optimization couples these operations because it optimizes the **end-to-end** loss, not separate encoder/decoder losses.

Result: W_gate, W_up, W_down are all nearly full-rank (need rank 2500-3500 for 99% variance).

### Why MESH Is Low-Rank

Attention has a different structure:

```
Q @ K.T = (input @ W_q) @ (input @ W_k).T
        = input @ (W_q.T @ W_k) @ input.T
        = input @ MESH @ input.T
```

This is a **bilinear form** - the same input appears on both sides. The MESH matrix is the product of two matrices:

```
MESH = W_q_head.T @ W_k_head
W_q_head: (128, 3584)  # head_dim × hidden
W_k_head: (128, 3584)
```

By the rank inequality: `rank(A @ B) ≤ min(rank(A), rank(B))`

So: `rank(MESH) ≤ min(128, 128) = 128`

**The bilinear structure creates natural low-rank!**

### Why MLP Cannot Be Unraveled

The MLP computation is:

```
output = W_down @ (SiLU(W_gate @ input) * (W_up @ input))
```

This is an **element-wise product**, not a matrix product. There's no way to rewrite this as:

```
output = input @ SOMETHING @ input.T  # NOT POSSIBLE
```

Each of the 18,944 intermediate dimensions operates independently. The element-wise product preserves full dimensionality.

## The MoE Solution

Qwen2-57B-A14B (the MoE version) explicitly decouples what the dense model couples:

| Property | Dense (7B) | MoE (57B-A14B) |
|----------|------------|----------------|
| Intermediate size | 18,944 | 2,560 per expert |
| Experts | 1 (monolithic) | 64 routed + 8 shared |
| Active per token | 18,944 dims | 16 × 2,560 = 40,960 dims |
| Structure | Coupled | Decoupled |

### Why MoE Might Be More Compressible

Each expert in MoE handles a **specific function**:
- Shared experts: Common patterns (like "structural encoding")
- Routed experts: Specialized patterns (like "domain-specific decoding")

Because each expert is specialized, it might have:
1. Lower effective rank (handles fewer patterns)
2. More structure (specialized = more regular)
3. Better φ-basis representation (less coupling)

## The Path Forward

### Option 1: Use MoE Architecture

Instead of trying to compress dense Qwen2-7B, work with Qwen2-57B-A14B (MoE):
- Each expert is smaller (2,560 vs 18,944 intermediate)
- Experts might be individually compressible
- The routing provides natural sparsity

### Option 2: Learn to Decouple

Train a model where encoder and decoder are **explicitly separate**:
- Encoder: input → latent (compressible)
- Decoder: latent → output (compressible)
- Compose at inference

This is essentially what autoencoders do, but applied to transformer layers.

### Option 3: Factorize the MLP

Even though MLP isn't bilinear, we might find structure:

```
W_gate ≈ U_g @ V_g  (low-rank approximation)
W_up ≈ U_u @ V_u
W_down ≈ U_d @ V_d
```

If the low-rank factors share structure (e.g., same V), we could compress.

Current analysis shows this doesn't work well (need rank 2500+ for 99% variance), but there might be structure we haven't found yet.

### Option 4: Train in φ-Basis from Scratch

The ultimate solution: train a model where the weights are **natively φ-encoded**:
- Constrain weights to φ-basis during training
- The model learns to use the φ structure
- No post-hoc compression needed

This is the "fail-fast" approach from the project philosophy: prove the hypothesis by building it correctly from the start.

## Connection to DA2

DA2 achieved extreme compression (32 weights) because:
1. It's a **decoder only** (not encoder-decoder)
2. The encoder (DINOv2) already compressed the input
3. The task is single-output (depth per pixel)

The φ-basis worked because DA2 was already **decoupled** by design.

## Conclusion

The chicken-and-egg problem is real:
- Dense transformers couple encoder-decoder during training
- This coupling creates full-rank matrices
- Full-rank matrices cannot be compressed without loss

Solutions:
1. Use architectures that decouple (MoE)
2. Train with explicit decoupling
3. Train natively in φ-basis

The MESH unraveling worked because attention has **bilinear structure** that creates natural low-rank. MLP has **element-wise structure** that preserves full-rank.

---

*Document created: January 18, 2025*
*Related: 129_phi_unraveled_transformer_engine.md, 130_phi_aig_compression.md*
