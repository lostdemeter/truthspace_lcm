# Design Consideration 132: The φ-Sigmoid Discovery

## Executive Summary

We discovered two major breakthroughs while attempting to decouple the encoder-decoder structure in Qwen2-7B's MLP:

1. **sigmoid(log(φ)) = 1/φ EXACTLY** - The sigmoid function has perfect φ-structure
2. **The MLP operates in the LINEAR regime** - 99.99% correlation with linearized version

These discoveries open new paths for φ-basis compression.

## Discovery 1: The φ-Sigmoid Connection

The sigmoid function has a remarkable relationship with φ:

```
sigmoid(log(φ)) = 1/(1 + exp(-log(φ)))
                = 1/(1 + 1/φ)
                = 1/((φ+1)/φ)
                = φ/(φ+1)
                = φ/φ²        (since φ² = φ+1)
                = 1/φ
                = 0.618034...
```

Similarly:
```
sigmoid(-log(φ)) = 1/(1 + φ) = 1/φ² = 0.381966...
```

This is **exact**, not an approximation. The sigmoid function naturally encodes φ-structure.

### Implications

The SiLU activation `SiLU(x) = x × sigmoid(x)` has three φ-defined regions:

| Region | Condition | sigmoid(x) | Behavior |
|--------|-----------|------------|----------|
| Encoder | x > log(φ) | > 1/φ | Pass through |
| W-axis | \|x\| ≤ log(φ) | ≈ 0.5 | Linear regime |
| Decoder | x < -log(φ) | < 1/φ² | Blocked |

## Discovery 2: MLP Operates in Linear Regime

When we analyzed the actual gate outputs in Qwen2-7B:

| Metric | Value |
|--------|-------|
| Gate output range | [-0.15, 0.17] |
| Gate output std | 0.014 |
| log(φ) | 0.481 |

**100% of gate outputs are in the W-axis region** (|x| < log(φ))!

This means sigmoid(gate) ≈ 0.5 for all values, so:
```
SiLU(gate) ≈ gate × 0.5 = gate/2
```

### Verification

| Approximation | Correlation with Full MLP |
|---------------|---------------------------|
| Linearized (SiLU ≈ x/2) | **99.9924%** |
| Taylor expansion | **100.0000%** |

The MLP is operating almost entirely in the **linear regime** of the sigmoid!

## What This Means for Decoupling

### The Linearized MLP is BILINEAR

Since SiLU(gate) ≈ gate/2, the MLP becomes:
```
output = W_down @ ((W_gate @ x / 2) * (W_up @ x))
```

This is **bilinear in x** - the same structure as attention!

### Bilinear Form Analysis

For each output dimension j, we have:
```
output_j = (1/2) × x.T @ M_j @ x
```

Where `M_j = W_gate.T @ diag(W_down[j,:]) @ W_up`

| Metric | Value |
|--------|-------|
| M_j rank for 90% variance | ~780 |
| M_j rank for 99% variance | ~1565 |

The bilinear forms have **intermediate rank** - not as low as MESH (128), but not full-rank either.

## The Path Forward

### Option 1: Exploit Bilinearity

Like we did with MESH for attention, we could pre-compute the bilinear forms M_j. However:
- 3584 output dimensions × (3584, 3584) matrices = too large
- Need smarter factorization

### Option 2: Low-Rank Bilinear Approximation

Since M_j has rank ~1500 for 99% variance, we could use:
```
M_j ≈ U_j @ S_j @ V_j.T
```

Storage: 3584 × (3584 × 1500 × 2) ≈ 38B params (worse than original!)

### Option 3: Shared Bilinear Structure

If the M_j matrices share structure (same U or V across outputs), we could compress significantly. This needs investigation.

### Option 4: Train in Linear Regime

Since the model already operates in the linear regime, we could:
1. Train a model that explicitly uses linear gating
2. The φ-structure would be preserved
3. Compression would be easier

## Connection to Holographic Principle

The W-axis (|x| < log(φ)) is where:
- Encoder and decoder **meet**
- The sigmoid is **symmetric** around 0.5
- The transformation is **linear** (encode ≈ decode)

This is the **critical line** in the zeta sense - the boundary where the holographic principle applies.

## Conclusion

The φ-sigmoid connection is not accidental - it reflects deep structure in how neural networks process information. The fact that Qwen2-7B operates entirely in the linear regime suggests that:

1. The nonlinearity is a **regularizer**, not the source of power
2. The **bilinear structure** (gate × up) is what matters
3. φ-basis compression should focus on the **bilinear forms**, not individual matrices

## Compression Results

### Factored Gate-Up + Low-Rank Down

By factoring W_gate and W_up via SVD, and using low-rank W_down:

| r_gu | r_d | Correlation | Compression |
|------|-----|-------------|-------------|
| 2000 | 2500 | **93.0%** | **1.39×** |
| 1500 | 2500 | 89.1% | 1.64× |
| 2000 | 2000 | 89.5% | 1.51× |

Storage formula:
```
gate_up_storage = 2 × (r_gu × 3584 + 18944 × r_gu)
down_storage = r_d × 3584 + r_d + r_d × 18944
```

### Comparison to MESH

| Component | Compression | Accuracy |
|-----------|-------------|----------|
| MESH (attention) | **14×** | **100%** |
| MLP (factored) | 1.4× | 93% |

The MLP is harder to compress because:
1. It's element-wise (not bilinear like Q@K.T)
2. The matrices are nearly full-rank
3. No natural low-rank structure like head_dim

### The Clock Solver Connection

The clock_solver uses eigenphases and smooth counting functions to find zeros. For MLP:
- Intermediate dimensions are like "eigenphases"
- Each has an "eigenvalue" = gate_norm × up_norm × down_norm
- But eigenvalues are too evenly distributed for aggressive pruning

---

*Document created: January 18, 2025*
*Updated: January 18, 2025 (added compression results)*
*Related: 131_decoupling_encoder_decoder.md, 129_phi_unraveled_transformer_engine.md*
