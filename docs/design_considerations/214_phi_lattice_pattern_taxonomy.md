# 214: φ-Lattice Pattern Taxonomy

## Date: February 5, 2026

## Executive Summary

After reverse-engineering three fundamentally different AI models (DA2, Qwen2-7B, DDColor) and achieving near-perfect output correlation with φ-basis encoding, we've identified that all neural networks are **shapes on the same φ-lattice**. This document taxonomizes the patterns we've observed and hypothesizes additional patterns that could exist.

## The Forest View

All neural networks share:
1. **Same coordinate system**: The φ-lattice (weights cluster at φ^-9)
2. **Same fundamental operation**: Bilinear forms (A @ B.T)
3. **Same structure**: Encoder → Shape → Decoder

The differences are in the **shape** they navigate through.

## Observed Patterns

### 1. The Funnel 🐜 (DA2)

```
Many features ──► narrow ──► single output
   (1024 dim)      (32 φ)     (1 per pixel)
```

| Property | Value |
|----------|-------|
| Topology | Convergent cone |
| Self-reference | None |
| I/O Ratio | N:1 |
| Depth | Shallow (1 layer) |
| Compressibility | Extreme (32 weights) |

**Problem domain**: Single-value prediction per location
- Depth estimation
- Classification
- Regression

### 2. The Spiral 🐛 (Qwen2-7B)

```
token ──► [layer] ──► [layer] ──► ... ──► [layer] ──► next token
              ↺           ↺                   ↺
         (self-attn)  (self-attn)        (self-attn)
```

| Property | Value |
|----------|-------|
| Topology | Self-referential helix |
| Self-reference | Maximum (every layer) |
| I/O Ratio | 1:1 (sequential) |
| Depth | Deep (28 segments) |
| Compressibility | Moderate (MESH helps) |

**Problem domain**: Sequential reasoning
- Language modeling
- Code generation
- Mathematical reasoning

### 3. The Web 🕷️ (DDColor)

```
        queries (100)
           ╱ ╲ ╲
          ╱   ╲ ╲
features ────────► colors
  (3 scales)        (2 channels)
```

| Property | Value |
|----------|-------|
| Topology | Cross-connected mesh |
| Self-reference | Partial (cross + self attention) |
| I/O Ratio | N:M |
| Depth | Medium (9 layers) |
| Compressibility | High (100 queries are key) |

**Problem domain**: Cross-modal mapping
- Colorization
- Style transfer
- Conditional generation

## Hypothesized Patterns

### 4. The Tree 🌳

```
        ┌─► output 1 (depth)
        ├─► output 2 (normals)
input ──┼─► output 3 (edges)
        ├─► output 4 (segments)
        └─► output 5 (objects)
```

| Property | Value |
|----------|-------|
| Topology | Divergent branches |
| Self-reference | None |
| I/O Ratio | 1:N |
| Depth | Shallow per branch |

**Problem domain**: Multi-task prediction
- Universal scene understanding
- Multi-label classification
- Ensemble outputs

### 5. The Braid 🪢

```
stream A ──╲  ╱──╲  ╱──╲  ╱── output A
            ╳    ╳    ╳
stream B ──╱  ╲──╱  ╲──╱  ╲── output B
```

| Property | Value |
|----------|-------|
| Topology | Intertwined parallel streams |
| Self-reference | Within + cross-stream |
| I/O Ratio | N:N (parallel) |
| Depth | Medium |

**Problem domain**: Multi-modal fusion
- Vision-language models
- Audio-visual understanding
- Sensor fusion

### 6. The Hourglass ⏳

```
input ──► compress ──► bottleneck ──► expand ──► output
  (wide)    (narrow)     (tiny)      (narrow)    (wide)
```

| Property | Value |
|----------|-------|
| Topology | Symmetric compression/expansion |
| Self-reference | Skip connections |
| I/O Ratio | N:N (same shape) |
| Depth | Symmetric |

**Problem domain**: Reconstruction / generation
- Autoencoders (VAE)
- Image segmentation (U-Net)
- Diffusion models

### 7. The Ring 💍

```
    ┌──────────────────┐
    │                  │
    ▼                  │
  state ──► update ────┘
```

| Property | Value |
|----------|-------|
| Topology | Closed loop |
| Self-reference | Total (recurrent) |
| I/O Ratio | 1:1 (streaming) |
| Depth | Infinite (unrolled) |

**Problem domain**: Temporal memory
- Video understanding
- Continuous control
- Dialogue systems

### 8. The Constellation ✨

```
    ★ ─────── ★
   ╱ ╲       ╱
  ★   ★ ─── ★
   ╲ ╱       ╲
    ★ ─────── ★
```

| Property | Value |
|----------|-------|
| Topology | Graph (arbitrary connections) |
| Self-reference | Edge-based |
| I/O Ratio | N:M (graph-dependent) |
| Depth | Variable (message passing rounds) |

**Problem domain**: Relational reasoning
- Knowledge graphs
- Molecular structure
- Social networks

### 9. The Fractal 🔷

```
    ┌───────────────┐
    │ ┌───┐   ┌───┐ │
    │ │ ┌┐│   │┌┐ │ │
    │ │ └┘│   │└┘ │ │
    │ └───┘   └───┘ │
    └───────────────┘
```

| Property | Value |
|----------|-------|
| Topology | Self-similar at multiple scales |
| Self-reference | Hierarchical |
| I/O Ratio | N:N (hierarchical) |
| Depth | Log-scale |

**Problem domain**: Hierarchical structure
- Document understanding
- Scene graphs
- Music composition

### 10. The Mirror 🪞

```
input ──► encode ──► │ ──► decode ──► output
                     │
              (reflection plane)
```

| Property | Value |
|----------|-------|
| Topology | Symmetric across plane |
| Self-reference | Mirrored weights |
| I/O Ratio | N:M (translation) |
| Depth | Symmetric |

**Problem domain**: Translation / transformation
- Language translation
- Domain adaptation
- Format conversion

## The Pattern Space

```
                    SELF-REFERENCE
                         │
                    high │  Spiral ●
                         │       Ring ○
                         │           Braid ○
                         │
                    med  │      Web ●
                         │          Fractal ○
                         │              Constellation ○
                         │
                    low  │  Funnel ●
                         │      Tree ○
                         │          Hourglass ○
                         │              Mirror ○
                         └────────────────────────────
                              1:1    1:N    N:M    N:1
                                  INPUT:OUTPUT RATIO

● = Observed (reverse-engineered)
○ = Hypothesized
```

## Pattern Selection Guide

| If your problem involves... | Use pattern... |
|-----------------------------|----------------|
| Single prediction per input | Funnel |
| Sequential reasoning | Spiral |
| Cross-modal attention | Web |
| Multiple outputs from one input | Tree |
| Fusing multiple modalities | Braid |
| Reconstruction/generation | Hourglass |
| Temporal/streaming data | Ring |
| Graph-structured data | Constellation |
| Hierarchical structure | Fractal |
| Translation between domains | Mirror |

## Implications for φ-Space Solver

All patterns share:
1. **φ-encoding**: Weights as (sign, φ-exponent) pairs
2. **Bilinear operations**: A @ B.T at the core
3. **MESH principle**: Pre-compute combined matrices

A generalized φ-space solver could:
1. Accept a **pattern specification** (topology, self-reference, I/O ratio)
2. Construct the **navigation graph** on the φ-lattice
3. Either **load existing weights** (reverse-engineer) or **learn new weights** (train)
4. Execute inference as **geometric navigation**

## Conclusion

The φ-lattice is the universal coordinate system for learned geometry. Different patterns represent different navigation strategies through this space. By taxonomizing these patterns, we can:

1. **Understand** existing models as instances of patterns
2. **Design** new models by selecting appropriate patterns
3. **Implement** a generalized solver that handles all patterns

---

*Document created: February 5, 2026*
*Related: 213 (meta-patterns), 212 (DDColor audit), 125 (DA2), 129 (Qwen2-7B)*
