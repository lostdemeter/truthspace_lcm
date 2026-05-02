# Design Consideration 253: Negative Zero as the Fourth Dimension

**Date:** February 19, 2026
**Status:** Discovery — experimentally validated on Qwen2-7B
**Prerequisites:** Doc 132 (φ-sigmoid), Doc 243 (GELU machine), Doc 245 (holographic gate field), Doc 247 (geometric φ-map), Phase 17C-D (push-pull / binary code)
**Finding:** 57 in FINDINGS.md

---

## 1. The Discovery

Neural network activation gates (SiLU, GELU) are not binary switches. They are
**4-state holographic encoders** where the sign at near-zero magnitude — "negative
zero" — carries essential information that previous analyses destroyed by treating
"dead" channels as empty.

In our φ-encoding system `(sign, φ-level)`, negative zero is:
- **sign = -1, φ-level ≈ 0** — a distinct point in φ-space from +0 (sign = +1, φ-level ≈ 0)
- The fourth coordinate in 4D φ-space, independent of the three spatial dimensions
- The "dark fringe" of the holographic interference pattern

This finding was validated independently on two architectures:
- **DDColor (ConvNeXt, GELU)**: Phase 17C — "dead" channels contribute 31.6% of output energy
- **Qwen2-7B (Transformer, SiLU)**: Finding 57 — CONTRACT channels contribute up to 42.4% of output energy

---

## 2. The 4-State Gate

The SiLU activation `SiLU(x) = x · σ(x)` creates four distinct operating regimes,
classified by the φ-lattice boundaries at ±log(φ) ≈ ±0.481:

```
     SiLU(x)
       │
  1.0  │                              ╱
       │                           ╱
  0.5  │                        ╱        +1 EXPAND: SiLU(x) ≈ x
       │                     ╱           Full fire, channels contribute proportionally
       │                  ╱
  log(φ)│- - - - - - - ╱ - - - - - - - - - - - - - boundary
       │            ╱                    +0 PRESERVE+: SiLU(x) ≈ x/2
       │         ╱                       Linear regime, positive side
  0.0  │──────╳─────────────────────
       │    ╱                            -0 PRESERVE-: SiLU(x) ≈ x/2
       │  ╱                              Linear regime, NEGATIVE side (THIS IS NEGATIVE ZERO)
 -log(φ)│╱- - - - - - - - - - - - - - - - - - - - boundary
       │                                 -1 CONTRACT: SiLU(x) ≈ x·exp(x)
 -0.5  │                                 Deep negative leakage
       │
      ─┼──────────────────────────── x
      -5  -3  -1   0   1   3   5
```

### Encoding: 2 bits per channel

| State | Sign bit | Magnitude bit | SiLU behavior | φ-encoding |
|-------|----------|--------------|---------------|------------|
| +1 | + | high | ≈ x (identity) | sign=+1, level=high |
| +0 | + | low | ≈ x/2 (linear) | sign=+1, level≈0 |
| -0 | - | low | ≈ x/2 (linear, negative) | sign=-1, level≈0 |
| -1 | - | high | ≈ x·exp(x) (leakage) | sign=-1, level=high |

The sign bit is the **more important** of the two bits (see §3).

---

## 3. Evidence: Sign > Magnitude

### Experiment: Remove sign vs remove magnitude in the PRESERVE region

In the PRESERVE region (|gate| ≤ log(φ)), we tested two ablations:
- **Remove sign**: replace SiLU(g) with |SiLU(g)| → destroys sign-at-zero
- **Keep only sign**: replace magnitude with constant, preserve sign → keeps sign-at-zero

| Layer | Remove sign (|SiLU|) | Keep ONLY sign | Sign advantage |
|-------|---------------------|----------------|---------------|
| 0 | 0.869 | **0.965** | 4.0× |
| 7 | 0.929 | **0.981** | 2.7× |
| 14 | 0.914 | **0.976** | 2.5× |
| 21 | 0.975 | **0.993** | 2.6× |
| 27 | 0.999 | **0.9997** | — |

At Layer 0, sign-only preserves 96.5% correlation while magnitude-without-sign drops to
86.9%. **The sign at zero carries approximately 4× more information than the magnitude.**

This extends Phase 17D's finding from DDColor: "sign pattern > magnitude for information
(5/6 blocks)" — the same principle holds in the transformer MLP.

### Why this matters

In IEEE floating point, -0 == +0 (they compare equal). In φ-space, they are **distinct
points**. The gate's sign at near-zero magnitude is a 1-bit decision that previous
approaches lost by:
- Treating channels as binary (active/dead)
- Skipping "dead" channels in sparse computation
- Using absolute values in SiLU approximations

---

## 4. Evidence: CONTRACT Energy Is Not Noise

### Energy decomposition by ternary region (through W_down)

| Layer | EXPAND | PRESERVE | **CONTRACT** | Total |
|-------|--------|----------|-------------|-------|
| 0 | 60.7% | 10.8% | **6.9%** | 78.4% |
| 7 | 69.2% | 4.4% | **20.2%** | 93.8% |
| 14 | 52.2% | 8.4% | **42.4%** | 103.0% |
| 21 | 74.4% | 2.3% | **24.7%** | 101.4% |
| 27 | 91.9% | 0.04% | **3.6%** | 95.5% |

**Layer 14: 42.4% of output energy from "dead" channels.**

Sum exceeds 100% at layers 14 and 21 because the positive and negative contributions
have NEGATIVE cross-terms — destructive interference. The anti-correlation between
positive and negative output contributions is -0.10 to -0.11 across middle layers.

This is the **push-pull architecture** from Phase 17C:
- Positive channels push the output in one direction
- Negative channels push the output in the OPPOSITE direction
- Together they create the complete interference pattern
- Neither alone is sufficient

---

## 5. The Holographic Interpretation

### Bright and dark fringes

A hologram encodes information in BOTH bright and dark fringes of an interference
pattern. Bright fringes alone give only half the information — you need the dark
fringes (where destructive interference occurs) to reconstruct the full wavefront.

The MLP gate operates identically:

```
HOLOGRAPHIC PLATE (the gate field)
══════════════════════════════════

Bright fringes (+1, +0):  Where the gate FIRES
  → Positive contribution to output
  → Carries the "what to say" signal

Dark fringes (-0, -1):    Where the gate LEAKS
  → Negative contribution to output
  → Carries the "what NOT to say" signal
  → Anti-correlated with bright fringes

Together:
  → Complete interference pattern
  → Full reconstruction of the intended output
```

### Connection to holographic gate field (Doc 245)

Doc 245 proposed that neural networks implement holographic computation:
- Weights define a **reference beam** (stable spatial structure)
- Input provides a **signal beam** (token-specific information)
- Activation creates an **interference pattern** (gate field)

Finding 57 adds: the interference pattern has FOUR regions, not two.
The -0 region (PRESERVE-) is where the reference and signal beams produce
**partial destructive interference** — not complete cancellation, but a
small negative residual. This residual IS the negative zero.

---

## 6. Connection to φ-Lattice and 4D Geometry

### The φ-encoding naturally supports negative zero

In our encoding system:
```
value = sign × φ^(exponent/128)
```

When exponent = 0: value = sign × φ^0 = sign × 1.0

But the CONCEPT of "near zero" in the gate means exponent is very negative:
```
+0: sign = +1, exponent → -∞  →  value → +0
-0: sign = -1, exponent → -∞  →  value → -0
```

These are distinct in φ-space (different sign bytes) even as their magnitudes converge.
The sign byte IS the fourth dimension:

```
4D φ-space coordinate:
  (dim_index, φ_level, weight_context, GATE_SIGN)
       ↑          ↑           ↑            ↑
      d=1..18944  φ^k     from W_gate     ±1
                                      ← THIS IS THE NEW DIMENSION
```

### Why 4D?

In 3D: (which channel, how much, which direction) — this loses +0 vs -0.
In 4D: (which channel, how much, which direction, **which side**) — complete.

The fourth dimension is binary (±1). It folds/unfolds based on the gate output:
- When gate > 0: fourth coordinate = +1
- When gate < 0: fourth coordinate = -1
- The magnitude of the gate determines "how far" along this fourth axis

This is exactly the "contextual dimension" from the Gated Regulatory Network
(geometric_patterns.py): a dimension that folds/unfolds based on content.

---

## 7. Distribution of 4 States Across Layers

The balance shifts from PRESERVE-dominated (early) to CONTRACT-dominated (late):

| Layer | +1 (EXPAND) | +0 (PRESERVE+) | -0 (PRESERVE-) | -1 (CONTRACT) |
|-------|------------|----------------|----------------|---------------|
| 0 | 2.0% | 24.7% | **44.8%** | 28.5% |
| 7 | 5.6% | 16.0% | 31.6% | **46.8%** |
| 14 | 7.1% | 16.9% | 29.5% | **46.5%** |
| 21 | 9.3% | 11.2% | 17.0% | **62.5%** |
| 27 | 8.2% | 3.7% | 5.4% | **82.7%** |

**Early layers** (0): The gate operates mostly in the PRESERVE region (±0 combined = 69.5%).
Most channels are in the linear regime. The MLP is nearly bilinear here.

**Late layers** (27): The gate is deeply contracted (82.7% in -1). Only ~12% of channels
are in the active region. The MLP is a highly selective filter — very few channels fire,
but those few carry almost all the energy (91.9% EXPAND energy at layer 27).

**Middle layers** (7-21): Mixed regime. About half the channels are contracted, half are
in the preserve/expand region. The CONTRACT contribution is largest here (20-42% of energy)
because there are enough contracted channels to contribute significant negative leakage,
AND their collective contribution has enough structure to matter.

---

## 8. Computational Implications

### What this means for MLP optimization

**1. You cannot skip dead channels.** (Finding 56 tried; Finding 57 proved why it fails.)
Binary (skip CONTRACT) drops correlation to 0.75 at layer 14. Including negative zero
recovers to 0.986.

**2. You CAN approximate dead channels cheaply — at late layers.**
Layer 27's CONTRACT output is low-rank (S[0]/S[1] = 4.508, rank 4 for 90% variance).
A rank-4 correction could replace the full CONTRACT computation at deep layers.

**3. The SIGN is the cheap part.** For PRESERVE channels, you only need to know which
side of zero the gate falls on (1 bit), not the exact magnitude. This is a binary
decision that the bias predicts 98-100% of the time (Phase 17D).

**4. The 4-state encoding maps to 2 bits per channel.** For 18,944 intermediate channels,
that's 37,888 bits = 4,736 bytes per token per layer. This could be precomputed or
transmitted as a tiny "gate code" that tells the receiver how to reconstruct the output.

### Potential architecture: Gate Code + Partial Compute

```
For each token:
  1. Compute gate = W_gate @ x                    (1 full matmul)
  2. Classify each channel: +1, +0, -0, -1        (FREE — just thresholds)
  3. EXPAND channels (~2-9%): compute exactly      (sparse matmul, very few)
  4. PRESERVE channels (~20-70%): use g/2 × u      (linearized, no sigmoid)
  5. CONTRACT channels (~28-83%): low-rank approx   (precomputed basis, few ops)
  6. Combine and project through W_down             (1 full matmul)

Savings: Eliminate sigmoid computation entirely.
         Reduce up_proj matmul to sparse (EXPAND only).
         Replace CONTRACT contribution with low-rank correction.
```

---

## 9. Summary

| Principle | Evidence |
|-----------|---------|
| Dead channels carry information | CONTRACT energy: 3.6-42.4% of output |
| Sign > magnitude at zero | Sign-only: 0.965-0.9997 correlation; magnitude-only: 0.869-0.999 |
| Push-pull architecture | Anti-correlation: -0.10 to -0.11 |
| 4 states, not 2 | +1, +0, -0, -1 cover 100% of channels with 2 bits |
| Negative zero = 4th dimension | φ-encoding (sign, level) distinguishes +0 from -0 |
| Consistent across architectures | DDColor GELU and Qwen2 SiLU show same structure |

> **"In a holographic system, the dark fringes carry as much information as the
> bright ones. Negative zero is not the absence of signal — it is the signal's
> shadow, and the shadow is half of the picture."**

---

## 10. References

- **Finding 57**: Full experimental results (explore_ternary_mlp.py)
- **Finding 55**: MLP linearization failure (explore_scaffold_mlp.py)
- **Finding 56**: Sparse MLP + cached Jacobian (explore_sparse_mlp.py)
- **Phase 17C**: Push-pull architecture in DDColor (ssm_phase17c_negative_space.py)
- **Phase 17D**: Binary code discovery, sign > magnitude (ssm_phase17d_gate_code_structure.py)
- **Doc 132**: φ-sigmoid discovery, SiLU linear regime
- **Doc 243**: The GELU Machine, GELU ≈ x·σ(φ·x)
- **Doc 245**: Holographic gate field
- **Doc 247**: Geometric φ-map, ternary classification (EXPAND/PRESERVE/CONTRACT)
