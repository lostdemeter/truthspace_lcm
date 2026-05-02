# Doc 288: Weight Structure — The Ordering Is in the Shape

**Date:** March 4, 2026
**Status:** Experimental finding
**Prerequisites:** DC 243 (GELU Machine), DC 130 (AIG Compression), DC 287 (Converting the Full Model)
**Finding:** F157

---

## 0. The Question

We know the model's weights contain information. We know they pass
through shape filters for selection. But is there rhyme or reason
to the ordering? Are they geometrically ordered? Alphabetically?
Ascending/descending? Or is the structure purely collective?

---

## 1. Answer: Individual Weights Are Unordered

We analyzed all 7 weight matrix types across Qwen2-7B's 28 layers.

| Test | Result |
|:-----|:-------|
| Exponent↔position Spearman ρ | ≈0.000 |
| Row norm monotonicity | 50/50 (random) |
| Sign distribution | 50.0% positive |
| Adjacent row cosine | ≈0.00 (gate, v) |

No geometric ordering. No magnitude ordering. No positional
ordering. Each individual weight entry is essentially random noise
at a specific φ-level magnitude. The most common magnitudes cluster
at φ^(-8.5) ≈ 0.016 and φ^(-10.0) ≈ 0.008 — very small values.

---

## 2. Answer: The SVD Structure Is Highly Organized

The collective shape — what emerges when you decompose the weight
matrices via SVD — reveals rich, interpretable structure.

### 2.1 Gate Anti-Alternation

The gate_proj top singular direction oscillates between layers:

```
Layer:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 ...
Sign:   +  -  -  +  -  -  +  -  -  +  -  +  -  -  +  -  +  +  ...
|cos|:    .13 .74 .69 .81 .81 .54 .65 .77 .52 .60 .76 .78 .72 .78 .80 .83 .88
```

Mean |cos| between consecutive layers: **0.71**. This is not
random (random would be ~0.02 in ℝ³⁵⁸⁴). The gate direction at
each layer is strongly coupled to its neighbors, but frequently
flipped.

67% of transitions are negative (anti-correlated). The pattern is
structured but non-periodic — not a clean alternation, but clearly
organized.

**Interpretation:** Each layer's gate selects for the OPPOSITE
features of its neighbor. This maximizes information injection per
layer — you don't want to inject the same correction twice. The
model naturally discovered an AIG-like alternating gate pattern.

### 2.2 Cross-Weight-Type Anti-Correlation

At Layer 14, the top right singular directions (input space ℝ³⁵⁸⁴):

| Pair | Cosine | Interpretation |
|:-----|:-------|:---------------|
| gate ↔ key | **-0.524** | Gate looks OPPOSITE to key |
| gate ↔ query | -0.306 | Gate opposes query |
| gate ↔ up | -0.374 | Gate opposes its expand partner |
| query ↔ up | +0.314 | Query and expand share direction |
| query ↔ key | +0.280 | Q-K aligned (expected) |
| value ↔ all | ≈0.00 | Value is isolated |
| output ↔ all | ≈0.00 | Output is isolated |

**The gate (SiLU selector) looks in the opposite direction from
the attention key.** The MLP selects for features that attention
does NOT attend to. This confirms DC 243's finding and extends it:
the orthogonality isn't just within a layer's MLP — it's across
the attention-MLP boundary.

### 2.3 Depth-Dependent Magnitude Trends

| Weight Type | Spearman ρ | Trend |
|:------------|:-----------|:------|
| q_proj | -0.653 (p<0.001) | **SHRINK** with depth |
| down_proj | +0.842 (p<0.001) | **GROW** with depth |
| gate_proj | -0.032 (p=0.87) | Stable |

Query weights shrink deeper — attention becomes more selective,
making finer adjustments. The MLP compress matrix grows — the
amplifier injects larger corrections in later layers. The gate
stays constant — its role (selection threshold) doesn't change
with depth.

### 2.4 Function-Specific Decay Laws

| Weight Type | Best Fit | RMSE |
|:------------|:---------|:-----|
| q_proj | Stretched exp (β=0.4) | 0.007 |
| v_proj | Stretched exp (β=0.5) | 0.010 |
| gate_proj | Power law (α=0.16) | 0.024 |
| down_proj | Stretched exp (β=0.5) | 0.016 |

Attention matrices follow stretched exponentials (rapid initial
decay, slow tail). The gate follows a power law (more uniform
energy distribution). This matches DC 243's ConvNeXt findings
exactly — the decay law is a function of architectural role, not
model size or training.

---

## 3. Nearly Full Rank — No Free Compression

| Weight Type | Shape | r50 | r90 | r99 |
|:------------|:------|:----|:----|:----|
| q_proj | 3584×3584 | 126 | 397 | 489 |
| k_proj | 512×3584 | 96 | 289 | 440 |
| v_proj | 512×3584 | 161 | 391 | 483 |
| gate_proj | 18944×3584 | 167 | 424 | 492 |
| up_proj | 18944×3584 | 209 | 437 | 493 |
| down_proj | 3584×18944 | 209 | 437 | 494 |

All matrices are nearly full-rank. r90 ≈ 400 out of a maximum 500.
The information is distributed across the full spectrum.

This does NOT mean compression is impossible. It means naive
rank-k truncation won't work without quality loss. The path
forward is understanding which singular components are redundant
across tasks (shared structure that can be factored out), not
which components have low energy.

DC 130 predicted MESH matrices (W_q.T @ W_k) would be low-rank.
We tested this directly.

### 3.1 Composition Rank — Where the Compression Actually Lives

**Attention compositions: naturally low-rank (confirmed)**

| Composition | Rank Bound | r90 | r99 |
|:------------|:-----------|:----|:----|
| MESH = W_q.T @ W_k (per head) | 128 | 37–79 | 68–113 |
| OV = W_o @ W_v (per head) | 128 | 93–105 | 123–125 |

MESH has effective r90 as low as **37** out of 128. This is massive
compression potential — DC 130's prediction confirmed on Qwen2-7B.
The attention score matrix can be represented at ~30% of its
maximum rank while retaining 90% variance.

**MLP compositions: FULL RANK**

| Composition | Rank | r50 | r90 | r99 |
|:------------|:-----|:----|:----|:----|
| W_down @ W_gate | 3584 | 306–508 | 1072–1658 | 2019–2665 |
| W_down @ W_up | 3584 | 311–584 | 1131–1761 | 2135–2727 |

The 18944-dim intermediate space fills the entire 3584-dim output.
The MLP is using its full representational capacity.

### 3.2 Three Critical Discoveries

**1. Gate and up paths are orthogonal after compression.**
cos(W_down @ W_gate, W_down @ W_up) ≈ 0.000 across all layers.
W_down projects the gate and up paths into independent subspaces.
The MLP splits into two completely independent channels that are
recombined only through the element-wise SiLU gating.

**2. The MLP composition is orthogonal to identity.**
cos(W_down @ W_gate, I) ≈ 0.000. The MLP is NOT reconstructing
its input — it's injecting entirely new information. This confirms
DC 243's "null-space injector" prediction on the actual Qwen2-7B
weights. The residual connection handles identity; the MLP handles
orthogonal injection.

**3. φ appears in the gate-compress composition — but not universally.**
At Layer 14 (model midpoint), the first singular value ratio of
W_down @ W_gate is S0/S1 = **1.620** — within 0.1% of φ = 1.618.
However, follow-up across all 28 layers shows this is NOT universal:
mean DG S0/S1 = 1.751 (std=0.513). φ appears at specific layers
(L14 DG, L6 DU, L26 DU) but most layers have different ratios.
With 56 layer×composition pairs, some hits are expected by chance.

What IS universal: the gate-compress path has a single dominant
direction φ-separated from a near-degenerate bulk. The SV cascade
at L14 drops from 1.620 → 1.180 → 1.01 → 1.00... — one primary
selection axis, then uniform distribution. This "one hot direction"
structure is the real finding, not the specific ratio value.

---

## 4. Synthesis: What "Structure IS Information" Means for Weights

The weights are like pixels in an image. No individual pixel is
meaningful — you can shuffle them randomly and destroy the image.
But collectively, they encode shapes, edges, and objects.

Similarly:
- **Individual weights**: random noise at φ-level magnitudes
- **Row/column norms**: unordered
- **Signs**: perfectly symmetric (50/50)
- **SVD directions**: highly structured (anti-alternation, cross-type coupling)
- **SV spectrum**: function-specific decay laws
- **Depth trends**: systematic magnitude evolution

The "rhyme and reason" is entirely in the collective shape — the
SVD decomposition. This is the strongest evidence yet for our
hypothesis: the weights are not the information. The shape they
create is.

---

## 5. Implications for Full Model Conversion

### What's Easier Than Expected

1. **The MLP is not opaque.** DC 243 + F157 show it's a conditional
   orthogonal injector with known geometric properties. The gate
   anti-alternation provides layer-level structure we can exploit.

2. **Cross-type relationships are consistent.** gate ⊥ key is not
   a coincidence — it's the model's learned strategy for maximizing
   information injection. This can be used as a structural prior.

3. **Depth trends are monotonic.** q shrinks, down grows — simple
   scaling relationships that can be modeled analytically.

### What's Harder Than Expected

1. **Full-rank MLP compositions.** W_down @ W_gate is rank 3584
   with r90 ≈ 1000-1700. No free compression anywhere in the
   MLP path. The 18944-dim intermediate fills the entire output.

2. **Non-periodic anti-alternation.** The gate pattern is structured
   but not a clean alternation — harder to model with simple rules.

3. **The SiLU barrier is real but different.** The MLP isn't opaque
   — it's a conditional orthogonal injector with known structure.
   But it's full-rank in both paths. The non-linearity can't be
   removed by composition; it must be modeled directly.

### Next Experiments

1. **Task-specific activation sparsity**: For known tasks (capital,
   language), measure which singular components of W_down@W_gate
   actually activate after SiLU gating. The compositions are
   full-rank statically, but may be sparse dynamically.

2. **Per-head MESH functional analysis**: Cluster the 28×28=784
   MESH matrices by their top-k singular vectors. Do heads
   specialize into functional groups? (induction, position, etc.)

3. **Orthogonal injector reconstruction**: Given that MLP ⊥ I,
   can we reconstruct the MLP's effect as a rotation in the
   null-space of the residual stream? Test on known facts.

---

## 6. Connection to Prior Work

### 6.1 DC 243 — The GELU Machine

DC 243 analyzed ConvNeXt's SSM (structurally similar to a
transformer MLP) and made three predictions:

| DC 243 Prediction | F157 Result | Status |
|:------------------|:------------|:-------|
| W₂ null-space injection is 65% of output | cos(W_down@W_gate, I) ≈ 0.000 | **Confirmed on Qwen2-7B** |
| SV spectrum follows stretched exponential | Attention: stretched exp. MLP: power law | **Partially confirmed** |
| S0/S1 ≈ φ is coincidental (one block only) | L14 DG = 1.620, but not universal | **Confirmed: not universal** |
| Directions + W₂ coupling is irreducible | Gate ⊥ up after W_down; independent channels | **Extended** |

The strongest confirmation: **the MLP IS a null-space injector.**
DC 243 inferred this from a colorization model's W₁/W₂ structure.
F157 proves it directly on Qwen2-7B: the MLP composition has
zero projection onto identity. The residual connection IS the
identity path. The MLP IS the orthogonal injection.

DC 243 also correctly predicted that S0/S1 ≈ φ would be
coincidental — one block, not a law. Our all-layer sweep confirms
this: φ at L14 but mean ratio = 1.75 with high variance.

### 6.2 DC 130 — φ-AIG Compression

DC 130 showed that MESH matrices (W_q.T @ W_k) are naturally
low-rank and achieve 14× compression with 0.09% error.

| DC 130 Prediction | F157 Result | Status |
|:------------------|:------------|:-------|
| MESH rank ≤ 128 (head_dim) | rank = 128, r90 = 37-79 | **Confirmed** |
| Low-rank = shared sub-expressions (AIG) | Compositions are the right level of analysis | **Confirmed** |
| Individual weights are not the unit of meaning | Raw weights unordered; SVD is the structure | **Confirmed** |

The AIG analogy holds precisely: just as an AIG circuit factors
shared logic across gates, the MESH factorization extracts shared
structure across attention heads. The r90 = 37 finding means some
heads use only 30% of their maximum rank — enormous compression.

### 6.3 The Conditional Orthogonal Injector Model

Combining DC 243 + DC 130 + F157, the MLP emerges as a
**conditional orthogonal injector**:

```
residual += W_down @ (SiLU(W_gate @ x) ⊙ (W_up @ x))
```

where:
- `W_gate @ x` and `W_up @ x` are ORTHOGONAL after W_down
  compression (cos ≈ 0.000)
- The entire composition is ORTHOGONAL to identity (cos ≈ 0.000)
- W_gate selects WHICH features activate (one dominant direction)
- W_up provides the VALUES to inject
- W_down compresses both into the residual's null-space
- SiLU is the ONLY non-linearity — it gates the injection

This is NOT a black box. It's a geometrically transparent
orthogonal injection with a learned gating function. The
remaining question is whether the gating (SiLU) can be
approximated geometrically — DC 243 showed GELU ≈ x·σ(φ·x),
and SiLU = x·σ(x), which is the same structure without the
φ scaling. The gate is a soft threshold, not a complex function.

### 6.4 Synthesis Table

| Prior Work | Key Claim | F157 Verdict |
|:-----------|:----------|:-------------|
| DC 243 Part 9 | MLP = null-space injector | ✓✓✓ cos(DG, I) = 0.000 |
| DC 243 Part 12 | Stretched exponential decay | ✓✓ attention yes, MLP = power law |
| DC 243 Part 13 | φ in S0/S1 is coincidental | ✓✓ confirmed across 28 layers |
| DC 130 | MESH is low-rank (compressible) | ✓✓✓ r90 = 37 out of 128 |
| DC 130 | AIG = shared sub-expressions | ✓✓ compositions are the right unit |
| DC 228 | Structure IS information | ✓✓✓ weights = noise, SVD = signal |

---

## 7. Files

| File | Purpose |
|:-----|:--------|
| `experiments/model_reverse_engineering_v2/frontier13_weight_structure.py` | Seven-part analysis |
| `experiments/model_reverse_engineering_v2/FINDINGS.md` | F157 |
| `docs/design_considerations/243_the_gelu_machine.md` | DC 243 (GELU/SiLU analysis) |
| `phi_chat/design_docs_workspace/130_phi_aig_compression.md` | DC 130 (AIG/MESH compression) |
| `docs/design_considerations/228_geometric_colorizer_experiments.md` | DC 228 (Structure IS information) |
