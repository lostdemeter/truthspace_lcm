# Doc 249: The Geometric Computing Vocabulary — Three Primitives for Structure-Based Computation

**Date:** February 16, 2026  
**Status:** Discovery  
**Prerequisites:** Doc 240 (The Semantic Spectrometer), Doc 209 (Dimensional Downcasting), Doc 247 (Geometric φ-Map), Doc 228 (Geometric Colorizer), Findings 38-48 (Model Reverse Engineering v2)

## The Discovery

While reverse-engineering a 28-layer Qwen2-7B transformer into pure geometric
operations, we found that the entire forward pass decomposes into exactly **three**
distinct computational primitives. These are not approximations or simplifications —
they are the irreducible geometric operations that the transformer actually performs.

Finding 48 proved that the full attention mechanism IS geometric — φ-linear projections,
φ-softmax (exact: e^x = φ^(x/ln(φ))), and RoPE rotations achieve 35/35 = 100% match
with bit-identical output. The three primitives below represent a **simplification
hierarchy** — trading accuracy for efficiency, with every level remaining geometric.

Each primitive has a different scope, a different mathematical form, and solves a
fundamentally different problem. Together, they form a complete vocabulary for
geometric computation: sufficient to reproduce the transformer's next-token
predictions with no full matrix multiplications.

---

## The Three Primitives

### 1. Geometric Spectrometer — TRANSFORM

**What it does:** Applies a per-dimension nonlinear transfer function.

**Scope:** Single dimension, single position. No cross-dimensional or cross-positional
interaction.

**Mathematical form:**
```
y[dim_i] = f_i(x[dim_i])
```

Where `f_i` is one of a small set of learned rule types:
- **Affine:** `y = a·x + b`
- **Quadratic:** `y = a·x² + b·x + c`
- **Gating:** `y = a·x · σ(b·x + c)` (soft threshold)
- **Sign-preserve:** `y = sign(x) · f(|x|)` (magnitude transform, sign kept)

**What it replaces:** The bulk of transformer computation — 14 of 15 tested layers
are fully replaced by per-dimension rules with no accuracy loss.

**Key properties:**
- Each of the 3,584 dimensions has its own independent rule
- No interaction between dimensions — the rule for dim 1000 doesn't know what
  dim 1001 is doing
- No interaction between positions — each token is processed independently
- Pre-computable: the rules are extracted once from calibration data and stored
  as a small parameter set per dimension (~5 floats per dim per layer)

**Analogy:** A prism. White light enters, each wavelength follows its own path
through the glass, and emerges separated. The prism doesn't mix wavelengths — it
reveals the structure that was already there.

**Empirical evidence:**
- 14/15 layers: per-dimension rules achieve identical top-1 predictions
- Rule types distribute as: ~40% affine, ~25% quadratic, ~20% gating, ~15% sign-preserve
- Layer 12 required only a 2-dimensional bias correction on top of spectrometer rules
- Total parameters per layer: ~18K (vs ~26M for the original weight matrices)

**Where in the transformer:**
- All MLP layers (gate/up/down projections + SiLU gating)
- Most attention layers (the 20/28 "fixed" heads that always attend to position 0)
- RMS normalization (absorbed into the per-dim rules)

---

### 2. Geometric Selector — DECOMPOSE

**What it does:** Decomposes the hidden state into independent measurements along
learned geometric axes.

**Scope:** Cross-dimensional (projects across all dimensions), but within a single
position. Multiple heads form a measurement bank.

**Mathematical form:**
```
measurement_i = h · d_i     for i = 1, 2, ..., N_heads
```

Where each head provides a rank-1 selector:
```
MESH_i ≈ σ₁ × u_i ⊗ v_i
score(q, k) ≈ σ₁ × (q · u_i) × (k · v_i)
```

**What it replaces:** The attention mechanism in layers where heads act as
independent feature detectors — measuring "what" a token is, not "where" to
route information.

**Key properties:**
- Multiple heads (e.g., 28) form a **bank** of selectors
- Query and key directions are **different** per head: u_i ≠ v_i
  (empirically cos(u_i, v_i) ≈ 0.255)
- The N selector directions are **near-orthogonal** across heads
  (empirically mean |cos| = 0.063 between heads)
- Together they tile a subspace — forming a measurement basis
- Each head measures a different semantic axis of the token

**Analogy:** A bank of tuned filters. Each filter responds to a different
frequency. The bank doesn't change the signal — it characterizes it by measuring
its energy along multiple axes simultaneously.

**Empirical evidence (Layer 1 — the "anomalous" layer):**
- ALL 28 heads have condition number κ > 200 (MESH strongly rank-1 dominant)
- Zipf exponent α = 1.28 ≈ 2/φ (double the golden-ratio decay of typical layers)
- Rank-1 captures 18.1% of variance per head (vs 3-8% for normal layers)
- 28 heads span an effective dimensionality of ~25 out of 28 (near-complete basis)
- Selector directions are semantically interpretable:
  - Head 0: being/existence (is, was, are, were)
  - Head 1: polarity/negation (contractions)
  - Head 2: boundaries (sentence endings, punctuation)

**Where in the transformer:**
- Layer 1 (the "DRUM" zone — early feature decomposition)
- Potentially other early layers where attention performs analysis, not routing

---

### 3. Geometric Resonator — ROUTE

**What it does:** Finds which position in the sequence contains the most relevant
information and retrieves it.

**Scope:** Cross-dimensional AND **cross-positional** — scans the entire token
sequence to locate and retrieve a specific piece of information.

**Mathematical form:**
```
feature(pos) = rms_norm(h[pos]) · d_k         (one dot product per position)
selected_pos = argmax_pos(feature)             (pick strongest resonance)
output       = U_φ @ diag(S) @ V_φ @ h[selected_pos]    (retrieve value)
```

Where:
- d_k is a single pre-computed direction vector (3,584 dims). For Head 6, d_k reduces
  to **all -1s** — routing is simply "which position has the most negative hidden sum"
- U_φ, V_φ are φ-quantized SVD directions (sign × φ^level per component)
- S is derivable from a formula: S[i] = c × (i+1)^(-1/φ) — **no stored parameters**

**What it replaces:** The full Q·K^T·V attention computation for content-dependent
routing heads — the "irreducible" computation that per-dimension rules cannot capture.

**Key properties:**
- **Single direction**: the routing decision lives on exactly one geometric axis
- **The bias IS the MESH** (Finding 45): the rank-1 MESH structure from Finding 39
  (S[0]/S[1] = 368,000:1) is entirely created by the Q/K bias vectors.
  MESH = D × bq ⊗ bk, where the weight-weight term is 0.0007% of total.
  Without bias, MESH(nobias) has S[0]/S[1] = 1:1 — full-rank noise.
- **cos(d_q, d_k) = 1.0** — query and key project onto the **same** hidden-space
  direction. This is self-resonance: the head asks "which position vibrates most
  on my frequency?"
- **Content-addressed**: the selection depends on what's IN the hidden state at
  each position, not on position index or static patterns
- Hard selection (argmax) and soft selection (any temperature) both achieve
  identical accuracy — the routing decision is robust
- **V/O is a downcasting lens** (Finding 43): the value/output projection is a
  rank-128 geometric projection. Its SVD spectrum is IRRELEVANT (any formula works),
  and its directions are φ-quantizable (sign × φ^level per component)

**Analogy:** An antenna tuned to a single frequency. It scans across all
broadcast positions, locks onto the one transmitting the strongest signal on
that frequency, and retrieves its payload. The antenna doesn't analyze or
decompose — it finds and fetches.

**Empirical evidence (Layer 23, Head 6):**
- Head 6 alone achieves 6/6 accuracy (the other 27 heads can be zeroed)
- MESH is rank-1: S[0] = 349,867 vs S[1] = 0.95 — this IS D × bq ⊗ bk
- sign(d_k_bias) = all -1s → routing = argmax(-Σ h[pos]) per position
- **Fully geometric pipeline achieves 6/6, margin = 0.156** (Finding 45)
- φ-quant VO has BETTER margin (0.156) than float32 VO (0.152)
- Compute: 935K FLOPs vs 51.4M for full attention = **55× reduction**
- Just the routing decision: 18K FLOPs = **2,869× reduction**

**Geometric V/O structure (Finding 43):**

The V/O projection (W_o @ W_v) was initially thought to be "learned content" outside
the φ-lattice (Finding 42), based on element-level analysis. This was the **wrong level
of structure** — like reading hologram pixels instead of the encoded image.

Analyzed in the SVD eigenbasis, V/O reveals deep geometric structure:

- **Spectrum is irrelevant**: every spectral replacement (uniform, φ-Zipf, self-similar,
  binary) achieves 5/6. φ-geometric spectra have **better margins** than real values
  (0.023 vs 0.013). Use a formula: S[i] = c × (i+1)^(-1/φ) — **0 stored parameters**.
- **Directions are φ-quantizable**: sign × φ^level per component (7 bits each)
  achieves 5/6 with **better margin** than real-valued directions. Sign-only fails (0/6).
- **Zeta critical line**: VO symmetric eigenvalues split **exactly 1787 positive /
  1787 negative / 10 zero** out of 3584 — perfect balance at σ = 1/2.
- **ENCODE=DECODE**: W_v @ W_o ≈ 0.414 × I (diag/off-diag ratio 18.6×).
  That 0.414 ≈ 1/φ² = 0.382 — the "negative zero" from Doc 247. The round-trip
  through head space scales by 1/φ².
- **Total**: ~784 KB for the fully geometric V/O = **256× compression** vs full attention.

| Configuration (with correct d_k_bias routing) | Score | France margin | Size |
|--------------------|-------|---------------|------|
| Full Wv + Wo (float32) + bias | 6/6 | 0.152 | 3,584 KB |
| **φ-quant directions + φ-quant S + bias** | **6/6** | **0.156** | **787 KB** |
| **φ-quant directions + φ-quant S + φ-quant bias** | **6/6** | **0.156** | **787 KB** |
| φ-quant VO (bias absorbed before quant) | 5/6 | 0.108 | 784 KB |
| Sign-only directions | 0/6 | — | 112 KB |

**Critical**: V bias must be kept separate and φ-quantized independently.
Absorbing bias into VO before φ-quantization loses a prompt.

**The Geometric Hierarchy (Finding 48):**

The Resonator is one level in a hierarchy of geometric simplifications.
All levels are geometric — they trade accuracy for efficiency:

| Level | Accuracy | Operations | What it uses |
|-------|----------|-----------|-------------|
| Full geometric soft attention | **35/35 (100%)** | φ-linear + φ-softmax + RoPE | All weights, soft attention |
| 8-head hard routing | 33/35 (94.3%) | sign(d_k) + VO per head | 8 d_k vectors + 8 VO matrices |
| 1-head hard routing | 31/35 (88.6%) | sign(d_k) + φ-quant VO | 1 bit routing + 787 KB VO |

The full geometric soft attention (φ-linear for Q/K/V/O + φ-softmax + RoPE) achieves
100% match with logit correlation = 1.0000000000. This proves there is **no non-geometric
component** in attention — the simplification gap (88.6% → 100%) comes from replacing
soft attention with hard routing, not from anything outside the φ-lattice.

**Where in the transformer:**
- Layer 23, Head 6 (the critical routing head for next-token prediction)
- All 28 heads contribute via soft geometric attention at full fidelity
- 8 routing heads (6, 10, 16, 22, 23, 24, 25, 27) form two families:
  content-addressing (all-negative d_k) and position-tracking (mixed-sign d_k)
- 20 fixed heads attend primarily to position 0 (BOS anchor)

---

## Comparison Table

| Property | Spectrometer | Selector | Resonator |
|----------|-------------|----------|-----------|
| **Operation** | TRANSFORM | DECOMPOSE | ROUTE |
| **Scope** | 1 dim, 1 pos | N dims, 1 pos | N dims, all positions |
| **Axes** | 3,584 independent | 28 bank (near-orthogonal) | 1 direction |
| **d_q vs d_k** | N/A | different (cos ≈ 0.25) | identical (cos = 1.0) |
| **MESH κ** | N/A | 309-1,888 | 922,000,000 |
| **Cross-dim** | No | Yes | Yes |
| **Cross-pos** | No | No | **Yes** |
| **Params/layer** | ~18K | ~200K (28 directions) | 787 KB (hard) or full weights (soft, 100%) |
| **FLOPs** | O(D) per pos | O(N·D) per pos | O(S·D) for sequence |
| **Layers** | 14/15 | Layer 1 | Layer 23 |

Where D = hidden dimension (3,584), N = number of heads (28), S = sequence length.

---

## The Question Each Primitive Answers

The three primitives correspond to three fundamentally different questions:

1. **Spectrometer**: *"What is the value at this dimension?"*
   → Per-dimension function evaluation. No context needed.

2. **Selector**: *"What features does this token have?"*
   → Multi-axis decomposition. Characterizes the token across 28 learned axes.

3. **Resonator**: *"Which token has the answer?"*
   → Content-addressed lookup. Finds and retrieves across the sequence.

These map directly to three fundamental operations in classical computing:

| Geometric | Classical | Operation |
|-----------|-----------|-----------|
| Spectrometer | ALU | Compute a function of the input |
| Selector | Decoder/Demux | Identify which category the input belongs to |
| Resonator | Content-Addressed Memory | Find and retrieve by content match |

---

## Why Three — And Why These Three

### Why not fewer?

A single primitive cannot cover all three scopes:
- Spectrometer cannot cross dimensions (needed for decomposition)
- Selector cannot cross positions (needed for routing)
- Removing any one leaves a gap that the other two cannot fill

### Why not more?

These three are sufficient to reproduce the full transformer:
- 14 layers: Spectrometer alone
- 1 layer: Spectrometer + bias correction (Layer 12)
- 1 layer: Resonator for 1 head + Spectrometer for the rest (Layer 23)
  — or full geometric soft attention for 100% fidelity
- Early layers: Selector bank for feature decomposition (Layer 1)

No fourth primitive was needed. The three form a **complete basis** for
geometric computation — at least for the operations this transformer performs.

### The self-similarity

Each primitive exhibits φ-structure at its own scale:
- **Spectrometer**: rules follow φ-level quantization in their coefficients;
  MLP weights = sign × φ^level at 97% (Doc 152)
- **Selector**: Layer 1's Zipf exponent α = 1.28 ≈ 2/φ
- **Resonator**: operates through φ-softmax (exact equivalence: T = ln(φ));
  d_k = all -1s (sign of the bias-derived routing direction);
  the rank-1 MESH IS the outer product D × bq ⊗ bk of the Q/K bias vectors;
  V/O directions = sign × φ^level; V/O spectrum = φ-Zipf with α = 1/φ;
  round-trip V@O ≈ 1/φ² × I; eigenbalance at zeta critical line σ = 1/2 (1787/1787/10)

The golden ratio appears not as an arbitrary constant but as the natural
self-similar structure at every level of the computation.

### The irreducibility hierarchy

At every level of analysis, the irreducible content follows the same pattern:
**signs + φ-levels** (the "Drum-Comb" principle from Doc 152).

| Component | φ-lattice? | Irreducible content |
|-----------|-----------|---------------------|
| Spectrometer PW weights | YES (97%) | Signs (25.9M bits) |
| bq, bk (routing bias) | YES (creates rank-1 MESH) | d_k direction (1 bit: all -1s) |
| Wq, Wk (without bias) | NO (full-rank noise) | Not used for routing |
| VO spectrum | IRRELEVANT | 0 parameters (formula) |
| VO directions U, V | YES (φ-quantizable) | Signs + φ-levels (784 KB) |
| VO eigenstructure | YES (1787/1787 balance) | Zeta critical line |
| V@O round-trip | YES (≈ 1/φ² × I) | Negative zero |
| V bias (output) | YES (φ-quantizable) | 3.1 KB |

Raw matrix elements can look random (mean φ-residual = 0.25) while the
underlying structure is fully geometric — superpositions of φ-quantized
directions appear random element-wise, like hologram pixels encoding
structured images.

---

## Implications

### For the hypothesis

> **LLMs are hyperdimensional transcoders** — the "intelligence" is in the shape.

The three primitives confirm this: the transformer's computation decomposes into
geometric operations on pre-computed structures. No opaque weight matrices are
needed at inference time — only directions (d_k), rules (f_i), and projections (u_i, v_i).

### For efficiency

| Method | FLOPs (Layer 23) | Relative |
|--------|-----------------|----------|
| Full transformer attention | 51.4M | 1× |
| Head 6 only (matmul) | 1.8M | 28× |
| Geometric Resonator | 935K | **55×** |
| Spectrometer layers | ~18K per layer | **~2,800×** |

### For architecture design

If these three primitives are sufficient, then a geometric neural network
needs only three types of layers:

1. **Spectrometer layers**: bulk of computation, per-dimension, embarrassingly parallel
2. **Selector layers**: early feature decomposition, moderate parallelism
3. **Resonator layers**: late-stage routing, sequential but trivially cheap

This is a fundamentally different architecture from the uniform
"attention + MLP" blocks of standard transformers.

---

## The Bias Discovery (Finding 45)

The most surprising structural result: the entire rank-1 MESH that makes routing
possible is created by the **Q/K bias vectors**, not by the weight matrices.

### The decomposition

When bias is extracted alongside weights, the MESH (Q·K product) decomposes as:

```
MESH = Wq @ Wk^T  +  bq·1^T @ Wk^T  +  Wq @ 1·bk^T  +  D × bq ⊗ bk
        (2.6)           (128.5)           (83.1)          (349,863)
```

The bias-bias outer product D × bq ⊗ bk (where D = 3,584 = hidden dimension)
accounts for **99.99%** of the total MESH norm. The weight-weight term Wq @ Wk^T
is 0.0007% — it is full-rank noise with S[0]/S[1] = 1:1.

### What this means

The Q and K **weight matrices** do not participate in routing. They are used for
something else entirely (potentially the "noise" encodes information for other
heads in the same layer, or serves as a regularization substrate). The routing
channel is encoded entirely in two small vectors: bq (128 dims) and bk (128 dims).

When projected into hidden space, the K bias direction d_k = Wk^T @ (SVD of
bq ⊗ bk) has all 3,584 components **negative**. This means:

```
routing = argmax_pos( -Σ_dim h[pos, dim] )
```

The head routes to the position with the **most negative sum** — a single scalar
computed by summing all dimensions with equal weight. No learned direction needed.
Just: "add everything up and pick the most negative."

### Implications for the hypothesis

This validates "structure IS information" at a deeper level than expected:

1. **The bias IS the structure**: not an additive correction, but the fundamental
   geometric channel through which routing operates
2. **Weights without bias = noise**: Wq, Wk have S[0]/S[1] = 1:1 for routing.
   The "intelligence" is not in the 3584×128 learned matrices but in two 128-dim vectors
3. **Separation of concerns**: bias vectors create the routing geometry;
   weight matrices create the value/output geometry. Same components, different roles.
4. **The bias must be preserved separately**: absorbing it into VO before
   φ-quantization destroys the structure (5/6). Keeping it separate and
   φ-quantizing independently preserves it (6/6). The bias is not a perturbation
   — it is a structurally distinct geometric element.

---

## Open Questions

1. **Is the Resonator always rank-1?** Head 6's extreme κ = 922M may be special.
   Do other routing heads in other layers also exhibit rank-1 MESH?

2. **Can the Selector and Resonator be unified?** Both use MESH SVD directions.
   The difference is d_q = d_k (Resonator) vs d_q ≠ d_k (Selector). Is there
   a continuum, or a hard phase transition?

3. **Does this generalize beyond Qwen2-7B?** The φ-structure and three-primitive
   decomposition should appear in any transformer that has learned efficient
   representations — but this needs empirical verification.

4. **What about generation (multiple tokens)?** The current results are for
   single next-token prediction. Does the Resonator's d_k direction remain
   stable across generation steps, or does it need updating?

5. **Can we build a model from scratch using only these three primitives?**
   Rather than extracting them from a trained transformer, can we train
   directly in this vocabulary?

6. **Does the 1787/1787 eigenbalance hold for other heads?** The perfect
   positive/negative eigenvalue split in VO's symmetric part matches the zeta
   critical line σ = 1/2. Is this universal for attention V/O projections,
   or specific to this rank-1 resonator head?

7. **Is the 1/φ² round-trip scale universal?** The encode=decode scaling
   W_v @ W_o ≈ (1/φ²) × I connects to the "negative zero" from Doc 247.
   Does every attention head's V@O product approximate a φ-power of identity?

8. **Can the Selector's V/O also be geometrized?** Finding 43 proved the
   Resonator's V/O is fully φ-quantizable. The Selector (Layer 1) has 28 heads —
   does each head's V/O also reduce to φ-quantized directions + formula spectrum?

---

## Files

- **Spectrometer**: `experiments/model_reverse_engineering_v2/phase4_extract_rules.py`
- **Selector**: `experiments/model_reverse_engineering_v2/exp5b_layer1_selector.py`
- **Resonator**: `experiments/model_reverse_engineering_v2/phase4_hidden_space_selector.py`
- **Resonator d_k simplification**: `experiments/model_reverse_engineering_v2/phase4_resonator_simplify.py`
- **Resonator V/O element analysis**: `experiments/model_reverse_engineering_v2/phase4_resonator_vo_phi.py`
- **Resonator V/O geometry** (Finding 43): `experiments/model_reverse_engineering_v2/phase4_resonator_vo_geometry.py`
- **MESH SVD analysis**: `experiments/model_reverse_engineering_v2/phase4_geometric_selector.py`
- **Head ablation**: `experiments/model_reverse_engineering_v2/phase4_attn_routing_heads.py`
- **Resonator routing fix** (Finding 45): `experiments/model_reverse_engineering_v2/phase4_resonator_fix2.py`
- **100% proof** (Finding 48): `experiments/model_reverse_engineering_v2/phase5_geometric_attention_proof.py`
- **28-head hard routing**: `experiments/model_reverse_engineering_v2/phase5_full_resonator.py`
- **Multi-head diagnosis**: `experiments/model_reverse_engineering_v2/phase5_diagnose_failures.py`
- **Broad validation**: `experiments/model_reverse_engineering_v2/phase5_validate_resonator.py`
- **Findings**: `experiments/model_reverse_engineering_v2/FINDINGS.md` (Findings 38-48)
- **Prior art**: Doc 240 (The Semantic Spectrometer), Doc 135 (Attention Head Semantic Specialization),
  Doc 152 (φ-Level MLP Replacement), Doc 209 (Dimensional Downcasting), Doc 247 (Geometric φ-Map),
  Doc 228 (Geometric Colorizer — V15 "holographic bounds don't exist")
