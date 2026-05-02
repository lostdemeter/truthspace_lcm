# Doc 276: A Taxonomy of Geometric Structures in Transformers

**Date:** February 28, 2026 | **Updated:** March 15, 2026
**Status:** Synthesis of Findings 39–126 + Day 41 (Expedition)
**Prerequisites:** DC 240 (Spectrometer), DC 249 (Resonator), DC 260 (Shadow Orbit/Gyroscope), DC 275 (Knowledge Extension), F40 (Selector), F45 (Resonator), F96 (Gyroscope), F122–126 (Lens, Aperture, Amplifier)

---

## 1. Overview

Across 126 findings, 25+ design considerations, and expedition days 39–41, we have identified
**eight** distinct geometric structures. Seven were characterised in Qwen2-7B; the eighth was found in Qwen2-1.5B and reveals that the routing function can be implemented via fundamentally different geometry across model architectures. Each structure has a
characteristic dimensionality, a physics analogy, and a specific
computational role.

This document formalizes the taxonomy and proposes that these
structures are not independent — they compose into a coherent pipeline
where each structure's output feeds the next. They combine via
orthogonal direct sum (⊕), not multiplicative product.

---

## 2. The Eight Geometric Structures

### 2.1 The Geometric Gyroscope

**Discovery:** Finding 96, DC 260
**Location:** Residual stream (all layers)
**Physics analogy:** A spinning gyroscope that settles to a fixed
precession angle regardless of how it was pushed.

**What it does:** When the residual stream is perturbed (e.g., by
replacing exact attention with approximate attention), the trajectory
does not diverge. It settles into a **stable displaced orbit** at a
fixed angular displacement from the true trajectory.

**Key properties:**
- Steady-state angle: 68.4° ≈ arccos(1/φ²) at the prediction-relevant
  last position
- Drift ratio ||ε||/||h|| ≈ 1.30 (prompt-independent)
- Underdamped settling: ~15 layers (half the network depth)
- The orbit is a φ-constant — the golden ratio governs the geometry

**Characteristic dimensionality:** 1 (a single angle)

**Role in pipeline:** INFRASTRUCTURE — maintains the highway. The
residual stream is self-correcting. Errors don't accumulate
catastrophically; they settle into a geometrically predictable orbit.
This is why approximate computation works.

---

### 2.2 The Geometric Spectrometer

**Discovery:** Findings 27–38, DC 240
**Location:** COMB layers (6–22)
**Physics analogy:** An optical spectrometer that decomposes white
light into spectral lines — each dimension is a separate channel.

**What it does:** Decomposes the hidden state into per-dimension
components using sign rules. Each dimension independently follows
a predictable pattern across layers (standing wave), enabling
96.4% of layer computation to be predicted from the channel-mode
structure.

**Key properties:**
- 96.4% of gate states predictable from standing wave (F62)
- L and R chirality channels 98.5% independent (F62)
- Per-dimension sign rules: each of 3584 dimensions has a simple rule
- 13/15 COMB layers pass with sign rules alone

**Characteristic dimensionality:** 3584 independent 1-d channels

**Role in pipeline:** INFRASTRUCTURE — decomposes the signal. The
Spectrometer reads the hidden state's spectral content dimension by
dimension, enabling efficient per-channel processing throughout the
COMB layers. It is the substrate on which everything else operates.

---

### 2.3 The Geometric Selector

**Discovery:** Finding 40, DC 249
**Location:** L23 H6 (QK mechanism)
**Physics analogy:** A polarization filter that passes or blocks based
on a single alignment axis.

**What it does:** Routes attention to the correct token position using
a single hidden-space direction d_k. The query and key project onto
the SAME direction (cos(d_q, d_k) = 1.0000), making Head 6 a
"same-feature detector" — it selects whichever token position has
the highest projection onto this shared axis.

**Key properties:**
- cos(d_q, d_k) = 1.0000 — Q and K align perfectly
- 28/28 layers: ONE_AXIS pattern, all routing heads share one d_k (F116)
- Hard selector (argmax) achieves 6/6, margin = 0.152
- Compute: 18K FLOPs for selector decision (2,869× reduction vs full attention)

**Characteristic dimensionality:** 1 (a single direction)

**Role in pipeline:** ROUTING — points the Lens. The Selector
determines which token position contains the entity whose identity
should be extracted. It is the "aiming mechanism" that precedes the
Lens.

---

### 2.4 The Geometric Resonator

**Discovery:** Findings 44–45, DC 249
**Location:** L23 H6 (QK bias structure)
**Physics analogy:** A tuned resonant cavity that amplifies a single
frequency while attenuating all others.

**What it does:** Creates the rank-1 score structure that enables the
Selector. The bias terms dominate the QK interaction by 42–72×,
producing a perfectly rank-1 MESH (S[0]/S[1] = 368,000:1) from the
outer product of query and key biases.

**Key properties:**
- MESH(bias): S[0]/S[1] = 368,000:1 (perfectly rank-1)
- MESH(nobias): S[0]/S[1] = 1:1 (full-rank noise)
- MESH ≈ D × b_q ⊗ b_k (bias-bias outer product = 99.99%)
- Weight-weight term: 0.0007% of total
- Bias must be kept separate (not absorbed before φ-quantization)

**Characteristic dimensionality:** 1 (rank-1 outer product)

**Role in pipeline:** AMPLIFICATION — tunes the Selector. The
Resonator creates the overwhelmingly rank-1 score matrix that makes
the Selector's binary choice clean and unambiguous. Without the
Resonator, the Selector would see a full-rank noise matrix.

---

### 2.5 The Geometric Lens

**Discovery:** Findings 122–124, DC 275
**Location:** L22–L23 (V·W_o mechanism)
**Physics analogy:** A converging lens that focuses light from any
source through its aperture, forming a faithful image.

**What it does:** Projects the entity's 3584-d hidden state through
a ~66-dimensional aperture (the effective rank of M_h = W_v_h^T @
W_o_h^T), producing the entity's full semantic identity cluster —
across languages, across fact types, at the appropriate hierarchical
level.

**Key properties:**
- M_h is a near-isometry: S[0]/S[1] ≈ 1.1, singular values nearly uniform
- Combined rank: 66 (W_v@W_o inner); individual W_o rank@90% ≈ 104
- Bias negligible (3–6% of signal) — unlike the Resonator
- Generalizes to unseen entities: 10/12 new countries at rank 2–10
- Fact-type agnostic: same M_h produces capitals from capital prompts,
  languages from language prompts — the question is in the INPUT
- L22 preserves structure (r = 0.75), L23 transforms it (r = 0.27)
- Obscure entities degrade gracefully → regional identity
- Sharp phase transition: only ~10 SVD dims needed for answers (F125)
- Answer tokens 87% orthogonal to M_h's column space

**Characteristic dimensionality:** ~10 (answers) / ~66 (identity) / 128 (bottleneck)

**Role in pipeline:** KNOWLEDGE — extracts identity. The Lens is where
the actual knowledge lives. It is a single geometric transformation
that maps the entire entity space to the identity space simultaneously.
All facts of all types about all entities are encoded in ONE matrix.

---

### 2.6 The Geometric Amplifier

**Discovery:** Finding 126, Phase 10z25
**Location:** MLP at every layer (dominant at L22–L27)
**Physics analogy:** A laser amplifier that coherently boosts a
specific signal direction while operating orthogonally to the input.

**What it does:** Reads the post-attention residual stream and
amplifies the answer signal direction. At L23, the MLP doubles the
answer signal projection (10.2 → 20.5). MLPs at L24–L27 continue
boosting, pushing the projection from 20 to ~47.

**Key properties:**
- ||Δmlp|| / ||Δattn|| = 2.1–5.3× (MLP changes dominate attention)
- cos(Δmlp, Δtotal) = 0.90–0.98 (MLP IS the layer's net change)
- cos(Δattn, Δmlp) ≈ 0 (operates **orthogonally** to attention)
- Architecture: SiLU(gate) ⊙ up → 18944-d intermediate → 3584-d output
- No biases (unlike Resonator) — structure is in weights
- All 6 test countries reach rank 0–3 by L23 post-MLP

**Characteristic dimensionality:** 18944 (intermediate) → 3584 (output)

**Role in pipeline:** AMPLIFICATION — boosts signal. The Amplifier
takes the weak answer signal extracted by the Lens (only 13% of
answer token energy, per F125) and coherently amplifies it until it
dominates the residual stream. Without the Amplifier, the Lens output
would produce rank 4–18 answers, not rank 0.

---

### 2.7 The Geometric Content Separator

**Discovery:** Finding 145, DC 279 §10 (COMB zone anatomy, Qwen2-7B)
**Location:** L10–L20 (COMB zone, MLP+attention interaction)
**Physics analogy:** An interference spectrometer that separates wavelengths
through controlled phase opposition — two beams push in opposite directions,
producing separation where they cancel and reinforcement where they align.

**What it does:** In the COMB zone (L10–L20), the MLP and attention operate
in **anti-correlated** directions, unlike the orthogonal Amplifier at L22–27.
Attention carries the structural scaffold (shared across all prompts of the
same template); MLP carries content-specific refinement (orthogonal between
prompts of different content). The push-pull interference separates different
content types into different subspaces.

**Key properties:**
- cos(Δattn, Δmlp) ≈ −0.1 to −0.36 (COMB zone) vs ≈0 (extraction zone)
- Attention: cross-structure cos = 0.3–0.75 (“same template scaffold”)
- MLP: cross-structure cos = 0.01–0.19 (“completely content-specific”)
- Same-structure convergence: same=0.901–0.989, cross=0.026–0.578
- PRESERVE channel hourglass: L5=20, L10=6620, L15=9279 (peak), L20=6275, L23=2084
- Same-structure Jaccard overlap: 0.60–0.76; cross-structure: 0.25–0.37
- Gyroscope is STRONGEST here: cos(h_in, h_out) = 0.95, std = 0.013

**Characteristic dimensionality:** Distributed (operates across all 18,944
MLP intermediate channels via gate selection)

**Role in pipeline:** SEPARATION — differentiates content types. The Content
Separator routes different prompt structures into different subspaces through
gate-mediated channel selection combined with push-pull interference. This is
self-similar with the channel-level push-pull from DC 253 §4: the same
interference principle operates at both the channel level (PRESERVE vs
CONTRACT, positive vs negative fringes) and the layer level (attention vs MLP,
structural vs content-specific).

**Contrast with Amplifier:** The Amplifier (L22–27) operates orthogonally to
attention (cos ≈ 0) and coherently boosts a specific signal. The Content
Separator operates anti-correlated to attention (cos ≈ −0.2) and creates
interference that differentiates content. Same MLP architecture, opposite
composition mode.

---

### 2.8 The Semantic Completeness Gate

**Discovery:** Expedition Day 41 (Chinese aspect routing, Qwen2-1.5B)
**Note:** DC 279 §10.5 counted the Content Separator as the "seventh structure". This is the eighth.
**Location:** L23 H01/H02 (QK attention weights)
**Physics analogy:** A phase detector that reads whether a received signal is
complete or still awaiting more input before it can be decoded.

**What it does:** At Layer 23, specific attention heads (H01/H02) test whether
a multi-token word's last piece has fully absorbed the semantic content of the
first piece. When the content is complete (semantically anchored), the head
drops backward attention and the token is promoted to the appropriate Zone C
semantic body. When the content is incomplete (phonemic fragment), the head
maintains backward attention and the token stays in the undifferentiated B001 zone.

**Key properties:**
- H01 attention (着→走 in 走着): 0.31 — drops (semantic completeness confirmed)
- H01 attention (inging→s in singing): 0.96 — holds (still phonemic fragment)
- Chinese/English cross-lingual correlation: 0.921 at L10, **0.029 at L23**, 0.958 at L27
- Layer 23 is the exact and complete divergence point
- No rank-1 MESH: eff_rank = 99–122, sq_ratio ≤ 1.35 (distributed, not narrow-beam)
- Architecture: GQA (2 KV heads, 12 Q heads) — structurally different from Qwen2-7B

**Characteristic dimensionality:** Distributed (no rank-1 structure)

**Role in pipeline:** ROUTING — tests semantic completion. The Completeness Gate
serves the same routing function as the Selector+Resonator pair in Qwen2-7B, but
via a distributed collective-head mechanism instead of a rank-1 MESH. It is the
evidence that the routing function is architecturally necessary but geometrically
free — training discovers whichever implementation is compatible with the model's
attention structure.

**Model specificity:** Found in Qwen2-1.5B. NOT present in Qwen2-7B (which uses
the Selector+Resonator pair instead). The two models implement the same routing
function via incompatible geometric mechanisms:
- Qwen2-7B: rank-1 MESH (368,000:1), single head, 1-bit direction
- Qwen2-1.5B: full-rank distributed gate (eff_rank 99–122), H01/H02 at L23

---

## 3. The Composition Pipeline

The core six Qwen2-7B structures compose into a complete computational pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE GEOMETRIC PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  L0-L5:   SPECTROMETER initializes per-dimension channels        │
│           (3584 independent 1-d sign rules)                      │
│                                                                  │
│  L6-L22:  SPECTROMETER + GYROSCOPE maintain stable channels      │
│           (96.4% predictable, errors → stable orbit)             │
│                                                                  │
│  L22:     LENS (preserve mode) — L22 H15/H19                    │
│           Carries entity identity forward (r = 0.67-0.75)        │
│                                                                  │
│  L23:     SELECTOR finds entity (d_k direction) → Head 6           │
│           RESONATOR makes selection clean (bias rank-1)             │
│           LENS extracts full identity (∞ fact types, r = 0.27)     │
│           AMPLIFIER doubles answer signal (10.2 → 20.5)           │
│                                                                  │
│  L24-L27: AMPLIFIER continues boosting (20 → 47)                  │
│           Answer stable at rank 0                                 │
│                                                                  │
│  OUTPUT:  LM HEAD reads the amplified answer                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 The Critical Layer: L23

Layer 23 is remarkable in concentrating four of the six structures:
- **Selector** (QK): routes attention to entity token
- **Resonator** (QK bias): creates rank-1 score matrix
- **Lens** (V·W_o): extracts entity identity
- **Amplifier** (MLP): amplifies answer signal

These four structures have fundamentally different geometries:
- Selector: rank 1, single direction
- Resonator: rank 1, outer product
- Lens: rank 66, near-isometric
- Amplifier: rank 18944, orthogonal to attention

Yet they compose seamlessly: the Resonator tunes the Selector, which
aims the Lens. The Amplifier boosts the Lens output.

### 3.2 The Two Modes of the Lens

The Lens operates in two modes across L22 and L23:

| Layer | Mode | Structure preservation | Role |
|:------|:-----|:----------------------|:-----|
| L22 | **Preserve** | r = 0.67–0.75 | Carry entity identity forward |
| L23 | **Transform** | r = 0.27 | Rotate entity → answer direction |

This preserve-then-transform pattern is analogous to a two-lens
optical system: the first lens collimates (preserves parallel rays),
the second lens focuses (maps to the focal plane).

---

## 4. Dimensionality Spectrum

The six structures span a striking range of dimensionalities:

```
Dimensionality:    1          1          1          66        18944       3584
Structure:     Gyroscope  Selector  Resonator   Lens    Amplifier  Spectrometer
               (angle)    (direction) (rank-1)  (subspace) (intermediate) (per-dim)
                  │          │          │          │           │           │
Physics:       attractor   filter    cavity      lens     laser amp   spectrum
Role:          stabilize   route     amplify    extract   boost     decompose
Category:      INFRA       ROUTE     ROUTE     KNOWLEDGE  AMPLIFY    INFRA
```

This is not accidental. The structures form a dimensional hierarchy:
- **d=1 structures** (Gyroscope, Selector, Resonator) handle binary
  decisions and stability — they collapse high-dimensional information
  to a single scalar or direction.
- **d=10/66 structure** (Lens) is the knowledge bottleneck — ~10
  dimensions carry the answer signal, ~66 carry entity identity.
  The aperture is architectural (128-d head → 104 per projection →
  66 combined), not determined by the number of facts stored.
- **d=18944 structure** (Amplifier) operates in an expanded intermediate
  space, boosting the answer signal orthogonally to attention.
- **d=3584 structure** (Spectrometer) operates on the full hidden
  space, treating each dimension independently.

The knowledge lives at the intermediate scale (d≈66). Amplification
at d=18944. Infrastructure operates at the extremes (d=1 and d=3584).

---

## 5. Contrast Table

| Property | Gyroscope | Spectrometer | Selector | Resonator | Lens | Amplifier | **Content Separator** | **Completeness Gate** |
|:---------|:----------|:-------------|:---------|:----------|:-----|:----------|:----------------------|:---------------------|
| **Rank** | 1 (angle) | 3584 (per-dim) | 1 (direction) | 1 (outer product) | 10/66 | 18944→3584 | **Distributed** | **Distributed (99–122)** |
| **Bias role** | N/A | N/A | N/A | IS the structure (99.99%) | Negligible | None | **Gate weights** | **N/A** |
| **Linearity** | Nonlinear | Linear | Linear | Linear | Linear | Nonlinear (SiLU) | **Nonlinear (gate+interference)** | **Nonlinear (gate)** |
| **Key constant** | 1/φ² | Standing wave | cos=1.0 | 368,000:1 | S[0]/S[1]≈1.1 | cos(Δattn,Δmlp)≈0 | **cos(Δattn,Δmlp)≈−0.2** | **corr(ZH,EN)=0.029 at L23** |
| **Failure mode** | N/A | L23 irreducible | N/A | N/A | Obscure → regional | N/A | **Unknown** | **Unknown** |
| **Layers** | All | L0–22 | L23 | L23 | L22–23 | L23–27 | **L10–20** | **L23** |
| **Model** | Qwen2-7B | Qwen2-7B | Qwen2-7B | Qwen2-7B | Qwen2-7B | Qwen2-7B | **Qwen2-7B** | **Qwen2-1.5B** |

---

## 6. The Connection to TruthSpace

Each geometric structure contributes to the TruthSpace hypothesis:

1. **Gyroscope** proves the geometry is **self-correcting** — the
   shape maintains itself under perturbation. Structure IS stable.

2. **Spectrometer** proves the geometry is **decomposable** — the
   3584 dimensions carry independent information channels. Structure
   IS information.

3. **Selector** proves the geometry is **navigable** — a single
   direction suffices to route attention. Structure IS computation.

4. **Resonator** proves the geometry is **self-amplifying** — the
   bias outer product creates overwhelming rank-1 structure from
   minimal parameters. Structure IS its own amplifier.

5. **Lens** proves the geometry IS **knowledge** — a single
   near-isometric transformation maps all entities to their full
   semantic identity across languages and fact types. Structure IS
   the knowledge.

6. **Amplifier** proves the geometry IS **coherent** — the MLP
   amplifies answer signals orthogonally to attention, boosting the
   13% alignment (F125) to dominance. Structure IS its own amplifier.

Together, these eight structures demonstrate that the transformer's
computational power emerges entirely from geometric relationships in
weight space. There are no opaque learned features — there are
gyroscopes, spectrometers, selectors, resonators, lenses, amplifiers,
content separators, and completeness gates, each with a well-defined
geometric character and computational role.

---

## 7. Open Questions (Status)

1. ~~**Are there more structures?**~~ **PARTIALLY RESOLVED:**
   - F126 (Amplifier, DC 276) = 6th structure in Qwen2-7B
   - F145 (Content Separator, DC 279 §10.5) = 7th structure in Qwen2-7B
   - Expedition Day 41 (Completeness Gate) = 8th structure in Qwen2-1.5B
   It is unknown whether additional structures exist in other architectures.

2. ~~**Cross-layer structure:**~~ **RESOLVED (F126):** Structures do
   NOT persist across layers. cos(d_k L22, d_k L23) = 0.095 for H6.
   Lens SVD subspace angle = 76.6°. Each layer constructs its own
   structures independently. Only the Gyroscope (dynamical attractor)
   operates cross-layer.

3. ~~**The 66-d aperture:**~~ **RESOLVED (F125):** 66 is an
   architectural constant from cascading two rank-104 projections
   through a 128-d bottleneck. Only ~10 dims are critical for
   answers; 10-66 carry identity; 66-128 contribute nothing.

4. ~~**Composition algebra:**~~ **RESOLVED (F126):** The algebra is
   ⊕ (orthogonal direct sum), not × (product). Head outputs are
   nearly orthogonal (mean cos = 0.006). Attention ⊥ MLP at every
   layer. Structures compose by additive superposition in
   approximately orthogonal subspaces.

5. ~~**Other heads:**~~ **RESOLVED (F126):** Only 3/28 heads (H3, H4,
   H6) produce useful capital-city bindings. The triad is NOT
   universal. Different heads likely specialize for different tasks.
   MESH weight-weight is full-rank for all heads; rank-1 structure
   comes entirely from bias (Resonator).

---

## 8. The Lens Aperture — Resolved (F125)

Phase 10z24 answered the aperture question:

### Why 66?

```
W_o_h alone:  rank@90% ≈ 104 (universal across ALL heads, L22-L23)
W_v_h alone:  rank@90% ≈ similar
Combined:     rank@90% ≈ 66 (product of two near-isometries)

128 → 104 → 66 (architectural narrowing)
```

66 is NOT determined by the number of facts or vocabulary size. It
is an architectural constant from the 128-d head dimension.

### The Three Zones of the Bottleneck

| SVD Dims | Energy | Role | Evidence |
|:---------|:-------|:-----|:---------|
| Top 10 | 14.8% | **ANSWER** | Phase transition: rank 5→10 |
| 10-66 | 53.7% | **IDENTITY** | Mean rank improves 23.8→8.0 |
| 66-128 | 31.5% | **NOISE** | No measurable contribution |

### Key Discovery

Answer tokens are 87% orthogonal to M_h's output space. The binding
vector contains a SMALL answer signal (13% of token energy) embedded
in a larger identity signal. The MLP layers (L24-L31) must amplify
this 13% before the LM head can read it.

This explains why direct M_h binding gives rank 4-18 (good but not
rank 0) — the answer signal is present but not yet amplified.

---

*This taxonomy will be updated as new structures are discovered. Current count: 8 (7 in Qwen2-7B, 1 additional in Qwen2-1.5B).*
