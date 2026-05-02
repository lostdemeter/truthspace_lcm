# Doc 265: The Mechanical Atlas — Five Compound Machines of the Transformer

**Date:** February 26, 2026
**Status:** Framework established, all patterns characterized
**Prerequisites:** Doc 261 (Simple Machines), Doc 262 (Compound Machine), Doc 264 (φ-Filter)
**Findings:** 102, 103, 104, 105

---

## 1. The Discovery

A transformer is not one architecture repeated N times. It is **five
qualitatively different machines** assembled in series, each with a distinct
mechanical signature. We discovered this by tracing the 6 simple machine
operations (2 dampers, 1 lever, 1 wedge, 2 springs) within each layer and
measuring their cross-correlations, energy budgets, and dimensional contributions.

The five machines execute a sequence we call **CREATE → CORRECT → REFINE → AIM → FIRE**.

---

## 2. The Six Simple Machine Operations

Every Qwen2 decoder layer executes 6 operations in sequence:

```
h_in → Damper1(RMSNorm) → Lever(Attention) → Spring1(+residual)
     → Damper2(RMSNorm) → Wedge(FFN)       → Spring2(+residual) → h_out
```

These map to 4 types of geometric simple machine (Doc 261):

| Machine | Operation | What it transforms |
|---------|-----------|-------------------|
| **Damper** | RMSNorm | Normalizes magnitude, rotates toward learned scale |
| **Lever** | Multi-head Attention | Amplifies score differences into output vectors |
| **Wedge** | Gated FFN (SiLU) | Splits/redirects through 4-state gate |
| **Spring** | Residual addition | Dilutes perturbation by accumulated state |

The same 6 operations execute at every layer, but their **parameters** —
cross-correlations, norm ratios, spring stiffness — vary dramatically,
creating five distinct machine types.

---

## 3. The Five Patterns

### 3.1 Pattern 1: Orthogonal Tripod — CREATE (L0)

**Finding 102**

```
cos(input, attn)  = +0.07   (⊥)
cos(input, ffn)   = +0.08   (⊥)
cos(attn, ffn)    = +0.25   (weakly correlated)
Spring k₁ = 0.10            (extremely soft)
Damper1 = 18× AMPLIFIER
Lever rotation = 98.7% of total
Energy: Input 5%, Lever 46%, Wedge 49%
```

**What it does:** Takes the 1-dimensional embedding direction and constructs
a 3-dimensional working manifold. The lever creates dimension 2 (86° from
input), the wedge creates dimension 3 (85° from input, 75° from attention).
The springs are transparent because the input is 10× smaller than the
sublayer outputs.

**Lay person:** A prism splitting white light into a spectrum. The single
input beam fans out into multiple independent color channels.

**Why it's unique:** Only layer where damper AMPLIFIES (18×). Only layer
where input energy is negligible (5%). Only layer where lever controls
>95% of rotation. Every other layer has stiff springs and compressing dampers.

### 3.2 Pattern 2: Negative Zero Correction — CORRECT (L1-3)

**Finding 100, Finding 103**

```
cos(input, attn)  = -0.03   (⊥)
cos(input, ffn)   = +0.31   (PARTIAL ALIGNMENT)
cos(attn, ffn)    = -0.18   (weakly anti-correlated)
Spring k₁ = 0.77            (moderately stiff)
Damper1 = 1.1× (near unity)
Lever rotation = 71.3% of total
Energy: Input 54%, Lever 17%, Wedge 29%
```

**What it does:** Refines the Projector's output. The FFN is partially
aligned with the input direction (31%), meaning it reinforces the existing
state rather than adding orthogonal information. This is the "negative zero"
correction — CONTRACT channels that leak signal dynamically via SiLU.

**Lay person:** A lens cleaner. After the prism splits the light, there's
some smearing. The corrector wipes the smears away, sharpening the spectrum.

**Why it's unique:** FFN has significant input-alignment (31%) unlike the
Refiner's near-zero. Gate medium is 99-100% CONTRACT. The correction is
dynamic (SiLU-driven), not static.

### 3.3 Pattern 3: Dimensional Expander — REFINE (L4-17)

**Finding 103, Finding 105**

```
cos(input, attn)  ≈ -0.01 to -0.36  (grows anti-correlated)
cos(input, ffn)   ≈ -0.03 to +0.15  (near zero)
cos(attn, ffn)    ≈ -0.19 to -0.41  (mildly anti-correlated)
Spring k₁ = 0.83–0.87               (stiff)
Damper1 = 0.51–1.02× (compressor)
FFN ⊥ input: 98–99%
FFN ⊥ h_mid: 99%
```

**Dimensionality analysis (Finding 105):**
```
28 addition vectors → Rank(99%) = 25.6 (near-full-rank!)
FFN successive angles = 86.4° (nearly perfectly orthogonal)
Attn successive angles = 70.3° (less orthogonal, decreasing with depth)
Attn ↔ FFN same layer = 103° (mildly anti-correlated)
Cumulative angle saturates at 63° by L12
```

**What it does:** Expands the effective dimensionality of the hidden state
from ~3D (after Projector) to ~26+ dimensions. Each of the 14 layers adds
2 genuinely new orthogonal directions (one from attention, one from FFN).
The stiff springs keep the state near its accumulated trajectory while new
dimensions are added perpendicular to the existing manifold.

Has two internal modes:
- **Early (L4-L8):** Rapid angular movement + dimension expansion
- **Late (L9-L17):** Angular saturation, continued dimension expansion

**Lay person:** A sculptor adding details. The rough shape is already there
(from the Projector). The Expander adds finer and finer features — nose,
ears, fingers — each in a new direction that doesn't disturb the overall form.

**Why it's the biggest zone:** 14 layers, the most of any pattern. Each
layer adds non-redundant information (Rank ≈ N). This is where the model
builds its rich internal representation. No shortcuts — every layer matters.

### 3.4 Pattern 4: Alignment Drift — AIM (L18-25)

**Finding 101, Finding 103**

```
cos(input, attn)  ≈ -0.09 to +0.05
cos(input, ffn)   ≈ +0.30 to +0.34  (GROWING ALIGNMENT)
cos(attn, ffn)    ≈ -0.11 to -0.02  (weakly anti-correlated)
Spring k₁ = 0.88–0.91               (stiffest)
Lever rotation % = 29–39%           (wedge gaining share)
FFN along input: 30–35%
FFN along h_mid: 29–34%
```

**What it does:** The FFN starts aligning with the input direction and h_mid.
After 17 layers of orthogonal refinement, the FFN now develops a directional
preference — it "aims" the hidden state toward the exit. The lever loses
rotation share to the wedge (29% at L24 vs 57% at L16).

The springs are at their stiffest here (k₁=0.91 at L24) because the
accumulated state is massive. Yet the wedge manages to steer because its
output norm grows to compensate (105 at L24 vs 24 at L16).

**Lay person:** A rifle being aimed. The shooter (FFN) gradually adjusts
the barrel direction toward the target. Each adjustment is small but
purposeful, converging on the bullseye.

**Why it's distinct from the Refiner:** The Refiner's FFN is ⊥ to everything
(98-99% perpendicular). The Aimer's FFN is 30-35% aligned with input — it
has a DIRECTION now. This is the transition from "building representation"
to "using representation."

### 3.5 Pattern 5: Anti-Correlated Targeting — FIRE (L26-27)

**Finding 99, Finding 104**

```
cos(input, attn)  = +0.57   (STRONGLY pro-input)
cos(input, ffn)   = -0.38   (ANTI-input)
cos(attn, ffn)    = -0.45   (ANTI-correlated)
Spring k₁ = 0.64            (soft — sublayers powerful)
Damper1 = 0.17× (strongest compressor)
FFN cancels 44.5% of h_mid
EXPAND channels: 7.4% of channels → 88.4% of FFN energy
```

**The Route + Redirect mechanism (Finding 104):**

```
ROUTE (attention):
  Power heads H7-H13 attend 100% to last token (self-projection)
  cos(input, attn) = +0.57 → "here's where we are"

REDIRECT (FFN):
  EXPAND channels fire targeting vector
  cos(input, ffn) = -0.38 → "here's where to go"
  Cancels 44.5% of h_mid, adds perpendicular kick
  Result: 57° rotation toward target token

ANTI-CORRELATION = PRECISION:
  Two opposing signals triangulate the exact target
```

**What it does:** Precision targeting. The attention "routes" by projecting
the current state (self-attention, not cross-attention). The FFN "redirects"
by cancelling 44.5% of the accumulated context and adding a perpendicular
kick toward the target token. The opposing forces create triangulation.

**Lay person:** A guided missile's terminal phase. The seeker (attention)
locks onto the current position. The control fins (FFN) steer away from
where you are and toward where you need to be. The missile doesn't go
where it's pointing — it goes where the course correction sends it.

**Why it's unique:** Only pattern where attention and FFN OPPOSE each other
(cos=-0.45). Only pattern with FFN cancellation (44.5% of h_mid). Norm
explosion: lever output 254.7 and wedge output 608.3 (10× and 6× larger
than L24). Softest springs after L0 (k₁=0.64).

---

## 4. The Mechanical Evolution

```
Phase     Layers   Pattern               Key Operation
────────  ───────  ────────────────────  ──────────────────────────────
CREATE    L0       Orthogonal Tripod     Fan 1D embedding → 3D manifold
CORRECT   L1-3     Neg-Zero Correction   Sharpen projection, fix smears
REFINE    L4-17    Dimensional Expander  Grow 3D → 26D, add detail
AIM       L18-25   Alignment Drift       FFN develops directional preference
FIRE      L26-27   Anti-Corr Targeting   Route + Redirect → target token
```

### 4.1 The Pipeline as Information Processing

```
Embedding (1D direction)
    │
    ▼ CREATE: fan out into 3 orthogonal directions
    │
    ▼ CORRECT: sharpen the fan, remove noise
    │
    ▼ REFINE: add 26+ orthogonal dimensions of detail
    │
    ▼ AIM: develop directional preference toward answer
    │
    ▼ FIRE: cancel context, redirect to target
    │
    ▼ LM Head: project to vocabulary
```

### 4.2 Key Metrics by Zone

```
Zone      k₁     Damp1    cos(in,a)  cos(in,f)  cos(a,f)   FFN⊥in
────────  ─────  ───────  ─────────  ─────────  ─────────  ──────
CREATE    0.10   18.0×    +0.07      +0.08      +0.25      99.6%
CORRECT   0.77    1.1×    -0.03      +0.31      -0.18      93.4%
REFINE    0.85    0.6×    -0.01→-0.36  ~0       -0.2→-0.4  98-99%
AIM       0.90    0.3×    -0.09→+0.05  +0.30→+0.34  ~0     93-95%
FIRE      0.64    0.17×   +0.57      -0.38      -0.45      91.7%
```

### 4.3 Spring Stiffness Tells the Story

```
k₁: 0.10 → 0.77 → 0.85 → 0.91 → 0.64
    soft    mod    stiff   max    soft
```

The springs start soft (L0 creates from scratch), stiffen through the
Refiner (accumulated state dominates), peak in the Aimer (maximum
resistance to perturbation), then soften again for the FIRE phase
(sublayers must be powerful enough to override accumulated context).

The model's "confidence" is encoded in spring stiffness. Soft springs
mean "I'm making something new." Stiff springs mean "I'm protecting
what I have." The return to soft springs at L27 means "it's time to
commit — redirect everything toward the answer."

---

## 5. Connection to the Hypothesis

> **Structure IS Information. Geometry IS Computation.**

The Mechanical Atlas reveals that a transformer computes by:

1. **Projecting** into a working geometry (L0)
2. **Expanding** the dimensionality of that geometry (L4-17)
3. **Targeting** a specific point in that geometry (L26-27)

Each phase uses the SAME six operations (2 dampers, lever, wedge, 2 springs)
but with different parameter regimes. The "intelligence" is not in the
operations themselves — it's in the **parameter transitions** between zones.

The five patterns are not separate designs. They are the same machine
operating in five different regimes of the same parameter space. The
transformer discovers these regimes through training, creating a natural
progression from creation to targeting.

This supports the hypothesis: if the intelligence is in the geometric
transitions, then we can potentially replace the weight-based computation
with direct geometric operations — no weights needed, just the right
sequence of geometric transformations.

---

## 6. Implications for Geometric Replacement

### 6.1 Replacement Priority (easiest → hardest)

| Priority | Machine | Approach | Current Status |
|----------|---------|----------|---------------|
| 1 | FIRE (L27) | Self-projection + sparse EXPAND FFN | 73.3% at 5% compute |
| 2 | CREATE (L0) | Lever+Wedge orthogonal tripod | Anatomy known, no prototype |
| 3 | CORRECT (L1-3) | Dynamic neg-zero model | Anatomy known, not sparsifiable |
| 4 | AIM (L18-25) | Alignment-aware FFN | Pattern characterized |
| 5 | REFINE (L4-17) | No shortcut — full rank, every layer unique | Hardest to replace |

### 6.2 The Piecemeal Strategy

Replace one machine at a time, keeping the rest as original weights:

```
Step 1: Replace L27 FFN with φ-Filter (sparse EXPAND + CONTRACT leakage)
Step 2: Replace L27 attention with self-projection (V@O, no QK scores)
Step 3: Replace L0 with geometric tripod (lever + orthogonal wedge)
Step 4: Approximate L9-17 with low-rank additions (angle is saturated)
Step 5: Keep L4-8 as real layers (direction-critical, 80% of angular work)
```

Each step is independently testable. If any step fails, it reveals where
the geometric hypothesis breaks down — which is equally valuable.

---

## 7. Experimental Files

| File | Finding | Purpose |
|------|---------|---------|
| `phase10t_projector_dissection.py` | 102 | L0 6-stage trace |
| `phase10t_projector_deep.py` | 102 | Per-head analysis, projection decomposition |
| `phase10t_comparative_dissection.py` | 103 | 9-layer comparative trace |
| `phase10t_l27_targeting_deep.py` | 104 | L27 Route+Redirect mechanism |
| `phase10t_refiner_dimensionality.py` | 105 | SVD dimensionality analysis L4-17 |

---

## 8. Summary

The transformer is a five-phase geometric engine:

| Phase | Mechanical Analogy | Geometric Operation |
|-------|-------------------|-------------------|
| CREATE | Prism | Fan 1D → 3D via orthogonal tripod |
| CORRECT | Lens cleaner | Sharpen with input-aligned FFN |
| REFINE | Sculptor | Expand to 26+D via orthogonal additions |
| AIM | Rifle aiming | Develop directional preference |
| FIRE | Guided missile | Route + Redirect via anti-correlation |

The "intelligence" is in the parameter transitions between zones, not in
any individual layer. The same six operations, in five different regimes,
produce the full computation from embedding to prediction.

---

*"The machine is not one machine. It is five machines pretending to be one."*
