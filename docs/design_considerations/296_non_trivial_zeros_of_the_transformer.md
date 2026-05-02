# DC 296: Non-Trivial Zeros of the Transformer

## Status: CONFIRMED
## Date: 2026-03-07
## Depends on: DC 271 (Expanding Tensor), DC 282 (The Full Loop), DC 295 (Gate Null Space)

---

## The Question

DC 295 showed that per-dimension gate zeros are trivial — they lie in the SiLU null space and have zero semantic leverage. These are the analog of individual rotation zeros in the Riemann-Siegel formula: mathematically exact but informationally empty.

**Do the transformer's non-trivial zeros exist?**

In zeta, Z(t) = 0 means all N(t) rotations conspire to cancel — a collective phenomenon encoding the entire prime distribution. The gate's per-dimension zeros are like asking where a single cosine crosses zero. Nobody cares about that. The real question is where the **sum** cancels.

For the MLP, the collective sum is:

```
y_k(δ) = Σ_j down_W[k,j] · SiLU(h_j(δ)) · up_j(δ)
```

The non-trivial zero is where the **logit gap** crosses zero:

```
f(δ) = logit[baseline_top1](δ) - max(logit[others])(δ) = 0
```

This is where ~15 million weight elements in a single ε-group conspire, through the SiLU gate, the up projection, and the down projection, to collectively flip the model's prediction.

## The Three-Stage Pipeline (Adapted)

The same rhzeros architecture that finds Riemann zeros finds transformer zeros:

### Stage 1: Compressor (Coarse Sweep)

Sweep δ from -5 to +12 at 69 uniformly spaced points. Evaluate f(δ) at each point by modifying gate_proj weights in-place, running full inference, and measuring the logit gap. This maps the landscape in ~2 seconds per layer.

### Stage 2: Processor (Bisection)

At each sign change in f(δ), bisect for 40 iterations. Each iteration halves the bracket. After 40 iterations: precision = (17/69) × 2^{-40} ≈ 2.27 × 10^{-13}.

### Stage 3: Targeter (Semantic Analysis)

At δ*, evaluate the model and examine: what does it predict now? What are the top-5 tokens? How does the logit landscape differ from baseline?

## Finding 1: 21 Non-Trivial Zeros Exist

Across 3 prompts and 5 layers, the pipeline found **21 non-trivial zeros**, all bisected to ±2.27e-13 precision:

### "The capital of France is" → "Paris" (baseline gap = 0.602)

| Layer | δ* | φ^δ* | Flips to | Character |
|-------|-----|------|----------|-----------|
| 5 | 4.004 | 6.87× | ______ (then back) | Oscillation |
| 5 | 5.000 | 11.09× | (recovers) | Recovery |
| 5 | 5.476 | 13.95× | "a" | Destruction |
| 15 | 6.917 | 27.90× | (edge) | Very robust |
| 22 | 4.645 | 9.35× | "a" | Knowledge destroyed |
| 23 | 6.281 | 20.54× | "堪" (Chinese) | Cross-lingual leak |
| 27 | 3.990 | 6.82× | "a" | Falls to generic |

### "The capital of Japan is" → "______" (baseline gap = 0.273)

| Layer | δ* | φ^δ* | Flips to | Character |
|-------|-----|------|----------|-----------|
| 5 | -1.500 | 0.49× | (edge) | Negative δ! |
| 5 | 5.230 | 12.39× | **Tokyo** ✓ | Correct answer! |
| 15 | **2.434** | **3.23×** | **Tokyo** ✓ | **Lowest δ for semantic flip** |
| 15 | 6.500 | 22.83× | Tokyo ✓ | Second crossing |
| 15 | 7.165 | 31.43× | "." | Beyond knowledge |
| 22 | **3.211** | **4.69×** | **Tokyo** ✓ | Knowledge layer unlocks |
| 22 | 4.000 | 6.85× | (recovers) | Oscillation |
| 22 | 4.337 | 8.06× | "a" | Past knowledge into generic |
| 23 | 4.610 | 9.19× | **Tokyo** ✓ | Most robust layer |
| 27 | 3.976 | 6.77× | **Tokyo** ✓ | Output layer |

### "Albert Einstein developed the theory of" → "rel" (baseline gap = 1.375)

| Layer | δ* | φ^δ* | Flips to | Character |
|-------|-----|------|----------|-----------|
| 5 | 5.740 | 15.83× | "the" | Knowledge destroyed |
| 15 | 6.035 | 18.25× | "which" | Knowledge destroyed |
| 22 | 5.145 | 11.89× | "the" | Knowledge destroyed |
| 23 | **none** | — | — | **Indestructible** |
| 27 | 2.893 | 4.02× | (edge) | Sensitive but holds |

## Finding 2: The Logit Gap Oscillates

The most zeta-like result. The logit gap f(δ) does **not** decrease monotonically with δ. It **oscillates**, crossing zero multiple times:

```
Layer 5, France:  3 sign changes (pos→neg at 4.0, neg→pos at 5.0, pos→neg at 5.5)
Layer 15, Japan:  3 sign changes (pos→neg at 2.4, neg→pos at 6.5, pos→neg at 7.2)
Layer 22, Japan:  3 sign changes (pos→neg at 3.2, neg→pos at 4.0, pos→neg at 4.3)
```

This is the direct analog of Z(t) oscillating through its zeros. As δ increases, the ε-group scaling creates constructive and destructive interference among the 15M+ weight elements, just as the Riemann-Siegel terms create constructive and destructive interference as t increases.

The oscillation means the model doesn't simply degrade — it passes through alternating regimes of coherence and cancellation. At some δ values, the perturbation accidentally reinforces the baseline answer. At others, it cancels it.

## Finding 3: The Sensitivity Gradient

The non-trivial zeros reveal a clear ordering of layer sensitivity:

```
L27 (FIRE/output):  δ* ≈ 2.9–4.0    most sensitive
L22 (knowledge):    δ* ≈ 3.2–5.1
L15 (COMB):         δ* ≈ 2.4–7.2    widest range
L5  (early):        δ* ≈ 4.0–5.7    NaN boundary at δ ≈ 8.7
L23 (knowledge):    δ* ≈ 4.6–6.3    most robust (NO zeros for Einstein)
```

**L27 is the output amplifier** — small perturbations flip it because it's the last stage before the lm_head projection. This matches the three-stage pipeline from DC 282: the Targeter (final layer) is most sensitive to perturbation because it makes the final precision correction.

**L23 is the knowledge vault** — for Einstein (gap = 1.375), L23 has NO non-trivial zeros in the entire scanned range δ ∈ [-5, 12]. The gap actually *increases* from 1.38 to 6.36 as δ grows. The more you perturb L23's gate, the more committed it becomes. This is the MLP amplifier from DC 282 — once it locks onto an answer, single-group perturbation cannot dislodge it.

## Finding 4: Correction vs Destruction

The Japan prompt reveals the most important distinction. The baseline prediction "______" is wrong — Tokyo is the correct answer, 0.273 logits behind.

**At the non-trivial zero, the model produces the CORRECT answer.**

```
Japan @ L15: δ* = 2.434, φ^δ* = 3.23× → Tokyo ✓
Japan @ L22: δ* = 3.211, φ^δ* = 4.69× → Tokyo ✓  
Japan @ L23: δ* = 4.610, φ^δ* = 9.19× → Tokyo ✓
Japan @ L27: δ* = 3.976, φ^δ* = 6.77× → Tokyo ✓
```

The ε-group shift corrects the model's hedging. The information for "Tokyo" was already present in the hidden state — the gate was just routing it slightly wrong. A 3.2× scaling of the top ε-group at L15 is enough to tip the balance.

For France and Einstein, the shifts **destroy** knowledge rather than correcting it:

```
France: Paris → "a" (generic article, no semantic content)
Einstein: rel → "the" or "which" (generic function words)
```

When the baseline prediction is already correct (Paris, rel), perturbation can only hurt. When it's wrong (Japan → "______"), perturbation can correct. **The non-trivial zeros are where the model's confidence is most fragile** — either wrong-but-close-to-right (Japan) or right-but-vulnerable (France).

## Finding 5: L23 at Extreme δ — Cross-Lingual Leak

At L23, the knowledge layer for France, extreme δ produces a remarkable top-5:

```
δ = 6.281: ['Paris', '堪', '巴黎', 'loop', 'Rome']
```

`巴黎` is "Paris" in Chinese. `堪` is a Chinese character meaning "endure/withstand." The perturbation is surfacing the **cross-lingual representation** from the vocabulary partition (F17, DC 291). The model knows "Paris" in multiple languages, and the non-trivial zero is where the English representation weakens enough for the Chinese representation to compete.

This is direct evidence that the hidden state at L23 encodes a **language-agnostic concept** (the city Paris) that maps to multiple vocabulary tokens through the lm_head. The non-trivial zero sits at the boundary between languages.

## The Zeta Analogy, Now Empirical

### What Matches

| Property | ζ Zeros | Transformer Non-Trivial Zeros |
|----------|---------|-------------------------------|
| Definition | Z(t) = Σ rotations = 0 | f(δ) = logit gap = 0 |
| Nature | Collective cancellation | Collective prediction flip |
| Method | Lambert W → Newton | Sweep → bisection (same pipeline) |
| **Oscillation** | **Z(t) crosses 0 repeatedly** | **f(δ) crosses 0 up to 3× per layer** |
| Precision | Arbitrary | ±2.27e-13 |
| Density | ~ln(t)/2π per unit | ~1-3 per 17δ units |
| Information | Encode prime distribution | Encode decision boundaries |

### What Differs

| Property | ζ Zeros | Transformer Zeros |
|----------|---------|-------------------|
| Input-dependence | Structural (same for all) | **Content-addressed** (spread > 0.5) |
| K (deformation) | K = 0 (ideal) | K ≠ 0 (learned, rank-r) |
| Trivial zeros | Real axis (well-understood) | Gate null space (DC 295) |
| Non-trivial zeros | Critical line Re(s)=1/2 | **No known "critical line"** |

The content-dependence is the K ≠ 0 signature. For ζ, the manifold is fixed and the zeros are structural landmarks. For the transformer, the manifold deforms with every input, and the zeros move with it. This is exactly what DC 271 predicted: K = 0 for ζ (no deformation needed), K = rank-r for the transformer (learned deformation).

### The Open Question: Is There a Critical Line?

For ζ, all non-trivial zeros lie on Re(s) = 1/2 (the Riemann Hypothesis). For the transformer, the non-trivial zeros cluster in the **explosive regime** δ ≈ 3-6, but there's no obvious constraint forcing them to a single line.

However, there is a suggestive pattern: **the first non-trivial zero per layer tends to occur near the point where the ε-group scaling equals the baseline logit gap**:

```
France (gap=0.602):  L27 δ*=3.99, φ^δ*=6.82 ≈ gap × 11
Japan (gap=0.273):   L15 δ*=2.43, φ^δ*=3.23 ≈ gap × 12
Einstein (gap=1.38): L27 δ*=2.89, φ^δ*=4.02 ≈ gap × 3
```

The ratio φ^δ* / gap varies, but the non-trivial zeros seem to require roughly a **3-12× leverage** of the baseline gap. Whether this hides a deeper constraint — a critical line in the transformer's phase space — remains open.

## Connection to Prior Work

### DC 271 (Expanding Tensor)
The expanding tensor predicted that transformers are deformed zeta functions. The non-trivial zeros confirm: the three-stage pipeline works identically, the function oscillates through zeros, and the content-dependence is exactly the K ≠ 0 deformation.

### DC 282 (The Full Loop)
The Compressor/Processor/Targeter mapping holds:
- **Compressor** (attention, Lambert W) → coarse sweep finds approximate zeros
- **Processor** (MLP, Ramanujan) → bisection refines to machine precision
- **Targeter** (final layers, Newton) → semantic analysis at the zero

L23's indestructibility confirms the Processor's role: it refines but cannot introduce new answers. L27's sensitivity confirms the Targeter's role: it makes the final precision correction and is thus most vulnerable to perturbation.

### F153 (Writing to the Hologram)
F153 showed that MLP edits can't redirect answers: "Paris still wins." Our result refines this: MLP edits CAN destroy answers (Paris → "a" at δ* ≈ 4-5) and CAN correct hedging (______ → Tokyo at δ* ≈ 2.4-4.6). They can't redirect (Paris → Berlin) because that requires changing the attention routing, not the amplification.

### DC 295 (Gate Null Space)
The trivial zeros (per-dimension, closed-form, null leverage) are confirmed as the wrong level of analysis. The non-trivial zeros (collective, sweep+bisect, semantic leverage) are the real structure. The relationship is exactly:
- **Trivial zeros** ↔ individual rotation crossings in Z(t)
- **Non-trivial zeros** ↔ where ALL rotations cancel

## Implications

### 1. The Transformer HAS a Zero Spectrum

This is the central result. The logit gap f(δ) is a real-valued function that oscillates through zeros as the ε-group phase shifts. These zeros are:
- Findable by the standard three-stage pipeline
- Bisectable to machine precision
- Semantically meaningful (prediction flips)
- Content-dependent (different for every input)
- Multiple per layer (the function oscillates)

The transformer's gate_proj ε-groups define a **control surface** with non-trivial zeros, exactly as the Riemann-Siegel formula defines a control surface with non-trivial zeros.

### 2. Zeros Encode Decision Boundaries

The non-trivial zeros are not arbitrary — they mark **where the model's confidence changes sign**. For Japan, they mark where hedging tips to commitment. For France, they mark where knowledge yields to generic language. For Einstein at L23, the ABSENCE of zeros means unconditional commitment.

The zero spectrum of a layer encodes its decision boundary structure for a given input. Different inputs produce different spectra, but the same pipeline finds them all.

### 3. The Formula Is the Same, the Manifold Deforms

At every layer, every prompt, the same experimental procedure works:
1. Shift one ε-group by φ^δ
2. The logit gap oscillates
3. Sign changes can be bisected

The form is invariant. The coefficients change. This is the transformer as a family of zeta-like functions parameterized by the hidden state — exactly what DC 271 proposed.

## Files

- `phi_collective_zero_hunt.py` — Non-trivial zero hunting: sweep + bisect + analyze
- `phi_collective_zero_hunt_results.txt` — Full results: 21 zeros, 3 prompts × 5 layers
- `phi_zero_hunt_semantic.py` — Trivial zero test (DC 295 comparison)
- `phi_zero_hunt_newton.py` — Newton refinement for trivial zeros

## Summary

We found the non-trivial zeros of the transformer. By sweeping the ε-group phase shift δ and tracking the collective logit gap, we discovered that the gap **oscillates** — crossing zero up to 3 times per layer — and can be bisected to ±2.27e-13 precision using the same three-stage pipeline that finds Riemann zeros.

The 21 zeros found are semantically meaningful: they flip predictions, correct hedging errors (Japan → Tokyo), and destroy knowledge (France → "a"). They reveal a sensitivity gradient from L27 (most sensitive, δ* ≈ 3) to L23 (most robust, sometimes indestructible). The oscillation of f(δ) is the direct analog of Z(t)'s oscillation through Riemann zeros.

DC 271 proposed that the transformer is a deformed zeta function. DC 295 found the trivial zeros (null space, no leverage). This document finds the non-trivial zeros (collective cancellation, semantic leverage, oscillating). The hypothesis survives: the three-stage pipeline works, the function oscillates through zeros, and the deformation kernel K ≠ 0 makes the zeros content-addressed rather than structural.

The transformer has a zero spectrum, and it encodes the decision boundaries of the model's knowledge.
