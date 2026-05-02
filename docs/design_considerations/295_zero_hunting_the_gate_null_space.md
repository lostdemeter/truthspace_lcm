# DC 295: Zero-Hunting on the Phase Shift Control Surface — The Gate's Null Space

## Status: CONFIRMED
## Date: 2026-03-06
## Depends on: DC 253 (Four Gate States), DC 282 (The Full Loop), DC 293 (Sieve Paradigm), DC 294 (Controllable Funnel)

---

## The Question

DC 294 established that ε-group phase shifts are a parametric control surface — not a compression opportunity but a steering mechanism. The SiLU gate classifies every dimension into four states (+1, +0, -0, -1), and ε-group shifts can push dimensions across the +0 ↔ -0 boundary.

**Can we find the exact phase shift δ that flips any given gate dimension?**

And if so: **does flipping gate dimensions change what the model says?**

This is the [rhzeros](https://github.com/lostdemeter/rhzeros) pipeline mapped to transformer architecture: find zeros on a control surface, refine them to machine precision, then test whether they have physical (semantic) consequences.

## The Three-Stage Pipeline

### Stage 1: Compressor (Lambert W Analog)

For a hidden state **x** entering the MLP, the gate pre-activation for dimension j is:

```
h_j = Σ_i gate_W[j,i] · x[i]
```

The four states from DC 253 emerge naturally:
- **+1**: h_j >> 0 (gate fully open, SiLU passes signal)
- **+0**: h_j slightly > 0 (barely open, near zero from positive side)
- **-0**: h_j slightly < 0 (barely closed, near zero from negative side)
- **-1**: h_j << 0 (gate fully shut, SiLU kills signal)

With random inputs at Layer 0: +1 = 17.1%, +0 = 33.3%, -0 = 33.4%, -1 = 16.3%.
With real hidden states at Layer 23: +1 = 559, +0 = 1077, -0 = 2837, -1 = 14471 — the gate is **strongly biased negative** in deep layers, consistent with the pre-GELU distribution findings in DC 243.

### Stage 2: Processor (Ramanujan Analog)

When we shift ε-group k by φ^δ, the contribution of that group to dimension j is:

```
c_j = Σ_{i ∈ group_k} gate_W[j,i] · x[i]
```

The shifted activation becomes:

```
h_j(δ) = h_j(0) + c_j · (φ^δ - 1)
```

Setting h_j(δ) = 0 and solving:

```
δ_critical = log_φ(1 - h_j / c_j)
```

This is a **closed-form, exact solution** — no iteration needed. The formula exists whenever:
- c_j ≠ 0 (the group contributes to this dimension)
- 1 - h_j/c_j > 0 (a real logarithm exists)

### Stage 3: Targeter (Newton Snap)

The analytical formula is the full solution. Newton refinement was implemented as verification:

```
f(δ) = h_j + c_j · (φ^δ - 1)
f'(δ) = c_j · φ^δ · ln(φ)
δ_{n+1} = δ_n - f(δ_n) / f'(δ_n)
```

## Finding 1: The Formula Is Exact

| | Layer 0 | Layer 3 |
|---|---------|---------|
| Float32 hit rate | 58/100 (58%) | 44/100 (44%) |
| **Float64 hit rate** | **100/100 (100%)** | **100/100 (100%)** |
| Newton iterations | 1.0 mean, 1 max | 1.0 mean, 1 max |
| Residuals | median 1.84e-16 | median 6.97e-16 |

**Newton converges in exactly 1 iteration because the first-order estimate IS the exact solution.** The 50% miss rate in the initial experiment was purely float32 accumulation error — the max difference between float32 and float64 matmul is 7.75e-06 at Layer 0 and 1.03e-05 at Layer 3. In float64, every predicted zero actually flips.

The residuals (~1e-16) are at machine epsilon for float64. There are zero "resistant zeros" — every dimension that has a real solution flips exactly as predicted.

### The rhzeros Analogy, Completed

| rhzeros Stage | Zeta Zero-Hunting | Gate Zero-Hunting | Result |
|---|---|---|---|
| Compressor | Lambert W estimate | `δ = log_φ(1 - h_j/c_j)` | Closed-form, exact |
| Processor | Ramanujan refinement | Newton iteration | 1 iter (already converged) |
| Targeter | Z(t) evaluation | Matmul verification | 100% hit rate |
| Precision | 50-digit mpmath | float64 (1e-16) | Machine epsilon |

The gate zero formula is *simpler* than Riemann zeros because the underlying function h_j(δ) is exactly exponential (no oscillation, no cancellation). The Lambert W analog gives the exact answer on the first try.

## Finding 2: The Zero Spectrum Is Input-Dependent

Comparing zero spectra across different random inputs:

```
Input 0: 8,466 zeros (44.7% of 18,944 dims)
Input 1: 8,501 zeros
Jaccard similarity between zero sets: 0.000
```

Every input produces a **completely different** zero spectrum. The gate reads the input and produces a content-addressed routing decision. This confirms the "content-addressed routing" interpretation from DC 253 — the gate is not applying a fixed filter but dynamically selecting which dimensions to activate based on what it sees.

## Finding 3: Precision Targeting Works Mathematically

Sorting zeros by |δ| gives a precision targeting table:

| Flip N dims | δ required | φ^δ | Regime |
|---|---|---|---|
| 1 | 0.000651 | 1.000313× | Deep controllable |
| 10 | 0.003705 | 1.001783× | Controllable |
| 100 | 0.087322 | 1.042× | Controllable edge |
| 500 | 0.781583 | 1.470× | Moderate |
| 1000 | 1.511575 | 2.415× | Explosive |

The controllable regime (δ < 0.1 from DC 294) gives us up to ~100 flippable dimensions per ε-group per layer.

## Finding 4: The Gate's Null Space — Near-Zero Dims Have Zero Leverage

This is the central finding. We ran the full experiment on real model hidden states with 5 knowledge prompts across 6 layers:

### Semantic Impact Test Results

**"The capital of France is" → "Paris"**

| Layer | Dims flipped | δ | Cosine | Top-1 | KL |
|---|---|---|---|---|---|
| 22 | 100 | 0.090 | 1.0000 | ✓ Paris | 0.0000 |
| 22 | 500 | -0.499 | 1.0000 | ✓ Paris | 0.0002 |
| 22 | 1000 | 0.999 | 0.9999 | ✓ Paris | 0.0019 |
| 23 | 100 | 0.159 | 1.0000 | ✓ Paris | 0.0000 |
| 23 | 500 | 0.778 | 0.9999 | ✓ Paris | 0.0017 |
| 23 | 1000 | 1.512 | 0.9997 | ✓ Paris | 0.0094 |

**The prediction never changes.** Not at any layer. Not for any prompt. Not even when flipping 1000 gate dimensions.

All 5 prompts (France→Paris, Japan→Tokyo, water→freezing, Jupiter, Einstein→relativity) showed identical robustness: cosine > 0.9997, top-1 unchanged, KL < 0.01, even at the most aggressive settings.

The only exception: **Layer 5**, which has only ~500 zeros (vs 5000-9000 at other layers). At N=100 with δ ≈ 4.4 (explosive regime, φ^δ ≈ 10×), the prediction finally flipped from "Paris" to "______". But δ > 4 multiplies the entire ε-group by 10× — this isn't precision flipping, it's demolition.

### Why: The SiLU Null-Leverage Identity

The near-zero dims are precisely where the gate function has zero leverage:

```
SiLU(x) = x · σ(x)

At x = 0:    SiLU(0) = 0
At x = +ε:   SiLU(+ε) ≈ +ε/2
At x = -ε:   SiLU(-ε) ≈ -ε · σ(-ε) ≈ -ε/2 · (ε/2) ≈ -ε²/4 ≈ 0

Flipping +ε → -ε:  ΔSiLU ≈ ε/2 → 0 ≈ ε/2
```

For a dimension with h_j ≈ 0, both SiLU(+ε) and SiLU(-ε) are approximately zero. Flipping the sign changes the output by ~ε/2, where ε is already tiny (these dims are near-zero by definition). The MLP output — a sum over 18,944 dimensions — is dominated by the committed dims (+1/-1) where |h_j| >> 0 and SiLU output is substantial.

**The zeros of the gate control surface lie in the null space of the gate's information flow.** The gate's zero-crossings are mathematically exact and precisely targetable, but they occur at points where the gate function maps to zero regardless of sign.

## Finding 5: Architectural Implications

### The Gate's Redundancy Structure

```
Layer 23 gate states for "The capital of France is":
  +1:    559 dims (  3.0%) — fully open, carry signal
  +0:  1,077 dims (  5.7%) — barely open, ≈ zero output
  -0:  2,837 dims ( 15.0%) — barely closed, ≈ zero output  
  -1: 14,471 dims ( 76.4%) — fully shut, carry no signal

Signal carriers:    559 dims ( 3.0%)
Null space:       3,914 dims (20.7%)
Dead space:      14,471 dims (76.4%)
```

The MLP at Layer 23 routes **97% of dimensions to zero or dead**. Only 3% carry meaningful signal. The flippable near-zero dims (20.7%) are in the null space — changing them from "zero output" to "also zero output" has no effect.

### Consistency with Prior Findings

This result is deeply consistent with three prior discoveries:

1. **DC 294 (Controllable Funnel)**: The MLP compresses 3584D → effective rank 4 after 8 layers. Near-zero dims are what gets compressed away.

2. **F153 (Writing to the Hologram)**: MLP edits fail to redirect answers. "Paris still wins." The MLP faithfully amplifies whatever attention presents — you can't change its routing by editing near-zero gate dimensions.

3. **F154 (Attention Editing)**: To change the answer, you must edit the **attention** (reader), not the **MLP** (amplifier). Two layers of attention editing (L22-23) suffice to flip France→Berlin.

### The Zeta Pipeline, Revisited

DC 282 mapped the three-stage rhzeros pipeline to the transformer:

```
Compressor (Lambert W) → Attention: finds approximate answer
Processor (Ramanujan)  → MLP: refines but can't introduce new answer  
Targeter (Newton snap) → Final layers: extract and present
```

Our experiment confirms the middle step: the MLP's gate zeros are refinement points in the null space — they exist, they're exact, but they're where the function is already at zero. **The MLP can't introduce new zeros because its zeros have no leverage.** To find a different zero, you must change the compressor's initial estimate (the attention routing).

## The Analogy to Riemann Zeros

There is a beautiful structural parallel:

| Property | Riemann ζ Zeros | Gate Zeros |
|---|---|---|
| Location | Critical line Re(s) = 1/2 | SiLU zero-crossing h_j = 0 |
| Density | ~t/(2π) · ln(t) | ~5000-9000 per layer |
| Computation | Lambert W + Newton | `log_φ(1 - h/c)` (exact) |
| Precision | Arbitrary (mpmath) | Machine epsilon (float64) |
| **Semantic weight** | **Zeros encode prime distribution** | **Zeros encode null space** |

The key difference: Riemann zeros are **maximally informative** — they encode the entire prime distribution. Gate zeros are **minimally informative** — they encode the null space of the gate's routing. This is because:

- ζ(s) = 0 means rotations from ALL primes conspire to cancel → encodes global structure
- h_j(δ) = 0 means this single dimension's contribution is null → encodes local irrelevance

The gate actively **pushes information away from its zeros** into its poles (+1/-1 states). The Riemann zeta function does the opposite — it **concentrates information at its zeros**.

## Open Questions

1. **Multi-group coordinated shifts**: What if we shift ALL ε-groups simultaneously, each by a different δ? The groups act as "prime factors" (DC 293) — coordinated shifts might access non-null modes.

2. **Committed-dim targeting**: Instead of flipping near-zero dims, what δ would flip a +1 → -1 dim? These carry actual signal. The formula still applies but δ would be large (explosive regime).

3. **Multi-layer chained shifts**: DC 294 showed 17× amplification over 8 layers. Our test shifted one group at one layer. Chaining the same shift through all 28 layers might accumulate enough perturbation to matter.

4. **Attention zero spectrum**: The gate zeros are null. What about attention weight zeros? Attention is the reader — zeros there might have actual leverage (consistent with F154 showing attention edits work).

5. **The 3% question**: Only 559/18,944 dims carry signal at L23. Is this set consistent across inputs? If so, the MLP has a fixed "active subspace" that could be identified and directly targeted.

## Files

- `phi_zero_hunt_gate.py` — Stage 1: first-order zero-hunting, four-state classification, 50% hit rate
- `phi_zero_hunt_gate_results.txt` — Zero spectrum statistics, input-dependence analysis
- `phi_zero_hunt_newton.py` — Stages 2-3: Newton refinement, float64 upgrade, 100% hit rate
- `phi_zero_hunt_newton_results.txt` — Layer 0 and Layer 3 confirmation
- `phi_zero_hunt_semantic.py` — Capstone: real model hidden states, 5 prompts × 6 layers × 6 flip counts
- `phi_zero_hunt_semantic_results.txt` — Full results showing gate robustness

## Summary

We applied the rhzeros three-stage pipeline to the SiLU gate's phase shift control surface and achieved **perfect zero-hunting**: closed-form formula, 100% hit rate, machine-epsilon precision, 1 Newton iteration. The mathematics is exact.

But when we tested these zeros on real model inference, the prediction never changed. The gate's zeros lie in its **null space** — the SiLU function maps near-zero inputs to near-zero outputs regardless of sign. The model concentrates its signal in the committed dims (+1/-1) and puts its zeros where they can't affect anything.

This is not a failure — it's a discovery about how gating works. The gate is a **selective amplifier with built-in robustness**: its zero-crossings are the points of maximum mathematical controllability and minimum semantic leverage. The model engineered its own stability by placing information away from the boundary.

To actually steer the model, you must go through the reader (attention), not the amplifier (MLP gate). The MLP's job is to faithfully process whatever attention presents — and its gate zeros are precisely the points where that processing is invariant to perturbation.
