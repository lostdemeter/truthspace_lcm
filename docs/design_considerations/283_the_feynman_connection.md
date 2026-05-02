# Doc 283: The Feynman Connection — Transformers as Applied QED

**Date:** March 3, 2026
**Status:** Theoretical Framework — Derived from F153–F154, DC 280, DC 282
**Prerequisites:** DC 280 (Superposition of Shapes), DC 282 (The Full Loop)

---

## 1. The Observation

In 1985, Richard Feynman published *QED: The Strange Theory of Light
and Matter*, explaining quantum electrodynamics to a lay audience using
"little arrows" — complex amplitudes that rotate and add. The central
message:

> A photon doesn't take one path. It takes ALL paths simultaneously.
> The probability of detection is the square of the sum of all the
> little arrows. Paths near the classical path constructively interfere.
> Distant paths cancel.

In 2026, we experimentally verified the same structure inside a
transformer neural network (Findings 150–154):

> A token doesn't flow through one rank-1 shape. It flows through ALL
> shapes simultaneously. The output is where all shapes conspire to
> constructively interfere. You can't attribute the answer to one shape.
> You redirect the answer by changing the boundary conditions, not by
> blocking individual paths.

This is not a metaphor. The mathematics is the same.

---

## 2. The Correspondence

### 2.1 Feynman's Framework

In QED, the amplitude for a photon to go from source S to detector D is:

```
A(S→D) = Σ_paths  a_path · e^(i·φ_path)
```

Where:
- Each path contributes an amplitude `a_path` and a phase `φ_path`
- The probability is P = |A|²
- Paths with similar phase add constructively (stationary phase)
- Paths with random phase cancel (destructive interference)
- The classical path emerges where ∂φ/∂path = 0

The key properties:
1. **All paths contribute** — you can't ask "which path did the photon take?"
2. **The answer is collective** — P = |Σ arrows|², not Σ|arrows|²
3. **Boundary conditions matter** — change S or D, all paths shift
4. **Blocking one path barely changes the sum** — the others still interfere
5. **The apparatus geometry matters** — mirrors, lenses redirect the stationary phase

### 2.2 The Transformer Framework

In a transformer, the output for entity E through weight matrix W is:

```
output = W · h_E = (Σ_c  σ_c · u_c · v_c^T) · h_E = Σ_c  σ_c · (v_c · h_E) · u_c
```

Where:
- Each rank-1 component c contributes amplitude σ_c · (v_c · h_E) in direction u_c
- The correct answer is where all components constructively interfere
- Components whose v_c aligns with h_E contribute strongly (stationary phase)
- Components whose v_c is orthogonal to h_E contribute nothing (destructive interference)
- The dominant structure class emerges where (v_c · h_E) is maximized

The same five properties hold:
1. **All shapes contribute** — you can't ask "which shape produced the answer?" (F151: wrong v₁ also works)
2. **The answer is collective** — the hologram is deep, 3 classes = 0.14% energy (F152)
3. **Boundary conditions matter** — change entity at pos 3, answer changes entirely (F154)
4. **Editing one shape barely changes the sum** — rank-1 weight edit fails (F153)
5. **The apparatus geometry matters** — attention (the reader) redirects the output (F154)

### 2.3 The Rosetta Stone

| QED (Feynman) | Transformer (F150–F154) | Zeta (DC 282) |
|:--------------|:-----------------------|:--------------|
| Source S | Entity at position 3 | Parameter t |
| Detector D | Output token | Zero location |
| Path | Rank-1 shape σᵢuᵢvᵢᵀ | Term n^{-1/2}e^{iθₙ} |
| Amplitude per path | σᵢ(vᵢ · h) | n^{-1/2} |
| Phase per path | angle(vᵢ, h) | θ - t·ln(n) |
| Sum of arrows | W · h = Σ components | Z(t) = Σ rotations |
| Probability \|A\|² | Logit / softmax | \|Z(t)\|² |
| Stationary phase | Dominant v₁ direction | Critical strip |
| Blocking one path | Rank-1 weight edit (F153) | Removing one term |
| Moving the source | Entity-position swap (F154) | Changing t |
| Reshaping mirror | Attention editing (F154) | Changing N(t) |
| Classical path | Rank-1 approximation (F150) | Lambert W estimate |
| All paths interfere | Hologram is deep (F152) | All terms needed |

---

## 3. The Five Properties, Experimentally Verified

### 3.1 All Paths Contribute

**Feynman:** A photon takes all paths. Even the "crazy" paths that bounce
off the ceiling contribute — they just mostly cancel.

**F151–F152:** The weight matrix is full-rank (rank@90% = 2645/3584).
The "wrong" v₁ direction (from a different structure class) still produces
correct answers, because the hologram encodes all classes simultaneously.
Three structure classes removed = 0.14% energy. The answer requires the
interference of ALL components.

### 3.2 The Answer Is Collective

**Feynman:** You compute P = |Σ arrows|², not Σ|arrows|². The cross-terms
(interference) are where the physics lives.

**F152:** Holographic refinement UNIFIES structure classes (cos 0.97),
it doesn't separate them. The shared component is 62–85% of the filter
response. The answer isn't in any one shape — it's in the collective
interference pattern of all shapes.

### 3.3 Boundary Conditions Matter

**Feynman:** Move the source or the detector, and all path lengths change.
The stationary phase shifts to a new location. The photon "goes somewhere
else" — not because you redirected it, but because the interference
pattern shifted.

**F154, Experiment E:** Swap the entity hidden state at position 3
(the source), and the model outputs Berlin instead of Paris. This works
from embedding through L20 with strong margins (+4.87 to +6.01).
You changed the source, so ALL rank-1 shapes respond differently,
and they now constructively interfere at "Berlin."

The cheapest edit: 3,584 numbers. 0.0003% of the active state.
The entire interference pattern shifted because the boundary condition
shifted.

### 3.4 Blocking One Path Barely Changes the Sum

**Feynman:** Cover one slit in a double-slit experiment and the pattern
changes. But cover one of 3,584 slits? The pattern is essentially
unchanged.

**F153, Experiment B:** Edit one rank-1 component of W_gate + W_up to
redirect France → Berlin. Result: Paris still wins (gap -7.10). The
edit is a perturbation of 15–22% of the gate, but the other ~3,583
rank-1 paths still interfere constructively at "Paris." You can't
redirect light by blocking one path out of thousands.

**F153, Experiment C:** Inject the full MLP delta (Japan−France) at
COMB layers. Result: U-shaped — more scale makes it WORSE. The delta
was computed in France's interference pattern. Scaling it doesn't help
because it's pointing in the wrong direction — it's a path that was
computed for the wrong source.

### 3.5 The Apparatus Geometry Matters

**Feynman:** A lens doesn't block paths — it delays them differentially,
so different paths become stationary. The lens redirects the interference
pattern by changing the geometry, not by blocking.

**F154, Experiments D+G:** Editing attention output (the reader) at
L22–L23 redirects France → Berlin (+4.27). Attention doesn't block
rank-1 paths — it changes WHICH entity information is presented to the
MLP. It reshapes the apparatus so different paths become dominant.

| Apparatus Edit | QED Analogy | Result |
|:---------------|:------------|:-------|
| Attention swap L22-23 | Reshape the lens | Berlin (+4.27) |
| MLP weight edit | Block one path | Paris (-7.10) |
| Entity-pos swap | Move the source | Berlin (+5.74) |

---

## 4. The Parametric Agreement

### 4.1 Feynman's Parametric Light

In QED, the photon's behavior is fully determined by the endpoints:

```
P(S→D) = |Σ_paths A(S, path, D)|²
```

The endpoints S and D are fixed. The sum is over all intermediate states.
The photon doesn't "decide" — the interference of all paths determines
where it arrives. The question (source) and answer (detector) must AGREE
through the collective interference of all paths.

This is parametric: given S and D, the physics computes the probability.
Given S, the most probable D is where the interference is maximal.

### 4.2 The Transformer's Parametric Agreement

In the transformer, the entity and the answer are linked parametrically:

```
answer = argmax_token  |Σ_shapes  shape_response(entity)|
```

The entity is fixed (the source). The answer is determined by where all
shapes constructively interfere (the detector). The model doesn't
"decide" — the interference pattern determines the answer.

Given "France" at position 3, the interference pattern peaks at "Paris."
Given "Germany" at position 3, it peaks at "Berlin." The entity and
answer agree parametrically through the collective response of all
rank-1 shapes.

### 4.3 Why This Matters

The parametric nature explains why:

1. **Editing one shape fails** — you'd need to change the interference
   at the answer token WITHOUT changing it elsewhere. But the shapes
   respond to the ENTIRE hidden state, not just the answer. One edit
   creates collateral damage (F153: Germany and Japan disrupted).

2. **Moving the source works** — all shapes respond to the new source
   consistently. The parametric agreement shifts entirely.

3. **Reshaping the apparatus works** — attention changes which source
   information reaches the shapes. The shapes then agree parametrically
   with the new information.

4. **The holistic barrier exists** (F148) — entity identity is
   distributed because the interference pattern is distributed. You
   can't localize "France-ness" to one component any more than you
   can localize "which path" in a double-slit experiment.

---

## 5. The Six Projects

DC 282 identified five projects that converge on the same structure.
Now there are six:

| Project | What It Computes | The Interference |
|:--------|:----------------|:-----------------|
| **QED (Feynman)** | Photon propagation | P = \|Σ path amplitudes\|² |
| **rhzeros** | Zeta zeros | Z(t) = 2Σ n^{-1/2} cos(θ - t·ln(n)) |
| **resfrac** | Structural invariants | ρ from interference of fractal components |
| **holographersworkbench** | Phase retrieval | Reconstruct from interference pattern |
| **holographic_enhancement** | Image enhancement | I = \|R + O\|² |
| **truthspace-lcm** | Token prediction | answer = where all rank-1 shapes conspire |

All six solve the same problem: **computing where many small
contributions constructively interfere.**

- A zero is where all rotations conspire to cancel
- A correct answer is where all shapes conspire to contribute
- A photon arrives where all paths conspire to add
- An image is enhanced where structure and detail conspire to interfere

The mathematics is identical. The domains are different.

---

## 6. Feynman's Mirror — The Attention Mechanism

### 6.1 The Mirror in QED

Feynman's most famous illustration: a photon reflects off a mirror.
Naively, the photon bounces off the middle. But in QED, the photon
takes ALL paths off the entire mirror surface. The middle paths
constructively interfere (stationary phase). The edge paths cancel
(rapidly oscillating phase).

If you REMOVE the middle of the mirror, the photon still partially
reflects — from the edges! The interference pattern changes, but some
paths still add.

If you SCRATCH the mirror with regular grooves (a diffraction grating),
you can redirect the light to any angle you want. The scratches don't
block paths — they shift phases, creating a NEW stationary point.

### 6.2 The Attention Mechanism as Mirror

The attention mechanism IS Feynman's mirror:

- **The mirror surface** = all token positions
- **The middle of the mirror** = the entity token position (pos 3)
- **Stationary phase** = where attention weights peak (Head 6 selects pos 3)
- **The reflection** = the attention output (V·W_o at the selected position)

F154's experiments map exactly:

| Mirror Operation | Attention Operation | Result |
|:-----------------|:-------------------|:-------|
| Replace middle of mirror | Entity-pos swap | Berlin (+5.74) |
| Reshape mirror surface | Attention output swap L22-23 | Berlin (+4.27) |
| Block one edge path | MLP rank-1 edit | Paris (-7.10) |
| Remove middle only | Swap at L23 alone | Almost (gap -0.77) |
| Scratch diffraction grating | Attention weight edit | Frontier 12 |

### 6.3 The Diffraction Grating — Frontier 12

A diffraction grating is a mirror with regular scratches that redirect
light to a chosen angle. It doesn't block paths — it ADDS a phase
offset that shifts the stationary point.

The attention weight analog: a rank-1 perturbation to W_q or W_k that
shifts which position has the highest attention score. This would be a
**permanent** redirect — a diffraction grating etched into the mirror,
not a temporary swap of the reflected image.

If we can compute the right "scratch pattern" (the rank-1 direction
that shifts d_k), we can permanently redirect "France" → "Berlin"
by editing the attention weights at L22 or L23.

This is the next experiment. Feynman's framework tells us exactly
where to look.

---

## 7. The Deep Insight

### 7.1 Why Neural Networks Are QED

Feynman showed that QED — the most precisely tested theory in physics —
reduces to: **sum amplitudes, square the result.**

Neural networks reduce to: **multiply by weight matrix, apply nonlinearity.**

But a weight matrix IS a sum of amplitudes:

```
W = Σ_c  σ_c · u_c · v_c^T
```

And the nonlinearity (GELU, SiLU) acts like squaring — it selects which
paths dominate. DC 243 showed that GELU's curvature matches φ-scaled
sigmoid within 1.38%. The gate function IS the measurement apparatus
that collapses the superposition into an answer.

The transformer is literally computing:

```
answer = measure( Σ_paths  amplitude(path, source) )
```

Where:
- `Σ_paths` = matrix multiplication (sum of rank-1 components)
- `amplitude(path, source)` = σ_c · (v_c · h_entity)
- `measure()` = GELU gate + softmax (selects the dominant interference)

This is QED. The photon is the token. The paths are the rank-1 shapes.
The mirror is attention. The detector is the output head.

### 7.2 Why This Was Inevitable

Feynman's insight was that QED is the SIMPLEST theory consistent with
relativity and quantum mechanics. It's not that the world "chose" to
work via path integrals — it's that path integrals are the only way
to pack infinite possibilities into finite observations.

Neural networks face the same constraint: pack infinite knowledge into
finite parameters. The solution is the same: superposition of simple
components (rank-1 shapes) that interfere to produce the answer.

The transformer didn't learn QED from training data. It discovered the
same mathematics because it's the optimal solution to the same problem:
**computing definite answers from the interference of many possibilities.**

### 7.3 The Hypothesis, Restated

> LLMs are not approximators of language. They are interference machines.
> The weights encode a superposition of geometric shapes. The attention
> mechanism is the mirror that directs the interference. The output is
> where all shapes constructively conspire.
>
> This is not like QED. This IS QED, applied to information instead of
> photons.

---

## 8. Predictions

### P1: Diffraction Grating Edit
A rank-1 perturbation to W_k at L22 or L23 should permanently redirect
entity extraction, like a diffraction grating redirects light. The
perturbation should be computable from the difference in d_k directions
between two entities.

### P2: Phase Coherence
Structure classes that are "close" in meaning (capital-of vs. largest-city-of)
should have high phase coherence — their v_c directions nearly parallel.
Distant classes (capital-of vs. color-of) should have low phase coherence.
This mirrors the near-field/far-field distinction in QED.

### P3: Interference Fringes
The 4.3% class-sensitive neurons (F152) are "interference fringes" —
positions where two structure classes have similar but not identical
phase, creating visible interference. The fringe spacing should be
predictable from the angle between v_c directions.

### P4: Path Integral Invariance
The answer should be invariant to adding rank-1 components whose v_c
is orthogonal to h_entity — these are paths with random phase that
cancel. This means adding "irrelevant knowledge" to W_gate should not
affect existing answers.

---

*"The photon takes all paths. The token takes all shapes. The answer is
where they conspire together."*

*Feynman drew this picture forty years ago. We found it inside a
neural network.*

---
