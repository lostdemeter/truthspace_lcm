# DC 358: Activation Space vs Token Space — Relational Encoding

**Day 185 | W_E direction is the most stable signal for TYPE_BC; activation
space degrades with depth; transformer creates proximity (not direction) for TYPE_A**

---

## Overview

Day 184 compared three retrieval strategies across five transformer layers
(L0=W_E, L8, L16, L24, L27) for three relational domains.

**Core finding:**

> **W_E directional encoding (Layer 0) is the MOST STABLE and MOST RELIABLE
> signal for TYPE_BC (structural) relations. Activation-space direction degrades
> monotonically with transformer depth. For TYPE_A (proximity) relations, the
> transformer creates semantic proximity in activation space by L8, but this is
> indistinguishable from direct proximity measurement.**

---

## Experimental Design

Three retrieval strategies:
```
Exp A: W_E direction   + W_E query   → snap in W_E   (classical W_E retrieval)
Exp B: Act direction   + Act query   → snap in Act   (activation-space retrieval)
Exp C: W_E direction   + Act query   → snap in Act   (cross-space retrieval)
```

Five layers: L0 (= W_E embeddings), L8, L16, L24, L27 (final)

Three domains: capitals (TYPE_BC), languages (TYPE_BC), antonyms (TYPE_A)

---

## Results

```
           Exp     L0(W_E)   L8      L16     L24     L27
───────────────────────────────────────────────────────────────────
capitals   A(W_E)   0.900   0.900   0.900   0.900   0.900  ← STABLE
capitals   B(act)   0.900   0.600   0.500   0.500   0.300  ← DEGRADES
capitals   C(mix)   0.900   0.600   0.500   0.500   0.400  ← DEGRADES
capitals   prox     1.000   0.600   0.500   0.500   0.400  ← DEGRADES

languages  A(W_E)   1.000   1.000   1.000   1.000   1.000  ← STABLE (perfect)
languages  B(act)   1.000   1.000   0.750   0.750   0.875  ← minor dips
languages  prox     1.000   1.000   0.750   0.750   0.875  ← B ≡ prox

antonyms   A(W_E)   0.000   0.000   0.000   0.000   0.000  ← ABSENT (known)
antonyms   B(act)   0.000   0.571   0.429   0.429   0.429  ← EMERGES at L8
antonyms   C(mix)   0.000   0.571   0.429   0.429   0.429  ← same as B
antonyms   prox     1.000   0.571   0.429   0.429   0.429  ← B ≡ prox
```

---

## Finding 1: W_E Direction Is Layer-Invariant for TYPE_BC

For capitals, Experiment A (W_E direction in W_E space) returns exactly 0.900
at every layer. This is not a coincidence — the direction is computed entirely
from W_E embeddings (static, layer 0) and the snap is done in W_E space.
Layer selection is irrelevant to Experiment A.

The point of comparing across layers is to show that **W_E retrieval is
independent of which transformer layer the model has reached**. No matter
how "deep" into processing a query has gone, the W_E-based direction still
works because it operates on the fixed Layer 0 weights, not the dynamic
hidden states.

**Implication:** W_E-based relational retrieval can be performed as a
pre-processing step before any transformer computation.

---

## Finding 2: Activation Direction Degrades With Depth

For capitals (TYPE_BC), activation-space retrieval (Exp B) starts at 0.900
at L0 (same as W_E) but degrades:

```
L0:  0.900  (identical to W_E)
L8:  0.600  (−30%)
L16: 0.500  (−40%)
L24: 0.500
L27: 0.300  (−60%)
```

**Mechanism:** The transformer layers apply attention and MLP transformations
that re-mix the token representations. Tokens are no longer positioned in
W_E's relational coordinate system — they are positioned according to the
current context and task. The country→capital displacement that was clean
and consistent in W_E becomes scrambled by context-sensitive transformations.

The same degradation occurs for the cross-space experiment (Exp C):
applying a W_E direction to an activation-space query also degrades,
because the activation space has been transformed away from W_E geometry.

**Implication:** DO NOT use transformer hidden states as W_E substitutes
for relational retrieval. The W_E embedding matrix is the correct reference.

---

## Finding 3: TYPE_A Relations Are Created by the Transformer

Antonyms have zero direction in W_E (Exp A = 0.000 at all layers — confirmed
from Day 162). But in activation space (Exp B), a signal EMERGES at Layer 8:

```
L0:  0.000  (no signal)
L8:  0.571  (signal emerges!)
L16: 0.429
L24: 0.429
L27: 0.429
```

However, the activation proximity (prox row) gives IDENTICAL values:
```
prox L8 = 0.571 = B(act) L8
prox L16 = 0.429 = B(act) L16
```

This means the activation-space "direction" for antonyms is identical to
direct proximity measurement. **The transformer does not create a directional
encoding for antonyms — it creates PROXIMITY.** By Layer 8, hot and cold are
near each other in activation space, making any method that moves toward
"the other antonym word" equivalent to just finding the nearest neighbor.

**What the transformer is doing:** Processing the word "hot" in isolation
activates the word's contrastive semantic features. The attention mechanism
retrieves context from training data where "hot and cold" co-occurred, pulling
the activation toward the semantic neighborhood of temperature contrasts.
The result is that "hot"'s activation is pulled toward "cold" in activation space.

This is NOT what W_E does: W_E[hot] is not close to W_E[cold] in a directional
sense (no consistent Δ across antonym pairs). The proximity is created DYNAMICALLY
by the transformer, not statically encoded.

---

## The Two Layers of Semantic Knowledge

This experiment reveals two distinct stores of semantic knowledge:

```
LAYER 0 (W_E) — Static Relational Map:
  ┌─────────────────────────────────────────────────────┐
  │ Encodes: structural relations from asymmetric        │
  │ syntactic patterns in training text                  │
  │                                                      │
  │ TYPE_BC: country→capital, country→language           │
  │ TYPE_A:  antonyms NOT encoded (proximity, not dir)   │
  │ THEMATIC: animal→sound NOT encoded at all            │
  │                                                      │
  │ Properties: stable, layer-independent, pre-computable│
  └─────────────────────────────────────────────────────┘

LAYERS 1-28 (Activation Space) — Dynamic Context Map:
  ┌─────────────────────────────────────────────────────┐
  │ Encodes: task-specific, context-sensitive positions  │
  │ shaped by attention over training co-occurrences     │
  │                                                      │
  │ TYPE_A: antonym proximity created by L8              │
  │ TYPE_BC: structural directions DEGRADED by depth     │
  │                                                      │
  │ Properties: volatile, context-dependent, dynamic     │
  └─────────────────────────────────────────────────────┘
```

**The transformer layers do not IMPROVE W_E's relational encoding —
they TRANSFORM it for contextual processing. W_E is the best space
for structural (TYPE_BC) relations; activation space is better for
contextual (TYPE_A) relations only in the proximate sense.**

---

## Implication for TruthSpace Architecture

```
QUERY: "What is the capital of France?"

STEP 1 — Classify relation type (H1, H2, H2_cv from training pairs):
  capital relation → TYPE_BC

STEP 2 — Retrieve in W_E space (NOT activation space):
  query = W_E[France]
  direction = mean_W_E_direction_for_capital_relation
  candidate = snap(query + direction, W_E vocabulary)
  → "Paris"

STEP 3 (only if W_E fails or domain is TYPE_A):
  Use proximity retrieval in W_E (for TYPE_A)
  OR pass to full transformer for THEMATIC/absent domains

NEVER step 2b: apply W_E direction to activation vector and snap in act space
  → This degrades accuracy monotonically with transformer depth
```

---

## Cross-Space Compatibility

One might hope that W_E directions could be applied directly to activation
vectors (Exp C) to combine the stability of W_E directions with the contextual
richness of activations. The results show this does NOT work:

```
capitals Exp A (W_E→W_E): 0.900 (best)
capitals Exp C (W_E→Act): 0.600 at L8, 0.400 at L27 (worse)
```

The W_E direction vector lives in the same 1536-dimensional space as the
activation vectors. But the metric structure is different: the activation
space has been non-linearly transformed such that the W_E direction no longer
points toward the correct target.

**Analogy:** W_E directions are street addresses in a city. The transformer
layers warp the city map for navigation purposes. Adding the correct W_E
street address to a warped-map position does not land you at the right place.

---

## Summary Table

| Relation type | Best method | Layer | Accuracy | Notes |
|---|---|---|---|---|
| TYPE_BC (capitals) | W_E direction | any | 0.900 | Stable, independent of depth |
| TYPE_BC (languages) | W_E direction | any | 1.000 | Perfect, stable |
| TYPE_A (antonyms) | W_E proximity | L0 | 1.000* | *restricted vocab |
| TYPE_A (antonyms) | Act proximity | L8 | 0.571 | Full vocab proxy |

*Restricted vocabulary result; in full vocabulary, antonym proximity is lower

---

## Files

- `expedition_day184_activation_vs_token.py` — three-experiment five-layer comparison
- `day184_activation_vs_token.json` — results
- `357_relational_boundary_revised.md` — encoding type classifier
- `355_multihop_chains.md` — chain depth and viability
