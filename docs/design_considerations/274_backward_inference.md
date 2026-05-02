# DC 274: Backward Inference — From Answer to Path

**Status**: Active — designing framework and first experiments
**Date**: 2025-02-27
**Depends on**: DC 272 (R-S Sum), DC 273 (Memory Editing), F114–F118
**Core question**: Can we start with an answer and work backwards to
discover, constrain, or CREATE the paths that lead to it?

---

## 1. The Insight

Normal inference: prompt → model → answer.
We've been doing this. It works.

But what if we invert it?

**Backward inference**: answer → model⁻¹ → viable paths.

Finding 118 already showed a hint of this: the LM head row W_lm[k]
IS both the forward readout direction AND the backward injection
direction. ENCODE = DECODE. The same vector works in both directions.

What if this principle extends deeper than the LM head?

---

## 2. The Framework

### 2.1 The Answer Cone

The final hidden state h must satisfy:

```
argmax(W_lm @ rms_norm(h)) = answer_token
```

Since rms_norm preserves direction (it only normalizes magnitude),
this is equivalent to:

```
For all k ≠ answer:
  W_lm[answer] · dir(h) > W_lm[k] · dir(h)
```

where dir(h) = h/‖h‖. This defines the **answer cone**: the region
of directions in ℝ³⁵⁸⁴ that decode to a specific token.

The answer cone is the intersection of ~152K half-spaces on the
unit sphere S³⁵⁸³. But most tokens are far from the answer in
hidden space, so the vast majority of constraints are slack.
The effective constraints come from the nearest neighbors in
vocabulary space.

### 2.2 Path Decomposition

The final hidden state is a sum:

```
h_final = h_embed + Σ_{L=0}^{27} (attn_L + mlp_L)
```

From Finding 117, we know each attn_L is a V·W_o output, and
~5 dominant layers determine the answer. So:

```
dir(h_final) ∈ answer_cone(token)
  ↔ h_embed + Σ attn_L + Σ mlp_L  points into the cone
  ↔ the SUM of 28 layer contributions reaches the target direction
```

Starting from the answer, we can ask: **what decompositions of the
answer direction into layer-wise terms are consistent with the
model's weights?**

Each layer's attn_L is constrained to lie in range(W_o_L), and
each mlp_L is constrained to lie in range(W_down_L). So not every
decomposition is achievable — the model's weights define which
paths exist.

### 2.3 The Reduction Principle

The user's key insight: **a known answer reduces the viable paths.**

If we know the output must be "Paris", then:
- h_final must lie in the Paris cone
- Most of ℝ³⁵⁸⁴ is NOT in the Paris cone
- Each layer's contribution must conspire to land in the cone
- This dramatically constrains what the intermediate states can be

The stronger the answer constraint (i.e., the narrower the cone),
the fewer paths are viable. This is information compression:
the answer IS a constraint on the computation.

---

## 3. Three Levels of Backward Inference

### Level 1: Answer → Path Verification

Given: a known answer (e.g., "Paris")
Find: which forward paths actually reach it

This is the simplest case. We already have the tools:
- Extract per-layer V·W_o for known prompts (F117)
- Project onto W_lm[Paris] to measure contribution
- Identify which layers/heads contribute most

**What we learn**: the "anatomy" of an answer — how it's assembled
from 28 additive terms.

### Level 2: Answer → Path Discovery

Given: a known answer (e.g., "Paris")
Find: what OTHER prompts could produce it

This inverts the question. Instead of "what does this prompt predict?",
we ask "what prompts predict this answer?"

Approach: project W_lm[Paris] back through W_o^{-1} at each layer
to find what V states (and therefore what input positions) would
produce a contribution toward Paris.

**What we learn**: the "input manifold" — the set of all prompts
that could lead to a given answer.

### Level 3: Structure → Knowledge Extension

Given: the geometric structure of known facts
Find: where UNKNOWN facts should live

This is the deepest level. If we know:
- France → Paris lives at position P_france in hidden space
- Japan → Tokyo lives at position P_japan
- Germany → Berlin lives at position P_germany

Then the "capital city" relationship defines a **manifold** in
hidden space. The structure of this manifold — its curvature,
its dimensionality, its relationship to the d_k axis — tells us
where a NEW capital should live, even if the model was never
trained on it.

```
Known:   {(country_i, capital_i)} → manifold M
Novel:   country_new → M predicts capital_new's location
```

This is **geometric knowledge completion**: using the shape of
known facts to predict unknown ones.

---

## 4. The Mathematical Structure

### 4.1 Fact Manifold

For a set of facts of the same type (e.g., capital cities),
extract the per-layer attention deltas:

```
Δ_L(fact_i) = attn_L(fact_i) - attn_L(baseline)
```

These deltas live in ℝ³⁵⁸⁴ for each layer. Stack them:

```
F_i = [Δ_0(i), Δ_1(i), ..., Δ_27(i)] ∈ ℝ^{28 × 3584}
```

The set {F_i} for all known facts of this type defines a manifold.
Its principal components tell us:
- How many "degrees of freedom" a fact type has
- Which dimensions vary (fact-specific) vs stay constant (structural)
- The geometric template for this fact type

### 4.2 Structural Template Extraction

Given facts F_1, ..., F_n, decompose:

```
F_i = T + S_i
```

Where T is the shared structural template (mean or PCA basis)
and S_i is the fact-specific deviation.

If the S_i are low-dimensional (e.g., rank 1-3), then the
entire fact type is parameterized by a few numbers. A new fact
needs only those few parameters to be placed on the manifold.

### 4.3 Answer-Guided Path Construction

Given a target answer token and a fact type template:

1. Get W_lm[answer] — the target direction
2. Get template T — the structural scaffold for this fact type
3. Solve for S_new: the fact-specific parameters that make
   T + S_new project onto W_lm[answer]
4. This gives the per-layer deltas needed to produce the answer
5. Inject these deltas — the model now "knows" the new fact

This is construction, not search. We're not looking for a path —
we're BUILDING one from the structural template and the answer
constraint.

### 4.4 Knowledge Extension Without a Known Answer

The most radical case: we don't know the answer, but we know the
STRUCTURE should admit one.

Example: "The capital of Elbonia is ___"

Elbonia is fictional. There is no correct answer. But:
1. The "capital city" manifold has a specific shape
2. "Elbonia" has an embedding (or we can construct one)
3. The manifold's structure PREDICTS what the answer's hidden
   state should look like
4. We can find the nearest vocabulary token to that predicted state
5. The model "invents" a plausible capital

This is the model reasoning geometrically: using structural
knowledge to generate novel but structurally consistent answers.

---

## 5. Connection to the R-S Sum

The Riemann-Siegel sum has a known analytic structure:

```
ζ(s) = Σ_{n=1}^{N} n^{-s} + χ(s) Σ_{n=1}^{N} n^{s-1} + R
```

The main sum, the correction term, and the remainder. If we know
ζ(s) (the answer) and χ(s) (the functional equation), we can
CONSTRAIN the individual terms. Not all decompositions are valid —
only those consistent with the number-theoretic structure.

In the transformer:
- The answer constrains h_final
- The model weights constrain each layer's contribution
- The R-S structure constrains how layers combine
- Together, these may uniquely determine the path

If the path is uniquely determined, then the answer IS the path.
ENCODE = DECODE at the level of the entire computation, not just
the LM head.

---

## 6. Experimental Plan

### Phase 10z18a: Answer Anatomy
- Take 6 known capital facts
- For each, extract per-layer attention deltas
- Project each delta onto W_lm[answer] to measure contribution
- Compare the "anatomy" across facts — is the structure consistent?

### Phase 10z18b: Fact Manifold Extraction
- Stack the per-layer deltas for all 6 facts
- PCA to find the manifold dimensionality
- Extract template T and fact-specific deviations S_i
- How many dimensions does "capital city" need?

### Phase 10z18c: Backward Path Construction
- Start with W_lm[answer] for a HELD-OUT fact
- Use the template to construct per-layer deltas
- Inject the constructed deltas
- Does the model predict the correct answer?

### Phase 10z18d: Knowledge Extension
- Construct a path for a completely novel fact
- Not by injecting W_lm[token] directly (F118 already did that)
- But by using the STRUCTURAL TEMPLATE to predict where the
  answer should be, then finding what token lives there

---

## 7. Why This Matters

If backward inference works:

1. **Knowledge becomes navigable**: Instead of probing the model
   forward, we can map the space of possible answers and trace
   paths to each one.

2. **Structure generates knowledge**: The geometric template for
   a fact type contains enough information to PREDICT new facts.
   Training data provides examples; structure provides the rule.

3. **ENCODE = DECODE at all scales**: Not just the LM head, but
   the entire computation is invertible. The transformer is not
   a one-way function — it's a geometric structure that can be
   traversed in either direction.

4. **The hypothesis is testable**: If backward-constructed paths
   produce correct answers, the geometric structure IS the
   knowledge. If they don't, we learn where the hypothesis breaks.

---

*This document will be updated as experiments progress.*
