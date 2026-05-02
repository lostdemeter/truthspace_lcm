# DC 304 — Transformers Discover Geometry, Not Understanding

*Date: March 2026*
*Follows from: DC 303 (attention IS context gravity)*

---

## The One Sentence

> The transformer doesn't "understand context."
> It discovers the geometry of co-occurrence through training,
> then applies it relationally via softmax competition.
> Our IRD gravity discovers the same geometry analytically
> and applies it in a single step.
> Same direction, different mechanism, different magnitude.

---

## 1. What Transformers Actually Learn

When a transformer is trained on a corpus, gradient descent minimises prediction
loss. To minimise prediction loss, the model must capture which tokens tend to appear
near which other tokens — the co-occurrence structure of language.

The attention weights (Q, K projections) are the model's encoding of that
co-occurrence geometry. A high Q·K score between two positions means those tokens
tend to be informationally relevant to each other in the training distribution.

But this is not understanding. It is **pattern compression**.

The model does not know that "cookie" can mean a baked good or an HTTP artefact.
It knows that cookie-tokens appear in contexts containing [recipe, flour, bake] and
also in contexts containing [browser, session, login], and that these two clusters
are geometrically distant in the high-dimensional space the weights encode. When
asked to process "cookie" in a new sentence, it resolves which cluster applies by
checking which context tokens are present — a softmax competition.

The geometry was always in the corpus. The transformer extracted it.

---

## 2. The IRD Path to the Same Geometry

Our IRD (Iterative Residual Decomposition) axes are computed by SVD of a concept
similarity matrix built from Qwen2-7B embeddings. Those embeddings are themselves
products of training — they encode the same co-occurrence geometry. The SVD then
finds the principal axes of that geometry: the directions along which concepts vary
most systematically.

Two paths to the same place:

```
Corpus co-occurrence structure
         │
         ├── via backpropagation → Transformer Q,K weights → applied by softmax
         │
         └── via SVD of embeddings → IRD axes → applied by geometric gravity
```

Neither path "creates" the geometry. Both paths **discover** the geometry that was
always latent in the corpus.

The transformer's Q and K matrices are a lossy, noise-contaminated compression of
the same signal that SVD extracts cleanly. The transformer's 3584-dimensional
attention space contains the same information as our 500-dimensional IRD space,
plus a large amount of statistical noise introduced by stochastic gradient descent,
mini-batch variance, and finite model capacity.

---

## 3. Mechanism Difference: Relational vs Absolute

The critical implementation difference is *how* the geometry is applied.

**Transformer (relational):**
The softmax normalises over ALL tokens in the current window. No token is attended
to in absolute terms — only relative to the other tokens present. A token's
attention weight is not a function of its proximity to the query; it is a function
of its proximity *compared to every other token*. The window creates a competition,
and the winner(s) reshape the query's representation.

This is why raw Q·K^T logits (DC 303 §3) showed near-zero discrimination: without
the softmax competition, the absolute logit for (cookie, recipe) ≈ (cookie, login)
≈ 153.4. The discrimination only emerges when both appear in the same sentence and
compete for attention mass.

**IRD gravity (absolute):**
Our `context_correct_proj()` applies a weighted sum of displacement vectors from
each context word. There is no competition between context words — each contributes
its gravity independently. The total shift is the sum of individual contributions,
scaled by distance-dependent falloff.

This is the correct mechanism when context words are already selected (i.e., when
you know what the context is). It is the wrong mechanism when you need to *identify*
which context words are relevant from a larger pool — because that identification
step requires the relational softmax.

**The division of labour:**

```
Context identification → softmax competition (relational, requires a window)
Context application    → geometric gravity (absolute, can be applied directly)
```

A complete geometric replacement for attention requires BOTH steps. DC 303 verified
that our gravity correctly performs the application step. The identification step —
how to select the relevant subset of a long document — is the remaining open problem.

---

## 4. Magnitude Difference: Iterative vs Single-Pass

The transformer applies 28 successive refinement steps. Each layer slightly adjusts
each token's representation based on the current state of all other tokens. By L27,
the accumulated adjustment has shifted "cookie" by 0.035 cosine units in food-space.

Our single-pass gravity shifted "cookie" by 0.115 cosine units in IRD space — a
larger magnitude in one step (with α = 0.5).

Two interpretations:

**Interpretation A — overcorrection:**
Our single-pass gravity overshoots because it applies the full correction at once.
The transformer distributes the same correction across 28 layers to avoid
instability. Reducing α would bring the magnitude closer to the transformer's
per-step correction.

**Interpretation B — different target:**
The transformer's 0.035 shift is not the "correct" final answer — it is the answer
after 28 layers with that specific parameterisation. A transformer trained with
different weights might shift by 0.12 or 0.005. The direction matters; the magnitude
is a hyperparameter of the training process. Our 0.115 shift is also a
hyperparameter of our α. Both are valid; they are not converging on the same number.

**The experiment to distinguish them** (Q1 in DC 303 §10): apply gravity
iteratively until convergence. If the magnitude converges to ≈ 0.035, Interpretation
A is correct. If it converges to some other value (and the direction holds), the
transformer's magnitude is incidental.

---

## 5. What This Means for the TruthSpace Hypothesis

The TruthSpace hypothesis claims: structure IS information, geometry IS computation.

DC 303 and this document together establish a specific, testable instance of this:

**Contextualisation IS displacement in geometric space.**

The "meaning" of "cookie" in a sentence is not a new object — it is the original
cookie vector displaced by geometric gravity from context words. There is no
learned contextualisation weight, no trained disambiguation module. The geometry
that the transformer spent billions of parameter-updates discovering was always
derivable analytically from the structure of the concept space.

This is the fail-fast principle applied to transformers: the transformer's
attention mechanism is not a clever architectural innovation that enables context.
It is the most direct implementation of gravity that gradient descent could find
given the constraint of differentiability. Remove that constraint, and you get
the same result more directly.

---

## 6. The Implication That Cannot Be Overstated

If geometry IS computation, then:

1. **Training is not knowledge acquisition.** Training is geometry discovery.
   The knowledge was always in the corpus. The model is a compressed readout
   of the corpus geometry.

2. **Weights are not the intelligence.** The geometry is the intelligence.
   Weights are one representation of the geometry — noisy, redundant, and opaque.
   The IRD axes are another representation — cleaner, interpretable, and compact.

3. **Larger models are not smarter.** They are better at extracting the same
   geometry. A 7B model extracts the corpus geometry more faithfully than a 1.5B
   model. But neither model created any geometry — both are compression algorithms
   applied to the same underlying geometric structure.

4. **Emergent capabilities are geometric phase transitions.** When a model
   becomes large enough to exhibit in-context learning, chain-of-thought reasoning,
   or tool use, it is not because the model learned a new capability. It is because
   the model's geometry became faithful enough to the corpus geometry that
   multi-hop traversal of the semantic manifold became reliable. The capability
   was latent in the corpus geometry; the model finally became a good enough
   approximation to express it.

5. **Our system does not need to discover geometry.** We have the geometry.
   What we build is not a substitute for training — it is a shortcut past training
   to the underlying structure that training was trying to find.

---

## 7. The Ground Truth

The ground truth — the corpus geometry — is unknowable in perfect form. No SVD
of any finite sample, no embedding model trained on any finite corpus, captures
it perfectly. Both the transformer and our IRD space are approximations.

But they are approximations of the same thing. When they agree — as they did in
DC 303 for both cookie and bass — it is not coincidence. It is two imperfect
instruments pointing at the same signal.

When they disagree — as they will — the disagreement is diagnostic. Either:
- The IRD space is missing structure that the transformer captured (needs more axes)
- The transformer learned a spurious pattern not present in the geometry (overfitting)
- The comparison methodology is inadequate

Every disagreement is an experiment. Every experiment reduces uncertainty about
the shape of the true underlying manifold.

---

## 8. Summary

| Claim | Evidence |
|---|---|
| Transformers discover geometry, not understanding | Q·K^T weights = IRD axes = both derived from corpus co-occurrence |
| Same geometry, different readout mechanism | DC 303: both make same directional prediction for polysemy |
| Relational (softmax) vs absolute (gravity) is the key mechanism difference | Raw logits can't discriminate pairs; in-context softmax can |
| Single-pass gravity approximates 28-layer attention | Direction matches; magnitude is a hyperparameter |
| Training is geometry discovery, not knowledge creation | Geometry is in the corpus; weights are a (noisy) compression of it |

---

## 9. What Comes Next

Three open questions follow directly. Each is now an experiment, not a speculation:

**Q1.** Does iterative IRD gravity converge to the transformer's hidden state?
**Q2.** Can geometric softmax competition reproduce context *selection* (not just application)?
**Q3.** Does bidirectional mutual gravity on a full sentence converge to a
         geometrically coherent representation?

These are DC 305 through DC 307.
