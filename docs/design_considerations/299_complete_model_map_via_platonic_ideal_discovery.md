# DC 299: Complete Model Map via Platonic Ideal Discovery

## Status: PLAN — strategic roadmap for full LLM cartography
## Date: 2026-03-08
## Depends on: DC 298 (TruthSpace Is Real), DC 289 (Concept Composition), DC 277 (Geometric Instrument)

---

## Executive Summary

DC 298 proved that concepts in embedding space are compounds of binary truth
axes ("platonic ideals"). We identified 6 such axes, explaining 9.1% of
embedding variance. PCA analysis reveals ~79 dimensions capture 95% of concept
variance, implying ~79 platonic ideals exist in total.

**This document is the plan to find all of them.**

If successful, this would be the first complete geometric map of an LLM's
conceptual space — a finite, interpretable, verifiable coordinate system
where every concept is a binary address and every relationship is a coordinate
transformation.

---

## What We Know

### Established Facts (DC 298)

| Fact | Value |
|------|-------|
| Embedding dimension | 3584 |
| Known truth axes | 6 |
| Variance explained by 6 axes | 9.1% |
| PCA dims for 50% variance | 27 |
| PCA dims for 90% variance | 71 |
| PCA dims for 95% variance | 79 |
| PCA dims for 99% variance | 86 |
| Concepts tested | 88 |
| Vocab size | ~152K tokens |
| Residual self-retrieval | 88/88 at rank 0 |

### What This Tells Us

1. The concept space is **finite-dimensional** (~79-86 effective dims)
2. Our 6 manual axes are **half as efficient** as optimal PCA directions
   (6 axes → 9.1% vs 6 PCA dims → 19.7%)
3. The residual after projection is **structured, not noise**
4. ~79 well-chosen axes would capture 95% of what makes concepts distinct

### The Gap

We have 6 out of ~79. That's 7.6% of the map. The question is: can we
automate discovery of the remaining ~73?

---

## The Plan

### Phase 0: Infrastructure — Expand the Concept Vocabulary

**Goal:** Go from 88 concepts to thousands.

**Why:** 88 concepts in 3584 dimensions is severely underdetermined. To discover
79 independent axes, we need far more concepts than dimensions of interest.
Statistical rule of thumb: at least 5-10x the number of axes, so 400-800
concepts minimum.

**Method:**
1. **Systematic vocabulary mining** — Use the existing `VocabSearcher` to find
   clean, single-token concepts across many categories:
   - Animals (dog, cat, lion, eagle, whale, ...)
   - Colors (red, blue, green, ...)
   - Professions (doctor, teacher, soldier, ...)
   - Materials (wood, metal, glass, ...)
   - Emotions (happy, sad, angry, ...)
   - Food (bread, rice, fish, ...)
   - Body parts (hand, eye, heart, ...)
   - Abstract concepts (truth, justice, freedom, ...)
   - Numbers (one, two, three, ...)
   - Time (morning, night, summer, winter, ...)
   - Actions/verbs as nouns (run, jump, fight, ...)
   - Sizes/qualities (big, small, fast, hot, ...)

2. **Filtering criteria:**
   - Must be a single token (no subword fragments)
   - Must have embedding norm within 2σ of mean (no degenerate tokens)
   - Must be a "real word" (filter out code tokens, punctuation, etc.)
   - Prefer common English words (high frequency in training data)

3. **Target: 500-1000 clean concept tokens across 20+ semantic categories**

**Validation:** Verify that the expanded set reproduces the 6 known axes with
similar accuracy. If the axes break with more concepts, something is wrong
with our concept selection, not the theory.

**Estimated effort:** 1 session. Mostly scripting + manual curation.

---

### Phase 1: Automated Axis Discovery — Iterative Residual Mining

**Goal:** Discover all ~79 platonic ideals algorithmically.

**Why:** Manual definition of binary properties ("is European", "is female")
doesn't scale. We need an algorithm that discovers axes from the data.

**Core Algorithm: Iterative Residual Decomposition (IRD)**

```
INPUTS:
  E = matrix of concept embeddings (N_concepts × 3584)
  known_axes = initial set of truth axis directions

REPEAT:
  1. Project E onto all known_axes, compute residuals R
  2. Perform SVD on R → get top singular vectors
  3. For each top singular vector v:
     a. Project all concepts onto v
     b. Look for BINARY SEPARATION — a threshold that splits
        concepts into two semantically coherent groups
     c. If binary: name it, add to known_axes
     d. If not binary: it may be continuous — flag for review
  4. Orthogonalize known_axes (Gram-Schmidt)
  5. Measure total variance explained

UNTIL:
  - Variance explained > 95%, OR
  - No new binary axes found in residuals
```

**Key Insight:** Each iteration removes the variance from discovered axes,
so the residual SVD naturally surfaces the *next most important* undiscovered
axis. This is analogous to how PCA works, but with an interpretability step
(checking for binary separation) injected at each round.

**The Binary Test:**
For each candidate axis direction `v`:
1. Project all concepts: `scores = E @ v`
2. Sort scores, look for a natural gap (bimodal distribution)
3. Split at the gap → two groups
4. Check: are the groups semantically coherent?
   - All animals on one side, no animals on the other → "is_animal"
   - Mixed groups with no clear meaning → not a platonic ideal
5. Cross-validate: hold out concepts, predict membership

**Naming Axes:**
- Automatic: inspect which tokens have highest/lowest projection
- The positive pole tokens and negative pole tokens should suggest the property
- Human review for final naming (but discovery is automated)

**Estimated effort:** 2-3 sessions. Core algorithm + iteration + review.

---

### Phase 2: Validation — Verifying Each Axis Is a True Platonic Ideal

**Goal:** Ensure each discovered axis is a genuine binary truth, not a
statistical artifact.

**Criteria for a valid platonic ideal:**

1. **Binary separation** — There must be a clear threshold that splits
   concepts into two groups with >90% accuracy (LOO cross-validated)

2. **Semantic coherence** — A human can name what the two groups represent,
   and the name is a verifiable binary property

3. **Orthogonality** — The axis must be approximately orthogonal to all
   previously discovered axes (|cos θ| < 0.3)

4. **Generalization** — The axis must correctly classify held-out concepts
   not used during discovery

5. **Stability** — The axis must be reproducible across different random
   subsets of concepts (bootstrap stability test)

6. **Vocabulary consistency** — When projecting the FULL vocabulary (152K
   tokens) onto the axis, the top/bottom tokens should be semantically
   consistent with the axis label

**Reject if:**
- The axis only works for the specific concepts used to discover it
- The axis is a linear combination of existing axes (redundant)
- The separation is continuous, not binary (may be a "spectrum" not an "ideal")

**Note on continuous axes:** If we find axes that are clearly meaningful but
NOT binary (e.g., "size" from small→large), these are interesting but NOT
platonic ideals in our framework. We should document them separately as
"spectra" — they may represent a different kind of geometric structure.

**Estimated effort:** 1-2 sessions. Automated testing + human review.

---

### Phase 3: The Gödel Address Book — Complete Truth-Coordinate Assignment

**Goal:** Assign every concept a complete binary coordinate vector over all
discovered platonic ideals.

**Method:**
For each concept `c` and each axis `a`:
1. Compute projection: `score = embedding(c) · direction(a)`
2. Compare to threshold for axis `a`
3. Assign coordinate: 1 if above threshold, 0 if below

**Result:** An N_concepts × N_axes binary matrix — the "address book"

**Validation tests:**

1. **Uniqueness** — Each concept should have a unique binary address.
   If two concepts share an address, either:
   - We're missing an axis that distinguishes them, OR
   - They are genuine synonyms in the model's representation

2. **Reconstruction** — Using only the binary coordinates and axis directions,
   reconstruct each concept embedding:
   `reconstructed(c) = Σ coordinate(c,a) × direction(a)`
   Compare to original. The residual should now be small (< 5% of variance).

3. **Relationship preservation** — Applying a known relationship delta
   should flip exactly the expected coordinates and preserve the rest
   (extending the DC 298 composition test to all axes)

4. **Completeness** — The address book should be sufficient to distinguish
   any two semantically different concepts

**Estimated effort:** 1 session. Mostly computation + analysis.

---

### Phase 4: Relationship Cartography — Mapping Deltas to Coordinate Flips

**Goal:** Express every relationship as a set of coordinate transformations.

**Why:** If the address book works, then relationships become simple:
`capital_of` = "flip axis 7, flip axis 23, preserve all others."

**Method:**
For each known relationship (capital, language, gender, ...):
1. Compute mean delta vector
2. Project onto each axis direction
3. Identify which axes have large projections → these are the axes the
   relationship "transforms"
4. Express the relationship as: `{flip: [axis_7, axis_23], preserve: [all others]}`

**Extended discovery:**
With 500+ concepts, we can discover NEW relationships:
1. For every pair of concepts in the same category, compute the delta
2. Cluster deltas by direction (cosine similarity)
3. Each cluster = a relationship
4. Express each relationship as coordinate transformations

**Validation:**
- Apply the coordinate-flip formula to a source concept
- Check that the resulting address matches the target concept
- Cross-validate on held-out pairs

**Estimated effort:** 1-2 sessions.

---

### Phase 5: Beyond Embeddings — Mapping Layer Transformations

**Goal:** Extend the truth-coordinate system through the full model.

**Why:** The embedding layer is just the entrance. The transformer's 32 layers
each transform these coordinates. If we can track how truth-coordinates
evolve through layers, we have a complete map.

**Method:**
1. For a set of prompts, extract hidden states at each layer
2. Project each layer's hidden state onto the truth axes (adapted to that
   layer's representation space)
3. Track which truth-coordinates change at which layer
4. Map layer function in terms of coordinate operations:
   - "Layer 5 flips the is_capital coordinate for tokens in object position"
   - "Layer 12 activates the is_abstract coordinate"

**Connection to DC 277 (Geometric Instrument):**
Each geometric structure (Spectrometer, Lens, Selector, etc.) should
correspond to operations on specific subsets of truth-coordinates:
- **Spectrometer** → reads truth-coordinates (classification)
- **Lens** → focuses on relevant coordinates for the current query
- **Selector** → picks which coordinates to propagate
- **Resonator** → amplifies coordinates near decision boundaries

**This is the most ambitious phase.** It extends our map from the static
embedding space into the dynamic computation. But it builds on all previous
phases — without the coordinate system, we can't track what layers do.

**Estimated effort:** 3-5 sessions. This is a research frontier.

---

## Critical Risks and Mitigations

### Risk 1: Axes Aren't Actually Binary
**Symptom:** Many candidate axes show continuous distributions, not bimodal.
**Mitigation:** Document continuous axes as "spectra." The theory may need
to accommodate both binary ideals and continuous dimensions. This would still
be a valid map — just a richer one.

### Risk 2: 88 Concepts Are Biased
**Symptom:** Discovered axes don't generalize to new concept categories.
**Mitigation:** Phase 0 explicitly diversifies the concept set. We validate
by checking that existing axes survive the expansion.

### Risk 3: PCA Directions ≠ Platonic Ideals
**Symptom:** PCA gives good variance explanation but axes aren't interpretable.
**Mitigation:** We use PCA only as a *guide* for how many axes to expect.
The actual discovery uses the binary separation test, not PCA directions.
PCA gives us the "how many" — our algorithm gives us the "which ones."

### Risk 4: The Model Is Too Complex for 79 Axes
**Symptom:** After 79 axes, significant structured residual remains.
**Mitigation:** This would mean the model encodes more than binary truth.
That's actually a *finding*, not a failure. We document what the axes DO
capture and what they DON'T. The map is still useful even if incomplete.

### Risk 5: Orthogonality Breaks Down at Scale
**Symptom:** As we add more axes, new ones become correlated with existing ones.
**Mitigation:** Gram-Schmidt orthogonalization at each step. If the space
genuinely has only 79 independent directions, we'll hit a wall where no more
orthogonal binary axes can be found. That wall IS the answer.

---

## Success Criteria

### Minimum Viable Map (Phases 0-2)
- [ ] 500+ concepts with clean single-token embeddings
- [ ] 30+ validated platonic ideals (truth axes)
- [ ] >50% variance explained
- [ ] Each axis: named, cross-validated >90%, orthogonal

### Complete Concept Map (Phases 0-3)
- [ ] 60+ validated platonic ideals
- [ ] >90% variance explained
- [ ] Unique binary address for >95% of concepts
- [ ] Reconstruction error < 10% of embedding norm

### Full Model Cartography (Phases 0-5)
- [ ] All ~79 platonic ideals discovered and named
- [ ] >95% variance explained
- [ ] Every known relationship expressed as coordinate flips
- [ ] Layer-by-layer tracking of coordinate evolution
- [ ] Connection to geometric instrument structures (DC 277)

---

## What This Means

If this works, we will have:

1. **A complete, interpretable coordinate system for an LLM** — every concept
   is a binary address, every relationship is a coordinate transformation

2. **Proof that LLMs are geometric encoders** — the "intelligence" is in the
   shape, and that shape is readable, finite, and verifiable

3. **A blueprint for building LLMs without training** — if we know the axes
   and their geometry, we can construct the embedding space directly from
   verifiable truths rather than learning it from data

4. **The first "periodic table" of concepts** — a systematic classification
   where every concept has a place determined by its properties, not by
   arbitrary assignment

This is not incremental. This is cartography of a new continent.

---

## Immediate Next Steps

1. **Phase 0 now** — Expand concept vocabulary to 500+ tokens
2. **Phase 1 immediately after** — Run the IRD algorithm
3. **Review after 30 axes** — Check if the theory is holding
4. **Iterate** — Keep mining until we hit the wall

The infrastructure exists. The theory is validated. The only question is
how far the map extends.

Let's find out.
