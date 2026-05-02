# DC 326: φ-Trie Navigability — Category vs Relational Axes

**Date:** 2026-03-17  
**Experiment series:** Days 92–94  
**Prerequisite:** DC 325 (Dimensionality Optimum, Days 87–91)

---

## Overview

DC 325 established the 12D φ-trie as optimal for generative lookup
(LOO=0.9443). A natural follow-on: can the trie be used for **semantic
navigation** — given a token's address, flip one axis bit and arrive at
its semantic partner?

Days 92–94 answer this with controlled traversal experiments, revealing
a fundamental distinction between axis types and a new finding about
morphological encoding in transformer layers.

---

## The Traversal Test (Day 92)

### Method

For each known semantic pair (src, tgt), compute src's 12D ternary address,
flip the axis bit corresponding to the expected transformation (e.g., gender
bit for king→queen), find all tokens at Hamming distance ≤ 5 from the flipped
address ranked by distance, and check whether tgt appears in top-5.

### Results

| Axis | Hits/Total | % | Notes |
|------|-----------|---|-------|
| gender | 4/8 | 50% | king→queen rank=0, brother→sister rank=0 |
| comparative | 2/6 | 33% | small→smaller, good→better |
| plural | 0/6 | 0% | all top5 are unrelated tokens |
| past_tense | 0/8 | 0% | all top5 are unrelated tokens |
| antonym | 0/8 | 0% | antonym bit doesn't separate pairs |
| **Overall** | **6/36** | **17%** | |

### Exact Matches

```
king    (gender=U) → flip → queen   (gender=H): rank=0  ← EXACT
brother (gender=U) → flip → sister  (gender=H): rank=0  ← EXACT
man     (gender=U) → flip → woman   (gender=H): rank=2
boy     (gender=H) → flip → girl    (gender=H): rank=1  (Hamming-1)
```

The trie **can** navigate when: (1) the axis bit differs between src and
tgt, and (2) the remaining 11 bits are identical enough to make tgt the
unique nearest token to the flipped address.

---

## The Two Axis Types

### Category Axes (navigable)

An axis is a **category axis** if, for an isolated token, the axis
projection reliably reflects an inherent property of the word:

| Axis | Navigation | Explanation |
|------|-----------|-------------|
| gender | 50% | "queen" has consistently high gender-axis projection |
| comparative | 33% | "faster" reliably differs from "fast" on comparative axis |

For category axes, the token's hidden state already encodes the feature.
"queen" is always at the "high gender" end of the gender axis, regardless
of context. Flipping the gender bit in king's address finds queen because
queen's address differs from king's primarily on that one bit.

### Relational Axes (not navigable)

An axis is a **relational axis** if the projection captures a contextual
transformation but NOT the token's inherent membership in a category:

| Axis | Navigation | Explanation |
|------|-----------|-------------|
| past_tense | 0% | "ran" doesn't project differently from "run" on the tense axis |
| plural | 0% | "dogs" doesn't reliably project higher than "dog" on plural axis |
| antonym | 0% | "hot" and "cold" project SIMILARLY on the antonym axis |

For relational axes, the T2 vector was computed from sentence-level
differences ("I walk" → "I walked"), not token-level. The axis captures
"how does a sentence's representation change when you apply this
transformation?" — not "does this token have this property?"

The address bit for "dog" and "dogs" often has the same value because their
isolated hidden states are semantically similar (same lemma, nearly same
contextual profile without sentence context).

### The Bit Analysis Evidence

```
axis        src_bits         tgt_bits       diff_bit pairs
─────────────────────────────────────────────────────────
gender      {U:5, H:3}       {H:8}          5/8  ← good separation
comparative {H:2,L:2,U:2}   {H:6}          4/6  ← reasonable
past_tense  {L:8}            {H:5,U:2,L:1}  7/8  ← EXCELLENT separation
plural      {U:3,H:3}        {U:1,H:5}      2/6  ← poor separation
antonym     mixed            mixed          5/8  ← mixed
```

Crucially: **past_tense has 7/8 pairs with different bits**, yet
navigation fails completely. The bit IS discriminative (run→L, ran→H),
but flipping the past_tense bit of "run"'s address finds random tokens
like "fish", "monkey" — not "ran". This is because the past_tense axis
direction at L28 doesn't have a clean token-level counterpart.

---

## Why Multi-Dimensional Navigation Works (Day 92)

The king→queen navigation (rank=0) is NOT because the gender bit alone
points to queen. It's because:

1. king and queen share very similar non-gender semantics (royalty,
   singular, noun, animate)
2. In the 12D trie, king and queen are nearly identical on bits 1-12
   EXCEPT the gender bit
3. Flipping king's gender bit produces an address that is uniquely
   close to queen's actual address in 12D Hamming space
4. No other token is as close to king's flipped address as queen

The navigation uses ALL 12 dimensions simultaneously. Individual axis
projections alone (1D ranking) give much worse results:

| Method | Gender accuracy |
|--------|----------------|
| 12D Hamming traversal (Day 92) | 4/8 (50%) |
| 1D projection ranking (Day 93) | 2/8 (25%) |

The extra precision comes from 11 other axes that narrow the search.

---

## Token-Level Axis Coherence (Day 93)

When T2 axes are computed from direct token pairs (hidden(" ran") −
hidden(" run")) instead of sentence pairs, a new finding emerges:

```
antonym_tok: mean pairwise cos = 0.008  (≈ random)
```

**There is no single antonym direction.** The vectors "cold−hot",
"small−big", "slow−fast" all point in completely different directions.
Antonymy is a semantic relation, not a geometric direction.

Compare with morphological axes:
```
gender_tok:      0.183  (weak signal)
past_tense_tok:  0.347
plural_tok:      0.488
comparative_tok: 0.483
```

Plural and comparative have moderate coherence — there IS some shared
direction from "base form" to "inflected form", but it's noisy.

The sentence-level and token-level gender axes are **77.6° apart** —
almost orthogonal. They capture completely different information.

---

## Middle-Layer Morphology Collapse (Day 94)

Sweeping token-level axis coherence over layers [1, 8, 15, 22, 27, 28]:

```
layer:      L1     L8     L15    L22    L27    L28
plural:     0.376  0.366  0.367  0.366  0.479  0.520
past_tense: 0.371  0.062  0.062  0.063  0.278  0.347
comparative:0.530 -0.011 -0.012 -0.011  0.297  0.480
```

Sharp discontinuity at L8 for comparative and past_tense:
- L1: strong morphological signal (embedding proximity preserved)
- L8–L22: signal collapses to near zero
- L27–L28: partial recovery

**Interpretation:** Transformer layers perform progressive semantic
abstraction. By L8, "run" and "ran" have been merged into approximately
the same contextual representation (same lemma, same semantic field).
The inflectional morphology that distinguishes them at the embedding level
(L1) is "forgotten" by the middle layers. At L27–L28, it partially
re-emerges as the model prepares to predict tokens.

Notable: comparative shows **negative** coherence at L8–L22, meaning
the difference vectors actively anti-correlate. The model may be
encoding the comparative's semantic content (bigger = larger) at these
layers, which opposes the morphological direction.

This is consistent with the **DRUM/COMB/MUSIC** layer structure:
- DRUM (L0–3): input encoding, form preserved
- COMB (L4–25): semantic processing, form abstracted
- MUSIC (L26–28): output targeting, form restored for prediction

---

## Hybrid Trie Results (Day 94)

Replacing sentence-level plural, past_tense, comparative with their
token-level optimal-layer versions:

| Config | LOO | r | Coverage | Traversal |
|--------|-----|---|----------|-----------|
| sentence-level | 0.9443 | 4 | 19.7% | 6/36 (17%) |
| hybrid | 0.9437 | 3 | 42.4% | 7/36 (19%) |

The hybrid trie:
- Trades -0.0006 LOO for +1 traversal pair and +22.7% coverage
- Coverage increase reflects that token-level axes discriminate differently
  (more tokens share leaves), but they're noisier axes
- The marginal traversal improvement does not justify the LOO cost

**Conclusion:** For generative lookup, sentence-level axes are superior.
For navigation/traversal, neither configuration is adequate for
relational axes (they require a fundamentally different approach).

---

## Toward Full Navigability

The φ-trie as currently built is a **semantic similarity index** — it
groups contextually similar tokens together for efficient generative lookup.
It is NOT a general semantic coordinate system for arbitrary traversal.

To achieve full navigability, two changes are needed:

### 1. Contextualized Addressing

Instead of classifying isolated tokens (" dog"), classify tokens in a
canonical sentence context:
```
"The [MASK] is an animal." → extract hidden state at [MASK] position
```
This would give each token a context-dependent address where relational
features (tense, number) are properly activated. The same token may have
different addresses in different contexts.

### 2. Category-Only Trie for Navigation

Use ONLY genuinely category-level axes (gender, animacy, concreteness,
person-number) that reflect inherent token properties, not transformations.
These axes enable reliable bit-flipping navigation while sacrificing the
richer contextual fingerprinting of the 12D sentence-level trie.

---

## Summary

| Property | 12D sentence-level φ-trie |
|----------|--------------------------|
| LOO generative lookup | 0.9443 at r≤4 |
| Coverage | 19.7% |
| Gender traversal | 4/8 (50%) — king→queen exact |
| Past_tense traversal | 0/8 (0%) — relational axis |
| Plural traversal | 0/6 (0%) — relational axis |
| Antonym traversal | 0/8 (0%) — no shared axis direction |
| Axis type insight | category axes navigable, relational are not |
| Morphology finding | collapses at L8–22, revives at L28 |

The trie succeeds at its core purpose — generative lookup of semantically
similar tokens — while revealing the geometry of lexical vs relational
features in the transformer's representational space.

---

## Connection to Prior Work

| Finding | DC | Value |
|---------|-----|-------|
| 8D trie LOO baseline | DC 323 | 0.9303 at r≤3 |
| 12D optimal | DC 325 | 0.9443 at r≤4 |
| Category vs relational axes | **DC 326** | gender 50%, past_tense 0% |
| Antonym: no shared axis | **DC 326** | coherence = 0.008 |
| Middle-layer morphology collapse | **DC 326** | L8–22 coherence ≈ 0 |
| Comparative@L1 highest coherence | **DC 326** | 0.530 |
