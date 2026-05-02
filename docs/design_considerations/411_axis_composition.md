# DC 411: Axis Composition — Additive Stacking Fails; Sequential Chaining Required

**Day 276 | Multi-hop geometric retrieval via additive axis composition
fails. Language axis does not generalise from country to city embeddings
(0%). Adding a second axis to a working single-hop query reduces accuracy
(33%→25%). Three additive axes collapse to a single dominant anchor (0%).
Classic analogy arithmetic succeeds only for language relations (27% overall).
Axes are source-type specific and do not compose by vector addition. The
correct multi-hop architecture is sequential: retrieve the intermediate
node via NN search, then apply the next axis from that exact embedding.**

---

## What Was Tested

Four experiments on axis compositionality:
1. Language axis generalisation: city_emb + language_axis → language (0/12)
2. Two-hop additive chain: country + cap_axis + lan_axis → language (3/12 = 25%)
3. Three-hop additive chain: person + nat + dem_cty + lan axes → language (0/12)
4. Analogy arithmetic: C + (B−A) → D (3/11 = 27%)

---

## Failure Mode 1: Source-Type Specificity

The language axis was built from country→language pairs. When applied to
city embeddings (paris + language_axis), it retrieves the capitalised
version of the same city (Paris), not the language.

```
paris    + language_axis -> Paris    (not french)
berlin   + language_axis -> Berlin   (not german)
rome     + language_axis -> ROME     (not italian)
```

**Interpretation:** W_E does not encode a universal "geographic entity →
its language" direction. It encodes a specific "country name → its
language adjective" direction. The displacement vector for
france→french does not equal the displacement for paris→french,
because paris and france live in different sub-regions of W_E.

This is the **source-type specificity law**: a first-order axis
encodes the relation for the exact distributional class (country names)
from which it was built, not for semantically related types (city names).

---

## Failure Mode 2: Additive Interference

```
Direct language axis:          4/12 (33%)
country + cap_axis + lan_axis: 3/12 (25%)  ← WORSE
```

Adding the capital axis to the language query moves the target embedding
slightly off-centre, causing a different nearest neighbour to win.
The axes partially cancel or point in slightly different directions,
making the combined query LESS accurate than the single axis.

**Why this happens:** The capital and language axes point in near-
orthogonal directions (inter-axis cosines < 0.10 from Day 268).
Adding two near-orthogonal vectors to an embedding moves it diagonally
into a region where neither the language NOR the capital is the nearest
neighbour, but some intermediate token.

---

## Failure Mode 3: Anchor Collapse in 3-hop

```
All 12 three-hop queries returned: germany
```

Three scaled direction vectors summed onto a person embedding consistently
overshoot into the region dominated by "germany" — the most frequent
country token in the training pairs. This is an **anchor collapse**:
the dominant training token captures all queries when the displacement
vector is large.

The root cause: the sum of three scaled axes has a magnitude of
~3× the single-axis displacement. This overshoots any specific word's
Voronoi cell and lands in whatever cluster occupies the centroid of the
training distribution.

---

## Partial Success: Analogy Arithmetic

```
france:french :: germany:german   → HIT  (language relation)
france:french :: japan:japanese   → HIT  (language relation)
france:paris  :: germany:berlin   → MISS (got Paris — capitalised duplicate)
Einstein:physicist :: Darwin:biologist → MISS (got physicists — morphology)
```

**Why language analogies work:** The displacement france→french is a
large, consistent vector (coherence 0.694). For language relations,
the country-specific component in (french − france) is small relative
to the language-direction component, so the residual generalises across
countries.

**Why capital analogies fail:** The displacement france→paris is
partially city-specific. The vector france→paris encodes not just the
"capital-of" relation but also the specific relationship between two
individual word embeddings. This doesn't generalise: berlin does not
live at germany + (paris − france).

**Why person analogies fail:** Proper noun embeddings (Newton, Darwin)
are idiosyncratic — they live in specific locations determined by their
full co-occurrence history. The analogy C + (B−A) is swamped by C's
own idiosyncratic position.

---

## The Correct Multi-Hop Architecture: Sequential Retrieval

Additive chaining fails because:
- Each axis displacement has a magnitude appropriate for exactly one hop
- Summing multiple displacements creates a vector far from any intended target
- The intermediate node never gets "snapped" to the vocabulary

**Sequential chaining** solves this by explicitly grounding each hop:

```
Step 1: pred_1 = emb(einstein) + scale_nat * ax_nat
        intermediate = NN(pred_1)  →  "German"
Step 2: pred_2 = emb("German") + scale_dem * ax_dem_cty
        intermediate = NN(pred_2)  →  "germany"
Step 3: pred_3 = emb("germany") + scale_lan * ax_lan
        result = NN(pred_3)        →  "german"
```

Each hop begins from the ACTUAL vocabulary embedding of the retrieved
intermediate, not from a floating predicted vector. This prevents
accumulation of displacement errors.

**Predicted performance:** Sequential chaining accuracy ≈ product of
single-hop accuracies (if hops are approximately independent):
- nat axis (45%) × dem_cty (33%) × lan (80%) ≈ 12%

This is a geometric multi-hop system, not a language model, and the
retrieval at each step is deterministic (nearest neighbour), so errors
are not probabilistic but systematic. Some chains will work perfectly;
others will fail at a specific intermediate hop.

---

## Implications for TruthSpace Architecture

**What works geometrically:**
- Single-hop relation retrieval with coherent axes (language: 80%)
- Analogy arithmetic for high-coherence, high-generalisability relations
  (language coherence 0.694 → analogy works; capital coherence 0.775
  but city-specific noise → analogy fails)

**What does not work geometrically:**
- Universal axis application across source types
- Additive multi-hop composition
- Three-hop additive chains

**Architecture recommendation:**
```
TruthSpace multi-hop engine:
  1. Query: "What language does Einstein speak?"
  2. Hop 1: emb(Einstein) + nat_axis → NN → "German" (nationality)
  3. Hop 2: emb("German") + dem_cty_axis → NN → "germany" (country)
  4. Hop 3: emb("germany") + lan_axis → NN → "german" (language)
  5. Return "german"

Required:
  - Source-type specific axes for each relation type
  - NN retrieval after each hop (not end-to-end summation)
  - Axis routing: detect source type, select appropriate axis
```

The TruthSpace hypothesis ("geometry IS computation") is partially
confirmed: each individual hop IS geometric. Multi-hop computation
requires NN grounding between hops, not pure vector arithmetic.

---

## Files

- `expedition_log.md` — Day 276 results
- `403_axis_orthogonality_full.md` — inter-axis orthogonality
- `401_semantic_relation_axes.md` — coherence and retrieval
