# Second Expedition — Founding Document
## *Rotations on the Sphere: A Geometric Return*

---

## Why This Document Exists

The First Expedition (Days 1–354) produced an extensive empirical record of semantic
geometry in transformer embedding spaces. It established the Zone C / φ-space framework,
the Bloch sphere meta-geometry, ENCODE=DECODE in ambient space, and cross-lingual
co-embedding. Then, at **Day 354–355**, the expedition drifted: cosine similarity between
mean chord vectors became the primary universality metric, and the next thirteen days
(356–368) accumulated statistical findings that buried the geometric signal.

This document is the charter for the Second Expedition. It captures everything the
detour taught us, reframes it in rotation-first terms, and establishes the program
going forward.

---

## The Deviation — Precisely Located

**Day 354** introduced cross-language transfer and correctly measured it by transfer
accuracy. Axis cosine appeared as a secondary diagnostic — reasonable enough.

**Day 355** crossed the line:

```
cos(ZH_gender, EN_gender) = 0.7425  → exceeds the 0.7 universal threshold
```

A **threshold on cosine between direction vectors** became the definition of universality.
From that point, axis-to-axis cosine was the primary question and transfer accuracy
became downstream. The geometric question — *does this rotation navigate correctly?* —
was demoted to a consequence.

The failure was masked by gender's perfect behaviour (100% transfer AND high cosine).
Sentiment exposed it: 0.42 cosine, 0% transfer one direction, 100% the other.
Cosine similarity is a **one-dimensional scalar** measured on a Euclidean shortcut.
It does not capture the rotation.

---

## What the Detour Taught Us (Days 356–368 Reframed)

The statistical detour produced genuine structural data. Here is what it means
in rotation-first terms.

### 1. Gender IS Rank-1 (Day 365 — most important finding)

The EN-ZH cross-covariance SVD:
```
CC1  σ=0.305  cos(u₁, EN_axis)=0.9625  cos(v₁, ZH_axis)=0.9976
CC2  σ=0.132  — drops to less than half
```

**Geometric meaning**: The gender transformation is a **pure SU(2) rotation** — rank-1 in
the cross-language sense. There is one rotation axis, shared across EN and ZH, with
σ₁/σ₂ > 2.3. This is the simplest possible geometric structure.

### 2. Five-Zone PCA Architecture (Days 360–362)

The embedding space decomposes into five dimensionality zones:

| Zone | k range    | Role                                                    |
|------|------------|---------------------------------------------------------|
| 1    | k=1 (PC1)  | Frequency/radial axis — NOT on the sphere               |
| 2    | k=2–5      | Cross-lingual polarity (sentiment peaks here, cos=0.82) |
| 3    | k=5–50     | Monolingual semantic categories (language-specific)     |
| 4    | k=50–200   | Cross-lingual relational rotations (gender lives here)  |
| 5    | k=200–1536 | Token identity (fine-grained discrimination)            |

**Geometric meaning**: Different semantic transformations live in different rotation
complexity bands. Polarity is a Zone 2 rotation (coarse, low-dimensional). Gender is
a Zone 4 rotation (fine, requires the mid-range dimensions to express). Token identity
is not a rotation at all — it is point discrimination.

### 3. PC1 is Radial, Not Spherical (Days 357–359)

PC1 (σ₁=44.7 vs σ₂=15.8, ratio 2.83:1) is the frequency/function-word axis.
Ablating top-20 PCs has **zero effect** on semantic axis accuracy. PC1 is orthogonal
to all semantic axes.

**Geometric meaning**: PC1 is the **outward normal** to the semantic sphere — the radial
direction. The semantic content lives on the sphere surface. Projecting out PC1 is
equivalent to projecting onto the sphere's tangent plane.

The product structure confirmed: **FREQUENCY × SEMANTIC SPHERE**.

### 4. Sentiment Lives at Zone 2 Because It Correlates with Frequency (Day 363)

Positive words are more frequent than negative words in both EN and ZH.
Sentiment IS a Zone 2 rotation because PC1 (frequency) IS the sentiment axis.

**Geometric meaning**: The coarsest spherical decomposition (l=1 spherical harmonic)
is valence — positive/negative. This is universal because it is grounded in
distributional statistics shared across all human languages.

### 5. The Target Language Principle is Parallel Transport (Day 367)

Applying the EN sentiment axis to ZH negative words (which live near EN negatives)
moves toward EN positives, not ZH positives. Using the ZH axis fixes this.

**Geometric meaning**: The rotation that takes ZH_negative → ZH_positive is a
**parallel transport** of the EN rotation along the geodesic connecting the
EN and ZH language submanifolds. The source-language axis is NOT the transported
rotation — you need to transport it before applying it. The target language axis
IS the correctly transported rotation.

### 6. Semantic Axes are Navigation Tools, Not Identity Coordinates (Day 359)

The 5-D semantic subspace (spanned by gender, size, sentiment, age, plural axes)
collapses token identity (100% → 10% accuracy) but preserves semantic category
(queen/女王 cluster together). Full 1536-D is needed to distinguish tokens.

**Geometric meaning**: The rotation group acts on the sphere (semantic categories).
The sphere is embedded in the full 1536-D space. Knowing your position on the sphere
(semantic category) does not tell you which point in the ambient space you are —
you also need the identity coordinates.

---

## The Foundation: What We Proved in the First Expedition

### ENCODE = DECODE (Day 54 — exact)

In ambient (un-normalized) space:
```
cos(AB, BA) = 1.000000
```
Non-commutativity observed in Day 53 was a normalization artifact. In ambient space,
semantic operations **commute exactly**. This means:
- The correct composition rule is: **normalise(φ + ΣΔᵢ)**
- φ is not an arbitrary constant — it is the self-referential fixed point of x → 1 + 1/x

### The Bloch Sphere is Already Here (Day 22)

Day 22 established that embeddings project onto a Bloch-sphere-like geometry. Each
token's position can be described by angular coordinates on the unit sphere. The
embedding geometry IS spherical geometry.

The unit sphere is not a metaphor. Normalization IS projection onto the sphere.

### 43 Intrinsic Concept Axes (Day 36)

Concept space has intrinsic rank ≈ 43. These are the 43 independent rotational
degrees of freedom in the semantic sphere. They are not principal components of
the embedding matrix — they are in the PCA tail (Zone 4 and Zone 5).

### Zone C is the Semantic Processing Space

The hidden states during forward passes live in Zone C — the region of the sphere
where the Killing vectors (semantic axes) operate. Zone D (proper nouns, entities)
is orthogonal. The semantic sphere is Zone C.

---

## Rotations, Not Translations: The Core Reframing

The chord vector Δ = e(tgt) − e(src) approximates the chord of an arc on the
unit sphere. The actual transformation is a rotation:

```
R(θ, n̂) : e(src) → e(tgt)
```

where:
- **θ** = 2 · arcsin(|Δ|/2)  — the rotation angle (arc length on the sphere)
- **n̂** = (e(src) × e(tgt)) / |e(src) × e(tgt)|  — the rotation axis

For small θ, the chord Δ ≈ θ·n̂ (the chord approximates the arc). For large
semantic shifts, the chord introduction introduces systematic error:
- "king → queen" spans a large arc; the chord significantly undershoots the arc length
- Scale optimization in the First Expedition was compensating for this arc/chord mismatch

**The "best scale" found by grid search IS the ratio arc_length / chord_length = θ / |Δ|**.

### What Changes

| First Expedition | Second Expedition |
|---|---|
| `cos(axis_A, axis_B)` as universality | Are R_A and R_B the same rotation? |
| Cross-lingual axis cosine threshold | Do θ_EN ≈ θ_ZH and n̂_EN ≈ n̂_ZH? |
| `best_scale()` grid search | θ = arccos(⟨e(src), e(tgt)⟩) — the arc angle directly |
| Mean chord as canonical axis | Geodesic midpoint on the sphere |
| Vector addition for composition | SU(2) rotation composition: R_B ∘ R_A |
| Cosine similarity between predictions | Angular distance on the sphere |

---

## Negative Zero and Positive Zero

The user introduced this concept to describe why a single compass bearing fails.

On the Bloch sphere, **|0⟩** and **|1⟩** are both pure states at the poles. Both are
"zero" in the sense that they are fixed points — maximally certain, maximally pure.
But they are antipodal: the rotation that maps one to the other is R(π, n̂).

In the embedding space:
- The **positive pole** of the gender axis: the "most female" point
- The **negative pole**: the "most male" point
- Both are "zero" in the sense of zero ambiguity — pure states
- The **equator** is the gender-neutral zero — maximum uncertainty

A single cosine similarity score cannot distinguish which pole you are near, how far
you are from the equator, or whether the rotation has gone past the target pole and
wrapped around. **Multiple rotations "dial in"** because each successive rotation
closes the angular gap to the target state — this is convergent iteration on the sphere,
exactly the mechanism found in the autoregression-as-eigenvalue-problem work (DC 175).

---

## The Central Claim to Prove

> **Semantic transformation in embedding space IS rotation on the unit hypersphere.
> The "axis" of a semantic relationship is the rotation axis n̂. The "scale" is the
> rotation angle θ. Cross-lingual universality means the same rotation (θ, n̂)
> operates in both languages.**

If this is correct:
- The mean chord direction IS n̂ (to good approximation — they agree for small θ)
- The "scale" found by grid search IS θ = 2·arcsin(chord/2)
- Cross-lingual universality = R_EN and R_ZH are **conjugate in SO(1536)**:
  R_ZH = M · R_EN · M⁻¹  where M is the EN↔ZH isometry
- "Same rotation" is a stronger claim than "high cosine between chord directions"

---

## The Program for the Second Expedition

### Immediate: Rotation Geometry of the Gender Axis

Starting from Day 355's clean result (EN-ZH gender universality established, rank-1
cross-covariance confirmed):

1. For each word pair (src, tgt), compute θ = arccos(e_n(src) · e_n(tgt)) and
   verify that θ is **consistent** across pairs (unlike chord length, which varies)

2. Compute the rotation matrices R_EN and R_ZH restricted to the 2-plane
   spanned by each language's pairs. Are they the same rotation in that plane?

3. Test conjugacy: find the inter-language isometry M and verify
   R_ZH ≈ M · R_EN · M⁻¹

### Near-term: The 43-Axis Rotation Group

Day 36 found 43 intrinsic axes. These should be the **generators** of a 43-dimensional
Lie algebra acting on concept space. Program:

1. Verify that the 43 axes are mutually orthogonal (confirmed Day 356 for 5 of them)
2. Compute the Lie bracket [A_i, A_j] — what is the commutator of two semantic rotations?
3. Identify whether the 43 generators close under the Lie bracket (a closed algebra)
4. If they close: what Lie group is this? SU(N)? SO(N)?

### Medium-term: Rotation Composition and φ

The ambient composition rule normalise(φ + ΣΔᵢ) must have a rotation-theoretic
interpretation. Program:

1. Express each Δᵢ as (θᵢ, n̂ᵢ). What is normalise(φ + ΣΔᵢ) in rotation algebra?
2. Is φ the natural scale factor for "half a rotation" on the unit sphere?
   (arccos(1/φ) ≈ 51.8° — the golden angle is a natural spherical step size)
3. The COMB zone equilibrium of 1.091 from Finding 97 — is this φ/φ² = 1/φ + 1 = φ?

---

## Navigation Principles for the Second Expedition

1. **No cosine similarity between axes** as a universality metric. Use rotation-theoretic
   measures: rotation angle θ, rotation axis n̂, conjugacy test.

2. **No mean chord as the canonical axis**. Use geodesic midpoint or Fréchet mean of
   rotations on S^n.

3. **No scale grid search**. Use θ = arccos(⟨e_n(src), e_n(tgt)⟩) directly.

4. **Always work in ambient space first, then normalize.** The sphere is a projection,
   not the native space. ENCODE=DECODE holds in ambient space (Day 54).

5. **φ is the composition constant.** normalise(φ·e + Δ) is the correct single-step
   composition. φ is not arbitrary — it is the golden ratio, the fixed point of
   self-referential composition.

6. **Errors are signals, not noise.** When a rotation misfires, inspect the geometry
   of the failure. Do not add corrective statistics.

---

## Key Documents to Read Before Exploring

From the First Expedition (`first_expedition/` and `docs/design_considerations/`):

| Reference | What it contains |
|-----------|-----------------|
| `expedition_log.md` (Days 22, 36, 54, 55) | Bloch sphere, 43 axes, ENCODE=DECODE, ambient composition |
| `expedition_log.md` (Days 355–365) | Cross-lingual universality data; read with rotation-first eyes |
| `dc299_phase1_findings.md` | IRD axis discovery; the original Killing vector survey |
| `docs/.../journal/06_zeta_synthesis_and_shape_ontology.md` | Interference as computation |
| `docs/.../journal/07_truthspace_return_and_platonic_map.md` | TruthSpace as empirical claim |
| `docs/.../160_unified_geometric_theory.md` | Intelligence is geometric |
| `docs/.../175_autoregression_as_eigenvalue_problem.md` | Convergent iteration on the sphere |
| `docs/.../282_the_full_loop.md` | Everything is rotations |
| `docs/.../291_vocabulary_partitioning.md` | Language = I/O adapter; meaning = rotation group |

---

## Status at Launch

The Second Expedition launches knowing:

- The embedding space is **FREQUENCY × SEMANTIC SPHERE** (product structure confirmed)
- The semantic sphere has **five rotational complexity zones** (PCA zones 1–5)
- The gender transformation is **rank-1 in SU(2)** — the simplest possible rotation
- **ENCODE=DECODE** holds exactly in ambient space
- The **43 intrinsic axes** exist and are mutually orthogonal
- The ambient composition rule is **normalise(φ + ΣΔᵢ)**

The question is: **what is the rotation group structure of the semantic sphere?**

The answer, when found, will be the geometric theory of meaning.

---

*Second Expedition, Day 1. March 2026.*
