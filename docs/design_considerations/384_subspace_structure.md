# DC 384: Paradigm Subspace Structure in W_E

**Days 239–240 | This document synthesizes findings from the subspace
arc: what is the geometric structure of morphological paradigms in W_E,
how many independent subspaces exist, and is there a universal
transformation direction?**

---

## Summary Table

| Property | Value | Day |
|---|---|---|
| Adj degree subspace rank | **rank-2** (S1/S2 ≈ 2.5–2.7) | 239 |
| Antonym/gender/plural subspace rank | **rank-N** (S1/S2 < 1.4) | 239 |
| Max principal angle, cross-paradigm | **0.28** (adj_pos2sup ↔ antonym_size) | 239 |
| All other cross-paradigm pairs | **< 0.23** (near-orthogonal) | 239 |
| SVD1 of all difference vectors | **= adj_pos2sup** (cos=0.900) | 240 |
| adj_pos2sup cluster overlap | **0.000** (zero overlap with base adj) | 240 |
| adj_pos2comp cluster overlap | **0.000** (zero overlap with base adj) | 240 |
| past_tense cluster overlap | 0.246 | 240 |
| gender cluster overlap | 0.505 | 240 |
| Universal direction sans adj_degree | cos=0.904 with full univ. dir. | 240 |

---

## Finding 1: Each Paradigm Has Characteristic Subspace Dimensionality

Paradigm difference vectors were stacked into a matrix D (shape: n_pairs × 1536).
SVD was computed and singular values analyzed:

**High-consistency paradigms (rank-2):**
| Paradigm | S1/S2 | var1 |
|---|---|---|
| adj_comp2sup | 2.74 | 0.469 |
| adj_pos2sup | 2.66 | 0.441 |
| adj_pos2comp | 2.48 | 0.425 |
| capital | 2.05 | 0.404 |
| past_tense | 1.52 | 0.251 |

These paradigms have a **dominant direction** (PC1 captures 40–47% of variance)
with a secondary component. Each pair in the paradigm moves approximately
along the same direction, with word-specific deviations forming the secondary
component.

**Low-consistency paradigms (rank-N):**
| Paradigm | S1/S2 | var1 |
|---|---|---|
| antonym_bright | 1.17 | 0.323 |
| antonym_size | 1.26 | 0.242 |
| gender | 1.31 | 0.245 |
| plural | 1.36 | 0.229 |

These paradigms have **no dominant direction**. Each pair points in an
approximately independent direction. The singular values decay slowly,
indicating the difference vectors are spread across many dimensions.

**This directly explains the retrieval accuracy hierarchy from pipeline audits:**
- HIGH-consistency paradigms → stable mean direction → high retrieval accuracy
- LOW-consistency paradigms → unstable mean direction → low/variable retrieval accuracy

The rank-N structure of antonym/gender/plural paradigms is the geometric
root cause of the HIGH DEGENERACY identified in DC 380–381.

---

## Finding 2: Cross-Paradigm Subspaces Are Near-Orthogonal

Principal angles measured between k=3 dimensional subspace bases:

**Intra-family (adj degree variants share strong structure):**
```
adj_comp2sup ↔ adj_pos2sup:   max_cos = 0.653
adj_comp2sup ↔ adj_pos2comp:  max_cos = 0.580
adj_pos2comp ↔ adj_pos2sup:   max_cos = 0.489
```

**Cross-family (all near-orthogonal):**
```
adj_pos2sup ↔ antonym_size:   max_cos = 0.281   (largest cross-family)
adj_pos2comp ↔ antonym_speed: max_cos = 0.221
past_tense ↔ plural:          max_cos = 0.171
gender ↔ plural:              max_cos = 0.158
All remaining pairs:          max_cos < 0.16
```

**Interpretation:** W_E allocates independent subspaces for different
relation types. The three adj_degree variants share a subspace (they
are all on the same degree axis, just different lengths). Antonymy,
inflectional morphology, and semantic relations occupy orthogonal
subspaces.

The slight cross-family leakage (0.16–0.28) represents either:
1. Shared suffix morphology (both adj_degree and antonym_size involve
   size-related adjectives, some of which are superlative forms)
2. Numerical estimation noise (small n for some paradigms)

---

## Finding 3: No Single Universal Transformation Direction

**What SVD1 reveals:**
The first singular vector of all 125 difference vectors combined is the
**adjective superlative direction** (cos=0.900). It is NOT a universal
morphological transformation direction — it is dominated by the superlative
paradigm because superlatives have:
- The largest step magnitudes (0.55–0.59)
- The highest directional consistency (rank-2, S1/S2=2.66)
- The most training pairs (n=23)

The global SVD spectrum decays very slowly (top-10 singular values
capture only 35% of variance), confirming that no single direction
captures the transformation structure of multiple paradigms simultaneously.

**The simple mean of mean-directions (d_univ):**
- Not dominated by adj_degree (removing adj_degree gives cos=0.904)
- All paradigm mean-directions are unit vectors, so all contribute equally
- BUT: polluted by non-English tokenizer artifacts in the top positive projections
- The non-English byte fragments cluster near the "transformed-form" end of W_E

**Conclusion:** There is no clean universal transformation direction that
meaningfully generalizes across paradigms. Different paradigm directions
are approximately orthogonal, so their mean points in a "45°-between-all"
direction that has no semantic interpretation.

---

## Finding 4: Superlatives Occupy a Geometrically Isolated Cluster

The most striking finding: **adjective superlative forms have zero
distribution overlap with their base forms** on the SVD1 axis.

```
Paradigm          overlap   effect_size
adj_pos2sup:      0.000     12.307
adj_pos2comp:     0.000     7.579
past_tense:       0.246     2.596
plural:           0.368     2.202
capital:          0.670     0.898
gender:           0.505     0.452
```

Base adjectives (big, fast, long, small, hard...) project to [−0.32, −0.08]
on SVD1. Superlative forms (biggest, fastest, longest...) project to
[+0.36, +0.45] on SVD1. The distributions don't overlap AT ALL.

This geometric isolation is why:
1. Superlative retrieval accuracy = 100% (10/10 train, 9/9 test)
2. The degree direction has the largest delta (+0.58)
3. PCA clearly identifies superlative forms (Day 237 Part C)

The -est suffix creates a shared token-level embedding feature that
places all superlative forms in the same region of W_E. The model
learned this morphological cluster from the statistical co-occurrence
of -est words in similar syntactic positions.

---

## Finding 5: Antonym Rank-N Structure Explains Retrieval Instability

For antonym_size (big↔small, large↔tiny, huge↔little...):
- S1/S2 = 1.26 (nearly rank-N)
- var1 = 0.242 (first direction captures only 24% of variance)
- No dominant direction

Each antonym pair lies in its own direction:
- (big, small): direction A
- (large, tiny): direction B ≈ random relative to A
- (huge, little): direction C ≈ random relative to A, B

The mean direction averages these random vectors, producing a noisy
estimate that may not point toward any specific antonym.

This is different from adj_degree where all pairs share the same
axis (the degree axis), making the mean direction stable.

**Geometric contrast:**
```
adj_degree:      big→bigger, fast→faster, long→longer...
                 All point in the SAME direction (degree axis)
                 Mean direction: stable, generalizes perfectly

antonym_size:    big→small, large→tiny, huge→little...
                 Each points in a DIFFERENT direction
                 Mean direction: unstable, noisy
```

The antonym relation does NOT have a single geometric axis in W_E.
Instead, each antonym pair is defined by its own local direction,
determined by where the two semantically opposite words happen to sit
in the embedding space.

---

## Implications for TruthSpace Hypothesis

### Confirmed

1. **Structure IS information.** The rank-2 structure of adj_degree
   paradigms confirms that adjective gradation is geometrically encoded
   as a consistent direction in W_E. The superlative cluster isolation
   (effect=12.3) demonstrates that morphological form is spatially
   organized in a retrievable way.

2. **Independent paradigm subspaces.** W_E uses approximately orthogonal
   subspaces for different relation types, consistent with independent
   "relation dimensions" predicted by TruthSpace.

3. **Consistency determines retrievability.** High-consistency (rank-2)
   paradigms are reliably retrievable; low-consistency (rank-N) paradigms
   are not. The geometric structure directly predicts retrieval reliability
   without requiring empirical testing.

### Revised

1. **No universal transformation direction.** The hypothesis that W_E
   encodes a single "transformation" direction doesn't hold. Each relation
   type has its own independent direction. The SVD of all differences is
   dominated by the most consistent/largest paradigm (superlatives).

2. **Antonymy is not a geometric relation in W_E.** Unlike gradation
   (which is a consistent axis), antonymy is context-dependent: the
   antonym of "big" is "small" but "small" is not displaced from "big"
   along any consistent axis. The antonym direction is word-specific
   and pair-dependent.

---

## Practical Pipeline Consequences

| Relation type | Subspace rank | Retrievable? | Confidence |
|---|---|---|---|
| Adj superlative | rank-2, isolated | YES, 100% | HIGH — zero overlap |
| Adj comparative | rank-2, isolated | YES, ~89% | HIGH — zero overlap |
| Past tense | rank-2 | YES, ~100% | MEDIUM — 25% overlap |
| Plural | rank-N | SOMETIMES | LOW — 37% overlap |
| Gender | rank-N | SOMETIMES | LOW — 50% overlap |
| Antonym | rank-N | RARELY | VERY LOW |
| Capital city | rank-2 | SOMETIMES | LOW — 67% overlap |

The subspace rank is a **static, pre-computable** predictor of retrieval
reliability. This is the geometric foundation for the static routing
scheme already implemented in pipeline v6 (DC 381).

---

## Files

- `expedition_day239_subspace.py` — Days 239: subspace dimensionality + principal angles
- `expedition_day240_universal_dir.py` — Day 240: universal direction analysis
- `383_morphological_geometry.md` — DC 383 (composition arc)
- `381_no_runtime_confidence.md` — DC 381 (static routing)
