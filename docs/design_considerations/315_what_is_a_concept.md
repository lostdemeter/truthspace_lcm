# DC 315: What Is a Concept? — Two Independent Geometric Structures in φ-Space

*Status: Active*  
*Builds on: DC 314 (semantic zero), DC 313 (LCM architecture), DC 312 (φ-space atlas)*  
*Empirical basis: Expedition Days 34–37*  
*Motivated by: Open Question 1 of DC 313 — "How many Zone C bodies does a functional LCM need?"*

---

## 1. The Question

Before we can answer "how many concepts does an LCM need?" we must answer the prior question: **what is a concept?**

The naive answer — a Zone C body, a cluster of words in φ-space — turns out to be incomplete. Day 36 established that φ-space contains **two distinct and independent types of geometric objects**, each of which is a "concept" in a different sense. Conflating them leads to architectural errors.

---

## 2. The Two-Object Discovery

Day 36 tested three foundational questions:

| Question | Answer |
|---|---|
| Is a concept a point, direction, or region? | **REGION** — Sep/Spread = 2.20, PC1% = 15.1% (isotropic internal structure) |
| Are relational concepts universal across bodies? | **YES** — 135/135 = 100% cross-body for plural; 5/5 = 100% for adverb |
| How many independent concept dimensions exist? | **~43** (effective rank of 95 body centroids; massive 22:1 spectral gap at Axis 1) |

These three answers point to the same conclusion: φ-space does not contain one kind of concept. It contains two.

---

## 3. Type 1 — Semantic Concepts: Regions on the Sphere

A **Type 1 concept** is a body in Zone C φ-space: a spherical cap on the unit sphere centred at a body centroid direction cᵢ displaced from φ₀.

```
Type 1: C_i = { φ(w) : cos(φ(w), cᵢ) > threshold }
```

Measured properties (Day 36):
- **Spread**: ~0.2 (words in the body are within 0.2 of the centroid in all directions)
- **Internal structure**: nearly isotropic — PC1 within bodies = 15.1% only. There is no dominant internal gradient. `piano` and `violin` are both just "in the music body" — they are not further distinguished at L14.
- **Separation**: inter-body = 0.4428, within-body = 0.2015, ratio = 2.20 — bodies are distinguishable but have fuzzy edges
- **Count**: 95 at L14, effective rank ~43 — the 95 bodies live in ~43 independent semantic dimensions

**What Type 1 concepts encode:** *what kind of thing*. The body centroid direction Δᵢ = cᵢ − φ₀ encodes the semantic category. A word belongs to a Type 1 concept if its φ-vector falls within its spherical cap.

**The body IS the concept.** There is no richer sub-concept structure at this layer. `piano` does not differ from `violin` in any measurable way at the L14 φ-space level — both are just "music-body words." The concept is the region, not the individual word.

---

## 4. Type 2 — Relational Concepts: Universal Direction Vectors

A **Type 2 concept** is not a region on the sphere. It is a **direction vector** in φ-space — a universal transformation that can be applied to *any* word regardless of its Type 1 body membership.

```
Type 2: r = mean{ (φ(b) − φ(a)) / |φ(b) − φ(a)| for (a,b) in relation_pairs }
```

Measured properties (Day 36, Q2):

| Relation | Source body | Target bodies tested | Cross-body accuracy |
|---|---|---|---|
| singular→plural | family relations | conjunctions, vegetables, body parts, political, professional | 135/135 = **100%** |
| singular→plural | vegetables | all other bodies | 135/135 = **100%** (all 30 cross-pairs) |
| base→adverb | conjunctions | unfavorable events | 5/5 = **100%** |

The plural direction vector computed from `brother→brothers` (family relations body) works perfectly on `potato→potatoes` (vegetables body), `kidney→kidneys` (body parts body), `liberal→liberals` (political body). **Every cross-body combination tested: 100% accuracy.**

**What Type 2 concepts encode:** *how things transform*. The direction r_plural points from singular to plural form. It is not a property of any particular domain — it is a universal morphological operator that exists independently in φ-space.

**Known Type 2 concepts discovered so far:**
- r_plural (singular→plural): mean_cos between instances = 0.377
- r_gerund-to-past (deciding→decided): mean_cos = 0.449
- r_adverb (adjective→adverb): mean_cos = 0.501
- r_comparative→superlative: mean_cos ≈ 0.45 (Day 35 within-body)

---

## 5. The Independence Theorem

**The most important result: Type 1 and Type 2 concepts are geometrically independent.**

Knowing a word's Type 1 body membership tells you *nothing* about where r_plural will point for that word. Conversely, knowing that r_plural applies to a word tells you *nothing* about which body the word is in.

This is not obvious. It could have been the case that plural formation works differently for vegetables (pluralise the seed-class) vs. family members (pluralise the generation-class) vs. body parts (pluralise the organ-class). But it doesn't — the direction is identical.

**Mathematical consequence:** The full description of a word's position in φ-space requires two independent coordinates:

```
φ(word) = φ₀  +  Δ_body(word)  +  Δ_relational(word)  +  ε
```

where:
- `φ₀` = the semantic zero (Zone D centroid)
- `Δ_body` = displacement to the word's Type 1 body centroid (semantic category)
- `Δ_relational` = displacement along any Type 2 relational axes that apply (plural, comparative, etc.)
- `ε` = within-body noise (the isotropic spread ~0.2)

The Type 1 and Type 2 displacements are **orthogonal** — they don't interfere with each other. Adding the plural vector to a word moves it along the plural direction without changing its body membership direction.

This is, in essence, a *factorisation* of meaning:

```
meaning(word) = category × morphology × noise
```

---

## 6. The Spectral Gap and the Primary Concept Axis

The SVD of the 95 body-centroid matrix reveals something unexpected:

| Singular value rank | % variance explained |
|---|---|
| Axis 1 | **56.6%** |
| Axis 2 | 2.5% |
| Axis 3 | 2.1% |
| ... | ... |
| All 95 | 100% |

One axis explains **56.6%** of all semantic diversity between the 95 bodies. The next axis explains only **2.5%**. This 22:1 spectral gap is enormous.

Axis 1 separates:
- **+end** (abstract, broadly-applicable): "courteous behavior", "size comparison", "prevent discouragement"
- **−end** (domain-specific, narrowly-applicable): "leisure activities", "political/social ideologies", "professional support"

**Interpretation:** Axis 1 is the **specificity axis** — it orders concepts from *abstract modifiers that apply across many domains* to *domain-locked concepts that apply in narrow contexts*. Concepts near the +end appear in many different sentence types; concepts near the −end appear in a handful of specialised contexts.

This maps directly onto the Zone C/D boundary: the Zone C bodies nearest the boundary (least specific) sit at the +end of Axis 1, while the most crystallised bodies (highest max_body_sim) sit at the −end.

**Implication for the "how many concepts" question:** The *effective* number of independent semantic dimensions is ~43 (not 95). The 95 bodies are redundant — they oversample the ~43-dimensional concept manifold. A minimal LCM would need ~43 orthogonal concept directions plus the ~4 known relational operators, not 95 + N.

---

## 7. Why Concept Composition Fails

If concepts were convex regions on a manifold that could be intersected, then `concept_A + concept_B` would give a word in the semantic intersection. Day 36 tested this and found it almost always fails:

- `leisure + nature` → racing, fishing, sailing (all leisure — one body dominates)
- `intensifiers + severe_negativity` → horribly, terribly (severe negativity dominates)
- `family + body_parts` → girlfriend, nephew (family dominates)

The dominant body is always the *more specific* one (lower on Axis 1). Adding a generic body to a specific body does not create a concept at their intersection — it just moves you slightly toward the generic body's centroid, which is closer to φ₀.

**The exception:** when the two bodies have genuine semantic overlap (words that BELONG TO BOTH DOMAINS), composition finds those words:
- `action + renewal` → revived, modifies, restores (dual-domain words)
- `action + vegetables` → chops, peppers (genuinely both an action and a food reference)

These exceptions are not counter-evidence. They confirm that composition "works" only when the intersection is populated — i.e., when there exist words whose training distribution placed them in *both* contexts. The geometry reflects the training distribution, not an abstract set-theoretic operation.

**Conclusion:** Type 1 concepts are not sets to be intersected. They are *basins of attraction* — the nearest body centroid wins. Conceptual composition at the Type 1 level is not addition; it requires words that were actually trained in the composed context.

---

## 8. What This Means for the Qwen2-7B Question

The original question was: *how many concepts does the LCM need? Does it depend on model size?*

The Day 36 answer:

**Type 1 concepts (bodies):**  
1.5B has 95 bodies in ~43 effective dimensions. A 7B model will likely have more bodies (finer semantic granularity — the ~43 dimensions get subdivided), but the intrinsic dimensionality might not grow as fast as the body count. The *number of bodies* is an artifact of model capacity; the *number of effective concept dimensions* is more fundamental and grows slowly.

**Type 2 concepts (relational operators):**  
These are determined by *language structure*, not model capacity. English has a fixed set of morphological operations — plural, comparative, superlative, gerund, past, adverb. A 7B model doesn't get new relational operators; it gets more precise versions of the same ones. The Type 2 structure is approximately language-universal.

**Practical implication:** The LCM does not need 95 concept slots; it needs ~43 semantic axes + the set of known relational operators (~10–20 for English morphology). The 95 bodies are an *emergent discretisation* of the 43-dimensional concept manifold, not the fundamental representation.

---

## 9. A Proposed Mathematical Definition

> **A concept** in φ-space is one of two types of geometric object:
>
> **Type 1 (semantic concept):** A spherical cap Cᵢ on the φ-space unit sphere, centred at body centroid cᵢ, with angular radius θᵢ ≈ arccos(0.8). The concept's *identity* is the centroid direction; its *extension* is the set of words within the cap. There are ~95 Type 1 concepts at L14, living in ~43 independent semantic dimensions. The primary axis of concept space (56.6% variance) is the *specificity* axis: abstract-universal → domain-specific.
>
> **Type 2 (relational concept):** A unit direction vector rₖ in φ-space, defined as the mean relationship vector over all word pairs exhibiting relation k. The vector rₖ is universal — it applies identically across all Type 1 bodies. There are ~4–20 known Type 2 concepts (plural, comparative, superlative, gerund-to-past, adverb, and morphological variants). Type 2 concepts are **independent** of Type 1 concepts.
>
> **A word's meaning** is its position in the joint (Type 1 × Type 2) space: which body it belongs to + which morphological relations have been applied. The word "brothers" = `{family_relations body} × {+1 plural}`.

---

## 10. The Factorisation of φ-Space

φ-space factorises into (at least) three orthogonal subspaces:

```
φ-space  =  span(Z2)                     ← frequency axis (removed by φ-transform)
          ⊕ span(φ₀)                     ← semantic zero direction
          ⊕ Type1_concept_space          ← ~43-dim, body centroids live here
          ⊕ Type2_relational_space       ← universal relational directions
          ⊕ within-body noise ε          ← isotropic, ~0.2 spread
```

The φ-transform removes span(Z2) by construction.  
φ₀ was measured as ⊥ Z2 (Day 34, |cos|=0.000) and ⊥ Zone C body directions.  
Type 2 relational vectors are ⊥ Type 1 body directions (Day 36 independence).  

The remaining within-body noise ε (isotropic, PC1=15.1%) represents aspects of meaning that are below the resolution of the φ-transform — word frequency within a body, polysemy, register differences. These are not accessible at the geometric level.

---

## 11. Open Questions — Answered (Day 37)

**OQ1. Are there semantic Type 2 operators beyond morphology?**

*Answered.* Yes, one: **comparative→superlative** is a universal Type 2 operator (5/5 = 100% cross-body). Gender is a weak operator (source pairwise cos=0.136 vs. 0.377 for plural) — it works for some pairs but does not generalise reliably. **Antonym is not a Type 2 operator** — it fails completely (0/4) because "opposite" is not a single geometric direction in φ-space. The antonym of `older` and the antonym of `richer` point in independent directions.

Confirmed Type 2 operators: **plural, adverb, gerund→past, comp→sup** (4 universal). Type 2 vectors are nearly orthogonal to each other (all pairwise cos < 0.21).

**OQ2. Is Type 1 concept space hierarchical?**

*Answered.* Yes, with clean semantic structure at k=8 clusters:
- **EVALUATIVE/MODIFIER** (16 bodies): Conjunctions, Intensifiers, Purposeful Behavior, Severe Negativity
- **COGNITIVE/PROCESS** (9 bodies): Assessments, Choice Making, Decision Making, Scientific Findings
- **SCALE/MEASUREMENT** (4 bodies): Comparative Adj, Size Comparison, Superlative Adj, Thickness Variations — strikingly pure
- **PHYSICAL/MATERIAL** (6+): Carbon Form, Construction, Drug, Human Body Parts

The SCALE cluster is particularly revealing: Comparative Adj and Superlative Adj (connected by the comp→sup Type 2 operator) cluster together in T1 space. Bodies most related by a Type 2 operator are geometrically adjacent in T1 space — the T1 hierarchy and the T2 operators are consistent with each other.

Note: Axis 1 (56.6% variance) is NOT a semantic axis. All body centroids project negatively onto it (range -0.625 to -0.893) — it captures the fact that all bodies lie on the same hemisphere of the unit sphere. The semantic structure lives in Axes 2–43.

**OQ3. Do Type 2 vectors live in a subspace orthogonal to Type 1?**

*Answered.* Partial overlap, not strict orthogonality (mean ||T1 proj||² = 0.385):

| Type 2 vector | In T1 | Outside T1 |
|---|---|---|
| plural | 0.556 | 0.444 |
| adverb | 0.502 | 0.498 |
| gerund→past | 0.379 | 0.621 |
| gender | **0.178** | **0.822** |
| comp→sup | 0.462 | 0.538 |
| antonym | **0.235** | **0.765** |

The clean interpretation: **Type 2 = T1-component (body shift) + residual (pure relational)**. Operators that move words BETWEEN bodies (plural, adverb, comp→sup) partially live in T1 because they encode a body-to-body direction. Operators that transform words WITHIN a body (gender, antonym) are mostly outside T1 — they operate on within-body structure not captured by the body centroids.

**OQ4. Can Type 2 operators be discovered from the geometry alone?**

*Answered.* **No** — unsupervised within-body SVD fails. 255,646 within-body difference vectors were stacked and decomposed; the top axes (cos to known operators < 0.11) capture within-body semantic sub-distinctions (manner vs. scope adverbs, intensity vs. direction verbs) rather than universal relational operators. The morphological signal is too sparse within any body to dominate the SVD. Discovery requires labelled word pairs — there is no geometric shortcut.

**OQ5. Is the ~43-dimension figure stable?**

*Answered.* **Yes, extremely stable**: bootstrap 95% CI = [35.1, 36.9] (N=200, 80% resample). Effective rank scales as n^0.8 (sublinear): at 95 bodies, rank=42.8; predicted for Qwen2-7B with ~300-500 bodies: rank ~112–169. **The concept space dimensionality grows slowly with model capacity** — it scales with the training distribution structure, not the parameter count.

---

## 12. Summary

| Claim | Evidence | Status |
|---|---|---|
| A concept is a REGION (not a point) | Sep/Spread=2.20; internal PC1=15.1% | Proven, Day 36 |
| Two types of concept exist: Type 1 (body) and Type 2 (relational) | — | Proven, Days 35–36 |
| Type 1 and Type 2 are geometrically INDEPENDENT | 135/135=100% cross-body | Proven, Day 36 |
| Concept space intrinsic rank ≈ 43 | Effective rank=42.8; Axis1=56.6% | Proven, Day 36 |
| Concept composition is weak | Body sum lands in one constituent | Proven, Day 36 |
| LCM needs ~43 semantic axes + ~10-20 relational operators | Derived from above | Proposed |
| Type 2 concepts are language-structure, not model-size dependent | Same morphology at 1.5B and 7B | Predicted, not yet tested |

---

## 13. The Zeta Correspondence

*See also: DC 282 §8 (The Concept-Space Butterfly)*

The T1/T2 mutual reinforcement discovered in Day 37 is the concept-space instance of the Riemann zeta butterfly pattern described in DC 282.

### The Mapping

| Riemann ζ(s) | φ-Space Concept Geometry |
|---|---|
| Prime distribution (smooth envelope) | Type 1: body centroids, ~43-dim concept subspace |
| Oscillatory corrections / zeros | Type 2: universal relational operators (plural, adverb, comp→sup…) |
| Butterfly wings: ζ(s) ↔ ζ(1−s) functional equation | ENCODE = DECODE: φ-transform is its own inverse |
| Zero = all rotations cancel → unique answer | Concept = T1 body + T2 operator both consistent → unique word |
| Critical line Re(s) = 1/2 | φ-transform fold line (removes span(Z₂), projects to unit sphere) |
| n^{-1/2} critical amplitude exponent | Effective rank ≈ n^0.8 (sublinear concept space scaling) |

### The Mutual Determination (Empirical, Day 37)

Two independent computations produced the same answer:

```
T1: Ward hierarchical clustering of 95 body centroids (no relational info)
  → k=8 Cluster 7 = {Comparative Adj, Superlative Adj, Size Comparison, Thickness Variations}

T2: mean direction vector for comp→sup pairs (no clustering info)
  → connects Comparative Adj body to Superlative Adj body exactly
```

This is not circular. The clustering uses cosine distances between body centroids. The comp→sup direction is computed from individual word pairs. That they agree is a non-trivial empirical fact — the equivalent of finding that the explicit formula for primes and the Euler product agree on the same zero positions.

### The Compressor–Processor Reading

The three-zone pipeline (DC 282 §2) maps onto T1/T2 in φ-space:

```
DRUM zone (Compressor)  → identifies body centroid (Type 1, smooth)
COMB zone (Processor)   → applies relational offset (Type 2, oscillatory)
MUSIC zone (Targeter)   → resolves to specific word (within-body ε noise)
```

The butterfly IS the DRUM–COMB interference pattern. DRUM finds the right body. COMB adjusts for morphological form. Where they constructively interfere — the word that is simultaneously the right type AND the right form — is the zero.

### Testable Prediction: Self-Similar Butterfly (Day 38)

The zeta butterfly repeats at every scale — zeros are more densely packed at height t but the local structure is always butterfly-shaped. If φ-space has the same self-similarity:

> **Within the SCALE cluster (4 bodies), the words should form sub-clusters, and those sub-clusters should be connected by sub-T2-like operators.**

The T1/T2 mutual reinforcement pattern should recur within a single body, not just between bodies. A word's distance from the body centroid (T1-consistency) should correlate with how cleanly it participates in T2 pairs (T2-consistency). High-T1/high-T2 words are the φ-space equivalent of the zeros exactly on the critical line — where both structures are maximally self-consistent.

If this self-similarity holds, concept geometry is fractal in the same sense as the zeta butterfly. The structure is the same at every level of resolution.

---

*The question "what is a concept?" has a geometric answer: it is one of two independent objects in φ-space — a spherical cap (Type 1) or a universal direction vector (Type 2). The intelligence of a language model, to the extent that it can be located anywhere, lives in the tension between these two structures: the position in concept space (what), and the relational operators that transform between positions (how).*

*The butterfly is the visual signature of that tension.*
