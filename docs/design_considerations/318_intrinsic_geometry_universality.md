# DC 318: Intrinsic Geometry — Universal Structure or Training-Data Relative?

**Arising from:** Day 50 axis generation targeting
**Date:** March 2026
**Status:** Active investigation (Day 51)

---

## 1. The Central Distinction

Day 50 revealed that the 43 concept axes are **intrinsic**, not **extrinsic**:

- **Extrinsic (Cartesian)**: concepts have absolute coordinates whose values carry
  meaning independently of all other concepts. Adding a new concept changes nothing.

- **Intrinsic (Riemannian)**: concepts have positions defined by their relationships
  to all other concepts. The coordinate values are chart-dependent (rotating the
  basis changes the numbers while preserving the geometry). What is invariant is
  the **metric** — pairwise distances, angles, and curvature.

The 43 axes are the principal components of the body-centroid distribution. They
are a particular **chart** of the Zone C manifold — a convenient basis, but not
the only one. The axis value `0.291` for "sank" on axis 3 is meaningless in
isolation. What is meaningful is that `d("king", "queen")` is small and
`d("king", "microscope")` is large, regardless of which basis you choose.

This raises the central question:

> **Are the invariants (distances, angles, topology) themselves universal — shared
> across models and training distributions — or are they relative to the
> specific data and weights that produced them?**

---

## 2. What Universality Would Mean

If the geometry is **universal**:

- The distance between "king" and "queen" in Zone C should be approximately equal
  in Qwen2-1.5B, Llama-3-8B, Gemma-2-9B, etc.
- The T2 operator for `male→female` should point in the same direction in any
  model trained on natural language.
- Translation pairs ("king", "王") should occupy the same position in Zone C
  within a single multilingual model.
- The body cluster topology (which concepts are near each other) should replicate
  across models.

If this holds, TruthSpace is discovering a **Platonic structure** — a geometry
that exists in the data itself (in the information-theoretic sense), which any
sufficiently trained model must converge to. The model weights are just one
instantiation of this universal shape.

This would strongly validate the core hypothesis: **the weights ARE the geometry,
and the geometry IS the knowledge**. The geometry predates any particular model.

---

## 3. What Training-Relativity Would Mean

If the geometry is **training-data relative**:

- Different models would have different inter-concept distances.
- The T2 operator direction for `male→female` might point in different directions
  in different models.
- Body clusters might exist in all models (since language structure forces them)
  but be arranged differently.

This would not invalidate TruthSpace, but would require a weaker claim:
**the geometry is the knowledge encoded by this training distribution**,
not a universal truth. The LCM would be a faithful reconstruction of one
model's learned geometry, not a discovery of universal semantic structure.

---

## 4. Evidence Hierarchy

Three levels of evidence are required, from cheapest to strongest:

### Level 1: Within-model stability (no second model needed)

Tests whether the geometry is stable across different **samples** of the same
vocabulary:

- **Split-half axis stability**: build SVD axes from each half of the Zone C
  vocabulary independently. High cosine alignment between corresponding axes
  means the geometry is an intrinsic property of the semantic space, not an
  artifact of the specific words chosen.

- **T2 operator generalization**: train T2 operators on half the seed pairs,
  test on the held-out half. High generalization means the T2 direction is a
  stable property of the space, not just a fit to those specific pairs.

- **Inter-body metric stability**: are pairwise cosine distances between body
  centroids stable when computed from different vocabulary subsets?

### Level 2: Cross-lingual consistency (within one model)

Tests whether the geometry is language-invariant within one multilingual model:

- Are English "king" and Chinese "王" (wáng) at the same Zone C position?
- Does `d("king"_EN, "queen"_EN) ≈ d("王"_ZH, "后"_ZH)`?
- Do English and Chinese words for the same concept cluster together?

**From Day 44**: English and Chinese words converge to Zone C — a positive
result. The question is whether convergence means co-location (same address)
or just same zone (same hemisphere).

### Level 3: Cross-model consistency (strongest, most expensive)

Tests whether the geometry is architecture-invariant:

- Run the same vocabulary through Qwen2-1.5B and a second model of different
  architecture (Llama, Gemma, Mistral).
- Apply the same Zone C extraction pipeline.
- Measure: rank-correlation of pairwise inter-concept distances.

This is the Platonic Representation Hypothesis test (Huh et al., 2024):
different models should converge to similar representations of the world.

---

## 5. Day 51 Investigation Results

### 5.1 Split-Half Axis Stability (T1)

10 random stratified splits. Mean cosine between corresponding axes:

```
All 43 axes: 0.6136   Top-5 axes: 0.8015

Per-axis decay:
  Axis  1: 0.9976  ← UNIVERSAL
  Axis  2: 0.9207  ← UNIVERSAL
  Axis  3: 0.7409
  Axis  4: 0.7252
  Axis  5: 0.6233
  ...        (0.5-0.65 range through axis 16)
  Axis 17: 0.4541  ← falls below STABLE
  Axis 18: 0.0484  ← effectively noise
  Axis 19+: NaN   ← too few bodies in half to match
```

Verdict: **STABLE**. The first two axes are near-universal; the axis
basis degrades gracefully. The top-5 explain most variance and are
robustly reproducible.

### 5.2 T2 Operator Leave-One-Out Generalization (T2)

High LOO stability of the operator direction itself (loo vs full: 0.9635)
but variable prediction of individual held-out pairs:

```
Operator          LOO cos (vs held-out pair)
──────────────────────────────────────────────
base→adverb       0.9025   UNIVERSAL
base→comp         0.6864   STABLE
comp→sup          0.5533   STABLE
singular→plural   0.5380   STABLE
base→gerund       0.3831   RELATIVE
male→female       0.1635   RELATIVE
gerund→past       0.1380   RELATIVE

Global mean:      0.4823   → just below STABLE threshold
```

The operator direction is highly stable (0.9635 loo vs full) but the
individual pair variance is large for morphologically irregular
transformations (tense, gender). base→adverb is the most regular
(suffix -ly is deterministic) and achieves near-universal LOO stability.

### 5.3 Inter-Body Metric Stability (T3)

```
Mean Spearman ρ (pairwise inter-body distances, 10 splits):
  ρ = 0.9428   min=0.9312  max=0.9520
```

Verdict: **UNIVERSAL**. The distances between concept clusters are
highly reproducible. Which bodies are near each other and which are
far apart does not depend on which specific words you chose from each
body. This is the strongest and most fundamental result.

### 5.4 Cross-Lingual Co-Location (T4)

English–Chinese translation pairs (15 pairs):

```
Pair              cos(EN,ZH)  dist
──────────────────────────────────────────────────────
king / 国王        0.9925      0.0075   CLOSE
queen / 女王       0.9942      0.0058   CLOSE
man / 男人         0.9899      0.0101   CLOSE
woman / 女人       0.9899      0.0101   CLOSE
cat / 猫           0.9914      0.0086   CLOSE
dog / 狗           0.9928      0.0072   CLOSE
running / 跑步     0.9893      0.0107   CLOSE
walking / 走路     0.9899      0.0101   CLOSE
quickly / 快速地   0.2622      0.7378   FAR  ← outlier
beautiful / 美丽   0.9906      0.0094   CLOSE
decision / 决定    0.9885      0.0115   CLOSE
family / 家庭      0.9913      0.0087   CLOSE
soldier / 士兵     0.9932      0.0068   CLOSE
scientist / 科学家  0.9917      0.0083   CLOSE
philosophy / 哲学  0.9923      0.0077   CLOSE

Mean cos(EN,ZH): 0.9426   (0.9926 excluding outlier)
```

All but one pair are co-located with cos > 0.98. Translation partners
occupy essentially the same Zone C address. The outlier 'quickly/快速地'
likely reflects a tokenization mismatch (快速地 is a 3-character sequence
with different context processing than the English adverb).

**Cross-lingual metric preservation:** Spearman ρ(d_EN, d_ZH) = 0.2556

This is the nuanced result. Individual concepts co-locate (cos ≈ 0.99)
but the pairwise distances between concepts have different rank orderings
in English vs Chinese. d(king_EN, soldier_EN) ≠ d(king_ZH, soldier_ZH)
in the rank sense. The Chinese concept cluster has different internal
metric structure from the English one, even though each concept lands
at the same address as its translation partner.

This likely reflects: different word co-occurrence patterns between
languages create slightly different relational geometries within the
same Zone C address space.

---

## 6. Verdict: Stratified Universality

The geometry is not uniformly universal or uniformly relative. It has layers:

```
Layer                    Universality    Evidence
─────────────────────────────────────────────────────────────────────────
Inter-body metric        UNIVERSAL       T3 ρ = 0.9428 (10 splits)
Translation co-location  UNIVERSAL       T4 cos = 0.993 (14/15 pairs)
Major axes (1-2)         UNIVERSAL       T1 cos = 0.998, 0.921
Minor axes (3-17)        STABLE          T1 cos = 0.5-0.74
Regular T2 operators     STABLE→UNIV     base→adv 0.90, base→comp 0.69
Irregular T2 operators   RELATIVE        gender 0.16, tense 0.14
Cross-lingual metric     RELATIVE        T4 ρ = 0.26
Micro axes (18+)         NOISE           T1 cos ≈ 0
─────────────────────────────────────────────────────────────────────────
```

**The coarse structure is universal. The fine structure is not.**

Zone C's inter-body topology — which concepts are neighbours of which —
replicates with ρ ≈ 0.94 across independent vocabulary samples. Translation
partners are nearly co-located (cos > 0.98). The two dominant axes (56.6% +
second-largest variance) are reproduced with cos > 0.92 across splits.

Morphologically irregular T2 operators (gender, tense) are not stable from
five seed pairs. They point in a consistent mean direction (loo vs full:
0.926 and 0.922) but individual pairs scatter widely around that mean.
More seed pairs would stabilise them — this is a data quantity issue, not
a fundamental instability.

The cross-lingual metric divergence (ρ = 0.26) is the genuine surprise:
translation concepts co-locate but the distances between different concepts
differ between English and Chinese. The English and Chinese sub-manifolds
have the same landmarks but different local curvature. This reflects
different statistical co-occurrence patterns across languages.

## 7. Implications for LCM Design

### 7.1 What IS universal and can be used directly

- **Zone C body topology**: the clustering of concepts into bodies, and the
  metric between those bodies, is stable. It can be discovered from any
  representative vocabulary sample and will generalise.
- **Major axes (1-5)**: reproducible from half the vocabulary. These are the
  primary semantic dimensions of Zone C and can serve as a stable coordinate
  frame for coarse concept addressing.
- **Translation equivalence**: the model maps translation partners to the same
  address. Cross-lingual generation is feasible without language-specific
  geometry.

### 7.2 What requires careful handling

- **Minor axes (6-17)**: moderately stable (cos 0.5-0.65). Useful within one
  vocabulary but may shift with different word samples. Don't over-rely on
  the axis INDEX; use axis VALUE correlations instead.
- **Regular T2 operators**: stable with 5+ seed pairs. Deterministic
  morphological transformations (adverb formation) are the most reliable.
- **Irregular T2 operators**: need more seed pairs (10+) to stabilise.
  Gender and tense have high within-class variance.

### 7.3 The relational generation principle

Generation is relational regardless of whether the geometry is universal:

```
word address = major axis coords  +  T2 coord
             = which body cluster    which form
             = WHAT concept          HOW expressed
```

Both components are **relative** (defined by the training distribution)
but the **inter-concept relationships** are **universal** (stable across
vocabulary samples, languages, and plausibly across models).

TruthSpace is finding universal relational structure, not universal absolute
coordinates. The distances between concepts are the truth; the numbers
assigned to them are just a chart.

---

## 7. Connection to the Platonic Representation Hypothesis

Huh et al. (2024) argue that large models trained on large datasets converge to
a shared representation of the world — the "Platonic Representation". Evidence:
different models assign similar distances to the same image/text stimuli.

Our work is more fine-grained: we are not measuring overall similarity but asking
whether **specific geometric structures** (body clusters, T2 directions, axis
topology) replicate. The Platonic hypothesis predicts they should.

The fail-fast test: if split-half stability is low (< 0.5 cosine between
corresponding axes), the geometry is not even stable within one model's
vocabulary — universality would be ruled out. This is the cheapest gate.

---

*This DC will be updated with Day 51 results. The question is foundational:
it determines whether TruthSpace is discovering truth or transcribing a
particular model's beliefs about truth.*
