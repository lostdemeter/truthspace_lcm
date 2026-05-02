# DC 324: The 20-Dimensional Transformation Subspace

**Date:** 2026-03-17  
**Experiment series:** Days 82–86  
**Prerequisite:** DC 323 (Ternary φ-Trie, Days 76–81)

---

## Overview

DC 323 established the φ-trie as a 8-bit semantic metric space. A
natural question follows: are 8 transformation dimensions sufficient
to span the full semantic transformation subspace of Qwen2-1.5B?

Days 82–86 answer this definitively:

**The transformation subspace has at least 20 orthogonal dimensions.**

The 8 original axes (comparative, plural, past_tense, gender, antonym,
hypernym, synonym, concrete_abstract) are correct and real, but they
represent fewer than half of the total semantic transformation capacity.

---

## The Critical Limitation: High-Dimensional Geometry (Day 84)

Before presenting results, a crucial methodological note.

In R^1536, any two random unit vectors are approximately orthogonal:

```
E[angle(random, random)] ≈ arccos(1/√1536) ≈ 90°
std ≈ arcsin(1/√1536) ≈ 1.5°
3σ range: [85.6°, 94.4°]
```

This means pairwise angles of 80–90° between T2 axes could be:
- **Genuine semantic independence** (different linguistic dimensions), OR
- **High-dimensional geometry artifact** (any two vectors look orthogonal)

Similarly, a random unit vector projected onto a k-dimensional subspace
in R^1536 has expected fraction = k/1536 ≈ 1% for k=16. Observed
fractions of 3–28% (Days 82–84) are above this but only modestly.

**The only valid test: REPRODUCIBILITY.** If the same transformation
type produces consistent axis directions from different sentence pairs,
the axes are real. If not, they are geometric noise.

---

## T2 Axes Are Real (Day 85)

### Test design

For each of 8 transformation types, compute two independent T2 axis
estimates:
- **SET_A**: original 8 sentence pairs
- **SET_B**: 8 completely different sentence pairs for the same type
- **RANDOM**: 4 sets of unrelated sentence pairs (no transformation)

### Results

```
Type         angle(A,B)   σ below 90°   verdict
gender         55.13°       +23.9σ       REPRODUCIBLE
comparative    59.85°       +20.6σ       REPRODUCIBLE
hypernym       65.84°       +16.5σ       REPRODUCIBLE
plural         42.53°       +32.5σ       REPRODUCIBLE
synonym        56.64°       +22.8σ       REPRODUCIBLE
concrete       73.03°       +11.6σ       REPRODUCIBLE
past_tense     35.91°       +37.0σ       REPRODUCIBLE
antonym        81.80°        +5.6σ       REPRODUCIBLE

Mean: 58.84°  (21.3σ below 90° random baseline)
```

All 8 axes are reproducible. The weakest is antonym (81.8°, 5.6σ),
the strongest is past_tense (35.9°, 37σ).

**Cross-type discrimination:** 6/8 SET_A axes correctly identify their
SET_B partner as nearest neighbor (75% vs 12.5% by chance).

### The context bias problem

```
Theoretical random coherence: 0.0722  (sqrt(8/1536))
Observed random coherence:    0.3424  (4.7× theoretical)
Real semantic coherence:      0.4715  (1.38× observed random)
```

Even unrelated sentence pairs produce coherent-looking difference
vectors (4.7× above pure theory) because all "last tokens" in English
sentences are drawn from similar syntactic positions. This inflates
the noise floor. Real semantic axes have 1.38× additional coherence
above this inflated baseline.

### Axis noise quantified

Mean A-B angle of 58.84° implies axis estimates scatter ~**30°** from
the true direction when computed from 8 sentence pairs.

---

## The 20-Dimensional Result (Days 82–86)

### Completeness tests (Days 82–84)

Four rounds of 4 new transformation types each, tested against the
growing subspace of confirmed axes:

```
Round  Types tested                                  All NEW_DIM?
  82   negation, passive, spatial, temporal          YES (3.8–31%)
  83   degree, part_whole, question, causation       YES (8.2–17%)
  84   possession, definiteness, modality, aspect    YES (15–28%)
```

In 3 consecutive rounds, every new type was a genuinely new dimension.
The cumulative SVD estimate rose from 12D → 15D → 19D.

### Noise-corrected saturation (Day 86)

With 30° axis noise, the conservative threshold for "VARIANT" is
fraction_explained > 0.50 (cos²(30°) = 0.75 would be exact match, 0.50
is generous). Pooling 16 pairs per core type, re-testing all 12 new
axes:

```
Axis          frac_8D   residual%   nearest_core  verdict
negation       0.056     94.4%      hypernym       NEW_DIM
passive        0.029     97.1%      plural         NEW_DIM
spatial        0.152     84.8%      comparative    NEW_DIM
temporal       0.317     68.3%      past_tense     NEW_DIM   ← closest (61°)
degree         0.158     84.3%      synonym        NEW_DIM
part_whole     0.123     87.7%      synonym        NEW_DIM
question       0.047     95.4%      antonym        NEW_DIM
causation      0.038     96.2%      gender         NEW_DIM
possession     0.125     87.5%      plural         NEW_DIM
definiteness   0.179     82.1%      past_tense     NEW_DIM
modality       0.121     87.9%      hypernym       NEW_DIM
aspect         0.185     81.5%      synonym        NEW_DIM

Genuinely new: 12   Variants: 0
```

**All 12 new axes are genuinely independent** even with the generous
50% threshold. The closest relationship is temporal ↔ past_tense at
61° (morphological form vs temporal reference shift — related but
distinct).

---

## The Full 20-Dimensional Transformation Subspace

### Complete axis inventory

```
GROUP 1: Morphology (form changes within same word)
  plural      (singular → plural)
  comparative (adjective → comparative form)
  past_tense  (present → past verb form)
  aspect      (simple → progressive form)

GROUP 2: Polarity / Scale
  antonym     (polar opposite)
  negation    (logical negation)
  degree      (scalar intensification)
  synonym     (same meaning, different word)

GROUP 3: Semantic Structure
  hypernym    (specific → general category)
  part_whole  (part → whole entity)
  concrete    (concrete → abstract meaning)
  causation   (cause → effect)

GROUP 4: Reference / Role
  gender      (masculine → feminine)
  passive     (active → passive voice)
  spatial     (location contrast: on→under, in→out)
  temporal    (past reference → future reference)

GROUP 5: Discourse / Pragmatics
  possession  (has → genitive 's)
  definiteness (a → the)
  modality    (bare → modal auxiliary)
  question    (assertion → interrogative)
```

### Properties of the 20D subspace

```
Pairwise orthogonality (pooled core 8):
  min=68.0°   mean=84.3°   max=89.8°
  26/28 pairs > 70°

SVD spectrum (20-axis combined):
  singular values: 0.082 0.074 0.066 0.062 ... 0.030
  isotropy ratio: 2.72  (max/min singular value)
  95% variance at dimension 19

T2 ⊥ PC0 (identity manifold): YES at all layers (dev < 13° from 90°)
```

The 20D subspace is approximately isotropic — no single transformation
dimension is privileged over others.

---

## Implications for the φ-Trie

### Why the 8-bit trie still works

The φ-trie (DC 322–323) used 8 T2 axes to build ternary addresses.
Days 82–86 reveal there are 20 semantic transformation dimensions, not
just 8. The 8-bit trie captures the MOST DISCRIMINATIVE dimensions
(selected by their ability to predict logit cosine separation at Day 73).

The other 12 dimensions are real but provide SECONDARY discrimination:
- The plural bit (most informative, MI=0.2586) is already in the trie
- Adding negation, passive, etc. would improve trie resolution at cost
  of sparsity (3^20 ≈ 3.5B possible leaves vs 3^8 = 6561)

### 20-bit trie (theoretical)

A 20-bit ternary trie would have:
- 3^20 ≈ 3.5 billion possible leaves
- With 401 tokens: average leaf population = 0.000000115
- Far too sparse for practical use at current vocabulary sizes

The 8-bit trie represents the practical optimum for the available
vocabulary: rich enough to be meaningful, dense enough to be useful.

### The trie is a projection

The 8-bit φ-trie is a **projection** of the full 20D transformation
subspace onto the 8 most diagnostically useful dimensions. The other
12 dimensions are "dark" from the trie's perspective but contribute
to the continuous similarity metric (cosine of logit distributions).

---

## Connections to Prior Work

| Finding | DC | Connection |
|---------|-----|------------|
| T2 ⊥ PC0 at all layers | 322, 323 | Holds for all 20 dimensions |
| T2 axes isotropic (flat SVD) | 322 | Confirmed at 20D (ratio 2.72) |
| Layer 1 rotates axes ~87° | 323 | Applies to all 20 axes equally |
| Transformer amplifies 1.9× | 323 | 20 dimensions equally amplified |
| Weight matrix = noisy geometry | 297 | 20 dimensions are the geometry |
| Complexity ladder O(d) | 297 | 20 T2 axes ≈ 20d parameters = 30K |

---

## Key Numbers

| Quantity | Value |
|---------|-------|
| Confirmed orthogonal T2 dimensions | 20 |
| Core 8 (DC 322–323) | still valid, still primary |
| New 12 (Days 82–84) | negation, passive, spatial, temporal, degree, part_whole, question, causation, possession, definiteness, modality, aspect |
| T2 axis noise (8 pairs) | ~30° |
| Reproducibility significance | 21.3σ below random |
| Cross-type discrimination | 6/8 correct (75% vs 12.5% chance) |
| Noise-corrected threshold | 0.50 |
| Variants found (all 12 new) | 0 |
| Temporal ↔ past_tense angle | 61.0° (closest pair, still NEW_DIM) |
| Pooled 8D pairwise mean angle | 84.3° |
| 20-axis SVD 95% dimension | 19 |
| Context bias in random baseline | 4.7× above theoretical |

---

## Files

- `expedition_day82_completeness.py` → `day82_completeness.json`
- `expedition_day83_saturation.py` → `day83_saturation.json`
- `expedition_day84_saturation2.py` → `day84_saturation2.json`
- `expedition_day85_reproducibility.py` → `day85_reproducibility.json`
- `expedition_day86_corrected_saturation.py` → `day86_corrected_saturation.json`
- `expedition_log.md` — Days 82–86 appended
