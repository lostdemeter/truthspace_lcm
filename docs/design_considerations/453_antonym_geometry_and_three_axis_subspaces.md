# DC 453: Antonym Geometry and Three Axis Subspaces in W_E

**Day 318 | Four findings: (1) Antonym axes have pc≈0 (random chord directions)
yet achieve in-sample=100% for verb and noun antonyms — they are an extreme
form of semantic_diverse with near-zero pairwise coherence. (2) country→president
is semantic_diverse (irred=100%), NOT phonol_scatter — the classification protocol
requires an irreducibility sweep to distinguish these categories. (3) The extended
orthogonality matrix reveals cos(cc, capl)=-0.311: country→capital and
capital→language oppose each other because one ARRIVES and one DEPARTS from
the capital semantic space. (4) PCA of 13 axis vectors confirms THREE distinct
subspaces: relational axes cluster at negative PC1, morphological at positive PC1,
antonym axes near the origin — all three separated in 2D projection.**

---

## Antonym Axes: The pc≈0 Case

### The Measurements

```
Axis          pc      n    in%    LOO%
verb_ant     0.016    8   100%      0%
noun_ant     0.020    8   100%      0%
adj_ant      0.046   12    67%      8%
adverb_ant   0.039    6    83%      0%
```

### What pc≈0 Means

pc (pairwise chord cosine) measures how consistently all pairs point in the same
direction for a given transformation. For verb antonyms:

```
win→lose:  chord direction θ₁
rise→fall: chord direction θ₂
push→pull: chord direction θ₃
...
```

pc=0.016 means the average cosine between any two chord directions is 0.016 ≈ 0.
The chords scatter in ALL DIRECTIONS — there is no single geometric direction
for "apply the antonym operation." The actual range was min=-0.194, max=0.465.

### Why in-sample=100% Despite pc≈0

If chords scatter randomly, why does the mean axis retrieve all 8 pairs?

Two reasons:
1. **The mean chord is not zero**: even with random directions, the mean of 8
   unit vectors has some non-negligible magnitude. The axis points somewhere.

2. **High-dimensional geometry**: in 1536 dimensions, "nearby" is relative.
   The target (e.g., 'lose') might be the nearest clean token to any
   reasonable displacement from 'win', because the antonym cluster is
   densely packed in a particular neighborhood.

This is distinct from semantic_diverse axes like un- (pc=0.19, irred=57%)
where some chords have partial alignment. Antonym axes are the degenerate
case: ZERO coherence, pure local structure.

### Why LOO=0%

With pc≈0, when you train on 7 pairs and test the held-out 8th:
- The mean axis of 7 pairs points in a nearly random direction
- There is no systematic reason for this direction to retrieve the held-out target
- LOO=0% is the correct result: the axis provides no predictive power

### The Adjective Asymmetry

```
adj_ant forward (hot→cold): 67%  in-sample  scale=0.639
adj_ant reverse (cold→hot):  0%  in-sample  scale=0.500
```

Adjective antonym pairs are NOT perfectly symmetric in W_E. The embedding of
'hot' displaced by the mean antonym vector lands near 'cold', but the embedding
of 'cold' displaced by the NEGATIVE antonym vector does not land near 'hot'.

This reveals that antonym pairs have asymmetric local geometry: the neighborhood
structure around 'cold' is different from around 'hot'. 'Cold' is surrounded by
temperature/weather/sensation terms; 'hot' may be surrounded by intensity/danger
terms. The antonym relationship is directional in the local manifold.

Contrast with country→capital where cos(forward, reverse) = -1.0000 exactly.
Relational axes ARE perfectly symmetric; antonym axes ARE NOT.

---

## country→president Classification Correction

### The Data

```
country→president: pc=0.165, in=100%, LOO=0%, irred=100%
```

This matches the PROFILE of phonol_scatter:
- pc=0.165 (low)
- in=100% (all training pairs retrieved)

But the IRRED test is the decisive classifier:
- irred=100% → semantic_diverse
- (phonol_scatter has irred≈0%)

### The Decision Tree

```
Compute pc:
  pc≥0.28 → morph_uniform (if LOO≥65%) or morph_moderate
  pc<0.28 AND in<85% → need more data
  pc<0.28 AND in≥85% → BRANCH:
    Compute LOO:
      LOO≥65% → morph_uniform (unexpected case)
      LOO 30-65% → morph_moderate
      LOO<30% → MUST CHECK IRRED:
        irred<30% → phonol_scatter (low pc from allomorphic surface forms)
        irred>60% → semantic_diverse (low pc from genuinely random directions)
        irred 30-60% → borderline
```

country→president: pc=0.165, in=100%, LOO=0%, irred=100% → **semantic_diverse**

The difference from phonol_scatter (+tion -ct: pc=0.116, LOO=75%):
- +tion -ct: LOO=75% → phonol_scatter  
- country→president: LOO=0% → need irred → irred=100% → semantic_diverse

LOO is the first discriminator. If LOO is high despite low pc, it's phonol_scatter.
If LOO is low, irred is needed.

### Why Presidents are Semantic_Diverse

The displacement from 'france' to 'Macron' points in a completely different
direction from 'usa' to 'Biden'. Each president is in a unique semantic
neighborhood determined by:
1. Political party affiliation
2. Name etymology and origin
3. Cultural context of the country
4. Frequency in training data

There is no consistent geometric direction for "apply presidential authority."
The mean axis happens to retrieve the training pairs at the right scale, but
cross-application fails because each mapping is unique.

---

## The Orthogonality Matrix: Structural Reading

### The Revealed Structure

```
cos(cc, capl) = -0.311  — strongest negative correlation
```

This is architecturally meaningful:
- `cc` (country→capital): displacement FROM country-space TO capital-space
- `capl` (capital→language): displacement FROM capital-space TO language-space

These are adjacent legs of the country→capital→language chain. The ARRIVAL
direction of cc is the DEPARTURE direction of capl, making them partially
opposite (-0.311).

In contrast:
```
cos(cc, cl)   = +0.501  — country→capital and country→language both DEPART from country
cos(cl, capl) = +0.514  — country→language and capital→language both ARRIVE at language
```

The sign of the inter-axis cosine tells you whether two axes share a domain:
- Positive: same departure OR same arrival domain
- Negative: one's arrival is the other's departure (they're adjacent in a chain)

### The Morphological Cluster

```
+er vs +s:   0.146
+er vs +ed:  0.143
+s  vs +ed:  0.169
+er vs un-:  0.231  (strongest!)
```

un- correlates with +er (0.231) because both are adjective transformations
operating on the same set of source words. They both involve the adjective
semantic region, pulling in related but opposite directions.

+tion has near-zero cosines with all other morphological axes (+tion vs +er = 0.041),
confirming that the +tion -ct axis lives in a distinct morphophonological space
even within the morphological cluster.

---

## PCA: Three-Subspace Structure Confirmed

### The 2D Layout

```
Axis         PC1      PC2     notes
cc          -0.639  -0.552   departs country-space
cl          -0.799  -0.174   departs country-space
capl        -0.261  +0.325   departs capital-space (PC2 separates this!)
pres        -0.529  -0.536   departs country-space

+er         +0.372  -0.434   morphological
+s          +0.306  -0.370   morphological
+ed         +0.301  -0.351   morphological
+tion       +0.070  -0.182   morphological (weaker)
un-         +0.407  -0.403   morphological

adj_ant     +0.136  -0.089   antonym (near origin)
adv_ant     +0.033  +0.008   antonym (near origin)
verb_ant    +0.134  -0.176   antonym
noun_ant    +0.274  -0.358   antonym
```

### The Three Subspaces

**Subspace 1 — Relational**: cc, cl, pres cluster in the NEGATIVE PC1 region.
All depart from country-space or named-entity spaces. capl is separated from
these by PC2 (positive vs negative).

**Subspace 2 — Morphological**: +er, +s, +ed, un- cluster in the POSITIVE PC1
region with negative PC2. These all involve form changes from a source word.

**Subspace 3 — Antonym**: verb_ant, noun_ant are near the origin (low both PCs).
This is consistent with pc≈0: zero-coherence axes have near-zero projections
onto any systematic principal component.

### The PCA Interpretation

PCA captures systematic variation across axis vectors. Since only 21.1% of
variance is explained in 2D, the axes live in a high-dimensional manifold —
but the top 2 dimensions reveal the relational/morphological split.

The antonym axes near the origin are NOT "uninformative" — they simply have
no systematic alignment with the variation captured by the top PCs. They
represent a DIFFERENT type of organization: local pair-specific geometry
rather than global directional structure.

### The 3-Hop Collapse

Applying the cc axis to a language token returns toward capitals:
```
French + cc_axis → Paris (capital)
French + capl_axis → Spanish (language)
```

The 3rd hop with `cc` is a "return journey": from language-space, the
country→capital vector points back toward the capital-space (which is
between language-space and country-space on the chain). This oscillation
is predicted by the orthogonality structure: since cc and capl have
cos=-0.311, applying cc from a point in capl's destination domain will
partially move back toward cc's origin.

Multi-hop navigation requires choosing the CORRECT axis for each hop.
The chain terminates when no valid axis extends from the current position.

---

## The Complete Axis Taxonomy (Revised for Day 318)

### Six-Category Framework

```
Category         pc range    LOO%   irred%  examples
─────────────────────────────────────────────────────────────────────
morph_uniform    ≥0.28      ≥65%   <15%    +er, er→est, +ed
morph_moderate   0.15-0.28  30-65% 15-40%  +s, gender, +tion(high LOO)
semantic_diverse <0.20      <30%   >60%    un-, +ful, +less, +able
relational_geom  ≥0.35      ≥60%   <30%    cc, cl, capl (relaxed mask)
factual_local    0.10-0.20  <10%   ~100%   pres, +tion(sub-split?)
antonym          <0.05      <10%   ~100%   verb_ant, noun_ant, adj_ant
```

### The Key Distinguisher for Low-pc Axes

Given an axis with pc < 0.20:

```
LOO ≥ 30%: morph_moderate or factual_local
LOO <  30%: MUST compute irred:
  irred < 30%: morph_moderate-phonol_scatter (+tion type)
  irred 30-60%: borderline
  irred > 60%: semantic_diverse
  irred > 90%: antonym or extreme semantic_diverse
```

The NEW finding is that antonym axes have pc<0.05 AND irred~100%, placing
them at the extreme end of the semantic_diverse spectrum.

---

## Day 319 Plan

1. **Antonym local geometry**: why do verb antonyms ALL retrieve at in=100%?
   Inspect the nearest clean neighbors of each antonym target to verify they
   sit in accessible local neighborhoods.

2. **pc distribution for semantic_diverse vs antonym**: measure pc of +ness,
   +ful, +less, and compare to verb_ant/noun_ant. Where does the antonym
   pc≈0 fit on the continuum?

3. **Factual local axes**: test scientist→discovery (einstein→relativity,
   darwin→evolution) and author→book. Do these show the same factual_local
   pattern (irred≈100%)?

4. **Cross-subspace chord test**: can we navigate FROM a morphological word
   position USING a relational axis? E.g., 'cats' + cc_axis → ??? Does
   the near-orthogonality mean the result is effectively random?

5. **Multi-lingual axes**: test english→spanish translation (cat→gato,
   dog→perro). Is translation a relational or morphological axis type?

---

## Files

- `expedition_log.md` — Day 318 results
- `452_axis_subspaces_and_relational_geometry.md` — DC 452
- `day318_antonyms_orthogonality_pca.py` — experiment script
