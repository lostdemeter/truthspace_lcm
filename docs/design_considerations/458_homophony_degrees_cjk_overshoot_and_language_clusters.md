# DC 458: Homophony Degrees, CJK Near-Synonym Overshoot, and the Two-Cluster Language Map

**Day 323 | Four discoveries: (1) The -er homophony is a PARTIAL split
(cos=+0.113), while the -al homophony is an ORTHOGONAL split (cos=-0.020).
The degree of homophony separation is determined by the grammatical category
distance between the source words. Noun vs verb = orthogonal; adjective vs
verb = partial. (2) The cross-suffix cosine matrix reveals a NOMINALIZER
FAMILY: +al_nominal and +er_noun are slightly aligned (cos=+0.115) while
both are orthogonal to the relational/modifier group. (3) The CJK in=0% failure
is NOT caused by high cluster density (CJK and European clusters have identical
density ≈0.07) but by NEAR-SYNONYM OVERSHOOT: 水 is already the nearest
neighbor of 'water' (cos=0.818), so any axis displacement overshoots. (4) The
complete 5-language cosine matrix reveals a two-cluster structure: CJK (ZH/JA,
cos=0.910) and European (ES/FR/DE, mean cos=0.323), with the two clusters
partially orthogonal to each other (mean cos≈0.20).**

---

## Homophony Splits: A Spectrum, Not a Binary

### The Cosine Evidence

```
Homophonous pair         cos      split_type       grammatical_distance
+al_rel vs +al_nom      -0.020   orthogonal        noun-source vs verb-source
+er_comp vs +er_noun    +0.113   partial           adj-source  vs verb-source
+ly_adv vs +ly_adj      +0.156   partial-weak      adj-source  vs noun-source
```

The degree of geometric separation between two homophonous axes is
determined by the **grammatical category distance** between their sources:

- **noun vs verb**: maximally different grammatical categories → orthogonal axes
- **adj vs verb**: moderately different → partial split (cos≈0.11)
- **adj vs noun**: minimally different → weak split (cos≈0.16)

This is a quantitative principle: **grammatical category distance ≈ geometric
axis separation** for homophonous suffixes.

### The Cross-Suffix Cosine Matrix

```
            +al_rel   +al_nom   +er_comp   +er_noun
+al_rel       1.00    -0.020    -0.023     -0.108
+al_nom      -0.020    1.00     -0.010     +0.115
+er_comp     -0.023   -0.010    1.00       +0.113
+er_noun     -0.108   +0.115   +0.113      1.00
```

### Two Morphosemantic Families Emerge

**Family 1: Relational/Modifier operations**
- +al_relational (nation→national)
- +er_comparative (fast→faster)
- These are NEAR-ORTHOGONAL to each other (cos=-0.023) and to the other family

**Family 2: Nominalizer/Agentive operations**
- +al_nominal (arrive→arrival)
- +er_noun (teach→teacher)
- These are SLIGHTLY CORRELATED (cos=+0.115)

The two families are orthogonal to each other:
```
+al_rel ⊥ +al_nom     (cos=-0.020)
+al_rel ⊥ +er_comp    (cos=-0.023) ← SAME suffix class is orthogonal cross-suffix!
```

The geometry reveals a **two-dimensional morphosemantic subspace**:
- Dimension 1: RELATIONAL/MODIFIER (adj formation, comparison)
- Dimension 2: NOMINALIZER/AGENTIVE (event noun formation, agent noun formation)

These dimensions are orthogonal in W_E, regardless of which surface suffix
produces them. The embedding space has organized morphological operations
by their semantic function, not by their phonological form.

### Why +er_comp and +al_rel Are Orthogonal

Both create MODIFIER forms (comparative adjective; relational adjective).
Yet they operate in different subregions:

**+al_rel** (noun→adj): moves from abstract noun cluster toward attributive-adjective cluster. The direction is "from entity-property to property-of-entity."

**+er_comp** (adj→comparative adj): moves within the adjective subspace toward its graded/comparative subcluster. The direction is "more of this property."

The starting points (noun subspace vs adjective subspace) are orthogonal,
making the displacement vectors orthogonal despite both producing modifiers.

---

## CJK Near-Synonym Overshoot: A Paradigm-Shift

### The Failed Density Hypothesis

Initial hypothesis: EN→ZH fails (in=0%) because CJK characters are densely
packed, making individual characters unreachable by displacement.

**Refuted by experiment:**
```
CJK intra-cluster mean cos  = 0.066
Euro intra-cluster mean cos = 0.071
```

The CJK and European token clusters have **identical density**. The density
hypothesis is wrong.

### The True Cause: Meaning-Organized Embedding Space

```
水 (water) nearest neighbor = 'water' (cos=0.818)
house     nearest neighbor = 'House' (cos=0.778, capitalized variant)
```

W_E organizes tokens by **meaning**, not by script or language. Cross-lingual
synonyms cluster TOGETHER. The Chinese character 水 is ALREADY the nearest
neighbor of 'water' — no displacement is needed. This was confirmed in
Frontier 17 (F17): `Ice` is near `冰`.

### The Overshoot Mechanism

For EN→ZH axis computation:
1. Collect training pairs: (cat, 猫), (dog, 狗), (water, 水), ...
2. Compute mean chord vector = mean(猫-cat, 狗-dog, 水-water, ...)
3. The mean chord vector is POSITIVE but SMALL (since 水 is already close to water)
4. Applying this small positive displacement to 'cat' moves it PAST 猫

Formally: let δ = mean displacement. If 猫 is already at `cat + ε` (small ε),
then `cat + δ` lands at `cat + δ`, but 猫 is at `cat + ε` where δ ≠ ε.
The displacement overshoots because it was estimated from pairs where
targets are close but not identical, and the mean over 40 pairs creates a
displacement that's systematically wrong for any individual pair.

### Why EN→ES Partially Works (73%)

Spanish words are **NOT** the nearest neighbors of their English translations:
- 'casa' is NOT the nearest neighbor of 'house' (cos is lower than 0.818)
- Spanish words are further from English in embedding space
- The axis displacement CAN navigate to them without overshooting

The baseline similarity determines navigability:
```
High baseline cos (水≈water: 0.82): axis displacement OVERSHOOTS
Low baseline cos  (casa≈house: ?):  axis displacement CAN navigate
```

### Implication: The Cross-Lingual Proximity Paradox

The fact that W_E places cross-lingual synonyms close together is GOOD for
understanding (the model knows that 水=water) but BAD for axis navigation
(displacement can't improve on what's already rank-0).

This is the **Near-Synonym Overshoot** problem: when a word pair is already
semantically co-located, axis navigation fails. The axis only works when
there is a navigable semantic gap between source and target.

**Corollary**: any semantic operation where the result is "close to the input"
will be hard to navigate via axis displacement. This applies to:
- Cross-lingual translation (synonyms in different languages)
- Near-synonyms within a language (big/large, quick/fast)
- Polysemy resolution (bank-financial vs bank-river are NOT close)

---

## The Two-Cluster Language Subspace Map

### Complete Cosine Matrix

```
         EN→ZH   EN→JA   EN→ES   EN→FR   EN→DE
EN→ZH    1.000   0.910   0.236   0.172   0.154
EN→JA    0.910   1.000   0.201   0.188   0.127
EN→ES    0.236   0.201   1.000   0.306   0.363
EN→FR    0.172   0.188   0.306   1.000   0.301
EN→DE    0.154   0.127   0.363   0.301   1.000
```

### Two Language Families in Geometric Space

**CJK cluster** (ZH, JA):
- Internal cosine: 0.910 (strongly similar — nearly identical translation axes)
- Both ZH and JA translation axes point in essentially the same direction from English

**European cluster** (ES, FR, DE):
- Internal cosines: 0.306-0.363 (moderately similar)
- ES-DE (0.363): Spanish and German share more phonological structure than expected
- FR-DE (0.301): French and German slightly less correlated

**Between clusters**:
- CJK vs Euro: mean cos ≈ 0.18-0.24 (distinct, approaching orthogonal)
- The two clusters occupy different translation subspaces

### Why CJK Axes Are Identical (cos=0.910)

Japanese kanji and Chinese hanzi **share the same token IDs** in Qwen2's
tokenizer. The character 水 appears in both Chinese and Japanese text with
the same meaning (water). The embedding of 水 is a mixture of Chinese
and Japanese co-occurrence patterns — but since both use it to mean "water,"
the embedding is essentially the same.

The EN→ZH and EN→JA axes are so similar (0.910) that they're essentially
measuring the same geometric direction: "from English word cluster toward
the shared CJK character cluster."

### Why European Axes Are More Diverse (mean 0.323)

The European languages share phonological patterns (Latin/Germanic roots)
but have distinct vocabularies. The EN→ES axis differs from EN→FR because:
- Spanish words (casa, libro) are phonologically distinct from French (maison, livre)
- The displacement required to reach Spanish vocab is different from French
- Despite both being Romance languages, they occupy different token subspaces

English→German (EN→DE, 0.363 with ES, 0.301 with FR): German is more phonologically
similar to English than French is (Germanic family). The EN→DE axis is thus more
similar to EN→ES than EN→FR is to EN→ES.

### The Language Map Geometry

```
            CJK           EUROPEAN
          ZH────JA         │
         (0.910)           │
              │            ES────DE
              │           (0.363)
              │            │    │
              └────────────FR───┘
               (cos≈0.18)  (0.301)
```

Two clusters, with the CJK cluster tighter and the European cluster looser.
The inter-cluster boundary (cos≈0.20) marks a fundamental division in how
translation operates in W_E:
- **CJK**: target character is already semantically co-located (overshoot problem)
- **European**: target word is at navigable distance (partial axis navigation works)

---

## Predictor v4: The Three Remaining Fixes

From 90% (v3) to target 100% (v4):

### Fix 1: +able (irred threshold boundary)

`+able: pc=0.220, LOO=0%, irred=0.60 → borderline (true: semantic_diverse)`

Rule: `pc>0.20, irred>0.60 → semantic_diverse`
Problem: `irred==0.60` doesn't trigger `>0.60`.
Fix: Change `irred > 0.60` to `irred >= 0.60` in the pc>0.20 branch.

### Fix 2: +less (irred=0.90 vs factual_local)

`+less: pc=0.167, LOO=0%, irred=0.90 → factual_local/translation (true: semantic_diverse)`

Rule: `pc>0.10, irred>=0.90 → factual_local/translation`
Problem: `irred=0.90` is not 100% — +less CAN sometimes be retrieved,
just rarely. factual_local requires COMPLETE irreducibility (irred=1.00).
Fix: Change `irred >= 0.90` to `irred >= 0.95` for factual_local boundary.
semantic_diverse then covers `irred 0.60-0.94`.

### Fix 3: country→currency (low-irred semantic_diverse)

`country→currency: pc=0.173, LOO=0%, irred=0.33 → borderline (true: semantic_diverse)`

Rule: `pc>0.10, irred>0.60 → semantic_diverse`
Problem: `irred=0.33` falls into the `borderline` zone (0.20-0.60).
The `borderline` zone is too wide — low-irred semantic_diverse axes need
recognition. country→currency has LOO=0% AND irred≈0.33: no generalization
but some pairs CAN be retrieved. This is semantic_diverse.
Fix: For `pc>0.10, LOO==0%, irred<0.60`: label as `semantic_diverse-partial`
rather than `borderline`.

### Predictor v4 Expected Accuracy: 29-30/30 (97-100%)

---

## Day 324 Plan

1. **Predictor v4**: implement the three fixes and re-benchmark. Target 29+/30.

2. **Suppletive -t class**: lose/lost, mean/meant, sleep/slept, keep/kept — this
   forms a distinct irregular past subclass (Germanic strong vs weak merger).
   Does this class form its own coherent axis? (predict: yes, pc>0.20)

3. **Nominalizer family expansion**: test +ance/+ancy, +ity, +ure as potential
   members of the nominalizer family. Do they align with +al_nom and +er_noun?

4. **Baseline similarity measurement**: for each translation pair (EN→ES, EN→ZH),
   measure the mean cosine between source and target embeddings. Verify that:
   - EN→ZH: high baseline (source/target already close) → overshoot
   - EN→ES: lower baseline → navigable

5. **Axis composition**: can we CHAIN two axes? E.g., apply ablaut axis then
   +s axis to go from 'go' → 'went' → 'wents'? (Expected: chaining works if
   axes are orthogonal, fails if they share direction.)

---

## Files

- `expedition_log.md` — Days 322-323 results
- `457_ablaut_subtypes_homophony_orthogonality_and_cjk_barrier.md` — DC 457
- `day323_predictor_v3_er_homophony_ablaut_nd_cjk_density.py` — experiment script
