# DC 464: GROUP E Expanded, +ly Standalone, EN→ZH Generalizes, Benchmark Flaw

**Day 329 | Four discoveries: (1) GROUP E (verb inflection) expands to include
+ing: cos(+ing,+3ps)=0.404, the highest single intra-GROUP E cosine. +ing and
+3ps share a PRESENT-TENSE source direction, which is why their cosine exceeds
the +ed_reg/ablaut cosines (0.28-0.30). (2) +ly (adj→adverb) is a standalone
adj-source axis — no GROUP F. Its cosines with adj-sourced axes (+ness=0.289,
+er_comp=0.293) reflect the source category correlation, not a new family.
(3) The EN→ZH translation axis generalizes beyond training: computer/music/
science correctly translate OOD, 3/5 success. Failures (table, garden) have
multi-token Chinese representations. (4) The 30-axis predictor benchmark has
a design flaw: self-holdout gives irred=0% for all axes, making the irred
dimension invisible. Predictor v6 improvements require proper separate holdout.**

---

## GROUP E Expanded: The Full Verb Inflection Cluster

### Complete Intra-GROUP E Cosine Matrix

```
                ed_reg  ablaut  +3ps   +ing
+ed_reg          —      0.411   0.384  0.284
ablaut          0.411    —      0.357  0.300
+3ps            0.384   0.357    —     0.404
+ing            0.284   0.300   0.404   —

Mean intra-group cosine: 0.349
```

### The Tense Sub-Structure Within GROUP E

The cosines reveal a sub-structure:

```
PAST TENSE pair:  {+ed_reg, ablaut}   cos=0.411  (strongest — both past)
PRESENT pair:     {+3ps, +ing}        cos=0.404  (second strongest — both present)
CROSS-TENSE:      {ed_reg, 3ps}=0.384, {ablaut, 3ps}=0.357,
                  {ed_reg, ing}=0.284, {ablaut, ing}=0.300
```

The pattern: within-tense cosines (past-past, present-present) are higher than
cross-tense cosines. This is tense sub-clustering within GROUP E.

### Interpretation

All four axes operate on BASE VERB forms. The direction they travel in W_E is
the "verb inflection direction." But within this direction, there are sub-axes:
- PAST sub-direction: "base verb → past tense cluster"
- PRESENT sub-direction: "base verb → present-time cluster"

These sub-directions are close (cos≈0.38 cross-tense) but not identical. The
present-time cluster (+3ps, +ing) is slightly different from the past-time cluster
(+ed_reg, ablaut). This makes sense: past tense forms and present participles
occupy distinct regions in W_E based on their semantic/syntactic roles.

### +plural as GROUP E Periphery

```
cos(+3ps, +s_plural) = +0.223
cos(+ing, +plural)   = +0.106
```

+s_plural is the "surface -s sibling" of +3ps. They share the -s suffix form,
creating a partial geometric alignment. But +s_plural is a NOUN inflection while
+3ps is VERB inflection — the source categories differ, pushing them apart.

```
FULL INFLECTION CLUSTER (sorted by group membership):
GROUP E core: +ed_reg, ablaut, +3ps, +ing (mean cos: 0.349)
GROUP E periphery: +s_plural (cos 0.11-0.22 with core)
```

---

## +ly: Standalone Adj-Source Axis

### Evidence Against GROUP F

```
cos(+ly, +ness)   = +0.289  (adj → quality noun)
cos(+ly, +er_comp)= +0.293  (adj → comparative adj)
cos(+ly, +able)   = +0.072  (verb → ability adj — DROPS because source differs)
cos(+ly, +ance)   = +0.038  (verb → event noun — near-zero)
```

The +able axis has adj-like TARGETS but VERB sources. Its cos with +ly is only
0.072 — much lower than +ness and +er_comp which have adj SOURCES. This confirms
the source category drives the positive cosine for adj-sourced axes.

### The Adj-Source Base Cosine

All adj-sourced axes share a base cosine of approximately 0.27-0.30:
```
cos(+ness, +er_comp) = +0.280
cos(+ness, +ity)     = +0.343
cos(+ly, +ness)      = +0.289
cos(+ly, +er_comp)   = +0.293
cos(+ity, +er_comp)  = +0.185
```

This "base" represents the common component of departing from the adjective
cluster. Any axis with adjective sources will share this base cosine with all
other adj-source axes, even when targets differ.

The remaining variance above 0.29 is explained by target similarity:
- cos(+ness, +ity) = 0.343 > 0.29 because both target ABSTRACT QUALITY NOUNS
- cos(+ness, +er_comp) = 0.280 ≈ 0.29 because targets differ (noun vs adj)
- cos(+ly, +er_comp) = 0.293 ≈ 0.29 because targets differ (adverb vs adj)

### +ly Classification

```
+ly: pc=0.152  LOO=81%  irred=0%  → phonol_scatter
```

Despite high LOO (81%), the +ly axis classifies as phonol_scatter due to moderate
pc (0.152). This is correct: the +ly transformation is very regular but not as
geometrically tight as morph_moderate axes (pc > 0.20). The adjective→adverb
transformation has moderate internal variation (happy→happily vs quick→quickly
take different paths due to -y→-ily vs simple -ly).

---

## EN→ZH Translation Axis: Generalization Beyond Training

### The Finding

```
Training pairs: sun/moon/water/fire/mountain/hand/eye/fish + 7 more
OOD test results:
  computer → 计算机 ✓  (expected 电脑, but 计算机 also means "computer")
  music    → 音乐   ✓  (exact match)
  science  → 科学   ✓  (exact match)
  table    → (table ✗  (Chinese 桌子 is multi-token)
  garden   → gardens ✗ (Chinese 花园 is multi-token, lands at English plural)
```

3/5 OOD translation accuracy. The axis trained on 15 common word pairs correctly
translates abstract/functional English nouns to their Chinese equivalents.

### Why Generalization Works

The EN→ZH axis points in the "English vocabulary → CJK vocabulary" direction in
W_E. This direction is consistent for:
- Concrete nouns (training set: sun, moon, water, fire)
- Abstract/technical nouns (OOD: computer, music, science)

The axis fails for:
- Words whose Chinese translations are multi-token (table→桌子/两字)
- Words with contextual Chinese alternatives (garden→花园 vs 园林)

### Cosine Structure

```
cos(EN→ZH, OOD→CJK_void) = +0.253
cos(EN→ZH, +al_rel)      = +0.063
```

The OOD→CJK void axis (computed from OOD words navigating to Chinese) is
partially aligned (0.253) with the direct EN→ZH training axis. This means:
- Morphological overshoot into CJK territory follows a TRANSLATION-LIKE direction
- Not random: there is a consistent geometric direction from English to Chinese
- This direction is partially learned from direct translation pairs AND from
  structural co-occurrence of English and Chinese text

The near-zero cos(EN→ZH, +al_rel) = 0.063 confirms that morphological axes
(+al_rel) are ORTHOGONAL to the translation direction. Applying +al_rel to
'computer' gives Chinese — but not because +al_rel IS a translation axis.
Rather, it's because +al_rel overshoots into a region where the nearest single
token happens to be Chinese.

---

## Benchmark Design Flaw: The Self-Holdout Problem

### What Happened

```python
irr_f, _, _ = irred_on_holdout(ax, pairs[:4], RELAXED_MASK)
```

Using `pairs[:4]` as holdout when the axis was trained on ALL pairs means:
- The axis has already "seen" these pairs during training
- The axis navigates its training distribution perfectly: irred≈0%
- The irred dimension carries no information

Result: v5 vs v6 BOTH get 18/30=60% because the irred-based rules never trigger.

### The Regression from v6

v6 lowered the top threshold from pc>0.35 to pc>0.32. This caused:
```
ablaut: pc=0.345
v5: pc=0.345 < 0.35 → morph_moderate        (WRONG but was consistently wrong)
v6: pc=0.345 > 0.32 → morph_uniform/re      (ALSO WRONG, differently)
```

The v6 change introduced a regression for ablaut. The correct approach:
ablaut should be phonol_scatter (high LOO from irregular past tense verbs),
but pc=0.345 pushes it above both thresholds. The true fix is to add a
"IRREGULAR past tense" category with pc=0.30-0.40, LOO>50%, irred=0-20%.

### The Properly Fixed Benchmark (for Day 330)

Split each 8-pair set: 5 train + 3 holdout. This gives:
- Proper irred estimates from genuinely unseen pairs
- The v6 irred override can trigger for appropriate axes
- The full 3D feature space (pc, LOO, irred) is measured correctly

---

## Day 330 Plan

1. **Fixed benchmark**: re-run 30-axis suite with 5 train + 3 proper holdout.
   Compare v5 vs v6 accuracy.

2. **Tense sub-clustering test**: verify that cos(+ing, +3ps) > cos(+ed_reg, +ablaut)
   consistently across multiple pair subsamples.

3. **Cross-lingual composition**: can the EN→ZH axis chain with +s_plural to give
   plural Chinese words? E.g., does 计算机 (computer) navigate to 计算机们 (computers)?

4. **+re- prefix axis**: is there a GROUP D counterpart for REVERSAL (+al_rel is
   noun→adj, but +re- is verb→reversed_verb: do/redo, write/rewrite)?
   Prediction: high pc (same morphological pattern), very low irred.

5. **GROUP map visualization summary**: compile all confirmed groups and their
   cosine relationships into a single reference table.

---

## Files

- `expedition_log.md` — Days 322-329 results
- `463_group_e_verb_inflection_probe_ceiling_and_bilingual_void.md` — DC 463
- `day329_v6_ing_ly_bilingual_group_e_probe.py` — experiment script
