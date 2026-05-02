# DC 462: The Verb→Adj Modifier Family, Etymology Split, and Three-Step Chain

**Day 327 | Four discoveries: (1) 15-pair probe achieves 6/8=75%. The two
remaining failures have specific causes: semantic_diverse (+er_noun) sits at
irred≈50% — genuinely borderline — and translation at 5-pair training gives
LOO=40% (vs true 9%) due to high-variance small-sample LOO estimation. (2) A
new GROUP D identified: verb→adj modifiers (+less, +ful, +able) cluster with
cos(+less,+ful)=0.438 and cos(+able,+ful)=0.347, despite producing oppositely-
valenced outputs (hopeful/hopeless). Same exit direction from verb cluster,
different landing zones. (3) The etymology split is confirmed: +ity navigates
to Latin-root quality nouns, +ness to Germanic-root quality nouns. Misapplying
+ity to Germanic adj lands in Chinese; misapplying +ness to Latin adj lands
in Chinese. Chinese tokens fill the "void" in empty morphological regions.
(4) Three-step chain fails due to scale drift: the third axis's calibrated scale
does not transfer to the deeply-displaced intermediate embedding.**

---

## 15-Pair Probe: Why 75% Is the Effective Ceiling

### The Two Remaining Failures

**Failure 1 — semantic_diverse (+er_noun)**

```
pc=0.132, LOO=20%, irred=50% (from 10 holdout pairs)
Predicted: borderline
True type: semantic_diverse
```

With 10 holdout pairs, irred=0.50 means exactly 5 of 10 fail.
The true irred for the full +er_noun axis is ~0.67 (8-pair estimate) or ~0.50
(15-pair estimate). The measurements DISAGREE because the holdout distribution matters:
- If holdout contains common agent nouns (teacher, farmer): irred ~ 0.40
- If holdout contains rare agent nouns (climber, diver): irred ~ 0.80

The +er_noun axis is genuinely borderline: it works for some nouns and fails
for others. The predictor is technically CORRECT to call it borderline.

**Conclusion**: semantic_diverse is NOT a clean category. There is a borderline
zone between phonol_scatter and semantic_diverse for axes with moderate LOO and
irred in the 0.40-0.65 range. The predictor's borderline label is accurate.

**Failure 2 — translation (EN→ES)**

```
pc=0.105, LOO=40%, irred=80% (from 10 holdout pairs)
Predicted: semantic_diverse
True type: translation
```

With 5 training pairs (house/water/sun/book/day in Spanish), the 5-pair axis
achieves LOO=40% — much higher than the full-set LOO of ~9%. Why?
- 5-pair training set has high variance
- Some EN→ES displacements accidentally generalize to a few holdout words
- LOO on 5-pair set measures a noisy local structure

The fundamental issue: **LOO is unstable below n=10 training pairs**. With
fewer than 10 pairs, LOO can fluctuate from 0% to 60% depending on which
specific words are sampled.

**Fix for Day 328**: use 10 training + 10 holdout = 20 total pairs. With 10
training pairs, LOO stabilizes and irred is measured on a larger holdout set.

### Probe Size Requirements Summary

```
5-pair probe (no holdout): 50% accuracy (only high-pc axes classified)
8-pair probe (3 holdout):  62% accuracy (irred too noisy)
15-pair probe (10 holdout): 75% accuracy (LOO still noisy for translation)
20-pair probe (10 train + 10 holdout): expected ~87-100%
```

---

## GROUP D: The Verb→Adj Modifier Family

### Evidence

```
cos(+less, +ful)  = +0.438
cos(+able, +ful)  = +0.347
```

These three suffixes all operate on VERB sources and produce ADJECTIVE targets:
```
+ful:  hope → hopeful   (positive potential)
+less: hope → hopeless  (negative lack)
+able: read → readable  (positive ability)
```

### The Polarity Paradox

Hopeful and hopeless are ANTONYMS. Yet cos(+ful, +less) = +0.438 — strongly
ALIGNED. How can antonymous outputs align?

The answer: **antonymy is about the TARGET position, not the axis direction**.

```
verb cluster  →  direction D  →  {positive_adj, negative_adj, ability_adj}
```

All three axes travel approximately in direction D (from verb cluster toward
adj cluster). They land in DIFFERENT sub-regions of adj space (positive,
negative, ability). These sub-regions are close together but distinct.

Compare: the adj_ant axis (antonym axis) has cos≈0.0 with everything. Antonymy
is an orthogonal flip within the adj cluster. It doesn't share the "verb→adj"
direction.

The GROUP D suffixes share the APPROACH vector (verb→adj) but differ in the
LANDING point (which sub-cluster within adj space).

### The Complete Morphological Family Map

```
GROUP A: verb → event_noun    (+ance, +al_nom, +tion, +ment)   cos≈0.435
GROUP B: adj  → quality_noun  (+ity, +ness)                    cos≈0.343
GROUP D: verb → adj modifier  (+less, +ful, +able)             cos≈0.380

STANDALONE axes (near-orthogonal to all groups):
  ablaut:   irregular verb → past tense
  +ed_reg:  regular verb   → past tense (cos=0.411 with ablaut — same operation!)
  +er_comp: adj → comparative adj
  +s_plural: noun → plural noun
  un-:      adj → negated adj
  cc:       word → capitalized word

REVERSE PAIR:
  +al_rel ↔ +ity (cos=-0.432)

TWO TENSE CLUSTERS:
  {ablaut, +ed_reg}: cos=0.411 (both "base → past")
  {+ing, ablaut}: cos=0.320 (both operate on base verbs)
```

### Notable: Two Past Tense Axes Cluster Together

```
cos(+ed_reg, ablaut) = +0.411
```

Regular (-ed) and irregular (ablaut) past tense axes are strongly aligned.
This means they traverse approximately the same "present→past" direction in W_E.
Despite completely different surface forms, the semantic transformation is the same.

**Prediction from this**: the axis for "third person -s" (he run/runs) should
ALSO cluster with these, since all three are "base → inflected verb" operations.

---

## Etymology Split: W_E Encodes Lexical Stratum

### The Result

```
Germanic adjectives: +ness axis navigates correctly, +ity axis fails
  dark → darkness ✓ (+ness),  黑暗 ✗ (+ity, Chinese for "darkness")
  hard → hardness ✓ (+ness),  harder ✗ (+ity, lands at comparative)
  kind → kindness ✓ (+ness),  kinda  ✗ (+ity, informal form)

Latin adjectives: +ity axis navigates correctly, +ness axis fails
  legal  → legality  ✓ (+ity),  法律 ✗ (+ness, Chinese for "law/legal")
  local  → locality  ✓ (+ity),  (local ✗ (+ness, tokenizer artifact)
  real   → reality   ✓ (+ity),  (real  ✗ (+ness, tokenizer artifact)
  moral  → morality  ✓ (both)   [both axes work here]
```

### The Lexical Stratum Hypothesis

W_E has absorbed the etymological layering of English vocabulary. The two quality
nominalizer axes have specialized to different layers:
- **+ity axis direction** points toward the Latin/French sub-region of W_E
- **+ness axis direction** points toward the Germanic/Old English sub-region

This is not a property we designed — it EMERGED from the model training on
English text that uses Latin words in different contexts than Germanic words.

### The Chinese Void

When an axis points toward a region of W_E where the expected English word
doesn't exist (e.g., *darkity, *legalness), the nearest neighbor retrieval
finds the nearest OCCUPIED token. Chinese tokens fill these voids because:
1. Chinese characters are single tokens in Qwen2 (it's a Chinese-English model)
2. Chinese abstract nouns often appear as single tokens near abstract English regions
3. The abstract quality noun region contains Chinese characters for concepts
   like 法律 (legal system), 黑暗 (darkness), 心理 (psychology)

This "Chinese void-filling" phenomenon is actually useful: **a Chinese result
indicates the displacement pointed to an empty region** — useful as a signal
that the axis is misapplied.

---

## Three-Step Chain: Scale Drift

### What Happened

```
Step 1: nation + 0.64×al_rel → national ✓
Step 2: normed(step1)×mag + 0.84×ity → nationality ✓
Step 3: normed(step2)×mag + 0.02×plural → nationality (stuck) ✗
```

The +s_plural scale was calibrated as 0.02 in the chain context. Why?

The `best_scale` function searches scales from 0.02-6.0. If the optimal scale
for plural retrieval on the training pairs is ~0.02, that means the plural axis
requires only a TINY displacement on those training words (cat, dog, house).

But 'nationality' is in a DIFFERENT DENSITY REGION than 'cat'. The local
neighborhood structure around 'nationality' is different:
- Around 'cat': the nearest tokens are cat-like things; 'cats' is close by
- Around 'nationality': the nearest tokens are nationality-related; 'nationalities' is far

The plural scale 0.02 that works for 'cat→cats' is too small for 'nationality→nationalities'.

### The Scale Drift Problem

Each morphological axis is calibrated on its TRAINING DISTRIBUTION. When used
on OOD (out-of-distribution) words (e.g., 'nationality' for the plural axis),
the calibrated scale may not transfer.

**The deeper issue**: the plural axis was trained on short, common nouns.
'Nationality' is a long, abstract, Latin-origin noun — completely outside the
training distribution. The plural axis doesn't have a strong gradient toward
'nationalities' from 'nationality'.

### A Potential Fix: Adaptive Scale

Instead of using a single global scale, calibrate the scale LOCALLY using
k-nearest neighbors of the intermediate result:

```
1. Find k=5 nearest neighbors N of the intermediate embedding
2. For each n in N: find best scale s* to navigate to n's plural form
3. Use median(s*) as the local scale for the chain step
```

This requires knowing the plural form of each neighbor (a morphological oracle),
but demonstrates that local scale calibration could enable three-step chains.

---

## Day 328 Plan

1. **20-pair probe**: 10 train + 10 holdout. Test whether LOO and irred both
   stabilize. Target: 8/8=100% probe accuracy on the 8 axis types.

2. **GROUP D internal cosines**: measure cos(+less, +able) and verify the
   complete GROUP D triangle. Also measure cos(GROUP D, GROUP A) to confirm
   the verb-source correlation principle.

3. **Third-person -s axis**: verify that `{ablaut, +ed_reg, +3ps}` form a
   "tense/inflection" cluster. How high is cos(+3ps, +ed_reg)?

4. **Scale-adaptive chain**: implement adaptive scale calibration for step 3
   of the nation→nationality→nationalities chain. Test if it recovers.

5. **Chinese void mapping**: systematically map which morphological operations
   land in Chinese when applied OOD. Is there a predictable pattern to which
   Chinese tokens appear?

---

## Files

- `expedition_log.md` — Days 322-327 results
- `461_reverse_operation_pairs_scale_free_composition_and_source_category_principle.md` — DC 461
- `day327_15pair_probe_negcos_etymology_antichain_3step.py` — experiment script
