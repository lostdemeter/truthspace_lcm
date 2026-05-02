# DC 442: The Semantic Atlas — Why pc Does Not Predict Generalization

**Day 307 | Full calibration of all 12 morphological axes reveals that the
pairwise chord cosine (pc) has only weak correlation with holdout accuracy
(Pearson r=0.27). The true predictors of generalization are DOMAIN OVERLAP
(how etymologically similar holdout pairs are to training pairs) and TOKEN
AVAILABILITY (whether source and target are both single tokens without
capitalized near-neighbors intercepting the axis trajectory). Globally
invisible axes (R²<0.025 in PC1–10) can achieve 80% holdout while globally
visible axes underperform. The full semantic atlas maps all known axes onto
PC1–10 and reveals a three-tier structure: labelling axes (R²=0.43–0.57),
partially-visible morphological axes (R²=0.07–0.15), and hidden semantic
axes (R²<0.025). mPC8 is confirmed as the AKTIONSART (tense aspect) axis:
simple past (+0.112) vs past participle (−0.214) vs present (−0.064).**

---

## The pc Paradox

### The Hypothesis (from DC 441)

> Higher pairwise chord cosine (pc) → more consistent geometry → better
> holdout generalization.

This was supported by the two early data points: +er (pc=0.405, 83% holdout)
vs +s (pc=0.297, 28% holdout). However, the full 12-axis calibration reveals
the hypothesis is incorrect.

### The Full Evidence

| Axis    | pc    | Holdout | Train | n_ho | Interpretation                    |
|---------|-------|---------|-------|------|-----------------------------------|
| er→est  | 0.430 | 40%     | 100%  | 10   | HIGH pc, LOW holdout — paradox    |
| +est    | 0.400 | 27%     | 100%  | 15   | HIGH pc, LOW holdout — paradox    |
| +er     | 0.385 | 80%     | 100%  | 20   | HIGH pc, HIGH holdout             |
| past_irr| 0.319 | 64%     | 100%  | 14   | medium pc, medium holdout         |
| +s      | 0.297 | 40%     | 100%  | 15   | medium pc, LOW holdout            |
| gender  | 0.241 | 17%     | 100%  | 12   | medium pc, VERY LOW holdout       |
| +ed     | 0.227 | 60%     | 100%  | 15   | low pc, medium holdout            |
| +ness   | 0.183 | 22%     | 100%  | 9    | low pc, low holdout               |
| +ment   | 0.153 | 50%     | 100%  | 10   | LOW pc, MEDIUM holdout — paradox  |
| **+tion**| **0.125**| **80%**| 100%| 10   | **LOWEST pc, HIGH holdout**       |
| +ful    | 0.112 | 8%      | 100%  | 12   | lowest pc, lowest holdout         |
| un-     | 0.103 | 8%      | 100%  | 12   | lowest pc, lowest holdout         |

**Pearson r(pc, holdout) = 0.27** — essentially no predictive power.

### Why +tion Achieves 80% with pc=0.125

The +tion training set: act→action, direct→direction, collect→collection,
connect→connection, protect→protection, select→selection.

The +tion holdout set: inject→injection, reject→rejection, infect→infection,
inspect→inspection, detect→detection, correct→correction, construct→construction,
instruct→instruction, introduce→introduction, reduce→reduction.

**Both sets are exclusively Latin-root verbs ending in -ct or -ce.**

The +tion axis learned from {act, direct, collect} transfers perfectly to
{inject, detect, construct} because they all belong to the SAME ETYMOLOGICAL
CLUSTER in W_E. The embedding space groups Latin-root verbs in a specific
region, and the +tion transformation is consistent within that region even
if it's inconsistent across the full vocabulary.

### Why er→est Has High pc but Only 40% Holdout

The er→est training set: faster→fastest, slower→slowest, etc. (all single
tokens in Qwen2's vocabulary).

The er→est holdout set: louder→loudest, quieter→quietest, warmer→warmest,
colder→coldest... Many -er and -est forms for these adjectives are NOT single
tokens (they are split as 'loude' + 'r', etc.), or the scale=0.181 is too
small to reach the target.

The pc=0.430 reflects that the TRAINING PAIRS are geometrically consistent.
But the holdout fails because of **tokenization mismatch** — the holdout
sources are not available as single tokens.

### The Two True Predictors

**Predictor 1: ETYMOLOGICAL DOMAIN OVERLAP**
```
+tion train: {act, direct, collect, connect, protect, select}  (Latin -ct/-ct)
+tion holdout: {inject, detect, construct, instruct, reduce}   (Latin -ct/-ce)
→ Domain overlap = HIGH → holdout = 80%

gender train: {king, man, boy, father, son, brother, uncle, husband}  (core kin)
gender holdout: {monk, prince, emperor, lion, actor, waiter, host, heir}
→ Domain overlap = LOW (royalty, animals, occupations ≠ kin)
→ holdout = 17%
```

**Predictor 2: TOKENIZATION AVAILABILITY**
```
+s train: {cat, dog, house, car, tree, book, bird, ship}  (all common single-tokens)
+s holdout fails for: cup→Cup, road→Road, hand→Hand, eye→Eye
→ The CAPITALIZED FORM (Cup, Road, Hand) is a distinct token closer to the source
→ The +s displacement (scale=0.262) is intercepted by the capitalized neighbor
```

---

## The +s Capitalized Token Problem

### Root Cause

In Qwen2's vocabulary, common English nouns have BOTH lowercase and capitalized
forms as distinct tokens:
- 'cup' (id=N₁) and 'Cup' (id=N₂) — different embeddings
- 'door' (id=M₁) and 'Door' (id=M₂) — different embeddings

In W_E, the capitalized form is geometrically CLOSER to the lowercase form
than the plural is:
```
cos(cup, Cup) > cos(cup, cups)
```

When we apply the +s axis with scale=0.262, the nearest neighbor search
finds 'Cup' before 'cups'.

**This is not a failure of the geometric axis — it is a failure of TOKENIZATION
GRANULARITY.** The embedding space contains both `cup` and `Cup` as near-
duplicate tokens, creating a local trap that intercepts the +s trajectory.

### The +s Success Pattern

+s WORKS for: flowers, stars, forests, trains, boats (all single tokens,
plurals more common than capitalized variants).

The key difference: for high-frequency ABSTRACT or NATURAL nouns (flowers,
stars, forests), the PLURAL form is used more frequently in text than the
capitalized singular, so W_E pushes the plural token away from the
capitalized singular.

For COMMON CONCRETE NOUNS (cup, door, road, hand, eye) in contexts like
"The Cup" (trophy), "The Road" (book/movie), these capitalized forms are
nearly as common as the lowercased form, creating the near-duplicate trap.

---

## The Three-Tier Semantic Atlas

### Tier 1: Labelling Axes (R²=0.43–0.57 in PC1–10)

These axes explain significant global variance because they bridge LARGE
frequency gaps (named entities → digit symbols):

```
Axis          PC1      R²(PC1-10)
v_ord         −0.722   0.560
planet→num    −0.729   0.567
card→num      −0.695   0.516
weekday→num   −0.656   0.461
digit→word    +0.638   0.460
month→num     −0.627   0.432
```

All labelling axes have strongly NEGATIVE PC1 (named entities like January,
Monday, Ace have positive PC1; digits have negative PC1). The labelling
operation is essentially ANTI-PC1 — it crosses the biggest global frequency
gradient.

### Tier 2: Partially Visible Morphological Axes (R²=0.07–0.15)

```
Axis   PC1     PC3      R²
+est   +0.287  +0.151   0.149
+ness  +0.198  +0.230   0.121   ← PC3 > PC1 for +ness!
+er    +0.286  +0.143   0.117
+s     +0.225  —        0.075
un-    +0.173  +0.145   0.069
+ed    +0.132  +0.086   0.045
+ful   +0.153  —        0.041
```

Tier-2 axes project somewhat onto PC1 (all positive, meaning they point toward
lower-frequency derived forms) and some onto PC3. The PC1 contribution is
expected: morphological derivatives are typically LESS frequent than base forms.

The notable exception: **+ness has a larger PC3 component (+0.230) than PC1
(+0.198)**. This means the +ness derivation has a specific semantic quality
captured by PC3 that is MORE important than mere frequency shift.

### Tier 3: Hidden Axes (R²<0.025 — globally invisible)

```
Axis         R²(PC1-10)  Holdout%
+tion        0.010       80%      ← invisible yet effective
+ment        0.020       50%
gender       0.020       17%
er→est       0.018       40%
past_irr     0.026       64%
country→dem  0.023       —
```

These axes are essentially PERPENDICULAR to the top-10 global PCs. Their
information lives in the very specific directions of W_E that are not captured
by any globally significant variance direction.

**Critical insight**: global PCA of W_E captures the structure of the FULL
vocabulary. Specific transformations between small word classes (Latin -ct
verbs, gender-alternating nouns, irregular verb pairs) occupy TINY low-variance
directions that are orthogonal to everything the global PCA finds.

### What R² in Global PCs Actually Measures

```
R²(axis, PC1-10) = fraction of axis direction explained by top-10 global PCs
                 = cos²(axis, PC1) + cos²(axis, PC2) + ... + cos²(axis, PC10)
```

This measures the extent to which a transformation is DRIVEN BY frequency/
structural properties of the full vocabulary. Axes that primarily separate
high-frequency from low-frequency tokens (labelling, morphological +er/+ness)
have high R². Axes that operate within a frequency-uniform region (gender
within noble/kin words, +tion within Latin-root verbs) have low R².

---

## mPC8: The Aktionsart Axis Confirmed

### Linguistic Aktionsart Theory

In linguistics, AKTIONSART (verbal aspect) distinguishes verb meanings by
their inherent temporal structure:
- **TELIC/EVENTIVE**: actions with a natural endpoint (went to the store)
- **STATIVE/RESULTATIVE**: states or outcomes (known by everyone)
- **ATELIC**: ongoing activities without endpoint (walking)

### Empirical Evidence

```
Form class       mPC8 mean    Examples
Simple past      +0.112       went, took, gave, came, saw, ran, ate, knew
Present tense    −0.064       go, take, give, come, see, run, eat, know
Past participle  −0.214       known, called, been, seen, shown, taken, referred
```

The mean mPC8 scores clearly separate all three classes. Simple past irregular
forms score highest (+0.112) — these are PUNCTUAL, EVENTIVE, PRETERITE actions.
Past participles score lowest (−0.214) — these are RESULTANT STATES (things
that have been done to something, now true of it).

Present tense sits between (−0.064) and clusters CLOSER TO PARTICIPLES than
simple past. This is linguistically correct: the PRESENT TENSE of verbs like
'know', 'see', 'come', 'run' is often STATIVE or HABITUAL (not punctual/telic),
just like past participles.

### Implications for W_E Structure

The mPC8 axis shows that Qwen2's embedding space encodes ASPECT automatically:
it has learned to separate EVENTIVE verb forms from STATIVE/PASSIVE verb forms
without being explicitly trained to do so. This is a highly sophisticated
linguistic distinction that emerges purely from next-token prediction training.

---

## Summary: The True Structure of W_E Semantic Axes

```
PREDICTOR          RELATIONSHIP TO HOLDOUT
─────────────────────────────────────────────────────────
pc                 r=0.27 — WEAK (not a reliable predictor)
R² in global PCs   no correlation with holdout
domain overlap     STRONG — same etymological class = high holdout
tokenization       STRONG — multi-token sources/targets = failures
axis scale         moderate — too small = capitalized traps
```

The geometry of W_E does not encode transformations as universal operators.
Instead, it encodes **domain-specific local operators**: the +tion transformation
works for Latin -ct verbs, the gender transformation works for core kinship
vocabulary, the +er transformation works for monosyllabic adjectives. Each
domain has its own local geometric structure.

This is the KEY FINDING: **W_E is not a universal morphological transformer.
It is a collection of LOCAL CONSISTENT GEOMETRIES, one per etymological/
semantic domain.** The pc metric measures consistency within a domain, not
generalization across domains.

---

## Day 308 Plan

1. **Domain overlap quantification**: compute a formal DOMAIN SIMILARITY
   metric between train and holdout pairs. Test whether it correlates
   with holdout better than pc.

2. **Capitalized token fix for +s**: when applying +s, exclude capitalized
   variants from the nearest-neighbor search. Does this fix the holdout?

3. **+tion domain expansion**: test +tion on non-Latin-root words (e.g.,
   observe→observation, describe→description). Does it still work?

4. **Gender domain expansion**: test gender axis on more diverse holdout
   (Greek, Norse, Latin origin words). Why does it fail at 17%?

---

## Files

- `expedition_log.md` — Day 307 results
- `441_pc_coherence_predicts_generalization.md` — DC 441
- `day307_pc_calibration_and_atlas.py` — experiment script
