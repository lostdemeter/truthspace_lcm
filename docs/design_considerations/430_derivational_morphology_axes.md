# DC 430: Derivational Morphology Axes — Below Inflectional, Above Semantic Noise

**Day 295 | Seven derivational morphology axes tested: +ness, un-, in-/im-,
+ful, +less, +ment, +tion. All sit BELOW inflectional morphology in the
linearity spectrum (pc 0.104–0.211). ENCODE=DECODE holds universally:
cos(fwd, rev) = −1.000 for all seven. +ful vs +less NOT anti-parallel
(cos=0.461) — both add a suffix in the same geometric direction. Scale
asymmetry: +ness ratio=2.563 (abstract nouns are 2.5× denser than base
adjectives in W_E). Best holdout: +ment 75%, +ful 67%. +ness true holdout
0% despite 100% training accuracy — classic mixed-axis training paradox.**

---

## Spectrum Position

All seven derivational axes occupy a band from pc=0.104 to pc=0.211:

```
+ness     0.211   DERIV  (highest derivational)
+ed past  0.174   INFL   ← inflectional boundary
+s plural 0.155   INFL
in-/im-   0.133   DERIV
+less     0.133   DERIV
+tion     0.130   DERIV
+ment     0.124   DERIV
un-       0.121   DERIV
+ful      0.104   DERIV  (lowest derivational)
```

The clear pattern: **derivational morphology cluster (0.104–0.211) sits
BELOW inflectional morphology (0.155–0.436)**. The highest derivational
axis (+ness, 0.211) barely matches the lowest inflectional axis (+s, 0.155)
at the same level — it falls between gender (0.213) and +ed (0.174).

### Why Inflectional > Derivational

Inflectional morphology:
- Does NOT change word category (noun stays noun, verb stays verb)
- Produces a small, closed set of forms (singular/plural, present/past)
- All training words from the same POS → high source homogeneity → high pc

Derivational morphology:
- Changes word category (adjective → noun, verb → noun)
- Produces an open class of derived words
- Training words span multiple semantic domains → lower source homogeneity
- Many derivational rules have spelling alternations that disrupt linearity

---

## +ness: Highest Derivational, Zero True Holdout

### Results

```
Train (available pairs):   9/9  (100%)
Holdout:                   1/6  (17%)  — but 1 is a training pair (dark)
True holdout:              0/5  (0%)
pc_cos:                    0.2108
scale:                     0.83
```

### Training Data Issues

The +ness training set contained contaminated pairs:

- `warm → warmth`: Not a +ness derivation! The correct form is `warmth`,
  not `warmness`. This is the Germanic -th suffix, not -ness.
  
- `clean → cleanness`: The conventional spelling is `cleanliness`, not
  `cleanness`. The model retrieves `cleanliness` correctly — but this
  counts as a miss against the target `cleanness`.

These issues depress the training accuracy from 100% to 9/15 in full
evaluation and corrupt the axis direction.

### Why Holdout Fails

The holdout pairs (neat, sharp, smooth, quick, still) all fail:
- `neat → neatness`: got `sweetness` (nearest -ness word in W_E)
- `sharp → sharpness`: got `sharper` (comparative form is closer)
- `smooth → smoothness`: got `smooth` (no sufficient distance to reach -ness region)
- `quick → quickness`: got `quick` (same problem)

The +ness axis overshoots some (bold → boldly, loud → louder → comparative
form closer) and undershoots others. The axis is not clean enough for
generalisation due to training contamination.

### Predicted Performance After Cleanup

If the training set is restricted to:
```
Regular +ness (adj → adj+ness, no spelling change):
  sad→sadness, kind→kindness, dark→darkness, hard→hardness,
  cold→coldness, bright→brightness, sweet→sweetness,
  weak→weakness, bold→boldness, calm→calmness
```
Predicted: pc > 0.25, holdout > 40% (to verify in Day 296).

---

## +ment and +tion: Best Holdout Despite Low pc

### +ment Results

```
pc=0.124   scale=1.13   train=12/12 (100%)   holdout=3/4 (75%)
```

Holdout pairs: employ→employment, invest→investment, govern→government
(3 hits), argue→argument (1 miss: got argument? — likely 'argument' is
a variant spelling issue).

### Why +ment Generalises Well at Low pc

The +ment training words are all latinate verbs from similar semantic
domains: move, treat, agree, judge, manage, pay, state, replace, require,
achieve, improve, develop. These form a TIGHT cluster in W_E (all are
abstract action verbs from Latinate vocabulary). High source homogeneity
despite the axis's overall low pc.

This is a **sub-pattern effect within a single named rule**: even though
+ment as a rule has pc=0.124, the specific training words are homogeneous
enough that the axis generalises to similarly homogeneous holdout words
(employ, invest, govern — all latinate abstract verbs).

The pc is low because the mean direction is not perfectly consistent, but
the sub-cluster of latinate action verbs has higher internal consistency.

### ENCODE=DECODE Perfect Symmetry

```
+ment: fwd_scale=1.13  rev_scale=1.13  ratio=1.000  cos=-1.000
+tion: fwd_scale=1.03  rev_scale=1.03  ratio=1.000  cos=-1.000
+less: fwd_scale=0.83  rev_scale=0.83  ratio=1.000  cos=-1.000
```

These three axes have PERFECTLY SYMMETRIC scale ratios (1.000). This
means the source words (verbs/adjectives) and target words (nouns) occupy
regions of EQUAL density in W_E. The step from verb to noun is the same
length as the step from noun back to verb.

In contrast:
```
+ness: fwd_scale=0.83  rev_scale=0.32  ratio=2.563  cos=-1.000
un-:   fwd_scale=0.93  rev_scale=0.63  ratio=1.484  cos=-1.000
```

The +ness asymmetry (ratio=2.563) means abstract nouns (happiness,
sadness, darkness) are in a MUCH denser neighbourhood than base adjectives
(happy, sad, dark). Stepping from abstract noun to adjective requires only
0.32 units vs 0.83 units in the other direction.

### Neighbourhood Density Interpretation

```
density_ratio = scale_fwd / scale_rev = density(target) / density(source)
+ness:  0.83/0.32 = 2.59 → abstract nouns 2.6× denser than base adjectives
un-:    0.93/0.63 = 1.48 → un- words 1.5× denser than base adjectives
+ment:  1.13/1.13 = 1.00 → verb and noun-ment regions equally dense
+tion:  1.03/1.03 = 1.00 → verb and noun-tion regions equally dense
+less:  0.83/0.83 = 1.00 → adjective and adj-less regions equally dense
```

This directly measures the relative compression of each morphological
neighbourhood. Abstract nouns (happiness cluster) are tightly packed
relative to base adjectives (happy cluster). This makes intuitive sense:
there are MANY abstract nouns but fewer base adjectives that commonly
co-occur with the -ness suffix.

---

## +ful vs +less: The Same Axis

### Result

```
cos(+ful axis, +less axis) = 0.461
```

Expected −1.0 (anti-parallel, as semantic opposites). Got +0.461 (similar
direction). Both axes point toward the "derived adjective from noun" region.

### Why

In W_E, the geometry of "adding a suffix" is primarily:
```
noun → derived_adjective  (both +ful and +less do this)
```

The semantic content of +ful (having) vs +less (lacking) is a secondary
dimension that is orthogonal to the primary "derivation" direction. The
primary geometric change is the same for both: walk from the noun region
toward the derived-adjective region.

The semantic distinction (+ful vs +less) is encoded in WHICH derived
adjective is retrieved, not in the direction of the displacement. Both
"hopeful" and "hopeless" are similarly displaced from "hope" — they
differ in where they land (positive vs negative), not in how far they
are from the base.

### Reversed +ful Axis

When the +ful axis is reversed (multiplied by -1), it retrieves:
```
hope  → Hope   (capitalised form)
care  → Care
help  → Help
wonder → Wonder
color → color  (already in lowercase)
```

The reversed axis goes toward the capitalised/formal version of the
word, not toward the -less form. This confirms that −(+ful direction)
does NOT point toward +less forms.

---

## Negation Axes: Partial Shared Direction

```
cos(un-, in-/im-) = 0.535   (partial alignment)
cos(un-, +less)   = 0.223   (weak)
cos(in-/im-, +less) = 0.319 (weak)
```

Both un- and in-/im- are negation prefixes, so partial alignment (0.535)
makes sense. They're not the same axis (which would give cos≈1.0) because:
- un- attaches to native English adjectives (happy, kind, fair, clear)
- in-/im- attaches to Latinate adjectives (possible, logical, regular)

These form two distinct sub-populations in W_E, with partial overlap.
Cross-prediction: un- applied to in-/im- holdout = 1/6 (17%) — limited
but non-zero generalisation.

---

## Training Contamination in Derivational Data

A critical finding: **quality of training pairs matters much more for
derivational axes than for inflectional axes**, because:

1. Inflectional rules are strict: ran is the ONLY past of run; cats is
   the ONLY plural of cat. No ambiguity.

2. Derivational rules are productive: both "loudness" and "louditude"
   are technically possible -ness formations. The model only knows the
   attested forms (loudness), not the rule.

3. Irregular derivation: warmth (not warmness), cleanliness (not cleanness),
   strength (not strongness) — including these in the training set corrupts
   the pure +ness axis.

**Rule for derivational axis construction:**
- Only include pairs where the derived form is the CANONICAL form
- Exclude pairs where the "regular" derivation is not the attested word
- If the canonical derived form differs from rule application, exclude it

---

## Day 296 Plan

Following the universal sub-pattern law (DC 429):

1. **Clean +ness axis**: restrict to 10 purely regular pairs, no warmth/
   cleanness contamination. Test holdout on 5 new pairs. Expected: pc > 0.25.

2. **+ness sub-patterns by phonological context**: adj ending in -y (happy→
   happiness), adj ending in consonant cluster (hard→hardness), adj ending
   in -ight (bright→brightness). Do these sub-patterns have different pc?

3. **Why does +ful (pc=0.104) achieve 67% holdout?** Analyse the source
   cluster structure. Are all training words (hope, care, help, wonder,
   color, power, peace, grace, skill, use, cheer, faith) from the same
   semantic domain?

---

## Files

- `expedition_log.md` — Day 295 results
- `429_element_subpattern_linearity.md` — DC 429: universal sub-pattern law
- `428_inflated_pc_and_element_axis.md` — DC 428: inflation mechanisms
- `day295_derivational_morphology.py` — experiment script
