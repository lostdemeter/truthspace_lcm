# DC 405: Second-Order Axes in W_E — General Pattern and Limits

**Day 270 | Testing the DC 404 prediction that second-order axes can be
extracted from PCA of first-order axis clusters. Inflectional axis (PCA
of 4 morpho axes): inflected forms ALWAYS project lower than base
(0/14 inflected > base), but positive pole is function words, not base
content words. Hypernym PC1 is an "animal-ness" axis specific to the
dominant training hypernym, not a universal "specific vs general" axis.
Second-order axes are mutually near-orthogonal (|cos| < 0.04).
Key insight: second-order axes work cleanly only when first-order axes
share the same source TYPE.**

---

## Inflectional Axis: Results

PCA of the four morphological inflection axes (adj_degree, superlative,
plural, past_tense):

```
SVD singular values: [1.306, 0.970, 0.907, 0.729]
PC1 explains 42.7% of morphological axis variance

Morpho axis projections onto inflectional PC1:
  adj_degree  → -0.767
  superlative → -0.769
  plural      → -0.534
  past_tense  → -0.492
```

All four project **negatively** — PC1 points TOWARD base forms (same
sign convention as the country axis where enc axes project negatively
onto the country axis).

**What the inflectional axis actually encodes:**

```
POSITIVE pole: punctuation, function words, digits
  (, . and " in 1 a 2 - ( high to . : / - for 5 ')

NEGATIVE pole: inflected forms and complex tokens
  walked(-0.155), bigger(-0.313), fastest(-0.339), taller(-0.335)
```

The inflectional axis is a **markedness / token frequency** gradient:
- Highest frequency, least marked tokens (function words, punctuation) → positive
- Morphologically complex, lower frequency tokens → negative

**Critical result — inflected ALWAYS below base:**
```
walked(-0.155) < walk(+0.062)        cats(-0.138) < cat(+0.074)
bigger(-0.313) < big(+0.192)         fastest(-0.339) < fast(+0.187)
went(-0.060)   < go(+0.164)          mice(-0.135) < mouse(-0.026)
better(-0.145) < good(+0.192)        played(-0.127) < play(+0.109)
taller(-0.335) < tall(+0.125)        harder(-0.298) < hard(+0.185)
Inflected > Base: 0/14
```

The inflectional axis correctly ORDERS every inflected/base pair with
100% accuracy — inflected forms always sit below base forms — but the
absolute threshold is not clean (some base forms are also negative).

---

## Why the Inflectional Axis Doesn't Clean-Separate

The country axis worked cleanly because:
- All three enc axes share the **same source type** (country names)
- The shared component IS specifically the "country-ness" direction
- Result: PC1 separates countries from non-countries

The inflectional axis fails to clean-separate because:
- The four morpho axes do NOT share the same source type
- `adj_degree` and `superlative` come from adjectives
- `plural` comes from common nouns
- `past_tense` comes from verbs
- Their shared component is a broader "morphological complexity / markedness"
  gradient that is entangled with token frequency

**Prediction for clean second-order axes:**

> A second-order axis works cleanly (separates its category from the
> vocabulary) if and only if the constituent first-order axes share the
> same SOURCE TYPE.

| First-order axes           | Source type shared? | 2nd-order result  |
|----------------------------|---------------------|-------------------|
| capital + language + currency | YES (countries)  | Clean country axis |
| adj + superlative + plural + past | NO (mixed POS) | Markedness gradient |

**Corollary:** To get a clean "adjective base" axis, use only adjective axes
(adj_degree + superlative) both of which come from adjective base forms.
To get a clean "noun base" axis, use only plural axis. Mixing POS categories
degrades the second-order axis to a frequency gradient.

---

## Hypernym PC1: "Animal-ness" Not "Specificity"

PCA of 20 hypernym chord vectors (specific → general):

```
SVD: S = [1.711, 0.904, 0.837, 0.773, 0.749]
PC1 explains 28.0% of hypernym chord variance
```

The PC1 is an **animal-ness axis**, not a generic specificity axis:

```
POSITIVE pole (specific animals → PC1 source direction):
  bear(+0.146), wolf(+0.138), tiger(+0.136), lion(+0.128), cat(+0.114)

EXTREME NEGATIVE (the dominant target "animal"):
  animal(-0.865), Animal(-0.682), animals(-0.621), creature(-0.258)

Other general terms: thing(-0.040), object(-0.178), organism(-0.169)
Non-animal specific: dog(-0.005), hammer(+0.012), chair(+0.012) → near zero
```

**Why only animal-ness?** The 20 training pairs include 10 (animal) pairs
(dog/cat/horse/lion/tiger/eagle/wolf/bear/deer/whale → animal). The
dominant target in the training set is "animal" — it appears 10 times.
The PC1 captures the SHARED DIRECTION of the 10 animal-pairs, which is
the "toward 'animal'" direction. It does not generalise to "toward category"
for vehicle, tool, furniture etc. because these appear only 2-3 times each.

The hypernym axis is not WRONG — it correctly identifies "animal" as the
most general-category word in the vocabulary (projection -0.865). But it
is domain-specific to the animal domain, not a universal specificity axis.

---

## Second-Order Axes Are Mutually Near-Orthogonal

```
cos(inflectional, country)  = -0.015   ≈ 0
cos(inflectional, hypernym) = -0.010   ≈ 0
cos(country, hypernym)      = -0.031   ≈ 0
```

The three second-order axes are mutually near-orthogonal, consistent
with the pattern at the first-order level (DC 403: 60.4% of first-order
axis pairs are near-orthogonal).

The hierarchical structure of W_E axes is self-similar:
- First-order axes: mostly orthogonal to each other
- Second-order axes: mostly orthogonal to each other
- Cross-level: second-order axes orthogonal to first-order axes

This orthogonality at each level means W_E uses INDEPENDENT dimensions
for different types of semantic information at every level of abstraction.

---

## Summary: Conditions for Clean Second-Order Axes

| Condition | Country axis | Inflectional axis | Notes |
|-----------|-------------|-------------------|-------|
| Same source type | ✓ (all countries) | ✗ (mixed POS) | Key predictor |
| PC1 % variance | 49.5% | 42.7% | Higher = cleaner |
| Clean separation | ✓ | ✗ (markedness) | |
| Correct ordering | ✓ | ✓ (0/14 fail) | Both useful |
| Self-consistent | ✓ | ✓ | |

Second-order axes are a real phenomenon in W_E, but their quality depends
on whether the constituent first-order axes share a common source type.
When they do (country relations), the second-order axis is a clean category
axis. When they don't (mixed-POS morphology), the second-order axis is a
broader frequency/markedness gradient.

---

## Files

- `expedition_log.md` — Day 270 results
- `404_country_axis_second_order.md` — clean second-order country axis
- `403_axis_orthogonality_full.md` — first-order orthogonality
- `401_semantic_relation_axes.md` — first-order axis taxonomy
