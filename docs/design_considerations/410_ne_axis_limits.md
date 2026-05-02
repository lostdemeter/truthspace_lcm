# DC 410: NE Axis Limits — Relative Ranker, Not Universal Filter

**Day 275 | Full-vocabulary precision/recall test of the NE axis reveals
it cannot serve as a universal named-entity classifier. The vocabulary
mean is −0.085 (negative-shifted), making absolute thresholds useless.
At every tested threshold, the false-positive rate for lowercase common
words equals or exceeds the recall for capitalized NE-like tokens. At
threshold < −0.15, 73% of retrieved tokens are Unicode/BPE fragments —
not named entities. The NE axis correctly ranks known NE categories
(countries, persons, companies, elements) relative to common words, but
only within the pre-filtered alphabetic word-token domain.**

---

## The Core Problem: Negative-Shifted Distribution

```
Full vocabulary NE axis scores (n=151,936):
  mean  = -0.085   std = 0.047
  p25   = -0.110   p50 = -0.084   p75 = -0.058
```

The ENTIRE vocabulary projects negatively on average. There is no
natural threshold that separates named entities (expected: more negative)
from common words (expected: less negative) because the baseline is
already at −0.085 — well into the "negative" region.

**Why is the mean negative?** The NE axis is extracted from second-order
axes that all point FROM category instances TOWARD attributes. All four
constituent second-order axes project negatively onto it (country=−0.62,
person=−0.74, element=−0.44, company=−0.30). The axis polarity is
defined such that named entities project most negatively — but the
"most negative" end of the vocabulary is dominated by BPE fragments,
not named entities at scale.

---

## Threshold Analysis

```
Threshold   Retrieved    Cap+alpha    Lower-alpha   BPE/Other
 < -0.10    51,219 (34%)   7,134 (14%)  26,593 (52%)  24,343 (47%)
 < -0.15    11,312 ( 7%)     779 ( 7%)   3,073 (27%)   8,221 (73%)
 < -0.20     4,701 ( 3%)     104 ( 2%)     327 ( 7%)   4,372 (93%)
```

At strict thresholds, the retrieved set is overwhelmingly BPE fragments
(CJK characters, Unicode pieces, subword tokens). The NE axis has learned
to score some class of non-alphabetic tokens very negatively — likely
high-frequency subword pieces from Chinese/Japanese/Korean text that
co-occur with named entities in Qwen's training data.

---

## Recall vs False-Positive Rate

```
Threshold   cap_recall   lower_FP    Lift (recall/FP)
 < -0.05      77.4%        79.4%       0.97×  (FP > recall)
 < -0.10      26.9%        30.7%       0.88×  (FP > recall)
 < -0.15       2.9%         3.6%       0.81×  (FP > recall)
 < -0.20       0.4%         0.4%       1.00×  (equal)
```

**Lift is below 1.0 at all thresholds.** The NE axis provides ZERO
discriminative power over capitalized vs lowercase tokens in the full
vocabulary. A random threshold would perform equally well.

---

## Why This Result Does Not Invalidate DC 409

DC 409 tested the NE axis on hand-selected word-level tokens from four
known NE categories. Those results are still valid:

```
countries:  -0.253   persons:  -0.211
companies:  -0.158   elements: -0.116
adj bases:  +0.018   func:     +0.080
```

The **relative ordering** is correct and consistent. The problem is not
the axis itself, but using it as an absolute threshold-based filter
across a 152k BPE vocabulary that includes non-word tokens.

**The two claims must be distinguished:**

| Claim | Valid? |
|-------|--------|
| "Country names score more negative than function words" | YES |
| "Element names score more negative than adj bases" | YES |
| "The NE axis uniquely identifies NE tokens in the full vocab" | NO |

---

## What the NE Axis Actually Measures

The NE axis measures the **degree to which a token participates in
named-entity attribute relations** in W_E. Words that frequently appear
as sources of nationality, field, type, or product relations (i.e., famous
people, countries, elements, companies) score more negative. Words that
frequently appear as TARGETS of such relations (British, German, metal,
technology) score more positive.

This is a **relational position on the named-entity-attribute axis**, not
a binary "is this a proper noun?" classifier. A binary classifier would
require sharp separation; what we have is a gradient.

The gradient is useful for:
1. **Re-ranking candidates**: Given N candidate tokens for a slot, rank
   by NE score to prefer actual named entities.
2. **Category disambiguation**: Given a known named entity, the category
   with the smallest signed distance from its NE score is its category.
3. **Relation direction inference**: A word that scores very negative is
   likely a source entity; a word that scores very positive is likely an
   attribute.

But the gradient is NOT useful for:
1. **Universal NE detection** from raw token projections
2. **Precision recall classification** across mixed vocabulary

---

## Implications for TruthSpace Architecture

The three-level hierarchy (DC 409) is structurally real:
- Level 1 → Level 2 → Level 3 extraction is valid
- Category-level axes (country, person, element, company) are clean
- The NE axis encodes relative NE-ness correctly for word-tokens

**TruthSpace should NOT use NE axis as a token filter.** Instead:
1. Pre-filter to single-token alphabetic words (eliminates BPE fragments)
2. Within the alphabetic candidate set, use NE score to rank by
   named-entity-ness
3. Use category-specific axes (country, person, element) for classification
4. Use relation axes (capital, language, field) for attribute retrieval

This is a two-stage architecture: **token filtering → geometric scoring**,
rather than a single geometric classifier.

---

## Revised Summary of NE Axis Capabilities

| Task | NE Axis | Category Axes |
|------|---------|---------------|
| Detect NE in raw vocab | NO (BPE noise) | Limited |
| Rank known NE candidates | YES (gradient) | YES |
| Classify NE into categories | NO (single axis) | YES |
| Retrieve NE attributes | Via relation axes | Via relation axes |
| Distinguish NE from attributes | YES (relative) | YES |

The NE axis is a genuine third-order geometric structure, but its practical
utility is as a discriminator among pre-filtered candidates, not as a
universal named-entity detector.

---

## Files

- `expedition_log.md` — Day 275 results
- `409_third_order_ne_axis.md` — third-order NE axis (hierarchy confirmed)
- `407_person_axis_named_entity.md` — second-order person axis
- `404_country_axis_second_order.md` — second-order country axis
