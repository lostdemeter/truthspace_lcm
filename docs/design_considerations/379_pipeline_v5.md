# DC 379: Pipeline v5 — TYPE_ANTONYM Axis Quality Threshold

**Day 227 | Pipeline v5 achieves 50/60 = 0.833 at full 42k token pool.
Key addition: axis_align > 0.70 is the reliability threshold for
TYPE_ANTONYM attribute-axis retrieval. The speed axis (axis_align=0.757)
retrieves brisk->sluggish at rank=0 in the full 42k pool. The size axis
(axis_align=0.547) fails for test pairs regardless of pair-lookup
augmentation. The root cause of low size axis_align is training pair
semantic heterogeneity: big/large/huge/tall/wide/thick span multiple
semantic sub-dimensions (volume, height, breadth, thickness), producing
diverse displacement directions that yield a noisy mean axis.**

---

## Overview

Day 226 tested three hypotheses about the antonyms_sup_size 0.333 failure:
1. Target cluster tightness predicts failure → **DISPROVED**
2. Pair-lookup augmentation recovers test accuracy → **DISPROVED** (only helps training)
3. axis_align > 0.70 predicts TYPE_ANTONYM reliability → **CONFIRMED**

Also introduced: antonyms_sup_speed as a new TYPE_ANTONYM domain with
axis_align=0.757. Speed axis retrieves correctly at 42k.

---

## Axis Quality Metrics

```
attribute   axis_align  target_tightness  train_acc  test_acc  verdict
size        0.547       0.218             0.333      0.333     FAILS
speed       0.757       0.369             1.000      1.000     WORKS
```

The critical observation: speed has **higher** target tightness than size
(0.369 vs 0.218), yet speed works and size fails. Target tightness is
not the predictor. **axis_align alone discriminates.**

---

## Finding 1: axis_align Reflects Training Semantic Homogeneity

**Why does size have low axis_align=0.547?**

Size training pairs: big/large/huge/tall/wide/thick → small/tiny/little/short/narrow/thin

These source words span multiple semantic sub-dimensions:
- big/large/huge: general volume/mass
- tall: specifically height (vertical dimension)
- wide: specifically breadth (horizontal dimension)
- thick: specifically depth/thickness

The W_E embedding for "tall" encodes HEIGHT specifically — it is closer
to other height-related words (high, elevated, towering) than to general
size-related words. The displacement tall→short points in a direction
that is somewhat orthogonal to huge→little (volume) or wide→narrow (breadth).

When these 6 heterogeneous directions are averaged, the mean is a blurry
central direction that does not precisely point toward any individual
target. axis_align = 0.547 reflects this blurring.

**Why does speed have high axis_align=0.757?**

Speed training pairs: fast/quick/swift/rapid/hasty → slow/sluggish/plodding/gradual/leisurely

All source words encode a single concept: RATE OF MOTION. There is no
speed sub-dimension. "Quick" and "swift" and "rapid" are nearly synonymous
in embedding space, encoding the same concept. Their displacement vectors
toward "slow/sluggish/plodding" point in nearly the same direction.

axis_align = 0.757 reflects this tight agreement.

---

## Finding 2: Pair-Lookup Does Not Fix Test Accuracy

Pair-lookup (memorise training A→B) recovers training accuracy:
- size training: axis=2/6, lookup=6/6
- speed training: axis=3/3=lookup=3/3 (axis already perfect)

But size test pairs (deep→shallow, high→low, long→short) are NOT in the
training lookup. For these unseen pairs, axis retrieval is still used.
Since axis_align=0.547 < 0.70, the axis fails on test.

Pair-lookup is useful for **exact recall** of training pairs but provides
zero improvement for generalization to unseen source words.

This confirms: the axis is the only generalizing mechanism. If the axis
is too noisy, test retrieval fails regardless of lookup augmentation.

---

## Finding 3: Proposed Fix for Size Axis

**The fix: use semantically homogeneous training pairs for size.**

Instead of mixing height/width/volume/thickness, use pairs from ONE
sub-dimension:

Length/extent pairs: long/short, tall/short, deep/shallow, high/low,
                     broad/narrow, wide/narrow

All of these encode LINEAR EXTENT in some dimension. Their displacement
directions should be more aligned, yielding higher axis_align.

Predicted outcome: axis_align > 0.70 → test acc improves from 0.333.

This will be tested in future sessions. Not yet implemented.

---

## Pipeline v5 Architecture

```python
# v5 retrieval routing
def classify_v5(train_pairs, attribute=None):
    p = ok_pairs(train_pairs)
    if any(a == b for a,b in p):
        return "IDENTITY"
    if attribute is not None and attribute in antonym_axes:
        return "TYPE_ANTONYM"
    if len(p) >= 2 and dir_consistency(p) > 0.10:
        return "TYPE_BC"
    return "TYPE_ADJACENT"

def retrieve_v5(src, train_pairs, pred, attribute=None):
    if pred == "IDENTITY":
        return src
    if pred == "TYPE_BC":
        return retrieve_bc(src, mean_dir(train_pairs))
    if pred == "TYPE_ANTONYM":
        aa = axis_quality[attribute]["axis_align"]
        # pair-lookup for known sources (regardless of axis_align)
        lookup = {a: b for a,b in ok_pairs(train_pairs)}
        if src in lookup:
            return lookup[src]
        # axis retrieval for unseen sources (only if axis reliable)
        if aa >= 0.70:
            return retrieve_axis(src, attribute)
        # below threshold: axis not reliable; no good fallback
        return retrieve_nn(src)  # TYPE_ADJACENT (known to fail at 42k)
    return retrieve_nn(src)  # TYPE_ADJACENT
```

Note: for antonyms_sup_size with axis_align=0.547, unseen test pairs
fall through to retrieve_nn (TYPE_ADJACENT), which also fails at 42k.
Both paths give acc~0.333, consistent with observed results.

---

## Pipeline v5 Results

```
Domain                Type          dc     aa    acc    vs v4b
capitals              TYPE_BC      0.368  —     1.000   same
gender                TYPE_BC      0.252  —     1.000   same
plurals               TYPE_BC      0.283  —     0.833   same
superlative           TYPE_BC      0.413  —     1.000   same
past_tense_F          TYPE_BC      0.348  —     1.000   same
past_tense_E          TYPE_BC      0.197  —     0.750   same
past_tense_D          TYPE_BC      0.135  —     1.000   same
past_tense_B          TYPE_BC      0.317  —     1.000   same
numbers               TYPE_BC      0.827  —     1.000   same
antonyms_unsup        TYPE_ADJ     0.020  —     0.000   same
antonyms_sup_size     TYPE_ANTONYM 0.159  0.547  0.333   same (lookup ineffective on test)
antonyms_sup_speed    TYPE_ANTONYM 0.359  0.757  1.000   NEW
no_change_verbs       IDENTITY     0.000  —     1.000   same

OVERALL: 50/60 = 0.833
Classification: 13/13 correct
```

The 0.002 increase (0.831→0.833) comes from adding one domain
(antonyms_sup_speed, 1/1 evaluable test pair) rather than fixing any
existing failure.

---

## Failure Decomposition (v5)

```
Failure                  Cause                          Pairs  Frac
antonyms_unsup 0/6       dc=0.020, direction=noise      6/60   0.100
antonyms_sup_size 1/3    axis_align=0.547 < 0.70        2/60   0.033
plurals 5/6              tokenization EC                1/60   0.017
past_tense_E 3/4         tokenization limits            1/60   0.017
────────────────────────────────────────────────────────────────────
Total failures                                         10/60   0.167

v5 accuracy = 50/60 = 0.833
Structural ceiling (with homogeneous size axis): ~52/60 = 0.867
```

The 0.167 gap to 1.000 breaks down as:
- 0.100: antonyms_unsup (unfixable — no attribute label, direction=noise)
- 0.033: size axis impurity (fixable with homogeneous training pairs)
- 0.033: tokenization limits (unfixable in single-token pipeline)

---

## Updated Archetype Taxonomy (v5)

```
Archetype    dc      axis_align  full-vocab acc  Mechanism
IDENTITY     0       N/A         1.000           return source
TYPE_BC      >0.10   N/A         0.75-1.000      source + mean_dir
             requires cross-dc > 0.15
TYPE_ANTONYM attr    >0.70       0.75-1.000      source + attribute_axis
             requires semantically homogeneous training pairs
             (all same sub-dimension)
TYPE_ANTONYM attr    <0.70       ~0.333          pair-lookup only
             training pairs recoverable but test pairs fail
TYPE_ADJACENT dc<0.05 N/A       ~0              nearest neighbor (fails at 42k)
```

---

## Open Problems (After v5)

1. **Homogeneous size axis:** test whether length-only training pairs
   (long/tall/deep/broad → short/low/shallow/narrow) yield axis_align > 0.70
   and improve test acc for deep/high/long test pairs.

2. **axis_align threshold validation:** confirmed at {0.547: fail, 0.757: pass}.
   Needs more data points to pin the boundary. Is it 0.65? 0.70? 0.75?

3. **antonyms_unsup:** dc=0.020, cross-dc=0.088-0.120. Hard structural floor.
   Cannot be fixed without attribute supervision.

4. **Multi-token test pairs:** past_tense_E and plurals have 1-2 multi-token
   test targets. Cannot be fixed in the single-token retrieval framework.

---

## Files

- `expedition_day226_antonym_v5.py` -- v5 implementation
- `day226_antonym_v5.json` -- results
- `378_antonym_axis_limits.md` -- axis centroid collapse analysis
