# DC 378: Antonym Axis Limits — Centroid Collapse

**Day 225 | The antonyms_sup_size acc=0.333 failure is caused by axis
centroid collapse, not vocabulary density. The size axis points to the
centroid of the {tiny, small, little, short, narrow, thin} synonym cluster.
Every size query retrieves "tiny" regardless of the specific intended
target. This is not a density problem: training acc=2/6 even at a 50-word
pool. The speed axis succeeds (acc=1.000 at 42k) because the "slow"
synonym cluster is geometrically more dispersed. TYPE_ANTONYM accuracy is
bounded by target synonym cluster tightness, not vocabulary size.**

---

## Overview

Day 224 investigated why antonyms_sup_size acc=0.333 despite the TYPE_ANTONYM
routing fix in v4b. Three experiments: (1) size axis training self-retrieval,
(2) five attribute axis comparison, (3) test accuracy vs pool size scan.

---

## Experiment Results

### Size Axis Training Self-Retrieval

```
pair          axis_align  500w_rank  42k_rank  top-1
big->small    0.608       1          1         tiny
large->tiny   0.631       1          1         small
huge->little  0.557       2          2         tiny
tall->short   0.438       3          3         tiny
wide->narrow  0.536       0 OK       0 OK      narrow
thick->thin   0.512       0 OK       0 OK      thin

Training acc: 500w=2/6  42k=2/6
Test acc:     500w=1/3  42k=1/3
```

The size axis is **not vocabulary-size sensitive**. Training accuracy is
identical at 50 words and 42,537 words. The failure is baked into the
axis construction itself.

### Five Attribute Axes

```
attribute    axis_align  train_acc  test_acc  notes
size         0.547       0.333      0.333     FAILS (centroid collapse)
temperature  0.596       0.500      0.000     partial
speed        0.757       1.000      1.000     WORKS fully
brightness   0.623       0.667      0.000     partial
age          0.535       0.400      0.000     partial
```

Speed is the only axis that works at both training and test at 42k.
All other axes fail on either training or test pairs.

### Density Scan (size test pairs)

```
pool     acc   deep_rank  high_rank  long_rank
50       0.667     0          1          0
100      0.667     0          1          0
200      0.667     0          1          0
500      0.667     0          1          0
1000     0.667     0          1          0
2000     0.667     0          1          0
5000     0.333     1          2          0
10000    0.333     1          3          0
20000    0.333     1          5          0
42537    0.333     1          6          0
```

- `high->low` fails at ALL pool sizes (rank=1 even at 50 words).
  Root cause: axis centroid collapse dominates.
- `deep->shallow` degrades from pool=5000 onward.
  Secondary cause: density pushes "tiny" above "shallow" at rank.
- `long->short` is stable at acc=1.000 across all pool sizes.
  "short" is more isolated in the size-synonym cluster.

---

## Finding 1: Axis Centroid Collapse

The size axis is constructed as:

```
axis = normed(mean([
    emb(big) - emb(small),
    emb(large) - emb(tiny),
    emb(huge) - emb(little),
    emb(tall) - emb(short),
    emb(wide) - emb(narrow),
    emb(thick) - emb(thin),
]))
```

This axis points in the direction of the **mean** of all "large-word"
positions minus all "small-word" positions. The query is:

```
target = normed(source_emb + target_dir)
```

where `target_dir = +axis` (if source is "big") or `-axis` (if source
is "small").

The problem: `emb(small) + emb(tiny) + emb(little) + emb(short) +
emb(narrow) + emb(thin)` forms a **tight cluster** in embedding space.
The axis terminal lands near the CENTROID of this cluster. The nearest
token to that centroid is "tiny" — the most semantically central
small-word in Qwen2's embedding space.

**The axis correctly identifies the DIRECTION (toward smallness) but
cannot identify the SPECIFIC target word within the small-word cluster.**

---

## Finding 2: Why Speed Works

```
speed axis_align=0.757  train=1.000  test=1.000
```

Speed training: fast/quick/swift -> slow/sluggish/plodding

The key difference: "slow", "sluggish", "plodding" are geometrically
dispersed in Qwen2's embedding space. They have distinct surface forms
and connotations:
- slow: neutral, general-purpose
- sluggish: implies heaviness/laziness
- plodding: implies effortful, mechanical

These words do not cluster as tightly as tiny/small/little/short.
The speed axis can point to specific targets because the targets are
not synonymous enough to form a tight cluster.

Also: `axis_align=0.757` (highest of all axes), meaning the training
displacements are unusually consistent. The slow/sluggish/plodding
cluster lies at the end of a very clean vector from fast/quick/swift.

---

## Finding 3: The Structural Limit of Single-Axis Retrieval

Single attribute-axis retrieval has a fundamental limitation:

```
One axis -> one direction -> one terminal point
Terminal = centroid of antonym cluster
Retrieval = nearest word to terminal
Correct only if: target == most-central member of antonym cluster
```

For SIZE: terminal = "tiny" (most central small-word)
- big->small: FAILS (target=small, retrieves tiny)
- large->tiny: FAILS (target=tiny, retrieves small, rank=1)
- wide->narrow: PASSES (narrow is not in tiny/small/little cluster)
- thick->thin: PASSES (thin is not in typical small-synonym cluster)

The only pairs that succeed are those whose target is geometrically
OUTSIDE the main synonym cluster, so the centroid happens to be
closer to the correct target.

**This is the same failure mode as TYPE_ADJACENT at full vocab:**
a cloud of near-synonyms surrounds the target, and the retrieval
query lands in the center of that cloud, not at the specific target.

---

## Finding 4: Taxonomy Update

```
Archetype      Mechanism               Vocab-robust?  Cluster-robust?
IDENTITY       exact lookup            YES            YES
TYPE_BC        source + mean_dir       YES            YES  (pairs are distinct)
TYPE_ANTONYM   source + attribute_axis YES            NO   (tight clusters fail)
TYPE_ADJACENT  source proximity        NO             NO   (both fail)
```

TYPE_ANTONYM sits between TYPE_BC and TYPE_ADJACENT in reliability:
- Like TYPE_BC: uses a directional vector, vocab-robust in principle
- Like TYPE_ADJACENT: fails when the target is not unique in the
  retrieval space (tight synonym cluster = same as dense vocab)

The critical difference between TYPE_BC and TYPE_ANTONYM success:
- TYPE_BC targets are UNIQUE (Paris is the only capital of France)
- TYPE_ANTONYM targets are NON-UNIQUE (small and tiny are both valid
  size antonyms of big, making the "correct" answer ambiguous)

---

## Finding 5: Axis Quality Metric (axis_align)

```
attribute    axis_align  train_acc
speed        0.757       1.000
brightness   0.623       0.667
temperature  0.596       0.500
size         0.547       0.333
age          0.535       0.400
```

axis_align = mean cosine(each training pair's A-B direction, axis).
Higher axis_align predicts higher training accuracy. Threshold appears
to be around 0.70 for reliable retrieval. Speed exceeds this; others do not.

axis_align is an axis-level analogue of cross-dc for TYPE_BC. It measures
whether the axis is self-consistent enough to distinguish targets.

---

## Implications for Pipeline

The v4b TYPE_ANTONYM route is correctly classified (12/12 classification)
but the retrieval quality is bounded by axis_align and target cluster
geometry. For the current size axis:

```
Expected acc at 42k = f(axis_align, target_cluster_tightness)
size:  axis_align=0.547, tight cluster -> acc=0.333 (structural ceiling)
speed: axis_align=0.757, loose cluster -> acc=1.000
```

**No amount of additional training pairs will fix the size axis ceiling**
for big/small/large/tiny pairs, because the problem is in the target
geometry, not the axis quality.

Fix options:
1. **Pair-lookup fallback:** if source is in training set, return
   stored B directly without axis retrieval. This would give acc=1.000
   on training pairs but requires memorisation (not geometric).
2. **Conditional axis:** build per-source axes (big-axis vs wide-axis).
   Computationally expensive and requires more training data.
3. **Accept the ceiling:** axis retrieval is reliable only for attributes
   with dispersed antonym clusters (speed, strong contrast pairs).

---

## Updated Archetype Taxonomy (v4b Final)

```
IDENTITY     dc=0         return source                    acc=1.000
TYPE_BC      dc>0.10      source + mean_dir               acc=0.75-1.000
             requires: cross-dc>0.15 for matched subclass
TYPE_ANTONYM attr label   source + attribute_axis         acc=0.33-1.000
             requires: axis_align>0.70, loose target cluster
TYPE_ADJACENT dc<0.05     proximity (UNSOLVED at full vocab) acc~0
```

---

## Open Problems

1. **Axis quality threshold:** axis_align=0.70 separates speed (works)
   from others (partial). Needs validation on more attribute axes.

2. **Target cluster tightness measure:** how to quantify "cluster tightness"
   for antonym targets at design time (before retrieval fails)?
   Possible measure: mean pairwise cosine among the target words in each
   training pair. High = tight cluster = will fail.

3. **Antonyms_unsup remains unsolvable:** no attribute label, dc=0.020,
   cross-dc=0.088-0.120. This is a hard floor.

4. **Pair-lookup vs axis for known pairs:** for production use, storing
   the specific A->B mapping as a lookup achieves acc=1.000 trivially.
   But this uses memorisation, not geometry. The geometric question is
   whether the AXIS retrieves the correct answer for UNSEEN pairs.

---

## Files

- `expedition_day224_antonym_axis.py` -- antonym axis investigation
- `day224_antonym_axis.json` -- results
- `377_crossdc_generalisation.md` -- cross-dc analysis
- `376_full_vocab_v4_final.md` -- full-vocab v4 results
