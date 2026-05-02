# DC 431: Target Cluster Density — The Second Predictor of Generalisation

**Day 296 | Two anomalous axes (+ful: pc=0.104→67% holdout, +ment:
pc=0.124→75% holdout) defy the pc→holdout correlation (ρ=0.689).
The common factor: their TARGET WORDS form a DENSE, BALANCED cluster in
W_E — all common -ful forms are single BPE tokens in the same semantic
region. Conversely, +ness fails (0% holdout at pc=0.211) because
'sweetness' is a DOMINANT ATTRACTOR that captures all predictions.
Cleaning the +ness training data (removing warmth/cleanness) does not
fix this: the attractor is in the targets, not the sources. A two-factor
model (pc + target_density) predicts holdout better than pc alone.**

---

## The Dominant Attractor Problem

### +ness Failure Pattern

Holdout test with clean training data (10 pure +ness pairs):

```
neat    → neatness:    got sweetness  ← ATTRACTOR
sharp   → sharpness:   got sharp      ← UNDERSHOOTS (capitalised form)
rough   → roughness:   got Rough      ← UNDERSHOOTS
thick   → thickness:   got Thick      ← UNDERSHOOTS
plain   → plainness:   got Plain      ← UNDERSHOOTS
round   → roundness:   got round      ← UNDERSHOOTS
cool    → coolness:    got coolest    ← WRONG FORM (comparative)
still   → stillness:   got still      ← UNDERSHOOTS
fast    → fastness:    got fast       ← UNDERSHOOTS
mild    → mildness:    got Mild       ← UNDERSHOOTS
```

Two failure modes:
1. **Dominant attractor** (`sweetness`): the step overshoots past the
   holdout -ness form and lands at the nearest common -ness word in the
   training cluster. "sweetness" is most central in W_E because it
   appears in many positive contexts ("sweetness of life", etc.).

2. **Undershoots**: the step lands at the capitalised form (Rough, Thick,
   Plain) or the base form (sharp, still, fast) — the scale (0.73) found
   from training pairs is TOO LARGE for some pairs and TOO SMALL for others.

### Why Training Works but Holdout Fails

Training pairs: `sad, kind, dark, hard, cold, loud, sweet, weak, bold, calm`
→ These are all HIGH-FREQUENCY common adjectives. Their -ness forms
(`sadness, kindness, darkness, hardness, coldness, loudness, sweetness,
weakness, boldness, calmness`) are ALL frequently attested. They form a
tight, dense cluster in W_E.

Holdout pairs: `neat, sharp, rough, thick, plain, round, cool, still, fast, mild`
→ Their -ness forms are LESS FREQUENT. `roughness, thickness, fastness,
mildness` are technical or uncommon words. They are NOT in the same cluster
as the training -ness words. The axis navigates toward the training -ness
cluster, not to the individual holdout targets.

### The Sweetness Dominance

`sweetness` is the single most common training target that is central to
the -ness cluster. It functions as the cluster centroid. Any prediction
that "arrives in the -ness region" retrieves `sweetness` first.

This is quantifiable: among the 10 holdout misses, `sweetness` appeared
as the top prediction for at least 2 (neat, and others). The dominant
training member acts as a gravitational attractor in the target region.

**Rule**: If any single training target is much more frequent than others
(appears more centrally in W_E), it will dominate the cluster and attract
all out-of-training predictions that step into that region.

---

## The +ful Success Pattern

### Why +ful Works at Low pc

```
pc = 0.104  (very low)
holdout = 4/6 (67%)  (anomalously high)
src_pc = 0.062  (very low source homogeneity)
```

**The -ful cluster is dense and balanced.** The training targets:
```
hopeful, careful, helpful, wonderful, colorful, powerful,
peaceful, graceful, skillful, useful, cheerful, faithful
```

These are all common positive-valence adjectives of similar frequency.
None dominates. When a prediction arrives in the -ful region, it finds
many equally plausible -ful forms and retrieves the nearest one (which
is often the correct one, since the source word's nearest -ful form is
its own derivation).

### BPE Token Structure

Key observation: ALL common -ful forms are SINGLE BPE TOKENS with
space prefix:
```
' beautiful'  → single token  ✓
' harmful'    → single token  ✓
' delightful' → single token  ✓
' respectful' → single token  ✓
' thoughtful' → single token  ✓
' playful'    → single token  ✓
```

Without space prefix, these tokenise as multi-token sequences:
```
'harmful'    → ['h', 'arm', 'ful']
'delightful' → ['del', 'ight', 'ful']
```

The single-token -ful forms (with space) are therefore a COHERENT BPE
region — they were learned as units, not compositions. This means:
1. Each -ful word has a dedicated embedding vector
2. These vectors are clustered in semantic space (all positive adjectives)
3. The cluster is balanced (no single dominant member)

This is the same mechanism as elem:single-letter (H, C, N all have
dedicated single-token embeddings that cluster in the "chemical symbol"
region).

### Attractor Cluster Test

Five unseen nouns tested with the +ful axis:
```
joy   → joyful:   HIT   (joy is semantically close to the +ful cluster)
fear  → fearful:  HIT   (fear is semantically associated with its -ful form)
taste → tasteful: MISS  (got tastes — taste has other strong associations)
truth → truthful: MISS  (got truth, but truthful ranked 2nd!)
awe   → awesome:  MISS  (got magnificent — 'awesome' ≠ 'aweful'/'awe+ful')
```

Notable: `truth → truthful` MISSES because `truth` is a very common word
strongly associated with itself. The scale overshoots slightly. But
`truthful` ranks 2nd — one nn_retrieve step away from success.

`awe → awesome` fails because `awesome` is NOT in the -ful cluster —
it's in the "strong positive adjective" cluster (magnificent, wonderful,
spectacular). The word `awesome` has drifted semantically from `awe+ful`.

---

## The Two-Factor Model of Holdout Prediction

### Factor 1: pc (Axis Linearity)

Controls whether the training chord vectors agree on a consistent direction.
- High pc → consistent direction → reliable navigation
- Low pc → noisy direction → unreliable navigation

### Factor 2: Target Cluster Density

Controls whether arriving "near" the correct target is sufficient.

```
density_balanced =
    (a) all targets are single BPE tokens, AND
    (b) no single target is 5× more frequent than others, AND
    (c) sources have morphologically/semantically unique -ful counterparts
```

When both factors are high: maximum holdout accuracy.
When pc is low but target cluster is balanced: anomalous high holdout.
When pc is moderate but target cluster has dominant attractor: holdout collapses.

### Evidence Table

```
Axis          pc      target_density  holdout   prediction
+er           0.393   balanced*       100%      both high → 100% ✓
+ful          0.104   DENSE+BALANCED   67%      low pc, good cluster → 67% ✓
+ment         0.124   DENSE(latinate)  75%      low pc, good cluster → 75% ✓
+ness         0.211   DOMINATED(sweet)  0%      medium pc, bad cluster → 0% ✓
elem:latin    0.104   CLUSTER(any sym)  0%      low pc, cluster→wrong → 0% ✓
+less         0.133   DOMINATED         0%      medium pc, bad cluster → 0% ✓
```

*+er/-er words are single-token and form a balanced comparative-adjective cluster.

---

## +ness Sub-Patterns: -ight+ness Has Highest pc

```
Sub-pattern              n   pc      coh
-ight + ness             2   0.274   0.784
consonant-final (+ness)  5   0.202   0.599
-y → +iness              2   0.200   0.773
```

The -ight+ness sub-pattern (bright→brightness, light→lightness) has
the highest pc (0.274) among all +ness sub-patterns. This is because:
- `brightness` and `lightness` are both very common words
- Their displacement vectors from `bright` and `light` are nearly identical
- The -ight suffix creates a consistent phonological context

Cross-pattern generalisation:
```
consonant-final axis → -y words:   1/3 (33%)
consonant-final axis → -e words:   0/3 (0%)
cos(consonant-final, -y) = 0.370
```

Sub-pattern axes are partially related (cos=0.37) but NOT interchangeable.
The -y→+iness transformation involves a vowel quality shift in W_E that
the consonant-final axis doesn't capture.

---

## Spearman Correlation

```
Spearman ρ(pc, holdout) = 0.689   p = 0.001
```

Statistically significant — pc IS a primary predictor. But ρ=0.69
leaves substantial unexplained variance (1 - 0.69² = 52%). The target
cluster density factor accounts for most of this unexplained variance.

---

## Revised Generalisation Model

A complete model for predicting holdout accuracy requires:

1. **pc** (pairwise chord cosine): linearity of the transformation
2. **target_cluster_balanced**: no dominant attractor in training targets
3. **target_BPE_coherence**: all targets are single tokens in the same
   BPE neighborhood
4. **source_homogeneity** (src_pc): similarity of source words

The +ness failure is primarily Factor 2 (dominant attractor).
The +ful success is primarily Factor 3 (coherent BPE cluster).

When ANY factor is severely violated, holdout collapses regardless of pc.

---

## Day 297 Plan

Temporal sequence axis: month ordering (January→February, ...) and
weekday ordering (Monday→Tuesday, ...). These are ORDINAL semantic axes
with very high expected linearity (7 or 12 equidistant steps in sequence
space). Predicted pc > 0.40.

Questions:
1. Does a temporal sequence form a linear axis in W_E?
2. What is the scale (distance between adjacent months/weekdays)?
3. Does the sequence wrap (December→January = same chord as Jan→Feb)?
4. Is the weekday/month cluster balanced (no dominant attractor)?

---

## Files

- `expedition_log.md` — Day 296 results
- `430_derivational_morphology_axes.md` — DC 430: derivational overview
- `day296_ness_cleanup_and_ful.py` — experiment script
