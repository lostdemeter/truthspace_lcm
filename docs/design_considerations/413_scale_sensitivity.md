# DC 413: Scale Sensitivity — Global Scale is a Bottleneck for Axis Chaining

**Day 278 | Attempting to fix the two failure modes from DC 412 (German
bias in nationality axis; capitalisation mismatch in demonym→country axis)
reveals a deeper problem: the global scale parameter is fragile. Extending
the language training pairs by 4 additional entries shifts the optimal
scale from 0.25 to 0.66, collapsing hop 3 accuracy from 50% to 23%.
Fix 1 (balanced nationality pairs) successfully improved hop 1 from
60%→69%. Fix 2 (capitalisation normalisation) correctly resolved
Aristotle and Plato's country-retrieval failures. But the scale collapse
in the language axis negates both improvements. The root problem:
a single global scale is a mean-field approximation over all pairs,
and its optimal value depends on the specific pair distribution in W_E.**

---

## What Changed Between Day 277 and Day 278

| Axis | Day 277 scale | Day 278 scale | Hop accuracy |
|------|--------------|--------------|-------------|
| language | 0.25 | 0.66 | 50% → 23% |
| person→nat | 0.41 | 0.30 | 60% → 69% |
| demonym→cty | 0.56 | 0.54 | 50% → 54% |

Only the language axis changed dramatically. The cause: four additional
training pairs were added to cover more languages:
```
('india','hindi'),  ('brazil','portuguese'),
('austria','german'), ('netherlands','dutch')
```

These pairs have longer chords in W_E (the source and target embeddings
are farther apart than for European country→language pairs). Including
them raises the mean chord length, shifting the optimal scale upward
from 0.25 to 0.66.

---

## Why Scale Sensitivity Matters

The scale parameter `s` in `target ≈ source + s × axis` is found by:
```
s* = argmax_{s} accuracy(pairs, s)
```

This is a discrete grid search, not a closed-form solution. The answer
depends on:

1. **The pair distribution**: If some pairs have short chords (need small s)
   and others have long chords (need large s), the optimal s is a compromise
   that is suboptimal for both extremes.

2. **The target neighbourhood radius**: If the target embedding has many
   near-synonyms or inflections in its neighbourhood, a larger s can still
   land in the correct neighbourhood; if the target is isolated, precision
   matters more.

3. **The axis coherence**: Low coherence → larger variance in predicted
   direction → scale matters more for precision.

The language axis trained on European pairs has coherence 0.694 and most
pairs have short chords (European languages are closely related in W_E).
Adding Hindi, Portuguese, Dutch creates longer chords. The optimal scale
for these is 0.50–0.70; for European pairs it is 0.20–0.30. A global
scale of 0.66 overshoots German, French, and Spanish.

---

## Per-Pair Scale Variation

To quantify this, consider: if we use each pair's own optimal scale:

```
Pair                        Chord length    Optimal scale (approx)
france → french             0.41            ~0.25
germany → german            0.38            ~0.25
japan → japanese            0.49            ~0.35
india → hindi               0.68            ~0.55
brazil → portuguese         0.60            ~0.50
austria → german            0.55            ~0.40
```

The chord lengths vary by ~1.7×. A global scale of 0.25 undershoots
India→Hindi; a global scale of 0.66 overshoots France→French.

---

## Solutions

### Solution 1: Source-Projection Scaling (Recommended)

The scale needed for a given source embedding is proportional to its
projection onto the axis direction:

```
s(source) = s_base × (1 + α × (proj - mean_proj))
```

Where `proj = dot(normed(source_emb), axis)` and `mean_proj` is the mean
projection over training sources. Sources far from the training distribution
need larger scales; sources close to the mean need the base scale.

This requires fitting `α` on training data but adds only one parameter.

### Solution 2: Per-Axis Scale Calibration from Chord Length

Estimate the optimal scale from the source embedding's distance to the
axis centroid:

```
s(source) = median_chord_length(training) / norm(source_component)
```

No additional parameters, but requires storing median chord length.

### Solution 3: Separate Axes Per Language Family

Build separate language axes for related groups:
- European axis (French, German, Spanish, Italian, …): scale ≈ 0.25
- Asian axis (Japanese, Chinese, Hindi, …): scale ≈ 0.45
- Slavic axis (Russian, Polish, …): scale ≈ 0.35

This increases the number of axes but keeps each scale clean.

### Solution 4: Use Cosine-Distance NN Instead of Euclidean

The current NN retrieval uses cosine similarity on normalised embeddings.
The scale matters because it shifts the normalised query vector. Using
a direction-only retrieval (always apply axis as a unit direction, retrieve
by cosine of normalised prediction) would decouple direction from magnitude.

```python
pred_direction = normed(source_emb + axis)  # not scaled
result = NN_cosine(pred_direction)
```

This eliminates the scale parameter entirely. The cost: the magnitude of
the displacement affects which cosine direction `normed(source + axis)`
points toward, so scale still matters implicitly. But in the cosine regime,
the effect is much weaker.

---

## What Still Works from Day 278

Despite the scale regression, two improvements are confirmed:

**Fix 1 (balanced nationality pairs): VALID**
- Hop 1 improved from 60% to 69%
- The German bias was genuinely reduced
- Marx→Marxist is a new failure, but it reflects a different issue:
  Marx is embedded near 'Marxist' in W_E, not near typical German names

**Fix 2 (capitalisation normalisation): VALID**
- Aristotle: Greek → greece (via normalisation) → CORRECTLY REACHES COUNTRY
- Plato: Greek → greece → CORRECTLY REACHES COUNTRY
- Without fix 2, both fail at hop 2 (Greek → Greece, which is not in dem_cty training)

Both fixes are real improvements. The scale problem is separable and solvable.

---

## Priority Order for Sequential Chain Improvements

| Issue | Impact | Fix difficulty | Priority |
|-------|--------|----------------|----------|
| Scale sensitivity | High (destroyed hop 3) | Medium (per-axis or per-family) | HIGH |
| Nationality bias (German) | Medium (hop 1 ~50%) | Low (more balanced pairs) | MEDIUM |
| Capitalisation mismatch | Low (2 cases) | Trivial (normalisation) | DONE |
| Proper noun leakage (Marx→Marxist) | Low (1 case) | Hard (require context) | LOW |

---

## Implications for TruthSpace

The axis chaining experiments (Days 276–278) establish the following
**engineering constraints** for a TruthSpace multi-hop query engine:

1. **Sequential NN grounding is mandatory** (Day 277: additive = 0%)
2. **Scale must be per-axis and per-family of source types** (Day 278)
3. **Capitalisation normalisation is a free 2% improvement** (Day 278)
4. **Axis training must be balanced** to avoid dominant-class bias (Day 278)
5. **Positive hop correlation** means the system reliability = hop 1 reliability
   (if hop 1 fails, entire chain fails; fixing hop 1 is the highest ROI)

---

## Files

- `expedition_log.md` — Day 278 results
- `412_sequential_chaining.md` — Day 277: sequential >> additive
- `411_axis_composition.md` — Day 276: additive composition failure
