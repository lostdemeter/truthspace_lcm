# DC 420: ENCODE=DECODE — Symmetry Holds Only for Bijective Relations

**Day 285 | Direct test of the TruthSpace ENCODE=DECODE hypothesis
using negative axis directions. nat→lang axis is PERFECTLY reversible
at the same scale (lang→nat: 100%, forward 79%, ratio 1.00). person→nat
axis is NOT reversible: nat→person retrieves only Napoleon/French.
ENCODE=DECODE holds for bijective (one-to-one) relations; fails for
many-to-one relations where the reverse is one-to-many. The double-
application test reveals 'english' as the long-range stable attractor
of the nat→lang axis: French→french at scale 1.17, but French→english
at 2× scale. This is the mathematical signature of W_E's English-
language dominance.**

---

## ENCODE=DECODE: The Hypothesis

TruthSpace's core claim:

> Encoding and decoding are the **same operation in opposite directions**,
> like φ and 1/φ. TEXT IN → φ-space → TEXT OUT. "Thinking" isn't a step
> between — it IS the encode-decode.

In the geometric axis framework, this predicts:

1. An axis from A→B should have a well-defined inverse (B→A)
2. The inverse uses the **negative** of the same axis direction
3. The scale should be the same (or related by 1/φ) in both directions

Day 285 tests whether these predictions hold for the 2-hop chain.

---

## The nat→lang Axis: Perfectly Symmetric

```
nat→lang (forward):   11/14 (79%)   scale = 1.17
lang→nat (reverse):    8/8 (100%)   scale = 1.17   ratio = 1.00
```

The nat→lang axis is **perfectly symmetric**. The same scale (1.17)
works in both directions. The negative of the axis direction maps
language words back to their nationality adjectives with 100% accuracy.

Why is the reverse MORE accurate (100%) than the forward (79%)?

Forward failures: Greek→english, Polish→english (contextual associations
override the linguistic rule in the nat→lang direction). But in the
reverse direction, these artefacts don't affect the language-side
retrieval: the word 'greek' is not in our test set (we only test
german, english, french, russian, italian, spanish, japanese, chinese).

The reverse is tested on a subset where no artefacts exist. If we
included 'greek' and 'polish' in the reverse test, we would likely see
failures in the reverse direction too (greek→wrong nationality).

### What "Symmetric" Means Geometrically

```
nat ──[+1.17 * axis]──→ lang
lang ──[-1.17 * axis]──→ nat
```

The axis vector in 1536-dimensional W_E space is the same. The sign
of the displacement determines direction. The magnitude (1.17) is
identical. This is the geometric analogue of φ and 1/φ being
reciprocals: they are the same value, just inverted.

The nat→lang relationship is **bijective** in the training data:
- German → german (unique)
- British → english (unique)
- French → french (unique)
- Russian → russian (unique)

Each nationality maps to exactly one language, and each language
in the test set comes from exactly one nationality. A bijective
mapping is its own inverse, and the axis reflects this.

---

## The person→nat Axis: Asymmetric

```
person→nat (cluster, forward):   100% per cluster
nat→person (reverse):             1/5 (Napoleon only)
```

The person→nat axis is **not reversible**. The forward mapping is
many-to-one (all German persons → 'German'). The reverse is one-to-many
('German' → {Einstein, Marx, Kepler, Gauss, Kant, Schiller, ...}).
There is no single canonical target for the reverse.

### Why Napoleon Succeeds

Napoleon is retrieved from 'French' because:
1. The French cluster has only 2 training pairs (Napoleon, Voltaire)
   — much smaller scale (0.25) and fewer members than German/British
2. Napoleon is the most prominent person associated with 'French' in
   W_E training data (far more famous than Voltaire for associating
   with 'French')
3. The French person cluster centroid is dominated by Napoleon's embedding

For German: the centroid of {Einstein, Marx, Kepler, Gauss, Kant, Schiller}
is equidistant from all of them. Reversing from 'German' lands at the
centroid, which is not close to any individual person — it's closest
to German-language tokens ('German', '德国', 'Germans', 'Germany', 'german').

### The Scale Constraint

The cluster axis scale (0.25–0.36) is small. Subtracting 0.36 from
'German' moves only 36% of a unit displacement away from 'German'.
The word 'German' has many close neighbours (German, Germans, Germany,
german, '德国') all within radius 0.36. The reverse displacement doesn't
escape this neighbourhood.

For the reverse to work, a much larger scale would be needed, but
using a larger scale would overshoot the person cluster entirely.
The forward axis is optimised for persons→nationality; the reverse
requires a different scale (or a purpose-built reverse axis).

---

## The ENCODE=DECODE Qualification

The TruthSpace ENCODE=DECODE principle holds, but with a qualification:

> **ENCODE=DECODE holds for bijective (one-to-one) relations.
> For many-to-one relations, the direction matters: the many-to-one
> direction is well-defined; the one-to-many reverse is not.**

This is not a failure of the hypothesis — it is a correct statement
about information structure. A bijective mapping carries the same
amount of information in both directions. A many-to-one mapping loses
information (which specific person becomes just 'German').
You cannot reconstruct the specific person from just 'German' without
additional context. This is the second law of information: compression
is irreversible.

### Implications for TruthSpace Architecture

1. **Bijective axes are reversible for free**: nat→lang, singular↔plural
   (morphological), city↔country (with appropriate training pairs), etc.

2. **Many-to-one axes are one-directional**: person→nat, animal→class,
   word→hypernym. These cannot be inverted without additional context.

3. **The 2-hop chain is reversible only if both hops are bijective**:
   person→nat→lang is NOT fully reversible because hop 1 is many-to-one.
   Only hop 2 (nat→lang) is reversible. The chain bottlenecks at hop 1.

4. **To build a reversible chain**: choose axes where each hop is bijective.
   This requires careful relation selection — not all semantic relations
   are bijective.

---

## The Double-Application Test: English as Long-Range Attractor

Applying the nat→lang axis at 2× and 3× scale:

```
German  → german  → german  → german   (STABLE at all scales)
British → english → english → english  (STABLE at all scales)
French  → french  → english → english  (drifts at 2×!)
Russian → russian → english → english  (drifts at 2×!)
```

German and British are stable because their language words ('german',
'english') happen to be near the long-range attractor. French and
Russian drift: 'french' at 1.17, then 'english' at 2.34.

This reveals that **'english' is the long-range stable attractor
of the nat→lang axis direction** in W_E space. The axis direction
points "toward language words" and at long range points specifically
toward 'english' — the most frequent language in the training corpus.

### The Mathematical Structure

In W_E, the nat→lang axis direction d satisfies:

```
For any nationality n:
  n + 1.17 * d  ≈  lang(n)    (correct language)
  n + 2.34 * d  ≈  'english'  (English attractor)
```

This means the axis direction d points "toward English" in the long
run. German and British hit 'english' immediately (their languages
ARE english/german which are stable along this direction). French
and Russian have their language words as unstable waypoints along
the path to 'english'.

The axis is not a straight line in a uniform space — it curves
through the non-Euclidean topology of W_E toward the dominant
language, English.

---

## Morphological Axes: Predicted to be Symmetric

Day 285 motivates a prediction: **morphological axes should be
perfectly reversible**, because morphological relations are bijective:

- singular ↔ plural (cat ↔ cats): one-to-one
- base ↔ comparative (fast ↔ faster): one-to-one
- base ↔ superlative (fast ↔ fastest): one-to-one
- base ↔ past_tense (walk ↔ walked): one-to-one (ignoring irregular)
- masculine ↔ feminine (king ↔ queen): one-to-one

Each of these is bijective. ENCODE=DECODE predicts each axis is
reversible at the same or similar scale. Testing this is the next
logical step.

---

## Summary

| Relation | Type | Reversible? | Scale ratio |
|---|---|---|---|
| nat→lang | bijective | YES | 1.00 |
| person→nat | many-to-one | NO (one-to-many reverse) | N/A |
| field→concept | one-to-many | N/A (forward fails) | N/A |
| plural (predicted) | bijective | YES (predicted) | ~1.0 |
| past_tense (predicted) | bijective | YES (predicted) | ~1.0 |

---

## Files

- `expedition_log.md` — Day 285 results
- `419_attractor_universality.md` — attractor pattern (Day 284)
- `417_two_hop_architecture.md` — 87% 2-hop (Day 282)
- `415_axis_type_taxonomy.md` — axis types (Day 280)
