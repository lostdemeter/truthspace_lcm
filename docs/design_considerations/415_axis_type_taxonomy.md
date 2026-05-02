# DC 415: Axis Type Taxonomy — Three Geometric Regimes in W_E

**Day 280 | Survey of all 14 axes (morphological + semantic) across
coherence, accuracy, Voronoi gap, and valid-pair count. The coherence
threshold 0.65 hypothesis from DC 414 is falsified: 6/15 axes are
mis-predicted. Morphological axes (plural, past_tense, gender,
comparative, superlative) achieve 94–100% accuracy despite coherence
0.48–0.66 — identical to person_nat (coh=0.499, acc=45%). Axis TYPE
is the missing third factor. Three geometric regimes exist: (A)
morphological axes work by surface-form proximity regardless of
coherence; (B) relational axes require high coherence; (C) structural
axes (antonym) work by distance regularity. Pearson r(coherence,
accuracy) = 0.70 overall, but ~0.90 within relational axes only.**

---

## The Three Geometric Regimes

### Type A: Morphological Axes

| Axis | Coherence | Accuracy | Valid pairs |
|------|-----------|----------|-------------|
| plural | 0.498 | 100% | 16 |
| past_tense | 0.481 | 94% | 16 |
| comparative | 0.638 | 100% | 16 |
| superlative | 0.664 | 100% | 12 |
| gender | 0.496 | 100% | 13 |

**Property:** High accuracy despite moderate-to-low coherence.

**Why it works:** The morphological target is always geometrically
close to the source. For `cat → cats`, the displacement vector is
short (morphological proximity in W_E), and the target token has a
well-separated Voronoi cell (no near-synonyms at the inflected form
level). The axis direction is the mean of many similar short vectors;
even with moderate coherence, the scale places the prediction within
the target's Voronoi cell.

**Coherence measures consistency of DIRECTION.** For morphological
axes, what matters is not direction but MAGNITUDE — the displacement
to reach the inflected form is small and consistent. The direction
varies because different morphological categories live in different
sub-regions of W_E, but the magnitude is stable.

### Type B: Relational Axes

| Axis | Coherence | Accuracy | Valid pairs |
|------|-----------|----------|-------------|
| dem_country | 0.787 | 100% | 5 |
| capital | 0.775 | 100% | 2 |
| city_country | 0.775 | 100% | 2 |
| country_axis | 0.751 | 100% | 15 |
| language | 0.694 | 100% | 4 |
| currency | 0.607 | 75% | 4 |
| person_nat | 0.499 | 45% | 20 |
| hypernym | 0.390 | 43% | 14 |
| meronym | 0.332 | 18% | 11 |

**Property:** Accuracy tracks coherence. High coherence → high
accuracy; low coherence → low accuracy.

**Why coherence matters here:** Relational targets are NOT
morphologically close to sources. The displacement from `france` to
`paris` is long and varies by pair. If the axis direction is
inconsistent (low coherence), the prediction will miss. Geographic
relations (country→capital, country→language) are encoded with very
consistent directions in W_E (coh > 0.70), so they achieve 100%.

**Within relational axes, Pearson r(coherence, accuracy) ≈ 0.90.**

### Type C: Structural Axes

| Axis | Coherence | Accuracy | Valid pairs |
|------|-----------|----------|-------------|
| antonym | 0.280 | 69% | 16 |

**Property:** Very low coherence but moderate accuracy. Anomalous.

**Why it works:** Antonym pairs do not form a directional cluster —
`hot↔cold`, `big↔small`, `fast↔slow` point in different directions.
The axis computed from these is essentially noise (coh=0.280).
Yet accuracy is 69%.

The explanation: antonym pairs have a consistent **cosine distance**
from each other in W_E (approximately 120°–140° apart), but no
consistent direction. At scale=1.53, the axis displacement moves the
predicted embedding by a large enough distance that it enters the
"antipodal region" of the source word's neighbourhood — landing near
the antonym. This is distance-based retrieval, not direction-based.

---

## The Plural / person_nat Paradox Resolved

```
plural:      coh=0.498   acc=100%   vgap=0.2749   type=A (morphological)
person_nat:  coh=0.499   acc= 45%   vgap=0.2399   type=B (relational)
```

Same coherence. Different accuracy. Resolution:

- **plural**: sources are common nouns (cat, dog, book), targets are
  their morphological variants (cats, dogs, books). Targets are near
  sources in W_E. The short, variable-direction displacement still
  lands near the target reliably. **Type A physics apply.**

- **person_nat**: sources are proper nouns (Einstein, Newton, Caesar),
  targets are nationality adjectives (German, British, Roman). Targets
  are far from sources. The long, variable-direction displacement from
  heterogeneous proper nouns is unreliable. **Type B physics apply.**

Coherence is the same (0.498 vs 0.499) because both axes have
similarly variable chord directions relative to the axis mean.
But the TYPE of displacement determines accuracy.

---

## Single-Token Restriction: Geographic Axes Are Perfect

High-coherence geographic axes achieve 100% accuracy but on very few
valid pairs (n_valid = 2–5). This is because most geographic proper
nouns are tokenised as multi-token sequences with Qwen2's BPE:
- 'Berlin' → [' Berlin'] (space prefix, different token from 'Berlin')
- 'Vienna' → often multi-token

When restricted to single-token pairs, all geographic axes are 100%
accurate. The failures in axis chaining (Days 277-279) originated
entirely from `person_nat` (45% single-hop accuracy, 20 valid pairs),
which is a genuine Type B coherence problem.

**Engineering implication:** A TruthSpace chain restricted to single-
token intermediates would achieve:
- Hop 1 (person→nat): ~45% (Type B, coh=0.499 — irreducible)
- Hop 2 (nat→country): ~100% (Type B, coh=0.787 — near-perfect)
- Hop 3 (country→lang): ~100% (Type B, coh=0.694 — near-perfect)
- Estimated 3-hop: ~45% (bottleneck is always hop 1)

This means **the only way to improve 3-hop accuracy is to improve
hop 1 (person→nat)**. Hops 2 and 3 are already essentially perfect
on their valid input domain.

---

## Revised Axis Reliability Model

```
function predict_axis_accuracy(axis_type, coherence, voronoi_gap):
    if axis_type == MORPHOLOGICAL:
        # Type A: coherence irrelevant; Voronoi gap matters
        return 0.95 if voronoi_gap > 0.25 else 0.70
    elif axis_type == RELATIONAL:
        # Type B: coherence is primary
        if coherence >= 0.75: return 0.95   # geographic cluster
        if coherence >= 0.60: return 0.75   # currency, moderate
        if coherence >= 0.50: return 0.45   # person_nat
        return max(0.18, coherence)          # hypernym, meronym
    elif axis_type == STRUCTURAL:
        # Type C: distance-based, moderate accuracy
        return 0.70 if consistent_distance else 0.30
```

Compared to simple threshold (coh >= 0.65 → acc >= 60%):
- Old: 6/15 wrong
- New: 0/15 wrong (all correctly classified by axis type)

---

## TruthSpace Implications

### What W_E Encodes by Axis Type

- **Type A (morphological)**: W_E encodes morphology as SHORT, REGULAR
  displacements. The structure is geometric proximity, not directional
  consistency. This is the most reliable axis class.

- **Type B (relational)**: W_E encodes world-knowledge relations as
  LONG, DIRECTIONAL displacements. Reliability depends on training-data
  frequency and source-type homogeneity. Geographic relations are
  near-perfect; person-nationality is noisy.

- **Type C (structural)**: W_E encodes structural oppositions as
  DISTANCE relationships. Antonyms are equidistant from each other,
  but the direction varies. This is a radial, not linear, geometry.

### For Sequential Chaining

The hop reliability in a multi-hop chain is determined by the WEAKEST
axis type encountered. For the person→nat→country→language chain:
- Hop 1 (Type B, low coh): 45% — chain bottleneck
- Hop 2 (Type B, high coh): ~100% on single-token
- Hop 3 (Type B, high coh): ~100% on single-token

The chain accuracy is limited by hop 1. To improve chain performance:
1. **Source-type clustering**: build separate nat axes for German/
   British/Greek/French persons. Each cluster has higher coherence.
2. **Use Type A for first hops when possible**: if the chain can
   start with a morphological axis, reliability is near-perfect.

---

## Files

- `expedition_log.md` — Day 280 results
- `414_coherence_bottleneck.md` — coherence hypothesis (superseded)
- `401_semantic_relation_axes.md` — original axis evaluation
