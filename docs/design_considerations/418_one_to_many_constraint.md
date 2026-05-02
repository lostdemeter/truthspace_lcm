# DC 418: The One-to-Many Constraint — When Axes Fail

**Day 283 | Testing the 2-hop architecture on a new domain
(person→field→concept) reveals a fundamental constraint. The
field→concept axis achieves only coh=0.401 and 15% accuracy.
Root cause: field→concept is a one-to-many relation (physics maps
to gravity, motion, energy, force, light, quantum, ...). The mean
chord direction averages over many divergent target directions,
yielding a noisy axis. The constraint: axes only work for many-to-one
or one-to-one relations. One-to-many relations violate the single
mean-direction assumption and produce low coherence, low accuracy
axes. person→field is salvageable with clustering (same attractor
problem as person→nat); field→concept is not.**

---

## The Three Relation Types and Their Axis Properties

| Relation Type | Example | Coherence | Accuracy | Why |
|---|---|---|---|---|
| Many-to-one | persons→nat (German cluster) | 0.91 | 100% | All chords parallel |
| One-to-one | nat→language | 0.58 | 83% | Each has one canonical target |
| One-to-many | field→concept | 0.40 | 15% | Chords diverge over many targets |

### Many-to-One (Highest Coherence)

Multiple sources → same target. All displacement chords point in
approximately the same direction (toward the shared target). The
mean direction is accurate; coherence is high.

Example: Russian cluster (coh=0.908). Lenin, Tolstoy, Dostoevsky,
Pushkin all live in a tight neighbourhood, and all point toward
the word 'Russian'. The displacement vectors are nearly parallel.

```
Lenin ──→ Russian
Tolstoy ──→ Russian      All chords ≈ parallel → coh=0.908
Dostoevsky ──→ Russian
Pushkin ──→ Russian
```

### One-to-One (Moderate Coherence)

Each source → unique target, but the direction is consistent because
the TYPE of displacement is shared. Nationality adjectives are all
displaced toward their respective language words by approximately
the same kind of transformation.

Example: nat→lang (coh=0.583). German→german, French→french,
Russian→russian. Different target words, but the displacement
direction "toward the language word from the nationality adjective"
is consistent in W_E.

```
German ──→ german
French ──→ french      Directions approximately consistent → coh=0.583
Russian ──→ russian
Italian ──→ italian
```

### One-to-Many (Low Coherence)

Each source → multiple valid targets, with no single canonical one.
The mean chord averages over divergent directions. Coherence is low.

Example: field→concept (coh=0.401).

```
physics ──→ {gravity, motion, force, energy, light, ...}
biology ──→ {cell, evolution, gene, DNA, species, ...}
economics ──→ {market, trade, capital, money, price, ...}
```

The displacement from 'physics' to 'gravity' points in a different
direction from 'physics' to 'motion', 'physics' to 'energy', etc.
The mean of these 20 training chords (2 per field × 10 fields)
is a nearly-null vector pointing vaguely toward "generic academic
concept" — the centroid of all concept words.

---

## The Attractor Phenomenon (Generalised)

In both person→nat (Day 281) and person→field (Day 283), the same
phenomenon appears: the axis direction is pulled toward the most
frequent target in the training data.

For person→nat: the German cluster has the most pairs (5-7), so the
axis is biased toward 'German'. Non-German persons are dragged toward
German.

For person→field: 'economics' appears as the target for 3+ persons
(Marx, Keynes, Adam Smith), and the W_E region around 'economics'
overlaps with philosophy, history, and social science thinkers. So
Einstein, Turing, Aristotle all retrieve 'economics'.

**The attractor is the dominant cluster in the training pairs.**

The fix is always the same: **source-type clustering**. Build a
separate axis for each field cluster:
- Physics cluster: Newton, Kepler (Einstein excluded — multi-token)
- Mathematics cluster: Euler, Gauss, Pythagoras (multi-token)
- Philosophy cluster: Aristotle, Plato, Kant
- Psychology cluster: Freud, Jung, Pavlov (multi-token)
- Economics cluster: Marx, Keynes

For field→concept, clustering cannot fix the problem because the
one-to-many nature means even a single-field axis has multiple
valid targets.

---

## What Can Be Done About One-to-Many Relations?

### Option 1: Reduce to One-to-One via specificity

Instead of field→concept (physics→gravity OR motion OR energy), use
field→PRIMARY concept (physics→energy by training data frequency).
This sacrifices completeness for axis quality.

```
physics    → energy    (most common association in training data)
biology    → life      (most common)
economics  → money     (most common)
mathematics → number   (most common)
```

The resulting axis would have higher coherence but only retrieves
the statistically dominant concept, not all valid ones.

### Option 2: Multiple axes per relation

Instead of one field→concept axis, build multiple:
- field→primary_concept axis
- field→secondary_concept axis
- field→example_concept axis

Each axis targets a consistent sub-type of concept. For retrieval,
apply all axes and take the union.

### Option 3: Inverse relation (concept→field)

If field→concept is one-to-many, concept→field is many-to-one.
'gravity' maps only to 'physics'. 'cell' maps only to 'biology'.
'market' maps only to 'economics'. This is a high-coherence,
reliable axis.

```
concept→field:   gravity→physics, motion→physics, force→physics
                 evolution→biology, cell→biology
                 market→economics, trade→economics
```

The inverse relation allows retrieval of field from concept, but
not concept from field. For the chain person→field→concept, we
would need a different strategy.

---

## The Reversibility Principle

A key finding across Days 276-283:

> **For a well-formed axis, the inverse relation tends to be more
> reliable than the forward relation.**

- person→nat (global, coh=0.49) vs nat→person (one-to-many, worse)
- nat→lang (coh=0.58) vs lang→nat (many-to-one, higher coh!)
- field→concept (one-to-many, coh=0.40) vs concept→field (many-to-one, predicted coh=0.70+)

The many-to-one direction of any relation is always recoverable.
For knowledge retrieval tasks, design chains that use MANY-TO-ONE
hops whenever possible.

---

## Axis Viability Checklist

Before building an axis for a new semantic relation, check:

1. **Multiplicity**: Does each source have ONE canonical target?
   - One canonical: proceed (one-to-one or many-to-one)
   - Multiple valid: expect coh<0.50, consider inverse or clustering

2. **Source homogeneity**: Are sources all the same type?
   - Homogeneous: high coherence expected (coh>0.70)
   - Mixed: expect attractor bias, apply clustering

3. **Tokenisation**: Are source AND target single-token?
   - Both single-token: include in axis
   - Either multi-token: exclude from training, handle separately

4. **Contextual override**: Does W_E encode contextual rather than
   definitional associations for this relation?
   - Greek→english vs Greek→greek (contextual override)
   - Polish→english vs Polish→polish (homograph override)
   - These cases cannot be fixed by scale or clustering

Following this checklist predicts whether an axis will achieve
high (>70%), moderate (40-70%), or low (<40%) accuracy before
building.

---

## Summary of the Axis Viability Landscape (All Tested)

| Axis | Relation type | Coherence | Accuracy | Viable? |
|------|--------------|-----------|----------|---------|
| plural | one-to-one (morph) | 0.498 | 100% | YES (Type A) |
| past_tense | one-to-one (morph) | 0.481 | 94% | YES (Type A) |
| comparative | one-to-one (morph) | 0.638 | 100% | YES (Type A) |
| superlative | one-to-one (morph) | 0.664 | 100% | YES (Type A) |
| gender | one-to-one (morph) | 0.496 | 100% | YES (Type A) |
| capital | one-to-one | 0.775 | 100% | YES (Type B) |
| language | one-to-one | 0.694 | 100% | YES (Type B) |
| dem_country | one-to-one | 0.787 | 100% | YES (Type B) |
| city_country | one-to-one | 0.775 | 100% | YES (Type B) |
| country_axis | many-to-one | 0.751 | 100% | YES (Type B) |
| nat→lang | one-to-one | 0.583 | 83% | YES (Type B) |
| person→nat (cluster) | many-to-one | 0.75-0.91 | 100% | YES (clustered) |
| person→nat (global) | mixed | 0.491 | 41% | PARTIAL |
| person→field (global) | mixed | 0.497 | 60% | PARTIAL |
| currency | one-to-one | 0.607 | 75% | PARTIAL |
| antonym | structural | 0.280 | 69% | PARTIAL (Type C) |
| hypernym | one-to-many | 0.390 | 43% | PARTIAL |
| meronym | one-to-many | 0.332 | 18% | NO |
| field→concept | one-to-many | 0.401 | 15% | NO |

---

## Files

- `expedition_log.md` — Day 283 results
- `417_two_hop_architecture.md` — 87% 2-hop result (Day 282)
- `415_axis_type_taxonomy.md` — three axis types (Day 280)
