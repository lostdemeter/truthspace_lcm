# DC 350: W_E Geometric Encoding Archetypes — Saturation Curves

**Day 166 | Four distinct learning curves reveal four types of relational geometry**

---

## Overview

Day 166 measures how direction-based accuracy changes as k (number of
training examples) increases from 0 to N-1 via leave-one-out evaluation.

**Core finding:**

> **W_E encodes relational facts via four distinct geometric mechanisms.
> Antonyms and gender are proximity-encoded (no direction needed, any
> direction hurts). Metals saturate at k=1. Capitals need k=8-10.
> Planets and colors have multi-pole geometry where direction averaging
> is anti-helpful.**

---

## Full Saturation Data

```
domain       k=0    k=1    k=2    k=3    k=4    k=5    k=6    k=7    k=8    max   type
────────────────────────────────────────────────────────────────────────────────────────
capitals     0.09   0.00   0.03   0.27   0.44   0.66   0.74   0.82   0.87  0.91   C
antonyms     0.75   0.00   0.00   0.00   0.04   0.09   0.12   0.16   0.19  0.75   A
gender       1.00   0.70   0.83   0.82   0.86   0.87   0.88   0.88   ---   1.00   A
metals       0.33   1.00   1.00   1.00   1.00   1.00   ---    ---    ---   1.00   B
planets      0.00   0.43   0.36   0.37   0.29   0.29   0.00   ---    ---   0.43   D
colors_temp  0.00   0.25   0.32   0.26   0.18   0.16   0.14   0.00   0.00  0.32   D
languages    1.00   0.78   1.00   1.00   1.00   1.00   1.00   1.00   1.00  1.00   A
```

---

## The Four Geometric Archetypes

### Type A: Proximity-Encoded (k=0 is optimal)

```
Antonyms:  k=0 → 0.75, k=1 → 0.00  (direction DESTROYS accuracy)
Gender:    k=0 → 1.00, k=1 → 0.70  (direction HURTS)
Languages: k=0 → 1.00, k=1 → 0.78  (direction HURTS temporarily)
```

The relational structure IS the proximity. Antonym pairs (hot/cold,
big/small, rich/poor) are already nearest neighbors in W_E without any
direction — they are encoded as opposite poles of the same semantic axis.
Adding a direction vector moves the query away from the true nearest neighbor.

**Geometric interpretation:**
```
hot ←────────────────────→ cold
     ANTONYM PROXIMITY AXIS

Adding direction vector:
hot + antonym_dir ≠ cold  (moves query off the axis)
```

This is the strongest form of W_E knowledge encoding: the relation IS
the embedding structure. No inference or direction needed.

### Type B: Fast-Saturation (k=1 achieves maximum)

```
Metals: k=0 → 0.33, k=1 → 1.00, k=2..5 → 1.00
```

One training example completely defines the direction. After seeing
{iron → metal}, the system correctly maps all other metals (copper,
aluminum, tin, zinc, lead) to "metal" with 100% accuracy.

**Geometric interpretation:**
The metal category label "metal" lies along a single coherent direction
from ALL metal instances. This direction is strongly defined in W_E
because "X is a metal" appears consistently in the training corpus.
One example is sufficient because the direction from iron to metal is
essentially identical to the direction from copper to metal.

### Type C: Slow-Saturation (needs k=8-10 examples)

```
Capitals: k=0 → 0.09, k=1 → 0.00, ..., k=8 → 0.87, k=10 → 0.91
```

k=1 is WORSE than k=0. A single country→capital pair gives a noisy
direction (e.g. France→Paris may point toward Parisian words rather
than purely toward the capital category). The direction only becomes
reliable when averaged over 8+ diverse examples.

**Geometric interpretation:**
```
Europe cluster: France, Germany, Spain, Italy → similar geographic context
Each country→capital direction has a large geographic component:
  France→Paris: direction contains Europe + city-of-France signal
  Russia→Moscow: direction contains Eurasia + Slavic signal

Average over 8+ examples cancels the geographic noise, revealing
the true "capital-of" subspace axis.
```

The slow saturation of capitals is a direct consequence of geographic
clustering: neighboring countries have similar W_E positions, creating
correlated noise in individual direction estimates.

### Type D: Multi-Pole Geometry (direction is anti-helpful)

```
Planets:     k=0 → 0.00, k=1 → 0.43, k=2 → 0.36, ..., k=6 → 0.00
Colors_temp: k=0 → 0.00, k=1 → 0.25, k=2 → 0.32, ..., k=7 → 0.00
```

The direction DEGRADES with more training examples and eventually reaches
0% (worse than no direction). This is caused by MULTI-POLE geometry:
the relation has two or more incompatible target clusters.

**For planets:**
```
Rocky planets:  Mercury, Venus, Earth, Mars  →  direction to "rocky"
Gas planets:    Jupiter, Saturn, Uranus, Neptune  →  direction to "gas"
```
These two directions are not just different — they point in incompatible
directions. Averaging them produces a vector that points toward neither
"rocky" nor "gas", and with more examples the confusion grows.

**For colors:**
```
Warm colors: red, orange, yellow  →  direction to "warm"
Cool colors: blue, green, purple  →  direction to "cool"
Neutral: white, black, gray  →  direction to "neutral"
```
Three incompatible directions. Any average is noise.

**Critical Distinction:**
Multi-pole geometry doesn't mean the knowledge is absent from W_E.
It means the relation cannot be captured by a single direction.
To handle Type D domains, you need:
1. A prior classifier to identify which sub-category the instance belongs to
2. Then apply the sub-category-specific direction

---

## Corrections to Prior Days

### Day 162 Correction

Day 162 concluded that planets (0%) and colors (0%) had "no W_E structure
accessible via direction". This was partially wrong:

- The structure IS accessible but requires ROUTING (not a single direction)
- k=1 gives 43% for planets — better than chance
- But k=6 gives 0% — more examples make it worse
- The Day 162 conclusion should be: "Type D multi-pole geometry"

### Day 164 Correction

Day 164 reported planets→50% and colors→80% with "own direction (4 pairs)".
Day 166 LOO reveals this was an artifact:
- The 4 specific pairs chosen in Day 164 happened to be rocky→rocky+gas→gas
  in a balanced way that produced a usable direction for that specific test set
- This doesn't generalize: different 4-pair subsets give widely different accuracy
- True ceiling for planets with any direction: ~43% (at k=1)

---

## Updated Accuracy Map

```
Domain encoding type and maximum achievable accuracy:

Type A (Proximity):
  antonyms:   75% (proximity, no examples needed)
  gender:     100% (proximity, no examples needed)
  languages:  100% (proximity, no examples needed)

Type B (Fast-Direction):
  metals:     100% (direction, k=1 sufficient)

Type C (Slow-Direction):
  capitals:   91% (direction, k=8-10 needed)

Type D (Multi-Pole):
  planets:    43% max (k=1), degrades to 0% with more
  colors_temp: 32% max (k=2), degrades to 0% with more
```

---

## Theoretical Framework: Why These Four Types?

**Type A: Bidirectional Proximity**
Antonyms, gender pairs, and country/language pairs all share one property:
they co-occur symmetrically in text. "hot and cold", "king and queen",
"Germany and German" appear in paired contexts. W_E places both words
near the same context, making them proximal. The direction estimate is
noisy because both words are equidistant — there's no consistent
"forward" direction.

**Type B: Unidirectional Category Membership**
Metal instances co-occur asymmetrically with the category label:
"iron is a metal" (not "metal is an iron"). The direction from instance
to category is consistent and non-noisy. One example is sufficient to
define this consistent direction.

**Type C: Noisy Unidirectional with Geographic Clustering**
Capitals face the same asymmetric structure as metals ("France is in
Europe, its capital is Paris") but the geographic clustering creates
directional noise. The direction has a "correct" component (toward
capital labels) and a "noisy" component (toward specific geographic
context). Many examples are needed to average out the noise.

**Type D: Opposing Directional Forces**
Planet types and color temperatures have two competing sub-categories
that pull in opposite directions. The relation "planet type" is not a
single function from planet to label — it's TWO different functions
(rocky→rocky, gas→gas) that point in opposing directions.

---

## Implications for TruthSpace Pipeline

### Routing Architecture

The saturation curve results suggest a 3-tier pipeline:

```
Tier 1: Proximity lookup (no direction)
  → Use for: antonyms, gender, language pairs
  → Condition: query is already nearest to answer in W_E

Tier 2: Direction-augmented lookup (domain-specific direction)
  → Use for: capitals, metals, any Type B/C domain
  → Condition: query needs to be moved toward a target cluster

Tier 3: Multi-pole routing + sub-category direction
  → Use for: planets, colors (Type D domains)
  → First classify which sub-category, then apply that sub-direction
  → Not yet implemented
```

### Training Example Requirements

```
Domain type     Examples needed
────────────────────────────────
Type A          0 (zero-shot)
Type B          1 (one-shot)
Type C          8-10 (few-shot)
Type D          2-3 per sub-pole (routing needed first)
```

### Confidence Gate Behavior

The confidence gate (cosine score threshold) should be calibrated
differently per encoding type:

- Type A: higher threshold (proximity scores are high for true answers)
- Type B: high threshold (metal direction is very sharp)
- Type C: lower threshold (direction is noisier, scores lower)
- Type D: threshold on sub-category classifier, not direction score

---

## Files

- `expedition_day166_fewshot_saturation.py` — full saturation curves
- `day166_fewshot_saturation.json` — results
- `349_direction_orthogonality.md` — prior arc
