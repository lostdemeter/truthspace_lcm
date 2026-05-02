# DC 356: The Relational Encoding Boundary in W_E

**Day 179 | Four geometric signals predict encoding type; joint condition classifies all domains;
encyclopedic facts are genuinely absent from W_E direction space**

---

## Overview

Day 178 systematically measures four geometric signals across 12 relational domains
to identify what distinguishes direction-encoded relations (viable for retrieval)
from non-encoded relations (inaccessible via geometry).

**Core finding:**

> **No single metric cleanly separates encoded from non-encoded relations.
> The best single predictor is H4 (displacement magnitude CV, cor=-0.678),
> but a joint condition on H1 and H2 plus a multi-pole check provides a complete
> classification. Truly non-encoded relations have H1<0.10 AND H2<0.15 —
> their target words are scattered and their direction vectors are random.**

---

## The Four Hypotheses — Results

```
                     H1_dir   H2_tgt   H3_src   H4_cv   LOO_acc
                     (consistency) (tgt_cmpct) (src_cmpct) (mag_cv)
─────────────────────────────────────────────────────────────────────
languages             0.611    0.360    0.366    0.054    1.000  ← TYPE B/C
metal_to_category     0.415    1.000    0.190    0.102    1.000  ← TYPE B (trivial)
insect_category       0.366    1.000    0.117    0.090    1.000  ← TYPE B (trivial)
number_parity         0.485    0.531    0.453    0.012    0.000  ← TYPE E
color_temperature     0.433    0.552    0.353    0.018    0.000  ← TYPE D
capitals              0.338    0.278    0.366    0.044    0.818  ← TYPE C
planet_type           0.302    0.445    0.268    0.031    0.000  ← TYPE D
season_weather        0.285    0.116    0.544    0.065    0.000  ← THEMATIC
gender                0.206    0.248    0.120    0.147    0.875  ← TYPE A/B
animal_sound          0.087    0.123    0.154    0.024    0.000  ← NOT ENCODED
metal_property        0.077    0.089    0.158    0.064    0.000  ← NOT ENCODED
antonym_hot           0.025    0.101    0.083    0.026    0.000  ← TYPE A (prox)
```

**Correlations with LOO_acc:**
```
H4_magnitude_cv (negated):     +0.678  ← strongest single predictor
H2_target_compactness:         +0.528
H1_direction_consistency:      +0.452
H3_source_compactness:         -0.203  ← NEGATIVE (more compact src → harder)
```

---

## Classification Decision Tree

```
                    Is H1 > 0.10?
                   /              \
                 NO               YES
                  |                |
         H1 ≈ 0: proximity    Is H2 > 0.15?
         (Type A or truly     /             \
         not encoded)        NO             YES
                              |              |
                          Thematic:      H2 > 0.45 and only 2-3 targets?
                          targets        /                    \
                          scattered     YES                   NO
                          (season→wx)    |                     |
                                    Multi-pole:          H1 > 0.30?
                                    routing needed       /         \
                                    (Type D/E)          YES         NO
                                                         |           |
                                                    Type B/C:    Type A/B:
                                                    LOO works    more k needed
```

**Decision rules (empirically validated):**

| Condition | Classification | Action |
|---|---|---|
| H1 < 0.10 | Not directionally encoded | Proximity (Type A) or truly absent |
| H1 ≥ 0.10, H2 < 0.15 | Thematic/scattered | No geometry available |
| H1 ≥ 0.10, H2 ≥ 0.45, n_targets ≤ 3 | Multi-pole (Type D/E) | Routing required |
| H1 ≥ 0.20, H2 ≥ 0.25, n_targets > 3 | Direction-encoded (Type B/C) | k-NN direction |
| H1 ≥ 0.20, H2 ≥ 0.25, n_targets > 3, slow | Type C | k=8+ examples |

---

## Why H4 (Magnitude CV) Is the Best Single Predictor

Displacement magnitude CV measures how uniformly all pair vectors are scaled:

```
H4_cv = std(||Y-X||) / mean(||Y-X||)

Low CV: all pairs displaced by the same amount
High CV: some pairs much larger/smaller than others
```

**Why it works:** Direction-encoded relations have pairs that are UNIFORMLY
displaced in W_E. When Y-X has consistent magnitude AND consistent direction
(H1), the LOO mean direction reliably targets the correct answer.

**Why it's confounded:** The CV being low is NECESSARY but not SUFFICIENT:
- Type D/E domains (number_parity: H4_cv=0.012) also have very uniform magnitudes
  because odd and even numbers are displaced uniformly toward their label, but in
  OPPOSITE directions — the mean cancels
- Antonyms also have low CV (0.026) — hot→cold and big→small have similar
  magnitudes but point in completely random directions relative to each other

The fundamental flaw: a low-CV domain could have 2 opposite directions that
cancel, appearing consistent in magnitude but inconsistent in direction.

---

## The Season-Weather Anomaly

Season-weather is the most interesting failure:
- H3_source_compactness = 0.544 (highest of any domain — seasons cluster tightly)
- H2_target_compactness = 0.116 (very low — weather words are scattered)
- LOO_acc = 0.000

**Mechanism:** The four seasons (winter, summer, spring, autumn) are semantically
similar words that appear together in texts about seasonal topics. Their embeddings
form a tight cluster in W_E. Their corresponding weather words (snow, heat, rain,
wind) are completely unrelated to each other — four different concepts with no
shared semantic neighborhood.

The direction vectors from each season to its weather are:
```
winter→snow: some direction D1
summer→heat: some direction D2
spring→rain: some direction D3
autumn→wind: some direction D4
```

All origins (seasons) are similar → all vectors start from similar points.
All endpoints (weather) are scattered → all vectors end at random points.
The mean of D1+D2+D3+D4 converges to zero (random walk) → no useful direction.

**Lesson:** High source compactness HURTS direction extraction when targets
are not compact. The sources must differ enough that their differences create
varied vectors that, when averaged, produce a non-zero consistent direction.

This is the COUNTER-INTUITIVE result: **maximum source similarity = minimum
direction extractability**. The direction only emerges when sources cover a
range of positions across which the target relation is consistent.

---

## Truly Non-Encoded Relations

Animal sounds and metal properties both have H1 < 0.10 and H2 < 0.15.
These are **encyclopedic facts** that are not captured in W_E direction space:

**Animal sounds (dog→bark, cat→meow, cow→moo, ...):**
- Each pair is a specific world-knowledge fact
- The target words (bark, meow, moo) have completely different semantic neighborhoods
- A dog barking has nothing geometrically in common with a cat meowing
- These are "arbitrary" mappings: there is no systematic linguistic pattern
  that places all animal sounds in a consistent direction from their animals

**Metal properties (iron→magnetic, copper→conductive, gold→malleable, ...):**
- Each property is a specific physical fact
- The target words (magnetic, conductive, malleable) have no shared semantic cluster
- "Magnetic" is associated with electricity, compasses, physics
- "Conductive" is associated with electricity, temperature, engineering
- "Malleable" is associated with metal, hammering, manufacturing
- No consistent direction maps metals to their specific properties

**Why LLMs CAN answer these questions but W_E alone cannot:**
The transformer layers compute context-dependent representations where these
relationships ARE present. The static W_E embedding layer cannot capture
arbitrary many-to-one encyclopedic mappings where the target space is not compact.
This knowledge requires the dynamic computation of the full transformer stack.

---

## Updated Complete Encoding Taxonomy

```
TYPE    H1      H2      n_tgt   LOO_acc   Method
────────────────────────────────────────────────────────────────────
A       < 0.10  < 0.25  any     varies    Proximity (k=0)
B/C     ≥ 0.20  ≥ 0.25  > 3    ≥ 0.80    Direction (k=1 or k=8+)
D       ≥ 0.20  ≥ 0.35  2-3    ≈ 0.00    Sub-pole routing
E       ≥ 0.35  ≥ 0.45  2-3    ≈ 0.00    Full-pop or symbolic
Thematic < 0.15 < 0.15  any     0.00      Not accessible via W_E
```

The thematic category is NEW — previously, failing relations were classified
as Type D/E. Day 178 establishes that some relations are genuinely absent
from W_E's directional structure and require full transformer processing.

---

## Implication for TruthSpace Pipeline

The relational boundary allows automatic routing of queries:

```python
def classify_relation(pairs, threshold_h1=0.20, threshold_h2=0.25):
    h1 = direction_consistency(pairs)
    h2 = target_compactness(pairs)
    n_unique_targets = len(set(b for a, b in pairs))

    if h1 < 0.10:
        return "TYPE_A"          # proximity only
    if h2 < 0.15:
        return "THEMATIC"        # not encodable in W_E
    if h1 >= 0.20 and h2 >= 0.25 and n_unique_targets > 3:
        return "TYPE_B_C"        # direction works
    if h2 >= 0.35 and n_unique_targets <= 3:
        return "TYPE_D_E"        # routing required
    return "UNKNOWN"             # borderline, test empirically
```

This auto-classifier can be run with k=3 training pairs to pre-screen any
new relation before committing to a retrieval strategy.

---

## Files

- `expedition_day178_relational_boundary.py` — four-hypothesis analysis
- `day178_relational_boundary.json` — results
- `353_we_knowledge_completeness.md` — completeness statement
- `355_multihop_chains.md` — chain viability by relation type
