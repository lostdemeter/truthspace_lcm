# DC 354: Multi-Relation Composition in W_E

**Day 175 | Directions don't compose additively; stronger dominates; multi-hop chains work**

---

## Overview

Day 174 tests whether two relational directions can be simultaneously applied
to a single entity, and whether geometric multi-hop reasoning is possible in W_E.

**Core findings:**

> **1. Directions from the same entity class are NOT orthogonal (cos=0.422 for
> capital vs language). DC 349's orthogonality claim holds only across different
> entity classes.**
>
> **2. Additive composition fails: the stronger/more salient direction dominates.
> Language direction (rank-1.0) drowns out capital direction (rank 2.6→7.3).**
>
> **3. Multi-hop chains work when each step is geometrically correct:
> France→Paris→French, Germany→Berlin→German, Japan→Tokyo→Japanese.**

---

## Direction Orthogonality Revisited

```
cos(capital, language) = 0.422   ← correlated (same entity class: countries)
cos(capital, gender)   = 0.006   ← orthogonal (different entity classes)
cos(capital, metal)    = 0.037   ← orthogonal
cos(language, metal)   = 0.023   ← orthogonal
cos(gender, language)  = 0.010   ← orthogonal
cos(gender, metal)    = -0.015   ← orthogonal
```

**Revised orthogonality rule:** Directions between different entity classes
are orthogonal (cos ≈ 0). Directions within the same entity class can be
correlated — capital and language are both "country→X" relations and share
a component reflecting the country's overall semantic position.

The capital direction and language direction both point "away from country
embeddings" in the same general region of the space. Their partial correlation
(0.422) reflects the shared "country context" component.

---

## Additive Composition: The Dominance Effect

```
Mean rank across 7 countries:
  capital direction alone:     rank = 2.6  (good)
  language direction alone:    rank = 1.0  (perfect)
  capital + language (dual):   cap rank = 7.3, lang rank = 1.0
```

**Mechanism:** When two directions are summed and the sum is applied to
an entity, the nearest neighbor is determined by the DOMINANT direction.
Since language direction consistently produces rank-1 results, it represents
a stronger geometric signal in the embedding space. Adding the capital
direction disturbs this but not enough to overcome it.

The query point `France + cap_dir + lang_dir` is closer to "French" than
to "Paris" because:
1. The language direction has larger effective magnitude in the relevant subspace
2. The two directions are partially correlated (cos=0.422), amplifying language
3. "French" is a stronger nearest-neighbor attractor than "Paris" for France contexts

**Commutativity:** cap+lang = lang+cap (vector addition is commutative), confirmed.

**Practical implication:** You cannot retrieve two attributes simultaneously
via additive direction composition. To get both capital AND language, you must
query each direction separately.

---

## Multi-Hop Chains

The key finding: **applying direction steps sequentially produces correct chains**.

```
France  + cap_dir → Paris   [correct: rank 1]
Paris   + lang_dir → French  [correct: rank 1]
France → Paris → French  ✓ (two-hop, both correct)

Germany + cap_dir → Berlin  [correct: rank 1]
Berlin  + lang_dir → German  [correct: rank 1]
Germany → Berlin → German  ✓

Japan   + cap_dir → Tokyo   [correct: rank 1]
Tokyo   + lang_dir → Japanese [correct: rank 1]
Japan → Tokyo → Japanese  ✓

Italy   + cap_dir → Berlin  [WRONG: Rome expected]
Berlin  + lang_dir → German  [chain breaks]
Italy → Berlin → German  ✗
```

**Chain accuracy = step accuracy:** Multi-hop chains work exactly when
each individual step is correct. There is no additional geometric noise
introduced by chaining — the error propagation is clean (correct step
leads to correct chain; wrong step propagates the error).

### What Multi-Hop Encodes

The successful chains reveal a deep property of W_E:

> **Capital city embeddings encode their country's language.**
> Paris is embedded near French; Berlin near German; Tokyo near Japanese.
> The language of a capital city's country is encoded in the city's embedding.

This is **relational transitivity in W_E**: Paris inherits France's language
property through co-occurrence in training data (texts about Paris always
describe it as the French capital, in French contexts).

---

## Disambiguation via Direction

Applying `cap_dir` to "Paris" moves it toward **other capital cities**, not
toward a different sense of Paris. This is because:
1. Paris is already a capital city — it is already in the capital city cluster
2. Adding cap_dir moves the query further in the "capital city" direction,
   landing near other capitals (Berlin, London, Tokyo)
3. Disambiguation requires the entity to be ambiguous (in two different clusters)
   and the direction to point from one cluster to its label

Direction-based disambiguation works when:
- The entity is at the boundary between two clusters
- The direction points away from the unwanted cluster's center

It does NOT work when:
- The entity is already deep in the "correct" cluster
- The direction just moves it further into the same cluster

---

## Updated Model: Direction Semantics

From Days 162-174, the following model of W_E direction semantics emerges:

**Directions are cluster-targeting vectors**, not universal shift vectors:

```
entity + direction = a point that is:
  - moved away from the entity's own cluster centroid
  - moved toward the TARGET cluster centroid
  - near the target cluster members
```

**Additive composition fails** because:
- direction_1 moves toward cluster_1 center
- direction_2 moves toward cluster_2 center
- The sum direction_1 + direction_2 points toward neither cluster center
  but toward the centroid of {cluster_1, cluster_2}
- The nearest neighbor at that point is whichever cluster has a stronger
  attractor (larger cluster, or stronger direction magnitude)

**Multi-hop works** because:
- Step 1: entity → cluster_1_member (discrete: snap to nearest neighbor)
- Step 2: cluster_1_member + direction_2 → cluster_2_member
- The snapping at each step prevents error accumulation in continuous space

The "snap to nearest neighbor" is the critical operation. Multi-hop
reasoning in W_E requires:
1. Apply direction
2. Snap to nearest neighbor (discrete step)
3. Apply next direction
4. Snap again

NOT: apply all directions at once, then snap once.

---

## Implications for TruthSpace System Design

### Querying Multiple Attributes
To retrieve multiple attributes of an entity:
```
# WRONG (simultaneous composition):
result = nearest_neighbor(entity + d1 + d2)

# RIGHT (sequential with snapping):
attr1 = nearest_neighbor(entity + d1)
attr2 = nearest_neighbor(entity + d2)
```

### Multi-Hop Reasoning
Multi-hop relational chains are supported geometrically:
```
step1 = nearest_neighbor(entity + relation1_dir)
step2 = nearest_neighbor(step1 + relation2_dir)
# chain accuracy = product of individual step accuracies
```

### Pipeline Architecture
This suggests a TruthSpace retrieval pipeline:

```
Query: "What language is spoken in France's capital?"

Step 1: France + capital_dir → Paris        (Type C, k=8)
Step 2: Paris  + language_dir → French      (Type B or C)

Each step is discrete: snap to nearest neighbor before next step.
```

The W_E geometry supports single-hop and multi-hop relational chains,
but NOT simultaneous multi-attribute retrieval via direction sum.

---

## Connection to Transformer Architecture

The "snap to nearest neighbor" in multi-hop chains corresponds to the
discrete token prediction at each generation step in an LLM. The model
generates one token at a time, then feeds it back as input — this IS
the snap operation. The transformer's auto-regressive generation is
precisely the "apply direction, snap, apply next direction" loop.

This suggests the transformer performs implicit W_E multi-hop chains
during generation, where each generated token is a "snap" step.

---

## Files

- `expedition_day174_multirelation.py` — composition and multi-hop experiments
- `day174_multirelation.json` — results
- `349_direction_orthogonality.md` — prior orthogonality analysis (needs revision)
- `353_we_knowledge_completeness.md` — completeness arc summary
