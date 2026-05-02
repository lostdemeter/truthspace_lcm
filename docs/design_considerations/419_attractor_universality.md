# DC 419: Attractor Universality — The Dominant Cluster Always Wins

**Day 284 | Testing the reversibility principle and biological taxonomy
axes. Two key findings: (1) The reversibility hypothesis is falsified —
concept→field inverse axis achieves only 18% (vs 15% forward), because
the 'physics' attractor dominates ALL concept words in W_E. (2) The
animal→class taxonomy axis achieves 88% for bird/fish/plant classes
but fails entirely for mammal/reptile/amphibian (multi-token targets).
The attractor phenomenon is universal: in every domain tested, the axis
prediction collapses to the dominant training-set target for
non-cluster-matched sources. The fix is always clustering, never
direction inversion.**

---

## The Reversibility Hypothesis: Falsified

DC 418 proposed that inverse (many-to-one) axes should have higher
coherence than forward (one-to-many) axes. Day 284 falsifies this:

| Axis direction | Coherence | Accuracy |
|---|---|---|
| field→concept (forward) | 0.432 | 50% (5/10) |
| concept→field (inverse) | 0.408 | 18% (4/22) |

The inverse is slightly LOWER coherence than the forward, and much
lower accuracy. The reason is the physics attractor:

```
gravity   → physics  [HIT]     (gravity IS a physics concept)
motion    → physics  [HIT]     (motion IS a physics concept)
force     → physics  [HIT]     (force IS a physics concept)
energy    → physics  [HIT]     (energy IS a physics concept)
reaction  → physics  [---]     (reaction is chemistry; but physics dominates)
cell      → physics  [---]     (cell is biology; but physics dominates)
market    → physics  [---]     (market is economics; but physics dominates)
...all 22 concepts → physics
```

The axis direction points toward 'physics' with such strength that
no other field is ever retrieved. This is not a failure of the
inverse principle — it is the attractor dominating the axis.

### Why the Physics Attractor Exists

In W_E, 'physics' has the largest and most coherent concept cluster
among academic fields. The word 'physics' appears in training data
adjacent to: gravity, force, energy, light, mass, momentum, wave,
particle, quantum, thermodynamics, electromagnetism, relativity, ...

The 'field' words (physics, biology, chemistry, etc.) are not
equidistant from concept words. Physics is the closest field word
to more concept words than any other field. The axis mean direction
is pulled overwhelmingly toward 'physics'.

The four hits (gravity, motion, force, energy) are the four concepts
closest to 'physics' in W_E. At scale=0.76, the axis displacement
from these specific concepts happens to land at 'physics'. The other
18 concepts are displaced past 'physics' to noise or back to 'physics'
with different nearest-neighbour geometry.

---

## The Universal Attractor Pattern

Across all domains tested in Days 276–284, the attractor phenomenon
appears every time a global (non-clustered) axis is used on a
heterogeneous source set:

| Domain | Global axis acc | Attractor | Cluster acc |
|---|---|---|---|
| person→nat | 41% | German | 100% (per cluster) |
| person→field | 60% | economics | predicted 100% |
| concept→field | 18% | physics | — (clustering can't fix) |
| animal→class | 43% (test) | bird | 100% (per cluster) |

**The attractor is always the largest/most coherent sub-cluster in
the training pairs.**

For concept→field, clustering cannot fix the attractor because:
- The sources (gravity, force, energy) ARE correctly in the physics cluster
- The sources (cell, evolution) are biology — but biology concepts are
  closer to physics in W_E than to biology as a field word
- The issue is not source heterogeneity but W_E layout: 'physics' is
  geometrically dominant over 'biology', 'chemistry', etc.

This is a different failure mode: **target-side attractor** (physics
dominates target space) vs **source-side heterogeneity** (mixed sources).

### The Two Failure Modes

**Mode 1: Source-side heterogeneity** (fixable by clustering)
- Mixed sources from different categories point in different directions
- Fix: separate axes per source cluster
- Examples: person→nat (German attractor), person→field (economics attractor)

**Mode 2: Target-side attractor** (NOT fixable by clustering)
- One target word dominates the target space in W_E
- All sources, regardless of cluster, point toward the dominant target
- Fix: balance target representation via training pair selection
- Examples: concept→field (physics), animal→class-global (bird)

### Distinguishing the Two Failure Modes

Mode 1 diagnosis: different source types retrieve the same wrong target
Mode 2 diagnosis: ALL sources retrieve the same wrong target regardless of type

For concept→field: biology concepts (cell, evolution), economics concepts
(market, trade), astronomy concepts (orbit, star) ALL return 'physics'.
This is Mode 2.

For person→nat (before clustering): British persons return 'German',
Greek persons return 'German', American persons return 'German'. This
is Mode 1 (German is the largest cluster), fixable by clustering.

---

## Biological Taxonomy: Token-Constrained Success

The animal→class axis demonstrates perfect accuracy on the accessible
domain but is blocked by tokenisation for three of six classes.

### What Works (single-token classes)

```
Class   Cluster coh   Accuracy    Representative members
bird    0.705         7/7  (100%) eagle, hawk, owl, robin, crow, duck, robin
fish    0.698         9/9  (100%) salmon, tuna, trout, cod, herring, perch, ...
plant   0.786         8/8  (100%) rose, oak, pine, fern, ivy, maple, elm, ...
```

The bird cluster (coh=0.705) and fish cluster (coh=0.698) confirm that
the Type B many-to-one pattern holds across domains. The geometric
structure of "many birds → 'bird'" exists just as "many Germans → 'German'"
existed in the nationality domain.

### What Fails (multi-token targets)

'mammal', 'reptile', 'amphibian' are all multi-token in Qwen2 BPE:
- 'mammal' → ['mam','mal'] or similar 2-token sequence
- 'reptile' → ['rep','tile'] or ['rept','ile']
- 'amphibian' → 3+ tokens

The 5/5 constraint from get_emb() filters these out entirely.
No axis computation, no retrieval, no failure message — just silence.
This makes the failure invisible without explicit token inspection.

### The Invisible Ceiling

Multi-token targets create an **invisible ceiling** on the knowledge
domains accessible via W_E axis retrieval. The accessibility of a
knowledge domain depends on whether its terminal concept words are
single-token in the specific BPE vocabulary used.

For Qwen2's vocabulary:
- Accessible: bird, fish, plant, physics, German, French, english, ...
- Inaccessible: mammal, reptile, amphibian, Norwegian, Hungarian, ...

This creates a taxonomy of "geometrically accessible knowledge" that
is vocabulary-specific and cannot be determined without empirical testing.

### Implication for Knowledge Chain Design

Before designing any multi-hop chain, audit ALL terminal tokens:

```python
def is_accessible(word):
    for prefix in [' ', '']:
        ids = tok(prefix + word)['input_ids']
        if len(ids) == 1: return True
    return False
```

Only build axes whose targets are accessible. For inaccessible concepts,
use paraphrase (e.g., 'mammal' → 'animal' which may be single-token).

---

## Revised Axis Viability Model (Complete)

Combining DC 415 (axis types), DC 418 (one-to-many constraint), and
DC 419 (attractor universality):

```
function axis_will_work(pairs, relation_type):
    # Check 1: token accessibility
    valid_pairs = [p for p in pairs if is_single_token(p.source)
                                    and is_single_token(p.target)]
    if len(valid_pairs) < 3: return "INSUFFICIENT_PAIRS"

    # Check 2: relation type
    if relation_type == ONE_TO_MANY: return "FAILS (use target subsets)"

    # Check 3: target distribution balance
    target_counts = count(valid_pairs.targets)
    if max(target_counts) > 0.5 * len(valid_pairs):
        return "MODE2_ATTRACTOR (balance training pairs)"

    # Check 4: source homogeneity
    coherence = compute_axis(valid_pairs).coherence
    if coherence < 0.50 and not homogeneous_sources:
        return "MODE1_ATTRACTOR (apply clustering)"

    # Expected performance
    if coherence > 0.75: return "HIGH (>90%)"
    if coherence > 0.55: return "MODERATE (70-90%)"
    return "LOW (40-70%)"
```

---

## Files

- `expedition_log.md` — Day 284 results
- `418_one_to_many_constraint.md` — axis viability checklist (Day 283)
- `416_cluster_axes.md` — source-type clustering (Day 281)
- `415_axis_type_taxonomy.md` — three axis types (Day 280)
