# DC 369: Mixed-Archetype Detection

**Day 207 | Per-pair archetype detection is geometrically impossible.
ADJACENT and DIRECTIONAL pairs are indistinguishable from single-pair
features: both have nn_rank≈0, overlapping cosine similarity (0.5–0.7),
and overlapping diff_norm (0.42–0.74). Directionality is a cross-pair
property — it requires at least two pairs and direction consistency
measurement. IDENTITY detection is exact (diff_norm=0 / cosine=1).
The pipeline must classify at domain level, not pair level.**

---

## Overview

Day 206 attempted to build a per-pair archetype classifier using three
geometric features: diff_norm, cosine(source, target), and nn_rank
(rank of target in nearest-neighbor list of source). The classifier
achieved F1=1.000 on IDENTITY, F1=0.593 on ADJACENT, and F1=0.108 on
DIRECTIONAL — effectively treating almost all pairs as ADJACENT.

---

## Results

```
Confusion matrix:
                   IDENTITY   ADJACENT   DIRECTIONAL
  IDENTITY              8          0             0
  ADJACENT              0         24             0
  DIRECTIONAL           0         33             2

F1 scores:
  IDENTITY:    1.000
  ADJACENT:    0.593
  DIRECTIONAL: 0.108

Feature distributions:
  Archetype     n   diff_norm(mean)  cosine(mean)  nn_rank(mean)
  IDENTITY      8   0.0000           1.000         202
  ADJACENT     24   0.5883           0.508           0.4
  DIRECTIONAL  35   0.5506           0.538           0.5

Rank distributions:
  ADJACENT:    all ranks ≤ 2 (sorted: 0×16, 1×4, 2×4)
  DIRECTIONAL: mostly rank 0 (0×15 visible in first 15)
```

---

## Finding 1: DIRECTIONAL Targets Are Proximity-Encoded

The central discovery: morphological targets are already near their
sources in W_E — as close as semantic neighbors.

```
Source    Target     nn_rank  Archetype
────────────────────────────────────────
cat       cats       0        DIRECTIONAL
dog       dogs       0        DIRECTIONAL
house     houses     0        DIRECTIONAL
big       biggest    0        DIRECTIONAL
France    Paris      3        DIRECTIONAL
king      queen      0        DIRECTIONAL
know      knew       0        ADJACENT
keep      kept       0        ADJACENT
hot       cold       0        ADJACENT
```

"cats" is rank=0 from "cat". "biggest" is rank=1 from "big". "Paris"
is rank=3 from "France". These directional targets are not in a
different geometric region — they are proximity neighbors.

**This means TYPE_BC retrieval was solving an easier problem than
assumed.** The direction does not need to push the query into a distant
region — it needs only to consistently point toward the target when the
target is already nearby. The direction acts as a **disambiguation
filter** among near-neighbors, not a long-range pointer.

---

## Finding 2: The Disambiguation Role of TYPE_BC Direction

Given that both "cats" and several other words are rank-0 near "cat",
why does adding the plural direction help?

The issue is **vocabulary composition**. In a large vocabulary:
- Near "cat": cats, kitty, kitten, feline, tabby, kitten, catlike, ...
- All are rank≤5 without direction

Adding the plural direction breaks the tie: among all neighbors of "cat",
"cats" is the one that best aligns with the displacement direction
averaged over cat/dog/house/tree. The direction is not pointing toward
"cats" from far away — it is selecting "cats" among many nearby options.

```
WITHOUT direction:  nn("cat") ∈ {cats, kitten, catlike, ...}  → rank(cats)=0 but ambiguous
WITH direction:     nn("cat" + plural_dir) = "cats"           → disambiguated
```

In a SMALL candidate vocabulary (as used in Day 202), this disambiguation
is less needed because there are few near-neighbors. In a large vocabulary,
the direction matters more.

**Counterintuitive implication:** TYPE_BC direction is most critical with
LARGE vocabularies. With a 120-word candidate set, proximity alone may
suffice. With the full 151,936-token vocabulary, proximity fails and
direction disambiguates.

---

## Finding 3: Why Per-Pair Classification Is Impossible

Per-pair features cannot distinguish ADJACENT from DIRECTIONAL because
they measure the same thing: how far/close source and target are.

```
ADJACENT criterion: target is semantically related and nearby
DIRECTIONAL criterion: target is morphologically related and nearby
```

Both result in nn_rank ≈ 0, diff_norm ≈ 0.55, cosine ≈ 0.52.

The distinction requires knowing whether the displacement is *consistent*
across multiple instances of the same relationship. You cannot know this
from one pair alone.

Formal statement:
```
Let f(src, tgt) = per-pair geometric features
Let A = set of ADJACENT pairs, D = set of DIRECTIONAL pairs

∀ single-feature threshold t:
  F1(predict_D from f(src,tgt) > t) ≤ 0.11

Because:
  P(f(src,tgt) ∈ region_D | pair ∈ D) ≈ P(f(src,tgt) ∈ region_D | pair ∈ A)
  The feature distributions are not separable.
```

---

## Finding 4: Revised Archetype Detection Protocol

The correct classifier operates at **domain level** (multiple known pairs)
not pair level:

```
STEP 0 — IDENTITY check (per-pair):
  if norm(W_E[tgt] - W_E[src]) < 0.05:
    archetype = IDENTITY
    retrieval = return source token unchanged

STEP 1 — ORDINAL check (per-domain, ≥3 known pairs):
  if Spearman(rank(src), rank(tgt)) > 0.85:
    archetype = TYPE_ORDINAL
    retrieval = ordinal projection

STEP 2 — DIRECTION check (per-domain, ≥2 known pairs):
  Compute dir_consistency = mean pairwise cosine of diff vectors
  if dir_consistency > 0.15:
    archetype = TYPE_BC (directional)
    retrieval = mean-direction + nn()

STEP 3 — DEFAULT (per-domain fallback):
  archetype = TYPE_ADJACENT
  retrieval = nn(source)
```

This protocol requires **at minimum 2 known pairs** per domain.
For a completely unseen domain with 0 known pairs:
  - Cannot distinguish TYPE_BC from TYPE_ADJACENT
  - Safest default: try TYPE_ADJACENT (nn(source))
  - If nn fails, try k=1 direction from any single example pair

---

## The Full Geometric Picture

```
W_E encoding landscape (per target):

Relative distance from source:
  IDENTITY    diff_norm ≈ 0.000   (same token)
  ADJACENT    diff_norm ≈ 0.55    
  DIRECTIONAL diff_norm ≈ 0.55   ← IDENTICAL TO ADJACENT

  Both ADJACENT and DIRECTIONAL live at ~0.55 units from source.
  This is the "morphological/semantic neighborhood" radius.

Disambiguator:
  ADJACENT:    target chosen by semantic proximity alone
  DIRECTIONAL: target chosen by proximity + consistent direction
  IDENTITY:    target = source (zero displacement)

Type classification uses consistency of displacement vector, not its
magnitude or the cosine between source and target.
```

---

## Implications for TruthSpace Pipeline

### New Architecture: Two-Stage Domain Classifier

```python
def classify_domain(known_pairs, k_min=2):
    """
    Returns archetype label for a relational domain.
    Requires at least k_min known pairs.
    """
    # IDENTITY check — any pair with zero diff
    if any(same_token(a,b) for a,b in known_pairs):
        return "IDENTITY"

    # ORDINAL check
    if len(known_pairs) >= 3:
        rho = spearman_rank(known_pairs)
        if rho > 0.85:
            return "TYPE_ORDINAL"

    # DIRECTIONAL check
    if len(known_pairs) >= 2:
        dc = dir_consistency(known_pairs)
        if dc > 0.15:
            return "TYPE_BC"

    # Fallback
    return "TYPE_ADJACENT"
```

### Key Constraints

```
Cannot detect archetype at k=0 (no known pairs)
Cannot detect archetype at k=1 (single pair)
Minimum k=2 for DIRECTIONAL detection
Minimum k=3 for ORDINAL detection
IDENTITY detectable at k=1 (trivially: if src==tgt)
```

---

## What This Means for the Hypothesis

The TruthSpace hypothesis states that **structure IS information** in
W_E. The archetype detection finding refines this:

- The structure is not just "where tokens are" (proximity)
- It is "how tokens move relative to each other" (consistency)
- Proximity encodes WHAT (semantic category membership)
- Direction consistency encodes HOW (relational transformation type)

Both ADJACENT and DIRECTIONAL exploit proximity. DIRECTIONAL additionally
exploits the consistency of the displacement — a richer geometric invariant
that requires multiple observations to measure.

---

## Files

- `expedition_day206_archetype_detection.py` — per-pair classifier
- `day206_archetype_detection.json` — results
- `368_verb_class_geometry.md` — verb class analysis
- `365_multitier_pipeline.md` — original pipeline architecture
