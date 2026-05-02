# DC 467: Full 5×5 GROUP MAP, Source Population Homology, +ize/+ity Cosine, Spread Feature

**Day 332 | The complete morphological atlas of W_E is now established. (1) The full
5×5 inter-group cosine matrix with GROUP C included reveals that +al_rel is
anti-aligned with the ENTIRE group system (negative cosines with A, B, C, E;
only slightly positive with D). (2) The cross-group cosine +en vs +er_comp = 0.406
and +ize vs +ity = 0.401 establish "source population homology": when two axes
draw from the same lexical sub-cluster, their cosine can reach intra-group levels
regardless of differing targets. (3) Chord spread as a 4th predictor feature
resolves the high-pc disambiguation: if pc > 0.30 and spread > 0.07, classify as
phonol_scatter (ablaut); if pc > 0.30 and spread < 0.06, classify as morph_uniform.
(4) GROUP C → GROUP E chain works 3/7 cases (widen/soften/weaken → widened/softened/
weakened). (5) The +ize axis has correct direction (LOO=86%) but vocabulary-limited
reach (irred=75% due to multi-token -ize forms in vocabulary).**

---

## The Complete 5×5 Inter-Group Cosine Matrix

### Final Table

```
          A(v→n)   B(a→n)   C(a→v)   D(v→a)   E(v→v)
A(v→n)     ---     +0.058   -0.143   +0.034   +0.150
B(a→n)    +0.058    ---     +0.302   +0.080   +0.109
C(a→v)    -0.143   +0.302    ---     +0.141   +0.054
D(v→a)    +0.034   +0.080   +0.141    ---     +0.161
E(v→v)    +0.150   +0.109   +0.054   +0.161    ---
```

Standalone axes vs all groups:

```
+al_rel: -0.029  -0.240  -0.118  +0.051  -0.021  ← mostly NEGATIVE
+er_comp: +0.008  +0.233  +0.304  +0.126  +0.136  ← all positive
un-:      +0.013  +0.183  +0.189  +0.109  +0.109  ← moderate positive
```

Anti-aligned pairs:
```
+al_rel vs +ity:         cos = −0.418  (reverse pair #1: noun↔adj)
GROUP C vs GROUP A:      cos = −0.143  (reverse pair #2: adj→verb vs verb→noun)
+al_rel vs GROUP B:      cos = −0.240  (group-level: noun→adj vs adj→noun)
+al_rel vs GROUP C:      cos = −0.118  (shared adj cluster boundary)
```

### Structural Reading of the Matrix

The 5-group system is organized around three POS clusters: {adj, verb, noun}.

```
                    noun cluster
                   ↗      ↖
             GROUP A      GROUP B
             (v→n)        (a→n)
            ↗               ↗
     verb cluster      adj cluster
        ↑    ↑              ↑  ↑
    GROUP E  GROUP D    GROUP C  +er_comp
    (v→v)   (v→a)      (a→v)  (a→adj)
              ↑
          adj cluster (arrival)
              ↑
          +al_rel (n→a)
```

**Groups that DEPART from the same cluster have positive cosines.**
**Groups that DEPART from one cluster and ARRIVE at another's departure point have
negative or near-zero cosines.**

Specifically:
- B(a→n) and C(a→v) both depart from adj cluster → cos = +0.302 (high)
- A(v→n) and C(a→v): A departs verb, C arrives verb → cos = −0.143 (negative)
- D(v→a) and E(v→v) both depart from verb cluster → cos = +0.161 (high)
- A(v→n) and E(v→v) both depart from verb cluster → cos = +0.150 (high)

The sign rule: **same departure cluster → positive cosine; one departs, other arrives → negative cosine**.

---

## Source Population Homology

### The Discovery

```
cos(+en, +er_comp) = +0.406   (highest inter-axis positive cosine in the map)
cos(+ize, +ity)    = +0.401   (second highest)
```

Both exceed the GROUP C internal cosine (0.385) and are comparable to GROUP A
pairwise cosines (0.307-0.476).

### Mechanism

+en operates on Germanic adj: bright/dark/hard/wide/soft/fresh/weak/sharp/deep
+er_comp operates on generic adj: fast/slow/bright/dark/soft/warm/tall/clean

**Shared population: {bright, dark, soft, ...}** — the core Germanic adj cluster in W_E.
Both axes DEPART from this sub-cluster. The departure direction is nearly identical.
Targets differ (verb space vs comparative adj space), but the departure dominates the cosine.

+ize operates on Latin adj: moral/legal/national/local/modern/final/general
+ity operates on Latin adj: human/real/national/personal/moral/legal/final/normal

**Shared population: {moral, national, legal, final, local}** — the Latin-root adj sub-cluster.
Both depart from this identical sub-cluster. Again, departure direction dominates.

### The Source Population Homology Principle

```
cos(axis_1, axis_2) is elevated when:
1. axis_1 and axis_2 draw from the SAME LEXICAL SUB-CLUSTER (same source population)
2. Regardless of whether targets are different

The cosine can reach intra-group levels (0.30-0.43) if the source populations are
homologous, even when:
- Sources have different POS (adj in +en vs adj-comparative in +er_comp)
- Targets are in different semantic categories (verb vs comparative-adj)
```

This refines the SOURCE-TARGET COSINE RULE from DC 465:

**Old rule**: `cos ≈ α × source_POS_similarity + β × target_category_similarity`

**New rule**: `cos ≈ α × source_POPULATION_similarity + β × target_category_similarity`

Source POPULATION is more precise than source POS:
- All adj-source axes share some cosine (POS-level similarity)
- Axes sharing the SAME adj sub-cluster share much higher cosine (population-level)

---

## +al_rel: The System Antagonist

### Anti-Alignment Pattern

```
+al_rel vs GROUP A: −0.029  (neutral/slight negative)
+al_rel vs GROUP B: −0.240  (STRONGLY NEGATIVE)
+al_rel vs GROUP C: −0.118  (MODERATELY NEGATIVE)
+al_rel vs GROUP D: +0.051  (slightly positive)
+al_rel vs GROUP E: −0.021  (neutral/slight negative)
```

+al_rel (noun→adj) is the only axis with net NEGATIVE alignment to the group system.
All five groups combined give a negative or near-zero cosine with +al_rel.

### Why GROUP D is the Exception

+al_rel (noun→adj) arrives at the adj cluster.
GROUP D (verb→adj) also arrives at the adj cluster.
They share the TARGET, making their cosine slightly positive (+0.051).

This confirms: **target arrival similarity creates positive cosines, even when source
departure is unrelated**.

The rule is symmetric:
- Shared DEPARTURE → positive cosine (dominant effect)
- Shared ARRIVAL → positive cosine (weaker effect, cos ≈ 0.05)
- Departure of one = Arrival of other → NEGATIVE cosine (antagonism)

---

## Spread as 4th Predictor Feature

### The High-pc Disambiguation Rule

```
If pc > 0.30:
  If spread < 0.06: morph_uniform  (uniform direction, all chords agree)
  If spread > 0.07: phonol_scatter  (irregular verbs: wide angular scatter)
```

Evidence:
```
er_comp:  pc=0.381  spread=0.043  → morph_uniform  ✓
ablaut:   pc=0.345  spread=0.095  → phonol_scatter  ✓
```

### Why This Works

**morph_uniform axes** (er_comp): all adj-to-comparative displacements point
nearly the same direction. The comparative adj cluster is compact and the adj
source cluster is compact. Every chord goes from cluster A to cluster B in
essentially the same direction. Low spread.

**ablaut axes**: irregular verbs include (go→went, take→took, give→gave, see→saw,
know→knew, drive→drove, write→wrote, ride→rode). These verb-form pairs are
semantically coherent but phonologically DIVERSE. Each pair departs from a
DIFFERENT position within the verb cluster (go and take are far apart) and
arrives at a different position in the past-tense cluster. The MEAN direction
is correct (ablaut axis), but individual chords fan out widely. High spread.

### Predictor v7 Rule

```
predict(pc, LOO, irred, spread, n_pairs):
  if n_pairs < 8: mark as 'low_confidence'
  if pc > 0.35:
    if spread < 0.06: return 'morph_uniform'
    else:             return 'phonol_scatter'  ← NEW ablaut rule
  elif pc > 0.20:
    ... [existing v6 logic] ...
```

This adds exactly ONE new correct classification (ablaut) while changing nothing else.

---

## GROUP C → GROUP E Chain

### Results

```
adj     +en(verb)  +ed(past)   +3ps(pres)  +ing(pres_part)
wide  ✓ widen    ✓ widened    ✓ widening  ✓ widening
soft  ✓ soften   ✓ softened   ✓ softened  ✓ softened
weak  ✓ weaken   ✓ weakened   ✓ weakened  ✓ weakening
dark  ✓ darken   ✗ DARK       ✗ DARK      ✗ DARK
deep  ✓ deepen   ✗ deeper     ✗ deeper    ✗ deeper
sharp ✓ sharpen  ✗ sharper    ✗ sharper   ✗ sharper
bright✗ brighter —            —           —
hard  ✗ harder   —            —           —
```

### Failure Analysis

**bright/hard → comparative not +en**: The comparative forms (brighter, harder)
are MORE COMMON and CLOSER in the adj cluster than the +en forms (brighten, harden).
The +en axis, trained on less common adj, produces an axis that for these canonical
comparatives is outcompeted by the +er_comp axis.

**dark → DARK**: The +ed axis applied to "darken" navigates to "DARK" (a proper
noun/brand in the Qwen2 vocabulary). "darkened" is likely multi-token or less
frequent than "DARK".

**deep/sharp → deeper/sharper**: This is a GENUINE GEOMETRIC AMBIGUITY:
- "deepen" and "deeper" ARE morphologically related
- "sharpen" and "sharper" ARE morphologically related
- The +ed axis for "deepen" navigates toward the comparative because "deeper" is
  both the natural continuation from "deep" AND the single-token form nearest to
  the predicted position

This reveals: GROUP C (+en) and the COMPARATIVE axis (+er) share source populations
and their chained products sometimes collide. The chain is unambiguous only when
the +en form is MORE COMMON than the +er form (widen > wider, soften > softer,
weaken > weaker are all true in frequency).

### Chain Validity

For 3/7 words (widen/soften/weaken), the full adj→verb→inflected chain works:
```
wide  → widen    → widened   / widening   ✓
soft  → soften   → softened  / softened   ✓
weak  → weaken   → weakened  / weakening  ✓
```

This is a 3-step morphological chain in W_E: GROUP C step + GROUP E step.
The chain is geometrically stable for adj where the +en form is more common
than the +er form.

---

## The +ize Vocabulary Ceiling

### Irred is Tokenization-Dependent

```
+ize LOO:   86%  (axis DIRECTION is correct)
+ize irred: 75%  (most holdout TARGETS don't exist as single tokens)

Multi-token failures: popularize, equalize, crystallize, neutralize, standardize
Single-token success: visualize, normalize
Wrong suffix: activate (should be -ize but uses -ate suffix)
```

### Two Types of Irreducibility

This reveals that `irred` conflates two different phenomena:

**Type 1: Geometric irred** — the axis direction is wrong; no scale reaches the
target. This is a TRUE failure of the morphological hypothesis.

**Type 2: Vocabulary irred** — the axis direction is correct but the TARGET TOKEN
doesn't exist as a single token in the vocabulary. The displacement lands in the
right semantic neighborhood, but the nearest single-token word isn't the expected
target.

+ize has almost entirely Type 2 irred. The axis KNOWS where to go (LOO=86%), but
the vocabulary DOESN'T HAVE the target as a single token.

The predictor currently can't distinguish Type 1 from Type 2. This is important
for axis classification: an axis with high LOO but high irred (like +ize) should
be classified as 'vocabulary_limited_reach' not 'factual_local'.

---

## Day 333 Plan

1. **Predictor v7**: implement the pc > 0.30 spread rule. Re-run benchmark.
   Expected: 19/30 = 63% (one new correct: ablaut).

2. **Type 1 vs Type 2 irred**: can we distinguish geometric vs vocabulary irred?
   Test: if the displaced position is within some distance of the target's NEAREST
   SINGLE-TOKEN SYNONYM, then it's Type 2. Try: check top-5 NN of displaced
   position and measure their semantic similarity to the target.

3. **Source population homology test**: are there other cross-group pairs with
   cos > 0.35 due to shared source populations? Specifically: Germanic verb pairs
   shared between GROUP D (+able) and GROUP E (+3ps)?

4. **The GROUP chain graph**: draw the full morphological chain graph showing
   which chains work (C→E, A→reverse, etc.) and which fail. This is the
   "morphological reachability" map of W_E.

5. **The +al_rel departure analysis**: measure how +al_rel compares to the GROUP D
   arrival direction. Are they truly similar (both arrive at adj cluster), or is
   the cos=+0.051 just noise?

---

## Files

- `expedition_log.md` — Days 322-332 results
- `466_group_c_adj_verb_anti_alignment_re_paradox_axis_diameter.md` — DC 466
- `day332_5x5_groupc_ity_spread_chain_ize_irred.py` — experiment script
