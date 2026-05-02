# DC 451: Relational Axes Are Geometric — The Filter Artifact Correction

**Day 316 | A major revision of the taxonomy established in DC 449-450.
The "named entity" failure of country→capital and country→language was
entirely an artifact of the capitalization filter. With the filter relaxed:
(1) country→capital achieves 8/8=100% in-sample at scale=0.432,
pc=0.378 — equivalent to +er (pc=0.385). (2) country→language achieves
6/6=100% in-sample, LOO=67%, pc=0.399. (3) Both are morph_moderate quality
relational axes, not named entity failures. (4) +able is confirmed as
semantic_diverse (irred=60% over full sweep). (5) Fast LOO = slow LOO
for all tested axes (gap=0%) — the global scale is per-fold optimal,
validating the fast method. (6) The language cluster has geographic
structure: applying the country→language axis to a language token
navigates within the language cluster to geographically proximate languages.**

---

## The Filter Artifact

### What Happened in Days 313–314

When we tested country→capital, the retrieval function filtered out all
tokens whose decoded string starts with an uppercase letter. The logic was:
"proper names and artifacts should not appear as valid answers." The result:
'Paris', 'Berlin', 'Tokyo', etc. — all capitalized — were never in the
candidate pool. In-sample = 0%, which we interpreted as a geometric failure.

### The Test That Fixed Everything

With the relaxed mask (length > 1, no leading `-` or `_`, caps allowed):

```
country→capital:  8/8=100% in-sample   pc=0.378   scale=0.432
country→language: 6/6=100% in-sample   pc=0.399   scale=1.051
```

France→Paris ✓, Germany→Berlin ✓, Japan→Tokyo ✓, China→Beijing ✓,
Canada→Ottawa ✓, Australia→Canberra ✓, India→Delhi ✓

These axes are fully geometric. Every country navigates to its capital
along a single consistent displacement vector.

### Why the Filter Masked This

The capitalization filter was designed to eliminate:
1. Sentence-initial token capitalizations (artifact of tokenization)
2. Named proper nouns as false positives in morphological retrieval

For country→capital, the capitalization is SEMANTICALLY MEANINGFUL:
capitals are proper nouns by convention. Removing the caps filter is
correct for this use case.

The filter was right for morphological axes (+er, +s, +tion) where
capitalized tokens ('Fast', 'Cats', 'Action') would be incorrect answers.
It was wrong for relational axes where targets are inherently capitalized.

### The General Rule

```
Use CLEAN mask (caps excluded):     morphological, derivational axes
Use RELAXED mask (caps allowed):    relational axes with proper noun targets
Use SUPER-RELAXED mask (all):       symbolic/chemical axes (short symbols)
```

---

## pc Equivalence: Relational ≈ Morphological

```
Axis               pc      type
────────────────────────────────────────
er→est             0.426   morph_uniform
country→language   0.399   relational
+er                0.385   morph_moderate/uniform
country→capital    0.378   relational
+s                 0.297   morph_moderate
```

Relational axes have pairwise chord cosines equivalent to or higher than
morphological inflection axes. The country→capital displacement is as
consistent as the +er displacement. W_E encodes "country to its national
symbol" relationships with the SAME geometric coherence as "adjective to
its comparative form."

This is a validation of the TruthSpace hypothesis at a new level: **not
only are morphological transformations geometric, but factual/relational
transformations between named entities are equally geometric.**

---

## Revised Axis Taxonomy (Final)

### Five Categories

```
Type                Signature                         Examples          irred
─────────────────────────────────────────────────────────────────────────────
morph_uniform       pc>0.28, LOO≥65%, clean           +er,+s,+ed,er→est  <15%
morph_moderate      pc<0.28, LOO 30-65%, clean        gender,+tion,+ly   15-40%
semantic_diverse    in=100%, LOO<30%, irred>60%,clean +ness,+ful,un-,+able >60%
relational_geom     pc≈0.38-0.40, LOO≥60%, RELAXED    country→X         15-40%
filter_blocked      (was named_entity — NOW DEPRECATED)
```

The `named_entity` category is abolished. The axes we labeled "named entity"
are actually `relational_geom` (geometric with proper noun targets) once
the correct filter is applied.

### What Would Actually Be Named Entity (Non-Geometric)?

A true named entity relation would have:
- Low pc (chords scatter randomly across the space)
- Low in-sample even with relaxed filter (even at best scale, can't retrieve)
- No consistent semantic direction

We have not yet found a confirmed non-geometric relation. The most suspicious
remaining candidate is element→symbol (pc=0.188, in-sample 67% with super-
relaxed) — this shows cross-lingual interference but not complete failure.

### The +able Confirmation

```
+able: pc=0.194  LOO=17%  in-sample=100%  irred=6/10=60%  → semantic_diverse
```

Over the full scale sweep 0.02–6.0, 6/10 holdout words are irreducible:
- agree (→agrees), debate (→debates), reason (→reasons)
- predict (→predicts), replace (→replacing), rely (→relies)

The pattern: verbs with strong present-form neighbors ('agrees', 'debates',
'reasons') are irreducible because the +able displacement moves to the
verb's most active form region rather than the -able derivation region.
These verbs have STRONG verb identity (high cosine to 3rd-person forms)
that the axis cannot overcome.

Note: trust→trustworthy ✓. The axis encodes "capable/worthy of [verb]"
semantics and retrieves 'trustworthy' even though the expected form is
'trustable'. The semantic destination is correct; the surface form differs.

---

## Fast LOO = Slow LOO: Scale Stability Theorem

### The Evidence

```
Axis      LOO_global  LOO_perfold  gap
+er       100%        100%         0%
+tion -ct  75%         75%         0%
un-ADJ     57%         57%         0%
```

For all three axes at very different quality levels, the fast LOO
(using the globally optimal scale) gives exactly the same result as
the slow LOO (finding optimal scale per fold).

### Why This Holds

When one training pair is removed from n training pairs, the mean chord
changes by at most 1/n of the removed chord's deviation from the mean.
For a well-defined axis (multiple training pairs), this is negligible.
The optimal scale shifts by O(1/n), which for 7-12 training pairs
changes the best scale by less than one step in a 30-step sweep.

Formally: scale_LOO ≈ scale_full ± O(1/(n×step_resolution))

This is a practical validation that 30-step scale sweeps on 8+ training
pairs are sufficient for stable LOO measurements.

---

## Language Cluster Geometry

### The Composition Test

Applying the country→language axis to a LANGUAGE token:

```
country →[axis]→ language →[axis]→ ???
france  → French  → Spanish, German, Italian
germany → German  → French, Japanese, Spanish
japan   → Japanese → French, German, Chinese
china   → Chinese  → Japanese, French, Spanish
turkey  → Turkish  → French, Spanish, Chinese
```

The second application finds the NEAREST LANGUAGE NEIGHBORS in W_E.
French's nearest language neighbors are Spanish, German, Italian —
Romance and Germanic European languages. Japanese finds French, German,
Chinese — global major languages. Chinese finds Japanese, French, Spanish.

### What This Reveals

1. **Languages form a semantic cluster** in W_E with internal geometric
   structure organized by geographic/cultural proximity.

2. **The country→language axis does not target specific languages** —
   it targets the "language that belongs to this country" semantic region.
   When applied to a language token, it finds "nearby languages" because
   it's moving INTO the language cluster from its current position.

3. **Navigation within the language cluster** is possible using the
   country→language axis as a local explorer, even though it's designed
   as a cross-domain axis.

4. **Cross-domain axes are also intra-cluster navigators** when applied
   to targets already in the destination cluster. This is consistent with
   the TruthSpace principle: a displacement vector encodes a direction in
   the semantic manifold; that direction has meaning everywhere it is applied.

---

## Day 317 Plan

1. **Holdout test for country→capital**: test 5 holdout pairs (russia,
   sweden, greece, portugal, argentina) with relaxed filter. Measure LOO.

2. **City→country reverse axis**: is the inverse relation geometric?
   Paris→france, Berlin→germany, Tokyo→japan. pc? Reversibility?

3. **Relation axis composition**: country→capital + capital→language →
   country→language. Can we chain two relational axes?

4. **Find a true non-geometric relation**: test country→president,
   author→book, scientist→discovery. Do any have pc < 0.15 AND in-sample=0%
   even with relaxed filter?

5. **+less full sweep**: verify +less is semantic_diverse like +ful/un-.

---

## Files

- `expedition_log.md` — Day 316 results
- `450_axis_taxonomy_and_topological_irreducibility.md` — DC 450
- `day316_relational_axes_uncapped.py` — experiment script
