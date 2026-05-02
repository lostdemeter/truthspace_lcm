# DC 380: Antonymy Is Not Functional — The Degeneracy Limit

**Day 229 | TYPE_ANTONYM reliability is bounded by antonym cluster
degeneracy, not by any training set or axis construction choice.
The core insight: TYPE_BC succeeds because its relations are FUNCTIONAL
(one source → one canonical target, e.g. France→Paris). TYPE_ANTONYM
fails for high-degeneracy attributes because antonymy is GRADED — each
source word has multiple equally valid antonyms (big→small/tiny/little/
minor/slight/compact) and the axis terminal lands at the cluster centroid,
not at any specific target. No homogeneous axis or quality threshold
fixes this. It is a structural property of W_E encoding.**

---

## Overview

Days 226-228 tested three hypotheses aimed at explaining the 0.333
ceiling for antonyms_sup_size:
1. Target cluster tightness → **DISPROVED** (speed has higher tightness, works)
2. axis_align > 0.70 threshold → **DISPROVED** (roughness=0.645 works, loudness=0.820 fails)
3. Homogeneous training pairs → **DISPROVED** (length axis variants give ≤0.667 via pair coverage, not generalisation)

All three hypotheses pointed to the wrong level of analysis. The actual
limit is at the SEMANTIC level: antonymy is not a functional relation.

---

## The Functional vs Non-Functional Distinction

### TYPE_BC: Functional Relations

A relation R is **functional** if for each source A, there is exactly
one correct target B = R(A).

```
capital(France)   = Paris       (unique, conventional)
past(go)          = went        (unique, grammatical)
plural(cat)       = cats        (unique, morphological)
number(seven)     = 7           (unique, definitional)
```

Every TYPE_BC relation in our pipeline is functional. The target is
**unique** by definition, convention, or grammar rule. This uniqueness
is what allows direction retrieval to succeed: the axis terminal lands
near a unique target, and that target ranks first.

### TYPE_ANTONYM: Non-Functional Relations

Antonymy is **not functional** for most attributes. For "big", all of
the following are valid antonyms depending on context and degree:

```
big → small    (general, most common)
big → tiny     (emphasises smallness)
big → little   (informal, childlike)
big → minute   (formal, very small)
big → compact  (specific context: rooms, cars)
big → petite   (specific context: humans)
big → slight   (emphasis on insubstantiality)
```

The word2vec analogy `king - man + woman = queen` works because
GENDER is a near-functional relation (one canonical gender flip per word).
SIZE antonymy is not: there is no single canonical "antonym of big".

**The attribute axis terminal lands at the centroid of all valid antonyms
of big. The nearest token to that centroid is whichever antonym is most
central — empirically "tiny" in Qwen2's W_E. "small" is rank=1.**

---

## Experimental Evidence

### High axis_align, Low acc (Loudness)

```
loudness pairs: (loud,quiet), (noisy,silent), (deafening,hushed),
                (boisterous,muted), (thunderous,whispered)
(ok pairs: loud/quiet, noisy/silent)
axis_align = 0.820   train_acc = 0.500
```

Despite axis_align=0.820 (the highest measured), loudness achieves only
0.500 training accuracy. The axis retrieves "quiet" for "loud" (correct)
but fails for "noisy→silent". Why?

- "quiet" is the most central member of the soft-sound cluster
- "noisy" has a softer-antonym centroid near "quiet" (not "silent")
- "silent" is at the periphery of the cluster (more extreme/absolute)

High axis_align means the training pairs agree on DIRECTION. But if the
antonym targets have high degeneracy (many near-synonyms), the axis
terminal still lands at the cluster centroid — which may not match any
specific training target.

### Low axis_align, High acc (Roughness)

```
roughness pairs: (rough,smooth), (coarse,fine), (jagged,polished),
                 (scratchy,silky), (rugged,sleek)
(ok pairs: rough/smooth, coarse/fine, jagged/polished)
axis_align = 0.645   train_acc = 1.000
```

Despite axis_align=0.645 (below the proposed 0.70 threshold), roughness
achieves 1.000 training accuracy. The antonym of "rough" is unambiguously
"smooth" in Qwen2's W_E. There are fewer competing near-synonyms for
TEXTURE SMOOTHNESS than for SIZE SMALLNESS. The axis terminal lands close
enough to the unique neighborhood of each specific target.

### Summary Table

```
attribute    aa    deg  train_acc  explanation
loudness    0.820  HIGH  0.500    quiet wins over silent (both valid)
speed       0.757  LOW   1.000    slow is unique; sluggish not synonymous with slow
weight      0.737  LOW   1.000    light is unique (few near-synonyms)
roughness   0.645  LOW   1.000    smooth is unique for texture dimension
temperature 0.585  MED   0.500    cold wins; freezing also valid
size        0.547  HIGH  0.333    tiny/small/little all compete
brightness  0.534  HIGH  0.250    dark/dim/gloomy/murky all compete

deg = estimated antonym cluster degeneracy (HIGH/MED/LOW)
```

The DEGENERACY column (estimated) predicts accuracy better than axis_align.

---

## Defining Antonym Cluster Degeneracy

**Antonym cluster degeneracy** for attribute X and source word A is the
number of tokens within cosine distance ε of the axis terminal that are
also semantically valid antonyms of A.

For practical measurement, we can proxy this as:

```
degeneracy(attribute) ≈ mean_k over training pairs,
  where k = number of tokens within cosine sim > 0.85 of each target token
```

High degeneracy attributes: size (tiny/small/little cluster), brightness
  (dark/dim/gloomy cluster), loudness (quiet/silent/hushed cluster)

Low degeneracy attributes: speed (slow/sluggish/plodding are more spread),
  roughness (smooth/fine/polished are distinct surface words),
  weight (light has mass-specific meaning; "featherlight" not common)

---

## The Formal Barrier

### Why this cannot be fixed

The axis computes: `query = normed(source_emb + target_direction)`

The target_direction points toward the centroid of all B-words in training:
`target_direction = normed(mean(normed(emb(B_i) - emb(A_i))))`

If training pairs have diverse B-words (small/tiny/little/short/narrow/thin),
the centroid direction is a weighted average of their individual directions
from their respective A-words. The query lands near the centroid of all B-words.

**There is no way to make the query land on a specific B if multiple B-words
are equidistant from the query and are near-synonyms of each other.**

This is not a flaw in the axis construction. It is a property of how
the LLM encodes semantics: synonymous words occupy nearby positions in W_E.
An axis pointing "toward small things" necessarily lands in the neighborhood
of ALL small things, not a specific one.

### The TYPE_BC contrast

TYPE_BC avoids this because:
- Training pairs: (France, Paris), (Germany, Berlin), (Italy, Rome)
- All B-words (Paris, Berlin, Rome) are DISTINCT in W_E — no city is a
  near-synonym of another city. The mean direction from source cluster
  to target cluster has a well-defined terminal near each specific target.
- Even at 42k tokens, no other token is near-synonymous with "Paris".

The B-word uniqueness of TYPE_BC is what makes direction retrieval work.
TYPE_ANTONYM lacks this uniqueness for high-degeneracy attributes.

---

## Updated Taxonomy: Degeneracy as a First-Class Property

```
Archetype     Relation type   B-uniqueness  max acc   mechanism
IDENTITY      identity        N/A           1.000     return source
TYPE_BC       functional      HIGH          0.75+     source + mean_dir
TYPE_ANTONYM  non-functional  HIGH (rare)   0.75+     source + attr_axis
              non-functional  LOW           ~0.33-0   axis fails (degeneracy)
TYPE_ADJACENT non-functional  LOW           ~0        proximity (degeneracy++)
```

The degeneracy spectrum is continuous:
- LOW degeneracy + functional → TYPE_BC (always works)
- LOW degeneracy + non-functional → TYPE_ANTONYM-like (works when degeneracy is low)
- HIGH degeneracy → TYPE_ADJACENT-like (no geometric retrieval possible)

TYPE_ANTONYM is TYPE_BC with partial degeneracy. TYPE_ADJACENT is fully
degenerate (no unique target exists in W_E for a given source).

---

## Predictive Metric: target_degeneracy

For a given attribute axis and test source A:

```
target_degeneracy(A, axis) = |{w in vocab : cos(w, axis_terminal(A)) > 0.85}|
```

where `axis_terminal(A) = normed(emb(A) + target_dir)`.

If target_degeneracy > k_threshold: retrieval will fail (return centroid, not target)
If target_degeneracy ≤ k_threshold: retrieval will succeed

This is measurable at runtime without knowing the true target B.

For size axis: target_degeneracy for "big" = many (tiny/small/little/compact...)
For speed axis: target_degeneracy for "fast" = few (slow is dominant, sluggish distinct)

**This metric can predict TYPE_ANTONYM success/failure before querying.**

---

## Final Accuracy Ceiling Analysis (v5)

```
Category                     Pairs  Correct  Ceiling  Note
TYPE_BC (9 domains)          48     45        49       plurals tokenization
TYPE_ANTONYM high-deg (size)  3      1         1       irreducible (degeneracy)
TYPE_ANTONYM low-deg (speed)  1      1         1       confirmed
IDENTITY (1 domain)           2      2         2       exact
TYPE_ADJACENT (1 domain)      6      0         0       irreducible
Total                         60     49*       53

(*v5 has 50/60 due to pair-lookup counting)
```

Achievable ceiling (fixing tokenization EC only): 53/60 = 0.883
Hard floor (unfixable): antonyms_unsup (6/60) + plurals EC (1/60) = 7/60

Note: "achievable ceiling" assumes low-degeneracy TYPE_ANTONYM domains
can be found and added. Size domain remains at 1/3.

---

## Open Problem: Can Degeneracy Be Reduced?

One approach: instead of using a single axis for SIZE, use **per-scale**
axes:

- "tiny_axis": very-small vs not-very-small (tiny/minute/microscopic)
- "small_axis": small vs not-small (small/little/minor)
- "short_axis": height-specific smallness (short/low/shallow)

When source "big" is queried, the system must know WHICH scale is intended.
This requires a secondary disambiguation mechanism, which puts us back
in the supervised setting.

The fundamental trade-off: **generalisation across the full size spectrum
requires accepting degeneracy. Eliminating degeneracy requires supervision
that specifies the intended scale.**

This trade-off is not specific to TruthSpace — it is inherent to any
unsupervised antonym retrieval system in distributed embeddings.

---

## Files

- `expedition_day228_homogeneous_axis.py` -- homogeneous axis tests
- `day228_homogeneous_axis.json` -- results
- `379_pipeline_v5.md` -- v5 pipeline
- `378_antonym_axis_limits.md` -- centroid collapse analysis
- `expedition_antonym_nn.py` -- Day 247 NN voting and chord variance analysis

---

## Day 247 Update: Chord Direction Variance Confirms Degeneracy

Day 247 extended the antonym analysis to the ARC GEOMETRY level,
quantifying the DIRECTION variance of antonym chord vectors directly.

### Chord Coherence Measurement

```
paradigm       mean_pair_cos   interpretation
antonym_size   0.036           NEAR-RANDOM (random baseline = 0.026)
adj_degree     0.360           HIGH COHERENCE
plural         0.160           MODERATE COHERENCE
```

`mean_pair_cos` = mean pairwise cosine between all (Bᵢ-Aᵢ, Bⱼ-Aⱼ) chord
vectors. A random set of unit vectors in R^1536 gives ≈0.026. Antonym
chords give 0.036 — essentially random.

### LOO Retrieval with All Methods

```
Method         antonym   adj_degree   plural
mean_dir       0/24 = 0%  18/24 = 75%   7/18 = 39%
1-NN analogy   0/24 = 0%  21/24 = 88%  13/18 = 72%
kNN-linear     0/24 = 0%  21/24 = 88%  10/18 = 56%
oracle         24/24=100%  24/24=100%  18/18=100%
```

With perfect direction (oracle): 100% for all paradigms.
With any learned direction: 0% for antonyms.

### Two Independent Barriers

**DC 380 (original)** identified BARRIER 1:
> Target degeneracy: big has many equally valid antonyms (small/tiny/little).
> The mean_dir terminal lands at the centroid of the antonym cluster,
> not near any specific token.

**Day 247** identifies BARRIER 2:
> Direction variance: the chord vectors (Bᵢ-Aᵢ) across antonym pairs
> are nearly random (mean_pair_cos=0.036). There is no shared direction
> to learn. Even with a word-specific method (1-NN analogy), the borrowed
> chord from the nearest neighbor points to a DIFFERENT antonym synonym
> (e.g., big→huge→predicts “little” instead of “small”).

Both barriers must be overcome simultaneously. Currently no method
solves either one within the geometric retrieval framework.

### Contrast: Why adj_degree Works

The adj_degree chord coherence (0.360 vs 0.026 random) is what enables
mean_dir to achieve 75% and 1-NN analogy to reach 87.5%. The private
plane rotation is consistent across words because:
- cos(pos, comp) ≈ 0.57 (consistent paradigm-specific cosine)
- The private plane orientation differs per word, but the ARC GEOMETRY
  (R, Ω, chord length) is consistent

For antonyms: cos(A, B) has std=0.12 (vs 0.08 for adj_degree), and the
chord directions are nearly random. There is no consistent geometric
structure to exploit.
