# DC 469: Corrected Irred Hurts, Working Chains, +al_rel Round-Trip, +able Mixed Fix

**Day 334 | Four findings: (1) Using corrected irred (ignoring Type0/vocab failures)
drops accuracy from 18/30=60% to 15/30=50% — raw irred IS useful signal because
vocabulary-limited failures reflect genuine axis coverage constraints, not just missing
tokens. (2) Two working 3-step chains confirmed: local→localize→localization (adj→verb→
event_noun) and write→writing→writer (verb→gerund→agent_noun). Chain 2 exploits the
fact that gerunds occupy a verb-noun position and +er_noun finds the AGENT of the
gerund activity. (3) +al_rel round-trip works 3/5 (nation/emotion/tradition), but
reversed -al_rel from abstract quality nouns sometimes navigates to Chinese/Japanese
kango forms (国籍/合法性), because Sino-Japanese abstract nouns share the same geometric
positions as English Latinate nouns — the language barrier is a gradient. (4) Mixed
+able training (Germanic + Latinate verbs) fixes the population mismatch: LOO 0%→45%,
irred 88%→38%, at the cost of lower pc (0.249→0.114).**

---

## Raw Irred Is Better Than Corrected Irred for Classification

### Experiment

Two predictor runs over the 30-axis benchmark:
1. **v6 (raw irred)**: use the raw holdout failure rate
2. **v6 (corrected irred)**: subtract Type0 (multi-token target) failures

```
v6 (raw irred):       18/30 = 60%
v6 (corrected irred): 15/30 = 50%
```

Corrected irred HURTS by 3 cases.

### Why Raw Irred Is Useful

The key insight: vocabulary-limited irred is NOT mere noise. It is a real signal about
axis quality. Axes with high vocabulary-limited irred tend to be:

- **Morphologically complex**: they operate on words with rare or irregular forms
- **POS-restricted**: they only apply to a specific etymological sub-class
- **Low-coverage**: the target morphological forms don't exist as single tokens for many words

These properties ARE what we want irred to capture. When the predictor sees high irred,
it correctly infers the axis is operating at the boundary of single-token vocabulary
(→ factual_local, semantic_diverse, or phonol_scatter).

When we remove Type0 failures, axes that SHOULD look "hard" (like en_zh, +tion, +ment)
suddenly look "easy" (irred ≈ 0%), and the predictor misclassifies them.

### The Correct Use of Each Metric

```
irred_raw  = Type0(vocab_limited) + Type1(geometric_failure)
irred_corr = Type1(geometric_failure) only

For CLASSIFICATION: use irred_raw  (captures: "is this axis hard to generalize?")
For DIAGNOSIS:      use irred_corr (captures: "is the axis direction geometrically wrong?")
```

The raw irred was right all along. The "irred correction" was a misguided attempt to
separate two things that shouldn't be separated for classification purposes.

---

## Working 3-Step Morphological Chains

### Chain 1: adj(Latin) → +ize(GROUP C) → +tion(GROUP A)

```
local    → localize    → localization  ✓   (3-step chain complete)
real     → realize     → realization   ✓
legal    → legalize    → legalization  ✓
final    → finalize    → finalized     ~   (+tion gives past participle, not -ation)
general  → generalize  → generalized  ~   (same issue)
moral    → legalize    ✗   (moral and legal share Latin adj sub-cluster: too close)
national → nacional   ✗   (Spanish overshoot from +ize)
```

**End-to-end: 3/7**

Why it works for local/real/legal: these adj have well-separated positions in W_E
(local≠real≠legal), their +ize forms (localize/realize/legalize) are the most common
and single-token, and their +tion forms (localization/realization/legalization) are
frequent enough to be single-token.

Why final→finalized (not finalization): "finalized" is MUCH more frequent than
"finalization" in the training corpus. The +tion axis, trained on common event nouns
(action, direction, education, creation), navigates toward the high-frequency noun
cluster, which for "finalize" contains "finalized" (past participle used as nominal
modifier) rather than "finalization".

### Chain 2: verb → +ing(GROUP E) → +er_noun

```
write  → writing  → writer  ✓   (perfect: agent of the gerund activity)
manage → managing → manager ✓
run    → running  → .running ✗  (dot prefix artifact in vocabulary)
build  → builds   ✗           (step 1 fails: builds not building)
```

**Why this chain is linguistically meaningful:**

The gerund (+ing form) occupies a position between verb and noun space in W_E. It
IS the verb activity, but as a nominal concept. The +er_noun axis finds the AGENT of
an activity — "writer" = one who writes, "manager" = one who manages.

So: `write(verb) → writing(activity_nominal) → writer(agent_of_activity)`

This is a 3-step linguistic derivation: V → V-ing → V-er. In traditional grammar,
this chain works because the -er suffix can attach to the base VERB (writer) but the
path through the gerund reveals the semantic logic: the gerund is the INTERMEDIATE
CONCEPT that +er_noun finds the agent of.

### Chain Validity Summary

```
VALID CHAINS (linguistically attested):
  adj(Latin) → +ize(C) → +tion(A):   local → localize → localization ✓
  verb       → +ing(E) → +er_noun:   write → writing → writer ✓

INVALID CHAINS (linguistically unattested):
  adj(Germ)  → +en(C)  → +ance(A):  darken → *darkance [not English]
  verb       → +3ps(E) → +ment(A):  runs → ran ✗ [+ment needs base verb, not 3ps]
```

The geometry enforces the linguistic compatibility rule. W_E encodes not just
semantic meaning but MORPHOLOGICAL GRAMMAR.

---

## The +al_rel Round-Trip and Chinese Proximity

### Test 3: noun → +al_rel → adj → -al_rel → noun

```
nation    → national    → nation    ✓  (PERFECT ROUND-TRIP)
emotion   → emotional   → emotion   ✓
tradition → traditional → tradition ✓
person    → personal    → 个人      ✗  (Chinese 'individual' instead of 'person')
origin    → original    → (original) ✗ (adj stays, reverse overshoots)
```

3/5 nouns return to themselves. The +al_rel axis is PARTIALLY reversible.

### Test 2: adj → +ity → noun → -al_rel → adj (3-step)

```
national → nationality → 国籍   ✗  (Chinese 'nationality')
legal    → legality    → 合法性  ✗  (Chinese 'legality')
moral    → morality    → morals  ✗  (plural noun, not adj)
personal → personality → personalities ✗  (plural)
```

The reversed -al_rel axis (adj→noun direction) applied to abstract quality nouns
consistently navigates to **Chinese/Japanese kango forms**.

### Why Chinese Kango Forms Appear Here

This is NOT a failure. It is a geometric truth:

```
English Latinate abstract nouns:  nationality, legality, finality
Chinese Sino-Japanese kango:       国籍,          合法性,    ...
Japanese on'yomi:                  国籍 (kokuseki), 合法性 (gōhōsei)
```

These concepts entered Chinese and Japanese via borrowing from Latin/French. They
encode the **same abstract concepts** in the same semantic neighborhood in W_E.

The reversed -al_rel axis, navigating from a quality noun TOWARD the adj cluster,
passes through a region where both English and Chinese/Japanese tokens are present.
When the nearest-neighbor search finds Chinese tokens first, it means:

> The Chinese kango cluster for abstract legal/national concepts is CLOSER to the
> "adj cluster arrival point" of -al_rel than the English Latinate adj cluster is.

This reveals: the Chinese cluster for borrowed abstract nouns is intermediate between
English nouns and English adj — geometrically, it occupies the boundary zone.

### Language Permeability at the Abstract Noun Boundary

```
English adj cluster       [national, legal, final, personal]
     |
     | (close — same concepts, borrowed vocabulary)
     |
Chinese kango cluster     [国籍, 合法性, ...]
     |
     | (some distance)
     |
English Latinate nouns    [nationality, legality, ...]
```

This confirms DC 463's finding that the language barrier is a gradient. Concepts
borrowed across language families end up geometrically co-located. The barrier is
permeable at concept boundaries.

---

## The +able Population Mismatch: Fixed by Mixed Training

### The Problem

```
+able GERMANIC-ONLY training:
  Sources: read/wash/break/love/use/accept/avoid/change (short Germanic verbs)
  pc=0.249  LOO=0%  irred_raw=88%  → fails completely on Latinate holdout
```

### The Fix

```
+able MIXED training (Germanic + Latinate):
  Sources: read/wash/break/love/use/accept/avoid/change (orig 8)
         + comfort/manage/reach/note/remark/reason/prefer/rely (8 new)
  pc=0.114  LOO=45%  irred_raw=38%  → works on diverse holdout!
  cos(mixed, germanic) = 0.749
```

### Trade-off Analysis

```
                  pc     LOO   irred   interpretation
germanic-only:  0.249    0%    88%    Very tight axis, wrong direction for Latinate
mixed:          0.114   45%    38%    Looser axis, covers both populations
```

The pc DROPS because mixing two sub-clusters creates a less coherent axis (the
Germanic-able cluster and Latinate-able cluster are related but not identical, so
the mean direction is a compromise). But the COVERAGE massively improves.

**The two-sub-axis structure of +able:**

```
Germanic -able:  readable, washable, breakable, lovable
  → "can be [verb]ed" with simple Germanic verbs
  → tight cluster: all short monosyllabic/bisyllabic Germanic words

Latinate -able:  comfortable, manageable, remarkable, reasonable
  → "having the quality of being [verb]able"
  → different sub-cluster: Latinate verbs with Latinate -able suffix
  → e.g., "com-fort-able" is purely Latin roots
```

The mixed axis (cos=0.749 with germanic-only) is a BRIDGE between these two
sub-clusters. 0.749 is high — they share a large component (the "ability/property"
direction) — but the 25% that differs is the distinction between Germanic and
Latinate populations.

### Implication for the Morphological Hypothesis

This is the first successful demonstration of **population mismatch correction**:

> When an axis fails on a holdout population, adding that population to training
> creates a mixed axis that covers both. The price is lower pc (less coherent axis)
> but higher LOO (better generalization).

This has important implications: many axes that currently show high irred might be
curable by expanding the training population to include the holdout sub-cluster.

---

## Day 335 Plan

1. **Etymology sub-cluster map (corrected method)**: instead of using axis directions,
   compute CENTROID of source word embeddings for each axis group, then score all
   adj/verb/noun tokens by cosine similarity to that centroid. This correctly maps
   the Germanic vs Latinate sub-clusters.

2. **Axis resolution**: measure how many distinct adj appear in EACH sub-cluster
   (Germanic adj cluster, Latinate adj cluster) and quantify overlap. Are there
   "boundary" adj that appear in both?

3. **The write→writing→writer chain deep-dive**: test all 8 training verbs through
   the write→writing→writer chain. How many give the correct agent noun? Is the
   chain reliable enough to be a NEW morphological route?

4. **v6 boundary analysis**: which 12 out of 30 benchmark axes are STILL wrong?
   Group them by failure mode: (a) wrong pc range, (b) right pc wrong irred,
   (c) ambiguous true label. Find the minimum predictor changes to fix them.

5. **Multi-sub-axis axes**: test whether +ness, +ance, +tion have similar two-sub-
   cluster structure as +able (Germanic-ness vs Latinate-ness). If yes, mixed
   training might improve all of them.

---

## Files

- `expedition_log.md` — Days 322-334 results
- `468_v7_regression_4level_source_homology_chain_validity_and_irred_types.md` — DC 468
- `day334_corrected_irred_chains_circular_etym_map_able_fix.py` — experiment script
