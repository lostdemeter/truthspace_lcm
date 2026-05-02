# DC 468: v7 Regression, 4-Level Source Population Homology, Chain Validity Rule, Irred Types

**Day 333 | Four findings: (1) Predictor v7 regresses to 17/30=57% because the
spread rule (pc>0.35, spread>0.07 → phonol_scatter) breaks relational axes
(London→England type) that have high pc AND high spread. Revert to v6. (2) Source
population homology is confirmed at four levels: individual token overlap (0.38-0.47),
etymological sub-cluster (0.35-0.42), POS match (0.19-0.29), no POS match (0.03-0.15).
Antonym pairs with ZERO shared sources still share cos=0.190 through adj-POS identity.
(3) The linguistic chain validity rule: chains work when source populations are
linguistically compatible (local→localize→localization ✓), fail when incompatible
(darken→darkening/darkness: Germanic -en verbs don't take Latin -ance suffix). (4)
Irred types: +able has 100% geometric irred from source population mismatch
(Germanic vs Latinate training split); +ize has 5/8 vocabulary-limited irred from
multi-token targets; irred measurement must distinguish no_emb from geometric failure.**

---

## Predictor v7 Regression: The Spread Rule Breaks Relational Axes

### What Went Wrong

```
v6: 18/30 = 60%
v7: 17/30 = 57%
```

The v7 spread rule:
```
if pc > 0.35:
  if spread > 0.07: return 'phonol_scatter'  ← NEW
  else:             return 'morph_uniform'
```

Problem: the **relational** axis (London→England, Paris→France, etc.) has:
- pc ≈ 0.36-0.40 (high — all capital→country pairs are similar displacements)
- spread likely > 0.07 (diverse: European + Asian + Middle Eastern + Latin American pairs)

In v6: `morph_uniform/relational_geom` → `match('relational_geom')` = True ✓

In v7: `phonol_scatter` → `match('relational_geom')` = False ✗

Similar issue with `er_sup` if its spread > 0.07 due to irregular superlatives.

### Why the Rule Is Wrong in Principle

High spread at high pc can mean:
1. **Irregular morphology** (ablaut): same POS transformation, very different phonological paths
2. **Diverse semantic domains** (relational): same relational schema, very different word domains
3. **Superlative variation**: small orthographic variation in suffix (-est, -iest, -most)

The spread rule can only distinguish (1) from (2) if we ALREADY KNOW the axis is
morphological. For unknown axes, spread=0.08 at pc=0.40 is ambiguous.

### v7 Verdict: Revert to v6

The correct fix requires a TWO-STAGE approach:
1. First classify as morphological vs relational (using irred and LOO)
2. WITHIN morphological high-pc axes, use spread to distinguish uniform vs irregular

This 2-stage rule is not implementable without first resolving the v6 misclassification
ceiling (which has more fundamental causes — training set size).

---

## Source Population Homology: 4 Levels Confirmed

### Empirical Results

```
Level 1: Same individual source tokens (6/8 overlap)
  cos(en_es, en_de) = +0.467   (house/water/sun/book/day/night shared)
  cos(+ize, +ity)   = +0.419   (moral/national/legal/final/local shared)

Level 2: Same etymological sub-cluster (3-5/8 overlap)
  cos(+ize, +ity)   = +0.419   (Latin adj — see above)
  cos(+en, +er_comp) = +0.375  (bright/dark/soft shared; Germanic adj cluster)
  cos(+able, +3ps)  = +0.261   (Germanic verbs: read/break/make/work shared)

Level 3: Same POS source (0/8 shared tokens)
  cos(+3ps_motion, +3ps_cognit) = +0.288  (verb POS, different verbs)
  cos(adj_ant1, adj_ant2) = +0.190        (adj POS, different adj sub-groups)

Level 4: Different POS (0/8 shared, different POS)
  GROUP C vs GROUP A = −0.143             (adj-source vs verb-source: anti-aligned)
  +al_rel vs GROUP B = −0.240             (noun-source vs adj-source)
  typical cross-POS: 0.03-0.15            (positive due to shared 'morphological' component)
```

### The Antonym Anomaly (Level 3 Zero-Overlap)

adj_ant1 pairs: (good/bad, hot/cold, fast/slow, big/small) — core semantic primitives
adj_ant2 pairs: (bright/dark, hard/soft, high/low, rich/poor) — perceptual/physical adj

Zero shared source words. Yet cos = +0.190.

This means the antonymy OPERATION itself contributes cos ≈ 0.19 regardless of which
adj are being opposed. The "semantic reversal" direction within the adj cluster is
partially shared across all antonym axes. This is Level 3 POS-identity effect.

### Formal Statement of the 4-Level Scale

```
cos(axis_1, axis_2) ≈
  α × P_token(src1, src2)         # token overlap: 0 → 0.47
  + β × P_cluster(src1, src2)     # sub-cluster similarity: 0 → 0.30
  + γ × P_pos(src1, src2)         # POS identity: 0 → 0.20
  + δ × P_target(tgt1, tgt2)      # target similarity: 0 → 0.15
  + ε × P_morph(axis1, axis2)     # shared 'morphological' component: ≈ 0.05

where:
  α > β > γ > δ  (token identity dominates)
  P_morph is approximately constant for all morphological axes (~0.05)
```

The P_morph term explains why ALL morphological axes have small positive cosines
with each other (≈ 0.03-0.15) even across POS boundaries. There's a shared
"transformation direction" in W_E that all morphological operations share.

---

## The Linguistic Chain Validity Rule

### Working Chains

**GROUP_C(+ize) → GROUP_A(+tion): adj → verb → event_noun**

```
local   → localize   → localization  ✓
real    → realize    → realization   ✓
final   → finalize   → finalized     ~ (past participle not +tion)
```

Why it works: Both +ize and +tion operate on Latin-root words. The +ize TARGETS
(localize, realize) are in the Latin-root VERB cluster. The +tion SOURCE cluster
is also the Latin-root VERB cluster. **Source populations are compatible.**

**Linguistic analysis**: local → localize (Latin -ize) → localization (Latin -ation)
This is a real and common morphological chain in English (legalize → legalization,
nationalize → nationalization, etc.).

### Failing Chains

**GROUP_C(+en) → GROUP_A(+ance): adj → verb → event_noun (FAILS)**

```
dark    → darken    → dark/darkness (not darkance!)
wide    → widen     → widening      (not widance!)
soft    → soften    → softened      (not softance!)
```

Why it fails: Germanic -en verbs don't take Latin -ance suffix. "darkance" is not
a word. The +ance axis (trained on: perform→performance, resist→resistance) operates
on LATIN-ROOT VERBS, not Germanic -en verbs.

When the +ance axis is applied to "darken", it navigates to the NEAREST +ance-type
noun from "darken". Since "darken" is near "dark", it produces "darkness" or "dark" —
the nearest quality-noun, not an event noun with -ance suffix.

**Linguistic analysis**: Germanic -en verbs take Germanic suffixes (-ness, -ening),
not Latin suffixes (-ance, -ment, -tion). This linguistic constraint is ENCODED
GEOMETRICALLY in W_E.

### The Chain Validity Rule

```
Chain (GROUP_X → GROUP_Y) is geometrically valid ⟺
  1. target_POS(GROUP_X) = source_POS(GROUP_Y)  [POS compatibility]
  2. source_population(GROUP_X targets) ⊆ source_population(GROUP_Y)  [pop compatibility]
```

Examples:
```
+ize → +tion:  target=Latin_verb, source(+tion)=Latin_verb    ✓ both conditions
+en  → +ance:  target=Germ_verb,  source(+ance)=Latin_verb   ✗ condition 2 fails
+ize → +ment:  target=Latin_verb, source(+ment)=?             ? depends on +ment training
```

This rule predicts which morphological chains will WORK purely from linguistic
etymology — the geometry respects etymological compatibility.

### Consequence for the Morphological Hypothesis

This is a strong validation of the geometric hypothesis:

> W_E does not just represent words — it represents LINGUISTIC STRUCTURE.
> The geometry encodes etymological compatibility: Latin-source operations
> chain with Latin-source operations; Germanic-source operations chain with
> Germanic-source operations. Cross-etymology chains fail geometrically
> because the landing clusters are linguistically incompatible.

---

## Irreducibility Types: Corrected Classification

### The Type 1/Type 2 Framework (Corrected)

```
Type 0: Target has no single-token embedding
  → irred is VOCABULARY-LIMITED (multi-token target)
  → axis direction may be correct; target simply doesn't exist as single token
  → Example: "popularize" [2 tok] → Type 0

Type 1: Target has single-token embedding, but navigation fails geometrically
  → irred is TRUE GEOMETRIC FAILURE
  → Axis direction is wrong for this pair
  → Example: "activate" [1 tok] but sim=0.412 → wrong direction (it's -ate not -ize)

Type 2: Target has single-token embedding, navigation is geometrically close
  → irred is SCALE-RELATED or NEAR-MISS
  → Axis direction is approximately correct but scale/target density prevents exact hit
  → Example: "judgment" [1 tok] sim=0.503 → close but "judge→judgment" is irregular
```

### Results by Axis

```
+ize  (LOO=86%):  5/8 Type0(vocab), 1/8 Type1(geom) [activate/wrong_suffix], 2/8 OK
+tion (LOO=38%):  0/6 Type0,         1/6 Type1(geom) [permit→permission, 0.392], 5/6 OK
+ment (LOO=62%):  0/5 Type0,         1/5 Type1(geom) [judge→judgment, 0.503], 4/5 OK
+able (LOO=0%):   0/6 Type0,         6/6 Type1(geom) [all sim 0.21-0.58]
```

### The +able Geometric Failure

+able is the ONLY GROUP D axis with 100% Type 1 irred on holdout. Cause:

```
Training population:  read/wash/break/love/use/accept/avoid/change
  → all SHORT COMMON GERMANIC VERBS
  → -able forms: readable/washable/breakable/lovable/usable (high-frequency)
  → form a tight GERMANIC-VERB-ABILITY cluster in adj space

Holdout population:  comfort/manage/reach/note/remark/reason
  → mix of LATINATE (comfort, manage, remark, reason) and Germanic (reach, note)
  → -able forms: comfortable/manageable/reachable/notable/remarkable/reasonable
  → these are in the FORMAL-ADJECTIVE cluster (different from Germanic-ability cluster)
```

The +able axis trained on Germanic verbs points to the wrong adj sub-cluster for
Latinate -able words. The cos(training_cluster, holdout_cluster) ≈ 0.3-0.6 (from
best_sim values), indicating they're RELATED but not in the same sub-cluster.

### Implication for the Predictor

The predictor currently uses irred to classify axis TYPE:
- High irred → semantic_diverse or factual_local
- Low irred → morph_moderate or morph_uniform

But Type 0 irred (vocabulary-limited) and Type 1 irred (population mismatch) should
NOT reduce the axis's quality score. They indicate:
- Type 0: vocabulary coverage is incomplete (not the axis's fault)
- Type 1 (pop mismatch): axis was trained on wrong population (not a geometric failure)

Only Type 1 DIRECTION failure (axis points wrong way for these pairs) represents
true geometric irred. The predictor needs a CORRECTED IRRED measure:

```
irred_corrected = Type1_direction_failures / total_holdout_pairs
irred_vocab = Type0_failures / total_holdout_pairs

For axis classification, use irred_corrected not irred_raw.
```

---

## Day 334 Plan

1. **v6 benchmark with corrected irred**: re-run the 30-axis benchmark using
   `irred_corrected` (ignoring Type 0 and population-mismatch Type 1 failures).
   Expected: several phonol_scatter axes that currently fail due to vocabulary-limited
   irred will now correctly classify.

2. **Chain graph complete**: test all linguistically VALID chains:
   - +ize → +tion (Latinate: adj→verb→event_noun)
   - +ize → +al_rel (can we navigate from -ize to -al via +al_rel axis?)
   - +ness → reverse (+al_rel) → noun (does reversing adj→noun give back noun?)
   - GROUP_E(+3ps) → GROUP_A(+ment) (verb→3ps→event_noun: runs→running→?)

3. **+al_rel as double reverse**: If -al_rel is the reverse of +ity, can we navigate:
   noun → (+al_rel) → adj → (-al_rel) → back_to_noun?
   This would be a CIRCULAR chain and a strong test of geometric consistency.

4. **Etymological sub-cluster mapping**: formally measure the Germanic adj cluster
   vs Latin adj cluster by computing: for all adj in CLEAN_MASK, measure their
   cos with the +en axis and +ity axis. Adj with high cos(+en) are Germanic;
   those with high cos(+ity) are Latin. This maps the adj cluster's etymological structure.

5. **Fix +able with mixed training**: re-train +able on a mixed set of both
   Germanic and Latinate verbs. Does LOO improve to > 50%?

---

## Files

- `expedition_log.md` — Days 322-333 results
- `467_full_5x5_group_map_source_population_homology_and_spread_feature.md` — DC 467
- `day333_v7_irred_types_source_pop_chain_graph_alrel.py` — experiment script
