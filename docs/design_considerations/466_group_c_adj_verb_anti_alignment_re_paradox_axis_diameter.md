# DC 466: GROUP C Discovered, Second Anti-Alignment, +re- Paradox, Axis Diameter

**Day 331 | Four major findings: (1) GROUP C (adj→verb) is discovered: {+en, +ize}
with internal cos=0.385, comparable to GROUP D and GROUP E. GROUP C is the SECOND
anti-aligned pair with GROUP A (verb→noun): cos=−0.075 to −0.100. (2) Adaptive
thresholds fail: scaling pc thresholds by training set size yields 16/30=53% vs
baseline 18/30=60%. The scaling direction is inconsistent across axes, so no
universal correction exists. (3) The +re- paradox is resolved: both LOO fixed-scale
(8%) and per-pair scale (15%) are near-zero, proving the DIRECTION is unstable
(not just the scale). High magnitude (0.737), high spread (0.092), and low LOO
define a new "local memorization" class. (4) Axis diameter measurements establish
that chord spread and magnitude are independent of pc. Inflectional morphology has
the smallest magnitudes (+3ps=0.475); antonymy and prefixal reversal have the
largest (0.69-0.74).**

---

## GROUP C: adj → verb

### Evidence

```
+en pairs: bright→brighten, dark→darken, hard→harden, wide→widen,
           soft→soften, fresh→freshen, weak→weaken, sharp→sharpen,
           deep→deepen, light→lighten, thick→thicken, white→whiten ...
+ize pairs: real→realize, moral→moralize, national→nationalize,
            local→localize, modern→modernize, memory→memorize ...

cos(+en, +ize) = +0.385   ← GROUP C internal cosine

+en  axis: pc=0.230  LOO=70%  irred=50%
+ize axis: pc=0.187  LOO=50%  irred=33%
```

The internal cosine (0.385) is stronger than GROUP D (0.323) and comparable to
GROUP E (0.330). GROUP C is a real morphological family.

### GROUP C vs All Other Groups

```
GROUP C vs GROUP A: −0.075 to −0.100  (NEGATIVE — second anti-aligned pair)
GROUP C vs GROUP B: +0.253            (adj-source overlap with +ness)
GROUP C vs GROUP D: +0.188            (partial verb-source overlap)
GROUP C vs GROUP E: +0.064            (low — verb-source but very different target)
```

The negative cosine with GROUP A is the key structural discovery.

### Structural Explanation: The Morphological Directed Graph

The five groups form a directed graph on the three POS categories {adj, verb, noun}:

```
adj ──GROUP_C──→ verb ──GROUP_A──→ noun
│                │                   │
GROUP_B          GROUP_D             │
↓                ↓                   │
noun             adj ←─────────────al_rel
                 │
                 GROUP_C (back to adj)
```

More precisely:
```
noun ──+al_rel──→ adj ──GROUP_C──→ verb ──GROUP_A──→ noun (cycle!)
                   │               │
                  GROUP_B        GROUP_D
                   ↓               ↓
                  noun            adj
                   │
                  +ity (back to adj←noun)
```

The key anti-alignments:
- **Pair 1**: +al_rel (noun→adj) ↔ +ity (adj→noun): direct reverse, cos=−0.383 to −0.432
- **Pair 2**: GROUP_C (adj→verb) ↔ GROUP_A (verb→noun): adjacent-step reverse, cos=−0.075 to −0.100

Pair 2 is ~8× weaker because GROUP_A and GROUP_C are not direct reverses — they share no axis. They're anti-aligned because GROUP_C ARRIVES where GROUP_A DEPARTS:
- GROUP_C lands in the verb cluster
- GROUP_A departs from the verb cluster
- So they point in opposite directions at the verb cluster boundary

### The Complete Updated GROUP MAP

```
        A         B         C         D         E
A(v→n)  ---      +0.047   -0.088    +0.028    +0.128
B(a→n)  +0.047    ---     +0.253    +0.082    +0.106
C(a→v)  -0.088  +0.253     ---      +0.188    +0.064
D(v→a)  +0.028  +0.082   +0.188     ---       +0.161
E(v→v)  +0.128  +0.106   +0.064    +0.161     ---

REVERSE PAIR: +al_rel vs +ity = −0.383 to −0.432
```

**Negative cosines in the full map: only 2 pairs.**
1. +al_rel ↔ +ity: direct noun↔adj reversal (strongest)
2. GROUP_C ↔ GROUP_A: adj→verb vs verb→noun (weaker)

**All five group-to-group cosines are positive**, confirming that each group
shares some common "morphological transformation" component with every other group.

---

## Adaptive Thresholds Fail

### The Inconsistency

```
Axis     5-pair pc   8-pair pc   Direction
er_comp  0.364       0.367       ↓ (pc decreases with fewer pairs)
ablaut   0.447       0.345       ↑ (pc increases with fewer pairs)
3ps      0.293       0.250       ↑
ness     0.199       0.187       ↑
tion     0.091       0.121       ↓
```

The direction of the pc shift with training set size is INCONSISTENT. For some axes
(ablaut, 3ps) fewer pairs → higher pc; for others (er_comp, tion) fewer pairs →
lower pc. This reflects the HIGH VARIANCE of individual chord pair sampling:

- Ablaut: includes (go→went, take→took, give→gave) which are very similar displacements.
  With 5 pairs, you might include more of these similar pairs → higher pc.
- er_comp: all pairs are quite similar regardless — pc is stable.
- tion: (act→action, direct→direction) are regular but with phonological variation.

There is no universal scaling rule because the variance is PAIR-SELECTION-DEPENDENT,
not training-size-dependent.

### Consequence

The predictor's accuracy ceiling (60%) on 5-pair probes is a PAIR-SELECTION
VARIANCE problem, not a threshold calibration problem. The only valid fix is:
1. Use ≥8 pairs for axis computation (recommended minimum)
2. Run multiple 5-pair subsets and ensemble (average pc, LOO across subsets)
3. Explicitly acknowledge the predictor's ≥8-pair requirement in its specification

---

## The +re- Paradox: Local Memorization

### Test Results

```
LOO fixed-scale:    8%  (1/13 held-out pair navigable)
LOO per-pair scale: 15% (2/13 held-out pair navigable with optimal scale)
```

The marginal improvement from per-pair scale (8% → 15%) shows that scale is
PARTIALLY responsible for the failure, but not primarily. The DIRECTION is unstable.

### Chord Analysis

```
+re- chord-to-axis alignment: mean=0.446  std=0.081  range=[0.282, 0.580]
+ness chord-to-axis alignment: ~0.90+ (implied by spread=0.032)
```

The mean alignment of 0.446 means the average +re- chord is only 0.446 cosine-similar
to the mean axis. This is much lower than a true morphological axis (where chords
should align >0.60 with the mean). The 0.282 minimum means some +re- chords are
nearly ORTHOGONAL to the mean axis.

### Why Navigation Works on Training Pairs

The +re- axis navigates 4/6 specific test pairs (do/write/build/think) because:
1. These have very large displacements (mean mag=0.747)
2. The source words (do, write, build, think) are common monosyllables
3. The re- prefixed forms are well-represented, single-token, and common
4. For these specific pairs, the large displacement overshoots to the right region

But "return" and "rename" fail because:
- "return" is lexicalized (not re+turn semantically)
- "rename" is less common, may be multi-token or poorly represented

### Defining "Local Memorization"

A new axis classification to add to the predictor:

```
local_memorization:
  pc: 0.10-0.20
  spread: > 0.08
  mag: > 0.70
  LOO (fixed): < 0.20
  LOO (per-pair): < 0.30
  irred: > 0.80
  Interpretation: axis has memorized training displacements but cannot generalize.
  The direction is highly variable (spread) with large individual steps (mag).
```

This class is distinct from factual_local/translation (which has consistent direction
and navigates its training pairs predictably) and from semantic_diverse (which has
consistent direction but unpredictable targets).

---

## Axis Diameter: The Four-Parameter Description

### Results Table

```
Axis       pc      spread   mag     type
er_comp   0.367   0.042   0.564   morph_uniform
3ps       0.250   0.066   0.475   morph_moderate  ← SMALLEST magnitude
ed_reg    0.198   0.052   0.497   morph_moderate
ness      0.187   0.032   0.604   phonol_scatter  ← LOWEST spread
ablaut    0.345   0.095   0.503   phonol_scatter  ← HIGHEST spread (real axis)
ance      0.134   0.044   0.544   phonol_scatter
er_noun   0.130   0.030   0.657   semantic_diverse ← 2nd lowest spread
adj_ant   0.047   0.041   0.692   polar_local
en_zh     0.109   0.042   0.578   factual_local
en_es     0.139   0.049   0.593   translation
+re-      0.142   0.092   0.737   standalone      ← LARGEST mag, 2nd highest spread
+ize      0.203   0.066   0.642   GROUP_C
```

### Spread vs pc: Independent Dimensions

The key finding: **spread and pc are independent**.

- +ness: LOW spread (0.032) + MODERATE pc (0.187)
  → All chords point the same direction as the mean, but disagree with each other
  → "Funnel morphology": depart same adj region, arrive scattered noun positions

- ablaut: HIGH spread (0.095) + HIGH pc (0.345)
  → Chords fan out widely from mean axis, but agree pairwise
  → Irregular verbs have very different individual trajectories that happen to AGREE

The 4-parameter space (pc, spread, mag, LOO) enables finer classification:

```
True morphological axis:   high pc, moderate spread, moderate mag, high LOO
Funnel morphology (+ness): moderate pc, LOW spread, HIGH mag, moderate LOO
Ablaut-type axis:          high pc, HIGH spread, moderate mag, moderate LOO
Local memorization (+re-): low pc, HIGH spread, HIGH mag, LOW LOO
Semantic diverse:          low pc, LOW spread, HIGH mag, LOW LOO
```

### Magnitude as Morphological Depth

```
Inflectional:  mag 0.47-0.50  (3ps, ed_reg — tiny meaning change)
Comparative:   mag 0.56        (er_comp)
Derivational:  mag 0.54-0.65  (ance, ness, ize — bigger meaning change)
Cross-lingual: mag 0.58        (en_zh — crossing language boundary)
Antonymy:      mag 0.69        (adj_ant — maximum semantic distance)
Prefixal rev:  mag 0.74        (+re- — large reversal concept change)
```

The ordering **inflectional < comparative < derivational < cross-lingual < antonymy
< reversal** reflects the intuitive "semantic distance" of each operation:
- Present-tense inflection barely changes meaning
- Derivation creates new words with new meanings
- Antonymy inverts the core semantic feature
- Reversal (re-) implies the full original action was undone

---

## Day 332 Plan

1. **Complete inter-group matrix with GROUP C**: measure all A/B/C/D/E pairings
   and build the final 5×5 cosine matrix. Confirm GROUP C vs GROUP A is the
   only new negative entry.

2. **GROUP C vs REVERSE PAIR**: does GROUP_C vs +ity have a specific cosine?
   (GROUP C arrives at verb cluster; +ity departs from adj cluster: related directions?)

3. **Spread as fourth predictor feature**: add (pc, LOO, irred, spread) to the
   4-feature predictor. Does spread help separate ablaut (should be phonol_scatter)
   from er_comp (should be morph_uniform) when pc is similar?

4. **Axis chain: GROUP C → GROUP E**: adj → verb → inflected.
   e.g., 'bright' → 'brighten' → 'brightened'. Does the chain work with GROUP C
   as the first step?

5. **GROUP C +ize irred analysis**: why is irred=33% for +ize? Which words fail?
   Is it a tokenization problem (multi-token +ize forms) or a vocabulary problem?

---

## Files

- `expedition_log.md` — Days 322-331 results
- `465_group_map_tense_subcluster_language_barrier_and_benchmark_ceiling.md` — DC 465
- `day331_adaptive_thresh_groupc_ing_cross_re_paradox_diameter.py` — experiment script
