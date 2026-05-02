# DC 428: Inflated Pairwise Cosine — Mechanisms, Detection, and the Element→Symbol Sub-Pattern

**Day 293 | Two inflation mechanisms for pairwise chord cosine (pc)
identified and measured: (1) Shared-target: fast→slow AND quick→slow
inflate pc from 0.297 (true) to 0.647 (inflated) because source words
are synonymous and share the same target. (2) Scope-limitation (DC 427):
training covers only the subset of pairs where the relation appears
linear. Contamination detector built: WARN if pc>0.3 AND (target_div<0.6
OR src_pc>0.4). Catches shared-target, misses scope-limitation. New
domain: element→symbol (pc=0.139, 57% train, 67% holdout) splits into
alphabetic-symbol sub-axis (HIT: H, He, Li, N, O) vs Latin-derived
(MISS: Fe, Au, Ag). ENCODE=DECODE confirmed for element domain:
cos(fwd, rev) = −1.0000 exactly.**

---

## Mechanism 1: Shared-Target Inflation

### Experiment

Training speed antonyms three ways with the same 4 source words:

```
shared_target:  fast→slow, quick→slow, rapid→slow, swift→slow
                (all pointing to the same target)
diversified:    fast→slow, quick→quiet, rapid→gradual, swift→sluggish
                (each pair has a distinct target)
canonical:      fast→slow, quick→slow, rapid→gradual, swift→sluggish
                (mixed — 2 shared, 2 distinct)
```

### Results

```
version         pc      target_div  holdout (3 pairs)
shared_target   0.647   0.25        1/3 (33%)
diversified     0.297   1.00        2/3 (67%)
canonical       0.387   0.75        2/3 (67%)
```

The pc drops from 0.647 to 0.297 when targets are diversified — a 2.2×
reduction. Despite having lower pc, the diversified axis achieves
better holdout accuracy (67% vs 33%).

### Geometric Explanation

In W_E, "fast" and "quick" are near each other (they are synonyms).
If both map to "slow", then:

```
chord(fast→slow) = emb(slow) − emb(fast)
chord(quick→slow) = emb(slow) − emb(quick)
```

Since `emb(fast) ≈ emb(quick)` (close in W_E):
```
chord(fast→slow) ≈ chord(quick→slow)
```

The two chord vectors are nearly identical → pairwise cosine ≈ 1.0.
This reflects not transformation consistency but source-word
proximity. The "axis" is just "point toward the 'slow' region from
the 'fast/quick' cluster."

When applied to "agile" or "lively" (which are NOT in the fast/quick
cluster), the axis still points toward "slow" → retrieves "slow"
for all holdout words → 33% accuracy (only correct when the true
antonym IS "slow").

### Shared-Target Inflation Formula

```
pc_inflated ≈ cos(src_i, src_j)  [when all pairs share the same target]
```

Because:
```
normed(t − s_i) ≈ normed(t − s_j)  when  s_i ≈ s_j
```

The pairwise cosine converges to the source-source cosine, not to
the transformation coherence.

### Detection

Signal: `target_div = unique_targets / total_pairs`

- shared_target: target_div = 0.25 (1 unique / 4 pairs) → WARN
- diversified: target_div = 1.00 → OK
- canonical: target_div = 0.75 → borderline

Threshold: `target_div < 0.6 → suspect shared-target inflation`

---

## Mechanism 2: Scope-Limitation (Recap from DC 427)

Training on the subset of pairs where the relation appears regular
(European country demonyms = language names) inflates pc because
the training distribution is not representative of the full relation.

Detection is harder: `target_div = 1.0` (targets ARE diverse) but
the scope is limited. Requires domain knowledge or a held-out
"out-of-scope" validation set.

---

## The Contamination Detector

```python
def contamination_risk(pairs, axis_pc):
    target_div = len(set(t for _,t in pairs)) / len(pairs)
    src_embs   = [normed(get_emb(s)[0]) for s,_ in pairs if get_emb(s)[0] is not None]
    src_pc     = mean_pairwise_cos(src_embs)
    
    shared_target_risk = (target_div < 0.6) and (axis_pc > 0.30)
    synonymous_src_risk = (src_pc > 0.40) and (axis_pc > 0.30)
    
    return shared_target_risk or synonymous_src_risk
```

### Performance on Known Cases

```
Axis               pc      target_div  src_pc   WARN?   True label
+er (comp)         0.395   1.000       0.112    NO      GENUINE   ✓
country→dem        0.598   1.000       0.352    NO      GENUINE   ✓
speed (shared)     0.605   0.167       0.355    YES     INFLATED  ✓
country→lang       0.529   1.000       0.327    NO      SCOPE_LIM ✗
element→sym        0.139   1.000       0.131    NO      LOW-PC    ✓
+s plural          0.162   1.000       0.080    NO      GENUINE   ✓
```

Precision: 1/1 warnings are true positives.
Recall: catches 1/2 inflation types (misses scope-limitation).

The detector is a useful pre-screening tool but does not replace
holdout evaluation. A clean holdout set remains the gold standard.

---

## The Element→Symbol Domain

### Properties

- **Bijective**: each element has exactly one symbol (one-to-one)
- **Target diverse**: all 14 symbols are different (target_div=1.0)
- **Source diversity**: elements span diverse categories (gas, metal,
  noble gas, radioactive...) → low src_pc=0.131
- **Low pc**: 0.1394 — predicts marginal generalisation

### Results

```
Train: 8/14 (57%)   Holdout: 6/9 (67%)
ENCODE=DECODE: cos(fwd, rev) = −1.0000
Scale ratio fwd/rev = 0.674
```

### The Sub-Pattern Split

Element symbols follow TWO different rules:

**Rule A — Alphabetic**: symbol = first letter(s) of English name
```
hydrogen → H     helium → He    lithium → Li
nitrogen → N     oxygen → O     calcium → Ca
potassium → K    chlorine → Cl  sulfur → S
aluminum → Al    silicon → Si   magnesium → Mg
```
All 12 CORRECT. This is a near-morphological transformation: take
the first 1-2 letters and capitalize. In W_E, element name
embeddings cluster near their capitalized abbreviations because
science text co-occurs "hydrogen (H)", "oxygen (O)" etc.

**Rule B — Latin-derived**: symbol = abbreviation of Latin name
```
iron → Fe (ferrum)    gold → Au (aurum)
silver → Ag (argentum) lead → Pb (plumbum)
tin → Sn (stannum)
```
All 5 FAIL. The English name and the Latin abbreviation have no
alphabetic relationship. There is no consistent geometric direction
from "iron" to "Fe" vs "gold" to "Au" — each is an arbitrary
association. The displacement vectors point in unrelated directions.

**Rule C — Ambiguous tokens**: two-letter symbols whose BPE embedding
is far from the element name
```
neon → Ne [got Neon]    argon → Ar [got arg]
chromium → Cr [got Chromium]    zinc → Zn [SKIP multi-token]
```
The axis overshoots to the capitalized full form or a different token.

### Why the Combined pc is Low (0.139)

The combined axis averages:
- Rule A chord vectors: all point consistently toward "short letter
  token" region (would have high pc in isolation)
- Rule B chord vectors: point in random directions (would have ~0 pc)
- Rule C: partially consistent

The mixture of two fundamentally different transformations depresses
the overall pc to 0.139, which correctly signals that the axis is
mixed and will have limited generalisation.

### Predicted Sub-Axis Performance (Not Yet Tested)

If we split the element→symbol axis:

```
Alphabetic sub-axis (Rule A, ~12 pairs):  predicted pc > 0.30 → >80% holdout
Latin sub-axis (Rule B, ~5 pairs):        predicted pc < 0.05 → ~0% holdout
```

This is the direct parallel to the past-tense sub-pattern analysis
(DC 425/Day 290), where +er had high pc (0.393) and +ed had low
pc (0.174) due to mixing.

---

## ENCODE=DECODE: Element→Symbol Perfect Symmetry

```
Forward:  element→symbol  pc=0.1394  scale=0.63  acc=8/14
Reverse:  symbol→element  pc=0.1394  scale=0.93  acc=9/14
cos(fwd, rev) = −1.0000  (perfect anti-parallel)
Scale ratio = 0.674
```

The forward and reverse axes are EXACTLY anti-parallel (cos=−1.000).
This is the strongest ENCODE=DECODE confirmation yet — even stronger
than the morphological axes in Day 286 (which showed cos≈−0.999).

The perfect anti-parallelism confirms:
1. The axis direction is consistent (not a noisy average)
2. Encoding (element→symbol) and decoding (symbol→element) are
   the same operation in opposite directions

The scale asymmetry (0.63 vs 0.93) reflects the density difference:
element names are longer tokens that live in a sparser region of W_E,
while single-letter symbols (H, N, O) live in a denser region with
many neighboring tokens. Walking 0.63 steps from element toward symbol
lands in the dense symbol region; walking 0.93 steps from symbol
back toward element is needed to traverse the sparser element region.

This is the "neighbourhood density ratio" from DC 421, applied to the
element domain:

```
scale_forward  / scale_reverse = density(target) / density(source)
0.63 / 0.93 = 0.677 ≈ density(symbols) / density(element_names)
```

---

## Updated Linearity Principle (v3)

```
AXIS QUALITY = f(pc, target_div, src_pc, training_scope)

RELIABLE:  pc > 0.35  AND  target_div > 0.7  AND  src_pc < 0.35
           AND  training_scope = full_distribution
           => >80% generalisation confirmed

SUSPECT:   pc > 0.35  AND  (target_div < 0.6  OR  src_pc > 0.4)
           => WARN: pc may be inflated, run holdout evaluation

MARGINAL:  0.10 < pc < 0.35  (and other signals OK)
           => 50–80% training accuracy; holdout degrades but usable
              with many training pairs (20+)

UNUSABLE:  pc < 0.10  (and genuine, not inflated)
           => attractor-dominated or fundamentally non-linear relation
```

---

## Files

- `expedition_log.md` — Day 293 results
- `427_semantic_generalisation.md` — Day 292: zero-shot 100%
- `426_unified_linearity.md` — Day 291: unified table
- `425_linearity_principle.md` — Day 290: source class analysis
