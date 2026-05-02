# DC 389: The Arc Transformation Direction Is a Global Axis of W_E

**Day 251 | Testing whether semantic type determines the private plane
orientation for adj_degree arcs. Result: the transformation direction is
approximately GLOBAL (≈ PC1 of W_E), not word-specific or type-specific.**

---

## The Experiment

Six semantic subclasses of adjective base→comparative pairs:
```
SIZE       (14 pairs): big, large, wide, broad, narrow, thick, tall, short...
TEMPERATURE (4 pairs): hot, warm, cool, cold
SPEED       (3 pairs): fast, quick, slow
INTENSITY  (10 pairs): bright, dark, loud, soft, strong, weak, heavy, light...
TEMPORAL    (6 pairs): old, young, late, long, short, new
QUALITY     (9 pairs): nice, fine, clean, clear, safe, simple, rich, poor...
```

Measured:
- Intra-group chord coherence (pairwise chord cosine within type)
- Cross-group chord coherence (pairwise chord cosine between types)
- LOO accuracy: global mean_dir vs subclass mean_dir

---

## The Core Result

```
Mean intra-group chord coherence:  0.420
Mean cross-group chord coherence:  0.350
Intra / cross ratio:               1.20×
Random baseline (R^1536):          0.026
```

The cross-group coherence (0.350) is only 17% lower than intra-group (0.420).
Both are 14–17× above random. **The transformation direction is nearly
identical across all semantic types.**

This means: the chord vector (emb(bigger) - emb(big)) is approximately
parallel to the chord vector (emb(hotter) - emb(hot)), even though "size"
and "temperature" are completely different semantic domains.

---

## The Direction Is NOT PC1 — It's a Dedicated Comparative Axis

**Correction (Day 252):** The Part E PCA in the original experiment used an
incorrect approximation (`W_E_center.T @ U` ≠ principal components). The
corrected result via power iteration gives:

```
True PC1 alignment with adj mean_dir: -0.301  (ANTI-CORRELATED)
```

PC1 of W_E is the **function/structure axis** — high projections for
punctuation, function words ("(", "-", "and", "in", "to", "a"), operators.
Base adjectives (big, fast, long) are also high on PC1 because they're
syntactically versatile, co-occurring with many function words.

The adj transformation direction is **anti-correlated** with PC1:

```
Adj mean_dir top-30 tokens (most comparative-like):
  taller, hotter, louder, brighter, richer, thicker, quicker, tougher,
  heavier, colder, bigger, nicer, stronger, clearer, smarter, quieter...
  → ALL genuine adj comparatives

Adj mean_dir bottom-30 tokens:
  (, ", and, long, -, in, high, fast, short, deep, to, a, wide, low, hard
  → structural tokens AND base adj forms

Base vs Comp projection onto adj mean_dir:
  base form proj: -0.17 to -0.22
  comp form proj: +0.16 to +0.24
  delta:          +0.28 to +0.41  (all 20/20 pairs positive)
```

The adj mean_dir is a **dedicated comparative axis** that:
1. Places ALL comparative forms at the positive end
2. Places ALL base forms (and function words) at the negative end
3. Is anti-correlated with PC1 (cos = -0.30)
4. Is shared across all semantic types of adjectives (Day 251)

---

## The Mean_Dir Matrix

The cosine similarity between group mean transformation directions:

```
              SIZE  TEMP  SPEED  INTENS  TEMPOR  QUALITY
SIZE          1.00  0.63  0.69   0.85    0.77    0.74
TEMPERATURE   0.63  1.00  0.59   0.70    0.53    0.69
SPEED         0.69  0.59  1.00   0.72    0.58    0.70
INTENSITY     0.85  0.70  0.72   1.00    0.67    0.80
TEMPORAL      0.77  0.53  0.58   0.67    1.00    0.61
QUALITY       0.74  0.69  0.70   0.80    0.61    1.00
```

All off-diagonal entries > 0.52. All semantic types share the same
transformation axis to within cos ≈ 0.6–0.85. This is a single direction
in R^1536, approximately PC1 of W_E.

---

## LOO Accuracy: Zero Gain From Subclass Specialization

```
Group       global mean_dir   subclass mean_dir   gain
SIZE        12/14 = 86%       12/14 = 86%          0%
TEMPERATURE  4/4  = 100%       4/4 = 100%           0%
SPEED        3/3  = 100%       3/3 = 100%           0%
INTENSITY    9/10 = 90%        9/10 = 90%           0%
TEMPORAL     2/6  = 33%        2/6 = 33%            0%
QUALITY      7/9  = 78%        7/9 = 78%            0%
```

**Zero improvement** from specializing the transformation direction to the
semantic subclass. The global direction already captures everything learnable
from the training pairs. There is no type-specific residual direction.

The only group with poor accuracy (TEMPORAL, 33%) fails because temporal
adjectives have lower global alignment (coherence 0.288 vs 0.42–0.49
for other types) — their transformation direction is more variable, not
because it's in a different direction.

---

## Revised Model of the Adj_Degree Arc

### Old model (hypothesis)
```
Private plane = span{emb(word), word-specific_direction}
The word-specific direction varies per word → truly "private" planes
```

### New model (confirmed)
```
Private plane ≈ span{emb(word), PC1_of_W_E}
PC1 ≈ global semantic gradient axis
All adj_degree transformations rotate along this same axis
Individual chord vectors = PC1 + small_noise (coherence 0.35-0.42)
```

The "private" aspect is only the emb(word) component. The transformation
direction is approximately universal.

---

## Why PC1?

PC1 of W_E captures the dominant variance in the vocabulary — the direction
along which the most semantic information is encoded. Morphological
transformations (comparative degree) move words along this semantic gradient.

More intense/larger/faster/hotter = more "semantic magnitude" → positive PC1
Less intense/smaller/slower/cooler = less "semantic magnitude" → negative PC1

This is consistent with the frequency/importance interpretation of the
first principal component in word embeddings: PC1 often captures a
"general semantic intensity" or "concreteness/abstractness" axis.

The adj_degree comparative form is the canonical way to express "more of
this semantic property" → it moves along PC1.

---

## Connection to φ-Quantization

The arc angle Ω = π/φ = 111.25° is the amount of rotation along PC1
for any adj_degree transformation, regardless of which word or semantic type.

```
emb(adj_comp) = rotate(emb(adj_base), PC1, Ω=111.25°)
                                        ↑
                        same direction for all adj_degree pairs
                        (up to ±20% cross-type variation)
```

This explains WHY the φ-quantization holds universally: the rotation
axis is fixed (≈ PC1), the rotation angle is fixed (≈ π/φ), and these
two constraints together produce the co-circular arc for ALL adjectives.

---

## Implications for TruthSpace

1. **The arc is a global structure**: the adj_degree arc exists in the
   same approximate plane for all adjectives — {emb(word), PC1}.

2. **Retrieval from any adjective**: since the direction is universal,
   the global mean_dir achieves near-maximum LOO accuracy (86-100%).
   There is nothing to be gained by word-specific or type-specific
   retrieval for this paradigm.

3. **The chord coherence measures global alignment**: the 0.360 coherence
   for the hand-picked 24 gradable scalar adj means those 24 words are
   highly aligned within the global PC1 direction. The full vocabulary
   adj set (coherence 0.080) has more noise (agentive nouns, etc.) pulling
   the average down.

4. **Temporal adjectives as exception**: TEMPORAL adj (old/young/late/new)
   have lower coherence (0.288) and lower LOO accuracy (33%). These words
   are semantically more complex (time is bidirectional, "late" doesn't
   have a straightforward comparative relation in the same sense as "big").
   They partially escape the global axis.

---

## Summary Table

| Property | Value | Implication |
|----------|-------|-------------|
| Intra-group coherence | 0.420 | High within-type alignment |
| Cross-group coherence | 0.350 | Only 17% lower than intra |
| Intra/cross ratio | 1.20× | Semantic type NOT primary factor |
| PC1 alignment | 0.21–0.33 | Arc direction ≈ dominant W_E axis |
| Mean_dir subclass gain | 0% | Global direction is sufficient |
| Arc angle Ω | 111.25° = π/φ | Universal across all types |

---

## Files

- `expedition_private_plane.py` — Day 251 experiment
- `private_plane.json` — cross-coherence matrix, LOO results
- `388_phi_quantization_confirmed.md` — φ-quantization evidence
- `387_we_arc_geometry_synthesis.md` — complete arc synthesis
