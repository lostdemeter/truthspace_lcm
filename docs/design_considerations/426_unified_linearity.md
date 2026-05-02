# DC 426: Unified Linearity Principle — Pairwise Chord Cosine Predicts Axis Quality Across All Domains

**Day 291 | The pairwise chord cosine (pc) predicts retrieval accuracy
across both morphological and semantic axes. Unified table of 13 axes:
country→demonym (pc=0.563) and country→lang (pc=0.474) rank above all
morphological axes except +est/+er, because country-to-demonym is a
near-morphological operation. country→capital (pc=0.317) achieves 100%
accuracy — the most geometrically linear factual relation tested.
word→antonym (pc=0.020) achieves 0% on holdout — antonyms have no
consistent geometric direction. All axes with pc>0.35 achieve >85%
accuracy on both training and holdout. corr(pc, accuracy)=0.48 across
both domains. Source embedding homogeneity does NOT predict accuracy
directly (corr=−0.11); chord homogeneity (pc) is the correct metric.**

---

## The Unified Table

```
Rank  Axis               pc_cos  coherence  accuracy  domain
1     country→demonym    0.563   0.799      100%      SEMANTIC
2     country→lang       0.474   ~0.75       89%      SEMANTIC
3     +est (sup)         0.436   0.698      100%      MORPH
4     +er (comp)         0.393   0.656      100%      MORPH
5     country→capital    0.317   0.605      100%      SEMANTIC
6     animal→class       0.254   0.599      100%      SEMANTIC
7     person→nat         0.246   0.574       56%      SEMANTIC
8     past_irr           0.230   0.527      100%*     MORPH
9     gender             0.213   0.527      100%*     MORPH
10    +ed (past_r)       0.174   0.473      100%*     MORPH
11    +s (plural)        0.155   0.454      100%*     MORPH
12    field→concept      0.087   0.404       25%      SEMANTIC
13    word→antonym       0.020   0.291        0%†     SEMANTIC
```

*100% on training pairs; holdout degrades (see DC 424, Day 289).
†0% on 12-word holdout; 67% on 15-word training due to attractor.

### The Threshold Structure

| pairwise_cos | category | generalisation |
|---|---|---|
| > 0.35 | HIGH | >85% on training AND holdout |
| 0.15 – 0.35 | MEDIUM | ~100% training; holdout varies (58–93%) |
| < 0.10 | LOW | training attractor-dominated; holdout ~0% |

Every axis with pc > 0.35 achieves high accuracy on both training data
and holdout data. Every axis with pc < 0.10 either has an attractor
dominating the training predictions or fails completely on holdout.

---

## Surprise: Semantic Axes Are More Linear Than Morphological

The top-ranked axes in the unified table are SEMANTIC, not morphological:

```
country→demonym   pc=0.563   RANK 1
country→lang      pc=0.474   RANK 2
```

Both outrank +est (0.436) and +er (0.393). Why?

### The Country Cluster Effect

Country names form an unusually tight semantic cluster in W_E:
- All are proper nouns
- All appear in similar distributional contexts (news, geography,
  geopolitics)
- All are single BPE tokens in Qwen2
- Their embeddings are densely clustered around the "country" region
  of W_E

When source words are tightly clustered, their chord vectors point in
consistent directions → HIGH pairwise cosine → HIGH linearity.

The country→demonym transformation is additionally near-morphological:
- France → French (suffix -ch)
- Germany → German (suffix -n)
- Spain → Spanish (suffix -ish)
- Italy → Italian (suffix -ian)

This regularity in the suffix pattern means the displacement vectors
are even more consistent than a random factual relation.

### country→capital (pc=0.317, 100%)

The capital city relation achieves 14/14 on training:
France→Paris, Germany→Berlin, Spain→Madrid, Italy→Rome, Japan→Tokyo,
China→Beijing, Russia→Moscow, Egypt→Cairo, India→Delhi, Turkey→Ankara,
Greece→Athens, Poland→Warsaw, Sweden→Stockholm, Norway→Oslo.

This is a completely factual (non-morphological) relation, yet it is
more linear than any past-tense axis (pc=0.17–0.23). Why?

Because:
1. Country names cluster tightly (as noted above)
2. Capital cities also cluster tightly (all are major proper nouns,
   all appear in similar distributional contexts)
3. The country→capital displacement is **geographically structured**:
   European capitals are systematically placed relative to their
   country names in W_E

The geometry of W_E encodes not just morphological relations but
geographic/political relations that are consistently expressed in text.

---

## The Antonym Axis: A Structural Impossibility

word→antonym achieves pc=0.020 — near-zero pairwise cosine.

```
Training: hot→cold, fast→slow, big→small, strong→weak, light→dark,
          high→low, old→young, rich→poor, happy→sad, love→hate,
          war→peace, good→bad, start→end, open→close, push→pull
Training accuracy: 67% (but 'poor' is attractor — it fires for
                        old, good, start, open; these pairs fail)
Holdout accuracy:  0/12 (0%)
```

### Why Antonyms Are Not a Linear Axis

The displacement from a word to its antonym depends on:
1. **Which semantic dimension is being reversed**: hot/cold reverses
   temperature; fast/slow reverses speed; good/bad reverses morality
2. **The magnitude of the reversal**: some antonyms are close in W_E
   (hot/cold); others are far (love/hate)
3. **The direction in W_E**: each semantic dimension has its own axis

There is no single direction in W_E that transforms all words to their
antonyms. The "antonym axis" is the mean of 15 unrelated direction
vectors — statistical noise.

The 67% training accuracy is deceptive: 'poor' appears as a nearest
neighbour for many words (it is a high-frequency word near the centre
of the W_E space), and by coincidence it is the antonym of 'rich'. The
mean axis lands somewhere near 'poor' and retrieves it for multiple
training words.

### The True Structure of Antonymy in W_E

Antonyms are not encoded as a single linear transformation. Instead:
- **Temperature antonyms**: hot↔cold, warm↔cool — one local axis
- **Size antonyms**: big↔small, large↔tiny — another local axis
- **Moral antonyms**: good↔bad, right↔wrong — another local axis

Each semantic dimension has its OWN antonym direction. The global
"antonym axis" is an average of these incompatible directions.

To handle antonymy geometrically would require:
1. Identifying the semantic dimension of the source word
2. Selecting the appropriate dimension-specific antonym axis
3. Applying it at the appropriate scale

This is fundamentally a CLUSTER problem (DC 416 applies here).

---

## Source Homogeneity vs Chord Homogeneity

A key finding from the correlation analysis:

```
corr(pairwise_cos, accuracy)    = +0.48   MODERATE POSITIVE
corr(src_homogeneity, accuracy) = −0.11   WEAK NEGATIVE
corr(src_homogeneity, pairwise) = +0.56   MODERATE POSITIVE
```

### Source homogeneity does NOT directly predict accuracy

This is counter-intuitive: we might expect that tightly clustered
source words → better axis. But the correlation is actually slightly
NEGATIVE.

Why? Because high source homogeneity does not guarantee consistent
TRANSFORMATION DIRECTIONS. Consider:

- field→concept: source words (physics, chemistry, biology...) have
  moderate-high homogeneity (all are field names), but each field maps
  to a completely different concept. The chords point in all directions.
  src_hom=0.247 but pc=0.087.

- word→antonym: source words (hot, fast, big, strong...) are diverse
  (low homogeneity), and chord cosines are near-zero.

- +s (plural): source nouns are very diverse (low homogeneity=0.072),
  but the +s transformation is consistent → pc=0.155.

### What predicts accuracy: chord homogeneity

The pairwise cosine between CHORD VECTORS (pc) directly measures
whether all transformations point in the same direction — regardless
of where the source words are.

pc high → all pairs transform in the same direction → axis is valid
pc low  → pairs transform in different directions → axis is noise

src_homogeneity can CORRELATE with pc (if tightly clustered sources
happen to have consistent transformations), but it is not the causal
factor.

---

## Design Rule for TruthSpace Geometric Parser

Based on the unified linearity principle, axis viability is:

```
VIABLE:    pc > 0.30   (use geometric axis retrieval)
MARGINAL:  pc 0.15-0.30 (use axis + verification)
UNUSABLE:  pc < 0.10   (use lookup table or cluster decomposition)
```

### Viable Axes (ready for deployment)

1. country→demonym (0.563) — near-morphological, generalises well
2. country→language (0.474) — similar properties
3. superlative (0.436) — any adjective, perfect generalisation
4. comparative (0.393) — any adjective, perfect generalisation
5. country→capital (0.317) — high accuracy, likely generalises

### Marginal Axes (needs testing)

6. animal→class (0.254) — 100% train, holdout unknown
7. person→nat (0.246) — 56% train accuracy, cluster-aware improves
8. irregular past (0.230) — 100% train, poor holdout
9. gender (0.213) — 100% train, suppletive pairs fail on holdout
10. regular past (0.174) — 100% train, 58% holdout
11. plural (0.155) — 100% train, 93% holdout (with 20 pairs)

### Unusable Axes (structural impossibilities)

12. field→concept (0.087) — attractor-dominated, 25% accuracy
13. word→antonym (0.020) — no consistent direction exists

---

## The Broader Principle: Consistency IS the Rule

The unified linearity principle generalises the TruthSpace core:

> **"Structure IS information"** requires that the structure be
> CONSISTENT across instances. A geometric axis can only encode a
> relation if that relation maps all source words to targets in the
> same geometric direction.

When relations are consistent (regular morphology, country demonyms,
country capitals), the geometry encodes the rule perfectly and it
generalises to any new instance.

When relations are inconsistent (antonyms, field→concept), there is
no rule to encode — only specific instances. The geometry faithfully
reflects this: the chord vectors point in all directions because the
relation itself is not directional.

The linearity principle (pairwise_cos) is a direct MEASUREMENT of
whether a relation "has structure" in the TruthSpace sense.

---

## Files

- `expedition_log.md` — Day 291 results
- `425_linearity_principle.md` — Day 290 source class analysis
- `424_generalisation.md` — Day 289 holdout tests
- `423_multi_axis_composition.md` — Day 288 orthogonality
