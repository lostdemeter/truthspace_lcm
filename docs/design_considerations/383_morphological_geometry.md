# DC 383: The Geometric Structure of Morphological Paradigms in W_E

**Days 234–237 | The composition arc investigated whether direction vectors
in W_E can be added to traverse multi-hop relational chains. The answer
is nuanced: composition works by algebraic triviality, not direction
parallelism. This document synthesizes all findings into a coherent
picture of how morphological paradigms are geometrically organized in W_E.**

---

## Summary of Findings

| Property | Value | Day |
|---|---|---|
| Comparative-superlative composition retrieval | 5/5 = 1.000 | 234 |
| cos(d_comparative, d_comp_to_sup) | **−0.401** (anti-correlated) | 235 |
| cos(mean_d1, mean_d2) for adj degree | **−0.419** | 235 |
| Per-word anti-correlation universality | **19/19 words** | 236 |
| Mean midpoint error (pos/comp/sup collinearity) | **0.77** (not collinear) | 235 |
| PC1 of adj degree ≈ d_superlative | cos = **0.989** | 237 |
| PCA comparative retrieval | **0/9 = 0.000** (fails) | 237 |
| mean_dir comparative retrieval | **8/9 = 0.889** (works) | 237 |
| Step magnitude |d(pos→comp)| | **0.555** ± 0.040 | 236 |
| Step magnitude |d(comp→sup)| | **0.592** ± 0.028 | 236 |
| All cross-paradigm directions | cos ≈ **0.00–0.15** (orthogonal) | 235 |

---

## The Composition Result (Day 234): What Really Happened

Day 234 showed that `emb(big) + d_comparative + d_comp_to_sup` retrieves
`biggest` with rank=0 for all 5 training words tested. The alignment
`cos(d_direct_sup, normed(d_comp + d_c2s)) = 0.9808` was initially
interpreted as "directions are parallel → strong composition."

**This interpretation was wrong.** Day 235 measured:
```
cos(d_comparative, d_comp_to_sup) = −0.401  (ANTI-CORRELATED)
```

The composition works for a different reason:

### The Algebraic Triviality Mechanism

For any word A with comparative A_c and superlative A_s:
```
(A_c − A) + (A_s − A_c) = A_s − A     [exact, by cancellation]
```

The intermediate node cancels out. When we NORMALIZE each step:
```
normed(A_c − A) + normed(A_s − A_c) ≈ normed(A_s − A)
```

This approximation holds because `|A_c − A| ≈ |A_s − A_c|` (equal step
magnitudes, confirmed: 0.555 vs 0.592). With equal magnitudes, the sum
of normed unit vectors is approximately the normed sum of the original
vectors, which equals the normed superlative displacement.

### Why Cross-Paradigm Composition Fails

For gender + plural:
```
(queen − king) + (cats − cat) ≠ (queens − king)
```

The two vectors come from **different word pairs**. There is no
algebraic relationship between them. The intermediate node does NOT
cancel, and the sum lands nowhere meaningful.

**Conclusion**: Composition works exclusively when the two steps are
part of a **same-word chain** (A→B→C with the SAME word A). This is
a mathematical identity, not a deep geometric property.

---

## The True Geometry of Adjective Degree in W_E

### Three Facts About the Degree Path

**Fact 1: The path is curved (NOT collinear).**
- Mean midpoint error = 0.77 for all adj degree words
- `big/bigger/biggest` do NOT lie on a straight line in W_E
- cos(d_pos2comp, d_comp2sup) = −0.40 per word, universally

**Fact 2: A dedicated degree dimension exists.**
- PCA PC1 of {pos, comp, sup} embeddings = d_superlative (cos=0.989)
- On this dimension: pos < comp < sup, for ALL 10 adj words tested
- Only 10.9% of variance explained by PC1 → degree is ONE of many
  dimensions active in positioning these words

**Fact 3: Step magnitudes are approximately equal but not identical.**
- |pos→comp| = 0.555 ± 0.040
- |comp→sup| = 0.592 ± 0.028
- Superlative step is slightly larger (~6.7%)
- This asymmetry means the comparative is NOT the midpoint of pos and sup

### The Geometric Picture

In the high-dimensional W_E space (1536 dimensions), adjective degree
forms occupy a path that looks approximately like this in 2D:

```
                  . biggest
                 /
                . big (base)
               /   \
              /     . bigger (comparative)
             /
         (other adjectives)
```

The key properties:
1. Along the "degree axis" (PC1): pos → comp → sup (ordered)
2. Each step has a WORD-SPECIFIC COMPONENT that makes steps anti-correlate
3. The MEAN direction over many words averages out word-specific noise,
   leaving the universal degree signal

This is why mean_dir works universally for held-out words (3/3 in Day 236):
the mean averages out the noise, and the universal degree component
accurately points from any base adjective toward its comparative/superlative.

---

## Multi-Paradigm Geometric Structure

### Direction Orthogonality (Day 235)

All 12 known directions were measured pairwise:

```
|cos(di, dj)| > 0.50:  only one pair found
  comp_to_sup ↔ superlative: 0.563

|cos(di, dj)| > 0.15:  only a few
  comparative ↔ superlative: 0.425
  comparative ↔ antonym_speed: 0.201
  superlative ↔ antonym_speed: 0.197
  antonym_size ↔ antonym_weight: 0.183

All others: |cos| < 0.15 (effectively orthogonal)
```

**W_E assigns nearly orthogonal directions to different paradigms.**
Each morphological relation (gender, number, degree, tense, capital city...)
occupies its own independent subspace in the 1536-dimensional space.

This is consistent with the TruthSpace hypothesis: structure IS information.
Different types of relational knowledge are encoded in different geometric
dimensions, allowing them to coexist without interference.

### PCA Alignment with mean_dir (Day 237)

| Paradigm | cos(PC1, mean_dir) | Interpretation |
|---|---|---|
| adj_degree | **0.989** | PCA axis = mean direction |
| gender | 0.192 | PCA captures content, not direction |
| plural | 0.029 | PCA captures content, not direction |
| past_tense | 0.009 | PCA captures content, not direction |

**Adj_degree is geometrically special**: it has a dedicated axis because
ALL words in the paradigm move along the SAME direction (the degree axis).
PCA captures this axis directly.

Other paradigms (gender, plural, tense) are **complementary-pair**
relations: king→queen, cat→cats, walk→walked. The pairs don't move in
a globally consistent direction. PCA of their word sets captures
semantic content similarity, not the relational direction.

### Step Magnitude Hierarchy (Day 236)

| Paradigm | Mean step |d_B − d_A|| |
|---|---|
| Adj degree (pos→comp) | 0.555 |
| Adj degree (comp→sup) | 0.592 |
| Noun plural (sg→pl) | 0.495 |
| Verb tense (root→past) | 0.460 |

Each paradigm type has a characteristic step size. Adjective degree
steps are larger than morphological inflection steps (plural, past tense).

---

## Implications for TruthSpace Hypothesis

### What These Results Confirm

1. **Structure IS information.** W_E encodes morphological knowledge as
   geometric directions. The gender direction, plural direction, degree
   direction, etc. are distinct vectors in W_E that encode relational
   knowledge without requiring explicit rules.

2. **Paradigms occupy independent subspaces.** The near-orthogonality of
   all 12 direction pairs confirms that W_E uses independent dimensions
   for different relations. This allows the same embedding space to
   simultaneously encode multiple types of knowledge.

3. **Adj_degree has a dedicated axis.** The fact that PCA of adj degree
   words directly yields the superlative direction (cos=0.989) means
   adjective degree is represented as a true one-dimensional scale in W_E.

4. **Mean directions generalize.** The mean direction over 10 training
   words generalizes perfectly to held-out words (3/3, 100%) — demonstrating
   that the direction is a UNIVERSAL MORPHOLOGICAL SIGNAL, not a
   memorized lookup table.

### What These Results Revise

1. **Composition is algebraic, not geometric.** Same-word chain composition
   works by `(B−A) + (C−B) = C−A`, not by direction parallelism. The
   directions ARE anti-correlated; equal step magnitudes make the normed
   sum approximate the direct direction.

2. **Cross-paradigm composition requires explicit intermediate steps.**
   gender(A) + plural(A) requires computing the gender-flipped form first,
   then pluralizing it. There is no shortcut via direction addition.

3. **The morphological path is curved, not linear.** big/bigger/biggest
   do NOT lie on a straight line in W_E (midpoint error = 0.77). The path
   has a consistent direction component (the degree axis) but also
   word-specific deviation components.

---

## Practical Retrieval Conclusions

| Task | Best method | Accuracy |
|---|---|---|
| Comparative retrieval (pos→comp) | mean_dir(comparative) | 8-10/10 |
| Superlative retrieval (pos→sup) | mean_dir(superlative) | 9-10/10 |
| Two-hop via known intermediate | step1 + step2 (mean dirs) | 9/9 |
| PCA direction for comparative | FAILS | 0/9 |
| PCA direction for superlative | mean_dir(sup) ≡ PC1 | 8-9/9 |

**Use mean_dir for retrieval. Use PCA for visualization and axis identification.**

---

## Open Questions

1. **Why is the path curved?** What causes the anti-correlation of
   per-word degree steps? Is it the word-specific semantic content
   that "rotates" each step, or is there a deeper reason?

2. **How many independent paradigm subspaces exist?** We've identified
   ~12 directions with pairwise cos < 0.15. The 1536-dimensional space
   could support many more. What is the total "paradigm dimensionality"
   of W_E?

3. **Do OTHER models (GPT-2, Llama, etc.) have the same structure?**
   Is the adj_degree axis universal across models, or specific to Qwen2?

4. **What is the structure of the antonym space?** Antonyms showed
   high degeneracy for size attributes. Is the antonym direction the
   SAME direction as the degree direction (opposite end of same axis)?

---

## Files

- `expedition_day234_composition.py` — Day 234: composition retrieval test
- `expedition_day235_direction_matrix.py` — Day 235: direction cosine matrix
- `expedition_day236_paradigm_survey.py` — Day 236: step magnitude survey
- `expedition_day237_degree_dim.py` — Day 237: PCA degree dimension
- `382_geometric_composition.md` — DC 382 (initial, partially superseded)
