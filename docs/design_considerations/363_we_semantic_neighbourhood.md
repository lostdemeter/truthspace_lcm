# DC 363: W_E Semantic Neighbourhood Structure

**Day 195 | Semantic categories form clusters of varying tightness in W_E.
Proper nouns (countries, numbers) are tight; abstract words (adjectives,
verbs, body parts) are diffuse. The relational step (country→capital) is
large (86% of embedding norm) but consistent (4% variation). CJK
translations co-locate with English words in the semantic subspace.**

---

## Overview

Day 194 measured intra- and inter-cluster cosine similarity for eight
semantic categories, and characterized the relational step size relative
to cluster diameter for the country→capital domain.

---

## Finding 1: Categorical Cluster Tightness Correlates with Semantic Specificity

```
Category         n    Mean pairwise cos    Centroid norm    Character
────────────────────────────────────────────────────────────────────────
numbers          15   0.335  ± 0.222       0.616            tight, high-var
countries        19   0.321  ± 0.068       0.598            tight, consistent
capitals         16   0.254  ± 0.062       0.549            moderate
colors           15   0.247  ± 0.095       0.545            moderate
animals          16   0.153  ± 0.060       0.454            loose
body_parts       16   0.100  ± 0.059       0.395            diffuse
common_verbs     16   0.106  ± 0.057       0.403            diffuse
adjectives       16   0.102  ± 0.066       0.398            diffuse
```

**The pattern:** categories where members have clear, distinct identities
(country names, number words) form tight clusters. Categories defined by
relationships or properties rather than intrinsic identity (body parts,
verbs, adjectives) are diffuse.

**Why numbers are tight but high-variance:** The number words "one" through
"twelve" are close to each other, but "hundred" and "thousand" are in a
different sub-region. Within-number distances span from very tight (1-3)
to moderate (1-1000), creating high variance.

**The centroid norm correlation:** tight clusters have high centroid norm
(the centroid is close to the surface of the sphere because all vectors
point similarly). Diffuse clusters have low centroid norm — vectors cancel
partially, leaving a short centroid. This gives a quick cluster tightness
signal: centroid_norm ≈ mean_intra_cosine.

---

## Finding 2: Inter-Cluster Structure — A Semantic Hierarchy

```
Inter-cluster centroid cosines:
  countries  ↔ capitals:    0.401   ← same geopolitical region
  capitals   ↔ animals:     0.185
  countries  ↔ animals:     0.175
  animals    ↔ body_parts:  0.251   ← both concrete physical nouns
  colors     ↔ adjectives:  0.217   ← both descriptive
  body_parts ↔ verbs:       0.190
  countries  ↔ numbers:     0.036   ← nearly orthogonal
  countries  ↔ adjectives: -0.001   ← orthogonal
  verbs      ↔ countries:   0.060   ← nearly orthogonal
```

This creates a rough W_E semantic hierarchy:

```
W_E Semantic Regions:
  
  GEOPOLITICAL (countries + capitals): cos ≈ 0.40 between groups
  │
  CONCRETE NOUNS (animals + body_parts): cos ≈ 0.25 between groups
  │
  DESCRIPTIVE (colors + adjectives): cos ≈ 0.22 between groups
  │
  ABSTRACT (verbs + numbers): cos ≈ 0.08 between groups
  
  All cross-region: cos ≈ 0.0-0.10  (approximately orthogonal)
```

Proper nouns (geopolitical) and abstract descriptors (adjectives) are
nearly orthogonal in W_E. The space partitions naturally by semantic type.

---

## Finding 3: The Relational Step Is Large but Consistent

```
Country→Capital relational step:
  Mean step L2 magnitude:  0.558
  Step / embedding norm:   0.858  (86% of a typical embedding)
  Std of step magnitude:   0.022  (4% variation)
  
Country cluster diameter (mean pairwise cos): 0.367
Capital cluster diameter (mean pairwise cos): 0.287
Centroid(countries) ↔ Centroid(capitals):     0.443
```

The country→capital step is **large**: it moves you 86% of an embedding
norm across W_E space. Yet it is **consistent**: all 10 tested country→capital
pairs produce steps within 4% of each other in magnitude.

**What this means geometrically:**

```
            country cluster
            (mean_cos = 0.37)
           ●●●●●●●●●●●●          ──────→ step (0.558 ≈ 86% of norm)
            ●●●●●●●●●            
                                capital cluster
                                (mean_cos = 0.29)
                                        ●●●●●●●
                                         ●●●●●●●
                                          ●●●●●●

  centroid-to-centroid: cos = 0.44
```

The relational step is roughly the same size as the separation between
the two cluster centroids (centroid-to-centroid cos=0.44). The translation
moves you from the country cloud into the capital cloud, and the consistency
(4% std) ensures you land near the correct capital.

**The retrieval challenge:** since the capital cluster has mean_cos=0.29
(some internal spread), and the step has 4% variation, the nearest-capital
from a query is correct 90% of the time (from LOO experiments). The 10%
failure cases are capitals that are geometrically atypical within the cluster
(e.g., Rome, which has strong non-geopolitical associations).

---

## Finding 4: CJK Translations Co-Locate in the Semantic Subspace

```
Nearest neighbours (top-8, W_E cosine similarity):
  France → [法国(Chinese), france, French, Germany, Spain, Italy]
  Paris  → [巴黎(Chinese), paris, France, French, London]
  king   → [King, kings, 国王(Chinese king), queen]
  red    → [红(red), Red, _red, 红色(red+color), -red]
```

For every English word tested, a CJK translation appears in the top-3
nearest neighbours. Despite the script axis (PC1 from DC 362) separating
CJK from Latin tokens globally, within the semantic subspace the Chinese
translation of "France" (法国) is co-located with the English "France".

**Why?** Qwen2 was trained on multilingual text where 法国 and France
appear in the same contexts (discussing the French Revolution, French
cuisine, etc.). The relational training signal forces their embeddings
to converge semantically, independent of the script dimension. The script
axis separates them along one dimension but their semantic coordinates
in the remaining ~1530 dimensions are nearly identical.

**Implication for TruthSpace:** semantic similarity search in W_E
automatically retrieves cross-script synonyms. This is a FREE multilingual
capability without any explicit cross-lingual training signal in TruthSpace.

---

## Finding 5: The Semantic Structure Predicts Retrieval Difficulty

Connecting cluster structure to LOO retrieval accuracy:

```
Category type        Intra-cluster cos    Expected retrieval difficulty
──────────────────────────────────────────────────────────────────────
tight (countries)    0.32                 EASY — members well separated
moderate (capitals)  0.25                 MODERATE — some blending
diffuse (adjectives) 0.10                 HARD — members nearly equidistant
```

For relational retrieval to succeed, the target must be:
1. Closer to the query+direction than all other targets
2. The target cluster must be tight enough to distinguish members

With capital cluster mean_cos=0.25, there's enough inter-capital separation
for the direction step (4% std) to reliably select the right capital.
For a domain with diffuse targets (mean_cos=0.10), retrieval would require
much higher directional precision.

**This explains TYPE_BC vs TYPE_A differences:** TYPE_BC domains work because
their source and target categories form moderately-tight clusters. If both
clusters were as diffuse as adjectives (0.10), directional retrieval would fail.

---

## The W_E Semantic Geometry — A Complete Picture

Combining findings from Days 190-194 (norm arc, anisotropy arc, neighbourhood arc):

```
W_E GLOBAL STRUCTURE:
  - Approximate unit sphere (norm 0.65 ± 0.09)
  - Dominant script axis (PC1 = 72% of variance): CJK vs Latin
  - Secondary axes (PC2-5): code/script formatting
  
W_E SEMANTIC SUBSPACE (residual after script axes):
  - ~1531 effective dimensions
  - Hierarchical clustering: geopolitical > concrete > descriptive > abstract
  - Relational directions live here (cos < 0.15 with script PCs)
  
W_E CLUSTERING:
  - Proper nouns: tight (cos=0.25-0.34)
  - Concrete nouns: moderate (cos=0.15-0.25)
  - Abstract words: diffuse (cos=0.10)
  - Cross-cluster: mostly near-orthogonal (cos=0.0-0.2)
  
W_E RELATIONS:
  - Step magnitude: 0.558 ≈ 86% of embedding norm
  - Step consistency: 4% variation
  - LOO retrieval accuracy: 90% (consistent with cluster/step ratio)
  - CJK synonyms co-locate in semantic subspace
```

---

## Summary

```
Finding                                   Value
─────────────────────────────────────────────────────────────────────
Tightest cluster                          numbers (0.335) / countries (0.321)
Loosest cluster                           body parts / verbs / adjectives (0.10)
countries ↔ capitals inter-centroid cos   0.401 (same region)
countries ↔ adjectives inter-centroid cos -0.001 (orthogonal)
Relational step magnitude                 0.558 (86% of emb norm)
Relational step consistency               4% variation
CJK translation in nearest neighbours    Yes (top-3 for all tested words)
Retrieval accuracy explained by          cluster_tightness / step_variation ratio
```

---

## Files

- `expedition_day194_semantic_neighbourhood.py` — cluster analysis
- `day194_semantic_neighbourhood.json` — results
- `361_we_norm_structure.md` — norm as semantic specificity
- `362_we_anisotropy.md` — script axis + relational orthogonality
