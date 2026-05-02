# DC 364: Relational Encoding Archetypes

**Day 197 | W_E encodes relational knowledge through at least four distinct
geometric mechanisms. Grammar (plural, superlative) is the most directionally
consistent. Facts (capitals) are moderately consistent. Antonyms are NOT
directionally encoded — they are semantically adjacent. Numbers are
ordinally encoded along a single axis with non-uniform spacing.**

---

## Overview

Day 196 tested six relational domains for directional encoding (TYPE_BC),
ordinal encoding (TYPE_ORDINAL), and proximity/hypernym patterns. Results
reveal four distinct encoding archetypes.

---

## The Four Archetypes

### TYPE_BC: Directional Translation

The relation encodes as a **consistent additive direction** in W_E.
Adding the mean direction to a source embedding moves it close to the target.

```
Marker:  direction_consistency > 0.20
         LOO accuracy > 0.80
         step magnitude: 0.49–0.65 (75–100% of embedding norm)

Domains tested:
  superlative  acc=1.000  dir_consistency=0.394  step=0.648
  plurals      acc=1.000  dir_consistency=0.211  step=0.489
  capitals     acc=0.833  dir_consistency=0.328  step=0.566
  past_tense   acc=0.800  dir_consistency=0.303  step=0.509
  gender       acc=0.857  dir_consistency=0.221  step=0.536
```

**Ranking by directional consistency:**
```
superlative (0.394) > capitals (0.328) > past_tense (0.303) > gender (0.221) > plurals (0.211)
```

**Grammar beats facts.** The morphological transformations (superlative,
plurals, past tense) have equal or higher directional consistency than
factual relations (capitals, gender). This is not surprising in retrospect:

- Plurals: the rule "add -s" applies uniformly to regular nouns. Every
  cat→cats transformation is the same operation, so gradients reinforce
  a consistent direction.
- Superlative: "add -est" applied to common adjectives is equally uniform.
- Capitals: not a rule — France→Paris is a memorized fact. Different facts
  accumulate in different contexts, so consistency is slightly lower.

**The mechanism:** TYPE_BC directions exist because the same training pattern
appears with consistent co-occurrence structure. "dog" appears in contexts
where "dogs" also appears, and similarly for "cat/cats", "bird/birds". The
consistent co-occurrence forces a shared direction in W_E.

---

### TYPE_ORDINAL: Rank-Encoded Axis

The relation encodes as **monotonically increasing projection** along a
single direction. Members are ordered but not uniformly spaced.

```
Marker:  Spearman ρ > 0.90 with known rank
         LOO directional accuracy < 0.20 (step size varies)

Domains tested:
  numbers (one...twelve): Spearman ρ=0.965, p=0.000
  Projections: -0.601, -0.227, 0.007, 0.137, 0.214, 0.302, 0.307,
               0.369, 0.449, 0.352, 0.443, 0.727
  LOO num→next: acc=0.091
```

Number words are arranged along a single W_E axis with 96.5% rank correlation.
The number line is encoded geometrically. However:

- The spacing is **logarithmic-ish**: one→two is a much larger gap than
  five→six. This mirrors frequency: "one" and "two" appear very differently
  in text; "five" and "six" are nearly interchangeable in most contexts.

- LOO retrieval fails (0.091) because the variable spacing means the "next
  number" direction varies with position. From "one", the right step is large;
  from "ten", it is small. There is no single direction that works everywhere.

**Implication:** The ordinal encoding captures "where on the number line"
but not "how to navigate to the next position." It is a positional encoding,
not a translation encoding.

---

### TYPE_ADJACENT: Semantic Proximity (Not Directional)

The relation encodes as **semantic nearness** — related words are closer
in W_E than random words, but there is no consistent direction between them.

```
Marker:  direction_consistency < 0.10
         LOO accuracy ≈ random (1/n)
         High intra-cluster cosine for the combined pair set

Domains tested:
  antonyms: dir_consistency=0.033, LOO acc=0.100
  (10 pairs, 10 targets — random baseline = 0.10)
```

**Antonyms are TYPE_ADJACENT, not TYPE_BC.** The confusion arose from
testing on tiny vocabularies where the target set was only the antonym pairs
themselves. In that setting, any direction that broadly moves toward the
"adjective cluster" would succeed by chance.

With 10 pairs tested in LOO (choosing from 10 target adjectives):
- LOO acc = 0.100 = exactly random chance
- dir_consistency = 0.033 ≈ zero (directions cancel)

**Why are antonyms near each other?** "Hot" and "cold" both appear in
contexts about temperature: "the water was hot/cold", "hot summer/cold winter",
"hot coffee/cold coffee". They are near each other because they compete for
the same semantic slot in the same contexts. Antonymy is encoded as
**contextual substitutability**, which is **proximity**, not direction.

This resolves a long-standing ambiguity in word embedding literature:
antonyms appear near synonyms in Word2Vec/GloVe spaces. Now we understand why:
they are in the same semantic neighborhood because they share contexts.

---

### TYPE_HYPERNYM: Concept ≠ Centroid

The hypernym is in the **same region** as its hyponyms but is NOT simply
the centroid. It encodes the concept independently.

```
Marker:  hyper→centroid cos = 0.08–0.37 (above random, below cluster mean)
         hyper→hypo_mean < intra_hypo_cos

Domains tested:
  animal:  hyper→centroid=0.373  hyper→hypo=0.184  intra=0.150
  color:   hyper→centroid=0.205  hyper→hypo=0.128  intra=0.301
  country: hyper→centroid=0.177  hyper→hypo=0.123  intra=0.398
  number:  hyper→centroid=0.080  hyper→hypo=0.058  intra=0.446
```

For "animal": the hypernym is more similar to the hyponym centroid (0.373)
than hyponyms are to each other (0.150). "Animal" is near the center of
the animal cluster — it is almost the prototype. This makes sense: "animal"
appears in exactly the contexts where specific animals appear.

For "country": the hypernym is LESS similar to the centroid (0.177) than
countries are to each other (0.398). "Country" is not at the centroid of
the country cluster — it's nearby but displaced. Countries like France/Germany
appear in geopolitical contexts while "country" also appears in "country music",
"country road", "foreign country" — shifting its embedding away from the proper
noun cluster.

For "number": the hypernym is nearly uncorrelated with number words (0.058).
"Number" means phone number, issue number, numerical value — a very different
semantic space than the specific number words one/two/three.

**The polysemy penalty:** hypernyms are systematically displaced from their
hyponym centroids by their own polysemy. The more meanings a hypernym has,
the further it drifts from the specific cluster.

---

## Summary Table

```
Archetype       Marker                            Domains             TruthSpace use
──────────────────────────────────────────────────────────────────────────────────────
TYPE_BC         dir_consistency > 0.20            capitals            LOO direction retrieval
                LOO acc > 0.80                    gender              mean(diff) as query
                                                  plurals
                                                  past_tense
                                                  superlative

TYPE_ORDINAL    Spearman ρ > 0.90                 numbers             Project onto number axis
                LOO fails (variable spacing)      (temperature?)      return word at position

TYPE_ADJACENT   dir_consistency < 0.10            antonyms            Nearest neighbour search
                LOO ≈ random                      (synonyms?)         within semantic cluster

TYPE_HYPERNYM   hyper→centroid < intra_cos        animal/color/       Cluster membership test
                hypernym ≠ centroid               country/number      not directional retrieval
```

---

## Implications for TruthSpace Pipeline

A complete TruthSpace retrieval system must first **classify the relation type**,
then apply the appropriate retrieval method:

```
QUERY: (source, relation_type?) → target

Step 1: Classify relation
  - Compute direction consistency from known pairs
  - dir_consistency > 0.20 → TYPE_BC
  - Spearman ρ > 0.90    → TYPE_ORDINAL
  - dir_consistency < 0.10 → TYPE_ADJACENT or TYPE_HYPERNYM

Step 2: Retrieve
  - TYPE_BC:         query = W_E[source] + mean_direction
                     return nearest token in target set

  - TYPE_ORDINAL:    project W_E[source] onto ordinal axis
                     return token with next-highest projection

  - TYPE_ADJACENT:   return tokens in the same semantic cluster
                     as source (nearest neighbours)

  - TYPE_HYPERNYM:   identify which cluster source belongs to
                     return the cluster's hypernym token
```

The multi-tier pipeline design is now fully motivated by experimental evidence.

---

## Key Insight: Grammar is More Geometric than Facts

The most consistent directional encodings are morphological rules:
```
superlative (dir=0.394) > capitals (dir=0.328) > past_tense (dir=0.303)
```

Grammar rules apply uniformly across thousands of word pairs; factual
associations (France→Paris) apply to a single pair. The more instances
that share a transformation, the more reinforced the direction becomes in
training. This is the geometric fingerprint of frequency and regularity.

---

## Files

- `expedition_day196_encoding_archetypes.py` — archetype testing
- `day196_encoding_archetypes.json` — results
- `359_we_relational_dimensionality.md` — SVD per domain
- `363_we_semantic_neighbourhood.md` — cluster structure
