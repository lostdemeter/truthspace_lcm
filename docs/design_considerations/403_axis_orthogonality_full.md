# DC 403: Full Inter-Axis Orthogonality — 14 Axes

**Day 268 | Expanding the Day 261 orthogonality analysis from 6 to 14 axes
(5 morphological + 3 encyclopedic + 5 domain antonyms + 1 hypernym).
60.4% of 91 axis pairs are near-orthogonal (|cos| < 0.05). Three structural
clusters emerge: an inflectional cluster, an encyclopedic cluster, and
isolated axes. SVD shows the 14 axes span ~14 independent dimensions —
the space is approximately full-rank at this scale.**

---

## Full Cosine Matrix

```
                  adj_deg  super   plural  past    gender  capital  curren  languag  speed  temp   size   quality  social  hypern
morph:adj_degree   1.000   0.469   0.188   0.164   0.056   0.026  -0.060  -0.041   0.183  0.044  0.122   0.160   0.057  -0.020
morph:superlative  0.469   1.000   0.190   0.167   0.066   0.032  -0.094  -0.041   0.170  0.081  0.152   0.159   0.046  -0.012
morph:plural       0.188   0.190   1.000   0.176   0.062   0.016  -0.060  -0.016   0.076  0.028  0.001   0.070   0.026  -0.000
morph:past_tense   0.164   0.167   0.176   1.000   0.026   0.003  -0.047  -0.015   0.047  0.028  0.006   0.083   0.083  -0.033
morph:gender       0.056   0.066   0.062   0.026   1.000  -0.006  -0.043  -0.015   0.010 -0.004  0.023   0.018  -0.003  -0.016
enc:capital        0.026   0.032   0.016   0.003  -0.006   1.000   0.156   0.294  -0.021 -0.013 -0.006  -0.028  -0.039  -0.011
enc:currency      -0.060  -0.094  -0.060  -0.047  -0.043   0.156   1.000   0.269  -0.026  0.025 -0.060  -0.047  -0.033  -0.007
enc:language      -0.041  -0.041  -0.016  -0.015  -0.015   0.294   0.269   1.000  -0.057 -0.012 -0.042  -0.042   0.019  -0.028
ant:speed          0.183   0.170   0.076   0.047   0.010  -0.021  -0.026  -0.057   1.000  0.068  0.065   0.157   0.090   0.014
ant:temperature    0.044   0.081   0.028   0.028  -0.004  -0.013   0.025  -0.012   0.068  1.000  0.035   0.048   0.070   0.021
ant:size           0.122   0.152   0.001   0.006   0.023  -0.006  -0.060  -0.042   0.065  0.035  1.000   0.123  -0.053   0.004
ant:quality        0.160   0.159   0.070   0.083   0.018  -0.028  -0.047  -0.042   0.157  0.048  0.123   1.000   0.050  -0.009
ant:social         0.057   0.046   0.026   0.083  -0.003  -0.039  -0.033   0.019   0.090  0.070 -0.053   0.050   1.000  -0.003
sem:hypernym      -0.020  -0.012  -0.000  -0.033  -0.016  -0.011  -0.007  -0.028   0.014  0.021  0.004  -0.009  -0.003   1.000
```

---

## Three Structural Clusters

### Cluster 1: Inflectional (morphological) axes

```
Pairs within cluster:
  adj_degree ↔ superlative   cos = +0.469  ★ strongest alignment
  plural     ↔ superlative   cos = +0.190
  adj_degree ↔ plural        cos = +0.188
  past_tense ↔ plural        cos = +0.176
  past_tense ↔ superlative   cos = +0.167
  adj_degree ↔ past_tense    cos = +0.164
  Within-cluster mean |cos| = 0.156
```

These four inflectional axes share a common "inflectional marking" component.
The direction encodes: "this word has been grammatically modified from its
base form". This is the same shared component identified in DC 396.

`gender` is isolated from this cluster (mean cos with the four = ~0.04).
Gender is a **derivational** axis (producing a new lexical item), not an
**inflectional** one (marking grammatical features of an existing item).
The distinction derivational vs inflectional is geometrically real.

### Cluster 2: Encyclopedic (country) axes

```
Pairs within cluster:
  capital  ↔ language   cos = +0.294  ★ strongest
  currency ↔ language   cos = +0.269
  capital  ↔ currency   cos = +0.156
  Within-cluster mean |cos| = 0.240
```

All three axes encode "country → {national property}":
- country → capital city
- country → national language
- country → currency

The shared component is a "country axis": a direction in W_E that
discriminates between word embeddings of countries vs non-countries.
Each encyclopedic axis = country_axis + domain_specific_component.

This is a **second-order structure**: a common high-level axis that
underlies multiple encyclopedic axes.

Encyclopedic axes are near-orthogonal to morphological axes (mean |cos|
= 0.034), confirming that grammatical and factual knowledge occupy
independent geometric subspaces.

### Cluster 3: Isolated axes

```
morph:gender        — no cluster affinity
ant:speed/temp/size/quality/social  — weakly clustered (mean 0.076)
sem:hypernym        — isolated from ALL other 13 axes
```

The antonym axes cluster weakly among themselves (mean |cos| = 0.076),
with the strongest inter-antonym alignment:
- speed ↔ quality: cos = +0.157
- quality ↔ size: cos = +0.123
- speed ↔ size: cos = +0.065

These share an "intensity-scaling" component (stronger/faster/larger
all point in related directions).

`sem:hypernym` is geometrically isolated — it has |cos| < 0.035 with
every other axis. The "specific-to-general" transformation occupies a
unique region of W_E.

---

## Cross-Category Orthogonality

```
Category A   Category B   Mean |cos|   Interpretation
────────────────────────────────────────────────────────────────────
enc          enc          0.240        tight cluster (country axis)
morph        morph        0.156        loose cluster (inflectional axis)
ant          ant          0.076        very loose (intensity component)
ant          morph        0.067        ant:speed shares adj-intensity
enc          morph        0.034        near-orthogonal
ant          enc          0.031        near-orthogonal
morph        sem          0.017        essentially orthogonal
enc          sem          0.015        essentially orthogonal
ant          sem          0.010        essentially orthogonal
```

**The hypernym axis (sem) is essentially orthogonal to everything.**
This means the `specific-to-general` transformation is encoded in a
dimension of W_E that is completely independent of all morphological,
encyclopedic, and antonym dimensions.

Encyclopedic and morphological knowledge are near-orthogonal (0.034),
confirming that W_E uses different dimensions for grammatical vs factual
knowledge.

---

## SVD: Dimensionality of the Axis Subspace

```
Normalised singular values: [0.103, 0.088, 0.077, 0.076, 0.073,
                              0.072, 0.070, 0.068, 0.067, 0.066, ...]
Cumulative variance:
  80% explained by: 11 dimensions
  95% explained by: 14 dimensions  (= full rank)
```

The nearly FLAT singular value spectrum is significant:
- No single axis dominates
- Each of the 14 axes contributes roughly equally to the subspace
- The 14 axes span approximately 14 independent directions

The near-full-rank result means: **these 14 semantic relations encode
14 different things**. They are not redundant representations of the
same information. Each relation type occupies its own geometric niche
in W_E.

---

## Implications for TruthSpace

### 1. Grammatical and factual knowledge are orthogonal subspaces

The mean |cos| between morphological and encyclopedic axes is 0.034.
This near-zero value means: knowing whether a word is in plural form
gives essentially no information about what country's capital it is.
The grammatical and factual dimensions of word meaning are stored in
orthogonal sub-dimensions of W_E.

### 2. W_E has hierarchical axis structure

The encyclopedic cluster reveals a hierarchy:
- Level 1: country axis (shared across capital/language/currency)
- Level 2: domain-specific axis (capital direction, language direction, etc.)

Any encyclopedic relation about countries is: country_axis + domain_axis.
This predicts that a new encyclopedic relation (e.g., country → continent)
would align with the country cluster (cos ~0.2) while remaining partially
distinct.

### 3. The 14 axes are not the complete picture

The flat SVD spectrum and near-full-rank result mean we are far from
exhausting the axes in W_E. The model has 1536 dimensions, and we have
found axes in 14 of them. The remaining ~1522 dimensions encode further
semantic information not yet characterised.

Each systematic, bijective relation in human language and knowledge is
likely encoded as a further axis. TruthSpace can in principle discover
and use all of them.

---

## Files

- `expedition_log.md` — Day 268 results
- `396_axis_orthogonality.md` — original 6-axis analysis (Day 261)
- `401_semantic_relation_axes.md` — full relation taxonomy
- `402_antonym_domain_axes.md` — domain antonym axes
