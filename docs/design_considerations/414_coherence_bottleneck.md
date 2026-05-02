# DC 414: Coherence is the Bottleneck — Scale Optimisation is a Dead End

**Day 279 | Scale-free voting (scanning alpha in [0.05, 3.0], taking the
most common top-1 result) ties or is worse than global scale across all
three tested axes (nat: 40% vs 45%, dem: 27% vs 33%, lan: 80% vs 80%).
The three-hop sequential accuracy is 40% with both methods. The bottleneck
is not scale but axis coherence: pairs that fail at the global scale also
fail at every tested alpha because the axis direction itself is wrong for
those source embeddings. The nat axis (coh=0.499) dominates German-nearby
embeddings and systematically fails for Greek, British, and ancient-history
persons. Improving axis coherence via source-type clustering is the correct
path; scale engineering is not.**

---

## The Scale-Free Null Result

### What was tested
For each source embedding, scan 13 alpha values [0.05, 0.10, 0.15, 0.20,
0.25, 0.30, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0, 3.0]. Retrieve the top-1
nearest neighbour at each alpha. Take the plurality vote as the answer.

### Why it doesn't help
Pairs that fail at the global scale fail because the predicted direction
`normed(source_emb + alpha * axis)` never points toward the correct target
for ANY alpha. As alpha → 0, the direction converges to `normed(source_emb)`,
which retrieves the word itself. As alpha → ∞, the direction converges to
`axis`, which retrieves the word most similar to the axis direction (i.e.,
the training centroid target). Neither extreme is the correct answer for
failing pairs; no intermediate alpha lands in the correct target's Voronoi
cell.

**The axis direction is wrong for failing pairs, not the scale.**

### New hazard: CJK contamination at high alpha
At large alpha (1.5–3.0), the predicted direction converges to the axis
direction, which is biased toward European country names and their
associated tokens. In the full 152k Qwen vocabulary, some CJK tokens
have high cosine similarity with the axis direction. Mozart's scale-free
hop 2 retrieved '奥地利' (Chinese: Austria) at a high alpha — technically
correct semantically, but a different tokenisation domain.

This is a reminder that the full vocabulary NN search is polluted by
multi-lingual tokens. Restricting NN search to ASCII alphabetic tokens
would eliminate this noise at the cost of missing legitimately useful
tokens.

---

## The Coherence-Accuracy Relationship

Across all three axes in the chaining experiments:

```
Axis         Coherence    Scale-optimal acc    3-hop contribution
nat          0.499        45% (9/20)           Primary failure point
language     0.694        80% (4/5)            High reliability
demonym→cty  0.787        33% (5/15)           Despite high coherence!
```

The surprising entry is demonym→country: coherence 0.787 (high) but
accuracy only 33% (low). This contradicts the simple "coherence predicts
accuracy" hypothesis. Investigation reveals why:

**High coherence + low accuracy = axis encodes a REAL direction, but
the target embeddings have small NN cells (many near-synonyms).**

For `German → germany`, the target 'germany' has near-synonyms:
'Germany', 'german', 'Germany.', 'germany,' — multiple capitalisations
and punctuation variants occupy adjacent Voronoi cells. Even if the
prediction points exactly at 'germany', it may fall in 'German'
or 'Germany' instead.

This means the **Voronoi structure of the target** is a second factor
independent of axis coherence.

### Two-Factor Model of Axis Accuracy

```
accuracy(axis) ≈ f(coherence, target_voronoi_size)
```

Where:
- `coherence` measures how consistently the axis direction is encoded
- `target_voronoi_size` measures how large the nearest-neighbour cell
  of the target is (larger cell = easier to land in)

**High coherence, large Voronoi cell → high accuracy** (language axis)
**High coherence, small Voronoi cell → moderate accuracy** (demonym axis)
**Low coherence, any Voronoi cell → low accuracy** (nat axis)

The language axis succeeds because: (1) coh=0.694, and (2) 'french',
'german', 'japanese' are well-separated words with large Voronoi cells
(no capitalization variants, no homophones).

The demonym axis fails despite high coherence because: 'germany',
'Germany', 'german', 'Germany.' are all adjacent in W_E, splitting the
target region into small cells that the axis frequently misses.

---

## Correct Path Forward: Improve Nat Axis Coherence

The nat axis coherence (0.499) is below the threshold for reliable
sequential chaining. The cause is heterogeneous training pairs:

**Easy pairs** (within German cluster): Einstein, Marx, Kepler, Gauss,
Mozart, Freud, Kant → all German or nearby → high chord alignment
**Hard pairs** (outliers): Newton, Darwin, Caesar, Aristotle, Plato,
Tesla, Gandhi → diverse nationalities, ancient or foreign names →
low chord alignment

The heterogeneous source distribution creates a high-variance axis
direction. The mean direction is biased toward German (dominant cluster)
and is unreliable for non-German outliers.

### Fix: Source-Type Clustering Before Axis Construction

Instead of one nat axis over all person-nationality pairs, build
**nationality-specific clusters**:

```
British scientists: Newton, Darwin, Turing, Faraday, Hooke, ...
German scientists:  Einstein, Gauss, Kepler, Planck, Ohm, ...
Greek philosophers: Aristotle, Plato, Socrates, Pythagoras, ...
```

Build a SEPARATE axis from each cluster. Each cluster-axis will have
higher coherence because the source embeddings lie in a tight cluster
and all point toward the same nationality.

Then use a two-stage retrieval:
1. Identify which cluster the query person belongs to (by NN to cluster centroids)
2. Apply the matching cluster-axis

This is source-type-aware axis application — a refinement of the
universal axis approach.

---

## Summary: Engineering Constraints for TruthSpace Axis Chaining

After four days of axis chaining experiments (Days 276–279):

| Component | Status | Requirement |
|-----------|--------|-------------|
| Sequential vs additive | RESOLVED | Always use sequential |
| Capitalisation normalisation | RESOLVED | Apply post-retrieval |
| Scale sensitivity | RESOLVED (negative) | Scale-free doesn't help |
| Axis coherence threshold | IDENTIFIED | Need coh ≥ 0.65 for reliability |
| Voronoi cell size | IDENTIFIED | Target must be well-separated |
| Source-type clustering | NEXT | Split heterogeneous axes by cluster |

**The next productive axis chaining experiment:** split the nat axis by
nationality cluster, measure per-cluster coherence, and re-test 3-hop.
Predicted improvement: hop 1 accuracy from 45% to 70%+.

---

## Connection to TruthSpace Hypothesis

The coherence finding connects to the core TruthSpace principle:
**"Structure IS information"**. 

A low-coherence axis is one where the "shape" of the relation (the
geometric direction) is not consistently encoded in W_E. This means:
1. The LLM did not learn a clean geometric structure for this relation
2. The relation is either weakly represented in training data, or
3. The source type is too heterogeneous (mixing multiple sub-clusters)

High coherence = the LLM has learned a clean, consistent geometric
structure for this relation. The geometry IS the information.

Low coherence = the geometry is noisy, the information is not cleanly
encoded. No amount of scale engineering can recover a poorly encoded
relation.

---

## Files

- `expedition_log.md` — Day 279 results
- `413_scale_sensitivity.md` — scale sensitivity (Day 278)
- `412_sequential_chaining.md` — sequential >> additive (Day 277)
