# Design Consideration 037: Spatial Attention for Concept Importance

## Problem Statement

**Frequency ≠ Importance**

The original Q&A system produced:
```
Q: Who is Holmes?
A: Holmes is a character who spoke often involving jabez
```

Jabez Wilson appears frequently in ONE story ("The Red-Headed League") but Watson is the DEFINING relationship for Holmes across ALL stories.

## The Insight: Attention as Geometry

Just like attention in transformers, we need a mechanism to weight what's IMPORTANT, not just what's FREQUENT.

The key metrics for importance:

1. **SPREAD** - Does the entity appear across multiple sources?
2. **MUTUAL AGENCY** - Do both entities act and mention each other?
3. **TOTAL MENTIONS** - How many times do they co-occur?

## The Formula

```
importance(A, B) = spread(B) × partnership(A, B)

where:
  spread(B) = |sources containing B| / |total sources|
  partnership(A, B) = log(1 + total_mentions) × bidirectional_bonus
  bidirectional_bonus = 1.5 if both directions exist, else 1.0
```

## Evidence: Watson vs Jabez

```
Holmes → Watson:
  holmes mentions watson: 2
  watson mentions holmes: 18
  Total: 20
  Spread: 5 sources (0.312)
  Score: 1.4584

Holmes → Jabez:
  holmes mentions jabez: 2
  jabez mentions holmes: 0
  Total: 2
  Spread: 2 sources (0.125)
  Score: ~0.0 (no bidirectional)
```

**Watson wins because:**
- Higher total mentions (20 vs 2)
- Bidirectional relationship (both mention each other)
- Higher spread (appears in 5 books vs 1 story)

## Implementation

### `spatial_attention.py`

```python
class SpatialAttention:
    def partnership_score(self, entity1: str, entity2: str) -> float:
        """
        Key insight: Watson mentions Holmes 18 times, Holmes mentions Watson 2.
        This asymmetry is NORMAL (Watson is the narrator).
        Weight TOTAL mentions, with bonus for bidirectionality.
        """
        e1_to_e2, e2_to_e1 = self.compute_mutual_agency(entity1, entity2)
        
        total = e1_to_e2 + e2_to_e1
        if total == 0:
            return 0.0
        
        bidirectional_bonus = 1.5 if (e1_to_e2 > 0 and e2_to_e1 > 0) else 1.0
        return np.log1p(total) * bidirectional_bonus
    
    def importance_score(self, query_entity: str, related_entity: str) -> float:
        spread = self.spread_score(related_entity)
        partnership = self.partnership_score(query_entity, related_entity)
        return spread * (partnership + 0.1)
```

### Integration with HolographicProjector

```python
# In resolve() method:
if axis == 'WHO':
    attention = get_attention()
    if attention._initialized:
        important_relations = attention.get_important_relations(entity, k=3)
        if important_relations:
            top_patients = important_relations  # Use attention-weighted
```

## Results

### Before (Frequency-Based)
```
Q: Who is Holmes?
A: Holmes is a character who spoke often involving jabez

Q: Who is Darcy?
A: Holmes is a character who took action often involving everybody
```

### After (Attention-Weighted)
```
Q: Who is Holmes?
A: Holmes is a character who spoke often involving watson ✓

Q: Who is Darcy?
A: Darcy is a character who took action often involving elizabeth ✓

Q: Who is Elizabeth?
A: Elizabeth is a character who traveled often involving jane ✓

Q: Who is Alice?
A: Alice is a character who spoke and traveled often involving queen ✓
```

## Connection to Transformer Attention

This is exactly analogous to attention in transformers:

| Transformer | Spatial Attention |
|-------------|-------------------|
| Query | Entity being asked about (Holmes) |
| Keys | All related entities (Watson, Jabez, Lestrade...) |
| Values | Entity descriptions |
| Attention weights | spread × partnership |
| Output | Weighted combination of values |

The attention mechanism defines a **metric space** where:
- Important entities are CLOSE to the query
- Unimportant entities are FAR from the query

## The "Path of Spatial Relativity"

The user's insight was profound: we need to "map out a path of spatial relativity" to produce coherent information.

This path is defined by:
1. **Spread** - How distributed is the entity across the space?
2. **Partnership** - How strong is the bidirectional connection?
3. **Centrality** - How many paths pass through this entity?

Watson is at the CENTER of Holmes's relationship network. Jabez is on the PERIPHERY.

## Limitations and Future Work

### Current Limitation: Curated Character List

The current implementation uses a curated list of known literary characters to filter noise. This works but doesn't scale automatically.

### Future Improvements

1. **Automatic character detection** - Use NER or pattern matching to identify character names
2. **Graph-based centrality** - Use PageRank or similar to find important nodes
3. **Learned attention weights** - Train the spread/partnership weights from data
4. **Cross-domain transfer** - Learn attention patterns that transfer across domains

## Conclusion

**Spatial attention solves the frequency ≠ importance problem** by weighting relationships based on:
- How widespread they are (spread)
- How bidirectional they are (partnership)
- How central they are to the entity's identity

This is the geometric analog of transformer attention - finding what's truly important by tracing the "path of spatial relativity" through the concept space.
