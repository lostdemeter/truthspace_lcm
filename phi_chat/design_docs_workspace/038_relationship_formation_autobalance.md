# Design Consideration 038: Relationship Formation and Autobalancing

## Problem Statement

How do meaningful relationships form in concept space, and how can we use this understanding to autobalance the model?

## Key Insight: Zipf Applies to Everything

The Zipf distribution isn't just about filtering noise from storage - it applies to the **entire concept space**:

1. **Storage**: Weight entities by specificity → filter noise from relationship graph
2. **Retrieval**: Weight frames by specificity → prioritize specific over generic frames  
3. **Answers**: Weight components by specificity → generate more meaningful output

This is the same principle at every level!

## Key Discoveries

### 1. Bidirectionality is the Strongest Signal

```
Bidirectional relationships: 257 (22%)
Unidirectional relationships: 898 (78%)
```

**Meaningful relationships are bidirectional:**
- Holmes ↔ Watson (2 → 18, bidirectional)
- Joe ↔ Pip (20 → 23, bidirectional)
- Darcy ↔ Elizabeth (bidirectional)

**Noise relationships are unidirectional:**
- Holmes → Jabez (one-way)
- Narrator → Subject (one-way)

### 2. Balance Indicates Relationship Type

| Balance | Type | Example |
|---------|------|---------|
| 0.8-1.0 | Equal partners | Holmes ↔ Sherlock (0.92) |
| 0.3-0.7 | Main + Supporting | Holmes ↔ Watson (0.11) |
| < 0.3 | Weak association | Alice ↔ King (0.11) |

**Key insight:** Watson's low balance (2:18) reflects the narrator↔subject pattern:
- Watson OBSERVES Holmes (THINK, PERCEIVE)
- Holmes ACTS, Watson REPORTS
- Asymmetric but deeply meaningful

### 3. Zipf Distribution Identifies Noise

```
Top words by frequency (Zipf):
  the    freq=33078  spec=0.0961  (structural)
  and    freq=18654  spec=0.1017  (structural)
  said   freq=8903   spec=0.1100  (structural)
  
Proper nouns:
  watson freq=~200   spec=0.3107  (meaningful)
  holmes freq=~600   spec=0.1683  (meaningful)
```

**Inverse Zipf weighting automatically filters noise:**
- High frequency → low specificity → low weight
- Low frequency → high specificity → high weight

### 4. Spread Indicates Universality

```
Multi-source proper nouns: 95
Single-source proper nouns: 110
```

Entities appearing in multiple sources are more universally important.

## The Autobalance Formula

```python
importance(A, B) = zipf(A) × zipf(B) × avg_spread(A,B) × bidir(A,B)

where:
  zipf(X) = 1 / log(1 + frequency(X))  # Inverse Zipf
  spread(X) = sources(X) / total_sources
  bidir(A,B) = log(1 + total_mentions) × (1.5 if bidirectional else 1.0)
```

### Test Results

| Entity1 | Entity2 | Importance | Zipf1 | Zipf2 | Spread | Bidir |
|---------|---------|------------|-------|-------|--------|-------|
| holmes | watson | **0.0746** | 0.168 | 0.311 | 0.312 | 4.57 |
| joe | pip | **0.0939** | 0.183 | 0.721 | 0.125 | 5.68 |
| darcy | elizabeth | 0.0182 | 0.229 | 0.200 | 0.094 | 4.25 |
| holmes | said | 0.0243 | 0.168 | 0.128 | 0.219 | 5.15 |
| holmes | the | **0.0000** | 0.168 | 0.096 | 0.156 | 0.00 |

**"the" gets zero importance** despite being the most frequent word!

## Connection to Qwen2 Findings

From earlier reverse engineering of Qwen2:
- Proper nouns took up **majority of model weight space**
- They existed on the **"zero axis"** - largely dormant
- BUT when activated, they carry the **most meaningful relationships**

This matches our findings:
- Proper nouns are SPARSE (low frequency, high specificity)
- They form BIDIRECTIONAL relationships (meaningful connections)
- They have high SPREAD (appear across contexts)

## Implementation for Autobalancing

### During Corpus Building

```python
class CorpusBuilder:
    def __init__(self):
        self.word_frequencies = Counter()
        self.entity_sources = defaultdict(set)
        self.bidirectional_graph = defaultdict(lambda: defaultdict(int))
    
    def add_frame(self, frame):
        # Track word frequencies for Zipf
        self.word_frequencies.update(frame['text'].split())
        
        # Track entity sources for spread
        self.entity_sources[frame['agent']].add(frame['source'])
        
        # Track bidirectional relationships
        for entity in self.known_entities:
            if entity in frame['text']:
                self.bidirectional_graph[frame['agent']][entity] += 1
    
    def compute_entity_weights(self):
        """Compute autobalance weights for all entities."""
        weights = {}
        for entity in self.entity_sources:
            zipf = 1.0 / np.log1p(self.word_frequencies[entity])
            spread = len(self.entity_sources[entity]) / self.total_sources
            
            # Sum bidirectional relationship strengths
            bidir_strength = 0
            for other, count in self.bidirectional_graph[entity].items():
                reverse = self.bidirectional_graph[other].get(entity, 0)
                if reverse > 0:  # Bidirectional
                    bidir_strength += np.log1p(count + reverse) * 1.5
            
            weights[entity] = zipf * spread * (1 + bidir_strength)
        
        return weights
```

### During Query Time

```python
def get_important_relations(self, entity, k=3):
    """Get top-k important relations using autobalance weights."""
    candidates = []
    
    for other in self.known_entities:
        if other == entity:
            continue
        
        importance = self.compute_importance(entity, other)
        candidates.append((other, importance))
    
    # Sort by importance (not frequency!)
    candidates.sort(key=lambda x: -x[1])
    return candidates[:k]
```

## The Geometric Principle

```
High-frequency words = Structural scaffolding (Zipf distribution)
                     = Low weight in autobalancing
                     
Proper nouns = Sparse but meaningful (relationship carriers)
             = High weight in autobalancing
             
Importance = Bidirectional + Spread + Specificity
```

This is the "path of spatial relativity" - the geometry of relationships that determines what's truly important.

## Future Work

1. **Automatic proper noun detection** - Use NER or pattern matching
2. **Learned Zipf thresholds** - Automatically determine cutoffs
3. **Graph centrality** - Use PageRank for importance
4. **Cross-domain transfer** - Apply learned weights to new domains

## Conclusion

Meaningful relationships in concept space are characterized by:

1. **Bidirectionality** - Both entities reference each other
2. **Specificity** - Inverse Zipf weighting (rare = important)
3. **Spread** - Appears across multiple sources
4. **Asymmetric balance** - Narrator↔Subject patterns are valid

The autobalance formula `importance = zipf × spread × bidir` automatically:
- Filters Zipf noise (common words)
- Prioritizes proper nouns (relationship carriers)
- Weights bidirectional relationships higher
- Accounts for multi-source universality

This enables the model to focus on what's truly meaningful, not just what's frequent.
