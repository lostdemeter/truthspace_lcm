# Design Consideration 115: Self-Assembling Corpus Roadmap

## Overview

This document outlines a roadmap for building a **self-assembling knowledge corpus** that:

1. **Compounds data** - Derives compound concepts from primitives geometrically
2. **Discovers dimensions** - New dimensions emerge from transformation pairs in ingested content
3. **Rebalances dynamically** - Existing positions adjust when new dimensions are discovered
4. **Fills unknowns** - Uses local LLM efficiently to populate gaps
5. **Remains geometrically pure** - No statistical weights, only φ-based structure

## The Core Challenges

### Challenge 1: Dynamic Dimension Discovery

When ingesting a first-person narrative after only seeing third-person content, "perspective" emerges as a new dimension. This requires:

- Detecting that a new dimension is needed
- Adding the dimension to the space
- Repositioning existing concepts on this new axis
- Identifying gaps (concepts that now need perspective variants)

### Challenge 2: Platonic Ideal Detection

Not all Platonic Ideals are obvious:

| Clear Case | Ambiguous Case |
|------------|----------------|
| house → cottage, mansion, palace | dog → mastiff, lapdog, puppy |
| person → child, elder, noble | large + dog → mastiff (but no word for "large dog") |

The neutral "dog" is the Platonic Ideal, but "mastiff" is a specific breed that happens to be large—not a pure "large dog" concept.

### Challenge 3: Efficient LLM Usage

LLM calls are expensive. We need to:

- Bootstrap primitives once
- Derive compounds geometrically (no LLM needed)
- Only call LLM for genuinely unknown concepts
- Batch queries efficiently

## The Roadmap

### Phase 1: Core Infrastructure

#### 1.1 Persistent Geometric Space

```
GeometricSpace
├── dimensions: List[Dimension]  # Emergent, named, introspectable
├── concepts: Dict[str, Position]  # Word → n-dimensional position
├── pairs: List[TransformationPair]  # The source of truth
├── ideals: List[PlatonicIdeal]  # Detected origin points
└── version: int  # For tracking rebalances
```

**Key property**: The space can be reconstructed entirely from `pairs`. Everything else is derived.

#### 1.2 Dimension Registry

```python
class Dimension:
    name: str  # e.g., "perspective", "size", "formality"
    index: int  # Position in the n-dimensional space
    pole_negative: List[str]  # Words at -φ end
    pole_positive: List[str]  # Words at +φ end
    source_pairs: List[TransformationPair]  # What defined this dimension
    discovered_at: datetime  # When this dimension emerged
```

#### 1.3 Platonic Ideal Registry

```python
class PlatonicIdeal:
    word: str  # e.g., "house", "dog", "person"
    dimensions_anchored: List[str]  # Which dimensions this ideal anchors
    variations: Dict[str, List[str]]  # dimension → [variation words]
    confidence: float  # How certain we are this is an ideal
```

### Phase 2: Ingestion Pipeline

#### 2.1 Text → Transformation Pairs

```
Raw Text → Tokenize → Extract Relationships → Transformation Pairs
```

**Relationship extraction strategies**:

1. **Explicit pairs**: "The king and queen ruled..." → (king, queen, gender_flip)
2. **Contextual pairs**: First-person "I went" vs third-person "He went" → (I, he, perspective_shift)
3. **Semantic pairs**: "The cottage was small, unlike the mansion" → (cottage, mansion, size_increase)
4. **LLM-assisted**: Ask LLM to identify transformation pairs in ambiguous cases

#### 2.2 Pair → Dimension Mapping

When a new pair arrives:

```python
def ingest_pair(pair: TransformationPair):
    # Check if this relationship type exists
    if pair.relationship in known_dimensions:
        # Add to existing dimension
        add_to_dimension(pair)
    else:
        # NEW DIMENSION DETECTED
        create_dimension(pair.relationship)
        trigger_rebalance()
```

#### 2.3 Rebalancing Protocol

When a new dimension is added:

```python
def rebalance(new_dimension: Dimension):
    # 1. Extend all positions with new axis (default: 0 = neutral)
    for concept in concepts:
        concept.position = extend(concept.position, 0)
    
    # 2. Position concepts that have pairs on this dimension
    for pair in new_dimension.source_pairs:
        position_on_dimension(pair.source, new_dimension, 0)  # Origin
        position_on_dimension(pair.target, new_dimension, PHI)  # +φ
    
    # 3. Identify gaps - concepts that might need variants
    gaps = identify_gaps(new_dimension)
    
    # 4. Queue gap-filling (LLM will handle these)
    queue_for_llm(gaps)
```

### Phase 3: Gap Detection and Filling

#### 3.1 Gap Types

| Gap Type | Example | Detection | Resolution |
|----------|---------|-----------|------------|
| **Missing variation** | "house" has size variants but no perspective variants | Ideal exists, dimension exists, but no pair | LLM: "What is a first-person word for house?" |
| **Missing ideal** | We have "cottage" and "mansion" but not "house" | Variations exist but no neutral anchor | LLM: "What is the neutral form of cottage/mansion?" |
| **Unnamed compound** | large + dog = ? (no single word) | Compound position has no word | Mark as compound, don't force a word |
| **Ambiguous variation** | "mastiff" = large + dog, or specific breed? | Word exists at compound position | LLM: "Is mastiff a general 'large dog' or a specific breed?" |

#### 3.2 LLM Query Batching

```python
class LLMQueryBatch:
    queries: List[Query]
    priority: int  # Higher = more urgent
    
    def execute(self, llm: LocalLLM):
        # Batch similar queries together
        # "What are the size variants of X?" for multiple X
        # Single prompt, multiple answers
        prompt = self.build_batch_prompt()
        responses = llm.generate(prompt)
        return self.parse_responses(responses)
```

#### 3.3 Efficient Query Strategies

1. **Compound before query**: Try geometric compounding first. Only query LLM if compound doesn't exist.

2. **Batch by dimension**: "For the 'formality' dimension, what are the formal/informal variants of: [list of words]?"

3. **Confidence thresholds**: Only query LLM when geometric confidence is below threshold.

4. **Cache aggressively**: Once LLM confirms a relationship, it becomes a permanent pair.

### Phase 4: Platonic Ideal Discovery

#### 4.1 Ideal Detection Algorithm

```python
def detect_ideals():
    # Count how many dimensions each word anchors
    anchor_counts = {}
    for pair in all_pairs:
        source = pair.source
        anchor_counts[source] = anchor_counts.get(source, set())
        anchor_counts[source].add(pair.relationship)
    
    # Words anchoring 2+ dimensions are candidate ideals
    candidates = [w for w, dims in anchor_counts.items() if len(dims) >= 2]
    
    # Verify: ideal should be at origin on its anchored dimensions
    ideals = []
    for word in candidates:
        pos = get_position(word)
        anchored_dims = anchor_counts[word]
        if all(pos[dim_index] == 0 for dim in anchored_dims):
            ideals.append(PlatonicIdeal(word, anchored_dims))
    
    return ideals
```

#### 4.2 Ideal Hierarchy

Some ideals are more fundamental than others:

```
Level 0: Universal ideals (anchor 5+ dimensions)
         └── "thing", "entity", "concept"

Level 1: Domain ideals (anchor 3-4 dimensions)
         └── "person", "place", "object"

Level 2: Category ideals (anchor 2 dimensions)
         └── "house", "dog", "food", "vehicle"

Level 3: Specific ideals (anchor 1 dimension)
         └── "cottage" (only size), "puppy" (only age)
```

#### 4.3 The "Unnamed Compound" Problem

When a compound position has no word:

```python
def handle_unnamed_compound(position: Position):
    # Check if any word exists at this position
    nearest = find_nearest(position)
    
    if distance(nearest, position) < THRESHOLD:
        # Close enough - use this word
        return nearest
    else:
        # No word exists - this is an unnamed compound
        # Generate a descriptor: "large dog", "formal greeting"
        components = decompose_to_dimensions(position)
        descriptor = " + ".join(f"{dim}:{val}" for dim, val in components)
        return UnnamedCompound(descriptor, position)
```

### Phase 5: Self-Assembly Loop

The complete self-assembly cycle:

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-ASSEMBLY LOOP                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. INGEST                                                   │
│     └── New text → Extract transformation pairs              │
│                                                              │
│  2. DETECT                                                   │
│     └── New relationship type? → Create dimension            │
│                                                              │
│  3. REBALANCE                                                │
│     └── New dimension → Extend all positions                 │
│                                                              │
│  4. POSITION                                                 │
│     └── Place concepts on dimensions (source=0, target=φ)   │
│                                                              │
│  5. DISCOVER                                                 │
│     └── Find Platonic Ideals (multi-dimension anchors)       │
│                                                              │
│  6. GAP-FILL                                                 │
│     └── Identify missing variations → Queue for LLM          │
│                                                              │
│  7. COMPOUND                                                 │
│     └── Derive compound positions geometrically              │
│                                                              │
│  8. VERIFY                                                   │
│     └── Check self-similarity, introspect dimensions         │
│                                                              │
│  └──────────────────── REPEAT ──────────────────────────────┘
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Phase 6: Persistence and Versioning

#### 6.1 Storage Format

```json
{
  "version": 42,
  "dimensions": [
    {"name": "gender", "index": 0, "poles": [["king", "man"], ["queen", "woman"]]},
    {"name": "size", "index": 1, "poles": [["small", "tiny"], ["large", "huge"]]},
    {"name": "perspective", "index": 2, "poles": [["I", "me"], ["he", "she"]]}
  ],
  "pairs": [
    {"source": "king", "target": "queen", "relationship": "gender"},
    {"source": "house", "target": "mansion", "relationship": "size"}
  ],
  "ideals": [
    {"word": "house", "dimensions": ["size", "regality"], "confidence": 0.95}
  ]
}
```

#### 6.2 Incremental Updates

- Pairs are append-only (never delete, only add)
- Dimensions can be added but not removed
- Positions are recomputed from pairs on load
- Version increments on each rebalance

### Implementation Priority

| Priority | Component | Effort | Value |
|----------|-----------|--------|-------|
| P0 | Persistent GeometricSpace | Medium | Foundation |
| P0 | Pair ingestion pipeline | Medium | Core loop |
| P1 | Dynamic dimension creation | Low | Enables growth |
| P1 | Rebalancing protocol | Medium | Maintains consistency |
| P2 | Gap detection | Medium | Improves coverage |
| P2 | LLM query batching | Low | Efficiency |
| P3 | Platonic Ideal discovery | Low | Insight |
| P3 | Unnamed compound handling | Low | Completeness |

## Success Criteria

1. **Scalability**: Ingest 10,000+ concepts without manual dimension definition
2. **Efficiency**: <1 LLM call per 100 concepts (99% derived geometrically)
3. **Self-similarity**: All transformations within a dimension have delta = φ
4. **Introspectability**: Every dimension can be described in English
5. **Recoverability**: Space can be reconstructed entirely from pairs

## Open Questions

1. **Dimension merging**: What if two dimensions are discovered to be the same? (e.g., "formality" and "register")

2. **Dimension splitting**: What if one dimension should be two? (e.g., "regality" → "formality" + "wealth")

3. **Cross-domain ideals**: Are there universal Platonic Ideals that appear in all domains?

4. **Temporal dimensions**: How do we handle dimensions that change over time? (e.g., word meanings that shift)

## Conclusion

The self-assembling corpus is built on three pillars:

1. **Transformation pairs** are the source of truth
2. **Dimensions emerge** from relationship types
3. **Platonic Ideals** anchor the structure

Everything else—positions, compounds, gaps—is derived geometrically. The LLM is only consulted when geometry alone cannot provide an answer.

## References

- Design 114: Emergent Dimensions and Platonic Ideals
- Design 039: φ-Zipf Duality
- Experiment: `/experiments/concept_compounding.py`
