# Design Consideration 047: Geodesic Generation

## Date: 2024-12-22

## Context

After implementing the 4D quaternion φ-dial and exploring holographic interference patterns, we investigated how to **generate** text geometrically, not just retrieve it. The goal: replace traditional LLM token prediction with geometric navigation.

## The Problem

Traditional LLMs generate via:
```
P(next_token | previous_tokens) → sequential, morphological
```

This requires:
- Training on massive corpora
- Token-by-token prediction
- Morphological awareness (word order, grammar)

Our system is **order-free** and **geometric**. We need a generation method that matches.

## The Solution: Geodesic Generation

### Core Insight

**Generation = Walking through concept space**

Instead of predicting the next token, we:
1. Start at a position in concept space (the query)
2. Walk a geodesic path through related concepts
3. Project the path to language (grammar applied only at the end)

### The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GEODESIC GENERATION                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. QUERY → Starting Position                                │
│     "Who is Holmes?" → [HOLMES, IDENTITY, PERSON]           │
│                                                              │
│  2. φ-DIAL → Direction & Depth                               │
│     x=-1 (formal): toward formal vocabulary                  │
│     z=+1 (elaborate): longer path, more concepts             │
│                                                              │
│  3. GEODESIC WALK → Concept Path                             │
│     [HOLMES] → [SPEAK, THINK] → [WATSON, CRIME] → [SOURCE]  │
│                                                              │
│  4. PROJECTION → English (grammar applied here)              │
│     "Holmes spoke and considered, often with Watson"         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why "Geodesic"?

In differential geometry, a geodesic is the shortest path between two points on a curved surface. In concept space:

- The "surface" is the manifold of stored knowledge
- The "path" connects query to answer
- The φ-dial controls the "curvature" (style, depth, certainty)

The path is **purely geometric** — no token sequences, no morphology.

## Implementation

### Entity Graph

We build a graph from concept frames:

```python
graph[entity] = {
    'actions': Counter(),    # What actions this entity performs
    'relations': Counter(),  # What other entities they relate to
    'source': str,           # Source text
}
```

### Concept Path Generation

```python
def get_concept_path(entity, dial):
    path = [(entity, 'ENTITY')]
    
    # Depth based on z-dial
    depth = 1 if dial.z < -0.3 else (3 if dial.z > 0.3 else 2)
    
    # Add actions (skip EXIST, POSSESS)
    for action in entity_actions[:depth]:
        path.append((action, 'ACTION'))
    
    # Add relations (filtered to real entities)
    for related in entity_relations[:depth]:
        path.append((related, 'RELATION'))
    
    path.append((source, 'SOURCE'))
    return path
```

### Projection to English

```python
def project_to_english(path, dial):
    # Style-based verb mapping
    verb_map = {
        'SPEAK': ('articulated', 'spoke', 'said'),  # formal, neutral, casual
        'THINK': ('contemplated', 'considered', 'thought'),
        ...
    }
    
    # Build sentence from path
    entity = path[0]
    verbs = [verb_map[a][style_index] for a in actions]
    relations = [r.title() for r in relations]
    
    return f"{entity} {verbs}, often with {relations}"
```

## The φ-Dial in Generation

| Axis | Effect on Path | Effect on Projection |
|------|----------------|---------------------|
| **x (style)** | — | Vocabulary selection (formal/casual) |
| **y (perspective)** | — | Framing (subjective/meta) |
| **z (depth)** | Path length | Amount of detail |
| **w (certainty)** | — | Modifiers (definitive/hedged) |

The z-axis directly affects the geodesic path length:
- `z = -1`: Short path (1 action, 1 relation) → terse answer
- `z = +1`: Long path (3 actions, 3 relations) → elaborate answer

## Relation Quality Filtering

### The Problem

Frame extraction picks up noise words as relations:
```
patient=holmes, theme=said, goal=you  ← "said" and "you" are noise
```

### The Solution

Filter relations to only include **character names**:
1. Must appear as agent ≥ 5 times
2. Must not be a common English word
3. Result: Clean entity-to-entity relationships

```python
# Before filtering
Holmes relations: said, man, white, sherlock, house

# After filtering  
Holmes relations: sherlock, london, jabez
Darcy relations: elizabeth, bingley, wickham
Watson relations: holmes, sherlock
Elizabeth relations: jane, charlotte, mary
```

## Comparison to Traditional LLMs

| Aspect | Traditional LLM | Geodesic Generation |
|--------|-----------------|---------------------|
| Unit | Token | Concept |
| Prediction | P(next \| prev) | Nearest in direction |
| Order | Sequential | Order-free |
| Grammar | Implicit in model | Explicit projection step |
| Training | Required | Not required |
| Morphology | Core to model | Only in projection |

## The Holographic Connection

Geodesic generation connects to holographic principles:

1. **Holographic Storage**: Knowledge stored as interference patterns
2. **Holographic Retrieval**: Query reconstructs relevant concepts
3. **Geodesic Path**: The "beam" that traverses the hologram
4. **Projection**: Rendering the 3D hologram to 2D image (concept to language)

The φ-dial controls the "viewing angle" of the hologram:
- Different angles (dial settings) → different projections (answers)
- Same underlying structure → consistent meaning

## Stereoscopic Generation

For richer generation, we can use **multiple geodesic paths** (stereoscopy):

1. Walk multiple paths from the same starting point
2. Find the **interference pattern** (concepts that appear in multiple paths)
3. Generate from the interference (constructive = include, destructive = exclude)

This produces answers that are:
- Grounded in multiple perspectives
- Robust to noise (single-path artifacts cancel out)
- Naturally weighted by importance (frequent concepts survive)

## Limitations

### What Works
- Entity-centric questions ("Who is Holmes?")
- Relationship questions ("Who does Darcy interact with?")
- Style/depth control via φ-dial

### What Doesn't Work (Yet)
- Complex reasoning ("Why did Holmes suspect the butler?")
- Temporal sequences ("What happened after the murder?")
- Novel combinations ("What if Holmes met Darcy?")

These require:
- Causal graph navigation (not just entity graph)
- Temporal ordering (breaks order-free assumption)
- Counterfactual reasoning (requires inference, not just retrieval)

## Future Directions

### 1. Causal Geodesics
Walk through causal relationships, not just entity relationships:
```
[MURDER] → [SUSPECT] → [EVIDENCE] → [DEDUCTION] → [SOLUTION]
```

### 2. Temporal Geodesics
Add time dimension to the graph:
```
[CHAPTER_1] → [CHAPTER_2] → [CHAPTER_3]
```

### 3. Counterfactual Paths
Branch from known paths to explore alternatives:
```
[HOLMES_INVESTIGATES] → [WHAT_IF_WATSON_INVESTIGATES]
```

### 4. Multi-Source Stereoscopy
Combine paths from different sources (books, perspectives) via interference.

## Conclusion

Geodesic generation provides a **purely geometric** approach to text generation:

| Layer | Geometric? | Description |
|-------|------------|-------------|
| Query | ✓ | Position in concept space |
| Path | ✓ | Geodesic through entity graph |
| φ-Dial | ✓ | Direction and depth control |
| Interference | ✓ | Multi-path combination |
| Projection | ✗ | Grammar (necessary for language output) |

The only non-geometric step is the final projection to language — and this is unavoidable since language has grammar. But it's a **thin layer** on top of a geometric core.

**"The sentence is a shadow of the path through concept space."**

---

## References

- Design 044: 4D Quaternion φ-Dial
- Design 045: The 4D Holographic Bound
- Design 046: Holographic Interference Patterns

---

*"Generation is not prediction — it's navigation."*
