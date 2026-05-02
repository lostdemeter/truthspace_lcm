# 092: JSON ↔ Structure Duality

## The Chicken and Egg Problem

On startup, we have no geometric structure. We need *something* to bootstrap from.
JSON provides that starting point - it's the **definition** of the initial structure,
not the structure itself.

```
STARTUP:   JSON → Bootstrap → Structure (empty → seeded)
RUNTIME:   Structure evolves via usage (positions move)
SHUTDOWN:  Structure → JSON (persist learned positions)
RESTART:   JSON → Structure (resume where we left off)
```

## The Key Insight

**JSON is serialization. Structure is truth.**

| Aspect | JSON | Structure |
|--------|------|-----------|
| Role | Definition/Serialization | The actual object |
| When used | Bootstrap, Persist | Runtime |
| What it contains | Position tuples, word lists | Living geometry |
| Mutability | Static file | Dynamic, evolving |

## The Hyperdimensional Transcoder Pattern

We're dynamically building a hyperdimensional transcoder:

```
┌─────────────────────────────────────────────────────────────┐
│                    JSON (Definition)                         │
│  {                                                          │
│    "concepts": [                                            │
│      {"words": ["king", "monarch"], "position": [0.8,...]}, │
│      {"words": ["queen", "monarch"], "position": [0.6,...]} │
│    ]                                                        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                           ↓ bootstrap
┌─────────────────────────────────────────────────────────────┐
│                  Structure (Runtime)                         │
│                                                             │
│     "king" ──→ ● (0.8, 0.1, 0.0, 0.0)                      │
│                    ╲                                        │
│     "monarch" ─────→● (shared reference)                   │
│                    ╱                                        │
│     "queen" ──→ ● (0.6, 0.1, 0.0, 0.0)                     │
│                                                             │
│  Positions ARE identity. Words are just indices.            │
└─────────────────────────────────────────────────────────────┘
                           ↓ usage
┌─────────────────────────────────────────────────────────────┐
│                  Structure (Evolved)                         │
│                                                             │
│     "king" ──→ ● (0.85, 0.12, 0.02, 0.0)  ← moved!         │
│                                                             │
│     "queen" ──→ ● (0.62, 0.11, 0.01, 0.0) ← moved!         │
│                                                             │
│  Successful uses pulled concepts toward query positions.    │
└─────────────────────────────────────────────────────────────┘
                           ↓ persist
┌─────────────────────────────────────────────────────────────┐
│                    JSON (Updated)                            │
│  {                                                          │
│    "concepts": [                                            │
│      {"words": ["king", "monarch"], "position": [0.85,...]},│
│      {"words": ["queen", "monarch"], "position": [0.62,...]}│
│    ]                                                        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

## Connection to Design 086: Emergent Gear Pattern

This is the same pattern we identified:

1. **STRUCTURE** - The geometric space (positions, relationships)
2. **BOOTSTRAP** - JSON → initial positions (solve chicken/egg)
3. **MATCH** - Find concepts by position similarity
4. **COMPOSE** - Use matched concepts to generate output
5. **LEARN** - Move positions based on success/failure

The JSON bootstrap is step 2. Everything else operates on structure.

## Connection to Design 091: Position Is Everything

The structure we're building is purely positional:

- **Position IS identity** - A concept is WHERE it is
- **Movement IS learning** - Success/failure moves positions
- **Critical line IS horizon** - σ=0.5 determines persistence

JSON just captures snapshots of this living geometry.

## Implementation Requirements

### Bootstrap (JSON → Structure)
```python
def load(path: str) -> GeometricKnowledgeStore:
    """Load structure from JSON definition."""
    data = json.load(path)
    store = GeometricKnowledgeStore()
    for concept_data in data['concepts']:
        concept = Concept(
            words=set(concept_data['words']),
            position=tuple(concept_data['position'])
        )
        store.add(concept)
    return store
```

### Persist (Structure → JSON)
```python
def save(self, path: str) -> None:
    """Save structure to JSON for later bootstrap."""
    data = {
        'concepts': [c.to_dict() for c in self.concepts],
        'metadata': {
            'persisting_count': len(self.get_persisting_concepts()),
            'dims': self.dims
        }
    }
    json.dump(data, path)
```

### Runtime (Structure only)
```python
def use(self, concept_id: str, query_position: tuple, success: bool):
    """THE learning operation - moves concept in position space."""
    concept = self.get(concept_id)
    if success:
        concept.move_toward(query_position)
    else:
        concept.move_away(query_position)
    # No JSON involved - pure structure manipulation
```

## What This Means for Word Overlap

The word overlap similarity we use during bootstrap is **fine** - it's how we
seed initial positions from the JSON definition. But once positions exist,
similarity should be computed **geometrically**:

```python
# Bootstrap: word overlap → initial positions
S[i,j] = word_overlap(concept_i.words, concept_j.words)
positions = eigendecompose(S)

# Runtime: positions → similarity
similarity = np.dot(concept_i.position, concept_j.position)
```

The word overlap is just the **definition language**. The positions are the
**actual structure** we work with.

## The Transcoder Analogy

Think of it like defining a neural network:

| Neural Network | Our System |
|----------------|------------|
| Architecture JSON | Corpus JSON |
| Weights | Positions |
| Forward pass | Query → match → compose |
| Backprop | use() with success/failure |
| Checkpoint | save() |
| Load checkpoint | load() |

The JSON defines the architecture. The positions ARE the learned weights.
We're building a transcoder that learns its own structure through use.

## Summary

```
JSON = Definition language (how we describe structure)
Structure = The actual object (what we compute with)

Bootstrap: JSON → Structure (solve chicken/egg)
Runtime:   Structure only (positions move, learn)
Persist:   Structure → JSON (save for next startup)
```

The structure is the hyperdimensional transcoder.
The JSON is just how we define and persist it.
