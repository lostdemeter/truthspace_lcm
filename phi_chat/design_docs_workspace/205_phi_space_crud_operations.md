# Doc 205: CRUD Operations on φ-Space

## Date: February 3, 2026

## Summary

We implemented Create, Read, Update, Delete operations on the geometric knowledge space. All operations reduce to three primitives: **vector addition**, **scalar multiplication**, and **cosine similarity**.

## The Model's Insight

When asked about CRUD operations, the model identified:

**Primitive Operations:**
1. Translation (movement)
2. Rotation (reorientation)
3. Scaling (magnitude change)
4. Projection (dimensionality)

**Key Insight:** "Can't truly delete in continuous space" - but we can isolate concepts by moving them away from their clusters.

## CRUD Specification

### CREATE (Add New Concept)

**Formula:**
```
position = Σ(weight_i × embed(parent_i))
```

Or via analogy:
```
position = A + (B - C)
```
Where "A is to C as new_concept is to B"

**Example:**
```python
# Weighted combination
crud.create("quantum-chef", parents=["quantum", "chef", "scientist"])

# Analogy: digital is to technology as artist is to creativity
crud.create("digital-artist", analogy=("digital", "artist", "creativity"))
```

**Results:**
| Concept | Method | Top Neighbors |
|---------|--------|---------------|
| quantum-chef | weighted | quant (0.65), scient (0.62), chef (0.58) |
| digital-artist | analogy | artist (0.61), digital (0.58) |

### READ (Query Concept)

**Formula:**
```
position = embed(concept)
neighbors = top_k(cosine_similarity(position, all_embeddings))
```

**Example:**
```python
info = crud.read("consciousness")
# Returns: position, φ-level, neighbors, norm
```

### UPDATE (Modify Concept)

**Formula:**
```
new_position = old_position + α × (new_property - old_property)
```

This is a **translation** in the direction of the property change.

**Example: Pluto reclassification**
```python
crud.update("Pluto", old_property="planet", new_property="dwarf", alpha=0.5)
```

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| Similarity to "planet" | 0.12 | **-0.25** |
| Similarity to "dwarf" | 0.13 | **0.52** |

The update successfully moved Pluto away from "planet" and toward "dwarf"!

### DELETE (Remove Concept)

**Methods:**

1. **Isolate**: Move concept far from its cluster
   ```
   isolated_position = position + β × (position - cluster_center)
   ```

2. **Null**: Project to near-zero
   ```
   null_position = position × 0.001
   ```

3. **Remove**: Delete from custom concepts (only for created concepts)

**Example:**
```python
crud.delete("unicorn", method='isolate', beta=2.0)
```

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| Top neighbor similarity | 0.21 | **0.14** |

The concept is now more isolated from its semantic cluster.

## All Operations Reduce To

```
1. Vector addition (translation)
   - CREATE: sum of parent embeddings
   - UPDATE: add direction vector
   - DELETE: add isolation vector

2. Scalar multiplication (scaling)
   - CREATE: weights on parents
   - UPDATE: α controls strength
   - DELETE: β controls isolation distance

3. Cosine similarity (reading relationships)
   - READ: find neighbors
   - All operations: verify results
```

## The φ-Space is a Vector Space

| Property | Meaning |
|----------|---------|
| Concepts | Points (vectors) |
| Relationships | Directions (differences) |
| Operations | Linear transformations |
| Distance | Semantic dissimilarity |
| Similarity | Cosine of angle |

## Implementation

`/home/thorin/truthspace-lcm/experiments/phi_space_crud.py`

```python
from phi_space_crud import PhiSpaceCRUD

crud = PhiSpaceCRUD(model, tokenizer)

# Create
info = crud.create("new-concept", parents=["a", "b", "c"])

# Read
info = crud.read("existing-concept")

# Update
crud.update("concept", old_property="old", new_property="new")

# Delete
crud.delete("concept", method='isolate')

# Save/Load state
crud.save_state("state.json")
crud.load_state("state.json")
```

## Persistence

The CRUD system maintains:
- `custom_concepts`: New concepts we've created
- `modifications`: Updates and deletions
- `operations`: Log of all operations

State can be saved to JSON and reloaded.

## Limitations

1. **No true deletion**: In continuous space, we can only isolate, not remove
2. **Model weights unchanged**: We're adding a layer on top, not modifying the model
3. **Vocabulary fixed**: New concepts exist in our index, not the tokenizer
4. **Linear operations**: Complex relationships may need nonlinear transforms

## Connection to Prior Work

- **Doc 200**: Universal bottleneck (φ-level validation)
- **Doc 204**: Reverse navigation (CREATE uses similar principles)
- **Doc 203**: Interface design (CRUD could be exposed via UI)

## Future Directions

1. **Validate through bottleneck**: Check that created concepts pass φ-27 ≈ φ
2. **Propagate updates**: When updating A, update concepts that depend on A
3. **Merge with generation**: Use modified embeddings during text generation
4. **Hierarchical operations**: Create/update entire concept hierarchies

## Conclusion

CRUD operations on φ-space are **geometrically simple**:
- CREATE = weighted sum or analogy
- READ = embedding lookup + similarity search
- UPDATE = translation in direction of change
- DELETE = isolation from cluster

All reduce to vector addition, scalar multiplication, and cosine similarity. The geometric structure of knowledge is not just readable - it's **writable**.

---

*"Knowledge is not just stored geometrically - it can be edited geometrically."*
