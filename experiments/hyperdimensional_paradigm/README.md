# Hyperdimensional Paradigm Experiment

## The Core Insight

We've separated two distinct concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                  HYPERDIMENSIONAL PARADIGM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │                     │      │                     │          │
│  │  STRUCTURE          │      │  TRANSCODER         │          │
│  │  (Data Structure)   │      │  (Execution Engine) │          │
│  │                     │      │                     │          │
│  │  - Positions        │◄────►│  - Encode           │          │
│  │  - Add/Remove       │      │  - Decode           │          │
│  │  - Query            │      │  - Transcode        │          │
│  │  - Learn            │      │  - Feedback         │          │
│  │  - Serialize        │      │                     │          │
│  │                     │      │                     │          │
│  │  DOMAIN-AGNOSTIC    │      │  DOMAIN-SPECIFIC    │          │
│  └─────────────────────┘      └─────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## The Structure (hyperdimensional_structure.py)

A pure data structure that:
- Stores nodes with N-dimensional positions
- Supports add, remove, query, update operations
- Provides learning via position movement (attract/repel)
- Maintains stability via reprojection
- Serializes to/from JSON

**Key property**: It knows NOTHING about what the positions represent.

## The Transcoder (hyperdimensional_transcoder.py)

An execution engine that:
- Encodes domain inputs to positions
- Queries the structure for matches
- Decodes matches back to domain outputs
- Provides feedback for learning

**Key property**: It provides domain-specific encoding/decoding.

## The Paradigm

```python
# Same structure, different transcoders

structure = HyperdimensionalStructure(dims=8)

# For text/chat
text_transcoder = TextTranscoder(structure)
text_transcoder.transcode("hello world")

# For classification
cat_transcoder = CategoricalTranscoder(structure)
cat_transcoder.transcode(feature_vector)

# For numeric interpolation
num_transcoder = NumericTranscoder(structure)
num_transcoder.transcode(input_vector)
```

## Why This Matters

1. **Separation of Concerns**
   - Structure handles geometry
   - Transcoder handles domain mapping

2. **Reusability**
   - Same structure code for any domain
   - Only transcoder changes per domain

3. **Flexibility**
   - Add dimensions as needed
   - Swap transcoders without changing structure

4. **Serialization is Clear**
   - Structure serializes positions
   - Transcoder is stateless (or minimal state)

## The Computer Science Perspective

This is a new kind of data structure:

| Traditional | Hyperdimensional |
|-------------|------------------|
| Hash table: key → value | Position → node |
| Tree: hierarchical | Continuous space |
| Graph: discrete edges | Proximity-based |
| Fixed schema | Flexible dimensions |

Operations:
- **Insert**: Add node at position
- **Query**: Find nearest neighbors
- **Update**: Move position (learning)
- **Delete**: Remove node (or let it decay below critical line)

Complexity:
- Insert: O(1) or O(n) with reprojection
- Query: O(n) naive, O(log n) with spatial index
- Update: O(1)
- Delete: O(1)

## Next Steps

1. **Improve encoding** - The text transcoder's bag-of-words is too simple
2. **Add spatial indexing** - k-d tree or ball tree for faster queries
3. **Test with real domains** - Images, audio, structured data
4. **Formalize the paradigm** - Mathematical foundations

## Files

- `hyperdimensional_structure.py` - The data structure
- `hyperdimensional_transcoder.py` - The execution engine + example transcoders
- `README.md` - This file
