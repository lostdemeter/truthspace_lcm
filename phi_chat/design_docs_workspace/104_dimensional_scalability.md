# Design Consideration 104: Dimensional Scalability

## Date: 2026-01-06

## Status: DESIGN DISCUSSION

## The Question

The current φ-lattice has 4 named dimensions:
- DOMAIN (what area)
- SPECIFICITY (how specific)
- INTENT (what response expected)
- FORMALITY (how formal)

But natural language requires many more nuanced dimensions:
- **Tense**: past/present/future
- **Aspect**: completed/ongoing/habitual
- **Mood**: indicative/subjunctive/imperative
- **Voice**: active/passive
- **Polarity**: positive/negative
- **Number**: singular/plural
- **Person**: 1st/2nd/3rd
- **Definiteness**: definite/indefinite
- **Evidentiality**: direct/reported/inferred

In prior testing, 12+ dimensions emerged just from English concepts. How does the φ-lattice scale?

## The Chinese Tense Example

> "I *went* to the store" vs "I *will go* to the store"
> In Chinese, tense is handled by a separate character - arguably a dimension.

This is exactly right. In Chinese:
- 我去商店 (I go store) - neutral
- 我去**了**商店 (I go-**LE** store) - completed aspect
- 我**会**去商店 (I **will** go store) - future

The **了** and **会** are essentially **dimensional markers**. They don't change the core concept (going to store) - they navigate along a temporal dimension.

## Current Architecture Analysis

### What We Have

```python
class PhiLattice:
    def __init__(self, dimensions: List[SemanticDimension]):
        self.dimensions = {d.index: d for d in dimensions}
        self.ndim = len(dimensions)
```

The lattice is **parameterized by dimensions**. It doesn't care how many there are - it just needs a list. The math works the same:

```python
def levels_to_position(self, levels: List[int]) -> np.ndarray:
    return np.array([PHI ** k for k in levels])

def distance(self, a: np.ndarray, b: np.ndarray, weights=None) -> float:
    diff = (a - b) * np.sqrt(weights)
    return float(np.linalg.norm(diff))
```

### The Good News

1. **The lattice scales to N dimensions** - no hardcoded limit
2. **Weights are per-dimension** - can emphasize what matters
3. **Positions are just φ^k vectors** - math doesn't change

### The Concern

1. **Named dimensions are static** - defined at initialization
2. **Primitives map to fixed dimensions** - hardcoded in `primitives.py`
3. **No mechanism for dimension discovery** - we define, not discover

## Three Approaches to Scalability

### Approach A: Static Expansion

Define all dimensions upfront. Add tense, aspect, mood, etc. to `semantic_dimensions.py`:

```python
TENSE = SemanticDimension(
    index=4, name='tense',
    level_meanings={2: 'future', 1: 'present', 0: 'past'},
    weight=PHI ** -2
)

ASPECT = SemanticDimension(
    index=5, name='aspect',
    level_meanings={1: 'perfective', 0: 'imperfective'},
    weight=PHI ** -2
)
```

**Pros:**
- Simple, predictable
- Weights can be tuned
- Primitives can target specific dimensions

**Cons:**
- Doesn't scale to unknown domains
- Every new dimension requires code changes
- Combinatorial explosion of primitives

### Approach B: Quaternion Layers

Use the φ-lattice as the **base layer** within a quaternion structure. Each quaternion component gets its own lattice:

```
Quaternion = (w, x, y, z)
           = (semantic_core, tense_aspect, mood_voice, context)

Each component is a φ-lattice position:
  w = [domain, specificity, intent, formality]  # Core meaning
  x = [tense, aspect, ...]                       # Temporal
  y = [mood, voice, ...]                         # Grammatical
  z = [register, domain_specific, ...]           # Contextual
```

**Pros:**
- Hierarchical organization
- 4 quaternion components × N dimensions each = 4N total
- Rotation/transformation between layers
- Matches the "free axis in a quaternion" intuition

**Cons:**
- More complex encoding
- Need to decide what goes in each component
- Quaternion multiplication semantics unclear for this use

### Approach C: Emergent Dimensions (Self-Assembly)

Don't predefine dimensions. Let them **emerge from data**:

1. Start with a high-dimensional space (e.g., 64 dims)
2. Primitives activate arbitrary dimensions
3. Use PCA/SVD to find the **effective dimensions**
4. Name them after the fact (or not at all)

```python
class EmergentLattice:
    def __init__(self, max_dims: int = 64):
        self.max_dims = max_dims
        self._active_dims = set()  # Which dims have been used
        
    def register_primitive(self, keyword: str, activations: Dict[int, int]):
        """
        Register a primitive with arbitrary dimension activations.
        
        activations: {dim_index: level}
        """
        for dim in activations:
            self._active_dims.add(dim)
        # ...
    
    def discover_structure(self):
        """
        After ingesting data, discover which dimensions matter.
        Returns the effective dimensionality and semantic clusters.
        """
        # PCA on all positions
        # Find dimensions with high variance (meaningful)
        # Cluster primitives by dimension activation patterns
```

**Pros:**
- Truly self-assembling
- Scales to any domain
- Discovers structure we didn't anticipate

**Cons:**
- Less interpretable
- Harder to debug
- May need lots of data to stabilize

## The Hybrid Insight

Looking at the memories:

> **Attractor/Repeller Dynamics**: The vocabulary EMERGES from dynamics, not design.

> **Forward Projection**: Seeds + Transforms → Generate → Verify

> **Holographic Pattern Space**: Positions are CONSTRUCTED from similarity.

The answer might be **both**:

1. **Bootstrap dimensions** (like our current 4) provide the initial structure
2. **Emergent dimensions** appear as data reveals new axes of variation
3. **Attractor dynamics** consolidate dimensions that co-vary
4. **Quaternion layers** organize dimensions hierarchically

## Proposed Architecture

### Layer 1: Core Semantic (4D φ-lattice)
- Domain, Specificity, Intent, Formality
- The "what is this about" layer
- Stable, well-defined

### Layer 2: Grammatical (4D φ-lattice)
- Tense, Aspect, Mood, Voice
- The "how is this expressed" layer
- Language-specific but structured

### Layer 3: Contextual (4D φ-lattice)
- Register, Evidentiality, Politeness, Emphasis
- The "in what context" layer
- More variable

### Layer 4: Emergent (ND)
- Dimensions discovered from data
- Domain-specific axes
- Grows as needed

### The Quaternion Connection

```
Q = w + xi + yj + zk

w = Core Semantic position (4D)
x = Grammatical position (4D)  
y = Contextual position (4D)
z = Emergent position (ND, projected to 4D)
```

Quaternion operations then have meaning:
- **Rotation** = perspective shift (active→passive, 1st→3rd person)
- **Conjugation** = negation or inversion
- **Multiplication** = composition of transformations

## The Tense Example Revisited

"I went to the store" vs "I will go to the store"

```
Core Semantic (w): [1, 2, 1, 0]  # Same for both - "going to store"

Grammatical (x):
  "went":     [0, 1, 0, 0]  # tense=past, aspect=perfective
  "will go":  [2, 0, 0, 0]  # tense=future, aspect=neutral

Full position:
  "I went to the store":     Q1 = [1,2,1,0] + [0,1,0,0]i + ...
  "I will go to the store":  Q2 = [1,2,1,0] + [2,0,0,0]i + ...
```

The **core meaning is identical**. The **grammatical layer differs**. This is exactly the "free axis" intuition - tense is a separate dimension that modifies without changing the core.

## Implementation Path

### Phase 1: Extend Current Lattice
- Add grammatical dimensions (tense, aspect, mood, voice)
- Keep as single flat lattice for now
- Test with tense-marked queries

### Phase 2: Layer Separation
- Separate core semantic from grammatical
- Implement as two coordinated lattices
- Explore quaternion representation

### Phase 3: Emergent Discovery
- Add high-dimensional "overflow" space
- Implement dimension discovery via PCA
- Let structure emerge from data

### Phase 4: Quaternion Integration
- Full quaternion representation
- Rotation/transformation semantics
- Self-similar operations across layers

## Key Questions

1. **Same lattice for every quaternion component?**
   - Probably yes for structure, but different dimensions per component
   - The φ^k math is universal; the semantic meaning varies

2. **How do primitives target layers?**
   - Primitives could specify which layer they activate
   - Or: primitives activate dimensions, layers are organizational

3. **What about 12+ dimensions?**
   - Hierarchical: 4 quaternion components × 4 dimensions = 16
   - Plus emergent dimensions in layer 4
   - Plenty of room

4. **Does distance still work?**
   - Within a layer: yes, same as now
   - Across layers: need to define (weighted sum? quaternion distance?)

## Conclusion

The φ-lattice **can scale** to 12+ dimensions through:

1. **Hierarchical organization** (quaternion layers)
2. **Emergent discovery** (data-driven dimensions)
3. **Self-assembly** (attractor dynamics)

The current 4D lattice is the **seed**. The structure will grow as we ingest more complex data. The key insight is that dimensions are **navigational axes**, not fixed categories - and the geometry works the same regardless of how many axes we have.

---

*"The lattice is the skeleton. The dimensions are the joints. The data teaches us where to bend."*
