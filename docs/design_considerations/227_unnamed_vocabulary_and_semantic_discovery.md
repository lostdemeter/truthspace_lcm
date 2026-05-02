# Design Consideration 227: Unnamed Vocabulary and Semantic Discovery

## Date: February 6, 2026

## The Challenge

We want to design geometric AI from scratch, but we face a gap:
- **Structure** (architecture, lattice) is derivable
- **Content** (semantic meaning) requires data

The question: Can we discover semantic content without traditional training?

---

## The Approach: Start from the Answer

Instead of training from scratch, we:
1. **Start from the destination** (DDColor's trained queries)
2. **Analyze the structure** (clustering, similarity, φ-levels)
3. **Self-assemble vocabulary** through attractor/repeller dynamics
4. **Build an interface** to explore and name the unknowns

---

## DDColor Query Structure Analysis

### Key Findings

| Property | Value | Meaning |
|----------|-------|---------|
| **Effective rank** | 94.4 | All 100 queries are distinct |
| **Mean similarity** | 0.0496 | Nearly orthogonal |
| **φ-level** | -1.58 | Same scale as other weights |
| **Natural clusters** | 10 | Queries group by similarity |

### The 10 Clusters

| Cluster | Size | Example Queries |
|---------|------|-----------------|
| 0 | 4 | 11, 28, 47, 93 |
| 1 | 8 | 16, 21, 33, 60, 65, 67, 70, 83 |
| 2 | 8 | 37, 42, 49, 54, 59, 72, 74, 80 |
| 3 | 6 | 15, 25, 35, 36, 40, 96 |
| 4 | 18 | 7, 10, 12, 13, 17, 18, 24, ... |
| 5 | 10 | 2, 6, 27, 51, 55, 57, 66, ... |
| 6 | 14 | 4, 9, 20, 23, 26, 43, 46, ... |
| 7 | 2 | 56, 79 |
| 8 | 6 | 41, 45, 48, 53, 81, 90 |
| 9 | 24 | 0, 1, 3, 5, 8, 14, 19, ... |

### Most Similar/Dissimilar Pairs

**Most similar:**
- (45, 53): 0.189
- (73, 92): 0.189
- (13, 59): 0.188

**Most dissimilar:**
- (27, 42): -0.214
- (51, 92): -0.209
- (21, 88): -0.203

---

## The Unnamed Vocabulary Framework

### Core Concept

Concepts don't need names upfront - they have **positions** and **behaviors**.

```python
@dataclass
class UnnamedConcept:
    id: int
    position: torch.Tensor  # Position in semantic space
    
    # Behavioral properties (discovered through usage)
    activation_count: int = 0
    co_activations: Dict[int, int] = {}
    spatial_affinity: Optional[str] = None
    color_tendency: Optional[Tuple[float, float]] = None
    
    # Emergent properties
    cluster_id: Optional[int] = None
    tentative_name: Optional[str] = None
```

### Self-Assembly Process

1. **Initialize** from DDColor's 100 queries
2. **Observe usage** on real images
3. **Apply attractor/repeller dynamics**:
   - Co-activating concepts ATTRACT
   - Non-co-activating concepts REPEL
4. **Cluster by behavior**
5. **Suggest names** based on patterns

### The Interface (Doc 203)

```
┌─────────────────────────────────────────────────────────────┐
│                    CONCEPT SPACE                            │
│                                                             │
│   ┌─────┐         ┌─────┐         ┌─────┐                  │
│   │ C0  │         │ C4  │         │ C9  │                  │
│   │ 4q  │         │ 18q │         │ 24q │                  │
│   └──┬──┘         └──┬──┘         └──┬──┘                  │
│      │               │               │                      │
│      ▼               ▼               ▼                      │
│   [unnamed]      [unnamed]       [unnamed]                  │
└─────────────────────────────────────────────────────────────┘

Operations:
- FOCUS: Examine a concept's behavior
- EXPLORE: Find similar concepts
- DISCOVER: Identify patterns
- NAME: Assign tentative labels
```

---

## Why This Matters

### The Semantic Gap

Traditional approach:
```
Random init → Training → Semantic content
```

Our approach:
```
Trained model → Extract structure → Self-assemble vocabulary → Name through exploration
```

### What We Can Skip

| Aspect | Traditional | Our Approach |
|--------|-------------|--------------|
| Architecture | Design | Derive from patterns |
| Lattice | Implicit | Explicit (φ-lattice) |
| Structure | Train | Extract from destination |
| Content | Train | Discover through exploration |

### What We Still Need

The **semantic meaning** of each query:
- Query 47 = sky? skin? vegetation?
- This requires observing actual usage

But we've reduced the problem from:
- "Learn 55M parameters" → "Name 100 concepts"

---

## The Attention Elimination Path

### What We Can Pre-compute

1. **MESH** = W_q.T @ W_k (per layer)
2. **query_MESH** = queries @ MESH
3. **Query structure** (clusters, similarity)

### What We Cannot Pre-compute

The **attention patterns** - which queries attend where:
- This depends on the input image
- This is the semantic content

### The Hybrid Solution

```
Pre-computed:                    Runtime:
─────────────                    ────────
query_MESH (all layers)          features = encoder(image)
cluster assignments              attention = softmax(query_MESH @ features.T)
φ-lattice positions              output = attention @ values
```

The query side is fully pre-computed. Only the feature side needs runtime computation.

---

## Connection to Prior Work

### Doc 189: Safe Dial Mechanism

The queries are the "dial", features are the "plates", attention is the "click".

### Doc 192: Boom-Newton Attention

89.5% of attention mass at 37% of positions. We can use sparse attention.

### Doc 208: Context Window Geometry

Context is compressible. The 100 queries are a compressed representation.

### Doc 213: Meta-Patterns

All models have the same structure: bilinear bottleneck on φ-lattice.

---

## The Path Forward

### Immediate Next Steps

1. **Run on real images** to get actual activation patterns
2. **Track which queries activate where** (spatial)
3. **Track which queries produce what colors**
4. **Build the semantic map** of 100 queries

### Longer Term

1. **Name the queries** through exploration
2. **Build direct routing** based on semantic map
3. **Skip attention** for known patterns
4. **Design new models** using the vocabulary

---

## Files

| File | Purpose |
|------|---------|
| `phi_geometric/evaluations/unnamed_vocabulary_assembly.py` | Framework |
| `phi_geometric/evaluations/analyze_query_structure.py` | Structure analysis |
| `phi_geometric/evaluations/query_structure.json` | Results |

---

## Conclusion

### What We Achieved

1. **Analyzed DDColor's 100 queries** - 94.4 effective rank, 10 clusters
2. **Built unnamed vocabulary framework** - attractor/repeller dynamics
3. **Created exploration interface** - focus, explore, discover, name
4. **Identified the semantic gap** - naming requires observation

### The Key Insight

**The vocabulary exists. We just don't know its names yet.**

DDColor's 100 queries form a complete color vocabulary:
- Each query is distinct (orthogonal)
- Queries cluster by similarity
- The structure is geometric (φ-lattice)

The semantic content is encoded in:
- Which queries attend where
- Which queries produce what colors
- How queries co-activate

This can be discovered through **observation**, not training.

### The Formula

```
Semantic Discovery = Structure Extraction + Usage Observation + Name Assignment

Where:
  Structure: From trained model (DDColor)
  Usage: From running on images
  Names: From human exploration
```

**We're not training. We're naming.**
