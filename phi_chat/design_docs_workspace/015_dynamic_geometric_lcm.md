# Design Consideration 015: Dynamic Geometric LCM

## Overview

This document describes the architecture for a **scalable, generalizable geometric language model** that can:
1. Bootstrap from minimal seed primitives
2. Discover new primitives from ingested knowledge
3. Organize knowledge into domain hierarchies automatically
4. Scale to arbitrary knowledge domains

This is the foundation for replacing traditional LLMs with pure geometric computation.

## The Core Insight

Traditional LLMs:
- Learn patterns from massive data through training
- Knowledge emerges from statistical regularities
- Fixed architecture after training

Geometric LCM:
- **Bootstrap** with intentional seed primitives (the "crystal seed")
- **Discover** new primitives from patterns in data
- **Position** knowledge geometrically, not statistically
- **Grow** dynamically by adding primitives and knowledge

The key realization: **Primitives themselves can emerge from data, not just be hand-coded.**

## Architecture

### Three-Layer Primitive Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  SEED PRIMITIVES (Universal, hand-coded)                    │
│  - Actions: CREATE, DESTROY, TRANSFORM, MOVE, READ, WRITE   │
│  - Relations: INTO, FROM, WITH, BEFORE, AFTER               │
│  - Structures: SEQUENCE, HIERARCHY, NETWORK, COLLECTION     │
│  - Social: GREETING, GRATITUDE, REQUEST                     │
│  - Epistemic: KNOW, BELIEVE, DOUBT, WANT                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  EMERGENT PRIMITIVES (Discovered from data)                 │
│  - Domain-specific: BASH_FILE, COOKING_RECIPE, MEDICAL_FEVER│
│  - Co-occurrence clusters: COPY_FILE, INFECTION_WOUND       │
│  - Automatically created as knowledge is ingested           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  KNOWLEDGE ENTRIES (Positioned in truth space)              │
│  - Each entry has: content, description, domain, position   │
│  - Position computed from seed + emergent primitives        │
│  - Organized into domain clusters                           │
└─────────────────────────────────────────────────────────────┘
```

### Seed Primitives

The seed primitives are the **irreducible core** - concepts that appear across ALL domains:

```python
# Actions (universal verbs)
CREATE, DESTROY, TRANSFORM, MOVE
READ, WRITE, SEARCH, COMPARE
CONNECT, SEPARATE, COMBINE, FILTER

# Relations (universal prepositions/connectors)
INTO, FROM, WITH, ABOUT
BEFORE, AFTER, DURING, UNTIL

# Structures (universal organizational patterns)
SEQUENCE, HIERARCHY, NETWORK, COLLECTION

# Quantities (universal amounts)
ONE, MANY, ALL, NONE

# Social (universal human interaction)
GREETING, GRATITUDE, APOLOGY, REQUEST

# Epistemic (universal knowledge states)
KNOW, BELIEVE, DOUBT, WANT

# Evaluation (universal judgments)
GOOD, BAD, IMPORTANT, TRIVIAL
```

These 40 seed primitives occupy dimensions 0-9 of truth space.

### Emergent Primitive Discovery

When knowledge is ingested, the system discovers new primitives:

```python
def discover_emergent_primitives():
    # 1. Find domain-specific words
    for word in vocabulary:
        if appears_mostly_in_one_domain(word):
            create_primitive(f"{DOMAIN}_{word}")
    
    # 2. Find co-occurring word clusters
    for cluster in find_cooccurrence_clusters():
        create_primitive(cluster)
```

**Domain-specific words**: Words that appear >70% in one domain become primitives for that domain.
- "file" → BASH_FILE
- "recipe" → COOKING_RECIPE
- "fever" → MEDICAL_FEVER

**Co-occurrence clusters**: Words that frequently appear together form compound primitives.
- "file" + "copy" → COPY_FILE
- "wound" + "infection" → INFECTION_WOUND

Emergent primitives occupy dimensions 10+ of truth space.

### Knowledge Ingestion Pipeline

```
Input: {content, description, domain}
           ↓
    Tokenize description
           ↓
    Update word statistics
           ↓
    Encode with current primitives
           ↓
    Add to domain
           ↓
    (After batch) Discover new primitives
           ↓
    Re-encode all entries
           ↓
    Update domain centroids
```

### Domain Management

Domains are **regions in truth space** defined by their entries:

```python
class Domain:
    name: str
    centroid: np.ndarray  # Mean position of entries
    entries: List[KnowledgeEntry]
    primitives: List[Primitive]  # Domain-specific emergent primitives
```

Domain centroids are computed as the mean of entry positions. This creates natural clustering - entries in the same domain will be geometrically close.

### Query Resolution

Two-stage resolution:

```python
def resolve(query):
    # Stage 1: Domain detection
    query_pos = encode(query)
    best_domain = argmax(similarity(query_pos, domain.centroid))
    
    # Stage 2: Entry matching within domain
    best_entry = argmax(similarity(query_pos, entry.position) 
                        for entry in best_domain.entries)
    
    return best_entry
```

## Experimental Results

### Experiment 6: Dynamic LCM Prototype

Tested with 4 domains: bash (22 entries), cooking (15), medical (14), social (12)

**Results:**
- Emergent primitives discovered: 33
- Domain detection accuracy: 54.2%
- Content match accuracy: 29.2%

**Key Findings:**

1. **Emergent primitives work**: "list the files" correctly routes to bash because "files" activates BASH_FILES.

2. **Domain-specific vocabulary is critical**: Queries without domain-specific words fail. "chop the onions" fails because "chop" and "onions" aren't in any primitive.

3. **Common words cause interference**: Medical domain has common words like "food", "water" that match many queries.

4. **Description quality matters**: Better descriptions with more domain-specific keywords improve accuracy.

## Challenges and Solutions

### Challenge 1: Vocabulary Coverage

**Problem**: Queries may use words not in any primitive.

**Solutions**:
1. **Larger seed vocabulary**: Add more synonyms to seed primitives
2. **Aggressive emergent discovery**: Lower thresholds for creating emergent primitives
3. **Embedding fallback**: Use word embeddings to find nearest primitive for unknown words

### Challenge 2: Domain Overlap

**Problem**: Some words appear in multiple domains ("create", "show", "make").

**Solutions**:
1. **Context weighting**: Weight domain-specific primitives higher than universal ones
2. **Phrase-level encoding**: Encode multi-word phrases, not just individual words
3. **Negative discrimination**: Penalize domains when query contains "anti-keywords"

### Challenge 3: Sparse Domains

**Problem**: Domains with few entries have unreliable centroids.

**Solutions**:
1. **Minimum entry threshold**: Require N entries before domain is active
2. **Hierarchical fallback**: Fall back to parent domain if child is sparse
3. **Prior weighting**: Weight domain by entry count

## Comparison to Traditional LLMs

| Aspect | Traditional LLM | Geometric LCM |
|--------|-----------------|---------------|
| Knowledge acquisition | Training on massive data | Ingestion + primitive discovery |
| Knowledge representation | Distributed weights | Geometric positions |
| Inference | Forward pass through network | Similarity computation |
| Scalability | Fixed after training | Grows with ingestion |
| Interpretability | Black box | Explicit primitives |
| Compute requirements | GPU clusters | CPU sufficient |
| Update mechanism | Fine-tuning | Add entries + primitives |

## Future Directions

### 1. Hierarchical Domains

Create domain hierarchies automatically:
```
TECHNICAL
├── BASH
├── PYTHON
└── NETWORKING

CREATIVE
├── WRITING
└── MUSIC
```

### 2. Contextual Priming

Use conversation history to bias domain selection:
```python
def resolve_with_context(query, history):
    prior = compute_prior_from_history(history)
    query_pos = encode(query) + prior_weight * prior
    ...
```

### 3. Active Learning

Identify knowledge gaps and request specific information:
```python
def identify_gaps():
    # Find queries that fall between domain centroids
    # Request knowledge to fill those regions
```

### 4. Cross-Domain Transfer

Share primitives between related domains:
```python
# COOKING and CHEMISTRY both use "mix", "heat", "combine"
# Share these primitives rather than duplicating
```

## Implementation

The prototype is implemented in:
- `truthspace_lcm/core/dynamic_lcm.py` - Core DynamicLCM class
- `experiments/exp06_dynamic_lcm.py` - Test with multiple domains

Key classes:
- `Primitive` - Seed or emergent primitive
- `KnowledgeEntry` - Positioned knowledge item
- `Domain` - Knowledge region with centroid
- `DynamicLCM` - Main class with ingestion and resolution

## Conclusion

The Dynamic Geometric LCM demonstrates that:

1. **Primitives can emerge from data** - Not all primitives need to be hand-coded
2. **Knowledge can be positioned geometrically** - No training required
3. **The system can scale** - Add domains and entries without retraining
4. **Interpretability is preserved** - Every decision traces to explicit primitives

Current accuracy (54% domain detection) is limited by vocabulary coverage, not the geometric approach itself. With better primitive discovery and larger seed vocabularies, this architecture could scale to general-purpose language understanding.

The key insight: **We're not replacing neural network computation with geometry. We're replacing learned representations with intentional structure that can grow.**
