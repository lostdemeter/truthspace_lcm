# Design Consideration 014: Hierarchical Knowledge Navigation

## Overview

This document explores a general-purpose architecture for **geometric knowledge navigation** - the ability to determine which knowledge domain to activate based on input, using the same φ-based geometric principles that drive concept resolution.

The goal is to create an extensible structure that could eventually replace traditional LLM knowledge retrieval with pure geometric navigation.

## The Problem

Consider these inputs:
- "create a file called test.txt" → Bash domain
- "I'm feeling a bit out of touch" → Conversational domain
- "create a poem about autumn" → Creative domain

All three contain words that might activate similar primitives (e.g., "create", "touch"), but they belong to fundamentally different knowledge domains. How do we geometrically determine which domain to activate?

## Current State

Our current system has:
- **Flat knowledge**: All entries in one list
- **Single-level resolution**: Query → best matching entry
- **Domain-agnostic primitives**: CREATE, FILE, PROCESS, etc.

This works for bash commands but doesn't scale to multiple knowledge domains.

## Proposed Architecture: Fractal Knowledge Regions

### Core Insight: Self-Similar Structure

The same geometric operation applies at every level of the knowledge hierarchy:

```
1. Encode input
2. Find which region of current space it falls into
3. Descend into that region
4. Repeat until reaching actionable knowledge
```

This is fractal - the structure is self-similar at every scale.

### Knowledge Region Definition

```python
class KnowledgeRegion:
    name: str                        # "technical", "social", "creative"
    centroid: np.ndarray             # Position in truth space
    signature_primitives: List[str]  # Primitives that define this region
    children: List[KnowledgeRegion]  # Sub-regions (recursive)
    entries: List[KnowledgeEntry]    # Leaf knowledge (if terminal)
    
    def is_terminal(self) -> bool:
        return len(self.children) == 0
```

### Hierarchical Structure Example

```
ROOT (Universe of Knowledge)
├── TECHNICAL
│   ├── BASH
│   │   ├── FILE_OPERATIONS
│   │   │   └── [touch, rm, cp, mv, ...]
│   │   ├── PROCESS_OPERATIONS
│   │   │   └── [ps, kill, top, ...]
│   │   └── NETWORK_OPERATIONS
│   │       └── [ping, curl, ssh, ...]
│   ├── PYTHON
│   │   └── [...]
│   └── SYSTEM_ADMIN
│       └── [...]
├── SOCIAL
│   ├── GREETING
│   │   └── [hello responses, ...]
│   ├── EMOTIONAL
│   │   └── [empathy responses, ...]
│   └── SMALLTALK
│       └── [...]
├── CREATIVE
│   ├── WRITING
│   │   └── [...]
│   └── MUSIC
│       └── [...]
└── INFORMATIONAL
    ├── MEDICAL
    │   └── [...]
    └── LEGAL
        └── [...]
```

### Navigation Algorithm

```python
def navigate(query: str, current_region: KnowledgeRegion) -> KnowledgeEntry:
    query_pos = encode(query)
    
    if current_region.is_terminal():
        # At leaf - find best matching entry
        return find_best_entry(query_pos, current_region.entries)
    
    # Find closest child region
    best_child = None
    best_similarity = -inf
    
    for child in current_region.children:
        sim = similarity(query_pos, child.centroid)
        if sim > best_similarity:
            best_similarity = sim
            best_child = child
    
    # Descend into best matching region
    return navigate(query, best_child)
```

## The φ-Level Hypothesis

### Observation

Our φ-based encoding creates natural hierarchical levels:
- φ⁰ = 1.000 (Level 0)
- φ¹ = 1.618 (Level 1)
- φ² = 2.618 (Level 2)
- φ³ = 4.236 (Level 3)
- φ⁴ = 6.854 (Level 4)

### Hypothesis

**Domain-level primitives should occupy higher φ levels than concept-level primitives.**

```
φ⁴+ : Domain primitives (TECHNICAL, SOCIAL, CREATIVE, INFORMATIONAL)
φ³  : Category primitives (BASH, PYTHON, GREETING, WRITING)
φ²  : Subcategory primitives (FILE_OPS, PROCESS_OPS, EMOTIONAL)
φ⁰-¹: Action primitives (CREATE, DELETE, READ, COPY)
```

### Why This Might Work

Higher φ levels have larger values, so they **dominate** the encoding:
- φ⁴ = 6.854 >> φ⁰ = 1.0

If "I'm feeling out of touch" activates:
- SOCIAL domain at φ⁴ level → 6.854
- CREATE action at φ⁰ level → 1.0

The SOCIAL activation dominates, routing to the correct domain despite "touch" activating CREATE.

## Key Questions to Answer

### Q1: How do we define domain centroids?

**Options:**
1. **From signature primitives**: Centroid = encode(signature_primitive_keywords)
2. **From example queries**: Centroid = mean(encode(example_queries))
3. **Emergent from entries**: Centroid = mean(encode(entry.description) for entry in entries)

**Experiment needed**: Compare these approaches on domain classification accuracy.

### Q2: How do we handle cross-domain overlap?

**Problem:**
- "Create a file" → TECHNICAL/BASH
- "Create a poem" → CREATIVE/WRITING
- Both use CREATE primitive

**Hypothesis**: The non-overlapping primitives (FILE vs POEM/WRITING) provide discrimination.

**Experiment needed**: Test discrimination accuracy on overlapping queries.

### Q3: What is the optimal hierarchy depth?

**Trade-offs:**
- Too shallow: Poor discrimination between similar domains
- Too deep: Slow navigation, sparse leaf regions, overfitting

**Experiment needed**: Compare accuracy and speed at different depths.

### Q4: Should navigation be hard (pick one) or soft (weighted)?

**Hard navigation:**
```python
best_child = argmax(similarity(query, child) for child in children)
return navigate(query, best_child)
```

**Soft navigation:**
```python
weights = softmax([similarity(query, child) for child in children])
results = [navigate(query, child) for child in children]
return weighted_combine(results, weights)
```

**Experiment needed**: Compare accuracy of hard vs soft navigation.

### Q5: How does context (previous queries) affect navigation?

**Hypothesis**: Recent queries create a "prior" that biases navigation.

If last 3 queries were in BASH domain, bias toward BASH interpretation.

**Experiment needed**: Test with and without contextual priming.

## Implementation Sketch

### Phase 1: Domain Primitives

Add high-level domain primitives:

```python
# Domain primitives at φ⁴ level (dimension 12+)
Primitive("TECHNICAL", 12, 0, ["code", "command", "system", "file", "process", "network"])
Primitive("SOCIAL", 12, 1, ["feel", "think", "want", "hello", "thanks", "sorry"])
Primitive("CREATIVE", 12, 2, ["create", "write", "compose", "design", "imagine", "story"])
Primitive("INFORMATIONAL", 12, 3, ["what", "how", "why", "explain", "define", "describe"])
```

### Phase 2: Region Definition

Define knowledge regions with centroids:

```python
REGIONS = {
    'TECHNICAL': KnowledgeRegion(
        name='TECHNICAL',
        signature_primitives=['TECHNICAL', 'EXECUTE', 'SYSTEM'],
        children=[BASH_REGION, PYTHON_REGION, ...],
    ),
    'SOCIAL': KnowledgeRegion(
        name='SOCIAL',
        signature_primitives=['SOCIAL', 'GREETING', 'EMOTIONAL_STATE'],
        children=[GREETING_REGION, EMOTIONAL_REGION, ...],
    ),
    ...
}
```

### Phase 3: Hierarchical Navigation

Implement recursive navigation:

```python
def resolve_hierarchical(self, query: str) -> Tuple[KnowledgeRegion, KnowledgeEntry]:
    query_pos = self._encode(query)
    return self._navigate(query_pos, self.root_region)

def _navigate(self, query_pos: np.ndarray, region: KnowledgeRegion):
    if region.is_terminal():
        return region, self._find_best_entry(query_pos, region.entries)
    
    best_child = max(region.children, 
                     key=lambda c: self._similarity(query_pos, c.centroid))
    return self._navigate(query_pos, best_child)
```

## Relationship to Neural Networks

This architecture mirrors aspects of neural network behavior:

| Neural Network | Hierarchical Navigation |
|----------------|------------------------|
| Early layers detect broad patterns | Domain-level matching |
| Later layers detect specific features | Concept-level matching |
| Neuron activation | Region similarity score |
| Softmax over classes | Soft navigation weights |
| Attention mechanism | Contextual priming |

The key difference: **No training required**. The structure is defined by primitives and φ-encoding, not learned from data.

## Success Criteria

1. **Domain classification accuracy**: >90% on test queries
2. **Cross-domain discrimination**: Correctly route "create file" vs "create poem"
3. **Graceful degradation**: Unknown domains fall back sensibly
4. **Extensibility**: Adding new domain requires only defining region + entries

## Next Steps

1. Create experiments to answer Q1-Q5
2. Implement Phase 1 (domain primitives)
3. Test domain classification accuracy
4. Iterate on hierarchy structure based on results

## Philosophical Note

This architecture embodies the core TruthSpace philosophy: **geometry over training**. 

Instead of learning domain boundaries from millions of examples, we define them through:
- Primitive signatures (what concepts define a domain)
- φ-level hierarchy (domains at higher levels than concepts)
- Geometric navigation (similarity-based descent)

The structure is **intentional**, not emergent - but the behavior that emerges from navigation may surprise us.
