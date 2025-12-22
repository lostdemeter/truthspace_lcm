# TruthSpace LCM Architecture

## Overview

TruthSpace LCM is a **Dynamic Geometric Language Model** that performs all semantic operations as geometric operations in vector space. No neural networks, no training - just mathematics.

## Core Principles

> **Structure IS the data. Learning IS structure update.**

- **Entities** = Positions in ℝ^256 (learned from context)
- **Relations** = Vector offsets between entities (learned from pairs)
- **Facts** = (subject, relation, object) triples
- **Learning** = Iterative refinement until relations are consistent

## Primary Component: GeometricLCM (`geometric_lcm.py`)

The main component that provides dynamic learning and reasoning.

### Data Structures

```python
GeoEntity:
  - name: str
  - position: np.ndarray (256D)
  - entity_type: str

GeoRelation:
  - name: str
  - vector: np.ndarray (256D offset)
  - consistency: float (0-1)
  - instance_count: int

GeoFact:
  - subject: str
  - relation: str
  - object: str
```

### Key Methods

```python
# Learning
lcm.add_fact(subject, relation, object)  # Add a fact
lcm.ingest(text)                          # Parse NL to facts
lcm.tell(statement)                       # NL learning interface
lcm.learn(n_iterations, target_consistency)  # Update structure

# Inference
lcm.query(subject, relation, k)           # subject --relation--> ?
lcm.inverse_query(object, relation, k)    # ? --relation--> object
lcm.analogy(a, b, c, k)                   # a:b :: c:?
lcm.similar(entity, k)                    # Find similar entities
lcm.multi_hop(start, [relations], k)      # Chain queries
lcm.find_path(start, end, max_hops)       # Find paths

# Natural Language
lcm.ask(question)                         # NL question answering
lcm.tell(statement)                       # NL fact learning
```

### Learning Algorithm

```
For each iteration:
    1. Update relation vectors from current entity positions
       relation.vector = average(object.position - subject.position)
    
    2. Update entity positions to align with relations
       object.position → subject.position + relation.vector
       subject.position → object.position - relation.vector
    
    3. Check consistency (pairwise similarity of offsets)
       If consistency > target: STOP
```

## Supporting Components

### Vocabulary (`vocabulary.py`)

Provides deterministic word positions for initial entity placement.

```
Word → Hash → Random Seed → Unit Vector in ℝ^256
```

### FactParser (in `geometric_lcm.py`)

Parses natural language into facts using pattern matching.

**Supported Patterns:**
- "X is the capital of Y" → (Y, capital_of, X)
- "X wrote Y" → (X, wrote, Y)
- "X is in Y" → (X, located_in, Y)
- "X is a Y" → (X, is_a, Y)

## Data Flow

```
                    ┌─────────────┐
                    │   Natural   │
                    │   Language  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  FactParser │
                    │  (patterns) │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Extract Facts:       │
              │   (subject, rel, obj)  │
              └───────────┬────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Learn     │
                   │  (iterate)  │
                   └──────┬──────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Updated Geometry:     │
              │  - Entity positions    │
              │  - Relation vectors    │
              └────────────────────────┘
```

## Formulas

### Relation Learning
```
relation.vector = (1/n) Σᵢ (objectᵢ.position - subjectᵢ.position)

Normalized to unit length after averaging.
```

### Query (subject --relation--> ?)
```
target = subject.position + relation.vector
answer = argmax_entity(cosine(target, entity.position))
```

### Analogy (a:b :: c:?)
```
relation = b.position - a.position
target = c.position + relation
answer = argmax_entity(cosine(target, entity.position))
```

### Consistency
```
consistency = mean(pairwise_cosine(all_relation_offsets))

Target: > 0.95 for reliable analogies
```

## Directory Structure

```
truthspace-lcm/
├── truthspace_lcm/           # Main package
│   ├── __init__.py           # Package exports
│   ├── chat.py               # Interactive GeometricChat
│   └── core/
│       ├── __init__.py       # Core exports
│       ├── geometric_lcm.py  # Dynamic Geometric LCM (main)
│       ├── vocabulary.py     # Word positions, IDF, encoding
│       ├── knowledge.py      # Facts, triples, Q&A pairs
│       └── style.py          # Style extraction/transfer
├── tests/
│   ├── test_core.py          # Core tests (29)
│   └── test_chat.py          # Chat tests (20)
├── design_considerations/    # Research journey
│   ├── 033_dynamic_geometric_lcm.md  # Current architecture
│   └── 032_vsa_binding_extension.md  # VSA exploration
├── experiments/              # Exploration and prototypes
│   ├── sparse_vsa_exploration_v3.py  # 100% analogy breakthrough
│   └── geometric_lcm_full.py         # Full system prototype
├── run.py                    # Entry point
└── requirements.txt          # Dependencies (numpy)
```

## Validation Results

### Analogy Accuracy
- **100%** on capital-country analogies (france:paris :: germany:berlin)
- **100%** on author-book analogies (melville:moby_dick :: shakespeare:hamlet)
- Works across domains without interference

### Relation Consistency
- **99%+** consistency achieved after learning
- Converges in **4-10 iterations** typically

### Query Accuracy
- **100%** on learned facts
- Similarity scores > 0.98 for correct answers

## Design Decisions

### Why Learned Positions (not just hash)?
- **Hash gives random positions** - no semantic structure
- **Learning aligns relations** - makes analogies work
- **Key insight**: Relations must be INVARIANT across instances

### Why Iterative Learning?
- **Single pass insufficient** - relations not consistent
- **Iteration aligns all pairs** - converges to stable structure
- **Fast**: 4-10 iterations, ~0.01s for 100 facts

### Why Vector Offsets for Relations?
- **Simple**: relation = object - subject
- **Invertible**: subject = object - relation
- **Composable**: multi-hop = sum of relations

## Future Work

1. **Hierarchical Relations** - Support sub-types (located_in → city_in_country)
2. **Temporal Dynamics** - Track how structure changes over time
3. **Larger Scale** - Test with thousands of entities
4. **Integration** - Combine with style engine for styled responses
