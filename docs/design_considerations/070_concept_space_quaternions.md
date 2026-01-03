# Design Consideration 070: Concept Space & Dual Quaternions

**Date**: December 28, 2024  
**Status**: Experimental  
**Author**: Lesley Gushurst

## Overview

This document captures discoveries from experimenting with concept space operations, including the insight that ConceptIdentity can be represented as a quaternion.

## The Dual Quaternion Model

Concepts in TruthSpace can be represented by **two complementary quaternions**:

### 1. SemanticQuaternion (Intrinsic Properties)

Encodes what a concept IS inherently:

```
x: Gender    (-1 female, +1 male, 0 neutral)
y: Age       (-1 young, +1 adult)
z: Agency    (-1 receiver, +1 initiator)
w: Animacy   (+1 human, 0.5 animal, -0.5 place, -1 abstract)
```

**Source**: Learned from word relationships (king-queen, man-woman)

**Example**:
```
king:   (x=+1.0, y=+1.0, z=+1.0, w=+1.0)  # male, adult, initiator, human
queen:  (x=-1.0, y=+1.0, z=+1.0, w=+1.0)  # female, adult, initiator, human
doctor: (x= 0.0, y=+1.0, z=+0.8, w=+1.0)  # neutral, adult, initiator, human
```

### 2. IdentityQuaternion (Relational Properties)

Encodes how a concept BEHAVES in context:

```
w: φ-direction  (agency from corpus usage)
x: Action signature (what it DOES)
y: Target signature (what it acts ON)
z: Category signature (what it IS-A)
```

**Source**: Learned from corpus frames (initiator → mediator → receiver)

**Example**:
```
holmes:   (w=+0.24, x=actions, y=targets, z=category)
watson:   (w=+0.73, x=actions, y=targets, z=category)
physics:  (w=+0.49, x=actions, y=targets, z=category)
```

## Key Discovery: Agency Correlation

The `z` component of SemanticQuaternion (predefined agency) and the `w` component of IdentityQuaternion (corpus-learned φ-direction) measure the **same thing from different sources**.

| Concept   | SQ.z (predefined) | IQ.φ (corpus) | Difference |
|-----------|-------------------|---------------|------------|
| detective | 0.80              | 1.00          | 0.20 ✓     |
| watson    | 0.30              | 0.73          | 0.43       |
| holmes    | 1.00              | 0.24          | 0.76       |
| doctor    | 0.80              | -1.00         | 1.80 ✗     |
| king      | 1.00              | -0.50         | 1.50 ✗     |

**Insight**: The corpus captures different usage patterns than our assumptions. "Doctor" in our corpus is mostly a receiver (acted upon), not an initiator.

## Analogies in Concept Space

### The Problem

Classic word2vec analogies use vector arithmetic:
```
king - man + woman = queen
```

This works because words are positioned by co-occurrence patterns.

### Why It Fails in Concept Space

In concept space, we encode **what concepts ARE and DO**, not their relative positions. The analogy "holmes:detective :: watson:?" asks "what category is Watson?" - but this requires the relationship to **exist in the corpus**.

**Test Results**:
```
holmes:detective :: watson:?  → (empty - no category found)
physics:science :: biology:?  → (empty - no category found)
```

**Root Cause**: The corpus lacks "X is a Y" frames. Watson's targets are:
```
['holmes', 'doorway', 'journal', 'victim', 'army']
```
None of these are category words like "doctor" or "companion".

### The Insight

> In concept space, analogies require **explicit relationships** in the corpus.
> Unlike word2vec which learns implicit relationships from co-occurrence,
> our concept space only knows what was explicitly stated.

This is both a limitation and a feature:
- **Limitation**: Can't infer unstated relationships
- **Feature**: Only returns relationships that are actually attested

### Solution: ConceptCorrector

To make analogies work, we need to inject the relationships:

```python
corrector.define_identity(ConceptIdentity(
    word='watson',
    category='doctor',
    primary_actions=['assist', 'accompany', 'treat'],
    primary_targets=['holmes', 'patients'],
    related_concepts=['holmes', 'medicine']
))
```

This adds "watson is a doctor" frames to the corpus, enabling the analogy to find the relationship.

## The 8D Concept Space

Together, the two quaternions form an **8-dimensional concept space**:

```
┌─────────────────────────────────────────────────────────────┐
│                    8D CONCEPT SPACE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   SemanticQuaternion          IdentityQuaternion            │
│   (INTRINSIC)                 (RELATIONAL)                  │
│                                                              │
│   ┌─────────────────┐         ┌─────────────────┐           │
│   │ x: Gender       │         │ w: φ-direction  │           │
│   │ y: Age          │         │ x: Actions      │           │
│   │ z: Agency       │         │ y: Targets      │           │
│   │ w: Animacy      │         │ z: Category     │           │
│   └─────────────────┘         └─────────────────┘           │
│                                                              │
│   WHO/WHAT it is              HOW it behaves                │
│   (static)                    (dynamic)                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implications

### 1. Concept Similarity

Two concepts are similar if both quaternions are similar:
- Same intrinsic properties (gender, age, animacy)
- Same behavioral properties (actions, targets, category)

### 2. Concept Correction

To change a concept's identity, we modify the IdentityQuaternion by injecting frames that shift its:
- Category (what it IS)
- Actions (what it DOES)
- Targets (what it acts ON)

### 3. Concept Navigation

The 8D space enables navigation:
- Find concepts with similar intrinsic properties
- Find concepts with similar behavioral properties
- Find concepts that bridge both

## Implementation

- `experiments/concept_space_experiments.py`: Experimental code
- `truthspace_lcm/core/semantic_quaternion.py`: SemanticQuaternion
- `truthspace_lcm/core/concept_correction.py`: ConceptIdentity and correction

## Next Steps

1. **Enrich corpus with category frames**: Add "X is a Y" statements
2. **Learn SemanticQuaternion from corpus**: Instead of predefined values
3. **Implement quaternion multiplication**: For proper 4D rotations
4. **Explore 8D operations**: Combined intrinsic + relational queries

## Conclusion

ConceptIdentity IS a quaternion - it has four components that encode a concept's relational identity. Combined with SemanticQuaternion, we get an 8D concept space that captures both what concepts ARE and how they BEHAVE.

The key insight: **analogies in concept space require explicit relationships**. This is fundamentally different from word2vec's implicit learning, and it means we need to either:
1. Ingest text that explicitly states the relationships
2. Use ConceptCorrector to inject the relationships manually

This is not a bug - it's a feature. The model only claims to know what it has evidence for.
