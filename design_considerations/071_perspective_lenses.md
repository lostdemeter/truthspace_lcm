# Design Consideration 071: Perspective Lenses for Truth Projection

**Date**: December 28, 2024  
**Status**: Experimental  
**Author**: Lesley Gushurst

## Overview

Truth itself needs a filter - like perspective. Just as we project geometric truth to English (or any language), we can project it through different **lenses** that reveal different aspects of the same underlying reality.

## The Watson Paradox

Watson is **literally** a doctor (his profession in the stories), but **behaviorally** he acts as a companion/assistant. The corpus captures his behavior (φ=0.73, assists, accompanies) not his credential.

This reveals a fundamental insight: **the same truth, viewed through different lenses, yields different answers**.

## The Five Lenses

### 1. LITERAL Lens
**Question**: What IS this thing? (category, definition)

- Looks for explicit "X is a Y" relationships
- High precision, requires explicit statements
- Watson: "(unknown)" - corpus lacks "Watson is a doctor"

### 2. BEHAVIORAL Lens  
**Question**: How does this thing ACT? (φ-direction, actions)

- Uses φ-direction from corpus usage patterns
- Captures how the concept behaves grammatically
- Watson: "initiator" (φ=0.73) - he assists, watches, accompanies

### 3. RELATIONAL Lens
**Question**: How does this thing CONNECT? (targets, relationships)

- Looks at what the concept acts on and relates to
- Reveals network position and associations
- Watson: "connected to Holmes" - his primary relationship

### 4. NARRATIVE Lens
**Question**: What ROLE does this play? (story structure)

- Infers narrative function from patterns
- Identifies protagonist, antagonist, helper, chronicler, etc.
- Watson: "helper/sidekick" - he assists/accompanies

### 5. INTRINSIC Lens
**Question**: What IS this thing inherently? (semantic quaternion)

- Uses predefined semantic properties
- Gender, age, agency, animacy
- Watson: "male, adult, human" (x=1.0, y=1.0, z=0.3, w=1.0)

## The Chronicler Effect

A fascinating discovery: **Holmes has lower behavioral agency (φ=0.24) than Watson (φ=0.73)** in the corpus.

This is because Watson **narrates** the stories. As narrator, Watson is the grammatical subject more often:
- "I watched Holmes..."
- "I accompanied Holmes..."
- "I observed that Holmes..."

Holmes is **described**, making him appear as a receiver in the text even though he's the protagonist.

| Concept | Intrinsic Agency (z) | Behavioral Agency (φ) | Difference |
|---------|---------------------|----------------------|------------|
| Holmes  | 1.0 (high)          | 0.24 (neutral)       | 0.76       |
| Watson  | 0.3 (helper)        | 0.73 (initiator)     | -0.43      |

The **intrinsic lens** captures what we know Holmes to be.
The **behavioral lens** captures how he appears in the text.

## Lens Tensions

When lenses disagree, it reveals something important:

| Tension | Meaning |
|---------|---------|
| Intrinsic ≠ Behavioral | Narrator bias or role vs. action mismatch |
| Literal ≠ Behavioral | Definition vs. actual usage |
| Narrative ≠ Relational | Story role vs. network position |

These tensions are **information**, not errors. They reveal:
- Narrator perspective effects
- Gaps between definition and usage
- Multiple valid interpretations

## Implementation

```python
class Lens(Enum):
    LITERAL = "literal"       # What something IS
    BEHAVIORAL = "behavioral" # How something ACTS
    RELATIONAL = "relational" # How something CONNECTS
    NARRATIVE = "narrative"   # What ROLE it plays
    INTRINSIC = "intrinsic"   # What it IS inherently
```

Each lens provides a `PerspectiveView`:
- `primary_answer`: The main answer from this lens
- `confidence`: How confident the lens is
- `supporting_evidence`: What supports this answer

## Connection to Language Projection

This is analogous to projecting to English:

```
GEOMETRIC TRUTH (φ-space)
        │
        ├─→ English projection → "Watson assists Holmes"
        ├─→ French projection  → "Watson aide Holmes"
        └─→ Formal projection  → "Dr. Watson provides assistance"
```

Similarly:

```
GEOMETRIC TRUTH (φ-space)
        │
        ├─→ Literal lens    → "(unknown)"
        ├─→ Behavioral lens → "initiator"
        ├─→ Relational lens → "connected to Holmes"
        └─→ Narrative lens  → "helper/sidekick"
```

The truth is the same. The **perspective** determines what aspect we see.

## Use Cases

### 1. Answering "What is X?"
Choose the appropriate lens based on context:
- For definitions: LITERAL lens
- For behavior analysis: BEHAVIORAL lens
- For relationship mapping: RELATIONAL lens
- For story analysis: NARRATIVE lens

### 2. Resolving Ambiguity
When answers conflict, present multiple perspectives:
> "Watson is literally a doctor, but behaviorally acts as Holmes's companion and narratively serves as the chronicler."

### 3. Detecting Bias
Lens tensions reveal narrator or corpus bias:
> "Holmes appears low-agency in text (φ=0.24) due to Watson's narration, but is intrinsically high-agency (z=1.0)."

## Files

- `experiments/perspective_lenses.py`: Implementation and demo
- `truthspace_lcm/core/semantic_quaternion.py`: Intrinsic properties

## Conclusion

Truth needs perspective. The geometric core is the same, but different lenses reveal different facets:

- **LITERAL**: What it's called
- **BEHAVIORAL**: How it acts
- **RELATIONAL**: How it connects
- **NARRATIVE**: What role it plays
- **INTRINSIC**: What it inherently is

The "correct" answer depends on which lens you're using. This is not relativism - it's multi-faceted truth.
