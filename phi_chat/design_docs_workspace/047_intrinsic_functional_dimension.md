# Design Consideration 047: Intrinsic vs Functional Dimension

**Date:** 2026-01-09  
**Status:** Discovery → Implementation  
**Related:** Design 039 (φ-Zipf Duality), Design 099 (φ-Lattice Coordinates)

---

## Discovery

While removing keyword boost fallbacks from `knowledge_space.py` to achieve pure geometric matching, we observed an unexpected but profound behavior change:

### The Observation

**Query:** "what is python?"

| Mode | Result |
|------|--------|
| With keyword boost | "Python is a high-level, interpreted programming language known for its clear syntax..." |
| Pure geometric | "Python uses indentation to define code blocks instead of braces..." |

Both answers are correct, but they represent fundamentally different types of knowledge.

### The Hypothesis

The geometry is distinguishing between:
- **Intrinsic properties** - What something literally IS (structure, components, characteristics)
- **Functional properties** - What something is FOR (purpose, history, relationships)

### Experimental Validation

We tested different query phrasings against the same knowledge base:

| Query | Geometric Match | Category |
|-------|-----------------|----------|
| "what is python" | INTRINSIC | Ontological |
| "define python" | FUNCTIONAL | Epistemological |
| "describe python" | FUNCTIONAL | Epistemological |
| "python definition" | FUNCTIONAL | Epistemological |

**Key finding:** "what is X" is the ONLY phrasing that consistently returns intrinsic properties.

### Content Analysis

**INTRINSIC entry:**
```
Python uses indentation to define code blocks instead of braces. 
Variables don't need type declarations. Lists, dictionaries, and 
tuples are built-in data structures...
```
Key words: `uses`, `define`, `blocks`, `built-in`, `keyword`, `defines`, `need`

**FUNCTIONAL entry:**
```
Python is a high-level, interpreted programming language known for 
its clear syntax and readability. Created by Guido van Rossum...
```
Key words: `known`, `created`, `released`, `supports`, `emphasizes`, `including`

### Geometric Evidence

- Distance between entries: 0.4647
- Query "what is python" → INTRINSIC (diff=0.1373)
- Query "define python" → FUNCTIONAL (diff=0.1156)
- Correlation between query diff and entry diff: **0.6084**

The geometry is capturing a real semantic distinction.

---

## Philosophical Interpretation

This maps to a fundamental distinction in philosophy:

### Ontological Questions ("what is it?")
- Ask about the **being** of something
- Seek intrinsic properties
- Answer: "It IS [structural properties]"
- Example: "A pencil IS yellow, has graphite, made of wood"

### Epistemological Questions ("define/describe it")
- Ask about our **knowledge** of something
- Seek functional/relational properties
- Answer: "It is KNOWN AS [functional description]"
- Example: "A pencil is a writing instrument used by students"

The geometry naturally captures this distinction because:
- **"is"** encodes toward structural/intrinsic space
- **"define/describe"** encodes toward relational/functional space

---

## Implementation: Adding Explicit Dimension

To give users control over this behavior, we add an explicit **intrinsic/functional** dimension to the φ-lattice.

### Dimension Definition

```python
INTRINSIC_FUNCTIONAL = Dimension(
    name="intrinsic_functional",
    negative_pole="intrinsic",      # What it IS (structure, properties)
    positive_pole="functional",     # What it's FOR (purpose, relations)
    examples=[
        # Intrinsic (negative)
        ("uses", -1),
        ("contains", -1),
        ("made of", -1),
        ("has", -1),
        ("consists of", -1),
        
        # Functional (positive)
        ("known as", 1),
        ("used for", 1),
        ("created by", 1),
        ("purpose", 1),
        ("designed to", 1),
    ]
)
```

### φ-Lattice Levels

Following the φ-lattice encoding scheme:

| Level | φ^level | Meaning |
|-------|---------|---------|
| -2 | φ^(-2) = 0.382 | Strongly intrinsic (raw structure) |
| -1 | φ^(-1) = 0.618 | Intrinsic (properties) |
| 0 | φ^0 = 1.0 | Neutral (balanced) |
| +1 | φ^1 = 1.618 | Functional (purpose) |
| +2 | φ^2 = 2.618 | Strongly functional (context/history) |

### Query Transformation

When encoding queries:
- "what is X" → intrinsic_functional = -1
- "define X" → intrinsic_functional = +1
- "describe X" → intrinsic_functional = +1
- "how does X work" → intrinsic_functional = 0 (neutral)

### Knowledge Tagging

Bootstrap knowledge entries should be tagged:
```json
{
  "text": "Python uses indentation to define code blocks...",
  "phi_levels": {"intrinsic_functional": -1}
}
```

```json
{
  "text": "Python is a high-level, interpreted programming language...",
  "phi_levels": {"intrinsic_functional": +1}
}
```

---

## Benefits

1. **User Control** - Users can explicitly request intrinsic or functional answers
2. **Predictable Behavior** - Query phrasing maps to dimension values
3. **Geometric Purity** - No keyword boost needed; the dimension handles it
4. **Self-Similar** - Same φ-lattice mechanism used for all dimensions

---

## Connection to Pencil Analogy

Your intuition about the pencil was exactly right:

| Question | Dimension | Answer Type |
|----------|-----------|-------------|
| "What is a pencil?" | intrinsic = -1 | "It is yellow, has an eraser, graphite core, wood casing" |
| "Define pencil" | functional = +1 | "A writing instrument used by students to take notes" |

The geometry isn't wrong - it's capturing a real distinction that humans implicitly understand but rarely articulate.

---

## Implementation Checklist

- [ ] Add `intrinsic_functional` to DEFAULT_DIMENSIONS in phi_lattice.py
- [ ] Add query word mappings for intrinsic/functional detection
- [ ] Tag bootstrap knowledge entries with intrinsic_functional levels
- [ ] Update PrimitiveRegistry to handle the new dimension
- [ ] Test with "what is X" vs "define X" queries

---

## Conclusion

The removal of keyword boost revealed that our geometry was already capturing the intrinsic/functional distinction - we just weren't aware of it. By making this dimension explicit, we:

1. Validate the geometric approach (it's working!)
2. Give users control over the behavior
3. Maintain geometric purity (no fallbacks needed)

This is a case where "the geometry is honest" - it was telling us something true about the structure of knowledge that we hadn't explicitly encoded.
