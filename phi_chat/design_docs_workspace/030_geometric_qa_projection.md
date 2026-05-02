# Geometric Q&A Projection Framework

## The Core Insight

A statement S can be **decomposed** into multiple (Q, A) pairs through geometric projection onto different question-type axes. This is the **holographic** aspect: a single statement contains multiple "views" (answers), just like a hologram contains multiple viewing angles.

## The Problem with Direct Matching

Traditional approach:
```
Store: "Captain Ahab is the monomaniacal captain of the Pequod"
Query: "Who is Captain Ahab?"
Match: word overlap / embedding similarity
```

**Problem**: Statement space ≠ Q&A space. We're matching apples to oranges.

## The Holographic Solution

### Level 1: Text → Sentences
Extract clean sentences from raw text.

### Level 2: Sentences → Semantic Triples
Parse each sentence into (Subject, Predicate, Object, Modifiers):

```
"Captain Ahab is the monomaniacal captain of the Pequod"
    ↓
Subject:   "Captain Ahab"
Predicate: "is"
Object:    "the monomaniacal captain"
Modifiers: {LOCATION: "of the Pequod", ATTRIBUTE: "monomaniacal"}
```

### Level 3: Triples → Q&A Pairs
Project each triple onto question-type axes:

```
Triple: (Captain Ahab, is, monomaniacal captain, {LOCATION: of the Pequod})
    ↓
IDENTITY axis:  Q: "Who is Captain Ahab?"     A: "the monomaniacal captain"
LOCATION axis:  Q: "Where is Captain Ahab?"   A: "of the Pequod"  
ATTRIBUTE axis: Q: "What kind of captain?"    A: "monomaniacal"
```

### Level 4: Query → Match → Answer
Encode the query with its axis direction, match against stored question vectors.

## Geometric Interpretation

### Question Axes

Each question type defines a **direction** in semantic space:

```
        IDENTITY axis (WHO/WHAT)
              ↑
              |
              |
REASON ←──────●──────→ LOCATION
(WHY)         |         (WHERE)
              |
              ↓
         TIME axis (WHEN)
```

### Projection as Answer Extraction

A statement S lives in high-dimensional space. Projecting S onto an axis extracts the answer for that question type:

```
v_S = statement vector
v_axis = question type axis

answer_relevance = v_S · v_axis  (dot product)
```

### The Holographic Principle

In holography:
- A 2D hologram encodes 3D information
- Different viewing angles reveal different views

In Q&A:
- A single statement encodes multiple Q&A pairs
- Different question types reveal different answers

```
Statement S
    /|\
   / | \
  /  |  \
 ↓   ↓   ↓
Q1  Q2  Q3   (different axis projections)
A1  A2  A3   (different answers)
```

## Mathematical Formulation

### Encoding

```python
def encode_question(question: str, axis: str) -> vector:
    content = encode_text(question)      # Word positions
    axis_dir = axis_vectors[axis]        # Axis direction
    return 0.7 * content + 0.3 * axis_dir  # Combined
```

### Matching

```python
def match(query: str, stored_pairs: List[QAPair]) -> QAPair:
    q_vec = encode_question(query, detect_axis(query))
    
    best_match = max(stored_pairs, key=lambda p: 
        dot(q_vec, p.question_vec) * axis_boost(query, p) * p.confidence
    )
    return best_match
```

### Axis Detection

```python
def detect_axis(question: str) -> str:
    if question.startswith("who "):   return "IDENTITY"
    if question.startswith("what "):  return "DEFINITION"
    if question.startswith("where "): return "LOCATION"
    if question.startswith("when "):  return "TIME"
    if question.startswith("why "):   return "REASON"
    if question.startswith("how "):   return "METHOD"
    return "IDENTITY"  # default
```

## The Recursive Aspect

The projection is **recursive**:

1. **Level 1**: Raw text → Sentences (boundary extraction)
2. **Level 2**: Sentences → Triples (semantic parsing)
3. **Level 3**: Triples → Q&A pairs (axis projection)
4. **Level 4**: Q&A pairs → Refined answers (query matching)

Each level is a holographic projection that extracts more structure.

## Why This is Geometric (Not Just Pattern Matching)

1. **Axes are orthogonal directions** in semantic space
2. **Projection** extracts the component along an axis
3. **Dot product** measures alignment between query and stored question
4. **The same statement** yields different answers depending on projection axis

This is fundamentally different from keyword matching or embedding similarity:
- Keywords: Does the answer contain the query words?
- Embeddings: Is the answer semantically similar to the query?
- **Geometric projection**: Does the answer fill the gap defined by the query's axis?

## Connection to Holographic Stereoscopy

From the holographer's workbench:

```
I_L = I - αE    (left eye view)
I_R = I + αE    (right eye view)
```

The **gap** between views IS the depth information.

In Q&A:
```
Question = Content - Gap    (has missing information)
Answer   = Content + Fill   (provides missing information)
```

The **gap** in the question IS the intent. The answer **fills** that gap.

## Implementation Files

- `papers/holographic_projection.py` - Basic projection system
- `papers/recursive_holographic_qa.py` - Full recursive implementation
- `papers/holographic_qa_general.py` - Generalized for any text

## Results

| Query | Axis | Answer |
|-------|------|--------|
| Who is Captain Ahab? | IDENTITY | "the monomaniacal captain of the Pequod" |
| What is Moby Dick? | DEFINITION | "a giant white sperm whale" |
| Why does Ahab hunt? | REASON | "the whale took his leg" |
| Who is Sherlock Holmes? | IDENTITY | "a famous detective in London" |
| Who is Elizabeth Bennet? | IDENTITY | "the protagonist of Pride and Prejudice" |

## Key Insight

**The question type defines the projection axis. The answer is what remains after projecting the statement onto that axis.**

This is a purely geometric operation - no LLM, no training, just:
1. Parse statements into triples
2. Project triples onto axes to generate Q&A pairs
3. Match queries by axis-aware vector similarity
