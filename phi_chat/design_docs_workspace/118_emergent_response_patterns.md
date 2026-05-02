# Design Consideration 118: Emergent Response Patterns

## The Problem with Templates

The roadmap (doc 117) proposed a template system:

```python
TEMPLATES = {
    "definition": [
        "{concept} is {definition}.",
        "A {concept} is {definition}.",
    ],
}
```

**This violates our core principle.** Templates are:
- Hard-coded patterns (not emergent)
- Filling in blanks (not geometric)
- Require massive enumeration to scale

## The Solution: Patterns as Pairs

If patterns are just another kind of knowledge, they should emerge geometrically like everything else.

### Key Insight: Utterances are Compound Concepts

Just like "linear algebra" compounds "linear" + "algebra", the sentence:

> "The queen is the female form of king"

compounds:
- `queen` (concept)
- `female` (dimension value)
- `king` (concept)  
- `"X is the Y form of Z"` (structural pattern)

**The structural pattern itself is a concept with a position!**

### Pattern Pairs

```python
# Instead of templates, we have transformation pairs:

# Input-output pattern pairs
corpus.add_pair("what_is_X", "X_is_definition", "response_pattern")
corpus.add_pair("how_do_I_X", "to_X_you_should", "response_pattern")
corpus.add_pair("tell_me_about_X", "X_is_a_Y_that", "response_pattern")

# Greeting pattern pairs
corpus.add_pair("hello", "hello_response", "greeting_pattern")
corpus.add_pair("how_are_you", "i_am_well_response", "greeting_pattern")
corpus.add_pair("goodbye", "goodbye_response", "farewell_pattern")

# Question-answer pattern pairs
corpus.add_pair("what_is", "definition_answer", "question_pattern")
corpus.add_pair("who_is", "person_answer", "question_pattern")
corpus.add_pair("where_is", "location_answer", "question_pattern")
```

### The φ-Duality

From concept compounding (doc 109), we have:

```
scale[i] = φ^(-rank[i])  # φ-Zipf scaling
```

This applies to patterns too:

```
Utterance = pattern * φ^0 + concept1 * φ^(-1) + concept2 * φ^(-2) + ...
```

The pattern is the "head" (most important), concepts are "modifiers".

## Architecture: Pattern Space

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED CONCEPT SPACE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CONTENT CONCEPTS          PATTERN CONCEPTS                  │
│  ─────────────────         ────────────────                  │
│  king, queen, man          what_is_X, X_is_Y                 │
│  dog, cat, house           how_do_I_X, to_X_you              │
│  happy, sad, angry         tell_me_about, here_is_info       │
│                                                              │
│  Same φ-geometry!          Same φ-geometry!                  │
│  Same dimensions!          Same dimensions!                  │
│                                                              │
│  king ──gender──→ queen    question ──response──→ answer     │
│  boy ──age──→ man          formal ──register──→ casual       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Pattern Dimensions

Just like content has dimensions (gender, age, size), patterns have dimensions:

| Dimension | Negative Pole | Positive Pole |
|-----------|---------------|---------------|
| `response_type` | question | answer |
| `register` | formal | casual |
| `verbosity` | terse | elaborate |
| `certainty` | uncertain | definite |
| `politeness` | direct | polite |

### Example: Generating a Response

**Input:** "What is a king?"

**Step 1: Parse to position**
```python
# "what is" → question pattern position
# "king" → content concept position
input_pos = compound([pattern_pos("what_is"), concept_pos("king")])
```

**Step 2: Traverse to response**
```python
# Apply response_type transformation (question → answer)
output_pos = input_pos + dimension_delta("response_type")
```

**Step 3: Find nearest pattern**
```python
# Find pattern concept nearest to output position
nearest_pattern = find_nearest(output_pos, pattern_space)
# → "X_is_definition" pattern
```

**Step 4: Compound with content**
```python
# Get content for the definition
king_def = get_definition("king")  # from corpus
# Compound pattern with content
response = compound(nearest_pattern, king_def)
# → "A king is a male monarch"
```

## Implementation: PatternCorpus

```python
class PatternCorpus(SelfAssemblingCorpus):
    """
    Corpus that treats patterns as first-class concepts.
    
    Patterns are transformation pairs just like content:
    - "what_is" → "definition_answer" (response_type dimension)
    - "formal_greeting" → "casual_greeting" (register dimension)
    """
    
    def __init__(self):
        super().__init__()
        self._pattern_examples: Dict[str, List[str]] = {}
    
    def add_pattern_pair(self, input_pattern: str, output_pattern: str,
                         dimension: str, examples: List[Tuple[str, str]] = None):
        """
        Add a pattern transformation pair with examples.
        
        Args:
            input_pattern: Pattern name for input (e.g., "what_is_X")
            output_pattern: Pattern name for output (e.g., "X_is_definition")
            dimension: The dimension of transformation
            examples: List of (input, output) example strings
        """
        self.add_pair(input_pattern, output_pattern, dimension)
        
        if examples:
            self._pattern_examples[input_pattern] = [e[0] for e in examples]
            self._pattern_examples[output_pattern] = [e[1] for e in examples]
    
    def match_pattern(self, utterance: str) -> Optional[str]:
        """
        Find which pattern best matches an utterance.
        
        Uses geometric similarity, not string matching.
        """
        # Tokenize utterance
        words = self._tokenize(utterance)
        
        # Compute position from known words
        known_positions = []
        for word in words:
            pos = self.get_position(word)
            if pos is not None:
                known_positions.append(pos)
        
        if not known_positions:
            return None
        
        # Average position
        utterance_pos = np.mean(known_positions, axis=0)
        
        # Find nearest pattern
        pattern_concepts = [p for p in self.concepts if p.startswith("pattern_")]
        if not pattern_concepts:
            return None
        
        nearest = self.find_nearest(utterance_pos, n=1)
        return nearest[0][0] if nearest else None
    
    def generate_response(self, input_pattern: str, 
                          content_concepts: List[str]) -> str:
        """
        Generate response by traversing from input pattern to output pattern.
        """
        # Get input pattern position
        input_pos = self.get_position(input_pattern)
        if input_pos is None:
            return None
        
        # Find output pattern via dimension traversal
        # (the response_type dimension connects question → answer patterns)
        for pair in self.pairs:
            if pair.source == input_pattern:
                output_pattern = pair.target
                break
        else:
            return None
        
        # Get example for output pattern
        if output_pattern in self._pattern_examples:
            template = self._pattern_examples[output_pattern][0]
            # Fill with content (this is the only "template-like" step,
            # but the template itself was learned, not hard-coded)
            return self._fill_pattern(template, content_concepts)
        
        return None
```

## The Key Difference

### Templates (OLD - BAD)
```python
# Hard-coded by programmer
TEMPLATES = {"definition": "{X} is {Y}"}
```

### Emergent Patterns (NEW - GOOD)
```python
# Learned from examples
corpus.add_pattern_pair(
    "what_is_X", 
    "X_is_definition",
    "response_type",
    examples=[
        ("What is a king?", "A king is a male monarch."),
        ("What is a dog?", "A dog is a domesticated canine."),
        ("What is love?", "Love is a deep affection."),
    ]
)
```

The pattern EMERGES from the examples. The geometry captures:
- What makes a "what is" question
- What makes a "definition" answer
- The transformation between them

## Scaling

With templates, you need O(n) templates for n response types.

With emergent patterns:
- Add examples → patterns emerge
- New patterns interpolate from existing ones
- The geometry generalizes

```
Known patterns:
  what_is_X → X_is_definition
  who_is_X → X_is_person
  where_is_X → X_is_location

New query: "when is X?"
→ Position is near other question patterns
→ Traverse response_type dimension
→ Find nearest: temporal_answer pattern
→ Generate: "X occurs at [time]"
```

## Connection to Music Box Principle

The music box has:
- **Pins** (fixed structure) = pattern positions in geometry
- **Cylinder** (interchangeable) = domain-specific content
- **Comb** (produces sound) = surface realization

The patterns ARE the pins. They're fixed geometric positions.
The content is the cylinder. It changes per domain.
The response is the sound. It emerges from the combination.

## Implementation Plan

1. **Add pattern dimension** to base corpus
   - `response_type`: question ↔ answer
   - `register`: formal ↔ casual
   - `verbosity`: terse ↔ elaborate

2. **Add pattern pairs** with examples
   - Greeting patterns
   - Question patterns (what, who, where, when, why, how)
   - Statement patterns
   - Acknowledgment patterns

3. **Implement pattern matching**
   - Parse utterance to position
   - Find nearest pattern concept

4. **Implement response generation**
   - Traverse from input pattern to output pattern
   - Compound with content concepts
   - Surface realization from examples

## The Principle

> **Patterns are concepts. Responses are traversals. Templates are forbidden.**

---

*"The pattern is not a template to fill. The pattern is a position to traverse to."*
