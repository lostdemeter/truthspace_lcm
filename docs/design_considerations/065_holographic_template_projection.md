# Design Consideration 065: Holographic Template Projection

## Date: 2024-12-28

## The Problem

Current approach: Hard-coded templates
```python
TEMPLATES = {
    "who_is": "{entity} is a {role} who {actions}.",
    "what_did": "{entity} {action} {target}.",
}
```

**Issues**:
- Rigid, unnatural
- Must anticipate all patterns
- Doesn't adapt to corpus style
- One-size-fits-all

## The Insight

**Templates are patterns that emerge from similar responses.**

If we have many Q&A pairs:
- "Who is Watson?" → "Watson is a loyal doctor who assists Holmes."
- "Who is Darcy?" → "Darcy is a proud gentleman who loves Elizabeth."
- "Who is Moriarty?" → "Moriarty is a criminal mastermind who challenges Holmes."

The **template structure** emerges from interference:
- Content words (Watson, Darcy, Moriarty) have different phases → cancel
- Structure words (is, a, who) have aligned phases → reinforce
- Pattern: `{X} is a {adj} {role} who {verb} {Y}`

## Mathematical Foundation

### Response as Complex Vector

Each response R can be encoded as a complex vector:
```
R = Σ w_i · e^(iθ_i)

Where:
  w_i = word at position i
  θ_i = phase based on word type and position
```

### Phase Assignment

| Word Type | Phase | Behavior |
|-----------|-------|----------|
| Entity (varies per response) | hash(word) | Cancels across responses |
| Structure (same across responses) | 0 | Reinforces |
| Role/Attribute | π/4 | Partially reinforces |
| Action | π/2 | Partially reinforces |

### Interference

Given N similar responses R₁, R₂, ..., Rₙ:
```
Template = (1/N) Σ Rᵢ

Words with aligned phases → high magnitude (keep)
Words with random phases → low magnitude (replace with slot)
```

## Algorithm

### Step 1: Retrieve Similar Q&A Pairs

```python
def find_similar_qa(query: str, k: int = 5) -> List[Tuple[str, str]]:
    """Find k most similar question-answer pairs."""
    query_type = detect_question_type(query)  # WHO, WHAT, etc.
    query_entity = extract_entity(query)
    
    # Find responses to similar questions
    similar = []
    for qa in corpus:
        if qa.question_type == query_type:
            similarity = compute_similarity(query, qa.question)
            similar.append((similarity, qa))
    
    return [qa for _, qa in sorted(similar, reverse=True)[:k]]
```

### Step 2: Encode Responses as Complex Vectors

```python
def encode_response(response: str) -> Dict[str, complex]:
    """Encode response as word → complex number mapping."""
    words = tokenize(response)
    encoding = {}
    
    for i, word in enumerate(words):
        # Position-based phase component
        pos_phase = (i / len(words)) * 2 * np.pi
        
        # Word-type phase component
        type_phase = get_word_type_phase(word)
        
        # Combined phase
        phase = pos_phase + type_phase
        
        # Magnitude based on importance
        magnitude = get_word_importance(word)
        
        # Complex encoding
        encoding[word] = magnitude * np.exp(1j * phase)
    
    return encoding
```

### Step 3: Compute Interference Pattern

```python
def compute_interference(responses: List[str]) -> Dict[str, complex]:
    """Compute interference pattern from multiple responses."""
    # Encode all responses
    encodings = [encode_response(r) for r in responses]
    
    # Sum complex values for each word
    interference = defaultdict(complex)
    for encoding in encodings:
        for word, value in encoding.items():
            interference[word] += value
    
    # Normalize
    n = len(responses)
    for word in interference:
        interference[word] /= n
    
    return interference
```

### Step 4: Extract Template

```python
def extract_template(interference: Dict[str, complex], 
                     threshold: float = 0.5) -> str:
    """Extract template from interference pattern."""
    # Sort by position (need to track positions)
    # Words with |z| > threshold → keep as literal
    # Words with |z| < threshold → replace with slot
    
    template_parts = []
    for word, z in sorted_by_position(interference):
        magnitude = abs(z)
        if magnitude > threshold:
            template_parts.append(word)
        else:
            # Determine slot type from word category
            slot_type = infer_slot_type(word)
            template_parts.append(f"{{{slot_type}}}")
    
    return " ".join(template_parts)
```

### Step 5: Fill Template

```python
def fill_template(template: str, query: str, knowledge: Knowledge) -> str:
    """Fill template slots with query-specific content."""
    entity = extract_entity(query)
    entity_profile = knowledge.get_entity(entity)
    
    filled = template
    filled = filled.replace("{entity}", entity)
    filled = filled.replace("{role}", entity_profile.primary_role)
    filled = filled.replace("{action}", entity_profile.primary_action)
    filled = filled.replace("{target}", entity_profile.primary_target)
    
    return filled
```

## Example Walkthrough

### Input
Query: "Who is Holmes?"

### Step 1: Find Similar Q&A
```
Q: "Who is Watson?"  → A: "Watson is a loyal doctor who assists Holmes."
Q: "Who is Darcy?"   → A: "Darcy is a proud gentleman who loves Elizabeth."
Q: "Who is Moriarty?" → A: "Moriarty is a cunning villain who opposes Holmes."
```

### Step 2: Encode Responses

Response 1: "Watson is a loyal doctor who assists Holmes"
```
Watson:   1.0 · e^(i·0.1)      # Entity, position 0
is:       1.0 · e^(i·0.0)      # Structure, phase 0
a:        0.5 · e^(i·0.0)      # Structure, phase 0
loyal:    0.8 · e^(i·π/4)      # Attribute
doctor:   1.0 · e^(i·π/4)      # Role
who:      0.5 · e^(i·0.0)      # Structure
assists:  1.0 · e^(i·π/2)      # Action
Holmes:   1.0 · e^(i·0.9)      # Entity, position ~1
```

Response 2: "Darcy is a proud gentleman who loves Elizabeth"
```
Darcy:    1.0 · e^(i·0.2)      # Different entity → different phase
is:       1.0 · e^(i·0.0)      # Same structure → same phase
a:        0.5 · e^(i·0.0)      # Same
proud:    0.8 · e^(i·π/4)      # Attribute (similar phase)
gentleman:1.0 · e^(i·π/4)      # Role (similar phase)
who:      0.5 · e^(i·0.0)      # Same
loves:    1.0 · e^(i·π/2)      # Action (similar phase)
Elizabeth:1.0 · e^(i·0.8)      # Different entity
```

### Step 3: Interference

```
Watson + Darcy + Moriarty → phases differ → magnitude ≈ 0.3 (SLOT)
is + is + is → phases align → magnitude ≈ 1.0 (KEEP)
a + a + a → phases align → magnitude ≈ 0.5 (KEEP)
loyal + proud + cunning → phases similar → magnitude ≈ 0.6 (SLOT or KEEP)
doctor + gentleman + villain → phases similar → magnitude ≈ 0.7 (SLOT)
who + who + who → phases align → magnitude ≈ 0.5 (KEEP)
assists + loves + opposes → phases similar → magnitude ≈ 0.6 (SLOT)
Holmes + Elizabeth + Holmes → phases differ → magnitude ≈ 0.5 (SLOT)
```

### Step 4: Extract Template

```
{entity} is a {attribute} {role} who {action} {target}
```

### Step 5: Fill Template

```
entity = "Holmes"
attribute = "brilliant" (from Holmes profile)
role = "detective" (from Holmes profile)
action = "investigates" (from Holmes profile)
target = "crimes" (from Holmes profile)

Result: "Holmes is a brilliant detective who investigates crimes."
```

## Advantages

1. **Dynamic**: Templates emerge from data, not hard-coded
2. **Adaptive**: Different corpora → different templates
3. **Natural**: Templates match corpus style
4. **Extensible**: Add more Q&A pairs → richer templates
5. **Geometric**: Pure interference, no ML

## Implementation Plan

### Phase 1: Basic Projection
- [ ] Store Q&A pairs with question type
- [ ] Implement response encoding
- [ ] Implement interference computation
- [ ] Implement template extraction

### Phase 2: Smart Slot Filling
- [ ] Infer slot types from cancelled words
- [ ] Use entity profiles for filling
- [ ] Handle multiple slot types

### Phase 3: Template Caching
- [ ] Cache common templates by question type
- [ ] Update cache incrementally
- [ ] Fallback to cached when few examples

### Phase 4: Multi-Sentence
- [ ] Extend to paragraph-level templates
- [ ] Handle transitions between sentences
- [ ] Maintain coherence

## Code Sketch

```python
class HolographicTemplateProjector:
    """Project response templates via holographic interference."""
    
    def __init__(self, knowledge: GeometricKnowledge):
        self.knowledge = knowledge
        self.qa_pairs: List[Tuple[str, str, str]] = []  # (q_type, question, answer)
        self.template_cache: Dict[str, str] = {}
    
    def add_qa_pair(self, question: str, answer: str):
        """Add a Q&A pair for template learning."""
        q_type = self._detect_question_type(question)
        self.qa_pairs.append((q_type, question, answer))
    
    def project_template(self, query: str, k: int = 5) -> str:
        """Project a template for the given query."""
        q_type = self._detect_question_type(query)
        
        # Find similar Q&A pairs
        similar = self._find_similar(q_type, query, k)
        if not similar:
            return self._fallback_template(q_type)
        
        # Extract responses
        responses = [answer for _, _, answer in similar]
        
        # Compute interference
        interference = self._compute_interference(responses)
        
        # Extract template
        template = self._extract_template(interference)
        
        return template
    
    def generate(self, query: str) -> str:
        """Generate a response for the query."""
        # Project template
        template = self.project_template(query)
        
        # Extract entity from query
        entity = self._extract_entity(query)
        
        # Fill template
        response = self._fill_template(template, entity)
        
        return response
    
    def _compute_interference(self, responses: List[str]) -> Dict[int, Tuple[str, complex]]:
        """Compute interference pattern preserving position."""
        # Track (position, word) → complex value
        position_values: Dict[int, Dict[str, complex]] = defaultdict(lambda: defaultdict(complex))
        
        for response in responses:
            words = self._tokenize(response)
            n = len(words)
            
            for i, word in enumerate(words):
                # Normalize position to [0, 1]
                pos = i / max(n - 1, 1)
                
                # Assign to position bucket
                bucket = int(pos * 10)  # 10 buckets
                
                # Compute phase
                phase = self._get_phase(word, pos)
                magnitude = self._get_magnitude(word)
                
                # Add to interference
                position_values[bucket][word] += magnitude * np.exp(1j * phase)
        
        # For each position, find dominant word
        result = {}
        for bucket, word_values in position_values.items():
            # Find word with highest magnitude
            best_word = max(word_values.items(), key=lambda x: abs(x[1]))
            result[bucket] = best_word
        
        return result
    
    def _extract_template(self, interference: Dict[int, Tuple[str, complex]], 
                          threshold: float = 0.5) -> str:
        """Extract template from interference pattern."""
        parts = []
        for bucket in sorted(interference.keys()):
            word, z = interference[bucket]
            magnitude = abs(z) / len(self.qa_pairs)  # Normalize
            
            if magnitude > threshold:
                parts.append(word)
            else:
                # Infer slot type
                slot = self._infer_slot(word)
                if slot not in [p for p in parts if p.startswith('{')]:
                    parts.append(slot)
        
        return " ".join(parts)
    
    def _infer_slot(self, word: str) -> str:
        """Infer slot type from word."""
        # Check if it's a known entity
        if word in self.knowledge.concepts:
            c = self.knowledge.concepts[word]
            if c.initiator_count > c.receiver_count:
                return "{entity}"
            elif c.mediator_count > 0:
                return "{action}"
            else:
                return "{target}"
        
        # Check word properties
        if word.endswith(('ed', 's', 'ing')):
            return "{action}"
        
        return "{content}"
    
    def _get_phase(self, word: str, position: float) -> float:
        """Get phase for word based on type and position."""
        word_lower = word.lower()
        
        # Structure words: phase 0 (always align)
        structure_words = {'is', 'a', 'an', 'the', 'who', 'that', 'which', 'was', 'were'}
        if word_lower in structure_words:
            return 0.0
        
        # Content words: phase based on hash (will cancel)
        return (hash(word_lower) % 1000) / 1000 * 2 * np.pi
    
    def _get_magnitude(self, word: str) -> float:
        """Get magnitude (importance) for word."""
        word_lower = word.lower()
        
        # Structure words: lower magnitude
        structure_words = {'is', 'a', 'an', 'the', 'who', 'that', 'which'}
        if word_lower in structure_words:
            return 0.5
        
        # Content words: higher magnitude
        return 1.0
```

## Connection to Existing System

This integrates with:
- `GeometricKnowledge` for entity profiles
- `GeometricMorphology` for verb form selection
- `GeometricConjugation` for output conjugation

The flow becomes:
```
Query → Project Template → Fill Slots → Conjugate → Response
```

## Next Steps

1. Implement `HolographicTemplateProjector` class
2. Seed with Q&A pairs from corpus
3. Test template emergence
4. Integrate with `GeometricQA`

---

*"Templates are not written, they are discovered."*
