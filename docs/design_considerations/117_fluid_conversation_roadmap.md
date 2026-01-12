# Design Consideration 117: Roadmap to Fluid Conversations

## The Challenge

Our current experimental corpus can answer transformation queries:
- "What's the female form of king?" → "queen" ✓
- "What's a younger version of dog?" → "puppy" ✓

But it can't handle natural conversation:
- "How are you?" → ??? 
- "Tell me about chess" → ???
- "What do you think about that?" → ???

## The Gap Analysis

### What We Have

1. **Self-Assembling Corpus** (Phases 1-6) ✓
   - Transformation pairs as source of truth
   - Emergent dimensions
   - φ-based geometry
   - Platonic Ideals
   - Self-assembly loop
   - Persistence

2. **Attachable Layers** ✓
   - Base layer (language fundamentals)
   - Domain layers (attachable/detachable)
   - Context layer (conversation state)
   - Geometric RAG

3. **Chat Interface** ✓
   - Query parsing
   - Intent detection
   - Response generation (basic)

### What We're Missing

1. **Conversational Primitives**
   - Greetings, farewells
   - Acknowledgments, confirmations
   - Questions, statements
   - Emotional expressions

2. **Discourse Structure**
   - Turn-taking patterns
   - Topic continuity
   - Reference resolution (pronouns)
   - Coherence across turns

3. **Response Generation**
   - Currently: retrieve concepts → format as text
   - Needed: traverse geometry → construct meaningful response

4. **Scale**
   - Current: ~30 pairs
   - Needed: ~1000+ pairs for basic fluency

## The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CONVERSATION SYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DISCOURSE LAYER                                     │    │
│  │  - Turn structure (greeting → body → closing)        │    │
│  │  - Topic tracking                                    │    │
│  │  - Reference resolution                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  INTENT LAYER                                        │    │
│  │  - Query type (question, statement, command)         │    │
│  │  - Speech act (inform, request, confirm)             │    │
│  │  - Emotional tone                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CORPUS STACK                                        │    │
│  │  - Context layer (this conversation)                 │    │
│  │  - Domain layer (current topic)                      │    │
│  │  - Base layer (language fundamentals)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  RESPONSE GENERATOR                                  │    │
│  │  - Geometric traversal                               │    │
│  │  - Template selection                                │    │
│  │  - Surface realization                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Expand Base Corpus

### Conversational Dimensions

Add dimensions for conversational structure:

```python
# Speech act dimension
corpus.add_pair("statement", "question", "speech_act")
corpus.add_pair("question", "command", "speech_act")
corpus.add_pair("command", "request", "speech_act")

# Formality dimension (already have)
corpus.add_pair("hello", "hi", "formality")
corpus.add_pair("goodbye", "bye", "formality")

# Politeness dimension
corpus.add_pair("please", "gimme", "politeness")
corpus.add_pair("thank_you", "thanks", "politeness")
corpus.add_pair("would_you", "do_it", "politeness")

# Certainty dimension
corpus.add_pair("definitely", "maybe", "certainty")
corpus.add_pair("always", "sometimes", "certainty")
corpus.add_pair("know", "think", "certainty")

# Temporal dimension
corpus.add_pair("past", "present", "temporal")
corpus.add_pair("present", "future", "temporal")
corpus.add_pair("was", "is", "temporal")
corpus.add_pair("is", "will_be", "temporal")

# Quantity dimension
corpus.add_pair("all", "some", "quantity")
corpus.add_pair("many", "few", "quantity")
corpus.add_pair("always", "never", "quantity")
```

### Conversational Patterns

Add transformation pairs for common patterns:

```python
# Greetings
corpus.add_pair("greeting", "hello", "instantiation")
corpus.add_pair("greeting", "hi", "instantiation")
corpus.add_pair("greeting", "hey", "instantiation")

# Farewells
corpus.add_pair("farewell", "goodbye", "instantiation")
corpus.add_pair("farewell", "bye", "instantiation")
corpus.add_pair("farewell", "see_you", "instantiation")

# Acknowledgments
corpus.add_pair("acknowledgment", "yes", "instantiation")
corpus.add_pair("acknowledgment", "okay", "instantiation")
corpus.add_pair("acknowledgment", "sure", "instantiation")

# Questions
corpus.add_pair("question", "what", "question_type")
corpus.add_pair("question", "who", "question_type")
corpus.add_pair("question", "where", "question_type")
corpus.add_pair("question", "when", "question_type")
corpus.add_pair("question", "why", "question_type")
corpus.add_pair("question", "how", "question_type")
```

### Scale Target

| Category | Current | Target | Notes |
|----------|---------|--------|-------|
| Gender pairs | 7 | 20 | More roles, professions |
| Age pairs | 6 | 20 | Life stages, maturity |
| Size pairs | 5 | 15 | Physical, abstract |
| Formality pairs | 4 | 30 | Vocabulary register |
| Sentiment pairs | 4 | 30 | Emotions, attitudes |
| Speech act pairs | 0 | 20 | Questions, statements |
| Temporal pairs | 0 | 15 | Tense, duration |
| Quantity pairs | 0 | 15 | Amount, frequency |
| **Total** | **~30** | **~165** | 5x increase |

## Phase 2: Discourse Structure

### Turn Types

```python
class TurnType(Enum):
    GREETING = "greeting"
    QUESTION = "question"
    ANSWER = "answer"
    STATEMENT = "statement"
    ACKNOWLEDGMENT = "acknowledgment"
    CLARIFICATION = "clarification"
    FAREWELL = "farewell"
```

### Conversation State Machine

```
GREETING → BODY → CLOSING

BODY can be:
  QUESTION → ANSWER → (FOLLOWUP | TOPIC_CHANGE | CLOSING)
  STATEMENT → ACKNOWLEDGMENT → (ELABORATION | TOPIC_CHANGE | CLOSING)
  COMMAND → RESPONSE → (FOLLOWUP | CLOSING)
```

### Implementation

```python
class DiscourseTracker:
    """Track conversation structure."""
    
    def __init__(self):
        self.state = "greeting"
        self.topic_stack = []
        self.turn_history = []
    
    def process_turn(self, utterance: str) -> TurnType:
        """Classify turn and update state."""
        turn_type = self._classify(utterance)
        self._update_state(turn_type)
        self.turn_history.append((utterance, turn_type))
        return turn_type
    
    def expected_response_type(self) -> TurnType:
        """What type of response is expected?"""
        if self.state == "greeting":
            return TurnType.GREETING
        if self.turn_history[-1][1] == TurnType.QUESTION:
            return TurnType.ANSWER
        # ... etc
```

## Phase 3: Response Generation

### Current Approach (Limited)

```python
# Find concepts → format as text
concepts = find_nearest(query_position)
response = f"Related concepts: {concepts}"
```

### Improved Approach: Template + Traversal

```python
class ResponseGenerator:
    """Generate responses by traversing geometry."""
    
    def __init__(self, stack: CorpusStack):
        self.stack = stack
        self.templates = self._load_templates()
    
    def generate(self, query: str, turn_type: TurnType) -> str:
        """Generate response based on turn type and geometry."""
        
        if turn_type == TurnType.GREETING:
            return self._respond_greeting(query)
        
        if turn_type == TurnType.QUESTION:
            return self._respond_question(query)
        
        if turn_type == TurnType.STATEMENT:
            return self._respond_statement(query)
        
        # ...
    
    def _respond_question(self, query: str) -> str:
        """Answer a question by geometric traversal."""
        # Parse question type
        q_type = self._parse_question_type(query)  # what, who, where, etc.
        
        # Extract focus concept
        focus = self._extract_focus(query)
        
        # Traverse geometry based on question type
        if q_type == "what":
            # Find definition (traverse to Platonic Ideal)
            ideal = self.stack.base.corpus.get_ideal(focus)
            if ideal:
                return f"{focus} is a concept that anchors {ideal.dimensions_anchored}"
        
        if q_type == "who":
            # Find person-like concepts near focus
            results = self.stack.query(focus)
            persons = [r for r in results if self._is_person(r.concept)]
            return f"Related people: {[p.concept for p in persons]}"
        
        # ... etc
```

### Template System

```python
TEMPLATES = {
    "greeting_response": [
        "Hello! How can I help you?",
        "Hi there! What would you like to know?",
        "Hey! What's on your mind?",
    ],
    "definition": [
        "{concept} is {definition}.",
        "A {concept} is {definition}.",
        "{concept} can be described as {definition}.",
    ],
    "transformation": [
        "The {dimension} of {source} is {target}.",
        "{source} becomes {target} along the {dimension} dimension.",
        "If you apply {dimension} to {source}, you get {target}.",
    ],
    "unknown": [
        "I don't have information about {concept} yet.",
        "I'm not familiar with {concept}. Can you tell me more?",
        "{concept} isn't in my knowledge base.",
    ],
}
```

## Phase 4: LLM-Assisted Expansion

### The Bootstrap Problem

We need a large corpus to have fluid conversations, but building it manually is slow.

### Solution: LLM as Corpus Builder

Use the LLM not for response generation, but for corpus expansion:

```python
class CorpusExpander:
    """Use LLM to expand the corpus."""
    
    def __init__(self, corpus: SelfAssemblingCorpus, llm: LLMInterface):
        self.corpus = corpus
        self.llm = llm
    
    def expand_dimension(self, dimension: str, n: int = 10):
        """Ask LLM for more pairs in a dimension."""
        prompt = f"""
        I have these transformation pairs for the '{dimension}' dimension:
        {self._get_existing_pairs(dimension)}
        
        Give me {n} more pairs that follow the same pattern.
        Format: source → target
        """
        response = self.llm.query(prompt)
        pairs = self._parse_pairs(response)
        for source, target in pairs:
            self.corpus.add_pair(source, target, dimension)
    
    def discover_dimension(self, concept: str):
        """Ask LLM what dimensions a concept participates in."""
        prompt = f"""
        The concept '{concept}' can be transformed along various dimensions.
        What are some transformations? Examples:
        - king → queen (gender)
        - boy → man (age)
        
        Give transformations for '{concept}':
        """
        response = self.llm.query(prompt)
        # Parse and add to corpus
```

### The Key Insight

**The LLM builds the corpus. The corpus generates responses.**

This keeps the geometric principle intact:
- LLM is used offline to expand knowledge
- Responses come from geometric traversal, not LLM generation
- The corpus IS the knowledge, the LLM is just a tool to build it

## Implementation Roadmap

### Week 1: Expand Base Corpus
- [ ] Add conversational dimensions (speech act, politeness, certainty, temporal)
- [ ] Add conversational patterns (greetings, farewells, acknowledgments)
- [ ] Scale to ~150 pairs

### Week 2: Discourse Structure
- [ ] Implement TurnType classification
- [ ] Implement DiscourseTracker
- [ ] Add conversation state machine

### Week 3: Response Generation
- [ ] Implement template system
- [ ] Add geometric traversal for questions
- [ ] Add response type selection

### Week 4: LLM-Assisted Expansion
- [ ] Implement CorpusExpander
- [ ] Expand each dimension to 20+ pairs
- [ ] Scale to ~500 pairs

### Week 5: Integration & Testing
- [ ] Integrate with chat interface
- [ ] Add domain attachment UI
- [ ] Test fluid conversation flow

## Success Criteria

A "fluid conversation" means:

1. **Natural greetings**: "Hi!" → "Hello! How can I help?"
2. **Topic handling**: "Tell me about chess" → [attach chess domain] → relevant response
3. **Follow-ups**: "What about the queen?" → uses context to understand "chess queen"
4. **Graceful unknowns**: "What's a flibbertigibbet?" → "I don't know that yet"
5. **Coherent multi-turn**: Maintains topic across 5+ turns

## Connection to Hypothesis

This approach validates the geometric LCM hypothesis:

- **Structure IS information**: The corpus geometry encodes knowledge
- **Geometry IS computation**: Traversal produces responses
- **The shape IS the knowledge**: What we "know" is the corpus shape

The LLM is a tool for building structure, not for generating responses.
The responses emerge from the geometry.

---

*"The corpus is the instrument. The LLM tunes it. The geometry plays the music."*
