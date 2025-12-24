# TruthSpace LCM Architecture

## Overview

TruthSpace LCM is a **Holographic Concept Language Model** that performs all semantic operations as geometric operations in concept space. No neural networks, no training - just mathematics.

## Core Principles

> **All semantic operations are geometric operations in concept space.**
> **Error = Construction Blueprint.**

- **Concept Frames** = Order-free semantic representations
- **Action Primitives** = Universal verbs (MOVE, SPEAK, THINK, etc.)
- **Holographic Projection** = Questions are gaps; answers fill them
- **φ-Based Navigation** = Golden ratio powers for importance and coherence
- **4D φ-Dial Control** = Style × Perspective × Depth × Certainty (quaternion)
- **Gradient-Free Learning** = Error-driven structure building, no backprop
- **Conversation Memory** = Multi-turn dialogue with pronoun resolution
- **Cross-Language** = Same concepts work across any language

## API Architecture

### OpenAI-Compatible API (`api/server.py`)

FastAPI server providing OpenAI-compatible endpoints:

```python
# Endpoints
POST /v1/chat/completions  # Chat with streaming support
GET  /v1/models            # List available models
GET  /health               # Health check

# Also available without /v1 prefix for Goose compatibility
POST /chat/completions
GET  /models
```

### Orchestrator (`core/orchestrator.py`)

Central routing and request management:

```python
Orchestrator:
  - handlers: list[Handler]     # Registered handlers
  - classifier: IntentClassifier # Intent detection
  - query_resolver: QueryResolver # Compound query handling
  - conversation_context: ConversationContext # Entity tracking
  
  def process(message, system_prompt) -> str:
      # 1. Resolve compound queries
      # 2. Route to best handler
      # 3. Combine responses
      # 4. Update context
```

### Handlers (`core/handlers/`)

Modular request handlers:

| Handler | Intents | Purpose |
|---------|---------|--------|
| KnowledgeHandler | QUESTION | Q&A, reasoning |
| CodeHandler | CODE | Python generation |
| ToolHandler | EXECUTE, CHART | Tools, calculations |
| ChatHandler | GREETING, FAREWELL, META | Conversation, self-knowledge |

### Tool System (`core/tools/`)

Extensible tool registry:

```python
Tool (abstract):
  - name: str              # Tool identifier
  - description: str       # What it does
  - triggers: list[str]    # Activation patterns
  - execute(query) -> ToolResult

ToolRegistry:
  - register(tool)         # Add a tool
  - find_best_tool(query)  # Match query to tool
  - list_tools()           # Get all tools

# Built-in tools:
- TimeTool      # Current time/date
- CalculatorTool # Math operations  
- ChartTool     # Matplotlib charts
```

### Query Resolution (`core/query_resolver.py`)

Compound query splitting and coreference resolution:

```python
QuerySplitter:
  # "Who is Holmes and what time is it?"
  # → ["Who is Holmes?", "What time is it?"]

CoreferenceResolver:
  # "Who is Darcy and how did he meet Elizabeth?"
  # → ["Who is Darcy?", "How did Darcy meet Elizabeth?"]
  # (pronoun "he" resolved to "Darcy")
```

### Self-Knowledge (`core/self_knowledge.py`)

Model identity and meta-information:

```python
SelfKnowledge:
  - identity: dict         # Name, creator, philosophy
  - capabilities: list     # What the model can do
  - knowledge_domains: list # Sherlock Holmes, Pride and Prejudice
  - limitations: list      # What it cannot do
  
  def answer_meta_question(query) -> str
  def get_system_prompt() -> str
  def get_full_introduction() -> str
```

### Conversation Context (`core/conversation_context.py`)

Entity and topic tracking across turns:

```python
ConversationContext:
  - entities: deque[EntityMention]  # Recently mentioned
  - current_topic: TopicState       # Active topic/domain
  
  def get_recent_entities(n) -> list[str]
  def resolve_reference("he") -> "darcy"
  def get_current_domain() -> "Pride and Prejudice"
```

## Primary Components

### ConceptFrame (`concept_language.py`)

Language-agnostic semantic representation.

```python
ConceptFrame:
  - agent: str       # Who performs the action
  - action: str      # Primitive (MOVE, SPEAK, THINK, etc.)
  - patient: str     # Who/what is affected
  - theme: str       # What the action is about
  - location: str    # Where it happens
  - goal: str        # Destination or purpose
  - source: str      # Origin
  - aspect: str      # PERFECTIVE or IMPERFECTIVE
```

### ConceptExtractor (`concept_language.py`)

Extracts concept frames from text in any language.

```python
extractor = ConceptExtractor()
frame = extractor.extract("Darcy walked to the garden.")
# ConceptFrame(agent='darcy', action='MOVE', goal='garden')
```

### ConceptKnowledge (`concept_knowledge.py`)

Language-agnostic knowledge storage and query.

```python
kb = ConceptKnowledge(dim=64)
kb.add_frame(frame, source_text, source)
results = kb.query_by_entity("darcy", k=10)
results = kb.query_by_action("SPEAK", k=10)
```

### HolographicProjector (`concept_knowledge.py`)

Resolves queries using holographic projection.

```python
projector = HolographicProjector(kb)
axis, entity = projector.detect_question_axis("Who is Darcy?")
# axis='WHO', entity='darcy'
answers = projector.resolve("Who is Darcy?", k=3)
```

### ConceptQA (`concept_knowledge.py`)

High-level Q&A interface with 2D φ-dial control.

```python
qa = ConceptQA(style_x=0.0, perspective_y=0.0)
qa.load_corpus('concept_corpus.json')
answer = qa.ask("Who is Darcy?")

# Change style/perspective dynamically
qa.set_dial(x=-1, y=1)  # Formal + Meta
answer = qa.ask("Who is Holmes?")
```

### ComplexPhiDial (`answer_patterns.py`)

2D control mechanism using complex φ-navigation.

```python
dial = ComplexPhiDial(x=-1, y=1)  # Formal + Meta
style = dial.get_style()          # 'formal'
perspective = dial.get_perspective()  # 'meta'
quadrant = dial.get_quadrant_label()  # 'Scholarly/Analytical'
```

### SpatialAttention (`spatial_attention.py`)

φ-based importance scoring for entity relationships.

```python
attention = SpatialAttention()
attention.initialize(frames, known_entities)

# Get important relations with navigation direction
relations = attention.get_important_relations('holmes', k=5, navigation='inward')
# [('watson', 0.19), ('lestrade', 0.12), ...]
```

### LearnableStructure (`learnable_structure.py`)

Gradient-free learning via error-driven structure building.

```python
from truthspace_lcm.core import LearnableStructure, train_from_examples

structure = LearnableStructure()
structure.add_known_entities(['holmes', 'watson', 'darcy', 'elizabeth'])

# Train from examples (entity, source, target_answer)
examples = [
    ('holmes', 'Sherlock Holmes', 'Holmes is a brilliant detective who investigates with Watson.'),
    ('darcy', 'Pride and Prejudice', 'Darcy is a proud gentleman who loves Elizabeth.'),
]
result = train_from_examples(structure, examples)
# {'epochs': 2, 'final_overlap': 0.96}

# Generate using learned structure
answer = structure.generate('holmes', 'Sherlock Holmes')
# "Holmes is a brilliant detective from Sherlock Holmes who investigates with Watson."
```

### ConversationMemory (`conversation_memory.py`)

Multi-turn dialogue with pronoun resolution and context decay.

```python
from truthspace_lcm.core import ConversationMemory

memory = ConversationMemory(max_turns=10)

# Add turns
memory.add_turn("Who is Holmes?", "Holmes is a detective...", entity="holmes")
memory.add_turn("What did he do?", "He investigated...", entity="holmes")

# Resolve pronouns
resolved = memory.resolve_pronouns("What did he do?")
# "What did Holmes do?"

# Get context with φ^(-n) decay
context = memory.get_recent_context(k=3)
# [(turn, weight), ...] - most recent has highest weight
```

### ReasoningEngine (`reasoning_engine.py`)

Multi-hop reasoning through concept graph traversal.

```python
from truthspace_lcm.core import ReasoningEngine

engine = ReasoningEngine(knowledge)

# WHY questions - find causal chains
path = engine.reason("Why did Holmes investigate?")
# path.answer: "Based on the reasoning chain: holmes → scotchman → ..."
# path.steps: [ReasoningStep(entity='holmes', action='INVERSE_EXIST', relation='scotchman')]

# HOW questions - find process chains
path = engine.reason("How did Darcy act?")
# path.answer: "The process involves: darcy → elizabeth → charlotte → bingley"

# Relationship paths - BFS between entities
path = engine.reason("What is the relationship between Darcy and Elizabeth?")
# path.answer: "Darcy is connected to Elizabeth through 1 steps."
```

### HolographicGenerator (`holographic_generator.py`)

Interference-based text generation using complex numbers.

```python
from truthspace_lcm.core import HolographicGenerator

hologen = HolographicGenerator(knowledge)

# Encode text as complex vector (magnitude + phase)
# - Magnitude = importance (IDF-weighted)
# - Phase = category (content/action/modifier/quality)

# Interfere multiple sources
pattern = hologen.interfere(sources)
# Constructive interference = high magnitude = include
# Destructive interference = low magnitude = exclude

# Generate with learned structure
output = hologen.generate("Who is Holmes?", entity="holmes", learnable=learnable)
# "Holmes is a brilliant detective from Sherlock Holmes who investigates with Watson."
```

### CodeGenerator (`code_generator.py`)

Generate Python code from natural language requests.

```python
from truthspace_lcm.core import CodeGenerator

codegen = CodeGenerator()

# Generate from request
code = codegen.generate("Write a function to add two numbers")
# def add(a, b):
#     """Add two numbers"""
#     return a + b

# Teach new functions
codegen.learn('factorial', ['n'], 'return 1 if n <= 1 else n * factorial(n-1)')

# 30+ built-in operations: add, subtract, multiply, reverse, greet, etc.
```

### Planner (`planner.py`)

Task decomposition and sandboxed execution.

```python
from truthspace_lcm.core import Planner

planner = Planner()

# Plan a task
plan = planner.plan("Calculate the sum of squares of [1, 2, 3, 4]")
# Step 1: Define data
# Step 2: Compute squares
# Step 3: Sum the squares
# Step 4: Return result

# Execute in sandbox
result = planner.execute(plan)
# result.final_result = 30
# result.success = True

# Sandbox provides:
# - Safe builtins (math, types, iteration)
# - Safe modules (math, random, statistics, matplotlib)
# - Timeout protection
# - No file/network access
```

## Action Primitives

Universal verbs that map surface forms to concepts:

| Primitive | English | Spanish |
|-----------|---------|---------|
| MOVE | walk, run, go, travel | caminó, corrió, fue |
| SPEAK | say, tell, ask, speak | dijo, habló, preguntó |
| THINK | think, consider, believe | pensó, creyó, consideró |
| PERCEIVE | see, hear, notice | vio, oyó, notó |
| FEEL | feel, love, hate | sintió, amó, odió |
| ACT | do, make, create | hizo, creó |
| EXIST | is, was, be | es, fue, está |
| POSSESS | have, own, hold | tiene, posee |

## Holographic Principle

From holographic stereoscopy:

```
Question = Content - Gap    (has missing information)
Answer   = Content + Fill   (provides missing information)
```

### Question Axes

| Axis | Gap | Fill |
|------|-----|------|
| WHO | Agent unknown | Identify the agent |
| WHAT | Action/patient unknown | Describe what happened |
| WHERE | Location unknown | Provide location |
| WHEN | Time unknown | Provide time |
| WHY | Purpose unknown | Explain reason |
| HOW | Manner unknown | Describe method |

## Data Flow

```
                    ┌─────────────────┐
                    │  Surface Text   │
                    │ (any language)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ConceptExtractor│
                    │ (verb mappings) │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │     CONCEPT FRAME        │
              │ {AGENT, ACTION, PATIENT} │
              │    (order-free)          │
              └────────────┬─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Vector    │
                    │ (64D hash)  │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   ConceptKnowledge     │
              │   (storage & query)    │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  HolographicProjector  │
              │   (fill the gap)       │
              └────────────┬───────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   English   │
                    │   Answer    │
                    └─────────────┘
```

## Formulas

### Word Position
```
pos(word) = hash(word) → ℝ^64 (deterministic unit vector)
```

### Frame Vector
```
vec(frame) = Σ hash(ROLE:value) for each filled slot
             (order-independent, normalized)
```

### Similarity
```
sim(a, b) = cos(θ) = (a·b) / (‖a‖·‖b‖)
```

### φ-Based Weighting
```
weight = φ^(-log(freq))    # Rare entities score higher
φ^(-n) × φ^(+n) = 1        # Conservation law (self-dual)
```

### 4D Quaternion φ-Dial
```
q = w + xi + yj + zk

Where:
  x = Style dial (-1 to +1): formal ↔ casual
  y = Perspective dial (-1 to +1): subjective ↔ meta
  z = Depth dial (-1 to +1): terse ↔ elaborate
  w = Certainty dial (-1 to +1): definitive ↔ hedged
  
  x (i-axis) = WHAT words we choose
  y (j-axis) = HOW we frame the content
  z (k-axis) = HOW MUCH detail we include
  w (scalar) = HOW SURE we are
```

### Query Resolution
```
1. Detect axis (WHO/WHAT/WHERE)
2. Determine navigation direction (inward/outward)
3. Extract entity from question
4. Query frames by entity with φ-weighted importance
5. Aggregate knowledge (action counts, relationships)
6. Apply 4D φ-dial (style × perspective × depth × certainty)
7. Project to English (fill the gap)
```

## Directory Structure

```
truthspace-lcm/
├── truthspace_lcm/              # Main package
│   ├── __init__.py              # Package exports (v0.7.0)
│   ├── chat.py                  # Holographic Q&A chat with φ-dial
│   ├── concept_corpus.json      # Knowledge corpus (11,214 frames)
│   ├── training_data.py         # Pre-defined training examples
│   ├── core/
│   │   ├── __init__.py          # Core exports
│   │   ├── vocabulary.py        # Word positions, IDF, encoding
│   │   ├── concept_language.py  # ConceptFrame, ConceptExtractor
│   │   ├── concept_knowledge.py # ConceptKnowledge, HolographicProjector
│   │   ├── answer_patterns.py   # QuaternionPhiDial, PatternAnswerGenerator
│   │   ├── spatial_attention.py # φ-based navigation, importance scoring
│   │   ├── learnable_structure.py # Gradient-free learning, EntityProfile
│   │   ├── conversation_memory.py # Multi-turn dialogue, pronoun resolution
│   │   ├── reasoning_engine.py    # Multi-hop reasoning, graph traversal
│   │   ├── holographic_generator.py # Interference-based generation
│   │   ├── code_generator.py      # Python code generation
│   │   └── planner.py             # Task planning and sandboxed execution
│   └── utils/
│       └── extractors.py        # Shared extraction utilities
├── tests/
│   ├── test_core.py             # Core tests (25)
│   └── test_chat.py             # Chat tests (12)
├── design_considerations/       # Research journey (50+ documents)
│   ├── 050_geometric_llm_roadmap.md   # Roadmap for LLM-level capability
│   ├── 049_gradient_free_learning.md  # Error-driven structure learning
│   ├── 048_clock_geodesic_unification.md  # Clock + geodesic connection
│   ├── 047_geodesic_generation.md     # Generation as concept space navigation
│   ├── 044_quaternion_phi_dial.md     # 4D quaternion φ-dial
│   ├── 043_3d_phi_dial_depth.md       # 3D φ-dial with depth
│   └── ...                            # Earlier design documents
├── scripts/                     # Utility scripts
├── run.py                       # Entry point
└── requirements.txt             # Dependencies (numpy)
```

## Validation Results

### Extraction Accuracy
| Language | Previous | Concept Language | Improvement |
|----------|----------|------------------|-------------|
| English | 0.524 | **0.852** | 1.6x |
| Spanish | 0.058 | **0.806** | **14x** |

### Cross-Language Queries
Query `{ACTION: SPEAK}` returns both:
- English: "Bingley," cried his wife..."
- Spanish: "También lo juro yo —dijo el labrador..."

### Corpus Statistics
- **11,214** concept frames
- **1,759** unique entities
- **3,029** entity relations
- **14** literary works (English and Spanish)

## Design Decisions

### Why Concept Frames (not triples)?
- **Order-free** - No SVO/VSO dependency
- **Richer** - Multiple slots (agent, patient, location, etc.)
- **Universal** - Same structure for any language

### Why Action Primitives?
- **Cross-language** - "walked" and "caminó" both → MOVE
- **Semantic** - Captures meaning, not surface form
- **Finite** - ~7 primitives cover most verbs

### Why Holographic Projection?
- **Gap-filling** - Natural model for Q&A
- **Geometric** - Pure vector operations
- **Interpretable** - Axis defines what's being asked

## The 4D Quaternion φ-Dial

The 4D φ-dial provides complete control over answer generation:

| Axis | Name | Range | Controls |
|------|------|-------|----------|
| **X** | Style | -1 to +1 | WHAT words (formal ↔ casual) |
| **Y** | Perspective | -1 to +1 | HOW framed (subjective ↔ meta) |
| **Z** | Depth | -1 to +1 | HOW MUCH detail (terse ↔ elaborate) |
| **W** | Certainty | -1 to +1 | HOW SURE (definitive ↔ hedged) |

### W-Axis: Certainty/Modality

| w | Certainty | Copula | Relationship |
|---|-----------|--------|--------------|
| -1 | Definitive | "is undoubtedly" | "closely tied to" |
| 0 | Neutral | "is" | "associated with" |
| +1 | Hedged | "appears to be" | "possibly connected to" |

### X,Y Plane: Style × Perspective

| Quadrant | x | y | Style | Perspective | Label |
|----------|---|---|-------|-------------|-------|
| Q1 | +1 | -1 | Casual | Subjective | Conversational |
| Q2 | +1 | +1 | Casual | Meta | Pop Culture |
| Q3 | -1 | -1 | Formal | Subjective | Literary |
| Q4 | -1 | +1 | Formal | Meta | Scholarly |

## φ-Navigation Modes

The φ-structure supports dual navigation:

| Mode | Formula | Use Case |
|------|---------|----------|
| INWARD | φ^(-n) | WHO/WHERE questions (specific entities) |
| OUTWARD | φ^(+n) | WHAT/HOW questions (universal patterns) |
| BALANCED | 1.0 | Similarity queries |
| OSCILLATING | alternate | WHY questions (causal chains) |

Key property: `φ^(-n) × φ^(+n) = 1` (conservation law)

## Gradient-Free Learning

The system learns without gradients or backpropagation:

```
Traditional ML: error = how wrong we are → adjust weights
Geometric LCM:  error = what's missing → add structure
```

### Learning Algorithm

```python
def train(structure, examples):
    for entity, source, target in examples:
        generated = structure.generate(entity, source)
        missing = target_words - generated_words
        
        for word in missing:
            category = classify(word)  # role, quality, action, relation
            structure.add(entity, category, word)
```

### Results

| Metric | Value |
|--------|-------|
| Starting overlap | ~50% |
| Final overlap | ~96% |
| Epochs needed | 2 |
| Gradients used | 0 |

### Comparison to Neural Networks

| Aspect | Neural Network | Geometric LCM |
|--------|----------------|---------------|
| Parameters | Weights (continuous) | Mappings (discrete) |
| Learning | Gradient descent | Error-driven addition |
| Convergence | Thousands of epochs | 1-2 epochs |
| Interpretability | Black box | Fully transparent |
| Memory | Forgets old data | Accumulates knowledge |
| Updates | Requires retraining | Incremental |

## Future Work

1. ~~**Multi-Hop Reasoning** - Chain multiple reasoning steps for WHY/HOW questions~~ ✓ DONE
2. ~~**Holographic Generation** - Replace templates with interference patterns~~ ✓ DONE
3. ~~**Code Generation** - Generate Python from natural language~~ ✓ DONE
4. ~~**Planning & Execution** - Decompose tasks, execute in sandbox~~ ✓ DONE
5. ~~**OpenAI-Compatible API** - REST API with streaming support~~ ✓ DONE
6. ~~**Tool System** - Extensible tools with plugin registry~~ ✓ DONE
7. ~~**Compound Queries** - Multi-part questions with coreference~~ ✓ DONE
8. ~~**Self-Knowledge** - Model identity and capabilities~~ ✓ DONE
9. **More Languages** - Add French, German, Chinese verb mappings
10. **Temporal Reasoning** - Track when events happened
11. **Causal Chains** - Improve oscillating navigation for WHY
12. **Scale Testing** - Benchmark against LLMs on larger corpora
13. **More Tools** - Web search, file operations, external APIs
