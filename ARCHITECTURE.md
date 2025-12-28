# TruthSpace LCM Architecture

## Overview

TruthSpace LCM is a **Geometric Language Model** that performs all semantic operations as geometric operations in concept space. No neural networks, no training, no hard-coded linguistic rules - just geometry.

**Version**: 1.0.0

## Core Principles

> **All semantic operations are geometric operations in concept space.**
> **ENCODE = DECODE - they are the same operation in opposite directions, like φ and 1/φ.**
> **Two quaternions: one for meaning, one for expression.**

### The Unified System

| Component | Purpose | Key Innovation |
|-----------|---------|----------------|
| **GeometricKnowledge** | Frame extraction | Position-based roles, no parsing |
| **HolographicTemplates** | Response generation | Interference patterns |
| **SemanticQuaternion** | Concept encoding | 100% analogy accuracy |
| **φ-Dial** | Output styling | 4D quaternion control |

---

## Architecture Diagram

```
INPUT: "Who is Holmes?"
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GeometricKnowledge                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Position-Based Frame Extraction                          │   │
│  │   [0, 0.33) → INITIATOR (subject)                       │   │
│  │   [0.33, 0.66) → MEDIATOR (verb)                        │   │
│  │   [0.66, 1.0] → RECEIVER (object)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐   │
│  │ GeometricConcept│  │GeometricMorpho- │  │ GeometricConj-│   │
│  │ - φ-direction   │  │logy             │  │ ugation       │   │
│  │ - role counts   │  │ - equivalence   │  │ - verb forms  │   │
│  │ - mean position │  │ - 109 clusters  │  │ - phase-based │   │
│  └─────────────────┘  └─────────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────┬──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Holographic  │    │    Semantic     │    │     φ-Dial      │
│   Templates   │    │   Quaternion    │    │  (Output Style) │
│               │    │                 │    │                 │
│ Project from  │    │ q = w+xi+yj+zk  │    │ q = w+xi+yj+zk  │
│ Q&A pairs via │    │                 │    │                 │
│ interference  │    │ x: Gender       │    │ x: Style        │
│               │    │ y: Age          │    │ y: Perspective  │
│ Slots emerge  │    │ z: Agency (φ!)  │    │ z: Depth        │
│ from content  │    │ w: Animacy      │    │ w: Certainty    │
│ word phases   │    │                 │    │                 │
│               │    │ 100% analogy    │    │ Controls output │
└───────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HolographicGeometricQA                        │
│                                                                 │
│  ask(query) → Holographic template + slot filling               │
│  complete_analogy(a, b, c) → Quaternion arithmetic              │
│  semantic_similarity(w1, w2) → Quaternion cosine                │
│  find_similar_relations(a, b) → Same rotation pairs             │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: "Holmes is a notable detective who examines, deduces, and observes"
```

---

## Core Components

### GeometricKnowledge (`core/geometric.py`)

Position-based frame extraction and storage.

```python
class GeometricKnowledge:
    concepts: Dict[str, GeometricConcept]  # Word statistics
    frames: List[Frame]                     # Extracted frames
    morphology: GeometricMorphology         # Verb equivalence
    conjugation: GeometricConjugation       # Verb forms
    
    def learn(text: str, source: str):
        """Extract frame from sentence using position bands."""
        # 1. Tokenize sentence
        # 2. Record position statistics for each word
        # 3. Extract frame:
        #    [0, 0.33) → Initiator
        #    [0.33, 0.66) → Mediator
        #    [0.66, 1] → Receiver
        # 4. Update role counts and φ-direction
```

### GeometricConcept

Word with geometric properties.

```python
class GeometricConcept:
    word: str
    positions: List[float]      # All positions [0, 1]
    initiator_count: int        # Times as subject
    mediator_count: int         # Times as verb
    receiver_count: int         # Times as object
    
    @property
    def phi_direction(self) -> float:
        """(initiator - receiver) / total. Range [-1, 1]."""
        # Positive = initiator (subject)
        # Negative = receiver (object)
        # Near zero = mediator (verb)
    
    @property
    def is_geometric_stop_word(self) -> bool:
        """No semantic role OR short+frequent."""
        # Emerges from data, not hard-coded
```

### GeometricMorphology

Verb equivalence learned from parallel structures.

```python
class GeometricMorphology:
    equivalence_classes: Dict[str, Set[str]]  # canonical → variants
    
    def bootstrap(text: str):
        """Learn from parallel structures like 'I love. He loves. I loved.'"""
        # Process sentences in groups of 3
        # → love ≡ loves ≡ loved
    
    def are_equivalent(word1: str, word2: str) -> bool
    def get_canonical(word: str) -> str
```

### GeometricConjugation

Output generation using learned verb forms.

```python
class GeometricConjugation:
    clusters: Dict[str, VerbCluster]  # canonical → forms
    
    def conjugate(word: str, phase: int) -> str:
        """
        Phase 0 = base (love)
        Phase 1 = 3rd singular (loves)
        Phase 2 = past (loved)
        """
```

---

## Holographic Templates (`core/holographic_templates.py`)

### HolographicTemplateProjector

Dynamic templates via interference patterns.

```python
class HolographicTemplateProjector:
    """
    GEOMETRIC ENCODING (not hash-based):
    - Phase = φ-direction × π (semantic role)
    - Magnitude = role_strength (how strongly typed)
    - Structure words: phase = 0 (always align)
    - Content words: phase from geometric knowledge
    """
    
    def project_template(query: str) -> ProjectedTemplate:
        """
        1. Find similar Q&A pairs
        2. Compute interference on responses
        3. Structure words reinforce → keep literal
        4. Content words cancel → become slots
        """
    
    def generate(query: str, slot_values: Dict) -> str:
        """Fill template slots with query-specific content."""
```

### HolographicResponseSynthesizer

Multi-source response synthesis.

```python
class HolographicResponseSynthesizer:
    def synthesize(query: str, sources: List[str]) -> str:
        """
        Combine multiple responses via interference.
        Common words reinforce, unique words cancel.
        """
```

---

## Semantic Quaternions (`core/semantic_quaternion.py`)

### SemanticQuaternion

4D encoding for concepts.

```python
class SemanticQuaternion:
    x: float  # Gender/Polarity (-1 male, +1 female)
    y: float  # Age/Maturity (-1 adult, +1 young)
    z: float  # Agency (-1 receiver, +1 initiator) ← FROM φ-DIRECTION!
    w: float  # Animacy (-1 place, +1 human)
    
    def __add__(self, other): ...  # Quaternion addition
    def __sub__(self, other): ...  # Quaternion subtraction
    def dot(self, other) -> float: ...  # Cosine similarity
```

### SemanticQuaternionNavigator

Analogy completion with 100% accuracy.

```python
class SemanticQuaternionNavigator:
    concepts: Dict[str, SemanticQuaternion]
    
    def complete_analogy(a: str, b: str, c: str, k: int = 5):
        """
        A : B :: C : ?
        
        ? = C + (B - A)  # Quaternion arithmetic
        
        Example: king:queen::man:? → woman
        """
    
    def similarity(w1: str, w2: str) -> float:
        """Quaternion cosine similarity."""
    
    def find_similar_relations(a: str, b: str, k: int = 5):
        """Find pairs with same rotation as A→B."""
```

### SemanticFeatureLearner

Learn x,y axes from parallel structures.

```python
class SemanticFeatureLearner:
    def learn_from_parallel(sentences: List[str]):
        """
        "The king rules" + "The queen rules" → king/queen differ in x
        "The man works" + "The boy plays" → man/boy differ in y
        """
```

---

## HolographicGeometricQA (`core/geometric.py`)

The unified Q&A system combining all components.

```python
class HolographicGeometricQA:
    knowledge: GeometricKnowledge
    template_projector: HolographicTemplateProjector
    semantic_navigator: SemanticQuaternionNavigator
    
    def ask(query: str) -> str:
        """Answer using holographic templates."""
    
    def ask_detailed(query: str) -> Dict:
        """Answer with full analysis."""
    
    def complete_analogy(a: str, b: str, c: str, k: int = 5):
        """A:B::C:? using semantic quaternions."""
    
    def semantic_similarity(w1: str, w2: str) -> float:
        """Quaternion cosine similarity."""
    
    def find_similar_relations(a: str, b: str, k: int = 5):
        """Find pairs with same rotation."""
    
    def add_semantic_concept(word: str, x: float, y: float, w: float):
        """Add concept with semantic features (z from φ-direction)."""
```

---

## Supporting Components

### ConversationMemory (`core/conversation_memory.py`)

Multi-turn dialogue with pronoun resolution.

```python
class ConversationMemory:
    turns: List[ConversationTurn]
    max_turns: int = 10
    
    def add_turn(query: str, response: str, entities: List[str])
    def resolve_pronouns(query: str) -> str
    def get_context() -> str
```

### ReasoningEngine (`core/reasoning_engine.py`)

Multi-hop reasoning for WHY/HOW questions.

```python
class ReasoningEngine:
    def reason(query: str) -> ReasoningPath:
        """Graph traversal for causal chains."""
```

### CodeGenerator (`core/code_generator.py`)

Python code from natural language.

```python
class CodeGenerator:
    def generate(request: str) -> CodeFrame:
        """Generate Python code from NL request."""
```

### Planner (`core/planner.py`)

Task decomposition and execution.

```python
class Planner:
    def plan(task: str) -> ExecutionPlan
    def execute(plan: ExecutionPlan) -> str
```

---

## API Architecture

### OpenAI-Compatible API (`api/server.py`)

```python
# Endpoints
POST /v1/chat/completions  # Chat with streaming
GET  /v1/models            # List models
GET  /health               # Health check
```

Uses `HolographicGeometricQA` for all requests.

---

## Key Formulas

### Geometric Encoding

```
Position: p(w) = normalized position in sentence [0, 1]
φ-direction: (initiator_count - receiver_count) / total_roles
Phase: φ-direction × π (geometric, not hash)
Magnitude: role_strength (how strongly typed)
```

### Semantic Quaternion Analogy

```
Given: A : B :: C : ?

Compute rotation: R = B - A
Apply to C: ? = C + R

Example:
  king = (x=-1, y=-1, z=1, w=1)   # male, adult, initiator, human
  queen = (x=1, y=-1, z=1, w=1)   # female, adult, initiator, human
  R = queen - king = (2, 0, 0, 0) # gender flip
  
  man = (x=-1, y=-1, z=0, w=1)    # male, adult, neutral, human
  ? = man + R = (1, -1, 0, 1)     # female, adult, neutral, human
  → woman ✓
```

### Holographic Interference

```
Template projection:
  1. Encode each word: z = magnitude × e^(i·phase)
  2. Sum across responses: Σ z / N
  3. High magnitude → keep (structure word)
  4. Low magnitude → slot (content word)
```

---

## File Structure

```
truthspace_lcm/
├── __init__.py                   # Package exports (v1.0.0)
├── chat.py                       # Interactive chat
├── core/
│   ├── __init__.py               # Core exports
│   ├── geometric.py              # GeometricQA, HolographicGeometricQA
│   ├── holographic_templates.py  # Template projection
│   ├── semantic_quaternion.py    # 4D quaternion encoding
│   ├── conversation_memory.py    # Multi-turn dialogue
│   ├── reasoning_engine.py       # Multi-hop reasoning
│   ├── code_generator.py         # Python generation
│   └── planner.py                # Task planning

api/
├── server.py                     # FastAPI server
└── models.py                     # Pydantic models

run.py                            # Main entry point
run_api.py                        # API server entry
ROADMAP.md                        # Development roadmap
```

---

## The Vision

**"Two quaternions: one for meaning, one for expression. Together they span the space of language."**

- **Semantic Quaternion**: Encodes WHAT concepts mean
- **φ-Dial Quaternion**: Controls HOW we express responses

The key insight: **ENCODE = DECODE**. They are the same operation in opposite directions, like φ and 1/φ.

---

*"Structure is the new training. Geometry is the new statistics."*
