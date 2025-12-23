# TruthSpace LCM

**Holographic Concept Language Model** - A conversational AI using holographic concept resolution. Gradient-free learning, no neural networks - just geometry.

## Philosophy

> *"All semantic operations are geometric operations in concept space."*
> *"Error = Construction Blueprint"*

This system demonstrates that **pure geometry can replace trained neural networks** for language understanding. Knowledge is stored as **concept frames** - language-agnostic semantic representations that can be queried across languages.

## Features

- **Concept Language** - Order-free semantic frames (like Chinese: no conjugation, flexible order)
- **Holographic Q&A** - Questions are gaps; answers fill them via geometric projection
- **4D φ-Dial Control** - Style × Perspective × Depth × Certainty (quaternion)
- **Gradient-Free Learning** - Error-driven structure building, no backprop needed
- **Conversation Memory** - Multi-turn dialogue with pronoun resolution
- **Multi-Hop Reasoning** - Graph traversal for WHY/HOW questions
- **Holographic Generation** - Interference-based text generation
- **Code Generation** - Generate Python from natural language
- **Planning & Execution** - Decompose tasks, execute in sandbox
- **Chart Generation** - Create matplotlib visualizations
- **φ-Based Navigation** - Golden ratio powers for entity importance and coherence
- **Cross-Language** - Same concepts work across English, Spanish, and more
- **Spatial Attention** - Zipf/φ-based weighting for meaningful relationships

## Installation

```bash
git clone https://github.com/lostdemeter/truthspace_lcm.git
cd truthspace-lcm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Interactive Chat

```bash
python run.py
```

```
============================================================
  TruthSpace LCM - Holographic Concept Q&A
============================================================

Loading corpus from truthspace_lcm/concept_corpus.json...
Loaded 11214 concept frames
Training GeometricLCM with quality examples...
  Epochs: 2
  Final overlap: 96.4%
  Learned: 14 entities, 14 roles, 10 qualities, 12 actions

You: Who is Holmes?

Bot: Holmes is a brilliant detective from Sherlock Holmes 
     who investigates with Watson.

You: What did he do?

Bot: Holmes investigated crimes and solved mysteries.
     (pronoun "he" resolved to Holmes via conversation memory)

You: Who is Darcy?

Bot: Darcy is a proud gentleman from Pride and Prejudice 
     who loves Elizabeth.

You: /memory

Conversation (3 turns):
  Q: Who is Holmes? [holmes]
  Q: What did he do? [holmes]
  Q: Who is Darcy? [darcy]
Focus: darcy
```

### Command Line Options

```bash
python run.py                              # Default (all neutral)
python run.py --style -1 --perspective 1   # Formal + Meta
python run.py -x 1 -y -1                   # Casual + Subjective
python run.py --depth -1                   # Terse mode
python run.py --certainty -1               # Definitive mode
python run.py -w 1                         # Hedged mode
```

### Run Tests

```bash
python run.py test
```

### Python API

```python
from truthspace_lcm import ConceptQA

# Create Q&A system with 4D φ-dial control
qa = ConceptQA(style_x=0.0, perspective_y=0.0, depth_z=0.0, certainty_w=0.0)
qa.load_corpus('truthspace_lcm/concept_corpus.json')

# Ask questions (holographic resolution)
answer = qa.ask("Who is Holmes?")
# "Holmes is a character from Sherlock Holmes who spoke..."

# Change dial dynamically
qa.set_dial(x=-1, y=1, z=1, w=-1)  # Formal + Meta + Elaborate + Definitive
answer = qa.ask("Who is Holmes?")
# "Thematically, Holmes is undoubtedly an archetypal figure..."

# Or set individually
qa.set_style(-1)        # Formal
qa.set_perspective(1)   # Meta
qa.set_depth(-1)        # Terse
qa.set_certainty(1)     # Hedged

# Detailed response with concept frame
result = qa.ask_detailed("What did Holmes do?")
# {'axis': 'WHAT', 'entity': 'holmes', 'answers': [...]}

# Ingest new text
qa.ingest_text("The detective examined the clues.", source="Mystery")

# Query by entity
frames = qa.knowledge.query_by_entity("darcy", k=5)

# Query by action primitive
frames = qa.knowledge.query_by_action("SPEAK", k=10)
```

### The 4D Quaternion φ-Dial

Control **style**, **perspective**, **depth**, and **certainty** using the 4D φ-dial:

| Axis | Name | Range | Controls |
|------|------|-------|----------|
| **X** | Style | -1 to +1 | WHAT words (formal ↔ casual) |
| **Y** | Perspective | -1 to +1 | HOW framed (subjective ↔ meta) |
| **Z** | Depth | -1 to +1 | HOW MUCH detail (terse ↔ elaborate) |
| **W** | Certainty | -1 to +1 | HOW SURE (definitive ↔ hedged) |

**Certainty Examples (w-axis):**

| w | Certainty | Example Output |
|---|-----------|----------------|
| -1 | Definitive | "Certainly, Holmes is undoubtedly a character... closely tied to Watson." |
| 0 | Neutral | "Holmes is a character... associated with Watson." |
| +1 | Hedged | "Perhaps Holmes appears to be a character... possibly connected to Watson." |

**Style × Perspective (x,y plane):**

| Quadrant | x | y | Style | Perspective | Voice |
|----------|---|---|-------|-------------|-------|
| Q1 | +1 | -1 | Casual | Subjective | Conversational |
| Q2 | +1 | +1 | Casual | Meta | Pop Culture |
| Q3 | -1 | -1 | Formal | Subjective | Literary |
| Q4 | -1 | +1 | Formal | Meta | Scholarly |

## Architecture

```
Surface Text (any language)
        ↓
   Language-Specific Parser
        ↓
   CONCEPT FRAME (order-free)
   {AGENT: X, ACTION: Y, PATIENT: Z, LOCATION: W}
        ↓
   Vector Representation (language-agnostic)
        ↓
   Storage / Query / Holographic Projection
        ↓
   English Answer
```

### Project Structure

```
truthspace_lcm/
├── __init__.py              # Package exports (v0.7.0)
├── chat.py                  # Holographic Q&A chat with φ-dial
├── concept_corpus.json      # Knowledge corpus (11,214 frames)
├── training_data.py         # Pre-defined training examples
├── core/
│   ├── __init__.py          # Core exports
│   ├── vocabulary.py        # Hash-based word positions, IDF weighting
│   ├── concept_language.py  # ConceptFrame, ConceptExtractor, primitives
│   ├── concept_knowledge.py # ConceptKnowledge, HolographicProjector, Q&A
│   ├── answer_patterns.py   # QuaternionPhiDial, PatternAnswerGenerator
│   ├── spatial_attention.py # φ-based navigation, importance scoring
│   ├── learnable_structure.py # Gradient-free learning, EntityProfile
│   ├── conversation_memory.py # Multi-turn dialogue, pronoun resolution
│   ├── reasoning_engine.py  # Multi-hop reasoning, graph traversal
│   ├── holographic_generator.py # Interference-based generation
│   ├── code_generator.py    # Python code generation
│   └── planner.py           # Task planning and sandboxed execution
└── utils/
    └── extractors.py        # Shared extraction utilities
```

## Core Concepts

### Concept Frames

Language-agnostic semantic representation with slots:
- **AGENT** - Who performs the action
- **ACTION** - Primitive (MOVE, SPEAK, THINK, PERCEIVE, FEEL, ACT, EXIST)
- **PATIENT** - Who/what is affected
- **LOCATION/GOAL/SOURCE** - Spatial relations

No word order - just slots filled with concepts.

### Holographic Principle

From holographic stereoscopy:
```
Question = Content - Gap    (has missing information)
Answer   = Content + Fill   (provides missing information)
```

The **axis** (WHO/WHAT/WHERE) defines the gap. The answer fills it.

### Action Primitives

Universal verbs that work across languages:
- **MOVE** - walk, run, go, travel, caminó, fue
- **SPEAK** - say, tell, ask, speak, dijo, habló
- **THINK** - think, consider, believe, pensó, creyó
- **PERCEIVE** - see, hear, notice, vio, oyó
- **FEEL** - feel, love, hate, sintió, amó
- **ACT** - do, make, create, hizo, creó
- **EXIST** - is, was, be, exist, es, fue

## Core Formulas

| Operation | Formula |
|-----------|---------|
| Word Position | `pos(w) = hash(w) → ℝ^64` (deterministic) |
| Frame Vector | `vec(frame) = Σ hash(ROLE:value)` (order-independent) |
| Similarity | `cos(θ) = (a·b) / (‖a‖·‖b‖)` |
| φ-Weighting | `weight = φ^(-log(freq))` (rare = important) |
| φ-Dial | `q = w + xi + yj + zk` (quaternion: style, perspective, depth, certainty) |
| Query | Find frames with highest similarity to query frame |
| Projection | Fill the gap slot based on question axis |

## Testing

```bash
python run.py test          # Run all tests (37 total)
python tests/test_core.py   # Core tests (25)
python tests/test_chat.py   # Chat tests (12)
```

## Design Documents

See `design_considerations/` for the research journey:
- `050_geometric_llm_roadmap.md` - Roadmap for LLM-level capability
- `049_gradient_free_learning.md` - Error-driven structure learning
- `048_clock_geodesic_unification.md` - Clock + geodesic connection
- `047_geodesic_generation.md` - Generation as concept space navigation
- `044_quaternion_phi_dial.md` - 4D quaternion φ-dial with certainty
- `043_3d_phi_dial_depth.md` - 3D φ-dial with depth/elaboration control
- `042_complex_phi_dial.md` - 2D complex φ-dial (style × perspective)
- `041_phi_dial_unified_control.md` - The φ-dial unified control
- `040_phi_inversion_navigation.md` - φ-inversion as navigation mechanism
- `039_phi_zipf_duality.md` - φ and Zipf as dual self-similar fractals
- `038_relationship_formation_autobalance.md` - Spatial attention and importance

## Corpus

The included `concept_corpus.json` contains **11,214 concept frames** extracted from 14 literary works:
- Pride and Prejudice, Dracula, Alice in Wonderland
- Sherlock Holmes, Frankenstein, Moby Dick
- Tale of Two Cities, Tom Sawyer, Great Expectations
- White Fang, Don Quixote (EN & ES), Les Misérables, War and Peace

## Gradient-Free Learning

The system learns without gradients or backpropagation:

```python
from truthspace_lcm import ConceptQA
from truthspace_lcm.training_data import train_model

qa = ConceptQA()
qa.load_corpus('truthspace_lcm/concept_corpus.json')

# Train with quality examples (2 epochs, 96% overlap)
train_model(qa)

# Before: "Holmes is a character who spoke, associated with watson."
# After:  "Holmes is a brilliant detective who investigates with Watson."
```

**Key insight**: Error = Construction Blueprint. Each error points to missing structure. Add the structure. Model improves.

| Aspect | Neural Network | Geometric LCM |
|--------|----------------|---------------|
| Parameters | Weights (continuous) | Mappings (discrete) |
| Learning | Gradient descent | Error-driven addition |
| Convergence | Thousands of epochs | 1-2 epochs |
| Interpretability | Black box | Fully transparent |

## Multi-Hop Reasoning

Chain multiple reasoning steps for complex questions:

```python
from truthspace_lcm.core import ReasoningEngine

engine = ReasoningEngine(qa.knowledge)

# Find relationship path
path = engine.reason("What is the relationship between Darcy and Elizabeth?")
# → Darcy is connected to Elizabeth through 1 steps.
#   → darcy --POSSESS--> elizabeth

# WHY questions
path = engine.reason("Why did Holmes investigate?")
# → Based on the reasoning chain: holmes → scotchman → ...

# HOW questions  
path = engine.reason("How did Darcy act?")
# → The process involves: darcy → elizabeth → charlotte → bingley
```

## Holographic Generation

Generate text using interference patterns:

```python
from truthspace_lcm.core import HolographicGenerator

hologen = HolographicGenerator(qa.knowledge)

# Generate with learned structure
output = hologen.generate("Who is Holmes?", entity="holmes", learnable=learnable)
# → "Holmes is a brilliant detective from Sherlock Holmes who investigates with Watson."
```

**The principle**: Multiple source texts interfere like light waves. Constructive interference (common concepts) = include. Destructive interference (noise) = exclude.

## Code Generation

Generate Python code from natural language:

```python
from truthspace_lcm.core import CodeGenerator

codegen = CodeGenerator()

code = codegen.generate("Write a function to add two numbers")
# → def add(a, b):
#       """Add two numbers"""
#       return a + b

# Teach new functions
codegen.learn('factorial', ['n'], 'return 1 if n <= 1 else n * factorial(n-1)')
```

## Planning & Execution

Plan and execute tasks in a sandboxed environment:

```python
from truthspace_lcm.core import Planner

planner = Planner()

# Plan and execute
plan = planner.plan("Calculate the sum of squares of [1, 2, 3, 4]")
result = planner.execute(plan)
# → Result: 30

# Show plan without executing
plan = planner.plan("Find even numbers in [1, 2, 3, 4, 5, 6]")
# → Step 1: Define data
# → Step 2: Filter by condition
# → Step 3: Return result
```

## Agent Demo

Run the full agent demo showing all capabilities:

```bash
python demo_agent.py
```

This demonstrates:
- Knowledge Q&A about Sherlock Holmes
- Task planning and execution
- Code generation
- Matplotlib chart generation
- Combined agent workflow

## License

MIT
