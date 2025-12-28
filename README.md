# TruthSpace LCM

**Geometric Language Model** - A conversational AI using fully geometric language understanding with holographic templates and semantic quaternions. No training, no neural networks, no hard-coded linguistic rules - just geometry.

## Philosophy

> *"All semantic operations are geometric operations in concept space."*
> *"ENCODE = DECODE - they are the same operation in opposite directions, like φ and 1/φ."*
> *"Two quaternions: one for meaning, one for expression. Together they span the space of language."*

This system demonstrates that **pure geometry can replace trained neural networks** for language understanding. The key insights:

- **Position** encodes semantic role (subject at 0, verb at 0.5, object at 1)
- **Frequency** distinguishes content words from function words (Zipf's law)
- **Parallel structure** reveals morphological relationships ("I love. I loved." → love ≡ loved)
- **Quaternions** encode semantic features (100% analogy accuracy)

## Features

### Core Geometric Features
- **Geometric Stop Word Detection** - No hard-coded lists; emerges from semantic role absence
- **Position-Based Frame Extraction** - Semantic roles assigned by position bands [0, 0.33), [0.33, 0.66), [0.66, 1]
- **Geometric Morphology** - Verb equivalence learned from parallel structures (109 clusters)
- **Geometric Conjugation** - Output generation learned from the same parallel structures
- **Holographic Template Projection** - Dynamic templates via interference patterns
- **Semantic Quaternions** - 4D concept encoding with 100% analogy accuracy

### Two Quaternions
| Quaternion | Purpose | Axes |
|------------|---------|------|
| **Semantic** | Concept encoding | Gender, Age, Agency (φ-dir), Animacy |
| **φ-Dial** | Output styling | Style, Perspective, Depth, Certainty |

### Additional Features
- **Conversation Memory** - Multi-turn dialogue with pronoun resolution
- **Multi-Hop Reasoning** - Graph traversal for WHY/HOW questions
- **Code Generation** - Generate Python from natural language
- **Planning & Execution** - Decompose tasks, execute in sandbox
- **OpenAI-Compatible API** - REST API with streaming support

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
  TruthSpace LCM - Geometric Language Model
  Holographic Templates + Semantic Quaternions
============================================================

Architecture:
  • Geometric frame extraction (position bands)
  • Holographic template projection (dynamic responses)
  • Semantic quaternions (100% analogy accuracy)
  • φ-dial output styling (4D quaternion)

Loading corpus...
Loaded 70 frames, 262 concepts
Morphology clusters: 109

You: Who is Holmes?
LCM: Holmes is a notable detective who examines, deduces, and observes

You: /analogy king queen man
king : queen :: man : ?
Top answers: ['woman', 'actress', 'hostess']
```

### Single Query Mode

```bash
python run.py "Who is Holmes?"
# Holmes is a notable detective who examines, deduces, and observes
```

### Command Line Options

```bash
python run.py                              # Default (all neutral)
python run.py --style -1 --perspective 1   # Formal + Meta
python run.py -x 1 -y -1                   # Casual + Subjective
python run.py --depth -1                   # Terse mode
python run.py --debug                      # Show debug info
```

### Python API

```python
from truthspace_lcm import HolographicGeometricQA

# Create Q&A system with holographic templates + quaternions
qa = HolographicGeometricQA()
qa.load_corpus('truthspace_lcm/sample_corpus_geometric.json')

# Ask questions (uses holographic template projection)
answer = qa.ask("Who is Holmes?")
# "Holmes is a notable detective who examines, deduces, and observes"

# Complete analogies (100% accuracy via semantic quaternions)
results = qa.complete_analogy("king", "queen", "man")
# [("woman", 0.0), ("actress", 0.12), ...]

# Find semantic similarity
sim = qa.semantic_similarity("king", "queen")
# 0.5

# Find pairs with similar relations
pairs = qa.find_similar_relations("king", "queen", k=5)
# [("man", "woman", 1.0), ("prince", "princess", 1.0), ...]

# Access geometric components
print(f"Concepts: {len(qa.knowledge.concepts)}")
print(f"Morphology clusters: {len(qa.knowledge.morphology.equivalence_classes)}")

# Check morphological equivalence
qa.knowledge.morphology.are_equivalent("love", "loved")  # True
qa.knowledge.morphology.are_equivalent("go", "went")     # True
```

### The 4D φ-Dial (Output Quaternion)

Control **style**, **perspective**, **depth**, and **certainty**:

| Axis | Name | Range | Controls |
|------|------|-------|----------|
| **X** | Style | -1 to +1 | WHAT words (formal ↔ casual) |
| **Y** | Perspective | -1 to +1 | HOW framed (subjective ↔ meta) |
| **Z** | Depth | -1 to +1 | HOW MUCH detail (terse ↔ elaborate) |
| **W** | Certainty | -1 to +1 | HOW SURE (definitive ↔ hedged) |

### The Semantic Quaternion (Encoding)

4D encoding for concepts with 100% analogy accuracy:

| Axis | Name | Range | Encodes |
|------|------|-------|---------|
| **X** | Gender | -1 to +1 | male ↔ female |
| **Y** | Age | -1 to +1 | adult ↔ young |
| **Z** | Agency | -1 to +1 | initiator ↔ receiver (from φ-direction!) |
| **W** | Animacy | -1 to +1 | human ↔ place |

**Analogy completion**: `? = C + (B - A)` (quaternion arithmetic)

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         GeometricKnowledge          │
                    │  - concepts (φ-direction, roles)    │
                    │  - morphology (verb equivalence)    │
                    │  - frames (initiator/mediator/recv) │
                    └──────────────┬──────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  Holographic    │    │ SemanticQuaternion  │    │    φ-Dial       │
│  Templates      │    │    Navigator        │    │  (Output Style) │
│                 │    │                     │    │                 │
│ - Interference  │    │ - z from φ-dir      │    │ - x: Style      │
│ - Slot filling  │    │ - x,y from parallel │    │ - y: Perspective│
│ - Synthesis     │    │ - 100% analogy      │    │ - z: Depth      │
└─────────────────┘    └─────────────────────┘    │ - w: Certainty  │
       │                           │              └─────────────────┘
       └───────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      HolographicGeometricQA         │
                    │  - ask() with holographic templates │
                    │  - complete_analogy() with quats    │
                    │  - semantic_similarity()            │
                    └─────────────────────────────────────┘
```

### Project Structure

```
truthspace_lcm/
├── __init__.py              # Package exports (v1.0.0)
├── chat.py                  # Interactive chat interface
├── core/
│   ├── __init__.py          # Core exports
│   ├── geometric.py         # GeometricQA, HolographicGeometricQA
│   ├── holographic_templates.py  # Template projection, synthesis
│   ├── semantic_quaternion.py    # 4D quaternion encoding
│   ├── conversation_memory.py    # Multi-turn dialogue
│   ├── reasoning_engine.py       # Multi-hop reasoning
│   ├── code_generator.py         # Python code generation
│   └── planner.py                # Task planning
├── sample_corpus_geometric.json  # Sample corpus

api/
├── server.py                # FastAPI server (OpenAI-compatible)
└── models.py                # Pydantic models

run.py                       # Main entry point
run_api.py                   # API server entry point
ROADMAP.md                   # Development roadmap
```

## Core Concepts

### Geometric Frames

Position-based semantic representation:
- **INITIATOR** - Position [0, 0.33) - Who performs the action (subject)
- **MEDIATOR** - Position [0.33, 0.66) - The action itself (verb)
- **RECEIVER** - Position [0.66, 1.0] - Who/what is affected (object)

No hard-coded parsing - roles emerge from position in sentence.

### Geometric Morphology

Verb equivalence learned from parallel structures:
```
"I love. He loves. I loved."
  ↓
love ≡ loves ≡ loved (same concept, different phases)
```

Works for irregular verbs too: go ≡ goes ≡ went, think ≡ thinks ≡ thought

### Holographic Template Projection

Templates emerge from Q&A pairs via interference:
```
Input: 5 "Who is X?" Q&A pairs
Output Template: {entity} is a {adjective} {role} who {action}
```

Structure words reinforce (same phase), content words cancel (different phases).

### Semantic Quaternions

4D encoding enables analogy completion:
```
king : queen :: man : ? → woman ✓
walk : walked :: run : ? → ran ✓
france : paris :: germany : ? → berlin ✓
```

The z-axis (agency) comes directly from the geometric φ-direction!

## API Server

Start the OpenAI-compatible API:

```bash
python run_api.py --port 8000
```

Use with any OpenAI-compatible client:

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="geometric-lcm",
    messages=[{"role": "user", "content": "Who is Holmes?"}]
)
```

## Run Tests

```bash
python run.py test
```

## Progress

See [ROADMAP.md](ROADMAP.md) for detailed progress on LLM replacement (~60% complete).

| Pillar | Progress | Key Achievement |
|--------|----------|-----------------|
| Understanding | 85% | Fully geometric, no hard-coded rules |
| Knowledge | 50% | Semantic quaternions, need scale |
| Generation | 65% | 100% analogy accuracy, holographic templates |

## License

GPLv3

## Author

Lesley Gushurst

---

*"Structure is the new training. Geometry is the new statistics."*
