# TruthSpace LCM

**Holographic Concept Language Model** - A conversational AI using holographic concept resolution. No training, no neural networks - just geometry.

## Philosophy

> *"All semantic operations are geometric operations in concept space."*

This system demonstrates that **pure geometry can replace trained neural networks** for language understanding. Knowledge is stored as **concept frames** - language-agnostic semantic representations that can be queried across languages.

## Features

- **Concept Language** - Order-free semantic frames (like Chinese: no conjugation, flexible order)
- **Holographic Q&A** - Questions are gaps; answers fill them via geometric projection
- **Cross-Language** - Same concepts work across English, Spanish, and more
- **64-dimensional semantic space** - Hash-based positions with conceptual primitives
- **No training** - Deterministic, interpretable, reproducible

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

Sample characters:
  - Darcy (Pride and Prejudice)
  - Holmes (Sherlock Holmes)
  - Alice (Alice in Wonderland)

You: Who is Darcy?

Bot: Darcy is a character from Pride and Prejudice who appears 
     and possesses things often involving elizabeth

You: What did Holmes do?

Bot: Holmes was present, had

You: /entity watson

Entity: Watson
  Source: Sherlock Holmes
  Actions: THINK, EXIST, POSSESS

You: /stats

Corpus Statistics:
  Total frames: 11214
  Unique entities: 1759
  Entity relations: 3029
```

### Run Tests

```bash
python run.py test
```

### Python API

```python
from truthspace_lcm import ConceptQA

# Create Q&A system
qa = ConceptQA()
qa.load_corpus('truthspace_lcm/concept_corpus.json')

# Ask questions (holographic resolution)
answer = qa.ask("Who is Darcy?")
# "Darcy is a character from Pride and Prejudice..."

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
├── __init__.py              # Package exports (v0.5.0)
├── chat.py                  # Holographic Q&A chat interface
├── concept_corpus.json      # Knowledge corpus (11,214 frames)
├── core/
│   ├── __init__.py          # Core exports
│   ├── vocabulary.py        # Hash-based word positions, IDF weighting
│   ├── concept_language.py  # ConceptFrame, ConceptExtractor, primitives
│   └── concept_knowledge.py # ConceptKnowledge, HolographicProjector, Q&A
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
- `035_autonomous_bootstrap.md` - Concept language breakthrough
- `030_geometric_qa_projection.md` - Q&A as holographic projection
- `031_unified_projection_framework.md` - Unified projection theory

## Corpus

The included `concept_corpus.json` contains **11,214 concept frames** extracted from 14 literary works:
- Pride and Prejudice, Dracula, Alice in Wonderland
- Sherlock Holmes, Frankenstein, Moby Dick
- Tale of Two Cities, Tom Sawyer, Great Expectations
- White Fang, Don Quixote (EN & ES), Les Misérables, War and Peace

## License

MIT
