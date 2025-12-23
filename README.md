# TruthSpace LCM

**Holographic Concept Language Model** - A conversational AI using holographic concept resolution. No training, no neural networks - just geometry.

## Philosophy

> *"All semantic operations are geometric operations in concept space."*

This system demonstrates that **pure geometry can replace trained neural networks** for language understanding. Knowledge is stored as **concept frames** - language-agnostic semantic representations that can be queried across languages.

## Features

- **Concept Language** - Order-free semantic frames (like Chinese: no conjugation, flexible order)
- **Holographic Q&A** - Questions are gaps; answers fill them via geometric projection
- **2D φ-Dial Control** - Style (formal↔casual) × Perspective (subjective↔meta)
- **φ-Based Navigation** - Golden ratio powers for entity importance and coherence
- **Cross-Language** - Same concepts work across English, Spanish, and more
- **Spatial Attention** - Zipf/φ-based weighting for meaningful relationships
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
φ-Dial: style=neutral (x=+0.0), perspective=objective (y=+0.0)

Sample characters:
  - Darcy (Pride and Prejudice)
  - Holmes (Sherlock Holmes)
  - Alice (Alice in Wonderland)

You: Who is Holmes?

Bot: Holmes is a character from Sherlock Holmes who spoke, 
     frequently associated with Watson.

You: /style -1
Style set to -1.0 (formal)

You: /perspective 1
Perspective set to +1.0 (meta)

You: Who is Holmes?

Bot: From a literary perspective, Holmes is an archetypal figure 
     from Sherlock Holmes who articulated, with Watson serving as 
     a narrative foil, embodying timeless literary patterns.

You: /dial

φ-Dial Settings:
  Style (x):       -1.0 (formal)
  Perspective (y): +1.0 (meta)
  Quadrant: Scholarly/Analytical
```

### Command Line Options

```bash
python run.py                              # Default (neutral/objective)
python run.py --style -1 --perspective 1   # Formal + Meta
python run.py -x 1 -y -1                   # Casual + Subjective
```

### Run Tests

```bash
python run.py test
```

### Python API

```python
from truthspace_lcm import ConceptQA

# Create Q&A system with 2D φ-dial control
qa = ConceptQA(style_x=0.0, perspective_y=0.0)  # Neutral/Objective
qa.load_corpus('truthspace_lcm/concept_corpus.json')

# Ask questions (holographic resolution)
answer = qa.ask("Who is Holmes?")
# "Holmes is a character from Sherlock Holmes who spoke..."

# Change style dynamically
qa.set_dial(x=-1, y=1)  # Formal + Meta
answer = qa.ask("Who is Holmes?")
# "From a literary perspective, Holmes is an archetypal figure..."

# Or set individually
qa.set_style(-1)        # Formal
qa.set_perspective(1)   # Meta

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

### The 2D φ-Dial

Control both **style** and **perspective** using the complex φ-dial:

```
                    UNIVERSAL (+x)
                         │
         Casual+Meta     │     Casual+Subjective
         "Holmes         │     "I find Holmes to be
          represents     │      quite the clever
          the archetype" │      fellow, really"
                         │
    ─────────────────────●─────────────────────────
                         │                    
         Formal+Meta     │     Formal+Subjective
         "Holmes is an   │     "One observes that
          archetypal     │      Holmes demonstrates
          figure..."     │      remarkable acuity..."
                         │
                    SPECIFIC (-x)
```

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
├── __init__.py              # Package exports (v0.6.0)
├── chat.py                  # Holographic Q&A chat with φ-dial
├── concept_corpus.json      # Knowledge corpus (11,214 frames)
├── core/
│   ├── __init__.py          # Core exports
│   ├── vocabulary.py        # Hash-based word positions, IDF weighting
│   ├── concept_language.py  # ConceptFrame, ConceptExtractor, primitives
│   ├── concept_knowledge.py # ConceptKnowledge, HolographicProjector, Q&A
│   ├── answer_patterns.py   # ComplexPhiDial, PatternAnswerGenerator
│   └── spatial_attention.py # φ-based navigation, importance scoring
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
| φ-Dial | `φ^(x + iy)` where x=style, y=perspective |
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
- `039_phi_zipf_duality.md` - φ and Zipf as dual self-similar fractals
- `040_phi_inversion_navigation.md` - φ-inversion as navigation mechanism
- `041_phi_dial_unified_control.md` - The φ-dial unified control
- `042_complex_phi_dial.md` - 2D complex φ-dial (style × perspective)
- `038_relationship_formation_autobalance.md` - Spatial attention and importance
- `035_autonomous_bootstrap.md` - Concept language breakthrough
- `030_geometric_qa_projection.md` - Q&A as holographic projection

## Corpus

The included `concept_corpus.json` contains **11,214 concept frames** extracted from 14 literary works:
- Pride and Prejudice, Dracula, Alice in Wonderland
- Sherlock Holmes, Frankenstein, Moby Dick
- Tale of Two Cities, Tom Sawyer, Great Expectations
- White Fang, Don Quixote (EN & ES), Les Misérables, War and Peace

## License

MIT
