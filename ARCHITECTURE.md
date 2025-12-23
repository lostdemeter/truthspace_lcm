# TruthSpace LCM Architecture

## Overview

TruthSpace LCM is a **Holographic Concept Language Model** that performs all semantic operations as geometric operations in concept space. No neural networks, no training - just mathematics.

## Core Principles

> **All semantic operations are geometric operations in concept space.**

- **Concept Frames** = Order-free semantic representations
- **Action Primitives** = Universal verbs (MOVE, SPEAK, THINK, etc.)
- **Holographic Projection** = Questions are gaps; answers fill them
- **φ-Based Navigation** = Golden ratio powers for importance and coherence
- **3D φ-Dial Control** = Style × Perspective × Depth
- **Cross-Language** = Same concepts work across any language

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

### 3D φ-Dial
```
φ^(x + iy) × scale(z)

Where:
  x = horizontal dial (-1 to +1): Style (formal ↔ casual)
  y = vertical dial (-1 to +1): Perspective (subjective ↔ meta)
  z = depth dial (-1 to +1): Elaboration (terse ↔ elaborate)
  
  Magnitude (φ^x) = WHAT words we choose
  Phase (y·ln(φ)) = HOW we frame the content
  Scale (z) = HOW MUCH detail we include
```

### Query Resolution
```
1. Detect axis (WHO/WHAT/WHERE)
2. Determine navigation direction (inward/outward)
3. Extract entity from question
4. Query frames by entity with φ-weighted importance
5. Aggregate knowledge (action counts, relationships)
6. Apply 3D φ-dial (style × perspective × depth)
7. Project to English (fill the gap)
```

## Directory Structure

```
truthspace-lcm/
├── truthspace_lcm/              # Main package
│   ├── __init__.py              # Package exports (v0.6.0)
│   ├── chat.py                  # Holographic Q&A chat with φ-dial
│   ├── concept_corpus.json      # Knowledge corpus (11,214 frames)
│   ├── core/
│   │   ├── __init__.py          # Core exports
│   │   ├── vocabulary.py        # Word positions, IDF, encoding
│   │   ├── concept_language.py  # ConceptFrame, ConceptExtractor
│   │   ├── concept_knowledge.py # ConceptKnowledge, HolographicProjector
│   │   ├── answer_patterns.py   # ComplexPhiDial, PatternAnswerGenerator
│   │   └── spatial_attention.py # φ-based navigation, importance scoring
│   └── utils/
│       └── extractors.py        # Shared extraction utilities
├── tests/
│   ├── test_core.py             # Core tests (25)
│   └── test_chat.py             # Chat tests (12)
├── design_considerations/       # Research journey
│   ├── 043_3d_phi_dial_depth.md     # 3D φ-dial with depth
│   ├── 042_complex_phi_dial.md      # 2D complex φ-dial
│   ├── 041_phi_dial_unified_control.md  # 1D φ-dial
│   ├── 040_phi_inversion_navigation.md  # φ-inversion navigation
│   ├── 039_phi_zipf_duality.md      # φ and Zipf as dual fractals
│   ├── 038_relationship_formation_autobalance.md  # Spatial attention
│   └── 035_autonomous_bootstrap.md  # Concept language breakthrough
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

## The 3D φ-Dial

The 3D φ-dial provides complete control over answer generation:

| Axis | Name | Range | Controls |
|------|------|-------|----------|
| **X** | Style | -1 to +1 | WHAT words (formal ↔ casual) |
| **Y** | Perspective | -1 to +1 | HOW framed (subjective ↔ meta) |
| **Z** | Depth | -1 to +1 | HOW MUCH detail (terse ↔ elaborate) |

### Z-Axis: Depth/Elaboration

| z | Depth | Max Actions | Relationship | Source | Elaboration |
|---|-------|-------------|--------------|--------|-------------|
| -1 | Terse | 1 | No | No | No |
| 0 | Standard | 2 | Yes | No | No |
| +1 | Elaborate | 4 | Yes | Yes | Yes |

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

## Future Work

1. **More Languages** - Add French, German, Chinese verb mappings
2. **Temporal Reasoning** - Track when events happened
3. **Causal Chains** - Implement oscillating navigation for WHY
4. **Dialogue Context** - Track conversation state
5. **Adaptive φ-Dial** - Learn user preferences for style/perspective
