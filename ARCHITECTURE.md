# TruthSpace LCM Architecture

## Overview

TruthSpace LCM is a **Holographic Concept Language Model** that performs all semantic operations as geometric operations in concept space. No neural networks, no training - just mathematics.

## Core Principles

> **All semantic operations are geometric operations in concept space.**

- **Concept Frames** = Order-free semantic representations
- **Action Primitives** = Universal verbs (MOVE, SPEAK, THINK, etc.)
- **Holographic Projection** = Questions are gaps; answers fill them
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

High-level Q&A interface.

```python
qa = ConceptQA()
qa.load_corpus('concept_corpus.json')
answer = qa.ask("Who is Darcy?")
result = qa.ask_detailed("What did Holmes do?")
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

### Query Resolution
```
1. Detect axis (WHO/WHAT/WHERE)
2. Extract entity from question
3. Query frames by entity
4. Aggregate knowledge (action counts, relationships)
5. Project to English (fill the gap)
```

## Directory Structure

```
truthspace-lcm/
├── truthspace_lcm/              # Main package
│   ├── __init__.py              # Package exports (v0.5.0)
│   ├── chat.py                  # Holographic Q&A chat
│   ├── concept_corpus.json      # Knowledge corpus (11,214 frames)
│   ├── core/
│   │   ├── __init__.py          # Core exports
│   │   ├── vocabulary.py        # Word positions, IDF, encoding
│   │   ├── concept_language.py  # ConceptFrame, ConceptExtractor
│   │   └── concept_knowledge.py # ConceptKnowledge, HolographicProjector
│   └── utils/
│       └── extractors.py        # Shared extraction utilities
├── tests/
│   ├── test_core.py             # Core tests (25)
│   └── test_chat.py             # Chat tests (12)
├── design_considerations/       # Research journey
│   ├── 035_autonomous_bootstrap.md  # Concept language breakthrough
│   └── 030_geometric_qa_projection.md  # Holographic Q&A
├── scripts/                     # Utility scripts
│   └── concept_chat.py          # Standalone chat script
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

## Future Work

1. **More Languages** - Add French, German, Chinese verb mappings
2. **Temporal Reasoning** - Track when events happened
3. **Causal Chains** - Link events by cause and effect
4. **Dialogue Context** - Track conversation state
