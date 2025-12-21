# TruthSpace LCM Architecture

## Overview

TruthSpace LCM is a **Geometric Chat System** that performs all semantic operations as geometric operations in vector space. No neural networks, no training - just mathematics.

## Core Principle

> **All semantic operations are geometric operations in vector space.**

- **Meaning** = Position in ℝ^64
- **Similarity** = Cosine of angle between vectors
- **Style** = Centroid (average position) of exemplars
- **Transfer** = Interpolation toward target centroid

## Components

### 1. Vocabulary (`vocabulary.py`)

Manages word positions and text encoding.

```
Word → Hash → Random Seed → Unit Vector in ℝ^64
```

**Key Functions:**
- `word_position(word)` - Deterministic position from hash
- `encode(text)` - IDF-weighted average of word positions
- `idf_weight(word)` - Rare words get higher weight: `1/log(1+count)`

### 2. Knowledge Base (`knowledge.py`)

Stores facts, triples, and Q&A pairs with their encodings.

**Data Types:**
- `Fact` - A statement with its encoding
- `Triple` - Subject-predicate-object with modifiers
- `QAPair` - Question, answer, and question type (WHO/WHAT/WHERE/WHEN/WHY/HOW)

**Key Functions:**
- `add_fact(content)` - Store a fact
- `add_qa_pair(question, answer)` - Store Q&A pair
- `search_qa(question)` - Find similar questions by cosine similarity
- `ingest_text(text)` - Extract facts and triples from raw text

### 3. Style Engine (`style.py`)

Extracts, classifies, and transfers styles.

**Key Insight:** A style IS its centroid - the average position of all exemplars.

**Key Functions:**
- `extract_style(exemplars, name)` - Compute centroid from examples
- `classify(text)` - Find nearest style by cosine similarity
- `transfer(content, style, strength)` - Interpolate toward style centroid

## Data Flow

```
                    ┌─────────────┐
                    │   Input     │
                    │   Text      │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Tokenize   │
                    │  (words)    │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   For each word:       │
              │   pos = hash → ℝ^64    │
              │   weight = 1/log(1+n)  │
              └───────────┬────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Weighted   │
                   │  Average    │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Text       │
                   │  Vector     │
                   └─────────────┘
```

## Formulas

### Text Encoding
```
enc(text) = Σᵢ wᵢ · pos(wordᵢ) / Σᵢ wᵢ

where:
  pos(word) = normalize(random_vector(hash(word)))
  wᵢ = 1 / log(1 + count(wordᵢ))
```

### Cosine Similarity
```
sim(a, b) = (a · b) / (‖a‖ · ‖b‖)

Range: [-1, 1]
  1  = identical direction
  0  = orthogonal (unrelated)
  -1 = opposite direction
```

### Style Centroid
```
centroid(style) = (1/n) Σᵢ enc(exemplarᵢ)
```

### Style Transfer
```
styled = (1 - α) · content + α · centroid

where α ∈ [0, 1] controls transfer strength
```

## Directory Structure

```
truthspace-lcm/
├── truthspace_lcm/           # Main package
│   ├── __init__.py           # Package exports
│   ├── chat.py               # Interactive GeometricChat
│   └── core/
│       ├── __init__.py       # Core exports
│       ├── vocabulary.py     # Word positions, IDF, encoding
│       ├── knowledge.py      # Facts, triples, Q&A pairs
│       └── style.py          # Style extraction/transfer
├── tests/
│   ├── test_core.py          # Core tests (28)
│   └── test_chat.py          # Chat tests (15)
├── gcs/                      # GCS specification
│   ├── docs/
│   │   ├── SRS_*.md          # Requirements spec
│   │   └── SDS_*.md          # Design spec (full math)
│   └── prototypes/           # Validated prototypes
├── design_considerations/    # Research journey (019-031)
├── papers/                   # Style/Q&A experiments
├── experiments/              # Earlier experiments
├── run.py                    # Entry point
└── requirements.txt          # Dependencies (numpy)
```

## Validation Results

### Style Classification
- **8/8 accuracy** on style classification (formal/casual/technical)
- Centroid approach outperforms vector arithmetic and co-occurrence methods

### Q&A Matching
- **0.83-1.0 confidence** on bootstrap questions
- Gap-filling principle: similar questions have similar gaps

## Design Decisions

### Why Hash-Based Positions?
- **Deterministic**: Same word always gets same position
- **No training**: Works immediately without data
- **Reproducible**: Results are consistent across runs

### Why IDF Weighting?
- **Meaningful words matter more**: "quantum" > "the"
- **Simple formula**: `1/log(1+count)`
- **Self-adjusting**: Weights update as vocabulary grows

### Why Centroid for Style?
- **Simplest possible approach**: Just average the exemplars
- **Validated**: 8/8 and 6/6 accuracy in experiments
- **Interpretable**: Style IS its average position

## Future Work

See `gcs/docs/SDS_geometric_chat_system.md` for the full implementation roadmap:

1. **Gutenberg Ingestion** - Ingest books from Project Gutenberg
2. **Advanced Q&A** - Multi-hop reasoning via projection chains
3. **Style Blending** - Interpolate between multiple styles
4. **Persistence** - Save/load knowledge bases and styles
