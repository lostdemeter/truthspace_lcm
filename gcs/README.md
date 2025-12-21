# Geometric Chat System (GCS)

A conversational AI system that replaces traditional neural network-based LLMs with a purely geometric approach based on holographic projection and semantic space navigation.

## Overview

The GCS is the reference implementation of TruthSpace's Language Concept Model (LCM). It demonstrates that meaningful conversation can emerge from pure geometry—no neural networks, no learned weights, no statistical inference.

**Core Principle:** All semantic operations are geometric operations in vector space.

## Directory Structure

```
gcs/
├── README.md                 # This file
├── docs/
│   ├── SRS_geometric_chat_system.md   # Software Requirements Specification
│   └── SDS_geometric_chat_system.md   # Software Design Specification (START HERE)
├── prototypes/
│   ├── style_centroid.py              # Style as centroid (validated 8/8)
│   ├── style_extractor.py             # On-the-fly style extraction (validated 6/6)
│   ├── recursive_holographic_qa.py    # Q&A projection system
│   └── holographic_qa_general.py      # Generalized Q&A system
├── design_docs/
│   ├── 030_geometric_qa_projection.md # Q&A as geometric projection
│   └── 031_unified_projection_framework.md # Unified style/Q&A theory
└── src/                      # Implementation (to be built)
    └── ...
```

## Quick Start for Developers

1. **Read the SDS first**: `docs/SDS_geometric_chat_system.md` contains all the math and algorithms
2. **Review prototypes**: Working code in `prototypes/` demonstrates each component
3. **Check validation**: Style centroid achieved 8/8, style extractor achieved 6/6 accuracy

## Key Concepts

### 1. Semantic Space
All text exists as points in ℝ^64. Words get deterministic positions via hash. Text is encoded as IDF-weighted average of word positions.

### 2. Style = Centroid
A style is fully characterized by the centroid (average position) of its exemplars:
```python
style_centroid = mean([encode(exemplar) for exemplar in exemplars])
```

### 3. Similarity = Cosine
Semantic similarity is cosine similarity between vectors:
```python
similarity = dot(a, b) / (norm(a) * norm(b))
```

### 4. Style Transfer = Interpolation
Apply style by interpolating toward the style centroid:
```python
styled = (1 - α) * content + α * style_centroid
```

### 5. Gap-Filling Q&A
Questions define gaps in semantic space. Answers fill those gaps. Matching is geometric similarity.

## Core Formulas

| Operation | Formula |
|-----------|---------|
| Word Position | `pos(w) = hash(w) → ℝ^dim` |
| Text Encoding | `enc(t) = Σᵢ wᵢ·pos(wordᵢ) / Σᵢ wᵢ` |
| IDF Weight | `w = 1 / log(1 + count)` |
| Cosine Similarity | `sim(a,b) = (a·b) / (‖a‖·‖b‖)` |
| Style Centroid | `c = (1/n) Σᵢ enc(exemplarᵢ)` |
| Style Transfer | `styled = (1-α)·content + α·centroid` |

## Features to Implement

- [ ] **Gutenberg Ingestion**: Auto-download and parse books
- [ ] **Style Extraction**: Learn style from any text source
- [ ] **Style Transfer**: Apply styles to responses
- [ ] **Style Analysis**: Classify text against known styles
- [ ] **Q&A Generation**: Auto-generate WHO/WHAT/WHERE/WHEN/WHY/HOW pairs
- [ ] **Gap-Filling Match**: Geometric Q&A matching
- [ ] **Chat CLI**: Interactive command-line interface

## Design Constraints

- **No Neural Networks**: Pure geometric operations
- **No Hardcoding**: Knowledge/styles emerge from data
- **Deterministic**: Same input → same output
- **NumPy Only**: Minimal dependencies

## Validation Results

| Component | Test | Result |
|-----------|------|--------|
| Style Centroid | 8 style classifications | 8/8 correct |
| Style Extractor | 6 author/format detections | 6/6 correct |
| Q&A Projection | Moby Dick questions | Working |

## License

Part of the TruthSpace LCM project.
