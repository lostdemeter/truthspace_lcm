# TruthSpace LCM

**Geometric Chat System** - A conversational AI using pure geometric operations in semantic space. No training, no neural networks - just geometry.

## Philosophy

> *"All semantic operations are geometric operations in vector space."*

This system demonstrates that **pure geometry can replace trained neural networks** for semantic understanding. Meaning is position. Similarity is angle. Style is centroid.

## Features

- **64-dimensional semantic space** - Hash-based word positions with IDF weighting
- **Q&A via similarity** - Questions find answers through cosine distance
- **Style extraction** - Learn any style from exemplars (centroid approach)
- **Style transfer** - Move content toward target style
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
  TRUTHSPACE GEOMETRIC CHAT SYSTEM
  All semantic operations are geometric operations
============================================================

Type /help for commands, /quit to exit.

You: What is TruthSpace?

GCS: TruthSpace is a geometric approach to language understanding 
     where meaning is position in semantic space.

You: /style formal

GCS: Style set to: formal

You: /analyze The methodology demonstrates significant improvements.

GCS: Style analysis:
       formal: 0.572
       technical: 0.431
       casual: 0.335
```

### Demo Mode

```bash
python run.py demo
```

### Python API

```python
from truthspace_lcm.core import Vocabulary, KnowledgeBase, StyleEngine

# Create vocabulary and knowledge base
vocab = Vocabulary(dim=64)
kb = KnowledgeBase(vocab)

# Add Q&A pairs
kb.add_qa_pair("What is Python?", "Python is a programming language.")
kb.add_qa_pair("Who is Einstein?", "Einstein was a physicist.")

# Query
results = kb.search_qa("Tell me about Python")
best_qa, similarity = results[0]
print(f"{best_qa.answer} (sim: {similarity:.2f})")

# Style operations
style_engine = StyleEngine(vocab)
style = style_engine.extract_style([
    "The methodology demonstrates improvements.",
    "Results indicate significant findings.",
], "formal")

classification = style_engine.classify("Hey, that's cool!")
print(classification)  # [('casual', 0.38), ('formal', 0.32), ...]
```

## Architecture

```
truthspace_lcm/
├── __init__.py          # Package exports
├── chat.py              # Interactive GeometricChat demo
└── core/
    ├── __init__.py      # Core exports
    ├── vocabulary.py    # Hash-based word positions, IDF weighting
    ├── knowledge.py     # Facts, triples, Q&A pairs, semantic search
    └── style.py         # Style extraction, classification, transfer
```

## Core Formulas

| Operation | Formula |
|-----------|---------|
| Word Position | `pos(w) = hash(w) → ℝ^dim` (deterministic) |
| Text Encoding | `enc(t) = Σᵢ wᵢ·pos(wordᵢ) / Σᵢ wᵢ` (IDF-weighted) |
| IDF Weight | `w = 1 / log(1 + count)` |
| Cosine Similarity | `sim(a,b) = (a·b) / (‖a‖·‖b‖)` |
| Style Centroid | `c = (1/n) Σᵢ enc(exemplarᵢ)` |
| Style Transfer | `styled = (1-α)·content + α·centroid` |

## Key Concepts

### Style = Centroid
A style is the average position of its exemplars in semantic space. This simple insight achieves 8/8 accuracy on style classification.

### Similarity = Cosine
Semantic similarity is the angle between vectors. Identical meaning = angle 0 (cosine 1). Unrelated = orthogonal (cosine 0).

### Gap-Filling Q&A
Questions define gaps in semantic space. Answers fill those gaps. Finding the right answer means finding the question with the most similar gap.

## Testing

```bash
# Run all tests (43 total)
python tests/test_core.py   # 28 tests
python tests/test_chat.py   # 15 tests
```

## Design Documents

See `design_considerations/` for the research journey:
- `030_geometric_qa_projection.md` - Q&A as geometric projection
- `031_unified_projection_framework.md` - Unified style/Q&A theory
- `019_holographic_resolution.md` - Holographic encoding principles

See `gcs/` for implementation specifications:
- `docs/SRS_geometric_chat_system.md` - Requirements specification
- `docs/SDS_geometric_chat_system.md` - Design specification with full math

## License

MIT
