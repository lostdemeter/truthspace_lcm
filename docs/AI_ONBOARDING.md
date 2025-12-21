# AI Onboarding: TruthSpace LCM

**Purpose**: This document enables AI assistants to quickly understand the project and continue development.

**Last Updated**: December 21, 2025

---

## Quick Start: What Is This Project?

TruthSpace LCM (Large Concept Model) is an experimental AI architecture that operates on **concepts** rather than tokens, using **geometric encoding**. The core thesis:

> **AI is fundamentally a geometric encoder-decoder. Meaning lives at intersection points in truth space, not in words.**

### Current State: Validated Geometric Framework

We have **validated the core geometric approach** with multiple working prototypes:
- **Style extraction/classification**: 8/8 and 6/6 accuracy on style detection
- **Q&A projection**: Geometric gap-filling for question answering
- **Holographic encoding**: Information distributed across representations
- **Centroid-based style transfer**: Styles as positions in semantic space

### Near-Term Goal: Geometric Chat System (GCS)

We are building a **production-quality chat system** that replaces traditional LLMs with pure geometry. See:
- `docs/SRS_geometric_chat_system.md` - Software Requirements Specification
- `docs/SDS_geometric_chat_system.md` - Software Design Specification (detailed math/algorithms)
- `gcs/` - Self-contained implementation directory

```bash
# Run prototype demos
cd /home/thorin/truthspace-lcm
source venv/bin/activate
python papers/style_extractor.py      # Style extraction demo
python papers/style_centroid.py       # Style classification demo
```

---

## The Core Breakthroughs

### 1. Style = Centroid (Validated 8/8, 6/6)

A style is fully characterized by the **centroid** (average position) of its exemplars:

```python
style_centroid = mean([encode(exemplar) for exemplar in exemplars])
similarity = cosine(encode(text), style_centroid)
```

This works for author styles, Q&A formats, technical docs—any text category.

### 2. Q&A as Geometric Projection

Questions define **gaps** in semantic space. Answers **fill** those gaps:
- WHO questions project onto identity axis
- WHAT questions project onto definition axis
- WHERE/WHEN/WHY/HOW project onto their respective axes

### 3. Style Transfer = Interpolation

Apply style by moving content toward the style centroid:

```python
styled = (1 - α) * content + α * style_centroid
```

### 4. Co-occurrence Builds Clusters

Words that appear together form attractor basins automatically:

```
Training: "How do I list files?" → "Use 'ls' to list files"
Result: files↔ls affinity emerges from data
```

---

## Project Evolution

### Phase 1: Composable Primitives (December 16)
- 100% accuracy on 50 bash queries using primitive composition
- Manual rule definition

### Phase 2: Attractor/Repeller Dynamics (December 17-18)
- Proved self-organization works: words cluster by co-occurrence
- Vocabulary emerges from usage patterns

### Phase 3: Auto-Ingestion with LLM (December 18-19)
- Few-shot prompting generates Q&A training data
- Co-occurrence builds clusters automatically

### Phase 4: Interactive Chatbot (December 19)
- Social context detection
- Command execution with sensible defaults

### Phase 5: Holographic Q&A System (December 20)
- Recursive projection: Text → Triples → Q&A pairs
- Gap-filling for question answering
- Generalized to any text source (Project Gutenberg demo)

### Phase 6: Style Framework (December 21)
- Universal style space with semantic axes
- Style = centroid of exemplars (validated 8/8)
- Style extractor for any data source (validated 6/6)
- Unified projection framework: Q&A and style are the same operation

### Phase 7: GCS Specification (December 21) ← CURRENT
- Software Requirements Specification (SRS)
- Software Design Specification (SDS) with full math
- Self-contained `gcs/` directory for implementation

---

## Current Architecture

### Key Files

```
truthspace_lcm/
├── core/
│   ├── engine.py                 # KnowledgeEngine, MultiDomainEngine
│   ├── matcher.py                # CooccurrenceTracker, AffinityMatcher
│   ├── social.py                 # Social context detection
│   └── stacked_lcm.py            # 128D hierarchical embedding

papers/                           # Validated prototypes
├── style_centroid.py             # Style = centroid (8/8 accuracy)
├── style_extractor.py            # On-the-fly style extraction (6/6 accuracy)
├── recursive_holographic_qa.py   # Q&A projection system
└── holographic_qa_general.py     # Generalized Q&A for any text

gcs/                              # Geometric Chat System (implementation target)
├── README.md                     # Overview and quick start
├── docs/
│   ├── SRS_geometric_chat_system.md   # Requirements
│   └── SDS_geometric_chat_system.md   # Design (full math)
├── prototypes/                   # Copied validated code
└── design_docs/                  # Theory documents

design_considerations/            # Research and theory
├── 019-028*.md                   # Holographic, attractor, ingestion theory
├── 030_geometric_qa_projection.md     # Q&A as projection
└── 031_unified_projection_framework.md # Unified style/Q&A theory
```

### Core Geometric Operations

All operations reduce to simple vector arithmetic:

```python
# 1. Text Encoding (IDF-weighted average)
def encode(text):
    words = tokenize(text)
    weights = [1/log(1 + count[w]) for w in words]
    return weighted_average([word_pos[w] for w in words], weights)

# 2. Similarity (cosine)
def similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# 3. Style Extraction (centroid)
def extract_style(exemplars):
    return mean([encode(e) for e in exemplars])

# 4. Style Transfer (interpolation)
def transfer(content, style, strength=0.5):
    return (1 - strength) * encode(content) + strength * style.centroid

# 5. Classification (nearest centroid)
def classify(text, styles):
    v = encode(text)
    return max(styles, key=lambda s: similarity(v, s.centroid))
```

### GCS Target Architecture

The Geometric Chat System (see `gcs/docs/SDS_geometric_chat_system.md`) implements:

1. **Vocabulary**: Hash-based word positions + IDF weighting
2. **Knowledge Base**: Facts, triples, Q&A pairs with encodings
3. **Style Engine**: Extract, classify, transfer styles
4. **Query Processor**: Parse, encode, match, generate responses
5. **Ingestion Pipeline**: Gutenberg, text files, Q&A pairs

---

## How to Add New Knowledge

### For GCS (Target System)

```python
from gcs import GeometricChatSystem

gcs = GeometricChatSystem()

# Ingest from Project Gutenberg
gcs.ingest_gutenberg(2701)  # Moby Dick

# Ingest from file
gcs.ingest_file("my_document.txt")

# Ingest Q&A pairs
gcs.ingest_qa_pairs([
    ("Who is Captain Ahab?", "Captain Ahab is the captain of the Pequod."),
    ("What is the white whale?", "Moby Dick is a great white sperm whale."),
])

# Extract style from any text
gcs.extract_style("path/to/author.txt", "AuthorStyle")
```

### For Current Prototypes

```python
# Style extraction
from papers.style_extractor import StyleExtractor
extractor = StyleExtractor()
style = extractor.extract_from_text(my_text, "MyStyle")

# Q&A system
from papers.holographic_qa_general import HolographicQA
qa = HolographicQA()
qa.ingest_text(my_text)
answer = qa.query("Who is the main character?")
```

---

## Key Design Documents

| Document | Purpose |
|----------|---------|
| `docs/SRS_geometric_chat_system.md` | **What to build** - Requirements for GCS |
| `docs/SDS_geometric_chat_system.md` | **How to build it** - Full math and algorithms |
| `design_considerations/030_geometric_qa_projection.md` | Q&A as geometric projection |
| `design_considerations/031_unified_projection_framework.md` | Unified style/Q&A theory |
| `design_considerations/019_holographic_resolution.md` | Holographic encoding principles |
| `design_considerations/022_attractor_repeller_dynamics.md` | Self-organization proof |

---

## Running Prototypes

### Style Demos

```bash
cd /home/thorin/truthspace-lcm
source venv/bin/activate

# Style classification (8/8 accuracy)
python papers/style_centroid.py

# Style extraction from any source (6/6 accuracy)
python papers/style_extractor.py

# Q&A projection demo
python papers/recursive_holographic_qa.py
```

### Legacy Chatbot (still works)

```bash
python experiments/phi_ingestion_prototype.py chat
```

---

## Mathematical Foundation

### Core Formulas (from SDS)

| Operation | Formula |
|-----------|---------|
| Word Position | `pos(w) = hash(w) → ℝ^dim` (deterministic) |
| Text Encoding | `enc(t) = Σᵢ wᵢ·pos(wordᵢ) / Σᵢ wᵢ` (IDF-weighted) |
| IDF Weight | `w = 1 / log(1 + count)` |
| Cosine Similarity | `sim(a,b) = (a·b) / (‖a‖·‖b‖)` |
| Style Centroid | `c = (1/n) Σᵢ enc(exemplarᵢ)` |
| Style Transfer | `styled = (1-α)·content + α·centroid` |

### Key Principles

1. **Style = Centroid**: A style is the average position of its exemplars
2. **Similarity = Cosine**: Semantic similarity is angle between vectors
3. **Transfer = Interpolation**: Move content toward style centroid
4. **Gap-Filling**: Questions define gaps, answers fill them

---

## Key Insights

1. **"All semantic operations are geometric"** - Vector arithmetic, not learned weights

2. **"Style = Centroid"** - Average position captures essence (validated 8/8, 6/6)

3. **"No hardcoding"** - Knowledge and styles emerge from data

4. **"Deterministic"** - Same input always produces same output

5. **"Q&A and Style are the same operation"** - Both are projections in semantic space

---

## Next Steps: Build GCS

**Completed:**
- ✅ Style centroid approach validated (8/8 accuracy)
- ✅ Style extractor validated (6/6 accuracy)
- ✅ Q&A projection system working
- ✅ SRS and SDS documents written
- ✅ Self-contained `gcs/` directory created

**Next: Implement GCS**
1. Build vocabulary system with hash-based positions
2. Implement knowledge base (facts, triples, Q&A pairs)
3. Build style engine (extract, classify, transfer)
4. Create query processor with gap-filling
5. Add Gutenberg ingestion
6. Build CLI chat interface

See `gcs/docs/SDS_geometric_chat_system.md` for full implementation details.

---

## Quick Reference

| Task | Command/File |
|------|--------------|
| Style classification | `python papers/style_centroid.py` |
| Style extraction | `python papers/style_extractor.py` |
| Q&A projection | `python papers/recursive_holographic_qa.py` |
| GCS requirements | `docs/SRS_geometric_chat_system.md` |
| GCS design (full math) | `docs/SDS_geometric_chat_system.md` |
| GCS implementation dir | `gcs/` |
| Design theory | `design_considerations/030-031*.md` |

---

*"All semantic operations are geometric operations in vector space."*
