# TruthSpace LCM

**Dynamic Geometric Language Model** - A conversational AI using pure geometric operations in semantic space. No training, no neural networks - just geometry.

## Philosophy

> *"Structure IS the data. Learning IS structure update."*

This system demonstrates that **pure geometry can replace trained neural networks** for language understanding. Knowledge is stored as geometry - entity positions and relation vectors - not as neural network weights.

## Features

- **Dynamic Learning** - Learn new facts from natural language in real-time
- **Relational Queries** - "What is the capital of France?" → "paris"
- **Analogical Reasoning** - "france:paris :: germany:?" → "berlin" (100% accuracy)
- **Multi-hop Reasoning** - Find paths between entities through relations
- **256-dimensional semantic space** - Hash-based positions with learned structure
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
  Structure IS the data. Learning IS structure update.
============================================================

Type /help for commands, /quit to exit.
Ask questions or teach me new facts!

You: What is the capital of France?

GCS: The capital of france is paris.

You: Beijing is the capital of China.

GCS: Learned: china --capital_of--> beijing

You: /analogy france paris china

GCS: Analogy: france:paris :: china:?
       beijing: 0.996
       berlin: 0.321
       tokyo: 0.288

You: /status

GCS: SYSTEM STATUS
     Entities: 26
     Relations: 4
     Facts: 16
```

### Demo Mode

```bash
python run.py demo
```

### Run Tests

```bash
python run.py test
```

### Python API

```python
from truthspace_lcm.core import GeometricLCM

# Create the geometric language model
lcm = GeometricLCM(dim=256)

# Learn facts from natural language
lcm.tell("Paris is the capital of France.")
lcm.tell("Berlin is the capital of Germany.")
lcm.tell("Tokyo is the capital of Japan.")

# Query relations
results = lcm.query("france", "capital_of", k=1)
print(results)  # [('paris', 0.99)]

# Solve analogies
results = lcm.analogy("france", "paris", "germany", k=1)
print(results)  # [('berlin', 0.98)]

# Natural language questions
answer = lcm.ask("What is the capital of France?")
print(answer)  # "The capital of france is paris."

# Multi-hop reasoning
paths = lcm.find_path("france", "europe", max_hops=2)
print(paths)  # [(['france', 'located_in', 'europe'], 0.98)]

# Save and load
lcm.save("knowledge.json")
lcm.load("knowledge.json")
```

## Architecture

```
truthspace_lcm/
├── __init__.py          # Package exports
├── chat.py              # Interactive GeometricChat demo
└── core/
    ├── __init__.py      # Core exports
    ├── geometric_lcm.py # Dynamic Geometric Language Model (main)
    ├── vocabulary.py    # Hash-based word positions, IDF weighting
    ├── knowledge.py     # Facts, triples, Q&A pairs
    └── style.py         # Style extraction, classification, transfer
```

## Core Concepts

### Structure IS the Data

Traditional LLMs store knowledge in billions of neural network weights. TruthSpace stores knowledge as **geometry**:

- **Entities** have positions in 256D space
- **Relations** are learned vector offsets between entities
- **Facts** are (subject, relation, object) triples

### Learning IS Structure Update

When you teach TruthSpace a new fact:
1. Entity positions are created/updated
2. Relation vectors are refined to be consistent across all instances
3. The geometric structure converges (typically in 4-10 iterations)

### Analogies Work Because Relations Are Invariant

After learning:
- `paris - france ≈ berlin - germany ≈ tokyo - japan`
- All point in the same "capital_of" direction
- So `france + (paris - france) ≈ paris` works for any country

## Core Formulas

| Operation | Formula |
|-----------|---------|
| Entity Position | `pos(e) = hash(e) → ℝ^256` (initial, then learned) |
| Relation Vector | `rel = avg(object - subject)` across all instances |
| Query | `target = subject + relation` → find nearest entity |
| Analogy | `answer = c + (b - a)` for a:b :: c:? |
| Consistency | Pairwise similarity of relation offsets (target: >95%) |

## Testing

```bash
python run.py test          # Run all tests (49 total)
python tests/test_core.py   # Core tests (29)
python tests/test_chat.py   # Chat tests (20)
```

## Design Documents

See `design_considerations/` for the research journey:
- `033_dynamic_geometric_lcm.md` - Dynamic Geometric LCM architecture
- `032_vsa_binding_extension.md` - VSA binding operations
- `030_geometric_qa_projection.md` - Q&A as geometric projection

See `experiments/` for exploration:
- `sparse_vsa_exploration_v3.py` - Breakthrough: 100% analogy accuracy
- `geometric_lcm_full.py` - Full system with NL parsing

## License

MIT
