# TruthSpace LCM

**Geometric Language Model** - A modular, interpretable AI system using geometric transformations. No training, no neural networks - just composable gear chains and quaternion geometry.

## Philosophy

> *"All semantic operations are geometric operations in concept space."*
> *"Each gear is a dimension - swap in/out at runtime."*
> *"Corpus is knowledge, gears are reasoning - separate what from how."*

This system demonstrates that **pure geometry can replace trained neural networks** for language understanding AND data transformation. The key insights:

- **Gear Chains** - Composable transformation units that can be swapped at runtime
- **Quaternions** - 4D rotation encoding for parameters and semantic features
- **Domain Agnostic** - Same architecture works for NLP, data pipelines, and more
- **Interpretable** - Every transformation is explicit and debuggable

## Features

### Gear Chain System (NEW)
- **Modular Gears** - Composable transformation units (Role, Action, Tense, Validation, etc.)
- **Domain Agnostic** - Same architecture for NLP chat AND data transformation pipelines
- **Runtime Flexibility** - Swap gears, change tenses, adjust ratios on the fly
- **Error Correction** - 71 irregular verb rules, 40 spelling corrections, pattern-based fixes
- **Corpus Tools** - Pruning, correction, and reinforcement learning for knowledge bases

### Core Geometric Features
- **Geometric Stop Word Detection** - No hard-coded lists; emerges from semantic role absence
- **Position-Based Frame Extraction** - Semantic roles assigned by position bands
- **Geometric Morphology** - Verb equivalence learned from parallel structures (109 clusters)
- **Holographic Template Projection** - Dynamic templates via interference patterns
- **Semantic Quaternions** - 4D concept encoding with 100% analogy accuracy

### Two Quaternions
| Quaternion | Purpose | Axes |
|------------|---------|------|
| **Semantic** | Concept encoding | Gender, Age, Agency (φ-dir), Animacy |
| **Gear** | Transformation params | Accumulated through gear chain |

### Additional Features
- **OpenAI-Compatible API** - REST API with streaming support
- **Data Transformation** - ETL pipelines using the same gear architecture
- **Conversation Memory** - Multi-turn dialogue with pronoun resolution

## Installation

```bash
git clone https://github.com/lostdemeter/truthspace_lcm.git
cd truthspace-lcm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Gear Chain Chat (NEW)

```bash
# Interactive chat using gear chain
python truthspace_lcm/gears/run.py

# Single query
python truthspace_lcm/gears/run.py "What is evolution?"
# Evolution is an entity that involves changing, dividing, and underscoring...

# Debug mode
python truthspace_lcm/gears/run.py --debug "What is Holmes?"
```

### Gear Chain API Server

```bash
# Start OpenAI-compatible API
python truthspace_lcm/gears/run_api.py --port 8000

# Available models: gear-chain, gear-chain-past, gear-chain-future
```

### Data Transformation Pipeline

```bash
# Run data pipeline demo
python truthspace_lcm/gears/practical_applications/data_pipeline.py
```

### Python API (Gear Chain)

```python
from truthspace_lcm.core import GearChain, GearState
from truthspace_lcm.practical_applications.nlp import RoleGear, ActionGear, TenseGear, OutputGear

# Build NLP pipeline
chain = GearChain("NLPPipeline")
chain.add(RoleGear())
chain.add(ActionGear())
chain.add(TenseGear(tense='present'))
chain.add(OutputGear())

# Process
state = GearState(entity="Evolution", role="concept", actions=["adapts", "changes"])
result = chain.process(state)
# "Evolution is a concept that involves adapting and changing."

# Change tense at runtime
chain.get("TenseGear").set_tense('past')
result = chain.process(state)
# "Evolution is a concept that adapted and changed."
```

### Data Pipeline API

```python
from truthspace_lcm.core import GearChain
from truthspace_lcm.practical_applications.data import (
    DataState, ValidationGear, NormalizationGear, FormatGear
)

# Build data pipeline
chain = GearChain("DataPipeline")
chain.add(ValidationGear().add_required("name").add_required("email"))
chain.add(NormalizationGear().trim("name").lowercase("email"))
chain.add(FormatGear(format="json"))

# Process records
state = DataState()
state.add_records(raw_data)
result = chain.process(state)
```

### Legacy API (Original System)

```python
from truthspace_lcm import HolographicGeometricQA

qa = HolographicGeometricQA()
qa.load_corpus('truthspace_lcm/sample_corpus_geometric.json')
answer = qa.ask("Who is Holmes?")
```

## Architecture

### Gear Chain System

```
┌─────────────────────────────────────────────────────────────────┐
│                        GEAR CHAIN                                │
│                                                                  │
│  Input → [RoleGear] → [ActionGear] → [TenseGear] → [OutputGear] │
│            ↓              ↓              ↓             ↓         │
│        Quaternion accumulates through chain                      │
│                                                                  │
│  Same architecture for NLP and Data pipelines!                   │
└─────────────────────────────────────────────────────────────────┘
```

| Neural Networks | Gear Chain |
|-----------------|------------|
| Knowledge + reasoning entangled | Knowledge (corpus) separate from reasoning (gears) |
| Opaque weights | Every transformation explicit |
| Retrain to change | Swap gears at runtime |
| Fixed architecture | Infinite composability |

### Project Structure

```
truthspace_lcm/
├── gears/                           # Modular Gear Chain System
│   ├── core/                        # Domain-agnostic base classes
│   │   ├── base.py                  # Gear, GearState, GearChain, Quaternion
│   │   └── error_correction.py      # ErrorCorrectionGear
│   │
│   ├── practical_applications/      # Domain-specific implementations
│   │   ├── nlp/                     # NLP gears + chat/API
│   │   │   ├── role.py, action.py, tense.py, output.py
│   │   │   ├── chat.py              # Interactive chat
│   │   │   └── api_server.py        # OpenAI-compatible API
│   │   │
│   │   └── data/                    # Data transformation gears
│   │       ├── validation.py, normalization.py, enrichment.py
│   │       └── format.py
│   │
│   ├── corpus/                      # Knowledge corpuses (45K+ frames)
│   ├── tools/                       # Pruner, Corrector, Reinforcer
│   ├── run.py                       # Chat entry point
│   └── run_api.py                   # API server entry point
│
├── core/                            # Original geometric system
│   ├── geometric.py                 # GeometricQA, HolographicGeometricQA
│   ├── holographic_templates.py     # Template projection
│   └── semantic_quaternion.py       # 4D quaternion encoding
│
├── api/                             # Original API server
└── experiments/                     # Research experiments
```

## Core Concepts

### Gear Chain

The fundamental abstraction:
- **Gear** - A transformation unit with `forward()` method
- **GearState** - Data object flowing through the chain
- **GearChain** - Composes gears sequentially
- **Quaternion** - 4D rotation encoding for parameters

Each gear transforms the state and accumulates quaternion rotations.

### Available Gears

**NLP Gears:**
| Gear | Purpose |
|------|---------|
| RoleGear | Transforms roles (character → concept) |
| ActionGear | Converts verbs to gerunds |
| TenseGear | Transforms verb tenses (present, past, future) |
| OutputGear | Assembles final text |

**Data Gears:**
| Gear | Purpose |
|------|---------|
| ValidationGear | Validates data types, ranges, patterns |
| NormalizationGear | Standardizes formats |
| EnrichmentGear | Adds derived fields |
| FilterGear | Filters records |
| FormatGear | Outputs JSON, CSV, etc. |

### Corpus Tools

- **CorpusPruner** - Remove bad frames (duplicates, typos, wrong roles)
- **CorpusCorrector** - Apply spelling and role corrections
- **CorpusReinforcer** - Additive learning by adding frames

## API Server

### Gear Chain API (NEW)

```bash
python truthspace_lcm/gears/run_api.py --port 8000
```

### Legacy API

```bash
python run_api.py --port 8000
```

## Run Tests

```bash
python run.py test
```

## License

GPLv3

## Author

Lesley Gushurst

---

*"Each gear is a dimension. Corpus is knowledge, gears are reasoning."*
