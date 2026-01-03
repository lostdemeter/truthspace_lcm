# TruthSpace Gear Chain System

A modular, extensible gear chain architecture for transformation pipelines.

## Philosophy

The Gear Chain paradigm represents a fundamental shift from monolithic models to **composable transformation units**. Each gear adds dimensions that can be swapped, tuned, or replaced at runtime - something impossible with traditional neural networks where everything is baked into opaque weights.

### Key Principles

1. **Interpretability**: Every transformation is explicit and debuggable
2. **Composability**: Add/remove gears without retraining
3. **Scalability**: Corpus grows additively, gears multiply expressiveness
4. **Runtime Flexibility**: Swap configurations on the fly
5. **Geometric Foundation**: Quaternions provide principled rotation/transformation semantics
6. **Domain Agnostic**: Same architecture works for NLP, data pipelines, and more

## Directory Structure

```
truthspace_lcm/gears/
├── core/                           # Domain-agnostic base classes
│   ├── base.py                     # Gear, GearState, GearChain, Quaternion
│   └── error_correction.py         # ErrorCorrectionGear
│
├── corpus/                         # Corpus management
│   └── corpus_*.json               # Knowledge corpuses
│
├── tools/                          # Corpus tools
│   ├── pruner.py                   # CorpusPruner
│   ├── corrector.py                # CorpusCorrector
│   └── reinforcer.py               # CorpusReinforcer
│
├── practical_applications/         # Domain-specific implementations
│   ├── nlp/                        # NLP gears + applications
│   │   ├── role.py                 # RoleGear
│   │   ├── action.py               # ActionGear
│   │   ├── tense.py                # TenseGear
│   │   ├── output.py               # OutputGear
│   │   ├── chat.py                 # Interactive chat
│   │   └── api_server.py           # OpenAI-compatible API
│   │
│   ├── data/                       # Data transformation gears
│   │   ├── validation.py           # ValidationGear
│   │   ├── normalization.py        # NormalizationGear
│   │   ├── enrichment.py           # EnrichmentGear
│   │   ├── filter.py               # FilterGear
│   │   └── format.py               # FormatGear
│   │
│   └── data_pipeline.py            # Data transformation demo
│
├── run.py                          # Chat entry point
└── run_api.py                      # API server entry point
```

## Quick Start

### NLP Pipeline (Chat)

```python
from truthspace_lcm.core import GearChain, GearState
from truthspace_lcm.practical_applications.nlp import RoleGear, ActionGear, TenseGear, OutputGear

chain = GearChain("NLPPipeline")
chain.add(RoleGear())
chain.add(ActionGear())
chain.add(TenseGear(tense='present'))
chain.add(OutputGear())

state = GearState(
    entity="Evolution",
    role="concept",
    actions=["adapts", "changes"],
    targets=["species"]
)

result = chain.process(state)
# → "Evolution is a concept that involves adapting and changing, particularly species."
```

### Data Pipeline (ETL)

```python
from truthspace_lcm.core import GearChain
from truthspace_lcm.practical_applications.data import (
    DataState, ValidationGear, NormalizationGear, 
    EnrichmentGear, FilterGear, FormatGear
)

chain = GearChain("DataPipeline")

# Add validation
validation = ValidationGear()
validation.add_required("name")
validation.add_required("email")
validation.add_pattern("email", r'^[\w\.-]+@[\w\.-]+\.\w+$')
chain.add(validation)

# Add normalization
normalization = NormalizationGear()
normalization.trim("name").titlecase("name").lowercase("email")
chain.add(normalization)

# Add enrichment
enrichment = EnrichmentGear()
enrichment.add_computed("name_length", lambda r: len(r.get("name", "")))
chain.add(enrichment)

# Add filter
filter_gear = FilterGear()
filter_gear.include_if_field_in("country", ["US", "UK"])
chain.add(filter_gear)

# Add format
chain.add(FormatGear(format="json"))

# Process
state = DataState()
state.add_records(raw_data)
result = chain.process(state)
```

## Running Applications

### Interactive Chat
```bash
python truthspace_lcm/gears/run.py                    # Interactive mode
python truthspace_lcm/gears/run.py "What is evolution?"  # Single query
python truthspace_lcm/gears/run.py --debug "What is Holmes?"
```

### API Server
```bash
python truthspace_lcm/gears/run_api.py                # Default: localhost:8000
python truthspace_lcm/gears/run_api.py --port 18000   # Custom port
```

### Data Pipeline Demo
```bash
python truthspace_lcm/gears/practical_applications/data_pipeline.py
```

## Creating Custom Gears

```python
from truthspace_lcm.core import Gear, GearState

class MyCustomGear(Gear):
    def __init__(self, config: str = "default"):
        super().__init__("MyCustomGear", ratio=1.0)
        self.config = config
    
    def forward(self, state: GearState) -> GearState:
        # Transform state based on config
        if self.config == "uppercase":
            state.entity = state.entity.upper()
        return state

# Use it
chain.add(MyCustomGear("uppercase"))
```

## The Vision

The gear chain can replace neural network functionality because:

| Neural Networks | Gear Chain |
|-----------------|------------|
| Knowledge + reasoning entangled | Knowledge (corpus) separate from reasoning (gears) |
| Opaque weights | Every transformation explicit |
| Retrain to change | Swap gears at runtime |
| Fixed architecture | Infinite composability |
| Domain-specific | Same architecture, any domain |

**Each gear is a dimension** - you can add formality, emotion, audience, domain gears and compose them in any order. The quaternion accumulation provides a geometric signature of the transformation path.

## License

GPLv3 - Lesley Gushurst
