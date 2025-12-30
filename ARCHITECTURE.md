# TruthSpace LCM Architecture

## Overview

TruthSpace LCM is a **Geometric Language Model** with a modular **Gear Chain** architecture. It performs all semantic operations as geometric transformations - no neural networks, no training, just composable gears and quaternion geometry.

**Version**: 2.0.0

## Core Principles

> **Each gear is a dimension - swap in/out at runtime.**
> **Corpus is knowledge, gears are reasoning - separate what from how.**
> **Same architecture works for NLP, data pipelines, and more.**

### The Gear Chain System

| Component | Purpose | Key Innovation |
|-----------|---------|----------------|
| **Gear** | Transformation unit | Composable, swappable at runtime |
| **GearState** | Data flowing through chain | Domain-agnostic state object |
| **GearChain** | Composes gears | Sequential processing with quaternion accumulation |
| **Quaternion** | 4D rotation encoding | Parameters and semantic features |

---

## Architecture Diagram

### Gear Chain Architecture

```
INPUT: GearState(entity="Holmes", role="character", actions=["investigates"])
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GEAR CHAIN                                │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ RoleGear │ → │ActionGear│ → │TenseGear │ → │OutputGear│     │
│  │          │   │          │   │          │   │          │     │
│  │ Classify │   │ Gerunds  │   │ Tense    │   │ Assemble │     │
│  │ roles    │   │ convert  │   │ transform│   │ text     │     │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘     │
│       │              │              │              │            │
│       └──────────────┴──────────────┴──────────────┘            │
│                              │                                   │
│                    Quaternion accumulates                        │
│                    through each gear                             │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: "Holmes is a character who investigating, particularly crimes."
```

### Data Pipeline Architecture

```
INPUT: DataState(records=[{name: "John", age: "25"}, ...])
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                               │
│                                                                  │
│  ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐     │
│  │Validation │→ │Normalizat- │→ │Enrichment│→ │ Format   │     │
│  │Gear       │  │ionGear     │  │Gear      │  │ Gear     │     │
│  │           │  │            │  │          │  │          │     │
│  │ Required  │  │ Trim, case │  │ Computed │  │ JSON/CSV │     │
│  │ Type/Range│  │ Date parse │  │ Lookups  │  │ output   │     │
│  └───────────┘  └────────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: [{"name": "John", "age": 25, "age_group": "young"}, ...]
```

---

## Core Components

### Gear (`gears/core/base.py`)

The fundamental transformation unit.

```python
class Gear(ABC):
    name: str
    ratio: float  # 0.0 to 1.0, controls transformation strength
    enabled: bool
    q: Quaternion  # Gear's quaternion parameters
    
    @abstractmethod
    def forward(self, state: GearState) -> GearState:
        """Transform the state. Must be implemented by subclasses."""
        pass
    
    def set_ratio(self, ratio: float) -> 'Gear'
    def enable(self) / disable(self)
```

### GearState (`gears/core/base.py`)

Data object flowing through the chain.

```python
@dataclass
class GearState:
    entity: str = ""           # Main entity being described
    role: str = ""             # Role (character, concept, field, etc.)
    actions: List[str]         # Verbs/actions
    targets: List[str]         # Objects/targets
    connector: str = "that"    # Connecting word
    accumulated_q: Quaternion  # Accumulated quaternion through chain
    metadata: Dict[str, Any]   # Additional data
```

### GearChain (`gears/core/base.py`)

Composes gears into a pipeline.

```python
class GearChain:
    name: str
    gears: List[Gear]
    
    def add(self, gear: Gear, position: int = None) -> 'GearChain'
    def remove(self, name: str) -> 'GearChain'
    def get(self, name: str) -> Optional[Gear]
    def process(self, state: GearState) -> Any
    def set_ratio(self, gear_name: str, ratio: float)
    def disable(self, gear_name: str)
```

### Quaternion (`gears/core/base.py`)

4D rotation encoding.

```python
@dataclass
class Quaternion:
    w: float = 1.0  # Scalar (formality/quality)
    x: float = 0.0  # Style/Gender
    y: float = 0.0  # Perspective/Age
    z: float = 0.0  # Depth/Agency
    
    def __mul__(self, other) -> 'Quaternion'  # Quaternion multiplication
    def normalize(self) -> 'Quaternion'
    def magnitude(self) -> float
```

---

## NLP Gears (`gears/practical_applications/nlp/`)

### RoleGear

Transforms roles based on concept type.

```python
class RoleGear(Gear):
    def forward(self, state: GearState) -> GearState:
        # Detect if entity is person, abstract, plural
        # Transform role accordingly (character → concept)
```

### ActionGear

Converts verbs to gerunds.

```python
class ActionGear(Gear):
    def forward(self, state: GearState) -> GearState:
        # When ratio > 0.5, convert verbs to gerunds
        # "investigates" → "investigating"
```

### TenseGear

Transforms verb tenses.

```python
class TenseGear(Gear):
    tense: str  # 'present', 'past', 'future', 'perfect'
    
    def set_tense(self, tense: str)
    def forward(self, state: GearState) -> GearState:
        # Transform actions to specified tense
```

### OutputGear

Assembles final text output.

```python
class OutputGear(Gear):
    def forward(self, state: GearState) -> str:
        # Assemble: "{entity} is a {role} {connector} {actions}, {targets}"
```

---

## Data Gears (`gears/practical_applications/data/`)

### ValidationGear

Validates data records.

```python
class ValidationGear(Gear):
    def add_required(self, field: str) -> 'ValidationGear'
    def add_type(self, field: str, expected_type: str) -> 'ValidationGear'
    def add_range(self, field: str, min_val, max_val) -> 'ValidationGear'
    def add_pattern(self, field: str, pattern: str) -> 'ValidationGear'
```

### NormalizationGear

Standardizes data formats.

```python
class NormalizationGear(Gear):
    def trim(self, field: str) -> 'NormalizationGear'
    def lowercase(self, field: str) -> 'NormalizationGear'
    def to_int(self, field: str) -> 'NormalizationGear'
    def to_date(self, field: str) -> 'NormalizationGear'
```

### EnrichmentGear

Adds derived fields.

```python
class EnrichmentGear(Gear):
    def add_computed(self, field: str, fn: Callable) -> 'EnrichmentGear'
    def add_lookup(self, target: str, source: str, table: str) -> 'EnrichmentGear'
```

### FormatGear

Outputs in various formats.

```python
class FormatGear(Gear):
    format: str  # 'dict', 'json', 'csv', 'summary'
    
    def forward(self, state: GearState) -> Any:
        # Format records according to specified format
```

---

## Corpus Tools (`gears/tools/`)

### CorpusPruner

Removes bad data from corpus.

```python
class CorpusPruner:
    def set_min_length(self, length: int)
    def set_max_duplicates(self, count: int)
    def prune(self, corpus: Dict) -> Tuple[Dict, PruneResult]
```

### CorpusCorrector

Applies corrections to corpus.

```python
class CorpusCorrector:
    def add_spelling(self, wrong: str, correct: str)
    def add_role_correction(self, concept: str, role: str)
    def correct(self, corpus: Dict) -> Tuple[Dict, CorrectResult]
```

### CorpusReinforcer

Additive learning by adding frames.

```python
class CorpusReinforcer:
    def reinforce(self, concept: str, role: str, actions: List[str], strength: int)
    def apply(self, corpus: Dict) -> Tuple[Dict, ReinforceResult]
```

---

## Legacy Components

The original geometric system is still available in `truthspace_lcm/core/`:

- **GeometricKnowledge** - Position-based frame extraction
- **HolographicTemplateProjector** - Dynamic templates via interference
- **SemanticQuaternionNavigator** - 100% analogy accuracy
- **HolographicGeometricQA** - Unified Q&A system

---

## File Structure

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
│   │       ├── validation.py, normalization.py
│   │       └── format.py
│   │
│   ├── corpus/                      # Knowledge corpuses (45K+ frames)
│   ├── tools/                       # Pruner, Corrector, Reinforcer
│   ├── run.py                       # Chat entry point
│   └── run_api.py                   # API server entry point
│
├── core/                            # Original geometric system
│   ├── geometric.py                 # GeometricQA, HolographicGeometricQA
│   └── semantic_quaternion.py       # 4D quaternion encoding
│
├── api/                             # Original API server
└── experiments/                     # Research experiments
```

---

## The Vision

**"Each gear is a dimension. Corpus is knowledge, gears are reasoning."**

| Neural Networks | Gear Chain |
|-----------------|------------|
| Knowledge + reasoning entangled | Knowledge (corpus) separate from reasoning (gears) |
| Opaque weights | Every transformation explicit |
| Retrain to change | Swap gears at runtime |
| Fixed architecture | Infinite composability |
| Domain-specific | Same architecture, any domain |

The key insight: The same gear chain architecture that powers NLP chat also powers data transformation pipelines. **Structure is the new training.**

---

*"Each gear is a dimension - swap in/out at runtime."*
