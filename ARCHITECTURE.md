# TruthSpace LCM - Architecture

**Version 0.3.0** - 12D Plastic-Primary Encoding

> *"Code is just a geometric interpreter. All logic lives in TruthSpace as knowledge."*

---

## Overview

TruthSpace LCM is a **Language-Code Model** that translates natural language to executable code using geometric knowledge encoding instead of neural networks.

| Aspect | TruthSpace LCM | Traditional LLM |
|--------|----------------|-----------------|
| **Knowledge** | Explicit geometric coordinates | Distributed in billions of weights |
| **Learning** | Direct coordinate assignment | Gradient descent |
| **Retrieval** | Cosine similarity search | Neural forward pass |
| **Interpretability** | Fully transparent | Black box |
| **Size** | ~100 knowledge entries | Billions of parameters |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                            │
│              "compress the logs folder"                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Resolver                                │
│   • No hardcoded patterns                                       │
│   • Fail fast: KnowledgeGapError if no match                    │
│   • Parameter extraction from natural language                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TruthSpace                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            PlasticEncoder (12D ρ-space)                 │   │
│  │  text → primitives → position in 12D semantic space     │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            SQLite Knowledge Store                        │   │
│  │  ~100 entries: primitives + intents + commands          │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Geometric Query                               │   │
│  │  cosine_similarity(query_pos, entry_pos) → best match   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Executor                                │
│   • Safe subprocess execution with timeout                      │
│   • Output capture (stdout, stderr)                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Output                                 │
│  tar -czvf logs.tar.gz logs                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### File Structure

```
truthspace_lcm/
├── core/
│   ├── truthspace.py      # Unified knowledge storage + query
│   ├── resolver.py        # NL → Knowledge → Output
│   ├── encoder.py         # PlasticEncoder (12D ρ-based)
│   ├── autotuner.py       # Dimension-aware autotuning
│   ├── ingestor.py        # Knowledge acquisition
│   ├── executor.py        # Safe code execution
│   └── clock.py           # 12D clock phase oracle
├── truthspace.db          # SQLite knowledge database
└── __init__.py            # Package exports
```

### Component Summary

| Component | Purpose |
|-----------|---------|
| **TruthSpace** | Unified knowledge storage, query, and resolution |
| **PlasticEncoder** | 12D semantic encoding using plastic constant (ρ) |
| **Resolver** | NL → Knowledge → Output with parameter extraction |
| **DimensionAwareAutotuner** | Analyzes and optimizes knowledge placement |
| **Ingestor** | Acquires knowledge from man pages, pydoc |
| **Executor** | Safe bash/python execution with timeout |

---

## 12D Plastic-Primary Encoding

### The Plastic Constant

We use the **plastic constant** (ρ ≈ 1.3247) instead of the golden ratio (φ ≈ 1.618):

```python
RHO = 1.3247179572447458  # Real root of x³ = x + 1
```

**Why ρ over φ?**
- Slower growth rate → finer semantic discrimination
- Cubic recurrence (x³ = x + 1) vs quadratic (x² = x + 1)
- Better separation between semantic levels

### 12D Semantic Space

```
Dimension │ Category   │ Primitives
──────────┼────────────┼─────────────────────────────────
   0-3    │ Actions    │ CREATE↔DESTROY, READ↔WRITE, MOVE/SEARCH, CONNECT/EXECUTE
   4-7    │ Domains    │ FILE/SYSTEM, PROCESS/DATA, NETWORK/USER, MODIFIERS
   8-11   │ Relations  │ TEMPORAL, CAUSAL, CONDITIONAL, COMPARATIVE
```

### Primitive Encoding

Each primitive occupies a specific dimension at a ρ-scaled level:

```python
# Example: CREATE primitive
position = [ρ⁰, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # dim 0, level 0

# Example: DESTROY primitive (opposite of CREATE)
position = [-ρ⁰, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # dim 0, level 0, negative

# Example: BEFORE primitive (temporal relation)
position = [0, 0, 0, 0, 0, 0, 0, 0, ρ⁰, 0, 0, 0]  # dim 8, level 0
```

### Orthogonality Property

Dimensions are **orthogonal** - primitives in different dimensions have zero similarity:

```
similarity(CREATE, FILE) = 0.0    # Different dimensions
similarity(CREATE, DESTROY) = -1.0  # Same dimension, opposite
similarity(CREATE, READ) = 0.0    # Different dimensions
```

This enables **independent tuning** - adding knowledge to dim 0 cannot interfere with dim 4.

---

## Knowledge Base

### Entry Types

```python
class EntryType(Enum):
    PRIMITIVE = "primitive"   # Semantic anchors (CREATE, READ, FILE)
    INTENT = "intent"         # NL → command mapping
    COMMAND = "command"       # Command reference with syntax
    CONCEPT = "concept"       # Parameter extraction patterns
    PATTERN = "pattern"       # Code templates
```

### Current Knowledge

| Type | Count | Examples |
|------|-------|----------|
| **Primitives** | 29 | CREATE, DESTROY, READ, WRITE, BEFORE, AFTER, CAUSE, IF, MORE |
| **Intents** | 29 | list_files, compress_directory, download_file |
| **Commands** | 30 | ls, grep, tar, curl, ssh |
| **Concepts** | 12 | FILENAME, DIRECTORY, URL, INTERFACE, IP_ADDRESS |

### Knowledge Files

```
knowledge/
├── bash_commands.json    # Command reference (30 commands)
├── bash_intents.json     # NL → command mappings (29 intents)
└── parameters.json       # Parameter extraction patterns (12 concepts)
```

---

## Resolution Pipeline

### Example: "compress the logs folder"

**1. Encode Query**
```python
encoder.encode("compress the logs folder")
# Detected primitives: TRANSFORM (dim 2), FILE (dim 4)
# Position: [0, 0, ρ, 0, ρ, 0, 0, 0, 0, 0, 0, 0]
```

**2. Query TruthSpace**
```python
ts.query("compress the logs folder")
# Best match: compress_directory (similarity: 0.89)
# Keywords: [compress, archive, tar, zip, gzip, tarball, tgz, logs, folder]
```

**3. Extract Output**
```python
entry.metadata["target_commands"][0]  # "tar -czvf <directory>.tar.gz <directory>"
entry.metadata["output_type"]         # "bash"
```

**4. Parameter Extraction**
```python
resolver._extract_parameters(request, template)
# Extracts "logs" from "compress the logs folder"
# Result: "tar -czvf logs.tar.gz logs"
```

---

## Dimension-Aware Autotuner

The autotuner analyzes new concepts and recommends optimal placement:

```python
autotuner = DimensionAwareAutotuner(encoder)

# Classify a new concept
analysis = autotuner.classify("synchronize")
# → dim=3 (INTERACTION), type=action, confidence=0.85

# Check for collisions
report = autotuner.check_collisions("sync", dim=3, level=2)
# → existing entries at this position, recommended alternatives

# Auto-place new primitive
primitive = autotuner.auto_place("SYNC", keywords=["sync", "synchronize"])
# → Creates SYNC at optimal position in dim 3
```

---

## Semantic Disambiguation

### Limitation

TruthSpace uses **compositional semantics** (keyword-based), not **distributional semantics** (learned from context). This means:

- 86.7% success rate on bash knowledge tests
- Some queries with keyword overlap match wrong intents
- No contextual word sense disambiguation

### Example Failure

```
Query: "ssh into server.com"
Expected: ssh command
Matched: copy_from_remote (scp)
Reason: "server" keyword overlap
```

### Mitigation Strategies

1. **Trigger phrases** - Exact match before geometric search
2. **N-gram keywords** - "ssh into" as a phrase
3. **Negative keywords** - Anti-associations
4. **Hybrid approach** - LLM for understanding, TruthSpace for retrieval

See `design_considerations/007_semantic_disambiguation.md` for full analysis.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/seed_truthspace.py` | Reset database with 29 primitives |
| `scripts/load_knowledge.py` | Load knowledge from JSON files |
| `run.py` | Interactive CLI |
| `chat.py` | Chat interface with learning |

### Quick Start

```bash
# Reset and seed database
python scripts/seed_truthspace.py
python scripts/load_knowledge.py

# Interactive mode
python run.py

# Single query
python run.py "list files in directory"
```

---

## API Reference

### Basic Usage

```python
from truthspace_lcm import TruthSpace, Resolver

# Initialize
ts = TruthSpace()
resolver = Resolver(ts, auto_learn=True)

# Resolve natural language
result = resolver.resolve("list files in directory")
print(result.output)       # ls -la
print(result.output_type)  # OutputType.BASH

# Resolve and execute
resolution, exec_result = resolver.resolve_and_execute("show current directory")
print(exec_result.stdout)  # /home/user/...
```

### Adding Knowledge

```python
from truthspace_lcm import TruthSpace, EntryType, KnowledgeDomain

ts = TruthSpace()

ts.store(
    name="my_command",
    entry_type=EntryType.INTENT,
    domain=KnowledgeDomain.PROGRAMMING,
    description="My custom command",
    keywords=["my", "custom", "command"],
    metadata={
        "target_commands": ["echo 'Hello from custom command'"],
        "output_type": "bash",
    }
)
```

### Using the Autotuner

```python
from truthspace_lcm import PlasticEncoder, DimensionAwareAutotuner

encoder = PlasticEncoder()
autotuner = DimensionAwareAutotuner(encoder)

# Classify new concept
analysis = autotuner.classify("backup")
print(f"Dimension: {analysis.primary_dimension}")  # 2 (SPATIAL)
print(f"Type: {analysis.primitive_type}")          # action

# Auto-place new primitive
primitive = autotuner.auto_place("BACKUP", keywords=["backup", "save", "archive"])
```

---

## Design Philosophy

### Principles

1. **Minimal Code** - ~3,000 lines of irreducible bootstrap
2. **Maximum Knowledge** - All logic lives in TruthSpace
3. **Fail Fast** - `KnowledgeGapError` if no match (no silent failures)
4. **Interpretable** - Every decision is traceable
5. **Deterministic** - Same input → same output always

### Tradeoffs

| We Get | We Give Up |
|--------|------------|
| Interpretability | Contextual understanding |
| Determinism | Probabilistic flexibility |
| Minimal size | Massive knowledge coverage |
| Fast execution | Neural network capabilities |
| Explicit knowledge | Implicit learning |

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.0 | Initial implementation with φ-based 8D encoding |
| 0.2.0 | Minimal architecture refactor (~6K → ~2.4K lines) |
| 0.3.0 | **12D plastic-primary encoding**, dimension-aware autotuner |

---

*Last updated: December 2024*
