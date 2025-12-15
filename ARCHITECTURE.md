# TruthSpace LCM - Architecture & Theory Deep Dive

**A comprehensive guide for understanding and continuing development of the TruthSpace Language-Code Model**

> **Version 0.2.0** - Minimal Architecture Refactor
> 
> The system has been consolidated from ~6,000 lines to ~2,400 lines.
> All logic now lives in knowledge space; code is just a geometric interpreter.

---

## Table of Contents

1. [Philosophy & Vision](#philosophy--vision)
2. [What is an LCM vs LLM?](#what-is-an-lcm-vs-llm)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Minimal Architecture](#minimal-architecture)
5. [Core Components](#core-components)
6. [Knowledge Base Structure](#knowledge-base-structure)
7. [Resolution Pipeline](#resolution-pipeline)
8. [Current State & Capabilities](#current-state--capabilities)
9. [Design Decisions & Rationale](#design-decisions--rationale)
10. [Development Guidelines](#development-guidelines)

---

## Philosophy & Vision

### The Core Insight

Traditional LLMs (Large Language Models) are statistical pattern matchers trained on massive datasets. They work, but they're:
- **Opaque**: We don't know *why* they produce specific outputs
- **Expensive**: Billions of parameters, massive compute requirements
- **Fragile**: Knowledge can interfere with other knowledge (catastrophic forgetting)

**TruthSpace LCM takes a different approach**: Instead of learning statistical patterns, we encode knowledge geometrically in a structured "truth space" where:
- Knowledge has explicit, interpretable coordinates
- Different domains are mathematically isolated
- New knowledge adds to the space without destroying existing knowledge
- Retrieval is based on geometric similarity, not neural activation

### The Name "TruthSpace"

The name comes from the idea that there exists a mathematical space where "truths" (facts, patterns, procedures) can be located by coordinates. Similar truths are geometrically close. Different domains occupy orthogonal subspaces.

### LCM = Language-Code Model

We call this an **LCM** (Language-Code Model) rather than LLM because:
1. It's not "large" - it's compact and interpretable
2. It's not purely "language" - it's specifically designed for code generation
3. The "model" is geometric/mathematical, not neural

---

## What is an LCM vs LLM?

| Aspect | LLM (Large Language Model) | LCM (Language-Code Model) |
|--------|---------------------------|---------------------------|
| **Knowledge Storage** | Distributed in billions of weights | Explicit geometric coordinates |
| **Learning** | Gradient descent on loss | Direct coordinate assignment |
| **Retrieval** | Forward pass through network | Geometric similarity search |
| **Interpretability** | Black box | Fully interpretable |
| **Domain Isolation** | Implicit, can interfere | Explicit orthogonal subspaces |
| **Size** | Billions of parameters | Hundreds of knowledge entries |
| **Compute** | GPU clusters | Single CPU |
| **Modification** | Requires retraining | Direct CRUD operations |

### The Key Tradeoff

LLMs can generate *anything* because they've seen *everything*. Our LCM can only generate what's in its knowledge base, but:
- We know exactly what it knows
- We can add/remove/modify knowledge precisely
- It will never hallucinate beyond its knowledge
- It's completely reproducible

---

## Mathematical Foundations

### Universal Constants as Semantic Dimensions

We use mathematical constants as the basis for our semantic space. This isn't arbitrary - these constants appear throughout nature and mathematics, suggesting they capture something fundamental:

```python
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PI = np.pi                   # ≈ 3.14159
E = np.e                     # ≈ 2.71828
SQRT2 = np.sqrt(2)          # ≈ 1.414
SQRT3 = np.sqrt(3)          # ≈ 1.732
LN2 = np.log(2)             # ≈ 0.693
GAMMA = 0.5772156649        # Euler-Mascheroni constant
ZETA3 = 1.2020569           # Apéry's constant
```

### Why These Constants?

1. **φ (phi)** - The golden ratio appears in growth patterns, aesthetics, and self-similarity. We use it for **identity/naming** - the most fundamental aspect of knowledge.

2. **π (pi)** - Appears in circles, waves, and periodic phenomena. We use it for **spatial relationships**.

3. **e** - The base of natural logarithms, appears in growth/decay. We use it for **temporal aspects**.

4. **γ (gamma)** - Euler-Mascheroni constant, appears in number theory. We use it for **causal relationships**.

5. **√2, √3** - Geometric constants. Used for **methods and attributes**.

### φ-Encoder: Semantic Primitives

The system uses a **φ-encoder** that maps natural language to geometric positions using hand-crafted semantic primitives:

```python
class PrimitiveType(Enum):
    ACTION = "action"      # Verbs: CREATE, DESTROY, READ, WRITE, etc.
    DOMAIN = "domain"      # Nouns: FILE, PROCESS, NETWORK, SYSTEM, etc.
    MODIFIER = "modifier"  # Adjectives: ALL, RECURSIVE, VERBOSE, etc.

# 17 core primitives with φ-anchored positions
PRIMITIVES = {
    # Actions
    "CREATE": (ACTION, [φ, 0, 0, 0, 0, 0, 0, 0]),
    "DESTROY": (ACTION, [-φ, 0, 0, 0, 0, 0, 0, 0]),
    "READ": (ACTION, [0, φ, 0, 0, 0, 0, 0, 0]),
    "WRITE": (ACTION, [0, -φ, 0, 0, 0, 0, 0, 0]),
    # ... etc
}
```

### The Encoding Formula

For a knowledge entry, we compute its position using the φ-encoder:

```python
def _compute_position(self, name: str, domain: KnowledgeDomain,
                      keywords: List[str]) -> np.ndarray:
    """Compute geometric position using φ-encoder."""
    # Combine name and keywords for encoding
    text_to_encode = f"{name} {' '.join(keywords)}"
    
    # Get semantic decomposition from φ-encoder
    decomposition = self._phi_encoder.encode(text_to_encode)
    position = decomposition.position.copy()
    
    # Add domain component for isolation
    domain_offset = DOMAIN_CONSTANTS[domain]
    position = position * 0.8  # Scale semantic components
    position[0] += domain_offset * 0.2  # Add domain signal
    
    return position
```

The φ-encoder:
1. Tokenizes the input text
2. Matches tokens to semantic primitives
3. Computes weighted sum of primitive positions
4. Returns 8-dimensional position vector

### Domain Isolation

Different knowledge domains occupy **orthogonal subspaces**:

```python
class KnowledgeDomain(Enum):
    PROGRAMMING = 1.0   # Code, libraries, patterns
    HISTORY = 2.0       # Historical facts
    SCIENCE = 3.0       # Scientific knowledge
    GEOGRAPHY = 4.0     # Geographic information
    GENERAL = 5.0       # General knowledge
    CUSTOM = 6.0        # User-defined
```

The domain value is multiplied by φ and placed in dimension 7, ensuring:
- Programming knowledge (1.0 × φ) is geometrically distant from
- History knowledge (2.0 × φ) which is distant from
- Science knowledge (3.0 × φ), etc.

This prevents "George Washington" from interfering with "Beautiful Soup" - they're in completely different regions of truth space.

### Similarity Search

To find relevant knowledge, we compute cosine similarity:

```python
def _compute_similarity(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute cosine similarity between positions."""
    dot = np.dot(pos1, pos2)
    norm1 = np.linalg.norm(pos1)
    norm2 = np.linalg.norm(pos2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)
```

---

## Geometric Knowledge Encoding

### The Knowledge Entry

Each piece of knowledge is stored as a `KnowledgeEntry`:

```python
@dataclass
class KnowledgeEntry:
    id: str                      # Unique identifier (hash)
    name: str                    # Human-readable name
    domain: KnowledgeDomain      # Which domain
    entry_type: str              # "function", "pattern", "command", etc.
    description: str             # What it does
    keywords: List[str]          # Semantic keywords for positioning
    position: np.ndarray         # 8D position in truth space
    metadata: Dict[str, Any]     # Additional data (syntax, examples, etc.)
    created_at: datetime
    updated_at: datetime
    version: int
```

### Example: Encoding "print" Function

```python
entry = manager.create(
    name="print",
    domain=KnowledgeDomain.PROGRAMMING,
    entry_type="function",
    description="Output text to console",
    keywords=["print", "output", "display", "console", "show", "text"],
    metadata={
        "syntax": "print(*args, sep=' ', end='\\n')",
        "example": "print('Hello, World!')"
    }
)
```

This creates an entry positioned in truth space based on the keywords ["print", "output", "display", "console", "show", "text"]. When someone asks to "show text on screen", the query keywords ["show", "text", "screen"] will have high similarity to this entry.

### Persistence

Knowledge is persisted in a **SQLite database** optimized for φ-based vectors:

```
truthspace_lcm/
└── knowledge.db          # SQLite database

Database Schema:
├── entries               # Main knowledge entries
│   ├── id TEXT PRIMARY KEY
│   ├── name TEXT
│   ├── domain TEXT
│   ├── entry_type TEXT
│   ├── description TEXT
│   ├── position BLOB     # 8D vector as binary
│   ├── position_norm REAL # Pre-computed for fast similarity
│   ├── metadata TEXT     # JSON
│   └── version INTEGER
├── keywords              # Normalized keyword table
│   ├── id INTEGER PRIMARY KEY
│   └── keyword TEXT UNIQUE
└── entry_keywords        # Many-to-many relationship
    ├── entry_id TEXT
    └── keyword_id INTEGER
```

Benefits over JSON files:
- **ACID transactions** for safe concurrent access
- **Indexed queries** for fast keyword filtering
- **Pre-computed norms** for efficient similarity search
- **Single file** instead of hundreds of JSON files

---

## Minimal Architecture

> **Design Principle**: Code is just a geometric interpreter. All logic lives in TruthSpace as knowledge.

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                            │
│              "write a hello world python program"               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Resolver                                 │
│   - NO hardcoded patterns                                       │
│   - NO fallbacks (fail fast)                                    │
│   - Just: query → extract → return                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TruthSpace                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              φ-Encoder (semantic math)                   │   │
│  │  text → primitives → position in 8D space               │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SQLite Knowledge Store                      │   │
│  │  ~40 entries: primitives + intents + commands           │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Geometric Query                             │   │
│  │  cosine_similarity(query_pos, entry_pos) → best match   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Executor                                │
│   - Subprocess execution with timeout                           │
│   - Output capture (stdout, stderr)                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Output                                 │
│  print("Hello, World!")                                         │
│  → Hello, World!                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### File Structure (~2,400 lines total)

| File | Lines | Purpose |
|------|-------|---------|
| `truthspace.py` | 647 | Unified knowledge storage + query |
| `resolver.py` | 321 | NL → Knowledge → Output |
| `ingestor.py` | 497 | Knowledge acquisition |
| `encoder.py` | 365 | φ-based semantic encoding |
| `executor.py` | 513 | Safe code execution |

### 1. TruthSpace (`truthspace.py`)

**Purpose**: Unified knowledge storage and query interface.

**Key Classes**:
```python
class EntryType(Enum):
    PRIMITIVE = "primitive"   # Semantic anchors (CREATE, READ, FILE)
    INTENT = "intent"         # NL trigger → command mapping
    COMMAND = "command"       # Executable command knowledge
    PATTERN = "pattern"       # Code templates
    CONCEPT = "concept"       # General knowledge

class KnowledgeEntry:
    id: str
    name: str
    entry_type: EntryType
    domain: KnowledgeDomain
    description: str
    position: np.ndarray      # 8D φ-encoded position
    keywords: List[str]
    metadata: Dict[str, Any]
```

**Key Methods**:
```python
# Store knowledge
entry = ts.store(name, entry_type, domain, description, keywords, metadata)

# Query (raises KnowledgeGapError if no match)
results = ts.query(text, entry_type=None, domain=None, threshold=0.3)

# Resolve to executable output
output, output_type, entry = ts.resolve(text)
```

### 2. Resolver (`resolver.py`)

**Purpose**: Thin NL → Knowledge → Output interface.

**Design**: NO hardcoded patterns. Everything comes from TruthSpace.

```python
class Resolver:
    def resolve(self, request: str) -> Resolution:
        # 1. Query TruthSpace
        # 2. If KnowledgeGapError and auto_learn, try to learn
        # 3. Extract output from best match
        # 4. Return Resolution with output + metadata
    
    def resolve_and_execute(self, request: str) -> Tuple[Resolution, ExecutionResult]:
        # Resolve + execute in one step
```

### 3. Encoder (`encoder.py`)

**Purpose**: φ-based semantic encoding.

**Primitives** (loaded from TruthSpace or bootstrap):
- **ACTIONS** (dims 0-3): CREATE, DESTROY, READ, WRITE, MOVE, CONNECT, SEARCH, EXECUTE
- **DOMAINS** (dims 4-6): FILE, PROCESS, NETWORK, SYSTEM, DATA, USER
- **MODIFIERS** (dim 7): ALL, RECURSIVE, FORCE, VERBOSE

```python
encoder = PhiEncoder()
result = encoder.encode("list files in directory")
# result.primitives = [READ, FILE]
# result.position = [0.62, 0.38, 0, 0, 0.85, 0, 0, 0]
```

### 4. Ingestor (`ingestor.py`)

**Purpose**: Knowledge acquisition from various sources.

```python
ingestor = Ingestor(ts)

# Auto-detect and ingest
entry = ingestor.ingest("ls")  # From man page

# Custom knowledge
entry = ingestor.ingest_custom(
    name="backup_dir",
    description="Create timestamped backup",
    keywords=["backup", "tar", "archive"],
    output_type="bash",
    syntax="tar -czf backup_$(date +%Y%m%d).tar.gz <dir>"
)
```

### 5. Executor (`executor.py`)

**Purpose**: Safe code execution.

```python
executor = CodeExecutor()
result = executor.execute_python('print("Hello")')
result = executor.execute_bash('ls -la')
```

---

## Knowledge Base Structure

### Minimal Seed (~40 entries)

The system starts with minimal knowledge that can be expanded:

| Type | Count | Examples |
|------|-------|----------|
| **Primitives** | 19 | CREATE, READ, WRITE, FILE, NETWORK, RECURSIVE |
| **Intents** | 17 | list_files → `ls -la`, hello_world → `print("Hello, World!")` |
| **Commands** | 5 | ls, cd, cat, grep, find |

### Entry Types

```python
PRIMITIVE   # Semantic anchors - the building blocks of meaning
INTENT      # NL trigger → command mapping (high precision)
COMMAND     # Executable command with syntax/examples
PATTERN     # Code templates
CONCEPT     # General knowledge
META        # Stop words, config, etc.
```

### Expanding Knowledge

1. **Seed script**: `python scripts/seed_truthspace.py` - Reset to minimal
2. **Auto-learning**: System learns from man pages on `KnowledgeGapError`
3. **Manual**: Use `TruthSpace.store()` or `Ingestor.ingest_custom()`

---

## Resolution Pipeline

### Step-by-Step Example

**Input**: "list files in directory"

**1. Encode**:
```python
encoder.encode("list files in directory")
# Primitives: [READ, FILE]
# Position: [0.62, 0.38, 0, 0, 0.85, 0, 0, 0]
```

**2. Query TruthSpace**:
```python
ts.query("list files in directory", entry_type=EntryType.INTENT)
# Best match: list_files intent (similarity: 0.87)
```

**3. Extract Output**:
```python
entry.metadata["target_commands"][0]  # "ls -la"
entry.metadata["output_type"]         # "bash"
```

**4. Execute** (optional):
```python
subprocess.run("ls -la", shell=True)
# stdout: "total 48\ndrwxr-xr-x..."
```

---

## Current State & Capabilities

### What Works ✅

1. **Hello World**: `"write a hello world python program"` → `print("Hello, World!")`
2. **File Operations**: List, create, delete files and directories
3. **Network**: Show interfaces, download files
4. **Search**: Find files, grep patterns
5. **System**: Disk usage, process list, system info
6. **Safe Execution**: Timeout protection, output capture

### Design Philosophy

- **Fail Fast**: No fallbacks - `KnowledgeGapError` if no match
- **Minimal Code**: ~2,400 lines of irreducible bootstrap
- **Maximum Knowledge**: Everything else is in TruthSpace
- **Auto-Learning**: Can acquire knowledge from man pages/pydoc

### Current Limitations ⚠️

1. **Minimal Knowledge**: Only ~40 seed entries (by design - expandable)
2. **No Context Memory**: Each request is independent
3. **English Only**: Primitives are English keywords

---

## Design Decisions & Rationale

### Why Minimal Code + Maximum Knowledge?

The previous architecture had ~6,000 lines with hardcoded patterns scattered across:
- `bash_generator.py` - 15 intent patterns
- `code_generator.py` - 9 intent patterns
- `task_planner.py` - 6 task patterns
- `phi_encoder.py` - 22 hardcoded primitives

**Problem**: Adding new capabilities required code changes.

**Solution**: Move everything to knowledge space:
- Primitives are now knowledge entries (type=PRIMITIVE)
- Intents are knowledge entries (type=INTENT)
- Commands are knowledge entries (type=COMMAND)

Now adding capabilities = adding knowledge, not code.

### Why Geometric Encoding Instead of Embeddings?

**Neural embeddings** (like word2vec, BERT):
- Require training data
- Are opaque (can't inspect why two things are similar)
- Can drift or change with retraining

**Our geometric encoding**:
- Deterministic: same keywords → same position always
- Interpretable: we know exactly which dimensions contribute
- Stable: adding new knowledge doesn't change existing positions
- Fast: just hash + arithmetic, no neural forward pass

### Why Fail Fast?

**Previous approach**: Multiple fallback layers
- Try intent patterns → Try knowledge query → Try auto-learn → Return generic error

**Problem**: Hard to debug, unpredictable behavior, code complexity.

**New approach**: Query succeeds or raises `KnowledgeGapError`
- Clear signal that knowledge is missing
- Triggers auto-learning opportunity
- No silent failures

---

## Development Guidelines

### Adding Knowledge

```python
from truthspace_lcm import TruthSpace, EntryType, KnowledgeDomain

ts = TruthSpace()

# Add an intent
ts.store(
    name="my_intent",
    entry_type=EntryType.INTENT,
    domain=KnowledgeDomain.PROGRAMMING,
    description="What this does",
    keywords=["keyword1", "keyword2"],
    metadata={
        "target_commands": ["actual_command"],
        "output_type": "bash",  # or "python"
    }
)
```

### Testing

```bash
# Interactive test
python run.py

# Single query
python run.py "your test request"

# Python API test
python -c "
from truthspace_lcm import TruthSpace, Resolver
ts = TruthSpace()
resolver = Resolver(ts)
result = resolver.resolve('your test request')
print(result.output)
"
```

### Resetting Knowledge

```bash
# Reset to minimal seed
python scripts/seed_truthspace.py
```

---

## Quick Reference

### File Locations

| File | Purpose |
|------|---------|
| `truthspace_lcm/core/truthspace.py` | Unified knowledge storage + query |
| `truthspace_lcm/core/resolver.py` | NL → Knowledge → Output |
| `truthspace_lcm/core/ingestor.py` | Knowledge acquisition |
| `truthspace_lcm/core/encoder.py` | φ-based semantic encoding |
| `truthspace_lcm/core/executor.py` | Safe code execution |
| `truthspace_lcm/truthspace.db` | SQLite knowledge database |
| `scripts/seed_truthspace.py` | Reset knowledge to minimal |
| `run.py` | Interactive runner |

### Key Classes

```python
from truthspace_lcm import (
    TruthSpace,          # Unified knowledge interface
    KnowledgeEntry,      # Single knowledge item
    KnowledgeDomain,     # Domain enum (PROGRAMMING, SYSTEM, GENERAL)
    EntryType,           # Entry type enum (PRIMITIVE, INTENT, COMMAND, etc.)
    KnowledgeGapError,   # Raised when no match found
    Resolver,            # NL → Knowledge → Output
    Resolution,          # Result of resolution
    OutputType,          # BASH, PYTHON, TEXT
    Ingestor,            # Knowledge acquisition
    PhiEncoder,          # Semantic encoding
    CodeExecutor,        # Safe execution
    ExecutionResult,     # Execution output
    ExecutionStatus,     # SUCCESS, FAILED, TIMEOUT
)
```

### Common Operations

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
print(exec_result.stdout)

# Add custom knowledge
ts.store(
    name="my_command",
    entry_type=EntryType.INTENT,
    domain=KnowledgeDomain.PROGRAMMING,
    description="My custom command",
    keywords=["my", "custom"],
    metadata={"target_commands": ["echo 'custom'"], "output_type": "bash"}
)
```

---

## Summary

TruthSpace LCM v0.2.0 is a **minimal, knowledge-first** system for code generation:

| Aspect | Description |
|--------|-------------|
| **Code** | ~2,400 lines of irreducible bootstrap |
| **Knowledge** | ~40 seed entries (expandable) |
| **Philosophy** | Code is a geometric interpreter; logic lives in knowledge |
| **Fail Mode** | Fast - `KnowledgeGapError` if no match |
| **Learning** | Auto-acquire from man pages/pydoc |

The result is a system that's:
- **Transparent**: We know exactly what it knows
- **Minimal**: Only essential code, everything else is knowledge
- **Extensible**: Add capabilities by adding knowledge, not code
- **Fast**: No GPU required, runs on any machine

---

*Last updated: December 2024*
*Version: 0.2.0*
