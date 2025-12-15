# TruthSpace LCM - Architecture & Theory Deep Dive

**A comprehensive guide for understanding and continuing development of the TruthSpace Language-Code Model**

---

## Table of Contents

1. [Philosophy & Vision](#philosophy--vision)
2. [What is an LCM vs LLM?](#what-is-an-lcm-vs-llm)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Geometric Knowledge Encoding](#geometric-knowledge-encoding)
5. [Architecture Overview](#architecture-overview)
6. [Component Deep Dives](#component-deep-dives)
7. [Knowledge Base Structure](#knowledge-base-structure)
8. [Code Generation Pipeline](#code-generation-pipeline)
9. [Current State & Capabilities](#current-state--capabilities)
10. [Design Decisions & Rationale](#design-decisions--rationale)
11. [Known Limitations](#known-limitations)
12. [Future Directions](#future-directions)
13. [Development Guidelines](#development-guidelines)

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

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                            │
│              "create a python project called myapp"             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Task Planner                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Detect task type (project setup, file op, web, etc.) │   │
│  │ 2. Decompose into atomic steps                          │   │
│  │ 3. Establish dependencies between steps                 │   │
│  │ 4. Determine step type (Python vs Bash)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Code Generators                            │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │   Python Generator   │    │    Bash Generator    │          │
│  │  ┌────────────────┐  │    │  ┌────────────────┐  │          │
│  │  │ Intent Detection│  │    │  │ Intent Detection│  │          │
│  │  │ Keyword Extract │  │    │  │ Path Extraction │  │          │
│  │  │ Knowledge Query │  │    │  │ Knowledge Query │  │          │
│  │  │ Code Composition│  │    │  │ Cmd Composition │  │          │
│  │  └────────────────┘  │    │  └────────────────┘  │          │
│  └──────────────────────┘    └──────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Knowledge Manager                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Geometric Similarity Search                 │   │
│  │  query_keywords → position → cosine_similarity → top_k  │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Knowledge Store                        │   │
│  │  77 Python entries │ 74 Bash entries │ Domain isolation │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Executor                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Isolated temp directory                                │   │
│  │ • Subprocess execution with timeout                      │   │
│  │ • Output capture (stdout, stderr)                        │   │
│  │ • Error diagnosis and suggestions                        │   │
│  │ • Validation rules                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Results                                │
│  ✅ Step 1: mkdir -p myapp                                      │
│  ✅ Step 2: mkdir -p myapp/src                                  │
│  ✅ Step 3: touch myapp/src/__init__.py                        │
│  ✅ Step 4: touch myapp/src/main.py                            │
│  ✅ Step 5: touch myapp/README.md                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dives

### 1. Knowledge Manager (`knowledge_manager.py`)

**Purpose**: Store, retrieve, and manage knowledge entries using geometric encoding.

**Key Methods**:
```python
# Create new knowledge
entry = manager.create(name, domain, entry_type, description, keywords, metadata)

# Query by keywords (returns sorted by similarity)
results = manager.query(keywords, domain=None, top_k=10)

# CRUD operations
entry = manager.read(entry_id)
entry = manager.update(entry_id, updates)
manager.delete(entry_id)  # Creates backup first

# Persistence
manager.save()  # Save all to disk
manager.load()  # Load from disk
```

**Design Principles**:
- **Additive**: New entries add to space, never overwrite
- **Versioned**: Updates create backups with version numbers
- **Safe Deletes**: Deleted entries are backed up, not destroyed

### 2. Code Generator (`code_generator.py`)

**Purpose**: Translate natural language to Python code.

**Key Components**:

1. **Intent Detection**: Regex patterns to identify what the user wants
```python
self.intent_patterns = {
    "fetch_url": [r"fetch\s+(?:the\s+)?(?:url|page|website)", ...],
    "read_file": [r"read\s+(?:the\s+)?(?:file|json|data)", ...],
    "write_file": [r"write\s+(?:to\s+)?(?:a\s+)?file", ...],
    # ... more patterns
}
```

2. **Keyword Extraction**: Pull relevant terms from request
3. **Knowledge Query**: Find matching entries in knowledge base
4. **Code Composition**: Assemble code from templates and knowledge

### 3. Bash Generator (`bash_generator.py`)

**Purpose**: Translate natural language to Bash commands.

**Similar structure to Code Generator but optimized for shell commands**:
- Path extraction from requests
- Command composition
- Flag handling

### 4. Task Planner (`task_planner.py`)

**Purpose**: Decompose complex tasks into executable steps.

**Task Types**:
```python
self.task_patterns = {
    "create_project": [...],    # Multi-step project setup
    "scrape_and_save": [...],   # Fetch → Parse → Save
    "backup_task": [...],       # Create dir → Compress → Verify
    "organize_files": [...],    # File organization
}
```

**Decomposition Example**:
```
"create a python project called myapp"
    ↓
Step 1: mkdir -p myapp
Step 2: mkdir -p myapp/src         [depends on 1]
Step 3: touch myapp/src/__init__.py [depends on 2]
Step 4: touch myapp/src/main.py     [depends on 2]
Step 5: touch myapp/README.md       [depends on 1]
```

### 5. Executor (`executor.py`)

**Purpose**: Safely execute generated code and validate results.

**Safety Features**:
- Isolated temporary directory
- Subprocess execution (not `exec()`)
- Timeout protection
- Output capture
- Error diagnosis

**Validation System**:
```python
ValidationRule(
    name="file_exists",
    check_type="file_exists",
    expected="output.json",
    message="Expected file was not created"
)
```

---

## Knowledge Base Structure

### Python Knowledge (77 entries)

| Category | Count | Examples |
|----------|-------|----------|
| Core Functions | 10 | print, input, len, range, type, str, int, list, dict, open |
| Requests Library | 8 | get, post, response handling, headers, JSON |
| JSON Library | 6 | load, dump, loads, dumps, file operations |
| OS Library | 8 | path operations, environment, file system |
| File Operations | 10 | read, write, append, binary, context managers |
| String Operations | 8 | split, join, format, strip, replace |
| List Operations | 8 | append, extend, sort, filter, map, comprehensions |
| Control Flow | 6 | if/else, for, while, try/except |
| Web Scraping | 6 | BeautifulSoup, selectors, parsing |
| Common Patterns | 7 | file reading, JSON handling, error handling |

### Bash Knowledge (74 entries)

| Category | Count | Examples |
|----------|-------|----------|
| File Operations | 6 | touch, cp, mv, rm, ln, file |
| Directory Operations | 7 | mkdir, rmdir, cd, pwd, ls, tree, find |
| Text Processing | 8 | grep, sed, awk, sort, uniq, wc, cut, tr |
| File Viewing | 5 | cat, less, head, tail, diff |
| Permissions | 3 | chmod, chown, chgrp |
| System Info | 9 | echo, whoami, hostname, uname, df, du, date, env, which |
| Process Management | 7 | ps, top, kill, killall, bg, fg, nohup |
| Networking | 6 | curl, wget, ping, ssh, scp, netstat |
| Compression | 5 | tar, gzip, gunzip, zip, unzip |
| Common Patterns | 18 | backup, search, loops, conditionals |

---

## Code Generation Pipeline

### Step-by-Step Example

**Input**: "fetch https://api.github.com and parse the JSON"

**1. Intent Detection**:
```python
intents = ["fetch_url"]  # Matches "fetch ... url" pattern
```

**2. URL Extraction**:
```python
url = "https://api.github.com"  # Extracted via regex
```

**3. Knowledge Query**:
```python
keywords = ["fetch", "url", "json", "parse", "api"]
results = manager.query(keywords, domain=PROGRAMMING)
# Returns: requests.get, json parsing patterns, etc.
```

**4. Code Composition**:
```python
code = '''
import requests

url = 'https://api.github.com'

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f'Error: {response.status_code}')
'''
```

**5. Import Inference**:
```python
# Scans code for library usage
imports = ["import requests"]  # Added automatically
```

---

## Current State & Capabilities

### What Works Well ✅

1. **Project Setup**: "create a python project called X" → Full directory structure
2. **File Operations**: Create, copy, move, delete files and directories
3. **Simple Code Generation**: Print statements, file I/O, JSON handling
4. **Web Requests**: Fetch URLs, parse JSON responses
5. **Multi-Step Tasks**: Automatic decomposition and dependency tracking
6. **Safe Execution**: Isolated environment, timeout protection
7. **Error Diagnosis**: Helpful error messages with fix suggestions

### Current Limitations ⚠️

1. **No Learning**: Knowledge must be manually added (by design, for control)
2. **Template-Based**: Code comes from templates, not generated creatively
3. **Limited Complexity**: Can't handle highly complex or novel requests
4. **No Context Memory**: Each request is independent (no conversation state)
5. **English Only**: Intent patterns are English-specific

### Test Results

```
✅ KnowledgeManager tests passed
✅ CodeGenerator tests passed  
✅ BashGenerator tests passed
✅ TaskPlanner tests passed
✅ CodeExecutor tests passed
✅ End-to-End tests passed

Results: 6 passed, 0 failed
```

---

## Design Decisions & Rationale

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

### Why Regex for Intent Detection?

**Pros**:
- Explicit and debuggable
- No training required
- Easy to add new patterns
- Predictable behavior

**Cons**:
- Doesn't handle paraphrasing well
- Requires manual pattern creation
- Can miss edge cases

**Future**: Could add fuzzy matching or simple ML classifier as enhancement.

### Why Separate Python and Bash Generators?

Different domains have different:
- Syntax requirements
- Common patterns
- Error handling
- Output expectations

Keeping them separate allows specialized handling while sharing the knowledge infrastructure.

### Why Template-Based Code Generation?

**Templates are**:
- Correct by construction
- Easy to verify
- Predictable
- Maintainable

**The tradeoff**: Less flexible than neural generation, but much more reliable for the supported use cases.

---

## Known Limitations

### Intent Detection Gaps

Some requests don't match patterns well:
- "show me what's in the folder" (informal phrasing)
- "make a backup of everything" (vague scope)
- "do the same thing but for CSV" (requires context)

### Code Generation Limitations

1. **No variable tracking**: Can't reference results from previous steps
2. **Fixed templates**: Can't combine patterns in novel ways
3. **No type inference**: Doesn't understand data types flowing through code

### Knowledge Base Gaps

- No database operations (SQL, MongoDB)
- No async/await patterns
- No class definitions
- No testing frameworks
- Limited error handling patterns

---

## Future Directions

### Short-Term Improvements

1. **More Knowledge Entries**: Add SQL, async, classes, testing
2. **Better Intent Detection**: Fuzzy matching, synonyms
3. **Variable Passing**: Track outputs between steps
4. **Interactive Refinement**: Ask clarifying questions

### Medium-Term Goals

1. **Learning from Execution**: If code fails, remember the fix
2. **User Customization**: Let users add their own patterns
3. **Context Window**: Remember recent requests for follow-ups
4. **Code Explanation**: Explain what generated code does

### Long-Term Vision

1. **Self-Improving Knowledge**: Automatically extract patterns from successful executions
2. **Multi-Language Support**: JavaScript, Go, Rust generators
3. **IDE Integration**: VS Code extension
4. **Collaborative Knowledge**: Share knowledge bases between users

---

## Development Guidelines

### Adding New Knowledge

```python
# In python_knowledge_builder.py or bash_knowledge_builder.py

def _build_new_category(self) -> int:
    entries = [
        ("entry_name", "entry_type",
         "Description of what it does",
         ["keyword1", "keyword2", "keyword3"],
         {"syntax": "...", "example": "..."}),
    ]
    for name, etype, desc, kw, meta in entries:
        self._create(name, etype, desc, kw, meta)
    return len(entries)
```

### Adding New Intent Patterns

```python
# In code_generator.py or bash_generator.py

self.intent_patterns["new_intent"] = [
    r"pattern1\s+with\s+groups?",
    r"alternative\s+pattern",
]

# Then add handler in generate() method
if "new_intent" in intents:
    code, entries = self._generate_new_intent(request)
    # ...
```

### Testing Changes

```bash
# Run test suite
python tests/test_basic.py

# Test specific functionality
python -c "
from truthspace_lcm import TaskPlanner
p = TaskPlanner()
plan = p.plan('your test request')
print(plan.steps)
"
```

### Code Style

- Type hints on all public methods
- Docstrings for classes and public methods
- Keep methods focused (single responsibility)
- Prefer explicit over implicit

---

## Quick Reference

### File Locations

| File | Purpose |
|------|---------|
| `truthspace_lcm/core/knowledge_manager.py` | Geometric knowledge storage |
| `truthspace_lcm/core/code_generator.py` | Python code generation |
| `truthspace_lcm/core/bash_generator.py` | Bash command generation |
| `truthspace_lcm/core/task_planner.py` | Multi-step task decomposition |
| `truthspace_lcm/core/executor.py` | Safe code execution |
| `truthspace_lcm/core/python_knowledge_builder.py` | Python knowledge entries |
| `truthspace_lcm/core/bash_knowledge_builder.py` | Bash knowledge entries |
| `truthspace_lcm/knowledge_store/` | Persisted knowledge (JSON) |
| `truthspace_lcm_cli.py` | Command-line interface |

### Key Classes

```python
from truthspace_lcm import (
    KnowledgeManager,    # Store and query knowledge
    KnowledgeDomain,     # Domain enum (PROGRAMMING, HISTORY, etc.)
    KnowledgeEntry,      # Single knowledge item
    CodeGenerator,       # Natural language → Python
    BashGenerator,       # Natural language → Bash
    TaskPlanner,         # Complex task decomposition
    TaskPlan,            # Plan with steps
    TaskStep,            # Single step in plan
    StepType,            # PYTHON or BASH
    StepStatus,          # PENDING, COMPLETED, FAILED, etc.
    CodeExecutor,        # Safe execution
    ExecutionResult,     # Execution output
    ExecutionStatus,     # SUCCESS, FAILED, TIMEOUT, etc.
)
```

### Common Operations

```python
# Plan and execute a task
planner = TaskPlanner()
plan = planner.plan("create a python project called myapp")
plan = planner.execute_plan(plan, dry_run=False)

# Generate Python code only
generator = CodeGenerator()
result = generator.generate("read a json file called config.json")
print(result.code)

# Generate Bash command only
bash_gen = BashGenerator()
result = bash_gen.generate("list all .py files")
print(result.command)

# Query knowledge directly
manager = KnowledgeManager()
results = manager.query(["http", "request", "get"])
for similarity, entry in results:
    print(f"{entry.name}: {similarity:.3f}")
```

---

## Summary

TruthSpace LCM is an experiment in **interpretable, geometric knowledge representation** for code generation. Instead of training a neural network on code, we:

1. **Encode knowledge geometrically** using mathematical constants
2. **Isolate domains** using orthogonal subspaces
3. **Retrieve by similarity** using cosine distance
4. **Generate from templates** for reliability
5. **Execute safely** with validation

The result is a system that's:
- **Transparent**: We know exactly what it knows
- **Controllable**: We can add/remove knowledge precisely
- **Reliable**: No hallucinations, predictable output
- **Fast**: No GPU required, runs on any machine

The tradeoff is flexibility - it can only do what's in its knowledge base. But for the supported use cases, it's rock solid.

---

*Last updated: December 2024*
*Version: 0.1.0*
