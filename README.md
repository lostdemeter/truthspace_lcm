# TruthSpace LCM

**Language-Code Model** - A minimal natural language to code system using φ-based geometric knowledge encoding.

## Philosophy

> *"Code is just a geometric interpreter. All logic lives in TruthSpace as knowledge."*

This system is designed with a **minimal code footprint** and **maximum knowledge-space usage**. The bootstrap code (~2,400 lines) is irreducible - everything else is knowledge that can be added, modified, or reset without changing code.

## Features

- **Minimal Bootstrap** - Only ~2,400 lines of irreducible code
- **φ-Based Semantic Encoding** - Golden ratio anchored primitives for geometric positioning
- **Knowledge-First Architecture** - Primitives, intents, and commands are all knowledge entries
- **Fail Fast** - No fallbacks; query succeeds or raises `KnowledgeGapError`
- **Auto-Learning** - System can acquire new knowledge from man pages and pydoc
- **SQLite Backend** - Fast, ACID-compliant knowledge storage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/truthspace-lcm.git
cd truthspace-lcm

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Seed the knowledge base
python scripts/seed_truthspace.py
```

## Quick Start

### Interactive Mode

```bash
python run.py
```

```
============================================================
TruthSpace LCM - Natural Language to Code
============================================================

Enter natural language requests (type 'quit' to exit)

>>> write a hello world python program

Request: "write a hello world python program"
----------------------------------------
Generated (python):
  print("Hello, World!")

Execute? (y/N): y

Output:
----------------------------------------
Hello, World!
```

### Single Query Mode

```bash
python run.py "list files in directory"
```

### Python API

```python
from truthspace_lcm import TruthSpace, Resolver

# Initialize
ts = TruthSpace()
resolver = Resolver(ts, auto_learn=True)

# Resolve natural language to code
result = resolver.resolve("list files in directory")
print(result.output)       # ls -la
print(result.output_type)  # OutputType.BASH

# Resolve and execute
resolution, exec_result = resolver.resolve_and_execute("show current directory")
print(exec_result.stdout)  # /home/user/...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      run.py                                 │
│                 Interactive Interface                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Resolver                                │
│   - NL → Knowledge → Output (no fallbacks)                  │
│   - Fail fast: KnowledgeGapError if no match                │
│   - Optional auto-learning on gaps                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TruthSpace                               │
│   - Unified knowledge storage + query                       │
│   - Everything is a KnowledgeEntry                          │
│   - Primitives, intents, commands all in one place          │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│     φ-Encoder       │     │   SQLite Database   │
│  (semantic math)    │     │   (persistence)     │
└─────────────────────┘     └─────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Executor                               │
│   - Safe bash/python execution                              │
│   - Timeout protection                                      │
│   - Output capture                                          │
└─────────────────────────────────────────────────────────────┘
```

### Core Components (~2,400 lines total)

| File | Lines | Purpose |
|------|-------|---------|
| `truthspace.py` | 647 | Unified knowledge storage + query |
| `resolver.py` | 321 | NL → Knowledge → Output |
| `ingestor.py` | 497 | Knowledge acquisition |
| `encoder.py` | 365 | φ-based semantic encoding |
| `executor.py` | 513 | Code execution |

## Knowledge Base

The system starts with minimal seed knowledge (~40 entries):

| Type | Count | Examples |
|------|-------|----------|
| **Primitives** | 19 | CREATE, READ, WRITE, FILE, NETWORK, RECURSIVE |
| **Intents** | 17 | list_files → `ls -la`, hello_world → `print("Hello, World!")` |
| **Commands** | 5 | ls, cd, cat, grep, find |

Knowledge can be expanded by:
1. **Seeding** - Run `python scripts/seed_truthspace.py` to reset to minimal
2. **Auto-learning** - System learns from man pages when it encounters gaps
3. **Manual addition** - Use `TruthSpace.store()` to add custom knowledge

## Project Structure

```
truthspace-lcm/
├── truthspace_lcm/           # Main package
│   ├── __init__.py
│   ├── truthspace.db         # SQLite knowledge database
│   └── core/                 # Core modules (~2,400 lines)
│       ├── truthspace.py         # Unified knowledge storage + query
│       ├── resolver.py           # NL → Knowledge → Output
│       ├── ingestor.py           # Knowledge acquisition
│       ├── encoder.py            # φ-based semantic encoding
│       └── executor.py           # Safe code execution
├── scripts/
│   └── seed_truthspace.py    # Reset knowledge to minimal state
├── run.py                    # Interactive runner
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## How It Works

### φ-Based Semantic Encoding

Knowledge is encoded using the **φ-encoder**, which maps natural language to geometric positions using semantic primitives:

**Primitive Types:**
- **ACTIONS** (dims 0-3) - CREATE, DESTROY, READ, WRITE, MOVE, CONNECT, EXECUTE, SEARCH
- **DOMAINS** (dims 4-6) - FILE, PROCESS, NETWORK, SYSTEM, USER, DATA
- **MODIFIERS** (dim 7) - ALL, RECURSIVE, VERBOSE, FORCE

**Position Computation:**
```
position = Σ (primitive_position × relevance_weight)
primitive_position[dim] = φ^level  (where φ = 1.618...)
```

This allows semantic similarity search without neural networks.

### Resolution Pipeline

```
"list files in directory"
        │
        ▼
   ┌─────────────┐
   │  φ-Encoder  │  → position = [0.62, 0.38, 0, 0, 0.85, 0, 0, 0]
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │ TruthSpace  │  → query(position) → best match: list_files intent
   │   Query     │
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │  Extract    │  → "ls -la"
   │   Output    │
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │  Execute    │  → stdout: "total 48\ndrwxr-xr-x..."
   └─────────────┘
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the TruthSpace geometric encoding framework
- Inspired by the idea that knowledge can be represented geometrically
- Part of the Holographer's Workbench project
