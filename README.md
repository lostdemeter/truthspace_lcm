# TruthSpace LCM

**Language-Code Model** - A natural language to code system that uses φ-based geometric knowledge encoding to translate human requests into executable Python and Bash code.

## Features

- **φ-Based Semantic Encoding** - Uses the golden ratio (φ) as the fundamental anchor for positioning knowledge in geometric space
- **Natural Language Understanding** - Describe what you want in plain English
- **Multi-Step Task Planning** - Complex tasks are automatically decomposed into steps
- **Python Code Generation** - Generate complete, runnable Python scripts
- **Bash Command Generation** - Generate shell commands for file/system operations
- **SQLite Knowledge Database** - Fast, ACID-compliant storage with optimized vector queries
- **Safe Execution** - Code runs in isolated environments with timeout protection
- **Autonomous Learning** - System can learn new commands and expand its knowledge base

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/truthspace-lcm.git
cd truthspace-lcm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Command Line

```bash
# Execute a task
python truthspace_lcm_cli.py "create a python project called myapp"

# Dry run (generate code without executing)
python truthspace_lcm_cli.py --dry-run "backup the project folder"

# Show generated code
python truthspace_lcm_cli.py --show-code "list all files"

# Interactive mode
python truthspace_lcm_cli.py --interactive
```

### Python API

```python
from truthspace_lcm import TaskPlanner, StepStatus

# Create planner
planner = TaskPlanner()

# Plan and execute a task
plan = planner.plan("create a python project called myapp")
plan = planner.execute_plan(plan, dry_run=False)

# Check results
if plan.status == StepStatus.COMPLETED:
    print("Task completed successfully!")
    for step in plan.steps:
        print(f"  {step.id}. {step.description}")
        print(f"     Code: {step.generated_code}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TruthSpace LCM CLI                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Task Planner                            │
│   - Decomposes complex requests into steps                  │
│   - Tracks dependencies between steps                       │
│   - Orchestrates execution                                  │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Python Generator  │     │    Bash Generator   │
└─────────────────────┘     └─────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    φ-Encoder                                │
│   - Semantic primitives (ACTIONS, DOMAINS, MODIFIERS)       │
│   - Golden ratio (φ) based position computation             │
│   - Bidirectional mapping (NL ↔ Code)                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 SQLite Knowledge Database                   │
│   - 160+ entries with φ-encoded positions                   │
│   - Indexed queries with keyword boosting                   │
│   - ACID transactions for safe updates                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Executor                               │
│   - Safe execution in isolated environment                  │
│   - Output capture and validation                           │
│   - Error diagnosis and suggestions                         │
└─────────────────────────────────────────────────────────────┘
```

## Supported Tasks

### Project Setup
```bash
python truthspace_lcm_cli.py "create a python project called webapp"
```
Creates: `webapp/`, `webapp/src/`, `__init__.py`, `main.py`, `README.md`

### File Operations
```bash
python truthspace_lcm_cli.py "create a directory called data"
python truthspace_lcm_cli.py "create a file called config.json"
python truthspace_lcm_cli.py "list all files in the current directory"
python truthspace_lcm_cli.py "copy file.txt to backup"
```

### Web Operations
```bash
python truthspace_lcm_cli.py "fetch https://api.github.com and parse JSON"
python truthspace_lcm_cli.py "scrape titles from https://example.com"
```

### Multi-Step Tasks
```bash
python truthspace_lcm_cli.py "create a directory called data, then create config.json inside it"
python truthspace_lcm_cli.py "backup the project folder"
```

## Knowledge Base

The system includes 160+ knowledge entries stored in a SQLite database with φ-encoded positions:

| Category | Examples |
|----------|----------|
| Python Core | print, input, len, range, open |
| Python Libraries | requests, json, os, sys |
| Python Patterns | file reading, JSON parsing, loops |
| Bash Commands | mkdir, cp, mv, rm, grep, find |
| Bash Patterns | backup, search, compress |
| Learned Commands | uptime, ifconfig, ping (autonomously acquired) |

## Project Structure

```
truthspace-lcm/
├── truthspace_lcm/           # Main package
│   ├── __init__.py
│   ├── knowledge.db          # SQLite knowledge database
│   └── core/                 # Core modules
│       ├── phi_encoder.py        # φ-based semantic encoder
│       ├── knowledge_db.py       # SQLite database backend
│       ├── knowledge_manager.py  # Knowledge CRUD operations
│       ├── code_generator.py     # Python code generation
│       ├── bash_generator.py     # Bash command generation
│       ├── task_planner.py       # Multi-step task planning
│       ├── executor.py           # Safe code execution
│       ├── intent_manager.py     # Intent pattern management
│       └── engine.py             # TruthSpace engine
├── tests/                    # Test suite
├── examples/                 # Example scripts
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── truthspace_lcm_cli.py     # Command line interface
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## How It Works

### φ-Based Semantic Encoding

Knowledge is encoded using the **φ-encoder**, which maps natural language to geometric positions using semantic primitives:

**Primitive Types:**
- **ACTIONS** - CREATE, DESTROY, READ, WRITE, MOVE, CONNECT, EXECUTE, TRANSFORM, SEARCH, COMPARE
- **DOMAINS** - FILE, PROCESS, NETWORK, SYSTEM, USER, DATA, TEXT
- **MODIFIERS** - ALL, RECURSIVE, VERBOSE, FORCE, QUIET

**Position Computation:**
```
position = Σ (primitive_position × relevance_weight)
```

Each primitive has a fixed position derived from the golden ratio (φ = 1.618...), ensuring:
- Semantically similar queries → nearby positions
- Domain isolation prevents cross-domain interference
- Deterministic, reproducible encodings

This allows semantic similarity search without neural networks.

### Task Decomposition

Complex requests are broken into atomic steps:
1. Detect task type (project setup, file operation, web scraping, etc.)
2. Decompose into steps with dependencies
3. Generate code for each step
4. Execute in dependency order
5. Validate results

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

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
