# TruthSpace LCM

**Language-Code Model** - A natural language to code system that uses geometric knowledge encoding to translate human requests into executable Python and Bash code.

## Features

- **Natural Language Understanding** - Describe what you want in plain English
- **Multi-Step Task Planning** - Complex tasks are automatically decomposed into steps
- **Python Code Generation** - Generate complete, runnable Python scripts
- **Bash Command Generation** - Generate shell commands for file/system operations
- **Safe Execution** - Code runs in isolated environments with timeout protection
- **Output Validation** - Verify results match expectations
- **Error Diagnosis** - Intelligent error messages with fix suggestions

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
│                   Knowledge Base                            │
│            77 Python + 74 Bash = 151 entries                │
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

The system includes 151 knowledge entries:

| Category | Entries | Examples |
|----------|---------|----------|
| Python Core | 20 | print, input, len, range, open |
| Python Libraries | 30 | requests, json, os, sys |
| Python Patterns | 27 | file reading, JSON parsing, loops |
| Bash Commands | 50 | mkdir, cp, mv, rm, grep, find |
| Bash Patterns | 24 | backup, search, compress |

## Project Structure

```
truthspace-lcm/
├── truthspace_lcm/           # Main package
│   ├── __init__.py
│   ├── core/                 # Core modules
│   │   ├── __init__.py
│   │   ├── knowledge_manager.py    # Geometric knowledge storage
│   │   ├── code_generator.py       # Python code generation
│   │   ├── bash_generator.py       # Bash command generation
│   │   ├── task_planner.py         # Multi-step task planning
│   │   ├── executor.py             # Safe code execution
│   │   ├── python_knowledge_builder.py
│   │   └── bash_knowledge_builder.py
│   └── knowledge_store/      # Persistent knowledge (JSON)
├── tests/                    # Test suite
├── examples/                 # Example scripts
├── docs/                     # Documentation
├── truthspace_lcm_cli.py     # Command line interface
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## How It Works

### Geometric Knowledge Encoding

Knowledge is encoded using semantic dimensions based on mathematical constants:
- **φ (phi)** - Identity/naming
- **π (pi)** - Spatial relationships
- **e** - Temporal aspects
- **γ (gamma)** - Causal relationships

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
