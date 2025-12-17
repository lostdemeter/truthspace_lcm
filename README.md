# TruthSpace LCM

**Geometric Language-Code Model** - A natural language to code system using hierarchical geometric embeddings. No training required.

## Philosophy

> *"AI is fundamentally a geometric encoder-decoder. Meaning lives at intersection points in truth space, not in words."*

This system demonstrates that **pure geometry can replace trained neural networks** for semantic resolution. No training data. No backpropagation. Just mathematics.

## Features

- **128-dimensional hierarchical embeddings** - 7 stacked geometric layers
- **Intent detection** - Automatically detects bash commands vs chat
- **Bash execution** - Execute commands with safety checks
- **No external dependencies** - No LLM API calls required
- **Interpretable** - Every dimension has semantic meaning

## Installation

```bash
git clone https://github.com/yourusername/truthspace-lcm.git
cd truthspace-lcm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Interactive Chat Mode

```bash
python run.py
```

```
============================================================
  TruthSpace LCM Chat
  Geometric knowledge resolution with bash execution
============================================================

Type 'help' for commands, 'exit' to quit.

You: list all files

LCM: I'll run: $ ls -la
     Execute? [y/N]: y

✓ $ ls -la
total 108
drwxrwxr-x 12 user user  4096 Dec 17 10:42 .
...

You: hello

LCM: I'm here to help with bash commands and answer questions.

You: show disk space

LCM: I'll run: $ df -h
     Execute? [y/N]: y

✓ $ df -h
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       1.8T  1.2T  606G  66% /
```

### Single Query Mode

```bash
python run.py "list files"
```

### Python API

```python
from truthspace_lcm.core import StackedLCM
from truthspace_lcm.chat import LCMChat

# Direct embedding usage
lcm = StackedLCM()
lcm.ingest("ls -la", "list files directory terminal")
lcm.ingest("df -h", "disk space usage storage")

content, similarity, cluster = lcm.resolve("show all files")
print(content)  # ls -la

# Chat interface
chat = LCMChat(safe_mode=True)
response = chat.process("list all files")
print(response)  # Prompts for confirmation, then executes
```

### Legacy Mode (12D TruthSpace)

```bash
python run.py --legacy
```

## Architecture

```
truthspace_lcm/
├── __init__.py              # Package exports
├── chat.py                  # Chat interface with bash execution
└── core/
    ├── __init__.py          # Core exports (StackedLCM, TruthSpace)
    ├── stacked_lcm.py       # PRIMARY: 128D hierarchical embeddings
    ├── truthspace.py        # Legacy: 12D φ-MAX encoding
    └── knowledge_generator.py  # LLM-based knowledge generation
```

### StackedLCM Layers (128D total)

| Layer | Dimensions | Purpose |
|-------|------------|---------|
| Morphological | 16 | Word structure (prefixes, suffixes, n-grams) |
| Lexical | 32 | Primitive activation (φ-MAX encoding) |
| Syntactic | 16 | Bigram pattern detection |
| Compositional | 24 | Domain signature detection |
| Disambiguation | 16 | Context-dependent meaning |
| Contextual | 16 | Co-occurrence statistics |
| Global | 8 | Prototype distances |

## How It Works

### Hierarchical Encoding

Each layer captures meaning at a different scale:

1. **Morphological**: "cooking" and "baking" share "-ing" suffix
2. **Lexical**: "chop" activates CUT primitive, "file" activates FILE primitive
3. **Syntactic**: "the file" vs "the vegetables" detected via bigrams
4. **Compositional**: Cooking domain = FOOD + HEAT + CUT patterns
5. **Disambiguation**: "cut the file" → tech, "cut the vegetables" → cooking
6. **Contextual**: Learns co-occurrence from ingested knowledge
7. **Global**: Distance to emergent domain prototypes

### Intent Detection

The chat interface detects intent geometrically:

```python
# "list files" → BASH intent (keywords: list, files)
# "hello" → CHAT intent (greeting patterns)
# "how do I find a file?" → QUESTION intent
# "ls -la" → BASH intent (direct command pattern)
```

### Disambiguation

Same word, different context:

```
"cut the file" vs "cut the vegetables"
  Similarity: 0.42 (correctly low - different domains)

"search for recipes" vs "search for files"  
  Similarity: 0.41 (correctly low - different domains)
```

## Key Concepts

### φ-MAX Encoding
- **φ^level** for each semantic primitive (golden ratio ≈ 1.618)
- **MAX per dimension** prevents synonym over-counting
- **Sierpinski property**: Overlapping activations don't stack

### Layer Weighting
- Morphological: 0.3 (reduced - word length shouldn't dominate)
- Lexical: 1.2 (semantic primitives)
- Disambiguation: 2.0 (high - context is critical)

## Testing

```bash
# Run all tests
python tests/test_stacked_lcm.py

# Run legacy tests
python tests/test_truthspace.py
```

## Design Documents

See `design_considerations/` for the research journey:
- `017_stacked_geometric_embeddings.md` - Hierarchical architecture
- `018_stacked_lcm_analysis.md` - What works and what doesn't
- `016_truly_dynamic_geometric_lcm.md` - Self-organizing domains
- `015_dynamic_geometric_lcm.md` - Dynamic primitive discovery

## License

MIT
