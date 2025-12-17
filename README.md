# TruthSpace LCM

**Hypergeometric Language-Code Model** - A natural language to code system using φ-MAX geometric encoding in 12D truth space.

## Philosophy

> *"AI is fundamentally a geometric encoder-decoder. Meaning lives at intersection points in truth space, not in words."*

This system demonstrates that **pure geometry can replace trained neural networks** for semantic resolution. No training data. No backpropagation. Just mathematics.

## Key Concepts

### φ-MAX Encoding
- **φ^level** for each semantic primitive (golden ratio ≈ 1.618)
- **MAX per dimension** prevents synonym over-counting
- **Sierpinski property**: Overlapping activations don't stack

### 12D Truth Space
- **Dims 0-3**: Actions (CREATE, READ, DELETE, COPY, SEARCH, etc.)
- **Dims 4-7**: Domains (PROCESS, NETWORK, FILE, STORAGE, etc.)
- **Dims 8-11**: Relations (INTO, FROM, BEFORE, AFTER, etc.)

### φ-Weighted Distance
- Actions weighted by φ² (most important)
- Domains weighted by 1
- Relations weighted by φ⁻² (least important)

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

### Interactive Mode

```bash
python run.py
```

```
============================================================
TruthSpace LCM - Hypergeometric Resolution
============================================================

Enter natural language requests (type 'quit' to exit)

>>> show disk space

Query: "show disk space"
----------------------------------------
Command: df
Match: read storage (similarity: 1.00)

Execute? (y/N): y

Output:
----------------------------------------
Filesystem     1K-blocks      Used Available Use% Mounted on
/dev/sda1      102400000  45000000  57400000  44% /
```

### Python API

```python
from truthspace_lcm import TruthSpace

ts = TruthSpace()

# Resolve natural language to command
output, entry, similarity = ts.resolve("list files in directory")
print(output)  # ls

# Explain resolution
print(ts.explain("show disk space"))

# Add new knowledge
ts.store("kubectl apply", "deploy application")
```

### Demo

```bash
python scripts/demo.py
```

## Architecture

```
truthspace_lcm/
├── __init__.py          # Package exports
└── core/
    ├── __init__.py      # Core exports
    └── truthspace.py    # The entire system (~400 lines)

That's it. One file. Pure geometry.
```

## How It Works

1. **Encode**: Query → 12D position vector using φ-MAX
2. **Distance**: φ-weighted Euclidean distance to all knowledge
3. **Match**: Return nearest knowledge entry

```python
# Encoding "show disk space"
# "show" → READ (dim 1, level 0) → 1.0
# "disk" → STORAGE (dim 5, level 3) → φ³ ≈ 4.24
# "space" → STORAGE (dim 5, level 3) → MAX(4.24, 4.24) = 4.24

# Position: [0, 1.0, 0, 0, 0, 4.24, 0, 0, 0, 0, 0, 0]
# Matches "read storage" (df) with similarity 1.0
```

## Why This Works

The golden ratio φ appears naturally in:
- Fibonacci sequences
- Optimal packing problems
- Self-similar fractals (Sierpinski)
- Dimensional hierarchies

By using φ for level separation and MAX for overlap handling, we achieve the **Sierpinski property**: overlapping semantic activations confirm the same region rather than stacking.

This is what trained LLMs learn implicitly. We encode it explicitly.

## Design Documents

See `design_considerations/` for the research journey:
- `012_geometric_overlap_handling.md` - φ-MAX encoding discovery
- `010_phi_dimensional_navigation.md` - φ-based weighting
- `009_projection_weighting.md` - Block weight optimization

## License

MIT
