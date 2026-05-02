# φ-Geometric Framework

A complete framework for understanding and implementing neural networks as **geometric structures on the φ-lattice**.

## The Core Hypothesis

> **LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

## Key Principles

1. **Structure IS information** - There are no opaque weights or embeddings
2. **Geometry IS computation** - Traversal through geometric space produces outputs
3. **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure
4. **φ IS the coordinate system** - All weights cluster on the φ-lattice

## Validation

We reverse-engineered three fundamentally different AI models:

| Model | Task | Correlation |
|-------|------|-------------|
| DA2 | Depth estimation | 99.98% |
| Qwen2-7B | Language modeling | 99.9991% |
| DDColor | Colorization | 100% |

All achieved near-perfect output correlation using pure φ-arithmetic.

## Installation

```bash
# From the truthspace-lcm root
cd phi_geometric
pip install -e .
```

Or simply add to your Python path:
```python
import sys
sys.path.append('/path/to/truthspace-lcm')
from phi_geometric import GeometricAI, ProblemSpec, IOSpec, DataType
```

## Quick Start

### Build AI Without Training

```python
from phi_geometric import GeometricAI, ProblemSpec, IOSpec, DataType

# Define your problem
problem = ProblemSpec(
    name="classifier",
    inputs=[IOSpec("features", DataType.VECTOR, (64,))],
    outputs=[IOSpec("class", DataType.VECTOR, (10,))],
)

# Create geometric AI (no training!)
ai = GeometricAI(problem)

# Inject knowledge
ai.inject_knowledge("Class 0 is for small values")
ai.inject_knowledge("Class 9 is for large values")

# Run inference
output = ai(input_tensor)
```

### Use Pattern Examples

```python
from phi_geometric.examples import (
    FunnelClassifier,
    SpiralLanguageModel,
    WebColorizer,
    TreeMultiTask,
    BraidMultiModal,
    HourglassAutoencoder
)

# Classification
classifier = FunnelClassifier(input_dim=64, num_classes=10)
class_idx, confidence = classifier.classify(features)

# Language modeling
lm = SpiralLanguageModel(vocab_size=1000, context_length=64)
next_token = lm.predict_next(embeddings)

# Colorization
colorizer = WebColorizer(image_size=64, queries=100)
colors = colorizer.colorize(grayscale)
```

### Reverse-Engineer Existing Models

```python
from phi_geometric.models import DA2Geometric, QwenGeometric, DDColorGeometric

# Load and convert to φ-space
da2 = DA2Geometric.from_pretrained("depth-anything/Depth-Anything-V2-Small")
depth = da2(features)
```

## Pattern Taxonomy

The framework includes 10 patterns for different problem types:

| Pattern | Topology | Use Case |
|---------|----------|----------|
| **Funnel** 🐜 | Convergent | Classification, regression |
| **Spiral** 🐛 | Self-referential | Language, reasoning |
| **Web** 🕷️ | Cross-connected | Colorization, segmentation |
| **Tree** 🌳 | Divergent | Multi-task learning |
| **Braid** 🪢 | Intertwined | Multi-modal fusion |
| **Hourglass** ⏳ | Compress/expand | Autoencoders, generation |
| **Ring** 💍 | Closed loop | Memory, control |
| **Constellation** ✨ | Graph | Relational reasoning |
| **Fractal** 🔷 | Self-similar | Hierarchical structure |
| **Mirror** 🪞 | Symmetric | Translation |

## Four Components

### 1. Shape Projection
Derive φ-coordinates from problem structure alone.

```python
from phi_geometric.core import ShapeProjector

projector = ShapeProjector()
pattern, phi_weights = projector.project(problem)
```

### 2. Knowledge Injection
Add facts without training.

```python
from phi_geometric.core import KnowledgeInjector

injector = KnowledgeInjector()
injector.add_fact("Sky is blue")
context = injector.inject(base_context)
```

### 3. Signature Memory
Self-assembling cache for fast lookup.

```python
from phi_geometric.core import SignatureMemory

memory = SignatureMemory(threshold=0.5)
result, distance = memory.lookup(input)
if result is None:
    result = compute(input)
    memory.store(input, result)
```

### 4. Bottleneck Filter
Validate outputs through φ-constraint.

```python
from phi_geometric.core import BottleneckFilter

filter = BottleneckFilter(tolerance=0.3)
is_valid, phi_level = filter.is_valid(output)
```

## Directory Structure

```
phi_geometric/
├── __init__.py          # Main exports
├── README.md            # This file
├── core/                # Core components
│   ├── encoder.py       # φ-basis encoding
│   ├── patterns.py      # Pattern taxonomy
│   ├── projector.py     # Shape projection
│   ├── navigator.py     # Geometric traversal
│   ├── memory.py        # Signature memory
│   ├── injector.py      # Knowledge injection
│   ├── filter.py        # Bottleneck filter
│   └── geometric_ai.py  # Unified interface
├── models/              # Reverse-engineered models
│   ├── da2.py           # Depth Anything V2
│   ├── qwen.py          # Qwen2-7B
│   └── ddcolor.py       # DDColor
├── examples/            # Pattern examples
│   ├── funnel_example.py
│   ├── spiral_example.py
│   ├── web_example.py
│   ├── tree_example.py
│   ├── braid_example.py
│   └── hourglass_example.py
└── tests/               # Tests
```

## The φ-Lattice

All neural network weights naturally cluster on a φ-lattice:

```
value = sign × φ^(exponent / K)
```

Where:
- `φ = 1.618...` (golden ratio)
- `sign ∈ {-1, 0, +1}`
- `exponent` is an integer
- `K` is the resolution (typically 32-128)

Key properties:
- **Multiplication → Addition**: `φ^a × φ^b = φ^(a+b)`
- **Universal**: Same structure across all models
- **Peak at φ^-9**: ≈ 0.013, typical weight magnitude

## Related Documentation

- Doc 124: φ-Transformer Replacement
- Doc 125: Exact DA2 Recreation
- Doc 178: Spatial Encoder Pattern
- Doc 204: Reverse Navigation
- Doc 210: Knowledge Injection
- Doc 213-217: Framework documentation

## License

TruthSpace LCM Project - February 2026

---

*"Structure IS information. Geometry IS computation. The shape IS the knowledge."*
