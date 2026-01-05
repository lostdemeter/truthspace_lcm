# HyperMapping

A bidirectional hyperdimensional data structure for geometric computation.

## Installation

```python
# From the truthspace-lcm root
from hypermapping import HyperMapping, TextEncoder, ImageEncoder
```

## Quick Start

```python
from hypermapping import HyperMapping, TextEncoder

# Create encoder and learn from corpus
encoder = TextEncoder(dims=12)
encoder.learn(["list files", "show files", "delete file", "kill process"])
encoder.add_synonyms([
    ["list", "show", "display"],
    ["delete", "remove", "erase"],
])

# Create mapping space
space = HyperMapping(dims=12, encoder=encoder)

# Add mappings
space.map("list files", "ls")
space.map("show files", "ls")
space.map("delete file", "rm")
space.map("kill process", "kill")

# Query forward (input → output)
result = space.forward("display files")
print(result.output)  # "ls"
print(result.similarity)  # 0.85

# Query backward (output → inputs)
results = space.backward("ls")
for r in results:
    print(f"{r.input} → {r.output}")

# Pipeline multiple spaces
pipeline = intent_space | command_space
result = pipeline("file operations")
```

## Core Concepts

### HyperMapping

A bidirectional mapping between inputs and outputs through geometric space.

```python
space = HyperMapping(dims=8, encoder=encoder, name="my_space")

# Add mappings
space.map(input, output)

# Query
result = space.forward(input)      # input → output
results = space.backward(output)   # output → inputs
results = space.query(value, k=5)  # Find nearest mappings

# Learning
space.feedback(input, output, success=True)  # Reinforce
space.attract(mapping1, mapping2)            # Move closer
space.repel(mapping1, mapping2)              # Move apart

# Persistence
space.save("path.json")
space = HyperMapping.load("path.json", encoder=encoder)
```

### Encoders

Encoders convert domain values to positions in hyperdimensional space.

| Encoder | Use Case | Description |
|---------|----------|-------------|
| `HashEncoder` | Key-value storage | Deterministic hash positions |
| `TextEncoder` | NLP, semantic search | Word co-occurrence positions |
| `NumericEncoder` | ML, regression | Non-linear feature expansion |
| `ImageEncoder` | Computer vision | Histogram + spatial features |
| `CategoricalEncoder` | Classification | Category positions |
| `CompositeEncoder` | Multi-modal | Combines multiple encoders |

### HyperPipeline

Chain multiple spaces together:

```python
pipeline = space1 | space2 | space3
result = pipeline(input)  # Flows through all spaces
```

## Comparison to Neural Networks

HyperMapping can solve the same problems as neural networks:

| Task | HyperMapping | Neural Network |
|------|--------------|----------------|
| XOR (non-linear) | 100% | 100% |
| Image Classification | 100% | ~99% |
| Sentiment Analysis | 71% | ~85% |
| Function Approximation | 15% | ~95% |

Key differences:
- **No gradient descent** - Learning is geometric (attract/repel)
- **Explicit positions** - Interpretable, not opaque weights
- **No training epochs** - Add mappings directly

## API Reference

### HyperMapping

```python
class HyperMapping:
    def __init__(dims: int, encoder: Encoder, name: str)
    def map(input, output, position=None, metadata=None) -> Mapping
    def forward(input, k=1) -> MatchResult
    def backward(output, k=5) -> List[MatchResult]
    def query(value, k=5) -> List[MatchResult]
    def feedback(input, output, success: bool, strength=0.1)
    def attract(mapping1, mapping2, strength=0.1)
    def repel(mapping1, mapping2, strength=0.05)
    def save(path: str)
    def load(path: str, encoder: Encoder) -> HyperMapping
```

### Mapping

```python
@dataclass
class Mapping:
    input: Any
    output: Any
    position: np.ndarray
    metadata: Dict[str, Any]
```

### MatchResult

```python
@dataclass
class MatchResult:
    mapping: Mapping
    similarity: float
    
    @property
    def input(self) -> Any
    @property
    def output(self) -> Any
```

## Examples

See `experiments/hypermapping_vs_neural_nets.py` for complete examples of:
- XOR problem
- Image classification
- Sentiment analysis
- Function approximation
- Sequence prediction

## License

GPLv3
