# 215: φ-Space Solver Library Design

## Date: February 5, 2026

## Vision

A generalized library that treats all neural network inference as **geometric navigation through the φ-lattice**. Instead of implementing models, users specify **patterns** and the solver handles the rest.

## Core Insight

All our reverse-engineered models share:
1. **φ-encoding**: `value = sign × φ^(exponent/K)`
2. **Bilinear core**: `A @ B.T` operations
3. **MESH principle**: Pre-compute combined matrices
4. **Pattern topology**: How information flows

A generalized solver abstracts these into reusable primitives.

## Library Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     φ-SPACE SOLVER                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Patterns  │  │  φ-Encoder  │  │    MESH     │         │
│  │   Library   │  │             │  │  Computer   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Navigation Engine                   │       │
│  │  (traverses φ-lattice according to pattern)     │       │
│  └─────────────────────────────────────────────────┘       │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │  Funnel   │    │  Spiral   │    │    Web    │          │
│  │  Solver   │    │  Solver   │    │  Solver   │          │
│  └───────────┘    └───────────┘    └───────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## API Design

### 1. Pattern Specification

```python
from phi_solver import Pattern, PhiSolver

# Define a pattern
funnel = Pattern(
    name="funnel",
    topology="convergent",
    self_reference=False,
    io_ratio="N:1",
    layers=[
        {"type": "linear", "in": 1024, "out": 32}
    ]
)

# Or use a preset
from phi_solver.patterns import Funnel, Spiral, Web

depth_pattern = Funnel(in_dim=1024, out_dim=1)
```

### 2. Loading Existing Weights (Reverse Engineering)

```python
from phi_solver import PhiSolver
from phi_solver.patterns import Spiral

# Load a pretrained model and convert to φ-space
solver = PhiSolver.from_pretrained(
    "Qwen/Qwen2-7B",
    pattern=Spiral(layers=28, heads=28, dim=3584)
)

# Weights are now φ-encoded
print(solver.phi_weights)  # Dict of (sign, exponent) tensors

# Run inference (geometric navigation)
output = solver.navigate(input_tokens)
```

### 3. Creating New Shapes

```python
from phi_solver import PhiSolver
from phi_solver.patterns import Web

# Define a new colorization pattern
colorizer = PhiSolver(
    pattern=Web(
        queries=100,
        feature_scales=3,
        layers=9,
        output_dim=2  # ab channels
    )
)

# Train the shape (learns φ-coordinates)
colorizer.learn(train_data, epochs=100)

# The learned weights are automatically φ-encoded
colorizer.save("my_colorizer.phi")
```

### 4. Composing Patterns

```python
from phi_solver import PhiSolver, compose
from phi_solver.patterns import Funnel, Tree

# Compose patterns: encoder (funnel) + multi-task (tree)
multi_task = compose(
    Funnel(in_dim=1024, out_dim=256),  # Shared encoder
    Tree(in_dim=256, branches=[
        ("depth", 1),
        ("normals", 3),
        ("edges", 1),
        ("segments", 20)
    ])
)

solver = PhiSolver(pattern=multi_task)
```

## Core Components

### 1. PhiEncoder

```python
class PhiEncoder:
    """Encode/decode values in φ-basis."""
    
    def __init__(self, K: int = 32, bias: int = 8192):
        self.K = K
        self.bias = bias
        self.phi_lut = precompute_phi_powers(K, bias)
    
    def encode(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Float tensor → (signs, exponents)"""
        signs = torch.sign(tensor)
        exponents = (self.K * torch.log(torch.abs(tensor)) / LN_PHI + self.bias).round()
        return signs, exponents.long()
    
    def decode(self, signs: Tensor, exponents: Tensor) -> Tensor:
        """(signs, exponents) → Float tensor"""
        return signs * self.phi_lut[exponents]
    
    def multiply(self, a_exp: Tensor, b_exp: Tensor) -> Tensor:
        """φ-multiplication = exponent addition"""
        return a_exp + b_exp - self.bias  # Adjust for bias
```

### 2. MESHComputer

```python
class MESHComputer:
    """Pre-compute combined matrices to eliminate error compounding."""
    
    def compute_mesh(self, W_q: Tensor, W_k: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute MESH = W_q.T @ W_k and encode in φ-basis."""
        mesh = W_q.T @ W_k
        return self.encoder.encode(mesh)
    
    def compute_bilinear(self, W_a: Tensor, W_b: Tensor) -> Tuple[Tensor, Tensor]:
        """Generic bilinear pre-computation."""
        combined = W_a.T @ W_b
        return self.encoder.encode(combined)
```

### 3. NavigationEngine

```python
class NavigationEngine:
    """Traverse the φ-lattice according to a pattern."""
    
    def __init__(self, pattern: Pattern, phi_weights: Dict):
        self.pattern = pattern
        self.weights = phi_weights
    
    def navigate(self, input: Tensor) -> Tensor:
        """Execute geometric navigation."""
        
        # Encode input
        x_signs, x_exps = self.encoder.encode(input)
        
        # Navigate according to pattern topology
        if self.pattern.topology == "convergent":
            return self._navigate_funnel(x_signs, x_exps)
        elif self.pattern.topology == "spiral":
            return self._navigate_spiral(x_signs, x_exps)
        elif self.pattern.topology == "web":
            return self._navigate_web(x_signs, x_exps)
        # ... etc
    
    def _navigate_funnel(self, x_signs, x_exps):
        """Funnel: weighted sum → single output."""
        w_signs, w_exps = self.weights['head']
        
        # φ-dot product: multiply = add exponents, then sum
        prod_exps = x_exps.unsqueeze(-1) + w_exps  # Broadcasting
        prod_signs = x_signs.unsqueeze(-1) * w_signs
        
        # Decode and sum (or use carry-save accumulation)
        products = self.encoder.decode(prod_signs, prod_exps)
        return products.sum(dim=-2)
```

### 4. Pattern Library

```python
# patterns/funnel.py
class Funnel(Pattern):
    """Convergent pattern: many inputs → one output."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(
            name="funnel",
            topology="convergent",
            self_reference=False,
            io_ratio="N:1"
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def build_graph(self):
        return [LinearNode(self.in_dim, self.out_dim)]


# patterns/spiral.py
class Spiral(Pattern):
    """Self-referential pattern: deep with attention."""
    
    def __init__(self, layers: int, heads: int, dim: int):
        super().__init__(
            name="spiral",
            topology="spiral",
            self_reference=True,
            io_ratio="1:1"
        )
        self.layers = layers
        self.heads = heads
        self.dim = dim
    
    def build_graph(self):
        nodes = []
        for i in range(self.layers):
            nodes.append(SelfAttentionNode(self.heads, self.dim))
            nodes.append(FFNNode(self.dim, self.dim * 4))
        return nodes


# patterns/web.py
class Web(Pattern):
    """Cross-connected pattern: queries attend to features."""
    
    def __init__(self, queries: int, feature_scales: int, layers: int, output_dim: int):
        super().__init__(
            name="web",
            topology="web",
            self_reference="partial",
            io_ratio="N:M"
        )
        self.queries = queries
        self.feature_scales = feature_scales
        self.layers = layers
        self.output_dim = output_dim
    
    def build_graph(self):
        nodes = []
        for i in range(self.layers):
            scale = i % self.feature_scales
            nodes.append(CrossAttentionNode(self.queries, scale))
            nodes.append(SelfAttentionNode(self.queries))
            nodes.append(FFNNode(self.queries))
        nodes.append(OutputNode(self.output_dim))
        return nodes
```

## Usage Examples

### Example 1: Reverse-Engineer Any Model

```python
from phi_solver import PhiSolver

# Load any HuggingFace model
solver = PhiSolver.from_pretrained("piddnad/ddcolor_paper_tiny")

# Automatically detects pattern (Web) and encodes weights
print(f"Pattern: {solver.pattern.name}")  # "web"
print(f"φ-encoding accuracy: {solver.encoding_accuracy}")  # 99.999%

# Run inference
colors = solver.navigate(grayscale_image)
```

### Example 2: Design a New Pattern

```python
from phi_solver import PhiSolver, Pattern
from phi_solver.nodes import CrossAttention, SelfAttention, FFN

# Custom pattern: Braid (two intertwined streams)
braid = Pattern(
    name="braid",
    topology="braid",
    streams=2,
    cross_every=2  # Streams cross every 2 layers
)

# Build the graph
braid.add_stream("vision", [
    CrossAttention(to="language"),
    SelfAttention(),
    FFN()
] * 6)

braid.add_stream("language", [
    CrossAttention(to="vision"),
    SelfAttention(),
    FFN()
] * 6)

# Create solver
multimodal = PhiSolver(pattern=braid)
multimodal.learn(vision_language_data)
```

### Example 3: Compose Patterns

```python
from phi_solver import compose
from phi_solver.patterns import Hourglass, Constellation

# Scene understanding: compress image, then reason over objects
scene_reasoner = compose(
    Hourglass(in_dim=512, bottleneck=64),  # Compress
    Constellation(nodes="detected_objects")  # Reason
)

solver = PhiSolver(pattern=scene_reasoner)
```

## Benefits

### 1. Unified Interface
- Same API for depth, language, vision, multimodal
- Pattern selection replaces architecture design

### 2. Automatic φ-Optimization
- Weights automatically encoded in φ-basis
- MESH pre-computation handled automatically
- Integer arithmetic where possible

### 3. Composability
- Patterns can be combined
- Reuse components across tasks

### 4. Hardware Portability
- Same φ-encoded weights work on GPU, FPGA, ASIC
- Navigation engine can be hardware-accelerated

## Implementation Roadmap

### Phase 1: Core Primitives
- [ ] PhiEncoder (encode/decode)
- [ ] MESHComputer (pre-computation)
- [ ] Basic patterns (Funnel, Spiral, Web)

### Phase 2: Reverse Engineering
- [ ] from_pretrained() for common architectures
- [ ] Automatic pattern detection
- [ ] Weight extraction and encoding

### Phase 3: Pattern Library
- [ ] All 10 patterns from taxonomy
- [ ] Pattern composition
- [ ] Custom pattern DSL

### Phase 4: Hardware Backends
- [ ] GPU (cuBLAS + φ-decode)
- [ ] FPGA (true integer arithmetic)
- [ ] ASIC design specs

## Conclusion

The φ-Space Solver library abstracts neural network inference into:
1. **Pattern selection** (what shape to navigate)
2. **Weight loading** (the specific shape on the φ-lattice)
3. **Navigation** (geometric traversal)

This unifies all our reverse-engineering work into a single, reusable framework.

---

*Document created: February 5, 2026*
*Related: 214 (pattern taxonomy), 213 (meta-patterns), 133 (φ-FPU)*
