# Design Consideration 099: Absolute φ-Lattice Coordinates

## Date: 2026-01-06

## Status: Proposed

## Problem Statement

The current knowledge matching system uses **eigenspace coordinates** derived from similarity matrix decomposition. While this achieves 88% accuracy with sqrt-inverse eigenvalue weighting, it suffers from a fundamental limitation:

**Similarity matrices give us RELATIVE positions, not ABSOLUTE positions.**

The DC component problem we identified is a symptom of this deeper issue:
- The first eigenvalue (λ₀) captures 58% of variance as "average similarity to everything"
- Query projection pulls toward the centroid (weighted average effect)
- Positions are compressed into a narrow range [0.1, 0.5] (only φ⁻² to φ⁻¹)
- Dimensions have no inherent semantic meaning - they're emergent

### Evidence from Deep Dive Analysis

```
EIGENSPACE:
  Position range: [-0.36, 0.48]  (compressed)
  DC dominance: 58%
  Dimensions: Emergent, no semantic meaning
  Verifiable: No - can't check if position is "correct"

Query "who are you?":
  Distance to centroid: 0.117
  Distance of correct answer to centroid: 0.204
  → Query is CLOSER to centroid than the correct answer!
```

This is the centroid pull problem: queries project toward the center because projection is a weighted average of all concept positions.

## The Insight

From the user:
> "Similarity matrices feel like they're pointing at some position in space, but we're having a hard time finding absolute values for."

The similarity matrix IS pointing at absolute positions. We just can't find them because we're using eigendecomposition, which gives relative coordinates.

### The Old TruthSpace Had It Right

The original TruthSpace (`temp/old_core/truthspace.py`) used absolute coordinates:

```python
# 12 dimensions with semantic meaning
PHI_BLOCK_WEIGHTS = [
    φ², φ², φ², φ²,     # Actions: dims 0-3
    1.0, 1.0, 1.0, 1.0,  # Domains: dims 4-7
    φ⁻², φ⁻², φ⁻², φ⁻²  # Relations: dims 8-11
]

# Positions at φ^level on semantic dimensions
def _encode(self, text):
    position = np.zeros(12)
    for word in words:
        if word in primitives:
            prim = primitives[word]
            value = φ ** prim.level
            position[prim.dimension] = max(position[prim.dimension], value)
    return position
```

**Key properties:**
1. Positions are **ABSOLUTE** - defined by φ^level on fixed dimensions
2. Positions are **VERIFIABLE** - you can check if they're valid lattice points
3. **No DC component** - positions aren't derived from similarity
4. **Semantic dimensions** - each axis has meaning (action, domain, relation)

### Supporting Evidence

**Zeta Line Method** (`docs/zeta_line_method.md`):
- Neural network weights naturally cluster at φ^(-k) levels
- 2.91x compression achieved by exploiting this structure
- Values align to the "zeta line" through truth space

**Kerr Truth Space Discovery** (`docs/kerr_truth_space_discovery.md`):
- Event horizon at σ = 1/(2φ) ≈ 0.309 separates regimes
- Frame dragging creates spiral structure through φ-levels
- The "natural curve" for navigation exists

**phi_bbp Discovery**:
- Connection between φ and π through constant-based lattice
- Mathematical constants form a navigable coordinate system
- Positions are mathematically verifiable

## Proposed Solution

### Core Principle: φ-Lattice Coordinates

Replace eigenspace with a **φ-lattice coordinate system** where:

1. **Positions are at φ^k for integer k** (absolute, verifiable)
2. **Dimensions have semantic meaning** (not emergent)
3. **Similarity is used for navigation, not positioning**
4. **Zeta zeros serve as waypoints** between φ-levels

### Coordinate System Design

```python
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618

# Semantic dimensions for knowledge matching
DIMENSIONS = {
    0: 'domain',      # What area? (physics=φ³, identity=φ⁰, social=φ⁻¹)
    1: 'specificity', # How specific? (quantum=φ³, physics=φ², science=φ¹)
    2: 'intent',      # What response? (explain=φ², info=φ¹, ack=φ⁰)
    3: 'formality',   # How formal? (academic=φ², casual=φ⁰, informal=φ⁻¹)
}

# φ-levels provide the lattice points
PHI_LEVELS = {k: PHI ** k for k in range(-10, 11)}
# φ⁻¹⁰ = 0.008, φ⁻¹ = 0.618, φ⁰ = 1.0, φ¹ = 1.618, φ¹⁰ = 122.99
```

### Position Assignment

Each concept is assigned to a φ-lattice point:

```python
CONCEPT_POSITIONS = {
    # Knowledge concepts (high domain, high specificity)
    'physics': [3, 2, 2, 2],      # φ³, φ², φ², φ² = [4.24, 2.62, 2.62, 2.62]
    'science': [3, 1, 2, 1],      # φ³, φ¹, φ², φ¹ = [4.24, 1.62, 2.62, 1.62]
    'math': [3, 2, 2, 2],         # Same as physics (same lattice point)
    
    # Identity concepts (meta domain)
    'who_are_you': [0, 0, 1, 0],  # φ⁰, φ⁰, φ¹, φ⁰ = [1.0, 1.0, 1.62, 1.0]
    'what_can_you_do': [0, 1, 1, 0],
    
    # Social concepts (negative domain)
    'hello': [-1, -1, 0, -1],     # φ⁻¹, φ⁻¹, φ⁰, φ⁻¹ = [0.62, 0.62, 1.0, 0.62]
    'thank_you': [-1, 0, 0, 0],
    'goodbye': [-1, -1, 0, -1],
}

def levels_to_position(levels):
    return np.array([PHI ** k for k in levels])
```

### Query Matching

Queries are encoded to φ-levels and matched by distance:

```python
def encode_query(text):
    """Encode query to φ-lattice position."""
    # Analyze query to determine levels
    domain_level = detect_domain(text)      # -1 to 3
    specificity_level = detect_specificity(text)  # -1 to 3
    intent_level = detect_intent(text)      # -1 to 2
    formality_level = detect_formality(text)  # -1 to 2
    
    levels = [domain_level, specificity_level, intent_level, formality_level]
    return levels_to_position(levels)

def match(query_position, concepts):
    """Find nearest concept on φ-lattice."""
    best_concept = None
    best_distance = float('inf')
    
    for name, levels in concepts.items():
        position = levels_to_position(levels)
        distance = np.linalg.norm(query_position - position)
        if distance < best_distance:
            best_distance = distance
            best_concept = name
    
    return best_concept, best_distance
```

### Lattice Snapping

New concepts snap to the nearest valid lattice point:

```python
def snap_to_lattice(position):
    """Snap position to nearest φ-lattice point."""
    snapped = np.zeros_like(position)
    for i, v in enumerate(position):
        # Find nearest φ^k
        best_k = round(np.log(abs(v) + 1e-10) / np.log(PHI))
        best_k = max(-10, min(10, best_k))  # Clamp to valid range
        snapped[i] = PHI ** best_k
    return snapped

def is_valid_position(position, tolerance=0.01):
    """Check if position is on the φ-lattice."""
    snapped = snap_to_lattice(position)
    return np.allclose(position, snapped, atol=tolerance)
```

### Zeta Zeros as Waypoints

Zeta zeros provide navigation waypoints between φ-levels:

```python
ZETA_ZEROS = [14.13, 21.02, 25.01, 30.42, 32.94, 37.59, 40.92, 43.33, 48.01, 49.77]

def find_waypoint(query_level, target_level):
    """Find zeta zero waypoint between two φ-levels."""
    query_value = PHI ** query_level
    target_value = PHI ** target_level
    
    # Find zeta zero between them
    for z in ZETA_ZEROS:
        if min(query_value, target_value) < z < max(query_value, target_value):
            return z
    
    return None  # Direct path, no waypoint needed
```

## Comparison: Eigenspace vs φ-Lattice

| Property | Eigenspace | φ-Lattice |
|----------|------------|-----------|
| **Coordinate type** | Relative | Absolute |
| **Position range** | [0.1, 0.5] | [φ⁻¹⁰, φ¹⁰] = [0.008, 123] |
| **Dynamic range** | 5x | 15,000x |
| **Verifiable** | No | Yes |
| **DC component** | 58% | 0% |
| **Dimension meaning** | Emergent | Semantic |
| **Centroid pull** | Yes | No |
| **Accuracy (current)** | 88% | TBD |

## Implementation Plan

### Phase 1: Semantic Dimension Definition

Define the semantic dimensions for knowledge matching:

```python
SEMANTIC_DIMENSIONS = {
    0: {
        'name': 'domain',
        'description': 'What area of knowledge',
        'levels': {
            3: ['physics', 'math', 'chemistry', 'biology'],
            2: ['programming', 'technology', 'engineering'],
            1: ['general', 'knowledge', 'information'],
            0: ['meta', 'identity', 'self'],
            -1: ['social', 'greeting', 'farewell'],
        }
    },
    1: {
        'name': 'specificity',
        'description': 'How specific is the concept',
        'levels': {
            3: ['quantum', 'relativistic', 'differential'],
            2: ['physics', 'chemistry', 'calculus'],
            1: ['science', 'math', 'programming'],
            0: ['general', 'basic', 'overview'],
            -1: ['vague', 'broad', 'any'],
        }
    },
    # ... etc
}
```

### Phase 2: Bootstrap Concept Placement

Place bootstrap concepts at φ-lattice positions:

```python
def bootstrap_knowledge():
    """Place core concepts at φ-lattice positions."""
    concepts = {}
    
    # Physics domain
    concepts['physics'] = {
        'levels': [3, 2, 2, 1],
        'input': 'Physics is the natural science...',
        'output': 'Physics is the natural science...',
    }
    
    # Identity domain
    concepts['identity'] = {
        'levels': [0, 0, 1, 0],
        'input': 'What can I do for you?...',
        'output': 'What can I do for you?...',
    }
    
    # Social domain
    concepts['hello'] = {
        'levels': [-1, -1, 0, -1],
        'input': 'Hello!',
        'output': 'Hello! How can I help you?',
    }
    
    return concepts
```

### Phase 3: Query Encoding

Encode queries to φ-lattice positions:

```python
def encode_query(text, primitives):
    """
    Encode query text to φ-lattice position.
    
    Uses primitive detection (like old TruthSpace) to determine
    which semantic dimensions are activated and at what level.
    """
    levels = [0, 0, 0, 0]  # Default: neutral position
    
    words = tokenize(text)
    for word in words:
        if word in primitives:
            prim = primitives[word]
            dim = prim.dimension
            level = prim.level
            # MAX aggregation (Sierpinski property)
            levels[dim] = max(levels[dim], level)
    
    return levels_to_position(levels)
```

### Phase 4: Hybrid Navigation

Use similarity for navigation direction, φ-lattice for position:

```python
def navigate(query, concepts, similarity_fn):
    """
    Navigate to concept using hybrid approach.
    
    1. Encode query to φ-lattice position
    2. Use similarity to determine direction
    3. Snap to nearest valid lattice point
    4. Return matched concept
    """
    # Step 1: Encode query
    query_pos = encode_query(query)
    
    # Step 2: Find direction via similarity
    similarities = {name: similarity_fn(query, c['input']) 
                   for name, c in concepts.items()}
    
    # Step 3: Weight lattice distance by similarity
    scores = {}
    for name, concept in concepts.items():
        concept_pos = levels_to_position(concept['levels'])
        distance = np.linalg.norm(query_pos - concept_pos)
        similarity = similarities[name]
        # Score combines geometric distance with similarity direction
        scores[name] = similarity / (1 + distance)
    
    # Step 4: Return best match
    return max(scores, key=scores.get)
```

## Connection to Existing Work

### Design 039: φ-Zipf Duality
- φ^(-rank) ≡ Zipf for ranking
- Low frequency → HIGH importance → prime position
- This maps directly to φ-lattice levels

### Design 057: Domain as t-coordinate
- Domain separation in eigenspace
- φ-lattice makes this explicit with semantic dimensions

### Design 098: Prime-Zeta Lattice
- Primes = irreducible concept positions
- Zeta zeros = navigation waypoints
- φ-lattice provides the coordinate system for this

### Kerr Truth Space Discovery
- Event horizon at σ = 1/(2φ)
- Natural curve through φ-levels
- Zeta zeros as resonance points

## Expected Benefits

1. **No DC Component**: Positions are absolute, not derived from similarity
2. **Full Dynamic Range**: φ⁻¹⁰ to φ¹⁰ instead of [0.1, 0.5]
3. **Verifiable Positions**: Can check if a position is valid
4. **Semantic Meaning**: Each dimension has clear interpretation
5. **No Centroid Pull**: Queries don't average toward center
6. **Mathematical Foundation**: Based on φ, connected to zeta zeros

## Risks and Mitigations

### Risk 1: Primitive Detection
The old TruthSpace relied on keyword-to-primitive mapping. This could be seen as "not purely geometric."

**Mitigation**: The primitives are bootstrapped and immediately transformed to geometry. The keywords are just the initial seed - the positions are what matter.

### Risk 2: Novel Concepts
How do we place concepts that don't match existing primitives?

**Mitigation**: Use similarity to existing concepts to determine approximate levels, then snap to nearest lattice point. The lattice provides the coordinate system; similarity provides the navigation.

### Risk 3: Dimension Choice
How do we know 4 dimensions is right? What if we need more?

**Mitigation**: Start with 4 semantic dimensions (domain, specificity, intent, formality). Add dimensions as needed based on discrimination failures. The φ-lattice scales naturally.

## Conclusion

The DC component problem is a symptom of using relative coordinates. The solution is to return to **absolute φ-lattice coordinates** with semantic dimensions.

This approach:
- Eliminates the DC component entirely
- Provides mathematically verifiable positions
- Connects to the zeta line method and Kerr truth space discoveries
- Maintains the geometric philosophy: structure IS information

The φ-lattice is the coordinate system. Similarity is the navigation. Zeta zeros are the waypoints.

---

*"Similarity matrices are pointing at absolute positions. The φ-lattice is where those positions live."*
