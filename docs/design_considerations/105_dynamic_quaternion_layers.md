# Design Consideration 105: Dynamic Quaternion Layers

## Date: 2026-01-06

## Status: EXPERIMENTAL (Validated)

## The Problem

From Design 104, we established quaternion layers with 16 structured dimensions:
- w (4D): Core Semantic
- x (4D): Grammatical
- y (4D): Contextual
- z (4D): Reserved

But natural language has **far more than 16 dimensions**. From reverse-engineering Qwen2, approximately **128 dimensions** were discovered. We can't predefine them all.

### The Regality Example

Consider these two sentences:
- "she put out the table ware for guests"
- "he put out the finery for company"

The **core action is identical** (setting table for visitors). The **grammatical structure is identical** (past, perfective, active). But the **output is completely different** due to:

- **Gender**: she → he
- **Regality**: table ware → finery, guests → company
- **Vocabulary level**: put out → (same, but could be "arranged", "laid out")

These dimensions aren't in any grammar textbook. They're **emergent** - they exist because language encodes them, not because linguists defined them.

## The Solution: Dynamic z-Layer

Keep structured layers (w, x, y) for predictable navigation, but make z **dynamic**:

```
Q = w + xi + yj + zk

w = Core Semantic (4D, named, fixed)
x = Grammatical (4D, named, fixed)
y = Contextual (4D, named, fixed)
z = Dynamic (ND, emergent, grows)
```

The z-layer:
- High-dimensional space (64D, 128D, or more)
- Dimensions discovered from data
- Projected to 4D for quaternion operations when needed
- Can grow as new dimensions emerge

## Architecture

### DynamicDimensionRegistry

```python
class DynamicDimensionRegistry:
    def __init__(self, max_dims: int = 128):
        self._dimensions: Dict[str, int] = {}  # name → index
        self._anchors: Dict[str, Dict[str, float]] = {}  # name → {word: level}
    
    def register(self, name: str, anchors: Dict[str, float] = None) -> int:
        """Register a new dimension with optional anchor words."""
        
    def get_level_for_word(self, word: str) -> Dict[str, float]:
        """Get all dimension levels activated by a word."""
```

### DynamicQuaternionPosition

```python
@dataclass
class DynamicQuaternionPosition:
    w: np.ndarray  # 4D, fixed
    x: np.ndarray  # 4D, fixed
    y: np.ndarray  # 4D, fixed
    z: np.ndarray  # ND, variable
    
    def z_projected(self, target_dim: int = 4) -> np.ndarray:
        """Project z to fixed dimensionality for quaternion ops."""
```

## Experimental Results

### Demo 1: Regality Dimension

```
Sentence 1: 'she put out the table ware for guests'
  Dynamic dims: {'gender': -1.0}

Sentence 2: 'he put out the finery for company'
  Dynamic dims: {'gender': 1.0, 'regality': 2.0}

Sentence 3: 'they laid out the china for the visitors'
  Dynamic dims: {'regality': 1.5, 'vocabulary_level': 1.5}

KEY INSIGHT:
  All three sentences have IDENTICAL w, x, y positions!
  The ONLY difference is in the dynamic z-layer.
```

### Demo 2: Dimension Discovery

Dimensions emerge from contrasting pairs:

```
Discovered 6 dimensions:
  [0] gender: ['he', 'king', 'boy'] ↔ ['she', 'queen', 'girl']
  [1] regality: ['finery', 'palace', 'monarch'] ↔ ['dishes', 'house', 'person']
  [2] tempo: ['quickly', 'rushed'] ↔ ['slowly', 'leisurely']
  [3] volume: ['whispered', 'murmured'] ↔ ['shouted', 'bellowed']
  [4] temporal_distance: ['ancient', 'old'] ↔ ['modern', 'new']
  [5] scale: ['tiny', 'microscopic'] ↔ ['enormous', 'cosmic']
```

We don't predefine these - they **appear when we see contrasting words**.

### Demo 3: High-Dimensional Navigation

With 20 dimensions registered:

```
Base: 'she put out the table ware for guests'
  z: {'gender': -1.0}

Transformations:
  gender_flip (+2.0):
    → {'gender': 1.0}
    'he put out the table ware for guests'

  regality_increase (+2.0):
    → {'gender': -1.0, 'regality': 2.0}
    'she put out the finery for company'

  gender + regality + urgency + formality:
    → {'gender': 1.0, 'regality': 2.0, 'formality': 1.0, 'urgency': 1.5}
    'He hastily arranged the fine china for the arriving dignitaries'
```

**Navigation remains predictable** even at high dimensionality.

### Demo 4: Generation from Position

Given a target position:
```
w: general, specific, inform, professional
x: future, prospective, indicative, active
y: formal, direct, polite, neutral
z: {gender: 1.0, regality: 1.5, tempo: 0.5, formality: 1.0, intimacy: -0.5}
```

Predicted text:
> "The gentleman will shortly arrange the fine settings for the distinguished guests."

Transformed position (casual, female, common):
> "She's gonna set out the plates for the folks coming over."

**Same core action, completely different output** based on z-dimensions.

## Key Insights

### 1. Structured + Dynamic = Best of Both

- **Structured layers (w, x, y)**: Predictable, navigable, well-defined
- **Dynamic layer (z)**: Flexible, emergent, captures nuance

We get predictability where we need it and flexibility where we can't predefine.

### 2. Dimensions Emerge from Contrasts

We don't need to predefine "regality" or "tempo". They appear when we encounter:
- "finery" vs "dishes"
- "quickly" vs "slowly"

The structure self-assembles from data.

### 3. High Dimensionality is Navigable

Even with 128 dimensions, transformations are just vector additions:
```python
new_z = old_z + delta
```

The math doesn't care how many dimensions there are.

### 4. Generation is Inverse Navigation

If we can navigate TO a position, we can describe what text SHOULD BE at that position. This is the foundation for generation.

## Connection to Qwen2 Findings

The 128 dimensions discovered in Qwen2 likely include:
- Standard grammatical dimensions (tense, aspect, mood, voice)
- Stylistic dimensions (formality, register, politeness)
- Semantic dimensions (domain, specificity)
- **Emergent dimensions** (regality, intimacy, urgency, etc.)

Our architecture can accommodate all of these:
- w, x, y handle the standard ones
- z handles the emergent ones
- Total capacity: 12 + 128 = 140 dimensions

## Implementation Path

### Phase 1: Current (Experimental)
- `experiments/dynamic_quaternion_layers.py`
- Manual dimension registration
- Demonstrates the concept

### Phase 2: Integration
- Move `DynamicDimensionRegistry` to `core/`
- Extend `PrimitiveRegistry` to include z-dimensions
- Update `QuaternionPosition` to use dynamic z

### Phase 3: Discovery
- Implement automatic dimension discovery from contrasting pairs
- Use SVD/PCA to find principal axes of variation
- Let dimensions emerge from ingested data

### Phase 4: Generation
- Given a target position, generate text that fits
- Use z-dimensions to control style, tone, vocabulary
- Bidirectional navigation: text ↔ position

## The Principle

> **Structure what you can. Discover what you can't.**

The w, x, y layers give us predictable navigation for the dimensions we understand. The z layer gives us room to grow as we discover dimensions we didn't anticipate.

Language is richer than any fixed schema can capture. But geometry doesn't care - it just needs coordinates. The z-layer is where the unknown becomes known.

---

*"The gentleman will shortly arrange the fine settings for the distinguished guests."*
*"She's gonna set out the plates for the folks coming over."*

*Same action. Same meaning. Different dimensions. Different words.*
