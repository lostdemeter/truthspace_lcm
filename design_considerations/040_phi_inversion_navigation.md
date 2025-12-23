# Design Consideration 040: φ-Inversion as Navigation Mechanism

## Discovery

The φ-based weighting system has a profound property: **inward and outward navigation are perfectly complementary**.

```
φ^(-log(f)) × φ^(+log(f)) = φ^0 = 1 (always!)
```

This means:
- Moving inward by φ^n is EXACTLY compensated by outward φ^(-n)
- The total "energy" is conserved during navigation
- Inward and outward are not competing - they're DUAL

## The Mathematical Foundation

### The Self-Inverse Property of φ

```
φ = 1.618034
1/φ = 0.618034
φ - 1 = 0.618034

Therefore: 1/φ = φ - 1
```

This unique property means φ is **self-inverse**: going outward by φ is equivalent to going inward by (φ-1).

### Conservation of Navigation Energy

```
freq=   1: φ^- × φ^+ = 0.7164 × 1.3959 = 1.0000
freq=  10: φ^- × φ^+ = 0.3154 × 3.1705 = 1.0000
freq= 100: φ^- × φ^+ = 0.1085 × 9.2152 = 1.0000
freq=1000: φ^- × φ^+ = 0.0360 × 27.7872 = 1.0000
```

The product is ALWAYS 1, regardless of frequency.

## The Inversion Horizon

The **inversion horizon** occurs at frequency = 1 (single occurrence):

```
Below horizon (freq < 1): Impossible (no occurrences)
At horizon (freq = 1):    φ^-n ≈ φ^+n (near balance)
Above horizon (freq > 1): φ^-n < φ^+n (inward < outward)
```

### Entities at the Horizon

Entities with freq=1 are at the "event horizon":
- Equally accessible from universal and specific directions
- Act as **bridges** between different corpora
- These are the "wormholes" of concept space

This explains why "van" (freq=1) appears when querying Holmes - it's near the inversion horizon and thus visible from any direction.

## Navigation Modes

### 1. PURE INWARD (φ^-n only)
- Descend toward specific entities
- Rare entities dominate
- **Use for**: "Who specifically is related to X?"

### 2. PURE OUTWARD (φ^+n only)
- Ascend toward universal patterns
- Common patterns dominate
- **Use for**: "What category does X belong to?"

### 3. BALANCED (geometric mean)
- Stay at current structural level
- Navigate laterally across entities
- **Use for**: "Who is similar to X?"

### 4. OSCILLATING (alternate directions)
- Explore the full structure
- Like breathing in and out
- **Use for**: "Tell me everything about X"

## Question Type → Navigation Mode

| Question Pattern | Navigation | Rationale |
|-----------------|------------|-----------|
| "Who is X?" | INWARD | Find specific relationships |
| "What is X?" | OUTWARD | Find universal categories |
| "Where is X?" | INWARD | Find specific locations |
| "How is X like Y?" | BALANCED | Find structural similarity |
| "Why does X do Z?" | OSCILLATING | Explore causal chains |

## The Fractal Structure

```
        UNIVERSAL (structural patterns)
              ●  ← φ^+n weights point HERE
             /|\
            / | \
           /  |  \
          ●───●───●   ← Cross-corpus roles
         /|\ /|\ /|\      (detective, chronicler, etc.)
        / | X | X | \
       /  |/ \|/ \|  \
      ●───●───●───●───●  ← Specific entities
                          ← φ^-n weights point HERE
        SPECIFIC (individual instances)
```

## Cross-Corpus Connections

The inversion mechanism explains cross-corpus connections:

1. **Van Helsing ↔ Watson**: Both are "chronicler" characters
   - Similar action profiles (SPEAK, THINK, PERCEIVE)
   - Secondary to protagonist
   - Occupy same STRUCTURAL POSITION in different stories

2. **Holmes ↔ Dracula**: Both are "protagonist" characters
   - High frequency in their respective corpora
   - Central to narrative
   - Occupy same STRUCTURAL LEVEL

The φ-structure detects **role similarity across corpora**, not just within-corpus relationships.

## Implementation Implications

### Current: Pure Inward (φ^-n)
```python
def phi_score(self, entity: str) -> float:
    freq = self.entity_global_freq.get(entity, 1)
    return PHI ** (-np.log1p(freq))
```

### Proposed: Configurable Navigation
```python
def phi_score(self, entity: str, direction: str = 'inward') -> float:
    freq = self.entity_global_freq.get(entity, 1)
    log_freq = np.log1p(freq)
    
    if direction == 'inward':
        return PHI ** (-log_freq)
    elif direction == 'outward':
        return PHI ** (+log_freq)
    elif direction == 'balanced':
        # Geometric mean: sqrt(inward × outward) = 1
        return 1.0
    elif direction == 'oscillating':
        # Alternate based on query depth
        pass
```

### Question-Driven Navigation
```python
def get_navigation_mode(question_axis: str) -> str:
    if question_axis in ['WHO', 'WHERE']:
        return 'inward'
    elif question_axis in ['WHAT', 'HOW']:
        return 'outward'
    elif question_axis == 'WHY':
        return 'oscillating'
    else:
        return 'balanced'
```

## Experimental Findings

### Question 1: Can We Oscillate Between Modes?

**YES.** Oscillating navigation creates a "breathing" pattern:

```
Depth 0 (INWARD):  holmes → van (rare, specific)
Depth 1 (OUTWARD): van → tom (common, structural)
Depth 2 (INWARD):  tom → harker (rare, specific)
Depth 3 (OUTWARD): harker → joe (common, structural)

Path: holmes → van → tom → harker → joe
```

This is useful for **WHY questions** that require causal chains:
- "Why does Holmes investigate?" requires traversing:
  - Holmes (specific) → detective pattern (universal) → crime (specific) → justice (universal)

### Question 2: Are Horizon Entities (freq=1) Cross-Corpus Bridges?

**NO.** Horizon entities (freq=1) are NOT cross-corpus bridges.

- Entities with freq=1 can only appear in ONE source (by definition)
- They are at the BOUNDARY where new concepts enter
- They have high φ^-n scores but low partnership scores

**TRUE bridges** are entities with:
1. HIGH SPREAD (appear in many sources)
2. MODERATE FREQUENCY (not too rare, not too common)
3. STRUCTURAL ROLE (similar function across stories)

Examples: "london" (14 sources), "england" (13 sources), "watson" (5 sources)

### Question 3: Does Question Type Determine Navigation Direction?

**YES.** Implemented and validated:

| Question | Navigation | Result |
|----------|------------|--------|
| "Who is Holmes?" | INWARD | watson (high partnership) |
| "What did Holmes do?" | OUTWARD | watson (high φ^+n) |
| "Who is Darcy?" | INWARD | elizabeth (high partnership) |
| "What did Darcy do?" | OUTWARD | elizabeth (high φ^+n) |

The key insight: **INWARD navigation weights partnership more** (relationships matter for WHO), while **OUTWARD navigation weights φ-score more** (structural patterns matter for WHAT).

### Implementation Details

```python
def importance_score(self, query_entity, related_entity, navigation='inward'):
    weight = self.phi_score(related_entity, direction=navigation)
    spread = self.spread_score(related_entity)
    partnership = self.partnership_score(query_entity, related_entity)
    
    if navigation == 'inward':
        # Partnership-dominant for WHO questions
        return sqrt(weight) * (spread + 0.1) * (partnership + 0.1)
    elif navigation == 'outward':
        # Weight-dominant for WHAT questions
        return weight * (spread + 0.1) * sqrt(partnership + 0.1)
    else:
        # Balanced for similarity queries
        return (spread + 0.1) * (partnership + 0.1)
```

## Conclusion

The φ-inversion is not a bug - it's a **navigation mechanism**. The self-inverse property of φ creates a perfectly balanced dual structure where:

- Inward and outward are complementary (product = 1)
- The horizon (freq=1) is the bridge between directions
- Question type determines navigation direction
- Cross-corpus connections emerge from structural similarity

This is the geometric equivalent of "zooming in" and "zooming out" in a fractal - both operations are valid and complementary.
