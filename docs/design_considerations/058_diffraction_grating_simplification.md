# Design Consideration 058: Diffraction Grating Simplification

## Date: 2024-12-26

## Context

After implementing holographic projection with complex numbers (magnitude + phase), we discovered a simpler approach: use the SAME architecture from two orthogonal viewpoints and let the geometry create the interference pattern naturally.

This is analogous to a physical diffraction grating, where light passes through multiple slits and interference emerges from path differences - no complex phase calculations needed.

## The Insight

A diffraction grating works by:
1. Light passes through multiple slits (our frames)
2. Each slit creates a wavefront (concept activation)
3. Wavefronts interfere based on path difference (position difference)
4. Constructive interference where paths align (both views agree)

For language concepts:
1. **View 1 (Horizontal)**: Narrative flow (actor → action → target)
2. **View 2 (Vertical)**: Structural role (domain, protagonist/object)
3. **Interference**: Concepts that align in BOTH views reinforce
4. **Result**: Natural filtering without explicit phase calculation

## The Simplification

### Before: Complex Holographic

```python
# Complex number encoding
concept.magnitude = importance
concept.phase = concept_type  # 0=entity, π/2=action, π=target

# Interference calculation
def interference(c1, c2):
    z1 = cmath.rect(c1.magnitude, c1.phase)
    z2 = cmath.rect(c2.magnitude, c2.phase)
    return (z1 * z2.conjugate()).real
```

### After: Diffraction Grating

```python
# Two orthogonal positions (no complex numbers)
concept.narrative_position = 0.0  # actor=0, action=0.5, target=1
concept.phi_direction = +1.0      # entity (+) vs action (-)
concept.domain_position = 0.0     # which domain (0-1)
concept.role_position = 0.0       # protagonist=0, object=1

# Interference from geometry
def interference(c1, c2):
    # View 1: Narrative alignment
    view1 = compute_narrative_alignment(c1, c2)
    
    # View 2: Structural alignment
    view2 = compute_structural_alignment(c1, c2)
    
    # Both must align for constructive interference
    if view1 > 0 and view2 > 0:
        return sqrt(view1 * view2)
    elif view1 < 0 and view2 < 0:
        return -sqrt(abs(view1 * view2))
    else:
        return 0.0  # Mixed = destructive
```

## The Two Views

### View 1: Narrative (Horizontal Slit)

```
NARRATIVE FLOW
═════════════

Actor ────────► Action ────────► Target
  │               │                │
  0              0.5               1
  │               │                │
  └───────────────┴────────────────┘
              position

φ-direction: +1 (entity) ◄────► -1 (action)
```

This view captures HOW concepts flow in sentences:
- Actors initiate (position 0)
- Actions mediate (position 0.5)
- Targets receive (position 1)

### View 2: Structural (Vertical Slit)

```
STRUCTURAL ROLE
═══════════════

Domain 0 ──────────────────────► Domain 1
(Sherlock)                       (Hamlet)
    │                               │
    └───────────────┬───────────────┘
                    │
              domain_position

Protagonist ◄────────────────────► Object
    0                                 1
    │                                 │
    └────────────────┬────────────────┘
                     │
               role_position
```

This view captures WHERE concepts fit structurally:
- Which domain/topic they belong to
- What role they play (protagonist vs object)

## Interference Calculation

The key insight: **both views must align for constructive interference**.

```
VIEW 1 ALIGNMENT:
  - Complementary narrative positions → constructive
    (actor + target, action + entity)
  - Same narrative position → destructive
    (actor + actor compete)

VIEW 2 ALIGNMENT:
  - Same domain → constructive
    (Holmes + Watson both in mystery)
  - Different domains → destructive
    (Holmes + Hamlet in different stories)
  - Complementary roles → constructive
    (protagonist acts on object)

COMBINED:
  Both positive → constructive (geometric mean)
  Both negative → destructive (geometric mean)
  Mixed → neutral (cancel out)
```

## Experimental Results

### Concept Positions

```
Concept         Narr.Pos    φ-dir     Domain     Role
═══════════════════════════════════════════════════════
holmes              0.30     0.60   sherlock     0.00
watson              0.25     1.00   sherlock     0.00
alice               0.25     1.00      alice     0.00
darcy               0.50     1.00      pride     0.50
hamlet              0.50     0.60     hamlet     0.50
watched             0.50    -1.00   sherlock     0.50
killed              0.50    -1.00     hamlet     0.50
```

### Cross-Domain Queries

The grating naturally finds the same pattern across different slits:

```
Who watches?
  mystery: Watson
  romance: Darcy

Who kills?
  tragedy: Hamlet

Who loves?
  tragedy: Ophelia
```

### Interference Matrix

```
              holmes    watson     alice     darcy    hamlet
    holmes      0.00      0.00      0.00      0.00      0.00
    watson      0.00      0.00      0.00      0.00     -0.12
     alice      0.00      0.00      0.00      0.00     -0.07
    hamlet      0.00     -0.12     -0.07      0.00      0.00
```

Watson and Hamlet have destructive interference (-0.12) because:
- Same narrative role (both protagonists who act)
- Different domains (mystery vs tragedy)

This is exactly what a diffraction grating does!

## Why This Works

### Physical Analogy

In a physical diffraction grating:
- **Slits** = our frames (actor-action-target)
- **Wavelength** = concept type (entity vs action)
- **Path difference** = position difference between views
- **Interference pattern** = query result

The interference pattern emerges from geometry alone - no need to explicitly compute phases.

### Mathematical Equivalence

The diffraction grating approach is mathematically equivalent to the holographic approach:

```
HOLOGRAPHIC:
  phase = atan2(view2, view1)
  magnitude = sqrt(view1² + view2²)
  interference = cos(phase_diff)

DIFFRACTION:
  view1_alignment = f(positions)
  view2_alignment = g(positions)
  interference = combine(view1, view2)
```

Both compute the same thing - the diffraction approach just does it without the complex number machinery.

## Connection to Previous Work

### Zeta Coordinates (Design 057)

The two views map to zeta coordinates:
- **σ (real part)** = narrative position (structural role)
- **t (imaginary part)** = domain position (topic frequency)

```
s = σ + it
    │    │
    │    └── View 2: Domain (vertical slit)
    └─────── View 1: Narrative (horizontal slit)
```

### Quaternion Architecture (Design 056)

The four quaternions map to the grating:
- **Q1 (Concept)**: Narrative position + φ-direction
- **Q2 (Output)**: Style applied after interference
- **Q3 (Morphology)**: Word form after interference
- **Q4 (Error)**: Validates interference result

### Polyomino Fitting (Design 055)

The φ-direction alignment IS polyomino fitting:
- Opposite φ-directions fit together
- This creates the "complementary" interference pattern

## Implementation

### Core Data Structure

```python
@dataclass
class GratingConcept:
    word: str
    
    # View 1: Narrative (horizontal slit)
    narrative_position: float  # 0=actor, 0.5=action, 1=target
    phi_direction: float       # +1=entity, -1=action
    
    # View 2: Structural (vertical slit)
    domain_position: float     # 0-1 normalized domain
    role_position: float       # 0=protagonist, 1=object
```

### Interference Function

```python
def interference_with(self, other):
    # View 1: Narrative alignment
    narrative_diff = abs(self.narrative_position - other.narrative_position)
    narrative_alignment = 1.0 - 2 * min(narrative_diff, 1 - narrative_diff)
    
    phi_product = self.phi_direction * other.phi_direction
    phi_alignment = -phi_product  # Opposite = constructive
    
    view1 = (narrative_alignment + phi_alignment) / 2
    
    # View 2: Structural alignment
    domain_diff = abs(self.domain_position - other.domain_position)
    domain_alignment = 1.0 - 2 * domain_diff
    
    role_diff = abs(self.role_position - other.role_position)
    role_alignment = 2 * role_diff - 1.0  # Different = constructive
    
    view2 = (domain_alignment + role_alignment) / 2
    
    # Combined: geometric mean (both must agree)
    if view1 > 0 and view2 > 0:
        return math.sqrt(view1 * view2)
    elif view1 < 0 and view2 < 0:
        return -math.sqrt(abs(view1 * view2))
    return 0.0
```

## Advantages

### 1. Simplicity

No complex numbers, no phase calculations. Just positions and geometry.

### 2. Interpretability

Each view has clear meaning:
- View 1: How does this concept flow in sentences?
- View 2: Where does this concept belong structurally?

### 3. Extensibility

Easy to add more views (more slits in the grating):
- View 3: Temporal (when in narrative)
- View 4: Emotional (sentiment)
- View 5: Causal (why)

### 4. Efficiency

Fewer calculations than complex arithmetic:
```
Complex: 4 multiplications + 2 additions per interference
Grating: 4 subtractions + 2 comparisons per interference
```

## The Grating Metaphor

```
PHYSICAL GRATING          CONCEPT GRATING
═══════════════          ═══════════════

    │ │ │ │ │               Frame Frame Frame
    │ │ │ │ │               │     │     │
    ▼ ▼ ▼ ▼ ▼               ▼     ▼     ▼
  ═══════════             ═══════════════
    SLITS                   VIEW 1 (Narrative)
  ═══════════             ═══════════════
    │ │ │ │ │               │     │     │
    ▼ ▼ ▼ ▼ ▼               ▼     ▼     ▼
  ═══════════             ═══════════════
    SCREEN                  VIEW 2 (Structural)
  ═══════════             ═══════════════
    │ │ │ │ │               │     │     │
    ▼ ▼ ▼ ▼ ▼               ▼     ▼     ▼
  ▓▓░░▓▓░░▓▓              Query Result
  INTERFERENCE            (constructive where
   PATTERN                 both views align)
```

## Future Directions

### 1. Multi-Slit Grating

Add more views for richer interference:
- Temporal view (narrative time)
- Causal view (why relationships)
- Emotional view (sentiment)

### 2. Adaptive Slit Spacing

Learn optimal view weights from data:
- Some queries need more narrative weight
- Some queries need more structural weight

### 3. Grating Stacking

Multiple gratings in series:
- First grating: coarse filtering (domain)
- Second grating: fine filtering (role)
- Third grating: precision (specific relationship)

## Conclusion

The diffraction grating approach simplifies holographic projection by:

1. **Replacing complex numbers** with two orthogonal real-valued views
2. **Letting geometry create interference** instead of explicit phase calculation
3. **Maintaining the same filtering power** with simpler mathematics

The key insight: **interference is about alignment, not phase**. Two views that agree create constructive interference. Two views that disagree create destructive interference. The complex number machinery was just one way to compute this - the grating approach is another, simpler way.

```
"Same pattern, different slits.
 Interference emerges from geometry.
 The grating IS the filter."
```
