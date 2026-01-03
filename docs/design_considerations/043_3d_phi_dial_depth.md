# Design Consideration 043: 3D φ-Dial with Depth Control

## Date: 2024-12-22

## Context

After implementing the 2D Complex φ-Dial (042), we explored what a third axis might control. The hypothesis was that **z controls depth/elaboration** — how much detail to include in the answer.

## The 3D φ-Dial

```
"We control the horizontal. We control the vertical. We control the depth."
```

### Axis Definitions

| Axis | Name | Range | Controls |
|------|------|-------|----------|
| **X** | Style | -1 to +1 | WHAT words we choose |
| **Y** | Perspective | -1 to +1 | HOW we frame content |
| **Z** | Depth | -1 to +1 | HOW MUCH detail we include |

### X-Axis: Style (Horizontal)
```
-1 = formal, specific, rare words
 0 = neutral, balanced
+1 = casual, universal, common words
```

### Y-Axis: Perspective (Vertical)
```
-1 = subjective, experiential, personal
 0 = objective, factual, neutral
+1 = meta, analytical, reflective
```

### Z-Axis: Depth (NEW)
```
-1 = terse, minimal, just the facts
 0 = standard, balanced
+1 = elaborate, detailed, full context
```

## What Z Controls

The depth dial affects multiple aspects of answer generation:

| z Value | Max Actions | Relationship | Source | Elaboration |
|---------|-------------|--------------|--------|-------------|
| -1 (terse) | 1 | No | No | No |
| 0 (standard) | 2 | Yes | No | No |
| +1 (elaborate) | 4 | Yes | Yes | Yes |

### Example Outputs

**Question: "Who is Holmes?"**

**TERSE (z=-1):**
```
Holmes is a character from Sherlock Holmes who spoke.
```

**STANDARD (z=0):**
```
Holmes is a character from Sherlock Holmes who spoke, 
frequently associated with Watson.
```

**ELABORATE (z=+1):**
```
Holmes is a character from Sherlock Holmes who spoke and 
considered and observed, frequently associated with Watson. 
Central to the story's development. (from Sherlock Holmes)
```

## The Octants

With 3 axes, we now have 8 octants (2³ = 8) instead of 4 quadrants:

| Octant | x | y | z | Style | Perspective | Depth | Label |
|--------|---|---|---|-------|-------------|-------|-------|
| 1 | - | - | - | Formal | Subjective | Terse | Literary/Minimal |
| 2 | - | - | + | Formal | Subjective | Elaborate | Literary/Rich |
| 3 | - | + | - | Formal | Meta | Terse | Scholarly/Brief |
| 4 | - | + | + | Formal | Meta | Elaborate | Scholarly/Full |
| 5 | + | - | - | Casual | Subjective | Terse | Conversational/Quick |
| 6 | + | - | + | Casual | Subjective | Elaborate | Conversational/Detailed |
| 7 | + | + | - | Casual | Meta | Terse | Pop Culture/Quip |
| 8 | + | + | + | Casual | Meta | Elaborate | Pop Culture/Analysis |

## Mathematical Structure

The 3D dial extends the complex φ structure:

```
2D: φ^(x + iy) = φ^x · e^(iy·ln(φ))
    - Magnitude (φ^x) = vocabulary selection
    - Phase (y·ln(φ)) = perspective framing

3D: φ^(x + iy) × scale(z)
    - Magnitude = vocabulary selection
    - Phase = perspective framing
    - Scale = depth/elaboration
```

The z-axis doesn't fit naturally into the complex number structure, but it does fit the **quaternion** structure:

```
q = φ^(x + iy + jz)
```

However, for practical purposes, we treat z as a separate scaling factor rather than a true quaternion component.

## Implementation

### ComplexPhiDial (now 3D)

```python
class ComplexPhiDial:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = max(-1.0, min(1.0, x))  # Style
        self.y = max(-1.0, min(1.0, y))  # Perspective
        self.z = max(-1.0, min(1.0, z))  # Depth
    
    def get_max_actions(self) -> int:
        if self.z < -0.3: return 1      # Terse
        elif self.z > 0.3: return 4     # Elaborate
        return 2                         # Standard
    
    def include_relationship(self) -> bool:
        return self.z > -0.5
    
    def include_source(self) -> bool:
        return self.z > 0.0
    
    def include_elaboration(self) -> bool:
        return self.z > 0.3
```

### Usage

```python
from truthspace_lcm import ConceptQA

# Terse mode
qa = ConceptQA(style_x=0, perspective_y=0, depth_z=-1)

# Elaborate mode
qa = ConceptQA(style_x=0, perspective_y=0, depth_z=1)

# Dynamic adjustment
qa.set_depth(-1)  # Switch to terse
qa.set_depth(1)   # Switch to elaborate
```

### CLI

```bash
python run.py --depth -1   # Terse
python run.py -z 1         # Elaborate
```

### Interactive Commands

```
/depth -1    # Set terse mode
/depth 1     # Set elaborate mode
/dial        # Show all three dial settings
```

## Geometric Interpretation

The 3D dial can be visualized as a cube:

```
                    ELABORATE (+z)
                         │
                         │
         ┌───────────────┼───────────────┐
        /│              /│              /│
       / │             / │             / │
      /  │            /  │            /  │
     ┌───────────────┼───────────────┐  │
     │   │           │   │           │  │
     │   │     FORMAL│   │CASUAL     │  │
     │   │    (-x)   │   │  (+x)     │  │
     │   └───────────┼───┼───────────┼──┘
     │  /            │  /            │ /
     │ /             │ /             │/
     │/              │/              │
     └───────────────┼───────────────┘
                     │
                     │
                  TERSE (-z)
                  
     (Y-axis goes into/out of page: SUBJECTIVE ↔ META)
```

## Connection to Information Theory

The z-axis has an interesting information-theoretic interpretation:

- **Terse (z=-1)**: Minimum viable information
- **Standard (z=0)**: Balanced information density
- **Elaborate (z=+1)**: Maximum available information

This maps to **compression ratio**:
- Terse = high compression (lossy)
- Elaborate = low compression (lossless)

## Future Directions

1. **Adaptive Depth**: Automatically adjust z based on question complexity
2. **User Preference Learning**: Track preferred depth per user
3. **Context-Sensitive Depth**: Short answers for simple questions, elaborate for complex
4. **W-Axis?**: Could there be a 4th dimension? (Certainty/hedging?)

## Conclusion

The z-axis for depth/elaboration is a natural extension of the 2D φ-dial. It controls **how much** information to include, complementing x (what words) and y (how framed).

The 3D dial provides complete control over answer generation:
- **X**: Vocabulary selection (formal ↔ casual)
- **Y**: Perspective framing (subjective ↔ meta)
- **Z**: Information density (terse ↔ elaborate)

Together, they form a **27-point control space** (3³ = 27 discrete settings) or a continuous cube of infinite settings.
