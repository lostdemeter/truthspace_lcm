# Design Consideration 044: 4D Quaternion φ-Dial

## Date: 2024-12-22

## Context

After implementing the 3D φ-dial (043), we explored what a 4th dimension might control. The hypothesis was that **w controls certainty/modality** — how sure we are about what we're saying.

## The 4D Quaternion φ-Dial

```
"We control the horizontal. We control the vertical. 
 We control the depth. We control the certainty."
```

### Quaternion Structure

The 4D dial follows the quaternion structure:

```
q = w + xi + yj + zk
```

Where:
- **x (i-axis)**: Style — WHAT words we choose
- **y (j-axis)**: Perspective — HOW we frame content
- **z (k-axis)**: Depth — HOW MUCH detail we include
- **w (scalar)**: Certainty — HOW SURE we are

### Axis Definitions

| Axis | Name | Range | Controls |
|------|------|-------|----------|
| **X** | Style | -1 to +1 | Vocabulary selection (formal ↔ casual) |
| **Y** | Perspective | -1 to +1 | Voice/framing (subjective ↔ meta) |
| **Z** | Depth | -1 to +1 | Detail level (terse ↔ elaborate) |
| **W** | Certainty | -1 to +1 | Epistemic stance (definitive ↔ hedged) |

## W-Axis: Certainty/Modality

The w-axis controls epistemic modality — how certain we are about our statements.

### Certainty Levels

| w | Certainty | Copula | Relationship | Opener |
|---|-----------|--------|--------------|--------|
| -1 | Definitive | "is undoubtedly" | "closely tied to" | "Certainly," |
| 0 | Neutral | "is" | "associated with" | (none) |
| +1 | Hedged | "appears to be" | "possibly connected to" | "Perhaps" |

### Example Outputs

**Question: "Who is Holmes?"**

**DEFINITIVE (w=-1):**
```
Certainly, Holmes is undoubtedly a character from Sherlock Holmes 
who spoke, closely tied to Watson.
```

**NEUTRAL (w=0):**
```
Holmes is a character from Sherlock Holmes who spoke, 
associated with Watson.
```

**HEDGED (w=+1):**
```
It seems that Holmes appears to be a character from Sherlock Holmes 
who spoke, possibly connected to Watson.
```

## The 16 Hexadecants

With 4 axes and 2 extremes each, we have 16 "hexadecants" (2⁴ = 16):

| # | x | y | z | w | Style | Perspective | Depth | Certainty |
|---|---|---|---|---|-------|-------------|-------|-----------|
| 1 | - | - | - | - | Formal | Subjective | Terse | Definitive |
| 2 | - | - | - | + | Formal | Subjective | Terse | Hedged |
| 3 | - | - | + | - | Formal | Subjective | Elaborate | Definitive |
| 4 | - | - | + | + | Formal | Subjective | Elaborate | Hedged |
| 5 | - | + | - | - | Formal | Meta | Terse | Definitive |
| 6 | - | + | - | + | Formal | Meta | Terse | Hedged |
| 7 | - | + | + | - | Formal | Meta | Elaborate | Definitive |
| 8 | - | + | + | + | Formal | Meta | Elaborate | Hedged |
| 9 | + | - | - | - | Casual | Subjective | Terse | Definitive |
| 10 | + | - | - | + | Casual | Subjective | Terse | Hedged |
| 11 | + | - | + | - | Casual | Subjective | Elaborate | Definitive |
| 12 | + | - | + | + | Casual | Subjective | Elaborate | Hedged |
| 13 | + | + | - | - | Casual | Meta | Terse | Definitive |
| 14 | + | + | - | + | Casual | Meta | Terse | Hedged |
| 15 | + | + | + | - | Casual | Meta | Elaborate | Definitive |
| 16 | + | + | + | + | Casual | Meta | Elaborate | Hedged |

## Mathematical Structure

### Quaternion Interpretation

The quaternion structure provides a natural 4D extension:

```
q = w + xi + yj + zk

Where:
  i² = j² = k² = ijk = -1
  
  The scalar part (w) is "real" — the grounding/certainty
  The vector part (xi + yj + zk) is "imaginary" — the style/perspective/depth
```

### Connection to Conformal Theory

The 4D conformal group has interesting properties:
- **Scale invariance**: Our φ-weighting
- **Rotations**: Style/perspective changes
- **Translations**: Depth changes
- **Special conformal**: Certainty changes (inversion-like)

The w-axis acts like a "conformal inversion" — it doesn't change the content, but changes our relationship to it (certain vs uncertain).

## Implementation

### QuaternionPhiDial Class

```python
class QuaternionPhiDial:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = max(-1.0, min(1.0, x))  # Style
        self.y = max(-1.0, min(1.0, y))  # Perspective
        self.z = max(-1.0, min(1.0, z))  # Depth
        self.w = max(-1.0, min(1.0, w))  # Certainty
    
    def get_certainty(self) -> str:
        if self.w < -0.3: return 'definitive'
        elif self.w > 0.3: return 'hedged'
        return 'neutral'
    
    def get_hexadecant(self) -> Tuple[str, str, str, str]:
        return (self.get_style(), self.get_perspective(), 
                self.get_depth(), self.get_certainty())
```

### Usage

```python
from truthspace_lcm import ConceptQA

# Definitive mode
qa = ConceptQA(certainty_w=-1)

# Hedged mode
qa = ConceptQA(certainty_w=1)

# Dynamic adjustment
qa.set_certainty(-1)  # Switch to definitive
qa.set_certainty(1)   # Switch to hedged
```

### CLI

```bash
python run.py --certainty -1   # Definitive
python run.py -w 1             # Hedged
```

### Interactive Commands

```
/certainty -1    # Set definitive mode
/certainty 1     # Set hedged mode
/dial            # Show all four dial settings
```

## Vocabulary

### Certainty Vocabulary

```python
CERTAINTY_VOCABULARY = {
    'copula': {
        'definitive': 'is undoubtedly',
        'neutral': 'is',
        'hedged': 'appears to be',
    },
    'relationship': {
        'definitive': 'closely tied to',
        'neutral': 'associated with',
        'hedged': 'possibly connected to',
    },
    'opener': {
        'definitive': ['Without question,', 'Certainly,', 'Undoubtedly,'],
        'neutral': [''],
        'hedged': ['Perhaps', 'Arguably,', 'It seems that'],
    },
}
```

## Design Decision: Opener Stacking

When both perspective and certainty have openers, we avoid stacking:
- If perspective opener exists (subjective/meta), use it
- Only use certainty opener when perspective is objective

This prevents awkward constructions like:
```
"Thematically, certainly, Holmes is..."  # BAD - stacked
"Thematically, Holmes is undoubtedly..." # GOOD - certainty in copula
```

## Why Certainty is Orthogonal

The w-axis is genuinely orthogonal to x, y, z because:

1. **Style (x)** changes WHAT words we use
2. **Perspective (y)** changes HOW we frame content
3. **Depth (z)** changes HOW MUCH we include
4. **Certainty (w)** changes HOW SURE we are

You can be:
- Formal AND definitive
- Formal AND hedged
- Casual AND definitive
- Casual AND hedged

These are independent dimensions.

## Connection to Epistemology

The w-axis maps to epistemic modality in linguistics:
- **Definitive**: Realis mood (factual)
- **Neutral**: Indicative mood (neutral)
- **Hedged**: Irrealis mood (uncertain, hypothetical)

This is a fundamental dimension of human language that exists independently of style, perspective, and detail level.

## Future Directions

1. **5th Dimension?** Could there be a 5th axis? Candidates:
   - Temporality (past/present/timeless)
   - Affect (neutral/emotional)
   - Agency (active/passive voice)

2. **Adaptive Certainty**: Adjust certainty based on confidence in the answer

3. **Query-Driven Certainty**: "Is Holmes a detective?" → definitive; "Was Holmes happy?" → hedged

## Conclusion

The 4D quaternion φ-dial provides complete geometric control over answer generation:

| Axis | Controls | Dimension |
|------|----------|-----------|
| **X** | WHAT words | Vocabulary |
| **Y** | HOW framed | Voice |
| **Z** | HOW MUCH detail | Density |
| **W** | HOW SURE | Modality |

The quaternion structure is mathematically natural and provides 16 discrete hexadecants or a continuous 4D hypercube of settings.

"Conformal theory inside conformal theory" — the w-axis acts as a conformal inversion on certainty, just as the x-axis acts on vocabulary, y on perspective, and z on depth.
