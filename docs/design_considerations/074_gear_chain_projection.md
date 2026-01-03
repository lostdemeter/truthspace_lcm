# Design Consideration 074: Gear Chain Projection

## Overview

This document describes the **Gear Chain Projection** methodology for transforming raw geometric truth output into polished natural language. Instead of using templates, patterns, or wave interference, we model the transformation as a chain of interlocking gears where each gear handles one specific transformation.

## The Problem

The geometric QA system produces raw truth output like:

```
Neutrons is a character who processes, seconds and causes. This relates to processe and energy.
```

This has several issues:
- "is a character" is inappropriate for abstract concepts
- Verb forms are inconsistent
- Structure is rigid and unnatural

Previous approaches tried:
1. **Templates** - Rigid, don't generalize
2. **Wave interference** - Too abstract, loses word order
3. **Nearest neighbor** - Inconsistent results
4. **Learned patterns** - Still template-like

## The Gear Chain Solution

### Core Insight

Instead of treating transformation as a single operation, we decompose it into a **chain of gears**:

```
Truth → [RoleGear] → [ActionGear] → [StructureGear] → [OutputGear] → Signal
```

Each gear:
- Has a **single responsibility**
- Receives **state** from the previous gear
- Applies **one transformation**
- Passes **modified state** to the next gear
- Has a **gear ratio** controlling transformation strength

### Why Gears?

The gear metaphor is apt because:

1. **Discrete meshing** - Semantic slots (entity, role, actions, targets) mesh like gear teeth
2. **Ratio control** - Gear ratios control how much each transformation applies
3. **Composition** - Gears chain naturally, each adding its transformation
4. **Reversibility** - Conceptually, gears can run backwards
5. **Quaternion encoding** - 4D quaternions encode transformation parameters

### The Gear Chain

#### 1. RoleGear

**Purpose**: Transform inappropriate roles to appropriate ones.

**Problem**: The corpus labels everything as "character" even abstract concepts.

**Transformation**:
- `character` → `concept` (for scientific terms)
- `someone` → `entity`
- Preserves `detective`, `doctor` for actual characters

**Quaternion encoding**:
- `x`: Person-ness (high = keep as character)
- `y`: Scientific-ness (high = transform to concept)

```python
# Detection heuristics
scientific_suffixes = ['ology', 'ics', 'istry', 'tion', 'ment', 'ness', 'ism']
person_names = {'holmes', 'watson', 'moriarty', 'lestrade'}
```

#### 2. ActionGear

**Purpose**: Transform verbs to appropriate forms.

**Problem**: Signal corpus prefers gerunds (-ing forms).

**Transformation**:
- `investigates` → `investigating`
- `processes` → `processing`
- Controlled by gear ratio

**Gear ratio effect**:
- `ratio=0.0`: Keep base forms ("confirms, articulates")
- `ratio=1.0`: Use gerunds ("confirming, articulating")

#### 3. StructureGear

**Purpose**: Determine sentence structure choices.

**Decisions**:
- **Prefix**: "X is" vs "It seems that X is"
- **Connector**: "who" vs "that" vs "that involves"
- **Target connector**: "particularly" vs "relating to"

**Rules**:
- Gerunds require "that involves" for grammar
- Person roles use "who"
- Abstract concepts use "that"

#### 4. OutputGear

**Purpose**: Assemble final output string.

**Assembly**:
```
{prefix} {article} {role} {connector} {actions}, {target_connector} {targets}.
```

### State Passing

State flows through the chain as a `GearState` object:

```python
@dataclass
class GearState:
    entity: str = ""
    role: str = "entity"
    actions: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    
    # Accumulated quaternion from gear chain
    accumulated_q: Quaternion = field(default_factory=Quaternion)
    
    # Style flags set by gears
    use_prefix: bool = False
    use_gerunds: bool = True
    connector: str = "that involves"
    target_connector: str = "particularly"
```

The **accumulated quaternion** carries style decisions through the chain, allowing later gears to make decisions based on earlier transformations.

## Results

### Before (Raw Truth)
```
Neutrons is a character who processes, seconds and causes. This relates to processe and energy.
```

### After (Gear Chain)
```
Neutrons is a concept that involves processing, seconding, and causing, particularly processe and energy.
```

### Gear Ratio Examples

```python
# Low action ratio - base verbs
projector.project("first", action_ratio=0.0)
# → "First is an entity that confirms, articulates, and presents"

# High action ratio - gerunds  
projector.project("first", action_ratio=1.0)
# → "First is an entity that involves confirming, articulating, and presenting"
```

## Advantages Over Previous Approaches

| Approach | Problem | Gear Chain Solution |
|----------|---------|---------------------|
| Templates | Rigid, don't generalize | Each gear is a transformation, not a template |
| Wave interference | Loses word order | Discrete slot meshing preserves structure |
| Learned patterns | Still template-like | Pure transformations, no patterns |
| Single gear | All-or-nothing | Fine-grained control via ratios |

## Connection to Quaternions

Each gear's transformation can be encoded as a quaternion:

- **w**: Transformation strength/confidence
- **x**: Primary transformation axis
- **y**: Secondary transformation axis
- **z**: Reserved/modifier

Quaternion multiplication chains transformations:
```python
state.accumulated_q = state.accumulated_q * gear.quaternion
```

This allows the accumulated transformation to influence later decisions.

## Future Extensions

### Additional Gears

- **TenseGear**: Transform verb tenses
- **FormalityGear**: Adjust formality level
- **VerbosityGear**: Control output length
- **DomainGear**: Adapt to specific domains (scientific, narrative, etc.)

### Gear Learning

Instead of hand-coded transformations, gears could learn their quaternion parameters from the signal corpus:

```python
gear.learn_from_corpus(signal_texts)
```

### Bidirectional Chains

Run the chain backwards to transform signal → truth (for analysis).

## Implementation

### Files

- `experiments/gear_chain_projection.py` - Main implementation
- `experiments/slot_gear_projection.py` - Single-gear variant
- `experiments/quaternion_gears.py` - Quaternion utilities

### Usage

```python
from experiments.gear_chain_projection import GearChainProjector

projector = GearChainProjector(
    'truthspace_lcm/corpus_experimental.json',
    'truthspace_lcm/corpus_signal_full.json'
)

# Default ratios
result = projector.project('evolution')

# Custom ratios
result = projector.project('evolution', 
    role_ratio=1.0,
    action_ratio=0.5,
    structure_ratio=1.0
)
```

## Conclusion

The gear chain methodology provides a clean, composable, and tunable approach to output polishing. By decomposing the transformation into discrete gears, we achieve:

1. **Modularity** - Each gear has one job
2. **Tunability** - Gear ratios control each transformation
3. **Composability** - Add/remove/reorder gears as needed
4. **Geometric purity** - No templates, just transformations
5. **Quaternion encoding** - 4D state flows through the chain

This approach aligns with the project's goal of geometric language processing without relying on morphological patterns or semantic similarity.
