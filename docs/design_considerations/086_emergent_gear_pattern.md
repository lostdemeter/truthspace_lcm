# 086: The Emergent Gear Pattern

## The Recurring Pattern

Across the codebase, we keep implementing the same fundamental pattern:

```
┌─────────────────────────────────────────────────────────────┐
│  1. STRUCTURE (the information IS the structure)            │
│     - Define what the space looks like                      │
│     - Patterns, signatures, templates, modules              │
│                                                             │
│  2. BOOTSTRAP (we don't have all the data yet)              │
│     - Seed with initial examples                            │
│     - Use LLM to generate missing pieces                    │
│     - Fill in the structure with real content               │
│                                                             │
│  3. MATCH (find the right structure for input)              │
│     - Project input into the space                          │
│     - Find nearest/best matching structure                  │
│     - Return confidence score                               │
│                                                             │
│  4. COMPOSE (adapt structure to specific request)           │
│     - Extract parameters from input                         │
│     - Modify the matched structure                          │
│     - Generate output that fits the request                 │
│                                                             │
│  5. LEARN (self-improve from usage)                         │
│     - Record successes and failures                         │
│     - Promote temporary structures to permanent             │
│     - Refine signatures/patterns based on data              │
└─────────────────────────────────────────────────────────────┘
```

## Examples in the Codebase

### PythonCodeGear
- **Structure**: `CodePattern` with templates and contact points
- **Bootstrap**: `_seed_patterns()` with basic patterns
- **Match**: Find pattern by contact point similarity
- **Compose**: `fill()` template with parameters
- **Learn**: Track `use_count` and `success_count`

### EmergentClassifier
- **Structure**: `CategorySignature` with frequency/position/morphology signals
- **Bootstrap**: Seed words for each category
- **Match**: `score()` word against signature
- **Compose**: Classify word into categories
- **Learn**: Update `word_stats` from data

### HolographicPatternSpace
- **Structure**: `HolographicModule` with word sets and positions
- **Bootstrap**: Seed modules, use LLM for unknowns
- **Match**: `find_best_match()` via geometric projection
- **Compose**: **MISSING** - this is the gap!
- **Learn**: `promote_temporary()` after success

### PlotCorpus
- **Structure**: `PlotPattern` with keywords and templates
- **Bootstrap**: Seed patterns for common plots
- **Match**: `find_pattern()` by keyword overlap
- **Compose**: `_apply_params_to_code()` - limited!
- **Learn**: `_learn_from_generation()` adds new patterns

## The Missing Piece: Composition

The current systems can **match** but struggle to **compose**. When a user says:

> "create a sine wave plot, but with the results being x+0.5"

We correctly match `sine_wave`, but we don't:
1. Parse the modification intent ("results being x+0.5")
2. Understand what part of the template to modify
3. Apply the modification correctly

## Proposed Solution: Template Composition Layer

A unified layer that can:

### 1. Parse Modification Intent
```python
def parse_modification(request: str, base_template: str) -> List[Modification]:
    """
    Extract what the user wants to change.
    
    "results being x+0.5" → Modification(target='y', operation='add', value=0.5)
    "amplitude of 2" → Modification(target='amplitude', operation='set', value=2)
    "in red" → Modification(target='color', operation='set', value='red')
    """
```

### 2. Locate Modification Points
```python
def find_modification_points(template: str) -> Dict[str, ASTNode]:
    """
    Parse template and identify what can be modified.
    
    For sine_wave template:
    - 'y' → the y = amplitude * np.sin(...) line
    - 'amplitude' → the amplitude parameter
    - 'color' → the 'b-' in plt.plot()
    """
```

### 3. Apply Modifications
```python
def apply_modification(template: str, mod: Modification) -> str:
    """
    Apply the modification to the template.
    
    Modification(target='y', operation='add', value=0.5)
    → Change "y = amplitude * np.sin(frequency * x)" 
    → To "y = amplitude * np.sin(frequency * x) + 0.5"
    """
```

## Connection to Holographic Projection

The composition layer fits naturally into the holographic model:

1. **Base Module** = the template (sine_wave)
2. **Modification** = a vector in the space
3. **Composed Result** = base + modification projected back to code

The modification intent creates a "delta" that we apply to the base structure.

## Implementation Strategy

### Phase 1: Modification Parser
- Use structural patterns to detect modification intent
- "but with X" → modification follows
- "shifted by N" → offset modification
- "in COLOR" → style modification

### Phase 2: Template Annotator
- Parse templates to find modifiable points
- Annotate with semantic meaning (y_value, color, title, etc.)
- Store annotations with the module

### Phase 3: Modification Applicator
- AST-level modifications for code templates
- Safe transformations that preserve structure
- Fallback to LLM for complex modifications

### Phase 4: Learning
- When LLM handles a modification, learn the pattern
- "x+0.5" → y_offset=0.5 becomes a known transformation
- Build vocabulary of modifications over time

## The Unified Gear Interface

```python
class EmergentGear(Protocol):
    """The unified interface for all emergent gears."""
    
    # Structure
    def define_structure(self, spec: StructureSpec) -> None: ...
    
    # Bootstrap
    def seed(self, examples: List[Example]) -> None: ...
    def bootstrap_with_llm(self, llm: LLMInterface) -> None: ...
    
    # Match
    def match(self, input: Any) -> Tuple[Structure, float]: ...
    
    # Compose
    def compose(self, structure: Structure, modifications: List[Mod]) -> Any: ...
    
    # Learn
    def record_outcome(self, success: bool) -> None: ...
    def promote(self, temporary: Structure) -> None: ...
```

## Files

- Pattern: `python_code_gear.py`, `emergent_classifier.py`, `holographic_pattern_space.py`
- Gap: Composition layer missing from holographic space
- Next: Implement `TemplateComposer` class
