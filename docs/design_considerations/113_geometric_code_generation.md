# Design 113: Geometric Code Generation

## The Problem

Current `PlotSpace` and `CodeSpace` use **template substitution**, not true code generation:

```python
# Current approach (template)
template = 'print("{message}")'
code = template.format(message="Hello")

# True generation (geometric)
# Position in code-space → token sequence → valid Python
```

This violates the core hypothesis: **Structure IS information**. Templates are hard-coded music in the comb.

## The Vision

Code generation should follow the Music Box Principle:
- **Python tokens** have positions in semantic space (the drum)
- **Syntax rules** are geometric constraints (valid paths)
- **Programs** emerge from traversing the space (the music)

## Requirements Analysis

### What We Need

| Feature | Description | Difficulty |
|---------|-------------|------------|
| **Multi-line chat input** | Accept code blocks in chat | Low |
| **Python token vocabulary** | Geometric positions for keywords, operators, etc. | Medium |
| **Syntax-aware generation** | Only produce valid token sequences | High |
| **Variable context** | Track what's defined in scope | Medium |
| **Compositional assembly** | Combine primitives into programs | High |

### What We Have

| Feature | Status | Location |
|---------|--------|----------|
| Geometric vocabulary | ✅ Done | `geometric_vocabulary.py` |
| Music box transformation | ✅ Done | `perspective.py`, `transformation_space.py` |
| Template matching | ✅ Done | `code_space.py`, `plot_space.py` |
| Code verification | ✅ Done | `CodeVerifier` class |
| Intent detection | ✅ Done | ACTION dimension |

## Proposed Architecture

### Phase 1: Multi-line Chat (Foundation)

Enable multi-line input in HyperChat:
- Detect code blocks (triple backticks or indentation)
- Buffer lines until block complete
- Pass complete blocks to pipeline

### Phase 2: Python Token Vocabulary

Create `PythonVocabulary` extending `GeometricVocabulary`:

```python
# Dimensions for Python tokens
# [category, scope, side_effect, complexity]

PYTHON_TOKENS = {
    # Keywords - control flow
    "if": [0, 0, 0, 0],      # control, local, pure, simple
    "else": [0, 0, 0, 0],
    "elif": [0, 0, 0, 0],
    "for": [0, 0, 0, 1],     # control, local, pure, medium
    "while": [0, 0, 0, 1],
    "def": [1, 1, 0, 2],     # definition, creates scope, pure, complex
    "class": [1, 2, 0, 3],   # definition, creates class scope, pure, very complex
    "return": [0, -1, 0, 0], # control, exits scope, pure, simple
    
    # Keywords - data
    "True": [2, 0, 0, 0],    # literal, local, pure, simple
    "False": [2, 0, 0, 0],
    "None": [2, 0, 0, 0],
    
    # Keywords - import
    "import": [3, 2, 1, 1],  # import, module scope, side effect, medium
    "from": [3, 2, 1, 1],
    
    # Operators
    "+": [4, 0, 0, 0],       # operator, local, pure, simple
    "-": [4, 0, 0, 0],
    "*": [4, 0, 0, 0],
    "/": [4, 0, 0, 0],
    "=": [4, 0, 1, 0],       # operator, local, side effect (assignment), simple
    
    # Built-ins
    "print": [5, 0, 1, 0],   # builtin, local, side effect (output), simple
    "len": [5, 0, 0, 0],     # builtin, local, pure, simple
    "range": [5, 0, 0, 0],
    "open": [5, 0, 1, 1],    # builtin, local, side effect (file), medium
}
```

### Phase 3: Syntax as Geometric Constraints

Valid Python follows patterns. These patterns are **paths through token space**:

```
STATEMENT → EXPRESSION | ASSIGNMENT | CONTROL | DEFINITION
ASSIGNMENT → NAME "=" EXPRESSION
CONTROL → "if" EXPRESSION ":" BLOCK
DEFINITION → "def" NAME "(" PARAMS ")" ":" BLOCK
```

Each production rule is a **valid direction** from a position:
- From `if`, you can go to EXPRESSION
- From `def`, you can go to NAME
- From `=`, you can go to EXPRESSION

### Phase 4: Compositional Generation

Generate code by:
1. Start at query position (what user wants)
2. Find nearest valid starting token
3. Follow syntax-constrained path
4. Accumulate tokens until complete statement

## Experiments Needed

### Experiment 1: Multi-line Chat
- Add code block detection to chat input
- Test with simple multi-line examples
- Verify blocks are passed intact to pipeline

### Experiment 2: Token Vocabulary
- Create `PythonVocabulary` with positioned tokens
- Test nearest-neighbor for token prediction
- Verify semantic clustering (all control flow tokens near each other)

### Experiment 3: Syntax Constraints
- Implement simple grammar as position constraints
- Test that invalid sequences are rejected
- Measure how often valid Python is produced

### Experiment 4: Simple Generation
- Generate single statements from descriptions
- "print hello" → `print("hello")`
- "add x and y" → `x + y`

## Connection to Core Hypothesis

> **Structure IS information** - There are no opaque weights or embeddings

If we can generate valid Python code using only:
- Token positions (structure)
- Syntax constraints (structure)
- Nearest-neighbor traversal (geometry)

Then we prove that code generation doesn't require neural networks - it requires the right geometric structure.

## Implementation Path

1. **Experiment: Multi-line chat** - Enable code block input
2. **Experiment: Python vocabulary** - Position tokens geometrically
3. **Experiment: Syntax paths** - Constrain generation to valid sequences
4. **Experiment: Simple statements** - Generate basic Python
5. **Integration** - Replace PlotSpace with geometric generation

## The Test

A system passes the geometric code generation test if:

1. **No templates are stored** - Code emerges from token positions
2. **Syntax is geometric** - Valid sequences are paths, not rules
3. **Output is valid Python** - Verified by AST parsing

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Syntax too complex | Start with subset (expressions only) |
| Token space too sparse | Bootstrap with common patterns |
| Generation too slow | Cache common paths |
| Invalid code produced | Verify before returning |

## Next Steps

1. Create `experiments/multiline_chat.py` - Test multi-line input
2. Create `experiments/python_vocabulary.py` - Token positions
3. Create `experiments/syntax_paths.py` - Grammar as geometry

---

*The code emerges from the geometry. The syntax is the path. The program is the music.*
