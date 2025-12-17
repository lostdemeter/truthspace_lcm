# AI Onboarding: TruthSpace LCM

**Purpose**: This document enables AI assistants (Claude, etc.) to quickly understand the project and continue development from where we left off.

**Last Updated**: December 16, 2025

---

## Quick Start: What Is This Project?

TruthSpace LCM (Large Concept Model) is an experimental AI architecture that operates on **concepts** rather than tokens, using **geometric encoding in hyperdimensional space**. The core thesis:

> **AI is fundamentally a geometric encoder-decoder. Meaning lives at intersection points in truth space, not in words.**

We've proven this works: **100% accuracy on 50 bash/Linux command queries** using pure geometric resolution with composable primitives.

---

## The Journey So Far

### Phase 1: Foundation (Early December 2025)

- Established the φ-based encoding framework
- Created the `PlasticEncoder` using the plastic constant (ρ ≈ 1.324718)
- Built the 12-dimensional truth space structure
- Defined semantic primitives as anchor points

### Phase 2: Semantic Disambiguation (Mid December 2025)

- Discovered that **synonyms converge perfectly (1.0 similarity)** in truth space
- Discovered that **Q/A pairs share dimensions** where the concept lives
- Implemented the BBP (Bailey-Borwein-Plouffe) inspired scoring approach

### Phase 3: Composable Primitive Resolution (December 16, 2025)

This is the breakthrough. Instead of matching queries to fixed concepts, we **compose** answers from primitive combinations:

```
Query: "copy file.txt to backup.txt"
  ↓ encode
Primitives: {COPY, FILE}
  ↓ match composition rule
Rule: (COPY, FILE) → cp
  ↓ resolve
Command: cp
```

**Key insight**: The composition rule `(COPY, FILE) → cp` is not arbitrary—it's the **name** we give to that intersection point in truth space. Different languages would have different names, but the intersection point is the same.

### Results Achieved

| Test Suite | Accuracy |
|------------|----------|
| Original 30 bash queries | 100% |
| Extended 50 queries (+ Linux tools) | 100% |

---

## Mathematical Background

### The Plastic Constant (ρ)

We use the plastic constant (ρ ≈ 1.324718) as our primary encoding constant:

```
ρ³ = ρ + 1
```

Why plastic over golden (φ)?
- **Slower growth** = finer discrimination between concepts
- **Cubic relationship** = richer structure in 3D+ spaces
- **Less "famous"** = fewer accidental collisions with other systems

### 12-Dimensional Truth Space

The encoding dimension (12) was chosen because:
- Aligns with the 12D clock for full phase coverage
- Provides sufficient orthogonality for semantic primitives
- Matches the φ-BBP dimensional structure

### Primitive Encoding

Each primitive has:
- **Dimension**: Which axis it lives on (0-11)
- **Level**: Position along that axis (0, 1, 2, ...)
- **Keywords**: Words that activate this primitive

Position calculation:
```python
position[dimension] = ρ^(-level)  # Plastic-weighted
```

---

## Current Architecture

### Core Files

```
truthspace_lcm/core/
├── encoder.py              # PlasticEncoder with ~25 primitives
├── composable_resolver.py  # Composition rules (80+ rules)
├── phi_lattice.py          # φ-lattice structure
├── intersection_resolver.py # Geometric intersection finder
└── signature_resolver.py   # Primitive signature matching
```

### Primitive Structure (Current)

```
ACTIONS (dims 0-4):
  CREATE, DESTROY           # Existence axis
  READ, WRITE               # Information flow
  COPY, RELOCATE, SEARCH    # Spatial operations
  COMPRESS, SORT, FILTER    # Transform operations
  CONNECT, EXECUTE          # Interaction

DOMAINS (dims 5-7):
  FILE, DIRECTORY           # Filesystem
  SYSTEM, STORAGE, MEMORY   # System resources
  TIME, UPTIME, HOST        # Temporal/identity
  PROCESS, DATA             # Runtime
  NETWORK, USER             # Access

MODIFIERS (dim 8):
  ALL, RECURSIVE, FORCE, VERBOSE

RELATIONS (dims 9-11):
  BEFORE, AFTER, DURING     # Temporal
  CAUSE, EFFECT             # Causal
  IF, ELSE                  # Conditional
```

### Composition Rules

Rules map primitive combinations to commands:

```python
(READ, FILE) → cat
(READ, FILE, BEFORE) → head
(READ, FILE, AFTER) → tail
(COPY, FILE) → cp
(RELOCATE, FILE) → mv
(SEARCH, FILE) → find
(SEARCH, DATA) → grep
(READ, STORAGE) → df
(READ, NETWORK) → ifconfig
(READ, PROCESS) → ps
...
```

### Scoring Algorithm

```python
def resolve(query):
    primitives = get_primitive_signature(query)
    
    for rule in rules:
        rule_set = set(rule.primitives)
        matches = len(rule_set & primitives)
        missing = len(rule_set - primitives)
        score = matches - (missing * 0.5)
        
        # Exact match bonus - critical for disambiguation
        if rule_set == primitives:
            score += 2
    
    return best_matching_rule
```

---

## Development Philosophy

### 1. Minimal Bootstrap

The code should only:
1. **Encode** - Convert input to geometric position
2. **Query** - Find nearest knowledge in TruthSpace
3. **Decode** - Convert knowledge back to executable output
4. **Execute** - Run the result

Everything else—patterns, intents, domain logic—should be **knowledge entries** with geometric positions.

### 2. Primitives Are Stable, Combinations Are Infinite

- ~25 primitives can cover hundreds of commands through composition
- Only split primitives when there's an unresolvable conflict
- Keywords can be expanded freely (safe)
- Composition rules are additive (no conflicts = no changes)

### 3. Geometry Does the Heavy Lifting

- Semantic proximity = geometric proximity
- Synonyms converge to the same point
- Q/A pairs share intersection dimensions
- Language is just I/O—the meaning lives in the geometry

### 4. Test-Driven Expansion

Every change must:
1. Pass all existing tests (no regressions)
2. Add new tests for new functionality
3. Validate with the 50-query test suite

---

## Where We're Going

### Immediate Next Steps

1. **Automated Knowledge Expansion** (Design doc 011)
   - Man page parser
   - Primitive inference engine
   - Conflict detection and resolution
   - Self-calibration loop

2. **Expand Test Suite**
   - More Linux commands
   - Edge cases and ambiguous queries
   - Multi-language support

### Medium Term

3. **Cross-Domain Expansion**
   - Python libraries and APIs
   - Git commands
   - Docker/Kubernetes
   - Domain-specific vocabularies

4. **Self-Learning Loop**
   - Read documentation → infer primitives → add rules → validate
   - No manual code edits required

### Long Term Vision

5. **Language-Agnostic Concept Space**
   - Same primitives, different language I/O filters
   - Spanish, Japanese, etc. map to same intersection points
   - Code generation in any language

6. **Emergent Reasoning**
   - Composition of compositions
   - Novel command synthesis
   - Semantic interpolation

---

## Key Design Documents

| Document | Purpose |
|----------|---------|
| `docs/PHILOSOPHY.md` | Core thesis and principles |
| `docs/TRUTHSPACE_LCM_PROPOSAL.md` | Original research proposal |
| `design_considerations/010_phi_dimensional_navigation.md` | φ-BBP and geometric resolution |
| `design_considerations/011_automated_knowledge_expansion.md` | Self-learning architecture |

---

## How to Continue Development

### Running Tests

```bash
cd /home/thorin/truthspace-lcm
source venv/bin/activate

# Run the composable resolver demo
python -m truthspace_lcm.core.composable_resolver

# Test a specific query
python -c "
from truthspace_lcm import PlasticEncoder
encoder = PlasticEncoder()
result = encoder.encode('show disk space')
print([p.name for p, _ in result.primitives])
"
```

### Adding a New Command

1. **Identify primitives**: What action + domain does it need?
2. **Check for conflicts**: Does the signature already exist?
3. **Add keywords** (if needed): Update `encoder.py`
4. **Add composition rule**: Update `composable_resolver.py`
5. **Test**: Verify no regressions

### Splitting a Primitive

When two commands have the same signature:

1. Analyze semantic difference
2. Create new primitive (e.g., SYSTEM → SYSTEM + STORAGE)
3. Update dimension assignments
4. Add keywords to new primitive
5. Update composition rules
6. Run full test suite

---

## Key Insights to Remember

1. **"Meaning lives at intersection points"** - The query and answer meet at the same geometric location

2. **"Primitives are the atoms of meaning"** - ~25 primitives cover 50+ commands

3. **"Composition is additive"** - Commands = combinations of primitives

4. **"Exact match disambiguates"** - +2 bonus for exact primitive match

5. **"The ribbon speech validation"** - When intersections produce coherent patterns, the truth space is aligned

6. **"Splitting resolves ambiguity"** - When commands collide, split the primitive

---

## Contact and Context

This project is being developed iteratively with AI assistance. Each session builds on the previous, with this document serving as the handoff point.

**Current state**: 100% accuracy on 50 queries, ready for automated expansion.

**Next priority**: Implement the self-calibration system from design doc 011.

---

*"The geometry does the heavy lifting. We just need to teach it where to look."*
