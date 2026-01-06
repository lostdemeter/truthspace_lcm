# Design Consideration 103: Self-Assembling Primitives

## Date: 2026-01-06

## Status: DESIGN

## The Problem

Our φ-lattice geometry is pure, but the outside world isn't. We need a system that:

1. **Transforms chaotic input into geometric structure**
2. **Self-assembles** - doesn't require manual primitive definition
3. **Works for bootstrap** (known keywords → known positions)
4. **Works for ingestion** (unknown text → discovered positions)

The keyword boost we implemented is a **fallback**, not a transformation. It pattern-matches strings instead of transforming them to geometry. This violates "Geometry IS computation."

## The Insight

From the memories and prior work:

> **Attractor/Repeller Dynamics**: Words that appear together attract. Words in different contexts repel. The vocabulary EMERGES from dynamics, not design.

> **Symmetry Determines Naming**: Languages name PAIRS, not isolated positions. The fundamental unit is the symmetric pair.

> **Forward Projection**: The structure IS the knowledge. We can build concept spaces without ever seeing text - just seeds + transforms.

The solution isn't to match keywords - it's to **transform keywords into geometric positions** that then participate in the lattice dynamics.

## The Design: Keyword-to-Primitive Bridge

### Core Principle

Every piece of text that enters the system must be **transformed to geometry immediately**. No text should participate in matching as text - only as position.

```
TEXT → PRIMITIVES → LEVELS → POSITION → GEOMETRY
```

### Phase 1: Bootstrap Transformation

At bootstrap time, transform keywords into primitives:

```python
# Bootstrap knowledge has:
{
    "text": "Physics is the natural science...",
    "keywords": ["physics", "what is physics"],
    "phi_levels": [3, 2, 1, 1]
}

# Transform keywords to primitives:
for keyword in item["keywords"]:
    # The keyword inherits the concept's phi_levels
    register_primitive(
        keyword=keyword,
        levels=item["phi_levels"],
        source="bootstrap"
    )
```

Now "what is physics" isn't a pattern to match - it's a **primitive that activates levels [3, 2, 1, 1]**.

### Phase 2: Query Encoding

When a query comes in:

```python
query = "what is physics?"

# Tokenize
words = ["what", "is", "physics"]

# Check for multi-word primitives first (longer = more specific)
# "what is physics" matches → levels [3, 2, 1, 1]

# If no multi-word match, fall back to single-word primitives
# "physics" → PHYSICS primitive → domain=3
# "what" → INFORM primitive → intent=1
# "is" → INFORM primitive → intent=1

# MAX aggregation per dimension
final_levels = [3, 0, 1, 0]  # or [3, 2, 1, 1] if multi-word matched
```

### Phase 3: Self-Assembly for New Data

When new text arrives (not bootstrap):

```python
def ingest_text(text: str, context: dict = None):
    """
    Transform new text into geometric structure.
    
    The structure self-assembles through:
    1. Primitive activation (what we know)
    2. Position inference (where it lands)
    3. Attractor dynamics (where it settles)
    4. Primitive emergence (what it teaches us)
    """
    
    # 1. Encode with known primitives
    position, levels = encoder.encode_with_levels(text)
    
    # 2. Find nearest existing concepts (attractors)
    neighbors = find_neighbors(position, k=3)
    
    # 3. If neighbors agree, position is stable
    # If neighbors disagree, position is at a boundary
    stability = compute_stability(position, neighbors)
    
    # 4. If stable and novel, extract new primitives
    if stability > threshold and is_novel(text, neighbors):
        # The text contains information not in our primitives
        # Extract candidate primitives from the gap
        new_primitives = extract_primitives(text, position, neighbors)
        
        # Register them (they become part of the geometry)
        for prim in new_primitives:
            register_primitive(prim)
```

### The Self-Assembly Mechanism

The key is **attractor/repeller dynamics**:

1. **Known primitives** define attractor basins
2. **New text** lands somewhere based on primitive activation
3. **Neighbors** pull it toward stable positions
4. **Gaps** (text that lands far from attractors) signal missing primitives
5. **Primitive extraction** fills the gaps

This is how the structure self-assembles:
- Bootstrap provides the initial attractors
- New data either falls into existing basins or creates new ones
- The geometry grows organically

### Multi-Word Primitive Handling

Keywords like "what is physics" should be treated as **phrases**, not word bags:

```python
class PrimitiveRegistry:
    def __init__(self):
        self.single_word = {}  # word → Primitive
        self.multi_word = {}   # tuple(words) → Primitive
        self.phrase_index = {} # first_word → [phrases starting with it]
    
    def register(self, keyword: str, levels: List[int], source: str):
        words = tokenize(keyword)
        
        if len(words) == 1:
            self.single_word[words[0]] = Primitive(keyword, levels, source)
        else:
            key = tuple(words)
            self.multi_word[key] = Primitive(keyword, levels, source)
            # Index by first word for efficient lookup
            self.phrase_index.setdefault(words[0], []).append(key)
    
    def encode(self, text: str) -> Tuple[np.ndarray, List[int]]:
        words = tokenize(text)
        levels = [0] * 4  # Default
        activated = [False] * 4
        
        i = 0
        while i < len(words):
            # Try multi-word matches first (greedy, longest match)
            matched = False
            if words[i] in self.phrase_index:
                for phrase in sorted(self.phrase_index[words[i]], 
                                    key=len, reverse=True):
                    if self._matches_phrase(words, i, phrase):
                        prim = self.multi_word[phrase]
                        # Multi-word primitive sets ALL dimensions
                        for dim, level in enumerate(prim.levels):
                            if not activated[dim] or level > levels[dim]:
                                levels[dim] = level
                                activated[dim] = True
                        i += len(phrase)
                        matched = True
                        break
            
            if not matched:
                # Single word primitive
                if words[i] in self.single_word:
                    prim = self.single_word[words[i]]
                    dim = prim.dimension
                    if not activated[dim] or prim.level > levels[dim]:
                        levels[dim] = prim.level
                        activated[dim] = True
                i += 1
        
        position = levels_to_position(levels)
        return position, levels
```

## The Transformation, Not Matching

The critical difference:

**OLD (Pattern Matching - IMPURE):**
```python
# Query: "what is physics?"
# Concept keywords: ["physics", "what is physics"]
# Match: "what is physics" in query → boost similarity
```

**NEW (Geometric Transformation - PURE):**
```python
# At bootstrap:
# "what is physics" → register as primitive with levels [3, 2, 1, 1]

# At query time:
# "what is physics?" → encode → finds primitive → levels [3, 2, 1, 1]
# Concept position: [3, 2, 1, 1]
# Distance: 0 (exact match in geometry)
```

The text is **transformed to geometry** at registration time. At query time, we're comparing **geometry to geometry**, not text to text.

## Connection to Prior Work

### Attractor Dynamics (Memory)
> Words that appear together attract. Words in different contexts repel.

Multi-word primitives ARE attractors. "what is physics" attracts queries about physics. The primitive defines the basin.

### Holographic Pattern Space (Memory)
> Positions are CONSTRUCTED from similarity.

We're constructing positions from primitives. The primitive registry IS the holographic projection - it defines where things land.

### Forward Projection (Memory)
> Seeds + Transforms → Generate → Verify

Bootstrap keywords are seeds. The primitive registry is the transform. New data is generated (positioned) and verified (stability check).

## Implementation Plan

### Step 1: PrimitiveRegistry Class
- Single-word and multi-word primitive storage
- Efficient phrase matching
- Registration from bootstrap keywords

### Step 2: Modify Bootstrap Loading
- Transform keywords to primitives at load time
- Each keyword inherits concept's phi_levels

### Step 3: Replace Keyword Boost
- Remove pattern matching from query_text
- Use PrimitiveRegistry for encoding
- Pure geometric distance/similarity

### Step 4: Add Ingestion Pipeline
- Stability detection for new text
- Primitive extraction from gaps
- Self-assembly dynamics

## Success Criteria

1. **100% accuracy** without keyword boost (pure geometry)
2. **Bootstrap keywords** become primitives (transformed, not matched)
3. **New data** can be ingested and positioned geometrically
4. **Primitives emerge** from ingested data (self-assembly)

## The Principle

> **Bootstrap is acceptable. Fallbacks are not.**
> **Transform everything to geometry. Match nothing as text.**

The outside world is chaotic. Our job is to transform that chaos into geometric structure. The transformation IS the intelligence - not pattern matching, not fallbacks, but geometric transformation.

---

*"The geometry is pure. The transformation makes the chaos pure too."*
