# Design Consideration 102: φ-Lattice Purity Audit

## Date: 2026-01-06

## Status: AUDIT COMPLETE

## The Maxims

From PROJECT_OVERVIEW.md:
> **LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

We aim to prove this by building a system where:
- **Structure IS information** - No opaque weights or embeddings
- **Geometry IS computation** - Traversal through geometric space produces outputs
- **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure

---

## Audit Scope

Audited files from the φ-lattice implementation:
- `core/phi_lattice.py`
- `core/semantic_dimensions.py`
- `core/primitives.py`
- `core/phi_encoder.py`
- `core/knowledge_space.py` (modifications)
- `corpus/bootstrap_knowledge.json` (modifications)

---

## Summary

| Category | Items | Severity | Status |
|----------|-------|----------|--------|
| ✅ GEOMETRIC | 6 | - | Good |
| ⚠️ BOOTSTRAP (acceptable) | 3 | LOW | Acceptable |
| ❌ NON-GEOMETRIC | 2 | HIGH | **Needs work** |

---

## ✅ GEOMETRIC (Good)

### `core/phi_lattice.py`

**Verdict: ✅ FULLY GEOMETRIC**

- ✅ Positions at φ^k - mathematically verifiable
- ✅ Distance is weighted Euclidean - pure geometry
- ✅ Similarity is 1/(1+distance) - geometric transformation
- ✅ Snapping to lattice uses log(φ) - geometric operation
- ✅ No hardcoded special cases
- ✅ No opaque weights

```python
# Good: Pure geometric operations
def levels_to_position(self, levels: List[int]) -> np.ndarray:
    return np.array([PHI ** k for k in levels])

def distance(self, a: np.ndarray, b: np.ndarray, weights=None) -> float:
    diff = (a - b) * np.sqrt(weights)
    return float(np.linalg.norm(diff))
```

### `core/semantic_dimensions.py`

**Verdict: ✅ GEOMETRIC with acceptable bootstrap**

- ✅ Weights are φ-powers: φ², φ, 1, φ⁻¹
- ✅ Level meanings are documentation only (not used in computation)
- ⚠️ The 4 dimensions are bootstrapped (acceptable)

```python
# Good: Weights follow φ-power pattern
DOMAIN.weight = PHI ** 2      # ≈ 2.618
SPECIFICITY.weight = PHI      # ≈ 1.618
INTENT.weight = 1.0           # φ^0
FORMALITY.weight = PHI ** -1  # ≈ 0.618
```

### `core/phi_encoder.py`

**Verdict: ✅ GEOMETRIC**

- ✅ MAX aggregation per dimension (Sierpinski property)
- ✅ Position computed from levels via φ^k
- ✅ Distance/similarity delegated to lattice
- ⚠️ Tokenization is simple word extraction (acceptable)

```python
# Good: MAX aggregation is geometric (Sierpinski)
for word in words:
    if word in self.keyword_map:
        prim = self.keyword_map[word]
        if not activated[dim] or prim.level > levels[dim]:
            levels[dim] = prim.level  # MAX
```

### `core/knowledge_space.py` - φ-Lattice Mode

**Verdict: ⚠️ MIXED - see NON-GEOMETRIC section**

The φ-lattice geometric core is good:
- ✅ Positions from explicit phi_levels
- ✅ Distance via φ-lattice geometry
- ✅ Similarity from distance

But the keyword boost is problematic (see below).

---

## ⚠️ BOOTSTRAP (Acceptable)

These are bootstrapped elements that are immediately transformed to geometry:

### 1. Semantic Dimensions (4 axes)

```python
DEFAULT_DIMENSIONS = [DOMAIN, SPECIFICITY, INTENT, FORMALITY]
```

**Why acceptable:** These define the coordinate system. Like choosing x/y/z axes - a necessary bootstrap. The dimensions themselves don't compute anything; they just define the space.

### 2. Primitives (keyword → level mappings)

```python
Primitive("PHYSICS", dimension=0, level=3, keywords=["physics", "quantum", ...])
```

**Why acceptable:** Primitives are the bootstrap seed. They are immediately transformed to geometric positions (φ^level). The keywords are just triggers - the geometry does the work.

### 3. phi_levels in bootstrap_knowledge.json

```json
{"topic": "physics", "phi_levels": [3, 2, 1, 1], ...}
```

**Why acceptable:** These are explicit position assignments for bootstrap concepts. Like placing landmarks on a map. The positions are geometric; the assignment is bootstrap.

---

## ❌ NON-GEOMETRIC (Needs Work)

### 1. Keyword Boost in `query_text` (SEVERITY: HIGH)

**Location:** `knowledge_space.py` lines 557-576

```python
# PROBLEM: This is pattern matching, not geometry
keyword_boost = 0.0
keywords = mapping.metadata.get("keywords", [])
if keywords:
    for kw in keywords:
        kw_words = set(kw_lower.split())
        overlap = len(query_words & kw_words)
        if overlap > 0 and overlap >= len(kw_words):
            keyword_boost = max(keyword_boost, 0.5 + 0.1 * len(kw_words))
```

**Why it violates the maxims:**
- This is **string pattern matching**, not geometry
- The boost values (0.5, 0.1, 0.3) are **arbitrary magic numbers**
- It's a **fallback** that masks geometric failures
- It violates "Geometry IS computation"

**The uncomfortable truth:** Without keyword boost, accuracy drops from 100% to 38%. The geometry alone isn't working.

### 2. Keyword Boost in `_query_phi_lattice` (SEVERITY: HIGH)

**Location:** `knowledge_space.py` lines 638-655

Same issue as above - duplicated in φ-lattice mode.

---

## Root Cause Analysis

### Why does geometry alone fail?

The φ-lattice encodes queries like this:

| Query | Encoded Levels |
|-------|---------------|
| "what is python?" | [2, 0, 1, 0] |
| "what is physics?" | [3, 0, 1, 0] |
| "hello" | [-1, 0, 0, 0] |

But the bootstrap concepts have these phi_levels:

| Concept | phi_levels |
|---------|-----------|
| Python | [2, 2, 1, 0] |
| Physics | [3, 2, 1, 1] |
| Greeting | [-1, -1, -1, -1] |

**The mismatch:** Query "what is python?" encodes to [2, 0, 1, 0] but Python concept is at [2, 2, 1, 0]. The specificity dimension (index 1) differs.

**Why?** The query doesn't contain words that trigger specificity primitives. "what is python" has:
- "what" → VERY_GENERAL (specificity=0)
- "is" → INFORM (intent=1)
- "python" → PROGRAMMING (domain=2)

No word triggers specificity level 2.

### The Geometric Solution

The keyword boost is a **workaround** for insufficient primitive coverage. The geometric solution would be:

1. **More primitives** - Add primitives that capture query patterns
2. **Keyword-to-primitive mapping** - Bootstrap keywords should map to primitives
3. **Query expansion** - Use synonyms geometrically

---

## Recommendations

### Option A: Accept the Hybrid (Pragmatic)

Keep keyword boost but document it as a **bootstrap assist**, not core geometry:

```python
# BOOTSTRAP ASSIST: Keyword matching helps bridge query encoding gaps
# This is acceptable bootstrap, not core geometry
# TODO: Replace with primitive expansion when coverage improves
```

**Pros:** Works now, 100% accuracy
**Cons:** Violates purity, masks geometric failures

### Option B: Expand Primitives (Geometric)

Add primitives that capture query patterns:

```python
# Query pattern primitives
Primitive("QUERY_PYTHON", 1, 2, ["python", "what is python"]),
Primitive("QUERY_PHYSICS", 1, 2, ["physics", "what is physics"]),
```

**Pros:** Pure geometry
**Cons:** Requires many primitives, approaches hardcoding

### Option C: Keyword-to-Primitive Bridge (Hybrid Geometric)

Transform bootstrap keywords into primitives at load time:

```python
# At bootstrap, convert keywords to primitives
for item in knowledge_items:
    for kw in item["keywords"]:
        # Create primitive from keyword → concept's phi_levels
        create_primitive_from_keyword(kw, item["phi_levels"])
```

**Pros:** Keywords become geometric, not pattern matching
**Cons:** Complexity, potential conflicts

### Option D: Fail Fast (Purist)

Remove keyword boost entirely. Accept 38% accuracy. Use failures to improve primitives.

**Pros:** Pure geometry, exposes real issues
**Cons:** System doesn't work well

---

## Verdict

### Current State: ⚠️ IMPURE

The φ-lattice implementation is **geometrically sound** in its core, but the **keyword boost is a non-geometric fallback** that violates the project maxims.

### Recommended Action: Option C

Transform keywords to primitives at bootstrap time. This keeps the geometric purity while leveraging bootstrap keywords:

1. Keywords become primitives (geometric)
2. Primitives activate dimensions (geometric)
3. Positions computed from levels (geometric)
4. Distance/similarity is geometric

This maintains the principle: **Bootstrap is acceptable, fallbacks are not.**

---

## Files Needing Work

| File | Issue | Severity |
|------|-------|----------|
| `knowledge_space.py` | Keyword boost in `query_text` | HIGH |
| `knowledge_space.py` | Keyword boost in `_query_phi_lattice` | HIGH |

---

## Conclusion

The φ-lattice implementation achieves 100% accuracy but **relies on non-geometric keyword matching**. This is a pragmatic solution but violates the project's core principle that "Geometry IS computation."

The geometric core is sound:
- φ^k positions ✅
- Semantic dimensions ✅
- MAX aggregation ✅
- Distance/similarity ✅

The non-geometric part:
- Keyword boost ❌

**Next step:** Implement Option C (keyword-to-primitive bridge) to make the system fully geometric while maintaining accuracy.

---

*"A fallback that works is still a fallback. The geometry should work on its own."*
