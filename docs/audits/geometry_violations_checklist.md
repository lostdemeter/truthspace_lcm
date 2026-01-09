# Geometry Violations Checklist

**Created:** January 9, 2026  
**Principle:** Structure IS information, Geometry IS computation, The shape IS the knowledge

---

## Progress Tracker

| # | File | Issue | Status | Notes |
|---|------|-------|--------|-------|
| 1 | `holographic_transformer.py:64-68` | Hard-coded FILLER_WORDS | ✅ DONE | Derived from document frequency |
| 2 | `chat_pipeline.py:212-233` | Keyword shortcuts bypass geometry | ⬜ TODO | Remove knowledge_prefixes and plot_keywords |
| 3 | `knowledge_space.py:560-579` | Keyword boost on top of geometry | ⬜ TODO | Use PrimitiveRegistry.encode() instead |
| 4 | `intent_classifier.py:46-132` | Regex patterns as primary classifier | ⬜ TODO | Replace with GeometricIntentClassifier |
| 5 | `geometric_intent_classifier.py:104-107` | Hard-coded FILLER set | ⬜ TODO | Derive from frequency distribution |
| 6 | `plot_space.py:499-502` | Keyword matching for plots | ⬜ TODO | Transform keywords to primitives |

---

## Detailed Tasks

### 1. holographic_transformer.py - FILLER_WORDS 🔄

**Current Code (lines 64-68):**
```python
FILLER_WORDS = {
    'a', 'an', 'the', 'of', 'with', 'for', 'to', 'and', 'or', 'in',
    'that', 'this', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'it', 'its', 'you', 'your', 'i', 'my', 'me', 'we', 'our', 'they',
}
```

**Problem:** Hard-coded word list violates "Structure IS information"

**Solution:** Derive filler words from corpus statistics:
- High frequency (top N% by count)
- Uniform distribution across contexts
- Short length (typically ≤4 chars)

**Approach:**
1. Build word frequency from transformation corpus
2. Identify words that appear in >50% of sentences
3. These are structurally "filler" - they don't carry transformation-specific meaning

**Status:** 🔄 IN PROGRESS

---

### 2. chat_pipeline.py - Keyword Shortcuts ⬜

**Current Code (lines 212-233):**
```python
knowledge_prefixes = ['what is', 'what are', 'who is', ...]
if any(query_lower.startswith(prefix) for prefix in knowledge_prefixes):
    return IntentResult(intent=Intent.KNOWLEDGE, ...)

plot_keywords = ['plot', 'graph', 'chart', 'sine', ...]
if any(kw in query_lower for kw in plot_keywords):
    return IntentResult(intent=Intent.PLOT_GENERATION, ...)
```

**Problem:** String matching bypasses geometric intent detection

**Solution:** Remove these shortcuts entirely. Let geometric matching handle all cases.

**Status:** ⬜ TODO

---

### 3. knowledge_space.py - Keyword Boost ⬜

**Current Code (lines 560-579):**
```python
keyword_boost = 0.0
keywords = mapping.metadata.get("keywords", [])
# ... keyword matching logic ...
similarity = geo_similarity + keyword_boost
```

**Problem:** Keyword matching compensates for geometric failures

**Solution:** Use `PrimitiveRegistry.encode()` for all queries (already shown in lines 617-639)

**Status:** ⬜ TODO

---

### 4. intent_classifier.py - Regex Patterns ⬜

**Current Code (lines 46-132):**
```python
GOOSE_TOOL_PATTERNS = { ... }
CODE_GENERATION_PATTERNS = { ... }
KNOWLEDGE_PATTERNS = { ... }
```

**Problem:** Uses regex as PRIMARY classification, holographic space only as fallback

**Solution:** Replace entire file usage with `GeometricIntentClassifier`

**Status:** ⬜ TODO

---

### 5. geometric_intent_classifier.py - FILLER Set ⬜

**Current Code (lines 104-107):**
```python
FILLER = {'a', 'an', 'the', 'of', 'with', ...}
```

**Problem:** Same as holographic_transformer.py

**Solution:** Share emergent filler detection with holographic_transformer.py

**Status:** ⬜ TODO

---

### 6. plot_space.py - Keyword Matching ⬜

**Current Code (lines 499-502):**
```python
for name, pattern in self.patterns.items():
    for keyword in pattern.keywords:
        if keyword in query_lower:
            ...
```

**Problem:** Uses keyword matching for plot type selection

**Solution:** Transform plot keywords to primitives, match geometrically

**Status:** ⬜ TODO

---

## Completion Log

| Date | Item | Change | Commit |
|------|------|--------|--------|
| 2026-01-09 | holographic_transformer.py | Replaced FILLER_WORDS with φ-Zipf duality (Design 039) | pending |

---

## Notes

- Work on items in order (1→6) to minimize breakage
- Each fix should include tests to verify geometric matching still works
- If geometric matching fails after removing fallbacks, improve the bootstrap data
