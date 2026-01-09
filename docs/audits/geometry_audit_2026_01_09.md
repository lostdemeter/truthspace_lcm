# Geometry Audit Report

**Date:** January 9, 2026  
**Scope:** `truthspace_lcm/` module  
**Principle:** Structure IS information, Geometry IS computation, The shape IS the knowledge

---

## Executive Summary

The audit identified **violations** where text matching bypasses geometry, and **acceptable bootstrap** patterns that transform to geometry immediately. The codebase has a mix of both, with some files adhering well to principles and others needing refactoring.

---

## Classification Legend

| Status | Meaning |
|--------|---------|
| ✅ ACCEPTABLE | Bootstrap data that transforms to geometry immediately |
| ⚠️ BORDERLINE | Could be improved but has geometric fallback |
| ❌ VIOLATION | Pattern matching that bypasses geometry |

---

## Findings by File

### 1. `core/classifiers/intent_classifier.py` ❌ VIOLATION

**Lines 46-132:** Hard-coded keyword sets and regex patterns for intent classification.

```python
GOOSE_TOOL_PATTERNS = {
    "Read": {
        "keywords": {"read", "show", "contents", "cat", "view", "display", "print", "open"},
        "patterns": [r"read\s+(?:the\s+)?(?:file\s+)?(.+)", ...],
    },
    ...
}
CODE_GENERATION_PATTERNS = {
    "keywords": {"plot", "chart", "graph", "histogram", ...},
    "patterns": [r"(?:create|make|generate|draw|plot)\s+...", ...],
}
KNOWLEDGE_PATTERNS = {
    "keywords": {"what", "how", "why", "explain", ...},
    "patterns": [r"^what\s+(?:is|are|does|do)\s+", ...],
}
```

**Problem:** Uses regex pattern matching and keyword sets as PRIMARY classification, with holographic space only as fallback (line 234-237).

**Recommendation:** Replace with `GeometricIntentClassifier` which already exists and uses pure geometric matching.

---

### 2. `core/classifiers/geometric_intent_classifier.py` ⚠️ BORDERLINE

**Lines 104-107:** Hard-coded FILLER word set.

```python
FILLER = {'a', 'an', 'the', 'of', 'with', 'for', 'to', 'and', 'or', 'in',
          'that', 'this', 'is', 'are', 'it', 'be', 'can', 'you', 'i', 'me',
          'my', 'your', 'please', 'could', 'would', 'should'}
```

**Problem:** Filler words are hard-coded rather than emergent.

**Mitigation:** This is used for word extraction before geometric matching, not as a classification mechanism. The actual classification is geometric.

**Recommendation:** Use `EmergentClassifierGear` to discover stopwords from frequency distribution (already exists in `legacy_gears/gears/emergent_classifier_gear.py`).

---

### 3. `core/chat_pipeline.py` ❌ VIOLATION

**Lines 212-233:** Hard-coded keyword matching for intent detection.

```python
knowledge_prefixes = ['what is', 'what are', 'who is', 'who are', 
                      'tell me about', 'explain', 'describe', 'define']
if any(query_lower.startswith(prefix) for prefix in knowledge_prefixes):
    return IntentResult(intent=Intent.KNOWLEDGE, ...)

plot_keywords = ['plot', 'graph', 'chart', 'sine', 'cosine', 'histogram', 
                 'scatter', 'bar chart', 'pie chart', 'visualize', 'wave']
if any(kw in query_lower for kw in plot_keywords):
    return IntentResult(intent=Intent.PLOT_GENERATION, ...)
```

**Problem:** String matching bypasses the geometric intent detection entirely.

**Recommendation:** Remove keyword shortcuts. Let geometric matching handle all cases. If geometric matching fails, that's a signal to improve the bootstrap data, not add fallbacks.

---

### 4. `core/knowledge_space.py` ❌ VIOLATION

**Lines 560-579:** Keyword boost in similarity calculation.

```python
keyword_boost = 0.0
keywords = mapping.metadata.get("keywords", [])
if keywords:
    for kw in keywords:
        kw_words = set(kw.lower().split())
        overlap = len(query_words & kw_words)
        if overlap > 0 and overlap >= len(kw_words):
            keyword_boost = max(keyword_boost, 0.5 + 0.1 * len(kw_words))
        elif overlap > 0:
            ratio = overlap / len(kw_words)
            keyword_boost = max(keyword_boost, 0.3 * ratio)

similarity = geo_similarity + keyword_boost
```

**Problem:** Keyword matching adds a boost ON TOP of geometric similarity. This is a fallback mechanism that compensates for geometric failures.

**Note:** Lines 617-639 show the CORRECT approach using `PrimitiveRegistry.encode()` for pure geometric matching.

**Recommendation:** Remove keyword boost. Use `PrimitiveRegistry` approach for all queries.

---

### 5. `core/primitives.py` ✅ ACCEPTABLE

**Lines 48-262:** Primitive definitions with keyword lists.

```python
Primitive("PHYSICS", 0, 3, [
    "physics", "quantum", "relativity", "mechanics", "thermodynamics",
    "electromagnetism", "particle", "wave", "energy", "force"
]),
```

**Status:** This IS bootstrap data. Keywords are transformed to geometric positions via `PrimitiveRegistry`. The keywords are seeds, not matching patterns.

---

### 6. `core/primitive_registry.py` ✅ ACCEPTABLE

**Lines 50-71:** Transforms keywords to geometry.

```python
# OLD: "what is physics" in query → boost similarity (pattern match)
# NEW: "what is physics" → primitive → levels → position (geometry)
```

**Status:** This is the correct approach. Keywords become geometric positions at registration time.

---

### 7. `corpus/bootstrap_knowledge.json` ✅ ACCEPTABLE

**Lines 1-179:** Bootstrap knowledge with keywords and phi_levels.

```json
{
  "text": "Physics is the natural science...",
  "keywords": ["physics", "what is physics", "study of physics"],
  "phi_levels": [3, 2, 1, 1]
}
```

**Status:** Bootstrap data with explicit geometric positions (phi_levels). Keywords are seeds for the primitive registry.

---

### 8. `core/holographic_transformer.py` ⚠️ BORDERLINE

**Lines 64-68:** Hard-coded FILLER_WORDS set.

```python
FILLER_WORDS = {
    'a', 'an', 'the', 'of', 'with', 'for', 'to', 'and', 'or', 'in',
    'that', 'this', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    ...
}
```

**Problem:** Same as geometric_intent_classifier.py - hard-coded stopwords.

**Recommendation:** Derive from Zipf distribution (high frequency + uniform distribution = filler).

---

### 9. `core/plot_space.py` ⚠️ BORDERLINE

**Lines 90-366:** PlotPattern with keywords.

```python
PlotPattern(
    name='sine_wave',
    keywords=['sine', 'sin', 'wave', 'sinusoidal', 'oscillation'],
    template='''import numpy as np...'''
)
```

**Lines 499-502:** Keyword matching for pattern selection.

```python
for name, pattern in self.patterns.items():
    for keyword in pattern.keywords:
        if keyword in query_lower:
            ...
```

**Problem:** Uses keyword matching for plot type selection.

**Mitigation:** Plot patterns are bootstrap data. However, the matching should be geometric.

**Recommendation:** Transform plot keywords to primitives, match geometrically.

---

### 10. `core/legacy_gears/` ⚠️ LEGACY

Multiple files contain keyword matching and pattern-based approaches:
- `bootstrap_gear.py`: Keyword frequency matching (lines 363-376)
- `intent_detector_gear.py`: CODE_KEYWORDS set (lines 148-153)
- `code_orchestrator.py`: PlotPattern keywords (lines 120-161)

**Status:** These are in `legacy_gears/` which suggests they're deprecated. However, they may still be imported.

**Recommendation:** Ensure legacy code is not used in active paths. Consider removal.

---

### 11. `core/legacy_gears/gears/emergent_classifier_gear.py` ✅ GOOD PATTERN

**Lines 161-183:** Shows the CORRECT approach for discovering word categories.

```python
# Instead of hardcoding lists like:
#     stopwords = {'the', 'a', 'an', ...}
# We define signatures that describe the STRUCTURAL properties:
#     stopwords: high_frequency + uniform_distribution + short_length
```

**Status:** This is the right approach. Word categories should emerge from distributional properties.

---

### 12. `practical_applications/chat/hyper_api.py` ⚠️ BORDERLINE

**Lines 200-246:** Command routing via `startswith()`.

```python
if user_message.lower().startswith("/learn "):
    ...
if user_message.lower().startswith("/transform_geo"):
    ...
```

**Status:** This is command parsing, not semantic classification. Commands are explicit user directives with `/` prefix.

**Verdict:** Acceptable - these are explicit commands, not semantic matching.

---

## Summary of Violations

| File | Severity | Issue |
|------|----------|-------|
| `intent_classifier.py` | ❌ HIGH | Regex patterns as primary classifier |
| `chat_pipeline.py` | ❌ HIGH | Keyword shortcuts bypass geometry |
| `knowledge_space.py` | ❌ MEDIUM | Keyword boost on top of geometry |
| `holographic_transformer.py` | ⚠️ LOW | Hard-coded filler words |
| `geometric_intent_classifier.py` | ⚠️ LOW | Hard-coded filler words |
| `plot_space.py` | ⚠️ LOW | Keyword matching for plots |

---

## Recommended Actions

### Priority 1: Remove Keyword Shortcuts in `chat_pipeline.py`

Delete lines 210-233 (knowledge_prefixes and plot_keywords checks). Let geometric matching handle all intent detection.

### Priority 2: Replace `IntentClassifier` with `GeometricIntentClassifier`

The geometric version already exists and follows the principles. Update imports to use it.

### Priority 3: Remove Keyword Boost in `knowledge_space.py`

Use `PrimitiveRegistry.encode()` for all queries. Remove the keyword_boost mechanism.

### Priority 4: Derive Filler Words Emergently

Use frequency distribution to identify filler words rather than hard-coding. The `EmergentClassifierGear` shows how.

### Priority 5: Audit Legacy Code Usage

Ensure `legacy_gears/` code is not imported in active paths. Mark for deprecation.

---

## Acceptable Patterns (Keep)

1. **Bootstrap data** (`primitives.py`, `bootstrap_knowledge.json`) - Seeds that transform to geometry
2. **Primitive Registry** (`primitive_registry.py`) - Keyword → geometry transformation
3. **Emergent Classifier** (`emergent_classifier_gear.py`) - Discovers categories from structure
4. **Command parsing** (`hyper_api.py`) - Explicit `/command` routing

---

## Conclusion

The codebase has good geometric infrastructure (`PrimitiveRegistry`, `GeometricIntentClassifier`, `EmergentClassifierGear`) but also has fallback mechanisms that bypass geometry. The main violations are in intent classification and knowledge retrieval where keyword matching is used as a shortcut.

**The fix is not to add more fallbacks, but to improve the geometric matching so fallbacks aren't needed.**
