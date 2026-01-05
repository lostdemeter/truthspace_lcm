# 097: Geometric Purity Audit (January 2025)

## The Maxims

From PROJECT_OVERVIEW.md:
> **LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

We aim to prove this by building a system where:
- **Structure IS information** - No opaque weights or embeddings
- **Geometry IS computation** - Traversal through geometric space produces outputs
- **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure

## Audit Scope

Audited: `/home/thorin/truthspace-lcm/truthspace_lcm/`

## Summary

| Category | Files | Severity | Status |
|----------|-------|----------|--------|
| ✅ GEOMETRIC | 8 | - | Good |
| ⚠️ BOOTSTRAP (acceptable) | 6 | LOW | Acceptable |
| ❌ NON-GEOMETRIC | 12 | HIGH/MEDIUM | Needs work |
| 📁 LEGACY | 15 | - | Deprecated |

---

## ✅ GEOMETRIC (Good)

These files follow the geometric principles correctly:

### `core/knowledge_space.py`
- ✅ Stop word detection via coverage (emergent, not hardcoded)
- ✅ φ^(-rank) importance formula for similarity
- ✅ Entity relationships tracked geometrically
- ✅ Critical line threshold (σ = 0.5)

### `core/chat_pipeline.py`
- ✅ Intent detection via bootstrapped templates
- ✅ Geometric matching after bootstrap
- ✅ Position-based routing
- ⚠️ Bootstrap patterns are hardcoded (acceptable - this IS bootstrap)

### `core/code_space.py`
- ✅ Patterns loaded from corpus file
- ✅ Geometric matching for pattern selection
- ✅ Learning through position reinforcement

### `core/ollama_space.py`
- ✅ LLM as resource, not decision maker
- ✅ Geometric routing to LLM

---

## ⚠️ BOOTSTRAP (Acceptable)

These files have hardcoded data that serves as bootstrap - this is acceptable per the Emergent Gear Pattern (Design 086).

### `core/bootstrap_knowledge.py`
- ⚠️ Loads from `corpus/bootstrap_knowledge.json`
- This IS the bootstrap - acceptable

### `core/chat_pipeline.py` - IntentSpace._bootstrap_intents()
```python
knowledge_patterns = [
    "what is", "who is", "how does", "why does",
    "tell me about", "explain", "describe",
]
```
- ⚠️ Hardcoded patterns, BUT they're bootstrap
- After bootstrap, detection is geometric
- **Verdict**: Acceptable

### `core/plot_space.py` - _bootstrap_patterns()
- ⚠️ Hardcoded plot templates (sine_wave, cosine_wave, etc.)
- These ARE the bootstrap patterns
- **Verdict**: Acceptable

---

## ❌ NON-GEOMETRIC (Needs Work)

### HIGH SEVERITY

#### `core/classifiers/intent_classifier.py`
```python
GOOSE_TOOL_PATTERNS = {
    "Read": {
        "keywords": {"read", "show", "contents", "cat", "view"...},
        "patterns": [
            r"read\s+(?:the\s+)?(?:file\s+)?(.+)",
            r"show\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+of\s+)?(.+)",
        ],
    },
    ...
}
```
- ❌ Hardcoded keyword sets (bag of words)
- ❌ Regex patterns for matching
- ❌ Not emergent, not geometric
- **Fix**: Move to HyperMapping with bootstrapped templates

#### `core/legacy_gears/gears/intent_detector_gear.py`
```python
ACTION_VERBS = {
    'create', 'make', 'touch', 'delete', 'remove', 'copy', 'move',
    'rename', 'write', 'read', 'open', 'save', 'edit', 'modify',
    ...
}

TOOL_PATTERNS = [
    r'\b(create|make)\s+(a\s+)?(file|directory|folder|dir)\b',
    r'\b(delete|remove|rm)\s+(the\s+)?(file|directory|folder)\b',
    ...
]
```
- ❌ Hardcoded verb sets
- ❌ Regex patterns
- **Fix**: This is in legacy_gears - should be deprecated

### MEDIUM SEVERITY

#### `core/legacy_gears/chains/semantic_chain.py`
```python
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    ...
}
```
- ❌ Hardcoded stopword set
- **Fix**: Use emergent coverage-based detection (like knowledge_space.py)

#### `core/legacy_gears/chains/conversational_chain.py`
- ❌ String `in` operator for topic matching
- ❌ Hardcoded book URLs (GUTENBERG_BOOKS)
- **Fix**: Move to corpus file or emergent discovery

#### `core/plot_space.py` - _bootstrap_modifiers()
```python
self.modifier_patterns = {
    'amplitude': re.compile(r'amplitude\s*(?:of\s*)?(\d+(?:\.\d+)?)', re.I),
    'color': re.compile(r'\b(red|blue|green|yellow|...)\b', re.I),
    ...
}
```
- ❌ Regex patterns for modifier extraction
- **Fix**: Use geometric modifier detection or LLM extraction

#### `core/utils/contact_point.py`
- ❌ Morphological suffix matching (`endswith`)
- ❌ Hardcoded suffix patterns
- **Fix**: Use geometric morphology detection

### LOW SEVERITY

#### `core/utils/template_composer.py`
- ⚠️ Regex for template parsing
- This is structural (parsing), not semantic
- **Verdict**: Acceptable for parsing

#### `core/utils/gear_improvement_loop.py`
- ⚠️ Regex for code extraction
- This is structural (parsing), not semantic
- **Verdict**: Acceptable for parsing

---

## 📁 LEGACY (Deprecated)

These files are in `legacy_gears/` and should be considered deprecated:

- `legacy_gears/gears/intent_detector_gear.py` - HIGH violations
- `legacy_gears/gears/emergent_classifier_gear.py` - MEDIUM violations
- `legacy_gears/chains/semantic_chain.py` - MEDIUM violations
- `legacy_gears/chains/conversational_chain.py` - MEDIUM violations
- `legacy_gears/chains/linguistic_chain.py` - LOW violations
- `legacy_gears/orchestrators/code_orchestrator.py` - HIGH violations
- `legacy_gears/orchestrators/gear_orchestrator.py` - HIGH violations

**Recommendation**: Mark as deprecated, do not use in new code.

---

## Recommendations

### Immediate Actions

1. **Mark legacy_gears as deprecated**
   - Add deprecation warnings
   - Document that new code should use HyperMapping-based components

2. **Fix intent_classifier.py**
   - Replace GOOSE_TOOL_PATTERNS with HyperMapping
   - Bootstrap tool patterns, then use geometric matching

3. **Fix plot_space.py modifier extraction**
   - Replace regex patterns with geometric modifier detection
   - Or use LLM for modifier extraction (as resource)

### Future Work

1. **Emergent morphology detection**
   - Replace suffix matching with geometric patterns
   - Learn morphological patterns from data

2. **Unified bootstrap system**
   - All bootstrap data in corpus files
   - Single pattern for bootstrap → geometric

### The Emergent Gear Pattern (Design 086)

All components should follow:
```
1. STRUCTURE - Define what the space looks like
2. BOOTSTRAP - Seed with initial examples (the ONLY hardcoding)
3. MATCH - Find structure via geometric projection
4. COMPOSE - Adapt structure to specific request
5. LEARN - Self-improve from usage
```

---

## Conclusion

The codebase has made good progress toward geometric purity:

- **Core components** (knowledge_space, chat_pipeline, code_space) are geometric
- **Bootstrap patterns** are acceptable per the Emergent Gear Pattern
- **Legacy gears** should be deprecated
- **Intent classifier** and **modifier extraction** need geometric refactoring

The key insight: **Bootstrap is acceptable, but runtime matching must be geometric.**
