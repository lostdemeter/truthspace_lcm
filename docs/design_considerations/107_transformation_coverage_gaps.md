# Design 107: Transformation Coverage Gap Detection

## Problem

Current transformation system has knowledge gaps:

```
/transform tense=future: Jack went up the hill
-> "Jack will go up the hill" (25% confidence, 1 change)

/transform tense=future: These charming tunes entertain children
-> "These charming tunes entertain children" (10% confidence, 0 changes)
```

The second sentence has verbs ("entertain") that should transform but we don't have patterns for them.

## Solution: Coverage-Aware Transformation

### 1. Better Confidence Calculation

Current: `confidence = len(word_changes) * 0.25`

Problem: Doesn't consider what *should* have changed.

Better approach:
```python
# Estimate expected changes based on sentence structure
expected_changes = estimate_transformable_words(text, dimension)
actual_changes = len(word_changes)

if expected_changes > 0:
    coverage = actual_changes / expected_changes
else:
    coverage = 1.0  # Nothing to transform

confidence = coverage
```

### 2. Transformable Word Detection

For each dimension, identify words that *should* transform:

**Tense dimension:**
- Verbs (went, sat, entertain, play, develop)
- Auxiliary verbs (will, shall, was, were)

**Regality dimension:**
- Names (Jack, Jill, Mary)
- Common nouns that could be elevated (cat → feline companion)
- Verbs that could be formalized (went → proceeded)

**Formality dimension:**
- Informal words (got, gonna, wanna)
- Contractions (don't, can't, won't)

### 3. LLM Fallback Strategy

When coverage is below threshold:

```python
COVERAGE_THRESHOLD = 0.5  # Below this, consider LLM fallback

if result.confidence < COVERAGE_THRESHOLD:
    if llm_available:
        # Ask LLM to transform
        llm_result = call_llm_transform(text, dimension, target_value)
        
        # Learn from the result
        learn_transformation(text, llm_result, dimension, target_value)
        
        return llm_result
    else:
        # Return partial result with warning
        result.needs_llm = True
        return result
```

### 4. Auto-Learning from LLM

When LLM provides a transformation:

1. **Extract word mappings** - Compare original and transformed
2. **Add to vocabulary** - Store new patterns
3. **Persist to corpus** - Save for future use

```python
def learn_transformation(source: str, target: str, dimension: str, value: str):
    # Extract what changed
    mappings = extract_word_mappings(source, target)
    
    # Add to patterns
    for src_word, tgt_word in mappings.items():
        self._patterns[dimension][value].append(
            (rf'\b{re.escape(src_word)}\b', tgt_word)
        )
    
    # Persist to learned corpus
    self._save_learned_patterns()
```

### 5. Coverage Statistics

Track what we know vs what we're missing:

```python
def coverage_stats(self) -> Dict[str, Any]:
    return {
        "dimensions": {
            "tense": {
                "patterns": len(self._patterns["tense"]),
                "known_verbs": ["went", "sat", "walked", ...],
                "coverage_estimate": 0.65,
            },
            ...
        },
        "total_patterns": sum(...),
        "llm_fallback_count": self._llm_fallback_count,
        "learned_from_llm": self._learned_count,
    }
```

## Implementation Plan

1. **Add verb detection** - Simple POS-like heuristics for verbs
2. **Improve confidence calculation** - Based on expected vs actual changes
3. **Add `needs_llm` flag** - Signal when LLM would help
4. **Implement LLM fallback** - Call Ollama for low-confidence transforms
5. **Auto-learn patterns** - Extract and store new mappings
6. **Add `/transform_stats` command** - Show coverage gaps

## Verb Detection Heuristics

Without a full POS tagger, we can use simple heuristics:

```python
# Common verb endings
VERB_PATTERNS = [
    r'\b\w+ed\b',      # walked, played, entertained
    r'\b\w+ing\b',     # walking, playing (gerunds)
    r'\b\w+s\b',       # walks, plays (3rd person)
]

# Known irregular verbs
IRREGULAR_VERBS = {
    "went", "sat", "stood", "ran", "came", "was", "were",
    "had", "did", "made", "took", "gave", "found", "said",
    "knew", "thought", "saw", "got", "be", "is", "are",
    "have", "has", "do", "does", "go", "goes", "play",
    "entertain", "develop", ...
}
```

## Success Criteria

1. **Accurate coverage detection** - Confidence reflects actual coverage
2. **Seamless LLM fallback** - User doesn't need to manually trigger
3. **Learning accumulates** - Each LLM call improves future coverage
4. **Transparency** - User can see what's known vs unknown

## Connection to Hypothesis

This approach maintains our geometric foundation while pragmatically using LLM to fill gaps:

- **Geometric patterns are primary** - Always try patterns first
- **LLM is a teacher, not a crutch** - We learn from LLM, not depend on it
- **Coverage grows over time** - System becomes more self-sufficient
- **Fail-fast philosophy** - Low confidence signals gaps, not hides them
