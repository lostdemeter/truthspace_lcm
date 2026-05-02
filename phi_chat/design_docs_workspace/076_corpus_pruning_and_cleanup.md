# Design Consideration 076: Corpus Pruning and Cleanup

**Date**: December 30, 2024  
**Author**: Lesley Gushurst  
**Status**: Implemented and Tested

## Executive Summary

This document describes the **corpus pruning and cleanup** system that removes bad data from the TruthSpace corpus. Rather than continuously adding reinforcement frames (which leads to model bloat), we now identify and remove problematic data at the source.

## The Problem

The experimental corpus had grown to **113,171 frames** with significant quality issues:

| Issue | Count | % of Corpus |
|-------|-------|-------------|
| Too short (<25 chars) | 53,832 | 47.6% |
| Reinforcement bloat | 37,275 | 32.9% |
| Excessive duplicates | 11,687 | 10.3% |
| **Total problematic** | **68,007** | **60.1%** |

Additionally, many concepts had incorrect roles:
- Abstract concepts labeled as "character" (e.g., "Development is a character who...")
- Plural nouns labeled as "character" (e.g., "Missions is a character who...")
- Nonsense verbs in action lists (e.g., "masaccioes", "gluons")

## The Solution: Multi-Stage Cleanup

### Stage 1: Corpus Pruning

The `CorpusPruner` removes obviously bad frames:

```python
pruner = CorpusPruner(corpus_path)
pruner.analyze_all(max_duplicates=10, min_length=25)
pruner.apply_pruning(dry_run=False)
```

**Rules:**
1. **Too short**: Frames < 25 characters are removed
2. **Duplicates**: Keep max 10 copies of identical frames
3. **Reinforcement bloat**: Remove excessive reinforcement frames (>5 copies)
4. **Typo fixes**: Fix known typos with word-boundary matching

**Results:**
- Original: 113,171 frames
- After pruning: 45,164 frames
- **Reduction: 60.1%**

### Stage 2: Role Fixing

The `CorpusRoleFixer` identifies and fixes incorrect roles:

```python
fixer = CorpusRoleFixer(corpus_path)
fixer.analyze_concepts()
fixer.apply_fixes(min_confidence=0.7)
```

**Detection rules:**
- Abstract suffixes: -ology, -tion, -ment, -ness, -ism, -ics, -istry, -ure, -ance, -ence
- Plural words: ending in -s (not -ss)
- Known person names: holmes, watson, etc.

**Results:**
- Analyzed: 10,415 concepts
- Wrong roles found: 4,121
- Fixed (confidence >= 0.7): 1,660

### Stage 3: Signal Frame Bypass

The gear chain now skips signal frames that have wrong roles:

```python
if concept_lower in self.signal_frames:
    signal = self.signal_frames[concept_lower]
    if 'is a character' in signal.lower():
        if self._should_be_concept(concept_lower):
            pass  # Skip signal frame, use gear chain
```

This ensures the RoleGear can fix roles even when signal frames exist.

### Stage 4: Deep Cleaning with Qwen2 (Optional)

The `CorpusDeepCleaner` uses Qwen2 for intelligent frame evaluation:

```python
cleaner = CorpusDeepCleaner(corpus_path)
cleaner.evaluate_batch(limit=100)
cleaner.apply_cleaning()
```

**Qwen2 evaluates:**
- Grammatical correctness
- Semantic clarity
- Factual plausibility
- Usefulness for QA

**Actions:** KEEP, REMOVE, or REWRITE

## Results

### Before Cleanup

```
MISSIONS: Missions is a character who provides and expands...
REFORMS: Reforms is a character that highlights, includes...
DEVELOPMENT: Development is a character that integrates...
```

### After Cleanup

```
MISSIONS: Missions is a concept that involves providing and expanding...
REFORMS: Reforms is a concept that involves highlighting, including...
DEVELOPMENT: Development is an entity that involves combining, integrating...
```

### Corpus Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total frames | 113,171 | 45,164 | -60.1% |
| Concepts | 31,667 | 31,667 | 0% |
| Wrong roles | 4,121 | ~2,461 | -40% |
| Avg frame length | ~45 chars | ~65 chars | +44% |

## Implementation Files

| File | Purpose |
|------|---------|
| `experiments/corpus_pruner.py` | Rule-based frame pruning |
| `experiments/corpus_role_fixer.py` | Role detection and fixing |
| `experiments/corpus_deep_cleaner.py` | Qwen2-powered evaluation |
| `experiments/gear_chain_feedback.py` | Signal frame bypass logic |

## Usage

### Quick Pruning
```bash
# Dry run
python3 experiments/corpus_pruner.py

# Apply
python3 experiments/corpus_pruner.py --apply
```

### Role Fixing
```bash
# Dry run
python3 experiments/corpus_role_fixer.py

# Apply with confidence threshold
python3 experiments/corpus_role_fixer.py --apply --min-confidence 0.7
```

### Deep Cleaning
```bash
# Evaluate 100 frames
python3 experiments/corpus_deep_cleaner.py --limit 100

# Apply cleaning
python3 experiments/corpus_deep_cleaner.py --apply
```

## Key Insights

### 1. Pruning > Adding

Adding reinforcement frames leads to bloat. The corpus grew from ~40K to 113K frames through reinforcement, but 62% of those frames were redundant or low-quality.

**Lesson:** Prune bad data rather than drowning it out with good data.

### 2. Role Detection is Heuristic

Determining if something is a "character" vs "concept" requires heuristics:
- Suffix patterns (-ology, -tion, etc.)
- Plural detection
- Known name lists

These work for ~70% of cases. The remaining 30% need Qwen2 or manual review.

### 3. Signal Frames Can Be Stale

Pre-computed signal frames bypass the gear chain, so they don't benefit from role fixes. The solution is to detect and skip bad signal frames.

### 4. Confidence Thresholds Matter

Not all fixes are equally confident:
- Abstract suffix match: 0.9 confidence
- Plural detection: 0.7 confidence
- Default/unknown: 0.5 confidence

Only apply fixes above a threshold (0.7 recommended).

## Future Improvements

### 1. Incremental Cleaning
Run cleanup periodically as new data is added, not just once.

### 2. Qwen2 Batch Processing
Use Qwen2 to evaluate all frames, not just samples.

### 3. Negative Examples
Maintain a list of "bad frame patterns" to automatically reject.

### 4. Signal Frame Regeneration
Regenerate signal frames through the gear chain to fix stale data.

## Conclusion

Corpus cleanup is essential for maintaining model quality. The multi-stage approach:

1. **Prune** obvious garbage (short, duplicate, bloated)
2. **Fix** detectable issues (wrong roles via heuristics)
3. **Bypass** stale cached data (signal frames)
4. **Evaluate** questionable cases (Qwen2)

This reduced the corpus by 60% while improving output quality. The gear chain now correctly identifies abstract concepts and plurals as "concept" rather than "character".

```
"A smaller, cleaner corpus beats a larger, noisier one.
 Prune the weeds, don't just plant more flowers."
```
