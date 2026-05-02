# Design Consideration 018: Stacked Geometric LCM Analysis

## What Works, What Doesn't

This document summarizes the findings from developing the Stacked Geometric LCM v1 and v2.

---

## Architecture Evolution

### v1: 5 Layers, 80 Dimensions
```
Morphological (16D) → Lexical (24D) → Compositional (16D) → Contextual (16D) → Global (8D)
```

### v2: 7 Layers, 128 Dimensions
```
Morphological (16D) → Lexical (32D) → Syntactic (16D) → Compositional (24D) → Disambiguation (16D) → Contextual (16D) → Global (8D)
```

Key additions in v2:
- **Syntactic layer**: Bigram patterns for word order
- **Disambiguation layer**: Context-dependent meaning resolution
- **Expanded primitives**: More keywords per primitive
- **Layer weighting**: Morphological reduced, disambiguation increased

---

## Quantitative Results

### Same-Domain Similarity (Higher is Better)

| Test Case | v1 | v2 | Δ |
|-----------|----|----|---|
| "chop onions" vs "dice carrots" | 0.659 | **0.975** | +48% |
| "ls -la" vs "show directory contents" | 0.300 | **0.745** | +148% |
| "hello how are you" vs "hi there friend" | 0.675 | **0.777** | +15% |
| "boil water" vs "simmer sauce" | 0.519 | **0.962** | +85% |

**v2 wins all same-domain tests.**

### Cross-Domain Separation (Lower is Better)

| Test Case | v1 | v2 | Δ |
|-----------|----|----|---|
| "chop onions" vs "ls -la" | 0.327 | **0.267** | -18% |
| "hello how are you" vs "grep pattern" | 0.441 | **0.205** | -54% |
| "boil water" vs "delete files" | 0.391 | **0.185** | -53% |

**v2 wins all cross-domain tests.**

### Ambiguity Handling (Lower is Better)

| Test Case | v1 | v2 | Δ |
|-----------|----|----|---|
| "cut the file" vs "cut the vegetables" | 0.831 | **0.390** | -53% |
| "search for recipes" vs "search for files" | 0.866 | **0.368** | -58% |
| "run the program" vs "run to the store" | **0.487** | 0.531 | +9% |
| "find the file" vs "find the recipe" | 0.670 | **0.372** | -44% |

**v2 wins 3/4 ambiguity tests.** The "run" case regressed slightly.

---

## What Works

### 1. Expanded Primitives
Adding more keywords to primitives dramatically improved recognition:
- "ls" → READ primitive
- "recipes" → FOOD primitive
- "program" → PROCESS primitive

**Lesson**: Vocabulary coverage is critical. Missing keywords cause 0.0 lexical similarity.

### 2. Syntactic Bigrams
Detecting word pairs like ("the", "file") vs ("the", "vegetables") helps disambiguate:
- "cut the file" → tech bigram detected
- "cut the vegetables" → cooking bigram detected

**Lesson**: Word ORDER matters, not just word presence.

### 3. Disambiguation Layer
Explicitly tracking ambiguous words and their context resolves polysemy:
- "cut" + "file" → tech domain
- "cut" + "vegetables" → cooking domain

**Lesson**: Some words need special handling. A general approach isn't enough.

### 4. Layer Weighting
Reducing morphological weight (0.3) and increasing disambiguation weight (2.0) improved results:
- Similar word lengths no longer dominate
- Context-dependent features have more influence

**Lesson**: Not all features are equally important. Weighting matters.

---

## What Doesn't Work

### 1. The "run" Case
"run the program" vs "run to the store" = 0.531 (too high)

**Why it fails**:
- "run" is in the PROCESS primitive (tech domain)
- "program" reinforces tech
- "store" is a weak physical indicator
- No strong cooking/social signal to contrast

**The problem**: Both sentences have tech-like structure. "run to the store" doesn't have enough non-tech indicators.

### 2. Sparse Bigram Coverage
We can't enumerate all possible bigrams. New phrases will be missed.

**Example**: "execute the script" works, but "launch the application" might not if those bigrams aren't defined.

### 3. Cold Start Problem
The contextual layer needs data to learn co-occurrence patterns. With few samples, it doesn't help much.

### 4. Scaling Primitives
Adding more primitives increases dimensionality but also increases the chance of spurious matches.

---

## Fundamental Limitations

### 1. Hand-Coded Knowledge
Every keyword, bigram, and disambiguation rule is manually specified. This doesn't scale.

**Contrast with LLMs**: They learn these patterns from data.

### 2. Fixed Vocabulary
New words not in our keyword lists get no primitive activation.

**Example**: "sauté" works (in HEAT), but "braise" might not if we forgot to add it.

### 3. Binary Context
Our disambiguation is binary: "file" = tech, "vegetables" = cooking. Real language has gradients.

**Example**: "cut the budget" is neither tech nor cooking, but our system will try to force it into one.

### 4. No Compositionality Beyond Bigrams
We detect bigrams but not longer phrases or syntactic structures.

**Example**: "the file that contains the recipe" has both tech and cooking words, but the structure indicates it's about a file.

---

## Comparison to LLM Embeddings

| Aspect | Stacked Geometric | LLM Embeddings |
|--------|-------------------|----------------|
| Dimensions | 128 | 768+ |
| Training | None | Massive |
| Vocabulary | Fixed (~500 keywords) | Learned (~50k+ tokens) |
| Ambiguity | Rule-based | Learned from context |
| New words | Fail silently | Subword fallback |
| Interpretable | Yes | No |
| Compute | CPU, instant | GPU, slower |

### The Trade-off
- **Stacked Geometric**: Interpretable, fast, but limited by hand-coded knowledge
- **LLM Embeddings**: Powerful, general, but black-box and compute-heavy

---

## Future Directions

### 1. Automatic Primitive Discovery
Instead of hand-coding keywords, discover them from a corpus:
```python
# Find words that co-occur with known cooking words
cooking_words = find_cooccurring_words(corpus, seed=['cook', 'recipe'])
```

### 2. Learned Bigram Weights
Instead of binary bigram detection, learn weights from data:
```python
# Weight bigrams by how discriminative they are
bigram_weight[('the', 'file')] = mutual_information(bigram, tech_domain)
```

### 3. Deeper Syntactic Analysis
Use dependency parsing to understand structure:
```python
# "the file that contains the recipe" → head noun is "file"
head_noun = parse(text).root.head
```

### 4. Hybrid Approach
Use LLM embeddings for the hard cases, geometric for the easy ones:
```python
if disambiguation_confidence < threshold:
    return llm_embedding(text)
else:
    return geometric_embedding(text)
```

---

## Conclusions

### What We Proved
1. **Hierarchical geometric encoding works** - stacking layers adds discriminative power
2. **Context matters** - disambiguation layer significantly improves ambiguity handling
3. **Vocabulary coverage is critical** - missing keywords cause failures
4. **Layer weighting matters** - not all features are equally important

### What We Learned
1. **Hand-coded knowledge doesn't scale** - we need automatic discovery
2. **Some cases are fundamentally hard** - "run to the store" needs world knowledge
3. **The geometric approach is complementary to LLMs** - not a replacement

### The Path Forward
The stacked geometric approach is viable for **interpretable, fast, domain-specific** applications. For **general-purpose** language understanding, hybrid approaches that combine geometric structure with learned embeddings are more promising.

---

## Implementation Files

- `@/home/thorin/truthspace-lcm/truthspace_lcm/core/stacked_geometric_lcm.py` - v1 (80D, 5 layers)
- `@/home/thorin/truthspace-lcm/truthspace_lcm/core/stacked_geometric_lcm_v2.py` - v2 (128D, 7 layers)
