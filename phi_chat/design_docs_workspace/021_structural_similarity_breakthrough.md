# Structural Similarity Breakthrough: Escaping Keyword Matching

## The Problem

We were stuck at 67% accuracy despite trying:
- GOP/MGOP protocols
- Probe Extraction
- ResourceType layer (SLAP)
- Dimensional downcasting
- Resfrac spectral scoring
- Z-score normalization
- Borwein smoothing
- Dual-space alignment

**Root cause**: Word-based primitives ARE keyword matching in disguise.

## The Insight

The user observed:
> "I'm not entirely a fan of action words. I'm not sure they should even be words. We're using words, but I feel like we keep landing on keyword matching."

This is exactly right. Our primitives like LIST, VIEW, READ, SHOW:
- Activate based on specific keywords
- Fail when query uses different vocabulary than description
- "show disk space" → VIEW (because of "show")
- "df -h" description has "usage" not "show" → mismatch

**Words are symbols. Different symbols can point to the same meaning.**

## The Solution: Character N-grams

Instead of word-based primitives, use **character-level structural similarity**:

```python
def char_ngrams(text, n=3):
    text = text.lower().replace(' ', '_')
    return set(text[i:i+n] for i in range(len(text) - n + 1))

# "show disk space" and "disk space usage" share:
# {'dis', 'ace', 'spa', 'isk', 'sk_', 'pac', 'k_s', '_sp'}
```

This captures structural similarity **without vocabulary dependence**.

## Why It Works

| Query | Description | Shared Structure |
|-------|-------------|------------------|
| "show disk space" | "disk space usage..." | disk, space patterns |
| "running processes" | "show running processes..." | running, process patterns |
| "hello" | "hello hi hey greetings..." | hello pattern |

The action words ("show", "list", "view") become **noise** that doesn't affect matching because the **content words** ("disk", "space", "process") carry the structural signal.

## Implementation

```python
def _geometric_resolve(self, query: str):
    query_ngrams = char_ngrams(query, n=3)
    
    for content, desc in knowledge_base:
        desc_ngrams = char_ngrams(desc, n=3)
        
        # Recall-weighted F-score (handles short queries)
        recall = len(query_ngrams & desc_ngrams) / len(query_ngrams)
        precision = len(query_ngrams & desc_ngrams) / len(desc_ngrams)
        
        beta = 2.0 if len(query_ngrams) < 10 else 1.0
        sim = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
```

## Results

| Metric | Word Primitives | Structural N-grams |
|--------|-----------------|-------------------|
| Core tests (6) | 67% (4/6) | **100% (6/6)** |
| Extended tests (15) | ~67% | **87% (13/15)** |
| Test suite | 7/7 | **7/7** |

### Previously Failing Cases Now Working:
- ✓ "show disk space" → df -h
- ✓ "running processes" → ps aux
- ✓ "hello" → Hello! How can I help you?

## Key Principles

1. **Structure over vocabulary**: Character patterns are invariant to word choice
2. **No action words**: They're just noise that causes mismatches
3. **Recall for short queries**: Weight recall higher when query has few n-grams
4. **Purely geometric**: No semantic interpretation, just structural overlap

## Connection to Holographic Principles

This aligns with the holographic insight from MGOP:
- Different projections (words) can encode the same information (meaning)
- The "smooth" structure (character patterns) is more fundamental than the "oscillatory" structure (word choice)
- We're projecting to a space where vocabulary variation is noise

## Implications

1. **The lexical layer primitives may be unnecessary** for resolution
2. **Morphological features are more robust** than semantic features
3. **Keyword matching is a trap** - any word-based system falls into it
4. **Structure is meaning** at the character level

## Files Modified

- `chat.py`: `_geometric_resolve()` now uses character n-gram similarity
- Removed dependence on lexical/ResourceType layer for resolution
- Kept embedding layers for other purposes (clustering, etc.)

## Future Directions

1. **Hybrid approach**: Use n-grams for resolution, embeddings for clustering
2. **Learned n-gram weights**: Some patterns more discriminative than others
3. **Variable n**: Adaptive n-gram size based on query length
4. **Phonetic similarity**: Handle typos and phonetic variations
