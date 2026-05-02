# Holographic Resolution: GOP, MGOP, and Probe Extraction

## Problem Statement

When using geometric embeddings for semantic matching, cosine similarity often produces dense, non-discriminative scores (e.g., all candidates score 0.85-0.95). This makes it impossible to reliably select the correct match.

## Protocols Applied

Three protocols from the Holographer's Workbench were applied:
1. **GOP** (Gushurst Optimization Protocol) - Recursive framework for extracting hidden structure
2. **MGOP** (Multifold Gushurst Optimization Protocol) - Multi-projection analysis
3. **Probe Extraction Protocol** - Direct measurement instead of optimization

## Inspiration: Additive Error Stereoscopy

From the holographer's workbench, the key insight is:

```
I_L = I - αE
I_R = I + αE
```

Where:
- `I` = original image (baseline)
- `E` = synthesis error (deviation from expected)
- `α` = amplification factor

**The error IS the signal.** Instead of trying to eliminate error, we exploit it.

## Application to Semantic Matching

### The Holographic Error Amplification Method

```python
def resolve(query):
    # Compute cosine similarities (dense, ~0.9 for all)
    sims = [cosine_sim(query_emb, candidate_emb) for candidate in candidates]
    
    # The "synthesis error" - deviation from baseline
    mean_sim = mean(sims)
    errors = sims - mean_sim
    
    # Amplify the error signal
    alpha = 20.0  # Amplification factor
    amplified = alpha * errors
    
    # Sigmoid to convert to probabilities
    probs = sigmoid(amplified)
    
    return candidates[argmax(probs)]
```

### Why It Works (In Theory)

1. **Dense similarities compress information**: When all scores are 0.85-0.95, the 0.10 range contains all discriminative signal
2. **Error amplification spreads the distribution**: Multiplying by α=20 turns 0.10 range into 2.0 range
3. **Sigmoid creates natural saturation**: Extreme values saturate, moderate values spread out

### Multi-View Extension

Like stereo vision uses two eyes, we can use multiple "views" (layer subsets):

```python
views = [
    (16, 48),   # Lexical: semantic primitives
    (48, 64),   # Syntactic: word order
    (64, 88),   # Compositional: combined meaning
]

# Compute error for each view
# Combine: views that AGREE reinforce, views that DISAGREE cancel
combined = product(1 + normalized_error for each view)
```

This is analogous to holographic interference patterns.

## Current Limitation: Signal-to-Noise Ratio

The holographic approach is mathematically sound but requires the underlying embeddings to have discriminative signal. Currently:

### The Problem

Query: "list files"
- Activates: LIST (dim 2), SEQUENCE (dim 24), FILE (dim 8)

Correct match "ls" description: "list files show directory contents folder listing"
- Activates: LIST, SEQUENCE, FILE, VIEW, DIRECTORY, READ_CONTENT

Incorrect match "head" description: "show first lines beginning top of file"  
- Activates: VIEW, SEQUENCE, FILE, PROCESS, FROM

Both have significant overlap with the query because:
1. Descriptions contain many words → many primitive activations
2. Common words like "file", "show" create spurious matches
3. The discriminative primitives (LIST vs VIEW) are drowned out by noise

### The Insight

**The holographic approach amplifies whatever signal exists.** If the raw embeddings have the wrong ranking, amplification makes the wrong answer win more decisively.

The fix must be at the representation level:
1. Better primitive definitions that distinguish actions (LIST ≠ VIEW ≠ READ)
2. Sparser descriptions that activate fewer, more specific primitives
3. Or: accept that some queries require more context to disambiguate

## Relationship to φ-MAX Encoding

The existing φ-MAX encoding (from design doc 012) already addresses synonym overlap:
- MAX prevents synonym stacking
- φ-levels create separation between primitive types

But this operates at the word level. The issue is at the description level - too many words creating too many activations.

## Future Directions

1. **Sparse descriptions**: Use minimal, discriminative keywords
2. **Negative primitives**: Explicitly encode what something is NOT
3. **Hierarchical matching**: First match on action, then on object
4. **Context accumulation**: Use conversation history to disambiguate

## MGOP Analysis Results

### Phase 1: Fractal Peel
- **Autocorrelation**: 0.9969 (highly structured errors)
- **Effective rank**: 10.1 (70 embeddings live in ~10D subspace)
- **Top 10 singular values**: 95.8% of variance

This indicates a **holographic bound** - information compressed into low-dimensional subspace.

### Phase 2: Holographic Scan (Multiple Projections)
Different layer projections give inconsistent rankings (variance=446):

| Projection | Expected Rank | Best Match |
|------------|---------------|------------|
| Morphological | 13 | whereis |
| Lexical | 2 | ls -la |
| **Syntactic** | **1** | **ls** |
| Compositional | 14 | git status |

**Key finding**: Syntactic layer (word order) is most discriminative for bash commands.

### Probe Extraction Results
Direct measurement on query's active dimensions:
- Works well when query activates unique dimensions
- Fails when primitives share dimensions (PROCESS/SYSTEM on dim 9)

## Final Implementation

```python
def _geometric_resolve(query):
    # 1. PROBE EXTRACTION: L1 pattern matching on lexical layer
    query_lex = query_emb[16:48]
    active_mask = abs(query_lex) > 0.1
    
    for candidate in candidates:
        l1_dist = sum(abs(query_lex[active] - emb_lex[active]))
        extra = sum(abs(emb_lex[~active]))
        score = 1.0 / (1.0 + l1_dist + 0.3 * extra)
    
    if gap > 0.05:  # Decisive probe
        return best_match
    
    # 2. FALLBACK: MGOP projection-weighted resolution
    # Weight projections by sharpness (how decisive each is)
    for projection in [lexical, syntactic, compositional]:
        sharpness = top_gap / std
        combined_scores += weight * errors
    
    return softmax(amplified_scores)
```

**Accuracy**: 67% (4/6 test cases)
- ✓ list files → ls
- ✓ hello → Hello!
- ✓ how are you → I'm doing well...
- ✓ thanks → You're welcome!
- ✗ show disk space (SYSTEM/PROCESS dimension collision)
- ✗ running processes (single-dimension query, many matches)

## Root Cause of Remaining Failures

Primitives share dimensions with different φ-levels:
- PROCESS (level 0, activation 1.0) and SYSTEM (level 1, activation 1.62) both on dim 9
- GREETING (level 0) and GRATITUDE (level 1) both on dim 16

When a query only activates one shared dimension, many candidates match.

## Future Directions

1. **Separate primitive dimensions**: Give PROCESS and SYSTEM distinct dimensions
2. **Multi-word primitives**: "disk space" as a single primitive
3. **Contextual disambiguation**: Use conversation history
4. **Learned projections**: Discover optimal projection weights from data

## Conclusion

The combination of GOP/MGOP analysis and Probe Extraction provides a principled geometric approach:
- MGOP reveals which projections are discriminative
- Probe Extraction directly measures overlap on active dimensions
- Holographic error amplification spreads dense similarity scores

The 67% accuracy represents the current limit of the primitive vocabulary. Improving accuracy requires architectural changes to the lexical layer, not algorithmic changes to resolution.
