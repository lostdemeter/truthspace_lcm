# Design 108: Pure Geometric Transformation (No LLM Fallback)

## EXPERIMENTAL RESULTS (Jan 7, 2025)

### Key Finding: Encoder Doesn't Distinguish Content

Testing revealed that the QuaternionEncoder gives **identical positions** to all sentences:

```
"Jack and Jill went up the hill" → [1.618, 1.618, 1.618, ...]
"The cat sat on the mat"         → [1.618, 1.618, 1.618, ...]
"The chef prepared a meal"       → [1.618, 1.618, 1.618, ...]
```

Pairwise distance = 0.0000 for all sentence pairs.

### What Works

1. **Deltas are self-similar** - The tense transformation delta is exactly φ² (2.618) in dimension 4, consistent across all word pairs
2. **Dimension-level encoding works** - Words like "will", "shall" activate the future tense dimension

### What Fails

1. **Content encoding** - Sentences without dimension anchor words all collapse to the same position
2. **Nearest neighbor decoding** - Can't find the right sentence when all sentences are at the same position
3. **Sentence-level transformation** - 21.1% accuracy on corpus (essentially random)

### Hypothesis Status

This is not a failure of the hypothesis - it's a **gap in the encoder**. The hypothesis says:
- Structure IS information
- Geometry IS computation
- ENCODE = DECODE

The encoder currently captures **dimensional structure** but not **content structure**. For the hypothesis to be fully tested, we need an encoder that gives each sentence a unique position based on its content.

### Next Steps

1. **Build content-aware encoder** - Use attractor/repeller dynamics to derive word positions from co-occurrence
2. **Or** accept that geometric transformation only works for dimension-level features, not content generation

---

## EXPERIMENTAL RESULTS: Holographic Transformer (Jan 7, 2025)

### Approach

Used `HolographicPatternSpace` approach:
- Positions CONSTRUCTED from word overlap similarity
- Eigendecomposition of similarity matrix gives positions
- `dot(P[i], P[j]) ≈ similarity(i, j)` by construction

### Results

**Accuracy: 8.2%** (18/220 transformations correct)

Worse than QuaternionEncoder (21.1%) despite content-aware encoding.

### Why It Fails

The geometry is **correct** - similar sentences are close:
- 75% word overlap → distance 0.08-0.13
- 25% word overlap → distance 0.83

But transformation fails because:
1. Source and target sentences have HIGH overlap (75%)
2. The delta is therefore SMALL (0.12)
3. Many OTHER sentences are equally close to `position(source) + delta`
4. Nearest neighbor finds the wrong sentence

Example:
```
Source: "Jack and Jill went up the hill to fetch a pail of water"
Expected: "Jack and Jill will go up the hill to fetch a pail of water"
Got: "Jack and Jill went up the hill to retrieve a pail of water"
```

The "retrieve" variant is closer to the transformed position than the "will go" variant.

### The Fundamental Issue

**Geometric transformation is a RETRIEVAL problem, not a GENERATION problem.**

For retrieval to work perfectly, we need:
- Each source sentence has EXACTLY ONE target at `position + delta`
- No other sentences are nearby

But our corpus has many similar sentences (variants of the same base), so retrieval is ambiguous.

### What This Tells Us About the Hypothesis

The hypothesis states: "Geometry IS computation"

For **classification/retrieval**, this works:
- Similar things are close
- Different things are far
- Nearest neighbor finds the right category

For **generation/transformation**, this fails:
- We can compute `new_position = position + delta`
- But decoding `new_position → text` requires a unique mapping
- With many similar sentences, the mapping is ambiguous

### Possible Solutions

1. **Larger corpus with unique sentences** - If every sentence is unique, retrieval is unambiguous
2. **Word-level transformation** - Transform individual words, not sentences (but needs word-level vocabulary)
3. **Accept limitation** - Use geometry for classification/routing, not generation

### Connection to LLMs

This reveals WHY LLMs work for generation:
- They don't retrieve from a fixed corpus
- They GENERATE token-by-token using learned distributions
- The "decoding" is probabilistic, not nearest-neighbor

Our geometric approach can match LLM-like behavior for:
- Intent detection ✓
- Similarity search ✓
- Classification ✓

But not for:
- Text generation ✗
- Sentence transformation ✗

Unless we build a geometric decoder that generates tokens, not retrieves sentences.

---

## EXPERIMENTAL RESULTS: Additional Approaches (Jan 7, 2025)

### Approaches Tested

| Approach | Accuracy | Issue |
|----------|----------|-------|
| QuaternionEncoder | 21.1% | All sentences encode to same position |
| HolographicTransformer (word overlap) | 8.2% | Similar sentences compete |
| Holographic + pair boost | 0% | Boosting doesn't create right structure |
| Probe extraction (W = Y @ X @ inv(X'X)) | 21.1% | Low MSE but nearest neighbor fails |
| Laplacian embedding (graph-based) | 0% | Deltas not self-similar |

### Probe Extraction Details

Used the exact formula from memory: `W = Y @ X @ (X^T X)^(-1)`

- MSE: 0.0026 (very low - good fit to training data)
- But nearest neighbor still finds wrong sentence
- The transformation matrix is correct, but decoding fails

### Laplacian Embedding Details

Treated transformation pairs as graph edges:
- Built adjacency matrix from transformation relationships
- Used normalized Laplacian eigenvectors as positions
- Result: Deltas NOT self-similar (deviation 0.45 vs norm 0.14)

### The Core Issue

**Geometric transformation requires unique decoding.**

For `position + delta → text` to work:
- The target must be the ONLY point at `position + delta`
- Or at least the NEAREST point

But with 169 sentences and high word overlap:
- Many sentences cluster together
- Nearest neighbor is ambiguous

### What Works vs What Doesn't

**Works geometrically:**
- Classification/retrieval (100% with keyword boost - Design 101)
- Concept-level transformations (king→queen, self-similar)
- Intent detection, similarity search

**Doesn't work geometrically:**
- Sentence-level transformation
- Text generation

### Connection to Design 084

Design 084 says: "We can construct the geometry we need."

For retrieval, we construct geometry where similar things are close.
For transformation, we'd need geometry where each source has EXACTLY ONE target nearby.

This is possible in principle but requires:
1. Each sentence has a unique position (not just based on word overlap)
2. Transformation pairs are explicitly encoded in the structure
3. No other sentences are nearby

With a small corpus (169 sentences) and high overlap, this isn't achievable.

---

## The Problem

The current `TransformationSpace` violates core philosophy:
- Uses regex patterns (hard-coded morphology)
- Falls back to LLM when patterns fail
- Word extraction is string matching, not geometric

This masks whether geometric transformation actually works.

## The Hypothesis Test

Can we transform sentences using **only geometry**?

```
ENCODE(source) + DELTA = ENCODE(target)
```

If this works, we validate that transformations are geometric operations.
If it fails, we learn where the hypothesis breaks down.

## Pure Geometric Architecture

### 1. The Transformation Corpus as Training Data

We have 180 transformation examples:
```json
{
  "source": "Jack and Jill went up the hill",
  "target": "Jack and Jill will go up the hill",
  "dimension_delta": {"tense": ["past", "future"]}
}
```

### 2. Compute Delta Vectors

For each transformation pair:
```python
source_pos = encoder.encode(source)  # QuaternionPosition
target_pos = encoder.encode(target)  # QuaternionPosition
delta = target_pos - source_pos      # The transformation IS this vector
```

### 3. Average Deltas Per Dimension

For dimension "tense" with value "future":
```python
deltas = []
for example in corpus:
    if example.dimension == "tense" and example.target_value == "future":
        deltas.append(example.delta)

# The canonical delta for tense=future
TENSE_FUTURE_DELTA = average(deltas)
```

### 4. Apply Transformation

```python
def transform(text: str, dimension: str, target_value: str) -> str:
    # Encode input
    position = encoder.encode(text)
    
    # Apply delta
    delta = get_delta(dimension, target_value)
    new_position = position + delta
    
    # Decode output
    return decoder.decode(new_position)
```

## The Decoder Challenge

This is where the hypothesis is tested. Options:

### Option A: Vocabulary-Based Decoder

Build a vocabulary where each word has a position:
```python
vocabulary = {
    "went": QuaternionPosition(...),
    "will go": QuaternionPosition(...),
    "sat": QuaternionPosition(...),
    "will sit": QuaternionPosition(...),
}

def decode(position: QuaternionPosition) -> str:
    # Find nearest word in vocabulary
    return nearest_neighbor(position, vocabulary)
```

**Problem**: This only works for single words, not sentences.

### Option B: Compositional Decoder

Sentences are compositions of word positions:
```python
def encode_sentence(text: str) -> QuaternionPosition:
    words = tokenize(text)
    positions = [encode_word(w) for w in words]
    return compose(positions)  # Some aggregation

def decode_sentence(position: QuaternionPosition) -> str:
    # Decompose position into word positions
    word_positions = decompose(position)
    words = [decode_word(p) for p in word_positions]
    return " ".join(words)
```

**Challenge**: How do we decompose? The composition must be invertible.

### Option C: Self-Similar Decoder

If transformations are self-similar (same delta at every scale):
```python
# king - man + woman = queen
# This works because the delta IS the transformation

# For sentences:
# "went up the hill" - PAST + FUTURE = "will go up the hill"
# The delta applies to each transformable word independently
```

**Insight**: We don't decode the whole sentence. We:
1. Identify which words are affected by the dimension
2. Apply the delta to those words individually
3. Decode each word back to text

### Option D: Probe Extraction (From Memory)

From the "Complete Path to 100%" memory:
```
Probe extraction = 100% (exact, no bound)
W = Y @ X @ (X^T X)^(-1)
```

If we have enough examples, we can extract the exact transformation:
- X = source positions (matrix)
- Y = target positions (matrix)  
- W = transformation matrix

Then: `new_target = W @ new_source`

## Proposed Implementation

### Phase 1: Build Word-Level Vocabulary

From the transformation corpus, extract word pairs:
```python
# From "went" -> "will go" examples
vocabulary["went"] = encode("went")
vocabulary["will go"] = encode("will go")
```

### Phase 2: Compute Canonical Deltas

```python
DELTAS = {
    ("tense", "future"): average_delta_from_corpus(...),
    ("tense", "past"): average_delta_from_corpus(...),
    ("regality", "noble"): average_delta_from_corpus(...),
}
```

### Phase 3: Transform via Nearest Neighbor

```python
def transform_word(word: str, delta: QuaternionPosition) -> str:
    word_pos = encode(word)
    new_pos = word_pos + delta
    return nearest_word(new_pos, vocabulary)

def transform_sentence(text: str, dimension: str, target_value: str) -> str:
    delta = DELTAS[(dimension, target_value)]
    words = tokenize(text)
    result = []
    for word in words:
        # Check if word is affected by this dimension
        if is_transformable(word, dimension):
            result.append(transform_word(word, delta))
        else:
            result.append(word)
    return " ".join(result)
```

### Phase 4: Test Self-Similarity

The key test: does the same delta work for all words?

```python
# If tense=future delta is truly self-similar:
delta = encode("will go") - encode("went")

# Then this should work:
encode("sat") + delta ≈ encode("will sit")
encode("walked") + delta ≈ encode("will walk")
```

If this holds, we have geometric transformation.
If not, we've found where the hypothesis breaks down.

## Success Criteria

1. **Geometric encoding works**: Words have consistent positions
2. **Deltas are self-similar**: Same delta transforms all words in a dimension
3. **Nearest neighbor decoding works**: Transformed positions map to correct words
4. **No LLM needed**: Pure geometry produces correct transformations

## Failure Modes (Valuable Data)

1. **Deltas not consistent**: Different word pairs have different deltas
   - This would mean transformations aren't purely geometric
   
2. **Nearest neighbor fails**: Transformed position doesn't map to expected word
   - This would mean the vocabulary space isn't structured correctly
   
3. **Composition breaks**: Sentence-level transforms don't decompose to words
   - This would mean sentences aren't compositional in our space

Each failure teaches us something about the limits of geometric representation.

## Connection to Core Philosophy

- **Structure IS information**: The delta vector IS the transformation
- **Geometry IS computation**: Adding delta IS the computation
- **ENCODE = DECODE**: Nearest neighbor in same space
- **Self-similarity**: Same delta at every scale

This is the true test of the hypothesis.
