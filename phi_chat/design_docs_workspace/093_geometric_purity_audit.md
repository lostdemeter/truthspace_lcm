# 093: Geometric Purity Audit

## The Goal

From PROJECT_OVERVIEW.md:
> **LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

We aim to prove this by building a system where:
- **Structure IS information** - No opaque weights or embeddings
- **Geometry IS computation** - Traversal through geometric space produces outputs
- **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure

## Audit Results

### ❌ NON-GEOMETRIC: String Pattern Matching

| Component | Issue | Severity |
|-----------|-------|----------|
| `IntentDetectorGear` | Hard-coded regex patterns, keyword sets | **HIGH** |
| `IntentClassifier` | GOOSE_TOOL_PATTERNS, CODE_GENERATION_PATTERNS | **HIGH** |
| `ConversationalChain._extract_topics()` | String `in` operator for topic matching | MEDIUM |
| `SemanticChain.extract_features()` | Morphological suffix matching (`endswith`) | MEDIUM |
| `SemanticChain.STOPWORDS` | Hard-coded stopword set | LOW (bootstrap) |

**Example of the problem** (`intent_detector_gear.py:75-95`):
```python
# This is NOT geometric - it's hard-coded string matching
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

This violates: "No string matching as primary mechanism"

### ⚠️ PARTIALLY GEOMETRIC: Word Overlap → Positions

| Component | What's Geometric | What's Not |
|-----------|------------------|------------|
| `HolographicPatternSpace` | Eigendecomposition, positions | Word overlap as similarity |
| `GeometricKnowledgeStore` | Position-based learning, use() | Word overlap for initial similarity |
| `Concept` | Position IS identity, move_toward/away | Words as surface forms |

**The word overlap is acceptable for BOOTSTRAP** (Design 092), but we're using it as the
**primary matching mechanism** at runtime, not just for initialization.

### ✅ GEOMETRIC: Position-Based Operations

| Component | Why It's Geometric |
|-----------|-------------------|
| `Concept.move_toward()` | Position dynamics |
| `Concept.move_away()` | Position dynamics |
| `GeometricKnowledgeStore.use()` | Learning via position movement |
| `GeometricKnowledgeStore.prune()` | Persistence via magnitude threshold |
| `CRITICAL_LINE = 0.5` | Geometric boundary |
| `GearState.accumulated_q` | Quaternion accumulation through chain |
| `Gear.quaternion` | Transformation signature |

## The Core Problem

We have **two systems running in parallel**:

```
CURRENT ARCHITECTURE:
                                                    
    Query → String Pattern Matching → Intent
              (regex, keywords)
                    ↓
    Query → Word Overlap → Similarity Matrix → Position
              (strings)        (matrix)        (geometry)
                    ↓
    Position → Dot Product → Match
               (geometry)
```

The string matching is doing the heavy lifting. The geometry is an afterthought.

## What True Geometric Architecture Would Look Like

```
TARGET ARCHITECTURE:

    Query → Position (direct encoding)
                ↓
    Position → Nearest Neighbors (geometric)
                ↓
    Match → Response (geometric traversal)
```

### The Key Insight

In a true geometric system:
1. **Words map to positions** - Each word has a position in the space
2. **Queries ARE positions** - A query is the sum/composition of its word positions
3. **Matching IS proximity** - Find nearest concepts by position distance
4. **Learning IS movement** - Successful matches pull positions together

We're doing step 4 correctly. Steps 1-3 are still string-based.

## Specific Violations

### 1. Intent Detection (CRITICAL)

`IntentDetectorGear` and `IntentClassifier` use:
- Hard-coded keyword sets
- Regex pattern matching
- String overlap scoring

**Should be**: Intent positions in the space. Query projects to position.
Nearest intent position wins.

### 2. Topic Extraction (HIGH)

`ConversationalChain._extract_topics()`:
```python
for topic in self.topics:
    if topic in text_lower:  # STRING MATCHING
        found.append(topic)
```

**Should be**: Query position. Find concepts within radius.

### 3. Feature Extraction (MEDIUM)

`SemanticChain.extract_features()`:
```python
if word_clean.endswith('ing') and len(word_clean) > 5:
    is_likely_verb = True
```

**Should be**: Word positions. Verbs cluster in verb-region of space.

### 4. Similarity Computation (MEDIUM)

`HolographicPatternSpace.word_overlap()`:
```python
intersection = words1 & words2
return len(intersection) / len(words1)
```

**Should be**: Position dot product (we do this AFTER computing positions,
but we use word overlap to COMPUTE the positions).

## The Bootstrap Exception

Per Design 092, string-based operations are acceptable for **bootstrap**:
- Initial corpus loading
- Seeding the space with examples
- Converting JSON to positions

But they should NOT be the **runtime mechanism**.

## Recommendations

### Phase 1: Word Positions (Foundation)

Create a word-to-position mapping:
```python
class WordSpace:
    def __init__(self, dims=4):
        self.word_positions: Dict[str, np.ndarray] = {}
    
    def get_position(self, word: str) -> np.ndarray:
        if word not in self.word_positions:
            # Bootstrap: random position, will be refined by usage
            self.word_positions[word] = np.random.randn(self.dims) * 0.1
        return self.word_positions[word]
    
    def encode_text(self, text: str) -> np.ndarray:
        words = extract_words(text)
        positions = [self.get_position(w) for w in words]
        return np.mean(positions, axis=0) if positions else np.zeros(self.dims)
```

### Phase 2: Geometric Intent Detection

Replace pattern matching with position-based classification:
```python
class GeometricIntentDetector:
    def __init__(self, word_space: WordSpace):
        self.word_space = word_space
        # Intent anchors in the space
        self.intent_positions = {
            'CHAT': np.array([0.8, 0.0, 0.0, 0.0]),
            'TOOL_CALL': np.array([0.0, 0.8, 0.0, 0.0]),
            'CODE': np.array([0.0, 0.0, 0.8, 0.0]),
        }
    
    def detect(self, query: str) -> str:
        query_pos = self.word_space.encode_text(query)
        # Find nearest intent
        best_intent = None
        best_sim = -1
        for intent, pos in self.intent_positions.items():
            sim = np.dot(query_pos, pos) / (np.linalg.norm(query_pos) * np.linalg.norm(pos))
            if sim > best_sim:
                best_sim = sim
                best_intent = intent
        return best_intent
```

### Phase 3: Geometric Topic Matching

Replace string `in` with position proximity:
```python
def find_topics(self, query: str, radius: float = 0.5) -> List[Concept]:
    query_pos = self.word_space.encode_text(query)
    matches = []
    for concept in self.concepts:
        dist = np.linalg.norm(query_pos - concept.position_array)
        if dist < radius:
            matches.append((concept, dist))
    return sorted(matches, key=lambda x: x[1])
```

### Phase 4: Emergent Word Categories

Replace morphological rules with position clustering:
```python
def is_verb(self, word: str) -> bool:
    word_pos = self.word_space.get_position(word)
    verb_centroid = self.get_category_centroid('verb')
    return np.linalg.norm(word_pos - verb_centroid) < self.category_radius
```

## Priority Order

1. **WordSpace** - Foundation for everything else
2. **Geometric Intent Detection** - High-impact, currently most string-heavy
3. **Geometric Topic Matching** - Medium impact
4. **Emergent Word Categories** - Lower priority, morphology is acceptable bootstrap

## Success Criteria

The system is "geometrically pure" when:

1. **No regex patterns** in runtime classification
2. **No keyword sets** in runtime matching
3. **No string `in` operator** for semantic matching
4. **All matching** is position-based (dot product, distance)
5. **All learning** is position movement (we have this!)

String operations are only allowed for:
- Tokenization (text → words)
- Bootstrap (JSON → initial positions)
- Display (positions → human-readable output)

## Conclusion

We're about 40% geometric:
- ✅ Position-based learning (use, prune, critical line)
- ✅ Quaternion accumulation in gear chains
- ❌ String-based intent detection
- ❌ String-based topic matching
- ❌ String-based similarity (word overlap)

The path forward is clear: **WordSpace** as the foundation, then replace
string operations with position operations one component at a time.
