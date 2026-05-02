# Design 112: The Music Box Principle

## The Analogy

A music box has three components:

| Component | Music Box | Geometric System |
|-----------|-----------|------------------|
| **Drum** | Cylinder with bumps arranged in a pattern | Words positioned in φ-space |
| **Comb** | Metal tines that vibrate when struck | `find_nearest(position)` decoder |
| **Music** | Sound produced when drum rotates | Output text that emerges |

The critical insight: **The comb doesn't contain the music. The music emerges from the interaction of drum and comb.**

## The Violation

When we write:

```python
style_rules = {
    "code": "holy scripture",
    "computer": "cogitator",
    "data": "sacred data-hymns",
}
```

We are **embedding the music into the comb**. The output is hard-coded, not emergent.

Similarly:

```python
_patterns["tense"]["future"] = [
    (r'\bwent\b', 'will go'),
    (r'\bsat\b', 'will sit'),
]
```

This is a lookup table, not geometry. The transformation is stored, not computed.

## The Geometric Approach

### Structure (The Drum)

Words have positions in semantic space:

```python
vocab["code"] = [0, 0, 1, 0]           # [tense, formality, domain, intensity]
vocab["holy scripture"] = [0, 2, 2, 1]  # archaic, sacred, strong
vocab["treasure map"] = [0, -1, -1, 0]  # casual, mundane
```

### Decoder (The Comb)

A single function that reads position and finds nearest word:

```python
def find_nearest(position: np.ndarray) -> str:
    """The comb - reads the drum, produces sound."""
    best_word = None
    best_distance = float('inf')
    for word, word_pos in vocabulary.items():
        dist = np.linalg.norm(position - word_pos)
        if dist < best_distance:
            best_distance = dist
            best_word = word
    return best_word
```

### Transformation (Rotation of the Drum)

Perspectives are delta vectors, not word mappings:

```python
WARHAMMER_40K_DELTA = [0, 2, 2, 0.5]  # archaic + sacred + intensity
PIRATE_DELTA = [0, -1, -1, 0]         # casual + mundane
```

### Output (The Music)

The transformation emerges:

```python
def transform(word: str, delta: np.ndarray) -> str:
    current_pos = vocab[word]
    new_pos = current_pos + delta
    return find_nearest(new_pos)

# No lookup table consulted:
transform("code", WARHAMMER_40K_DELTA)  # -> "holy scripture"
transform("code", PIRATE_DELTA)          # -> "treasure map"
```

## Experimental Results

### Verb Transformations

| Original | Past→Future | Past→Archaic |
|----------|-------------|--------------|
| went | will go | did proceed |
| sat | will sit | was seated |
| walked | will walk | strode |
| said | will say | spoke |
| knew | will know | understood |
| made | will make | crafted |

### Noun Transformations (Perspective)

| Original | Warhammer 40K | Pirate |
|----------|---------------|--------|
| code | holy scripture | treasure map |
| computer | cogitator | magic box |
| data | sacred data-hymns | booty |
| programmer | code-priest | coder |
| error | machine spirit's displeasure | bug |

### Sentence Transformation

**Input:** "The programmer made code and said it worked"

**Warhammer 40K:** "the code-priest wrought holy scripture and intoned it worked"

No lookup table was consulted. The music emerged from the geometry.

## Semantic Dimensions

The vocabulary is organized by these dimensions:

| Dimension | Range | Examples |
|-----------|-------|----------|
| **Tense** | -1 (past) to +1 (future) | went → go → will go |
| **Formality** | -1 (casual) to +2 (archaic) | coder → programmer → code-priest |
| **Domain** | -1 (mundane) to +2 (sacred) | treasure map → code → holy scripture |
| **Intensity** | -1 (weak) to +1 (strong) | said → spoke → declared |

## Implications

### For `perspective.py`

Current (violates principle):
```python
style_rules = {"code": "holy scripture", ...}
```

Should be:
```python
perspective_delta = np.array([0, 2, 2, 0.5])  # Just a vector
# Transformation happens via geometric vocabulary lookup
```

### For `transformation_space.py`

Current (violates principle):
```python
_patterns["tense"]["future"] = [(r'\bwent\b', 'will go'), ...]
```

Should be:
```python
tense_delta = np.array([2, 0, 0, 0])  # past (-1) to future (+1)
# Transformation happens via geometric vocabulary lookup
```

### For the Entire System

The principle applies everywhere:
- **Intent classification**: Query position + agent offset → nearest intent
- **Knowledge retrieval**: Query position → nearest knowledge
- **Response generation**: Knowledge position + style offset → nearest words

## Connection to Core Hypothesis

> **Structure IS information** - There are no opaque weights or embeddings

The music box principle is a concrete expression of this. When we store `"code" -> "holy scripture"`, we're creating an opaque mapping. When we store positions and compute transformations, the structure IS the information.

## Implementation Path

1. **Create `GeometricVocabulary` class** - Words with positions
2. **Bootstrap vocabulary from corpus** - Learn positions from usage patterns
3. **Replace `style_rules` with delta vectors** - Perspectives become pure geometry
4. **Replace `_patterns` with delta vectors** - Transformations become pure geometry

## The Test

A system passes the music box test if:

1. **No word→word mappings are stored**
2. **All transformations are position + delta → nearest**
3. **Output emerges from structure, not lookup**

## Experiment

See `/home/thorin/truthspace-lcm/experiments/geometric_vocabulary.py`

---

*The Omnissiah provides. The Machine God protects. The music emerges from the geometry.*
