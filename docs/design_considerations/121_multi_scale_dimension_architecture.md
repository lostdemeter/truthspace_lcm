# Design Consideration 121: Multi-Scale Dimension Architecture

## The Insight

Dimensions don't just exist at one level - they form a **fractal hierarchy** across scales:

```
CHARACTER → WORD → PHRASE → SENTENCE → PARAGRAPH → SECTION → DOCUMENT
    ↓         ↓       ↓         ↓           ↓          ↓         ↓
 spacing   gender   idiom     tone      structure  argument   genre
 case      size    register   meter     coherence  narrative  audience
 leet    formality verbosity  voice     topic_flow  pacing    purpose
```

Each scale is roughly **φ² ≈ 2.6x** larger than the previous.

## The Scale Hierarchy

| Scale | Typical Length | Example Dimensions |
|-------|----------------|-------------------|
| CHARACTER | ~1 char | spacing, case, substitution |
| WORD | ~3 chars | gender, size, formality |
| PHRASE | ~8 chars | idiom, register, collocation |
| SENTENCE | ~20 chars | tone, meter, speech_act |
| PARAGRAPH | ~50 chars | structure, coherence, density |
| SECTION | ~130 chars | argument, narrative, pacing |
| DOCUMENT | ~340 chars | genre, audience, purpose |

## Fractal Self-Similarity

The same dimension types appear at multiple scales:

### "Formality" Across Scales

| Scale | Negative Pole | Positive Pole |
|-------|---------------|---------------|
| WORD | guy | gentleman |
| PHRASE | what's up | how do you do |
| SENTENCE | casual register | formal register |
| DOCUMENT | blog post | academic paper |

### "Complexity" Across Scales

| Scale | Negative Pole | Positive Pole |
|-------|---------------|---------------|
| WORD | simple | complex |
| SENTENCE | short | elaborate |
| PARAGRAPH | sparse | dense |
| DOCUMENT | brief | comprehensive |

This is **true fractal structure** - the same pattern repeating at every scale.

## Cross-Scale Composition

Using φ-weighted composition, higher scales dominate lower scales:

```python
position = (document_pos * φ^0 +      # Dominant: genre
            section_pos * φ^(-1) +     # Major: argument structure
            paragraph_pos * φ^(-2) +   # Medium: coherence
            sentence_pos * φ^(-3) +    # Minor: tone
            word_pos * φ^(-4) +        # Detail: word choice
            character_pos * φ^(-5))    # Micro: stylization
```

### Example Composition

```
paper@DOCUMENT                           → [0, 0, 0, 1.618]
paper@DOCUMENT + formal@SENTENCE         → [0, 0, 1, 1.618]
paper@DOCUMENT + formal@SENTENCE + queen@WORD → [0, 0.618, 1, 1.618]
paper@DOCUMENT + formal@SENTENCE + queen@WORD + vaporwave@CHARACTER 
                                         → [0.382, 0.618, 1, 1.618]
```

Document-level choices constrain all lower levels.

## Auto-Detection of Relevant Scales

Given text, we can detect which scales are relevant:

| Input | Relevant Scales |
|-------|-----------------|
| "hello" | CHARACTER, WORD |
| "The king ruled wisely." | CHARACTER, WORD, PHRASE, SENTENCE |
| Two sentences | + PARAGRAPH |
| Multiple paragraphs | + SECTION |
| Full document | ALL SCALES |

## Implementation

### Scale Enum

```python
class Scale(Enum):
    CHARACTER = 0
    WORD = 1
    PHRASE = 2
    SENTENCE = 3
    PARAGRAPH = 4
    SECTION = 5
    DOCUMENT = 6
    
    @property
    def typical_length(self) -> int:
        return int(PHI ** (2 * self.value))
```

### Scaled Dimension

```python
@dataclass
class ScaledDimension:
    name: str           # e.g., "formality"
    scale: Scale        # e.g., Scale.WORD
    negative_pole: str  # e.g., "guy"
    positive_pole: str  # e.g., "gentleman"
    
    @property
    def full_name(self) -> str:
        return f"{self.scale.name.lower()}:{self.name}"
```

### Multi-Scale Corpus

```python
class MultiScaleCorpus(SelfAssemblingCorpus):
    def add_scaled_pair(self, source, target, dimension, scale):
        full_dim = f"{scale.name.lower()}:{dimension}"
        return self.add_pair(source, target, full_dim)
    
    def compose_multi_scale(self, *concepts_with_scales):
        # Sort by scale (highest first)
        # Weight by φ^(-rank)
        # Compose into single position
```

## The Meta-Insight

> **The scale IS a dimension.**

We can treat scale as just another axis in the geometry:
- Moving along the scale axis = zooming in/out
- Each position has coordinates at multiple scales
- The same φ-geometry applies at every scale

This makes the system **infinitely scalable**.

## Extended Universal Dimension Principle

From Design Doc 120:
> "ANY transformation can be a dimension."

Extended:
> **"ANY transformation at ANY scale can be a dimension."**

## Implications

### 1. Hierarchical Constraints

Higher scales constrain lower scales:
- If document is "academic paper", sentences must be formal
- If paragraph is "argument", sentences must support claims
- Constraints flow downward through the hierarchy

### 2. Emergent Coherence

By composing across scales, coherence emerges:
- Document-level genre sets the frame
- Section-level structure provides organization
- Paragraph-level flow maintains continuity
- Sentence-level tone creates voice
- Word-level choices add precision
- Character-level styling adds flair

### 3. Scale-Aware Self-Assembly

The self-assembly loop can operate at any scale:
1. Detect scale of input text
2. Find relevant dimensions at that scale
3. Discover new dimensions from examples
4. Propagate constraints up/down the hierarchy

### 4. Fractal Compression

Because the same patterns repeat at every scale:
- Learn "formality" once, apply at all scales
- Cross-scale dimensions share structure
- Massive parameter efficiency

## Connection to LLM Hypothesis

This validates another aspect of our hypothesis:

> LLMs learn transformations at multiple scales simultaneously

The attention mechanism in transformers naturally handles multi-scale:
- Local attention = character/word scale
- Medium attention = phrase/sentence scale
- Global attention = paragraph/document scale

Our explicit scale hierarchy makes this structure visible.

## Next Steps

1. **Integrate into unified self-assembly loop**
2. **Test scale propagation** - do constraints flow correctly?
3. **Implement scale-aware transforms** - apply at right level
4. **Explore cross-scale ideals** - concepts that anchor multiple scales

---

*"The scale is the dimension. The dimension is the scale."*
