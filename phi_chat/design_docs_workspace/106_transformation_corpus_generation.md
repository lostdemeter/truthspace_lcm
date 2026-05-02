# Design 106: Transformation Corpus Generation

## Goal

Build a corpus of sentence transformations that enables fluid dimensional manipulation without LLM calls. The system should be able to handle requests like:

- "Rewrite this sentence in future tense"
- "Make this sound more regal"
- "Convert to past tense with high formality"

## The Problem

We have quaternion dimensions (tense, regality, formality, etc.) but no vocabulary of transformations. To transform sentences geometrically, we need:

1. **Source sentences** - diverse examples covering different structures
2. **Transformed versions** - the same sentence shifted along each dimension
3. **Geometric relationships** - learned from the transformation pairs

## Strategy: LLM-Assisted Corpus Generation

Use our local LLM (Ollama) to generate transformation examples at scale, then store them for geometric learning.

### Phase 1: Seed Sentences

Create a diverse set of seed sentences that cover:

- **Different structures**: simple, compound, complex
- **Different subjects**: people, objects, abstract concepts
- **Different actions**: physical, mental, social
- **Different tenses**: past, present, future (as starting points)

Example seed sentences:
```
"Jack and Jill went up the hill to fetch a pail of water."
"The cat sat on the mat."
"She believes that honesty is the best policy."
"They will arrive at noon tomorrow."
"The ancient castle stands on the cliff overlooking the sea."
```

### Phase 2: Dimension Transformations

For each seed sentence, generate transformations along each dimension:

#### Grammatical Dimensions
| Dimension | Transformations |
|-----------|-----------------|
| Tense | past → present → future |
| Voice | active → passive |
| Mood | indicative → subjunctive → imperative |

#### Semantic Dimensions  
| Dimension | Transformations |
|-----------|-----------------|
| Regality | common → noble → royal |
| Formality | casual → neutral → formal |
| Certainty | uncertain → neutral → certain |

#### Combined Transformations
| Combination | Example |
|-------------|---------|
| future + regal | "Sir John and Lady Jill shall ascend the hill..." |
| past + casual | "Jack and Jill headed up the hill..." |
| formal + certain | "It is established that Jack and Jill proceeded..." |

### Phase 3: LLM Prompts

Structured prompts for consistent output:

```
Rewrite the following sentence in {dimension_value}:
Original: "{sentence}"
Rewritten:
```

For combined transformations:
```
Rewrite the following sentence to be {dim1_value} and {dim2_value}:
Original: "{sentence}"
Rewritten:
```

### Phase 4: Storage Format

Store in `corpus/transformation_corpus.json`:

```json
{
  "version": 1,
  "generated": "2026-01-07T...",
  "transformations": [
    {
      "source": "Jack and Jill went up the hill to fetch a pail of water.",
      "source_dimensions": {
        "tense": "past",
        "regality": "common",
        "formality": "neutral"
      },
      "target": "Jack and Jill shall go up the hill to fetch a pail of water.",
      "target_dimensions": {
        "tense": "future",
        "regality": "common", 
        "formality": "neutral"
      },
      "dimension_delta": {
        "tense": ["past", "future"]
      }
    }
  ]
}
```

### Phase 5: Geometric Learning

Once we have transformation pairs:

1. **Encode source and target** using QuaternionEncoder
2. **Compute delta vectors** - the geometric shift for each transformation
3. **Learn transformation patterns** - which words change, how positions shift
4. **Build transformation vocabulary** - "went" → "shall go" for tense shift

## Implementation Plan

### Script: `scripts/generate_transformation_corpus.py`

```python
# Pseudocode structure
class TransformationGenerator:
    def __init__(self, ollama_model="qwen2.5:14b"):
        self.model = ollama_model
        self.dimensions = self._load_dimensions()
        self.seeds = self._load_seed_sentences()
    
    def generate_all(self):
        for seed in self.seeds:
            for dim, values in self.dimensions.items():
                for target_value in values:
                    transformed = self._transform(seed, dim, target_value)
                    self._store(seed, transformed, dim, target_value)
    
    def _transform(self, sentence, dimension, target_value):
        prompt = f"Rewrite in {target_value}: {sentence}"
        return self._call_ollama(prompt)
```

### Dimensions to Cover

From our existing quaternion encoder:

**Structured (GrammaticalDim):**
- tense: past, present, future
- aspect: simple, progressive, perfect
- voice: active, passive
- mood: indicative, subjunctive, imperative

**Dynamic (from DynamicDimensionRegistry):**
- regality: common, noble, royal
- formality: casual, neutral, formal
- certainty: uncertain, neutral, certain
- age: young, adult, old (for subject transformation)
- gender: masculine, feminine, neutral

### Estimated Output

- 10 seed sentences
- 10 dimensions × 3 values each = 30 transformations per seed
- 10 × 30 = 300 transformation pairs (single dimension)
- Add combined transformations: ~500 total pairs

This should provide a solid baseline for geometric transformation learning.

## Success Criteria

1. **Coverage**: At least 300 transformation pairs covering all major dimensions
2. **Quality**: LLM outputs are grammatically correct and semantically appropriate
3. **Diversity**: Seed sentences cover different structures and topics
4. **Usability**: Corpus can be loaded and used for transformation requests

## Next Steps

1. Create seed sentence list (diverse, interesting examples)
2. Implement generator script with Ollama integration
3. Run generation (estimate: 10-15 minutes with local LLM)
4. Validate output quality
5. Integrate into ChatPipeline for transformation requests

## Connection to Hypothesis

This approach aligns with our core hypothesis:

> **Structure IS information** - The transformations reveal the geometric structure of language dimensions

By generating many examples, we're not training a model - we're **discovering the geometric relationships** that already exist. The LLM is a tool for rapid corpus generation, but the knowledge lives in the geometric structure we extract.

The goal is to reach a point where:
```
ENCODE(source) + DELTA(dimension) = ENCODE(target)
```

And we can decode back to text without needing the LLM.
