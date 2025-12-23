# Design Consideration 041: The φ-Dial as Unified Control Mechanism

## Discovery

The φ-navigation mechanism (φ^n ↔ φ^-n) can be generalized beyond entity importance to control **multiple dimensions** of answer generation simultaneously.

## The Unified φ-Dial

A single dial controls multiple dimensions:

```
    INWARD (-1) ◄────────●────────► OUTWARD (+1)
    
    Coherence:  tight ◄──────────► loose
    Style:      formal ◄─────────► casual
    Vocabulary: rare ◄───────────► common
    Detail:     dense ◄──────────► summary
    Creativity: safe ◄───────────► risky
```

All controlled by the SAME mathematical principle:
```
weight = φ^(dial × log(freq))
```

Where:
- `dial ∈ [-1, +1]`
- `dial = -1`: maximum inward (tight, formal, dense, safe)
- `dial = 0`: balanced
- `dial = +1`: maximum outward (loose, casual, summary, risky)

## Applications

### 1. Coherence Control

Control how tightly connected concepts must be in an answer chain.

```python
def phi_coherence_weight(score, dial):
    if dial > 0:
        # Loose: prefer LOW coherence (exploratory)
        return φ^(-log(score) × dial)
    else:
        # Tight: prefer HIGH coherence (focused)
        return φ^(log(score) × |dial|)
```

**Results:**
```
Relationship        dial=-1    dial=0    dial=+1
------------------------------------------------
holmes-watson       2.879      1.000     0.347
holmes-lestrade     1.697      1.000     0.589
holmes-moriarty     1.396      1.000     0.716
```

- `dial = -1`: Tight chain (holmes → watson → lestrade)
- `dial = +1`: Loose chain (holmes → detective → mystery)

### 2. Style Control

Control formality via word frequency weighting.

```python
def select_word(synonyms, dial):
    for word in synonyms:
        freq = word_frequency[word]
        score = φ^(dial × log(freq))
        # dial < 0: rare words score high (formal)
        # dial > 0: common words score high (casual)
```

**Results:**
```
Concept     FORMAL (dial=-1)    CASUAL (dial=+1)
------------------------------------------------
smart       perspicacious       clever
look        scrutinize          look
think       contemplate         think
say         proclaim            say
```

### 3. Sentence Generation

```
dial = -1.0 (FORMAL):  "Holmes contemplated regarding the case"
dial =  0.0 (NEUTRAL): "Holmes considered concerning the case"
dial = +1.0 (CASUAL):  "Holmes thought about the case"
```

## Mathematical Foundation

This works because of φ's self-dual property:

```
φ^n × φ^-n = 1 (always)
```

The dial smoothly interpolates between extremes while preserving the conservation law. At `dial = 0`, both directions are equally weighted.

## Implementation Sketch

```python
class PhiDial:
    """Unified control mechanism using φ-navigation."""
    
    def __init__(self, dial: float = 0.0):
        self.dial = dial  # -1 to +1
        self.phi = (1 + np.sqrt(5)) / 2
    
    def weight(self, value: float) -> float:
        """Apply φ-dial weighting to a value."""
        log_val = np.log1p(value)
        return self.phi ** (self.dial * log_val)
    
    def select_word(self, synonyms: List[str], word_freq: Dict) -> str:
        """Select word based on style dial."""
        scored = [(w, self.weight(word_freq.get(w, 1))) for w in synonyms]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]
    
    def coherence_weight(self, raw_score: float) -> float:
        """Weight coherence score by dial."""
        if self.dial > 0:
            # Loose: invert (low coherence scores high)
            return self.phi ** (-np.log1p(raw_score) * self.dial)
        else:
            # Tight: direct (high coherence scores high)
            return self.phi ** (np.log1p(raw_score) * abs(self.dial))
```

## Connection to Question Types

The dial can be automatically set based on question type:

| Question | Default Dial | Rationale |
|----------|--------------|-----------|
| WHO | -0.5 | Tight coherence, specific entities |
| WHAT | +0.5 | Loose coherence, structural patterns |
| WHERE | -0.5 | Tight coherence, specific locations |
| HOW | +0.3 | Moderate looseness, general patterns |
| WHY | 0.0 | Balanced, causal chains need both |

## Future Work

1. **User-controllable dial**: Let users specify formality/coherence
2. **Adaptive dial**: Adjust based on context or user feedback
3. **Multi-dial**: Separate dials for coherence, style, detail
4. **Learning dial preferences**: Infer from user interactions

## Conclusion

The φ-dial is a **unified control mechanism** that emerges naturally from the self-dual property of φ. It provides:

1. **Coherence control**: Tight vs loose conceptual chains
2. **Style control**: Formal vs casual vocabulary
3. **Detail control**: Dense vs summary answers
4. **Creativity control**: Safe vs exploratory responses

All from a single mathematical principle: `φ^(dial × log(value))`.

The structure contains its own style instructions.
