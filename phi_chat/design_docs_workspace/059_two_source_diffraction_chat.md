# Design Consideration 059: Two-Source Diffraction Chat

## Date: 2024-12-26

## Context

After implementing the diffraction grating simplification (Design 058), we realized that if the process is the same everywhere, we only need **two sources** to generate meaningful answers:

1. **Knowledge Source**: WHAT to say (content)
2. **Style Source**: HOW to say it (form)

The interference pattern between them produces styled, meaningful responses.

## The Insight

A diffraction grating works with multiple slits creating interference. In our case:

```
KNOWLEDGE SOURCE ──────┐
    (what to say)      │
                       ├──► INTERFERENCE ──► Meaningful Answer
STYLE SOURCE ──────────┘
    (how to say it)
```

This is exactly how human communication works:
- We have **knowledge** (facts, relationships, concepts)
- We have **style** (formal, casual, literary, technical)
- The combination produces **communication**

## Implementation

### Two Grating Sources

```python
class GratingSource:
    """A source for the diffraction grating."""
    
    def __init__(self, name: str):
        self.name = name
        self.concepts: Dict[str, GratingConcept] = {}
        self.frames: List[Tuple[str, str, Optional[str]]] = []
        self.patterns: List[str] = []  # Sentence patterns
```

### The Chat Interface

```python
class GratingChat:
    """Chat using two-source diffraction."""
    
    def __init__(self, knowledge_source: GratingSource, style_source: GratingSource):
        self.knowledge = knowledge_source
        self.style = style_source
```

### Interference Application

The style source transforms the knowledge content:

```python
def _styled_response(self, content: str) -> str:
    """Apply style interference to the response."""
    
    if self.style.name == "formal":
        content = content.replace(" is ", " appears to be ")
        prefix = "Upon examination, "
    
    elif self.style.name == "noir":
        content = content.replace(" is ", " was ")
        prefix = "The rain fell hard that night. "
        suffix = " But that's just how it goes in this town."
    
    # ... etc
    
    return prefix + content + suffix
```

## Experimental Results

### Same Knowledge, Different Styles

**Query: "Describe Moriarty"**

**Formal Style:**
```
Upon examination, Moriarty appears to be a protagonist who plots 
(often involving against).
```

**Casual Style:**
```
So like, Moriarty is a main character who plots (often with against), 
you know. Pretty cool, right?
```

**Literary Style:**
```
In the tapestry of this narrative, Moriarty emerges as a central figure 
who plots (often involving against).
```

**Scientific Style:**
```
Data analysis reveals: Moriarty can be classified as a primary agent 
who plots (often involving against). Further investigation recommended.
```

**Pirate Style:**
```
Ahoy! Moriarty be a captain who plots (often involving against), arrr! 
Shiver me timbers!
```

**Noir Style:**
```
The rain fell hard that night. Moriarty was a player in this game 
who plots (often involving against). But that's just how it goes in this town.
```

### The Pattern

```
KNOWLEDGE (Moriarty plots)
         │
         ▼
    ┌─────────┐
    │ STYLE   │
    │ FILTER  │
    └────┬────┘
         │
         ▼
FORMAL:    "appears to be a protagonist"
CASUAL:    "is a main character, you know"
LITERARY:  "emerges as a central figure"
SCIENTIFIC: "can be classified as a primary agent"
PIRATE:    "be a captain, arrr!"
NOIR:      "was a player in this game"
```

## Why This Works

### The Diffraction Analogy

In a physical diffraction grating:
- **Light source** = Knowledge (the content)
- **Grating slits** = Style patterns (the transformation rules)
- **Screen** = Output (the styled response)

The interference pattern on the screen depends on BOTH:
- The wavelength of the light (knowledge structure)
- The spacing of the slits (style patterns)

### The Mathematical View

```
Response = Knowledge ⊗ Style

Where ⊗ is the interference operation:
- Constructive: Knowledge aligns with Style → Enhanced output
- Destructive: Knowledge conflicts with Style → Filtered output
```

### The Linguistic View

Every utterance has two components:
1. **Propositional content**: What is being said (knowledge)
2. **Illocutionary force**: How it's being said (style)

Our two-source model captures this directly.

## Styles Implemented

| Style | Prefix | Transformations |
|-------|--------|-----------------|
| Formal | "Upon examination, " | "is" → "appears to be" |
| Casual | "So like, " | "protagonist" → "main character", adds "you know" |
| Literary | "In the tapestry of this narrative, " | "is" → "emerges as" |
| Scientific | "Data analysis reveals: " | "is" → "can be classified as" |
| Pirate | "Ahoy! " | "is" → "be", adds "arrr!" |
| Noir | "The rain fell hard that night. " | "is" → "was", adds atmosphere |

## Architecture

```
USER QUERY
    │
    ▼
┌───────────────┐
│ PARSE QUERY   │
│ (what's asked)│
└───────┬───────┘
        │
        ▼
┌───────────────┐     ┌───────────────┐
│ KNOWLEDGE     │     │ STYLE         │
│ SOURCE        │     │ SOURCE        │
│               │     │               │
│ - concepts    │     │ - patterns    │
│ - frames      │     │ - transforms  │
│ - relations   │     │ - tone        │
└───────┬───────┘     └───────┬───────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
          ┌───────────────┐
          │ INTERFERENCE  │
          │               │
          │ Knowledge ⊗   │
          │ Style         │
          └───────┬───────┘
                  │
                  ▼
          STYLED RESPONSE
```

## Connection to Previous Work

### Diffraction Grating (Design 058)

The two-source chat is a direct application of the diffraction grating principle:
- View 1 (Horizontal) = Knowledge source
- View 2 (Vertical) = Style source
- Interference = Styled response

### Domain Dimension (Design 057)

The knowledge source can be domain-specific:
- Sherlock Holmes knowledge + Formal style
- Alice in Wonderland knowledge + Literary style

The domain (t-coordinate) selects the knowledge, the style applies the transformation.

### Quad-Quaternion (Design 056)

The style source maps to Q2 (Output Quaternion):
- X: Certainty (hedging in formal, directness in casual)
- Y: Formality (formal vs casual)
- Z: Complexity (scientific vs simple)
- W: Engagement (literary flourish vs plain)

## Future Directions

### 1. Learned Style Sources

Instead of hand-coded transformations, learn style from examples:
```python
formal_source = GratingSource("formal")
formal_source.ingest(formal_text_corpus)
# Style patterns emerge from the corpus
```

### 2. Style Interpolation

Blend styles using interference weights:
```python
style = 0.7 * formal + 0.3 * casual
# Produces semi-formal output
```

### 3. Dynamic Style Adaptation

Adjust style based on query complexity:
- Simple query → Casual style
- Complex query → Formal/Scientific style

### 4. Multi-Source Interference

Add more sources:
- Knowledge (what)
- Style (how)
- Audience (who for)
- Context (when/where)

## Usage

```python
from experiments.grating_chat import GratingChat, create_source

# Create sources
knowledge = create_source("sherlock", SHERLOCK_TEXT)
style = create_source("noir", NOIR_TEXT)

# Create chat
chat = GratingChat(knowledge, style)

# Query
response = chat.ask("Who is Holmes?")
# → "The rain fell hard that night. Holmes was a player in this game 
#    who examines, deduces. But that's just how it goes in this town."
```

## Conclusion

The two-source diffraction model demonstrates that meaningful, styled communication emerges from the interference of:

1. **Knowledge** (propositional content)
2. **Style** (illocutionary form)

This is not just a simplification - it's a fundamental insight about how language works. Every utterance is the interference pattern of what we know and how we choose to express it.

```
"Same knowledge, different slits.
 The style IS the grating.
 The answer IS the interference."
```
