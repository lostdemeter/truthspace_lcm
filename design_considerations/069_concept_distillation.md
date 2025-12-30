# Design Consideration 069: Concept Distillation & Correction Learning

**Date**: December 28, 2024  
**Status**: Implemented  
**Author**: Lesley Gushurst

## Overview

This document describes two related advances in the GeometricLCM:

1. **Concept Distillation**: Extracting pure geometric concepts from the frame-based corpus, eliminating text storage while preserving inference capability.

2. **Correction Learning**: A mechanism for the model to learn from corrected outputs, enabling supervised refinement without traditional backpropagation.

## Part 1: Concept Distillation

### The Insight

Text is *evidence* for structure. Once we've extracted the geometric relationships from text, we don't need the text for inference—only for citation.

The frame-based corpus stores:
```json
{
  "initiator": "holmes",
  "mediator": "examine",
  "receiver": "evidence",
  "source": "Grokipedia-Sherlock_Holmes",
  "text": "Holmes carefully examines the evidence left at the crime scene"
}
```

But for inference, we only need:
- Holmes → examines → evidence (relationship)
- Holmes has φ-direction +0.714 (active agent)
- Holmes frequency: 847 (well-attested)

### The Distilled Format

Each concept becomes a compact array:
```
[φ_direction, frequency, i_count, m_count, r_count, [[action, count]...], [[target, count]...]]
```

Example:
```json
"holmes": [0.714, 847, 612, 89, 146, [["examine", 38], ["deduce", 27]], [["evidence", 32], ["mystery", 24]]]
```

### Compression Results

| Metric | Full Corpus | Distilled Model |
|--------|-------------|-----------------|
| Size | 7.5 MB | 785 KB |
| Compression | — | **9.6x smaller** |
| Concepts | 23,936 | 13,529 (freq ≥ 3) |
| Text stored | Yes | **No** |

### What's Preserved

The distilled model retains:
- **φ-direction**: Agency/passivity of each concept
- **Role counts**: How often concept appears as initiator/mediator/receiver
- **Actions**: What the concept does (top-k verbs)
- **Targets**: What the concept acts upon (top-k nouns)
- **Relationships**: Concept-to-concept edges with weights
- **Morphology**: Word form equivalences

### Inference Without Text

The distilled model can:
- Describe concepts from relationships
- Find related concepts
- Compute similarity
- Navigate concept space
- Answer questions

All without storing a single sentence.

### Implementation

- `scripts/distill_concepts.py`: Distillation script
- `truthspace_lcm/core/distilled_lcm.py`: Loader and inference
- `truthspace_lcm/concepts_distilled.json`: Distilled model

---

## Part 2: Correction Learning

### The Problem

The model produces: "Sherlock Holmes is a teacher"  
We want: "Sherlock Holmes is a consulting detective"

How do we correct this without traditional gradient descent?

### The Geometric Solution

In our model, this error means:
- Holmes's **actions** don't include "consult" or "detect" strongly enough
- Holmes's **targets** don't include "case" or "crime" strongly enough
- The relationship Holmes → detective is weak or missing

### Correction as Frame Injection

A correction is simply a new frame with high confidence:

```python
correction = {
    "initiator": "holmes",
    "mediator": "be",  # or "consult", "detect"
    "receiver": "detective",
    "source": "Correction",
    "confidence": 2.0,  # Higher weight than normal frames
}
```

### The Correction Protocol

```
INPUT:  (question, wrong_answer, correct_answer)
OUTPUT: Updated concept relationships

1. Parse wrong_answer → extract (subject, predicate, object)
2. Parse correct_answer → extract (subject, predicate, object)
3. Compute delta:
   - What relationships are WRONG (should be weakened)
   - What relationships are CORRECT (should be strengthened)
4. Apply corrections:
   - Decrease weight of wrong relationships
   - Increase weight of correct relationships
   - Add new relationships if missing
```

### Example Correction

**Wrong**: "Sherlock Holmes is a teacher"
**Correct**: "Sherlock Holmes is a consulting detective"

**Delta Analysis**:
```
WEAKEN:  holmes → be → teacher
STRENGTHEN: holmes → be → detective
ADD: holmes → consult → (implicit)
ADD: detective → (target of) → holmes
```

**Applied Changes**:
```python
# In concept "holmes":
actions["teach"] -= 1  # or remove if count becomes 0
actions["consult"] += 2  # correction weight
targets["teacher"] -= 1
targets["detective"] += 2

# In concept "detective":
# Strengthen bidirectional relationship
```

### Confidence Weighting

Corrections should have higher confidence than regular ingestion:
- Normal frame: weight = 1.0
- Correction frame: weight = 2.0 (or configurable)
- Repeated corrections: weight compounds

This ensures corrections "stick" even against large corpus evidence.

### The Correction API

```python
class CorrectionLearner:
    def correct(self, question: str, wrong: str, right: str):
        """Learn from a correction."""
        # Extract frames from both answers
        wrong_frame = self.extract_frame(wrong)
        right_frame = self.extract_frame(right)
        
        # Compute what changed
        if wrong_frame.initiator == right_frame.initiator:
            # Same subject, different predicate/object
            self.weaken_relationship(
                wrong_frame.initiator,
                wrong_frame.mediator,
                wrong_frame.receiver
            )
            self.strengthen_relationship(
                right_frame.initiator,
                right_frame.mediator,
                right_frame.receiver
            )
        
        # Re-distill affected concepts
        self.update_distilled_model()
```

### Batch Corrections (RLHF-like)

For systematic improvement:

```python
corrections = [
    ("Who is Holmes?", "Holmes is a teacher", "Holmes is a consulting detective"),
    ("What does Watson do?", "Watson is a cook", "Watson is a doctor and Holmes's companion"),
    ("Where does Holmes live?", "Holmes lives in Paris", "Holmes lives at 221B Baker Street"),
]

for question, wrong, right in corrections:
    learner.correct(question, wrong, right)
```

### Connection to RLHF

This is analogous to Reinforcement Learning from Human Feedback, but geometric:

| RLHF (Neural) | Correction Learning (Geometric) |
|---------------|--------------------------------|
| Reward model | Correction signal |
| Gradient update | Relationship weight adjustment |
| Policy optimization | φ-direction refinement |
| Catastrophic forgetting risk | Localized updates only |

**Key advantage**: Our corrections are *localized*. Correcting "Holmes is a detective" doesn't affect unrelated concepts like "physics" or "philosophy".

### Preventing Overcorrection

To avoid oscillation:
1. **Decay**: Old corrections decay over time unless reinforced
2. **Bounds**: Relationship weights have min/max bounds
3. **Consensus**: Multiple corrections needed to override strong corpus evidence

---

## Part 3: The Unified Vision

### Text → Concepts → Corrections → Better Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE LIFECYCLE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   RAW TEXT ──────► FRAMES ──────► DISTILLED CONCEPTS        │
│   (evidence)       (structure)    (pure geometry)           │
│                                                              │
│                         ▲                                    │
│                         │                                    │
│                    CORRECTIONS                               │
│                    (refinement)                              │
│                         │                                    │
│                         ▼                                    │
│                                                              │
│   QUESTION ──────► INFERENCE ──────► ANSWER                 │
│                    (geometric)       (generated)             │
│                                                              │
│                         │                                    │
│                         ▼                                    │
│                                                              │
│                    EVALUATION                                │
│                    (human/auto)                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### The Goal

A self-improving system where:
1. **Ingestion** builds initial knowledge from text
2. **Distillation** compresses to pure concepts
3. **Inference** generates answers from geometry
4. **Correction** refines based on feedback
5. **Re-distillation** incorporates corrections

The text becomes disposable scaffolding. The concepts are the knowledge.

---

## Implementation Status

### Completed
- [x] Concept distillation script (`scripts/distill_concepts.py`)
- [x] Distilled model loader (`truthspace_lcm/core/distilled_lcm.py`)
- [x] Basic inference from distilled model
- [x] 9.6x compression achieved
- [x] Keyword-level CorrectionLearner (`truthspace_lcm/core/correction_learner.py`)
- [x] Concept-level ConceptCorrector (`truthspace_lcm/core/concept_correction.py`)
- [x] Answer generation updated to use category from targets

---

## Part 4: Key Discovery - Changing Ideas Post-Training

### The Problem: Keywords vs Concepts

Our initial correction approach injected frames like:
```
holmes → consult → client
holmes → detect → crime
```

This added "consults" as a keyword in Holmes's actions, but **didn't change Holmes's identity**. The model still said:

> "Holmes is a notable entity who includes, examines, and deduces"

The word "consults" appeared, but Holmes wasn't recognized as a *detective*.

### The Root Cause

Answer generation used φ-direction to assign generic roles:
```python
if c.phi_direction > 0.3:
    role = "protagonist"
elif c.phi_direction < -0.3:
    role = "concept"
else:
    role = "entity"
```

This produced generic labels ("protagonist", "entity") regardless of what the concept actually *is*.

### The Solution: Category from Targets

When we say "Holmes is a detective", the frame is:
```
holmes → be → detective
```

The word "detective" goes into Holmes's **targets**. The fix:

1. **Look at targets for category words** (detective, doctor, scientist, etc.)
2. **Require count >= 3** to avoid incidental mentions
3. **Fall back to φ-direction** if no category found

```python
category_words = {'detective', 'doctor', 'scientist', 'teacher', 'writer',
                 'philosopher', 'artist', 'leader', 'hero', 'villain', ...}

role = None
if c.targets:
    for target, count in c.targets.most_common(10):
        if target in category_words and count >= 3:
            role = target
            break

if not role:
    # Fall back to φ-direction
    role = "protagonist" if c.phi_direction > 0.3 else "entity"
```

### ConceptIdentity: Defining What a Concept IS

To change an entire idea, we define its **identity**:

```python
@dataclass
class ConceptIdentity:
    word: str                    # The concept
    category: str                # What it IS (detective, doctor, science)
    primary_actions: List[str]   # What it DOES (investigate, deduce)
    primary_targets: List[str]   # What it acts ON (crime, evidence)
    related_concepts: List[str]  # What it's CONNECTED to (watson, moriarty)
```

The `ConceptCorrector` converts this identity into multiple weighted frames:

```python
def define_identity(self, identity: ConceptIdentity):
    # Generate frames for category (10x weight)
    for _ in range(10):
        self._add_frame(identity.word, 'be', identity.category)
    
    # Generate frames for actions (5x weight each)
    for action in identity.primary_actions:
        for _ in range(5):
            self._add_frame(identity.word, action, 'cases')  # or appropriate target
    
    # Generate frames for relationships
    for related in identity.related_concepts:
        self._add_frame(identity.word, 'associate', related)
```

### Results: Before and After

**Before Correction:**
```
Q: Who is Holmes?
A: Holmes is a notable entity who includes, examines, and deduces

Q: Who is Watson?
A: Watson is a notable protagonist who assists, watches, and adventures
```

**After Applying ConceptIdentity:**
```
Q: Who is Holmes?
A: Holmes is a notable detective who associates, is, and investigates

Q: Who is Watson?
A: Watson is a notable doctor who associates, is, and assists
```

The concepts now have their correct **identities**, not just keywords.

### The Geometric Insight

In concept space, an **idea** is defined by:
1. **Category** → What the concept IS (stored in targets via "X is a Y" frames)
2. **Actions** → What the concept DOES (stored in actions counter)
3. **Relations** → What the concept connects to (stored in targets/initiators)

Changing an idea requires changing all three, with sufficient weight to override existing corpus evidence.

### Why This Matters

This is fundamentally different from neural network fine-tuning:

| Neural Fine-Tuning | Geometric Correction |
|-------------------|----------------------|
| Adjusts all weights | Adjusts specific relationships |
| Risk of catastrophic forgetting | Localized, no side effects |
| Requires gradient computation | Direct weight injection |
| Black box | Interpretable |
| Statistical | Surgical |

We can literally **see** what changed: Holmes's targets now include "detective" with count >= 3.

---

### Next Steps
- [ ] Add correction persistence across sessions
- [ ] Create correction API endpoint
- [ ] Build evaluation/correction UI
- [ ] Implement automatic category expansion (learn new category words)
- [ ] Explore demotion (weakening incorrect associations)

---

## Conclusion

Concept distillation proves that text is evidence, not knowledge. The geometric structure *is* the knowledge.

Correction learning extends this: we can refine the geometry directly, without re-ingesting text. This is a fundamentally different approach from neural network fine-tuning—it's *surgical* rather than *statistical*.

The combination enables a knowledge system that:
- Stores meaning, not tokens
- Learns from corrections, not gradients
- Improves locally, not globally
- Scales by concepts, not parameters

This is the path to a truly geometric language model.
