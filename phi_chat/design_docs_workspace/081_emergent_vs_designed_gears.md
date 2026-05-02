# Design Consideration 081: Emergent vs Designed Gears

**Date**: December 30, 2024  
**Status**: Experimental Analysis

## Executive Summary

We ran an experiment to let the system discover its own gears from LLM-generated behavioral data, then compared what emerged vs what we intentionally designed.

**Key Finding**: The emergent system discovered **14 dimensions** explaining **87.2% of variance**, including recognizable concepts like Agency, Maturity, and Morality - but these emerged from CHARACTER BEHAVIOR, not linguistic theory.

## The Experiment

### Method
1. Generated 294 behavioral sentences using Ollama (qwen2) for 20 characters
2. Let the system discover dimensions via SVD on verb-usage patterns
3. Automatically built gears for each discovered dimension
4. Compared with our intentionally designed gears

### LLM-Generated Corpus
- 20 characters (Holmes, Watson, Moriarty, Alice, King, Servant, etc.)
- ~15 sentences per character
- Each sentence shows the character DOING something
- No predefined dimensional labels

## Results

### Emergent Dimensions Discovered

| Dimension | Variance | Poles | Key Features |
|-----------|----------|-------|--------------|
| Dim1 | 15.0% | holmes ↔ hearts | methodically, logically vs dances, servant |
| Dim2 | 13.1% | king ↔ hearts | king, servant vs negotiations, instructs |
| Dim3 (Maturity) | 10.0% | king ↔ servant | kingdoms vs whispers, guides |
| Dim4 | 8.5% | idea ↔ servant | patterns, mysteries vs holmes, hearts |
| Dim5 (Agency) | 5.5% | hero ↔ widow | bravely vs writes, organizes |
| Dim6 | 4.9% | alice ↔ moriarty | dances, whispers vs expertly, meticulously |
| Dim7 (Maturity) | 4.8% | alice ↔ idea | dances vs systems, contemplates |
| Dim8 (Agency) | 4.5% | villain ↔ bennet | secrets, devices vs listens, writes |
| Dim9 (Maturity) | 4.1% | sage ↔ child | navigates vs plays, energetically |
| Dim10 | 3.8% | alice ↔ child | negotiates vs carefully, stars |
| Dim11 | 3.4% | storm ↔ hero | across, devices vs trains, memories |
| Dim12 (Morality) | 3.3% | dog ↔ villain | conducts, performs vs patterns, negotiates |
| Dim13 | 3.2% | villain ↔ storm | plants, devices vs bolts, causes |
| Dim14 (Maturity) | 3.1% | robot ↔ widow | analyzes vs attends, uncovers |

**Total: 14 dimensions, 87.2% variance explained**

### Designed Gears (Intentional)

| Gear | Purpose | Based On |
|------|---------|----------|
| RoleGear | Semantic roles (agent, patient) | Linguistic theory |
| TenseGear | Temporal aspects | Grammar |
| MoodGear | Modality (certainty, possibility) | Grammar |
| VoiceGear | Active/passive transformations | Grammar |
| AspectGear | Perfective/imperfective | Grammar |
| PolarityGear | Negation/affirmation | Grammar |

### Comparison

| Aspect | Emergent | Designed |
|--------|----------|----------|
| **Basis** | Data variance | Linguistic theory |
| **Number** | 14 (auto-determined) | 6 (predefined) |
| **Focus** | Character behavior | Sentence structure |
| **Discovered** | Agency, Maturity, Morality | Roles, Tense, Mood |
| **Overlap** | None direct | None direct |

## Key Insights

### 1. Different Levels of Analysis

The emergent system operates at the **semantic/character level**:
- Who does what? (Agency)
- How mature are they? (Maturity)
- Are they good or evil? (Morality)

The designed system operates at the **syntactic/sentence level**:
- What role does this word play? (Role)
- When did this happen? (Tense)
- How certain is this? (Mood)

**These are complementary, not competing.**

### 2. Emergent Dimensions Are Character-Centric

The system discovered dimensions that separate CHARACTERS based on behavior:
- Holmes: methodical, logical, investigative
- Watson: supportive, documenting, assisting
- Villain: secretive, scheming, manipulative
- Child: playful, energetic, learning

This is exactly what we'd want for understanding narrative and character dynamics.

### 3. Designed Gears Are Sentence-Centric

Our designed gears handle transformations WITHIN sentences:
- "Holmes investigates" → "Holmes investigated" (Tense)
- "Holmes investigates" → "The case is investigated by Holmes" (Voice)
- "Holmes investigates" → "Holmes might investigate" (Mood)

### 4. The Gap Reveals Opportunity

Neither system discovered:
- **Emotion** (happy/sad/angry)
- **Certainty** (sure/unsure)
- **Formality** (formal/casual)
- **Perspective** (1st/2nd/3rd person)

These might emerge with different data or different extraction methods.

## Concept Analysis Results

### Holmes
- Similar to: watson, stranger, dog
- Opposite of: servant
- Dimensions: negative on holm_hear (toward holmes)

### Watson
- Similar to: darcy, widow, bennet
- Opposite of: alice
- Dimensions: negative on holm_hear, positive on Agency (toward bennet)

### Villain
- Similar to: watson, moriarty, adler
- Opposite of: hearts
- Dimensions: negative on Agency, positive on Morality (toward villain)

### Hero
- Similar to: stranger, adler, watson
- Opposite of: alice
- Dimensions: positive on Maturity, Agency, Morality

## Implications

### 1. Use Both Systems Together

```
Query → Emergent Gears (character understanding)
     → Designed Gears (sentence transformation)
     → Response
```

### 2. Let Data Guide Discovery

The emergent system found dimensions we didn't think to design:
- Character-level agency (who acts vs who is acted upon)
- Maturity patterns (sage vs child behaviors)
- Moral alignment (hero vs villain behaviors)

### 3. Designed Gears Fill Gaps

The designed gears handle things the emergent system can't discover from behavioral data:
- Grammatical transformations
- Temporal shifts
- Modal variations

## Files Created

| File | Purpose |
|------|---------|
| `experiments/llm_live_corpus_generator.py` | Real LLM corpus generation |
| `experiments/self_discovering_gears.py` | Self-building gear system |
| `truthspace_lcm/gears/corpus/corpus_llm_live.json` | LLM-generated corpus |

## Conclusion

**The emergent system discovers CHARACTER-LEVEL dimensions from behavior.**
**The designed system handles SENTENCE-LEVEL transformations from theory.**

Both are valuable. The emergent system tells us WHAT dimensions exist in the data.
The designed system tells us HOW to transform along known dimensions.

The ideal system combines both:
1. Discover dimensions from data (emergent)
2. Apply known transformations (designed)
3. Let errors guide new dimension discovery (self-improving)

---

*"The system discovered Agency, Maturity, and Morality from behavior alone."*

*"Designed gears handle grammar. Emergent gears handle meaning."*
