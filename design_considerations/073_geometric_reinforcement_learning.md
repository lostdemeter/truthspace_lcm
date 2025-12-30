# Design Consideration 073: Geometric Reinforcement Learning

**Date**: December 29, 2024  
**Author**: Lesley Gushurst  
**Status**: Experimental - Proof of Concept Working

## Executive Summary

We successfully demonstrated that corrections can propagate backward through the output lens to modify the underlying corpus. This is **geometric reinforcement learning** - learning through structural modifications rather than gradient descent.

## The Insight

From Design 072, we know that transformations in TruthSpace are **bidirectional**:
- If you can go king → queen, you can go queen → king
- If you can project forward through a lens, you can project backward

This means the output lens isn't just a one-way transformation - it's a **reversible projection**.

## The Architecture

```
FORWARD PATH (Generation):
    Corpus → GeometricQA → Raw Output → Lens → Natural Text

REVERSE PATH (Learning):
    Correction → Parse → Modifications → Corpus Update
         ↑                                    ↓
         └────────── Feedback Loop ───────────┘
```

## Experimental Results

### Before Corrections

| Query | Answer |
|-------|--------|
| What does Holmes do? | "examines and deduces" |
| What does Watson do? | "assists, watches and adventures" |

### After Corrections

| Query | Answer |
|-------|--------|
| What does Holmes do? | "**investigates, solves** and includes" |
| What does Watson do? | "assists, **provides and documents**" |

The new verbs from the corrections ("investigates", "solves", "documents", "provides") successfully propagated into the answers.

## How It Works

### 1. Parse the Correction

Extract structure from the corrected answer:
```python
corrected = "Holmes is a detective who investigates, deduces, and solves mysteries."

parsed = {
    'role': 'detective',
    'actions': ['investigates', 'deduces', 'solves'],
    'targets': ['mysteries', 'crime']
}
```

### 2. Compare with Original

Find what changed:
```python
original_actions = {'examines', 'deduces'}
corrected_actions = {'investigates', 'deduces', 'solves'}

to_add = corrected_actions - original_actions  # {'investigates', 'solves'}
```

### 3. Generate Corpus Modifications

Create frames that reinforce the corrections:
```python
modifications = [
    CorpusModification('add_action', 'holmes', {'action': 'investigates'}),
    CorpusModification('add_action', 'holmes', {'action': 'solves'}),
    CorpusModification('add_target', 'holmes', {'target': 'crime'}),
]
```

### 4. Apply to Corpus

Add reinforcement frames:
```python
for _ in range(strength):  # strength=10 by default
    knowledge.learn("Holmes investigates.", source="reinforcement")
    knowledge.learn("Holmes solves.", source="reinforcement")
```

### 5. Future Answers Improve

The new frames shift the concept's action distribution, causing future answers to include the corrected verbs.

## Key Insights

### 1. Reinforcement Through Repetition

Just like human learning, repetition strengthens associations. Adding multiple frames (strength=10) ensures the correction has enough weight to influence future answers.

### 2. Structure Preserving

The corrections don't break the geometric structure - they add to it. The concept's position in TruthSpace remains valid; only its action/target distributions change.

### 3. No Gradient Descent

Unlike neural network RL, this is purely structural:
- No loss function
- No backpropagation
- No optimizer
- Just frame addition and role count adjustment

### 4. Interpretable Changes

Every modification is human-readable:
```
add_action: holmes → investigates
add_target: holmes → crime
adjust_role: holmes → detective
```

## Comparison to Traditional RL

| Aspect | Neural RL | Geometric RL |
|--------|-----------|--------------|
| Learning signal | Reward scalar | Structured correction |
| Update mechanism | Gradient descent | Frame addition |
| Interpretability | Black box | Fully transparent |
| Sample efficiency | Low (many samples) | High (few corrections) |
| Catastrophic forgetting | Yes | No (additive) |

## Limitations

### 1. Verb Conjugation

The morphology system sometimes produces incorrect conjugations ("investigateses"). This is a known issue with the geometric morphology.

### 2. Action Extraction

Parsing natural language corrections is imperfect. Complex sentences may not parse correctly.

### 3. No Negative Learning

Currently we only add frames, not remove them. "Reduce action" modifications are noted but not applied.

### 4. Strength Tuning

The reinforcement strength (10 frames per correction) is arbitrary. Too low = no effect, too high = overfitting.

## Future Directions

### 1. Automatic Correction Generation

Use the output lens to automatically detect and correct awkward outputs:
```python
if "is a entity who" in raw_output:
    suggest_correction("is someone who")
```

### 2. Negative Reinforcement

Add "anti-frames" that reduce the weight of incorrect associations:
```python
# Instead of "Holmes teaches" (wrong)
# Add frames that contradict: "Holmes does not teach"
```

### 3. Confidence-Weighted Learning

Weight corrections by confidence:
```python
strength = int(correction.confidence * 20)  # 0-20 frames
```

### 4. Interactive Learning Loop

Real-time correction during chat:
```
User: What does Holmes do?
Bot: Holmes is someone who examines and deduces.
User: Actually, Holmes investigates and solves mysteries.
Bot: [applies correction] Thanks! I'll remember that.
```

## Connection to Design 072

This work builds directly on the self-similar transformation insights:

1. **Bidirectional transformations** → Reversible lens projection
2. **Structure-first approach** → Modifications preserve geometry
3. **Self-verification** → Corrections can be verified by re-asking

## Files

- `experiments/geometric_reinforcement.py` - Main experiment
- `truthspace_lcm/core/output_lens.py` - Forward projection lens
- `truthspace_lcm/corpus_experimental.json` - Modified corpus (for testing)

## Conclusion

**Geometric reinforcement learning works.** Corrections can propagate backward through the output lens to modify the corpus, and future answers reflect those corrections.

This opens the door to:
1. Interactive learning from user feedback
2. Automatic self-correction based on output quality
3. Continuous improvement without retraining

The key insight: **The lens is reversible. What we project forward, we can project backward.**

```
"Same lens, different direction.
 Forward projects knowledge to language.
 Backward projects corrections to knowledge.
 The structure learns from its own output."
```
