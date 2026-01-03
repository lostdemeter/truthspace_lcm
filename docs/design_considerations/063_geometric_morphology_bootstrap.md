# Design Consideration 063: Geometric Morphology Bootstrap

## Date: 2024-12-26

## Problem

The morphological transformer (Q3) uses a hard-coded lookup table:
```python
self.irregulars = {
    'love': {'past': 'loved', 'present_3rd': 'loves'},
    ...
}
```

This is:
1. **Language-specific** - Only works for English
2. **Non-geometric** - Breaks the principle of geometric understanding
3. **Incomplete** - Requires manual addition of every verb

## Solution

Learn morphological equivalence from **parallel structures** in concept space.

## The Geometric Principle

**Parallel sentences reveal morphological equivalence:**

```
"I love. I loved."
"He runs. He ran."
"She sees. She saw."
```

Words in the **same frame slot** across **consecutive parallel sentences** 
are the **same concept at different temporal phases**.

This is purely geometric:
- **Position** = concept identity
- **Phase** = temporal aspect

No suffix patterns. No word length heuristics.

## Implementation

```python
def learn_parallel(self, sentences: List[str]):
    """
    CONSECUTIVE sentences with the same initiator reveal that
    their mediators are morphological variants of the same concept.
    """
    prev_initiator = None
    current_group: List[str] = []
    
    for sentence in sentences:
        frame = self._extract_frame(sentence)
        initiator, mediator, receiver = frame
        
        if initiator == prev_initiator:
            # Same initiator = same parallel group
            current_group.append(mediator)
        else:
            # New initiator = end previous group, start new one
            if len(current_group) > 1:
                self._create_equivalence(current_group)
            current_group = [mediator]
            prev_initiator = initiator
```

## Bootstrap Text

The bootstrap is a small set of parallel sentences that teach morphological patterns:

```
I love. I loved.
He runs. He ran.
She sees. She saw.
They watch. They watched.
We go. We went.
It falls. It fell.
...
```

This bootstrap:
1. Translates directly to concept space
2. Teaches patterns, not knowledge
3. Is language-independent in principle (any language could have its own bootstrap)

## Results

From the bootstrap, the system learns:

```
MORPHOLOGICAL EQUIVALENCES (Geometric Bootstrap)
============================================================
  love ≡ loved
  runs ≡ ran
  sees ≡ saw
  watch ≡ watched
  go ≡ went
  falls ≡ fell
  speak ≡ spoke
  write ≡ wrote
  think ≡ thought
  ...
```

## The Chicken-and-Egg Solution

The user noted: "When humans learn there's usually combined stimuli. Both visual 
cues and verbal ones work to teach young children words and concepts."

The bootstrap serves as this "combined stimulus":
- The **parallel structure** is the visual/structural cue
- The **words themselves** are the verbal cue
- Together, they teach the pattern without hard-coding rules

## Integration

The geometric morphology is integrated into `phi_geometric.py`:

```python
class PhiGeometric:
    def __init__(self):
        # Geometric morphology - learned from parallel structures
        self.geo_morpho = GeometricMorphology()
        self.geo_morpho.bootstrap(MORPHOLOGY_BOOTSTRAP)
    
    def _who_does(self, action: str) -> str:
        # Use geometric morphology for matching
        equivalents = self.geo_morpho.get_equivalents(action)
        
        for act in c.actions:
            if act in equivalents or self.geo_morpho.are_equivalent(act, action):
                actors.append((name, c.actions[act]))
```

## The Fully Geometric Pipeline

```
BOOTSTRAP (parallel structures)
    │
    ▼
┌─────────────────────────────────────┐
│ GEOMETRIC MORPHOLOGY                │
│                                     │
│ love ≡ loved (same position)        │
│ runs ≡ ran (same position)          │
│ sees ≡ saw (same position)          │
└─────────────────────────────────────┘
    │
    ▼
CORPUS (knowledge)
    │
    ▼
┌─────────────────────────────────────┐
│ FRAME EXTRACTION (position-based)   │
│                                     │
│ Position 0.0-0.33 → Initiator       │
│ Position 0.33-0.66 → Mediator       │
│ Position 0.66-1.0 → Receiver        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ STOP WORD DETECTION (role-based)    │
│                                     │
│ No semantic role → Stop word        │
│ Has semantic role → Content word    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ QUERY → ENCODE → DECODE → RESPONSE  │
│                                     │
│ Uses geometric morphology for       │
│ verb form matching                  │
└─────────────────────────────────────┘
```

## What's Now Geometric

| Component | Before | After |
|-----------|--------|-------|
| Stop words | Hard-coded list | Detected by semantic role |
| Frame slots | Ordinal (1st, 2nd, 3rd) | Position bands (0-0.33, 0.33-0.66, 0.66-1) |
| Morphology | Lookup table | Learned from parallel structures |

## Remaining Non-Geometric Layer

The **verb conjugation output** still uses Q3 lookup:
```python
verb = self.morpho.transform(canonical, MorphoQuaternion(...))
```

This is the "thin linguistic layer" - the final step that converts 
concept space back to language. It's acceptable because:
1. It's only used for OUTPUT, not understanding
2. The understanding is fully geometric
3. Any language could have its own output layer

## Conclusion

Morphological patterns can emerge from parallel structures without 
hard-coded rules. The bootstrap provides the "combined stimulus" that 
teaches the system to recognize equivalence geometrically.

```
"Parallel structure is the teacher.
 Position is the concept.
 Phase is the tense.
 The geometry knows the rest."
```
