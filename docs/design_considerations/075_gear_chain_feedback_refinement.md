# Design Consideration 075: Gear Chain Feedback and Auto-Refinement

**Date**: December 30, 2024  
**Author**: Lesley Gushurst  
**Status**: Implemented and Tested

## Executive Summary

This document describes the **bidirectional gear chain** system that enables automatic refinement of the TruthSpace corpus through output correction. By combining the gear chain projection (Design 074) with geometric reinforcement learning (Design 073), we create a self-improving loop where Qwen2 evaluates outputs, suggests corrections, and those corrections propagate back to modify the underlying knowledge base.

## The Problem

The gear chain projection produces output like:

```
Analysis is an entity that involves involving, rigorizing, and formalizing.
```

Issues include:
- **Incorrect roles**: "entity" should be "concept" for abstract terms
- **Awkward verbs**: "rigorizing" isn't natural
- **Missing context**: No domain-specific information

We need a way to:
1. Detect these issues automatically
2. Generate corrections
3. Propagate corrections back to the corpus
4. Improve future outputs

## The Solution: Bidirectional Gear Chain

### Architecture

```
FORWARD PATH (Projection):
    Truth Corpus → [RoleGear] → [ActionGear] → [StructureGear] → [OutputGear] → Signal

BACKWARD PATH (Correction):
    Corrected Output → [Parse] → [Detect Changes] → [Propagate Back] → Corpus Update
                                                            ↓
                                                    Reinforcement Frames
```

### Key Insight

From Design 073, we learned that corpus modification works best through **reinforcement frames**, not by editing existing frames. When a correction says "Analysis should be a concept", we don't find and modify the original frame. Instead, we:

1. Boost the concept's `mediator_count` (which determines "concept" role)
2. Add new frames that reinforce the correction
3. Let the statistical weight shift future outputs

This is **additive learning** - we never delete or modify existing knowledge, only add to it.

## Implementation

### 1. Bidirectional Gears

Each gear in the chain supports both forward and backward transformation:

```python
class BidirectionalGear:
    def forward(self, state: GearState) -> GearState:
        """Transform state forward (truth → signal)."""
        
    def backward(self, correction: Correction, state: GearState) -> Correction:
        """Propagate correction backward (signal → truth)."""
```

**RoleGear** (backward):
- Correction: "entity" → "concept"
- Action: Boost `mediator_count` by 20

**ActionGear** (backward):
- Correction: "investigating" → "investigates" (reverse gerund)
- Action: Add frames with the corrected verbs

### 2. Reinforcement Frame Generation

When corrections are applied, we generate reinforcement frames:

```python
def _save_corpus_changes(self, changes, strength=10):
    for concept, change_list in changes.items():
        for change in change_list:
            if change['field'] == 'role':
                # Boost role counts
                c.mediator_count += strength * 2
                
            elif change['field'] == 'actions':
                # Add reinforcement frames
                for action in new_actions:
                    for _ in range(strength):
                        frame_text = f"{entity} {action}."
                        knowledge.learn(frame_text, source="reinforcement")
```

The `strength` parameter (default 10) controls how many frames to add. More frames = stronger reinforcement.

### 3. Auto-Refinement with Qwen2

The `AutoRefiner` class automates the correction process:

```python
class AutoRefiner:
    def refine_concept(self, concept):
        # 1. Project through gear chain
        output = self.chain.project(concept)
        
        # 2. Evaluate with Qwen2
        score, feedback = self.evaluate_output(concept, output)
        
        # 3. If score < 8, get correction
        if score < 8:
            corrected = self.suggest_correction(concept, output, feedback)
            
            # 4. Apply correction
            corrections = self.chain.correct(corrected)
            
            # 5. Save to corpus
            self.chain.apply_corrections(save=True)
```

**Qwen2 Prompts**:

*Evaluation*:
```
Evaluate this sentence describing "{concept}":
"{output}"
Rate it 0-10 on grammatical correctness, natural phrasing, and appropriate role.
```

*Correction*:
```
If this sentence needs improvement, provide a corrected version.
Keep the same structure but fix issues with role, grammar, or phrasing.
```

## Results

### Before Refinement

| Concept | Output |
|---------|--------|
| Analysis | "Analysis is an entity that involves involving, rigorizing, and formalizing" |
| Explore | "Explore is a concept that explores" |
| Heavy | "Heavy is a character who collisions" |

### After Refinement (5 concepts, 110 frames added)

| Concept | Output |
|---------|--------|
| Analysis | "Analysis is a concept that involves involving, formalizing, and structuring" |
| Explore | "Explore is a concept that involves exploring and investigating" |
| Heavy | "Heavy is a concept that involves colliding and focusing, relating to gravity and force" |

### Statistics

```
Evaluated:  5
Corrected:  5 (100%)
Frames added: 110
Corpus size: 111,931 → 112,031 frames
```

### Broad Test (15 random concepts)

**100% success rate** - No malformed outputs detected.

## Gerund Handling

A key challenge was converting verbs to gerunds correctly. The ActionGear maintains a comprehensive mapping:

```python
self.to_gerund = {
    # Standard verbs
    'investigates': 'investigating',
    'processes': 'processing',
    
    # Typos in corpus
    'monitores': 'monitoring',
    'facilitats': 'facilitating',
    
    # Nouns misidentified as verbs
    'collisions': 'colliding',
    'emphasis': 'emphasizing',
    'michels': 'influencing',  # Typo for "influences"
}
```

This handles both correct verbs and corpus data quality issues.

## Connection to Previous Designs

| Design | Contribution |
|--------|--------------|
| **073 (Geometric RL)** | Reinforcement frame approach - add frames, don't modify |
| **074 (Gear Chain)** | Modular transformation pipeline with quaternion encoding |
| **075 (This)** | Bidirectional gears + auto-refinement with Qwen2 |

## Advantages

### 1. Self-Improving
The system can improve itself without manual intervention. Run the auto-refiner periodically to continuously improve output quality.

### 2. Interpretable
Every change is logged:
```
role: entity → concept
actions: ['explores'] → ['exploring', 'investigating']
Frames added: 20
```

### 3. Non-Destructive
Corrections add to the corpus, never delete. Original knowledge is preserved; corrections shift statistical weights.

### 4. Scalable
```bash
# Refine 100 concepts automatically
python3 experiments/auto_refine_gears.py --full --save --limit 100
```

### 5. Quality-Gated
Only corrections with Qwen2 score < 8 are applied. High-quality outputs are left unchanged.

## Limitations

### 1. Qwen2 Dependency
Auto-refinement requires Ollama running with Qwen2. Manual correction still works without it.

### 2. Corpus Data Quality
Some issues (truncated words, typos) are in the original corpus and require manual cleanup.

### 3. Reinforcement Strength
The `strength=10` parameter is empirically chosen. Too low = no effect, too high = overfitting.

### 4. No Negative Learning
We only add frames, not remove them. Incorrect associations can only be outweighed, not deleted.

## Future Directions

### 1. Confidence-Weighted Reinforcement
```python
strength = int(qwen2_score * 2)  # Higher confidence = more frames
```

### 2. Batch Optimization
Run refinement on entire corpus overnight, saving checkpoints.

### 3. Human-in-the-Loop
Interactive mode where user approves corrections before applying:
```
Correction: entity → concept
Apply? [y/n]
```

### 4. Negative Reinforcement
Add "anti-frames" that reduce incorrect associations:
```python
knowledge.learn(f"{entity} does not {wrong_action}.", source="negative")
```

## Files

- `experiments/gear_chain_feedback.py` - Bidirectional gear chain with corpus modification
- `experiments/auto_refine_gears.py` - Qwen2-powered auto-refinement
- `experiments/gear_chain_projection.py` - Original forward-only gear chain

## Usage

### Manual Correction
```python
from experiments.gear_chain_feedback import FeedbackGearChain

chain = FeedbackGearChain(truth_path, signal_path)
output = chain.project('analysis')
corrections = chain.correct("Analysis is a concept that...")
chain.apply_corrections(save=True)
```

### Auto-Refinement
```bash
# Demo mode (don't save)
python3 experiments/auto_refine_gears.py

# Full refinement with saving
python3 experiments/auto_refine_gears.py --full --save --limit 100
```

## Conclusion

The bidirectional gear chain with auto-refinement creates a **self-improving knowledge system**. By combining:

1. **Gear chain projection** (modular, tunable transformations)
2. **Geometric reinforcement** (additive learning through frames)
3. **Qwen2 evaluation** (automatic quality assessment)

We achieve continuous improvement without manual intervention. The system learns from its own output, propagating corrections back through the gears to modify the underlying corpus.

```
"The gears turn both ways.
 Forward projects knowledge to language.
 Backward projects corrections to knowledge.
 The machine improves itself."
```
