# Planning Experiments Analysis

## Executive Summary

We ran 6 different planning configurations to understand what makes autonomous planning work. The key finding: **discrete milestone feedback produces the most consistent results**.

| Approach | Success Rate | Avg Steps | Consistency |
|----------|--------------|-----------|-------------|
| **Milestone Planner** | **100%** | **3.0** | **Perfect** |
| Reflection-Guided | 100% | 4.0 | High |
| Minimal 3-Tools | 100% | 4.3 | Medium |
| Geometric-Guided | 100% | 6.7 | Medium |
| Granular 6-Tools | 100% | 7.3 | Low |
| φ-Space Navigation | **0%** | 15+ | Failed |

## Experiment Results Summary

| Experiment | Goals Completed | Avg Steps | Avg Output | Errors | Consistency |
|------------|-----------------|-----------|------------|--------|-------------|
| **minimal_3_tools** | 3/3 (100%) | 4.3 | 2,697 chars | 0 | Medium |
| **granular_6_tools** | 3/3 (100%) | 7.3 | 1,783 chars | 2 | Low |
| **reflection_guided** | 3/3 (100%) | 4.0 | 2,543 chars | 0 | **High** |
| **geometric_guided** | 3/3 (100%) | 6.7 | 2,165 chars | 1 | Medium |

## Key Findings

### 1. All Configurations Eventually Succeeded
Every configuration completed all 3 goals. The verification check (requiring artifacts before `done`) was crucial - it caught premature completion attempts and forced the model to actually create output.

### 2. Reflection-Guided Was Most Consistent
The `reflection_guided` configuration showed the most consistent behavior:
- **Exactly 4 steps every time**: search → reflect → generate_and_save → done
- **Zero errors**
- **Good output quality** (2,543 chars average)

The `reflect` tool acted as a "checkpoint" that helped the model understand its state.

### 3. Granular Tools = More Chaos
The `granular_6_tools` configuration had:
- **Most errors** (2 premature `done` attempts)
- **Longest step counts** (up to 10 steps)
- **Smallest output** (1,783 chars average)
- **Inconsistent tool sequences**

More tools ≠ better planning. The model struggled to choose the right tool.

### 4. Geometric Guidance Increased Search
The `geometric_guided` configuration led to more search calls (3-5 per goal) before generating output. The "navigation" metaphor may have encouraged more exploration, but also caused some confusion (2 "No tool call" instances).

### 5. Minimal Tools = Efficient but Variable
The `minimal_3_tools` worked well but showed variability:
- Goal 1: 3 steps (optimal)
- Goal 2: 7 steps (5 searches before generating)
- Goal 3: 3 steps (optimal)

## Tool Sequence Patterns

### Optimal Pattern (Reflection-Guided)
```
search → reflect → generate_and_save → done
```
This pattern was 100% consistent across all goals.

### Chaotic Pattern (Granular)
```
search → think → plan → write → done  (Goal 1)
search → reflect → search×6 → write → done  (Goal 2)
search×4 → done(ERROR) → write → done  (Goal 3)
```
The model couldn't find a consistent strategy.

## Implications

### For Tool Design
1. **Fewer, more powerful tools** work better than many granular tools
2. **Mandatory reflection** creates consistency
3. **Verification gates** (like requiring artifacts) prevent shortcuts

### For Autonomous Planning
The model can follow a workflow if:
- The workflow is simple (3-4 steps)
- There's a reflection/checkpoint mechanism
- There are guardrails preventing premature completion

The model struggles when:
- It must choose between many similar tools
- It must generate content directly (vs. having a tool do it)
- There's no feedback about its current state

## Recommendations

### Best Configuration: Reflection-Guided
Use the `reflection_guided` pattern for reliable autonomous planning:
```python
tools = {
    "search": "Gather information",
    "reflect": "Analyze current state (REQUIRED after each action)",
    "generate_and_save": "Create output from knowledge",
    "done": "Complete (only after output exists)"
}
```

### Future Experiments
1. **φ-Space Navigation**: Use actual geometric distance to goal as feedback
2. **Self-Improvement**: Let the planner learn from its errors
3. **Hierarchical Planning**: Break complex goals into subgoals
4. **Memory Integration**: Use PhiMemory to remember successful patterns

## New Experiments: φ-Space Navigation and Milestone Planner

### φ-Space Navigation (FAILED)

Attempted to use embedding distance as progress feedback:
- Compute goal embedding and current state embedding
- Show distance to goal after each action
- Hypothesis: decreasing distance = progress

**Results**: 0/3 success rate
- Initial distance was very small (0.004-0.04)
- After first action, distance jumped to ~0.37 and stayed constant
- Model got stuck in infinite search loops
- Never completed any goals

**Why it failed**: The embedding space doesn't capture "progress toward completion" in a useful way. "Starting: goal" and "Completed: goal" have nearly identical embeddings.

### Milestone Planner (BEST RESULTS)

Used discrete milestones instead of continuous distance:
```
START → KNOWLEDGE_GATHERED → OUTPUT_CREATED → COMPLETE
```

**Results**: 3/3 success rate, exactly 3 steps each
```
Goal 1: search → generate_and_save → done
Goal 2: search → generate_and_save → done  
Goal 3: search → generate_and_save → done
```

**Why it worked**:
1. Clear, discrete checkpoints (not fuzzy distances)
2. Explicit "NEXT:" guidance at each milestone
3. Automatic milestone advancement based on state
4. Model always knew exactly what to do next

## Key Insight: Discrete > Continuous

The model performs better with:
- **Discrete milestones** vs continuous distance
- **Explicit next-action hints** vs open-ended choices
- **State-based feedback** vs embedding-based feedback

This aligns with how transformers work - they're good at pattern matching and following instructions, not at gradient-based navigation.

## Recommendations

### Best Configuration: Milestone Planner
```python
milestones = [START, KNOWLEDGE_GATHERED, OUTPUT_CREATED, COMPLETE]

# After each action:
1. Check which milestone we're at based on state
2. Show milestone progress (✓ achieved, → current, ○ pending)
3. Provide explicit "NEXT:" hint for current milestone
```

### Tool Design Principles
1. **Fewer tools** - 3 is optimal
2. **Smart defaults** - tools should work even with missing params
3. **State-aware feedback** - tell the model where it is
4. **Guardrails** - prevent premature completion

## Safe Dial Planner Experiments (Feb 4, 2026)

### The Hypothesis

Based on Doc 189 (Safe Dial Mechanism), we hypothesized that:
- Layers are like rotors that click into place
- Progress could be measured by "rotor alignment"
- Geometric feedback would guide planning decisions

### What We Tried

1. **v1**: Single-layer distance at layer 27
   - Result: Distance stayed constant (~0.37)
   - Problem: "Starting: goal" ≈ "Completed: goal" in embedding space

2. **v2**: Multi-layer trajectory with rotor alignment
   - Measured: Click φ (L3), Bottleneck φ (L27), trajectory smoothness
   - Result: Alignment increased (0.14 → 0.57) but model got stuck

3. **v3**: State-based rotor alignment
   - Rotors: knowledge gathered, output created, trajectory quality
   - Result: 0/3 success - model ignored feedback

### Results

| Version | Success | Problem |
|---------|---------|---------|
| v1 (distance) | 0/3 | Distance doesn't capture progress |
| v2 (trajectory) | 2/3 | Model got stuck in loops |
| v3 (state-based) | 0/3 | Model ignored geometric feedback |

### Key Finding

**Geometric feedback doesn't help LLM planning.**

The safe dial analogy is correct for *understanding* the transformer:
- Layer 3 is the click point
- Layers 4-27 are post-click processing
- The bottleneck at layer 27 converges to φ

But it's wrong for *guiding* the model's decisions:
- Continuous metrics don't translate to discrete action choices
- The model can't interpret "alignment: 0.57 ↑" as "do X next"
- Explicit hints ("NEXT: use search") work; metrics don't

### Comparison: Milestone vs Safe Dial

| Approach | Success | Why |
|----------|---------|-----|
| **Milestone** | 3/3 | Discrete checkpoints + explicit hints |
| **Safe Dial** | 0/3 | Continuous metrics, no actionable guidance |

### Implications for TruthSpace

1. **For understanding**: φ-geometry explains transformer behavior
2. **For planning**: Use discrete milestones, not continuous metrics
3. **For self-control**: The Conceptual Nexus (Doc 206) should use milestone-based goals

### Future Directions

1. **Hybrid approach**: Use geometry to *detect* milestones, not to *guide* actions
2. **Threshold-based transitions**: "When alignment > 0.8, advance milestone"
3. **Self-improvement**: Let the planner learn which actions advance milestones

## Raw Data

See `experiment_results.json`, `phi_space_results.json`, `milestone_results.json`, and `safe_dial_results.json` for full details.
