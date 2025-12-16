# Experiments to Test the Vacuum Forming Hypothesis

## Overview

These experiments aim to validate or refute the hypothesis that:
1. LLMs learn the "surface" of a semantic structure through training
2. There exists an "interior" structure that generates this surface
3. Phase shifts can help us probe and infer this interior structure
4. Our φ-based geometry approximates (or is) this interior structure

---

## Experiment 1: Phase-Shift Consistency

**Hypothesis**: If the 12D clock captures fundamental structure, different phase shifts should reveal consistent relationships.

**Method**:
```python
for concept_pair in semantic_pairs:
    similarities_by_phase = []
    for phase in range(1, 1000):
        v1 = encode_with_phase(concept1, phase)
        v2 = encode_with_phase(concept2, phase)
        similarities_by_phase.append(cosine_sim(v1, v2))
    
    # Check: do similar concepts stay similar across phases?
    variance = np.var(similarities_by_phase)
```

**Expected Result**: 
- Related concepts should have LOW variance (consistently similar)
- Unrelated concepts should have HIGH variance (similarity fluctuates)

**What It Tests**: Whether phase shifts preserve semantic structure or scramble it.

---

## Experiment 2: LLM Embedding Alignment

**Hypothesis**: Our φ-positions should correlate with LLM embedding positions.

**Method**:
```python
# Get LLM embeddings for concepts
llm_embeddings = get_llm_embeddings(concepts)

# Get our φ-positions for same concepts  
phi_positions = get_phi_positions(concepts)

# Compute pairwise distances in both spaces
llm_distances = pairwise_distances(llm_embeddings)
phi_distances = pairwise_distances(phi_positions)

# Correlation between distance matrices
correlation = spearman_correlation(llm_distances, phi_distances)
```

**Expected Result**:
- Positive correlation suggests our geometry captures similar structure
- No correlation suggests our geometry is arbitrary
- Negative correlation suggests we're doing something wrong

**What It Tests**: Whether intentional φ-geometry aligns with emergent LLM geometry.

---

## Experiment 3: Phase-Shift Probing of LLM

**Hypothesis**: Different phase shifts reveal different aspects of LLM's learned structure.

**Method**:
```python
prompt = "The relationship between {concept1} and {concept2} is"

for phase in clock_phases:
    # Modulate the prompt embedding by phase
    modulated_prompt = apply_phase_shift(prompt, phase)
    
    # Get LLM completion
    completion = llm.complete(modulated_prompt)
    
    # Analyze: do different phases elicit different relationship types?
    relationship_type = classify_relationship(completion)
```

**Expected Result**:
- Different phases should emphasize different relationship types
- Golden ratio phases → hierarchical relationships
- Silver ratio phases → lateral/sibling relationships
- This would suggest phase structure maps to semantic structure

**What It Tests**: Whether phase shifts can "steer" LLM attention to different relationship types.

---

## Experiment 4: Surface Reconstruction

**Hypothesis**: Multiple phase-shifted observations can reconstruct the "interior" structure.

**Method**:
```python
# Collect observations from multiple phases
observations = []
for phase in range(12):  # One per clock dimension
    phase_vector = clock.get_12d_phase(phase * 100)
    
    # Query LLM with phase-modulated prompts
    for concept in concepts:
        response = probe_llm(concept, phase_vector)
        observations.append((concept, phase, response))

# Use tomographic reconstruction
# (multiple angles → interior structure)
interior_model = reconstruct_from_observations(observations)

# Compare to our φ-geometry
alignment = compare_geometries(interior_model, phi_geometry)
```

**Expected Result**:
- If reconstruction aligns with φ-geometry, our hypothesis is supported
- If reconstruction reveals different structure, we learn what the "true" interior looks like

**What It Tests**: Whether we can infer interior structure from surface observations.

---

## Experiment 5: Prediction Test

**Hypothesis**: If we understand the interior, we can predict LLM behavior on novel concepts.

**Method**:
```python
# Train: learn mapping from φ-position to LLM behavior
training_concepts = get_known_concepts()
for concept in training_concepts:
    phi_pos = get_phi_position(concept)
    llm_behavior = probe_llm_behavior(concept)
    model.fit(phi_pos, llm_behavior)

# Test: predict LLM behavior for novel concepts
test_concepts = get_novel_concepts()
for concept in test_concepts:
    phi_pos = get_phi_position(concept)
    predicted_behavior = model.predict(phi_pos)
    actual_behavior = probe_llm_behavior(concept)
    
    accuracy = compare(predicted, actual)
```

**Expected Result**:
- High accuracy suggests φ-geometry captures generative structure
- Low accuracy suggests our geometry is surface-level only

**What It Tests**: Whether our interior model has predictive power.

---

## Experiment 6: Autotuner Validation

**Hypothesis**: The autotuner's suggestions improve alignment with LLM behavior.

**Method**:
```python
# Before autotuning
initial_entry = create_entry(keywords_v1)
initial_alignment = measure_llm_alignment(initial_entry)

# Run autotuner
result = autotuner.analyze_entry(initial_entry, test_cases)
tuned_entry = apply_suggestions(initial_entry, result.suggestions)

# After autotuning
tuned_alignment = measure_llm_alignment(tuned_entry)

improvement = tuned_alignment - initial_alignment
```

**Expected Result**:
- Autotuner suggestions should improve LLM alignment
- This validates that our tuning moves us toward the "true" surface

**What It Tests**: Whether our optimization process converges toward LLM's learned structure.

---

## Implementation Priority

### Phase 1 (Immediate)
1. **Experiment 6**: Autotuner validation - we have the tools now
2. **Experiment 1**: Phase-shift consistency - uses only our clock

### Phase 2 (Requires LLM API)
3. **Experiment 2**: LLM embedding alignment
4. **Experiment 3**: Phase-shift probing

### Phase 3 (Research)
5. **Experiment 4**: Surface reconstruction
6. **Experiment 5**: Prediction test

---

## Success Criteria

The vacuum forming hypothesis is **supported** if:
- Phase shifts reveal consistent structure (Exp 1)
- φ-geometry correlates with LLM geometry (Exp 2)
- Phase shifts can steer LLM behavior (Exp 3)
- Reconstruction aligns with φ-geometry (Exp 4)
- φ-positions predict LLM behavior (Exp 5)

The hypothesis is **refuted** if:
- Phase shifts scramble structure randomly
- No correlation between φ and LLM geometry
- Phase shifts have no effect on LLM
- Reconstruction reveals fundamentally different structure

Either outcome is valuable - we either validate our approach or learn what the true structure is.

---

## Data Requirements

### For Phase 1
- Our existing TruthSpace knowledge base
- Test cases for autotuner

### For Phase 2+
- Access to LLM embedding API (OpenAI, local model, etc.)
- Corpus of concept pairs with known relationships
- Ability to probe LLM with modified prompts

---

## Next Steps

1. Implement Experiment 1 (phase-shift consistency) using clock.py
2. Run autotuner validation (Experiment 6) on existing knowledge
3. Design LLM probing infrastructure for Phase 2 experiments
