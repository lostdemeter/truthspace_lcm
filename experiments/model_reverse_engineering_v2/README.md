# Model Reverse Engineering v2: Spectrometer Approach

## Date: February 11, 2026

## Premise

v1 asked: **"What are the weights made of?"** → φ^level × sign  
v2 asks: **"What transformations do the layers perform?"** → PhaseDiscovery archetypes

## Key Insight

The PhaseDiscovery engine ("geometric spectrometer") identifies transformation
structure from (input, output) pairs. Instead of encoding individual weight matrices,
we treat each layer of Qwen2-7B as a transformation and classify its archetype.

## Experiments

### Experiment 1: Layer Archetype Identification
- Capture hidden states at each layer boundary
- Quantize to φ-lattice levels (discrete tokens)
- Run PhaseDiscovery on (layer_N, layer_N+1) pairs
- Classify: map, collapse, expand, context, geometric_context?

### Experiment 2: Attention Head Classification  
- Extract per-head transformations
- Each head is a separate (input → output) mapping
- What archetype is each head?

### Experiment 3: Meta-Pipeline Discovery
- Compose layer archetypes into a cascade
- Does PhaseDiscovery on the full 28-layer pipeline find structure?
- Can we identify the DRUM/COMB/MUSIC boundary automatically?

## What We Expect

From v1 findings:
- Layers 0-2 (DRUM): Semantic — should show different archetype than layers 3-24
- Layer 3: Phase transition — should be a distinct archetype
- Layers 3-24 (COMB): Linear transcoder — might all be same archetype
- The scaffolding/content split might map to different phase types

## Connection to v1

v1 files: `../model_reverse_engineering/`  
v1 key findings: `../model_reverse_engineering/QWEN2_ARCHITECTURE.md`

## Model

Using Qwen2-7B (cached locally at `~/.cache/huggingface/hub/models--Qwen--Qwen2-7B`)
