# Model Reverse Engineering with φ-Basis

This directory contains experiments in reverse engineering AI models to understand their geometric structure and map them to φ-basis representations.

## The Hypothesis

**LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

## Methodology

### Phase 1: Architecture Analysis
1. Load the model and enumerate all layers/modules
2. Document shapes, types, and connections
3. Identify the computational graph

### Phase 2: Weight Geometry Analysis
1. Extract weight matrices from each layer
2. Compute SVD to find principal directions
3. Look for φ-patterns in eigenvalues and singular values
4. Identify self-similar structures across layers

### Phase 3: Attention Pattern Analysis
1. Extract Q, K, V projection matrices
2. Compute MESH = W_q.T @ W_k (pre-computed attention)
3. Analyze angles between Q and K spaces
4. Look for φ-expressible rotations

### Phase 4: φ-Basis Mapping
1. Find the minimal set of φ-angles that span the attention space
2. Create lookup tables for residuals
3. Implement AIG-optimized integer arithmetic version

### Phase 5: Validation
1. Compare outputs between original and φ-basis version
2. Measure correlation, error distribution
3. Profile performance gains

## Completed Models

### Depth Anything V2 (DA2)
- **Status**: ✅ Complete
- **Key Findings**:
  - Transformer attention uses only 17 unique φ-angles
  - 100% mesh reconstruction with 1.1KB error LUT
  - AIG integer decoder achieves ~150 FPS
- **Files**: `experiments/vr_video_converter/phi_depth_estimation.py`

## In Progress

### Qwen2.0
- **Status**: 🔄 Starting
- **Goal**: Map language model to φ-basis structure
- **Files**: `qwen2_analysis.py`, `QWEN2_ARCHITECTURE.md`

## Key Insights from DA2

1. **Structure IS information** - The 17 φ-angles encode all attention patterns
2. **Geometry IS computation** - Traversal through φ-space produces outputs
3. **The shape IS the knowledge** - What DA2 "knows" about depth is in its geometric structure

## References

- `docs/design_considerations/124_phi_transformer_replacement.md`
- `docs/design_considerations/122_self_assembly_on_weights.md`
