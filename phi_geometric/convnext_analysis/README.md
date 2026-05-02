# ConvNeXt Reverse Engineering

## Goal

Reverse engineer ConvNeXt to understand if it can be represented geometrically,
enabling us to build vision encoders from first principles.

## Context

DDColor uses ConvNeXt-Tiny as its encoder. Our colorizer experiments (V9-V11)
showed that the encoder is the bottleneck - geometric features (Gabor, position,
local stats) don't capture semantic understanding.

## Questions to Answer

1. **Architecture**: What are the key components of ConvNeXt?
2. **Weights**: Do the weights follow geometric patterns (φ-Zipf, etc.)?
3. **Features**: What semantic information does each layer capture?
4. **Replacement**: Can we build an equivalent from first principles?

## Approach

1. Analyze ConvNeXt architecture (layers, operations)
2. Extract and analyze weights (SVD, distribution, patterns)
3. Probe intermediate features for semantic content
4. Attempt geometric replacement layer by layer

## Files

- `01_architecture_analysis.py` - Understand ConvNeXt structure
- `02_weight_analysis.py` - Analyze weight patterns
- `03_feature_probing.py` - What does each layer encode?
- `04_geometric_replacement.py` - Attempt geometric equivalent

## Related Work

- Doc 228: Geometric Colorizer Experiments
- Doc 229: Reverse Engineering Procedure
- Memory: φ-Zipf pattern in attention (α ≈ 1/φ)
- Memory: SVD-based LOD not viable for MLP (α ≈ 0.12)
