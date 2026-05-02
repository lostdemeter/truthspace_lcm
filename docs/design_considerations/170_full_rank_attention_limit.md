# 170: The Full-Rank Attention Limit

## The Discovery

Single-token attention is **inherently full-rank**. The V→O transformation cannot be efficiently compressed because it's doing a full linear transformation.

This finding emerged from testing the geometric comb (Doc 112's Music Box Principle) on Qwen2-7B-Instruct.

## The Experiment

We attempted to approximate single-token attention using various low-rank decompositions:

| Approach | Correlation | Why It Failed |
|----------|-------------|---------------|
| MESH SVD (Q.T @ K) | 0.01 | Wrong matrix - captures attention scores, not output |
| V→O SVD (k=106) | 0.70 | V→O has rank 512, needs all dims |
| V→O SVD (k=256) | 0.80 | Still missing 35% of variance |
| V→O SVD (k=512) | 1.00 | Full rank required |

## Why Single-Token Is Different

For **multi-token** attention:
```
output = softmax(Q @ K.T / sqrt(d)) @ V @ O
```
The softmax creates sparsity - attention focuses on specific tokens. The Q-K interaction (captured by MESH) determines WHERE to attend.

For **single-token** attention:
```
output = softmax([score]) @ [V] @ O = 1 * V @ O = V @ O
```
The softmax is always 1 (only one token to attend to). The Q-K interaction is **irrelevant**. Only the V→O linear transformation matters.

## The V→O Path

```
V projection: (512, 3584) - projects to 4 KV heads × 128 dims
O projection: (3584, 3584) - projects back from 28 Q heads × 128 dims
Combined A = O @ expand(V): rank 512, no low-rank structure
```

The V→O matrix has **rank 512** (limited by V's output dimension). This is the fundamental bottleneck - you cannot compress a rank-512 linear transformation below 512 dimensions without losing information.

## Implications for Geometric LCM

### What This Means

1. **Single-token generation requires full computation** - no geometric shortcuts
2. **The model's organization determines compressibility** - Qwen2 wasn't trained for low-rank V→O
3. **Multi-token attention may still be compressible** - Q-K scores create sparsity

### What This Does NOT Mean

1. ~~LLMs aren't geometric~~ - The geometry is still there, just not low-rank
2. ~~Spatial computing doesn't work~~ - It works, but requires full-rank operations
3. ~~φ-encoding is useless~~ - Still valuable for storage compression

## The Deeper Question

> "The goal is to prove/disprove that LLMs are 'hyperdimensional transcoders' where the geometry IS the computation."

This finding doesn't disprove the hypothesis. It reveals a **constraint**: the geometry of attention is full-rank for single tokens.

The question becomes: **Is this fundamental, or is it an artifact of how the model was trained?**

### Hypothesis: Training Creates the Rank

A model trained with explicit low-rank constraints might:
- Learn to use fewer effective dimensions in V→O
- Develop sparse attention patterns even for single tokens
- Organize its geometry to be more compressible

Current LLMs are trained to maximize next-token prediction, not geometric efficiency. The full-rank V→O might be **learned behavior**, not fundamental necessity.

## Connection to Doc 132 (φ-Sigmoid)

Doc 132 claimed "100% of gate outputs are in |x| < log(φ)" based on Depth Anything v2 analysis.

**This claim is WRONG for Qwen2-7B early layers:**

| Layer | Gate Range | % in Linear Regime | MLP Correlation |
|-------|------------|-------------------|-----------------|
| 0 | [-5.5, 3.6] | 62-76% | 0.91-0.96 |
| 1 | [-12, 5.7] | **0.1-0.2%** | **0.32-0.39** |
| 2 | [-23, 5.3] | **0.6-0.7%** | **0.14-0.48** |
| 3+ | varies | varies | 0.75-0.99 |

**Key insight**: Different models have different internal dynamics. What works for Depth Anything v2 may not work for Qwen2-7B. The bilinear MLP approximation fails catastrophically in Qwen2's early layers.

## The Path Forward

### Option 1: Accept Full-Rank for Single-Token
Use exact computation for single-token, geometric approximation for multi-token context.

### Option 2: Train for Geometric Efficiency
Design a model architecture and training objective that encourages:
- Low-rank V→O transformations
- Sparse attention patterns
- Linear-regime MLP activations

### Option 3: Hybrid Approach
- φ-encoding for weight storage (compression)
- Exact computation for inference (accuracy)
- Geometric analysis for interpretability (understanding)

## Conclusion

Single-token attention is full-rank in Qwen2-7B. This is a **constraint**, not a refutation of geometric principles. The geometry is there - it's just not low-rank.

The question for future work: Can we design models where the geometry IS low-rank? Or is full-rank attention fundamental to language modeling?

---

## Related Documents

- Doc 112: Music Box Principle (Drum + Comb = Music)
- Doc 132: φ-Sigmoid Discovery (bilinear MLP hypothesis)
- Doc 135: Attention Head Semantic Specialization
- Doc 137: Integer φ-Encoding

## Experimental Code

See `src/phi_navigator/geometric_comb.py` for the exact reproduction and approximation tests.
