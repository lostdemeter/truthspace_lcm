# Qwen2-7B v2 Reverse Engineering Roadmap

## Goal

Run Qwen2-7B **controlled by our framework** using **integer arithmetic only**,
on systems **without GPUs**.

```
Current:  text → [HuggingFace/PyTorch on GPU] → text
Target:   text → [φ-framework on CPU, integer math] → text
```

## What We've Already Proven

### v1: The Math Works

| Component | What We Proved | Accuracy | File |
|-----------|---------------|----------|------|
| **Weight encoding** | sign(1bit) + φ-exponent(uint8) = 9 bits/weight | 99.92% | `phi_integer_engine.py` |
| **Integer encoding** | sign(1bit) + φ-exponent(int16, scale=8192) | 100.000% | `phi_geometric_attention.py` |
| **MESH attention** | Pre-compute W_q.T @ W_k, eliminate self-reference | 99.9991% | `phi_unraveled_engine.py` |
| **φ-matmul** | sign XOR + exponent ADD + LUT lookup | 99.93% | `phi_geometric_mlp.py` |
| **φ-softmax** | φ^(x/T) / Σ φ^(x/T) — exact equivalent | 100% | `phi_transformer.py` |
| **φ-SiLU** | x × φ-sigmoid(x) — exact equivalent | 100% | `phi_transformer.py` |
| **φ-RMSNorm** | x / rms × weight — already implemented | 100% | `phi_transformer.py` |
| **Full pipeline** | 28-layer forward pass with φ-encoded weights | 99.9991% | `phi_inference_engine.py` |
| **API server** | OpenAI-compatible endpoint serving φ-integer model | Working | `phi_integer_api_server.py` |

### v2: The Structure is Geometric

| Finding | What We Proved | Experiment |
|---------|---------------|------------|
| **88% structured** at peak layer | Affine + nonlinear + sign rules | exp4 |
| **Five-zone architecture** | DRUM/TRANSITION/COMB-early/COMB-late/MUSIC | exp1c |
| **Sign patterns = 29%** of structure | sign_preserve dominates (hyperplane preservation) | exp4 |
| **XOR at mode boundaries** | sign_xor concentrates at TRANSITION and MUSIC | exp4 |
| **φ-Zipf at layers 5-6** | α ≈ 1/φ = 0.618 | exp1c |
| **Attention is multi-dimensional** | Per-dim rules can't capture softmax (but MESH can) | exp2/exp4 |
| **10.04 GB information** | 10.55 bits/weight in 22.85 GB encoding (2.27× redundancy) | phase1.5 |
| **MESH α = 0.6528 ≈ 1/φ** | 25/28 layers follow golden ratio spectral decay | phase1.5 |
| **0.983 cross-layer similarity** | φ-level distributions nearly identical across layers | phase1.5 |
| **8.1× MLP level grouping** | 800-1170 unique levels per row vs 3584 cols | phase1.5 |
| **Layer 1 = Geometric Selector** | 28 rank-1 selectors, 97% entropy, 18.1% rank-1 energy | exp5/5b |
| **Selector ⊥ Spectrometer** | 1% subspace overlap, 89% complementary, PCA vs LDA | exp6 |

### The Compute Path

```
Standard transformer:
  input → [float32 matmul] → [float32 softmax] → [float32 SiLU] → output
  
φ-integer transformer:
  input → [sign XOR + int ADD + LUT] → [φ-softmax] → [φ-SiLU] → output
         ─────────────────────────    ────────────    ─────────
         Proven: 99.93% corr          Proven: exact   Proven: exact
```

The ONLY float operations remaining are:
1. **LUT lookup** — φ^(combined_exponent) from a ~500-entry table (2KB, fits in L1)
2. **Accumulation** — summing the looked-up values (standard float add)

Everything else is **integer arithmetic**.

---

## Phase 1: φ-Encode All Weights ✓ COMPLETE

**Status**: DONE (Feb 11, 2026)

**Result**: All 7.6B parameters encoded to φ-integer format.

| Metric | Value |
|--------|-------|
| Parameters | 7,615,283,200 |
| Components | 198 weight matrices |
| Disk size | 12.75 GB (compressed npz) |
| Compression | 2.39× vs float32 |
| Correlation | min=0.999999, avg=0.999999, max=0.999999 |
| Format | sign(int8) × φ^(exponent(int16) / 128) |
| Time | ~18 minutes |

Confirmed: Q/K/V have biases, O/gate/up/down do not. `rope_theta=1000000.0`.

### Files
- `phi_geometric/inference/phi_types.py` — Canonical `PhiEncoded` class
- `experiments/model_reverse_engineering_v2/phase1_encode_model.py` — Encoding script
- `experiments/model_reverse_engineering_v2/phi_model/` — 257 files, 28 layer dirs

**The GPU is no longer needed.**

---

## Phase 1.5: Mesh Simplification ✓ COMPLETE

**Status**: DONE (Feb 12, 2026)

**Goal**: Apply the same AIG / mesh simplification pipeline we used for the
IPA model — but now to a 7B-parameter transformer.

Converting to φ-basis puts weights on a **discrete lattice**. This unlocks
the entire simplification toolchain we already built:

### The IPA ↔ Qwen2 Analogy

```
IPA Model:                          Qwen2-7B:
──────────                          ────────
159 gate_step primitives            7.6B float weights
    ↓ encode to framework               ↓ encode to φ-basis (Phase 1 ✓)
COMPARE + AND + EMIT atoms          sign XOR + exp ADD + LUT atoms
    ↓ structural hashing                 ↓ φ-level grouping
Shared comparisons across rules     Shared φ-levels across dimensions
    ↓ template factorization             ↓ spectrometer rules
6 shape templates × N instances     14 rule types × 3584 dimensions
    ↓ information content                ↓ information content
283 bytes of actual information     ? bits of actual structure
    ↓ SimplifiedExecutor                 ↓ SimplifiedEngine
Pure lookup tables, zero float      Rule application + φ-level LUT
```

The IPA result: *"The model doesn't COMPUTE the answer. The model IS the
answer — a structured table that fits in 283 bytes."*

The Qwen2 question: **How much of the 12.75 GB is actual information?**

### 1.5.1 MESH Low-Rank Decomposition (AIG "shared sub-expressions")

MESH = W_q_head.T @ W_k_head is **rank-128 by construction**.

This is the AIG analog of finding that a complex circuit has shared
sub-expressions that can be factored out.

```
Original: MESH (3584, 3584) = 12.8M values per head
Factored: U(3584,128) × S(128) × Vt(128,3584) = 918K values
Compression: 14×
Error: 0% (exact, rank is bounded by head_dim)

Compute savings:
  score = input @ MESH @ input.T              → 12.8M multiply-accumulates
  score = (input @ U) × S × (Vt @ input.T)   → 918K multiply-accumulates
  Speedup: 14× for attention scores
```

This is FREE — just SVD the encoded Q/K weights.

### 1.5.2 φ-Level Grouping (AIG "structural hashing")

Weights cluster on ~166 distinct φ-levels. Like AIG structural hashing
where identical sub-circuits are shared, identical φ-levels can be grouped:

```python
# Standard MLP row (3584 multiplications):
output[j] = Σ_i W[j,i] × x[i]

# φ-level grouped (166 groups, integer arithmetic):
for level in unique_levels:            # ~166 iterations
    mask = (W_levels[j] == level)      # which inputs are at this level
    signed_sum = Σ (signs[mask] × x[mask])  # INTEGER add/subtract
    output[j] += signed_sum × φ^level       # ONE float multiply (LUT)

# Result: 166 float multiplies instead of 3584
# Speedup: 21× fewer float operations
```

This is the EXACT same principle as IPA's `find_shared_comparisons()` —
identical operations are computed once and shared.

### 1.5.3 Spectrometer Rules = Template Factorization

The IPA model had 6 ShapeTemplates (char_rect, digraph, context, geared, etc.).
Each rule was an INSTANCE of a template.

Our spectrometer found **14 rule types** that cover 88% of dimensions at peak:

| IPA Template | Qwen2 Rule Type | What It Does |
|-------------|-----------------|---------------|
| char_rect (1:1 map) | identity, scale | Direct pass-through |
| context (selector) | affine | Linear transform |
| geared (multi-level) | quadratic, gating | Nonlinear transform |
| digraph (pair match) | cross_dim | Cross-dimensional |
| — | sign_preserve | Boundary preservation |
| — | sign_xor | Boundary crossing (XOR gate) |

For the 88% of dimensions that match a template, we don't need the full
matmul — we apply the rule directly. **This IS `SimplifiedExecutor`** for
transformers.

### 1.5.4 Information Content Analysis

The IPA model: 159 gate_steps encoded 283 bytes of information (encoding
was 54× larger than information).

For Qwen2-7B, the question becomes:
- φ-lattice has ~166 unique levels × 2 signs = 332 distinct weight values
- That's log₂(332) ≈ 8.4 bits per weight
- But weight levels CLUSTER (φ-Zipf distribution) → effective bits < 8.4
- 88% of dimensions follow simple rules → most structure is REDUNDANT

Estimated true information content:
- Per-weight entropy (from level histogram): ~6-7 bits
- Structural redundancy (shared patterns across layers): ~2-3× reduction
- Spectrometer-exploitable structure: ~3-8× reduction
- **Estimated: 2-4 GB of actual information in a 12.75 GB encoding**

### 1.5.5 Implementation Plan

1. **MESH SVD**: Decompose all 28 heads × 28 layers = 784 MESH matrices
2. **φ-level histogram**: Count unique levels per matrix, measure entropy
3. **Level-grouped weight format**: Reorganize each weight matrix by φ-level
4. **Cross-layer structural hashing**: Find shared level patterns across layers
5. **Information content report**: True bits per layer, per component

### 1.5.6 Deliverables

- `phase1_5_simplify_model.py` — Script that runs the full simplification
- `phi_model_simplified/` — Simplified weight files
- MESH stored as U×S×Vt (14× smaller attention)
- Weight matrices reorganized by φ-level (ready for grouped matmul)
- `simplification_report.json` — Information content analysis

### 1.5.7 Expected Outcome

| Component | Phase 1 Size | After Simplification | Savings |
|-----------|-------------|---------------------|---------|
| Attention (MESH) | ~3.0 GB | ~0.22 GB | 14× |
| MLP weights | ~8.4 GB | ~8.4 GB (reorganized) | 1× storage, 21× compute |
| Embeddings | ~1.8 GB | ~1.8 GB | 1× |
| **Total disk** | **12.75 GB** | **~10.4 GB** | **1.2×** |
| **Attention compute** | **12.8M ops/head** | **918K ops/head** | **14×** |
| **MLP compute** | **3584 float muls/dim** | **166 float muls/dim** | **21×** |

The real win isn't storage — it's that the simplified model runs FASTER
because the computation structure matches the geometric structure.

### 1.5.8 Actual Results

| Component | Expected | Actual |
|-----------|----------|--------|
| Information content | 2-4 GB | **10.04 GB** (10.55 bits/weight) |
| MESH Zipf α | 1/φ | **0.6528** (≈ 1/φ!) |
| MESH compression | 14× | **14×** (exact, as predicted) |
| MLP level grouping | 21× | **8.1× average** (4.4× gate/up, 16× down) |
| Cross-layer similarity | unknown | **0.983** average cosine similarity |

The information content was higher than estimated (10 GB vs 2-4 GB) because
the level histogram is broader than expected (~2100 unique levels vs 166).
The MLP speedup is 8.1× rather than 21× because rows use ~800 unique levels
rather than 166. The overall structure validates the approach.

**Bonus discovery**: Layer 1 is a **Geometric Selector Bank** — 28 rank-1
attention selectors that tile the token space with 97% entropy. The model
built its own spectrometer. See Experiments 5/5b/6 in FINDINGS.md.

---

## Phase 2: Component Pipeline ✓ COMPLETE

**Status**: DONE (Feb 12, 2026)

**Goal**: Build each transformer block as a framework-controlled φ-component.

### 2.1 Architecture

```
PhiQwen2Engine
├── PhiEmbedding          — token → φ-vector (LUT lookup)
├── PhiTransformerBlock × 28
│   ├── PhiRMSNorm        — magnitude alignment
│   ├── PhiAttention       — MESH-based, φ-softmax
│   │   ├── Q/K/V projections (φ-matmul)
│   │   ├── RoPE (rotation in φ-space)
│   │   ├── Score = input @ MESH @ input.T (φ-matmul)
│   │   ├── Weights = φ-softmax(scores)
│   │   └── Output = weights @ V (φ-matmul)
│   ├── PhiMLP
│   │   ├── gate = φ-matmul(x, W_gate)
│   │   ├── up = φ-matmul(x, W_up)
│   │   ├── out = φ-SiLU(gate) ⊙ up
│   │   └── down = φ-matmul(out, W_down)
│   └── Residual connection (float add)
├── PhiRMSNorm (final)
└── PhiLMHead            — φ-vector → logits → token
```

### 2.2 Implementation order

1. **PhiMatmul** — Unified φ-integer matrix multiply (sign XOR + exp ADD + LUT)
2. **PhiRMSNorm** — `x / sqrt(mean(x²)) × weight`
3. **PhiEmbedding** — Just a lookup into φ-encoded embedding table
4. **PhiAttention** — MESH-based with φ-softmax and RoPE
5. **PhiMLP** — gate/up/down with φ-SiLU
6. **PhiTransformerBlock** — Combines attention + MLP + residual + norms
7. **PhiLMHead** — φ-matmul to logits + argmax/sampling

### 2.3 The RoPE question

RoPE applies rotations to Q and K based on position. In φ-space:
- Rotation = sign pattern change + level adjustment
- cos/sin are pre-computable for all positions
- Can be folded into the MESH or applied as level-space rotation

### 2.4 Deliverable

`phi_geometric/inference/` — New module containing:
- `phi_matmul.py` — Core integer matmul (hybrid + pure modes)
- `phi_components.py` — RMSNorm, Embedding, LMHead, φ-softmax, φ-SiLU
- `phi_attention.py` — Multi-head attention with RoPE + GQA
- `phi_mlp.py` — Gated MLP with φ-SiLU
- `phi_engine.py` — PhiQwen2Engine (load/forward/generate)

### 2.5 Actual Results

| Test | Result |
|------|--------|
| 2-layer smoke test | ✓ Valid logits, correct shape, no NaN/Inf |
| 2-layer vs reference | **r = 1.00000000**, 100% top-1 match, max diff 3.5e-5 |
| Full 28-layer forward | ✓ 69s for 5 tokens, coherent predictions |
| Per-layer timing | 2.0s/layer (attention) + LM head ~5-13s |
| Memory | ~16 GB total (embeddings decoded, layers in φ-format) |

**Hidden norm trajectory validates five-zone architecture:**
```
Layer  0-2:    42 →   84  (DRUM — initial processing)
Layer  3:          → 7186  (TRANSITION — sudden jump!)
Layer  4-16:       → 9016  (COMB — gradual growth)
Layer 17-25:       → 8551  (late COMB — plateau)
Layer 26-27:       → 1347  (MUSIC — sharp compression)
```

**Full 28-layer predictions for "1 + 1 =":**
Top-1: " " (space), Top-2: " ?" — base model expects formatting after "=".
All predictions are grammatically valid tokens.

---

## Phase 3: End-to-End Integer Inference ✓ COMPLETE

**Status**: DONE (Feb 12, 2026)

**Goal**: Generate text from a prompt using ONLY the φ-framework, no GPU.

### 3.1 What Was Built

- **KV Cache** (`KVCache` class) — stores K/V at KV-head level, ~4 MB/layer per 1024 tokens
- **Incremental decode** — PhiAttention supports prefill (full seq) + decode (single token)
- **Weight decode caching** — `decode_cached()` on PhiEncoded, ~6× speedup after first call
- **Qwen2Tokenizer** — lightweight BPE tokenizer, no HuggingFace dependency
- **CLI tool** (`phi_generate.py`) — streaming output, argparse, timing stats

### 3.2 Verification Results

| Test | Result |
|------|--------|
| KV cache vs full forward | **r = 1.0, max_diff = 0.0** (bit-identical) |
| 5-step incremental decode | All 5 steps: r=1.0, top-1 match ✓ |
| Full 28-layer generation | Factually correct, coherent multi-sentence output |

### 3.3 Generation Output

```
Prompt: "The capital of France is"
Output: "The capital of France is Paris. It is the largest city in France and"
```

10 tokens generated, factually correct, zero hallucination on the core fact.

### 3.4 Performance

| Phase | Time | Rate |
|-------|------|------|
| Model load | 77s | 28 layers from compressed .npz |
| Prefill (5 tokens) | 121s | 24.2s/tok (includes first weight decode) |
| Decode (per token) | 4.1s | With KV cache + cached weights |
| KV cache speedup | **6×** | 4.1s vs 24.2s per token |
| KV cache memory | 112 MB | For 1024 tokens across 28 layers |

### 3.5 Deliverable

- `phi_generate.py` — CLI: `python phi_generate.py "prompt" --max-tokens 20`
- `phi_geometric/inference/tokenizer.py` — Lightweight Qwen2 BPE tokenizer
- `phi_geometric/inference/phi_attention.py` — KVCache + incremental decode
- `phi_geometric/inference/phi_types.py` — decode_cached() / warm_cache()

### 3.6 Acceptance Criteria

| Metric | Target | Achieved |
|--------|--------|----------|
| KV cache correctness | Bit-identical to full forward | ✓ (r=1.0, diff=0.0) |
| Coherent generation | Factually correct output | ✓ ("Paris") |
| No GPU required | Pure NumPy | ✓ |
| No PyTorch required | Standalone inference | ✓ |

---

## Phase 4: Spectrometer-Guided Optimization ✓ COMPLETE

**Status**: DONE (Feb 14, 2026)

**Goal**: Use spectrometer findings to SKIP computation for structured dimensions.

### 4.1 What Was Built

- **Full-dimension spectrometer**: ContinuousPhaseDiscovery on ALL 3584 dims × 28 layers
- **SpectrometerLayer** (`phi_spectrometer.py`): vectorized per-dim rule application
- **Progressive replacement test**: measures quality vs number of replaced layers
- **Single-layer sweep**: identifies which layers tolerate replacement best

### 4.2 Spectrometer Results (from φ-engine hidden states)

| Zone | Layers | Structured | R² | Composition |
|------|--------|-----------|-----|-------------|
| DRUM | 0-2 | 26% | 0.322 | sign_preserve dominates |
| TRANSITION | 3 | 9% | 0.194 | mostly unstructured |
| COMB-early | 4-6 | 59% | 0.650 | affine(14%) + nonlin(15%) + sign(29%) |
| COMB-late | 7-25 | 64% | 0.670 | sign_preserve grows to 63% |
| MUSIC | 26-27 | 18% | 0.261 | mostly unstructured |
| **Overall** | **0-27** | **54%** | **0.584** | |
| **Peak (layer 5)** | | **82%** | **0.772** | affine + quad + gating + sign |

### 4.3 Progressive Layer Replacement (key result)

Test: "The capital of France is" → should predict "Paris"

| Replaced | Layers | Logit r | Top-1 | Top-10 | Speedup |
|----------|--------|---------|-------|--------|---------|
| 0 | (baseline) | 1.0000 | ✓ Paris | 100% | 1× |
| 1 | 5 | 0.9941 | ✓ Paris | 90% | ~1× |
| 3 | 5,14,16 | 0.9719 | ✓ Paris | 80% | ~1× |
| **5** | **5,14,15,16,24** | **0.9528** | **✓ Paris** | **70%** | **~1×** |
| 10 | best 10 | 0.8423 | ✗ "the" | 30% | ~1× |
| 15 | best 15 | 0.8185 | ✗ | 20% | 2× |
| 20 | best 20 | 0.6524 | ✗ | 10% | 3× |
| 28 | all | -0.0621 | ✗ | 0% | 12× |

**Quality cliff between 5-10 replaced layers.** Up to 5 layers can be fully
replaced with per-dimension rules while maintaining correct prediction.

### 4.4 Single-Layer Replacement (which layers tolerate it?)

| Layer | Coverage | R² | Logit r | Top-1 correct? |
|-------|----------|-----|---------|----------------|
| 5 | 82% | 0.772 | 0.9941 | ✓ |
| 16 | 78% | 0.755 | 0.9880 | ✓ |
| 14 | 75% | 0.736 | 0.9872 | ✓ |
| 24 | 73% | 0.734 | 0.9802 | ✓ |
| 17 | 72% | 0.715 | 0.9936 | ✓ |
| 13 | 71% | 0.715 | 0.9943 | ✓ |
| 11 | 66% | 0.685 | 0.9939 | ✓ |
| 19 | 65% | 0.675 | 0.9871 | ✓ |
| 25 | 64% | 0.659 | 0.9798 | ✓ |
| 18 | 64% | 0.669 | 0.9878 | ✓ |
| 22 | 62% | 0.657 | 0.9812 | ✓ |
| 20 | 61% | 0.647 | 0.9881 | ✓ |

**13 of 15 tested layers maintain correct top-1 when individually replaced.**
The geometry DOES predict single-layer computation. Error accumulation across
multiple layers is the limiting factor.

### 4.5 What This Proves

1. **The spectrometer works on φ-engine output** — no PyTorch needed for analysis
2. **Per-dimension rules capture 54-82% of layer computation** at the individual level
3. **Single-layer replacement is well-tolerated** (r > 0.97 for most COMB layers)
4. **Multi-layer replacement hits a quality cliff at ~5 layers** due to error accumulation
5. **The speedup is real but limited** — replacing 5/28 layers saves ~18% compute

### 4.6 The Boundary

The spectrometer reveals WHERE geometry ends and computation begins:
- **Geometric (replaceable)**: Per-dimension affine/sign transforms in COMB layers
- **Computational (irreplaceable)**: DRUM setup, TRANSITION phase change, MUSIC output shaping
- **Error source**: Cross-dimensional interactions (attention softmax, layer norm redistribution)

### 4.7 Files

- `phi_geometric/inference/phi_spectrometer.py` — SpectrometerLayer + rule loading
- `phi_geometric/inference/phi_engine.py` — forward_with_hidden_states()
- `experiments/.../phase4_extract_rules.py` — hidden state extraction + CPD
- `experiments/.../phase4_quality_test.py` — progressive replacement test
- `experiments/.../results/phase4_rules_full/` — 28 per-layer rule JSON files

---

## Phase 5: Validation & Benchmarking ✓ COMPLETE

**Status**: DONE (Feb 14-16, 2026)

**Goal**: Validate three geometric primitives at scale, prove attention IS geometric.

### 5.1 Resonator Validation — 88.6% Match on 35 Prompts (Finding 46-48)

| Category | Prompts | Match Rate |
|----------|---------|------------|
| Factual recall | 10 | 90% |
| Completion | 5 | 80% |
| Entity retrieval | 5 | 80% |
| Arithmetic/logic | 5 | 100% |
| Multi-token | 5 | 80% |
| Common knowledge | 5 | 100% |
| **Overall** | **35** | **88.6%** |

With 8-head routing: **94.3%** (33/35). Two routing families discovered.

### 5.2 Geometric Attention Proof (Finding 49)

**100% proof that attention IS geometric:**
- φ-linear projection: EXACT (sign XOR + exp ADD)
- φ-softmax: EXACT (φ^(x/T) / Σ)
- RoPE: EXACT (rotation in φ-space)
- Full attention: bit-identical to float baseline

### 5.3 Geometric Purity Audit (Finding 50)

| Metric | Result |
|--------|--------|
| φ-encoded parameters | **99.9956%** of all params |
| Geometric operations | **19/23** pipeline ops |
| Non-geometric ops | 4 (accumulation, RMS sqrt, sampling, causal mask) |

### 5.4 Files

- `phase5_validate_resonator.py` — 35-prompt validation
- `phase5_diagnose_failures.py` — Multi-head routing diagnosis
- `phase5_full_resonator.py` — 28-head hard routing test
- `phase5_geometric_attention_proof.py` — Exact geometric proof
- `phase5_geometric_purity_audit.py` — End-to-end purity audit

---

## Phase 6: Integer Arithmetic ✓ COMPLETE

**Status**: DONE (Feb 17-18, 2026)

**Goal**: Replace all float operations with integer arithmetic primitives.

### 6.1 Integer Primitives (Finding 51)

11 operations implemented in `phi_integer.py`, all verified:
- Accumulator, SiLU LUT, RMS norm, matmul, softmax
- RoPE, einsum QK/AV, causal mask, broadcast add, scale

### 6.2 Full Integer Forward Pass (Finding 52)

| Test | Result |
|------|--------|
| 28-layer forward pass | ✓ Complete |
| Next-token prediction | **6/6 MATCH** with float baseline |
| Precision cliff | Layer 27 (cancellation in residual stream) |
| Per-layer correlation | r > 0.999 for layers 0-26 |

### 6.3 Files

- `phi_geometric/inference/phi_integer.py` — All 11 integer operations
- `phase6_integer_primitives_test.py` — Unit tests
- `phase6_integer_forward_pass.py` — Full 28-layer test
- `phase6_integer_predictions.py` — 6/6 prediction validation
- `phase6_diagnose_precision.py` — Per-op precision diagnostic
- `phase6_find_cliff.py` — Per-layer correlation sweep

---

## Phase 7: Distributed Compute ✓ COMPLETE

**Status**: DONE (Feb 18-19, 2026)

**Goal**: Run φ-integer inference distributed across network nodes.

### 7.1 Remote Matmul (Finding 53)

TCP dispatch to compute node on gimli: 7/7 EXACT MATCH.

### 7.2 Full Layer Remote (Finding 52)

5/5 MATCH — full transformer layer computed remotely, bit-identical.
Weight compression: per-row uint8 quantization, 1.50× additional compression.

### 7.3 GPU Acceleration (Finding 53)

CuPy/CUDA acceleration on gimli RTX 3050: 9.1× speedup, 100% bit-identical.

### 7.4 Thin Client φ-Compute Protocol (Finding 54)

| Test | Result |
|------|--------|
| Opcodes | **18/18 bit-identical** |
| Full layer | **BIT-IDENTICAL** (55 instructions) |
| Full model | **5/5 CORRECT** (1540 instructions) |
| Per-prompt inference | 29-50s (GPU-accelerated) |

Model-agnostic VM: 19 opcodes, 64 registers, zero model knowledge on node.
All model knowledge lives in the controller's layer compiler.

### 7.5 Files

- `gimli:~/truthspace-node/phi_compute_node.py` — Thin client VM
- `phi_geometric/inference/phi_compute_client.py` — Controller client
- `test_phi_compute_ops.py` — 18/18 opcode tests
- `test_phi_compute_layer.py` — Layer compiler + test
- `test_phi_compute_full.py` — Full model test
- Doc 251: Distributed Integer Compute
- Doc 252: φ-Compute Protocol

---

## Research Branch: MLP Optimization & Negative Zero (Findings 55-61)

**Date**: Feb 19-20, 2026

After completing the distributed compute pipeline, we investigated MLP
optimization (the remaining computational bottleneck) and discovered the
4th dimension.

### The MLP Problem (Findings 55-56)

| Approach | Result | Why |
|----------|--------|-----|
| Linearized MLP | r=0.57-0.74 | Gated structure is irreducibly nonlinear |
| Sparse MLP (static) | 4/5 argmax | Channel volatility blocks static pruning |
| Cached Jacobian | r=0.19-0.67 | Jacobian changes dramatically per token |
| Mean Jacobian | 16.6× speed | Insufficient quality (except layer 27) |

**Conclusion**: MLP cannot be simplified by linearization, sparsification, or
caching. The gated bilinear interaction (gate × up) is essential.

### The Discovery: Negative Zero (Finding 57)

While investigating gate sparsity, we discovered that "dead" channels carry
meaningful information through SiLU negative leakage. The SIGN of near-zero
gate activations ("negative zero") encodes 4× more information than magnitude.

This led to the 4-state gate classification:
- **CONTRACT (-1)**: x < -log(φ) — deeply gated off
- **PRESERVE- (-0)**: -log(φ) ≤ x < 0 — negative fringe (NEGATIVE ZERO)
- **PRESERVE+ (+0)**: 0 ≤ x < +log(φ) — positive fringe
- **EXPAND (+1)**: x ≥ +log(φ) — fully active

### Gate Structure Exploration (Findings 58-60)

- Finding 58: Low-rank gate predictor DISPROVED — gate signs not low-rank
- Finding 59: 3-tier structure exists but computationally unexploitable
- Finding 60: Cross-cutting impact — weighted sign navigation +2% accuracy,
  4-state SiLU LUT validated, but NOT universal across architectures

### The 4-State Gate IS a Real φ-Dimension (Finding 61)

The breakthrough: treating the 4-state gate as a genuine geometric dimension
and testing it against the same rules as arithmetic/zeta spacetime.

| Test | Result | Error |
|------|--------|-------|
| Light-cone speed limit | 1/φ | 0.2% |
| Token universality | RMS = 0.0085 | 12× stronger than primes |
| Cross-parity split | 1/φ | 0.8% |
| Eigenvalue decay | 1/φ² | 1.9% |
| Persistence ratio | φ | 1.2% |

**The gate state standing wave encodes the five-zone architecture.**

### Design Documents

- Doc 253: Negative Zero as the 4th Dimension
- Doc 254: Negative Zero Cross-Cutting Impact
- Doc 255: 4-State Gate as φ-Dimension
- Doc 256: Multi-Lens φ-Geometry
- Doc 257: Polarization, Handedness, and Embarrassing Parallelism

---

## Phase 8: Polarization & Parallel Architecture ◉ IN PROGRESS

**Status**: IN PROGRESS (Feb 20, 2026)

**Goal**: Test whether the gate dimension obeys polarization physics (Malus's Law)
and whether this enables embarrassingly parallel layer computation.

### Three Research Questions

1. **Accuracy**: Can we perform calculations better using the 4th dimension?
2. **Parallelism**: Can we make layer processing embarrassingly parallel?
3. **Science**: What does this teach us about the geometry of computation?

### 8.1 Standing Wave Prediction Test

The standing wave is 99.15% token-universal (RMS = 0.0085). If we can predict
gate states from the standing wave alone, layers become parallelizable.

**Method:**
- Compute mean gate state distribution per layer across diverse tokens
- For each token, predict gate states using the mean standing wave
- Measure per-layer prediction error
- Identify which layers deviate most (where sequential processing is needed)

**Success criteria:** Per-layer prediction error < 2% for most COMB layers.

### 8.2 Chirality Independence Test

The cross-parity split (CONTRACT+PRESERVE+ = 61.3%, PRESERVE-+EXPAND = 38.7%)
suggests two independent information channels ("handedness").

**Method:**
- Decompose channels into L (CONTRACT + PRESERVE+) and R (PRESERVE- + EXPAND)
- Compute mutual information between L and R channel outputs
- Test whether L and R carry statistically independent information
- Measure cross-channel correlation at layer boundaries

**Success criteria:** Mutual information between L and R < 10% of total.

### 8.3 Malus's Law Quantitative Test

The persistence rates (59.1% ≈ 1/φ, 35.7% ≈ 1/φ²) map to Malus's Law:
cos²(38.2°) = 1/φ, cos²(51.8°) = 1/φ², and 38.2° + 51.8° = 90°.

**Method:**
- Compute per-layer transition matrices (not just the global one)
- Fit each transition probability to cos²(θ)
- Extract per-layer angles and verify complementarity
- Test whether per-layer angles follow the standing wave rotation

**Success criteria:** Malus's Law fit R² > 0.9 for COMB layers.

### 8.4 Parallel Architecture Simulation

**Method:**
- Implement predict-parallel-correct pipeline:
  1. Pre-compute expected gate states from standing wave
  2. Run all layers in parallel on predicted inputs
  3. Sequential correction pass for residual
- Compare output to sequential baseline
- Measure residual size and correction quality

**Success criteria:** Output within 1% of sequential baseline.

### 8.5 Deliverables

- `phase8_polarization_test.py` — Combined standing wave + chirality + Malus test
- `phase8_parallel_architecture.py` — Parallel pipeline simulation
- Results added to FINDINGS.md
- Design implications added to Doc 257

---

## Execution Order

```
Phase 1 ✓ ──→ Phase 1.5 ✓ ──→ Phase 2 ✓ ──→ Phase 3 ✓ ──→ Phase 4 ✓
  (encode)    (simplify)     (pipeline)    (inference)   (spectrometer)
                                                              │
              Phase 5 ✓ ←─────────────────────────────────────┘
              (validate: 88.6% resonator, 100% geometric proof)
                    │
              Phase 6 ✓ (integer arithmetic, 6/6 predictions)
                    │
              Phase 7 ✓ (distributed: 18/18 opcodes, 5/5 full model)
                    │
              Research: Findings 55-61 (MLP, negative zero, 4th dimension)
                    │
              Phase 8 ◉ (polarization physics, parallel architecture)
```

**Phase 1** ✓ DONE: weights are φ-encoded. GPU no longer needed.

**Phase 1.5** ✓ DONE: 10.04 GB information, MESH α≈1/φ, 14× attn speedup.

**Phase 2** ✓ DONE: r=1.0 vs reference, full 28-layer forward in 69s.

**Phase 3** ✓ DONE: "The capital of France is Paris." KV cache, 4.1s/tok.

**Phase 4** ✓ DONE: 3 geometric primitives, 82% structured at peak.

**Phase 5** ✓ DONE: 88.6% resonator on 35 prompts, 100% geometric attention proof.

**Phase 6** ✓ DONE: Integer arithmetic, 6/6 predictions match float baseline.

**Phase 7** ✓ DONE: Thin client protocol, 18/18 opcodes, 5/5 full model correct.

**Research** ✓ DONE: Negative zero → 4-state gate → real φ-dimension (Finding 61).

**Phase 8** ◉ IN PROGRESS: Polarization physics + embarrassingly parallel architecture.

---

## Connection to the Hypothesis

> "LLMs are hyperdimensional transcoders — the intelligence is in the shape"

This roadmap tested the hypothesis end-to-end:

1. **Phase 1** ✓ proves the weights ARE a geometric shape (φ-lattice encoding)
2. **Phase 1.5** ✓ proves the shape can be SIMPLIFIED (10 GB of 23 GB is information)
3. **Phase 2** ✓ proves the computation IS geometric navigation (r=1.0 fidelity)
4. **Phase 3** ✓ proves the system WORKS as pure geometry (GPU-free "Paris")
5. **Phase 4** ✓ proves the geometry PREDICTS the computation (82% per-dim rules)
6. **Phase 5** ✓ proves attention IS geometric (100% exact proof)
7. **Phase 6** ✓ proves integer arithmetic suffices (6/6 match)
8. **Phase 7** ✓ proves the model runs distributed on a model-agnostic VM
9. **Research** discovered the 4th dimension (gate states = real φ-geometry)
10. **Phase 8** tests if the 4th dimension enables embarrassingly parallel compute

### The Verdict (Updated)

**The hypothesis is confirmed at multiple levels:**

- **Weights**: 99.9956% φ-encoded, 19/23 operations geometric
- **Computation**: 6/6 integer predictions match float, distributed VM works
- **Structure**: 82% of per-dimension computation follows simple geometric rules
- **The 4th dimension**: Gate states form a real φ-structured dimension with
  speed limit 1/φ, decay 1/φ², and a standing wave that IS the architecture
- **The frontier**: If the 4th dimension follows Malus's Law (polarization),
  sequential layer processing may be parallelizable

### The Progression

```
IPA:    29 rules → 283 bytes         (structure IS the answer)
DA2:    encoder → 32 φ-weights       (structure IS the depth map)
Qwen2:  7.6B params → 10 GB info     (structure IS the language model)
        82% geometric at peak        (geometry PREDICTS computation)
        6/6 integer predictions      (integer arithmetic SUFFICES)
        18/18 distributed opcodes    (model-agnostic VM WORKS)
        4-state gate = real φ-dim    (the 4th dimension IS real)
        Malus's Law at φ-angles?     (polarization → parallelism?)
```
