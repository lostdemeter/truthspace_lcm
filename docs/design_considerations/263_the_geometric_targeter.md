# Design Consideration 263: The Geometric Targeter

**Date:** February 26, 2026
**Status:** Design + Implementation
**Prerequisites:** Doc 253 (negative zero / 4th dimension), Doc 255 (4-state gate), Doc 261 (simple machines), Doc 262 (compound machine), Finding 98 (compound machine verification)

---

## 1. The Opportunity

Finding 98 proved that the Targeter (L26-27) is **100% independent** — its
attention can be freely approximated with zero prediction loss. Combined with:

- Gate is 98% bias-predicted (Finding 59)
- Only 8.2% of channels are EXPAND at L27
- Those 8.2% carry 91.9% of output energy (Doc 253 §7)
- FFN is 4.4× stronger than attention (Finding 97)
- LN damper compresses 92% of incoming error (Finding 97)

This means the Targeter is almost entirely a **static sparse projection**. The
gate classification is known in advance (from bias), the active channels are
few, and attention is irrelevant. We can replace L26-27 with a geometric data
structure that performs the same operation without the full transformer layer
machinery.

---

## 2. What the Targeter Computes

### 2.1 The Full Computation (Current)

For each of L26 and L27, the transformer computes:

```
h_normed = LayerNorm(h)                         [damper: 92% compression]
attn_out = Attention(h_normed, KV_cache)         [lever: irrelevant]
h_mid = h + attn_out                             [spring: residual add]
h_normed2 = LayerNorm2(h_mid)                    [damper: second compression]
gate = SiLU(gate_proj(h_normed2))                [4-state classifier]
up = up_proj(h_normed2)                          [full projection: 3584→18944]
ffn_out = down_proj(gate * up)                   [full projection: 18944→3584]
h_out = h_mid + ffn_out                          [spring: residual add]
```

Cost: 2 LayerNorms + 1 Attention (28 heads) + 3 matmuls (gate, up, down) of
size 3584×18944 each.

### 2.2 What Actually Matters

At L27:
- **LayerNorm**: Essential (the damper). But it's cheap (element-wise).
- **Attention**: Irrelevant (100% accuracy when approximated).
- **gate_proj**: 82.7% of outputs are CONTRACT (below -log(φ)). The gate
  classification is 98% determined by the bias alone. Only ~2% of channels
  need input-dependent computation.
- **up_proj**: Only the 8.2% EXPAND channels contribute meaningfully.
- **down_proj**: Only the columns corresponding to EXPAND channels matter.

### 2.3 The Geometric Equivalent

```
h_normed = LayerNorm(h)                          [keep: cheap]
attn_out = BiasAwareAttn(h_normed)               [replace: precomputed tables]
h_mid = h + attn_out                             [keep: addition]
h_normed2 = LayerNorm2(h_mid)                    [keep: cheap]

# SPARSE FFN: only compute active channels
active_idx = precomputed_expand_mask             [static: from gate bias]
up_active = up_proj_sparse @ h_normed2           [sparse: 3584 → ~1553]
gate_active = gate_bias[active_idx]              [static: precomputed]
gate_vals = SiLU(gate_active + gate_proj_sparse @ h_normed2)  [sparse]
ffn_out = down_proj_sparse @ (gate_vals * up_active)  [sparse: ~1553 → 3584]

h_out = h_mid + ffn_out                          [keep: addition]
```

Savings per layer:
- Attention: from O(S²·d) to O(S·d) (bias-aware, precomputed tables)
- gate_proj: from 3584×18944 to 3584×1553 (8.2% sparse)
- up_proj: from 3584×18944 to 3584×1553 (8.2% sparse)
- down_proj: from 18944×3584 to 1553×3584 (8.2% sparse)
- Total FFN: **~12× reduction** in compute (8.2% of channels)

---

## 3. The Data Structure: φ-Filter

> **See Doc 264 (The φ-Filter) for the full standalone treatment**: in-depth
> mathematical derivation, noise amplification theorem, threshold proof from
> first principles, and lay person analogy.

### 3.1 Definition

A **φ-Filter** is a sparse geometric projection parameterized by the 4th
dimension (gate state). It selects a small subset of dimensions based on a
φ-structured threshold, transforms through those dimensions, and projects back.

```
φ-Filter(h) = h + down_sparse @ (SiLU(gate_sparse(h)) * up_sparse(h))
```

Where:
- `gate_sparse`: projects h onto ~k active dimensions (k << d_intermediate)
- `up_sparse`: projects h onto the same ~k dimensions
- `down_sparse`: projects the k-dimensional result back to d_model
- The threshold for "active" is ±log(φ), the natural boundary of the 4-state
  gate (Doc 253)

### 3.2 Interface

```python
class PhiFilter:
    """Sparse geometric projection through φ-gated active channels.
    
    The generalization of a transformer's FFN layer when the gate is
    predominantly CONTRACT. Applicable whenever >70% of gate channels
    are below -log(φ) (the CONTRACT threshold).
    
    Analogous to: Bloom filter (approximate membership) + sparse matmul.
    The gate acts as a hash function mapping input to active dimensions.
    """
    
    # --- Construction (offline, from pre-trained weights) ---
    
    active_mask: bool[d_intermediate]    # True for EXPAND channels
    gate_bias: float[n_active]           # Bias values for active channels
    gate_weight: float[n_active, d_model]  # gate_proj rows for active only
    up_weight: float[n_active, d_model]    # up_proj rows for active only
    down_weight: float[d_model, n_active]  # down_proj cols for active only
    ln_weight: float[d_model]              # LayerNorm parameters
    ln_bias: float[d_model]
    
    # --- Properties ---
    
    @property
    def n_active(self) -> int:
        """Number of active (EXPAND) channels. ~8% of d_intermediate."""
        return self.active_mask.sum()
    
    @property
    def sparsity(self) -> float:
        """Fraction of channels that are active."""
        return self.n_active / len(self.active_mask)
    
    @property
    def compression_ratio(self) -> float:
        """Compute reduction: 1/sparsity."""
        return 1.0 / self.sparsity
    
    # --- Operation (online, per token) ---
    
    def forward(self, h: float[d_model]) -> float[d_model]:
        """Apply the φ-Filter to a hidden state vector.
        
        Complexity: O(d_model × n_active) instead of O(d_model × d_intermediate)
        """
        h_normed = self.layer_norm(h)
        
        # Sparse gate: bias + input-dependent correction
        gate_vals = self.gate_bias + self.gate_weight @ h_normed
        gate_activated = silu(gate_vals)  # SiLU on n_active values only
        
        # Sparse up-projection
        up_vals = self.up_weight @ h_normed  # n_active values
        
        # Gated combination
        active_vals = gate_activated * up_vals  # n_active values
        
        # Sparse down-projection back to d_model
        ffn_out = self.down_weight @ active_vals  # d_model values
        
        return h + ffn_out  # Residual connection (spring)
```

### 3.3 Computational Complexity

| Operation | Full Layer | φ-Filter | Reduction |
|-----------|-----------|----------|-----------|
| gate_proj | d × d_int | d × n_act | ~12× |
| up_proj | d × d_int | d × n_act | ~12× |
| down_proj | d_int × d | n_act × d | ~12× |
| Attention | O(S² × d) | O(S × d) or skip | S× |
| LayerNorm | O(d) | O(d) | 1× |
| **Total** | **~114M ops** | **~9.5M ops** | **~12×** |

For Qwen2-7B: d=3584, d_int=18944, n_active≈1553 (8.2%).

### 3.4 Generalization: When Can a Layer Be a φ-Filter?

A transformer layer can be replaced by a φ-Filter when:

1. **High CONTRACT fraction**: >70% of gate channels are CONTRACT
2. **Bias-dominant gate**: gate classification is >90% determined by bias
3. **Low attention dependence**: attention approximation doesn't hurt accuracy
4. **Energy concentration**: EXPAND channels carry >80% of output energy

From our measurements:

| Layer | CONTRACT% | Bias pred% | Attn-free? | EXPAND energy% | φ-Filter? |
|-------|-----------|-----------|------------|---------------|-----------|
| L0 | 28.5% | — | No | — | **No** |
| L7 | 46.8% | — | No | — | **No** |
| L14 | 46.5% | — | No | — | **No** |
| L21 | 62.5% | — | No | — | **Borderline** |
| L26 | 67.3% | ~95% | Yes | ~80% | **Yes** |
| L27 | 82.7% | ~98% | Yes | 91.9% | **Yes** |

The Targeter layers (L26-27) clearly qualify. Some late COMB layers (L21-25)
might qualify partially. Early/mid layers do not.

---

## 4. Connection to Classical Data Structures

### 4.1 The φ-Filter as a Bloom Filter Variant

A Bloom filter tests approximate set membership using hash functions. The
φ-Filter does something analogous:

| Bloom Filter | φ-Filter |
|-------------|----------|
| Hash functions | Gate projection (gate_weight @ h) |
| Bit array | Active channel mask |
| Membership test | Gate value > log(φ) threshold |
| False positives | PRESERVE channels misclassified as EXPAND |
| No false negatives | CONTRACT channels are truly inactive |

The key difference: Bloom filters are read-only (test membership). φ-Filters
are read-write (they transform the input through the active dimensions).

### 4.2 The φ-Filter as a Sparse Lookup Table

The active channels form a **basis** for the output correction. Each active
channel is a learned direction in d_model space (a column of down_weight).
The φ-Filter:

1. **Addresses** the table: gate activation selects which entries are active
2. **Reads** the table: up_weight extracts the input's projection onto each entry
3. **Modulates** the read: gate activation scales each entry
4. **Combines** the reads: down_weight sums the scaled entries

This is a **content-addressable memory (CAM)** with φ-structured addressing.
The gate is the address decoder, the up/down projections are the memory banks,
and the SiLU is the read amplifier.

### 4.3 The φ-Filter as a Geometric Projection

In the geometric vocabulary:

- **Input**: a point in d_model-dimensional space
- **Active basis**: n_active directions in d_model space
- **Operation**: project onto active basis, scale by gate, project back
- **Output**: input + correction along active directions

This is a **rank-n_active update** to the input vector. The φ-Filter applies
a low-rank correction (rank ~1553 out of 3584) that targets the output.

The correction is not arbitrary — it is **φ-structured**:
- Only dimensions where the gate exceeds log(φ) contribute
- The gate activation is SiLU, which is x·σ(x) ≈ x·σ(φ·x) (Doc 243)
- The threshold log(φ) is the natural boundary of the 4-state classification

---

## 5. The Targeter as Two φ-Filters

The complete Geometric Targeter is:

```
GeometricTargeter = PhiFilter(L26) → PhiFilter(L27)
```

Two sequential sparse projections with a residual connection between them.
The first filter (L26) does coarse aiming, the second (L27) does precision
targeting. Together they push the hidden state from ~57° to ~68° to hit
arccos(1/φ²).

### 5.1 Full Pipeline

```
Input h (from Processor output, L25)
  │
  ├─ LayerNorm₂₆ ──────────────────────── [damper]
  ├─ BiasAwareAttn₂₆(h_normed) ─────────── [lever: cheap, irrelevant]
  ├─ h += attn_out ──────────────────────── [spring]
  ├─ LayerNorm₂₆' ─────────────────────── [damper]
  ├─ PhiFilter₂₆(h_normed) ────────────── [wedge: sparse FFN]
  ├─ h += ffn_out ──────────────────────── [spring]
  │
  ├─ LayerNorm₂₇ ──────────────────────── [damper]
  ├─ BiasAwareAttn₂₇(h_normed) ─────────── [lever: cheap, irrelevant]
  ├─ h += attn_out ──────────────────────── [spring]
  ├─ LayerNorm₂₇' ─────────────────────── [damper]
  ├─ PhiFilter₂₇(h_normed) ────────────── [wedge: sparse FFN]
  ├─ h += ffn_out ──────────────────────── [spring]
  │
  └─ Final LayerNorm → LM Head → logits
```

### 5.2 What Can Be Precomputed

- **Active masks**: Static from gate bias (offline)
- **Sparse weights**: Extract once from full weights (offline)
- **Attention tables**: Precompute bias-aware vectors per relative position (offline)
- **Gate bias activations**: SiLU(gate_bias) for the static component (offline)

The only online computation is:
1. LayerNorm (element-wise, O(d))
2. Sparse gate correction (n_active × d matmul)
3. Sparse up projection (n_active × d matmul)
4. Sparse down projection (d × n_active matmul)
5. Residual adds (O(d))

---

## 6. Experimental Program

### Phase 10r: Geometric Targeter Verification

1. Extract L26 and L27 gate biases
2. Classify channels: EXPAND where gate_bias > log(φ)
3. Build sparse weight matrices for active channels
4. Run the full model through L25, then apply GeometricTargeter
5. Compare logits and top-1 predictions to baseline
6. Measure: accuracy, cosine similarity, angle deviation

### Success Criteria

- **Top-1 accuracy ≥ 90%** (vs 100% baseline)
- **Cosine similarity of logits ≥ 0.99**
- **Compute reduction ≥ 10×** for the Targeter layers

---

## 7. Beyond the Targeter: The Simple Machine Library

If the φ-Filter works for the Targeter, it defines a pattern:

| Machine | Data Structure | Key Operation |
|---------|---------------|--------------|
| **Targeter** | φ-Filter | Sparse projection through ~8% of channels |
| **Compressor** | φ-Damper? | Layer Norm compression + residual |
| **Processor** | φ-Lens? | Balanced attention routing + equilibrium |

Each machine in the compound becomes a **data structure** with:
- A well-defined interface (input type, output type)
- Known computational complexity
- Measurable parameters (sparsity, compression ratio, etc.)
- A classical CS analogy (filter, damper, router)

The LLM is then: `φ-Damper(4 layers) → φ-Lens(22 layers) → φ-Filter(2 layers)`

Three data structures, composed sequentially. No black box.

---

## 8. Summary

| Property | Value |
|----------|-------|
| Data structure | φ-Filter (sparse geometric projection) |
| Active channels | ~8.2% (1553 / 18944) |
| Compute reduction | ~12× per layer |
| Attention | Irrelevant (bias-aware approximation, or skip) |
| Gate prediction | 98% from bias alone |
| Energy concentration | 91.9% in EXPAND channels |
| CS analogy | Content-addressable memory with φ-structured addressing |
| Geometric analogy | Low-rank correction along φ-selected basis vectors |

> **The Targeter is not a neural network layer. It is a sparse geometric
> filter that selects ~8% of learned directions, scales them by a φ-gated
> activation, and adds the result to the residual stream. This is a data
> structure, not a mystery.**
