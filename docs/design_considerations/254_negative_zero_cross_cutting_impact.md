# Design Consideration 254: Negative Zero — Cross-Cutting Impact

**Date:** February 19, 2026
**Status:** Analysis complete, implementation pending
**Prerequisites:** Doc 253 (Negative Zero as 4th Dimension), Doc 247 (Geometric φ-Map), Doc 245 (Holographic Gate Field)
**Findings:** 57 (MLP ternary decomposition), 59 (3-tier gate structure)

---

## 1. Summary

Design Consideration 253 established that the SiLU/GELU gate is a **4-state holographic
encoder** where negative zero (PRESERVE-) carries essential information. This document
traces the impact of that discovery through the existing codebase and identifies where
current code is blind to the 4th dimension.

**Scale of impact:** 123 occurrences of `signs[signs == 0] = 1` across 88 files.

Not all are equally important. This document categorizes them by impact level and
proposes specific fixes for the critical paths.

---

## 2. The Systemic Pattern

Every φ-encoding function in the codebase uses this pattern:

```python
signs = np.sign(tensor).astype(np.int8)
signs[signs == 0] = 1  # Zero → positive
```

This forces all zero values to positive sign, destroying the distinction between
approaching zero from above (+0) and approaching zero from below (-0).

The φ-encoding format `value = sign × φ^(exp/K)` with `sign ∈ {-1, +1}` CAN represent
negative zero as `(sign=-1, exp=very_negative)`. But the encoding functions destroy
this by forcing `np.sign(0.0) == 0 → +1`.

---

## 3. Impact Levels

### Level 1: CRITICAL — Sign Navigation Weighting

**Files affected:**
- `src/phi_navigator/sign_only_server.py`
- `src/phi_navigator/unnamed_concept_space.py`
- `src/phi_navigator/navigation_chat_server.py`
- `src/phi_navigator/phi_compressed_navigation.py`
- `src/phi_navigator/geodesic_navigator.py`

**The problem:** All sign-based navigation uses unweighted Hamming distance:

```python
agreement = (cand_signs.float() == target_signs).float().sum()
```

Every dimension contributes equally to similarity. But Finding 57 proved that
**sign near zero carries 4× more information** than sign far from zero.

Dimensions near zero are the fringe boundaries of the holographic interference
pattern — they encode fine detail. Deep positive/negative dimensions encode only
coarse "bright/dark" structure.

**The fix:** Weight sign comparisons by inverse magnitude (φ-level proximity to zero):

```python
# Current: unweighted
agreement = (signs_a == signs_b).float().sum()

# Proposed: level-weighted
# Near-zero dims (|level| small) get higher weight
weights = 1.0 / (1.0 + torch.abs(levels.float()))
agreement = ((signs_a == signs_b).float() * weights).sum()
```

The levels are already computed and stored (`self.all_levels`) but never used for
weighting. The information exists and is discarded.

**Expected effect:** Navigation should become more sensitive to semantic distinctions
that live near hyperplane boundaries — exactly where meaning changes.

### Level 2: HIGH — Static Gate Direction in Geodesic Navigator

**File:** `src/phi_navigator/geodesic_navigator.py`

**The problem:** The geodesic navigator summarizes each layer's gate as a static
direction by averaging W_gate across output dimensions:

```python
gate_direction = np.mean(gate_weight, axis=0)
gate_signs = np.sign(gate_direction).astype(np.int8)
```

Finding 59 showed the gate has 3-tier structure:
- **Tier 1 (bias):** Per-channel majority sign, 73-93% of gate sign prediction
- **Tier 2 (scaffold):** Low-rank input-dependent signal, +3 to +7 points in layers 18-22
- **Tier 3 (content):** Full-rank, requires complete matmul

The static average captures only Tier 1. The input-dependent scaffold and content
tiers — where all the negative zero information lives — are invisible.

**Implication:** The geodesic navigator's per-layer gate representation is a lossy
projection that discards the most informative component. Any navigation through
"gate space" is navigating the scaffold only, not the content.

### Level 3: MEDIUM — SiLU LUT Sign Handling

**File:** `phi_geometric/inference/phi_integer.py`

**The problem:** The SiLU LUT forces positive sign for near-zero outputs:

```python
if abs(y) < 1e-20:
    self.out_signs[s_idx, e_idx] = 1  # Forces positive
    self.out_exps[s_idx, e_idx] = np.int16(EXP_MIN)
```

The 1e-20 threshold is so extreme that practical PRESERVE-region values are unaffected
(SiLU(-0.3) ≈ -0.128, SiLU(-0.001) ≈ -0.0005, both well above threshold).

However, the LUT only knows 2 sign states {-1, +1}. The actual gate has 4 information
states: EXPAND (+1), PRESERVE+ (+0), PRESERVE- (-0), CONTRACT (-1).

**The fix:** Extend the LUT to track the 4-state gate code alongside the standard
sign+exponent output. This preserves backward compatibility while adding the
4th dimension as metadata:

```python
# New: 4-state gate code
self.gate_code = np.zeros((2, EXP_RANGE), dtype=np.int8)
# 0=CONTRACT, 1=PRESERVE-, 2=PRESERVE+, 3=EXPAND
```

This enables downstream operations to distinguish between "barely negative gate"
and "deeply negative gate" — the key insight from Finding 57.

### Level 4: LOW — The Encoding Pattern Itself

**All 88 files with `signs[signs == 0] = 1`**

For **weights:** Exact floating-point zeros are extremely rare in trained models.
The forced positive sign is a non-issue. No fix needed.

For **activations after matmul:** When integer accumulation sums to exactly 0
(int64 cancellation), the direction of approach is lost. This is rare with 3584
terms but possible. The fix is to track the sign of the largest cancelled term,
but the cost-benefit is marginal.

**Recommendation:** Leave the weight encoding unchanged. For the critical activation
paths (phi_accumulate, phi_matmul_integer), consider tracking the pre-cancellation
sign for diagnostic purposes.

---

## 4. The Colorizer Parallel — DISPROVEN

Phase 17C found "dead" channels (pre_gelu < 0) contribute 31.6% of output energy
in the DDColor/ConvNeXt colorizer. This LOOKED like the same phenomenon as
Finding 57's negative-zero insight.

**But the 4-state decomposition reveals a fundamentally different structure:**

```
explore_colorizer_4state.py results (10 images, 18 ConvNeXt blocks):

State        Fraction   Energy%   Info Density
CONTRACT       80.2%     29.3%     0.362
PRESERVE-       9.0%      1.3%     0.225
PRESERVE+       5.8%      1.6%     0.460
EXPAND          5.0%     67.9%    30.270

PRESERVE / BOUNDARY info density ratio: 0.02×
```

**The colorizer is bimodal, NOT quadrimodal:**
- **80% of channels are CONTRACT** (deeply negative, mostly OFF)
- **5% are EXPAND** (fully ON) carrying **68% of energy**
- PRESERVE region (near zero) carries only **2.9% of energy**
- Information density ratio is **0.02×** (PRESERVE is LESS informative, not more)

**Why the difference:**
- **Transformer SiLU**: Gate-up architecture with balanced gate, ~60% PRESERVE,
  gate explicitly modulates a parallel pathway → fringe boundaries matter
- **Colorizer GELU**: Sparse activation in 4× expanded channels, ~80% CONTRACT,
  acts as a selective switch → only the spikes matter, not the fringes

**What Phase 17C actually found:** The 31.6% dead energy comes from the massive
number of CONTRACT channels (80%) each contributing a tiny amount, NOT from
information-rich PRESERVE channels near zero.

**Conclusion:** The negative zero / 4-state insight is specific to **gated MLP
architectures** (SiLU in gate-up, gate-down), NOT universal to all GELU activations.
The critical factor is whether the gate creates a balanced fringe pattern (transformer)
or a sparse activation pattern (ConvNeXt).

---

## 5. The Deeper Principle

### 5.1 Why Near-Zero Dimensions Are Special

In the holographic model (Doc 245), the gate field creates an interference pattern:
- **Bright fringes** (large positive): Dominant signal, low information per bit
- **Dark fringes** (large negative): Suppressed signal, low information per bit
- **Fringe boundaries** (near zero): Maximum gradient, maximum information per bit

This is exactly Young's double-slit experiment: the interesting physics happens at
the edges between light and dark, not in the bright or dark regions themselves.

A sign flip at |x| = 0.01 means the holographic pattern shifted by half a fringe
width. A sign flip at |x| = 5.0 means... essentially nothing changed structurally.

### 5.2 Connection to Scaffold/Content Decomposition

The scaffold/content decomposition (Doc 247, Finding 59) maps directly:
- **Scaffold** = the fringe pattern (which dimensions are near zero)
- **Content** = the sign choices at each fringe (±0 at each boundary)

The scaffold tells you WHERE the boundaries are. The content tells you WHICH SIDE
of each boundary the information falls on. Both are needed. The current sign-only
navigation has the content (signs) but not the scaffold (which signs matter).

### 5.3 Self-Similarity

The negative zero insight exhibits the same self-similarity as all φ-structure:
- At the **activation level**: SiLU creates 4 states per channel
- At the **layer level**: Some layers have more PRESERVE channels (balanced gate)
- At the **model level**: COMB→MUSIC transition (layers 18-22) has the strongest
  low-rank scaffold signal
- At the **architecture level**: Gated MLPs (SiLU gate-up) exhibit 4-state structure;
  sparse activations (GELU in ConvNeXt) do NOT — the pattern requires balanced gates

The pattern repeats at every scale within gated MLP architectures, but does NOT
transfer to sparse activation architectures. The critical factor is whether the
gate creates a balanced fringe pattern or a sparse on/off pattern.

---

## 6. Implementation Plan

### Phase 1: Sign Navigation Weighting (Level 1 fix) — DONE + VALIDATED
1. ✓ Added `_compute_level_weights()` and `_weighted_sign_agreement()` to `sign_only_server.py`
2. ✓ Wired into `navigate_holographic()`, `navigate()`, and `find_similar()`
3. ✓ Weight function: w = φ^(-|level|/K) — φ-geometric decay from zero
4. ✓ A/B test on 40 held-out pairs (80 directions): **WEIGHTED WINS**
   - MRR: 0.0547 vs 0.0358 (+53%)
   - Top-5/10 accuracy: 2.3× improvement
   - 100% higher confidence, 3.6:1 win ratio on disagreements

### Phase 2: 4-State SiLU LUT (Level 3 fix) — DONE
1. ✓ Extended `PhiSiLULUT` with `gate_codes` array (int8, 4 states)
2. ✓ Gate code boundaries at ±log(φ) on the INPUT
3. ✓ Backward compatible — `phi_silu_int()` unchanged
4. ✓ Added `phi_silu_4state()` returning (sign, exp, gate_code)
5. ✓ Fixed near-zero sign: preserves input direction instead of forcing +1
6. ✓ All 5 verification tests pass
7. ✓ Exported constants: GATE_CONTRACT, GATE_PRESERVE_N, GATE_PRESERVE_P, GATE_EXPAND

### Phase 3: Colorizer Validation — DISPROVEN
1. ✓ Ran 4-state decomposition on ConvNeXt GELU (10 images, 18 blocks)
2. ✗ PRESERVE region carries only 2.9% of energy (info density ratio 0.02×)
3. ✗ Colorizer is bimodal (80% CONTRACT, 5% EXPAND carrying 68% energy)
4. Conclusion: 4-state insight is gated-MLP-specific, NOT architecture-universal

### Phase 4: Documentation and Propagation
1. ✓ Doc 254 created with full cross-cutting analysis
2. For weights: keep `signs[signs == 0] = 1` (correct behavior)
3. For activations: preserve sign from pre-encoding value direction (done in SiLU LUT)

---

## 7. Files Referenced

### Core library (Level 1-3 impact)
- `src/phi_navigator/sign_only_server.py` — Sign navigation, unweighted Hamming
- `src/phi_navigator/geodesic_navigator.py` — Static gate direction
- `phi_geometric/inference/phi_integer.py` — SiLU LUT, accumulation sign
- `phi_geometric/inference/phi_types.py` — PhiEncoded.encode()
- `src/phi_navigator/coordinates.py` — PhiCoordinates.encode()

### Colorizer parallel (4-state DISPROVEN here)
- `phi_geometric/evaluations/lattice_navigator/ssm_phase17c_negative_space.py`
- `phi_geometric/evaluations/v20_gate_discovery.py`
- `phi_geometric/evaluations/v20_4th_dimension_trace.py`
- `experiments/model_reverse_engineering_v2/explore_colorizer_4state.py` — 4-state test
- `experiments/model_reverse_engineering_v2/results/colorizer_4state.json` — results
- `experiments/model_reverse_engineering_v2/verify_4state_lut.py` — LUT verification

### Related design docs
- Doc 253: Negative Zero as the Fourth Dimension (MLP-specific)
- Doc 247: Geometric φ-Map (PRESERVE region definition)
- Doc 245: Holographic Gate Field (interference pattern model)
- Doc 243: The GELU Machine (activation analysis)
- Doc 132: φ-Sigmoid Discovery (SiLU linearization)
