# DC 294: Multi-Layer Phase Shift — The Controllable Funnel

## Status: CONFIRMED
## Date: 2026-03-06
## Depends on: DC 276 (Geometric Structures), DC 288 (Weight Structure), DC 292 (Binary Phase Hologram), DC 293 (Sieve Paradigm)

---

## The Question

DC 293 proved that the weight matrix is an exact sieve — ε-groups act as "prime factors" that uniquely determine the output. DC 292 showed the sign matrix is a binary phase hologram. The single-layer phase shift experiment (phi_phase_shift_funnel.py) demonstrated that ε-group knobs are linear, stable, and coherent across weight types within a layer.

**But what happens when you chain phase shifts through multiple layers with nonlinear gating?**

Single matmul is linear — perturbations scale proportionally. The SiLU gate is the nonlinearity that could either amplify or dampen differences. The residual connection could bound or compound them. Which wins?

## The Experiments

Three scripts, escalating in scope:

1. **phi_multi_layer_phase.py** — Chain 8 layers, compare full/macro/shifted
2. **phi_phase_sweep.py** — Sweep δ from 0.01 to 1.0, sweep group counts 3-30, test where to apply shift

## Finding 1: The Gate Amplifies Everything

Chaining 5 macro groups (top ε-groups only, ~87% energy) through 8 MLP layers with SiLU gating:

```
Layer 0:  22.7° from full (cos = 0.922)
Layer 1:  66.8° from full (cos = 0.392)
Layer 2:  80.1° from full (cos = 0.170)
Layer 3:  89.5° from full (cos = 0.009)  ← ORTHOGONAL
Layer 4+: ~90.0° (saturated)
```

**In 3 layers, macro becomes completely orthogonal to full.** The SiLU gate makes different kill/pass decisions for macro vs full weights — only 52.6% binary agreement at 5 groups — and these different routing decisions cascade exponentially.

The effective rank after 8 layers tells the story:
- Full: rank 4 (MLP compresses 3584D → 4D!)
- Macro: rank 17 (can't compress — wrong gate decisions lose focus)
- Shifted: rank 5 (shift preserves compression)

**The detail groups enable precision gating that collapses output to a low-dimensional manifold.** Without them, the funnel can't focus.

## Finding 2: Three Regimes of Phase Shift

Sweeping φ^δ scaling on the top ε-group through 8 layers:

| δ | φ^δ | L0→L7 angle | Amplification | Magnitude | Regime |
|---|-----|-------------|---------------|-----------|--------|
| 0.01 | 1.005× | 0.1° → 1.7° | 17× | 1.02× | CONTROLLABLE |
| 0.05 | 1.024× | 0.5° → 8.7° | 17× | 1.15× | CONTROLLABLE |
| **0.10** | **1.049×** | **1.0° → 16.9°** | **17×** | **1.39×** | **SWEET SPOT** |
| 0.20 | 1.101× | 2.1° → 26.8° | 13× | 1.89× | MODERATE |
| 0.50 | 1.272× | 5.7° → 45.6° | 8× | 3.73× | EXPLOSIVE |
| 1.00 | 1.618× | 13.3° → 59.4° | 4.5× | 11.84× | EXPLOSIVE |

### The 17× Amplification Constant

In the controllable regime (δ ≤ 0.1), **the angle amplification factor is constant at ~17×** regardless of δ. This means:

- The system is **LINEAR in δ** for small perturbations
- 8 layers of SiLU gating amplify the initial deflection by exactly 17×
- The amplification drops at larger δ because the system enters the nonlinear/saturating regime

### Sigmoid Growth Pattern

The layer-by-layer progression for δ=0.10 shows a characteristic sigmoid:

```
L0: 1.0°  (+1.0°)
L1: 1.8°  (+0.8°)  ← slow start
L2: 4.4°  (+2.6°)  ← accelerating
L3: 8.4°  (+4.0°)  ← peak growth
L4: 12.4° (+4.0°)  ← peak growth
L5: 14.9° (+2.5°)  ← decelerating
L6: 16.3° (+1.4°)  ← saturating
L7: 16.9° (+0.6°)  ← saturated
```

The residual connection bounds the divergence. The system amplifies through layers 2-4 (where the SiLU routing differences compound) but the residual stream anchors the signal, preventing unbounded divergence. This creates a **bounded, predictable knob**.

### The Sweet Spot

**δ = 0.10 (φ^0.1 = 1.049× scaling)** is the largest shift that stays controllable:
- 16.9° deflection after 8 layers (meaningful but bounded)
- 1.39× magnitude ratio (no explosion)
- Linear relationship with δ preserved

## Finding 3: Gate Agreement Has Discrete Phase Transitions

Testing how many ε-groups are needed for the SiLU gate to make the same binary decisions as the full weight matrix:

| Groups | Gate Agreement | cos_sim | Interpretation |
|--------|---------------|---------|---------------|
| 3 | 56.3% | 0.61 | Coin flip |
| 5 | 57.1% | 0.62 | Still coin flip |
| **7** | **71.9%** | **0.76** | **Jump! Coarse gear engages** |
| 10 | 71.9% | 0.76 | Plateau |
| 15 | 81.2% | 0.80 | Gradual |
| 20 | 83.0% | 0.80 | Plateau |
| **30** | **100.0%** | **1.00** | **Perfect. Fine gear completes.** |

The agreement does NOT improve gradually — it has **discrete thresholds**:

1. **5→7 groups**: 57% → 72% (coarse gear engages)
2. **15→20 groups**: gradual improvement
3. **20→30 groups**: 83% → 100% (fine gear completes)

This maps directly to the gearing mechanism from DC 293:
- **Coarse gear** (7 groups): resolves ~72% of gate decisions
- **Fine gear** (30 groups): resolves the remaining 28%
- The structure is NOT a smooth continuum — it's **quantized**

## Finding 4: Multi-Layer Divergence Requires 20+ Groups

Chaining macro weights through 4 layers:

| Groups | L0 | L1 | L2 | L3 |
|--------|-----|-----|-----|-----|
| 3 | 29.8° | 86.3° | 88.5° | 90.2° |
| 5 | 22.2° | 64.7° | 77.5° | 87.7° |
| 7 | 15.9° | 46.1° | 70.6° | 83.6° |
| 10 | 12.9° | 42.8° | 72.1° | 84.1° |
| 15 | 5.6° | 23.9° | 54.4° | 76.0° |
| **20** | **3.5°** | **9.3°** | **17.4°** | **43.6°** |
| **30** | **1.3°** | **2.4°** | **6.0°** | **22.2°** |

The critical transition is at **20 groups**. Below 15 groups, everything diverges past 76° by L3. At 20 groups, L3 is 43.6° — still divergent but manageable. At 30, it's 22.2° — similar to the full shift at δ=0.2.

This means **for multi-layer processing, you need ~30 of ~37 ε-groups** (81% of the alphabet) to maintain gate fidelity. The sieve is exact, but you can't shortcut through the MLP — the gate needs almost everything.

## Finding 5: Shift Effects Are Additive in the Controllable Regime

Testing where to apply a δ=0.1 shift through 4 layers:

| Configuration | L0 | L1 | L2 | L3 |
|--------------|-----|-----|-----|-----|
| gate only | 0.56° | 0.90° | 1.93° | 3.90° |
| up only | 0.49° | 0.81° | 1.70° | 3.51° |
| down only | 0.48° | 0.77° | 1.52° | 1.93° |
| gate+up | 0.80° | 1.39° | 3.24° | 6.92° |
| all three | 1.02° | 1.75° | 4.40° | 8.44° |

Key observations:

1. **Gate ≈ up in effect** (3.90° vs 3.51° at L3) — the gate does NOT dominate at small δ
2. **Down has half the effect** (1.93° at L3) — it only transforms, doesn't route
3. **Effects are ~90% additive**: gate+up predicted 7.41°, actual 6.92°; all predicted 9.34°, actual 8.44°
4. **In the controllable regime, perturbations add, not multiply**

The gate's special role (controlling SiLU routing) only matters at larger δ where routing decisions actually flip. At δ=0.1, the shift is too small to change binary routing — the effect is purely through magnitude modulation.

## Synthesis: The Controllable Funnel

### What We Now Know

1. **The ε-group structure provides a precision knob system**
   - Each δ=0.01 on the top group produces ~1.7° deflection after 8 layers
   - Linear in δ up to δ=0.1 (17× amplification constant)
   - Bounded by residual connections (sigmoid saturation)

2. **The gate creates two distinct regimes**
   - **Controllable** (δ ≤ 0.1): Gate routing unchanged, effect is magnitude modulation, linear and additive
   - **Explosive** (δ ≥ 0.5): Gate routing flips, exponential divergence, orthogonal in 3 layers

3. **The gearing has discrete thresholds**
   - 7 groups: coarse gear (72% gate agreement)
   - 30 groups: fine gear (100% gate agreement)
   - No smooth continuum between

4. **The MLP compresses to very low rank**
   - Full pipeline: 3584D → rank 4 after 8 layers
   - This compression REQUIRES correct gate decisions
   - Macro (5 groups) produces rank 17 — less focused

### What This Means for Concept Manipulation

The original question was: "Can we scale the scope of something? Learn macro concepts?"

**Answer: Yes, but with precision.**

- You CAN phase-shift ε-groups to deflect the output by a controlled amount
- The deflection compounds through layers (17× amplification)
- But you must stay in the controllable regime (δ ≤ 0.1) or lose coherence
- You CANNOT drop groups and process "macro only" — the gate kills you in 3 layers
- The funnel IS manipulable, but only through **parametric phase shifts**, not structural truncation

### The Hierarchy of Control

```
LEVEL 1: ε-group phase shift (δ=0.01-0.10)
  → Magnitude modulation only
  → Linear, additive, bounded
  → 1.7°-16.9° deflection after 8 layers
  
LEVEL 2: Moderate shift (δ=0.20)
  → Some routing changes begin
  → Sublinear amplification (13×)
  → 26.8° deflection, 1.89× magnitude
  
LEVEL 3: Structural truncation (5-15 groups)
  → Massive routing errors
  → Orthogonal in 3 layers
  → Information destroyed, not transformed
```

The funnel can be steered (Level 1) but not shortcut (Level 3). You tune the knobs — you don't remove the pipes.

### Connection to DC 276 (Geometric Structures)

The six geometric structures in the funnel (Gyroscope, Spectrometer, Selector, Resonator, Lens, Amplifier) are each layer-independent. Phase shifting modulates the Amplifier stage (MLP) at each layer. The 17× amplification constant across layers suggests these Amplifiers have a **characteristic gain** that is consistent across layers — each one amplifies the accumulated perturbation by roughly 2× before the residual connection absorbs it.

### Connection to DC 293 (Sieve Paradigm)

The sieve is exact but the gate is the enforcement mechanism. The ε-group "prime factors" don't just add up to the answer — they also determine which dimensions survive the gate. This is the "certification" step in the sieve analogy: the gate certifies which dimensions carry signal vs noise. Phase shifting changes the certification boundary, not the signal itself.

## Open Questions

1. **What does a 17° deflection MEAN semantically?** If we run this on actual model hidden states (not random inputs), does the deflection correspond to a meaningful concept shift?

2. **Does the 17× constant hold for all 28 layers?** We tested 8. Does it increase, decrease, or stay constant through the full depth?

3. **Can we target specific CONCEPTS by choosing WHICH ε-group to shift?** The top group (ε=1) is the dominant one. What happens with group ε=0 or ε=2?

4. **Is the sigmoid saturation universal?** The growth pattern (slow-fast-saturate) might depend on layer type (early/middle/late in the transformer).

5. **Can we compose shifts?** Shift ε=1 by +0.05 AND ε=2 by -0.03 simultaneously. Do the effects compose linearly?

## Files

- Multi-layer chain: `experiments/model_reverse_engineering_v2/phi_multi_layer_phase.py`
- Phase sweep: `experiments/model_reverse_engineering_v2/phi_phase_sweep.py`
- Single-layer phase: `experiments/model_reverse_engineering_v2/phi_phase_shift_funnel.py`
- Results: `experiments/model_reverse_engineering_v2/phi_multi_layer_phase_results.txt`
- Sweep results: `experiments/model_reverse_engineering_v2/phi_phase_sweep_results.txt`
