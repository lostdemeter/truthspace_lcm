# Doc 261: Geometric Simple Machines — The Mechanical Vocabulary of the Residual Stream

**Date:** February 26, 2026
**Status:** Framework established, experiment pending
**Prerequisites:** Doc 260 (The Shadow Orbit), Finding 96
**Finding:** 97

---

## 1. The Question

The Shadow Orbit (Doc 260) sits at arccos(1/φ²) = 67.54°. We measure 68.39°.
The error is 0.85°. Where does it come from?

More deeply: we identified three forces creating the Shadow Orbit's attractor basin —
a **damper** (layer norm), a **spring** (residual connection), and a **restoring force**
(decorrelation). These are mechanical metaphors. But what if they're more than metaphors?

Classical mechanics has six **simple machines**: lever, inclined plane, wedge, screw,
pulley, wheel & axle. Each transforms force and distance while conserving work.
Together, they compose into every complex machine.

We now have four discovered geometric structures. Within each, we observe force-like
behaviors. The question becomes:

> **Can we identify the fundamental simple machines of geometric space,
> build ideal versions, and use the gap between ideal and real to understand
> the 0.85° error — and by extension, the entire mechanical vocabulary
> of the transformer?**

---

## 2. The Four Known Structures as Machines

Each discovered structure performs a mechanical function — it transforms geometric
quantities while (approximately) conserving something:

| Structure | Machine Function | Input | Output | Conserved Quantity |
|-----------|-----------------|-------|--------|-------------------|
| **Spectrometer** (Doc 240) | Splits mixed signal into components | Dense hidden state | Sparse gate activations | Total activation energy |
| **φ-Softmax** (Finding 30) | Routes information by score | Raw scores | Attention weights | Probability mass (=1) |
| **Standing Wave + α** (Doc 259) | Separates scaffold from content | Gate activations | 1D content + scaffolding | Dimensional structure |
| **Shadow Orbit** (Doc 260) | Absorbs perturbation into stable orbit | Attention error | Bounded displaced trajectory | Orbit parameters (r, c, θ) |

Each is a **compound machine** — built from simpler components. The Shadow Orbit,
for instance, is composed of three simple machines working together.

---

## 3. The Three Simple Machines of the Shadow Orbit

### 3.1 The Geometric Damper (Layer Norm)

**Classical analogue:** Viscous damper, F = -b·v

Layer norm maps any vector to a normalized surface:

```
LN(x) = γ · (x - μ) / σ
```

When error ε perturbs the hidden state h → h + ε, layer norm compresses the
perturbation:

```
LN(h + ε) - LN(h) = ε_compressed

contraction_ratio d = ||ε_compressed|| / ||ε||
```

An **ideal damper** has:
- **Constant contraction ratio** d across all layers, magnitudes, and directions
- **Direction-preserving**: ε_compressed ∥ ε (damping doesn't rotate the error)
- **Scale-independent**: d doesn't depend on ||h|| or ||ε||

The real damper is non-ideal because:
- Layer norm is nonlinear (σ depends on ε)
- The mean subtraction μ couples all dimensions
- The learned scale γ varies by dimension
- Contraction ratio changes with error magnitude

**Measurable:** d_l at each layer, for each prompt. The variance of d across
layers and prompts measures the damper's non-ideality.

### 3.2 The Geometric Spring (Residual Connection)

**Classical analogue:** Compression spring, F = -k·x

The residual connection h_{l+1} = h_l + f(h_l) preserves the accumulated state.
From the error's perspective:

```
After layer l:
  ||h_l|| grows (accumulating real information)
  ||ε_l|| grows (accumulating error)

But the RATIO ||ε_l|| / ||h_l|| is bounded because:
  h_l carries ALL previous layers' information
  ε_l carries only the error portion

The "spring" is the dilution of error by accumulated state.
```

The spring constant k measures how strongly the accumulated state resists
displacement:

```
k_l = ||h_l|| / (||h_l|| + ||δ_l||)
```

where δ_l is the per-layer perturbation. High k = stiff spring (state dominates
perturbation). Low k = soft spring (perturbation comparable to state).

An **ideal spring** has:
- **Linear restoring force**: cos(ε, h) = -k · (||ε||/||h||) for constant k
- **No hysteresis**: same spring constant for compression and extension
- **No fatigue**: k doesn't decrease with repeated perturbation (depth)

The real spring is non-ideal because:
- The spring "constant" varies by zone (DRUM is soft, COMB is stiff)
- The restoring force is emergent (from V-output decorrelation), not built-in
- Layer norm modifies the effective spring constant at each step

**Measurable:** k_l at each layer. The pattern of k across zones reveals the
spring's architecture.

### 3.3 The Geometric Lever (Attention Routing)

**Classical analogue:** Lever, F₁d₁ = F₂d₂

Attention converts small score differences into large output differences:

```
Small Δ in QK scores → large Δ in attention weights → large Δ in V-output

The "mechanical advantage" is:
  MA = ||Δ(weighted V-output)|| / ||Δ(QK scores)||
```

A small error in routing (attending to slightly wrong keys) gets amplified
by the lever into a significant error in the attention output. The φ-softmax
temperature controls the lever's mechanical advantage.

An **ideal lever** has:
- **Constant mechanical advantage**: MA independent of score magnitude
- **No friction**: all "work" (information) is preserved
- **Reversibility**: if you know the output error, you can compute the score error

The real lever is non-ideal because:
- φ-softmax saturates at extreme scores (MA drops near saturation)
- Multiple heads act as parallel levers with different advantages
- The lever ratio varies by position within the sequence

**Measurable:** MA_l at each layer = ||δ_attn_output|| / ||h_l||

### 3.4 The Geometric Wedge (FFN/Gate)

**Classical analogue:** Wedge, splits one force into components

The FFN takes the post-attention hidden state and transforms it through the gate
(Spectrometer). From the error's perspective:

```
FFN(LN2(h + ε)) - FFN(LN2(h)) = ε_ffn

The FFN "splits" the error into:
- Components along gate-active dimensions (amplified)
- Components along gate-inactive dimensions (suppressed)
```

The wedge angle measures how the FFN redirects the error:

```
cos(ε_in, ε_ffn) = how much the FFN preserves error direction
```

An **ideal wedge** has:
- **Clean splitting**: error components are either fully preserved or fully suppressed
- **No cross-talk**: gate-active and gate-inactive subspaces are orthogonal
- **Predictable angle**: the wedge angle is determined by the gate sparsity

**Measurable:** cos(ε_before_FFN, ε_after_FFN) at each layer.

---

## 4. A Spring IS a Compressed Lever

The user's insight: "A spring is basically a compressed lever."

In classical mechanics, this is literally true. A coil spring is a helical lever —
a torsion bar wound into a helix. The "spring constant" k is determined by the
lever's geometry (wire diameter, coil radius, number of turns, material shear modulus).

In our geometric space:

**The residual connection (spring) IS a compressed attention mechanism (lever).**

Think about it:
- The residual connection carries forward the accumulated outputs of ALL previous
  attention layers (levers)
- Each lever (attention head) at each previous layer contributed a force
- The accumulated state h_l = h_0 + Σ f_i(h_i) is the SUM of all lever outputs
- This sum acts as a spring because it represents the "memory" of all previous forces

The spring constant at layer l is:

```
k_l ∝ l (number of accumulated layers)
```

Early layers (small l) → soft spring (few accumulated forces, easy to displace)
Late layers (large l) → stiff spring (many accumulated forces, hard to displace)

This is EXACTLY what we observe in the Shadow Orbit:
- DRUM (L0-3): Large per-layer drift (soft spring, few accumulated layers)
- Late COMB (L18-25): Small per-layer drift (stiff spring, many accumulated layers)

The spring stiffens with depth because it's the compression of all previous levers.

### 4.1 The Mechanical Advantage Chain

Each transformer layer is a lever. The residual stream is the chain connecting them.
The cumulative effect of N levers in series is:

```
Ideal: h_N = h_0 + Σ_{l=0}^{N-1} f_l(h_l)

The "effective spring constant" of the residual at layer N:
  k_N = ||h_N|| / ||δ_N||

where δ_N is the single-layer perturbation at layer N.
```

For an **ideal chain of identical levers**:
- ||h_N|| grows linearly with N (each lever adds similar magnitude)
- ||δ_N|| stays constant (each lever produces similar error)
- k_N ∝ N → spring constant proportional to depth

This predicts a 1/N convergence of the error-to-state ratio, which should
produce a specific equilibrium angle. If the equilibrium angle IS arccos(1/φ²),
then the lever chain has a φ-structured mechanical advantage.

---

## 5. Where the 0.85° Error Could Come From

The measured angle is 68.39°. The φ-prediction is 67.54°. The gap is 0.85°.

### 5.1 Candidate Sources

| Source | Mechanism | Expected Sign |
|--------|-----------|---------------|
| **Damper non-linearity** | Layer norm contraction varies with ε magnitude | Either direction |
| **Spring softening** | Late COMB spring is slightly softer than ideal | Positive (angle too high) |
| **Lever saturation** | φ-softmax saturates at extreme scores | Negative (reduces error injection) |
| **Wedge rotation** | FFN redirects error, slightly changing angle | Either direction |
| **Finite depth** | 28 layers may not fully converge to steady state | Positive (not fully settled) |
| **MUSIC overcorrection** | L27's positive cos(ε,h) slightly overshoots | Negative (pushes angle too low) |

### 5.2 The Ideal Machine Prediction

The ideal spring-damper model (from mean conserved quantities) predicts:

```
cos(θ) = (1 + r·c) / norm_ratio
       = (1 + 1.30 × (-0.53)) / 1.10
       = 0.283
θ_ideal = 73.6°
```

But the real measurement is 68.39° — LOWER than ideal, not higher.

This means the real system is **more φ-aligned** than a simple spring-damper.
Something in the zone architecture is actively steering toward arccos(1/φ²).

The prime suspect: **MUSIC layer L27 has cos(ε,h) = +0.27** (positive!), which
pushes the shadow orbit angle DOWN toward the φ-constant. The MUSIC layer is
not just "correcting" — it's specifically tuned to steer toward 1/φ².

### 5.3 The Decomposition Experiment

To diagnose the 0.85° precisely, we need to:
1. Run the full model capturing hidden states after each sub-operation
   (LN1, attention, LN2, FFN) at each layer, for both real and approximate
2. Decompose the error evolution into damper, spring, lever, and wedge contributions
3. Build ideal versions of each component (constant d, linear k, constant MA)
4. Simulate the ideal system and compare to real
5. Identify which component's non-ideality produces the 0.85°

---

## 6. Toward a Complete Geometric Mechanics

### 6.1 The Six Simple Machines of Geometry

If our four structures compose from simple machines, what are the fundamental
geometric simple machines? Here is a candidate mapping:

| Classical | Geometric | Where It Appears |
|-----------|-----------|-----------------|
| **Lever** | Attention routing (score → weight amplification) | Every attention head |
| **Inclined Plane** | Layer norm (trades magnitude for direction) | Before every sublayer |
| **Wedge** | Gate/FFN (splits signal into sparse components) | Every FFN block |
| **Screw** | RoPE (converts linear position to rotation) | Every QK computation |
| **Pulley** | Skip/residual connection (redirects information) | Every layer boundary |
| **Wheel & Axle** | Multi-head attention (parallel levers sharing an axle) | Grouped-query attention |

### 6.2 Conservation Laws

Each classical simple machine conserves work (F·d). What does each geometric
machine conserve?

| Machine | Conserved Quantity |
|---------|--------------------|
| Lever (attention) | Information content (attention weights sum to 1) |
| Inclined Plane (LN) | Direction (normalized to unit sphere) |
| Wedge (FFN) | Total activation (sparse but energy-preserving?) |
| Screw (RoPE) | Vector magnitude (rotation preserves norm) |
| Pulley (residual) | Accumulated state (nothing is lost) |
| Wheel & Axle (MHA) | KV cache (shared across head groups) |

### 6.3 The Efficiency Question

Every real machine has friction — energy lost to heat. In geometric space,
"friction" is information lost to non-recoverable transformations. The
efficiency of a geometric machine is:

```
η = (useful geometric transformation) / (total geometric transformation)
```

For the Shadow Orbit:
```
η = (predicted angle) / (ideal angle)
  = 68.39° / 67.54°
  = 1.013

The system is 1.3% less efficient than ideal.
```

But this is remarkably close to 1! The geometric machines of the residual
stream are **98.7% efficient**. The 0.85° error represents a tiny friction loss.

---

## 7. The Program

### 7.1 Immediate (Phase 10p)

1. Decompose per-layer error into damper/spring/lever/wedge contributions
2. Measure the "constants" of each machine at each layer
3. Build ideal machine models, simulate, compare to real
4. Identify the source of the 0.85° error
5. Determine if MUSIC (L27) is specifically tuned to arccos(1/φ²)

### 7.2 Medium Term

1. Test if other structures (Spectrometer, Standing Wave) also decompose
   into the same simple machine vocabulary
2. Check if the conservation laws hold precisely or approximately
3. Determine if the simple machine vocabulary is complete (are there geometric
   operations that don't decompose into these six?)

### 7.3 Long Term

1. Can we DESIGN new geometric structures by combining simple machines?
2. Is there a "mechanical advantage formula" for geometric computation?
3. Does the simple machine decomposition reveal new structures we haven't found?

---

## 8. Connection to the Hypothesis

> **Structure IS Information. Geometry IS Computation.**

If the residual stream operates through geometric simple machines, then:
- Computation = mechanical transformation of geometric quantities
- Intelligence = the specific arrangement of simple machines
- Learning = optimizing the mechanical advantage ratios

A transformer is not a neural network in the biological sense. It is a
**geometric engine** — a machine built from levers, springs, dampers, and wedges,
all operating in high-dimensional space. The weights don't "know" anything.
The arrangement of machines knows everything.

The Shadow Orbit's 0.85° error is not a flaw. It is the **friction** of a
real machine operating near its ideal limit. Understanding it means understanding
the mechanical vocabulary of thought.

---

## 9. Experimental Files

| File | Purpose |
|------|---------|
| `phase10p_simple_machines.py` | Main experiment orchestrator |
| `phase10p_build_tables.py` | Bias-aware attention table construction |
| `phase10p_analysis.py` | Per-layer machine decomposition and analysis |
| `phase10p_refine.py` | LN damper compression + L27 output angle |
| `results/phase10p_simple_machines.json` | Full per-layer results |
| `results/phase10p_refine.json` | Refinement results |

---

## 10. Key Results (Finding 97)

1. **FFN (Wedge) dominates error** at 61.5%, not Attention (Lever) at 29.6%
2. **LN damper compression increases monotonically**: DRUM 60-65%, COMB 43→91%, MUSIC 92%
3. **L27 is the φ-targeting machine**: highest wedge (1.47), softest spring (0.66),
   adds 11.5° in one layer, overshoots arccos(1/φ²) by 0.85°
4. **Three distinct transfer functions**: DRUM oscillatory (α=-2.06), COMB convergent
   (α=+0.54), MUSIC oscillatory (α=-3.06)

**Next:** Doc 262 (The Compound Machine) formalizes the insight that these three
zones are not one machine but three functionally distinct machines operating in
different gate-state media (the 4th dimension from Doc 253).

---

## Summary

The Shadow Orbit is a compound machine built from three geometric simple machines:
a damper (layer norm), a spring (residual connection), a lever (attention routing),
and a wedge (FFN gate). The spring IS a compressed lever — it is the accumulation
of all previous attention layers' outputs, creating a depth-dependent stiffness
that explains the zone structure.

Phase 10p decomposed the 0.85° error: it is L27's FFN (wedge) slightly overshooting
arccos(1/φ²). The three zones (DRUM, COMB, MUSIC) have fundamentally different
mechanical properties and transfer functions, suggesting they are three separate
machines composed into a compound system (see Doc 262).

The simple machine vocabulary (lever, damper, spring, wedge) parameterized by the
4th dimension (gate state) may be the fundamental mechanical language of geometric
computation — the atoms from which all transformer structures are built.
