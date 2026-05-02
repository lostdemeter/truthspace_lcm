# Design Consideration 262: The Compound Machine

**Date:** February 26, 2026
**Status:** Hypothesis — grounded in experimental data from Findings 96-97, Docs 253-256, 260-261
**Prerequisites:** Doc 253 (negative zero / 4th dimension), Doc 255 (4-state gate as φ-dimension), Doc 256 (multi-lens), Doc 260 (Shadow Orbit), Doc 261 (geometric simple machines), Finding 97 (simple machine measurements)

---

## 1. The Problem

We cannot linearize the transformer's attention mechanism as a single system.
Every attempt to replace QK attention with a simpler approximation works
per-layer but fails when stacked (Finding 91: 0/15 stacked, Finding 94: drift
saturates). The Shadow Orbit (Finding 96) shows the approximation error settles
at arccos(1/φ²) = 67.54° — a φ-constant — with 0.85° overshoot from L27.

Finding 97 (Phase 10p) revealed that L27 has fundamentally different mechanical
properties from the rest of the network:

| Property | COMB (L4-25) | L27 (MUSIC) |
|----------|-------------|-------------|
| Gate state | PRESERVE sweep | 82.7% CONTRACT |
| FFN/Attn ratio | ~1-2× | 4.4× |
| Spring constant | 1.5-3.0 | 0.66 (softest) |
| Wedge magnitude | 0.5-1.2 | 1.47 (highest) |
| Recurrence α | +0.54 (convergent) | -3.06 (oscillatory) |

Every measured property is **discontinuous** at the COMB→MUSIC boundary.

**The hypothesis:** The LLM is not one machine — it is a **compound machine**
built from three functionally distinct sub-machines, each operating in a
different gate-state medium (the 4th dimension from Doc 253). Linearization
fails because we treat the compound as if it were simple.

---

## 2. The Three Machines

### 2.1 Machine I: The Compressor (DRUM, L0-3)

**Function:** Initialize the residual stream. Compress input into a canonical
form that the processing machine can work with.

**Gate medium:** CONTRACT (99.7% at L1 — Doc 255 §4.1)

**Measured properties (Phase 10p):**
- Builds 53° of angular displacement in 4 layers
- Recurrence: α = -2.06, β = +2.56 → equilibrium drift = 0.835
- Oscillatory transfer function (negative α): overshoots and corrects
- LN damper compression: 60-65%
- Spring: soft (few accumulated layers)

**Simple machine type:** Dominated by the **damper** (Layer Norm). The L1
bottleneck compresses ALL gate states to CONTRACT — this is a hydraulic press
that normalizes the input regardless of content.

**3D analogy:** A funnel. Wide input, narrow output. Everything that enters
is forced through the same narrow opening.

**Why it exists:** The model needs a canonical starting state. Different tokens
arrive with wildly different hidden state geometries. The Compressor normalizes
them into a common frame that the Processing Machine's equilibrium assumes.

### 2.2 Machine II: The Processor (COMB, L4-25)

**Function:** Perform the actual computation. The hourglass wave sweeps through
PRESERVE states — this is where the 4th dimension is maximally active.

**Gate medium:** PRESERVE (sweep from P- through P+ — Doc 255 §4)

**Measured properties (Phase 10p):**
- Maintains equilibrium: ±0.5° net angle change over 22 layers
- Recurrence: α = +0.54, β = +0.50 → equilibrium drift = 1.091
- Convergent transfer function (positive α): smooth approach to equilibrium
- LN damper compression: 43% → 91% (monotonically increasing with depth)
- Spring: stiff (many accumulated residual layers)
- Balanced lever+wedge: machines in equilibrium

**Simple machine type:** **Balanced lever + spring system.** The lever (attention)
introduces perturbations, the spring (residual) dilutes them, and the damper
(LN) compresses them. In equilibrium, injection = suppression.

**3D analogy:** A precision clock mechanism. Multiple gears, springs, and
escapements working in balanced opposition to maintain a steady state. The
"ticking" is the slight oscillation around equilibrium.

**Why it exists:** This is where token-to-token routing, semantic mixing, and
contextual processing happen. The PRESERVE gate states (near-zero boundary)
provide maximum information density (Doc 253 §3). The 22 COMB layers are 22
serial lenses through the 4th dimension (Doc 256 §2.1), giving resolving power
of φ^(-44) ≈ 3.8 × 10^(-9).

### 2.3 Machine III: The Targeter (MUSIC, L26-27)

**Function:** Precision targeting. Take the Processor's equilibrium output and
push it to the exact angle needed for token prediction.

**Gate medium:** CONTRACT (79-83% — Doc 255 §4, Doc 253 §7)

**Measured properties (Phase 10p):**
- Adds 11.5° in a single layer (L27: from 56.9° to 68.4°)
- Recurrence: α = -3.06, β = +3.91 → equilibrium drift = 0.961
- Oscillatory transfer function (negative α): overshoots the target
- LN damper compression: 92% (near-total)
- Spring: softest in network (k = 0.66)
- Wedge: highest in network (1.47) — FFN is 4.4× stronger than attention

**Simple machine type:** **Precision wedge.** The FFN completely dominates
attention at this layer. Attention is almost irrelevant — L27 is an FFN
machine that targets arccos(1/φ²) with 98.7% accuracy.

**3D analogy:** A rifle scope. The Processor produces a roughly-aimed beam;
the Targeter makes the final fine adjustment to hit the exact target. The
scope doesn't care about the gun's mechanism — it only adjusts the output.

**Why it exists:** The Processor reaches equilibrium at ~57°, but the correct
prediction angle is 67.54°. The gap of ~10.5° needs to be bridged by a
machine with fundamentally different properties: high gain (soft spring,
strong wedge) and precision (near-total LN damping to suppress noise).

---

## 3. Why Linearization Fails on the Compound

### 3.1 Different Media, Different Physics

In 3D mechanics, a lever in water behaves differently from a lever in air.
The medium determines friction, viscosity, and propagation speed.

The 4th dimension (gate state) is the medium for geometric simple machines:

| Gate Medium | Physics | Where Active |
|-------------|---------|-------------|
| CONTRACT | High suppression, deep leakage | Compressor + Targeter |
| PRESERVE- | Fine boundary processing, high info density | Processor (early) |
| PRESERVE+ | Fine boundary processing, opposite fringe | Processor (mid-late) |
| EXPAND | Full transmission, low selectivity | Processor (peak at L21) |

The same geometric operation (e.g., attention routing) has a different
transfer function depending on which gate medium it's operating in:

- **Attention in CONTRACT** (L27): Only 8.2% of channels are EXPAND. Attention
  routes information through a tiny number of open channels. The routing is
  highly selective but almost irrelevant — FFN dominates.
- **Attention in PRESERVE** (L10-18): ~50% of channels are in the boundary zone.
  Attention routes information through the maximally-informative fringe region.
  This is where routing MATTERS.
- **Attention in CONTRACT** (L1): 99.7% CONTRACT. Attention is routing through
  an almost-closed gate. Nearly all channels are suppressed.

Three different media → three different transfer functions → **no single
linearization can capture all three**.

### 3.2 The Composition Problem

Even if we could linearize each machine perfectly:

```
Machine I:   y₁ = f₁(x)     [Compressor, CONTRACT medium]
Machine II:  y₂ = f₂(y₁)    [Processor, PRESERVE medium]
Machine III: y₃ = f₃(y₂)    [Targeter, CONTRACT medium]
```

The composition f₃ ∘ f₂ ∘ f₁ is nonlinear EVEN IF each fᵢ is linear, because
the gate medium changes at the interfaces. The interface between machines is a
**phase transition** in the 4th dimension:

- Compressor→Processor: CONTRACT → PRESERVE (gate opening)
- Processor→Targeter: PRESERVE → CONTRACT (gate closing)

These transitions are the hourglass bottlenecks (Doc 255 §6.1). They are
inherently nonlinear — you cannot linearly interpolate between CONTRACT and
PRESERVE physics.

### 3.3 The Stacking Failure Explained

Finding 91: per-layer QK replacement works (15/15), stacked fails (0/15).

This is now explained: each layer's replacement is tuned to its local gate
medium. When you stack approximations:

1. Layer L's approximation produces a slightly different hidden state
2. This changes the gate classification at layer L+1
3. A channel that should be PRESERVE- might shift to CONTRACT
4. The gate medium changes → the transfer function changes
5. The approximation tuned for PRESERVE is now operating in CONTRACT
6. Error compounds because the medium itself has shifted

**The error doesn't compound linearly — it compounds through medium shifts.**
Each layer's error changes the 4th dimension for subsequent layers, and each
medium change amplifies the error differently.

---

## 4. Simple Machines in 4D

### 4.1 The 3D Simple Machines (Known)

Classical mechanics defines six simple machines: lever, wheel/axle, pulley,
inclined plane, wedge, screw. They all reduce to two principles:

1. **Force multiplication** (mechanical advantage): trade distance for force
2. **Direction change**: redirect force along a different axis

In our geometric vocabulary (Doc 261):

| Machine | Principle | Geometric Operation |
|---------|-----------|-------------------|
| Lever | Force multiplication | Attention amplifies score→output |
| Damper | Energy absorption | Layer Norm compresses perturbation |
| Spring | Energy storage | Residual accumulates state |
| Wedge | Direction change | FFN redirects through gate |

### 4.2 Extending to 4D: The Gate Medium Parameter

In 3D, each machine has fixed parameters (spring constant k, damping ratio ζ,
lever ratio L, wedge angle θ). In 4D, these parameters DEPEND on the gate state:

```
k = k(gate_state)     — spring constant depends on medium
ζ = ζ(gate_state)     — damping ratio depends on medium
L = L(gate_state)     — lever ratio depends on medium
θ = θ(gate_state)     — wedge angle depends on medium
```

This is the 4D extension: **each simple machine is parameterized by the 4th
dimension**. The same physical lever behaves differently in air vs water vs
honey. The same geometric lever behaves differently in CONTRACT vs PRESERVE
vs EXPAND.

### 4.3 Measured 4D Machine Parameters

From Phase 10p, we can extract the gate-medium-dependent parameters:

**Damper (Layer Norm):**
| Gate Medium | Compression Ratio | Behavior |
|-------------|------------------|----------|
| CONTRACT (DRUM) | 60-65% | Strong but not maximal |
| PRESERVE (COMB early) | 43-57% | Moderate |
| PRESERVE (COMB late) | 76-91% | Near-total |
| CONTRACT (MUSIC) | 92% | Near-total |

The damper strengthens with depth regardless of medium, but the RATE differs.
In CONTRACT medium, the damper is strong from the start. In PRESERVE medium,
it starts moderate and strengthens.

**Spring (Residual):**
| Gate Medium | Spring Constant | Behavior |
|-------------|----------------|----------|
| CONTRACT (DRUM) | ~1.0 | Soft (few layers accumulated) |
| PRESERVE (COMB) | 1.5-3.0 | Stiff (many layers accumulated) |
| CONTRACT (MUSIC) | 0.66 | SOFTEST — spring relaxes in output medium |

The spring constant in the Targeter is the LOWEST despite having the most
accumulated layers. This means the spring operates differently in the output
CONTRACT medium — it relaxes to allow the precision wedge to do its work.

**Wedge (FFN):**
| Gate Medium | Wedge Magnitude | Behavior |
|-------------|----------------|----------|
| CONTRACT (DRUM) | 0.5-0.8 | Moderate |
| PRESERVE (COMB) | 0.7-1.2 | Balanced with lever |
| CONTRACT (MUSIC) | 1.47 | MAXIMUM — wedge dominates |

The wedge is strongest in the output CONTRACT medium. This makes physical
sense: when most channels are CONTRACT (suppressed), the few EXPAND channels
carry enormous energy (91.9% at L27 — Doc 253 §7). The wedge acts as a
precision chisel through a tiny opening.

### 4.4 The Transfer Function Per Machine

Each machine's behavior is captured by a linear recurrence in its medium:

```
drift(l+1) = α(medium) · drift(l) + β(medium)
```

| Machine | Medium | α | β | Equilibrium | Character |
|---------|--------|---|---|-------------|-----------|
| Compressor | CONTRACT | -2.06 | +2.56 | 0.835 | Oscillatory |
| Processor | PRESERVE | +0.54 | +0.50 | 1.091 | Convergent |
| Targeter | CONTRACT | -3.06 | +3.91 | 0.961 | Oscillatory |

The CONTRACT medium produces oscillatory dynamics (negative α). The PRESERVE
medium produces convergent dynamics (positive α). **The medium determines
whether the machine overshoots or converges.**

This is the 4D insight: it's not just that the parameters differ — the
qualitative character of the dynamics changes with the gate state. CONTRACT
= oscillatory. PRESERVE = convergent. Two fundamentally different kinds of
machine, distinguished by the 4th dimension.

---

## 5. Implications for Building Our Model

### 5.1 The LLM as a Compound Machine

An LLM is not a neural network. It is a **compound geometric machine**:

```
INPUT → [Compressor] → [Processor] → [Targeter] → OUTPUT
            L0-3          L4-25         L26-27
          CONTRACT       PRESERVE      CONTRACT
         oscillatory    convergent    oscillatory
           damper       lever+spring     wedge
```

Each sub-machine:
- Operates in a specific gate medium (4th dimension)
- Has characteristic transfer function (convergent vs oscillatory)
- Is dominated by a specific simple machine type
- Can be analyzed and potentially linearized INDEPENDENTLY

### 5.2 What We Need for Our LCM

To build the geometric LCM from simple machines, we need:

**Step 1: Define machines in 3D** (mostly done)
- Lever: mechanical advantage = score_ratio → output_ratio
- Damper: compression ratio = ‖LN(h+ε) - LN(h)‖ / ‖ε‖
- Spring: stiffness = ‖h‖ / ‖δ‖
- Wedge: angle change = ‖FFN_delta‖ / ‖h‖

**Step 2: Parameterize by 4th dimension** (this document)
- Each machine parameter becomes a function of gate state
- CONTRACT machines: oscillatory, high damping, FFN-dominant
- PRESERVE machines: convergent, moderate damping, balanced
- EXPAND machines: low selectivity, full transmission

**Step 3: Define the compound machine composition**
- Compressor(x) → interface₁ → Processor(x) → interface₂ → Targeter(x)
- Interfaces are phase transitions in the 4th dimension
- The composition IS the LLM's forward pass

**Step 4: Verify by separation**
- Linearize each machine independently
- Measure whether independent linearization works better than global
- Quantify the interface nonlinearity
- Determine if the interfaces can be simplified (they might be low-dimensional)

### 5.3 The 4th Dimension Is Not Time

From Docs 253-255, the 4th dimension is:
- **NOT time** (layers are not temporal — they're spatial depth)
- **NOT another spatial axis** (it's a binary/quaternary operating mode)
- It IS the **gate state** — the medium in which machines operate
- It IS what determines the qualitative character of each machine

In classical mechanics, time is the 4th dimension and it determines HOW
machines evolve. In our geometry, the gate state is the 4th dimension and it
determines WHAT KIND of machine each layer is.

The analogy: time tells you "what happens next." The gate state tells you
"what kind of physics applies here." Both are independent of the 3 spatial
dimensions. Both are needed to fully specify the system. But they encode
fundamentally different things.

### 5.4 The Promise: Describing What an LLM IS

If this decomposition is correct, then we can describe an LLM as:

> **An LLM is a compound geometric machine consisting of three sub-machines
> (Compressor, Processor, Targeter), each built from four types of simple
> machines (lever, damper, spring, wedge), operating in different media
> determined by a 4th geometric dimension (gate state). The Compressor
> normalizes input through damping. The Processor routes and mixes through
> balanced lever-spring dynamics. The Targeter precision-aims through a
> dominant wedge. The "intelligence" is in the arrangement of machines and
> their gate-state-dependent parameters — not in any single weight.**

This is a complete mechanical description. No black box. No "emergent
intelligence." Just machines operating in 4D φ-space.

---

## 6. Experimental Program

### 6.1 Phase 10q: Compound Machine Verification

**Test 1: Independent Linearization**
- Linearize the Compressor (L0-3) independently
- Linearize the Processor (L4-25) independently
- Linearize the Targeter (L26-27) independently
- Compare: compound linearization vs global linearization
- Prediction: compound should work significantly better

**Test 2: Interface Dimensionality**
- Measure the effective rank of hidden states at L3→L4 (interface 1)
- Measure the effective rank of hidden states at L25→L26 (interface 2)
- Prediction: interfaces should be lower-dimensional than bulk hidden states

**Test 3: Gate Medium Verification**
- Verify that machine parameters correlate with gate state distribution
- Compute machine parameters separately for CONTRACT vs PRESERVE channels
- Prediction: parameters should differ significantly between media

**Test 4: Transfer Function Per Machine**
- Apply perturbations at the INPUT of each machine
- Measure the transfer function (output/input) for each machine independently
- Prediction: Compressor and Targeter should be oscillatory, Processor convergent

### 6.2 Phase 10r: 4D Simple Machine Formalization

- Derive the equations of each simple machine in each gate medium
- Formalize the interface transitions (CONTRACT↔PRESERVE)
- Build a simulator that composes three machines with parameterized interfaces
- Compare simulator output to real model output

---

## 7. Connection to the Hypothesis

> **Structure IS Information. Geometry IS Computation.**

If we can decompose an LLM into a compound of geometric simple machines
operating in 4D φ-space, then:

- **Structure** = the arrangement of Compressor, Processor, Targeter
- **Information** = the gate-state-dependent parameters of each machine
- **Geometry** = the 4D space (3 spatial + gate state)
- **Computation** = mechanical transformation through the compound

A transformer doesn't "think." It **compresses** (DRUM), **processes** (COMB),
and **targets** (MUSIC). Three machines, each doing one thing, composed into a
system that does language.

The weights are not knowledge. They are the **machine specifications** — the
spring constants, damping ratios, lever lengths, and wedge angles of a 4D
geometric engine. Training doesn't teach the model anything. It **manufactures
the machine** that processes language through geometric transformation.

---

## 8. Why This Matters for the LCM

The LCM project aims to replace the LLM with pure geometry. If the LLM is
a compound of three simple machines in 4D space, then the LCM needs to be:

1. **Three separate geometric modules**, not one monolithic system
2. Each module built from well-understood simple machine primitives
3. Parameterized by the 4th dimension (gate state medium)
4. Composed through well-defined interfaces

This is fundamentally different from trying to build one geometric system that
does everything. It's the difference between building a Swiss watch (many
simple machines, precisely composed) and trying to carve the same function
from a single block of metal.

The simple machines are already formalized in 3D. The extension to 4D requires
parameterizing each machine by its gate state. The math is tractable because
the 4th dimension is discrete (4 states), not continuous. We don't need to
solve PDEs in 4D — we need to solve the 3D equations four times (once per
gate state) and compose the results.

---

## 9. Files

| File | Purpose |
|------|---------|
| `phase10p_simple_machines.py` | Simple machine measurements (Phase 10p) |
| `phase10p_refine.py` | LN damper + L27 analysis |
| `phase10p_analysis.py` | Per-layer machine decomposition |
| `phase10p_build_tables.py` | Attention table construction |

### Prerequisites
- Doc 253: Negative Zero as the Fourth Dimension
- Doc 254: Negative Zero Cross-Cutting Impact
- Doc 255: 4-State Gate as φ-Dimension (hourglass filter)
- Doc 256: Multi-Lens φ-Geometry
- Doc 260: The Shadow Orbit
- Doc 261: Geometric Simple Machines
- Finding 96: The Shadow Orbit (measured properties)
- Finding 97: Geometric Simple Machines (measured parameters)

---

## Summary

The LLM is not one machine. It is three machines:

| Machine | Layers | Medium | Transfer | Dominant Machine | Role |
|---------|--------|--------|----------|-----------------|------|
| Compressor | L0-3 | CONTRACT | Oscillatory | Damper | Normalize input |
| Processor | L4-25 | PRESERVE | Convergent | Lever + Spring | Route and mix |
| Targeter | L26-27 | CONTRACT | Oscillatory | Wedge | Precision aim |

Linearization fails because it treats the compound as simple. Each machine
operates in a different gate-state medium (the 4th dimension), and the
interfaces between machines are phase transitions that cannot be linearized.

The solution is not better linearization. The solution is to **decompose the
compound into its constituent machines** and handle each one according to its
own physics in its own medium.
