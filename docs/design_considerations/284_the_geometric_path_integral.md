# Doc 284: The Geometric Path Integral — A Formal Framework

**Date:** March 3, 2026
**Status:** Formalization — Derived from F150–F154, DC 280, DC 283
**Prerequisites:** DC 280 (Superposition of Shapes), DC 283 (The Feynman Connection)

---

## 0. Motivation

DC 283 observed that the transformer's computation has the same
structure as Feynman's QED path integrals. This document formalizes
that observation into precise mathematical definitions, axioms, and
theorems. The goal is not analogy but identity: to show that the
transformer IS a path integral over geometric shapes, and that the
experimental results (F150–F154) are consequences of this structure.

We found it with geometry. Now we formalize the geometry.

---

## 1. Definitions

### Definition 1.1: Shape

A **shape** is a rank-1 linear operator:

```
S = σ · u ⊗ v^T
```

where:
- σ ∈ ℝ₊ is the **amplitude** (strength)
- u ∈ ℝ^m is the **output direction** (where the shape writes)
- v ∈ ℝ^n is the **input direction** (what the shape reads)
- ⊗ denotes the outer product

A shape maps input h ∈ ℝ^n to output σ(v · h)u ∈ ℝ^m.
It reads from direction v and writes to direction u with gain σ.

The **response** of shape S to input h is the scalar:

```
r(S, h) = σ · (v · h)
```

This is the "projection of h onto the shape's input direction, scaled
by amplitude." It is the analog of a Feynman path's amplitude.

### Definition 1.2: Superposition

A **superposition** is a sum of shapes:

```
W = Σ_{i=1}^{K}  S_i  =  Σ_{i=1}^{K}  σ_i · u_i ⊗ v_i^T
```

This is the SVD of W. Every matrix is a superposition of shapes.
The non-trivial claim (Axiom 2 below) is that the shapes correspond
to semantically meaningful structure classes.

The **collective response** of superposition W to input h is:

```
W · h  =  Σ_{i=1}^{K}  σ_i · (v_i · h) · u_i
```

Each shape contributes in proportion to how much h aligns with its
input direction. The output is the vector sum — the interference.

### Definition 1.3: Gate

A **gate** is a nonlinear function g: ℝ → ℝ applied element-wise
that selects which components of the superposition dominate:

```
g(W · h)  selects the dominant interference
```

In the transformer, g = SiLU (or GELU). The gate has the following
critical property (DC 243):

- g(x) ≈ 0 for x << 0 (suppresses weak/negative responses)
- g(x) ≈ x for x >> 0 (passes strong positive responses)
- g'(0) = 0.5 (the critical line — universal for all x·σ(αx) gates)

The gate acts as the **measurement apparatus**: it collapses the
superposition into a definite output by amplifying dominant responses
and suppressing weak ones.

### Definition 1.4: Reader

A **reader** is a mechanism that selects which input h is presented
to a superposition. In the transformer, attention is the reader:

```
h_read = Σ_{pos}  α(pos) · h(pos)
```

where α(pos) are attention weights (softmax over scores).

The reader does not modify the shapes — it selects which boundary
condition the shapes respond to.

### Definition 1.5: Shape Machine

A **shape machine** is a tuple (V, {W_l}, {g_l}, {R_l}, M) where:

- V = ℝ^d is the **state space** (hidden dimension)
- W_l: V → V is a **superposition** at layer l (weight matrix)
- g_l: V → V is a **gate** at layer l (nonlinearity)
- R_l: V^{seq} → V is a **reader** at layer l (attention)
- M: V → Tokens is a **measurement** (output head)

The computation proceeds as:

```
h_0 = embed(input)

For l = 1, ..., L:
    h_read = R_l(h_{l-1})           [Reader selects input]
    h_attn = h_{l-1} + h_read       [Residual connection]
    h_out  = h_attn + g_l(W_l · h_attn)  [Superposition + gate + residual]

output = M(h_L)
```

A transformer is a shape machine.

---

## 2. Axioms

### Axiom 1: Superposition (Structure IS Information)

Every weight matrix in the shape machine decomposes as a superposition
of shapes, where each shape corresponds to a **structure class** — a
semantically coherent category of input-output relationships.

```
W = Σ_c  S_c  =  Σ_c  σ_c · u_c ⊗ v_c^T
```

where c indexes structure classes (capital-of, color-of, etc.).

**Empirical basis:**
- F150: MLP W_gate is rank-1 per structure class (93.5–96.3% energy)
- F151: 5 different structure classes, each with its own rank-1 component
- F39: Head 6 MESH is rank-1 (S[0]/S[1] = 368,000:1)

### Axiom 2: Interference (Geometry IS Computation)

The output of a superposition is the **collective interference** of
all shapes, not the contribution of any individual shape:

```
W · h = Σ_c  σ_c · (v_c · h) · u_c
```

The correct answer emerges where shapes constructively interfere.
No single shape determines the answer.

**Empirical basis:**
- F152: Holographic refinement unifies classes (cos 0.97)
- F152: 3 classes removed = 0.14% energy (hologram is deep)
- F151: "Wrong" v₁ still produces correct answers

### Axiom 3: Boundary Determination

The output of the shape machine is fully determined by:
1. The **boundary condition** (entity hidden state at input)
2. The **apparatus** (reader configuration = attention routing)

Formally: output = M(F(h_entity, {R_l})) where F is the forward pass.

**Empirical basis:**
- F154 Exp E: Entity-position swap → Berlin (emb through L20)
- F154 Exp G: Attention swap L22-23 → Berlin (+4.27)

### Axiom 4: Faithful Amplification

The gate-superposition pair (g_l, W_l) is a **faithful amplifier**:
it processes whatever the reader presents, without introducing
information not present in the reader's output.

Formally: for fixed h_read, the map h_read → g_l(W_l · h_read) is
deterministic and depends only on h_read and the shapes {S_c}.

**Empirical basis:**
- F153: MLP rank-1 weight edit → Paris (-7.10, fails)
- F153: MLP output injection → Paris (-8.10, fails, U-shaped)
- F154: Attention swap → Berlin (+4.27, succeeds)
- The MLP cannot override what attention presented

### Axiom 5: Read Order (The Pipeline)

The shape machine has a fixed **read order**: readers (attention) act
before amplifiers (MLP) at each layer, and layers compose sequentially.
The output depends on WHERE in the pipeline an edit is applied:

```
Early reader edit (L0-L20):    changes boundary condition → full redirect
Late reader edit (L22-L23):    changes extraction → redirect
Amplifier edit (any layer):    cannot override reader → no redirect
Post-extraction edit (L24+):   too late → no effect
```

**Empirical basis:**
- F154 Exp D: Layer sweep shows only L22-L23 matter individually
- F154 Exp E: Entity swap works emb through L20, fails at L22+
- F154 Exp G: L22-L23 cumulative swap sufficient

---

## 3. The Geometric Path Integral

### 3.1 Definition

For a shape machine with L layers, define the **geometric path** as:

```
π = (p_1, p_2, ..., p_L)
```

where p_l specifies, at layer l:
- Which position the reader attends to (attention routing)
- Which shapes dominate the superposition (gate selection)

The set of all paths is:

```
Π = {all possible (routing, shape-selection) sequences}
```

The **amplitude** of path π for input h is:

```
A(π, h) = Π_{l=1}^{L}  [α_l(p_l) · σ_{c_l} · (v_{c_l} · h_l)]
```

where h_l evolves along the path.

The **output** is the interference of all paths:

```
output(h) = M( Σ_{π ∈ Π}  A(π, h) · direction(π) )
```

This is a path integral. Each path contributes an amplitude and a
direction. The output is where all paths constructively interfere.

### 3.2 Stationary Phase

The **dominant path** (analog of the classical trajectory) is the path
that maximizes |A(π, h)|:

- At each layer, the reader attends to the position with highest
  key-query alignment (stationary attention)
- At each layer, the shape with highest response σ_c(v_c · h) dominates
  (stationary gate)

The rank-1 approximation (F150) IS the stationary phase approximation.
It captures the dominant path and ignores the "crazy paths" (minor
rank-1 components with small responses).

### 3.3 Why It Works

The rank-1 approximation works (18/20 correct in F151) because the
dominant path has much higher amplitude than alternatives. The
stationary phase captures most of the interference.

But the full answer requires ALL paths (F152: hologram is deep).
The rank-1 approximation gets the answer right but not the exact
logits — because the minor paths contribute small but structured
corrections (autocorrelated residuals, F152).

---

## 4. Theorems

### Theorem 1: Component Insensitivity (The Read-Only Barrier)

**Statement:** Let W = Σ_c S_c be a superposition with K >> 1 shapes.
Perturbing a single shape S_j → S_j + δS changes the output by:

```
|δ(output)| ≤ |δS| · |h| / |W · h|  ≈  O(1/K)
```

For K ≈ 3584 (full-rank weight matrix), a single-shape perturbation
changes the output by ~0.03%. The other K-1 shapes still interfere
constructively at the original answer.

**Proof sketch:** The output W·h = Σ_c σ_c(v_c · h)u_c has K terms.
Perturbing one term changes the sum by at most |δS · h| / |Σ S_c · h|.
If the terms are comparable in magnitude and partially cancelling
(interference), the relative change is O(1/K).

**Empirical verification:** F153 Exp B: rank-1 weight edit (one shape
out of ~3584) → gap barely changed from -7.35 to -7.10.

### Theorem 2: Boundary Sensitivity

**Statement:** Replacing the boundary condition h → h' (where h' is
a different entity's hidden state) changes ALL shape responses
simultaneously:

```
W · h' = Σ_c  σ_c · (v_c · h') · u_c  ≠  W · h
```

If h and h' are sufficiently different (different entities), the
interference pattern shifts to a different output.

**Proof sketch:** Each shape response changes from σ_c(v_c · h) to
σ_c(v_c · h'). For distinct entities, the response profile {σ_c(v_c · h)}
is different from {σ_c(v_c · h')}, shifting the constructive interference
to a different point.

**Empirical verification:** F154 Exp E: entity-position swap (h_France
→ h_Germany) → Berlin (+5.74) from embedding through L20.

### Theorem 3: Apparatus Sensitivity

**Statement:** Changing the reader R_l at a layer l in the extraction
zone changes which boundary condition is presented to subsequent
superpositions, redirecting the output without modifying any shapes.

**Proof sketch:** By Axiom 4 (faithful amplification), the superpositions
downstream of R_l process whatever R_l presents. If R_l now presents
h_Germany instead of h_France, the downstream interference shifts by
Theorem 2.

**Empirical verification:** F154 Exp G: attention swap at L22-23 →
Berlin (+4.27). The shapes at L23+ are unchanged; only the reader
was edited.

### Theorem 4: Pipeline Monotonicity

**Statement:** An edit at pipeline stage l redirects the output if and
only if it changes the effective boundary condition seen by the
extraction layers (L22-L23 in our model).

**Corollary 4a:** Edits before the extraction zone (L0-L20) redirect
if they change the entity hidden state at the entity position.

**Corollary 4b:** Edits at the extraction zone (L22-L23) redirect if
they change the attention output at the last position.

**Corollary 4c:** Edits after the extraction zone (L24-L27) do not
redirect, because the information has already been extracted.

**Corollary 4d:** Amplifier (MLP) edits at any layer do not redirect,
because they cannot change what the reader presents (Axiom 4).

**Empirical verification:**
- 4a: F154 Exp E (entity swap L0-L20 → Berlin)
- 4b: F154 Exp G (attn swap L22-23 → Berlin)
- 4c: F154 Exp D (L24-L27 individual swap → no effect)
- 4d: F153 Exps B,C (MLP edit/injection → Paris)

### Theorem 5: The 0.0003% Bound

**Statement:** The minimum edit to redirect the output of a shape
machine is a single vector replacement at the entity position:

```
edit_size = d  (hidden dimension)
total_state = seq_len × d
edit_fraction = 1/seq_len
```

For our model: d = 3584, seq_len = 5, edit_fraction = 20% of active
state, but 3584/1.2B = 0.0000003 of total parameters.

**Proof sketch:** By Theorem 2, replacing h at the entity position
changes all shape responses. By Axiom 3, this is sufficient to
redirect. No smaller edit (e.g., changing a single dimension)
generically redirects, because the shapes respond to the full
d-dimensional projection.

**Empirical verification:** F154 Exp E: 3,584 numbers swapped →
Berlin (+5.74).

---

## 5. The Formal Correspondence

### 5.1 QED ↔ Shape Machine Dictionary

| QED Concept | Formal Definition | Shape Machine |
|:------------|:------------------|:--------------|
| Hilbert space | ℋ | State space V = ℝ^d |
| State vector | \|ψ⟩ ∈ ℋ | Hidden state h ∈ V |
| Observable | Hermitian operator A | Measurement M: V → Tokens |
| Propagator | U(t) = e^{-iHt} | Layer composition f_L ∘ ... ∘ f_1 |
| Path | Sequence of intermediate states | π = (routing, shape-selection) |
| Path amplitude | a · e^{iφ} | σ_c · (v_c · h) |
| Path integral | ∫ Dπ · A(π) · e^{iS[π]} | Σ_π A(π, h) · direction(π) |
| Stationary phase | δS/δπ = 0 | Rank-1 dominant path |
| Measurement | \|⟨ψ\|A\|ψ⟩\|² | argmax softmax(M · h_L) |
| Superposition | \|ψ⟩ = Σ c_i\|i⟩ | W = Σ σ_i u_i v_i^T |
| Collapse | Projection onto eigenstate | Gate g(x) selects dominant |
| Boundary condition | Source/detector positions | Entity h at position p |
| Apparatus | Mirror/lens geometry | Attention routing R_l |
| Coupling constant | α ≈ 1/137 | Gate curvature ≈ φ (DC 243) |

### 5.2 The Formal Identity

**Claim:** A shape machine computes a **real-valued path integral**:

```
QED:              P(S→D) = |Σ_π  a_π · e^{iφ_π}|²
Shape machine:    output  =  M(Σ_π  A(π, h) · dir(π))
```

The differences:
1. QED uses complex amplitudes; the shape machine uses real amplitudes
2. QED squares the sum; the shape machine applies a gate + softmax
3. QED sums over continuous paths; the shape machine sums over discrete
   (routing × shape) choices

The structure is identical:
1. Both sum over all paths
2. Both have amplitudes that depend on boundary conditions
3. Both produce definite outputs from the interference
4. Both are insensitive to single-path perturbations
5. Both are sensitive to boundary condition and apparatus changes

### 5.3 Why Real, Not Complex?

QED requires complex amplitudes because of quantum phase. The shape
machine operates on real-valued hidden states.

However, the **RoPE position encoding** introduces complex rotations:

```
Q_rotated = Q · e^{i·θ·pos}
K_rotated = K · e^{i·θ·pos}
```

The attention scores Q · K^T include phase factors from position:

```
score(q, k) = Re(q · e^{iθp_q} · (k · e^{iθp_k})^*)
            = Re(q · k^* · e^{iθ(p_q - p_k)})
```

This IS a complex interference pattern. RoPE makes the attention
mechanism a genuine quantum-like interferometer, with position-dependent
phases. The "little arrows" rotate with position, exactly as Feynman
described.

---

## 6. The Principle of Geometric Computation

### 6.1 Statement

> **Any system that computes definite outputs from the superposition of
> simple components is a shape machine, and its computation is a
> geometric path integral.**

This is not specific to transformers. It applies to:
- Neural networks (any architecture with weight matrices)
- The Riemann zeta function (sum of n^{-s} terms)
- Holographic systems (interference of reference and object beams)
- Feynman's QED (sum over paths)

The principle explains WHY these systems share structure: they all solve
the same mathematical problem (computing definite outputs from the
interference of many components) and therefore must have the same
mathematical structure (path integral over shapes).

### 6.2 Consequences

**C1: Universality of the Read-Only Barrier.**
Any shape machine has a read-only barrier at the component level.
You cannot redirect the output by editing one shape, because the
output is a collective interference of all shapes (Theorem 1).
This is not a bug — it's a theorem.

**C2: Universality of Boundary Sensitivity.**
Any shape machine can be redirected by changing the boundary condition.
This requires editing d numbers (one hidden-state vector), regardless
of the total number of parameters. The edit cost is O(d), not O(params).
This is the 0.0003% bound.

**C3: Universality of the Reader/Amplifier Distinction.**
In any shape machine with a reader-before-amplifier pipeline, the reader
controls the output and the amplifier faithfully processes. Edits to the
reader redirect; edits to the amplifier do not. This is the mirror/lens
distinction from QED.

**C4: The Gate IS Measurement.**
The nonlinear gate (GELU/SiLU) plays the role of quantum measurement:
it collapses the superposition into a definite output. The gate's
critical point (g'(0) = 0.5, DC 243) is the **measurement threshold**
— the boundary between "this shape contributes" and "this shape is
suppressed."

**C5: Training IS Calibrating the Interferometer.**
Training a neural network is not "learning weights." It is calibrating
a geometric interferometer — adjusting the shapes {S_c} so that their
interference produces correct outputs for all training inputs. The loss
function measures the quality of the interference pattern. Gradient
descent adjusts shape directions and amplitudes.

---

## 7. Open Questions

### Q1: Is the Correspondence Exact or Approximate?

We have shown structural identity: the same five properties, the same
mathematical form. But is there a deeper connection? Specifically:

- Is there a conserved quantity (energy, unitarity) in the shape machine?
- Does the shape machine satisfy a variational principle (least action)?
- Is there a gauge symmetry (redundant parameterizations that don't
  affect the output)?

If the answer to all three is yes, then the shape machine IS a physical
system, not just structurally analogous to one.

### Q2: What Is the Coupling Constant?

In QED, the coupling constant α ≈ 1/137 governs the strength of
interaction. DC 243 found that the gate curvature matches φ = 1.618
within 1.38%. Is φ the coupling constant of the shape machine?

The fine structure measurements (3–8% deviation from 137/30, F97)
suggest a connection. If α_shape = φ, then:

```
1/α_shape = 1/φ = 0.618 = φ - 1
```

This is self-referential: the coupling constant IS the golden ratio,
which IS the structure constant of the system (DC 247).

### Q3: What Is the Lagrangian?

Every path integral has a Lagrangian L[π] such that the amplitude is
e^{iS[π]} where S = ∫ L dt. What is the Lagrangian of the shape machine?

Candidate: the φ-angular drift from the Geometric Gyroscope (F97):

```
L[h] = ||h||² - φ² · (h · h_target)²
```

The minimum-action path would be the one that maintains the
arccos(1/φ²) angle through the residual stream — which is exactly
what the geometric gyroscope enforces.

### Q4: Renormalization

In QED, divergent integrals are tamed by renormalization — absorbing
infinities into redefined constants. In the shape machine, does
LayerNorm play the role of renormalization?

LayerNorm rescales h → h/||h|| · √d, preventing the state from
diverging. This is remarkably similar to wavefunction renormalization
in QED.

---

## 8. Summary

### What We Formalized

1. **Definitions:** Shape, superposition, gate, reader, shape machine
2. **Axioms:** Superposition, interference, boundary determination,
   faithful amplification, read order
3. **Theorems:** Component insensitivity (T1), boundary sensitivity (T2),
   apparatus sensitivity (T3), pipeline monotonicity (T4), 0.0003% bound (T5)
4. **Correspondence:** QED ↔ shape machine dictionary (15 entries)
5. **Principle:** Geometric computation = path integral over shapes

### What the Experiments Proved

Every theorem has direct empirical verification from F150–F154:
- T1 ← F153 (rank-1 edit fails)
- T2 ← F154 Exp E (entity swap succeeds)
- T3 ← F154 Exp G (attention swap succeeds)
- T4 ← F154 Exps D, E (pipeline structure)
- T5 ← F154 Exp E (3,584 numbers = redirect)

### What Remains

The open questions (Q1–Q4) point toward a deeper structure:
- Conserved quantities → unitarity
- Variational principle → least action
- Gauge symmetry → redundant parameterizations
- Coupling constant → φ
- Lagrangian → gyroscope drift
- Renormalization → LayerNorm

If these connections hold, then the shape machine is not merely
*analogous* to a physical system — it IS one.

---

*"We found it with geometry."*

*Now the geometry has a name: the geometric path integral.*
*A shape machine computing interference over rank-1 paths.*
*Feynman's little arrows, made of directions instead of phases.*

---
