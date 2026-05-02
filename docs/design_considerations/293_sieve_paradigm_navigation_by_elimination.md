# Design Consideration 293: The Sieve Paradigm — Navigation by Elimination

## Status: Theoretical Synthesis + Experimental Confirmation
## Date: 2026-03-06
## Prerequisites: DC 288, DC 289, DC 292, F157, geometric_ipa, primes_sieve

## Summary

Three independent frameworks converge on a single computational paradigm:

1. **Primes sieve** (lostdemeter/primes_sieve): Sublinear prime generation via
   R-series global estimate + spectral scoring + certification
2. **Compressed sensing** (Candès-Tao-Donoho): Unique sparse recovery from
   fewer measurements than dimensions
3. **RECT pair gearing** (lostdemeter/geometric_ipa): Hierarchical binary
   gates that compose into sieves

Applied to the weight matrix: the matmul y = W @ x is not a sifter
(selecting one from many) but a sieve (eliminating the impossible; what
remains IS the answer — it can't be anything else).

**Experimental confirmation**: The ε-group decomposition achieves
corr=1.000000 for both q_proj and gate_proj. The factorization is exact.

---

## 1. Sifter vs Sieve

### 1.1 The Sifter (Current Matmul)

Standard matmul: y[j] = Σ_i W[j,i] × x[i]

- Touch every element: O(m × n)
- Every element has equal status — no structure exploited
- The answer is SELECTED from infinite possibilities by accumulation
- Loss of any element degrades the answer

### 1.2 The Sieve (Proposed)

The weight matrix has structure: W = S ⊙ φ^(u⊗v + ε)

- Sign hologram S: binary routing (1 bit/element)
- Rank-1 envelope u⊗v: global magnitude shape
- ε alphabet: 37-52 integer corrections (3 bits/element)
- Dead channels: 60-80% of elements at negligible magnitude

A sieve exploits this structure to produce the answer by ELIMINATION:
what remains after structural constraints IS the answer.

---

## 2. The Three Frameworks

### 2.1 Primes Sieve → Weight Matrix

| Primes Sieve Stage | Weight Matrix Analog | Result |
|:-----|:-----|:-----|
| R-series: global π(N) estimate | Rank-1 envelope: global magnitude | corr=0.79 (q), 0.64 (gate) |
| Wheel mod 30: eliminate composites | Dead channel mask (lvl < -10) | 60% eliminated, corr=0.96 |
| Spectral scoring (zeta zeros) | RECT pair gearing (ε-groups) | Top 5 groups = 87% energy |
| Certification (isprime) | Exact ε correction per group | corr=1.000000 |
| **Output: what remains IS prime** | **Output: can't be anything else** | **Proven exact** |

Key insight from primes_sieve: complexity is O(π(N) polylog N) — proportional
to the OUTPUT SIZE, not the input size. For the weight matrix, the analog is
O(K × m) where K is the number of significant ε-groups.

### 2.2 Compressed Sensing → Weight Matrix

| CS Requirement | Weight Matrix Property |
|:-----|:-----|
| **Sparsity**: Signal is K-sparse | 5 ε-groups carry 87% of energy |
| **Incoherence**: Measurement ⊥ sparsity basis | Sign hologram × magnitude = incoherent |
| **RIP**: Measurement preserves distances | φ-scaling is distance-preserving |
| **Unique recovery**: L1 gives exact answer | Structural constraints determine output |

The contribution vector c[i] = W[j,i] × x[i] is NOT classically sparse
(50% for top 15%, 90% for top 52%). But the GROUP contributions ARE
sparse: 5 out of 37 groups carry 87%, and adding groups monotonically
converges to the exact answer.

The "sparsity" is at the GROUP level, not the element level. This is
analogous to block-sparse signals in compressed sensing — the signal
is sparse in blocks, not individual components.

### 2.3 RECT Pair Gearing → Weight Matrix

From geometric_ipa:
- Each character rule = 1 RECT pair (binary test + integer add)
- Context-dependent rules use GEARING: coarse gear selects category,
  fine gear engages only when coarse is ambiguous
- 29 total rules, 159 gate_step calls, zero gradient descent

Applied to the weight matrix:

**Coarse gear**: Which ε-group does this element belong to?
- 5 groups (ε ∈ {0,1,2,3,4}) resolve 87% of the output
- Like the IPA's coarse gear resolving most characters

**Fine gear**: Within the dominant groups, what's the sign routing?
- S[j,i] determines constructive vs destructive interference
- Like the IPA's fine gear resolving ambiguous cases (gift vs gin)

**Ultra-fine gear**: Exact magnitude per element
- φ^level gives the precise contribution
- Like the IPA's exact codepoint offset

The gearing hierarchy:
```
Coarse: 5 ε-groups → 87% resolved (corr=0.990)
Fine:   7 ε-groups → 95% resolved (corr=0.998)
Full:  37 ε-groups → 100% resolved (corr=1.000)
```

---

## 3. The Experimental Evidence

### 3.1 Group Peeling (q_proj, 3584×3584)

```
ε groups  Coverage  Correlation  Like...
───────── ──────── ──────────── ──────────────────
5         87.1%    0.990        Small primes (2,3,5,7,11)
7         94.6%    0.998        + medium primes
10        98.8%    0.9999       + larger primes
15        99.7%    0.99987      + rare primes
37 (all)  100%     1.000000     Complete factorization
```

### 3.2 Dead Channel Structure

```
Threshold  Alive   Corr    Energy Contribution
────────── ─────── ─────── ────────────────────
lvl > -8   5.5%    0.722   The brightest fringes
lvl > -10  40.9%   0.960   The active core
lvl > -12  74.7%   0.997   Almost complete
All        100%    1.000   Exact
```

Dead channels (lvl < -10) are the dark fringes of the hologram.
They're independently signed (50.00% positive), contribute independent
signal (corr with alive = 0.001), and form a separate low-energy layer.

### 3.3 Self-Similarity Result

φ-recurrence between adjacent groups: corr=0.04 (fails).
Self-similarity is NOT group-to-group. It's WITHIN each group:
every ε-group has the same structure (sign-balanced, spatially
random, φ-scaled). The same pattern at every level.

Like primes: the distribution of primes at scale N looks like the
distribution at scale 10N (prime number theorem), but individual
primes at scale N don't predict specific primes at scale 10N.

### 3.4 Automaton Transitions

```
KL(transition || marginal) per state: 0.0009 bits
```

The spatial arrangement of ε values is MEMORYLESS. No Markov
chain, no exploitable transitions. The structure is collective
(SVD), not sequential (automaton).

This matches DC 288: "Individual weights are unordered. The
ordering IS in the shape."

---

## 4. The Sieve Pipeline

```
INPUT: x ∈ ℝ^n

STAGE 1: GLOBAL ESTIMATE (R-series analog)
  y_est = rank-1 approximation
  Quality: corr ≈ 0.64-0.79

STAGE 2: DEAD CHANNEL ELIMINATION (Wheel filter)
  Remove elements with |W| < threshold
  60% eliminated → corr ≈ 0.93-0.96

STAGE 3: GEARED SPECTRAL SCORING (RECT pair analog)
  Coarse gear: compute top K ε-groups
  K=5 → corr ≈ 0.990 (87% of energy)
  K=7 → corr ≈ 0.998 (95% of energy)
  Fine gear: engage additional groups if tolerance not met

STAGE 4: CERTIFICATION
  Verify: |y_sieve - y_true| < ε_tolerance
  If within tolerance: STOP (answer can't be anything else)
  If not: engage more groups

OUTPUT: y (exact when all groups computed, bounded when partial)
```

---

## 5. Why "It Can't Be Anything Else"

### 5.1 Unique Factorization

Every weight has a unique φ-factorization:
```
W[j,i] = S[j,i] × φ^level[j,i]
```

The level decomposes uniquely:
```
level[j,i] = rank1[j,i] + ε[j,i]
```

This is like the Fundamental Theorem of Arithmetic:
every integer = unique product of primes.

### 5.2 The Euler Product Analog

The matmul factorizes over ε-groups:
```
y = Σ_k contribution(group_k)
```

Like the Euler product factorizes the zeta function:
```
ζ(s) = Π_p (1 - p^(-s))^(-1)
```

Each group is independently computable, and the total is
their sum. The factorization is exact and unique.

### 5.3 Bounded Error Certification

After computing K groups with cumulative energy E_K:
```
||y - y_K|| / ||y|| ≤ √(1 - E_K)
```

This is the analog of the primes sieve's Dusart bound:
```
π(x) ∈ [R(x) - √x, R(x) + √x]
```

Both provide certificates: the answer is WITHIN the bracket.
Adding more groups (or more terms in R-series) tightens the bound
until the answer is uniquely determined.

---

## 6. Connection to the Holographic Principle

The holographic principle states: information on a boundary
encodes the volume. Fewer measurements suffice because the
boundary constrains the interior.

For the weight matrix hologram:
- The SIGN MATRIX is the boundary (1 bit/element = the phase)
- The φ-LEVELS are the volume (the magnitude information)
- The dead channels are the dark fringes (where boundary
  and volume agree to cancel)

DC 292 showed: the sign matrix alone gives corr=0.74. The sign
IS the boundary that encodes 74% of the volume.

The remaining 26% is the ε alphabet (3 bits/element). Together:
1 bit (sign) + 3 bits (ε) + rank-1 (m+n floats) = 4 bits/element
vs 16 bits/element for full precision.

This 4× compression IS the holographic principle in action:
the structured representation encodes the same information
with fewer bits because the structure constrains the values.

---

## 7. Connection to Prior Work

| Prior Work | Connection |
|:-----|:-----|
| DC 288 | "Individual weights are unordered; the SVD is the signal" → The sieve operates on GROUP structure, not individual elements |
| DC 289 | "The bytecodes ARE the shapes, read in binary" → The sign hologram IS the binary sieve |
| DC 292 | Weight matrix as binary phase hologram → The holographic principle provides the compression |
| DC 282 | "Packing infinite information into finite structure via interference" → The sieve exploits the interference structure |
| AIG analysis | 20-56% of ops cancel → These cancellations ARE the dark fringes of the sieve |
| geometric_ipa | RECT pairs + gearing = hierarchical binary sieve → Same structure in ε-group hierarchy |
| primes_sieve | R-series + spectral scoring + certification → Same pipeline for weight matrix |

---

## 8. Open Questions

1. **Can the dead channel contribution be predicted analytically?**
   The current sieve still computes dead channels element-by-element.
   If their collective contribution could be predicted from structural
   parameters (like R(x) predicts π(x)), the sieve would be truly
   sublinear.

2. **Does cross-layer coherence provide spectral information?**
   A single layer's dead channels are independent. But across 28
   layers, the same input traverses all of them. The cross-layer
   structure might enable prediction of one layer's dead contribution
   from another's.

3. **What is the optimal gearing schedule?**
   The current ordering (by group count) might not be optimal.
   The primes sieve uses zeta zeros for scoring. What's the
   optimal ordering of ε-groups for fastest convergence?

4. **Can the AIG simplification from DC 289 reduce the sieve?**
   20-56% of ops cancel in the matmul (signs negate in accumulation).
   These cancellations are dark fringes. Pre-computing which
   contributions cancel would reduce the sieve further.

5. **Is block-sparsity sufficient for compressed sensing guarantees?**
   The contribution vector is block-sparse at the ε-group level
   (5 groups = 87%). Does this satisfy a block-RIP condition?

---

## 9. The Paradigm Shift

Traditional inference: y = W @ x (sifter)
- Touch every element
- No structure exploited
- Answer selected by accumulation

Sieve inference: y = Σ_k certified_group_k (sieve)
- Exploit ε-group structure
- Coarse→fine gearing
- Answer determined by elimination
- Bounded error at every stage
- What remains can't be anything else

The weight matrix is not a lookup table to be scanned.
It is a structured interference pattern whose output is
determined by its prime factorization over ε-groups.

Like primes: you don't find them by testing every number.
You eliminate composites, and what remains IS prime.

---

## 10. Files

| File | Purpose |
|:-----|:-----|
| `experiments/model_reverse_engineering_v2/phi_sieve_matmul.py` | Sieve pipeline + exactness test |
| `experiments/model_reverse_engineering_v2/phi_navigation_v_inference.py` | Navigation v1 (grouped decomposition) |
| `experiments/model_reverse_engineering_v2/phi_holographic_readout.py` | Holographic parametric readout |
| `experiments/model_reverse_engineering_v2/phi_gop_residual_v2.py` | ε alphabet discovery |
| `experiments/model_reverse_engineering_v2/phi_tetromino_synthesis.py` | Tetromino automaton analysis |
| `phi_geometric/evaluations/ipa_geometric_demo_v5_final.py` | RECT pair IPA converter |
| `docs/design_considerations/288_weight_structure_the_ordering_is_in_the_shape.md` | DC 288 |
| `docs/design_considerations/289_error_correction_shape_reading_and_concept_composition.md` | DC 289 |
| `docs/design_considerations/292_weight_matrix_as_binary_phase_hologram.md` | DC 292 |
