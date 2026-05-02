# Design Consideration 271: The Expanding Tensor

## The Discovery

While building a geometric zeta zero hunter (F112-113), we found that the
Riemann-Siegel formula isn't just a computational tool — it's a window into
how the **ideal transformer** works. The zeta function is a tensor that
literally grows with time, and its structure maps exactly onto the
transformer architecture.

```
Z(t) = 2 Σ_{n=1}^{N(t)} n^{-1/2} cos(θ(t) - t·ln(n))

where N(t) = floor(√(t/2π))
```

Each term in this sum is a **rotation**. The number of rotations increases
with height t. The tensor expands.

---

## 1. The Tensor Structure

### 1.1 Dimensions

The zeta function on the critical line has four geometric dimensions plus time:

| Axis | Meaning | Curvature |
|------|---------|-----------|
| **n** (term index) | Which rotation axis | Dirichlet: amplitude ∝ n^{-1/2} |
| **θ(t)** (global phase) | Position on the manifold | Log-warped: θ = Im(log Γ) |
| **t·ln(n)** (local phase) | Angle of each rotation | Prime-frequency: ln(2), ln(3), ln(5), ... |
| **amplitude** | Strength of each term | φ-scaled: n^{-1/2} ≈ φ^{-ln(n)/ln(φ²)} |
| **t** (height) | Time — the tensor expands | Density grows as ln(t/2π)/2π |

Every axis is curved. None are flat. The log-warping of θ(t) is exactly
the φ-warped manifold from DC 048. The amplitude decay n^{-1/2} is exactly
the Dirichlet series structure from F107.

### 1.2 The Expansion

```
     t      N_terms   spacing   density    tensor_size
    14.0        1      7.84      0.13      1 rotation
    50.0        2      3.03      0.33      2 rotations
   100.0        3      2.27      0.44      3 rotations
   500.0        8      1.44      0.70      8 rotations
  1000.0       12      1.24      0.81      12 rotations
  5000.0       28      0.94      1.06      28 rotations
 10000.0       39      0.85      1.17      39 rotations
```

At t = 14.13 (first zero), the tensor has **one** rotation. Z(t) = 0 means
that single rotation lands on a null. Trivial.

At t = 10000 (zero #~10000), the tensor has **39** rotations. Z(t) = 0
means 39 rotations conspire to cancel. This is a collective phenomenon —
the geometric analog of attention across 39 "positions."

N_terms grows as √(t/2π). The tensor gains one rotation axis every time
t passes the next perfect square × 2π:

```
t = 2π·1² → N = 1
t = 2π·2² → N = 2
t = 2π·3² → N = 3
...
t = 2π·k² → N = k
```

---

## 2. The Transformer Mapping

### 2.1 Each Term IS a Token

In a transformer, attention operates over a sequence of tokens.
In the zeta tensor, the Riemann-Siegel sum operates over a sequence
of terms. The mapping is exact:

| Zeta Term | Transformer Token |
|-----------|-------------------|
| n^{-1/2} | Token embedding (amplitude) |
| θ(t) - t·ln(n) | Positional encoding (phase) |
| cos(...) | Attention score projection (real part) |
| Σ over n | Attention aggregation (sum) |
| N(t) terms | Context window length |

The context window **grows** with t — exactly as a transformer's context
window grows with sequence length. But in the zeta case, the growth is
geometric (√t), not linear.

### 2.2 The Phase IS Position Encoding

Each term's phase is θ(t) - t·ln(n). The t·ln(n) part depends on BOTH
the "position" t and the "token identity" n. This is exactly what RoPE
does — it entangles position with content:

```
RoPE:  phase = position × frequency_i
Zeta:  phase = t × ln(n)
```

The frequencies in RoPE are φ-geometric: freq_i = φ^{-i × 0.4486} (F88).
The frequencies in zeta are prime-logarithmic: freq_n = ln(n).

Both are **non-uniform** frequency ladders on curved axes. RoPE uses
the φ-ladder. Zeta uses the prime-ladder. The structure is the same;
only the specific curvature differs.

### 2.3 The Zero IS the Output

A transformer produces output by finding where the accumulated
information "points." A zeta zero is where the accumulated rotations
cancel — where the tensor's net output is zero.

```
Transformer: output = argmax(Σ attention-weighted values)
Zeta:        zero   = where(Σ amplitude-weighted rotations = 0)
```

Both are finding a specific point on a manifold by aggregating
contributions from all positions/terms.

---

## 3. Static vs Dynamic: The K = 0 Insight

From F112, we know that the deformation kernel K determines how much
"attention" a problem needs:

```
ζ (ideal):        K = 0     → no deformation → static manifold
Modular arith:    K = rank-1 → one "head"     → bilinear kernel
Language:         K = rank-r → r heads         → learned kernel
```

For the zeta function, K = 0. The manifold doesn't deform with input
because the "input" (the index n) maps directly to the manifold
coordinate via the smooth counting function θ(T)/π + 1 = n.

This means: **the zeta function is the computation that needs no
attention**. The reference geometry IS the answer. Every other
computation is a deformation of this reference, and the rank of
that deformation determines the computational complexity.

This is why ζ is the *ideal* transformer — it's the transformer
with perfect geometry, where attention would be wasted because
the manifold is already aligned with the answer.

---

## 4. The Three Stages on the Expanding Tensor

The zero-hunting pipeline (F113) maps onto the expanding tensor:

### Stage 1: Compressor (Lambert W)

```
n → t ≈ 2π(n - 7/8) / W((n - 7/8)/e)
```

This inverts the **global** structure of the tensor — the smooth counting
function that tells us roughly where we are on the manifold. It ignores
the individual rotations entirely. Like the transformer's early layers
(L0-3, DRUM), it captures >95% of the answer from the global shape alone.

### Stage 2: Processor (Ramanujan Refinement)

```
Newton iteration: t_{k+1} = t_k + (n - N_smooth(t_k)) / N'_smooth(t_k)
```

This refines the global coordinate by iterating on the exact smooth
counting function θ(T)/π + 1. Like the transformer's middle layers
(L4-25, COMB), it performs oscillatory corrections that converge
conditionally. The Processor doesn't evaluate Z(t) — it works entirely
in the smooth geometry.

### Stage 3: Targeter (Z(t) + Newton)

```
Z(t) = 2 Σ rotations → find sign change → bisect → Newton polish
```

This is where the **full tensor** is evaluated for the first time.
All N(t) rotations contribute. The Targeter finds where they cancel.
Like the transformer's final layers (L26-27, FIRE), it makes a single
precision correction using the complete information.

The key insight: **Stages 1 and 2 never see the individual rotations.**
They work entirely on the manifold geometry. Only Stage 3 evaluates
the tensor. This matches the transformer exactly — the early and middle
layers build representation (geometry), and only the final layers
produce the output (evaluate).

---

## 5. Why the Tensor Expands

The expansion N(t) = floor(√(t/2π)) is not arbitrary. It comes from
the saddle-point approximation of the functional equation:

```
ζ(s) = χ(s) · ζ(1-s)
```

The Riemann-Siegel formula splits the Dirichlet series at the saddle
point n = √(t/2π). Below the saddle, terms contribute to the main sum.
Above the saddle, their contribution is captured by the remainder.

In transformer terms: the saddle point determines the **effective
context window**. At low t (early in the sequence), few terms matter —
the tensor is small, the computation is simple. At high t (deep in
the sequence), many terms contribute — the tensor is large, the
computation requires aggregating many rotations.

This is exactly what transformers do:
- Short sequences: simple attention patterns, few effective positions
- Long sequences: complex attention patterns, many positions contributing

The √t growth is optimal — it's the balance point between the
Dirichlet series (growing) and the functional equation symmetry
(reflecting). Any other growth rate would either miss terms or
double-count them.

---

## 6. The Prime Frequency Ladder

Each rotation in the tensor operates at frequency ln(n). For prime n,
these frequencies are **independent** — they form a basis. For composite
n, the frequency ln(ab) = ln(a) + ln(b) is a sum of prime frequencies.

This means the tensor has a natural **factorization structure**:

```
Prime terms:     fundamental rotations (basis vectors)
Composite terms: combinations of prime rotations
```

The Euler product ζ(s) = Π_p (1 - p^{-s})^{-1} expresses this directly:
the zeta function is a product over primes, each contributing an
independent geometric factor.

In transformer terms: the prime frequencies are like the **attention
heads** — independent channels that each capture a different aspect
of the structure. The composite frequencies are like **multi-head
combinations** that emerge from the interaction of heads.

The number of primes below N(t) grows as N(t)/ln(N(t)) ≈ √(t/2π) / ln(√(t/2π)).
These are the independent "heads." The rest are compositions.

---

## 7. Implications

### 7.1 Attention IS Curvature

F112 showed that the deformation kernel K(a,b) = ab/(φ² + φ(a+b)) IS
attention — bilinear interaction with φ-normalization, derived from
manifold curvature. The expanding tensor shows WHY:

On a **flat** manifold, rotations don't interact — each term is
independent, and the sum is trivial. On a **curved** manifold, the
terms interact through the curvature, creating the bilinear kernel
that we call attention.

For ζ (K = 0): the manifold is perfectly curved so that the rotations
cancel at exactly the right places — the zeros. No additional curvature
(attention) is needed.

For other problems (K ≠ 0): the manifold must be deformed to align
the rotations with the desired output. The deformation IS attention.
Its rank determines how many independent deformations (heads) are needed.

### 7.2 The Transformer's Hidden Scaling Law

The expanding tensor predicts a scaling relationship:

```
Effective attention ∝ √(sequence_length)
```

At length L, the "natural" number of contributing terms is √(L/2π).
This suggests that the effective rank of attention should grow as √L,
not L. Transformers that achieve this scaling would be "zeta-optimal."

### 7.3 ζ as the Universal Reference

Every computational problem can be characterized by its deformation
from the zeta reference:

```
Problem complexity = rank of K = rank of deformation from ζ
```

This gives a geometric complexity theory:
- rank 0: trivially on the manifold (zeta zeros, simple lookups)
- rank 1: one-dimensional deformation (modular arithmetic, factual QA)
- rank r: r-dimensional deformation (language, reasoning)
- rank ∞: incomputable (the manifold can't be deformed to fit)

---

## 8. Connection to Prior Work

| Document | Connection |
|----------|-----------|
| DC 048 | Curved arithmetic axis = the φ-warped manifold that θ(t) follows |
| DC 124 | φ-transformer replacement = deformation kernel at rank 1 |
| DC 160 | Unified geometric theory = the expanding tensor IS the unified structure |
| Doc 270 | ζ as ideal transformer = the conceptual proof this DC makes concrete |
| F107 | Three-stage pipeline = Compressor/Processor/Targeter on the tensor |
| F108 | Lambert W captures 95% = Stage 1 reads the global tensor shape |
| F111 | Darwin II recipe = the architecture that creates this tensor |
| F112 | Deformation kernel = K = 0 for ζ, rank-1 for mod arith |
| F113 | Zero hunter = the pipeline that navigates the expanding tensor |

---

## 9. Summary

The zeta function is a tensor that expands with time. Each Riemann-Siegel
term is a rotation axis. The number of axes grows as √(t/2π). The zero
is where all rotations cancel.

This expanding tensor IS the ideal transformer:
- Terms = tokens
- Phases = position encodings (RoPE analog)
- Amplitudes = embeddings
- Zero = output
- Expansion = growing context window

The three-stage pipeline navigates this tensor:
- **Compressor** reads the global shape (95% accuracy)
- **Processor** refines the smooth geometry (machine precision)
- **Targeter** evaluates the full tensor (exact zero)

And the deformation kernel K measures how far any problem is from
this ideal: K = 0 for ζ, K = rank-1 for modular arithmetic,
K = rank-r for language.

The tensor doesn't just expand — it expands in a way that encodes
the optimal structure for information processing. That structure is
the zeta function. That structure is what transformers learn.

---

*References: F107-113, DC 048, DC 124, DC 160, Doc 270*
*Empirical basis: phase10z9 series (mod arith, 100%), phase10z10 (zero hunter, 100/100)*
