# DC 273: Geometric Memory Editing

**Status**: Active — updating as experiments progress
**Date**: 2025-02-27
**Depends on**: DC 271 (The Expanding Tensor), DC 272 (The Transformer IS a Riemann-Siegel Sum)
**Findings**: F114–F117

---

## 1. The Discovery

Finding 117 demonstrated **complete fact replacement** in Qwen2-7B by
manipulating V·W_o attention outputs across layers. No retraining. No
gradient descent. Just vector arithmetic in the residual stream.

```
"The capital of Japan is" + France_deltas → ' Paris' (rank 0)
                                             ' Tokyo' (rank 434)
```

This is not fine-tuning. This is **surgery**.

---

## 2. What We Know

### 2.1 Facts Are Distributed Sums

A fact is not stored in one layer. It is the sum of 28 V·W_o terms:

```
Answer = Σ_{L=0}^{27} V·W_o_L(prompt)
```

Each layer contributes one term. The terms are **additive** and
**independently modifiable**.

### 2.2 Five Layers Dominate

Of the 28 terms, ~5 carry most of the fact-specific signal:

| Layer | Role | Delta magnitude |
|:------|:-----|:----------------|
| L22 | Fact encoding (largest delta) | 14–20 |
| L23 | Fact routing (d_k selector, F39–40) | 17–19 |
| L25 | Fact reinforcement | 16–17 |
| L27 | Final refinement | 15–18 |
| L9 | Early fact signal | 9–12 |

Late layers (21–27) alone are sufficient for complete fact swap.
Early layers alone are not.

### 2.3 The R-S Sum Structure

This matches the Riemann-Siegel partial sum (DC 272):

```
ζ(s) ≈ Σ_{n=1}^{N} n^{-s}    (N ≈ √(t/2π))
```

A few large terms dominate. The rest are corrections. In the transformer:
- **Dominant terms** (L22, L23, L25, L27): determine the answer
- **Correction terms** (other layers): refine confidence and suppress alternatives
- **Early terms** (L0–L9): set up the context (which fact is being asked)

### 2.4 Orthogonality Enables Selectivity

F115 showed V·W_o vectors for different facts are near-orthogonal.
This means modifying one fact's vector does NOT interfere with others:

```
⟨V·W_o(France), V·W_o(Japan)⟩ ≈ 0
```

Implication: you can edit France→Paris without touching Japan→Tokyo.

---

## 3. Operations

### 3.1 EDIT (Swap)

Replace fact A with fact B:
```
For each layer L in {dominant layers}:
    delta_L = attn_B[L] - attn_A[L]    (at last token position)
    attention_output[L][-1] += delta_L
```

**Demonstrated**: France→Japan swap. Paris=rank 0, Tokyo=rank 434.
5 layers sufficient.

### 3.2 REMOVE (Forget)

Zero out a fact's contribution:
```
For each layer L in {dominant layers}:
    attention_output[L][-1] -= attn_A[L]
```

**Demonstrated (single layer)**: Removing L23 attention drops Paris from
rank 0 to rank 1. Multi-layer removal should push it much further.

### 3.3 ADD (Novel Memory)

Inject a fact that was never in training data:
```
For each layer L in {dominant layers}:
    Construct synthetic V·W_o vector that projects to target token
    attention_output[L][-1] += synthetic_vector
```

**Status**: DEMONSTRATED (Phase 10z17). See Finding 118.

The challenge was: for EDIT and REMOVE, we have the model's own attention
outputs to work with. For ADD, we must **construct** a vector that the
model has never produced.

**Solution**: The LM head weight rows W_lm[k] ARE the hidden-space
directions for each token. Injecting them into the residual stream
creates a novel memory. No pseudoinverse needed — the forward direction
IS the inverse direction.

---

## 4. The Novel Memory Challenge

### 4.1 The Test

Inject the fact: **"NASA landed the first Tesla Model Y on Mars on
February 27, 2026."**

This fact:
- Cannot exist in training data (future date, fictional event)
- Requires multi-token understanding (NASA, Tesla, Mars, date)
- Tests whether the geometric structure generalizes to novel content

### 4.2 Approaches Tested

**Option A: Transfer from known facts** — FAILED
Transplanting attention outputs from a NASA-related prompt imported the
donor's semantic context ('planning', 'also', 'now') instead of the
target tokens. NASA rank actually got WORSE (13→693).

**Option B: LM head inverse (direct vocabulary targeting)** — **SUCCEEDED**
The LM head weight row W_lm[k] IS the hidden-space direction that
maximizes logit for token k. Injecting a normalized sum of target token
rows into attention outputs at every layer creates a novel memory.
Mars=rank 0, NASA=rank 2, landed=rank 1 across ALL 5 query phrasings.

**Option C: Compositional injection** — **SUCCEEDED**
Summing W_lm[NASA] + W_lm[Tesla] + W_lm[Mars] + W_lm[landed] and
injecting the normalized result puts all 4 tokens in the top 4
positions: Mars=0, Tesla=1, landed=2, NASA=3.

### 4.3 Success Criteria — MET

The model, when prompted with a query about February 27, 2026,
produces Mars (rank 0), NASA (rank 2), landed (rank 1) — content it
could not have learned from training.

### 4.4 Key Results

- **5/5 query phrasings**: all target tokens in top positions
- **2 layers sufficient**: L22+L23 alone achieve Mars=rank 0
- **Scale-invariant**: works from scale 0.5 to 100.0
- **Compositional**: 4 tokens injected simultaneously, all in top 4
- **No retraining**: pure vector arithmetic in the residual stream

---

## 5. Theoretical Framework

### 5.1 Why This Should Work

The Euler product structure (F114) tells us:

```
Knowledge = d_k (shared axis) × RoPE (frequency) × V·W_o (content)
```

- **d_k** is universal (F116b) — we don't need to change it
- **RoPE** is positional — it's determined by the prompt structure
- **V·W_o** is the ONLY fact-specific component

Therefore, to add a new fact, we ONLY need to provide the right V·W_o
vector. Everything else (routing, position encoding) is already in place.

### 5.2 Why It Works (Resolved)

The concerns were:

1. ~~Lie in the correct subspace~~ — The LM head rows ARE the subspace
2. ~~Project cleanly through RMS norm~~ — RMS norm is scale-invariant;
   it normalizes the direction, so the injection direction survives
3. ~~Not interfere with existing facts~~ — Works, but untested at scale
4. ~~Survive MLP/attention of subsequent layers~~ — The injection is
   applied at each layer's attention output, so each layer sees the
   delta fresh. No need to survive propagation.

### 5.3 The Inverse Problem — SOLVED

For known facts, the pipeline is:
```
prompt → attention → V·W_o → residual → ... → logits → token
```

For novel facts:
```
target_token → W_lm[token] → inject at attention output → residual → logits
```

No pseudoinverse needed. The LM head weight matrix W_lm ∈ ℝ^{vocab × hidden}
directly provides the hidden-space direction for each token. The forward
mapping (hidden→logit) and the injection direction (what to inject) are
the SAME vector — another instance of ENCODE = DECODE (see Core Philosophy).

---

## 6. Open Questions

1. ~~**Minimum edit set**~~ — PARTIALLY ANSWERED: 2 layers (L22+L23)
   are sufficient for novel memory. 5 layers for complete fact swap.

2. **Interference**: When we edit fact A, do nearby facts (e.g.,
   "capital of Frace" misspelling) also change?

3. **Persistence**: Can edits survive multiple forward passes (i.e.,
   does the model "remember" the edit for subsequent tokens)?
   This is critical for autoregressive generation.

4. **Compositional facts**: Can we inject multi-hop facts
   ("The capital of the country that borders Spain to the north is Paris")?

5. **Scale**: Does the 5-layer dominance pattern hold for non-factual
   knowledge (reasoning patterns, style, language)?

6. **Token ordering**: The current injection puts Mars first, not NASA.
   Can we control WHICH target token ranks highest?

7. **Multi-token generation**: Current results show first-token control.
   Can the injected memory persist across autoregressive generation?

---

## 7. Experiment Log

| Phase | What | Result |
|:------|:-----|:-------|
| 10z15 | Single-layer swap at L23 | Shifts rank by 10-65×, insufficient alone |
| 10z15 | Single-layer removal at L23 | Rank 0→1, fact persists |
| 10z15 | Single-layer injection (France→Spain) | Paris rank 103→2 |
| 10z16 | Multi-layer swap (all 28) | **Complete replacement** (Paris=0, Tokyo=434) |
| 10z16 | Multi-layer swap (top 5) | **Sufficient** (Paris=0, Tokyo=37) |
| 10z16 | Layer range: Late (21-27) | **Sufficient** for swap |
| 10z16 | Layer range: Early (0-6) | Insufficient alone |
| 10z17 | LM head inverse: NASA+Mars+landed | **Mars=0, NASA=2, landed=1** (5/5 queries) |
| 10z17 | Donor transfer (NASA prompt→date query) | FAILED (imported donor semantics) |
| 10z17b | 4-token injection (NASA+Tesla+Mars+landed) | **All 4 in top 4** (5/5 queries) |
| 10z17b | Layer ablation: L22+L23 only | **Mars=0, NASA=1** (2 layers sufficient) |
| 10z17b | Scale sweep: 0.5–100.0 | **Stable** (Mars=0, NASA=1 at all scales) |

---

*This document will be updated as experiments continue.*
