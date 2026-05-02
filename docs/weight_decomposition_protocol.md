# Weight Decomposition Protocol
## Reverse Engineering Transformer Sub-Machines into Geometric Data Structures

---

## Overview

This protocol converts a contiguous range of transformer layers into a **named
geometric data structure** with known interface, complexity, and classical CS
analogy. It was developed from the successful decomposition of the Targeter
(L26-27) into a φ-Filter (Doc 264, Finding 99).

The protocol is applied once per **sub-machine** — a contiguous range of layers
that share a gate medium and transfer function (Doc 262).

**Input**: Layer range, model weights, test prompts
**Output**: Named data structure, formal definition, prototype implementation

---

## The Seven Steps

```
┌──────────────────────────────────────────────────────────┐
│                WEIGHT DECOMPOSITION PROTOCOL              │
│                                                          │
│  ┌────────────────┐                                      │
│  │ 1. BOUNDARY    │  Confirm layer range, capture        │
│  │    LOCK        │  input/output hidden states           │
│  └──────┬─────────┘                                      │
│         │                                                │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ 2. GATE        │  Run 4-state classification          │
│  │    CENSUS      │  per layer, measure stability         │
│  └──────┬─────────┘                                      │
│         │                                                │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ 3. SIMPLE      │  Measure lever, damper, wedge,       │
│  │    MACHINES    │  spring per layer                     │
│  └──────┬─────────┘                                      │
│         │                                                │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ 4. INDEPENDENCE│  Test: can attention be approximated? │
│  │    TEST        │  Can FFN be approximated?             │
│  └──────┬─────────┘                                      │
│         │                                                │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ 5. TRANSFER    │  Classify: oscillatory, convergent,  │
│  │    FUNCTION    │  or step? Fit linear recurrence.      │
│  └──────┬─────────┘                                      │
│         │                                                │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ 6. SPARSE      │  Extract energy-carrying channels.   │
│  │    EXTRACTION  │  Build minimal weight set.            │
│  └──────┬─────────┘                                      │
│         │                                                │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ 7. PROTOTYPE   │  Build geometric replacement.        │
│  │    & NAME      │  Verify. Name the structure.          │
│  └──────────────────                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Step 1: Boundary Lock

**Purpose**: Confirm the sub-machine boundaries and capture its I/O signature.

**Process**:
1. Register forward hooks on the first and last layers of the range
2. Run 10+ diverse prompts through the full model
3. Capture hidden states at input (before first layer) and output (after last layer)
4. Compute per-prompt: norm change, angle change, cosine similarity in→out

**Script template**:
```python
captures = {}
def hook_in(mod, args, output):
    captures['in'] = output[0].detach().clone()
def hook_out(mod, args, output):
    captures['out'] = output[0].detach().clone()

model.model.layers[first_layer].register_forward_hook(hook_in)  # use pre-hook or capture layer[first-1] output
model.model.layers[last_layer].register_forward_hook(hook_out)
```

**Output**:
- Input/output hidden state pairs for each prompt
- Summary statistics: mean angle change, mean norm change
- Confirmation that boundaries match the expected sub-machine role

**Stopping criterion**: If angle change is near-zero or highly variable across
prompts, the boundary may be wrong. Re-examine with Doc 262's gate medium
transitions.

---

## Step 2: Gate Census

**Purpose**: Classify every FFN channel at every layer in the range into the
4-state gate system (Doc 253).

**Process**:
1. For each layer in range, hook `gate_proj` to capture pre-SiLU activations
2. Run 10+ prompts, collect gate activations for the LAST token position
3. Classify each channel by its MEAN activation:
   - EXPAND: mean > +log(φ) ≈ +0.481
   - PRESERVE+: 0 < mean < +log(φ)
   - PRESERVE-: -log(φ) < mean < 0
   - CONTRACT: mean < -log(φ) ≈ -0.481
4. Measure stability: fraction of prompts agreeing with mean classification

**Script template**:
```python
gate_acts = {li: [] for li in range(first, last+1)}
for prompt in prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    hooks = []
    acts = {}
    for li in range(first, last+1):
        def make_hook(idx):
            def hk(mod, inp, output):
                acts[idx] = output[0, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].mlp.gate_proj.register_forward_hook(make_hook(li)))
    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()
    for li in range(first, last+1):
        gate_acts[li].append(acts[li])
```

**Output per layer**:
```
L_i: EXPAND=N₁ (X%), PRESERVE=N₂ (Y%), CONTRACT=N₃ (Z%)
     Stability: EXPAND S_E%, CONTRACT S_C%
```

**Key indicators**:
- High CONTRACT% (>70%) + high stability (>80%) → φ-Filter candidate
- High PRESERVE% (>30%) → information-dense, routing-heavy layer
- Oscillating CONTRACT% across layers → medium transitions within the machine

---

## Step 3: Simple Machines

**Purpose**: Measure the four geometric simple machines at each layer (Doc 261).

For each layer, measure:

| Machine | Component | Metric |
|---------|-----------|--------|
| **Lever** | Attention | ‖attn_out‖ / ‖h_in‖ (magnification) |
| **Damper** | LayerNorm | ‖h_post_LN‖ / ‖h_pre_LN‖ (compression) |
| **Wedge** | FFN | ‖ffn_out‖ / ‖h_in‖ (force multiplication) |
| **Spring** | Residual | ‖residual‖ / ‖h_out‖ (dilution) |

Also measure **drift** (angle change per layer) and **cumulative angle**
from the first layer's input.

**Process**:
1. Hook each sublayer (input_layernorm, self_attn, post_attention_layernorm, mlp)
2. Run prompts, capture intermediate states
3. Compute ratios and angles

**Output per layer**:
```
L_i: lever=X.XX  damper=X.XX  wedge=X.XX  spring=X.XX  drift=X.X°  cum_angle=XX.X°
```

**Key indicators**:
- Dominant wedge → FFN-controlled (like Targeter)
- Dominant lever → attention-controlled (routing layer)
- Negative drift → overcorrection (oscillatory behavior)
- Monotonic drift → convergent behavior

---

## Step 4: Independence Test

**Purpose**: Determine which components (attention, FFN) can be approximated
or skipped without affecting the sub-machine's output.

**Tests** (run each independently):
1. **Skip attention**: Set attn_out = 0 for layers in range, measure top-1 accuracy
2. **Approximate attention**: Use bias-aware tables, measure accuracy
3. **Skip FFN**: Set ffn_out = 0 for layers in range (CAUTION: usually destructive)
4. **Sparse FFN**: Use only EXPAND channels, measure accuracy

**Process**:
1. Run full model → baseline logits
2. Capture hidden state at layer[first-1] output
3. Apply modified sub-machine (skip/approx/sparse variants)
4. Apply remaining layers (final LN + lm_head) → variant logits
5. Compare: top-1 match, cosine similarity, angle

**Output**:
```
Variant                    Top-1%   cos(logits)   Angle
Baseline                   100%     1.000         0.0°
Skip attn                  XX%      X.XXX         X.X°
Approx attn (bias-aware)   XX%      X.XXX         X.X°
Skip FFN                   XX%      X.XXX         X.X°
Sparse FFN (EXPAND only)   XX%      X.XXX         X.X°
```

**Decision tree**:
- Skip attn ≥ 90% → attention is irrelevant, focus on FFN
- Skip attn < 50% → attention is critical, focus on attention decomposition
- Sparse FFN ≥ 80% → φ-Filter structure, proceed to Step 6
- Sparse FFN < 50% → FFN needs all channels, different structure

---

## Step 5: Transfer Function

**Purpose**: Classify the sub-machine's input→output transformation.

**Process**:
1. Compute per-layer drift: Δθ_l = angle(h_l, h_{l-1})
2. Fit linear recurrence: drift(l+1) = α·drift(l) + β
3. Classify by α:
   - **α < 0**: Oscillatory (alternating correction). Seen in Compressor.
   - **0 < α < 1**: Convergent (steady approach to equilibrium). Seen in Processor.
   - **α ≈ 0, β large**: Step function (single strong correction). Seen in Targeter.
4. Compute equilibrium: drift_eq = β/(1-α) for convergent case

**Output**:
```
Transfer: [oscillatory|convergent|step]
Recurrence: drift(l+1) = α·drift(l) + β
  α = X.XXX, β = X.XXX
  Equilibrium = X.X° (if convergent)
```

**Key indicators**:
- Oscillatory + high CONTRACT → damping machine (compresses input)
- Convergent + high PRESERVE → routing machine (maintains equilibrium)
- Step + high EXPAND energy → targeting machine (aims output)

---

## Step 6: Sparse Extraction

**Purpose**: Identify the minimal set of weights that reproduces the
sub-machine's behavior.

**Process**:
1. From Step 2, identify channels by state. Focus on energy-carrying channels.
2. For EXPAND channels: extract gate_proj, up_proj, down_proj rows/columns
3. Measure energy concentration: what % of output energy is in top-K% of channels?
4. Sweep sparsity: test 5%, 10%, 20%, 50%, 100% of channels by energy rank
5. Find the knee: minimum channels for ≥90% accuracy

**Script template**:
```python
for pct in [0.05, 0.10, 0.20, 0.50, 1.00]:
    k = int(D_INT * pct)
    # Sort channels by |mean_gate| descending (most active first)
    top_k = mean_gate.abs().argsort(descending=True)[:k]
    # Build sparse weights, run, measure accuracy
```

**Output**:
```
Sparsity   Channels   Top-1%   cos     Compute
5%         947        XX%      X.XXX   5%
10%        1894       XX%      X.XXX   10%
20%        3789       XX%      X.XXX   20%
50%        9472       XX%      X.XXX   50%
100%       18944      XX%      X.XXX   100%
```

**Decision**:
- If there's a sharp knee (e.g., 5% → 73%, 10% → 90%) → sparse structure exists
- If accuracy scales linearly with channels → distributed, no sparse structure
- If accuracy is flat until 50%+ → dense computation, different structure needed

---

## Step 7: Prototype & Name

**Purpose**: Build the geometric replacement and formalize it.

**Process**:
1. Combine findings from Steps 1-6 into a geometric operation
2. Implement the prototype (use captured hidden states, apply geometric version)
3. Verify: top-1 accuracy, cosine similarity, angle deviation
4. If accuracy ≥ 90%: **name the structure**
5. Write design doc: formal definition, complexity, CS analogy, lay person analogy
6. If accuracy < 90%: identify the gap, return to Step 4 with modified variants

**Naming convention**:
- φ-Filter: sparse projection through EXPAND gated channels (Targeter L26-27)
- φ-Projector: attention-dominated embedding→hidden space map (Compressor L0)
- φ-Corrector: FFN-dominated negative zero leakage refinement (Compressor L1-3)
- φ-Lens: attention-dominated routing through equilibrium (Processor?)
- φ-[Name]: the pattern that emerges from the decomposition

**Output**:
- Named data structure with formal interface
- Design consideration document
- Finding in FINDINGS.md
- Prototype script in experiments/

---

## Step 8: Deep Dissection (Simple Machine Composition)

**Purpose**: Trace all 6 internal operations of each layer to understand how
the 4 simple machine types compose into the compound behavior.

**The 6 stages of a Qwen2 decoder layer**:
```
h_in → Damper1(RMSNorm) → Lever(Attn) → Spring1(+resid)
     → Damper2(RMSNorm) → Wedge(FFN) → Spring2(+resid) → h_out
```

**Measurements per stage** (last token):
1. Norm (absolute magnitude)
2. Norm ratio (||out|| / ||in|| for this stage)
3. Rotation (angle between this stage's input and output)
4. Cumulative angle from layer input h_in

**Key derived quantities**:
- **Rotation budget**: what % of total rotation comes from lever vs wedge?
- **Energy budget**: what fraction of output norm comes from input/lever/wedge?
- **Cross-correlations**: cos(attn_out, ffn_out), cos(input, attn_out), cos(input, ffn_out)
- **Projection decomposition**: project FFN output onto/perpendicular to h_in and h_mid
- **Per-head contributions**: decompose lever into individual head vectors via O-proj slicing

**Known compound machine patterns** (Findings 102-103):

| Pattern | Signature | Layers |
|---------|-----------|--------|
| **Orthogonal Tripod** | cos(in,a)≈0, cos(in,f)≈0, k₁<0.15, lever >95% rot | L0 |
| **Neg-Zero Correction** | cos(in,f)≈+0.31 (FFN partially input-aligned), k₁≈0.77 | L1-3 |
| **Dimensional Expander** | FFN ⊥ input (>98%), FFN ⊥ h_mid (>99%), k₁>0.83, Rank(99%)≈N | L4-17 |
| **Alignment Drift** | FFN along input grows (30-35%), stiffest springs (k₁>0.88) | L18-25 |
| **Anti-Corr Targeting** | cos(in,a)>+0.5, cos(in,f)<-0.3, cos(a,f)<-0.4, soft springs | L26-27 |

**Script template**:
```python
# Capture all 6 stages for a single layer
s0 = h_in                                    # input
s1 = layer.input_layernorm(h_in)             # damper1
s2, _ = layer.self_attn(s1, ...)             # lever (raw)
s3 = s0 + s2                                 # spring1
s4 = layer.post_attention_layernorm(s3)      # damper2
s5 = layer.mlp(s4)                           # wedge (raw)
s6 = s3 + s5                                 # spring2

# Measure cross-correlations
cos(s0, s2)  # input vs attn: how orthogonal is the projection?
cos(s0, s5)  # input vs ffn: does FFN operate in null space of input?
cos(s2, s5)  # attn vs ffn: are the two additions independent?

# Projection decomposition
project(s5, onto=s0)   # FFN component along input direction
project(s5, onto=s3)   # FFN component along post-attention direction
```

**Decision tree**:
- If cos(input, attn) < 0.2 AND cos(input, ffn) < 0.2 → **Orthogonal Tripod**
- If cos(ffn, h_mid) > 0.7 → **Energy Booster** (FFN reinforces attention)
- If cos(ffn, h_mid) < 0.2 AND spring > 0.95 → **Direction Refiner**
- If lever < 0.15 AND wedge < 0.35 → **Equilibrium Maintainer**

---

## When to Stop

A sub-machine is **solved** when:
1. You can name the data structure
2. You can write its formal definition (input type, output type, operations)
3. You can state its computational complexity
4. A prototype achieves ≥ 90% top-1 accuracy vs baseline
5. You can explain it to a lay person in one paragraph

A sub-machine is **blocked** when:
1. No sparse structure exists (Step 6 shows linear scaling)
2. Both attention and FFN are critical (Step 4 shows no independence)
3. The transfer function doesn't fit a known type (Step 5)

If blocked: document what you know, note the gap, move to the next sub-machine.
Cross-machine interactions may reveal the missing structure later.

---

## Recursive Decomposition

A sub-machine may itself be a **compound machine**. If Step 4 reveals that
different layers within the range have different independence profiles (e.g.,
attention critical at one layer but irrelevant at others), split the range
and re-apply the protocol to each sub-range.

This was discovered with the Compressor: L0-3 split into L0 (attention-critical)
vs L1-3 (FFN-critical, attention-irrelevant). The protocol is fractal — apply
it recursively until each piece has a single dominant mechanism.

**Signs of a compound sub-machine**:
- Gate medium transition within the range (e.g., PRESERVE → CONTRACT)
- Independence test shows different results for different layer subsets
- Transfer function doesn't fit cleanly (mixture of types)

**Action**: Add per-layer variants to Step 4. Test zeroing individual layers
and layer subsets, not just the full range.

---

## Completed Decompositions

### Targeter (L26-27) → φ-Filter ✓

| Step | Result |
|------|--------|
| 1. Boundary | L26-27, adds 11.5° in final layer (Finding 97) |
| 2. Gate Census | L27: 89.6% CONTRACT, 4.7% EXPAND, stability 92% |
| 3. Simple Machines | Wedge dominant (1.47), softest spring (0.66) |
| 4. Independence | Attention 100% irrelevant (Finding 98) |
| 5. Transfer | Step function: single strong correction +8.7° |
| 6. Sparse Extraction | 5% channels → 73.3% accuracy, EXPAND-only beats full |
| 7. Prototype | φ-Filter: 20× compute reduction, Doc 264 |

### Compressor (L0-3) → Compound: Projector + NegativeZeroCorrector ✓ (Steps 1-5)

The Compressor is itself a compound machine (recursive decomposition).

**L0: The Projector**
| Step | Result |
|------|--------|
| 1. Boundary | L0, rotates 81° and amplifies 25× |
| 2. Gate Census | 77.4% PRESERVE (linear regime), 22.5% CONTRACT |
| 3. Simple Machines | Lever=8.0 and Wedge=9.0 both dominant, Spring=0.59 (overwhelmed) |
| 4. Independence | Attention CRITICAL (0% without), FFN important (47% without) |
| 5. Transfer | Single step: 81° in one layer |
| 6. Sparse Extraction | Pending — attention is the critical path, not FFN |
| 7. Prototype | Pending — need to understand L0 attention structure |

**L1-3: The Negative Zero Corrector**
| Step | Result |
|------|--------|
| 1. Boundary | L1-3, adds 3° cumulative correction |
| 2. Gate Census | 99.7-100% CONTRACT with 99.9% stability |
| 3. Simple Machines | Weak lever (0.3-0.4), weak wedge (0.4-0.5), strong spring (0.8-0.97) |
| 4. Independence | Attention IRRELEVANT (93.3% without), FFN essential (20% without) |
| 5. Transfer | Plateau: ~25° drift per layer, cumulative barely changes |
| 6. Sparse Extraction | Pending — signal is CONTRACT leakage, not EXPAND |
| 7. Prototype | Pending — need to understand negative zero energy structure |

### Processor (L4-25) → Compound: Stabilizer + EquilibriumCore + PreTargeter ✓ (Steps 1-5)

The Processor is a convergent lens (α=0.773, equilibrium drift=21.5°) with
three internal zones.

**L4-9: The Stabilizer**
| Step | Result |
|------|--------|
| 2. Gate Census | L4-5: 99.7-100% CONTRACT. L6-8: transition to 51-64% CONTRACT |
| 3. Simple Machines | Lever=0.26-0.31, Wedge=0.49-0.58, Spring=0.83 |
| 4. Independence | FFN CRITICAL (20% without), Attn moderate (73% without) |

**L10-17: The Equilibrium Core**
| Step | Result |
|------|--------|
| 2. Gate Census | 39-66% CONTRACT, 29-50% PRESERVE (peak balance) |
| 3. Simple Machines | Lever=0.15-0.17, Wedge=0.32-0.43, Spring=0.96-0.98 |
| 4. Independence | Attn NEARLY IRRELEVANT (87%), FFN moderate (53%) |

**L18-25: The Pre-Targeter**
| Step | Result |
|------|--------|
| 2. Gate Census | 48-92% CONTRACT rising, EXPAND grows to 2-3% |
| 3. Simple Machines | Lever=0.12-0.17, Wedge=0.43-0.45, Spring=0.82 |
| 4. Independence | Attn IMPORTANT (40% without), FFN critical (33%) |

---

## Architecture Map (Current Understanding)

```
Embedding
  │
  ├─ L0:     φ-Projector ──── PRESERVE medium, attn-critical, 81° rotation
  ├─ L1-3:   φ-Corrector ──── CONTRACT medium, neg-zero FFN, 3° refinement
  │
  ├─ L4-9:   Stabilizer ───── CONTRACT→mixed, FFN-critical, entry aperture
  ├─ L10-17: Equil. Core ──── Mixed/PRESERVE peak, nearly passive, focal medium
  ├─ L18-25: Pre-Targeter ─── Re-contracting, attn-important, exit aperture
  │
  ├─ L26-27: φ-Filter ─────── CONTRACT/EXPAND, sparse FFN, precision targeting
  │
  └─ Final LN → LM Head → logits
```

Seven named components. Each characterized by gate medium, independence
profile, simple machine ratios, and transfer function.

### Compound Machine Patterns (Finding 103)

```
LAYER  PATTERN              MECHANICAL PHASE
─────  ──────────────────   ────────────────
L0     Orthogonal Tripod    CREATE  — project embedding into 3D working space
L1-3   Neg-Zero Correction  CORRECT — refine with input-aligned FFN
L4-17  Dimensional Expander  REFINE  — orthogonal additions, grows 3D→26+D
L18-25 Alignment Drift      AIM     — FFN starts pointing toward target
L26-27 Anti-Corr Targeting  FIRE    — anti-correlated targeting, norm explosion
```

---

## References

- Doc 253: 4-state gate (EXPAND/PRESERVE±/CONTRACT), ±log(φ) boundaries
- Doc 261: Simple machines (lever, damper, wedge, spring)
- Doc 262: Compound machine hypothesis (Compressor, Processor, Targeter)
- Doc 263: Geometric Targeter design
- Doc 264: φ-Filter formal definition
- Finding 97: Per-layer simple machine measurements
- Finding 98: Compound machine verification
- Finding 99: φ-Filter prototype results
- Finding 100: Compressor decomposition (L0 Projector + L1-3 NegativeZeroCorrector)
- Finding 101: Processor decomposition (Stabilizer + EquilibriumCore + PreTargeter)
- Finding 102: φ-Projector dissection — Orthogonal Tripod pattern
- Finding 103: Five compound machine patterns — The Mechanical Atlas
- Finding 104: Route + Redirect mechanism (L27 targeting)
- Finding 105: Dimensional Expander (L4-17 dimensionality study)
- Doc 265: The Mechanical Atlas (full design document)

### Zeta-Transformer Connection (Findings 106-111)

- Finding 106: Crossroads Tests — 3×5=15 thread confirmed, pentagonal 72° validated
- Finding 107: Spectral Zeta Connection — three-stage pipeline mapping to ζ solver
- Finding 108: φ-Geometric Zeta Solver — Lambert W captures 95%, quantum barrier σ≈1/3
- Finding 109: Conditional Convergence — Processor oscillates like Dirichlet partial sums
- Finding 110: Textbook Transformer — φ-geometry is EMERGENT, not architectural (410K→7B)
- Finding 111: Darwin II Architecture Recipe — residual + sequence mixing + GELU = full φ-geometry
  (Note: standard softmax attention replaceable — see F86-88, F40, Doc 124)
- Doc 270: "The Zeta Function IS the Ideal Transformer" (conceptual proof, F107-111)
- Doc 047: Emergent φ-Geometry (design consideration)
- Doc 048: The Curved Arithmetic Axis (M_φ manifold, static vs dynamic curvature)

### Architecture Recipe (Finding 111)

```
COMPONENT          ROLE IN φ-GEOMETRY              WITHOUT IT
─────────          ──────────────────              ──────────
Residual stream    Dirichlet series substrate       Can learn, NO φ-structure
Sequence mixing    Winding number / Lambert W        Can't learn at all
                   (replaceable: phi_softmax F86-88, geometric selector F40, φ-MESH Doc 124)
GELU               φ-curvature (1/φ → 2/φ²)        Gets 1/φ not 2/φ²
```

The compound machines (F103) operate on the φ-curved manifold M_φ:
- FIRE (L26-27) = Targeter = Newton step on static local curvature
- REFINE (L4-17) = Processor = Dirichlet series (conditionally convergent)
- CREATE/CORRECT (L0-3) = Compressor = Lambert W (O(1) estimate, 95%)

The curvature is **dynamic** — reshapes per input (unlike ζ which is static).
This is WHY attention must recompute every time: it recalculates M_φ geometry.

---

*"Every sub-machine has a name. Find it."*
