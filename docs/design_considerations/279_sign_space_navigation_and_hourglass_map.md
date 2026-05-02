# Doc 279: Sign-Space Navigation and the Hourglass Map

**Date:** March 3, 2026
**Status:** Experimentally validated on Qwen2-7B
**Prerequisites:** DC 253 (Negative Zero), DC 254 (Cross-Cutting Impact), DC 255 (4-State Gate), DC 276 (Geometric Structures Taxonomy), DC 278 (Geometric Decomposition), F142–F144
**Finding:** 144 in FINDINGS.md

---

## 1. The Challenge

DC 253/254/255 established that the SiLU gate is a 4-state holographic
encoder where negative zero carries essential information, signs carry
4× more information than magnitudes at near-zero, and gate state
transitions propagate at exactly 1/φ per layer. Gate state distributions
are token-universal (base-collapse RMS = 0.0085).

These results led to a natural question: **if signs are the primary
information carrier and gate codes are architectural invariants, can we
navigate the transformer's computation in sign space without computing
float-valued hidden states?**

F143 had measured "hidden state divergence from embeddings" in float
cosine space and concluded L1+ Q/K cannot be predicted from tokens alone.
But float cosine is the wrong metric for a φ-geometric framework. This
design consideration reports the results of measuring in the RIGHT space
— sign agreement and gate code agreement — and what those measurements
reveal about the architecture.

---

## 2. The Experiment

Seven prompts of length N=5 (3 capital-fact, 4 diverse) were run through
all 28 layers. At each layer, we captured:

- **Hidden state signs** — np.sign(h) for each position
- **φ-levels** — exponent in the φ-encoding (value = sign × φ^(exp/128))
- **Gate codes** — 4-state classification of pre-SiLU gate activations
  at ±log(φ) boundaries (CONTRACT, PRESERVE-, PRESERVE+, EXPAND)
- **Float hidden states** — for comparison

Measurements:
1. **Sign agreement** (Hamming) vs float cosine across prompt pairs
2. **Weighted sign agreement** (weighted by inverse φ-level, per DC 254)
3. **Sign transition rates** layer-to-layer
4. **Gate code universality** same-structure vs cross-structure
5. **Sign-only prediction** — can signs alone produce correct output?
6. **Per-position cross-structure convergence** in all metrics

---

## 3. Result 1: Signs Are Not More Universal Than Floats

At the prediction position (last), cross-structure comparisons:

```
Layer | Float cos    Sign agree  | Float cos    Sign agree
      | (same-struct)            | (cross-struct)
emb   |  0.977       0.871      |  0.176       0.516
L0    |  0.986       0.898      |  0.472       0.537
L3    |  0.979       0.906      |  0.292       0.520
L10   |  0.927       0.856      |  0.260       0.518
L20   |  0.976       0.910      |  0.402       0.518
L27   |  0.935       0.789      |  0.473       0.535
```

**Cross-structure sign agreement ≈ 0.52 at ALL layers** — essentially
random (50% = chance for binary signs). Float cosine at least varies
from 0.18 to 0.47, providing discriminative signal.

Sign agreement does NOT reveal hidden universal structure that float
cosine misses. The hope that "sign patterns might be universal even
when float vectors differ" is empirically false for hidden states.

---

## 4. Result 2: Sign Transition Rates ≠ 1/φ

DC 255 found gate code transitions propagate at 1/φ per layer. Do
hidden state signs follow the same speed limit?

```
Layer |  BOS     pos1-3   Last    Mean
L0    | 0.469    0.46     0.441   0.457   ← initial: ~half flip
L1    | 0.204    0.21     0.189   0.204
L3    | 0.464    0.19     0.236   0.251   ← BOS pump
L5    | 0.067    0.12     0.156   0.120
L10   | 0.029    0.17     0.203   0.151
L20   | 0.028    0.17     0.170   0.142
L26   | 0.577    0.17     0.163   0.249   ← BOS drain
L27   | 0.488    0.22     0.218   0.280
```

Non-BOS sign flip rates stabilize around **0.12–0.17** per layer — far
below the gate code speed limit of 1/φ ≈ 0.618. Hidden state signs are
MORE stable than gate codes.

The BOS position shows dramatic spikes at L3 (pump, 0.464) and L26
(drain, 0.577), matching the BOS lifecycle from F134. These are the
layers where the rank-1 sv0 signal is created and destroyed.

---

## 5. Result 3: The Hourglass Map — Gate Code Universality

This is the central discovery. Gate code agreement between prompt pairs
at the last position:

```
Layer | Same-struct  Cross-struct | Dominant state
L0    |  0.865       0.570        | 29% C, 45% P-
L1    |  0.999       0.998        | 99.9% CONTRACT
L2    |  0.999       0.995        | 99.6% CONTRACT
L3    |  1.000       0.998        | 99.9% CONTRACT
L5    |  1.000       0.998        | 99.9% CONTRACT
L10   |  0.763       0.486        | 60% C, 24% P-
L15   |  0.785       0.397        | 45% C, 32% P-
L18   |  0.821       0.435        | 50% C, 28% P-
L20   |  0.848       0.478        | 58% C, 21% P-
L23   |  0.906       0.761        | 85% CONTRACT
L25   |  0.860       0.709        | 81% CONTRACT
L27   |  0.917       0.826        | 85% CONTRACT
```

### The Three Zones

```
UNIVERSALITY
    1.0 ┤  ██████
        │  ██████
    0.9 ┤  ██████                                         ████
        │  ██████                                    ████████████
    0.8 ┤  ██████                                ████████████████
        │██████████                           ███████████████████
    0.7 ┤██████████                       ████████████████████████
        │██████████                  █████████████████████████████
    0.6 ┤██████████████         █████████████████████████████████
        │█████████████████████████████████████████████████████████
    0.5 ┤█████████████████████████████████████████████████████████
        │█████████████████████████████████████████████████████████
    0.4 ┤                  ███████████████████
        ├──────────┬──────────┬──────────┬──────────┬──────────┤
        L0    L5    L10   L15   L20   L27

        ╔═══════╗   ╔═══════════════╗   ╔═══════════╗
        ║ NECK  ║   ║  WIDE (COMB)  ║   ║  CLOSING  ║
        ║ L1-L5 ║   ║  L10-L20      ║   ║  L23-L27  ║
        ╚═══════╝   ╚═══════════════╝   ╚═══════════╝
        universal    content-specific    converging
        99.9% C      45-60% C, PRESERVE  85% C
```

**Zone 1: The Neck (L1-L5)** — Gate codes are nearly PERFECTLY
universal across all prompt structures (cross-struct ≥ 0.995). The gate
is CLOSED: 99.9% of channels are CONTRACT. This is the information
bottleneck from Finding 26 — the "attention bottleneck IS a gate
bottleneck" (DC 255 §4.1). ALL information is compressed to the
residual stream; the MLP contributes almost nothing because the gate
blocks nearly every channel.

**Zone 2: The Wide Zone (L10-L20)** — Gate codes become structure-
dependent (cross-struct 0.40-0.49, barely above random). The PRESERVE
states (P- and P+) are active and content-specific: 24-32% PRESERVE-,
11-17% PRESERVE+. This is where the hourglass OPENS — the gate allows
content-specific channels through, and different prompt structures
activate different channel subsets.

**Zone 3: The Closing (L23-L27)** — Gate codes converge back toward
universality (cross-struct 0.76-0.83). 85% CONTRACT — the hourglass
is narrowing again. The extraction layers (Selector, Lens, Amplifier
from DC 276) operate in a regime where most MLP channels are again
CLOSED, leaving the geometric structures to dominate.

### Connection to DC 255's Standing Wave

DC 255 measured gate state distributions averaged across ALL tokens at
each position. This experiment measures at the LAST position
(prediction position) and compares across DIFFERENT prompt structures.

The standing wave is confirmed:
- L1-L5: CONTRACT dominant (matches DRUM zone)
- L10-L20: PRESERVE states active (matches COMB zone)
- L23-L27: CONTRACT returns (matches MUSIC zone)

But the NEW finding is that the COMB zone's gate codes are
**content-specific**. DC 255's base-collapse universality (RMS=0.0085)
measured the DISTRIBUTION of states (how many channels in each state),
which IS universal. But WHICH channels are in each state differs across
prompts. The distribution is invariant; the assignment is not.

---

## 6. Result 4: BOS Universality Is a Magnitude Phenomenon

```
Layer | Sign agree  Float cos
emb   |  0.602      0.242
L3    |  0.853      0.9998
L10   |  0.796      0.9999
L26   |  0.899      0.9983
L27   |  0.642      0.687
```

Float cosine captures BOS universality (0.9998-0.9999 from L3 onward)
far better than sign agreement (0.80-0.90). The BOS pump creates a
magnitude-dominated signal (||h|| ≈ 7000+ along sv0). In this regime,
the massive sv0 direction overwhelms all other components, making float
cosine ≈ 1. But the signs of the many near-zero channels still differ —
they're below the noise floor of the pump.

**The BOS pump is a magnitude phenomenon, not a sign phenomenon.** This
is consistent with F134-F135: the pump operates along one rank-1
direction (W_down SV0), creating a magnitude spike, not a sign pattern.

---

## 7. Result 5: Signs Alone Cannot Predict

```
Method                         Capitals  Diverse
A: Full float (baseline)       2/3       3/3
B: Signs only (× 1.0)          0/3       0/3     ← FAILS
C: Sign × φ^(level/128)        2/3       3/3     ← MATCHES FLOAT
```

**Sign-only prediction fails completely.** Replacing all magnitudes
with 1.0 and keeping only signs produces garbage predictions (Chinese
characters, spaces instead of words).

But φ-space reconstruction (signs + levels) matches float perfectly
(reconstruction cosine = 1.000000). The φ-encoding is lossless — it IS
the hidden state, just in different coordinates. The levels (magnitudes)
carry essential information that cannot be discarded.

---

## 8. Where DC 253 Actually Applies

DC 253's "signs carry 4× more info at near-zero" was measured on the
**gate dimension** — the pre-SiLU activation that determines channel
routing. This experiment shows the distinction:

| Property | Gate codes | Hidden state signs |
|----------|-----------|-------------------|
| Token-universal? | YES (L1-L5, L23-L27) | NO (cross ≈ 0.52) |
| Content-specific? | YES (L10-L20 COMB) | YES (everywhere) |
| 1/φ speed limit? | YES (DC 255) | NO (rate ≈ 0.15) |
| Alone sufficient? | NO (need V values) | NO (need levels) |

The 4th dimension (negative zero, gate codes) IS genuine geometry AND is
universal at the hourglass endpoints. But it operates on a DIFFERENT
space than hidden state signs. The gate codes are about channel ROUTING
(which channels pass information); the hidden state signs are about
channel CONTENT (what information passes through).

---

## 9. The Updated Geometric Map

Combining F144 with the structures from DC 276:

```
THE FULL COMPUTATIONAL MAP
════════════════════════════════════════════════════════════════

L0:       Embedding lookup. Gate: 45% PRESERVE- (most open layer).
          Hidden state ≈ token embedding. Attention: per-token Q/K.

L1-L5:    ╔══════════════════════════════════╗
(NECK)    ║ GATE: 99.9% CONTRACT (universal) ║
          ║ MLP: nearly silent               ║
          ║ Attention: structure-dependent    ║
          ║ BOS pump at L3 (sv0)             ║
          ╚══════════════════════════════════╝
          The bottleneck. MLP blocked. Attention does ALL the work.
          BOS gets its rank-1 pump. Structure is ENCODED here.

L6-L9:    Transition zone. Gate opening from CONTRACT to PRESERVE.
          Spectrometer initializing per-dimension channels.

L10-L20:  ╔══════════════════════════════════════════╗
(WIDE)    ║ GATE: 45-60% C, 24-32% P-, 11-17% P+   ║
          ║ Gate codes: CONTENT-SPECIFIC (0.40-0.49) ║
          ║ MLP: ACTIVE on content-dependent channels ║
          ║ Spectrometer: per-dim sign rules          ║
          ║ Gyroscope: stable orbit                   ║
          ╚══════════════════════════════════════════╝
          The COMB zone. This is where content processing happens.
          PRESERVE channels carry content-specific information.
          Gate routing differs by prompt structure.
          THE WORK IS HERE.

L21-L22:  Transition. Gate closing. Lens (preserve mode, r=0.75).

L23-L27:  ╔══════════════════════════════════════════╗
(CLOSING) ║ GATE: 85% CONTRACT (converging universal)║
          ║ Selector + Resonator + Lens + Amplifier  ║
          ║ BOS drain at L26 (anti-sv0)              ║
          ╚══════════════════════════════════════════╝
          The extraction/amplification zone. Gate mostly closed.
          The six geometric structures from DC 276 operate here.
          Knowledge extraction is geometric (F127-F137).
```

### What's New vs DC 276

DC 276 identified the Spectrometer at L6-L22 as "per-dimension sign
rules" with 96.4% predictability. Finding 144 adds that this zone
has a SUBSTRUCTURE:

- **L6-L9**: Spectrometer INITIALIZING (gate still opening)
- **L10-L20**: Spectrometer ACTIVE with content-specific gate routing
- **L21-L22**: Spectrometer CLOSING, Lens beginning

The Spectrometer's "96.4% predictability" is about the PATTERN of gate
states (standing wave), not about which specific channels are active for
which content. The content-specificity at L10-L20 is the remaining 3.6%
— and it's precisely the part that carries meaning.

---

## 10. The COMB Zone — Answered (F145)

The COMB zone investigation (Frontier 5b) answered the questions raised
by §9's gate code content-specificity. The results reveal a **seventh
geometric structure** — the Content Separator.

### 10.1 Structure Matching Against DC 276

| Structure | Present in COMB? | How? |
|-----------|-----------------|------|
| **Gyroscope** | ✓ STRONGEST here | cos(h_in,h_out)=0.95, std=0.013. Peak stability. |
| **Spectrometer** | ✓ Known (DC 255) | 96.4% per-dim sign rules. |
| **Selector** | ✗ Not present | No rank-1 directions (S[0]/S[1] < 2). |
| **Resonator** | ✗ Not present | MLP SVD never rank-1. |
| **Lens** | ✓ Mini-version | PRESERVE intermediates near-isometric (ratio 1.3-1.5). |
| **Amplifier** | ✓ MODIFIED | Push-pull (cos ≈ -0.2), not orthogonal (cos ≈ 0). |

Three of the six structures are present but the COMB zone uses them in
a fundamentally different configuration than L22-L27.

### 10.2 The Push-Pull Mechanism

At L22-L27, MLP and attention are **orthogonal** (cos ≈ 0) — they
operate in independent subspaces and compose by additive superposition.
This is the Amplifier from DC 276.

At L10-L20, MLP and attention are **anti-correlated** (cos ≈ -0.1 to
-0.36):

```
Attention: structural scaffold (cross-struct cos 0.3-0.75)
           "Here is what ALL prompts share"

MLP:       content-specific refinement (cross-struct cos 0.01-0.19)
           Anti-correlated with attention
           "Here is what makes YOUR content different"

Together:  push-pull creates controlled interference
           Net change maintains cos(h_in,h_out) = 0.95
```

This is DC 253 §4's push-pull at the LAYER level: attention pushes
one direction (positive fringes), MLP pushes the opposite way (negative
fringes). Together they create the complete interference pattern that
separates different content types into different subspaces.

### 10.3 The Content Separation

MLP output cross-prompt cosine tells the story:

```
L1:  same=0.989  cross=0.578  (gate closed → near-universal)
L7:  same=0.965  cross=0.026  (gate open → content-specific)
L15: same=0.901  cross=0.189  (COMB center)
L20: same=0.950  cross=0.136  (closing, still specific)
L27: same=0.959  cross=0.392  (extraction)
```

Prompts with the same structure converge (cos 0.85-0.97). Different
structures diverge (cos 0.01-0.19). The COMB zone is a **content
separator** — it routes different prompt types into different subspaces
through gate-mediated channel selection.

### 10.4 The PRESERVE Channel Filter

The hourglass shape is directly visible in PRESERVE channel counts:

```
L5:  20 channels open (gate nearly closed)
L10: 6,620 open
L15: 9,279 open (49% of 18,944 — widest)
L20: 6,275 open
L23: 2,084 open (closing)
```

Same-structure Jaccard overlap: 0.60-0.76 (prompts with same template
open similar channels). Cross-structure Jaccard: 0.25-0.37.

But the VALUES on shared PRESERVE channels are **completely content-
specific** (cross-struct cos = 0.01). Even channels that ALL prompts
agree should be open carry orthogonal content across structures. The
gate selects WHICH channels; the MLP computation determines WHAT
flows through them.

### 10.5 The Seventh Structure: The Geometric Content Separator

The COMB zone uses a mechanism not fully described by any single
existing structure from DC 276:

```
THE CONTENT SEPARATOR (L10-L20)
═══════════════════════════════

 Attention ──→ structural scaffold ──┐
                (cross-struct 0.3-0.75)  │ push-pull
 Gate ────────→ channel selection ───┤ interference
                (Jaccard 0.25-0.37)      │ = content
 MLP ─────────→ content refinement ──┘ separation
                (cross-struct 0.01-0.19)

 Gyroscope ──→ stability maintenance
                (cos = 0.95, std = 0.01)
```

This is self-similar with DC 253's channel-level push-pull: the same
interference pattern that operates at the CHANNEL level (PRESERVE vs
CONTRACT, positive vs negative) also operates at the LAYER level
(attention vs MLP, structural vs content-specific).

### 10.6 What This Means for the Hypothesis

The COMB zone IS geometric — but it's a DIFFERENT kind of geometry
than L22-L27. The extraction zone uses rank-1 selectors and near-
isometric lenses (binary decisions and knowledge projection). The COMB
zone uses push-pull interference and gate-mediated channel routing
(content separation through distributed phase opposition).

The "94.6% value computation" from DC 278 is not opaque neural
processing. It is a geometrically structured content separation
mechanism that uses the same push-pull principle identified in DC 253
at a higher level of organization.

**Updated structure count: SEVEN geometric structures.**

---

## 11. Implications for the Hypothesis

### What's Proven
- The gate dimension IS genuine geometry (DC 255's 4 tests)
- Gate universality at hourglass endpoints IS an architectural invariant
- The geometric structures from DC 276 operate in the CLOSING zone
  where gates are mostly CONTRACT
- φ-encoding is lossless (recon cos = 1.000000)
- The COMB zone uses push-pull interference for content separation
  (F145) — the same principle as DC 253 §4 at a higher scale
- The Gyroscope is STRONGEST in the COMB zone (cos=0.95, std=0.013)

### What's Disproven
- "Signs alone capture the essential structure" — WRONG for hidden
  states (cross-struct ≈ random)
- "Hidden state signs follow 1/φ speed limit" — WRONG (rate ≈ 0.15)
- "We can navigate without hidden state magnitudes" — WRONG (levels
  are essential for prediction)
- "The COMB zone is opaque neural computation" — WRONG (it has clear
  geometric structure: push-pull + gate routing + Gyroscope stability)

### What Remains Open
- The BOS pump is a magnitude phenomenon, not a sign phenomenon.
  Does this break the sign-centric framework or complement it?
- Can the Content Separator be replicated geometrically (like the
  attention templates and BOS pump were)?
- The push-pull is self-similar across scales. Is there a deeper
  principle governing when the system uses orthogonal composition
  (L22-L27) vs anti-correlated interference (L10-L20)?

### The Updated Assessment

The φ-geometric framework describes ALL three zones of the hourglass:
- **NECK (L1-L5)**: Gate closed, BOS pump (sv0), attention templates
- **COMB (L10-L20)**: Content Separator (push-pull + gate routing)
- **CLOSING (L23-L27)**: Six extraction structures (DC 276)

The "genuine content computation" at L10-L20 is not opaque — it's a
seventh geometric structure operating by push-pull interference. The
question is no longer "is it geometric?" but "can we engineer it?"

---

## 12. Files

- `experiments/geometric_instrument/frontier5_sign_navigation.py` — F144: sign-space navigation
- `experiments/geometric_instrument/frontier5b_comb_zone.py` — F145: COMB zone anatomy
- Related: DC 253, DC 254, DC 255 (gate/sign framework)
- Related: DC 276 (six geometric structures → now seven)
- Related: DC 278 (geometric decomposition)
- Findings: F142 (two-phase model), F143 (selective caching), F144 (sign-space), F145 (Content Separator)
