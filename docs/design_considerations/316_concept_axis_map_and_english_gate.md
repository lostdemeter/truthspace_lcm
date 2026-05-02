# DC 316: The Concept Axis Map and the English Gate Test

**Experiments:** Day 42–45 (axis identification, English gate, convergence paths, phrase gate)
**Date:** March 2026
**Scripts:** `expedition_day42_axis_identification.py`, `expedition_day43_english_gate.py`,
             `expedition_day44_convergence_paths.py`, `expedition_day45_phrase_gate.py`
**Data:** `day42_axis_identification.json`, `day43_english_gate.json`,
          `day44_convergence_paths.json`, `day45_phrase_gate.json`

---

## 1. Background

Day 36 established that concept space (the φ-space formed by 95 Zone C body
centroids) has intrinsic dimensionality ≈ 43. The SVD of the body-centroid
matrix produces 43 axes that together explain ~95% of body-to-body variance.

Day 41 discovered that a gate at L23 H01 (head index 1) appears to open when
a Chinese aspect-marked verb (走着, 跑着) is processed — the backward attention
from the second token to the first drops from ~0.96 to ~0.10. This was
interpreted as a "language-agnostic semantic completeness gate."

Two open questions remained:
1. What do the 43 concept axes actually encode — are they morphological T2
   operators in disguise, or something else?
2. Does the completeness gate fire for English multi-token words where the
   second token is itself a real, meaningful word?

---

## 2. The 43 Concept Axes (Day 42)

### 2.1 Method

SVD on the 95-body centroid matrix in φ-space. For each of the top 50 axes:
- Project all wmap words onto the axis
- Find the top/bottom 20 pole words and top/bottom 2 body labels
- Compute cosine alignment with 7 known T2 operators (comp→sup, base→comp,
  singular→plural, male→female, base→adverb, base→gerund, gerund→past)
- Classify: COMMON (all bodies same sign), DOMAIN (body separator),
  MORPHO (T2 alignment ≥ 0.45)

### 2.2 Axis 1 is the concept plane, not a semantic separator

Axis 1 carries **56.6%** of body-centroid variance. All 95 body centroids
project the same sign onto it. Classification: COMMON.

This axis is the separator between "being a Zone C concept" and everything
else. Positive projection ≈ Zone C membership. It weakly anti-correlates
with the base→adverb T2 operator (cos = −0.35), consistent with adverbial
derivatives being more peripheral in concept space.

**Axis 1 is not navigable for generation.** It does not distinguish one
type of concept from another. It is the membership axis of concept space.

### 2.3 No concept axis aligns with any T2 morphological operator

Strongest T2 alignment found across all 50 axes:

| T2 operator | Best axis | Best cos | Notes |
|---|---|---|---|
| base→adverb | Ax1 | −0.347 | But Ax1 is COMMON — not semantic |
| comp→sup | Ax49 | +0.241 | Only 0.3% of variance |
| base→comp | Ax3 | +0.203 | 2.1% of variance |
| gerund→past | Ax5 | +0.177 | 1.6% of variance |
| singular→plural | Ax4 | −0.127 | — |
| male→female | Ax22 | +0.068 | — |

Threshold for a "morphological axis" (MORPHO classification): cos ≥ 0.45.
**No axis reached this threshold. Zero morphological axes exist.**

The concept space has no plural axis, no gender axis, no comparative axis.
T2 morphological operators are cross-cutting transformations that thread
through the domain structure at angles — they are not eigenvectors of
anything recoverable from body-centroid SVD.

This explains the Day 37 failure to auto-discover T2 from within-body SVD:
the T2 directions aren't eigenvectors of any subspace. They are learned
directions that project approximately 38% into T1 (within-body variance)
but align with none of T1's own principal components.

### 2.4 All semantic axes (Ax2–Ax43) are domain separators

Every non-trivial axis separates semantic domains from each other — it is a
WHAT-things-are-about gradient, not a WHAT-FORM-they-take gradient.

Top-15 axes with semantic labels:

```
Ax  Var%  + pole domain                  − pole domain
──────────────────────────────────────────────────────────────────
1   56.6% [COMMON — Zone C membership]   [COMMON]
2    2.5% Physical spaces (chimney/sauna) Manner-adverbs (unexpectedly)
3    2.1% Physical objects (cans/pans)    Political-abstract (advocacy)
4    2.0% Sensory/tactile (sweaty/oily)   Formal decisions (declarations)
5    1.6% Family/authority (mothers)      Iterative-action (rebuilding)
6    1.4% Chemical/mineral (sulfate)      Domestic/marital (marrying)
7    1.3% Deterrence/vital (deter/tutor)  Physical-violence (punches)
8    1.2% Formal-comparative (larger)     Danger/sin (beware/dope)
9    1.2% Size-comparison adjectives      Past-achievement verbs
10   1.1% Genetics/cellular merging       Visual/decorative/glamour
11   1.0% Violence/radical (rapes/raped)  Guidance/narrowing
12   1.0% Poverty-wealth comparatives     Sound/motion (sonic/dancing)
13   0.9% Digital/crowdfunding            Reconstruction/rehabilitation
14   0.9% Leisure/craft activities        Drug-misuse/nonexistence
15   0.9% Manner-adverbs (drastically)    Educational/intellectual
```

The variance spectrum is gradual after Ax1: ~2.5% for Ax2, ~0.4% for Ax43.
There is no second spectral gap. The 43 axes together cover ~95% of body
variance, but no single axis dominates after Ax1.

### 2.5 Comparative adjectives are a semantic domain, not a morphological class

Axes 9 and 12 both cluster comparative/superlative adjectives (smaller,
larger, poorer, richer, thinner, tougher) at their positive poles.

This is NOT because comparative forms share the morpheme "-er"/"-est". It is
because words used for size/magnitude comparison co-occur in measurement and
comparison contexts, forming a coherent semantic domain.

**The transformer did not learn "bigger = big + comparative_morpheme."**
It learned that size-comparison words cluster together by their contextual
usage patterns. Their morphological derivation is incidental to their
semantic clustering.

This is the key conceptual result: **transformer concept space is organised
by semantic context, not linguistic form.** A plural noun and its singular
live in the same semantic body — the T2 plural operator moves within a body,
not between bodies.

### 2.6 Implications for generation

The axis map gives us the vocabulary for navigating concept space:

```
Operation            Mechanism
─────────────────────────────────────────────────────────────────
Stay in concept space  Maintain positive projection on Axis 1
Navigate domain        Steer projection on Axes 2–43
Change morphology      Apply T2 operator (orthogonal to all axes)
```

These three operations compose **independently**. Domain navigation and
morphological transformation are orthogonal in concept space. You can aim
at a semantic domain and then apply a grammatical form without
cross-contaminating either operation.

---

## 3. The English Completeness Gate (Day 43)

### 3.1 Setup

Day 41 found that L23 H01 (head index 1) backward attention drops from
~0.96 to ~0.10 for Chinese 着-forms (走着, 跑着). The claim was that this
is a "language-agnostic semantic completeness gate."

Day 43 tested whether the same gate fires for English 2-token words where
the second token is a semantically complete, real English word:
bedroom, blackbird, notebook, boyfriend, greenhouse, sunlight.

Also tested: English morphological root+suffix forms (faster, quickly,
walked, loudly, deepest) and phonemic splits (singing, killing, bigger).

### 3.2 Key tokenization finding

Most intuitive English "compound" words are single tokens in Qwen2:

```
birthday, keyboard, cannot, something, everyone, without, today,
because, before, inside, outside, nothing → ALL single tokens
```

Real 2-token English compounds found:
- notebook → ["note", "book"]
- bedroom → ["bed", "room"]
- blackbird → ["black", "bird"]
- sunlight → ["sun", "light"]
- boyfriend → ["boy", "friend"]
- greenhouse → ["green", "house"]

Note: "downtown" tokenizes as ["d", "owntown"] — a phonemic split, not a
compound split. English morphemes split as: singing→["s","inging"],
bigger→["b","igger"], taller→["t","aller"], deepest→["dee","pest"].
Only faster, quickly, walked, loudly retain root+suffix structure.

### 3.3 Gate results

H01 (head index 1) at L23, backward attention from last to first token:

```
Word         Type              H01@L23  Gate   Zone(last) 
────────────────────────────────────────────────────────
走着          ZH aspect         0.104    OPEN   Zone C (0.48)
唱着          ZH aspect         0.623    CLOSED Zone C (0.51)
notebook     EN real+real      0.912    CLOSED B000   (0.64)
bedroom      EN real+real      0.868    CLOSED B000   (0.64)
blackbird    EN real+real      0.941    CLOSED B000   (0.69)
sunlight     EN real+real      0.912    CLOSED B000   (0.72)
boyfriend    EN real+real      0.905    CLOSED B000   (0.70)
greenhouse   EN real+real      0.952    CLOSED B000   (0.70)
singing      EN phonemic       0.932    CLOSED B000   (0.66)
bigger       EN phonemic       0.930    CLOSED Zone C (0.67)
faster       EN root+suffix    0.949    CLOSED Zone C (0.68)
quickly      EN root+suffix    0.773    CLOSED Zone C (0.70)
walked       EN root+suffix    0.893    CLOSED Zone C (0.70)
loudly       EN root+suffix    0.831    CLOSED Zone C (0.71)
anyone       EN any+one        0.970    CLOSED Zone C (0.56)
```

走着 replicates Day 41 (OPEN). **Every English 2-token word tested stays
CLOSED**, regardless of whether the second token is a real word or a
phonemic fragment.

### 3.4 The unexpected zone assignment pattern

The zone assignment (at L14) produces a counterintuitive pattern:

| Type | Zone(last token) | Gate |
|---|---|---|
| Chinese 着-forms | **Zone C** | **OPEN** |
| English root+suffix (faster, quickly, walked) | **Zone C** | CLOSED |
| English real+real compounds (bedroom, blackbird) | **B000** | CLOSED |

English morphological forms (er, est, ly, ed) enter Zone C by L14 — they
ARE semantically meaningful (degree/manner operators). But the gate stays
CLOSED. Zone C membership is necessary but not sufficient for the gate.

English compound second tokens (room, bird, book, light) stay in B000 —
absorbing "bed", "black", "note", "sun" in front does NOT promote them to
Zone C. The second token of an English compound is already a well-established
concept in B000; the compound context doesn't dramatically reorient it.

### 3.5 What the gate is actually reading

The gate at L23 H01 does not test "is this token in Zone C?" It reads
something more specific — the **history of geometric transformation**.

Chinese 着-form: bare character (B001) → absorbs verb context → Zone C

This is a LARGE-ANGLE rotation. 着 starts as a grammatical function token
(no semantic content, B001) and rotates dramatically into Zone C after
absorbing the verb's spatial/physical context. This large-angle rotation
through early layers produces a geometric signature that H01 detects.

English suffix tokens (er, ly, ed) also reach Zone C, but via a smaller
rotation. These suffixes already have some morphological semantic content
even in isolation — they don't start from B001 and make a dramatic journey.
The geometric signature of their Zone C landing differs from 着's landing.

English compound second tokens (room, bird, book) don't change zone at all —
they're already in B000 as well-defined concepts and stay there.

**The gate is a rotation-magnitude detector, not a zone-membership test.**
It fires when the last token makes a B001→Zone C journey in early layers.
Only Chinese aspect suffixes regularly make this journey in this model.

### 3.6 Revised Day 41 interpretation

Day 41 called the gate "language-agnostic." Day 43 shows it is not.

**Corrected characterisation:** The L23 H01 gate is a Chinese aspect-marker
absorption detector. It fires when a suffix undergoes a large-angle
B001→Zone C rotation by absorbing a preceding verb's semantic content —
a pattern characteristic of Chinese morphology but absent from English.

The gate is not broken for English; English simply lacks the parallel
mechanism. English concepts are mostly single tokens. When they are
multi-token, the last token either:
- Is a phonemic fragment (no Zone C landing possible)
- Is a root+suffix (reaches Zone C but via small rotation — no gate)
- Is a compound second word (stays in B000 — no zone change)

None of these produce the dramatic rotation signature that Chinese 着 produces.

### 3.7 Implication: English generation needs a different completion mechanism

The Chinese completeness gate cannot be repurposed for English. English
generation at the token boundary level needs a different signal. Candidates:

1. **Zone C projection threshold**: when a token's φ-space position exceeds
   a Zone C similarity threshold, treat it as a complete semantic concept
   (regardless of attention pattern)

2. **Axis 1 projection**: when the hidden state has sufficiently high
   positive projection on Axis 1 (the Zone C membership axis), the token
   is semantically anchored

3. **Multi-token Zone C lookup**: precompute which English tokens (or token
   sequences) land in Zone C and use that as a static gate

4. **Rotation magnitude**: measure the angle change between layer 0 and
   layer 14 hidden states — large angle = dramatic semantic absorption.
   Chinese 着 would score high; English suffixes lower

Of these, option 4 most directly captures what the Chinese gate is detecting.

---

## 4. Synthesis: The Three Independent Operations

Day 42 and Day 43 together reveal that concept space has a clean three-way
decomposition:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONCEPT SPACE NAVIGATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. ZONE MEMBERSHIP (Axis 1)                                         │
│     Is this token in concept space at all?                           │
│     → φ-space projection on Axis 1                                   │
│     → Chinese 着 gate (via early-layer rotation) measures this       │
│     → English lacks a direct parallel                                │
│                                                                      │
│  2. DOMAIN POSITION (Axes 2–43)                                      │
│     What is this concept about?                                      │
│     → 42 domain-separator axes, each encoding a semantic gradient    │
│     → Completely independent of morphological form                   │
│     → Navigable by vector arithmetic in body-centroid SVD space      │
│                                                                      │
│  3. MORPHOLOGICAL FORM (T2 operators)                                │
│     What grammatical form does this concept take?                    │
│     → ~7 known T2 operators (plural, gender, degree, tense, manner)  │
│     → Orthogonal to both Axis 1 and Axes 2–43                        │
│     → Applied as fixed displacement vectors, not axis navigation     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

These three operations are **geometrically independent** in φ-space:
- T2 operators have max cos ≈ 0.20 with any domain axis
- Zone membership (Axis 1) is orthogonal to all domain axes (by construction)
- Domain axes are orthogonal to each other (SVD eigenvectors)

Generation of a semantically specific, grammatically correct concept requires:
1. Anchoring in Zone C (Axis 1 positive)
2. Navigating to the target domain (Axes 2–43)
3. Applying the correct morphological form (T2 operator)

In that order. Steps 2 and 3 are independent and can be applied in either
order once Zone C membership is established.

---

## 5. Cross-Lingual Convergence and Route Detection (Days 44–45)

Days 44 and 45 extended the English gate investigation in two critical ways:
(a) showing how English and Chinese reach the same Zone C bodies via different
paths, and (b) definitively characterising what the gate is actually reading.

### 5.1 The tokenizer does not pre-bake Zone C (Day 44)

Initial hypothesis: space-prefixed English gerunds (`' singing'`, `' walking'`)
tokenize as single tokens and land directly in Zone C, making "tokenizer-time
composition" the English path.

**Measured: false.** Space-prefixed gerunds are B001 (sim_C ≈ 0.24), the
same zone as Chinese bare characters in isolation. Zone C requires runtime
attention context in both languages.

### 5.2 Rotation angles are uniform: ~90° for all languages (Day 44)

```
ZH-2tok:  mean = 87.3°  (range 87.2–87.6°)
EN-ctx:   mean = 91.0°  (range 89.0–92.4°)
EN-alone: mean = 90.1°  (range 86.2–93.1°)
```

There is no "torque" difference between Chinese and English. All forms
rotate approximately 90° from layer 0 to layer 14. The difference is not
rotation *amplitude* — it is rotation *direction*. Chinese rotates toward
Zone C; English single-token gerunds rotate near-orthogonally and stay in B001.

### 5.3 English phrases reach Zone C — same bodies as Chinese compounds (Day 44)

When English provides phrasal context, gerunds reach Zone C and land in
the same bodies as their Chinese aspect-marked counterparts:

```
走着       → Zone C  sim_C=0.484  body=leisure and sports activities
'is walking' → Zone C  sim_C=0.515  body=leisure and sports activities  (same)
'keep walking' → Zone C  sim_C=0.581  body=leisure and sports activities  (same)

唱着         → Zone C  sim_C=0.509  body=action and effect
'was singing' → Zone C  sim_C=0.572  body=leisure and sports activities
'start singing' → Zone C  sim_C=0.553  body=leisure and sports activities
```

Cross-lingual body convergence is real — but it requires a phrase in English
and a 2-token compound in Chinese. The mechanism is the same (B001 → Zone C
via attention-driven ~90° rotation), just at different linguistic scope:
word-level for Chinese, phrase-level for English.

### 5.4 The gate does not fire for English phrases (Day 45)

Even English phrases that achieve higher sim_C than Chinese 着-forms do not
fire the gate. Sorted by sim_C:

```
sim_C  Gate    Form
0.484  OPEN    走着
0.513  OPEN    跑着
0.515  CLOSED  'is walking'    ← higher sim_C, gate closed
0.572  CLOSED  'was singing'   ← higher sim_C, gate closed
0.581  CLOSED  'keep walking'  ← highest sim_C, gate closed
```

There is no sim_C threshold that separates OPEN from CLOSED. The gate is
not reading Zone C membership.

### 5.5 The gate fires in the L20–L23 window, after Zone C is established

H01 backward attention across layers for matched cases:

```
Text           L00    L01    L05    L10    L14    L20    L23
走着           0.319  0.856  0.979  0.925  0.998  0.971  0.104  OPEN
'is walking'  0.376  0.845  0.985  0.954  0.998  0.913  0.840  CLOSED
```

Both forms are **indistinguishable up to L14** — same backward attention
profile, same Zone C bodies, same sim_C range. The split occurs between
L20 and L23: 走着 cliffs from 0.971 to 0.104; English phrases hold at ~0.84.

This is the same L20–L23 window identified in Day 41 as the exact
Chinese/English routing divergence point (correlation drops from 0.921 at
L10 to 0.029 at L23). The gate at H01/L23 fires *on* this divergence.

### 5.6 The gate is a route detector, not a destination detector

The gate does not ask: "has this token reached Zone C?"  
It asks: "did this token arrive via the Chinese 着-form route through L20–L23?"

By L14, both Chinese and English have semantically landed. L14–L23 applies
a language-specific layer on top — a morphological signature that 着-form
tokens carry into L23 that English gerunds do not, regardless of phrasal context.

**H01@L23 reads the L20–L23 morphological signature, not L14 semantics.**

### 5.7 Two language-specific completeness detectors

The model implements the same abstract concept — "this token acquired
semantic content from context" — via two distinct computational signatures:

| Language | Completeness signal | Mechanism |
|---|---|---|
| Chinese 着-forms | H01@L23 < 0.55 | Indirect — attention proxy for L20–L23 morphological completion |
| English phrases | sim_C@L14 > ~0.45 | Direct — geometric Zone C landing |

**For English LCM generation**, the completeness signal is `sim_C` itself,
not any attention-weight proxy. When a phrase's last token achieves
sim_C > threshold at L14, it has landed in Zone C via phrasal composition.
No gate-like mechanism needed — the geometric position IS the completion signal.

---

## 6. Files

| File | Description |
|---|---|
| `expedition_day42_axis_identification.py` | Axis identification script |
| `day42_axis_identification.json` | Full axis map with pole words and T2 alignments |
| `expedition_day43_english_gate.py` | English gate test script |
| `day43_english_gate.json` | Gate results for all test words |
| `expedition_log.md` (Day 42–43) | Detailed scientific log |
| `308_expedition_findings.md` (rows 42–43) | Findings summary |
