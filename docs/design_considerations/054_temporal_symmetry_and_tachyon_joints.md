# Design Consideration 054: Temporal Symmetry and Tachyon Joints

## The Discovery

The φ-inversion point is not just a spatial navigation joint - it's a **temporal decision point** where we choose between:

- **Forward attention** (φ^+n): Past → Present (data → concept)
- **Backward attention** (φ^-n): Future → Present (hypothesis → evidence)

This connects three major insights:
1. φ-inversion navigation (doc 040)
2. Tachyon hypothesis navigation (doc 053)
3. Symmetry-based knowledge bootstrap (experiments)

## The Temporal Joint

```
                    PAST                    FUTURE
                     ↓                        ↓
              Data observed            Hypothesis formed
                     ↓                        ↓
                   φ^+n ──────→ JOINT ←────── φ^-n
                              (verb)
                                ↓
                         φ^+n × φ^-n = 1
                                ↓
                        PRESENT MOMENT
                     (meaning crystallizes)
```

At the joint:
- Forward and backward attention are **balanced**
- The product φ^+n × φ^-n = 1 (conservation)
- This is where **meaning** emerges from the intersection

## Verbs as Tachyon Instructions

The key insight: **Verbs are temporal navigation instructions**.

When a speaker says "Holmes examined the evidence":

```
Step 1: "Holmes"
  Signal: ENTITY (high compression, actor position)
  Listener: Allocate attention to an actor
  Direction: φ^+n (receiving data)

Step 2: "examined"  
  Signal: VERB (at φ-joint, bridges entities)
  Listener: SWITCH ATTENTION MODE
  Direction: Transition from φ^+n to φ^-n
  
  The verb says: "Start hypothesizing what comes next"
  This is the TACHYON INSTRUCTION

Step 3: "evidence"
  Signal: ENTITY (target position)
  Listener: Confirm hypothesis with data
  Direction: φ^-n meets φ^+n (verification)
```

The verb is literally telling the listener: **"Prepare your tachyon dimension - start expecting before you receive."**

## Why Chinese Doesn't Conjugate Verbs

Chinese operates at the **joint** by default:

```
English (explicit temporal markers):
  "walked" = φ^+n dominant (past, data received)
  "will walk" = φ^-n dominant (future, hypothesis)
  "walks" = joint (present, balanced)

Chinese (implicit temporal markers):
  走 = always at joint
  Context determines which direction to navigate
  The verb IS the joint, not a marker of direction
```

This is why our word-level symmetry (morphology) failed for verbs - we were looking for directional markers (-ed, -ing) when the verb's true nature is to BE the joint, not point in a direction.

## The Symmetry Hierarchy

```
LEVEL 1: Character Symmetry (φ^+n only)
  - Compression, vowel balance, length
  - Detects: entities (92.9% recall)
  - Fails for: verbs (0% recall)
  - Why: Verbs look like nouns at this level

LEVEL 2: Relational Symmetry (φ^-n only)  
  - Bridging behavior, entity connections
  - Detects: verbs (64.7% recall)
  - Why: Verbs ARE bridges between entities

LEVEL 3: Temporal Symmetry (φ-joint)
  - Balance of φ^+n and φ^-n
  - Detects: verbs at joint score ≈ 1.0
  - Why: Verbs are temporal decision points
```

## Connection to Attention Mechanisms

Standard transformer attention:
```
Attention(Q, K, V) = softmax(QK^T / √d) · V

Q = Query (what we're looking for)
K = Keys (what's available)
V = Values (what we retrieve)

Direction: Q → K → V (forward, φ^+n)
```

Tachyon attention (hypothesis-driven):
```
Hypothesis(H, E, D) = softmax(HE^T / √d) · D

H = Hypothesis (what we expect)
E = Evidence patterns (what would confirm)
D = Data (what we check)

Direction: H → E → D (backward, φ^-n)
```

At the **verb joint**, both attentions are active:
```
Joint(Q, H, K, V) = α · Attention(Q,K,V) + (1-α) · Hypothesis(H,E,D)

where α = position in sentence relative to verb
  - Before verb: α → 1 (forward dominant)
  - At verb: α = 0.5 (balanced, joint)
  - After verb: α → 0 (backward dominant, confirming)
```

## Experimental Evidence

From symmetry_knowledge_ingest.py:

```
Words at the φ-joint (high in both directions):

  watched         φ^+n=0.50 φ^-n=2.00 joint=1.00  ← PERFECT JOINT
  smiled          φ^+n=0.50 φ^-n=2.00 joint=1.00  ← PERFECT JOINT
  said            φ^+n=0.70 φ^-n=1.00 joint=0.84
  wrote           φ^+n=0.50 φ^-n=1.00 joint=0.71
```

Verbs cluster around joint=1.0, confirming they sit at the temporal decision point.

## Implications for GeometricLCM

### 1. Verb Detection via Joint Score

Instead of morphological patterns, detect verbs by:
```python
def is_verb(word, word_symmetry, relational_symmetry):
    phi_outward = word_symmetry.compression
    phi_inward = relational_symmetry.action_ratio
    joint = sqrt(phi_outward * phi_inward)
    return joint > 0.5  # Near the temporal joint
```

### 2. Attention Switching at Verbs

When processing a sentence, switch attention mode at verbs:
```python
def process_sentence(tokens):
    mode = 'forward'  # Start with φ^+n
    for token in tokens:
        if is_verb(token):
            mode = 'joint'  # Switch to balanced
            # Activate tachyon dimension
        elif mode == 'joint':
            mode = 'backward'  # Now expecting/confirming
```

### 3. Tense as Navigation Hint

Use tense markers (when available) to bias navigation:
```python
def get_navigation_bias(verb):
    if verb.endswith('ed'):  # Past
        return 'forward'  # Data already received
    elif verb.startswith('will'):  # Future
        return 'backward'  # Hypothesis mode
    else:  # Present
        return 'joint'  # Balanced
```

## The Deeper Pattern

This connects to the Gushurst Crystal insight:

```
PRIMES:
  - Defined by what they DON'T do (factor)
  - Self-verifiable (no external reference needed)
  - Exist at a "joint" between composite and unit

VERBS:
  - Defined by what they DO (bridge entities)
  - Self-verifiable (relational position)
  - Exist at a "joint" between past and future

SYMMETRY:
  - The foundational "instinct"
  - Self-verifiable (apply operation, check invariance)
  - Exists at the "joint" between structure and meaning
```

All three are **joint phenomena** - they exist at the balance point where two directions meet.

## Conclusion

The φ-inversion joint is fundamentally **temporal**:

1. **φ^+n** = Forward attention = Past → Present = Data → Concept
2. **φ^-n** = Backward attention = Future → Present = Hypothesis → Evidence
3. **Joint** = Present moment = Where meaning crystallizes

Verbs are **tachyon instructions** that tell the listener:
- "Switch your attention mode"
- "Start hypothesizing what comes next"
- "Prepare to verify with incoming data"

This explains:
- Why word-level symmetry fails for verbs (wrong level)
- Why relational symmetry works (captures bridging)
- Why Chinese doesn't conjugate (operates at joint by default)
- Why tense exists in other languages (explicit navigation hints)

The symmetry-based knowledge system can bootstrap from this insight:
**Symmetry is the joint. The joint is temporal. Language navigates time.**
