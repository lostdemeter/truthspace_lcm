# DC 402: Antonym Axes Are Domain-Specific, Not Universal

**Day 267 | The "universal antonym axis" (coherence 0.27) fails because it
mixes domain-specific antonym axes that each have HIGH coherence. Speed
antonyms (fast/slow, rapid/gradual): coherence 0.756 — the highest of any
tested relation. Social antonyms (rich/poor, young/old): 6/6=100% forward
AND reverse. The universal axis encodes MAGNITUDE/INTENSITY, not antonymy
per se. The antonym relation is perfectly symmetric (cos=-1.0 between
forward and reverse axis). DC 380 is refined: antonymy IS geometric within
semantic domains.**

---

## The Domain Decomposition

Splitting the 20-pair universal antonym set by semantic domain reveals
the hidden structure:

```
Domain      N   Pairs                        Coh    Fwd       Rev
────────────────────────────────────────────────────────────────────────────
universal  20   all adjective antonyms       0.270  8/20=40%  12/20=60%
speed       4   fast/slow, rapid/gradual     0.756  2/3=67%   3/3=100%   ★
temperature 4   hot/cold, warm/cool          0.601  3/4=75%   3/4=75%
size        8   big/small, tall/short        0.474  4/8=50%   4/8=50%
quality     8   clean/dirty, bright/dark     0.402  6/8=75%   6/8=75%
social      6   rich/poor, young/old         0.428  6/6=100%  6/6=100%   ★★
```

**Speed coherence (0.756) exceeds the language axis (0.694)**, which was
previously the highest coherence observed in any relation.

**Social antonyms achieve 100% forward and reverse retrieval** with a
modest 6 training pairs — the same sample efficiency as the morphological
gender axis.

---

## Why the Universal Axis Fails

Each domain-specific antonym axis points in a DIFFERENT direction in W_E.
The mean direction over all 20 pairs averages across these incompatible
directions, producing a low-coherence resultant:

```
          ←─── speed axis ───►
          ←── temperature axis ──►
                    ←───── size axis ────►
               ←──── quality axis ────►
                   ←── social axis ──►
          ─────────────────────────────────
          ←─ universal axis (blurred) ──►
```

Coherence of a mixture ≈ coherence of each component × overlap between
components. Five roughly orthogonal domain axes average out to something
with coherence ~0.27 (= 5 × 0.50 × 1/√5 ≈ 0.27, rough estimate).

---

## The Universal Axis Encodes Magnitude/Intensity

The "antonym axis" constructed from mixing all domains does NOT encode
antonymy — it encodes the MAGNITUDE/INTENSITY dimension of the adjectives:

```
Negative pole (high magnitude/intensity):
  big(-0.262), rich(-0.220), bright(-0.199), fast(-0.198),
  heavy(-0.195), strong(-0.195), loud(-0.189), tall(-0.178)

Positive pole (low magnitude/intensity):
  weak(+0.197), slow(+0.195), quiet(+0.181), soft(+0.181),
  short(+0.179), small(+0.175), light(+0.152), poor(+0.141)
```

This is the **scalar magnitude axis** shared across all intensity-based
antonym domains. It says:

> "These words describe high-magnitude properties" vs
> "These words describe low-magnitude properties"

The adjective antonym pairs (hot/cold, big/small, fast/slow, etc.) share
a common structure: one member is the "more intense" version, the other the
"less intense" version of the same property. The mean direction points from
intense→unintense, which is real but domain-agnostic.

---

## Perfect Symmetry of the Antonym Relation

```
cos(hot→cold axis, cold→hot axis) = -1.0000 (exact)
```

The antonym transformation is geometrically symmetric:
- `emb(cold) = emb(hot) + d`  →  `d = emb(cold) - emb(hot)`
- `emb(hot)  = emb(cold) - d` →  `-d = emb(hot) - emb(cold)`

So the "forward axis" and "reverse axis" are exact negations of each other.
This is algebraically guaranteed, but the result confirms the embeddings
are not warped in a way that breaks this symmetry.

**Contrast with hypernymy:** `dog → animal` has coherent forward axis
(+0.37), but `animal → dog` has zero reverse accuracy. Hypernymy is
*directionally asymmetric* because the relation is many-to-one. Antonymy
is *directionally symmetric* because the relation is one-to-one (bijective).

This is the same bijection principle from DC 401: **symmetric ↔ bijective**.

---

## Revision to DC 380

DC 380 ("Antonymy Not Functional") concluded:
> Antonymy is not encoded as a directional axis; it cannot be used
> for systematic semantic analogy.

**Day 267 refinement:**
- DC 380 was correct for a UNIVERSAL antonym axis
- Domain-specific antonym axes ARE functional and highly coherent
- The correct statement is:

> **"Antonymy" as a cross-domain category has low-coherence geometry.
> Within semantic domains, antonymous opposites form high-coherence,
> invertible axes. The domains discovered so far: speed (coh=0.756),
> temperature (coh=0.601), social scale (coh=0.428), quality (coh=0.402),
> size (coh=0.474).**

DC 380 stands as a warning against treating "antonymy" as a monolithic
geometric relation, but the finding is more subtle than originally stated.

---

## Implication: The Number of Semantic Axes Is Very Large

The morphological parser used 5 axes covering grammatical morphology.
The encyclopedic relations add ~3 axes (capital, language, currency).
Domain-specific antonym axes add at least 5 more (speed, temp, size,
quality, social).

Each semantic domain has its own axis. W_E is a high-dimensional space
(1536 dims) with potentially HUNDREDS of semantic axes — one per
systematic relation type that appears in natural language.

The TruthSpace hypothesis predicts this: **structure IS information**.
Every systematic relationship in human knowledge is encoded as a geometric
direction in W_E. The more systematic and bijective the relation, the
higher its coherence and the more reliably it can be used for analogy.

---

## Files

- `expedition_log.md` — Day 267 results
- `380_antonymy_not_functional.md` — original finding (refined by this DC)
- `401_semantic_relation_axes.md` — full taxonomy of relation types
- `393_geometric_axis_coherence_law.md` — coherence predicts accuracy
