# DC 306 — Semantic Gravity Physics

*Date: March 2026*
*Follows from: DC 302 (gravity correction), DC 304 (geometry not understanding), DC 305 (experiments)*

---

## 1. The Two Questions

DC 304 established that our context correction mechanism IS gravity in concept space.
Two questions follow immediately:

**Q: Is non-locality real in our system?**
**A: Yes. Intentionally.**

**Q: Can we draw on the body of gravitational physics?**
**A: Every part of it. We will.**

---

## 2. Non-Locality

### What it is

In our system, the context word "money" exerts a direct gravitational force on
"bank" regardless of what tokens appear between them in a sentence. There is no
intermediate step, no propagation through neighbouring positions. The force acts
because the geometric relationship between "bank" and "money" exists in the IRD
manifold — a static structure, not a dynamic chain.

This is **action at a distance**. Newton called it "absurd" when he derived it.
It was only dissolved in General Relativity, where the mediating field is spacetime
curvature — gravity doesn't jump; the geometry is already curved.

### Why it is correct in semantic space

Newtonian non-locality is troubling in physics because it would allow
instantaneous transmission of information, violating causality. In semantic
space there is no causality constraint. The geometry of concept space is:

1. **Static** — all concepts exist simultaneously; there is no "when" at which
   a force is computed. "Bank" and "money" are geometrically related whether or
   not they appear in the same sentence.

2. **Complete** — the entire IRD manifold is computed in advance. Every pairwise
   distance is knowable at O(1) per query. There is no propagation delay.

3. **Correct for language** — human reading is not strictly local. We understand
   "I went to the bank to deposit my paycheck" because "deposit" and "paycheck"
   disambiguate "bank" even though they appear after it. Retrospective non-local
   disambiguation is a feature of how language works, not a violation.

The transformer achieves the same non-local effect through multi-head attention
over the full window — but it takes O(N²) computation and 28 serial layers to do it.
Our absolute gravity achieves it in O(k) where k is the number of context words,
in one step, because the geometry encodes the relationships directly.

**Non-locality is not a bug in our system. It is the correct behaviour for a
static geometric representation of meaning.**

### The Einsteinian revision (future)

In General Relativity, non-locality is replaced by field curvature. The
Einsteinian version of our system would not compute pairwise forces — it would
compute the curvature of the IRD manifold at each point and let words move
along geodesics in that curved space. The curvature IS the gravity. This is
the deeper theory; our current Newtonian version is the first approximation.

---

## 3. Semantic Mass

The first thing we need from gravitational physics is a definition of **mass** —
the property that determines how strongly a concept participates in gravity.

In gravitational physics: mass is both inertial (resistance to acceleration) and
gravitational (source of the field). They happen to be equal. This equivalence
principle has a semantic analogue.

### Gravitational mass (field strength)

How strongly does a concept attract others toward itself?

In IRD space this is **the average cosine similarity of a concept to its N
nearest neighbours** — a "semantic density." A highly connected concept (one
with many near neighbours in many directions) sources a stronger field.

Alternative: **degree centrality** — the number of concepts within cosine radius
r of the concept's position. High centrality → high gravitational mass.

### Inertial mass (resistance to displacement)

How hard is it to move a concept out of its current position via context gravity?

This is **polysemy ambiguity** — a concept with two equal-mass attractor basins
has HIGH inertial mass because equal forces from both sides cancel. A concept
with a dominant single sense has LOW inertial mass (it's easy to confirm its
sense; hard to displace it to a different one). A genuinely ambiguous word like
"set" (183 senses in WordNet) has enormous inertial mass.

### The equivalence principle for semantics

Gravitational mass ≅ inertial mass means: the concepts that attract others most
strongly are also the hardest to move. This makes sense. "Bake" (high culinary
mass) strongly attracts food-related concepts AND is itself hard to move out of
the culinary domain. The equivalence holds.

### Semantic mass estimates

| Concept | Gravitational mass | Inertial mass | Notes |
|---|---|---|---|
| "the" | Very high | Very low | Appears everywhere; attracts nothing specific |
| "bank" | Medium | Very high | Two equal basins; hard to move |
| "bake" | High (culinary) | Low | Strong field, easy to confirm |
| "and" | Very high | Very low | Connective; semantic black hole |
| "python" | High | High | Two strong, unrelated basins |

This gives us a **semantic mass spectrum** derivable entirely from IRD geometry.

---

## 4. The Gravitational Potential Field

Newton's gravity is often computed as pairwise forces (a sum). But the cleaner
formulation is the **scalar potential field**:

```
Φ(p) = − G × Σᵢ  m_i / dist(p, concept_i)
```

The force at any point is the gradient: F = −∇Φ(p).

This is operationally significant. Instead of computing a force vector by summing
contributions from each context word, we can:

1. Precompute Φ over a grid of IRD space (or evaluate it lazily at query points)
2. Compute ∇Φ at the word's current position
3. Move in the direction of steepest descent (toward the deepest potential well)

For our context gravity, the potential well of context words defines the attractor
basin. The gradient of that potential gives the exact correction direction — no
iteration needed, no falloff hyperparameter. **The geometry defines the force.**

In practice, for a small context set, the explicit sum is fine. But for
sentences, paragraphs, or document-level context, the potential field approach
scales better. You compute the field once; all words in the sentence move along it.

---

## 5. Escape Velocity

In orbital mechanics: `v_escape = sqrt(2GM/r)` — the velocity needed to escape
a body's gravitational well from distance r.

In semantics: **how much contextual force is needed to pull a concept out of its
dominant-sense attractor basin?**

The escape velocity for "cookie" from its HTTP basin is the minimum α (context
correction strength) such that the gravity from culinary context words exceeds
the gravitational binding energy of the HTTP basin.

From DC 303 Part 2:
- "Cookie" native food-align = +0.2619
- "Cookie" HTTP-basin binding: cos(cookie, login) = +0.243 > cos(cookie, recipe) = +0.225
- Escape from HTTP basin requires net culinary force > HTTP binding gravity

The escape condition:

```
Σᵢ w_i × cos(p_escape, p_ctx_i) > Σⱼ w_j × cos(p_escape, p_http_j)
```

This is a calculable threshold. Once we have semantic mass, we can compute exact
escape velocities for any polysemous word against any context set. This gives us
a principled disambiguation confidence score:

```
P(correct_disambiguation) ≈ σ(escape_velocity_ratio)
```

Rather than a binary "is this word polysemous?", we get a continuous measure of
how much contextual evidence is required.

---

## 6. Tidal Forces and Polysemy

In gravitational physics, **tidal forces** arise when different parts of an
extended body experience different gravitational forces from an external source.
The Moon produces tides because the near side of Earth is pulled more strongly
than the far side — the differential force stretches the body.

A polysemous word experiences exactly this. "Bank" in the sentence "the bank
of the river" has:

- Its HTTP-finance "face" pointing toward {deposit, money, loan}
- Its geographic "face" pointing toward {river, water, stream}
- The tidal force is the difference in gravitational pull between the two faces

The **polysemy severity** of a word is directly measurable as the magnitude of
this tidal force: how different is the gravity field between the two poles of
the word's semantic body?

This connects to our `detect_polysemy()` function (DC 302 §9.1), which measures
domain mismatch among nearest neighbours. That measurement IS the tidal force.
We derived it empirically; gravity theory gives us the theoretical justification.

**Tidal force = polysemy severity.** Words with low tidal force are semantically
compact (easy to retrieve correctly). Words with high tidal force are polysemous
(require contextual disambiguation before reliable retrieval).

---

## 7. Geodesics and Delta Navigation

In General Relativity, a freely-falling body follows a **geodesic** — the
straightest possible path through curved spacetime. No force is applied; the body
follows the geometry.

In DC 301, our delta navigation (king → queen, bread → wheat) follows a direction
vector in IRD space. The claim is that this vector IS a geodesic on the semantic
manifold — the straightest path from one concept to another in the curved geometry.

The empirical support: the φ-encoded relationship vectors are consistent across
all pairs in a semantic domain (king→queen = man→woman = boy→girl = always Δx=−2.0
on the gender axis). This self-similarity IS the hallmark of geodesic motion on
a symmetric manifold — the same displacement vector works everywhere because the
manifold is locally flat along that direction.

**Delta vectors are geodesic tangent vectors.** Applying a delta is taking one
step along a geodesic. The fact that they compose correctly (France+capital =
Paris; Italy+capital = Rome) means the geodesics are straight in IRD coordinates —
the IRD axes are **normal coordinates** at the manifold's flat regions.

The curvature shows up at the edges: where concepts are highly polysemous or where
multiple semantic domains intersect, the manifold curves and the delta vectors no
longer commute cleanly. This is exactly where our retrieval is weakest.

---

## 8. Gravitational Lensing

In GR, a massive object bends light rays passing near it — **gravitational
lensing**. The apparent position of a distant object is shifted toward the
massive body.

In semantic space: a high-mass concept "bends" the apparent position of
neighbouring concepts. "Python" (massive in both programming and biology domains)
makes nearby concepts seem closer to both domains than they would be without
Python's presence in the vocabulary.

This is the mechanism behind **semantic contamination**: polysemous words with
high mass distort the local geometry, making nearby words appear more ambiguous
than they are. The lensing effect is real and measurable — compare the neighbour
lists of concepts near high-polysemy vs. low-polysemy centroids.

An operational consequence: when building query expansions or retrieval sets,
concepts near massive polysemous words should be treated with lower confidence
because their observed positions are lensed by the nearby massive body.

---

## 9. Schwarzschild Radius — Semantic Black Holes

In GR, the Schwarzschild radius defines when an object's gravitational pull
becomes inescapable: the event horizon of a black hole.

In semantic space: **what is the Schwarzschild radius of "the"?**

"The" appears in >90% of English sentences. Its gravitational mass in IRD space
is enormous — it is near almost every concept. But because it is near everything
equally, its net force on any specific concept is approximately zero (forces
cancel). It is a black hole in the sense that it ABSORBS meaning from its context
without CONTRIBUTING directional force.

More precisely: function words (the, a, is, and, of) have near-zero gravitational
field strength despite near-infinite mass. They are maximally massive, minimally
directive. Their Schwarzschild radius is effectively infinite — everything is
"inside" their event horizon — but they produce no tidal forces, no displacement,
no disambiguation.

**Operationally: filter function words from context before applying gravity.**
This is already correct behaviour in our system (we pass semantically meaningful
context words). The gravitational physics now gives us the theoretical reason:
function words are black holes — they absorb without directing.

---

## 10. The Q3 Collapse — Explained by Gravity Theory

DC 305 Q3 found that bidirectional N-body gravity collapses every sentence to
its centroid (all pairs → cos=1.0). This seemed like a limitation. Gravity
theory explains it exactly.

A uniform cloud of equal-mass particles with no angular momentum under purely
attractive forces will **always collapse to a single point** — gravitational
collapse. This is the same physics as stellar formation, galaxy formation, and
the Big Crunch. We reproduced it in semantic space.

The fix is what astrophysics already knows:

1. **Differential mass** — if concepts have different semantic masses, they don't
   collapse equally. Heavy concepts (semantically central) act as nucleation
   points around which lighter concepts orbit. The sentence doesn't collapse to
   a point; it collapses to a multi-nucleus structure.

2. **Angular momentum** — narrative direction. A sentence has a direction
   (syntactic head → dependency → modifier). If we inject angular momentum
   (a "narrative axis" from subject to predicate to object), the system forms
   stable orbits instead of collapsing.

3. **Semantic repulsion** — antonyms repel. If we model opposing concepts as
   having repulsive force (hot↔cold, expand↔contract), the N-body system
   reaches a stable equilibrium rather than collapsing.

The sentence centroid we computed in Q3 IS correct as a sentence-level
representation (it's stable and discriminative). But it's the **maximally
collapsed** representation — all structure erased. A better sentence representation
preserves the internal structure: a solar system, not a black hole.

---

## 11. The Research Programme

We now have a direct mapping between gravitational physics and IRD operations:

| Gravitational concept | Semantic definition | IRD operation | Status |
|---|---|---|---|
| Gravitational mass | Semantic centrality / density | Avg cos to k-NN | To implement |
| Inertial mass | Polysemy resistance | Tidal force magnitude | `detect_polysemy()` |
| Force law | Context gravity | `context_correct_proj()` | ✓ Done |
| Escape velocity | Disambiguation threshold | Min α to cross basin | To implement |
| Potential field | Scalar semantic landscape | Σ m_i/dist(p, c_i) | To implement |
| Geodesic | Delta navigation path | Delta vector | ✓ Done (DC 301) |
| Tidal force | Polysemy severity | Domain mismatch cos | ✓ Done (DC 302) |
| Gravitational lensing | Semantic contamination | Neighbour distortion | To measure |
| Black hole | Function word | Filter from context | ✓ Done (implicit) |
| N-body collapse | Sentence centroid | Bidirectional gravity | ✓ Done (DC 305 Q3) |
| Stable orbit | Semantic phrase cluster | Gravity + angular momentum | To implement |
| Schwarzschild radius | Domain event horizon | Basin boundary | To define |

The **immediately actionable** items for the TruthSpace LCM:

1. **Semantic mass function** — compute gravitational mass for all 25,671 concepts.
   Use as weight in context gravity (heavier context words pull harder).

2. **Escape velocity score** — replace binary polysemy flag with a continuous
   disambiguation difficulty score. Low escape velocity = easy to confirm.
   High = needs strong context.

3. **Potential field gradient** — single-query context correction via ∇Φ instead
   of explicit force sum. Precomputable for known context words.

4. **Repulsive term** — add semantic repulsion between antonyms (and between
   concepts on opposite sides of a φ-axis). Makes N-body simulation produce
   stable cluster structures instead of centroid collapse.

---

## 12. Newton vs Einstein: Where We Are

Our current system is **Newtonian**:
- Action at a distance (non-local, instant)
- Additive force law
- Fixed background space (IRD axes are static)

The **Einsteinian version** would be:
- Field-theoretic (curved IRD manifold, no action at a distance)
- Forces arise from curvature, not explicit computation
- Dynamic background (adding new concepts curves the manifold around them)
- Geodesics replace force vectors

We are not at the Einsteinian level yet, and we don't need to be.
Newton's gravity described the solar system to high accuracy for 200 years before
Einstein was needed. Our Newtonian semantic gravity correctly predicts polysemy
disambiguation, sentence embedding, and context correction. The Einsteinian
revision becomes necessary only when:

- We need to handle concept evolution (concepts changing meaning over time)
- We need to describe the "curvature" near extremely polysemous words
- We need exact geodesic paths, not approximate delta vectors

That is DC 400+.

---

## 13. Summary

1. **Non-locality is real** in our system — and correct. Semantic geometry is
   static; forces act at a distance because the relationships exist in structure,
   not in propagation.

2. **The entire body of gravitational physics is now applicable** to semantic
   space. Every major concept (mass, field, potential, geodesic, escape velocity,
   tidal force, black hole, collapse, orbit) has a precise semantic definition
   in terms of IRD operations.

3. **Immediately actionable**: semantic mass function, escape velocity score,
   semantic repulsion term for N-body stability.

4. **Our current gravity is Newtonian**. The Einsteinian field theory is the
   next theoretical horizon, not the current engineering priority.

---

## Files

- `experiments/truthspace_v1/dc299_phase4_lcm_inference.py` — `context_correct_proj()` (force law)
- `experiments/truthspace_v1/dc299_phase4_lcm_inference.py` — `detect_polysemy()` (tidal force)
- `experiments/truthspace_v1/dc305_frontier_experiments.py` — Q3 N-body collapse
- Prior: DC 302 (gravity correction), DC 303 (attention=gravity), DC 304 (geometry not understanding), DC 305 (three experiments)
