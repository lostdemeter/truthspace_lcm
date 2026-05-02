# DC 307 — The Gravitational Expedition

*Date: March 2026*
*Mode: Naturalist explorer. Record what you observe. Name what you find.*
*Precedes: DC 308, 309, ... (expedition logs)*

---

> "I had gradually come to see that the struggle for existence had done for organic
> beings what man had done for domestic productions: selected the best adapted forms."
> — Charles Darwin, reflecting not on a clever theory but on what he actually saw.

We are not starting from a theory. We are starting from two observations:

1. Our context gravity is non-local — and it's correct.
2. If gravity is the right word, then two centuries of gravitational physics are
   available as a map.

The question is not whether the analogy holds. The question is: **how far does
it hold, and where does it break?** Breakage is as informative as agreement.
Darwin found that the finches on different islands had different beak shapes.
He didn't expect that. We will find things we don't expect. That is the point.

---

## The Two New Ideas

### Idea A: Non-Locality as Compression

If "money" and "bank" are geometrically related regardless of whether they share
a sentence, then the relationship is *structural* — it exists in the geometry
and doesn't need to be stored in the token sequence or the model weights.

This means: concepts that are geometrically derivable from other concepts don't
need to be stored independently. Store the archetype. Store the transformation.
Derive the rest.

We already proved this works for individual pairs: king→queen, man→woman, all
country→capital pairs use the same Δ vectors. The compression is: instead of
one embedding per concept, store one *basis concept* per semantic family plus
one *transformation vector* per relationship type.

The extreme version: a "periodic table of concepts" — a finite basis set from
which all of language can be derived by composition and transformation. If
concept space has intrinsic dimension D, the basis has D elements and everything
else is a linear combination with small residuals.

**The key prediction**: a model built on archetypes + transformations would be
dramatically smaller than a model that stores all concepts independently, yet
produce the same semantic behaviour — because the *information* isn't in the
individual concept storage, it's in the geometry that relates them.

**What would falsify it**: if residuals are large — if most concepts require
many archetypes and large correction terms to reconstruct — then the basis is
no more compact than the original and the idea fails.

### Idea B: Gravitational Physics as an LLM Lens

Trained language models have implicit semantic gravity. They never name it,
measure it, or exploit it directly. The training process discovers the geometry
and encodes it into weights — but the geometric structure is then buried under
billions of floating-point numbers.

Two questions:
1. Which gravitational features *explain* how current LLMs work?
2. Which gravitational features do LLMs *not have*, and what would they unlock?

These are different questions. The first is diagnostic. The second is a design
proposal: what would a "gravitationally explicit" model look like?

---

## The Expedition Islands

We will visit these in order. Each island gets a day of experiments, a log
of observations, and a conclusion about what it means.

### Island 1: The Mass Spectrum
*What things weigh in concept space.*

Compute semantic mass for all 25,671 concepts under multiple definitions.
Find the heavyweights. Find the black holes. Find the vacuum. Map the distribution.

Does the mass distribution follow a power law (like galaxy masses)?
Are there "stellar" concepts (massive, stable, many satellites)?
Are there "rogue planets" (isolated, low-mass, no gravitational context)?

### Island 2: The Compression Coast
*How small can a basis get?*

Pick the highest-mass concepts as the archetype basis.
Reconstruct all other concepts as: archetype + correction.
Measure reconstruction error as a function of basis size.
Find the "knee" — the basis size at which error stops falling rapidly.

If the knee is at B=100, we need 100 archetypes to represent 25,671 concepts.
That's a 250× compression of the concept space.

### Island 3: The Orbital Archipelago
*Stable semantic solar systems.*

Find concept pairs with stable mutual orbits: both pull on each other, neither
escapes. Identify multi-body systems: clusters where removing one member
destabilises the others. These are the "solar systems" of concept space.

The hypothesis: every polysemous word is a binary star — two attractors in
mutual orbit. Disambiguation is choosing which star the sentence orbits.

### Island 4: The Escape Velocity Atlas
*How hard is it to leave home?*

Compute escape velocity for every polysemous concept: the minimum context
force to cross from the dominant-sense basin to the secondary-sense basin.
Map this as an atlas — some words are easy to redirect (low escape velocity),
some require overwhelming context (high escape velocity).

**LLMs don't have this.** They don't know how much context they need before
they're confident about a disambiguation. A gravitationally-explicit model would.

### Island 5: The Lensing Survey
*How massive words distort their neighbourhood.*

Pick the top-20 highest-mass polysemous words.
For each, measure: do their neighbours appear closer to multiple domains than
they should (lensing distortion)?
Compare neighbour-domain distribution for concepts near high-mass polysemous
words vs concepts far from them.

**If lensing is real**: this explains why some retrievals fail even for
non-polysemous words — they're in the gravitational shadow of a nearby massive
ambiguous word.

### Island 6: The N-body Stability Experiments
*Making sentences into solar systems, not black holes.*

Fix the Q3 collapse by introducing:
a) Differential mass (concepts weighted by gravitational mass from Island 1)
b) Semantic repulsion (antonym pairs repel with inverse-square law)
c) Angular momentum from syntax (subject→predicate→object as orbital direction)

Measure: does the N-body system converge to a stable structure instead of a
point? Does the structure encode sentence meaning better than the centroid alone?

### Island 7: The Geodesic Completions
*Inferring the unknown from the shape of the known.*

Given a set of known delta vectors (semantic transformations), can we complete
partial knowledge?

- Known: France→Paris, Germany→Berlin, Spain→Madrid
- Unknown: Portugal→?

Can we compute the geodesic in concept space that satisfies the known
examples, and follow it to the unknown position?

**LLMs interpolate but can't extrapolate along a geodesic.** A geometric model
can, because geodesics are derivable from the curvature of the manifold —
which is derivable from the data we have.

### Island 8: The Periodic Table
*What are the irreducible elements of meaning?*

After compression (Island 2), what are the archetype concepts that span the
space? Do they form a natural taxonomy? Are they discoverable from the geometry
alone (without external labels)? Does the number of archetypes match any known
cognitive or linguistic theory?

This is the deepest island. We may not reach it on this expedition.
It may require a second voyage.

---

## The Darwin Principle

Darwin didn't leave the Galápagos with a finished theory. He left with notebooks.
The theory came later, from the weight of accumulated observations.

We will follow the same discipline:

1. **Observe first.** Run the experiment. Print the numbers. Look at what the
   data actually says before deciding what it means.

2. **Name what you find.** If a pattern appears that doesn't have a name yet,
   give it one. The name should describe what it is, not what we hoped it would be.

3. **Record anomalies.** The most important observations are the ones that
   contradict expectations. If mass distribution is NOT a power law, that is a
   finding. If concepts don't form stable orbits, that is a finding.

4. **No graceful fallbacks.** If a prediction fails, we record the failure.
   We don't adjust the methodology to make it succeed. We adjust our understanding.

5. **Follow the geometry.** If the data points somewhere we didn't plan to go,
   we go there.

---

## What LLMs Are Missing

Running list, to be extended as we find things:

| Feature | Gravitational description | LLM status | Value if added |
|---|---|---|---|
| Semantic mass | How much a word anchors its region | Implicit, never measured | Disambiguation confidence, attention weighting |
| Escape velocity | Min context to change sense | Not available | Context sufficiency score |
| Gravitational lensing map | Which words distort their neighbourhood | Unknown, untested | Retrieval confidence correction |
| Orbital stability | Stable concept pairs, phrase structures | Not modelled | Structural phrase binding |
| Geodesic completion | Deriving unknown facts from manifold shape | Not available | Structured knowledge extension |
| Schwarzschild filter | Identify and skip semantic black holes | Not implemented | Attention efficiency |
| N-body sentence | Sentence as gravitational system, not sequence | Not modelled | Richer sentence representations |
| Periodic table | Basis decomposition of all meaning | Not available | Maximum compression, generative basis |

---

## The Compact Model Prediction

The compression idea (Idea A) makes a falsifiable quantitative prediction:

> The concept space of 25,671 IRD concepts can be represented by a basis of B
> archetypes plus N transformation vectors, where B + N << 25,671, with
> reconstruction error < ε.

Specifically: we predict B ≈ 100-500 and ε < 0.05 cosine error on 95% of
concepts.

If this holds: the entire vocabulary of a language model does not need to be
stored. Only the archetypes and transformations need explicit weights. Everything
else is derived. This is a ~50–250× compression of the embedding layer.

The deeper implication: if all concepts are derivable, **the "knowledge" in a
language model is not in the vocabulary weights — it's in the transformation
vectors**. The vocabulary is just an efficient lookup into the geometric basis.
This is the strongest version of the hypothesis: language models are
transformation stores, not fact stores.

---

## Files

Each island will produce:
- An experiment script: `expedition_dayN_*.py`
- An expedition log entry: appended to `expedition_log.md`
- A design consideration: DC 308, 309, ... as warranted

Starting: Day 1 — The Mass Spectrum.
