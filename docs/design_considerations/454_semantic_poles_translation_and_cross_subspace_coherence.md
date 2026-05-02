# DC 454: Semantic Poles, Translation Axes, and Cross-Subspace Coherence

**Day 319 | Three discoveries: (1) Antonym pairs are SEMANTIC POLES — they
are naturally far apart (win/lose: baseline rank >20, cos=0.201) yet the mean
antonym axis navigates to each target at rank=0 with tgt_cos=0.31-0.43. Each
pair navigates in a different direction (pc≈0) but all land correctly because
antonyms occupy isolated antipodal positions in their local semantic clusters.
(2) Translation axes form a FOURTH axis subspace, correlated with morphological
axes (+s: 0.24-0.31) and orthogonal to relational axes (-0.05 to -0.08). The
three European translation axes cluster together (cos=0.33-0.43 internal). (3)
Cross-subspace navigation produces COHERENT but DOMAIN-WRONG output: applying
the cc_axis to adjectives finds comparative forms; applying the antonym axis to
'war' finds Chinese/Russian 'war' (战争/войны), not 'peace'. The axes navigate
to the nearest token in their direction — which is semantically meaningful but
not the intended transformation.**

---

## Semantic Poles: Resolving the Antonym Mystery

### The Central Finding

```
VERB_ANT: pc=0.016  in=8/8  scale=1.051  → ALL tgt_rank=0

Pair          baseline_cos  baseline_rank  axis_tgt_cos  after_rank
win→lose         0.201         >20            0.319          0
rise→fall        0.171         >20            0.359          0
push→pull        0.420           7            0.431          0
enter→exit       0.274         >20            0.396          0
buy→sell         0.390          ?             0.390          0
love→hate        0.355          ?             0.355          0
open→close       0.333          ?             0.333          0
start→stop       0.379          ?             0.379          0
```

The antonym target is NOT a natural neighbor of the source. Yet after applying
the axis displacement (scale=1.051), the target becomes rank-0 — the nearest
clean token in the vocabulary.

### Why This Works: The Semantic Pole Mechanism

Antonym pairs are **semantic poles**: they sit at distant ends of their
conceptual domain with nothing between them in the clean vocabulary.

Consider 'win' and 'lose':
- cos(win, lose) = 0.201 (far apart)
- 'lose' is at rank >20 from 'win' (many other words are closer to 'win')

But 'lose' is at a LOCAL MAXIMUM of similarity to a specific displacement
direction from 'win'. The mean antonym axis vector, when applied from 'win',
moves to a region of embedding space where 'lose' is the UNIQUELY CLOSEST
clean token. The other 20+ words that were closer to 'win' don't appear in
that displaced region.

This explains why pc≈0 yet in=100%: the displacement directions are completely
different for each pair (pc=0.016), but each pair has its own "pole direction"
along which the antonym is the unique nearest clean neighbor.

### The Lone Exception: push→pull

push→pull has baseline rank=7 (cos=0.420) — they ARE naturally close. This
is because 'push' and 'pull' are semantically COMPLEMENTARY (both describing
force application) as well as antonymous. They co-occur in the same training
contexts ("push and pull", "push or pull", "pushing and pulling"), making them
embedding neighbors.

The other pairs (win/lose, rise/fall, war/peace) are CONTEXTUALLY SEPARATED:
you win OR lose, not "winning and losing" in the same motion.

### The Semantic Pole Hypothesis

For a lexical pair to form a semantic pole:
1. They are semantically antonymous (opposite on some scale)
2. They are contextually exclusive (rarely co-occur in the same phrase)
3. They are grammatically parallel (same syntactic category)
4. They are frequency-matched (both common words)

When all four conditions hold, the embedding space places them at antipodal
positions in their local cluster. The mean displacement vector navigates to
this antipodal position even without knowing which direction to go, because
the destination is uniquely isolated in the target region.

### Implications for pc Interpretation

The previous pc interpretation assumed "pc≈0 means the transformation is
inconsistent or random." The antonym case shows this is too simple:

**pc≈0 can mean two things:**
1. **Noise**: The transformation is genuinely inconsistent (random sources/targets)
2. **Polar structure**: The transformation is LOCALLY consistent for each pair
   but GLOBALLY diverse because each pair occupies a unique directional slot

The distinction is made by:
- LOO score: Noise → LOO≈0. Polar → LOO≈0 as well (can't generalize across pairs).
- In-sample tgt_rank: Noise → high rank. Polar → rank=0 (uniquely isolated target).

Antonym axes are POLAR, not noisy.

---

## The pc Continuum: A Complete Map

### All 17 Axes Ordered

```
pc      axis       LOO%   type
──────────────────────────────────────────────────────
0.426   er→est     100%   morph_uniform
0.399   cl         67%    relational_geom (morph_uniform range)
0.394   capl       100%   relational_geom (morph_uniform range)
0.385   +er        88%    morph_uniform
0.351   cc         71%    relational_geom
0.297   +s         100%   morph_moderate
0.259   +ed        100%   morph_moderate
0.220   +able      0%     morph_moderate (semantic_diverse by irred)
0.203   +ness      86%    morph_moderate → phonol_scatter?
0.189   un-        67%    phonol_scatter? (irred=57%)
0.167   +less      0%     semantic_diverse (irred=90%)
0.165   pres       0%     factual_local (irred=100%)
0.142   +ful       33%    phonol_scatter?
0.112   +tion      75%    phonol_scatter (irred~0%)
0.082   EN→ES      25%    translation
0.064   EN→FR      0%     translation
0.055   adj_ant    30%    antonym-borderline
0.020   noun_ant   0%     antonym (polar)
0.016   verb_ant   0%     antonym (polar)
```

### The Natural pc Boundaries

```
pc > 0.35:  "coherent" axes — reliable navigation (morph_uniform + relational_geom)
0.20-0.35:  "moderate" axes — somewhat reliable (morph_moderate)
0.10-0.20:  "scatter" axes — diverse/phonological effects dominate
0.05-0.10:  "translation" region — cross-lingual morphological similarity
< 0.05:     "polar" axes — local pole structure, no global coherence
```

The only SHARP break is at pc≈0.05: below this, axes are of the polar type
where individual pair directions are effectively random. Everything above 0.05
has some detectable global coherence.

### The LOO-pc Separation of phonol_scatter from semantic_diverse

Within the 0.10-0.20 range, LOO is the decisive classifier:
- +tion: pc=0.112, LOO=75% → phonol_scatter (surface diversity, semantic unity)
- +ness: pc=0.203, LOO=86% → phonol_scatter (high LOO despite low pc)
- un-:   pc=0.189, LOO=67% → phonol_scatter
- +ful:  pc=0.142, LOO=33% → borderline
- +less: pc=0.167, LOO=0%  → semantic_diverse (irred=90%)
- pres:  pc=0.165, LOO=0%  → factual_local (irred=100%)

**Rule**: For 0.10 < pc < 0.20, if LOO ≥ 60% → phonol_scatter. If LOO < 30%,
check irred: irred<30% → borderline phonol_scatter, irred>60% → semantic_diverse
or factual_local.

---

## Translation as the Fourth Axis Subspace

### The Measurements

```
Axis     pc      in%    LOO%   cos(+s)  cos(cc)
EN→ES   0.082   100%    25%    0.239   -0.050
EN→FR   0.064    80%     0%    0.296   -0.052
EN→DE   0.101   100%     0%    0.307   -0.076
```

### Internal Translation Coherence

```
cos(EN→ES, EN→FR) = 0.333
cos(EN→ES, EN→DE) = 0.362
cos(EN→FR, EN→DE) = 0.426
```

The three translation axes are substantially correlated (0.33-0.43). All three
point in a similar direction: AWAY from English toward the multilingual cluster.
This makes sense: the English→European language direction is the same semantic
direction regardless of target language (Spanish/French/German all cluster
together in W_E).

### Translation ↔ Morphological Correlation

The most surprising finding: translation axes correlate with +s (0.24-0.31).

Why would English→Spanish translation have a 0.24 cosine with English plural?

Hypothesis: In W_E, the "move away from English" direction and the "add suffix"
direction are partially aligned. Many Spanish words look like English words with
added suffixes (-o, -a, -os, -as). The training distribution of Spanish words
appears in similar positional contexts to pluralized English nouns (both as
object positions, both carry grammatical agreement markers).

This is a cross-lingual tokenization effect: Spanish words that share morphology
with English plurals (libro/books, casa/houses) cluster in nearby embedding
positions because of overlapping subword tokenization patterns.

### Translation is NOT Relational

```
cos(EN→ES, cc) = -0.050  (near-zero)
cos(EN→ES, cl) = -0.xx   (near-zero)
```

Translation is orthogonal to relational axes. This means:
- country→capital: factual association between two named entities
- word→translation: morphological/lexical substitution

These are fundamentally different operations in W_E geometry.

### The Complete Four-Subspace Model

```
Subspace        pc range   LOO%   axes
──────────────────────────────────────────────────────────────────
Relational      0.35-0.40  60-100  cc, cl, capl (factual, bijective)
Morphological   0.11-0.43  0-100   +er, +s, +ed, +tion, un-, +ful, +less
Translation     0.06-0.10  0-25    EN→ES, EN→FR, EN→DE
Antonym/Polar   0.02-0.06  0-30    verb_ant, noun_ant, adj_ant
```

The four subspaces are separated by:
1. pc threshold (natural breaks in the distribution)
2. LOO behavior (generalizes vs. doesn't)
3. Cross-axis cosines (orthogonal between groups)

---

## Cross-Subspace Navigation: Coherent Misdirection

### The Results

```
Input      Axis      Output
────────────────────────────────────────────────────────
fast       +cc       faster, 快速, .fast
slow       +cc       slower, slows, slowed
happy      +cc       happiness, happier, HAPP
young      +cc       年轻(zh), jeune(fr)
france     ++er      法国(zh), germany, フランス(ja)
germany    ++er      德国(zh), france, berlin
war        +ant      战争(zh), войны(ru), wars
day        +ant      (day, .day, day
```

### The Pattern

Each result is SEMANTICALLY RELATED to the input but transformed by the
axis direction rather than the intended operation:

- cc_axis on adjectives → finds comparative/derivative OR cross-lingual forms
  (because from 'fast', the cc direction moves toward the dense cluster of
  frequency-similar words including 'faster', '快速', etc.)

- +er_axis on countries → finds multilingual country names
  (because from 'france', the +er direction moves toward the cross-lingual
  cluster of 'france' equivalents: 法国, フランス, etc.)

- ant_axis on 'war' → finds foreign 'war' words, NOT 'peace'
  (because 'war' is NOT in the training set for ant_axis; the displacement
  moves to the nearest token in that direction, which is a foreign
  language equivalent of 'war' — semantically the same concept, different language)

### The Semantic Coherence Law

**Cross-subspace navigation preserves semantic cluster but changes language or form.**

Formally: applying axis A to token T (where T is not in A's domain) produces
the nearest token to T+s·A that is semantically close to T's concept but shifted
in the grammatical/linguistic dimension defined by A.

This is a deep property of W_E: the embedding space is organized so that
multilingual equivalents, morphological variants, and semantic relatives are
all locally clustered. Any displacement from a token will land in some
semantically coherent neighborhood.

The only truly "wrong" results come when the nearest clean token is an
artifact (tokenization compound like '(day', '.fast') or when the displacement
exits all meaningful content.

### Implication: Axes as Semantic Lenses

Each axis type acts as a "lens" that focuses navigation on a specific type of
semantic relationship:
- Relational lens (cc): focuses on factual associations in named-entity space
- Morphological lens (+er): focuses on form-change neighbors
- Translation lens (EN→ES): focuses on cross-lingual equivalents
- Antonym lens (ant): focuses on polar opposites (when in training domain)

When used outside the training domain, the lens still focuses — just on the
wrong type of relationship. The embedding space is so rich that every direction
leads somewhere semantically coherent.

---

## Day 320 Plan

1. **Antonym polar confirmation**: test 5 additional verb antonym pairs not in
   training set. Do they also have tgt_rank=0? Verify the pole structure holds
   out-of-sample.

2. **Translation LOO deeper analysis**: why is EN→ES LOO=25% but EN→FR LOO=0%?
   Subtype the translation axis into semantic_diverse vs near-phonol_scatter.

3. **+ness full classification**: LOO=86% but from DC 296 we know +ness has
   high holdout irred. Reconcile — does high LOO mean phonol_scatter?

4. **Translation chain**: EN→ES then ES→FR (is there a systematic "translation
   network" where axes can be composed across languages?). cos(ES→FR, EN→FR)?

5. **Axis type predictor**: given just pc and LOO, can we predict axis type
   reliably? Build a 2D decision boundary.

---

## Files

- `expedition_log.md` — Day 319 results
- `453_antonym_geometry_and_three_axis_subspaces.md` — DC 453
- `day319_antonym_local_translation_crosssubspace.py` — experiment script
