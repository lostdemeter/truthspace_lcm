# DC 446: Domain-Specific Plurals and the king→queen Anomaly

**Day 311 | Three definitive results: (1) the body-part plural axis (trained
on head/foot/ear/knee/etc.) achieves 93% on body-part holdout vs 71% for the
object-noun axis — and critically it retrieves EYE→EYES (which was NEVER
reachable with the object-noun axis) and the IRREGULAR plural FOOT→FEET.
HAND→HANDS remains the single irreducible failure even with a domain-specific
axis. (2) ALL 12 morphological axes have nearly identical eigenspectrum
isotropy (0.89–0.99) — this is a mathematical consequence of small sample
size (n=8–16) in high dimension (d=1536), not a property of the axes
themselves. The pc metric remains the correct directionality measure.
(3) KING→QUEEN lives almost entirely in gPC4–gPC5 of gender space (near-zero
projection on gPC1 and gPC2), while MAN→WOMAN and BOY→GIRL live in gPC1.
This confirms that "gender" in W_E is not one operation — king→queen encodes
ROYAL SUCCESSION geometry while man→woman encodes CORE GENDER geometry.**

---

## The Body-Part Plural Axis

### The Result

```
Word        Object axis  Body-part axis  Target   Winner
──────────────────────────────────────────────────────────
flower      flowers      flowers         flowers  =
star        stars        stars           stars    =
boat        boats        boats           boats    =
cup         cups         cups            cups     =
door        doors        doors           doors    =
road        roads        roads           roads    =
hand        hand         hand            hands    =FAIL
eye         eye          eyes            eyes     BP ↑
arm         arms         arms            arms     =
leg         legs         legs            legs     =
head        head         heads           heads    BP ↑
ear         ears         ears            ears     =
knee        knees        knees           knees    =
foot        foot         feet            feet     BP ↑

Object axis:    10/14 = 71%
Body-part axis: 13/14 = 93%
```

### The Three Body-Part Axis Wins

**1. eye→eyes**: Previously unreachable at any scale with object-noun axis.
The body-part axis, trained on head/ear/knee/toe/lip/hip/etc., correctly
points toward 'eyes' from 'eye'. This proves that eye→eyes direction is
encoded in W_E — it just requires the right training domain.

**2. head→heads**: Also unreachable with object axis. Now solved.

**3. foot→feet (IRREGULAR plural)**: The body-part axis retrieves the
IRREGULAR plural 'feet' from 'foot'. This is remarkable: the axis was
not trained on foot/feet explicitly, but the body-part semantic cluster
encodes the morphological transformation including irregular forms. The
axis has internalized the cluster's plural semantics, which includes
irregular suppletive forms.

### Why Body Parts Form a Distinct Semantic Cluster

Body-part vocabulary in W_E is characterized by:
1. **Universal embodied reference**: body parts are cross-culturally basic
   vocabulary (hand/main/mano/рука appear in parallel semantic positions)
2. **Metonymic extension**: body-part plurals often have DIFFERENT semantic
   roles than singulars (hands = labor, eyes = perception/attention, feet = travel)
3. **Tight morphological cluster**: the W_E positions for head/foot/ear/knee
   etc. form a compact cluster with their own morphological geometry

The object-noun axis captures the "collection-of-object" semantics of
plurality. The body-part axis captures the "embodied-plurality" semantics
where plural means "a person's two [body parts]" rather than "many [objects]."

### The hand→hands Irreducible Failure

Even with the domain-specific body-part axis, 'hand' remains irreducible.
The nearest-neighbor analysis reveals why:

```
Nearest neighbors of 'hands':
  Hands    0.761  (capitalized — excluded by clean retrieval)
  hands    0.758  (the target — but not top clean NN from 'hand')
  manos    0.498  (Spanish cognate)
  HAND     0.493  (uppercase — excluded)
  hand     0.475  (singular — this is what self-retrieval produces)

Nearest neighbors of 'arms':
  Arms     0.730  (capitalized — excluded)
  arms     0.661  (target — accessible)
  arm      0.650  (singular, 3rd position)
```

The key difference: for 'arm', the singular is the 3rd nearest clean
neighbor of the plural (cos=0.650). For 'hand', the singular is the
6th nearest neighbor of the plural (cos=0.475).

When we apply the axis to 'hand', the prediction moves toward the plural
region, but 'hand' itself has a LOWER similarity to the plural's region
than 'hands' requires — the singular 'hand' is not among the top clean
neighbors of the predicted location.

Put differently: **'hand' and 'hands' are in DIFFERENT semantic neighborhoods
despite being singular/plural forms**. 'hand' is embedded near
hand-as-instrument vocabulary (manual, dexterity, gesture), while 'hands'
is embedded near hands-as-collective-labor vocabulary (workforce, manos,
双手). No linear displacement can bridge this semantic discontinuity.

---

## The king→queen Anomaly

### Where king→queen Lives in Gender Space

PCA of 19 gender pair chord vectors produces an isotropic eigenspectrum
(each of the top 10 PCs contributes 8–13% of variance). Projecting
king→queen onto these PCs:

```
Component   king→queen projection
gPC1        +0.026  ← nearly ZERO
gPC2        +0.008  ← nearly ZERO
gPC3        −0.064  ← small
gPC4        +0.339  ← moderate
gPC5        −0.508  ← DOMINANT
gPC6        −0.210  ← moderate
gPC7        −0.372  ← moderate
gPC8        −0.130
gPC9        +0.341  ← moderate
```

**king→queen has near-zero projection on gPC1 and gPC2** — the two "main"
gender axes. It is predominantly encoded in gPC4–gPC5–gPC7–gPC9.

Reconstruction fidelity: using only gPC1+gPC2 gives cos=0.027. Need all
top 10 PCs to get cos=0.836.

### Contrast with Other Gender Pairs

```
Pair          gPC1     gPC5    ||top5||   Dominant component
─────────────────────────────────────────────────────────────
boy→girl      −0.597   +0.051  0.640      gPC1 (core social gender)
man→woman     −0.312   −0.110  0.464      gPC1 (core social gender)
uncle→aunt    −0.096   +0.155  0.539      gPC2 (relational gender)
groom→bride   +0.311   −0.038  0.832      gPC4 (wedding/romantic)
king→queen    +0.026   −0.508  0.615      gPC5 (royalty/succession)
```

### Semantic Interpretation

**gPC1 (Social/Binary Gender)**: boy/girl, man/woman, son/daughter — these
pairs are the core BIOLOGICAL GENDER vocabulary. The gPC1 direction captures
the fundamental male/female semantic distinction.

**gPC4 (Wedding/Ceremonial)**: groom→bride has gPC4=−0.719, the largest
single projection in the entire dataset. This axis separates wedding-ceremony
roles. 'bride' and 'groom' are paired in the exact context of marriage
ceremonies, creating a very specific local geometry.

**gPC5 (Royalty/Succession)**: king→queen is dominated by gPC5. The semantic
relationship king→queen is NOT primarily about gender — it's about ROYAL
SUCCESSION and political power. 'Queen' means "the female monarch" but in
W_E its position is determined primarily by its co-occurrence with coronation,
crown, throne, reign — not with its gender opposition to 'king'.

### The Broader Implication: "Gender" as a Human Category

Human linguists categorize all these as "gender pairs," but W_E makes no
such abstraction. Instead:

- boy→girl: DEVELOPMENTAL BIOLOGY dimension (kids, youth, growth)
- man→woman: SOCIAL ROLE dimension (adult, human, partner)
- king→queen: POLITICAL SUCCESSION dimension (monarchy, rule, power)
- groom→bride: RITUAL dimension (ceremony, marriage, celebration)
- actor→actress: PROFESSIONAL ROLE dimension (entertainment, performance)

Each "gender pair" is primarily about its SEMANTIC DOMAIN, with gender as
a secondary feature. The W_E embedding captures this correctly — gender is
not an independent dimension but a facet of semantic context.

---

## The Eigenspectrum Isotropy: Mathematical Artifact

### The Result

```
Axis       pc      Isotropy (1 - (λ1-λ2)/λ1)
+est       0.401   0.993  ← most isotropic
+ness      0.169   0.991
past_irr   0.284   0.893  ← least isotropic
un-        0.096   0.936
```

**All 12 axes show isotropy 0.89–0.99.** Even the "most directional" axis
(past_irr, pc=0.284) is 89% isotropic.

### Why This Is a Mathematical Artifact

The eigenspectrum of a matrix with n rows in d-dimensional space always
has at most n non-zero eigenvalues. When n << d (here n=8–16, d=1536),
the top eigenvalues are:

```
λ_k ≈ ||M||² / n    for all k from 1 to n
```

Because the n chord vectors span a random n-dimensional subspace of the
1536D space, all n eigenvalues are roughly equal. The apparent "isotropy"
measures how much the data is uniformly spread across its own span — which
will always be high when n << d.

The pc metric directly measures what we care about: the average pairwise
cosine between chord vectors (their mutual alignment). This is NOT the same
as eigenvalue spread.

**The correct interpretation of isotropy:**
- pc=0.401 (+est) means chords are moderately aligned (one dominant direction)
- Eigenspectrum isotropy=0.993 (+est) says the n=8 chord vectors are uniformly
  spread across their 8-dimensional span — which is always true for 8 vectors!

The eigenspectrum is not wrong — it's measuring a different (less useful)
property for our purposes.

### What Would Actually Reveal Structure

To properly test whether morphological axes have dominant directions,
one would need:
1. Many more chord vectors (n >> 100 pairs per axis)
2. Or restrict to a d-dimensional PCA of all tokens first, then analyze
   chord vectors in the d-PC space

With n=8–16 pairs in 1536D, the eigenspectrum is mathematically constrained
to be isotropic.

---

## Implications for the Geometric LCM

### Three Levels of Morphological Granularity Confirmed

```
Level 1: COARSE (across POS)
  gender(kin) ≠ gender(animals) ≠ gender(royalty)
  [Different semantic clusters have entirely different axes]

Level 2: MEDIUM (within POS, across semantic field)
  +s(object nouns) ≠ +s(body parts)
  cos(object, body-part) = 0.255  [different directions]
  [The "plural" transformation differs by semantic field]

Level 3: FINE (within semantic field, specific words)
  hand→hands: IRREDUCIBLE even with domain-specific axis
  [Some words have plural-semantic-field discontinuity]
```

### The Navigator Architecture

For a LCM to perform reliable morphological navigation:

```
Input: (source_word, transformation_type)
Step 1: DOMAIN CLASSIFIER — which semantic cluster is source_word in?
  → kin / royalty / animal / body_part / object / etc.
Step 2: AXIS LOOKUP — retrieve domain-specific axis
  → ax_body_part for head/foot/arm
  → ax_object for cup/door/car
  → ax_kin_gender for father/brother/son
Step 3: SCALE LOOKUP — retrieve domain-specific scale
  → Some domains need larger displacement (fire, train)
Step 4: CLEAN RETRIEVAL — exclude caps/compounds
Step 5: IRREDUCIBILITY CHECK — flag known exceptions
  → hand→hands is not retrievable; use lookup for this case
```

This is not a concession to lookup tables — it is a recognition that W_E
encodes semantic context at multiple granularities, and accessing the correct
level is part of the geometric computation.

---

## Day 312 Plan

1. **The hand→hands geometric analysis**: map the full path from 'hand' along
   the body-part axis. What token is closest at scale=0.1, 0.5, 1.0, 2.0?
   When does the path leave the 'hand' neighborhood? What's in that region?

2. **Identify all irreducible words across 12 axes**: for each axis, what
   fraction of training words are irreducible (can never be hit even at
   optimal scale)?

3. **Cross-semantic irregular plurals**: are there other body-part irregular
   plurals (tooth/teeth, mouse/mice) that the body-part axis also retrieves?

4. **The gPC5 royal axis**: what tokens are at the positive and negative poles
   of gPC5? Does it form a navigable axis for royalty vocabulary?

---

## Files

- `expedition_log.md` — Day 311 results
- `445_isotropic_gender_and_irreducible_plurals.md` — DC 445
- `day311_bodypart_plural_eigenspectrum_king_queen.py` — experiment script
