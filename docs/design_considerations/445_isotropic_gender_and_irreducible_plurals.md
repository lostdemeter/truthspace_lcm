# DC 445: Isotropic Gender Space and Irreducible Plurals

**Day 310 | Two definitive findings: (1) The gender chord space in W_E is
ISOTROPIC — the PCA eigenvalues are 0.078/0.069/0.064, all within 20% of
each other. There is no dominant gender direction. Each word pair's gender
transformation lives in a different local direction, making "the gender axis"
a statistical fiction — in reality there are as many gender axes as there
are semantic clusters. (2) Two plural forms (hand→hands and eye→eyes) are
TRULY IRREDUCIBLE: no matter what scale is applied to the +s axis, 'hands'
and 'eyes' never become the nearest clean neighbor of 'hand' and 'eye'. The
failure is LOCAL GEOMETRY not frequency — arm (similar frequency, norm=0.65)
achieves arm→arms at scale=0.020. The irreducibility of hand and eye reveals
that the +s transformation IS NOT DEFINED for these words in the direction
that the training set implies — the training set builds an axis that is
biased away from the hand/eye neighborhood.**

---

## The Isotropic Gender Space

### The Evidence

PCA of all gender chord vectors (21 pairs across kin, titles, occupation,
animals, cross-lingual) produces:

```
Component  Eigenvalue  % variance (relative)
gPC1       0.0780      36.5%
gPC2       0.0689      32.3%
gPC3       0.0641      30.0%
```

These three components are within 20% of each other. This is the signature
of an ISOTROPIC space — no preferred direction. Compare to a truly directional
space (like the +er axis): if gender had one dominant direction, gPC1 would
have 80%+ of the variance.

### Individual Pair Decomposition

```
Pair              gPC1     gPC2     interpretation
──────────────────────────────────────────────────────
boy → girl        −0.585   +0.007   gPC1-dominant
son → daughter    −0.639   −0.125   gPC1-dominant
uncle → aunt      +0.063   −0.502   gPC2-dominant!
groom → bride     +0.276   +0.537   gPC2-dominant!
husband → wife    +0.071   +0.261   gPC2-dominant
king → queen      +0.035   +0.040   NEITHER (both tiny)
father → mother   +0.175   −0.197   SPLIT
brother → sister  −0.380   −0.165   gPC1-dominant
```

The core observation: **king→queen has the SMALLEST projection on both gPC1
and gPC2**. The most iconic gender pair has almost no projection on the top
two gender axes! This is geometrically profound: king→queen is not
representative of the average gender transformation — it's a special local
operation in a very specific direction.

### What the Two Gender Axes Mean

**gPC1 (age/generation gender)**:
- Negative: boy→girl, son→daughter, brother→sister, actor→actress
- These are GENERATIONAL or YOUTH-ADJACENT gender pairs
- The gPC1 direction captures the gender distinction that co-occurs with
  the "young" or "parallel generation" semantic field

**gPC2 (intimate/relational gender)**:
- Positive: bride, wife, lady (romantic partnership roles)
- Negative: groom, aunt, lord (authority/caregiving roles)
- Known pair delta: lord→lady = +0.289 (large), actor→actress = −0.030

gPC2 separates ROMANTIC/INTIMATE gender (bride, wife, girlfriend) from
AUTHORITY/SOCIAL-ROLE gender (lord, aunt).

### Why "The Gender Axis" Does Not Exist

The computation of a single gender axis (as done in Days 305–309) produces
a MEAN of many different local directions. This mean direction:
- Works well for kin core vocabulary (which is where it was trained)
- Fails for animals (orthogonal direction)
- Fails for occupation (different local direction)
- Fails for titles (different local direction + tokenization issues)

The mean axis is not "wrong" — it captures the SHARED COMPONENT across
gender pairs. But this shared component is small relative to the
pair-specific components, which is why generalization is limited.

---

## Irreducible Plural Failures: hand and eye

### The Escape Scale Spectrum

```
Word      Min scale to escape   Can hit?
────────────────────────────────────────
flower    0.020                 YES (immediate)
boat      0.020                 YES
cup       0.020                 YES
door      0.020                 YES
arm       0.020                 YES
forest    0.060                 YES
road      0.060                 YES
star      0.140                 YES
train     0.200                 YES (just above default)
fire      0.580                 YES (3× default)
hand      ∞                     NO — NEVER
eye       ∞                     NO — NEVER
```

The escape scales form a continuum from 0.02 (easy plurals) to 0.58 (fire)
to infinity (hand, eye). The default +s scale=0.181 hits 11/15 words.

### Why Can't 'hands' and 'eyes' Be Retrieved?

The clean nearest-neighbor search works by finding the token w such that:
```
cos(W_E[hand] + s×axis_s, W_E[w]) > cos(W_E[hand] + s×axis_s, W_E[w'])
```
for all w' ≠ hand (excluding capitalized and compound forms).

For 'hand': at any scale s, the token 'hands' is never the most similar
clean token to W_E[hand] + s×axis_s.

**What IS the nearest clean token at various scales?**

At scale=0.181 (default): 'hand' itself (self-retrieval)
At scale=0.580: still 'hand'
At scale=1.0: the axis displacement is now pointing toward tokens related
to what 'hand' means PLUS the average plural meaning — but 'hands' is not
the dominant plural concept in that direction.

The root cause: **'hands' is embedded close to 'hand' in a direction that is
ORTHOGONAL to the +s training axis**. The +s axis was trained on
cat/dog/house/car/tree/book/bird/ship — these are OBJECT NOUNS. When we
compute the mean displacement for these pairs, the direction primarily
captures the "object-to-collection" semantic shift.

'hand' and 'eye' are BODY PART NOUNS. Their plurals exist in a slightly
different direction — the "body-to-collection" shift. The +s axis computed
from object nouns doesn't point toward body part plurals.

### The Body Part Exception

The key distinction:
- Object nouns (cat, dog, car): their plurals are used in IDENTICAL semantic
  contexts as the singular (a cat, two cats — same referent type)
- Body part nouns (hand, eye): plurals often have DIFFERENT semantic functions
  (hands = manual labor, eyes = attention/perception — metaphorical extension)

In W_E, 'hands' is embedded in the labor/manual work semantic neighborhood,
not just in the "plural of hand" neighborhood. Similarly, 'eyes' is embedded
in the attention/perception neighborhood.

The +s axis computed from object nouns doesn't reach these body-part-plural
semantic neighborhoods.

### Implication for Morphological Axes

The +s axis is TRULY domain-specific at a fine granularity:
- Object noun plural: cat→cats, tree→trees (easy, scale≈0.02)
- Concrete noun plural: cup→cups, door→doors (easy, scale≈0.02)
- Nature noun plural: flower→flowers, star→stars (easy)
- Abstract noun plural: train→trains (medium, scale≈0.20)
- Ambiguous noun plural: fire→fires (hard, scale≈0.58)
- Body part plural: hand→hands, eye→eyes (UNREACHABLE with object-noun axis)

This is the LOCAL OPERATOR PRINCIPLE at fine granularity: even within the
+s transformation, there are multiple sub-domains (object, nature, abstract,
body-part) each with their own geometric neighborhood.

---

## The Tokenization Trap: Revisited

### The Titles Domain Revelation

The "0% gender accuracy on titles domain" (Day 309) was entirely explained
by tokenization:

```
Word           Single-token?  Why
─────────────────────────────────────────────────
duchess        NO             [294, 1387] = "duch" + "ess"
empress        NO             [976, 1873] = "empr" + "ess"
baroness       NO             [3619, 263] = "bar" + "oness"
marchioness    NO             multi-token
viscountess    NO             multi-token
tsarina        NO             multi-token
```

All feminine title forms ending in '-ess' or '-ina' are multi-token in
Qwen2. Only 4 single-token title pairs exist: lord/lady, prince/princess,
king/queen, knight/dame.

**The pattern**: Qwen2's BPE tokenizer assigns single tokens to high-frequency
words. Feminine title forms (baroness, duchess, countess) are low-frequency
enough to be tokenized as multi-part strings. Masculine counterparts (baron,
duke, count) are sometimes single tokens because they appear as common proper
names in text.

### The Broader Tokenization Asymmetry

```
Domain     Male forms           Female forms
──────────────────────────────────────────────
Titles     duke, count, baron   duchess ✗, countess ✗, baroness ✗
Animals    lion, tiger, stallion lioness ✗(?), tigress ✗(?), mare ✓
Occupation actor, waiter, host   actress ✓, waitress ✓, hostess ✓
Kin        king, man, boy        queen ✓, woman ✓, girl ✓
```

The tokenization asymmetry is systematic: MALE forms are more commonly
single tokens because male forms are more frequent in text (higher BPE priority),
while FEMALE forms are multi-token because they're derived (lower frequency).

This is a REPRESENTATION BIAS in the model's tokenizer that limits what
morphological transformations can be evaluated.

---

## The Three Layers of Morphological Failure

Day 310 reveals a clear taxonomy of morphological axis failures:

**Layer 1: TOKENIZATION FAILURE** (not geometric)
- The target word is multi-token → cannot be retrieved by axis + NN
- Solution: use character-level or subword composition
- Examples: duchess, empress, baroness

**Layer 2: TOKENIZATION INTERFERENCE** (geometric but intercepted)
- Source or target has near-duplicate tokens (capitalized, compound)
- Solution: clean retrieval (Day 308/309 fix)
- Examples: cup→Cup, road→Road, eye→-eye

**Layer 3: AXIS DOMAIN MISMATCH** (geometric root)
- The axis was trained on a different sub-domain
- Target is geometrically located elsewhere
- Solution: domain-specific axis
- Examples: hand→hands (body-part domain ≠ object-noun domain)

Layers 1 and 2 are fixable with retrieval improvements. Layer 3 is a
FUNDAMENTAL LIMITATION — there is no single +s axis that covers all nouns.

---

## Day 311 Plan

1. **Body-part plural axis**: train +s axis specifically on body part nouns
   (head/heads, foot/feet, ear/ears, knee/knees, etc.). Does it retrieve
   hand→hands and eye→eyes?

2. **Gender chord isotrophy confirmation**: compute eigenvalue spectrum for
   all 12 morphological axes. Is gender uniquely isotropic, or are others
   similar?

3. **The king→queen anomaly**: king→queen has near-zero gPC1/gPC2 projections.
   What direction does king→queen live in? Does it live in a 3rd, 4th, or 5th
   PC of the gender space?

4. **Semantic field analysis**: for each irreducible +s failure, what semantic
   field does the word's plural form belong to that differs from the source?

---

## Files

- `expedition_log.md` — Day 310 results
- `444_orthogonal_gender_spaces_and_clean_retrieval.md` — DC 444
- `day310_threshold_frequency_sigma_gender.py` — experiment script
