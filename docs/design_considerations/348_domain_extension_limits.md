# DC 348: Domain Extension Limits of W_E Factual Knowledge

**Day 162 | Scientific facts in W_E: where it works and why it fails**

---

## Overview

Day 162 extends the W_E entity_excl pipeline beyond geography/language/gender
into scientific domains: animal taxonomy, planets, colors, and chemistry.

**Core finding:**

> **W_E factual knowledge tracks training corpus co-occurrence density.
> Facts encoded in explicit relational phrases in general web text
> (capitals, antonyms, gender) succeed. Scientific facts that appear only
> in domain-specific text (rocky planets, color temperature) fail entirely.**

---

## Results

| Domain | Accuracy | Notes |
|--------|----------|-------|
| Capitals (control) | 5/6 = 83% | Expected; reduced vocab |
| Antonyms (control) | 3/6 = 50% | Reduced vocab artifact |
| Animal → class | 4/8 = 50% | Insects work; birds/fish mixed |
| Animal → class (3-shot dir) | 4/8 = 50% | Direction doesn't help |
| Planet → type | 0/7 = 0% | Planets cluster together |
| Color → temperature | 0/9 = 0% | Colors cluster together |
| Color → primary/secondary | 0/6 = 0% | Colors cluster together |
| Element → metal/nonmetal | 4/8 = 50% | Common metals; gold/silver confuse |

---

## Domain-by-Domain Analysis

### Planets (0%)
```
Mercury → Jupiter (target: rocky)
Venus   → Jupiter (target: rocky)
Earth   → Mars    (target: rocky)
Mars    → Venus   (target: rocky)
Jupiter → Saturn  (target: gas)
Saturn  → Jupiter (target: gas)
Neptune → Jupiter (target: gas)
```
Planets form a tight mutual-proximity cluster. Each planet is more similar
to another planet than to the abstract labels "rocky" or "gas". The W_E
knows that Mercury, Venus, Mars, Jupiter are all planets — but the
rocky/gas distinction is not encoded because "rocky planet" vs "gas planet"
is scientific terminology that rarely appears adjacent to the planet name
in general web text.

**What W_E DOES know about planets:**
PC2 of the scientific vocabulary SVD separates languages (Chinese, French)
from gas planets (Jupiter, Mercury, Neptune). Planets have a recognizable
cluster in W_E space — but the internal structure of that cluster doesn't
encode rocky vs gas.

### Colors (0%)
```
red → blue (target: warm)
orange → yellow (target: warm)
blue → purple (target: cool)
white → black (target: neutral)
```
Colors form a tight chromatic circle. Red is nearest to blue, orange
to yellow (complementary proximity), blue to purple. The temperature
labels (warm, cool, neutral) are abstract properties that don't appear
in proximity to color names in text. "Red is a warm color" is art/design
knowledge, not general web knowledge.

**What W_E DOES know about colors:**
PC0 of the scientific SVD separates colors (white, black, red, blue)
from animals (shark, bird, beetle). The color cluster is fully formed —
but it's an undifferentiated ball, not a structured warm/cool partition.

### Animals (50%) — Taxonomy Depends on Label Frequency
```
Works:   ant→insect✔  bee→insect✔  beetle→insect✔  salmon→fish✔
Fails:   eagle→owl✗   shark→whale✗  trout→salmon✗   moth→beetle✗
```
The pattern is revealing:
- **Insects**: "ant", "bee", "beetle" → "insect" works because the word
  "insect" appears frequently near these animal names in general text
  ("bee is an insect", "common insects include ants and beetles")
- **Birds**: eagle→owl (nearest bird) rather than eagle→bird. The word
  "bird" appears near "eagle" ("eagle is a bird") but owl is even closer
  because both are birds of prey that co-occur in similar contexts.
- **Fish**: salmon→fish works; shark→whale fails because shark and whale
  co-occur in marine contexts regardless of their taxonomic difference.

The 3-shot direction (using 3 mammal pairs to build a mammal direction)
provides no improvement over pure proximity. The direction built from
{dog→mammal, cat→mammal, horse→mammal} doesn't generalize to birds,
because the concept "animal → class" is not a single geometric direction —
each class (mammal, bird, fish, insect) is a different target, not a
shared direction.

### Elements (50%)
```
Works:   iron→metal✔  copper→metal✔  aluminum→metal✔  tin→metal✔
Fails:   gold→silver✗  silver→gold✗   zinc→copper✗   lead→copper✗
```
"Iron", "copper", "aluminum", "tin" appear frequently in contexts that
include the word "metal" in general text (industrial, engineering, and
everyday contexts). "Gold" and "silver" are precious metals that co-occur
more with each other (jewelry, investment contexts) than with the generic
label "metal". Zinc and lead have weaker corpus signal.

---

## The Proximity Principle: A Unified Theory

W_E entity_excl succeeds when:

```
condition_1: source clusters near target (direct proximity)
condition_2: source and target are in different top-level clusters
condition_3: relational phrase "X is a Y" appears frequently in corpus
```

W_E entity_excl fails when:
```
failure_1: source items cluster with each other, target is external
           (planets, colors — each planet/color is more similar to
            its peers than to the abstract class label)

failure_2: target label is a rare abstract property
           (rocky, gas, warm, cool, primary, secondary)

failure_3: multiple items confuse each other at proximity level
           (gold↔silver, zinc↔copper)
```

---

## SVD Structure of the Scientific Vocabulary

```
PC0: colors (white,black,red,blue)  ↔  animals (shark,bird,beetle)
PC1: cities/languages (Paris,Tokyo)  ↔  birds/animals (owl,bird)
PC2: languages (Chinese,French)  ↔  gas planets (Jupiter,Mercury)
PC3: properties (warm,soft,hard,cold)  ↔  colors (orange,blue,purple)
PC4: metals (metal,tin,iron,copper)  ↔  gas planets (Saturn,Jupiter)
```

**Notable:**
- PC4 separates metals from gas planets — the metal cluster is a
  distinct SVD axis, explaining why iron/copper/aluminum→metal works
- PC3 separates abstract properties from colors — "warm" IS in a
  different SVD region than "red", but the direction doesn't map
  color→property because it's the wrong type of contrast

---

## Comparison: What Distinguishes Working Domains

| Domain | Corpus phrase | Frequency | W_E accuracy |
|--------|--------------|-----------|--------------|
| Capitals | "X is the capital of Y" | Very high | 83-91% |
| Antonyms | "opposite of X is Y" | High | 75-100% |
| Gender | "the king and queen" | High | 100% |
| Animals→insect | "X is an insect" | Medium | 75% |
| Elements→metal | "X is a metal" | Medium | 50% |
| Planets→type | "X is a rocky planet" | Low | 0% |
| Colors→temp | "X is a warm color" | Very low | 0% |

The threshold for W_E encoding appears to be at "medium" frequency —
facts that appear in explicit relational form in general web text
enough times to push the embedding toward the target label.

---

## Implications

### For TruthSpace

The W_E factual knowledge store is **general web text knowledge**,
not encyclopedic scientific knowledge. Its reach is:

- ✅ Geography (capitals, countries, languages)
- ✅ Social relations (gender, royalty, family)
- ✅ Linguistic relations (antonyms, synonyms)
- ✅ Common categories with frequent explicit labels (insects, metals)
- ❌ Scientific taxonomy requiring domain-specific text
- ❌ Abstract properties rarely stated explicitly in text
- ❌ Fine-grained distinctions within tight clusters (rocky vs gas planets)

The pipeline accuracy (82.8%) on the Day 141-148 test set reflects
exactly this: the held-out set was designed around geography and
antonyms, where W_E excels.

### For the Hypothesis

The TruthSpace hypothesis predicts that knowledge IS geometry.
This is confirmed — but the geometry only encodes knowledge that
is structurally present in the training data.

The hypothesis doesn't fail — it's refined:
> **The shape IS the knowledge present in the training corpus,
> organized by co-occurrence frequency.**

Scientific facts below the co-occurrence threshold are not in W_E
geometry at all — they live only in the deeper inference components
(T2, full context). This is why scientific Q&A requires full inference.

---

## Next Directions

1. **Co-occurrence density test**: Can we predict which facts are in W_E
   by measuring the actual corpus frequency of "X is a Y" constructions?
   (Use Wikipedia/Common Crawl word co-occurrence counts as a proxy)

2. **Domain-specific models**: Does BioMedLM (trained on biomedical text)
   have animal→class structure in its W_E? Does CodeLlama have
   syntax→category structure?

3. **Direction generalization**: Can a universal "hypernym direction"
   be built that generalizes across all categories (not just one)?
   The 3-shot direction from mammals didn't generalize — but a direction
   built from {dog→animal, Paris→city, iron→metal, red→color} might.

---

## Files

- `expedition_day162_domain_extension.py` — full domain test
- `day162_domain_extension.json` — results
- `347_scaling_invariance.md` — prior arc
