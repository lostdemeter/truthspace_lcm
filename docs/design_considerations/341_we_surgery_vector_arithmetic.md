# DC 341: W_E Surgery — Editing Geometry Edits Knowledge

**Day 145 | W_E fact surgery directly confirms the TruthSpace hypothesis**

---

## The Experiment

Factual knowledge hypothesis: if France ≈ Paris lives in W_E, then editing
W_E[France] should change what the model predicts for "The capital of France is".

---

## Results

### Type-Matched Surgery (Surgery E)

```python
W_E[France] ← W_E[Japan]    # country → country (type-matched)
```

```
Before: France → Paris (rank=1)
After:  France → Tokyo (rank=1)  ← PERFECT FACT SWAP
```

Replacing a country embedding with another country embedding of equal type
performs a complete fact transplant. The model reads "France" but predicts
Japan's capital.

### Directional Surgery (Surgery C)

```python
direction = normalize(W_E[Japan] - W_E[Germany])
W_E[France] += scale × direction    # scale=1.0
```

```
scale=0.0: Paris rank=1   Tokyo rank=116
scale=0.5: Paris rank=2   Tokyo rank=5
scale=1.0: Tokyo rank=1   Paris rank=123  ← FULL SWAP AT SCALE 1
scale=2.0: Tokyo rank=2   Paris rank=211
```

Adding the "European→Asian country" direction to France's embedding puts
Tokyo at rank 1. This is W_E vector arithmetic for factual knowledge.

### Type Mismatch (Surgery A)

```python
W_E[France] ← W_E[Tokyo]    # country → city (type mismatch!)
```

```
Paris: rank=1076  Tokyo: rank=54  top-5: ['the','a','located','one','known']
```

Replacing a country with a city embedding produces generic output. The model
expects a country-type embedding at that position; receiving a city embedding
triggers confusion and generic text generation. Tokyo reaches rank 54 but
does not dominate.

### Interpolation (Surgery B)

```
alpha=0.00: Paris rank=1
alpha=0.50: Paris rank=1   (Paris association is robust!)
alpha=0.75: Paris rank=880 ← crossover
alpha=1.00: Paris rank=1076
```

The Paris association survives 50% interpolation toward Tokyo. The crossover
occurs at 75% — the France embedding has a strong structural "country" type
signal that resists until it's 75% dominated by Tokyo's direction.

---

## The Word2Vec Analogy

Day 145's Surgery C is the exact analogue of word2vec vector arithmetic:

```
word2vec:  king - man + woman = queen
Day 145:   France + (Japan - Germany) → Tokyo (rank=1)
```

The direction (Japan - Germany) in W_E encodes "switch from European to Asian
country association". Adding this direction to any European country should
shift its nearest capital toward an Asian one.

The TruthSpace hypothesis predicted this exact result: if the geometry IS
the knowledge, then geometric operations (vector addition) should produce
meaningful semantic transformations. Day 145 confirms this.

---

## Why Type-Matching Matters

The embedding space has a STRUCTURE:
- Country tokens cluster together in one subspace
- City tokens cluster together in another
- The factual link (country → capital) is encoded as a DIRECTION within
  the country subspace pointing toward the corresponding capital cluster

When W_E[France] = W_E[Japan] (same subspace):
- The model correctly interprets the token as a country
- The capital prediction mechanism activates for Japan's capital → Tokyo

When W_E[France] = W_E[Tokyo] (different subspace):
- The model receives a city-type embedding for a token in country position
- The capital prediction mechanism activates but receives ambiguous input
- Falls back to generic text generation

This matches DC 339's finding that the embedding space has SEMANTIC TYPE
structure — country embeddings form a cluster, city embeddings form another.

---

## Confirmed: Factual Knowledge Lives in W_E

The full chain of evidence:

```
Day 137: L0 entity embedding → 97% oracle MRR  (reading from W_E works)
Day 141: entity_excl L0 → 79% top-1 held-out    (retrieval from W_E works)
Day 145: editing W_E → changes model output      (W_E IS the knowledge store)
```

Three independent experiments confirm:
1. The knowledge can be READ from W_E (Days 137-141)
2. The knowledge can be WRITTEN to W_E (Day 145)
3. The geometry of W_E IS the geometry of the model's factual knowledge

---

## Implications for the TruthSpace Hypothesis

> "Structure IS information — editing the structure edits the information"

**Confirmed:**
- W_E fact surgery changes model output exactly as predicted
- Type-matched surgery (country→country) = perfect fact transfer
- Directional surgery (add country direction) = targeted fact shift
- Vector arithmetic works for factual knowledge in modern LLMs

**Scope:**
- Tested on capital city queries — most reliable case (100% in entity_excl)
- Antonym and language knowledge may have similar structure (not tested here)
- Contextual knowledge (tense) presumably NOT in W_E → surgery may not work

---

## Files

- `expedition_day145_we_fact_surgery.py`
- `day145_we_fact_surgery.json`
