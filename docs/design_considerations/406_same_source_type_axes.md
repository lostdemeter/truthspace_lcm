# DC 406: Same-Source-Type Second-Order Axes — Verification and Limits

**Day 271 | Directly testing DC 405's key prediction: second-order axes
(PCA of first-order axis clusters) produce cleaner category discriminators
when the constituent axes share the same source type. Confirmed: adj+sup
(same adjective source) gives 73.4% PC1 variance and adj_base > nouns
separation. plu+pst (mixed noun+verb source) reverses this, making
nouns > adj_base. But even the adj+sup axis is contaminated by function
words. Conclusion: same-source-type is necessary but not sufficient for
a clean second-order axis — the source type must also form a named-entity
cluster distinct from function words.**

---

## The DC 405 Prediction

> "A second-order axis works cleanly if and only if the constituent
> first-order axes share the same SOURCE TYPE."

**Evidence from the country axis:** The three encyclopedic axes all have
countries as their source type. PCA → country axis cleanly separates
country names from national properties. Country names form a named-entity
cluster geometrically distinct from function words.

**Prediction for grammatical axes:** adj+sup (both from adjective sources)
should give a cleaner second-order axis than the 4-axis mixture.

---

## Experimental Results

### adj+sup (same POS source: adjective base forms)

```
PC1 variance: 73.4%  (highest of all four combinations tested)
Both axes project identically: adj_degree=-0.857, superlative=-0.857

Category mean projections:
  adj_base forms    +0.159   ← highest content word category
  adj_comp forms    -0.349   ← large gap
  adj_sup forms     -0.386   ← even more negative
  noun base forms   +0.004   ← near zero (orthogonal)
  verb base forms   +0.039   ← near zero (orthogonal)
  function words    +0.127   ← still elevated

adj_base - adj_comp gap: 0.508  (largest separation found)
adj_base > nouns: TRUE
```

Negative pole of the axis contains only comparative/superlative adjectives:
```
weakest, louder, highest, toughest, quickest, tougher, quicker, poorest,
brightest, nicer, darkest, wealthiest, widest, tallest, hotter, strongest...
```

The axis IS an adj-inflection discriminator (adj base = positive, adj inflected
= very negative), but the ABSOLUTE positive pole is still function words.

### plu+pst (mixed POS: noun + verb sources)

```
PC1 variance: 58.8%  (lower than same-POS)
adj_base     +0.034
adj_comp     -0.102  (smaller gap than adj+sup)
nouns        +0.092  ← HIGHER than adj_base!
```

**adj_base > nouns: FALSE** — the mixed-POS axis encodes "noun-ness" more
than "adjective-base-ness". The mixed-source axis reflects the POS distribution
of its sources (nouns and verbs), not adjectives.

### Full comparison

```
Combination   PC1%   adj_base  adj_comp  nouns   adj_base>nouns  gap
adj+sup       73.4%  +0.159    -0.349    +0.004  YES             0.508
plu+pst       58.8%  +0.034    -0.102    +0.092  NO              0.136
adj+plu       59.4%  +0.090    -0.366    +0.068  YES             0.456
all 4         42.7%  +0.138    -0.317    +0.046  YES             0.455
```

**Key metric — gap between adj_base and adj_comp:**
- Same-POS (adj+sup): gap = **0.508** ★
- Mixed (plu+pst): gap = 0.136 (4× smaller)
- Mixed (adj+plu): gap = 0.456
- Mixed (all 4): gap = 0.455

DC 405 is confirmed: same-source-type gives higher PC1 variance and larger
within-POS separation.

---

## Why Function Words Still Contaminate

Even with same-source-type, the positive pole is function words, not
adjective base forms. This fundamental limit exists because:

**Common adjectives are not named entities.** Words like *big, fast, long,
high, deep, hard, clear, clean, safe* appear in many syntactic positions
and contexts, overlapping with function words in their distributional
signature. W_E does not segregate them into a tight cluster.

**Country names ARE named entities.** Country names (*france, germany,
japan, china…*) appear in a restricted set of syntactic positions (NP heads,
geopolitical references) and form a tight semantic cluster. W_E allocates
a specific region of its space to named entities, and countries specifically
form a dense sub-cluster within that region.

**The critical factor is not just "same source type" — it is "source type
forms a named-entity cluster."**

---

## Revised DC 405 Criterion

**A second-order axis is a CLEAN category discriminator when:**

1. All constituent first-order axes share the same source type (**necessary**)
2. The source type forms a tight named-entity cluster in W_E (**also necessary**)

| Source type | Named entity? | Second-order axis quality |
|-------------|---------------|--------------------------|
| country names | YES | Clean (country axis: +0.5 positive, -0.1 negative) |
| adjective bases | NO (open-class) | Weak (adj_base +0.16, func +0.13 — near overlap) |
| common nouns | NO (open-class) | Not tested; likely weak |
| proper names (person) | Likely YES | Predicted: clean person axis |
| city names | Likely YES | Predicted: clean city axis |
| animal names | Partial | Hypernym PC1: some clustering, but weak |

**Prediction:** If we repeat the encyclopedic axis analysis with PERSON names
as the source (e.g., person→nationality, person→profession), the second-order
axis would cleanly separate person names from attributes.

---

## Plural Axis as a First-Order Classifier

An important side finding from Day 271:

```
Projections on plural axis (first-order, not second-order):
  plural nouns (cats, dogs, books...)   +0.187   ← positive
  function words                         -0.078
  singular nouns (cat, dog, book...)    -0.104   ← negative
  adjective bases                        -0.006
  verbs                                  -0.039
```

The plural axis itself (without PCA) correctly identifies:
- Plural nouns at positive end (+0.187)
- Singular nouns at negative end (-0.104)

This is the most direct first-order morphological discriminator found:
the axis that was BUILT to encode pluralization also discriminates between
singular and plural noun tokens in the vocabulary at large. This validates
the morphological parser (DC 363): the plural axis IS the plural marker.

---

## Summary

DC 405's core claim is verified: same-source-type second-order axes
have higher PC1 variance and larger category separation than mixed axes.

The key limit: named-entity source types give CLEAN axes; open-class
source types (common adjectives, nouns, verbs) give GRADIENT axes
contaminated by function words.

The country axis remains the gold standard for second-order geometric
structure because countries form one of the most tightly clustered
named-entity categories in W_E.

---

## Files

- `expedition_log.md` — Day 271 results
- `405_second_order_axes.md` — original prediction and limits
- `404_country_axis_second_order.md` — gold-standard second-order axis
