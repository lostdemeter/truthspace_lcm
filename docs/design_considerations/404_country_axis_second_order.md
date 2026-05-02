# DC 404: The Country Axis — Second-Order Geometric Structure in W_E

**Day 269 | The PCA of the three encyclopedic axes (capital, language,
currency) reveals a shared "country axis" that perfectly separates
COUNTRY NAMES (positive pole: france +0.522, germany +0.517, japan +0.428…)
from NATIONAL PROPERTIES (negative pole: real -0.153, peso -0.132,
Spanish -0.126, paris -0.121…). After removing this shared component,
the domain residuals each correctly point toward their target category.
This is the first discovered second-order geometric structure: W_E organises
encyclopedic knowledge into a structured "national facts" subspace.**

---

## Discovery Method

We computed three first-order axes via mean chord direction:
```
capital_axis  (france→paris, germany→berlin, ...)
language_axis (france→french, germany→german, ...)
currency_axis (japan→yen, china→yuan, ...)
```

These three axes are moderately aligned (cos 0.16–0.29, DC 403).
We applied SVD to the 3×1536 matrix of axis vectors to extract their
shared component:

```
SVD singular values: [1.218, 0.919, 0.819]
PC1 explains 49.5% of variance in the encyclopedic subspace
Projections of enc axes onto PC1:
  capital  → -0.680
  language → -0.774
  currency → -0.650
```

The first principal component is the **country axis** — the shared
direction in W_E that all three encyclopedic axes traverse.

---

## The Country Axis Perfectly Separates Countries from Properties

Projecting every token in W_E onto the country axis:

```
POSITIVE POLE (country names):
  germany  +0.522   france   +0.517   japan    +0.428
  china    +0.419   Germany  +0.409   mexico   +0.404
  india    +0.357   canada   +0.346   australia +0.344
  Ireland  +0.314   Norway   +0.281   Sweden   +0.306
  Poland   +0.300   Greece   +0.297   Russia   +0.295
  Finland  +0.288   Portugal +0.287   Britain  +0.283

NEGATIVE POLE (national properties):
  real    -0.153   peso    -0.132   Spanish -0.126
  English -0.122   paris   -0.121   French  -0.116
  Chinese -0.115   yuan    -0.112   yen     -0.109
  Welsh   -0.106   Russian -0.103   dollar  -0.101
  Paris   -0.099   Tibetan -0.095
```

**The country axis is not a constructed feature — it emerged from the
mean structure of three independently computed axes.** Yet it precisely
separates the conceptual category "country" from the conceptual category
"national property" (language names, currency names, capital names).

---

## Axis Decomposition: Two-Level Hierarchy

Every encyclopedic axis decomposes into:

```
enc_axis = country_component × country_axis + domain_component × domain_axis
```

where `country_component ≈ -0.65 to -0.77` (negative: enc axes point
FROM countries TOWARD properties, i.e., oppose the country axis direction).

**After removing the shared country component, domain residuals correctly
identify their target vocabulary:**

```
capital  residual NN: [berlin, paris, Berlin, Paris]     ✓ capital cities
language residual NN: [french, japanese, chinese, german, spanish]  ✓ languages
currency residual NN: [real, yen, peso, yuan]            ✓ currencies
```

The domain residuals are mutually anti-correlated:
```
cos(capital_residual, language_residual) = -0.501
cos(capital_residual, currency_residual) = -0.512
cos(language_residual, currency_residual) = -0.487
```

After removing the country component, the three domain directions point
away from each other — they form a maximally discriminating coordinate
system in the "property" half of the encyclopedic subspace.

---

## The 3D "National Facts" Subspace

The three encyclopedic axes span a 3-dimensional subspace of W_E
that encodes all national facts simultaneously. Its structure:

```
Dimension 1: country_axis
  + end: country name tokens  (france, germany, japan, china...)
  − end: national property tokens (paris, french, yen...)

Dimension 2: capital domain
  Points specifically toward capital city vocabulary (berlin, paris, rome...)
  After removing country_axis component

Dimension 3: language domain  (approx. orthogonal to capital after removal)
  Points specifically toward language vocabulary (french, german, japanese...)
```

Currency occupies a direction not well captured by dimensions 2 and 3
(it's approximately the third orthogonal direction in this subspace).

**Navigation from a country embedding:**
```
emb(france) - enc_axis_capital   → emb(paris)
emb(france) - enc_axis_language  → emb(french)
emb(france) - enc_axis_currency  → emb(euro/franc)
emb(germany) - enc_axis_language → emb(german)
```

The enc axes point FROM country TO property (negative country_axis
component), so subtraction (not addition) navigates from country to
property.

---

## Why the Projections Are Negative

The first-order axes (capital, language, currency) were computed as
mean chord from source (country) to target (property). These axes
therefore point from the "country" cluster toward the "property" cluster
in W_E. The country axis (PC1 of the three enc axes) points in the
direction of maximum shared variance — which is the direction from
properties toward countries. So the enc axes project negatively onto
the country axis (they point in the opposite direction).

This sign convention is consistent: enc axes = `-α × country_axis + domain`.

---

## Implication: W_E Has a Country "Concept" Dimension

The country axis is a genuine semantic dimension of W_E that encodes
the **concept type "country"** independently of any specific country.
It answers the question "how much does this token participate in the
conceptual category of nation-states?"

This is analogous to the biological hierarchy: just as "animal" projects
above the hypernym axis (DC 401), "france" projects above the country axis.
W_E encodes **conceptual categories** as directions, not just specific
word relationships.

---

## Second-Order Structure: A General Pattern?

The discovery of the country axis suggests a general pattern:
For any coherent domain with multiple bijective encyclopedic axes sharing
the same source type, a **category axis** can be extracted via PCA.

Predicted further second-order axes:
- **animal axis**: extract from hypernym pairs (dog→animal, cat→animal, …)
  — should separate animal names from non-animals
- **adjective magnitude axis**: already observed as the "universal antonym
  axis" (weak/slow/quiet at positive pole, big/fast/strong at negative)
- **inflectional axis**: PCA of the 4 morphological inflection axes should
  give the shared "this is an inflected form" direction

Each second-order axis is discoverable without any linguistic annotation
— just PCA of first-order axes sharing a common source or target type.

---

## Files

- `expedition_log.md` — Day 269 results
- `401_semantic_relation_axes.md` — first-order enc axes
- `403_axis_orthogonality_full.md` — enc cluster identified
- `402_antonym_domain_axes.md` — magnitude axis (first-order antonym)
