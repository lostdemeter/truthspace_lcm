# DC 408: Third-Order Named-Entity Axis — Negative Result and Lessons

**Day 273 | Attempt to extract a third-order "named-entity vs common word"
axis via PCA of [country_axis, person_axis, city_axis] fails. The city
axis anti-correlates with the country axis (cos=−0.548) due to geographic
semantic entanglement: cities and countries occupy opposite ends of the
same geographic axis in W_E. The PCA yields a city-vs-country discriminator
(cities=+0.150, countries=−0.276), not a universal NE axis. The correct
approach requires semantically INDEPENDENT named-entity categories
(countries, persons, chemical elements, biological species, companies)
that do not have systematic first-order relations to each other.**

---

## What Was Tested

PCA of three second-order named-entity axes:
- `country_axis` (PC1 of capital+language+currency enc axes)
- `person_axis` (PC1 of scientist_nat+scientist_field+leader_nat axes)
- `city_axis` (PC1 of city→country and city→continent axes)

**Expected:** PC1 of [country, person, city] = shared "named entity"
direction separating all proper nouns from common words.

**Found:** city axis is anti-correlated with country axis (cos=−0.548),
so PCA captures the city-country CONTRAST rather than a shared NE direction.

---

## The City-Country Anti-Correlation

```
City axis construction:
  city→country chords: (paris, france), (berlin, germany), (tokyo, japan)…
  mean chord direction = axis pointing FROM cities TOWARD countries
  coherence = 0.775 (high)

Country axis (from Day 269):
  PC1 of capital+language+currency enc axes
  positive pole = country names (france, germany, japan…)
  enc axes point FROM countries TOWARD properties (negative projection)

cos(city_axis, country_axis) = -0.548
```

**Why they anti-correlate:** Both city→country chords and the country axis
itself pass through the same geographic cluster in W_E. The city axis
points FROM the city cluster TOWARD the country cluster. The country axis
points TOWARD the country cluster from the other direction (from national
properties). They are arriving at the same destination (country names)
from OPPOSITE directions — so they anti-correlate.

Geometrically: cities, countries, and national properties are not three
independent locations; they form a linear arrangement along a geographic
axis:

```
[national properties] ←——— [country names] ←——— [city names]
       currencies                france               paris
       languages                 germany              berlin
       capitals                  japan                tokyo
```

The country axis points LEFT→RIGHT (properties to countries).
The city axis points RIGHT→LEFT (cities to countries).
They both point TOWARD the same region from opposite sides.

---

## Why the Third-Order Test Failed

The fundamental assumption behind PCA-based third-order axis extraction is:

> If second-order axes all point TOWARD the same semantic class, their
> shared component IS the category axis for that class.

This worked for the country axis (Day 269) because capital, language, and
currency enc axes all point FROM countries TOWARD different targets. PCA
extracted the shared "FROM countries" component.

It worked for the person axis (Day 272) because scientist_nat,
scientist_field, and leader_nat axes all point FROM persons TOWARD different
attributes. PCA extracted the shared "FROM persons" component.

It **failed** for the NE axis because:
- Country axis points TOWARD country names
- Person axis points TOWARD person names (partially — also toward attributes)
- City axis points TOWARD country names (from cities)

The country and city axes are pointing toward the SAME THING from different
starting points, not toward DIFFERENT named-entity categories from a shared
starting class. The PCA finds the city-country contrast, not a shared
named-entity feature.

---

## The Correct Architecture for a Third-Order NE Axis

**Required:** Second-order axes all extracted FROM semantically independent
named-entity categories:

```
country_axis  →  anchored at country names
person_axis   →  anchored at famous person names
element_axis  →  anchored at chemical element names (hydrogen, oxygen, carbon…)
species_axis  →  anchored at biological species (canis, quercus, homo…) [multi-token issue]
company_axis  →  anchored at company names (Apple, Google, Microsoft…)
```

These categories share NO systematic first-order relations to each other:
- Countries are not systematically related to persons (except historically)
- Persons are not systematically related to elements
- Elements are not systematically related to companies

Their second-order axes will each point FROM their respective named-entity
clusters TOWARD their respective attribute clusters. The ONLY shared
component across all five will be the "starting from a named entity"
direction — the genuine named-entity axis.

**Prediction for the correct experiment:**
```
PCA of [country_axis, person_axis, element_axis, company_axis]:
  PC1 = named-entity axis
  positive pole: named entities (all categories)
  negative pole: attributes (nationalities, fields, properties…)
```

---

## What the City-vs-Country Axis Reveals

Even though the third-order NE axis failed, the city-vs-country axis
(Day 273 NE axis) is informative:

```
positive pole: paris, berlin, Paris, Berlin, London, Beijing, Nairobi,
               Moscow, Boston, Miami, Madrid, Stockholm, Vienna…
negative pole: france, germany, europe, Germany, France, Italy, Europe,
               Spain, china, japan, Ireland, canada, Sweden…
```

This is the **geographic containment axis** — the direction in W_E that
distinguishes CONTAINER entities (countries, continents) from CONTAINED
entities (cities, regions). The anti-correlation of city_axis and
country_axis (cos=−0.548) is itself a meaningful geometric structure:
W_E encodes the city-country containment relation as a genuine geometric
direction.

**Practical application:** This axis can classify geographic tokens as
"country-level" (negative) vs "city-level" (positive) without any
lookup table.

---

## Lessons for Axis Hierarchy Construction

1. **Second-order axes pointing FROM the same category are composable.**
   PCA of axes all anchored at the same source type extracts the
   category axis. (Days 269, 272)

2. **Second-order axes pointing TOWARD semantically related categories
   anti-correlate.** PCA finds the contrast, not a shared direction.
   (Day 273)

3. **For a K-th order axis, the constituent (K−1)-th order axes must
   all be anchored at independent named-entity categories.** Independence
   means: no systematic first-order relations between the categories.

4. **Anti-correlation encodes semantic proximity.** The cos=−0.548
   between city and country axes is not noise — it encodes the
   geographic containment relation (cities are geographically and
   semantically "inside" countries).

---

## Revised Plan for Third-Order NE Axis

**Day 274 experiment:**
Build element→property axis (hydrogen→element, oxygen→element, carbon→compound…)
and company→attribute axis (Apple→technology, Google→technology, Microsoft→software…).
Then PCA of [country_axis, person_axis, element_axis, company_axis].

If the third-order NE axis emerges, it should:
- Score named entities highly (countries, persons, elements, companies)
- Score common words near zero
- Provide a universal "is this a proper noun?" filter

If it doesn't emerge, it means the "named entity" concept is not a geometric
axis in W_E at all — different named-entity categories are represented in
different, incompatible directions — which would itself be an important finding.

---

## Files

- `expedition_log.md` — Day 273 results
- `407_person_axis_named_entity.md` — second-order person axis
- `404_country_axis_second_order.md` — second-order country axis
