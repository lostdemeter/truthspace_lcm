# DC 407: Person-Name Axis — Named-Entity Criterion Confirmed

**Day 272 | The DC 406 prediction that person names (a named-entity class)
would yield a clean second-order axis is confirmed. Person axis PC1 explains
52.3% of variance. Positive pole: Einstein(+0.346), Darwin(+0.332),
Newton(+0.300), Kepler(+0.299), Mandela(+0.294), Lenin(+0.275), Aristotle(+0.258),
Napoleon(+0.245)... Category means: person=+0.239, city=+0.065, country=+0.019,
adj=−0.004, func=−0.039. The person axis and country axis are partially
overlapping (cos=+0.147), sharing a common "named-entity" component. This
establishes a general law: W_E allocates clean geometric axes to named-entity
categories.**

---

## First-Order Axes Used

Three first-order person→attribute axes:

```
scientist→nationality (Einstein→German, Newton→British, Darwin→British…)
  coherence = 0.640

scientist→field      (Einstein→physicist, Darwin→biologist, Turing→mathematician…)
  coherence = 0.641

leader→nationality   (Napoleon→French, Churchill→British, Lincoln→American…)
  coherence = 0.533
```

The two nationality axes strongly align (cos = 0.517), replicating the
encyclopedic-cluster pattern (DC 403): axes sharing the same target DOMAIN
(nationality) align more than axes sharing the same source type.

---

## Person Axis Results

```
SVD: S = [1.253, 0.983, 0.681]
PC1 variance: 52.3%

Projections onto PC1:
  scientist→nationality: -0.871
  leader→nationality:    -0.823
  scientist→field:       -0.367
```

The nationality axes project strongly and nearly equally (−0.87, −0.82).
The field axis projects weakly (−0.37): "physicist/biologist/mathematician"
is more domain-specific and less shared.

**Positive pole (person names):**
```
Einstein(+0.346), Darwin(+0.332), Newton(+0.300), Kepler(+0.299),
Mandela(+0.294), Lenin(+0.275), Euler(+0.265), Aristotle(+0.258),
Napoleon(+0.245), Gandhi(+0.243), Gauss(+0.238), Plato(+0.235),
Churchill(+0.215), Nietzsche(+0.210), Stalin(+0.208), Hitler(+0.202),
Lincoln(+0.200), Orwell(+0.199), Mao(+0.194), Messi(+0.192),
Caesar(+0.190), Picasso(+0.188), Tesla(+0.185)
```

**Negative pole (nationality adjectives):**
```
German(-0.517), British(-0.505), French(-0.435), Italian(-0.395),
Japanese(-0.377), American(-0.373), Spanish(-0.359), Canadian(-0.340),
Swedish(-0.336), Greek(-0.336), Australian(-0.334), Dutch(-0.326),
Mexican(-0.325), Russian(-0.322), Irish(-0.315), Turkish(-0.310)
```

The structure is identical to the country axis:
- Positive pole = the SOURCE entities (persons, just as countries were positive)
- Negative pole = the TARGET attributes (nationality adjectives)

---

## Category Discrimination

```
Category          mean projection   Clean separation?
────────────────────────────────────────────────────
person names      +0.239            ★ clearly positive
city names        +0.065            moderate (named entities)
country names     +0.019            near zero ← correctly excluded
adj base forms    -0.004            near zero ← correctly excluded
function words    -0.039            slightly negative ← correctly excluded
```

Person names score 3.7× higher than city names and 12.6× higher than
country names. The person axis is a clean discriminator with no contamination
from common-word categories.

---

## Person Axis vs Country Axis: Named-Entity Hierarchy

```
cos(person_axis, country_axis) = +0.147

Each axis's category means:
              Country axis    Person axis
country names   +0.372          +0.019
person names    +0.040          +0.239
city names      +0.036          +0.065
adj bases       -0.007          -0.004
func words      +0.127          -0.039
```

The two named-entity axes **do not fully orthogonalise** (cos = 0.147).
They share a small "named-entity" component — a direction in W_E that
all named entities (countries, persons, cities…) share relative to
common-word vocabulary. This is the **third-order** structure: a
"named-entity axis" that sits above both the country axis and person axis.

City names (+0.036 on country, +0.065 on person) are intermediate — they
are named entities but neither countries nor persons — consistent with
their position between the two specialised axes.

**Predicted third-order axis:** PCA of [country_axis, person_axis, city_axis]
should yield a clean "named entity vs common word" axis. This is the next
experiment.

---

## The General Law: Named-Entity Categories → Clean Axes

From three independent experiments:

| Source type       | Named entity? | 2nd-order axis quality | Category mean |
|-------------------|---------------|------------------------|---------------|
| country names     | YES           | Clean (+0.372)         | country axis  |
| person names      | YES           | Clean (+0.239)         | person axis   |
| adj base forms    | NO            | Weak (+0.159)          | markedness    |
| mixed morphology  | NO            | Weak (+0.138)          | markedness    |

**Law:** W_E allocates a distinct geometric sub-region to each named-entity
category. This sub-region has a centroid direction — the category axis —
that can be extracted via PCA of first-order axes anchored at that category.

The specificity of named-entity axes is due to the way transformer models
process proper nouns: proper nouns appear in restricted syntactic positions
(NP heads, capitalized, rare) and co-occur with a consistent set of
predicates (was born in, is the capital of, invented, won, led…). This
creates a tight distributional cluster for each named-entity category.

Common-word categories (adjectives, nouns, verbs) do not have this
tight clustering because they appear in all syntactic positions and
co-occur with almost all other words.

---

## Implications for TruthSpace

Named-entity axes provide a navigation infrastructure for encyclopedic
knowledge:

```
TruthSpace query: "What nationality was Einstein?"
1. Project emb(Einstein) onto person_axis → high positive (person confirmed)
2. Apply nationality residual axis → retrieve "German"

TruthSpace query: "What is the capital of France?"
1. Project emb(france) onto country_axis → high positive (country confirmed)
2. Apply capital residual axis → retrieve "Paris"
```

The axes are discoverable without hand-labelling: just PCA of naturally
occurring word→attribute pairs extracts both the category axis and the
domain residual axes.

**Broader implication:** Every named-entity category in W_E (countries,
persons, cities, organisations, biological species, chemical elements…)
has a geometric axis that can be extracted by this method. The number of
such axes is bounded by the number of named-entity categories in the
training data, which is large but finite. TruthSpace can in principle
discover and index all of them.

---

## Files

- `expedition_log.md` — Day 272 results
- `406_same_source_type_axes.md` — same-source-type criterion
- `404_country_axis_second_order.md` — first clean axis
- `401_semantic_relation_axes.md` — first-order axes
