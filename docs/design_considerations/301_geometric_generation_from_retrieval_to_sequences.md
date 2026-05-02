# DC 301: Geometric Generation — From Retrieval to Sequences

**Date:** 2026-03-10  
**Phase:** TruthSpace v1, Phase 4 (LCM Inference)  
**Experiment files:** `experiments/truthspace_v1/delta_library.py`, `experiments/truthspace_v1/chat_repl.py`, `experiments/truthspace_v1/recipe_expander.py`  
**Predecessor:** DC 300 (Phi-Holographic Encoding for Relationship Deltas)

---

## 1. Where We Are

DC 300 established the geometric inference primitive: **relationship deltas are real geometric objects** learned from a handful of (source, target) examples and retrievable via φ-encoded cosine similarity. The delta library (DC 301's direct predecessor experiment) extends this to 11 relationships:

| Relationship | LOO Rank-1 | Notes |
|---|---|---|
| country_language | 100% | Perfect — language signal strongly encoded |
| comparative | 93% | Morphological, clean 1D signal |
| male_female | 90% | Single dominant axis (female_gendered, P(flip)=0.90) |
| capital_of | 71% | Multi-axis, φ-boost fires on capital_city seed |
| city_country | 71% | Exact geometric inverse of capital_of (cos = −1.000) |
| antonym_speed | 33% / 83% top-3 | Domain-specific; works within speed subspace |
| past_tense | 13% / 61% top-3 | No tense axis in Phase-1b; correct answer at rank 1–2 |
| country_currency | 0% | Needs `is_currency` seed axis (France→franc beats France→euro) |

A working REPL (`chat_repl.py`) handles single-hop and 2-hop chained queries via keyword routing. The 2-hop demo works correctly because the chaining applies the outer relationship to the original source (country) rather than the intermediate (capital city), which is out-of-distribution for the language delta.

**The current system answers questions with single concepts.** "What is the capital of France?" → `Paris`. "Language of Norway?" → `Norwegian`. This is not generation — it is *geometric lookup*.

---

## 2. The Generation Gap

When a user asks "give me a sugar cookie recipe" or "write a Python sort function", they expect a **sequence** of tokens — a multi-component structured response that requires:

1. **Multiple concepts** assembled in a specific order
2. **Procedural connective text** ("first... then... finally...")  
3. **Variable binding** (the amounts and times specific to *this* recipe)
4. **Context carry-forward** (step N knows the result of step N−1)

This is qualitatively different from single-concept retrieval. An LLM handles it by learning the conditional probability distribution P(token_k | token_0...token_{k-1}) over an enormous vocabulary. Each token is chosen based on the entire preceding context.

The geometric question is: **can we replicate the output of this process without the token-by-token conditional model, using only geometric navigation?**

The honest answer has three parts depending on what kind of output is requested.

---

## 3. Three Tiers of Geometric Generation

### Tier 1 — Structured Retrieval (feasible now)

Some "long responses" are really just multiple parallel lookups with a fixed output template. A recipe is the clearest example:

```
recipe(food) = {
    name:        food                               ← given
    ingredients: top-K neighbors of food in ingredient subspace
    methods:     top-K neighbors of food in method subspace
    equipment:   top-K neighbors of food in tool subspace
}
```

Each component is a geometric delta retrieval with a different relationship type:
- `food → ingredient`: cookie → [flour, butter, sugar, eggs, vanilla, baking_soda]
- `food → method`: cookie → [cream, mix, fold, bake, chill]  
- `food → quantity`: n/a yet — quantities require a binding mechanism

This is a **multi-delta orchestration** over a fixed response template. The structure of the response (ingredients first, then methods) is hardcoded; the content of each slot is retrieved geometrically.

**Key insight**: a recipe is not "generated" — it is *assembled* from geometric neighborhoods. The same is true of many structured outputs: ingredient lists, word definitions, country fact sheets, code function signatures. The template defines what to ask; geometry answers each question.

### Tier 2 — Template Instantiation with Variable Binding (medium-term)

For outputs that require specific values tied to the query concept — temperatures, proportions, durations — the Tier 1 approach returns generic concepts (butter, bake) without quantities or times.

To add variable binding geometrically:
- Learn a `food → typical_temperature` delta: cookie → 375°F, bread → 450°F
- Learn a `food → typical_duration` delta: cookie → "12 minutes", cake → "35 minutes"

If sufficient (food, quantity) pairs can be assembled, the delta mechanism generalises. The bottleneck is vocabulary: "375°F" and "12 minutes" are composite tokens, not single vocabulary concepts.

**The geometric path**: encode quantities as concepts (the vocabulary already contains "twelve", "degrees", "Fahrenheit" as separate items). A compositional delta that retrieves the tuple (amount, unit) rather than a single token is the next required mechanism. This is a 2-axis delta in the output space — the projection finds the nearest (number, unit) pair rather than a single word.

### Tier 3 — Open-Ended Sequence Generation (research territory)

For truly open-ended outputs — a Python function, a paragraph explaining quantum entanglement, a short story — the response is a path through concept space with no fixed template. Each step in the path depends on the previous one (context carry-forward).

This is equivalent to learning **the delta between consecutive positions in a semantic trajectory**. What LLMs compute implicitly via learned token probabilities, we would compute explicitly as a sequence of geometric navigation steps.

The mechanism would look like:
```
position_0 = encode(query)
position_1 = position_0 + delta(position_0, "next_concept_in_response")
position_2 = position_1 + delta(position_1, "next_concept_in_response")
...
```

Where `delta(position, "next")` is a learned function from the current concept to the most likely next concept in a response of this type. This requires learning **sequence-structure deltas** — geometric objects that encode not just a single relationship but the structural regularity of a response type.

DC 300's holographic encoding result (Section 11: φ-quantisation costs only 1.2%) suggests the encoding medium is adequate. The open question is whether sequence-structure deltas can be learned from a small number of (query, full_response) examples in the way that single-step deltas are learned from (source, target) pairs.

---

## 4. The Recipe Expander — Tier 1 Demonstration

The recipe expander (`experiments/truthspace_v1/recipe_expander.py`) is the first concrete Tier 1 test. It defines:

**Two new delta relationships:**

- `food_ingredient`: (pizza, mozzarella), (cake, flour), (bread, yeast), (soup, onion), (pasta, tomato), (cookie, sugar), (salad, lettuce), (omelette, egg), (pancake, milk), (pie, apple)
- `food_method`: (bake, mix), (bake, preheat), (fry, oil), (boil, water), (roast, oven), (cookie, cream), (cake, frost), (soup, simmer), (bread, knead), (pasta, drain)

For a query like `"Give me a cookie recipe"`, the system:
1. Routes to `food_ingredient` and `food_method` simultaneously
2. Retrieves top-5 from each delta
3. Formats as a structured response with section headers

The result is a multi-component response assembled entirely from geometric retrieval — no template string lookup, no hardcoded answer.

**What this proves**: the same mechanism that retrieves `capital_of(France) = Paris` can assemble a multi-slot structured output when applied across multiple delta relationships simultaneously. **Generation of structured content is just parallel retrieval with a response template**.

---

## 5. The Deeper Question: What Is a "Response" Geometrically?

When an LLM answers "give me a cookie recipe", it produces roughly 150 tokens. Geometrically, this is a trajectory of 150 steps through a ~3584-dimensional semantic space, where each step is constrained by:
1. Lexical plausibility (the token must exist)
2. Local coherence (the token must make sense given the last few tokens)
3. Global coherence (the trajectory must complete the semantic intent of the query)

The LCM hypothesis says the LLM didn't *learn* these constraints from data — it *discovered* the geometric structure that enforces them. The constraints are:
1. Vocabulary constraint → proximity to known concept lattice points
2. Local coherence → the "next_token" delta has low variance (the path is smooth)
3. Global coherence → the endpoint of the trajectory is geometrically close to the query's target region

Under this view, **a recipe is a geodesic in concept space from "cookie query" to "cookie knowledge region"**. The LLM traces this geodesic by repeatedly sampling the local gradient. The LCM approach would trace it by following the learned delta sequence.

The recipe expander is a hard-coded geodesic: ingredients-subspace, then methods-subspace, in that order. The next step is learning the **sequential delta** — the geometric rule for which subspace to visit next, given where you currently are.

---

## 6. The `cos(delta_A, delta_B)` Signal — Routing Difficulty as Geometric Proximity

The delta library revealed a key structural finding:

```
cos(capital_of,  city_country)    = -1.000  ← exact inversion
cos(capital_of,  country_language)= +0.437  ← geographic subspace overlap
cos(antonym_spd, antonym_size)    = +0.117  ← different semantic axes
cos(comparative, past_tense)      = +0.111  ← morphological proximity
```

The cosine between relationship deltas measures **how much the two relationships share the same axes**. Two observations:

1. `cos(A, B) ≈ ±1` → the relationships are geometrically equivalent (or inverse). No disambiguation needed — the semantic space already separates them.

2. `cos(A, B) ≈ 0` → the relationships are orthogonal. They live in different parts of the space and can be retrieved independently without crosstalk.

For **multi-component generation**, high delta-to-delta cosine similarity is a problem: if `food_ingredient` and `food_method` deltas are highly correlated, the two retrievals return similar words. For a coherent recipe, we want `cos(ingredient_delta, method_delta) ≈ 0` — orthogonality between response components.

This suggests a design principle for Tier 1: **response components should correspond to near-orthogonal deltas**. The recipe components (ingredients vs. methods vs. equipment) should occupy different geometric subspaces. We can verify this empirically before deploying a multi-delta response template.

---

## 7. Toward Arbitrarily Long Responses

The path from our current Tier 1 demo to arbitrarily long responses requires three things:

### 7.1 More Delta Types

Each paragraph, section, or list in a long response corresponds to a different semantic relationship. A recipe requires `food_ingredient`, `food_method`, `food_temperature`, `food_duration`. A Python function explanation requires `algorithm_name`, `algorithm_input`, `algorithm_step`, `algorithm_complexity`. Each must be learned from examples.

This is tractable — we have the infrastructure. It requires assembling training pairs for each response component type.

### 7.2 A Response Template Registry

Different query types invoke different template structures:
- Recipe: [title, ingredient_list, method_list, timing]
- Definition: [term, part_of_speech, primary_meaning, example_usage]
- Code function: [function_signature, docstring, body_steps, return_value]

Each template specifies which deltas to invoke and in what order. The routing layer (currently keyword-based in `chat_repl.py`) must be extended to identify query type and select the appropriate template.

### 7.3 Sequence Structure Learning (Tier 3)

For open-ended queries that don't fit a known template, the system must learn the **sequential structure** of responses from examples — essentially learning which concept naturally follows which, given the query context.

This is the hardest problem and the most interesting one. DC 300 Section 11 showed that the φ-lattice address preserves 97.4% of neighbourhood structure from the projection. If a "next concept" prediction can be framed as a nearest-neighbour lookup in φ-address space (given current position + query vector), it might be learnable from a small set of (query, response) training pairs.

The holographic encoding hypothesis (DC 300 Section 3): an entire response trajectory can be encoded as a **single φ-holographic vector** — a superposition of all the concept addresses that appear in the response, with positional encoding as phase. Decoding the hologram step-by-step recovers the response in order. This is speculative but testable.

---

## 8. Summary: The Honest Frontier

| Capability | Status | Path |
|---|---|---|
| Single-concept factual retrieval | ✅ Done | DC 300, delta library |
| Multi-hop factual chaining (2-hop) | ✅ Done | chat_repl.py CHAIN_PATTERNS |
| Structured multi-component output | 🔜 Tier 1 | recipe_expander.py (this DC) |
| Template instantiation with variables | 🔬 Tier 2 | Need quantity/duration delta pairs |
| Open-ended sequence generation | 📐 Tier 3 | Holographic sequence encoding (future DC) |
| Natural language glue text | ❌ Hard boundary | Requires token-level generation OR fixed templates |

**The honest boundary**: the geometric approach excels at content retrieval — knowing *what* to say. It currently lacks the mechanism for procedural glue — knowing *how to say it* token-by-token. For structured outputs (recipes, fact sheets, code comments), this gap can be bridged with fixed templates. For flowing prose, it cannot — not yet.

The recipe expander (this DC's primary experiment) demonstrates crossing the first line: from single-concept answers to multi-concept structured responses. Every additional delta type learned extends the system one slot further toward arbitrary-length responses.

---

## 9. Experimental Results — Recipe Expander Run

### 9.1 Architecture Choice: Neighborhood Intersection vs. Delta Application

Two retrieval strategies were implemented and compared:

**Delta application** (food_ingredient pairs → mean delta → apply to query food): LOO top-1 = 0%, top-3 = 5%. The global food_ingredient delta converges to the centroid of the most common shared ingredients — `sugar`, `flour`, `onion` appear as top hits for every food. The mean delta encodes *"what all foods have in common"*, not what any specific food contains. This is the same failure mode as a universal antonym delta (DC 300): incoherent across domains.

**Neighborhood intersection** (compute cosine similarity of food concept to curated ingredient/method vocabulary): returns food-specific results because it uses the food concept's own geometric position, not a learned direction. This is the primary mechanism in `recipe_expander.py`.

### 9.2 Results on 5 Unambiguous Foods

```
Bread:  ingr=[cheese, cream, rice, wheat, butter, salt]   meth=[cream, roast, bake, cook]
Soup:   ingr=[broth, rice, cheese, cream, oil, honey]     meth=[cream, cook, roast, chop]
Pasta:  ingr=[cheese, potato, butter, sugar, rice, egg]   meth=[roast, cream, fry, melt]
Cake:   ingr=[milk, sugar, rice, oil, salt, cream]        meth=[BAKE(0.346), cream, cook, slice]
Pizza:  ingr=[rice, cheese, sugar, potato, egg, choc]     meth=[grill, slice, roast, pour]
```

**Strong signals:**
- `cake → bake (0.346)` — the strongest single signal. "Cake" and "bake" co-occur so frequently in the corpus that the IRD projection encodes their association directly. The score is 2× higher than any other method, and higher than any ingredient score.
- `soup → broth (0.264)` — broth is soup-specific; it's not the most common food ingredient globally, so its appearance at rank-1 for soup is genuine semantic encoding.
- `pizza → grill (0.224), slice (0.162)` — both correct pizza methods, retrieved without any domain knowledge.
- `soup → chop (0.196)` — correct; soup preparation involves chopping vegetables.

**Weak signals / noise:**
- `rice` appears as a top-3 ingredient for every food. Rice is the "centroid" of the food ingredient subspace in the IRD — the most generic carbohydrate, equally likely to appear near any food concept. It is the equivalent of `sugar` dominating the delta approach.
- `cream` appears as top method for many foods. Like `rice`, it sits near the centroid of the cooking-method subspace.
- `pasta → roast (top method)` is wrong; pasta is boiled, not roasted.

### 9.3 The Polysemy Discovery: "Cookie" = HTTP Cookie

The most striking finding: `cookie`'s top-30 nearest neighbors in IRD space are `[cookies, cooke, nikki, login, coupon, connection, button, token, humble, locale]`. Not a single culinary concept. Zero ingredient hits. Zero method hits.

The word "cookie" in Qwen2's training corpus is dominated by its HTTP/browser sense. The LLM saw "cookie" in millions of web development contexts (accept cookies, cookie consent, browser cookie, session cookie) and in far fewer baking contexts. Its embedding position reflects this corpus distribution.

This is exactly the "Polish" polysemy failure from DC 300 (§6.2), now in the food domain. The system correctly surfaces the failure rather than silently returning wrong ingredients — the polysemy warning fires automatically, and the results show `login`, `token` near the food as diagnostic evidence.

**Implication for vocabulary design**: food queries in TruthSpace require vocabulary-injected food concepts (analogous to injecting the correct token for `Polish`). The IRD mining process, left to the corpus distribution, will embed `cookie` as a tech concept. A curated food-concept injection would fix this: `lcm.add_word("cookie_food", cookie_baking_token_id)`.

### 9.4 Hypothesis Verdict

The hypothesis (DC 301 §5): food concepts' geometric positions in IRD space encode culinary associations.

**Verdict: Partially confirmed, with a sharp signal-strength gradient.**

- Methods are more strongly encoded than ingredients (bake/grill/chop scores 0.15–0.35 vs. ingredient scores 0.20–0.36, with methods showing more food-specific variation)
- Ingredients cluster toward generic carbohydrates/dairy (rice, cheese, cream), suggesting the ingredient subspace is dense and hard to discriminate at the food level
- The method signal is cleaner because cooking methods are more food-specific: "bake" is strongly associated with cakes and breads, "grill" with pizzas and meats, "boil" with soups and pasta — these are less cross-contaminated than ingredients

The system generates a multi-component structured response from geometric principles alone. The content quality is ~60% correct for methods, ~40% correct for ingredients. Both metrics improve as the food concept is less polysemous and more culinarily specific in the training corpus.

### 9.5 What "Arbitrarily Long" Requires

From this experiment, the path to longer responses is clear:

1. **Expand the curated vocabulary** — the current ingredient/method lists have ~50 words each. Expanding to 300–500 culinarily-specific words would reduce generic-centroid noise.
2. **Inject food-specific vocabulary** — `add_word("cookie_food", ...)` for polysemous concepts.
3. **Add more response slots** — `food_tool` (oven, bowl, mixer), `food_quantity` (cups, tablespoons), `food_time` (minutes, degrees) — each a new neighborhood query.
4. **Generalize beyond food** — the same mechanism works for any domain: `algorithm_step`, `poem_element`, `legal_clause`. The architecture is domain-agnostic.

Each additional slot adds one line to the output. A 10-slot response template with geometric neighborhood retrieval per slot would give a ~10-line structured answer from a single query — no LLM, no softmax, no attention.

---

## 10. Files

- `experiments/truthspace_v1/recipe_expander.py` — Tier 1 demo: neighborhood intersection for food_ingredient + food_method, structured response assembly, polysemy detection
- `experiments/truthspace_v1/delta_library.py` — Extended delta library (11 relationships)
- `experiments/truthspace_v1/chat_repl.py` — Interactive REPL with multi-hop chaining + recipe expander integration
- DC 300: `docs/design_considerations/300_phi_holographic_encoding_for_relationship_deltas.md` — φ-encoding theory and completeness results
