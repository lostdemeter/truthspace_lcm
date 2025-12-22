# Autonomous Bootstrap: Beyond the Coupon Collector Problem

## The Current Situation

We've built a pattern-based extraction system that:
1. Uses regex patterns defined in JSON
2. Discovers patterns through corpus analysis
3. Requires manual curation to fix edge cases

**Results so far:**
- 70 patterns covering 9 books
- 140-325% improvement in extraction
- But: diminishing returns as we add more patterns

## The Coupon Collector Problem

If there are N distinct "pattern types" in language, collecting them all requires O(N log N) observations on average. But language is generative - N is effectively infinite. We'll never cover every case.

**Current approach limitations:**
1. Each new pattern adds less value (diminishing returns)
2. Patterns are brittle (regex breaks on variations)
3. No mechanism to handle unseen patterns
4. Manual intervention required for edge cases

## Key Insight: Error-Driven Construction

From our previous work, we know: **Error = Where to Build**

Instead of trying to enumerate all patterns upfront, we should:
1. Start with minimal patterns
2. Let the system encounter failures
3. Use failures to guide construction
4. Build structure where it's needed

## The Fundamental Shift

**From:** Pattern enumeration (coupon collecting)
**To:** Error-driven construction (building where needed)

### What This Means Practically

Instead of:
```
patterns = [pattern1, pattern2, ..., patternN]
for sentence in text:
    for pattern in patterns:
        if pattern.matches(sentence):
            extract(sentence, pattern)
```

We need:
```
for sentence in text:
    # Try to understand the sentence
    result = understand(sentence)
    
    if result.confidence < threshold:
        # This is where we need to build structure
        learn_from_failure(sentence, result)
```

## Three Approaches to Consider

### 1. Geometric/Statistical Approach
Use the geometric structure we already have:
- Words have positions in semantic space
- Sentences are trajectories through this space
- Relations are geometric patterns (not regex patterns)

**Advantage:** Generalizes naturally, no enumeration needed
**Challenge:** How to extract discrete facts from continuous geometry?

### 2. Template Induction
Learn templates from examples rather than hand-coding:
- Given: "Alice loves Bob" → (Alice, loves, Bob)
- Induce: "[ENTITY] [VERB] [ENTITY]" template
- Apply to new sentences

**Advantage:** Learns from data, not hand-coded
**Challenge:** Still requires labeled examples

### 3. Compositional Semantics
Build meaning compositionally from word meanings:
- Each word contributes meaning
- Sentence meaning = composition of word meanings
- Relations emerge from composition

**Advantage:** Truly generative, handles novel sentences
**Challenge:** Requires understanding of composition rules

## Proposed Architecture: Hybrid Approach

Combine all three:

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS BOOTSTRAP                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. GEOMETRIC LAYER (continuous)                            │
│     - Words → positions in semantic space                   │
│     - Sentences → trajectories                              │
│     - Similarity = geometric distance                       │
│                                                              │
│  2. PATTERN LAYER (discrete, learned)                       │
│     - Templates induced from successful extractions         │
│     - Patterns ranked by success rate                       │
│     - Low-confidence patterns trigger learning              │
│                                                              │
│  3. COMPOSITION LAYER (generative)                          │
│     - Word classes (ENTITY, VERB, RELATION)                 │
│     - Composition rules (learned from data)                 │
│     - Handles novel combinations                            │
│                                                              │
│  4. ERROR-DRIVEN LEARNING                                   │
│     - Track extraction failures                             │
│     - Cluster similar failures                              │
│     - Induce new patterns from clusters                     │
│     - Validate on held-out data                             │
│     - Integrate successful patterns                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## The Key Innovation: Confidence-Weighted Learning

Instead of binary "matched/didn't match":

```python
def extract(sentence):
    # Try all approaches
    geometric_result = geometric_extract(sentence)
    pattern_result = pattern_extract(sentence)
    composition_result = composition_extract(sentence)
    
    # Combine with confidence weighting
    combined = weighted_combine([
        geometric_result,
        pattern_result, 
        composition_result
    ])
    
    if combined.confidence > HIGH_THRESHOLD:
        return combined.facts
    elif combined.confidence > LOW_THRESHOLD:
        # Tentative extraction - flag for validation
        return combined.facts, needs_validation=True
    else:
        # Failure - this is where we learn
        record_failure(sentence, combined)
        return []
```

## Handling the Coupon Collector Problem

The coupon collector problem assumes we need to collect all coupons. But we don't need all patterns - we need **enough coverage**.

**Strategy: Pareto Principle**
- 20% of patterns cover 80% of cases
- Focus on high-value patterns first
- Accept that some cases won't be covered
- Use geometric fallback for edge cases

**Strategy: Graceful Degradation**
- High confidence → extract facts
- Medium confidence → extract with uncertainty
- Low confidence → don't extract, but learn

**Strategy: Active Learning**
- Track which sentences fail extraction
- Cluster failures by similarity
- Prioritize learning patterns for large clusters
- Ignore rare edge cases (not worth the complexity)

## Implementation Roadmap

### Phase 1: Geometric Fallback
When patterns fail, use geometric similarity to find related known facts.
- "X verbed Y" similar to "A verbed B" → infer same relation

### Phase 2: Template Induction
Automatically induce templates from successful extractions.
- Track (sentence, extracted_facts) pairs
- Find common structures
- Generalize to templates

### Phase 3: Composition Rules
Learn how word classes combine to form relations.
- ENTITY + VERB + ENTITY → (E1, V, E2)
- ENTITY + "is a" + NOUN → (E, is_a, N)

### Phase 4: Autonomous Learning Loop
```python
while True:
    # Process new text
    results = extract(new_text)
    
    # Learn from failures
    failures = get_recent_failures()
    clusters = cluster_failures(failures)
    
    for cluster in clusters:
        if cluster.size > threshold:
            new_pattern = induce_pattern(cluster)
            if validate(new_pattern):
                add_pattern(new_pattern)
    
    # Prune low-value patterns
    prune_patterns(min_success_rate=0.1)
```

## Success Metrics

1. **Coverage**: % of sentences where we extract something
2. **Precision**: % of extractions that are correct
3. **Recall**: % of true facts that we extract
4. **Autonomy**: Time between manual interventions

Goal: Maximize coverage and precision while minimizing manual intervention.

## Connection to Prior Work

This connects to several insights from our previous experiments:

1. **Error = Where to Build** (from encoder work)
   - Failures point to missing structure
   
2. **Attractor/Repeller Dynamics** (from vocabulary work)
   - Similar patterns attract, dissimilar repel
   - Patterns self-organize through use
   
3. **Holographic Encoding** (from VSA work)
   - Meaning is distributed, not localized
   - Similarity is geometric, not symbolic

## Empirical Finding: Zipf's Law Applies

Analysis of 4500 sentences across 9 books shows:

| Rank | Pattern | % | Cumulative |
|------|---------|---|------------|
| 1 | is_a | 29.4% | 29.4% |
| 2 | speaks | 18.7% | 48.1% |
| 3 | moves | 17.7% | 65.7% |
| 4 | associated_with | 12.1% | 77.9% |
| 5 | acts | 6.4% | 84.3% |
| 6 | perceives | 4.5% | 88.7% |
| 7 | thinks | 4.5% | 93.2% |
| 8 | role | 4.0% | 97.2% |

**8 semantic categories cover 97% of extractions.**

### Implications

1. **The coupon collector problem is bounded** - we don't need infinite patterns
2. **Semantic categories > specific verbs** - "moves" covers walked/ran/turned/entered
3. **Diminishing returns are STEEP** - after top 8, each new pattern adds <1%
4. **Focus on coverage, not completeness** - 97% is good enough

### Revised Strategy

Instead of collecting more patterns, focus on:

1. **Robust semantic categories** - Make the 8 core categories bulletproof
2. **Geometric fallback** - Use similarity for edge cases
3. **Graceful degradation** - Accept that 3% won't be covered

The autonomous system doesn't need to learn new patterns - it needs to:
1. Correctly classify verbs into the 8 categories
2. Handle variations within each category
3. Use geometric similarity when classification fails

## The Real Problem: Verb Classification

The bottleneck isn't pattern discovery - it's **verb classification**.

Given a sentence like "Elizabeth pondered the situation", we need to:
1. Recognize "pondered" as a verb
2. Classify it into "thinks" category
3. Extract (Elizabeth, thinks, true)

This is a **classification problem**, not a **pattern enumeration problem**.

### Solution: Geometric Verb Classification

Use the geometric structure we already have:
- Each verb has a position in semantic space
- Categories are regions in that space
- Classification = finding nearest category

```python
def classify_verb(verb):
    verb_position = vocabulary.get_position(verb)
    
    category_centers = {
        'speaks': vocabulary.encode('said asked replied'),
        'moves': vocabulary.encode('walked ran turned'),
        'thinks': vocabulary.encode('thought believed knew'),
        # etc.
    }
    
    best_category = max(category_centers, 
                        key=lambda c: cosine_similarity(verb_position, category_centers[c]))
    
    return best_category
```

This handles novel verbs automatically - no pattern enumeration needed.

## Implementation: Conceptual Vocabulary

Created `truthspace_lcm/core/conceptual_vocabulary.py` with:

### Conceptual Primitives
Orthogonal semantic dimensions that span the meaning space:
- **ACTION** vs **STATE** vs **THING** (ontological)
- **MOTION**, **SPEECH**, **MENTAL**, **PERCEPTION**, **EMOTION**, **PHYSICAL**, **SOCIAL** (domains)
- **POSITIVE** vs **NEGATIVE** (valence)

### Seed Words
~110 words with manually assigned primitives (verbs and nouns).

### Syntactic Position Learning
Automatically learns word categories from text:
- Words after ENTITY with -ed/-ing → likely verbs → assign ACTION primitive
- Words after "the/a/an" → likely nouns → assign THING primitive
- Learned **1,285 words** from Dracula with **94% accuracy**

### Verb Classification
Classifies verbs into semantic categories by geometric similarity:
- **speaks**: ACTION + SPEECH
- **moves**: ACTION + MOTION
- **thinks**: STATE + MENTAL
- **perceives**: ACTION + PERCEPTION
- **feels**: STATE + EMOTION
- **acts**: ACTION + PHYSICAL
- **is_state**: STATE (pure state verbs)

### Results
- Geometric extraction: 0.072 facts/sentence
- Regex extraction: 0.074 facts/sentence
- **Comparable performance, but geometric is learned, not hand-coded**

## Key Insight: The Real Bottleneck

Both approaches find ~0.07 facts/sentence because the fundamental pattern (ENTITY + verb) is rare in literature. Most verbs follow pronouns ("He said", "She looked"), not capitalized entities.

To improve extraction, we need:
1. **Coreference resolution** - Link pronouns to entities
2. **Sentence-level parsing** - Understand full sentence structure
3. **Multi-sentence context** - Track entities across sentences

## The Autonomous Bootstrap Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 AUTONOMOUS BOOTSTRAP                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. CONCEPTUAL PRIMITIVES (fixed, orthogonal)               │
│     - 12 semantic dimensions                                │
│     - Define the basis of meaning space                     │
│                                                              │
│  2. SEED VOCABULARY (minimal, hand-coded)                   │
│     - ~110 words with known primitives                      │
│     - Bootstrap the learning process                        │
│                                                              │
│  3. SYNTACTIC POSITION LEARNING (automatic)                 │
│     - Learn word categories from text structure             │
│     - No manual labeling required                           │
│     - Grows vocabulary autonomously                         │
│                                                              │
│  4. GEOMETRIC CLASSIFICATION (generalizes)                  │
│     - Classify new words by similarity to known words       │
│     - 8 semantic categories cover 97% of extractions        │
│     - Handles novel verbs automatically                     │
│                                                              │
│  5. ERROR-DRIVEN REFINEMENT (future)                        │
│     - Track extraction failures                             │
│     - Adjust primitive assignments based on errors          │
│     - Self-correcting through use                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## What We Solved

1. **Coupon Collector Problem**: Bounded by Zipf's Law - 8 categories cover 97%
2. **Manual Pattern Curation**: Replaced with syntactic position learning
3. **Brittle Regex**: Replaced with geometric similarity
4. **Novel Word Handling**: Automatic via primitive inference

## What Remains

1. **Pronoun Resolution**: Most verbs follow pronouns, not entities
2. **Sentence Parsing**: Need to understand full structure
3. **Cross-sentence Context**: Track entities across sentences
4. **Error-driven Refinement**: Self-correction loop

## Files Created

- `truthspace_lcm/core/conceptual_vocabulary.py` - Conceptual primitive vocabulary
- `truthspace_lcm/core/geometric_attention.py` - Pronoun resolution and entity tracking
- `truthspace_lcm/core/semantic_extractor.py` - Geometric extraction (prototype)
- `truthspace_lcm/conceptual_bootstrap.json` - Learned vocabulary from 9 books
- `scripts/corpus_discovery.py` - Cross-book pattern discovery
- `scripts/build_corpus_knowledge.py` - Verb usage analysis
- `scripts/auto_improve_bootstrap.py` - Auto-improvement loop

---

## Experimental Results: Cross-Language Generalization

### English Literary Works (9 books, 500 sentences each)

| Book | Facts/sentence | Direct | Resolved |
|------|----------------|--------|----------|
| Alice in Wonderland | **0.884** | 159 | 283 |
| Tom Sawyer | **0.670** | 98 | 237 |
| Pride & Prejudice | **0.624** | 109 | 203 |
| Tale of Two Cities | 0.524 | 50 | 212 |
| Sherlock Holmes | 0.494 | 52 | 195 |
| Dracula | 0.480 | 37 | 203 |
| Frankenstein | 0.414 | 34 | 173 |
| Great Expectations | 0.392 | 60 | 136 |
| Moby Dick | 0.236 | 44 | 74 |
| **Average** | **0.524** | - | - |

### Held-Out English Book

| Book | Facts/sentence |
|------|----------------|
| White Fang | 0.190 |

### Translated Works (English translations)

| Book | Origin | Facts/sentence |
|------|--------|----------------|
| Don Quixote | Spanish→EN | 0.238 |
| War and Peace | Russian→EN | 0.178 |
| Les Misérables | French→EN | 0.144 |

### Original Non-English (Spanish)

| Book | Language | Facts/sentence |
|------|----------|----------------|
| Don Quixote | Spanish | 0.058 |

### Key Findings

1. **Conceptual primitives are language-agnostic** - ACTION, STATE, THING, MOTION, SPEECH work across languages
2. **Syntactic position learning generalizes** - Works for Spanish verb detection (Entity + verb-ending patterns)
3. **Translation preserves conceptual structure** - Translated works perform ~30-50% of native English
4. **Original non-English needs language-specific tuning** - Spanish word order differs from English

### Why Spanish Extraction is Lower

1. **Word order flexibility** - Spanish allows "Respondió Don Quijote" (verb-first)
2. **Pro-drop** - Spanish often omits subject pronouns ("Dijo que..." vs "He said that...")
3. **Verb conjugation complexity** - More endings to detect
4. **Our patterns assume English SVO order** - Need Spanish-specific patterns

### Implications for Multilingual Bootstrap

The conceptual approach **can** generalize to other languages because:
- Primitives (ACTION, MOTION, SPEECH) are universal semantic categories
- Syntactic position learning adapts to each language's patterns
- The geometric structure is language-independent

To fully support a new language, we need:
1. Language-specific verb ending patterns
2. Language-specific article/determiner patterns
3. Seed words with primitive assignments
4. Pronoun gender mappings

The **conceptual bootstrap file** (`conceptual_bootstrap.json`) provides a template that can be adapted for any language.

---

## Breakthrough: Concept Language (Order-Free Interlingua)

### The Insight

Instead of trying to match English SVO or Spanish VSO patterns, we work in **concept space** - like Chinese, which has no verb conjugation and flexible word order.

### The Architecture

```
Surface Text (any language)
        ↓
   Language-Specific Parser
        ↓
   CONCEPT FRAME (order-free)
   {AGENT: X, ACTION: Y, PATIENT: Z, LOCATION: W}
        ↓
   Vector Representation (language-agnostic)
        ↓
   Storage / Query / Match
```

### Concept Frame

A language-agnostic semantic frame with slots:
- **AGENT**: Who performs the action
- **ACTION**: Primitive (MOVE, SPEAK, THINK, PERCEIVE, FEEL, ACT, EXIST)
- **PATIENT**: Who/what is affected
- **LOCATION/GOAL/SOURCE**: Spatial relations

No word order - just slots filled with concepts.

### Results

| Language | Previous Rate | Concept Language | Improvement |
|----------|---------------|------------------|-------------|
| English | 0.524 | **0.852** | 1.6x |
| Spanish | 0.058 | **0.806** | **14x** |

### Cross-Language Queries

Query `{ACTION: SPEAK}` returns both:
- `[EN] "Bingley," cried his wife...`
- `[ES] "También lo juro yo —dijo el labrador..."`

The same conceptual primitive (SPEAK) matches "cried" and "dijo".

### Why This Works

1. **Primitives are universal** - MOVE, SPEAK, THINK exist in all languages
2. **Order doesn't matter** - We identify components by TYPE, not position
3. **Conjugation is abstracted** - "walked/walks/walking" and "caminó/caminaba" all map to MOVE
4. **Storage is language-agnostic** - English and Spanish frames live in the same vector space

### Files Created

- `truthspace_lcm/core/concept_language.py` - Concept frame extraction and storage
- Verb mappings for English (~150 verbs) and Spanish (~100 verbs)
- Relation mappings for spatial/semantic roles

### Connection to Holographic Projection

From design doc 030:
- A statement can be **projected** onto different question-type axes
- The concept frame IS the holographic representation
- Different queries (WHO/WHAT/WHERE) project onto different slots

The concept language is the **interlingua** that enables holographic Q&A across languages.
