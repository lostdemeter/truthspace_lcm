# Self-Assembling Corpus: Progress Tracker

**Last Updated:** January 11, 2025
**Status:** Phase 3 Complete, Phase 4-6 Pending

---

## Goal

Build a **self-assembling knowledge corpus** that:

1. Compounds data geometrically (derive compounds from primitives)
2. Discovers dimensions emergently (not predefined)
3. Rebalances dynamically when new dimensions are added
4. Fills unknowns efficiently using a local LLM
5. Remains **geometrically pure** (no statistical weights, no lookup tables)

### The Ultimate Vision

Scale knowledge ingestion to large corpora by:
- Bootstrapping primitives once
- Deriving all compounds geometrically
- Only calling LLM for genuinely unknown concepts
- Target: <1 LLM call per 100 concepts (99% derived geometrically)

---

## Core Principles

### 1. The Music Box Principle (Design 112)

**The comb doesn't contain the music. The music emerges from the interaction of drum and comb.**

| Component | Music Box | Our System |
|-----------|-----------|------------|
| **Drum** | Cylinder with bumps | Words positioned in φ-space |
| **Comb** | Metal tines | `find_nearest(position)` decoder |
| **Music** | Sound produced | Output text that emerges |

**Violations to avoid:**
- ❌ Hard-coded word→word mappings (`"code" -> "holy scripture"`)
- ❌ Lookup tables for transformations
- ❌ Statistical weights that obscure structure

**Correct approach:**
- ✅ Words have positions in semantic space
- ✅ Transformations are delta vectors
- ✅ Output emerges from `position + delta → find_nearest`

### 2. Geometric Purity

From the project philosophy:

> **Structure IS information** - There are no opaque weights or embeddings
> **Geometry IS computation** - Traversal through geometric space produces outputs
> **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure

### 3. Fail-Fast Development

- **No graceful fallbacks** - If geometry fails, we see the error
- **No hard-coded workarounds** - Hard-coding violates the hypothesis
- **Prove or disprove** - Every component must work emergently

### 4. φ as the Fundamental Unit

- φ (golden ratio) = 1.618...
- All semantic distances are multiples of φ
- Source → Target delta = φ on each dimension
- Compound distance = φ√n where n = number of dimensions

---

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Core Infrastructure | ✅ Complete |
| **Phase 2** | Ingestion Pipeline | ✅ Complete |
| **Phase 3** | LLM Integration | ✅ Complete |
| **Phase 4** | Platonic Ideal Discovery | ⏳ Pending |
| **Phase 5** | Self-Assembly Loop | ⏳ Pending |
| **Phase 6** | Persistence & Versioning | ⏳ Pending |

---

## Phase 1: Core Infrastructure ✅

**Goal:** Build the foundational data structures and geometric positioning.

### Completed Components

1. **TransformationPair** - Source of truth for relationships
   - `(source, target, relationship, confidence)`
   - Everything derives from pairs

2. **Dimension** - Emergent dimension with introspectable metadata
   - `name, index, pole_negative, pole_positive`
   - Dimensions emerge from relationship types

3. **PlatonicIdeal** - Multi-dimension anchor detection
   - Concepts that sit at origin of multiple dimensions
   - Example: "house" anchors size AND regality

4. **SelfAssemblingCorpus** - Main orchestration class
   - `add_pair()` - adds pairs, creates dimensions automatically
   - `recompute()` - positions all concepts using φ-geometry
   - `find_nearest()` - the "comb" that reads positions
   - `transform()` - apply delta along dimension
   - `save()/load()` - persistence from pairs alone

### Key Findings

1. **Dimensions emerge automatically** from relationship types
2. **Platonic Ideals detected** by multi-dimension anchoring
3. **Positions extend automatically** when new dimensions added
4. **Self-similarity preserved** - all deltas = φ
5. **Corpus reconstructable** from pairs alone

### Files

- `/experiments/self_assembling_corpus.py` - Main implementation
- `/docs/design_considerations/114_emergent_dimensions_platonic_ideals.md`
- `/docs/design_considerations/115_self_assembling_corpus_roadmap.md`

---

## Phase 2: Ingestion Pipeline ✅

**Goal:** Extract transformation pairs from text and handle edge cases.

### Completed Components

1. **ConceptType Enum** - Distinguishes concept types
   - `CATEGORY` - General concept (e.g., "large dog", "mansion")
   - `INSTANCE` - Specific example (e.g., "mastiff", "labrador")
   - `IDEAL` - Platonic ideal (e.g., "dog", "house")
   - `UNKNOWN` - Not yet classified

2. **Concept** - Word with type information
   - `word, concept_type, parent, attributes`

3. **Gap** - Detected missing variation
   - `ideal, dimension, direction, description`
   - `to_query()` - generates LLM query

4. **IngestionPipeline** - Orchestrates text ingestion
   - `extract_pairs_from_text()` - regex-based pair extraction
   - `classify_concept()` - determines category vs instance
   - `detect_gaps()` - finds missing variations
   - `generate_llm_queries()` - batches queries by dimension

### The Mastiff Problem ⚠️

**Challenge:** "mastiff" is a specific breed, not a general "large dog".

```
Q: What is 'large + dog'?
A1: 'mastiff' - but this is a SPECIFIC BREED ❌
A2: 'large dog' - this is a GENERAL CATEGORY ✓
```

**Key insight:** Not every large dog is a mastiff, but every mansion IS a large house.

**Solution:** Instance vs Category detection:
- Known breeds/types → INSTANCE
- Compound descriptors ("large dog") → CATEGORY
- Context indicators ("breed of") → INSTANCE

### Classification Results

```
mastiff         for dog+size_increase: INSTANCE ✗
chihuahua       for dog+size_decrease: INSTANCE ✗
mansion         for house+size_increase: CATEGORY ✓
cottage         for house+size_decrease: CATEGORY ✓
large dog       for dog+size_increase: CATEGORY ✓
labrador        for dog+friendliness_increase: INSTANCE ✗
```

### Gap Detection & LLM Batching

```
Detected 8 gaps:
  house missing age_decrease variation
  dog missing regality_increase variation
  person missing size_increase variation

Batched LLM queries:
  For the 'age_decrease' dimension, what are variations of: house, dog?
  For the 'regality_increase' dimension, what are variations of: dog, person?
```

### Files

- `/experiments/self_assembling_corpus.py` - Extended with Phase 2 classes

---

## Phase 3: LLM Integration ✅

**Goal:** Connect to local LLM for gap filling and validation.

### Completed Components

1. **LLMInterface Class**
   - `is_available()`: Check if Ollama is running
   - `query()`: Send prompt, get response
   - `query_variation()`: Get variation word for ideal+dimension
   - `validate_instance_vs_category()`: Solve the mastiff problem
   - `query_batch_variations()`: Efficient batched queries

2. **LLMEnhancedPipeline Class**
   - `fill_gaps_with_llm()`: Query LLM for missing variations
   - `validate_existing_pairs()`: Check for instance issues

3. **Instance vs Category Validation (WORKING)**
   ```
   mastiff         (dog): INSTANCE ✗  (specific breed)
   mansion         (house): CATEGORY ✓ (general concept)
   labrador        (dog): INSTANCE ✗  (specific breed)
   cottage         (house): CATEGORY ✓ (general concept)
   chihuahua       (dog): INSTANCE ✗  (specific breed)
   palace          (house): CATEGORY ✓ (general concept)
   ```

4. **Gap Filling Results**
   - LLM suggested "dwarf" and "giant" for person size variations
   - Both correctly REJECTED as instances (mythological/specific)
   - LLM suggested "cottage" for house age_decrease → ACCEPTED

### Efficiency Achieved

- **2.5 pairs per LLM query** in demo
- Batch queries by dimension working
- Geometry guides LLM (not the other way around)

### Key Principle Maintained

The LLM is a **tool** used by the geometric system, not the driver:
- Geometry identifies gaps
- LLM suggests candidates
- Geometry validates and positions

---

## Phase 4: Platonic Ideal Discovery ⏳

**Goal:** Automatically identify fundamental concepts.

### Planned Components

1. **Ideal Detection Algorithm**
   - Count dimensions each word anchors
   - Words anchoring 2+ dimensions are candidates
   - Verify position is at origin

2. **Ideal Hierarchy**
   - Level 0: Universal ideals (5+ dimensions)
   - Level 1: Domain ideals (3-4 dimensions)
   - Level 2: Category ideals (2 dimensions)
   - Level 3: Specific ideals (1 dimension)

3. **Cross-Domain Ideals**
   - Are there universal Platonic Ideals?
   - Do they appear across all domains?

---

## Phase 5: Self-Assembly Loop ⏳

**Goal:** Continuous self-improvement cycle.

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-ASSEMBLY LOOP                        │
├─────────────────────────────────────────────────────────────┤
│  1. INGEST    → New text → Extract transformation pairs      │
│  2. DETECT    → New relationship type? → Create dimension    │
│  3. REBALANCE → New dimension → Extend all positions         │
│  4. POSITION  → Place concepts (source=0, target=φ)          │
│  5. DISCOVER  → Find Platonic Ideals                         │
│  6. GAP-FILL  → Identify missing → Queue for LLM             │
│  7. COMPOUND  → Derive compound positions geometrically      │
│  8. VERIFY    → Check self-similarity, introspect            │
│  └──────────────────── REPEAT ──────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 6: Persistence & Versioning ⏳

**Goal:** Robust storage and incremental updates.

### Planned Components

1. **JSON Storage Format**
   - Pairs as source of truth (append-only)
   - Dimensions, ideals as derived data
   - Version tracking for rebalances

2. **Incremental Updates**
   - Add pairs without full recompute
   - Lazy recomputation on query

3. **Reconstruction**
   - Space reconstructable entirely from pairs
   - No external dependencies

---

## Challenges Overcome

### 1. The Mastiff Problem

**Challenge:** Specific instances (breeds) vs general categories.

**Solution:** ConceptType enum + heuristic detection + LLM validation (Phase 3).

### 2. Dimension Explosion

**Challenge:** Each relationship type creates a dimension. Could explode.

**Solution:** 
- Dimensions emerge only from explicit pairs
- Compound dimensions detectable (regality = formality + wealth?)
- Future: dimension merging/splitting

### 3. Unnamed Compounds

**Challenge:** Some compound positions have no single word (e.g., "large dog").

**Solution:**
- Accept compound descriptors as valid categories
- Mark as `UnnamedCompound` with position
- Don't force a word where none exists

### 4. Article/Stopword Pollution

**Challenge:** Regex patterns captured "a", "the", "most" as concepts.

**Solution:** SKIP_WORDS set + improved regex patterns.

---

## Open Questions

1. **Dimension Merging:** What if two dimensions are the same? (e.g., "formality" and "register")

2. **Dimension Splitting:** What if one dimension should be two? (e.g., "regality" → "formality" + "wealth")

3. **Temporal Dimensions:** How do we handle word meanings that shift over time?

4. **Cross-Domain Ideals:** Are there universal Platonic Ideals across all domains?

5. **Scaling:** How does this perform with 10,000+ concepts?

---

## Key Files

| File | Purpose |
|------|---------|
| `/experiments/self_assembling_corpus.py` | Main implementation |
| `/experiments/concept_compounding.py` | Earlier experiments |
| `/docs/design_considerations/112_music_box_principle.md` | Core principle |
| `/docs/design_considerations/114_emergent_dimensions_platonic_ideals.md` | Platonic Ideals |
| `/docs/design_considerations/115_self_assembling_corpus_roadmap.md` | Full roadmap |

---

## How to Continue

If starting a new session, run:

```bash
cd /home/thorin/truthspace-lcm
source ./venv/bin/activate
python -m experiments.self_assembling_corpus
```

This will demonstrate all Phase 1 and Phase 2 functionality.

### Next Steps

1. **Phase 3:** Implement LLM integration for gap filling
2. **Phase 4:** Automatic Platonic Ideal discovery from corpus
3. **Phase 5:** Full self-assembly loop
4. **Phase 6:** Robust persistence

### Guiding Principles (Repeat Often)

- **Music Box Principle:** Output emerges from geometry, not lookup
- **φ is fundamental:** All distances are φ-based
- **Pairs are truth:** Everything derives from transformation pairs
- **Fail fast:** No graceful fallbacks that hide geometric failures
- **Structure IS information:** The shape IS the knowledge

---

*The geometry provides. The structure protects. The knowledge emerges.*
