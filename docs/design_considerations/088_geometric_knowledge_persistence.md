# Design Consideration 088: Geometric Knowledge Persistence

## Abstract

This document outlines the architecture for **persistent geometric knowledge storage** in TruthSpace LCM. The core insight is that the geometry itself—similarity matrices, eigendecompositions, positions—IS the knowledge. We should persist the geometry directly, not just the text that generates it.

Key design decisions:
1. **Hierarchical granularity**: Facts, fact-clusters, and topics as nested geometric structures
2. **Two-tier persistence**: Temporary (session) corpus vs curated (permanent) corpus
3. **Gear-native knowledge**: Knowledge management as an inherited capability of all Gears and GearChains
4. **Uniform architecture**: Entry point as a ChatGearChain for conceptual consistency

---

## Problem Statement

### Current State

Today, knowledge in TruthSpace LCM is:

1. **Ephemeral**: `ConversationalChain` builds corpus via LLM calls at startup; semantic dimensions discovered via SVD each time
2. **Scattered**: Multiple JSON files (`chat_corpus.json`, `holographic_patterns.json`, `plot_corpus.json`) with different schemas
3. **Text-centric**: We store text and reconstruct geometry, rather than storing geometry directly
4. **Gear-external**: Knowledge management is done by specific gears, not inherited by all gears

### The Tension

Our philosophy states:
- *"Structure IS information"*
- *"Geometry IS computation"*

But we persist text and reconstruct geometry. This is backwards.

### Goals

1. Persist geometric structure (similarity matrices, positions, quaternions)
2. Support hierarchical knowledge (facts → clusters → topics)
3. Separate temporary (learning) from permanent (curated) knowledge
4. Make knowledge management a core Gear/GearChain capability
5. Maintain JSON portability while planning for future optimization

---

## Architecture

### Three-Layer Knowledge Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    GEOMETRIC KNOWLEDGE STORE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LAYER 1: GEOMETRIC CORE (the actual knowledge)                │
│  ──────────────────────────────────────────────                │
│  - Similarity matrices S[i,j]                                  │
│  - Eigendecomposition (V, Λ)                                   │
│  - Positions P = V @ sqrt(Λ)                                   │
│  - Quaternion signatures per concept                           │
│  - Attractor basins and stability metrics                      │
│                                                                 │
│  LAYER 2: CONCEPT ANCHORS (what the geometry represents)       │
│  ────────────────────────────────────────────────────          │
│  - Word sets (content words per concept)                       │
│  - Co-occurrence counts                                        │
│  - Use/success statistics                                      │
│  - Parent/child relationships (hierarchy)                      │
│                                                                 │
│  LAYER 3: SURFACE TEXT (optional, for debugging/display)       │
│  ─────────────────────────────────────────────────────         │
│  - Original text snippets                                      │
│  - Source attribution                                          │
│  - Human-readable descriptions                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hierarchical Granularity

Knowledge scales from atomic facts to topic clusters:

```
TOPIC (e.g., "American Revolution")
├── CLUSTER (e.g., "Key Figures")
│   ├── FACT: "George Washington was Commander-in-Chief"
│   ├── FACT: "Benjamin Franklin was a diplomat"
│   └── FACT: "Thomas Jefferson wrote the Declaration"
├── CLUSTER (e.g., "Major Events")
│   ├── FACT: "Boston Tea Party occurred in 1773"
│   └── FACT: "Declaration signed July 4, 1776"
└── CLUSTER (e.g., "Outcomes")
    ├── FACT: "Treaty of Paris 1783 ended the war"
    └── FACT: "Constitution ratified 1788"
```

**Geometric representation**:
- Each FACT has a position in the space
- CLUSTER = centroid of its facts + cluster-level metadata
- TOPIC = centroid of its clusters + topic-level metadata
- Hierarchy encoded via parent_id references

This allows:
- Querying at any granularity (find a specific fact, or a whole topic)
- Facts about facts (meta-knowledge) as higher-level clusters
- Natural scaling as knowledge grows

### Two-Tier Persistence

```
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐     ┌─────────────────────┐           │
│  │  TEMPORARY CORPUS   │     │  PERMANENT CORPUS   │           │
│  │  (session memory)   │     │  (curated store)    │           │
│  ├─────────────────────┤     ├─────────────────────┤           │
│  │ - New facts from    │     │ - Bootstrapped on   │           │
│  │   LLM responses     │     │   startup           │           │
│  │ - Unverified        │     │ - Verified/curated  │           │
│  │ - High churn        │     │ - Stable            │           │
│  │ - Lost on restart   │     │ - Persisted to disk │           │
│  │   (unless promoted) │     │                     │           │
│  └──────────┬──────────┘     └──────────▲──────────┘           │
│             │                           │                       │
│             │    PROMOTION CRITERIA     │                       │
│             │    ─────────────────────  │                       │
│             │    - use_count >= N       │                       │
│             │    - success_rate >= 0.8  │                       │
│             │    - stability >= 0.9     │                       │
│             │    - manual curation      │                       │
│             └───────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Temporary Corpus**:
- Lives in memory during session
- Receives all new knowledge from LLM, user input, tool results
- Geometry updated incrementally (no full reproject)
- Cleared on restart unless explicitly promoted

**Permanent Corpus**:
- Loaded on startup (bootstrap)
- Contains curated, verified knowledge
- Updated cautiously via promotion or manual curation
- Geometry pre-computed and persisted

**Promotion Criteria**:
- `use_count >= 5`: Concept has been accessed multiple times
- `success_rate >= 0.8`: Queries using this concept succeeded
- `stability >= 0.9`: Position hasn't drifted significantly
- Manual override: User/admin can force promotion

---

## Gear-Native Knowledge Management

### The Insight

Currently, knowledge management is done by specific components (`ConversationalChain`, `CodeOrchestrator`). But if *"Structure IS information"*, then every Gear that transforms state should be able to:
- Read from a knowledge store
- Write to a knowledge store
- Have its own corpus

### Proposed Inheritance

```python
class Gear(ABC):
    """Base class for all gears."""
    
    # Existing
    def forward(self, state: GearState) -> GearState:
        pass
    
    # NEW: Knowledge management
    @property
    def knowledge_store(self) -> Optional[GeometricKnowledgeStore]:
        """This gear's knowledge store (if any)."""
        return self._knowledge_store
    
    def load_knowledge(self, path: str):
        """Load knowledge from file."""
        self._knowledge_store = GeometricKnowledgeStore.load(path)
    
    def save_knowledge(self, path: str):
        """Save knowledge to file."""
        if self._knowledge_store:
            self._knowledge_store.save(path)
    
    def add_knowledge(self, concept: Concept, temporary: bool = True):
        """Add a concept to this gear's knowledge."""
        if self._knowledge_store:
            self._knowledge_store.add(concept, temporary=temporary)
```

### GearChain Knowledge Aggregation

```python
class GearChain:
    """Chain of gears with aggregated knowledge."""
    
    def __init__(self, gears: List[Gear]):
        self.gears = gears
        self._chain_knowledge = GeometricKnowledgeStore()
    
    @property
    def knowledge_store(self) -> GeometricKnowledgeStore:
        """Aggregated knowledge from all gears in chain."""
        return self._chain_knowledge
    
    def aggregate_knowledge(self):
        """Merge knowledge from all gears into chain-level store."""
        for gear in self.gears:
            if gear.knowledge_store:
                self._chain_knowledge.merge(gear.knowledge_store)
```

### Entry Point as ChatGearChain

For conceptual uniformity, the main entry point should be a `ChatGearChain`:

```python
class ChatGearChain(GearChain):
    """
    The main chat application as a GearChain.
    
    This makes the entry point uniform with all other chains,
    enabling consistent knowledge management across the system.
    """
    
    def __init__(self):
        super().__init__([
            IntentClassifierGear(),
            ConversationalGear(),
            CodeOrchestratorGear(),
            ResponseComposerGear(),
        ])
        
        # Load permanent corpus on startup
        self.load_knowledge("corpus/permanent.json")
        
        # Initialize temporary corpus
        self._temporary_store = GeometricKnowledgeStore()
    
    def chat(self, message: str) -> str:
        """Process a chat message through the chain."""
        state = GearState(input=message)
        state = self.forward(state)
        
        # Any new knowledge goes to temporary store
        if state.new_knowledge:
            for concept in state.new_knowledge:
                self._temporary_store.add(concept)
        
        return state.output
    
    def promote_knowledge(self):
        """Promote qualifying temporary knowledge to permanent."""
        for concept in self._temporary_store.concepts:
            if self._qualifies_for_promotion(concept):
                self.knowledge_store.add(concept, temporary=False)
                self._temporary_store.remove(concept)
        
        self.save_knowledge("corpus/permanent.json")
```

---

## File Format

### Schema (JSON)

```json
{
  "version": "1.0",
  "type": "geometric_knowledge_store",
  "metadata": {
    "created": "2025-01-04T04:25:00Z",
    "modified": "2025-01-04T04:25:00Z",
    "concept_count": 150,
    "dims": 12,
    "gear_id": "conversational_chain",
    "tier": "permanent"
  },
  "geometry": {
    "similarity_matrix": [
      [1.0, 0.3, 0.1],
      [0.3, 1.0, 0.5],
      [0.1, 0.5, 1.0]
    ],
    "eigenvalues": [1.8, 1.2, 0.5],
    "eigenvectors": [
      [0.5, 0.7, 0.5],
      [0.7, -0.5, 0.5],
      [0.5, 0.5, -0.7]
    ],
    "positions": [
      [0.67, 0.77, 0.35],
      [0.94, -0.55, 0.35],
      [0.67, 0.55, -0.49]
    ]
  },
  "concepts": [
    {
      "id": "george_washington_001",
      "parent_id": "founding_fathers",
      "level": "fact",
      "words": ["george", "washington", "president", "founding", "first"],
      "quaternion": [1.0, 0.0, 0.0, 0.0],
      "position_index": 0,
      "use_count": 5,
      "success_count": 4,
      "stability": 0.95,
      "created": "2025-01-04T04:20:00Z",
      "source": "llm_knowledge"
    },
    {
      "id": "founding_fathers",
      "parent_id": "american_revolution",
      "level": "cluster",
      "words": ["founding", "fathers", "framers", "founders"],
      "quaternion": [0.9, 0.1, 0.0, 0.0],
      "position_index": 1,
      "use_count": 12,
      "success_count": 10,
      "stability": 0.98,
      "created": "2025-01-04T04:15:00Z",
      "source": "bootstrap"
    }
  ],
  "text_cache": {
    "george_washington_001": [
      "George Washington was the first President of the United States.",
      "He served as Commander-in-Chief during the Revolutionary War."
    ],
    "founding_fathers": [
      "The Founding Fathers were the leaders who founded the United States."
    ]
  },
  "co_occurrence": {
    "george": {"washington": 15, "president": 8, "first": 6},
    "washington": {"george": 15, "president": 10, "commander": 5}
  }
}
```

### File Naming Convention

```
corpus/
├── permanent/
│   ├── chat_chain.knowledge.json      # ChatGearChain's permanent store
│   ├── code_orchestrator.knowledge.json
│   └── intent_classifier.knowledge.json
├── temporary/
│   └── session_2025_01_04.knowledge.json  # Current session's temp store
└── backups/
    └── chat_chain.knowledge.backup_20250104_042500.json
```

### Future: Custom File Type

When JSON becomes a bottleneck (large matrices), we can introduce `.gks` (Geometric Knowledge Store):

```
file.gks (zip archive containing):
├── manifest.json       # Metadata, concept list
├── geometry.npz        # NumPy arrays (similarity, positions)
├── text_cache.json     # Optional text for debugging
└── co_occurrence.npz   # Sparse matrix of co-occurrences
```

This maintains portability (JSON manifest) while enabling efficient storage (NumPy binaries).

---

## Operations

### Core Operations

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `add(concept)` | Add concept to store, update geometry incrementally | O(n) for similarity row |
| `query(text)` | Project text, find nearest concepts | O(n) for projection |
| `remove(concept_id)` | Remove concept, reproject if needed | O(n²) if reproject |
| `merge(other_store)` | Combine two stores | O(n²) for reproject |
| `promote(concept_id)` | Move from temporary to permanent | O(1) |
| `save(path)` | Persist to disk | O(n) |
| `load(path)` | Load from disk | O(n) |

### Incremental Geometry Updates

When adding a concept, we don't need full eigendecomposition:

```python
def add_concept_incremental(self, concept: Concept):
    """Add concept with incremental geometry update."""
    n = len(self.concepts)
    
    # Compute similarity to all existing concepts
    new_similarities = [
        self.word_overlap(concept.words, c.words) 
        for c in self.concepts
    ]
    
    # Extend similarity matrix
    new_row = new_similarities + [1.0]  # Self-similarity = 1
    self.similarity_matrix = np.vstack([
        np.hstack([self.similarity_matrix, np.array(new_similarities).reshape(-1, 1)]),
        new_row
    ])
    
    # Approximate new position via weighted average
    if sum(new_similarities) > 0:
        weights = np.array(new_similarities) / sum(new_similarities)
        new_position = weights @ self.positions
    else:
        new_position = np.zeros(self.dims)
    
    self.positions = np.vstack([self.positions, new_position])
    self.concepts.append(concept)
    
    # Mark for full reproject if drift detected
    self._check_drift()
```

### Promotion Logic

```python
def qualifies_for_promotion(self, concept: Concept) -> bool:
    """Check if concept should be promoted to permanent."""
    return (
        concept.use_count >= 5 and
        concept.success_rate >= 0.8 and
        concept.stability >= 0.9
    )

def promote_all_qualifying(self):
    """Promote all qualifying concepts from temporary to permanent."""
    promoted = []
    for concept in self.temporary_store.concepts:
        if self.qualifies_for_promotion(concept):
            self.permanent_store.add(concept, temporary=False)
            promoted.append(concept.id)
    
    for concept_id in promoted:
        self.temporary_store.remove(concept_id)
    
    if promoted:
        self.permanent_store.save()
        logger.info(f"Promoted {len(promoted)} concepts to permanent store")
```

---

## Implementation Phases

### Phase 1: GeometricKnowledgeStore Class
**Goal**: Single class that handles all knowledge persistence

Files to create:
- `truthspace_lcm/core/knowledge/geometric_store.py`
- `truthspace_lcm/core/knowledge/concept.py`
- `truthspace_lcm/core/knowledge/__init__.py`

Key features:
- Load/save JSON
- Add/remove concepts
- Query by text
- Incremental geometry updates

### Phase 2: Gear Integration
**Goal**: Make knowledge management a Gear capability

Files to modify:
- `truthspace_lcm/core/gear.py` - Add knowledge_store property
- `truthspace_lcm/core/protocol.py` - Add new_knowledge to GearState

Key features:
- Optional knowledge_store per gear
- Knowledge flows through GearState
- Chains aggregate gear knowledge

### Phase 3: Two-Tier System
**Goal**: Separate temporary and permanent stores

Files to create:
- `truthspace_lcm/core/knowledge/knowledge_manager.py`

Key features:
- Temporary store for session
- Permanent store bootstrapped on startup
- Promotion logic
- Periodic save of permanent store

### Phase 4: ChatGearChain Refactor
**Goal**: Entry point as uniform GearChain

Files to modify:
- `truthspace_lcm/practical_applications/chat/chat.py`
- `run.py`
- `run_api.py`

Key features:
- ChatGearChain wraps existing functionality
- Consistent knowledge management
- Same interface, cleaner architecture

### Phase 5: Migration
**Goal**: Move existing corpus files to new format

Tasks:
- Convert `chat_corpus.json` → `chat_chain.knowledge.json`
- Convert `holographic_patterns.json` → `code_orchestrator.knowledge.json`
- Update all load/save calls
- Remove deprecated corpus code

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Startup time | < 2s | Time to load permanent corpus |
| Query latency | < 50ms | Time to find nearest concepts |
| Memory usage | < 100MB | For 10K concepts |
| Promotion accuracy | > 90% | Promoted concepts remain useful |
| Code reduction | > 30% | Lines removed by unifying corpus code |

---

## Open Questions

1. **Garbage collection**: How do we handle concepts that are never used? Decay over time? Manual pruning?

2. **Conflict resolution**: What if temporary and permanent stores have conflicting concepts? Permanent wins? Merge?

3. **Versioning**: How do we handle schema changes? Migration scripts? Version field in JSON?

4. **Distributed stores**: Future consideration - can stores be sharded across nodes?

---

## References

- Design Consideration 026: Generalized Knowledge Ingestion
- Design Consideration 084: Holographic Pattern Projection
- Design Consideration 085: Temporary Module Injection
- `truthspace_lcm/core/utils/holographic_pattern_space.py` - Current pattern storage
- `truthspace_lcm/core/chains/conversational_chain.py` - Current corpus handling

---

## Conclusion

This design unifies knowledge management across TruthSpace LCM by:

1. **Persisting geometry directly** - The structure IS the knowledge
2. **Scaling granularity** - Facts, clusters, and topics in one hierarchy
3. **Separating concerns** - Temporary for learning, permanent for reliability
4. **Making it native** - Every Gear can manage knowledge
5. **Staying uniform** - Entry point as ChatGearChain

The result is a system where knowledge truly is geometric structure, persisted and managed consistently across all components.
