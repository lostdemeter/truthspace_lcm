# Design Consideration 116: Attachable Corpus Layers

## The Problem

A single corpus that contains all knowledge has several issues:

1. **Saturation** - Domain-specific knowledge pollutes base language understanding
2. **No Forgetting** - Can't "detach" a topic when it's no longer relevant
3. **No Modularity** - Can't share base knowledge across different applications
4. **RAG Limitation** - Traditional RAG retrieves text, not geometric structure

## The Solution: Layered Corpora with Shared Dimensions

```
┌─────────────────────────────────────────────────────────────┐
│                    CORPUS STACK                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CONTEXT LAYER (ephemeral)                          │    │
│  │  - Current conversation                              │    │
│  │  - Temporary concepts                                │    │
│  │  - Discarded after session                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓ hooks                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DOMAIN LAYER (attachable)                          │    │
│  │  - Topic-specific knowledge                          │    │
│  │  - Chess, Medicine, Law, etc.                        │    │
│  │  - Attach/detach at will                             │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓ hooks                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  BASE LAYER (permanent)                             │    │
│  │  - Language fundamentals                             │    │
│  │  - Core dimensions (gender, age, size, etc.)         │    │
│  │  - Platonic Ideals                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## The Music Box Principle

```
BASE CORPUS = The music box mechanism (fixed gears, pins, comb)
DOMAIN CORPUS = Interchangeable cylinder (different tune)
CONTEXT CORPUS = Current rotation position (where we are in the tune)

The mechanism (base) is always the same.
The cylinder (domain) determines what can be played.
The position (context) determines what IS being played.
```

## Shared Dimensions = Hooks

The key insight: **dimensions are the hooks between layers**.

```
BASE LAYER defines:
  - gender dimension (index 0)
  - age dimension (index 1)
  - size dimension (index 2)
  - regality dimension (index 3)
  ...

DOMAIN LAYER (Chess) uses:
  - SAME dimensions, SAME indices
  - "chess king" uses regality dimension
  - "pawn" uses regality dimension (low)
  - "promotion" uses age dimension (pawn → queen)

The dimensions ARE the zinc fingers that bind layers together.
```

### Example: Chess Domain

```python
# Base corpus has:
base.add_pair("king", "queen", "gender")      # gender dim
base.add_pair("person", "king", "regality")   # regality dim

# Chess domain corpus has:
chess.add_pair("pawn", "queen", "promotion")  # NEW dimension
chess.add_pair("king", "pawn", "regality")    # SHARED dimension

# When chess is attached:
# - "king" in base and "king" in chess are DIFFERENT concepts
# - But they share the "regality" dimension
# - Query for "high regality" finds BOTH
```

## Query Resolution

When a query arrives, it traverses ALL attached layers:

```
Query: "What is a king?"

1. Parse query → extract "king"
2. Search BASE layer → find "king" (person, high regality)
3. Search DOMAIN layers → find "chess king" (piece, high regality)
4. Search CONTEXT layer → find any recent mentions
5. Combine results weighted by:
   - Layer priority (context > domain > base)
   - Geometric distance
   - Activation (promoters/enhancers from DNA model)
```

## Layer Interface

```python
class CorpusLayer:
    """A layer in the corpus stack."""
    
    name: str
    priority: int  # Higher = checked first
    corpus: SelfAssemblingCorpus
    
    # Dimension mapping to base layer
    dimension_map: Dict[str, int]  # local_name → base_index
    
    def attach_to(self, base: 'CorpusLayer'):
        """Hook this layer to a base layer via shared dimensions."""
        pass
    
    def detach(self):
        """Unhook this layer."""
        pass
    
    def query(self, position: np.ndarray) -> List[Tuple[str, float]]:
        """Find concepts near this position."""
        pass


class CorpusStack:
    """Stack of corpus layers with unified query interface."""
    
    base: CorpusLayer
    domains: List[CorpusLayer]
    context: CorpusLayer
    
    def attach_domain(self, domain: CorpusLayer):
        """Attach a domain layer."""
        domain.attach_to(self.base)
        self.domains.append(domain)
    
    def detach_domain(self, name: str):
        """Detach a domain layer by name."""
        pass
    
    def query(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Query all layers, return (concept, distance, layer_name).
        Results sorted by priority then distance.
        """
        pass
```

## Dimension Alignment

When attaching a domain layer, dimensions must align:

```python
def align_dimensions(domain: CorpusLayer, base: CorpusLayer):
    """
    Align domain dimensions to base dimensions.
    
    Cases:
    1. Domain uses existing base dimension → map to same index
    2. Domain introduces new dimension → extend base (or keep local)
    3. Domain dimension conflicts with base → resolve or reject
    """
    for dim_name in domain.corpus.dimensions:
        if dim_name in base.corpus.dimensions:
            # Case 1: Shared dimension - map to base index
            domain.dimension_map[dim_name] = base.corpus.dimensions[dim_name].index
        else:
            # Case 2: New dimension - keep local or extend base
            # Option A: Keep local (domain-specific)
            # Option B: Extend base (if dimension is universal)
            pass
```

## Geometric RAG

Traditional RAG:
```
Query → Embed → Search vectors → Retrieve TEXT → Inject into prompt
```

Geometric RAG:
```
Query → Parse concepts → Compute position → Search ALL LAYERS → 
Retrieve STRUCTURE → Traverse geometry → Generate response
```

The key difference: we retrieve **geometric structure**, not text chunks.

```python
class GeometricRAG:
    """RAG using geometric corpus layers instead of vector embeddings."""
    
    def __init__(self, stack: CorpusStack):
        self.stack = stack
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve relevant concepts from all layers.
        
        Unlike traditional RAG:
        - We retrieve CONCEPTS, not text chunks
        - We preserve GEOMETRIC RELATIONSHIPS
        - We can traverse the space, not just retrieve
        """
        # Parse query to position
        concepts = self._parse_concepts(query)
        position = self._compute_position(concepts)
        
        # Search all layers
        results = []
        for layer in self.stack.all_layers():
            nearby = layer.query(position)
            for concept, distance in nearby:
                results.append(RetrievalResult(
                    concept=concept,
                    distance=distance,
                    layer=layer.name,
                    position=layer.corpus.get_position(concept)
                ))
        
        # Sort by distance, respecting layer priority
        results.sort(key=lambda r: (r.layer_priority, r.distance))
        return results[:k]
    
    def traverse(self, start: str, dimension: str, steps: int = 1) -> List[str]:
        """
        Traverse the geometry from a starting concept.
        
        This is what traditional RAG CAN'T do:
        - "What's the opposite of X?"
        - "What's more Y than X?"
        - "What's between X and Z?"
        """
        pass
```

## Example: Medical Domain

```python
# Base corpus (language fundamentals)
base = SelfAssemblingCorpus()
base.add_pair("person", "patient", "role")
base.add_pair("person", "doctor", "role")
base.add_pair("healthy", "sick", "health")
base.add_pair("mild", "severe", "intensity")

# Medical domain corpus
medical = SelfAssemblingCorpus()
medical.add_pair("symptom", "fever", "manifestation")
medical.add_pair("symptom", "cough", "manifestation")
medical.add_pair("fever", "high_fever", "intensity")  # SHARED dimension!
medical.add_pair("diagnosis", "treatment", "medical_flow")
medical.add_pair("patient", "doctor", "medical_role")

# Create stack
stack = CorpusStack(base)
stack.attach_domain(CorpusLayer("medical", medical))

# Query traverses both
results = stack.query("What should a doctor do for severe fever?")
# Finds: doctor (base), severe (base), fever (medical), treatment (medical)
# Can traverse: fever → high_fever (intensity dimension)
```

## Session Context Layer

The context layer is ephemeral - it captures the current conversation:

```python
class ContextLayer(CorpusLayer):
    """Ephemeral layer for current conversation context."""
    
    def __init__(self, base: CorpusLayer):
        super().__init__("context", priority=100)  # Highest priority
        self.corpus = SelfAssemblingCorpus()
        self.attach_to(base)
    
    def add_mention(self, concept: str, position: np.ndarray):
        """Track a concept mentioned in conversation."""
        # Add to context with recency boost
        pass
    
    def decay(self, factor: float = 0.9):
        """Decay old mentions (recency weighting)."""
        pass
    
    def clear(self):
        """Clear context (new conversation)."""
        self.corpus = SelfAssemblingCorpus()
```

## Benefits

1. **Modularity** - Base corpus is reusable across applications
2. **No Saturation** - Domain knowledge stays in domain layer
3. **Forgetting** - Detach domain layer to "forget" a topic
4. **Geometric RAG** - Retrieve structure, not just text
5. **Traversal** - Can navigate the space, not just retrieve
6. **Consistency** - Shared dimensions ensure geometric consistency

## Implementation Plan

1. **CorpusLayer class** - Wrapper around SelfAssemblingCorpus with dimension mapping
2. **CorpusStack class** - Manages layer stack with unified query
3. **Dimension alignment** - Algorithm to align domain dimensions to base
4. **GeometricRAG class** - RAG-like retrieval using geometry
5. **ContextLayer class** - Ephemeral conversation context
6. **Demo** - Base + Chess domain + conversation

## Connection to DNA Mechanics

From design doc 077:

| DNA Concept | Corpus Layer Equivalent |
|-------------|------------------------|
| Zinc Fingers | Shared dimensions (hooks) |
| Major Groove | Semantic query (text matching) |
| Minor Groove | Geometric query (position matching) |
| Promoters | Query patterns that activate concepts |
| Enhancers | Context that boosts domain relevance |
| Silencers | Context that suppresses irrelevant domains |

The domain layer attachment is like **gene expression** - the base genome (base corpus) is always there, but which genes are expressed (which domains are attached) determines the phenotype (conversation capability).

---

*"The base corpus is the genome. Domain corpora are gene expression. Context is the current cellular state."*
