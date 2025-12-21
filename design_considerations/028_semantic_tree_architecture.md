# Design Consideration 028: Semantic Tree Architecture

## The Problem

We need a knowledge encoding system that:
1. Works with small amounts of data (bootstrap)
2. Gets **more useful** as data grows (scaling)
3. Handles queries using different words than stored facts (bridging)
4. Self-organizes without manual structure design (emergence)

Previous attempts revealed:
- **Pure dynamics** from random positions: ~15% accuracy, doesn't converge
- **Pareto bootstrap alone**: 52-76% accuracy, needs domain-specific clusters
- **Aggressive error correction**: Destabilizes with diverse data
- **Recursive tree alone**: Self-organizes but fails on vocabulary mismatch

## The Solution: Two Complementary Layers

### Layer 1: Semantic Clusters (Vocabulary Bridging)

Clusters group words with similar meanings:

```python
SEMANTIC_CLUSTERS = {
    'DEATH': ['die', 'died', 'death', 'assassinated', 'killed', 'passed'],
    'BIRTH': ['born', 'birth', 'birthday', 'birthplace'],
    'CAPITAL': ['capital', 'city', 'largest', 'main'],
    'COOK': ['cook', 'cooking', 'recipe', 'prepare'],
    'LIST_CMD': ['ls', 'list', 'files', 'directory'],
    ...
}
```

This answers: **"We know where new information goes"**

When a query says "How did Lincoln die" and the fact says "Lincoln was assassinated", the DEATH cluster bridges `die ↔ assassinated`.

### Layer 2: Recursive Tree (Self-Organizing Structure)

The tree grows organically as data arrives:

```
ROOT
  BIRTH_WASHINGTON {BIRTH,WASHINGTON}
    LEAD_WASHINGTON {LEAD,WASHINGTON}
      DEATH_WASHINGTON {DEATH,WASHINGTON}
    BIRTH_LINCOLN {BIRTH,LINCOLN}
  CAPITAL_FRANCE {CAPITAL,FRANCE}
    CAPITAL_ENGLAND {CAPITAL,ENGLAND}
      CAPITAL_JAPAN {CAPITAL,JAPAN}
```

This answers: **"We organize as we receive"**

New facts either:
- **Fit existing structure** (most common, ~80%)
- **Create new branches** (rare, high-info events, ~20%)

This is the Pareto principle at every level of the hierarchy.

## The Tree Analogy

The architecture mirrors how trees actually grow:

| Tree Component | Encoder Component | Function |
|----------------|-------------------|----------|
| DNA/Seed | Semantic Clusters | Initial structure, vocabulary bridges |
| Cambium (living) | New data ingestion | Where growth happens |
| Heartwood (dead) | Crystallized structure | Frozen, stable, fast lookup |
| Growth rings | Snapshots | History of crystallizations |
| Branches | Tree nodes | Self-similar at every scale |

## Results

### Test: 26 facts across 5 domains

| Category | Accuracy |
|----------|----------|
| Geography | **100%** |
| Linux | **100%** |
| Cooking | 80% |
| Science | 75% |
| History | 67% |
| **Overall** | **85%** |

### Why It Works

1. **Semantic clusters** provide vocabulary bridges
   - Query: "How did Lincoln die"
   - Stored: "Lincoln was assassinated"
   - Bridge: `die → DEATH → assassinated` ✓

2. **Recursive tree** self-organizes by topic
   - Capitals cluster together
   - Presidents cluster together
   - Commands cluster together

3. **Combined similarity** uses both:
   ```python
   score = 0.6 * cluster_similarity + 0.4 * position_similarity
   ```

## Scaling Behavior

As data grows, **both layers improve**:

| Data Size | Semantic Layer | Tree Layer |
|-----------|----------------|------------|
| Small (~50 facts) | Clusters provide bridges | Tree is shallow (depth 2-3) |
| Medium (~500 facts) | Co-occurrence adds bridges | Tree deepens (depth 4-5) |
| Large (~5000+ facts) | Bridges are rich | Tree is deep, fine-grained |

The structure gets more useful because:
- More data → deeper tree → finer organization
- More data → richer co-occurrence → semantic bridges emerge naturally
- More queries → access counts reveal Pareto distribution → optimize hot paths

## Implementation

### Key Files

- `experiments/semantic_tree.py` - Full implementation
- `experiments/recursive_pareto.py` - Tree-only version
- `experiments/pareto_bootstrap_demo.py` - Cluster-only version

### Core Algorithm

```python
def store(text, fact_id):
    # 1. Extract semantic clusters from text
    clusters = get_clusters(text)  # {DEATH, LINCOLN}
    
    # 2. Encode position from word vectors
    position = encode_text(text)
    
    # 3. Find path through tree using combined similarity
    path = find_path(text)  # ROOT → LINCOLN → ...
    
    # 4. Either use existing branch or create new one
    if best_similarity < threshold:
        create_new_branch(clusters, position)
    else:
        merge_into_existing(clusters, position)
    
    # 5. Store fact at leaf node
    current_node.facts.append((text, fact_id))

def query(text):
    # 1. Extract clusters and position
    clusters = get_clusters(text)
    position = encode_text(text)
    
    # 2. Navigate tree
    path = find_path(text)
    
    # 3. Search path nodes and children for best match
    for node in path + children:
        for fact in node.facts:
            score = combined_similarity(query, fact)
            if score > best:
                best_match = fact
    
    return best_match
```

## Path to 100% Accuracy

Current failures fall into categories:

### 1. Missing Cluster Bridges (fixable)
- "Darwin evolution" → fails because DARWIN cluster doesn't include "evolution"
- Fix: Add missing words to clusters

### 2. Ambiguous Queries (harder)
- "What about Japan" → matches wrong domain
- Fix: Context tracking, conversation state

### 3. Sparse Clusters (needs more data)
- "How to roast vegetables" → ROAST cluster too small
- Fix: More cooking data, or expand cluster

### Proposed Improvements

1. **Automatic cluster expansion** from co-occurrence:
   ```python
   if cooccurrence[word1][word2] > threshold:
       merge_into_same_cluster(word1, word2)
   ```

2. **Error-driven cluster refinement**:
   ```python
   if query_failed:
       # Add query words to correct fact's clusters
       for word in query_words:
           correct_fact_clusters.add(word)
   ```

3. **Hierarchical clusters** (clusters of clusters):
   ```
   LIFE_EVENTS
     BIRTH: [born, birth, ...]
     DEATH: [died, death, ...]
   COOKING
     BOIL: [boil, pasta, ...]
     BAKE: [bake, bread, ...]
   ```

4. **Confidence thresholds**:
   ```python
   if best_score < confidence_threshold:
       return "I'm not sure about that"
   ```

## Connection to Previous Work

This builds on several earlier discoveries:

- **DC 012**: φ-MAX encoding prevents synonym stacking
- **DC 022**: Attractor/repeller dynamics for self-organization
- **DC 025**: Co-occurrence cluster matching
- **DC 027**: Pareto bootstrap for initial structure

The Semantic Tree is the synthesis: **bootstrap structure + emergent organization**.

## Conclusion

The Semantic Tree architecture solves the scaling problem:

> **Start with semantic clusters (vocabulary bridges), let the tree grow organically (self-organization), and both layers improve as data accumulates.**

Current: 85% accuracy with 26 facts
Target: 100% accuracy through cluster refinement and error-driven learning

The key insight: your two intuitions are complementary, not competing:
1. "We know where new info goes" → Semantic clusters
2. "We organize as we receive" → Recursive tree

Both are needed. Both improve with scale.
