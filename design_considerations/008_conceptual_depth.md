# Design Consideration 008: Conceptual Depth in the LCM

## The Question

The LCM (Large Concept Model) is supposed to work because **concepts overlap** in meaning. When we say "download file", the concept of "download" should already contain information about network, remote source, etc. that distinguishes it from "copy file".

**Are our primitives capturing this conceptual overlap, or are they too shallow?**

## Analysis

### Current State: Compositional Semantics

Our current encoding is **compositional**:
```
meaning("download file") = meaning("download") + meaning("file")
                         = CONNECT(dim 3) + FILE(dim 4)
```

This treats concepts as independent building blocks that sum together.

### What LCM Should Be: Conceptual Semantics

A true LCM should have **conceptual overlap**:
```
meaning("download") = {
    primary: CONNECT(dim 3),
    implies: NETWORK(dim 6),  // download implies network context
    excludes: COPY(dim 2),    // download is not local copy
    expects: {
        object: FILE or URL,
        source: REMOTE,
    }
}
```

The concept of "download" should **already contain** the network context, not just be associated with it.

## Experiment: Primitive Implications

We implemented a primitive implications system:

```python
Primitive("CONNECT", PrimitiveType.ACTION,
    keywords={"connect", "ssh", "curl", "download", ...},
    implies={"NETWORK"},   # Connect implies network context
    excludes={"COPY"},     # Connect excludes local copy
)
```

### Results

| Metric | Before Implications | After Implications |
|--------|--------------------|--------------------|
| Success Rate | 83.3% | 83.3% |
| "download file" vs "copy file" similarity | 0.4260 | 0.3966 |

The implications **did help** separate concepts (lower similarity = better), but didn't improve the overall success rate.

### Why Implications Aren't Enough

The problem is that implications work at **encoding time**, but the stored knowledge was also encoded with implications. This means:

1. Query "download file" gets NETWORK implied
2. Intent "download_file" also got NETWORK implied when stored
3. But so did "scp" and "copy_from_remote" (they have CONNECT keywords)

The implications don't create **unique** signatures - they create **similar** signatures for related concepts.

## The Deeper Issue

### What's Missing: Semantic Roles

Consider these two phrases:
- "download file from URL" - file is the OBJECT, URL is the SOURCE
- "copy file to backup" - file is the SOURCE, backup is the DESTINATION

Our encoding treats both as:
```
[ACTION] + [FILE] + [other stuff]
```

It doesn't capture that "file" plays **different roles** in each phrase.

### What's Missing: Verb-Argument Structure

Different verbs expect different argument types:
- "download X" expects X to be a remote resource
- "copy X" expects X to be a local resource
- "ssh into X" expects X to be a server
- "scp X to Y" expects X to be a file and Y to be a remote path

Our encoding doesn't capture these expectations.

### What's Missing: Contextual Priming

The presence of certain patterns should **prime** the interpretation:
- "http://" should prime toward CONNECT/NETWORK
- ".txt" should prime toward FILE
- "server.com" should prime toward NETWORK
- "/tmp" should prime toward FILE/LOCAL

We added some of this (http, server in keywords), but it's not systematic.

## The Fundamental Limitation

The LCM concept assumes that **concepts naturally overlap** in ways that enable disambiguation. But our primitives are:

1. **Orthogonal by design** - dimensions are independent
2. **Flat** - no hierarchical structure
3. **Context-free** - same encoding regardless of surrounding words

This is the opposite of how human concepts work, where:
1. Concepts **share features** (download and network share "remote")
2. Concepts are **hierarchical** (download is-a transfer is-a action)
3. Concepts are **context-sensitive** (file means different things in different contexts)

## Proposed Solutions

### Option 1: Non-Orthogonal Primitives

Instead of orthogonal dimensions, allow primitives to **share** dimensional space:

```python
DOWNLOAD = [0, 0, 0, 0.8, 0, 0, 0.5, 0, 0, 0, 0, 0]  # dim 3 + dim 6
COPY     = [0, 0, 0.8, 0, 0.3, 0, 0, 0, 0, 0, 0, 0]  # dim 2 + dim 4
```

This creates natural overlap but loses the orthogonality benefits.

### Option 2: Hierarchical Encoding

Encode concepts at multiple levels:
```
Level 0: ACTION (most general)
Level 1: TRANSFER (more specific)
Level 2: DOWNLOAD (most specific)
```

Each level contributes to the position, creating richer signatures.

### Option 3: Frame-Based Encoding

Use semantic frames that capture argument structure:
```python
Frame("DOWNLOAD",
    action=CONNECT,
    object=FILE,
    source=URL,
    destination=LOCAL,
)
```

The frame encodes not just what concepts are present, but their relationships.

### Option 4: Contextual Encoding

Different encodings based on context:
```python
encode("file", context="download") → FILE + REMOTE_OBJECT
encode("file", context="copy")     → FILE + LOCAL_OBJECT
```

This requires understanding context before encoding.

## Resolution (December 2024)

**UPDATE**: We achieved 100% success rate WITHOUT needing "deep conceptual semantics."

The breakthrough was realizing that the 12D encoding already contained sufficient information - we just needed to **weight dimensions appropriately** when comparing vectors.

### The Real Issue

It wasn't shallow vs deep semantics. It was that cosine similarity treats all dimensions equally, when ACTION dimensions (0-3) are more discriminative than DOMAIN dimensions (4-7) for command disambiguation.

### The Fix

```python
dim_weights = [3.0, 3.0, 3.0, 3.0,  # Actions: 3x weight
               1.0, 1.0, 1.0, 1.0,  # Domains: 1x weight
               0.3, 0.3, 0.3, 0.3]  # Relations: 0.3x weight
```

This is a form of **dimensional downcasting** - projecting high-dimensional information through a weighted lens that emphasizes discriminative features.

See `009_projection_weighting.md` for the mathematical framework.

## Original Conclusion (Superseded)

~~The 83.3% success rate represents the ceiling for shallow compositional semantics.~~

**Corrected**: The ceiling was an artifact of equal-weight projection, not a fundamental limit of the encoding.

## Next Steps

1. **Accept 83% baseline** for keyword-based encoding
2. **Explore frame-based encoding** for richer structure
3. **Consider hybrid approach** - use LLM for context understanding, LCM for knowledge retrieval
4. **Research non-orthogonal encodings** that allow natural concept overlap
