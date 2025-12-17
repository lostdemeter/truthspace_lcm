# Experiment Results Summary

## Overview

These experiments tested hierarchical knowledge navigation for domain selection.
The goal: geometrically determine which knowledge domain (BASH, SOCIAL, CREATIVE, etc.) to activate based on input.

## Key Findings

### Experiment 1: Centroid Definition

**Result**: Examples-based centroids (60%) > Hybrid (55%) > Keywords-only (50%)

**Critical Issue**: The INFORMATIONAL domain dominated because its keywords ("what", "how", "why", "describe") are too common. This reveals that **domain centroids need discriminative keywords, not just descriptive ones**.

**Insight**: Domain centroids should be defined by what makes them *unique*, not what they contain.

---

### Experiment 2: Cross-Domain Discrimination

**Result**: 34.4% accuracy (11/32) - Very poor!

**Critical Issue**: CREATIVE domain won almost everything because:
1. "create" is in CREATIVE keywords
2. CREATIVE centroid is too generic

**Key Insight**: The current primitive set doesn't capture domain-level distinctions. We need **domain-specific primitives** that don't overlap with action primitives.

For example:
- "create a file" should activate BASH because of "file"
- "create a poem" should activate CREATIVE because of "poem"

But currently, "create" dominates and routes to CREATIVE.

**Solution Direction**: Domain primitives should be at a *different level* than action primitives - they shouldn't compete directly.

---

### Experiment 3: Hierarchy Depth

**Result**: Flat (83%) > Shallow (75%) > Deep (58%)

**Surprising Finding**: Deeper hierarchies performed *worse*!

**Why**: Each navigation step is a chance to make a wrong turn. With current centroid quality, more steps = more errors.

**Insight**: Hierarchy depth should match centroid discrimination quality. Don't add depth until centroids are reliable.

---

### Experiment 4: Navigation Strategy

**Result**: Hard (50%) = Beam (50%) > Soft-0.5 (40%) > Soft-1.0 (20%)

**Interesting**: Soft navigation with temperature=0.5 was the *only* strategy that correctly handled "I'm feeling out of touch" → SOCIAL.

**But**: Soft navigation also made more mistakes on clear cases.

**Insight**: Soft navigation helps with ambiguity but hurts with clear cases. A hybrid approach might work: use hard navigation when confidence is high, soft when ambiguous.

---

### Experiment 5: Contextual Priming

**Result**: No priming (70%) > All priming strategies (65%)

**Surprising**: Priming actually *hurt* performance!

**Why**: Priming biased toward the previous domain, which hurt context switches. When switching from SOCIAL to BASH, the SOCIAL prior made "list files" route incorrectly.

**Insight**: Priming is a double-edged sword. It helps maintain context but hurts context switches. Need a mechanism to detect when context is switching.

---

## Root Cause Analysis

The fundamental issue across all experiments: **domain centroids are not discriminative enough**.

Current approach:
```
BASH centroid = encode("command file directory process system")
CREATIVE centroid = encode("write poem story compose imagine")
```

Problem: These overlap too much in truth space because:
1. Action primitives (CREATE, WRITE) appear in multiple domains
2. No primitives specifically encode "this is a technical context" vs "this is a creative context"

---

## Proposed Solutions

### Solution 1: Domain-Level Primitives at Higher φ Levels

Add primitives that encode domain context, not actions:

```python
# At φ^4 level (dimension 12+)
Primitive("TECHNICAL_CONTEXT", 12, 0, ["terminal", "shell", "command", "execute", "system"])
Primitive("EMOTIONAL_CONTEXT", 12, 1, ["feeling", "emotion", "mood", "sense", "experience"])
Primitive("CREATIVE_CONTEXT", 12, 2, ["artistic", "imaginative", "poetic", "narrative", "aesthetic"])
Primitive("INQUIRY_CONTEXT", 12, 3, ["question", "curious", "wondering", "asking"])
```

These would activate at a higher level than action primitives, providing domain signal.

### Solution 2: Negative Discrimination

Define what a domain is NOT:

```python
BASH_NOT = ["feeling", "emotion", "poem", "story", "imagine"]
SOCIAL_NOT = ["file", "directory", "command", "process"]
```

Penalize centroids when query contains "NOT" keywords.

### Solution 3: Two-Stage Resolution

1. **Stage 1**: Detect domain using domain-level primitives only
2. **Stage 2**: Within domain, use action primitives for concept resolution

This separates the concerns and prevents action primitives from interfering with domain detection.

### Solution 4: Structural Signatures

Detect sentence structure:
- Imperative mood (verb first) → likely BASH
- First-person ("I feel", "I think") → likely SOCIAL
- Question words ("what", "how") → likely INFORMATIONAL

Add structural primitives:
```python
Primitive("IMPERATIVE", 13, 0, [])  # Detected by structure, not keywords
Primitive("FIRST_PERSON", 13, 1, ["i", "me", "my", "myself"])
Primitive("INTERROGATIVE", 13, 2, ["what", "how", "why", "when", "where"])
```

---

## Next Steps

1. **Implement domain-level primitives** at higher φ levels
2. **Test two-stage resolution** (domain first, then concept)
3. **Add structural detection** for sentence patterns
4. **Re-run experiments** with improved architecture

---

## Philosophical Reflection

These experiments reveal that **the current primitive set was designed for bash command resolution, not general knowledge navigation**.

To scale to multiple domains, we need:
1. Primitives that encode *context*, not just *actions*
2. Hierarchical primitive levels (domain > category > action)
3. Structural analysis beyond keyword matching

The good news: the geometric framework is sound. The issue is primitive design, not the navigation algorithm.
