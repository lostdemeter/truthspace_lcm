# Design Consideration 007: Semantic Disambiguation in TruthSpace LCM

## The Problem

After migrating to 12D plastic-primary encoding, we achieved 86.7% success on bash knowledge tests. The remaining 4 failures reveal a fundamental question:

**Does TruthSpace LCM lack the semantic understanding that traditional LLMs possess?**

### The Failing Cases

```
Query: "move file.txt to /tmp"      → matched copy_recursive (not move_file)
Query: "follow the log file"        → matched count_words (not follow_log)
Query: "download file from http://..."  → matched find_python_files (not download_file)
Query: "ssh into server.com"        → matched copy_from_remote (not remote_login)
```

## Analysis: What's Actually Happening

### In TruthSpace LCM

Our system encodes meaning through **keyword co-occurrence in geometric space**:

```
"move file.txt to /tmp"
    ↓
Keywords: [move, file, txt, to, tmp]
    ↓
Encode each keyword → sum weighted positions
    ↓
Find nearest neighbor in knowledge base
```

The problem: "move file.txt to /tmp" and "copy directory to backup" share keywords like "to", "file", and action-related terms. The 12D encoding places them nearby because they share surface-level features.

### In a Traditional LLM

An LLM encodes meaning through **contextual co-occurrence learned from billions of examples**:

```
"move file.txt to /tmp"
    ↓
Transformer attention over all tokens
    ↓
Contextual embedding that "knows":
  - "move" in file context → mv command
  - "to /tmp" → destination path
  - "file.txt" → source file
    ↓
Generate: "mv file.txt /tmp"
```

The LLM has seen millions of examples where "move X to Y" in a file context means `mv`, not `cp`. It has learned the **distributional semantics** of these patterns.

## The Fundamental Difference

| Aspect | TruthSpace LCM | Traditional LLM |
|--------|----------------|-----------------|
| **Learning** | Explicit keyword assignment | Implicit from data distribution |
| **Context** | Bag-of-keywords (position-independent) | Full sequence context |
| **Disambiguation** | Keyword uniqueness | Attention patterns |
| **Training** | Manual knowledge curation | Gradient descent on loss |
| **Generalization** | Limited to keyword overlap | Interpolates in embedding space |

### The Core Issue: Compositional vs Distributional Semantics

**TruthSpace uses compositional semantics:**
- Meaning = sum of primitive meanings
- "move file to X" = MOVE + FILE + destination
- "copy file to X" = COPY + FILE + destination
- Problem: MOVE and COPY are both in the SPATIAL dimension (dim 2)

**LLMs use distributional semantics:**
- Meaning = where the phrase appears in context
- "move" and "copy" have different distributions in training data
- The model learns that "move" implies deletion of source

## Why Our Failures Occur

### Case 1: "move file.txt to /tmp" → copy_recursive

```
move_file keywords:  [move, mv, rename, relocate]
copy_recursive:      [copy, duplicate, clone, cp, recursive, directory, folder]
```

The query "move file.txt to /tmp" contains:
- "move" → matches move_file
- "file" → matches many things
- "to" → common word, matches copy_to_remote, move_file
- "/tmp" → matches nothing specific

The 12D encoding sums these and finds copy_recursive closer because "file" and "to" create overlap with copy-related intents.

### Case 2: "ssh into server.com" → copy_from_remote

```
remote_login:     [ssh, login, into, shell, terminal, connect]
copy_from_remote: [scp, download, remote, fetch, from, server]
```

The query contains:
- "ssh" → matches remote_login
- "into" → matches remote_login
- "server" → matches copy_from_remote
- "com" → matches nothing

The word "server" pulls the encoding toward copy_from_remote because it's a strong keyword there.

## What LLMs Have That We Don't

### 1. Contextual Word Sense Disambiguation

LLMs learn that "file" in "move file to" means a filesystem object, while "file" in "file a report" means something different. Our system treats "file" the same everywhere.

### 2. Syntactic Structure Awareness

LLMs understand that "move X to Y" has a different structure than "copy X to Y" even though they share words. We only see keyword overlap.

### 3. Implicit World Knowledge

LLMs know that `mv` deletes the source while `cp` doesn't. This knowledge influences their embeddings. We have no such implicit knowledge.

### 4. Negative Examples

LLMs learn from what words DON'T appear together. We only encode positive associations.

## Potential Solutions

### 1. N-gram Keywords (Compositional Phrases)

Instead of single keywords, use phrases:
```json
{
  "name": "move_file",
  "keywords": ["move file", "mv", "move to", "rename"]
}
```

This captures some syntactic structure without full parsing.

### 2. Negative Keywords (Anti-associations)

Add keywords that should DECREASE similarity:
```json
{
  "name": "move_file",
  "keywords": ["move", "mv", "rename"],
  "anti_keywords": ["copy", "duplicate", "recursive"]
}
```

### 3. Context-Dependent Encoding

Encode the RELATIONSHIP between keywords, not just their presence:
```
"move file to /tmp"
  → (MOVE, FILE, TO) as a triple
  → Different from (COPY, FILE, TO)
```

### 4. Hierarchical Disambiguation

First classify the ACTION type, then resolve within that category:
```
Query → ACTION classifier → MOVE detected
                         → Search only MOVE-related intents
```

### 5. Trigger Phrase Matching (Current Partial Solution)

We already have "triggers" in our intents:
```json
"triggers": ["move file", "move to", "mv file"]
```

These could be used for exact/fuzzy matching BEFORE geometric search.

## The Deeper Question

**Can geometric encoding ever match distributional semantics?**

The answer may be: **not without additional structure**.

Our 12D encoding captures:
- Action type (dims 0-3)
- Domain (dims 4-7)
- Relations (dims 8-11)

But it doesn't capture:
- Word order
- Syntactic roles (subject, object, destination)
- Contextual word sense
- Implicit world knowledge

### A Hybrid Approach

Perhaps the solution is not to replace LLM-style understanding, but to use TruthSpace as a **structured knowledge layer** on top of it:

```
User Query
    ↓
LLM extracts: {action: "move", source: "file.txt", dest: "/tmp"}
    ↓
TruthSpace resolves: action="move" + domain="file" → mv command
    ↓
Template: "mv {source} {dest}"
    ↓
Output: "mv file.txt /tmp"
```

This separates:
- **Understanding** (LLM's strength) from
- **Knowledge retrieval** (TruthSpace's strength)

## Experimental Analysis (December 2024)

Deep analysis of the 4 failing cases reveals specific root causes:

### Finding 1: Primitive Keyword Overlap

The MOVE primitive includes both "move" AND "copy":
```
MOVE (dim 2): {move, copy, mv, cp, transfer, backup, duplicate, clone}
```

This means "copy file" and "move file" encode **identically**! The encoder cannot distinguish them.

### Finding 2: Misclassified Keywords

| Keyword | Expected | Actual | Why |
|---------|----------|--------|-----|
| `log` | FILE/DATA | USER | Matches "login" in USER primitive |
| `tail` | ACTION | VERBOSE | Matches verbose modifier |
| `to` | SPATIAL | CREATE | Matches creation keywords |
| `scp` | CONNECT | MOVE | Matches copy/move keywords |
| `follow` | ACTION | (none) | No primitive match at all |

### Finding 3: Query-Intent Dimension Mismatch

```
Query "ssh into server.com":
   dim 3 (INTERACTION): +0.752  ← "ssh" → CONNECT
   dim 6 (NETWORK_USER): +0.000  ← "server.com" has no USER signal

Expected "remote_login":
   dim 3 (INTERACTION): +0.591
   dim 6 (NETWORK_USER): +0.783  ← "remote", "login" → USER
```

The query lacks dim 6 signal because "server.com" doesn't encode to USER!

### Finding 4: Dimension Collision

Multiple intents occupy the same region:
- "move file to /tmp" → dim 2 + dim 4
- "du" (disk usage) → dim 2 + dim 4
- "copy directory" → dim 2 + dim 4

### Root Cause Summary

1. **Primitive keywords are too broad** - "copy" and "move" share a primitive
2. **Missing discriminative primitives** - "follow", "log", "tail" lack specific matches
3. **Stop words add noise** - "to", "the", "from" encode to wrong primitives
4. **Domain keywords missing** - "server.com", "http://" don't trigger NETWORK

## Attempted Fixes (December 2024)

### Fix 1: Split MOVE/COPY Primitives ✓
```python
# Before: MOVE included both move AND copy
MOVE: {move, copy, mv, cp, transfer, backup, clone, duplicate}

# After: Separate primitives
MOVE: {move, mv, rename, relocate}
COPY: {copy, cp, duplicate, clone, backup}
```
**Result**: Helped distinguish move vs copy operations.

### Fix 2: Add FOLLOW Primitive ✓
```python
FOLLOW: {follow, tail, watch, monitor, stream, live, realtime}
```
**Result**: "follow the log file" now correctly matches `follow_log` intent.

### Fix 3: Add Stop Word Filtering ✓
```python
STOP_WORDS = {"the", "a", "an", "to", "from", "in", "on", ...}
```
**Result**: Reduced noise from common words.

### Fix 4: Expand CONNECT with URL Keywords ✓
```python
CONNECT: {connect, link, ssh, curl, fetch, download, upload, sync, http, https, url, remote}
```
**Result**: Better matching for download/network operations.

### Fix 5: Add "server" to NETWORK Primitive ✓
```python
NETWORK: {network, interface, ip, port, socket, connection, host, server, client}
```
**Result**: "ssh into server.com" now triggers NETWORK dimension.

### Remaining Failures (5 cases)

Despite improvements, 5 cases remain problematic:

| Query | Expected | Actual | Root Cause |
|-------|----------|--------|------------|
| "move file.txt to /tmp" | mv | cp | "file" pulls toward copy_recursive |
| "search for error in log files" | grep | find | "files" dominates over "error" |
| "count lines in file.txt" | wc -l | pwd | "count" has no primitive match |
| "download file from http://..." | curl | scp | "file" pulls toward scp |
| "ssh into server.com" | ssh | scp | "server" pulls toward scp |

### Key Insight: The Tuning Dilemma

Fixing one failure often causes regressions elsewhere:
- Adding "lines" to DATA primitive → broke head/tail queries
- Adding "server" to NETWORK → helped ssh but not enough

This reveals a **fundamental limitation**: keyword-based encoding cannot capture the nuanced relationships that distinguish similar operations.

## Resolution (December 2024)

**UPDATE**: The disambiguation problem was SOLVED by implementing dimension-weighted scoring.

The issue wasn't a fundamental limitation - it was that our scoring treated all dimensions equally when ACTION dimensions should be weighted more heavily.

### The Fix

```python
dim_weights = [3.0, 3.0, 3.0, 3.0,  # Actions: highest weight
               1.0, 1.0, 1.0, 1.0,  # Domains: normal weight  
               0.3, 0.3, 0.3, 0.3]  # Relations: low weight
```

### Results

| Stage | Success Rate |
|-------|-------------|
| Initial 12D | 60% |
| After keyword tuning | 83.3% |
| After dimension weighting | **100%** |

See `009_projection_weighting.md` for the full mathematical analysis.

## Original Conclusion (Superseded)

~~The 86.7% success rate isn't a failure of the 12D encoding—it's a fundamental limitation of keyword-based compositional semantics.~~

**Corrected**: The 12D encoding contains all necessary information. The issue was the **projection method** (how we compare vectors), not the encoding itself.

TruthSpace LCM's approach is different by design:
- **Explicit** rather than implicit knowledge
- **Interpretable** rather than black-box
- **Deterministic** rather than probabilistic
- **Minimal** rather than massive

The remaining 13.3% of failures represent cases where **surface-level keyword overlap** creates ambiguity that only **contextual understanding** can resolve.

### Recommendations

1. **Accept the limitation** for pure TruthSpace queries
2. **Implement trigger phrase matching** as a pre-filter
3. **Consider n-gram keywords** for common ambiguous patterns
4. **Explore hybrid architectures** where LLMs handle disambiguation and TruthSpace handles knowledge retrieval

The question isn't whether TruthSpace can match LLM understanding—it's whether the tradeoffs (interpretability, minimalism, determinism) are worth the disambiguation limitations for specific use cases.
