# Design Consideration 013: Compound Phrase Resolution

## Overview

This document describes the geometric approach to resolving compound natural language queries into multiple executable commands with extracted parameters. The system uses sliding window encoding for concept extraction and semantic void detection for parameter identification.

## The Problem

Users naturally express compound requests:

```
"create a directory called test and put a file in it called test.txt"
```

This single query contains:
- **Two concepts**: create directory, create file
- **Two parameters**: "test", "test.txt"
- **Structural words**: "and", "called", "in it"

The challenge: How do we extract multiple concepts and their parameters **geometrically**?

## Solution Architecture

### Two-Phase Resolution

```
Query → [Phase 1: Geometric Concept Extraction] → [Phase 2: Parameter Detection] → Commands
              ↓                                           ↓
        Sliding window encoding                    Semantic void analysis
        Action-boosted scoring                     Context-based detection
        Non-overlapping selection                  Position assignment
```

**Phase 1** identifies WHAT to do (the concepts/commands).
**Phase 2** identifies WITH WHAT (the parameters/values).

Both phases use geometric principles.

---

## Phase 1: Sliding Window Concept Extraction

### The Insight

A compound query contains multiple **concept regions** - contiguous word sequences that encode to positions near known knowledge entries.

```
"create directory test and create file readme.txt"
     └──────────┘              └─────────┘
      Concept 1                 Concept 2
```

### Algorithm

1. **Slide windows** of sizes [2, 3, 4] across the tokenized query
2. **Encode each window** using φ-MAX encoding
3. **Find best match** for each window against knowledge entries
4. **Score candidates** with action-primitive boosting
5. **Select greedily** non-overlapping windows by score

### Implementation

```python
def resolve_compound(self, text: str, window_sizes=[2, 3, 4], threshold=0.65):
    words = self._tokenize(text)
    candidates = []
    
    for window_size in window_sizes:
        for i in range(len(words) - window_size + 1):
            window = ' '.join(words[i:i+window_size])
            pos = self._encode(window)
            
            # Find best matching knowledge entry
            best_entry, best_sim = self._find_best_match(pos)
            
            if best_sim >= threshold:
                candidates.append({
                    'start': i,
                    'end': i + window_size,
                    'window': window,
                    'command': best_entry.output,
                    'similarity': best_sim
                })
    
    # Score with action boosting
    for c in candidates:
        window_pos = self._encode(c['window'])
        has_action = any(window_pos[d] > 0.5 for d in ACTION_DIMS)
        c['score'] = c['similarity'] * (1.5 if has_action else 1.0)
    
    # Greedy non-overlapping selection
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected = []
    covered = set()
    
    for c in candidates:
        positions = set(range(c['start'], c['end']))
        if not positions & covered:
            selected.append(c)
            covered.update(positions)
    
    return sorted(selected, key=lambda x: x['start'])
```

### Action Boosting

Commands are action-oriented. Windows containing action primitives (CREATE, READ, DELETE, COPY, etc.) are more likely to be valid concepts than windows with only domain primitives.

```python
ACTION_DIMS = {0, 1, 2, 3}  # CREATE, READ/LIST/WRITE/DELETE, COPY/RELOCATE/SEARCH, EXECUTE/etc

# Boost score by 1.5x for windows with action primitives
score = similarity * (1.5 if has_action else 1.0)
```

This prevents spurious matches like "directory called" (which only activates DIRECTORY) from outscoring "create directory" (which activates CREATE + DIRECTORY).

### Threshold Selection

The threshold (0.65) balances:
- **Too low** (0.5): Spurious matches like "processes and" → ping
- **Too high** (0.8): Valid queries like "create a directory called X" fail

Empirically, 0.65 provides good discrimination while catching natural language variations.

---

## Phase 2: Geometric Parameter Detection

### The Key Insight

Parameters are **semantic voids** - words that don't activate any primitives in truth space.

```
CONCEPT: "directory" → activates DIRECTORY primitive (dim 5)
PARAMETER: "myproject" → activates nothing (semantic void)
```

This is a geometric property: concepts have positions, parameters don't.

### Detection Rules

1. **After NAMING keywords** (`called`, `named`, `as`)
   - The next word is definitely a parameter
   - "create directory **called** `test`" → `test` is a parameter

2. **After DOMAIN primitives** (FILE, DIRECTORY, PROCESS, etc.)
   - If the next word is a semantic void, it's likely a parameter
   - "create **directory** `myproject`" → `myproject` is a parameter

### Implementation

```python
def detect_parameters_geometric(self, text: str):
    words = self._tokenize(text)
    parameters = []
    naming_keywords = {'called', 'named', 'as'}
    
    for i, word in enumerate(words):
        # Rule 1: After NAMING keyword
        if word in naming_keywords and i + 1 < len(words):
            parameters.append((i + 1, words[i + 1], "after_naming"))
            continue
        
        # Rule 2: Semantic void after domain word
        word_pos = self._encode(word)
        word_activates = np.any(word_pos > 0.01)
        
        if i > 0 and not word_activates:
            prev_pos = self._encode(words[i - 1])
            prev_activates_domain = any(prev_pos[d] > 0.01 for d in DOMAIN_DIMS)
            
            if prev_activates_domain:
                parameters.append((i, word, "after_domain"))
    
    return parameters
```

### Structural Primitives

To support geometric detection, we added structural primitives:

```python
Primitive("NAMING", 11, 1, ["called", "named", "as"])
Primitive("SEQUENCE", 11, 2, ["and", "then", "also", "next", "after"])
```

These encode the **structure** of natural language, not just content.

---

## Parameter Assignment

Parameters are assigned to concepts based on **positional proximity**:

```python
for concept in concepts:
    concept['params'] = []
    concept_end = concept['end']
    next_concept_start = get_next_concept_start(concept)
    
    # Collect params between this concept and the next
    for pos, value in param_positions:
        if concept_end <= pos < next_concept_start:
            concept['params'].append(value)
```

This handles queries like:
```
"create directory src and create file main.py"
         ↓      ↓           ↓     ↓
      concept  param     concept param
```

---

## Tokenization

Proper tokenization is critical for parameter preservation:

```python
def _tokenize(self, text: str) -> List[str]:
    # Match: filenames (word.ext), paths (/foo/bar), hyphenated words, or regular words
    tokens = re.findall(r'[\w./]+\.[\w]+|/[\w./]+|\b[\w]+-[\w-]+\b|\b\w+\b', text.lower())
    return tokens
```

This preserves:
- **Filenames**: `test.txt`, `main.py`
- **Paths**: `/tmp/output.txt`, `/home/user/file`
- **Hyphenated words**: `my-project`, `test-data`

---

## Results

### Test Suite

26 comprehensive tests covering:
- Basic compound queries
- Natural language variations
- Compound with parameters
- Three+ command sequences
- File, system, process, network operations
- Edge cases (hyphens, paths)

### Accuracy

```
RESULTS: 26/26 passed (100%)
```

### Example Resolutions

| Query | Result |
|-------|--------|
| "create directory test and create file readme.txt" | `mkdir -p test && touch readme.txt` |
| "list files and show disk space" | `ls && df` |
| "create a directory called src and create a file called main.py" | `mkdir -p src && touch main.py` |
| "show disk space then show memory then show processes" | `df && free && ps` |

---

## Philosophical Notes

### Why This Is Geometric

1. **Concept extraction** uses φ-MAX encoding and distance-based matching
2. **Parameter detection** uses the geometric property of "semantic voids"
3. **Action boosting** uses dimensional analysis of the encoding
4. **Structural primitives** encode grammar geometrically

### What Remains Non-Geometric

- **Tokenization**: Regex-based (syntactic, not semantic)
- **Quoted strings**: Explicit markers detected via pattern matching
- **Position assignment**: Sequential logic based on word indices

This is acceptable because these are **syntactic** operations, not **semantic** ones. The semantic understanding (what concepts, what parameters) is geometric.

### The Sierpinski Property

The sliding window approach exhibits a fractal-like property: each window is a self-contained encoding that can be matched independently. Overlapping windows don't interfere because we select non-overlapping regions greedily.

---

## Future Directions

1. **Context tracking**: "in it" referring to previous directory
2. **Piping**: "find files | count them" → `find . | wc -l`
3. **Conditionals**: "if file exists, delete it"
4. **Loops**: "for each file, show contents"

These will require additional structural primitives and more sophisticated parameter assignment.

---

## Summary

Compound phrase resolution combines:
- **Sliding window encoding** for multi-concept extraction
- **Action boosting** for command-oriented scoring
- **Semantic void detection** for geometric parameter identification
- **Positional assignment** for parameter-to-concept mapping

The result is a system that translates natural language compound queries into executable command sequences using primarily geometric methods.
