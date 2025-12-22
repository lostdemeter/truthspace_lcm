# Design Consideration 034: Bootstrapped Instinct Knowledge

## Overview

This document captures observations from scaling GeometricLCM to ingest a full book (Moby Dick) and identifies what "bootstrapped instinct knowledge" would need to make the system work without hard-coded patterns.

## Current State

### What Works
- **Analogy reasoning**: `ahab:captain :: starbuck:?` → `first_mate` (90% confidence)
- **Relation learning**: 90%+ consistency achieved
- **Entity extraction**: Finds characters, places, things
- **Pattern-based fact extraction**: Extracts ~150 relations from 1000 sentences

### What Doesn't Work Well
- **Question understanding**: "Who is Ahab?" treats "who" as an entity
- **Coreference**: "He said" doesn't link to the actual speaker
- **Context**: No understanding of narrative flow
- **Temporal reasoning**: No sense of before/after

## Hard-Coded Knowledge (Current)

Currently, we have hard-coded:

### 1. Question Patterns (`geometric_lcm.py:ask()`)
```python
# "What is the X of Y?"
match = re.search(r"what\s+(?:is\s+)?the\s+(\w+)\s+of\s+(\w+)", q)

# "Who wrote X?"
match = re.search(r"who\s+wrote\s+(.+?)[\?]?$", q)
```

### 2. Relation Patterns (`ingest_book.py`)
```python
# "X is the Y of Z" → (Z, has_Y, X)
(r"(\w+)\s+is\s+the\s+(\w+)\s+of\s+(?:the\s+)?(\w+)", ...)

# "X is a Y" → (X, is_a, Y)
(r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", ...)
```

### 3. Character Knowledge
```python
MOBY_DICK_CHARACTERS = {
    "ahab": "captain",
    "ishmael": "narrator",
    ...
}
```

## What Bootstrapped Knowledge Would Need

### Level 1: Word Classes (Parts of Speech)

The system needs to know:
- **Question words**: who, what, where, when, why, how
- **Pronouns**: he, she, it, they (for coreference)
- **Verbs**: action words that create relations
- **Prepositions**: in, on, at, to, from (spatial/temporal relations)

**How to learn this geometrically:**
- Question words appear at sentence START before a verb
- Pronouns appear where nouns would appear
- Verbs appear between noun phrases
- These patterns could be learned from co-occurrence

### Level 2: Sentence Structure Templates

Common patterns that indicate relations:
```
[ENTITY] is the [ROLE] of [ENTITY]  → role relation
[ENTITY] [VERB]ed [ENTITY]          → action relation
[ENTITY] is a [TYPE]                → type relation
[ENTITY] and [ENTITY]               → association
```

**How to learn this geometrically:**
- Template = sequence of word-class positions
- Learn templates from examples with known relations
- Match new sentences to learned templates

### Level 3: Question-Answer Mapping

Questions map to relation queries:
```
"Who is X?"     → find all relations where X is subject
"What is X?"    → find X's type (is_a relation)
"Where is X?"   → find X's location (located_in relation)
"Who did X?"    → find subject of action X
```

**How to learn this geometrically:**
- Question word → relation type mapping
- Learn from Q&A pairs: "Who is the captain?" → "Ahab is the captain"
- The question structure predicts the answer structure

### Level 4: Coreference Resolution

Track entity references across sentences:
```
"Ahab stood on the deck. He looked at the sea."
→ "He" refers to "Ahab"
```

**How to learn this geometrically:**
- Pronouns inherit position from recent matching entities
- Gender/number agreement as geometric constraint
- Recency bias in position inheritance

## Proposed Architecture for Bootstrapped Knowledge

### Phase 1: Word Class Learning
```python
class WordClassLearner:
    """Learn word classes from position in sentences."""
    
    def learn_from_corpus(self, sentences):
        # Words at position 0 before "?" → question_word
        # Words that follow "the/a/an" → noun
        # Words between nouns → verb/preposition
        pass
```

### Phase 2: Template Learning
```python
class TemplateExtractor:
    """Learn sentence templates that indicate relations."""
    
    def extract_template(self, sentence, known_relation):
        # Given: "Ahab is the captain of the Pequod"
        # Known: (pequod, has_captain, ahab)
        # Learn: [ENTITY] is the [ROLE] of [ENTITY]
        pass
```

### Phase 3: Question Understanding
```python
class QuestionParser:
    """Parse questions using learned patterns."""
    
    def parse(self, question):
        # Match against learned question templates
        # Return: (query_type, entities, expected_relation)
        pass
```

### Phase 4: Coreference Tracker
```python
class CoreferenceTracker:
    """Track entity references across sentences."""
    
    def resolve(self, pronoun, context):
        # Find most recent entity matching pronoun's constraints
        # Return: resolved entity name
        pass
```

## Implementation Roadmap

### Immediate (This Session)
1. ✅ Basic book ingestion working
2. ✅ Character bootstrap knowledge
3. ⬜ Improve question handling for book queries

### Short-term
1. Learn word classes from corpus statistics
2. Extract templates from known relations
3. Build question-template mapping

### Medium-term
1. Coreference resolution
2. Context window for narrative flow
3. Temporal relation extraction

### Long-term
1. Self-improving pattern discovery
2. Cross-domain knowledge transfer
3. Emergent question understanding

## Key Insight

The goal is to replace hard-coded patterns with **learned geometric structures**:

| Current (Hard-coded) | Future (Learned) |
|---------------------|------------------|
| Regex patterns | Template vectors |
| Character lists | Entity type clusters |
| Question keywords | Question-type positions |
| Relation names | Relation direction vectors |

The bootstrapped knowledge is essentially **meta-knowledge about language structure** that allows the system to learn domain knowledge (like Moby Dick characters) automatically.

## Connection to Prior Work

### Attractor Dynamics
- Word classes could emerge as attractor basins
- Question words cluster together, nouns cluster together
- The structure emerges from usage patterns

### Error-Driven Construction
- Failed queries indicate missing patterns
- Each error points to a template that should be learned
- The system builds its own pattern library

### Holographic Encoding
- Phase could encode word class (noun vs verb)
- Magnitude could encode specificity (proper noun vs common noun)
- Complex encoding enables richer word representations

## Conclusion

Bootstrapped instinct knowledge is **meta-knowledge about language** that enables learning domain knowledge. The key components are:

1. **Word classes** - What role does each word play?
2. **Templates** - What patterns indicate relations?
3. **Question mapping** - How do questions map to queries?
4. **Coreference** - How do pronouns link to entities?

All of these can potentially be learned geometrically rather than hard-coded, making the system truly self-improving.
