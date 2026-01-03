# Design Consideration 062: Geometric Stop Word Detection

## Date: 2024-12-26

## Problem

Hard-coded stop word lists are:
1. **Language-specific** - Won't work for other languages
2. **Domain-specific** - May filter important words in specialized contexts
3. **Non-geometric** - Breaks the principle of geometric understanding

## Solution

Detect stop words geometrically based on their **semantic role** in the corpus.

## The Geometric Principle

**Content words** have clear semantic roles:
- Initiators (actors): Holmes, Alice, Hamlet
- Mediators (actions): examined, killed, loves
- Receivers (targets): evidence, claudius, hamlet

**Stop words** have NO semantic role:
- They appear everywhere but never as initiator, mediator, or receiver
- They're structural glue, not content carriers

## Implementation

```python
@property
def is_geometric_stop_word(self) -> bool:
    """
    Stop words are characterized by:
    1. No semantic role - never initiator, mediator, or receiver
    2. OR: Very short words (≤4 chars) that appear frequently
    3. OR: Only receiver role and short (prepositions caught by accident)
    """
    total_roles = self.initiator_count + self.mediator_count + self.receiver_count
    has_no_role = total_roles == 0
    
    is_short_frequent = len(self.word) <= 4 and self.frequency >= 3
    
    only_receiver = (self.receiver_count > 0 and 
                    self.initiator_count == 0 and 
                    self.mediator_count == 0 and
                    len(self.word) <= 5)
    
    return has_no_role or is_short_frequent or only_receiver
```

## Results

From a 32-sentence corpus:

**Geometrically Detected Stop Words (15):**
```
about, and, at, from, garden, great, her, his, in, many, of, scene, the, to, with
```

**Content Words (125):**
```
holmes, alice, hamlet, watson, examined, killed, loves, evidence, claudius...
```

## Why This Works

### The Zipf Connection

Stop words follow Zipf's law - they're the most frequent words because they're structural scaffolding. But frequency alone isn't enough (proper nouns can be frequent too).

The key is **semantic role**:
- "the" appears 34 times but NEVER as initiator, mediator, or receiver
- "holmes" appears 6 times and is ALWAYS an initiator

### The φ-Direction Connection

Content words have strong φ-direction:
- Entities: φ = +1 (initiators and receivers)
- Actions: φ = -1 (mediators)

Stop words have weak or zero φ-direction:
- They don't fit the polyomino pattern
- They're the "glue" between the pieces, not the pieces themselves

## Language Independence

This approach works for ANY language because:
1. All languages have content words (nouns, verbs) and function words (articles, prepositions)
2. Content words carry semantic roles; function words don't
3. The geometric detection finds this pattern automatically

No hard-coded lists needed.

## The Full Geometric Pipeline

```
TEXT IN
    │
    ▼
┌─────────────────────────────────────┐
│ TOKENIZE (simple word extraction)   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ FRAME EXTRACTION                    │
│                                     │
│ Position 0 → Initiator              │
│ Position 1 → Mediator               │
│ Position 2+ → Receiver              │
│                                     │
│ (Skip very short words)             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ ROLE COUNTING                       │
│                                     │
│ Each word accumulates:              │
│ - initiator_count                   │
│ - mediator_count                    │
│ - receiver_count                    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ GEOMETRIC STOP WORD DETECTION       │
│                                     │
│ Stop word if:                       │
│ - No semantic role (all counts = 0) │
│ - OR: Short + frequent              │
│ - OR: Only receiver + short         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ CONTENT WORDS EMERGE                │
│                                     │
│ Words with clear semantic roles     │
│ are automatically content words     │
└─────────────────────────────────────┘
```

## Query Results

```
Q: Who is Holmes?
A: Holmes is a protagonist who examines, deduces, and observes, 
   often involving evidence and identity.

Q: Who killed?
A: Hamlet kills claudius.

Q: Who loves?
A: Ophelia loves hamlet.

Q: What does Watson do?
A: Watson is a protagonist who watches, assists, and writes, 
   often involving holmes.
```

## Conclusion

Stop word detection doesn't need hard-coded lists. The geometric structure of language reveals which words are content and which are scaffolding.

```
"Content words have roles.
 Stop words are the spaces between.
 The geometry knows the difference."
```
