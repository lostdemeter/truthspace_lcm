# Design Consideration 076: Emergent Classifier Protocol

**Date**: December 30, 2024  
**Author**: Lesley Gushurst  
**Status**: Implemented

## Executive Summary

This document describes the **Emergent Classifier Protocol** - a systematic approach for converting hardcoded word lists and pattern-based detection into emergent, self-learning classifiers. This generalizes the pattern we kept solving across stopwords, verbs, pronouns, entities, and gender detection.

## The Problem

Throughout development, we repeatedly wrote code like:

```python
# Hardcoded stopwords
stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', ...}

# Hardcoded verbs
common_verbs = {'said', 'went', 'came', 'made', 'took', ...}

# Pattern-based detection
if word.endswith('ed') or word.endswith('ing'):
    is_verb = True
```

Problems with this approach:
1. **Maintenance burden** - Lists need manual updates
2. **Domain brittleness** - Lists don't transfer across domains
3. **Language dependence** - Patterns are language-specific
4. **Missing coverage** - Can't discover new members

## The Insight

Word categories are **STRUCTURAL**, not semantic. They can be identified by:

| Category | Structural Signal |
|----------|-------------------|
| Stopwords | High frequency, uniform distribution, short length |
| Verbs | Position after subject, -ed/-ing morphology |
| Pronouns | Very high frequency, specific positions, closed class |
| Entities | Low frequency, capitalization, specific contexts |
| Gender | Co-occurrence patterns, suffix patterns (-ess, -ine) |

## The Solution: CategorySignature

Instead of hardcoding lists, we define **signatures** that describe structural properties:

```python
@dataclass
class CategorySignature:
    name: str
    
    # Frequency signals
    min_frequency: float = 0.0
    max_frequency: float = 1.0
    
    # Distribution signals
    min_document_frequency: float = 0.0
    max_document_frequency: float = 1.0
    
    # Position signals
    typical_positions: List[str] = []  # 'start', 'middle', 'end'
    position_weight: float = 0.0
    
    # Morphological signals
    positive_suffixes: List[str] = []
    negative_suffixes: List[str] = []
    suffix_weight: float = 0.0
    
    # Length signals
    min_length: int = 0
    max_length: int = 100
    length_weight: float = 0.0
    
    # Seed examples (for bootstrapping)
    seeds: Set[str] = set()
```

## The Protocol

When you find yourself writing a hardcoded list, follow this protocol:

### Step 1: IDENTIFY the Structural Signal

Ask: "What makes these words different from others?"

Examples:
- **Stopwords**: Very frequent, appear everywhere, short
- **Verbs**: Appear after subjects, have -ed/-ing forms
- **Entities**: Capitalized, less frequent, specific contexts
- **Pronouns**: Very frequent, specific positions, closed class

### Step 2: CREATE a Signature

```python
my_signature = CategorySignature(
    name="my_category",
    min_frequency=0.01,        # Frequency range
    max_length=5,              # Length constraint
    positive_suffixes=['ed'],  # Morphological patterns
    typical_positions=['middle'],  # Syntactic position
    seeds={'example1', 'example2'},  # Bootstrap examples
)
```

### Step 3: ADD to Classifier

```python
classifier = EmergentClassifierGear()
classifier.add_signature(my_signature)
```

### Step 4: LEARN from Data

```python
for document in corpus:
    classifier.learn_from_text(document, document_id=doc_id)
```

### Step 5: USE Emergently

```python
if classifier.is_category(word, 'my_category'):
    # Word belongs to category
```

### Step 6: VALIDATE and Refine

- Check precision/recall against known examples
- Adjust signature parameters
- Add more seeds if needed

## Implementation

### EmergentClassifierGear

```python
class EmergentClassifierGear(Gear):
    """
    A gear that discovers word categories from data patterns.
    """
    
    def __init__(self):
        self.signatures = {
            'stopword': STOPWORD_SIGNATURE,
            'verb': VERB_SIGNATURE,
            'pronoun': PRONOUN_SIGNATURE,
            'entity': ENTITY_SIGNATURE,
        }
        self.word_stats = {}
    
    def learn_from_text(self, text, document_id=None):
        """Learn word statistics from text."""
        # Track frequency, position, document frequency
        
    def classify(self, word) -> Dict[str, float]:
        """Score word against all signatures."""
        
    def is_category(self, word, category, threshold=0.5) -> bool:
        """Check if word belongs to category."""
```

### Pre-defined Signatures

**STOPWORD_SIGNATURE**:
```python
CategorySignature(
    name="stopword",
    min_frequency=0.01,
    max_length=4,
    length_weight=0.9,
    seeds={'the', 'a', 'an', 'is', 'are', ...},
)
```

**VERB_SIGNATURE**:
```python
CategorySignature(
    name="verb",
    typical_positions=['middle'],
    position_weight=0.3,
    positive_suffixes=['ed', 'ing'],
    negative_suffixes=['ness', 'ment', 'tion'],
    suffix_weight=0.7,
    seeds={'said', 'went', 'came', ...},
)
```

## Results

### Before (Hardcoded)
```python
common_verbs = {
    'said', 'saw', 'went', 'came', 'made', 'took', 'got', 'gave', 'found',
    'thought', 'told', 'asked', 'looked', 'seemed', 'felt', 'knew', 'wanted',
    # ... 50+ more hardcoded verbs
}

if action in common_verbs:
    is_verb = True
```

### After (Emergent)
```python
if self.classifier_gear:
    is_verb = self.classifier_gear.is_verb(action)
else:
    # Fallback to morphological patterns
    is_verb = action.endswith('ed') or action.endswith('ing')
```

### Test Results (Moby Dick)
```
Classifier learned: 218,461 words, 16,874 unique

Classification accuracy:
  said: verb=True ✓
  went: verb=True ✓
  called: verb=True ✓
  whale: verb=False ✓
  captain: verb=False ✓
  the: stopword=True ✓
  he: stopword=True ✓
  ocean: stopword=False ✓
```

## Benefits

### 1. Self-Improving
The classifier improves as it sees more data. No manual updates needed.

### 2. Domain Adaptive
Works across different domains - learns the vocabulary of each corpus.

### 3. Transparent
You can inspect why a word was classified:
```python
scores = classifier.classify('called')
# {'stopword': 0.0, 'verb': 1.0, 'pronoun': 0.0, 'entity': 0.0}
```

### 4. Extensible
Add new categories by defining signatures:
```python
ADJECTIVE_SIGNATURE = CategorySignature(
    name="adjective",
    positive_suffixes=['ful', 'less', 'ous', 'ive'],
    typical_positions=['middle'],
    ...
)
classifier.add_signature(ADJECTIVE_SIGNATURE)
```

### 5. Graceful Degradation
Falls back to morphological patterns when insufficient data.

## Connection to Previous Designs

| Design | Contribution |
|--------|--------------|
| **073 (Geometric RL)** | Additive learning - signatures can be refined, not replaced |
| **074 (Gear Chain)** | Modular architecture - classifier is a composable gear |
| **075 (Feedback)** | LLM can suggest signature refinements |
| **076 (This)** | Generalizes the pattern into a reusable protocol |

## The Meta-Pattern

This protocol itself follows a meta-pattern:

```
PROBLEM: We keep hardcoding X
    ↓
INSIGHT: X has structural properties
    ↓
SOLUTION: Define signature for X's structure
    ↓
RESULT: X emerges from data
```

This can be applied to ANY concept we find ourselves hardcoding:
- Sentence boundaries
- Paragraph structure
- Topic boundaries
- Sentiment words
- Domain-specific terminology

## Files

- `truthspace_lcm/gears/core/emergent_classifier.py` - EmergentClassifierGear implementation
- `truthspace_lcm/gears/core/conversational_chain.py` - Integration with chat system

## Usage

```python
from truthspace_lcm.gears.core.emergent_classifier import (
    EmergentClassifierGear,
    CategorySignature,
    create_custom_signature,
)

# Create classifier
classifier = EmergentClassifierGear()

# Learn from text
classifier.learn_from_text(document, document_id="doc1")

# Use emergent classification
if classifier.is_verb(word):
    process_verb(word)

if classifier.is_stopword(word):
    skip_word(word)

# Add custom category
my_sig = create_custom_signature(
    name="technical_term",
    seeds={'algorithm', 'function', 'variable'},
    length_range=(5, 20),
    suffixes=['tion', 'ment', 'ity'],
)
classifier.add_signature(my_sig)
```

## Conclusion

The Emergent Classifier Protocol transforms hardcoded knowledge into self-learning systems. By identifying the **structural signals** that distinguish word categories, we can:

1. **Replace** hardcoded lists with learnable signatures
2. **Adapt** to new domains automatically
3. **Improve** with more data
4. **Extend** to new categories easily

The key insight: **Structure is discoverable. Hardcoding is a sign that we haven't yet identified the structural signal.**

```
"When you find yourself writing a list,
 ask what makes those items similar.
 The answer is a signature.
 The signature is emergent knowledge."
```
