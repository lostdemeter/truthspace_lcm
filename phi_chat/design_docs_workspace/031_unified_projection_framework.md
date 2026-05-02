# Unified Projection Framework

## The Core Insight

**Q&A and style transfer are the SAME operation** - projection onto axes in universal semantic space.

- Q&A axes: WHO, WHAT, WHERE, WHEN, WHY, HOW
- Style axes: FORMAL, EMOTIONAL, ABSTRACT, etc.
- Both are directions in the same space
- Both use the same projection operation

## The Universal Semantic Space

All language exists in a universal semantic space with fixed axes:

```
CONCRETE ←────────────────────────→ ABSTRACT
CASUAL   ←────────────────────────→ FORMAL
NEGATIVE ←────────────────────────→ POSITIVE
PASSIVE  ←────────────────────────→ ACTIVE
GENERAL  ←────────────────────────→ SPECIFIC
RATIONAL ←────────────────────────→ EMOTIONAL
SIMPLE   ←────────────────────────→ COMPLEX
```

These are like X, Y, Z in physical space - they're built into the structure of language.

## Styles as Positions

A style is a POSITION in this universal space:

| Style | ABSTRACT | FORMAL | NEGATIVE | ACTIVE | SPECIFIC | EMOTIONAL | COMPLEX |
|-------|----------|--------|----------|--------|----------|-----------|---------|
| Q&A | -0.3 | +0.2 | 0.0 | +0.3 | +0.5 | -0.5 | -0.3 |
| Warhammer | +0.7 | +0.8 | -0.9 | +0.6 | +0.3 | +0.8 | +0.7 |
| Romance | +0.2 | 0.0 | +0.3 | +0.2 | +0.4 | +0.9 | +0.3 |
| Technical | -0.6 | +0.5 | 0.0 | +0.4 | +0.8 | -0.8 | 0.0 |

## Q&A Axes are Style Axes

The Q&A question types (WHO, WHAT, WHERE) are **specific directions** in this same space:

- **WHO axis**: Points toward SPECIFIC + IDENTITY (person/entity)
- **WHAT axis**: Points toward SPECIFIC + DEFINITION (thing/concept)
- **WHERE axis**: Points toward SPECIFIC + LOCATION (place)
- **WHY axis**: Points toward SPECIFIC + REASON (cause)
- **HOW axis**: Points toward SPECIFIC + METHOD (process)

These are not separate from style axes - they're **combinations** of universal axes.

## The Holographic Principle

```
Question = Content - Gap
Answer   = Content + Fill

Gap = direction from question position to answer position
Fill = movement along that direction
```

The **gap in a question IS an axis direction**. Different question types define different axis directions.

## Unified Projection Operation

All projection uses the same formula:

```python
def project(content, axis_direction):
    """Project content onto an axis."""
    content_vec = encode(content)
    return dot(content_vec, axis_direction) / norm(axis_direction)
```

For Q&A:
```python
# "Who is Captain Ahab?" projects onto IDENTITY axis
answer_relevance = project(candidate_answer, IDENTITY_axis)
```

For style:
```python
# "Make this more formal" projects onto FORMAL axis
formality = project(content, FORMAL_axis)
```

## Style Transfer = Axis Movement

To transfer content from one style to another:

```python
def style_transfer(content, source_style, target_style):
    """Move content from source style position to target style position."""
    content_vec = encode(content)
    
    # Compute movement vector
    movement = target_style.position - source_style.position
    
    # Move content
    new_vec = content_vec + movement
    
    return decode(new_vec)
```

This is the same as Q&A matching:
- Question defines a "source position" (has gap)
- Answer defines a "target position" (fills gap)
- Matching = finding content that bridges the gap

## The Chicken-and-Egg Solution

**Problem**: To project into a style, we need axes. To get axes, we need examples. To recognize examples, we need to know the style.

**Solution**: Universal axes are FIXED. Styles are just positions measured on these fixed axes.

To define a new style:
1. Collect exemplars
2. Measure their average position on universal axes
3. That position IS the style

No circular dependency - the axes are always there.

## Geometric Operations

| Operation | Formula | Description |
|-----------|---------|-------------|
| Measure position | `dot(content, axis)` | Where is content on this axis? |
| Style similarity | `1 - distance(content, style)` | How close to this style? |
| Style transfer | `content + (target - source)` | Move toward target style |
| Q&A matching | `dot(question, answer)` | Does answer fill question's gap? |
| Gap detection | `target - source` | What's missing? |

## Implementation Files

- `papers/recursive_holographic_qa.py` - Q&A projection
- `papers/style_projection.py` - Style projection via PCA
- `papers/universal_style_space.py` - Universal axes framework
- `papers/style_centroid.py` - **Centroid approach (validated)**

---

## Experimental Validation: Style Vector Arithmetic

We tested three approaches to style projection via vector arithmetic:

### Approach 1: Random Hash Vectors + Contrastive Direction

```python
style_direction = average(encode(styled) - encode(neutral))
styled_content = encode(content) + style_direction
```

**Result**: Mixed - classification unreliable. Random vectors don't capture semantic relationships.

### Approach 2: Co-occurrence Vectors + Contrastive Direction

Built word vectors from co-occurrence statistics (SVD on co-occurrence matrix).

**Result**: Better - some correct classifications. Words that appear together get similar vectors:
- `ship ~ cruiser`: 0.982
- `rain ~ tears`: 0.729
- `inquisitor ~ emperor`: 0.604

### Approach 3: Centroid Approach (VALIDATED)

**Style = centroid (average position) of exemplars.**

```python
# Define style
style_centroid = average([encode(exemplar) for exemplar in exemplars])

# Classify by cosine similarity
similarity = cosine(encode(text), style_centroid)

# Transfer by interpolation
new_vec = (1 - α) * encode(content) + α * style_centroid
```

**Result: 8/8 correct classifications!**

| Test Text | Top Match | Score |
|-----------|-----------|-------|
| "The Inquisitor purged the heretics with holy fury" | Warhammer | 0.367 |
| "Her heart raced as their eyes met across the ballroom" | Romance | 0.297 |
| "The function returns a boolean value based on input" | Technical | 0.321 |
| "The rain fell like tears on the dark city streets" | Noir | 0.513 |
| "He burned with zealous wrath for the Emperor" | Warhammer | 0.116 |
| "She ached for him with every fiber of her being" | Romance | 0.325 |
| "I lit a cigarette and watched the smoke curl upward" | Noir | 0.185 |
| "Initialize the array with default values" | Technical | 0.332 |

### Style Directions Reveal Semantic Differences

The direction between centroids shows what makes styles different:

| Direction | Toward | Away From |
|-----------|--------|-----------|
| Neutral → Warhammer | fury, purpose, faith | the, was, he |
| Neutral → Romance | forbidden, awakened | the, and, was |
| Neutral → Noir | sins, bodies, dead | the, was, and |
| Romance → Warhammer | purpose, chapter, faith | he, she, as |

### The Validated Formula

```python
# 1. Define style from exemplars
style_centroid = mean([encode(e) for e in exemplars])

# 2. Classify content
best_style = argmax(cosine(encode(text), centroid) for centroid in styles)

# 3. Transfer content toward style
transferred = (1 - strength) * encode(content) + strength * target_centroid

# 4. Find style direction
direction = centroid_B - centroid_A  # What makes B different from A
```

### Why Centroids Work Better Than Contrastive Directions

1. **Robustness**: Averaging many exemplars reduces noise
2. **No pairing required**: Don't need (neutral, styled) pairs
3. **Interpretable**: Centroid IS the style, geometrically
4. **Composable**: Direction between centroids = style difference

---

## Key Insight

**The gap in a question and the direction of a style are the same thing** - both are axis directions in universal semantic space.

Q&A is just one style. Warhammer 40k is another. Romance is another. They all live in the same space, and projection onto their axes uses the same geometric operation.

This is why the system is generalizable:
- The axes are universal (built into language)
- Styles are just positions (easy to define)
- Projection is just dot product (purely geometric)
- No training needed - just measurement and arithmetic

---

## Connection to Holographic Principle

The centroid approach connects directly to holographic encoding:

```
Holographic:  I = I_L + I_R  (interference pattern stores both views)
Style:        centroid = average(exemplars)  (centroid stores the "essence")

Holographic:  I_R - I_L = depth information
Style:        centroid_B - centroid_A = style difference
```

The **style centroid IS a holographic encoding** of the style - it captures the interference pattern of all exemplars, and the difference between centroids reveals the "depth" (what makes styles different).

## Future Directions

1. **Q&A as Style**: Treat question types (WHO, WHAT, WHERE) as styles with their own centroids
2. **Hierarchical Styles**: Styles within styles (e.g., "Warhammer Inquisitor" vs "Warhammer Ork")
3. **Style Interpolation**: Blend styles by interpolating between centroids
4. **Dynamic Style Discovery**: Cluster content to discover emergent styles automatically
