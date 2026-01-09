# Design 111: Perspective as Layer Manipulation

## Discovery

Building on Design 110 (Agent-Grounded Interpretation), we realized that agent identity might not require capability projection. Instead, perspective can be implemented as a simple **layer offset** - matching how LLM system prompts actually work.

## The Insight

### Traditional LLM Approach

```
"You are an expert physicist" + "What is gravity?"
         ↓
    Sets initial hidden state / attention bias
         ↓
    All subsequent layers traverse from this "perspective"
         ↓
    Output is colored by the physicist framing
```

### Geometric Equivalent

Instead of projecting queries through a capability structure, we **offset the query position by a perspective vector** before traversal:

```
Query: "What is gravity?"
Perspective: "AI assistant with knowledge capabilities"
         ↓
Query position + Perspective offset = Adjusted position
         ↓
Traverse from adjusted position
         ↓
Output reflects the perspective
```

## Key Realization

**Perspective doesn't FILTER the query - it SHIFTS it.**

"You are an expert physicist" doesn't mean:
- "Only return physics answers"

It means:
- "Start from a physics-expert position in the space"

## Experimental Results

Same query "List files" from different perspectives:

| Perspective | Adjusted Position | Answer Found |
|-------------|-------------------|--------------|
| Child | [0, -1, 0, -1, 0, 2] | "A list is like when you write down all your toys..." |
| AI Assistant | [2, 1, 0, 0, 0, 2] | "I can list files in a directory for you..." |
| Software Developer | [2, 2, 0, 0, 1, 2] | "Lists in Python are mutable sequences..." |

**Same query, different perspectives, different answers** - all geometrically valid.

## Implementation

### Perspective as Vector

```python
@dataclass
class Perspective:
    name: str
    description: str
    offset: np.ndarray  # [domain, specificity, intent, formality, intrinsic_functional, action]
    
    def apply(self, query_position: np.ndarray) -> np.ndarray:
        return query_position + self.offset
```

### HyperChat Identity

```python
HYPERCHAT_IDENTITY = Perspective(
    name="HyperChat",
    description="AI assistant that can answer questions, generate plots, and use tools",
    offset=np.array([
        2,   # domain: technical
        1,   # specificity: moderate
        0,   # intent: neutral
        0,   # formality: neutral
        0,   # intrinsic_functional: balanced
        0,   # action: responds to user's action
    ])
)

# In query processing:
query_position = encode(query)
adjusted = HYPERCHAT_IDENTITY.apply(query_position)
result = find_nearest(adjusted)
```

## Comparison with Design 110

| Approach | Mechanism | Complexity |
|----------|-----------|------------|
| **Capability Projection** (Design 110) | Project query through capability structure | Higher - need to define capabilities |
| **Perspective Offset** (Design 111) | Add identity vector to query position | Simpler - just vector addition |

## Why This is Simpler

1. **No capability structures needed** - just a vector
2. **Works with existing φ-lattice** - same encoding, just offset
3. **Matches LLM intuition** - system prompts work this way
4. **Composable** - can stack perspectives (expert + formal + technical)

## Composability

Perspectives can be combined:

```python
# Base identity
base = HYPERCHAT_IDENTITY.offset

# Add "expert mode"
expert_modifier = np.array([0, 2, 0, 1, 0, 0])  # boost specificity and formality

# Combined perspective
combined = base + expert_modifier
```

This is like stacking system prompts:
- "You are HyperChat" + "You are an expert" + "Be formal"

## Connection to LLM Architecture

In transformer architecture:
- System prompt → Sets initial hidden state
- Each layer → Transforms based on that initial state
- Output → Colored by the accumulated perspective

In our geometric model:
- Perspective → Offset vector in φ-space
- Query encoding → Position in φ-space
- Adjusted position → Query + Perspective
- Nearest neighbor → Answer from that viewpoint

The mathematics are different, but the **conceptual operation is the same**: perspective sets the starting point for all subsequent processing.

## Implications

### For Intent Classification

The ACTION dimension (Design 110) encodes what the user wants to DO.
The perspective offset encodes WHO is interpreting.

Together:
```
final_position = query_position + action_encoding + perspective_offset
```

### For Knowledge Retrieval

Different perspectives find different knowledge:
- Physicist asking about energy → E=mc²
- Economist asking about energy → market dynamics
- Child asking about energy → "what makes things go"

The knowledge base doesn't change. The perspective determines which part is "nearest".

### For Response Generation

The perspective could also influence output style:
- Formal perspective → technical language
- Casual perspective → conversational tone
- Expert perspective → detailed explanations

## Next Steps

1. Define HyperChat's default perspective vector
2. Allow perspective modification via system prompts
3. Test perspective composition (stacking modifiers)
4. Integrate with existing φ-lattice encoding

## Experiment

See `/home/thorin/truthspace-lcm/experiments/perspective_layer_manipulation.py`

---

## Key Quote

> "The query encodes WHAT is being asked. The perspective encodes WHO is asking (or being asked). Together they determine WHERE in the space we look."
