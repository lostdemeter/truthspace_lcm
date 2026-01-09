# Design 048: Agent-Grounded Geometric Interpretation

## Discovery

When testing the `GeometricIntentClassifier`, we found that queries like "what is a histogram" were being misclassified as CODE instead of KNOWLEDGE because:

1. The word "histogram" overlaps with code generation patterns
2. The word "what" was being filtered as a filler word (high-frequency)
3. Without "what", the query loses its intent signal

## The Deeper Problem

The issue isn't just filler word filtering. It's that **our model has no concept of itself**.

When a human asks "can you pick up a cup?":
- A human answers "yes" (has physical capabilities)
- An AI answers "no" (lacks physical capabilities)
- A robot answers "yes" (has physical capabilities)

The query is the same. The answer differs because each agent has a different **self-model** - a geometric structure representing their capabilities, environment, and identity.

## The Insight

**Intent verbs encode the relationship between user and agent, not just content.**

| Verb | Relationship | Intent |
|------|--------------|--------|
| "what is" | User wants to KNOW | KNOWLEDGE |
| "create" | User wants agent to MAKE | CODE |
| "list" | User wants agent to SHOW | TOOL |

These verbs are NOT filler words. They're the **intent dimension** of the query.

The φ-Zipf duality correctly identifies them as high-frequency, but high-frequency ≠ unimportant for intent classification. They're high-frequency precisely BECAUSE they're fundamental to human-agent interaction.

## Geometric Model

### Query Encoding

A query encodes to a position with multiple dimensions:

```
Query: "what is a histogram"

Content dimensions:
  - domain: TECHNICAL (histogram is a technical concept)
  - specificity: MODERATE (specific but not expert-level)

Intent dimension:
  - action: QUERY (-1) ← encoded by "what is"

Full position: [domain=2, specificity=1, action=-1, ...]
```

### Agent Projection

The agent has a capability structure - a geometric representation of what it can do:

```
HyperChat capabilities:
  - KNOWLEDGE: answer questions, explain concepts
  - CODE: generate plots, write scripts
  - TOOL: list files, read documents
```

When a query is projected through the agent's capability structure:

```
Query position → Agent projection → Intent + Answer

"what is a histogram" → KNOWLEDGE capability → KNOWLEDGE intent
"create a histogram"  → CODE capability → CODE intent
```

### Why This Works

The same query, projected through different agents, yields different results:

```
"list files"
  → Human (no file system access): "I don't understand"
  → AI (has file system tools): "Here are the files..."
  → Robot (no file system): "I cannot do that"
```

The geometry is the same. The agent structure determines the path.

## Implementation

### Option 1: Action Dimension

Add an "action" dimension to the φ-lattice:

```python
ACTION = SemanticDimension(
    name="action",
    index=5,
    weight=PHI,
    level_meanings={
        -2: "strongly_query",    # what, why, how
        -1: "query",             # explain, describe
         0: "neutral",           # ambiguous
        +1: "action",            # create, make
        +2: "strongly_action",   # execute, run, delete
    }
)
```

### Option 2: Agent Capability Space

Define the agent's capabilities as positions in the same space:

```python
class AgentCapability:
    name: str
    position: np.ndarray  # Position in φ-lattice
    
hyperchat_capabilities = [
    AgentCapability("answer_questions", position=[0, 0, -1, 0, 0]),  # KNOWLEDGE
    AgentCapability("generate_plots", position=[2, 1, +1, 0, 0]),    # CODE
    AgentCapability("list_files", position=[0, 0, +2, 0, 0]),        # TOOL
]
```

Query intent = closest capability after projection.

### Option 3: Hybrid

Use the action dimension for encoding, then project through capabilities for routing.

## Connection to Existing Dimensions

We already have:
- **intrinsic_functional** (Design 047): "what it IS" vs "what it's FOR"
- **intent** (existing): inform, request, command

The action dimension is related but distinct:
- **intrinsic_functional**: About the CONTENT (ontological vs relational)
- **intent**: About the SPEECH ACT (statement vs question vs command)
- **action**: About the AGENT RELATIONSHIP (query vs create vs execute)

## Philosophical Implication

This connects to the core hypothesis:

> If structure IS information, then **self-awareness must also be geometric**.

The agent's understanding of itself emerges from its position in capability space. Its identity IS its geometric structure. The tools it can call, the knowledge it has access to, the transformations it can perform - these collectively define "who it is" geometrically.

## Next Steps

1. Add "action" dimension to φ-lattice primitives
2. Define HyperChat's capability structure geometrically
3. Modify intent classification to project through capabilities
4. Test disambiguation cases

## Experiment

See `/home/thorin/truthspace-lcm/experiments/agent_grounded_interpretation.py`

---

## Key Quote

> "The verbs 'what', 'create', 'list' are NOT filler words. They encode the RELATIONSHIP between the user and the agent. These verbs are the INTENT DIMENSION of the query."
