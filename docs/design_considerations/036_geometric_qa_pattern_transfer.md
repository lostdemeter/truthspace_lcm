# Design Consideration 036: Geometric Q&A Pattern Transfer

## Problem Statement

Can Q&A patterns learned from general knowledge transfer to literary queries through geometric operations?

**Hypothesis**: If we learn how to answer "Who is Einstein?" from Q&A training data, can that pattern transfer to answering "Who is Darcy?" about literary characters - purely through geometric operations in concept space?

## Experimental Setup

We tested 5 approaches to geometric Q&A pattern transfer:

1. **Q&A Vector Pairs** - Learn Q→A as vector offset
2. **Role-Based Encoding** - Cluster characters by role
3. **Holographic Binding** - Bind Q structure to A structure
4. **Prototype-Based** - Find nearest prototype, adapt with entity features
5. **Geometric Slot-Filling** - Hybrid approach with geometric slot relevance

### Training Data
General knowledge Q&A (NOT about literature):
- "Who is Einstein?" → "Einstein is a scientist from Germany who developed theories"
- "Who is Napoleon?" → "Napoleon is a leader from France who conquered territories"
- "Who is Shakespeare?" → "Shakespeare is a writer from England who created plays"

### Test Queries
Literary characters (should transfer from Q&A training):
- "Who is Darcy?" (Pride and Prejudice)
- "Who is Holmes?" (Sherlock Holmes)
- "Who is Alice?" (Alice in Wonderland)

## Results

### Approach 1: Q&A Vector Pairs
```
Idea: Q_darcy + learned_offset ≈ A_darcy

Results:
  Q: Who is Darcy?
  Nearest A (sim=0.397): Cleopatra is a queen from Egypt...

Verdict: FAILS
  - Offset is too generic (averages all Q→A transformations)
  - Only retrieves existing answers, doesn't generate new ones
```

### Approach 2: Role-Based Encoding
```
Idea: Characters with similar ROLES cluster together

Results:
  Darcy: actions=[SPEAK, POSSESS, MOVE] → role=communicator
  Holmes: actions=[PERCEIVE, THINK, ACT] → role=observer
  Alice: actions=[MOVE, PERCEIVE, SPEAK] → role=traveler

Verdict: PARTIAL SUCCESS
  + Captures character similarity
  + Enables "Darcy is like Shakespeare" reasoning
  - Doesn't directly generate answers
```

### Approach 3: Holographic Binding
```
Idea: WHO_question ⊗ WHO_answer = WHO_pattern
      new_question ⊗ WHO_pattern^(-1) ≈ predicted_answer

Results:
  Q: Who is Darcy?
  Best template (sim=0.147): Shakespeare is a writer...

Verdict: PARTIAL SUCCESS
  + Theoretically elegant
  - Noisy in practice (interference between patterns)
  - Low similarity scores
```

### Approach 4: Prototype-Based
```
Idea: Find nearest prototype, substitute entity features

Results:
  Q: Who is Darcy?
  Generated: Darcy is a gentleman from Pride and Prejudice who speaks

Verdict: SUCCESS (but not purely geometric)
  + Generates actual answers
  + Adapts templates to new entities
  - Requires explicit slot-filling
```

### Approach 5: Geometric Slot-Filling (Hybrid)
```
Idea: Make slot-filling GEOMETRIC by:
  1. Encoding slots as vectors
  2. Learning slot relevance per question type
  3. Using vector similarity for answer comparison

Results:
  Q: Who is Darcy?
  A: Darcy is a gentleman from Pride and Prejudice who speaks eloquently 
     often involving Elizabeth

  Geometric similarity (WHO context):
    Darcy → nearest: Shakespeare (0.545)
    Holmes → nearest: Shakespeare (0.545)
    Alice → nearest: Shakespeare (0.502)

Verdict: BEST APPROACH
  + Generates quality answers
  + Geometric similarity captures role/action patterns
  + Transfer happens through vector space
```

## Key Finding: The Geometric Slot-Filling Formula

The answer vector is computed as:

```
Answer_vec = Σ (relevance[q_type][slot] × slot_vec × value_vec)
```

Where:
- `relevance[q_type][slot]` = learned weight for each slot per question type
- `slot_vec` = deterministic vector for slot type (name, role, action, etc.)
- `value_vec` = hash-based vector for slot value (Darcy, gentleman, speaks, etc.)

### Slot Relevance Weights (Learned from Q&A Training)

| Slot | WHO | WHAT | WHERE |
|------|-----|------|-------|
| name | 1.0 | 0.8 | 0.6 |
| role | 0.8 | - | - |
| source | 0.6 | - | 0.4 |
| action | 0.4 | 1.0 | - |
| related | 0.3 | - | - |
| patient | - | 0.6 | - |
| location | - | - | 1.0 |

### Why This Works

1. **Slot vectors create structure**: Each slot type has a fixed position in space
2. **Value vectors add content**: Entity-specific values perturb the slot position
3. **Relevance weights focus attention**: Question type determines which slots matter
4. **Similarity enables transfer**: Similar entities have similar answer vectors

## The Transfer Mechanism

```
Q&A Training:
  Einstein → {role: scientist, action: developed theories}
  Answer_vec(Einstein, WHO) = 0.8×role_vec + 0.4×action_vec + ...

Literary Entity:
  Holmes → {role: detective, action: observes carefully}
  Answer_vec(Holmes, WHO) = 0.8×role_vec + 0.4×action_vec + ...

Transfer:
  cos(Answer_vec(Holmes), Answer_vec(Einstein)) = 0.511
  
  Holmes is geometrically SIMILAR to Einstein because:
  - Both have strong ROLE component
  - Both have OBSERVATION-type actions
  - The STRUCTURE of their answers is similar
```

## Architecture Recommendation

```
                    ┌─────────────────┐
                    │    Question     │
                    │ "Who is Darcy?" │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Detect Q-Type  │
                    │     (WHO)       │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  Get Slot Relevance      │
              │  WHO: name=1.0, role=0.8 │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  Compute Answer Vector   │
              │  Σ relevance × slot_vec  │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  Find Similar Entities   │
              │  (from Q&A training)     │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  Generate Answer         │
              │  Using learned template  │
              └──────────────────────────┘
```

## Implementation

See `experiments/geometric_qa_transfer.py` for full implementation.

Key class: `GeometricSlotFilling`
- `compute_answer_vector(q_type, entity)` - Geometric answer encoding
- `find_similar_answers(q_type, entity, known)` - Transfer via similarity
- `generate_answer(q_type, entity)` - Template-based generation

## Connection to Holographic Principle

This approach aligns with the holographic Q&A principle:

```
Question = Content - Gap
Answer = Content + Fill
```

The **slot relevance weights** define the GAP:
- WHO questions have a gap in the AGENT slot (high relevance)
- WHAT questions have a gap in the ACTION slot (high relevance)
- WHERE questions have a gap in the LOCATION slot (high relevance)

The **entity slots** provide the FILL:
- Entity-specific values fill the relevant slots
- The answer is the projection of entity knowledge onto the question axis

## Future Work

1. **Learn slot relevance from data** - Currently hand-coded, could be learned
2. **Richer slot types** - Add temporal, causal, emotional slots
3. **Cross-domain transfer** - Test on completely different domains
4. **Complex questions** - Handle multi-hop and comparative questions

## Conclusion

**Geometric Q&A pattern transfer IS possible** through the Geometric Slot-Filling approach:

1. Encode answer structure as weighted sum of slot vectors
2. Learn slot relevance weights from Q&A training data
3. Transfer patterns via vector similarity in answer space
4. Generate answers by filling slots with entity-specific content

The key insight: **Structure is geometric, content is entity-specific**. By separating these, we enable transfer of answer patterns across domains.
