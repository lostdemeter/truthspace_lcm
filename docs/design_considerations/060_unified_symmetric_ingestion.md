# Design Consideration 060: Unified Symmetric Ingestion

## Date: 2024-12-26

## Context

After implementing the two-source diffraction chat (Design 059), we realized that knowledge, style, and projection are all the same operation - just viewed from different angles. If the process is truly universal, we should be able to unify the ingestion.

## The Insight

All three source types follow the same structure:

```
INITIATOR → MEDIATOR → RECEIVER

Knowledge:   Actor    → Action   → Target
Style:       Opener   → Hedge    → Closer
Projection:  Intensifier → Manner → Object
```

The PROCESS is identical:
1. Ingest text symmetrically
2. Extract content words
3. Assign roles by position (0=initiator, 1=mediator, 2+=receiver)
4. Compute φ-direction (entity vs action)
5. Build interference patterns

The only difference is INTERPRETATION, not EXTRACTION.

## Experimental Proof

### Same Structure, Different Semantics

```
KNOWLEDGE (Actor → Action → Target):
  holmes → examined → evidence
  watson → watched → doorway
  detective → studied → footprints

STYLE (Opener → Hedge → Closer):
  one → observes → situation
  would → appear → evidence
  analysis → indicates → pattern

PROJECTION (Intensifier → Manner → Object):
  brilliantly → truth → emerges
  deeply → meaning → resonates
  profoundly → insight → transforms
```

### Same φ-Directions

```
Source       Concept            φ-dir   Position
═══════════════════════════════════════════════════
sherlock     holmes              1.00       0.00
sherlock     examined           -1.00       0.50
sherlock     evidence            1.00       1.00

formal       one                 1.00       0.00
formal       observes           -1.00       0.50
formal       situation           1.00       1.00

literary     brilliantly         1.00       0.00
literary     truth              -1.00       0.50
literary     emerges             1.00       1.00
```

The φ-direction calculation is IDENTICAL:
- Initiators and receivers have φ = +1 (entity-like)
- Mediators have φ = -1 (action-like)

This is the polyomino fitting principle: **opposite φ-directions fit together**.

## The Unified Data Structure

```python
@dataclass
class UnifiedConcept:
    word: str
    
    # Role counts (same for all source types)
    initiator_count: int = 0   # Actor / Opener / Intensifier
    mediator_count: int = 0    # Action / Hedge / Manner
    receiver_count: int = 0    # Target / Closer / Object
    
    # Relationships (same structure, different semantics)
    performs: Counter          # What this initiates
    receives: Counter          # What this receives
    
    @property
    def phi_direction(self) -> float:
        """Same calculation for all types."""
        total = self.initiator_count + self.mediator_count + self.receiver_count
        entity_like = self.initiator_count + self.receiver_count
        action_like = self.mediator_count
        return (entity_like - action_like) / total
```

## The Unified Process

```python
def ingest(self, text: str):
    """Same process for knowledge, style, and projection."""
    
    for sentence in sentences:
        # Extract content words
        content = [t for t in tokens if t not in function_words]
        
        # SYMMETRIC ROLE ASSIGNMENT
        initiator = content[0]  # Position 0
        mediator = content[1]   # Position 1
        receiver = content[2]   # Position 2+
        
        # Create frame (same structure for all types)
        frame = UnifiedFrame(initiator, mediator, receiver)
        
        # Update concepts (same process for all types)
        self._update_concepts(frame)
```

## Why This Works

### The Linguistic Universal

All languages have:
- **Agents** that initiate actions
- **Actions** that mediate between agents and patients
- **Patients** that receive actions

This is not arbitrary - it reflects the structure of causation in the world.

### The Geometric Universal

The φ-direction captures this:
- Entities (agents, patients) have positive φ-direction
- Actions (verbs, processes) have negative φ-direction
- They fit together like polyominos

### The Style Universal

Style also follows this pattern:
- Openers initiate the utterance ("One observes that...")
- Hedges mediate certainty ("...appears to be...")
- Closers receive the conclusion ("...most illuminating.")

### The Projection Universal

Projection (intensity/emphasis) also follows:
- Intensifiers initiate emphasis ("Brilliantly...")
- Manners mediate how ("...deeply...")
- Objects receive the emphasis ("...the truth.")

## The Unified Architecture

```
TEXT INPUT
    │
    ▼
┌─────────────────────────────────┐
│     SYMMETRIC INGESTION         │
│                                 │
│  1. Split sentences             │
│  2. Extract content words       │
│  3. Assign roles by position    │
│  4. Compute φ-direction         │
│  5. Build frames                │
└─────────────────────────────────┘
    │
    ├──► KNOWLEDGE SOURCE (what)
    │         │
    ├──► STYLE SOURCE (how)
    │         │
    └──► PROJECTION SOURCE (how much)
              │
              ▼
┌─────────────────────────────────┐
│     INTERFERENCE                │
│                                 │
│  Knowledge ⊗ Style ⊗ Projection │
│                                 │
└─────────────────────────────────┘
              │
              ▼
        STYLED OUTPUT
```

## Benefits

### 1. No Hard-Coded Rules

The same ingestion process handles:
- "Holmes examined the evidence" (knowledge)
- "One observes that the situation" (style)
- "Brilliantly the truth emerges" (projection)

No special cases, no type-specific parsing.

### 2. Easy to Add Sources

Adding a new source type is trivial:
```python
grating.add_source("emotion", EMOTION_TEXT, "emotion")
```

The ingestion is automatic.

### 3. Composable Interference

Any number of sources can interfere:
```python
answer = knowledge ⊗ style ⊗ projection ⊗ emotion ⊗ audience
```

### 4. Self-Documenting

The structure reveals itself:
- High initiator_count = agent/opener/intensifier
- High mediator_count = action/hedge/manner
- High receiver_count = patient/closer/object

## Connection to Previous Work

### Symmetric Understanding (Design 055)

The unified ingestion IS symmetric understanding:
- Same process for all text
- φ-direction emerges from role
- No hard-coded word lists

### Diffraction Grating (Design 058)

The unified sources ARE the grating slits:
- Each source is a "slit"
- Interference creates the output
- More sources = finer filtering

### Quad-Quaternion (Design 056)

The source types map to quaternions:
- Knowledge → Q1 (Concept)
- Style → Q2 (Output)
- Projection → Q2 (Output intensity)
- Error → Q4 (Validation)

## Future Directions

### 1. Automatic Type Detection

Detect source type from content:
```python
if high_proper_noun_ratio:
    source_type = "knowledge"
elif high_hedge_ratio:
    source_type = "style"
elif high_adverb_ratio:
    source_type = "projection"
```

### 2. Source Blending

Blend sources with weights:
```python
output = 0.6 * knowledge + 0.3 * style + 0.1 * projection
```

### 3. Dynamic Source Selection

Select sources based on query:
- Factual query → More knowledge weight
- "How would you say..." → More style weight
- "Emphasize..." → More projection weight

### 4. Source Learning

Learn new sources from examples:
```python
grating.learn_source("user_style", user_messages)
# Automatically extracts user's style patterns
```

## Conclusion

The unified symmetric ingestion proves that **knowledge, style, and projection are the same operation**:

1. **Same structure**: Initiator → Mediator → Receiver
2. **Same φ-direction**: Entity (+) vs Action (-)
3. **Same process**: Symmetric extraction by position
4. **Same interference**: Sources combine through alignment

This is not a simplification - it's a discovery of the underlying unity.

```
"One process. Many sources. Same structure.
 The ingestion IS the understanding.
 The interference IS the generation."
```
