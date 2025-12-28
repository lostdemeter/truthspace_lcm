# Design Consideration 057: Domain Dimension as Zeta t-Coordinate

## Date: 2024-12-25

## Context

While implementing holographic projection for target selection in the GeometricLCM, we discovered that all protagonists (Holmes, Alice, Darcy, Hamlet) looked structurally identical - they all "resonate" with the same actions because they share the same role (actor with positive φ-direction).

This wasn't a bug - it was revealing a missing dimension: **domain/topic separation**.

## The Discovery

The holographic interference pattern showed:

```
'examined' resonates with: Holmes, Alice, Darcy, Hamlet
'watched' resonates with:  Holmes, Alice, Darcy, Hamlet
'killed' resonates with:   Holmes, Alice, Darcy, Hamlet
```

All protagonists look the same structurally because they ARE the same structurally - they're all entities that perform actions. The difference is **which story they belong to**.

## The Zeta Connection

This maps directly to the zeta function coordinates:

```
s = σ + it

σ (real part):      Distance from critical line - STRUCTURAL role
                    σ = 0.5 for all protagonists (they're all actors)

t (imaginary part): Frequency on critical line - DOMAIN/TOPIC
                    t = 0: Sherlock Holmes
                    t = 1: Alice in Wonderland
                    t = 2: Pride and Prejudice
                    t = 3: The Great Gatsby
                    t = 4: Hamlet
```

The critical line (σ = 0.5) represents **shared structure** - all protagonists share this. The t-coordinate represents **which domain** - separating topics while maintaining structural similarity.

## Implementation

### Domain Detection

Domains are detected by keyword matching:

```python
DOMAINS = {
    'sherlock': {
        'keywords': {'holmes', 'watson', 'detective', 'lestrade', 'moriarty'},
        'genre': 'mystery',
        't_value': 0.0,
    },
    'alice': {
        'keywords': {'alice', 'rabbit', 'queen', 'cheshire', 'hatter'},
        'genre': 'fantasy',
        't_value': 1.0,
    },
    # ... etc
}
```

### Domain-Aware Concepts

Each concept tracks its domain membership:

```python
@dataclass
class DomainConcept:
    word: str
    domains: Counter                    # Which domains it appears in
    actor_by_domain: Dict[str, int]     # Role counts per domain
    actions_by_domain: Dict[str, Counter]  # Actions per domain
    targets_by_domain: Dict[str, Counter]  # Targets per domain
```

### Query Types Enabled

1. **Domain-specific**: "Who is Holmes?" → Sherlock domain only
2. **Cross-domain**: "Who watches?" → Watson (mystery), Darcy (romance)
3. **Domain transfer**: "What would Alice do in a mystery?"
4. **Comparison**: "Compare Holmes and Hamlet"

## Experimental Results

```
Learned 197 concepts
Frames: 88
Cross-domain concepts: 14

Sentences by domain:
  sherlock: 24
  alice: 16
  pride: 18
  gatsby: 14
  hamlet: 16
```

### Cross-Domain Queries

```
Q: Who watches?
A: Across domains: Watson (mystery); Darcy (romance) all watches.

Q: Who kills?
A: Hamlet kills in the tragedy.

Q: Who loves?
A: Ophelia loves in the tragedy.
```

### Domain-Specific Queries

```
Q: Who is Holmes?
A: Holmes says, observes, deduces in the mystery genre.

Q: Who is Hamlet?
A: Hamlet kills in the tragedy genre.
```

## The Architecture

```
QUAD-QUATERNION + DOMAIN
════════════════════════

Q1 (Concept)     Q2 (Output)     Q3 (Morpho)     Q4 (Error)
────────────     ───────────     ───────────     ──────────
What fits?       How to say?     Word form?      What's wrong?
     │                │               │               │
     └────────────────┴───────────────┴───────────────┘
                              │
                      SHARED W-AXIS (σ = 0.5)
                              │
                      ┌───────┴───────┐
                      │               │
                  STRUCTURE       DOMAIN
                  (what role)     (which topic)
                      │               │
                      σ               t
                      │               │
                      └───────┬───────┘
                              │
                         s = σ + it
                    (ZETA COORDINATE)
```

## Why This Matters

### 1. Geometric Overlap Preserved

All protagonists share σ = 0.5 because they're structurally similar:
- They're all actors (positive φ-direction)
- They all perform actions
- They all have targets

This enables cross-domain queries: "Who investigates?" finds both Holmes AND Hamlet.

### 2. Topic Clarity Achieved

The t-coordinate separates domains:
- Holmes is at t = 0 (mystery)
- Hamlet is at t = 4 (tragedy)

This enables domain-specific queries: "Who in Sherlock Holmes?" only returns Holmes, Watson, etc.

### 3. Domain Transfer Possible

Because structure is shared (same σ), we can ask:
- "What would Alice do in a mystery?" → Find similar entity in mystery domain
- "Compare Holmes and Hamlet" → Both investigate, but in different genres

## Connection to Zeta Zeros

The zeta zeros are at s = 0.5 + it where t takes specific values. Each zero is a "resonant frequency" where structure naturally exists.

In our model:
- **σ = 0.5**: The critical line where all meaningful concepts live
- **t = domain**: The frequency that distinguishes topics
- **Zeros**: The specific t-values where domains "crystallize"

```
ZETA CRITICAL LINE
       │
       │    ●  t=0 (Sherlock)
       │    
       │    ●  t=1 (Alice)
       │    
       │    ●  t=2 (Pride)
       │    
       │    ●  t=3 (Gatsby)
       │    
       │    ●  t=4 (Hamlet)
       │
    σ=0.5
```

Each domain is like a zeta zero - a resonant frequency where a coherent narrative structure exists.

## Cross-Domain Concepts

14 concepts were automatically detected as cross-domain:

These are words that appear in multiple stories with similar roles:
- **watched**: Watson watches Holmes, Darcy watches Elizabeth
- **smiled**: Cheshire Cat smiles, Jane smiles
- **fell**: Alice fell, Bingley fell (in love)

These cross-domain concepts are the **bridges** between narratives - they share structure (σ) but span multiple domains (t).

## The Full Picture

```
                    DOMAIN (t-axis)
                         │
    Sherlock ─●─────────●─────────●─────────●─────────● Hamlet
              │         │         │         │         │
              │    Alice│    Pride│   Gatsby│         │
              │         │         │         │         │
              └─────────┴─────────┴─────────┴─────────┘
                              │
                        STRUCTURE (σ = 0.5)
                              │
                    All protagonists share this
                    (actors, positive φ-direction)
```

## Implementation Files

```
experiments/
├── domain_aware_lcm.py      # Domain-aware GeometricLCM
├── geometric_lcm_v3.py      # Holographic projection (revealed the need)
├── geometric_lcm_v2.py      # Symmetric ingestion
└── geometric_lcm.py         # Original quad-quaternion
```

## Usage Example

```python
from experiments.domain_aware_lcm import DomainAwareLCM

model = DomainAwareLCM()
model.ingest(corpus)

# Domain-specific query
model.set_domain('sherlock')
print(model.ask("Who is Holmes?"))
# → "Holmes says, observes, deduces in the mystery genre."

# Cross-domain query
model.set_domain(None)
print(model.ask("Who watches?"))
# → "Across domains: Watson (mystery); Darcy (romance) all watches."

# Domain transfer
print(model.ask("What would Alice do in a mystery?"))
# → "In a mystery, Alice would likely examine, similar to Holmes."
```

## Theoretical Implications

### 1. Topics as Frequencies

Different topics/domains are different frequencies on the critical line. This explains why:
- Similar topics have close t-values (Gatsby and Hamlet are both tragedies)
- Different topics have distant t-values (Sherlock and Alice are far apart)

### 2. Cross-Domain Reasoning

The shared σ-coordinate enables reasoning across domains:
- "All protagonists act" (shared structure)
- "Holmes investigates mysteries, Hamlet investigates murder" (different domains, same action)

### 3. Analytic Continuation

Moving between domains is like analytic continuation in the zeta function:
- Same function (narrative structure)
- Different regions (different t-values)
- Connected by the critical line (σ = 0.5)

## Future Directions

### 1. Automatic Domain Detection

Currently domains are predefined. Could detect them automatically by:
- Clustering concepts by co-occurrence
- Finding natural "gaps" in the t-coordinate
- Using zeta zero spacing as a guide

### 2. Domain Hierarchy

Domains could have sub-domains:
- Literature → Mystery → Sherlock Holmes
- Literature → Fantasy → Alice in Wonderland

This would create a tree structure on the t-axis.

### 3. Domain Blending

Generate text that blends domains:
- "Holmes in Wonderland" - mystery + fantasy
- "Hamlet's Pride" - tragedy + romance

This would involve interpolating between t-values.

## Conclusion

The domain dimension completes the geometric picture:

| Coordinate | Meaning | Example |
|------------|---------|---------|
| σ (real) | Structural role | σ = 0.5 for all protagonists |
| t (imaginary) | Domain/topic | t = 0 for Sherlock, t = 4 for Hamlet |
| φ-direction | Entity vs action | +1 for actors, -1 for verbs |
| W-axis | Certainty/balance | Definitive ↔ hedged |

The zeta coordinate s = σ + it captures both **what something is** (σ) and **where it belongs** (t).

```
"Structure is shared. Topics are frequencies.
 The critical line connects them all."
```
