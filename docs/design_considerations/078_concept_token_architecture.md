# Concept Token Architecture

## The Fundamental Shift

Instead of frames being primary and concepts being implicit, we make **concept tokens** the atomic unit of our system.

## Why "Tokens"?

The term "token" is already familiar in computational contexts:
- **Lexical tokens**: The output of a tokenizer
- **Authentication tokens**: A representation that grants access
- **Game tokens**: A piece that represents something else

A **concept token** is similar—it's a computational unit that **represents** a concept. The token isn't the concept itself, just as a chess piece isn't literally a king. But it carries the concept's properties and can be manipulated computationally.

## What is a Concept Token?

A concept token is the smallest meaningful unit in our geometric space. It has:

1. **Position** - A quaternion coordinate in concept space
2. **Surface Forms** - One or more text representations
3. **Activation Properties** - How easily it's triggered by context

```
ConceptToken:
  position: Quaternion(w, x, y, z)    # WHERE it lives in concept space
  forms: ["king", "monarch", "ruler"] # HOW it appears in text
  tension: float                      # HOW MUCH context needed to activate
```

## The Key Insight: Position IS Identity

In traditional systems, a word's identity is its string: "king" ≠ "monarch".

In our system, a token's identity is its **position**: 
- "king" and "monarch" can be the SAME token (same position, different forms)
- "bank" (river) and "bank" (financial) are DIFFERENT tokens (different positions, same form)

This is the fundamental inversion.

## Atomizing Concepts

The key question: **What are the "atoms" of meaning?**

### Option A: Semantic Primitives

Break concepts into orthogonal dimensions:

```
"king" = HUMAN + MALE + ADULT + HIGH_AGENCY + AUTHORITY

Each primitive is a concept token:
  HUMAN:       q = (1, 0, 0, 1)     # w=animacy, z=agency
  MALE:        q = (0, -1, 0, 0)    # x=gender
  ADULT:       q = (0, 0, -1, 0)    # y=age
  HIGH_AGENCY: q = (0, 0, 0, 1)     # z=agency
  AUTHORITY:   q = (1, 0, 0, 0.5)   # compound
```

Composite concepts are **combinations** of tokens:
```
king = HUMAN ⊗ MALE ⊗ ADULT ⊗ HIGH_AGENCY ⊗ AUTHORITY
     = quaternion_compose([...])
```

### Option B: Emergent Positions

Don't predefine primitives—let positions emerge from usage:

```
Observe: "king" appears in contexts with:
  - high agency verbs (rules, commands, decrees)
  - male pronouns (he, his)
  - authority targets (kingdom, subjects)

Infer position from statistical patterns:
  king.position = average_context_quaternion(all_king_frames)
```

### Option C: Hybrid (Recommended)

Some tokens are **primitive** (the axes), others are **composite** (combinations):

```
Primitive Tokens (the basis vectors):
  GENDER_MALE:   q = (0, -1, 0, 0)
  GENDER_FEMALE: q = (0, +1, 0, 0)
  AGE_ADULT:     q = (0, 0, -1, 0)
  AGE_YOUNG:     q = (0, 0, +1, 0)
  AGENCY_HIGH:   q = (0, 0, 0, +1)
  AGENCY_LOW:    q = (0, 0, 0, -1)
  ANIMACY_HUMAN: q = (+1, 0, 0, 0)
  ANIMACY_PLACE: q = (-1, 0, 0, 0)

Composite Tokens (combinations):
  king = compose(ANIMACY_HUMAN, GENDER_MALE, AGE_ADULT, AGENCY_HIGH)
       = Quaternion(1, -1, -1, 1)  # normalized
```

## The Token Registry

A central registry of all concept tokens:

```python
class ConceptToken:
    """The atomic unit of meaning."""
    id: str                      # Unique identifier
    position: Quaternion         # Location in concept space
    forms: List[str]             # Surface text forms
    is_primitive: bool           # True if basis vector
    tension: float               # Activation threshold
    
    # Relationships (inspired by DNA regulation, but computational terms)
    activators: List[str]        # Tokens that boost this one
    inhibitors: List[str]        # Tokens that suppress this one


class TokenRegistry:
    """The complete vocabulary of concept tokens."""
    primitives: Dict[str, ConceptToken]   # Basis vectors
    composites: Dict[str, ConceptToken]   # Derived tokens
    
    def register(self, token: ConceptToken) -> None
    def lookup_by_form(self, text: str) -> Optional[ConceptToken]
    def lookup_by_position(self, q: Quaternion, tolerance: float) -> List[ConceptToken]
    def compose(self, *tokens: ConceptToken) -> ConceptToken
    def decompose(self, token: ConceptToken) -> List[ConceptToken]
```

## Encoding and Decoding

### Encode: Text → Tokens

```python
def encode(text: str) -> List[ConceptToken]:
    """Convert text to concept tokens."""
    words = tokenize(text)
    tokens = []
    for word in words:
        token = registry.lookup_by_form(word)
        if token:
            tokens.append(token)
        else:
            # Unknown word - infer position from context
            token = infer_token(word, context=tokens)
            tokens.append(token)
    return tokens
```

### Decode: Tokens → Text

```python
def decode(tokens: List[ConceptToken], style: str = "default") -> str:
    """Convert concept tokens back to text."""
    words = []
    for token in tokens:
        # Choose surface form based on style and context
        form = select_form(token, style, context=words)
        words.append(form)
    return assemble(words)
```

### The Duality

```
TEXT ──encode──► TOKENS ──decode──► TEXT'

The same tokens can decode to different text:
  [KING_TOKEN] + style="formal" → "His Majesty the King"
  [KING_TOKEN] + style="casual" → "the king"
  [KING_TOKEN] + style="archaic" → "the sovereign"

Different text can encode to same tokens:
  "king" ──encode──► [KING_TOKEN]
  "monarch" ──encode──► [KING_TOKEN]
  "ruler" ──encode──► [KING_TOKEN]
```

## Frames as Token Sequences

With concept tokens as fundamental, frames become **sequences of tokens**:

```python
class Frame:
    """A structured sequence of concept tokens."""
    entity: ConceptToken         # WHO/WHAT
    role: ConceptToken           # CLASSIFICATION
    actions: List[ConceptToken]  # VERBS
    targets: List[ConceptToken]  # OBJECTS
    
    def to_tokens(self) -> List[ConceptToken]:
        """Flatten to token sequence."""
        return [self.entity, self.role] + self.actions + self.targets
    
    @classmethod
    def from_tokens(cls, tokens: List[ConceptToken]) -> 'Frame':
        """Parse token sequence into frame structure."""
        # Use position/role to determine structure
        pass
```

## Token Composition

How do tokens combine? Several options:

### Quaternion Multiplication
```python
def compose_multiply(a: ConceptToken, b: ConceptToken) -> Quaternion:
    """Compose via quaternion multiplication (rotation)."""
    return a.position * b.position
```

### Quaternion Addition (Averaging)
```python
def compose_add(tokens: List[ConceptToken]) -> Quaternion:
    """Compose via averaging (centroid)."""
    total = Quaternion(0, 0, 0, 0)
    for t in tokens:
        total = total + t.position
    return total.normalize()
```

### Weighted Composition
```python
def compose_weighted(tokens: List[ConceptToken], weights: List[float]) -> Quaternion:
    """Compose with importance weights."""
    total = Quaternion(0, 0, 0, 0)
    for t, w in zip(tokens, weights):
        total = total + (t.position * w)
    return total.normalize()
```

## Activation and Context

### Token Tension

Each token has a "tension" value—how much context is needed to activate it:

```python
def compute_tension(token: ConceptToken) -> float:
    """
    Low tension = common, easily activated (like "the", "is")
    High tension = rare, needs strong context (like "defenestration")
    """
    frequency = count_occurrences(token)
    specificity = token.position.magnitude()  # How far from origin
    
    return specificity / (1 + log(1 + frequency))
```

### Activation Spread

When a token activates, it can spread activation to related tokens:

```python
def spread_activation(active: ConceptToken, registry: TokenRegistry) -> Dict[ConceptToken, float]:
    """Spread activation to nearby tokens in concept space."""
    activations = {active: 1.0}
    
    for token in registry.all_tokens():
        if token == active:
            continue
        
        # Activation decays with distance
        distance = active.position.distance(token.position)
        activation = exp(-distance)
        
        # Boost from explicit activators
        if active.id in token.activators:
            activation *= 2.0
        
        # Suppress from explicit inhibitors
        if active.id in token.inhibitors:
            activation *= 0.1
        
        if activation > 0.01:  # Threshold
            activations[token] = activation
    
    return activations
```

## Corpus as Token Graph

The corpus becomes a graph of token relationships:

```
Nodes: ConceptTokens
Edges: Co-occurrence in frames

     HOLMES ──investigates──► CRIME
        │                       │
     works_with              involves
        │                       │
        ▼                       ▼
     WATSON                  VICTIM
```

This enables:
- **Path finding**: How are two concepts related?
- **Clustering**: What concepts form natural groups?
- **Analogy**: A:B :: C:? becomes graph traversal

## Migration Path

How to get from current system to concept tokens:

### Phase 1: Extract Tokens from Existing Corpus
```python
def extract_tokens(corpus: Dict) -> TokenRegistry:
    """Build token registry from existing frames."""
    registry = TokenRegistry()
    
    # Add primitives (the axes)
    registry.add_primitive("MALE", Quaternion(0, -1, 0, 0))
    registry.add_primitive("FEMALE", Quaternion(0, 1, 0, 0))
    # ... etc
    
    # Extract composites from frames
    for frame in corpus["frames"]:
        entity = frame["entity"]
        if not registry.has(entity):
            # Infer position from frame context
            position = infer_position(frame)
            token = ConceptToken(entity, position, forms=[entity])
            registry.add_composite(token)
    
    return registry
```

### Phase 2: Dual Representation
Run both systems in parallel:
- Old: Frame-based corpus
- New: Token registry

Verify they produce equivalent results.

### Phase 3: Token-First Operations
New operations work on tokens directly:
- Query → Encode → Token matching → Decode → Response

### Phase 4: Deprecate Frame-First
Frames become a view over tokens, not primary storage.

## Open Questions

1. **Granularity**: Is "king" one token or should it decompose to primitives?
   - Tradeoff: Fewer tokens = simpler, more tokens = more expressive

2. **Unknown tokens**: What happens when we encounter a word not in registry?
   - Option A: Infer position from context
   - Option B: Create new token at inferred position
   - Option C: Map to nearest existing token

3. **Token evolution**: Should token positions drift based on new data?
   - DNA analogy: Mutations are usually bad, but sometimes beneficial
   - Maybe: Allow drift within tolerance, flag large changes for review

4. **Composition operator**: Multiply vs add vs something else?
   - Multiply: Rotation (order matters)
   - Add: Averaging (order doesn't matter)
   - Maybe: Different operators for different relationships

## Summary

**Concept tokens** are the atomic units of meaning in our system:
- Each has a **position** (quaternion) and **surface forms** (text)
- **Primitives** are the basis vectors (gender, age, agency, animacy)
- **Composites** are combinations of primitives
- **Frames** become sequences of tokens
- **Encoding** maps text to tokens
- **Decoding** maps tokens to text (style-dependent)
- **Activation** spreads through the token graph based on context

This gives us:
- Explicit concept representation (not implicit in frames)
- Geometric queries (find by position, not just text)
- Bidirectional translation (text ↔ tokens)
- Context-sensitive activation (tension, activators, inhibitors)

## Integration with Gear Chain

The gear chain system processes `GearState` objects. With concept tokens, we have two integration points:

### Option 1: Token-Aware GearState

```python
@dataclass
class TokenAwareGearState(GearState):
    """GearState that carries concept tokens alongside text."""
    entity_token: Optional[ConceptToken] = None
    role_token: Optional[ConceptToken] = None
    action_tokens: List[ConceptToken] = field(default_factory=list)
    target_tokens: List[ConceptToken] = field(default_factory=list)
    
    # The text fields (entity, role, actions, targets) become
    # derived from tokens via decode()
```

### Option 2: Token Gears

New gears that operate on tokens:

```python
class TokenEncoderGear(Gear):
    """Converts text fields to concept tokens."""
    def forward(self, state: GearState) -> GearState:
        state.entity_token = self.registry.encode(state.entity)
        state.action_tokens = [self.registry.encode(a) for a in state.actions]
        return state

class TokenDecoderGear(Gear):
    """Converts concept tokens back to text."""
    def forward(self, state: GearState) -> GearState:
        state.entity = self.registry.decode(state.entity_token, self.style)
        state.actions = [self.registry.decode(t, self.style) for t in state.action_tokens]
        return state

class TokenTransformGear(Gear):
    """Operates directly on token positions."""
    def forward(self, state: GearState) -> GearState:
        # Example: shift all tokens toward higher agency
        for token in state.action_tokens:
            token.position.z += 0.1 * self.ratio
        return state
```

### The Pipeline

```
Text Input
    │
    ▼
┌─────────────────┐
│ TokenEncoderGear│  text → tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RoleGear      │  operates on tokens
│   ActionGear    │  (position-aware)
│   TenseGear     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TokenDecoderGear│  tokens → text
└────────┬────────┘
         │
         ▼
    Text Output
```

## Practical First Step: ConceptToken Prototype

Let's start with a minimal implementation to test the ideas:

```python
# truthspace_lcm/gears/core/concept_token.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .base import Quaternion

@dataclass
class ConceptToken:
    """A concept represented by its position in quaternion space."""
    id: str
    position: Quaternion
    forms: List[str] = field(default_factory=list)
    is_primitive: bool = False
    tension: float = 0.5
    
    # Context sensitivity
    activators: List[str] = field(default_factory=list)
    inhibitors: List[str] = field(default_factory=list)
    
    def distance_to(self, other: 'ConceptToken') -> float:
        """Euclidean distance in quaternion space."""
        dw = self.position.w - other.position.w
        dx = self.position.x - other.position.x
        dy = self.position.y - other.position.y
        dz = self.position.z - other.position.z
        return (dw**2 + dx**2 + dy**2 + dz**2) ** 0.5
    
    def primary_form(self) -> str:
        """The canonical text representation."""
        return self.forms[0] if self.forms else self.id


class TokenRegistry:
    """Registry of all known concept tokens."""
    
    def __init__(self):
        self.tokens: Dict[str, ConceptToken] = {}
        self.form_index: Dict[str, str] = {}  # form → token_id
        self._init_primitives()
    
    def _init_primitives(self):
        """Initialize the basis vector tokens."""
        primitives = [
            ("GENDER_MALE", Quaternion(0, -1, 0, 0), ["male", "man", "he"]),
            ("GENDER_FEMALE", Quaternion(0, 1, 0, 0), ["female", "woman", "she"]),
            ("AGE_ADULT", Quaternion(0, 0, -1, 0), ["adult", "mature"]),
            ("AGE_YOUNG", Quaternion(0, 0, 1, 0), ["young", "child", "youth"]),
            ("AGENCY_HIGH", Quaternion(0, 0, 0, 1), ["active", "initiator"]),
            ("AGENCY_LOW", Quaternion(0, 0, 0, -1), ["passive", "receiver"]),
            ("ANIMACY_HUMAN", Quaternion(1, 0, 0, 0), ["human", "person"]),
            ("ANIMACY_PLACE", Quaternion(-1, 0, 0, 0), ["place", "location"]),
        ]
        for id, pos, forms in primitives:
            self.register(ConceptToken(id, pos, forms, is_primitive=True))
    
    def register(self, token: ConceptToken) -> None:
        """Add a token to the registry."""
        self.tokens[token.id] = token
        for form in token.forms:
            self.form_index[form.lower()] = token.id
    
    def lookup_by_form(self, text: str) -> Optional[ConceptToken]:
        """Find token by surface form."""
        token_id = self.form_index.get(text.lower())
        return self.tokens.get(token_id) if token_id else None
    
    def lookup_by_position(self, q: Quaternion, tolerance: float = 0.5) -> List[ConceptToken]:
        """Find tokens near a quaternion position."""
        results = []
        for token in self.tokens.values():
            dist = ((token.position.w - q.w)**2 + 
                    (token.position.x - q.x)**2 +
                    (token.position.y - q.y)**2 +
                    (token.position.z - q.z)**2) ** 0.5
            if dist <= tolerance:
                results.append((token, dist))
        return [t for t, d in sorted(results, key=lambda x: x[1])]
    
    def encode(self, text: str) -> Optional[ConceptToken]:
        """Convert text to concept token."""
        return self.lookup_by_form(text)
    
    def decode(self, token: ConceptToken, style: str = "default") -> str:
        """Convert concept token to text."""
        # For now, just return primary form
        # Later: style-aware form selection
        return token.primary_form()
```

## What This Enables

1. **Geometric Queries**: "Find all tokens with high agency" → filter by z > 0.5
2. **Synonym Resolution**: "king" and "monarch" → same token
3. **Polysemy Handling**: "bank" → different tokens based on context
4. **Analogical Reasoning**: king:queen :: man:? → find token at man + (queen - king)
5. **Style Variation**: Same token → different surface forms based on style

## Next Steps

1. **Prototype `ConceptToken` and `TokenRegistry`** in `gears/core/`
2. **Extract tokens from existing corpus** - build registry from frames
3. **Add `TokenEncoderGear` and `TokenDecoderGear`** to pipeline
4. **Test encode/decode roundtrip** - verify text → tokens → text works
5. **Implement geometric queries** - find by position, not just form

---

*"Tokens are positions. Positions are meaning. Text is just one projection."*
