# 094: Gear Serialization Architecture

## The Problem

We keep regressing to non-geometric methods because:

1. **JSON role is unclear** - Is it bootstrap data? Serialization? Source of truth?
2. **No standard serialization interface** - Each component does its own thing
3. **Gear vs GearChain distinction is blurry** - What's the actual difference?

## Analysis: Current State

### JSON Files Today

| Usage | Example | Role |
|-------|---------|------|
| Corpus files | `corpus/feature_labels.json` | Bootstrap data |
| Pattern files | `holographic_patterns.json` | Learned state |
| Knowledge store | `knowledge.json` | Serialized structure |
| Config files | Various | Configuration |

The problem: **no consistent interface**. Each component has its own `save()`/`load()`.

### Gear vs GearChain Today

```python
class Gear(ABC):
    """Single transformation unit."""
    def forward(self, state: GearState) -> GearState
    def backward(self, state: GearState) -> GearState
    # Has: name, ratio, quaternion, enabled, _knowledge_store

class GearChain:
    """Array of gears with sequential processing."""
    def process(self, state: GearState) -> Any
    def process_backward(self, state: GearState) -> GearState
    # Has: name, gears: List[Gear], _knowledge_store
```

**The distinction is minimal:**
- `Gear` = single transformation
- `GearChain` = `List[Gear]` with `process()` that calls `forward()` on each

A `GearChain` is essentially just an array of gears with a loop.

## Proposal: Unified Serialization

### The `Serializable` Protocol

Every gear and chain should implement a standard serialization interface:

```python
from typing import Protocol, Dict, Any
from pathlib import Path

class Serializable(Protocol):
    """Standard serialization interface for all gears and chains."""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state to dictionary.
        
        Returns a dict containing:
        - 'type': Class name for deserialization
        - 'version': Schema version
        - 'state': The actual state (positions, patterns, etc.)
        """
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Deserialize from dictionary."""
        ...
    
    def save(self, path: Path) -> None:
        """Save to JSON file."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> 'Serializable':
        """Load from JSON file."""
        ...
```

### What Gets Serialized

**The geometric structure IS the state.** JSON is just the serialization format.

```python
def to_dict(self) -> Dict[str, Any]:
    return {
        'type': self.__class__.__name__,
        'version': '1.0',
        'state': {
            # Geometric state
            'positions': [c.position for c in self.concepts],
            'words': [list(c.words) for c in self.concepts],
            
            # Metadata (for debugging/inspection only)
            'name': self.name,
            'dims': self.dims,
        }
    }
```

### The Serialization Contract

1. **JSON is serialization** - Not the source of truth, just a snapshot
2. **Structure is truth** - The geometric positions ARE the knowledge
3. **Round-trip guarantee** - `load(save(x)) == x` for geometric state
4. **Bootstrap vs Resume**:
   - Bootstrap: Create structure from scratch using JSON as seed
   - Resume: Restore exact structure from previous session

## Proposal: Simplify Gear/GearChain

### Option A: GearChain = List[Gear]

Make `GearChain` a simple type alias:

```python
GearChain = List[Gear]

def process_chain(chain: GearChain, state: GearState) -> GearState:
    for gear in chain:
        if gear.enabled:
            state = gear.forward(state)
    return state
```

**Pros:**
- Simpler mental model
- No special class needed
- Chains are just arrays

**Cons:**
- Lose chain-level knowledge store
- Lose chain-level methods (aggregate_knowledge, etc.)

### Option B: Gear Can Contain Gears

Make `Gear` recursive - a gear can contain sub-gears:

```python
class Gear(ABC):
    def __init__(self, name: str):
        self.name = name
        self.sub_gears: List[Gear] = []  # Optional sub-gears
        self._knowledge_store = None
    
    def forward(self, state: GearState) -> GearState:
        # Process through sub-gears first
        for sub in self.sub_gears:
            state = sub.forward(state)
        # Then apply this gear's transformation
        return self._transform(state)
    
    @abstractmethod
    def _transform(self, state: GearState) -> GearState:
        """The gear's own transformation."""
        pass
```

**Pros:**
- Unified model (everything is a Gear)
- Recursive composition
- Each gear can have its own knowledge store

**Cons:**
- More complex
- Might be over-engineering

### Option C: Keep Both, Clarify Roles (Recommended)

Keep the current structure but clarify:

```
Gear = Atomic transformation with optional knowledge store
GearChain = Composition of gears with aggregated knowledge

Gear implements: forward(), backward(), to_dict(), from_dict()
GearChain implements: process(), to_dict(), from_dict()
```

The key insight: **GearChain is a Gear that contains other Gears**.

```python
class GearChain(Gear):
    """A gear composed of other gears."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.gears: List[Gear] = []
    
    def forward(self, state: GearState) -> GearState:
        for gear in self.gears:
            if gear.enabled:
                state = gear.forward(state)
        return state
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'GearChain',
            'version': '1.0',
            'state': {
                'name': self.name,
                'gears': [g.to_dict() for g in self.gears],
                'knowledge_store': self._knowledge_store.to_dict() if self._knowledge_store else None,
            }
        }
```

## Implementation Plan

### Phase 1: Add Serializable Protocol

```python
# truthspace_lcm/core/serialization.py

from typing import Protocol, Dict, Any, TypeVar
from pathlib import Path
import json

T = TypeVar('T', bound='Serializable')

class Serializable(Protocol):
    def to_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T: ...

def save(obj: Serializable, path: Path) -> None:
    """Save any serializable object to JSON."""
    data = obj.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load(path: Path) -> Any:
    """Load a serializable object from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Dispatch to correct class based on 'type' field
    type_name = data.get('type')
    cls = get_class_for_type(type_name)
    return cls.from_dict(data)
```

### Phase 2: Update Gear Base Class

```python
class Gear(ABC, Serializable):
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'version': '1.0',
            'state': {
                'name': self.name,
                'ratio': self.ratio,
                'quaternion': [self.quaternion.w, self.quaternion.x, 
                               self.quaternion.y, self.quaternion.z],
                'enabled': self.enabled,
                'knowledge_store': self._knowledge_store.to_dict() if self._knowledge_store else None,
            }
        }
```

### Phase 3: Make GearChain a Gear

```python
class GearChain(Gear):
    """A gear composed of other gears."""
    
    def __init__(self, name: str = "GearChain"):
        super().__init__(name)
        self.gears: List[Gear] = []
    
    def forward(self, state: GearState) -> GearState:
        for gear in self.gears:
            if gear.enabled:
                state = gear.forward(state)
        return state
```

## The Key Insight

**JSON is to Structure as Source Code is to Running Program.**

- JSON defines the structure (like source code defines a program)
- Structure is the runtime object (like a running program)
- Serialization saves the structure (like saving program state)
- Deserialization restores the structure (like resuming from checkpoint)

We don't "run" JSON. We **instantiate** structure from JSON, then work with the structure.

## Summary

1. **Add `Serializable` protocol** - Standard interface for all gears
2. **Make `GearChain` extend `Gear`** - Unified model, chains are just composite gears
3. **Clarify JSON role** - Serialization format, not source of truth
4. **Centralized save/load** - One place for all serialization logic

This prevents regression to non-geometric methods by making the geometric structure
the explicit, serializable state of every gear.
