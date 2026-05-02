# 078: Gear Message Protocol

## Overview

This document defines the standardized communication protocol for gears in the TruthSpace-LCM system. The goal is to enable clean inter-gear communication without complex feedback loops.

## The Problem

Before standardization, gears communicated inconsistently:
- `base.py` gears use `GearState` with `forward(state: GearState)`
- `gear_improvement_loop.py` uses `Dict[str, Any]` with `process(input_data)`
- `bootstrap_gear.py` uses raw `str` with `process(input_text: str)`
- Various ad-hoc patterns with duck-typing (`hasattr(gear, 'chat')`)

This made chaining gears fragile and required constant type checking.

## The Solution: GearMessage

A universal message envelope that all gears understand:

```python
@dataclass
class GearMessage:
    content: str                    # The main payload
    source: str                     # Which gear sent this
    intent: MessageIntent           # What the sender wants
    context: Dict[str, Any]         # Additional context
    history: List[str]              # Chain of gears that processed this
    confidence: float               # How confident (0.0-1.0)
    errors: List[str]               # Any errors encountered
```

## GearProtocol

The standard interface all gears should implement:

```python
class GearProtocol:
    name: str
    
    def receive(self, data) -> GearMessage:
        """Normalize any input to GearMessage."""
        
    def process_message(self, message: GearMessage) -> GearMessage:
        """Core gear logic. Override in subclasses."""
        
    def send(self, message: GearMessage, content: str = None) -> GearMessage:
        """Forward message to next gear."""
        
    def process(self, data) -> Dict[str, Any]:
        """Compatibility wrapper returning dict."""
        
    def chat(self, text: str) -> str:
        """Convenience wrapper for string I/O."""
        
    def __call__(self, data) -> GearMessage:
        """Allow gear to be called directly."""
```

## Usage Examples

### Creating a Gear

```python
from truthspace_lcm.core import GearProtocol, GearMessage

class MyGear(GearProtocol):
    def __init__(self):
        self.name = "MyGear"
    
    def process_message(self, message: GearMessage) -> GearMessage:
        # Do something with message.content
        result = message.content.upper()
        return self.send(message, result)
```

### Chaining Gears

```python
msg = GearMessage.from_string("hello world", "user")
msg = gear1(msg)  # Uses __call__
msg = gear2(msg)
msg = gear3(msg)

print(msg.content)  # Final result
print(msg.history)  # ['user', 'gear1', 'gear2', 'gear3']
```

### Backward Compatibility

Gears can still accept any input format:

```python
gear.chat("hello")              # String in, string out
gear.process({"input": "hi"})   # Dict in, dict out
gear(message)                   # GearMessage in, GearMessage out
```

## Bridging with GearState

For gears that use the formal `GearState` protocol (with quaternions, etc.):

```python
from truthspace_lcm.core import adapt_to_gear_state, adapt_from_gear_state

# GearMessage → GearState
state = adapt_to_gear_state(message)

# GearState → GearMessage  
message = adapt_from_gear_state(state, source="MyGear")
```

## Emergent Intent Space

Instead of hardcoded intents, the system discovers intents from usage:

```python
from truthspace_lcm.core import EmergentIntentSpace

space = EmergentIntentSpace()

# Detect intent
intent, confidence = space.detect("Who is Ahab?")  # ('query', 1.0)

# Record observations
space.observe("deploy to prod", "query", "DeployGear", success=True)

# Discover new intents from low-confidence patterns
new_intent = space.discover_new_intent(min_observations=5)
# → "deploy" (emergent=True)
```

## Message Flow

```
User Input
    ↓
GearMessage.from_string("Who is Ahab?", "user")
    ↓
IntentDetector.receive() → normalize to GearMessage
IntentDetector.process_message() → detect intent, set message.intent
IntentDetector.send() → forward with history
    ↓
Router.receive() → GearMessage with intent
Router.process_message() → route based on intent
Router.send() → forward to appropriate gear
    ↓
ConversationalChain.receive() → GearMessage
ConversationalChain.process_message() → generate response
ConversationalChain.send() → final result
    ↓
GearMessage(
    content="Ahab is the captain of the Pequod...",
    history=["user", "IntentDetector", "Router", "ConversationalChain"],
    confidence=0.85
)
```

## Key Principles

1. **Common Language** - All gears speak GearMessage
2. **History Tracking** - Every gear adds itself to the history
3. **Context Preservation** - Original input survives the chain
4. **Intent Awareness** - Gears know what the sender wants
5. **Backward Compatible** - Works with existing dict/string interfaces
6. **Emergent Intents** - System discovers what intents work best

## Files

- `/truthspace_lcm/gears/core/gear_message.py` - Core protocol
- `/truthspace_lcm/gears/core/__init__.py` - Exports

## Migration Guide

To update an existing gear:

1. Inherit from `GearProtocol` (or implement the interface)
2. Set `self.name` in `__init__`
3. Implement `process_message(self, message: GearMessage) -> GearMessage`
4. Use `self.send(message, new_content)` to forward
5. Existing `process()` and `chat()` methods work automatically

```python
# Before
class OldGear:
    def process(self, data):
        text = data.get('input', '')
        return {'output': text.upper()}

# After
class NewGear(GearProtocol):
    def __init__(self):
        self.name = "NewGear"
    
    def process_message(self, message):
        return self.send(message, message.content.upper())
```

The new gear still works with `process()` and `chat()` for backward compatibility.
