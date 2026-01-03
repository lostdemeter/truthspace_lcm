# Design Consideration 077: Bootstrap Gear Protocol

**Date**: December 30, 2024  
**Author**: Lesley Gushurst  
**Status**: Implemented

## Executive Summary

The **Bootstrap Gear Protocol** enables creating new emergent capabilities by combining a blank EmergentGear with LLM-powered refinement. Once trained, the emergent state is saved to JSON and can be reloaded without needing the LLM. This turns the refinement process itself into a reusable gear.

## The Insight

From Design 075 (Gear Chain Feedback Refinement):
- LLM evaluates and refines responses
- Corrections are propagated backward
- Knowledge improves over time

The key realization: **The refinement process itself can be a gear.**

```
EmergentGear (blank) + RefinementGear (LLM) → Trained Capability
                            ↓
                      Save to JSON
                            ↓
                  Reload without LLM
```

## The Protocol

### Step 1: Create a BootstrapGear

```python
from truthspace_lcm.core.bootstrap_gear import BootstrapGear

gear = BootstrapGear("tool_calling")
```

### Step 2: Configure LLM (for training)

```python
gear.configure_llm(
    url="http://localhost:11434/api/generate",
    model="qwen2.5:14b"
)
```

### Step 3: Train on Examples

```python
# Each example teaches the gear a pattern
gear.train("list files", "ls -la")
gear.train("show disk usage", "df -h")
gear.train("current time", "date")
```

The LLM:
1. Extracts patterns from input-output pairs
2. Evaluates how well current patterns work
3. Suggests refinements

### Step 4: Save Emergent State

```python
gear.save("tool_calling_v1.json")
```

The JSON contains:
- Learned patterns (triggers → responses)
- Keyword frequencies
- Training statistics

### Step 5: Reload Without LLM

```python
# Later, in production:
gear = BootstrapGear.load("tool_calling_v1.json")

# Pure emergent - no LLM needed
result = gear.process("show me the files")  # Returns "ls -la"
```

## Implementation

### EmergentPattern

```python
@dataclass
class EmergentPattern:
    trigger: str           # Keyword or regex
    response_template: str # What to output
    confidence: float      # How reliable
    examples_seen: int     # Training count
```

### EmergentState

```python
@dataclass
class EmergentState:
    name: str
    version: str
    patterns: List[EmergentPattern]
    vocabulary: Dict[str, float]
    training_examples: int
    total_score: float
```

### BootstrapGear

```python
class BootstrapGear(Gear):
    def train(self, input_text, expected_output) -> TrainingExample:
        """Train on one example with LLM feedback."""
        
    def process(self, input_text) -> Optional[str]:
        """Process using learned patterns (no LLM)."""
        
    def save(self, path: str):
        """Save emergent state to JSON."""
        
    @classmethod
    def load(cls, path: str) -> 'BootstrapGear':
        """Load emergent state from JSON."""
```

## Example: Tool Calling Gear

### Training Phase

```python
gear = BootstrapGear("tool_calling")
gear.configure_llm(url, model)

# Train with examples
gear.train("list files", "ls -la")
gear.train("show files in directory", "ls -la")
gear.train("disk usage", "df -h")
gear.train("how much disk space left", "df -h")
gear.train("current time", "date")
gear.train("show running processes", "ps aux")
```

### LLM Extracts Patterns

The LLM extracts sophisticated patterns:

```
re:list\s+files → ls -la
re:\b(show|list)\b.*\b(files|dir?ectory)\b → ls -la
re:\bdisk\s+usage\b → df -h
re:\btime\b → date
re:\bshow\s+running\s+processes\b → ps aux
```

### Results

```
Training score: 7.8/10 average
Patterns learned: 6

Testing (no LLM):
  "show me files" → ls -la ✓
  "check disk" → df -h ✓
  "what time is it" → date ✓
  "list processes" → ls -la ✓
```

## The Power: Creating Any Capability

This protocol can bootstrap ANY emergent capability:

### Sentiment Analysis

```python
gear = BootstrapGear("sentiment")
gear.train("I love this!", "positive")
gear.train("This is terrible", "negative")
gear.train("It's okay", "neutral")
gear.save("sentiment.json")
```

### Intent Classification

```python
gear = BootstrapGear("intent")
gear.train("What's the weather?", "weather_query")
gear.train("Set an alarm for 7am", "set_alarm")
gear.train("Play some music", "play_music")
gear.save("intent.json")
```

### Entity Extraction

```python
gear = BootstrapGear("entities")
gear.train("Meet John at 3pm", '{"person": "John", "time": "3pm"}')
gear.train("Call Sarah tomorrow", '{"person": "Sarah", "time": "tomorrow"}')
gear.save("entities.json")
```

### Domain-Specific Commands

```python
gear = BootstrapGear("git_commands")
gear.train("save my changes", "git commit -am 'update'")
gear.train("get latest code", "git pull")
gear.train("show history", "git log --oneline -10")
gear.save("git_commands.json")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
│                                                             │
│   Input/Output    ──►  LLM Refinement  ──►  Pattern         │
│   Examples             (extract patterns)    Extraction     │
│                                                             │
│                              │                              │
│                              ▼                              │
│                     EmergentState                           │
│                     (patterns, vocab)                       │
│                              │                              │
│                              ▼                              │
│                        Save to JSON                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION PHASE                         │
│                                                             │
│   Load from JSON  ──►  Pattern Matching  ──►  Response      │
│   (no LLM needed)      (pure emergent)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Connection to Previous Designs

| Design | Contribution |
|--------|--------------|
| **075 (Feedback Refinement)** | LLM evaluates and refines responses |
| **076 (Emergent Classifier)** | Patterns emerge from structural signals |
| **077 (This)** | Refinement becomes a reusable, saveable gear |

## The Meta-Pattern

```
PROBLEM: Need new capability X (tool calling, sentiment, etc.)
    ↓
CREATE: BootstrapGear("X")
    ↓
TRAIN: gear.train(input, expected_output) with LLM feedback
    ↓
SAVE: gear.save("X.json")
    ↓
DEPLOY: gear = BootstrapGear.load("X.json")  # No LLM needed
```

## Benefits

### 1. No Hardcoding
Capabilities emerge from training, not hardcoded rules.

### 2. LLM-Free Production
Once trained, the gear runs without LLM calls.

### 3. Versionable
JSON files can be versioned, shared, and refined.

### 4. Composable
Multiple BootstrapGears can be chained together.

### 5. Transparent
Patterns are human-readable and debuggable.

## Future Extensions

### Continuous Learning
```python
# Add new examples to existing gear
gear = BootstrapGear.load("tool_calling.json")
gear.configure_llm(url, model)
gear.train("show network connections", "netstat -an")
gear.save("tool_calling_v2.json")
```

### Pattern Merging
```python
# Merge patterns from multiple gears
gear1 = BootstrapGear.load("file_commands.json")
gear2 = BootstrapGear.load("network_commands.json")
merged = BootstrapGear.merge([gear1, gear2], name="all_commands")
```

### Confidence Decay
```python
# Patterns that aren't used lose confidence over time
gear.decay_unused_patterns(factor=0.95)
```

## Files

- `truthspace_lcm/gears/core/bootstrap_gear.py` - BootstrapGear implementation
- `design_considerations/077_bootstrap_gear_protocol.md` - This document

## Usage

```python
from truthspace_lcm.core.bootstrap_gear import (
    BootstrapGear,
    create_tool_calling_gear,
    create_sentiment_gear,
)

# Quick start with pre-seeded examples
gear = create_tool_calling_gear(llm_url, llm_model)
gear.save("tool_calling.json")

# Or build from scratch
gear = BootstrapGear("my_capability")
gear.configure_llm(url, model)
for inp, out in my_training_data:
    gear.train(inp, out)
gear.save("my_capability.json")

# Production use
gear = BootstrapGear.load("my_capability.json")
result = gear.process(user_input)  # No LLM needed
```

## Conclusion

The Bootstrap Gear Protocol transforms the refinement process into a reusable pattern:

1. **Train** with LLM feedback
2. **Save** emergent state
3. **Deploy** without LLM

This enables creating new capabilities on-demand, without hardcoding, and with the ability to continuously improve through additional training.

```
"The refinement process itself is a gear.
 Train it once, use it forever.
 The LLM is the teacher, not the runtime."
```
