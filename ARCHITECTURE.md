# TruthSpace LCM Architecture

## Overview

TruthSpace LCM uses a **gear chain architecture** where composable transformation units (gears) process state through pipelines. This replaces traditional neural network approaches with explicit, interpretable geometric transformations.

## Core Components

### 1. Base Classes (`core/gear.py`)

```python
Gear          # Abstract base class for transformations
GearState     # State object flowing through chains
GearChain     # Container for composing gears
Quaternion    # 4D rotation encoding for parameters
```

### 2. Protocol (`core/protocol.py`)

```python
GearProtocol  # Interface for message-aware gears
GearMessage   # Standardized message format
MessageIntent # Intent enumeration
```

### 3. Gears (`core/gears/`)

Individual transformation units:
- `bootstrap_gear.py` - Pattern bootstrapping
- `chat_improvement_gear.py` - Response improvement
- `corpus_builder_gear.py` - Self-building corpus
- `emergent_classifier_gear.py` - Emergent classification
- `emergent_gear.py` - Dimension discovery
- `factory_gear.py` - Gear creation
- `intent_detector_gear.py` - Intent detection
- `python_code_gear.py` - Code generation

### 4. Chains (`core/chains/`)

Sequences of gears:
- `base_chain.py` - EmergentDimensionChain (abstract)
- `conversational_chain.py` - Main chat chain
- `semantic_chain.py` - Semantic understanding
- `linguistic_chain.py` - Linguistic processing

### 5. Orchestrators (`core/orchestrators/`)

Multi-gear coordination:
- `code_orchestrator.py` - Code generation orchestration
- `gear_orchestrator.py` - General gear orchestration

### 6. Classifiers (`core/classifiers/`)

Intent classification:
- `intent_classifier.py` - Geometric intent detection

### 7. Utilities (`core/utils/`)

Support classes:
- `holographic_pattern_space.py` - Pattern matching
- `template_composer.py` - Template generation
- `folding_deficiency.py` - Shape analysis
- `gear_improvement_loop.py` - Self-improvement
- `contact_point.py` - Contact structures

## Data Flow

```
User Input
    ↓
EmergentChatEngine (api_server.py)
    ↓
IntentClassifier → Intent (KNOWLEDGE, CODE_GENERATION, etc.)
    ↓
┌─────────────────────────────────────────┐
│ KNOWLEDGE → ConversationalChain.chat()  │
│ CODE_GENERATION → CodeOrchestrator      │
│ TOOL_CALL → GearOrchestrator            │
└─────────────────────────────────────────┘
    ↓
Response (text or tool_calls)
```

## Key Design Principles

### 1. Fail-Fast
No graceful fallbacks. If geometric classification fails, we see the error rather than hiding it with pattern matching.

### 2. Emergent Structure
Dimensions are discovered via SVD, not designed top-down. The system learns its own structure from data.

### 3. ENCODE = DECODE
Encoding and decoding are the same operation in opposite directions. The transformation IS the understanding.

### 4. Separation of Concerns
- **Corpus** = Knowledge (what)
- **Gears** = Reasoning (how)
- **Chains** = Composition (flow)

## API Layer

The `practical_applications/chat/` module provides:
- `api_server.py` - OpenAI-compatible REST API
- `chat.py` - EmergentChat class for programmatic use
- `run_api.py` - Server runner with corpus options

## Corpus Structure

Knowledge is stored in JSON corpuses (`corpus/`):
- Topic-based organization
- Definitions and facts
- Relationship patterns

The corpus is built at startup using LLM calls, then chat operates without LLM (truly emergent).
