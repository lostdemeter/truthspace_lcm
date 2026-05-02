# Design Consideration 096: Gear to HyperMapping Migration

## Overview

This document outlines the migration from the current Gear-based architecture to a HyperMapping-based architecture for the TruthSpace LCM system.

**Goal**: Replace `Gear`, `GearChain`, and `GearState` with `HyperMapping` and `HyperPipeline` while preserving all functionality and improving geometric purity.

## Current Architecture Analysis

### Core Components

| Component | File | Purpose | Lines |
|-----------|------|---------|-------|
| `Gear` | `core/gear.py` | Abstract base class for transformations | ~100 |
| `GearState` | `core/gear.py` | State object passed between gears | ~50 |
| `GearChain` | `core/gear.py` | Sequential composition of gears | ~180 |
| `Quaternion` | `core/gear.py` | 4D rotation encoding | ~80 |
| `GearProtocol` | `core/protocol.py` | Message-aware gear interface | ~600 |
| `GeometricKnowledgeStore` | `core/knowledge/` | Position-based concept storage | ~600 |

### Gear Implementations

| Gear | File | Purpose |
|------|------|---------|
| `ChatGearChain` | `gears/chat_gear_chain.py` | Main chat entry point |
| `KnowledgeLearningGear` | `gears/chat_gear_chain.py` | Learn from interactions |
| `IntentDetectorGear` | `gears/intent_detector_gear.py` | Detect query intent |
| `PythonCodeGear` | `gears/python_code_gear.py` | Code generation |
| `ChatImprovementGear` | `gears/chat_improvement_gear.py` | Response improvement |
| `CorpusBuilderGear` | `gears/corpus_builder_gear.py` | Build knowledge corpus |
| `EmergentGear` | `gears/emergent_gear.py` | Emergent pattern matching |
| `BootstrapGear` | `gears/bootstrap_gear.py` | Bootstrap templates |
| `FactoryGear` | `gears/factory_gear.py` | Create gears dynamically |
| `EmergentClassifierGear` | `gears/emergent_classifier_gear.py` | Classify words/intents |

### Chain Implementations

| Chain | File | Purpose |
|-------|------|---------|
| `ConversationalChain` | `chains/conversational_chain.py` | Main chat logic |
| `SemanticChain` | `chains/semantic_chain.py` | Semantic understanding |
| `LinguisticChain` | `chains/linguistic_chain.py` | Language processing |
| `EmergentDimensionChain` | `chains/base_chain.py` | Dimension discovery |

### Practical Applications

| Application | File | Purpose |
|-------------|------|---------|
| `EmergentChat` | `practical_applications/chat/chat.py` | Interactive chat |
| `api_server` | `practical_applications/chat/api_server.py` | REST API |

## HyperMapping Equivalents

### Direct Mappings

| Gear Concept | HyperMapping Equivalent |
|--------------|------------------------|
| `Gear.forward()` | `HyperMapping.forward()` |
| `Gear.backward()` | `HyperMapping.backward()` |
| `GearChain` | `HyperPipeline` (via `\|` operator) |
| `GearState` | Input/Output values (no separate state object) |
| `Quaternion` | `QuaternionEncoder` (4D positions) |
| `GeometricKnowledgeStore` | `HyperMapping` with `TextEncoder` |
| `Gear.ratio` | Encoder weights / position scaling |

### Functionality Gaps

| Gear Feature | HyperMapping Status | Solution |
|--------------|---------------------|----------|
| `GearState.metadata` | ❌ Missing | Add metadata to `Mapping` dataclass |
| `GearState.errors` | ❌ Missing | Add error tracking to `MatchResult` |
| `Gear.enabled` | ❌ Missing | Add enable/disable to pipeline stages |
| `GearChain.get(name)` | ❌ Missing | Add named stage lookup to `HyperPipeline` |
| `knowledge_store` | ✅ Exists | `HyperMapping` IS a knowledge store |
| `feedback()` | ✅ Exists | `HyperMapping.feedback()` (deprecated) or `reproject()` |
| `prune()` | ❌ Missing | Add pruning based on magnitude |

## Migration Plan

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Extend HyperMapping for Chat Use Cases

```python
# Add to hypermapping.py

@dataclass
class Mapping(Generic[I, O]):
    # Existing fields...
    metadata: Dict[str, Any] = field(default_factory=dict)
    use_count: int = 0
    success_count: int = 0
    created: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.use_count if self.use_count > 0 else 0.0
    
    @property
    def persists(self) -> bool:
        """Concept persists if past critical line."""
        return self.magnitude > CRITICAL_LINE
```

#### 1.2 Add Pipeline Stage Management

```python
# Add to hypermapping.py

class HyperPipeline:
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.stages: List[Tuple[str, HyperMapping]] = []
        self._enabled: Dict[str, bool] = {}
    
    def add(self, name: str, space: HyperMapping) -> 'HyperPipeline':
        self.stages.append((name, space))
        self._enabled[name] = True
        return self
    
    def get(self, name: str) -> Optional[HyperMapping]:
        for n, space in self.stages:
            if n == name:
                return space
        return None
    
    def enable(self, name: str) -> 'HyperPipeline':
        self._enabled[name] = True
        return self
    
    def disable(self, name: str) -> 'HyperPipeline':
        self._enabled[name] = False
        return self
```

#### 1.3 Add Pruning Support

```python
# Add to HyperMapping

def prune(self, threshold: float = CRITICAL_LINE) -> int:
    """Remove mappings below the critical line."""
    before = len(self._mappings)
    self._mappings = [m for m in self._mappings if m.magnitude >= threshold]
    self._rebuild_indices()
    return before - len(self._mappings)

def get_persisting(self) -> List[Mapping]:
    """Get mappings past the critical line."""
    return [m for m in self._mappings if m.magnitude >= CRITICAL_LINE]

def get_fading(self) -> List[Mapping]:
    """Get mappings below the critical line."""
    return [m for m in self._mappings if m.magnitude < CRITICAL_LINE]
```

### Phase 2: Replace GeometricKnowledgeStore (Week 1-2)

The `GeometricKnowledgeStore` is essentially a `HyperMapping` with `TextEncoder`. Replace it:

```python
# New: knowledge_space.py

from hypermapping import HyperMapping, TextEncoder

class KnowledgeSpace(HyperMapping):
    """
    Knowledge storage using HyperMapping.
    
    Replaces GeometricKnowledgeStore with cleaner API.
    """
    
    def __init__(self, name: str = "knowledge", dims: int = 8):
        encoder = TextEncoder(dims=dims)
        super().__init__(dims=dims, encoder=encoder, name=name)
    
    def add_from_text(self, text: str, source: str = "unknown") -> Mapping:
        """Add a concept from text."""
        mapping = self.map(text, text)  # Self-mapping for concepts
        mapping.metadata['source'] = source
        return mapping
    
    def query(self, text: str, top_k: int = 5) -> List[MatchResult]:
        """Query for similar concepts."""
        return self.query_k(text, k=top_k)
    
    def use(self, mapping_id: str, success: bool) -> bool:
        """Update mapping based on success/failure."""
        # Find mapping and update position
        for m in self._mappings:
            if id(m) == mapping_id or str(m.input) == mapping_id:
                if success:
                    # Move toward critical line
                    m.position = m.position * 1.1
                else:
                    # Move away from critical line
                    m.position = m.position * 0.9
                m.use_count += 1
                if success:
                    m.success_count += 1
                return True
        return False
```

### Phase 3: Replace Chat Components (Week 2)

#### 3.1 Replace ChatGearChain

```python
# New: chat_pipeline.py

from hypermapping import HyperMapping, HyperPipeline, TextEncoder, QuaternionEncoder

class ChatPipeline:
    """
    Chat pipeline using HyperMapping.
    
    Replaces ChatGearChain with cleaner geometric architecture.
    """
    
    def __init__(self, dims: int = 8):
        # Intent detection space
        self.intent_space = HyperMapping(
            dims=4,
            encoder=QuaternionEncoder(),
            name="intent"
        )
        self._bootstrap_intents()
        
        # Knowledge space
        self.knowledge_space = KnowledgeSpace(dims=dims)
        
        # Response templates
        self.template_space = HyperMapping(
            dims=dims,
            encoder=TextEncoder(dims=dims),
            name="templates"
        )
        
        # Build pipeline
        self.pipeline = HyperPipeline("chat")
        self.pipeline.add("intent", self.intent_space)
        self.pipeline.add("knowledge", self.knowledge_space)
        self.pipeline.add("templates", self.template_space)
    
    def _bootstrap_intents(self):
        """Bootstrap intent detection."""
        self.intent_space.bootstrap("knowledge question", "KNOWLEDGE")
        self.intent_space.bootstrap("how do I", "KNOWLEDGE")
        self.intent_space.bootstrap("what is", "KNOWLEDGE")
        self.intent_space.bootstrap("create file", "TOOL_CALL")
        self.intent_space.bootstrap("run command", "TOOL_CALL")
        self.intent_space.bootstrap("write code", "CODE_GENERATION")
        self.intent_space.bootstrap("python function", "CODE_GENERATION")
    
    def chat(self, query: str) -> str:
        """Process a chat query."""
        # Detect intent
        intent_result = self.intent_space.forward(query)
        intent = intent_result.output if intent_result else "KNOWLEDGE"
        
        # Route based on intent
        if intent == "KNOWLEDGE":
            return self._handle_knowledge(query)
        elif intent == "TOOL_CALL":
            return self._handle_tool(query)
        elif intent == "CODE_GENERATION":
            return self._handle_code(query)
        
        return self._handle_knowledge(query)  # Default
    
    def _handle_knowledge(self, query: str) -> str:
        # Find relevant knowledge
        results = self.knowledge_space.query(query, top_k=3)
        if results:
            # Compose response from templates
            template_result = self.template_space.forward(query)
            if template_result:
                return template_result.output
            # Fallback: return best match
            return results[0].output
        return "I don't have information about that."
    
    def feedback(self, success: bool) -> bool:
        """Provide feedback on last response."""
        # Update positions based on success
        pass
```

### Phase 4: Replace Remaining Gears (Week 2-3)

| Gear | Replacement Strategy |
|------|---------------------|
| `IntentDetectorGear` | `HyperMapping` with `QuaternionEncoder` + bootstrap |
| `PythonCodeGear` | `HyperMapping` with pattern templates |
| `ChatImprovementGear` | Position-based learning in `HyperMapping` |
| `CorpusBuilderGear` | `KnowledgeSpace.learn()` method |
| `EmergentGear` | `HyperMapping.forward()` with `TextEncoder` |
| `EmergentClassifierGear` | `HyperMapping` with category positions |

### Phase 5: Update Practical Applications (Week 3)

#### 5.1 Replace EmergentChat

```python
# Updated: practical_applications/chat/chat.py

from hypermapping import HyperMapping, TextEncoder
from .chat_pipeline import ChatPipeline

class EmergentChat:
    """Interactive chat using HyperMapping."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.pipeline = ChatPipeline()
    
    def query(self, question: str) -> str:
        return self.pipeline.chat(question)
    
    def feedback(self, success: bool) -> bool:
        return self.pipeline.feedback(success)
```

### Phase 6: Cleanup (Week 3-4)

1. Remove deprecated gear files
2. Update imports throughout codebase
3. Update documentation
4. Run full test suite
5. Performance benchmarking

## File Changes Summary

### Files to Create

| File | Purpose |
|------|---------|
| `truthspace_lcm/core/knowledge_space.py` | HyperMapping-based knowledge store |
| `truthspace_lcm/core/chat_pipeline.py` | HyperMapping-based chat pipeline |
| `truthspace_lcm/core/intent_space.py` | HyperMapping-based intent detection |

### Files to Modify

| File | Changes |
|------|---------|
| `hypermapping/hypermapping.py` | Add metadata, pruning, use tracking |
| `truthspace_lcm/__init__.py` | Update exports |
| `truthspace_lcm/core/__init__.py` | Update exports |
| `truthspace_lcm/practical_applications/chat/chat.py` | Use ChatPipeline |

### Files to Deprecate/Remove

| File | Reason |
|------|--------|
| `core/gear.py` | Replaced by HyperMapping |
| `core/gears/*.py` | Replaced by HyperMapping spaces |
| `core/knowledge/geometric_store.py` | Replaced by KnowledgeSpace |
| `core/chains/*.py` | Replaced by HyperPipeline |

## Key Architectural Decisions

### 1. No Separate State Object

**Gear**: Uses `GearState` to pass data between gears.
**HyperMapping**: Input/output values flow directly through pipeline.

**Rationale**: State is implicit in positions. Metadata can be attached to `Mapping` objects.

### 2. Learning Through Position Movement

**Gear**: `feedback()` method on individual gears.
**HyperMapping**: `reproject()` for exact learning, position scaling for incremental.

**Rationale**: Position IS the learned state. Movement IS learning.

### 3. Composition via Pipeline

**Gear**: `GearChain.add()` with sequential processing.
**HyperMapping**: `|` operator for composition, `HyperPipeline` for named stages.

**Rationale**: Cleaner API, functional composition.

### 4. Intent Detection via Bootstrap

**Gear**: `IntentDetectorGear` with hardcoded patterns.
**HyperMapping**: `QuaternionEncoder` with bootstrapped templates.

**Rationale**: Templates are explicit, positions are learned.

## Testing Strategy

### Unit Tests

1. `test_knowledge_space.py` - KnowledgeSpace CRUD operations
2. `test_chat_pipeline.py` - Chat routing and response generation
3. `test_intent_detection.py` - Intent classification accuracy

### Integration Tests

1. Full chat conversation flow
2. Knowledge persistence (save/load)
3. Learning from feedback

### Regression Tests

1. Compare responses before/after migration
2. Verify 100% accuracy on existing test cases

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Core Infrastructure | Extended HyperMapping, HyperPipeline |
| 1-2 | Knowledge Store | KnowledgeSpace replacing GeometricKnowledgeStore |
| 2 | Chat Components | ChatPipeline replacing ChatGearChain |
| 2-3 | Remaining Gears | All gears replaced |
| 3 | Applications | EmergentChat updated |
| 3-4 | Cleanup | Deprecated code removed, docs updated |

## Success Criteria

1. ✅ All existing functionality preserved
2. ✅ 100% accuracy on test suite maintained
3. ✅ Cleaner, more geometric API
4. ✅ No magic numbers or bags of words
5. ✅ Full serialization support
6. ✅ Reduced code complexity

## Open Questions

1. **Quaternion multiplication**: Should we preserve Hamilton product for chaining?
   - **Decision**: Keep as utility, but positions are the primary mechanism.

2. **GearProtocol**: Should we keep the message protocol?
   - **Decision**: Simplify to input/output, metadata in Mapping.

3. **Auto-build**: How to handle background corpus building?
   - **Decision**: Add `learn_async()` method to KnowledgeSpace.

## Conclusion

The migration from Gears to HyperMapping simplifies the architecture while preserving all functionality. The key insight is that **position IS state** - we don't need a separate `GearState` object when positions encode all the information.

The HyperMapping approach is:
- **Cleaner**: No separate state object
- **More geometric**: Position-based everything
- **More composable**: Pipeline operator
- **More serializable**: All state explicit
- **More testable**: Clear input/output contracts
