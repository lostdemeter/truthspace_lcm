# 085: Temporary Module Injection for Unknown Queries

## The Problem: "George Washington"

When a query contains words not present in any existing module, the holographic projection returns zero overlap and correctly rejects the query. But this creates a problem:

```
Query: "write a hello world program that prints 'Hello George Washington'"
Result: NONE (no module contains "George" or "Washington")
```

The system correctly identifies that it doesn't know about this query, but it has no way to handle it.

## The Solution: Temporary Module Injection

Instead of rejecting unknown queries, we:

1. **Inject a temporary module** from the query itself
2. **Reproject the space** to include the new module
3. **Query now matches** the temporary module
4. **LLM handles** the request
5. **If successful**: Promote temporary → permanent (learning!)
6. **If failed**: Remove temporary module

This is exactly how the system should behave:
- Known patterns → use existing modules
- Unknown patterns → create temporary, let LLM handle, learn from success

## Implementation

### Core Methods

```python
class HolographicPatternSpace:
    
    def inject_temporary_module(self, query_text: str, 
                                 fallback_effects: Dict = None) -> HolographicModule:
        """
        Inject a temporary module based on the query itself.
        
        The temporary module:
        - Has words extracted from the query
        - Gets projected into the space via eigendecomposition
        - Can be promoted to permanent if successful
        """
        query_words = self.extract_words(query_text)
        
        temp_module = HolographicModule(
            text=query_text,
            words=query_words,
            module_type='temporary',
            effects=fallback_effects or {'task': 'unknown'},
        )
        
        self.modules.append(temp_module)
        self._reproject()  # Recompute all positions
        
        return temp_module
    
    def find_or_inject(self, query_text: str, 
                       fallback_effects: Dict = None,
                       min_similarity: float = 0.3) -> Tuple[Module, float, str, bool]:
        """
        Main entry point: find match or inject temporary.
        
        Returns: (module, confidence, reason, was_injected)
        """
        # First, try to find a match
        module, confidence, reason = self.find_best_match(query_text)
        
        if module is not None and confidence >= min_similarity:
            return module, confidence, reason, False
        
        # No good match - check if we should inject
        query_words = self.extract_words(query_text)
        max_overlap = max(
            (self.word_overlap(query_words, m.words) for m in self.modules),
            default=0.0
        )
        
        if max_overlap == 0:
            # No overlap at all - inject temporary module
            temp_module = self.inject_temporary_module(query_text, fallback_effects)
            return temp_module, 1.0, "injected temporary module", True
        
        # Some overlap but below threshold
        return module, confidence, f"weak match", False
    
    def promote_temporary(self, module: HolographicModule, 
                          new_type: str = 'enhancer',
                          new_effects: Dict = None):
        """
        Promote a temporary module to permanent.
        Called when LLM successfully handled the query.
        """
        if module.module_type == 'temporary':
            module.module_type = new_type
            if new_effects:
                module.effects = new_effects
    
    def remove_temporary_modules(self):
        """Remove all temporary modules (cleanup after failures)."""
        self.modules = [m for m in self.modules if m.module_type != 'temporary']
        if self.modules:
            self._reproject()
```

### Integration with CodeOrchestrator

```python
class CodeOrchestrator:
    def __init__(self):
        self.pattern_space = HolographicPatternSpace(dims=12)
        self._load_initial_modules()
    
    def generate(self, request: str) -> CodePlan:
        # Find or inject module
        module, confidence, reason, was_injected = self.pattern_space.find_or_inject(
            request,
            fallback_effects={'task': 'llm_generation'}
        )
        
        if module and not was_injected:
            # Known pattern - use module's effects
            return self._generate_from_module(module, request)
        
        # Unknown pattern or injected - use LLM
        plan = self._generate_with_llm(request)
        
        if plan.verified and was_injected:
            # Success! Promote the temporary module
            self.pattern_space.promote_temporary(
                module,
                new_type='enhancer',
                new_effects={'code_template': plan.complete_code}
            )
            self._save_modules()  # Persist learning
        elif was_injected:
            # Failed - remove temporary
            self.pattern_space.remove_temporary_modules()
        
        return plan
```

## Why This Works

### Holographic Projection Makes It Natural

When we add a module, we just reproject the entire space:
```python
S = compute_similarity_matrix(modules)  # Include new module
eigenvalues, eigenvectors = np.linalg.eigh(S)
positions = eigenvectors @ np.sqrt(eigenvalues)
```

The new module gets a position based on its similarity to existing modules. If it's truly novel (no word overlap), it gets an orthogonal position.

### The Learning Loop

```
Query → No Match → Inject Temp → LLM Handles → Success?
                                                  ↓
                                           YES: Promote
                                           NO:  Remove
```

Over time, the system learns:
- Successful patterns become permanent modules
- Failed patterns are discarded
- The space grows to cover more of the query distribution

### Connection to "Error = Where to Build"

From previous work, we know that errors tell us where to add structure. Here:
- "No match" = error signal
- Temporary injection = building structure at the error location
- Promotion = confirming the structure is correct

## Test Results

```
Query: 'create a sine wave'
  Result: sine module (0.667)
  Injected: NO

Query: 'read a csv file'
  Result: file module (1.000)
  Injected: NO

Query: 'analyze the data'
  Result: INJECTED (1.000)
  Injected: YES
  → Promoted to permanent after success
  → Now matches without injection

Query: 'write hello world'
  Result: INJECTED (1.000)
  Injected: YES
```

## Advantages

1. **No hardcoded fallbacks**: Unknown queries are handled dynamically
2. **Learning from success**: Good patterns become permanent
3. **Graceful degradation**: Unknown → LLM → learn
4. **Geometric consistency**: New modules integrate naturally via reprojection

## Open Questions

1. **Persistence**: How to save/load the learned modules?
2. **Merging**: Should similar temporary modules be merged?
3. **Pruning**: When to remove rarely-used modules?
4. **Synonyms**: How to handle "analyze" vs "analyse"?

## Files

- Implementation: `/home/thorin/truthspace-lcm/experiments/geometric_patterns.py`
- Classes: `HolographicPatternSpace`, `HolographicModule`
- Methods: `find_or_inject()`, `inject_temporary_module()`, `promote_temporary()`
