# Archived Core Modules

These files represent the evolution of the geometric LCM architecture.
They are preserved for reference but are not part of the active codebase.

## Files

- `dynamic_lcm.py` - Early dynamic primitive discovery (supervised domains)
- `geometric_lcm.py` - Density-based clustering with word-hash encoding
- `hybrid_lcm.py` - LLM embeddings + geometric clustering (validation)
- `self_organizing_lcm.py` - Unsupervised clustering attempt
- `stacked_geometric_lcm.py` - v1 of stacked architecture (80D, 5 layers)

## Current Primary Module

The active implementation is `stacked_lcm.py` in the parent directory:
- 128 dimensions
- 7 layers (morph, lex, syn, comp, disamb, ctx, global)
- No external LLM dependencies

## Import

```python
from truthspace_lcm.core import StackedLCM
```
