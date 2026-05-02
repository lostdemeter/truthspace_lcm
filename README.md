# φ-Geometric Transformation Engine

**Automatic discovery of sequence transformation pipelines from examples.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Feed example (input, output) pairs — get an executable, interpretable, deterministic transformation pipeline. No training. No GPU. No neural networks.

## Quick Start

### Python API

```python
from phi_geometric import PhaseDiscovery

pd = PhaseDiscovery()
pd.add_pair(list('cat'),  list('kæt'))
pd.add_pair(list('ship'), list('ʃɪp'))
pd.add_pair(list('shin'), list('ʃɪn'))
pd.add_pair(list('thin'), list('θɪn'))
pd.add_pair(list('that'), list('θæt'))
pd.add_pair(list('bat'),  list('bæt'))

result = pd.discover()
print(result.archetype)  # 'collapse_map'

nav = result.to_navigator()
trace = nav.execute(list('shat'))
print(trace.output_elements)  # ['ʃ', 'æ', 't']
```

PhaseDiscovery automatically found:
- **Collapse phase**: `sh→ʃ`, `th→θ` (digraph patterns)
- **Map phase**: `a→æ`, `i→ɪ`, `c→k` (single-token substitution)

### Command Line

```bash
# Discover from a TSV file (input<TAB>output, tokens space-separated)
python -m phi_geometric discover pairs.tsv -o pipeline.json

# Execute on new input
python -m phi_geometric execute pipeline.json -i "s h a t"
# Output: ʃ æ t

# Inspect a saved pipeline
python -m phi_geometric info pipeline.json
```

### Save & Load

```python
from phi_geometric import save_pipeline, load_pipeline

# Save (tiny JSON files, ~1KB)
save_pipeline(nav, 'my_pipeline.json')

# Load anywhere — no re-discovery needed
nav = load_pipeline('my_pipeline.json')
trace = nav.execute(list('shop'))
```

### Generation (Reverse, Complete, Navigate)

```python
from phi_geometric import PhaseDiscovery, ReverseEngine

# ... discover and build nav as above ...
engine = ReverseEngine(nav)

# Reverse: "What input produces this output?"
inputs = engine.reverse(list('ʃæt'))      # → [['s','h','a','t']]

# Complete: Fill in wildcards
completions = engine.complete(['?','a','t'], target_output=list('kæt'))

# Navigate: Generate novel valid pairs (Ribbon Math pattern)
novel_pairs = engine.navigate(seed_pairs=training_pairs, steps=200)
```

## What It Does

Given example pairs of (input sequence → output sequence), PhaseDiscovery automatically identifies the transformation structure:

| Phase Type | What It Does | Example |
|---|---|---|
| **Map** | 1→1 token substitution | `a→æ`, `ROT13` |
| **Collapse** | N→M token merging | `sh→ʃ`, `BPE` |
| **Expand** | 1→N token expansion | `x→ks`, abbreviation expansion |
| **Context** | Neighbor-dependent rules | `c→k/s` depending on next vowel |

These phases compose. PhaseDiscovery handles any combination automatically.

### Proven Archetypes (8/8, 100% accuracy)

| Archetype | Phases | Toy Domain |
|---|---|---|
| A: map | [map] | Elvish Cipher |
| B: context→map | [context, map] | Traffic Signals |
| C: collapse→map | [collapse, map] | Musical Chords |
| D: collapse→context→map | [collapse, context, map] | Alien Language |
| E: expand→map | [expand, map] | Phonetic Spelling |
| F: expand→collapse→map | [expand, collapse, map] | Chemical Notation |
| G: expand→context→map | [expand, context, map] | Morse Encoding |
| H: φ-context→map | [context, map] | Vowel Harmony |

### Geometric Context (φ-decay)

For long-range dependencies, enable geometric context:

```python
pd = PhaseDiscovery(geometric=True)
```

This uses φ-level binning (inspired by our [Qwen2-7B reverse engineering](docs/design_considerations/161_attention_spigot.md)) to cover distance 1-12 with just 4 features per direction, mirroring how attention naturally decays.

## Installation

```bash
pip install numpy   # Only dependency
```

No torch, no GPU, no heavy frameworks. The engine is pure Python + numpy.

## Project Structure

```
phi_geometric/
├── __init__.py              # Public API (PhaseDiscovery, save/load)
├── __main__.py              # python -m phi_geometric
├── cli.py                   # Command-line interface
├── core/
│   ├── discovery.py         # StructureDiscovery — information-gain rules
│   ├── cascade_navigator.py # CascadeNavigator — phase pipeline execution
│   ├── phase_discovery.py   # PhaseDiscovery — automatic structure discovery
│   └── serialization.py     # JSON save/load for pipelines
└── examples/
    └── archetypes.py        # 8 ready-to-use archetype examples
```

## API Reference

### PhaseDiscovery

```python
pd = PhaseDiscovery(context_window=1, geometric=False)
pd.add_pair(input_seq, output_seq)   # Add training pair
pd.add_pairs([(inp, out), ...])      # Add multiple pairs
result = pd.discover()               # Run discovery
```

### PhaseDiscoveryResult

```python
result.n_phases          # Number of discovered phases
result.n_rules           # Total rules across all phases
result.archetype         # String signature (e.g., 'collapse_context_map')
result.describe()        # Human-readable summary
result.validate()        # Check accuracy on training data
result.to_navigator()    # Build executable pipeline
```

### CascadeNavigator

```python
nav = result.to_navigator()
trace = nav.execute(input_seq)
trace.output_elements    # The transformed sequence
trace.elements           # Per-element trace with rule details
nav.describe()           # Pipeline structure description
```

### Serialization

```python
from phi_geometric import save_pipeline, load_pipeline

save_pipeline(nav, 'pipeline.json')
nav = load_pipeline('pipeline.json')
```

## The Hypothesis

> **LLMs are hyperdimensional transcoders** — they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights, but in the **shape** those weights create.

This engine is one piece of evidence: transformation structure can be discovered from examples using information geometry (entropy, information gain) and φ-decay attention — the same mathematical primitives found in LLM internals.

See [docs/design_considerations/](docs/design_considerations/) for 240+ design documents exploring this hypothesis, including Qwen2-7B reverse engineering, φ-lattice attention, and the geometric computation framework.

## License

GPLv3 — Lesley Gushurst
