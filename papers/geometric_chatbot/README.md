# Geometric Chatbot: A Fully Geometric Approach to Natural Language Understanding

This directory contains a complete paper and standalone implementation of a chatbot system that operates entirely on geometric principles, eliminating hard-coded linguistic rules.

## Contents

| File | Description |
|------|-------------|
| `paper.md` | The full paper with abstract, methodology, and results |
| `mathematical_foundations.md` | Detailed mathematical derivations for each component |
| `geometric_chatbot.py` | Standalone implementation (no external dependencies beyond Python stdlib) |
| `demo.py` | Interactive demonstration with tutorial and benchmark modes |

## Key Contributions

1. **Geometric Stop Word Detection** - Stop words emerge from semantic role absence, not hard-coded lists
2. **Position-Based Frame Extraction** - Semantic roles assigned by normalized position bands [0, 0.33), [0.33, 0.66), [0.66, 1]
3. **Geometric Morphology** - Verb form equivalence learned from parallel structures
4. **Geometric Conjugation** - Output generation learned from the same parallel structures

## Quick Start

```bash
# Run the main demo
python geometric_chatbot.py

# Interactive chat mode
python demo.py --interactive

# Step-by-step tutorial
python demo.py --tutorial

# Run benchmark tests
python demo.py --benchmark
```

## Benchmark Results

```
Query Tests:      8/8 passed (100%)
Morphology Tests: 6/6 passed (100%)
Conjugation Tests: 6/6 passed (100%)
```

## The Core Insight

Language has geometric structure that can be exploited without explicit linguistic rules:

- **Position** encodes semantic role (subject at 0, verb at 0.5, object at 1)
- **Frequency** distinguishes content words from function words (Zipf's law)
- **Parallel structure** reveals morphological relationships ("I love. I loved." → love ≡ loved)

## Mathematical Foundation

The system is built on the golden ratio φ = 1.618034...:

- φ-weighted position encoding creates self-similar structure
- Zipf weighting (1/log(f+2)) is φ-powers turned inward
- The encode-decode duality mirrors φ and 1/φ

See `mathematical_foundations.md` for detailed derivations.

## Citation

```bibtex
@article{gushurst2024geometric,
  title={Geometric Chatbot: Emergent Language Understanding from Position and Parallel Structure},
  author={Gushurst, Lesley},
  year={2024}
}
```

## License

GPLv3
