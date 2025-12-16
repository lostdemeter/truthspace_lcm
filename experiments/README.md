# Experiments

This directory contains validation and testing scripts for TruthSpace LCM.

## Active Tests

- **`final_validation.py`** - Validates the 12D plastic-primary encoding and dimension-aware autotuner
- **`live_fire_test.py`** - End-to-end test: ingest ifconfig man page and test natural language queries

## Archive

The `archive/` directory contains research experiments that informed the v2 design:

- `test_hypothesis.py` - Vacuum forming hypothesis experiments
- `dimensionality_analysis.py` - Analysis of optimal dimensions and plastic constant
- `compare_encoders.py` - Comparison of v1 (φ-8D) vs v2 (ρ-12D) encoders

## Running Tests

```bash
# Validate core system
python experiments/final_validation.py

# Run live fire test
python experiments/live_fire_test.py
```
