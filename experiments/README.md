# TruthSpace Experiments

This directory contains experiments to validate and refine the hierarchical knowledge navigation architecture.

## Experiments

### Experiment 1: Domain Centroid Definition
**Question**: How should we define domain centroids?
**File**: `exp01_centroid_definition.py`

### Experiment 2: Cross-Domain Discrimination  
**Question**: Can we discriminate overlapping concepts across domains?
**File**: `exp02_cross_domain_discrimination.py`

### Experiment 3: Hierarchy Depth
**Question**: What is the optimal hierarchy depth?
**File**: `exp03_hierarchy_depth.py`

### Experiment 4: Hard vs Soft Navigation
**Question**: Should navigation be deterministic or probabilistic?
**File**: `exp04_navigation_strategy.py`

### Experiment 5: Contextual Priming
**Question**: How does conversation history affect domain selection?
**File**: `exp05_contextual_priming.py`

## Running Experiments

```bash
# Run all experiments
python -m experiments.run_all

# Run specific experiment
python experiments/exp01_centroid_definition.py
```

## Results

Results are logged to `results/` subdirectory with timestamps.
