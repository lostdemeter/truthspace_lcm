# DC299 Phase 1 — Visualization Summary

**Input:** `dc299_phase1_axes.json`  
**Plot:** `dc299_phase1_viz.png`

## Key Numbers

| Metric | Value |
|--------|-------|
| Total axes | 1500 (6 seeds + 1494 discovered) |
| Max cumulative variance | 0.9067 (90.7%) |
| Axes to reach 50% variance | 292 |
| Axes to reach 75% variance | 779 |
| Axes to reach 90% variance | 1456 |
| Axes to reach 95% variance | Not reached (need more) |
| Semantic axes (quality ≥ 0.5) | 305 (20%) |
| Structural axes (quality < 0.5) | 1189 (80%) |
| Semantic quality cliff | ~axis 214 |
| Last axis binary_acc | 0.987 |
| Last axis gap | 0.0783 |

## Variance Extrapolation

Mean step variance (last 50 axes): 0.001584

If decay rate stays constant:
- Need ~28 more axes to reach 95%
- Total axes needed: ~1528

## Semantic Quality Summary

Quality score = fraction of top-20 + bottom-20 vocab tokens that are
pure ASCII alphabetic strings of length ≥ 3 (proxy for 'real English word').

Rolling mean of quality score drops below 0.4 permanently at ~axis 214.
This is the approximate boundary between semantic and structural axes.
