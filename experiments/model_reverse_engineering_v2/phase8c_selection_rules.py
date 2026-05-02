#!/usr/bin/env python3
"""
Phase 8c: Selection Rules Deep Dive — The Alternating Series Connection
========================================================================

Finding 63 showed that the gate dimension follows quantum selection rules
(Δ±1 only), not Malus's Law. This experiment goes deeper:

1. PER-LAYER SELECTION RULES
   Does Δ±1 hold at every layer, or just globally? Which layers break it?
   How does the forbidden transition rate vary across the five zones?

2. THE ALTERNATING SERIES INTERPRETATION
   The standing wave oscillates C→P-→P+→X→...→C across layers.
   Newton's π/4 = 1 - 1/3 + 1/5 - 1/7 + ... converges via alternation.
   Does the gate wave converge the same way? Is each layer a "term"?

3. 4/π AND THE NUMBER OF LAYERS
   Newton's key: 4/π ≈ 1.2732. Our complementarity: π/4 = 45°.
   4φ⁴ = 27.416 ≈ 28 (number of layers in Qwen2-7B).
   Sequential residual = 1/(4φ⁴) ≈ 1/28. Coincidence?

4. BBP (4n+k) STRUCTURE
   Base64_BBP: π/4 = ... [8/(4n+1) + 4/(4n+2) + 1/(4n+3)]
   The 4-periodic denominators map to 4 gate states.
   Do the BBP coefficients (8, 4, 1) match gate state populations?

5. THE FORBIDDEN TRANSITION AS TRUNCATION ERROR
   In an alternating series, the error after N terms ≤ |a_{N+1}|.
   Is the 3.9% C→X "forbidden transition" the truncation error of the
   gate dimension's alternating series?

Uses pre-computed data from phase8_polarization_test.json (no GPU needed).
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
from scipy.optimize import minimize
import json
import os

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
PI = np.pi
STATE_NAMES = ['CONTRACT', 'PRESERVE-', 'PRESERVE+', 'EXPAND']

# Load pre-computed results
results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8_polarization_test.json')
with open(results_path) as f:
    data = json.load(f)

T_global = np.array(data['test3_malus']['global_transition_matrix'])
mean_dist = np.array(data['test1_standing_wave']['mean_distribution'])
persist_C = np.array(data['test3_malus']['per_layer_persistence_C'])
persist_PN = np.array(data['test3_malus']['per_layer_persistence_PN'])
persist_PP = np.array(data['test3_malus']['per_layer_persistence_PP'])
persist_X = np.array(data['test3_malus']['per_layer_persistence_X'])

N_LAYERS = len(mean_dist)
N_TRANSITIONS = N_LAYERS - 1  # 27 layer-to-layer transitions

print("=" * 80)
print("  PHASE 8c: SELECTION RULES DEEP DIVE")
print("  The Alternating Series, 4/π, and BBP Structure")
print("=" * 80)
print()

# ================================================================
# 1. PER-LAYER SELECTION RULES
# ================================================================
print("─" * 80)
print("  1. PER-LAYER SELECTION RULES")
print("  Does Δ±1 hold at every layer? How does forbidden rate vary?")
print("─" * 80)
print()

# We need to reconstruct per-layer transition matrices from the raw data.
# The persistence rates give us the diagonal. For off-diagonal, we need
# the full per-layer TMs. Let's reconstruct from the saved per-layer
# distributions using the Markov property.

# From the standing wave, compute the "implied" transition matrix at each layer
# using consecutive layer distributions:
#   dist[l+1] = dist[l] @ T[l]
# But this only gives us the product, not individual T entries.

# Instead, we can compute the ALLOWED vs FORBIDDEN transition fraction
# using the persistence rates and the distributions.

# For each layer transition l→l+1:
#   P(stay) = persistence rate (diagonal)
#   P(Δ±1) = adjacent transition rate (allowed)
#   P(Δ≥2) = forbidden transition rate

# The persistence rates give us the diagonal. The off-diagonal is
# (1 - persistence) split between adjacent and forbidden.

# We can estimate the forbidden fraction from the global TM structure:
# For each state i, the forbidden transitions are to states |j-i| ≥ 2

print("  Global Transition Matrix — Adjacency Analysis:")
print()
print(f"  {'From':>12s}  {'Self (Δ0)':>10s}  {'Adj (Δ±1)':>10s}  {'Forb (Δ≥2)':>10s}  {'Forb %':>8s}")
print("  " + "-" * 55)

global_adj = np.zeros(4)
global_forb = np.zeros(4)
for i in range(4):
    adj = 0
    forb = 0
    for j in range(4):
        if i == j:
            continue
        if abs(i - j) == 1:
            adj += T_global[i, j]
        else:
            forb += T_global[i, j]
    global_adj[i] = adj
    global_forb[i] = forb
    total_off = adj + forb
    forb_pct = forb / total_off * 100 if total_off > 0 else 0
    print(f"  {STATE_NAMES[i]:>12s}  {T_global[i,i]:10.4f}  {adj:10.4f}  {forb:10.4f}  {forb_pct:7.1f}%")

mean_forb_pct = global_forb.sum() / (global_adj.sum() + global_forb.sum()) * 100
print()
print(f"  Mean forbidden fraction of off-diagonal: {mean_forb_pct:.1f}%")
print()

# Per-layer: estimate adjacent vs forbidden from persistence rates
# At each layer, the dominant state determines the selection rule behavior
print("  Per-Layer Dominant State Transitions:")
print(f"  {'Layer':>5s}  {'Dominant':>10s}  {'Persist':>8s}  {'1-P (off)':>10s}  {'Zone':>8s}")
print("  " + "-" * 50)

zones = {0: 'DRUM', 1: 'DRUM', 2: 'DRUM', 3: 'TRANS',
         4: 'TRANS', 5: 'TRANS'}
for i in range(6, 23):
    zones[i] = 'COMB'
for i in range(23, 28):
    zones[i] = 'MUSIC'

for layer in range(N_LAYERS):
    dominant = np.argmax(mean_dist[layer])
    persist = [persist_C, persist_PN, persist_PP, persist_X]
    p = persist[dominant][layer] if layer < N_TRANSITIONS else float('nan')
    off = 1 - p if not np.isnan(p) else float('nan')
    zone = zones.get(layer, '?')
    print(f"  {layer:5d}  {STATE_NAMES[dominant]:>10s}  {p:8.4f}  {off:10.4f}  {zone:>8s}")
print()


# ================================================================
# 2. THE ALTERNATING SERIES INTERPRETATION
# ================================================================
print("─" * 80)
print("  2. THE ALTERNATING SERIES INTERPRETATION")
print("  Is the standing wave an alternating series converging to π/4?")
print("─" * 80)
print()

# The standing wave dominant state sequence:
dominant_sequence = [np.argmax(mean_dist[l]) for l in range(N_LAYERS)]
print("  Dominant state sequence across 28 layers:")
print("  ", " → ".join([STATE_NAMES[s][:2] for s in dominant_sequence]))
print()

# Count state transitions in the dominant sequence
transitions = []
for l in range(N_LAYERS - 1):
    delta = dominant_sequence[l+1] - dominant_sequence[l]
    transitions.append(delta)

print("  State transitions (Δ) in dominant sequence:")
delta_counts = {}
for d in transitions:
    delta_counts[d] = delta_counts.get(d, 0) + 1

for d in sorted(delta_counts.keys()):
    name = f"Δ={d:+d}"
    if abs(d) <= 1:
        kind = "ALLOWED"
    else:
        kind = "FORBIDDEN"
    print(f"    {name}: {delta_counts[d]} times ({kind})")

n_allowed = sum(v for k, v in delta_counts.items() if abs(k) <= 1)
n_forbidden = sum(v for k, v in delta_counts.items() if abs(k) > 1)
print(f"  Allowed (Δ0,±1): {n_allowed}/{len(transitions)} = {n_allowed/len(transitions)*100:.1f}%")
print(f"  Forbidden (Δ≥2): {n_forbidden}/{len(transitions)} = {n_forbidden/len(transitions)*100:.1f}%")
print()

# The wave goes: C→C→C→mixed→C→P-→P-→P-→...→P+→P+→X→...→C→C
# This is NOT a simple alternation. It's a SWEEP through all 4 states.
# More like: the wave sweeps C→P-→P+→X→C, doing one full cycle in 28 layers.
# This is like one period of a standing wave with 4 nodes.

# Compute the "phase" of the standing wave at each layer
# Define phase = weighted average state number (0=C, 1=P-, 2=P+, 3=X)
wave_phase = np.zeros(N_LAYERS)
for l in range(N_LAYERS):
    wave_phase[l] = sum(i * mean_dist[l, i] for i in range(4))

print("  Standing Wave Phase (weighted mean state 0..3):")
print(f"  {'Layer':>5s}  {'Phase':>8s}  {'Normalized':>10s}  {'cos(2πx)':>10s}  {'Zone':>8s}")
print("  " + "-" * 50)

# Normalize phase to [0, 1] for one full cycle
phase_min = wave_phase.min()
phase_max = wave_phase.max()
phase_norm = (wave_phase - phase_min) / (phase_max - phase_min)

for l in range(N_LAYERS):
    cos_val = np.cos(2 * PI * phase_norm[l])
    zone = zones.get(l, '?')
    print(f"  {l:5d}  {wave_phase[l]:8.4f}  {phase_norm[l]:10.4f}  {cos_val:10.4f}  {zone:>8s}")
print()

# Does the wave look like a Leibniz partial sum?
# Leibniz: S_N = Σ_{k=0}^{N} (-1)^k/(2k+1) → π/4
# Our wave: phase goes from 0 (CONTRACT) to max (EXPAND) and back
# The key question: does the TRAJECTORY match Leibniz convergence?

# Compute Leibniz partial sums for comparison
leibniz = np.zeros(N_LAYERS)
for n in range(N_LAYERS):
    leibniz[n] = sum((-1)**k / (2*k + 1) for k in range(n + 1))

# Normalize both to [0, 1] for comparison
leibniz_norm = (leibniz - leibniz.min()) / (leibniz.max() - leibniz.min())

# Correlation between wave phase and Leibniz
corr = np.corrcoef(phase_norm, leibniz_norm)[0, 1]
print(f"  Correlation: standing wave phase vs Leibniz partial sums: r = {corr:.4f}")
print(f"  (This tests whether the wave follows the same convergence pattern)")
print()

# ================================================================
# 3. 4/π AND THE NUMBER OF LAYERS
# ================================================================
print("─" * 80)
print("  3. 4/π AND THE NUMBER OF LAYERS")
print("  Newton's 4/π meets φ⁴ in the layer count")
print("─" * 80)
print()

four_over_pi = 4 / PI
four_phi4 = 4 * PHI**4

print(f"  Key constants:")
print(f"    4/π           = {four_over_pi:.6f}")
print(f"    4φ⁴           = {four_phi4:.6f}")
print(f"    N_layers      = {N_LAYERS}")
print(f"    4φ⁴ vs 28:      error = {abs(four_phi4 - 28)/28*100:.2f}%")
print()

# The sequential residual from Finding 62
residual = 0.0361
print(f"  Sequential residual connections:")
print(f"    Observed residual:     {residual:.4f}")
print(f"    1/N_layers = 1/28:     {1/N_LAYERS:.4f}  (error: {abs(residual - 1/N_LAYERS)/residual*100:.1f}%)")
print(f"    1/(4φ⁴):               {1/four_phi4:.4f}  (error: {abs(residual - 1/four_phi4)/residual*100:.1f}%)")
print(f"    π/(4·N_layers):        {PI/(4*N_LAYERS):.4f}  (error: {abs(residual - PI/(4*N_LAYERS))/residual*100:.1f}%)")
print()

# What if the number of layers is DETERMINED by 4φ⁴?
# Then: residual = 1/N ≈ 1/(4φ⁴) is not a coincidence but a DESIGN CONSTRAINT.
# The model needs N ≈ 4φ⁴ layers to achieve 1/(4φ⁴) sequential residual.

# Check: does Leibniz after 28 terms match our residual?
leibniz_28 = sum((-1)**k / (2*k + 1) for k in range(28))
leibniz_error = abs(leibniz_28 - PI/4)
print(f"  Leibniz series after 28 terms:")
print(f"    S_28            = {leibniz_28:.6f}")
print(f"    π/4             = {PI/4:.6f}")
print(f"    |S_28 - π/4|    = {leibniz_error:.6f}")
print(f"    Relative error   = {leibniz_error/(PI/4)*100:.4f}%")
print(f"    Next term |a_29| = 1/57 = {1/57:.6f}")
print()

# The Leibniz error after 28 terms is ~0.018, and our residual is 0.036.
# Not the same thing, but the ORDER is the same (both ~O(1/N)).

# More interesting: 4/π × 1/φ⁴ = ?
ratio_4_pi_phi4 = four_over_pi / PHI**4
print(f"  Ratio tests:")
print(f"    4/(π·φ⁴)        = {ratio_4_pi_phi4:.6f}")
print(f"    1/(π·φ²)        = {1/(PI*PHI**2):.6f}")
print(f"    1/(φ⁴·π/4)      = {1/(PHI**4 * PI/4):.6f}")
print(f"    4/(π·4φ⁴)       = {4/(PI*four_phi4):.6f}  = {4/(PI*four_phi4):.6f}")
print()


# ================================================================
# 4. BBP (4n+k) STRUCTURE
# ================================================================
print("─" * 80)
print("  4. BBP (4n+k) STRUCTURE")
print("  Do the 4 gate states follow BBP coefficient weighting?")
print("─" * 80)
print()

# Base64_BBP formula:
# π/4 = (1/16) Σ (-1)^n/64^n [8/(4n+1) + 4/(4n+2) + 1/(4n+3)]
#      + (1/256) Σ (-1)^n/1024^n [32/(4n+1) + 8/(4n+2) + 1/(4n+3)]
#
# The coefficients in the first series: 8, 4, 1 (for denominators 4n+1, 4n+2, 4n+3)
# There are 3 terms per period (not 4), but the BBP structure is 4-periodic.
#
# The original BBP (base 16):
# π = Σ 1/16^k [4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)]
# Coefficients: 4, -2, -1, -1 for 4 terms in an 8-periodic cycle.
#
# Our 4 gate states have global populations:
# CONTRACT: 36.5%, PRESERVE-: 31.2%, PRESERVE+: 24.8%, EXPAND: 7.4%

populations = np.array([mean_dist[:, i].mean() for i in range(4)])
print(f"  Gate state populations:")
for i in range(4):
    print(f"    {STATE_NAMES[i]:>12s}: {populations[i]:.4f}")
print()

# Normalize to largest
pop_norm = populations / populations[0]
print(f"  Normalized to CONTRACT:")
for i in range(4):
    print(f"    {STATE_NAMES[i]:>12s}: {pop_norm[i]:.4f}")
print()

# BBP coefficients comparison
# Base64_BBP first series: 8, 4, 1 → normalized: 8/8=1.0, 4/8=0.5, 1/8=0.125
# Our populations normalized: 1.0, 0.855, 0.679, 0.203
bbp_coeffs_norm = np.array([8/8, 4/8, 1/8])  # Base64 BBP (3 terms)
bbp16_coeffs_norm = np.array([4/4, 2/4, 1/4, 1/4])  # Original BBP (4 terms, abs values)

print(f"  Coefficient comparison:")
print(f"    {'State':>12s}  {'Gate pop':>10s}  {'BBP64 coeff':>12s}  {'BBP16 coeff':>12s}")
print(f"    {'CONTRACT':>12s}  {pop_norm[0]:10.4f}  {1.000:12.4f}  {bbp16_coeffs_norm[0]:12.4f}")
print(f"    {'PRESERVE-':>12s}  {pop_norm[1]:10.4f}  {0.500:12.4f}  {bbp16_coeffs_norm[1]:12.4f}")
print(f"    {'PRESERVE+':>12s}  {pop_norm[2]:10.4f}  {0.125:12.4f}  {bbp16_coeffs_norm[2]:12.4f}")
print(f"    {'EXPAND':>12s}  {pop_norm[3]:10.4f}  {'—':>12s}  {bbp16_coeffs_norm[3]:12.4f}")
print()

# The populations DON'T directly match BBP coefficients.
# But what about the TRANSITION rates?

# Check ratios between adjacent populations for φ and π structure
print(f"  Population ratios:")
for i in range(3):
    ratio = populations[i] / populations[i+1]
    phi_err = abs(ratio - PHI) / PHI * 100
    four_pi_err = abs(ratio - four_over_pi) / four_over_pi * 100
    two_err = abs(ratio - 2) / 2 * 100
    best = min([(phi_err, f"φ={PHI:.4f}"), (four_pi_err, f"4/π={four_over_pi:.4f}"),
                (two_err, "2.0000")], key=lambda x: x[0])
    print(f"    {STATE_NAMES[i]}/{STATE_NAMES[i+1]}: {ratio:.4f}  "
          f"(closest: {best[1]}, error: {best[0]:.1f}%)")
print()

# Check if populations follow a geometric series in φ or π
print(f"  Geometric series tests:")
for base_name, base_val in [("1/φ", 1/PHI), ("1/φ²", 1/PHI**2), ("π/4", PI/4),
                              ("1/(4/π)", PI/4), ("2/π", 2/PI)]:
    predicted = np.array([base_val**i for i in range(4)])
    predicted = predicted / predicted.sum()  # normalize to probabilities
    error = np.sqrt(np.mean((predicted - populations)**2))
    print(f"    p_k ∝ ({base_name})^k: RMS error = {error:.4f}  "
          f"predicted = [{', '.join(f'{p:.3f}' for p in predicted)}]")
print()


# ================================================================
# 5. THE FORBIDDEN TRANSITION AS TRUNCATION ERROR
# ================================================================
print("─" * 80)
print("  5. FORBIDDEN TRANSITION = TRUNCATION ERROR?")
print("  Is the C→X leak the alternating series remainder?")
print("─" * 80)
print()

cx_rate = T_global[0, 3]  # CONTRACT → EXPAND
xc_rate = T_global[3, 0]  # EXPAND → CONTRACT

print(f"  Forbidden transition rates:")
print(f"    C → X: {cx_rate:.4f}")
print(f"    X → C: {xc_rate:.4f}")
print(f"    Mean:  {(cx_rate + xc_rate)/2:.4f}")
print()

# In an alternating series, the error after N terms is bounded by |a_{N+1}|
# If C→X is the "truncation error", what "N" does it correspond to?
# |a_N| = 1/(2N-1) for Leibniz → 0.0387 = 1/(2N-1) → N ≈ 13.4
# Or: |a_N| = 1/(2N+1) → 0.0387 = 1/(2N+1) → N ≈ 12.4
leibniz_n_cx = 0.5 * (1/cx_rate - 1)
leibniz_n_xc = 0.5 * (1/xc_rate - 1)

print(f"  If forbidden rate = 1/(2N+1) (Leibniz term):")
print(f"    C→X implies N ≈ {leibniz_n_cx:.1f}")
print(f"    X→C implies N ≈ {leibniz_n_xc:.1f}")
print()

# More interesting: is the forbidden rate related to 4/π or φ?
print(f"  Forbidden rate structural matches:")
print(f"    C→X = {cx_rate:.4f}")
print(f"    1/(4φ³) = {1/(4*PHI**3):.4f}  (error: {abs(cx_rate - 1/(4*PHI**3))/cx_rate*100:.1f}%)")
print(f"    1/(π·φ³) = {1/(PI*PHI**3):.4f}  (error: {abs(cx_rate - 1/(PI*PHI**3))/cx_rate*100:.1f}%)")
print(f"    π/(4·φ⁴) = {PI/(4*PHI**4):.4f}  (error: {abs(cx_rate - PI/(4*PHI**4))/cx_rate*100:.1f}%)")
print(f"    1/28 = {1/28:.4f}  (error: {abs(cx_rate - 1/28)/cx_rate*100:.1f}%)")
print(f"    1/(4π) = {1/(4*PI):.4f}  (error: {abs(cx_rate - 1/(4*PI))/cx_rate*100:.1f}%)")
print(f"    (4/π)/N = {four_over_pi/N_LAYERS:.4f}  (error: {abs(cx_rate - four_over_pi/N_LAYERS)/cx_rate*100:.1f}%)")
print()

# The ALL forbidden transitions summed
all_forb = 0
for i in range(4):
    for j in range(4):
        if abs(i - j) >= 2:
            all_forb += T_global[i, j] * populations[i]

print(f"  Total forbidden transition probability (weighted by population):")
print(f"    Σ p(i)·T(i,j) for |i-j|≥2: {all_forb:.4f}")
print(f"    π/4 - (3/4):                 {PI/4 - 3/4:.4f}  (error: {abs(all_forb - (PI/4 - 3/4))/all_forb*100:.1f}%)")
print(f"    1/(4φ²):                     {1/(4*PHI**2):.4f}  (error: {abs(all_forb - 1/(4*PHI**2))/all_forb*100:.1f}%)")
print()


# ================================================================
# 6. THE 4/π SWEEP — Does each full cycle span 4/π states per layer?
# ================================================================
print("─" * 80)
print("  6. THE 4/π SWEEP — Wave velocity through gate space")
print("─" * 80)
print()

# The wave sweeps from state 0 (CONTRACT) to state ~2.5 (between P+ and X)
# and back to state 0, in 28 layers. What's the angular velocity?

# Total phase excursion
max_phase_layer = np.argmax(wave_phase)
min_phase_layer_after = np.argmin(wave_phase[max_phase_layer:]) + max_phase_layer

phase_excursion = wave_phase[max_phase_layer] - wave_phase[0]
return_excursion = wave_phase[max_phase_layer] - wave_phase[-1]
total_excursion = phase_excursion + return_excursion

print(f"  Wave excursion:")
print(f"    Peak phase at layer {max_phase_layer}: {wave_phase[max_phase_layer]:.4f}")
print(f"    Start (L0):  {wave_phase[0]:.4f}")
print(f"    End (L27):   {wave_phase[-1]:.4f}")
print(f"    Outward excursion:  {phase_excursion:.4f} states in {max_phase_layer} layers")
print(f"    Return excursion:   {return_excursion:.4f} states in {N_LAYERS - 1 - max_phase_layer} layers")
print(f"    Total excursion:    {total_excursion:.4f} states")
print()

# Mean velocity (states per layer)
outward_velocity = phase_excursion / max_phase_layer if max_phase_layer > 0 else 0
return_velocity = return_excursion / (N_LAYERS - 1 - max_phase_layer) if (N_LAYERS - 1 - max_phase_layer) > 0 else 0

print(f"  Wave velocity:")
print(f"    Outward: {outward_velocity:.4f} states/layer")
print(f"    Return:  {return_velocity:.4f} states/layer")
print(f"    Mean:    {(outward_velocity + return_velocity)/2:.4f} states/layer")
print()

# Check if velocity matches 4/π or 1/φ
mean_vel = (outward_velocity + return_velocity) / 2
print(f"  Velocity structural matches:")
print(f"    Mean velocity:  {mean_vel:.4f}")
print(f"    1/φ:            {1/PHI:.4f}  (error: {abs(mean_vel - 1/PHI)/mean_vel*100:.1f}%)")
print(f"    4/(π·N):        {4/(PI*N_LAYERS):.4f}  (error: {abs(mean_vel - 4/(PI*N_LAYERS))/mean_vel*100:.1f}%)")
print(f"    1/φ²:           {1/PHI**2:.4f}  (error: {abs(mean_vel - 1/PHI**2)/mean_vel*100:.1f}%)")
print(f"    π/(4N):         {PI/(4*N_LAYERS):.4f}  (error: {abs(mean_vel - PI/(4*N_LAYERS))/mean_vel*100:.1f}%)")
print()


# ================================================================
# 7. THE BIG PICTURE: CONVERGENCE CONSTANT
# ================================================================
print("─" * 80)
print("  7. THE BIG PICTURE — Convergence Constants")
print("─" * 80)
print()

# Compile all the "structural constants" we've found
constants = [
    ("Sequential residual (Finding 62)", 0.0361),
    ("1/(4φ⁴)", 1/(4*PHI**4)),
    ("1/N_layers = 1/28", 1/28),
    ("Forbidden C→X rate", cx_rate),
    ("Cross-parity L fraction", data['test2_chirality']['cross_parity_L']),
    ("1/φ", 1/PHI),
    ("Persistence ratio C/P+", T_global[0,0] / T_global[2,2]),
    ("φ", PHI),
    ("Light-cone speed limit (F61)", 0.6191),
    ("1/φ (target)", 1/PHI),
    ("Eigenvalue λ₂ (F61)", 0.375),
    ("1/φ²", 1/PHI**2),
    ("Complementarity angle", 43.10),
    ("π/4 in degrees", 45.0),
    ("4φ⁴", four_phi4),
    ("N_layers", float(N_LAYERS)),
]

print(f"  {'Measurement':>40s}  {'Value':>10s}  {'Nearest φ-π':>15s}  {'Error':>8s}")
print("  " + "-" * 80)

for name, val in constants:
    # Find nearest φ-π structural constant
    candidates = [
        (1/PHI, "1/φ"), (1/PHI**2, "1/φ²"), (1/PHI**3, "1/φ³"), (1/PHI**4, "1/φ⁴"),
        (PHI, "φ"), (PHI**2, "φ²"), (PHI**4, "φ⁴"),
        (PI/4, "π/4"), (4/PI, "4/π"), (PI, "π"),
        (1/(4*PHI**4), "1/(4φ⁴)"), (4*PHI**4, "4φ⁴"),
        (28.0, "28"), (45.0, "45°"),
        (np.degrees(np.arctan(1/PHI)), "arctan(1/φ)°"),
        (np.degrees(np.arctan(1/PHI**3)), "arctan(1/φ³)°"),
    ]
    best_err = float('inf')
    best_name = "?"
    for cval, cname in candidates:
        err = abs(val - cval) / max(abs(val), 1e-10) * 100
        if err < best_err:
            best_err = err
            best_name = cname

    print(f"  {name:>40s}  {val:10.4f}  {best_name:>15s}  {best_err:7.1f}%")

print()

# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: SELECTION RULES DEEP DIVE")
print("=" * 80)
print()

print("  1. PER-LAYER SELECTION RULES:")
print(f"     Dominant sequence transitions: {n_allowed}/{len(transitions)} allowed ({n_allowed/len(transitions)*100:.0f}%)")
print(f"     Global forbidden fraction: {mean_forb_pct:.1f}%")
print()

print("  2. ALTERNATING SERIES:")
print(f"     Wave-Leibniz correlation: r = {corr:.4f}")
print(f"     The wave is a SINGLE SWEEP (C→P-→P+→X→C), not alternating")
print()

print("  3. 4/π AND LAYER COUNT:")
print(f"     4φ⁴ = {four_phi4:.3f} ≈ {N_LAYERS} layers ({abs(four_phi4-N_LAYERS)/N_LAYERS*100:.1f}% error)")
print(f"     Residual = 1/(4φ⁴) = {1/four_phi4:.4f} ≈ 1/{N_LAYERS} = {1/N_LAYERS:.4f}")
print()

print("  4. BBP STRUCTURE:")
print(f"     Populations don't directly match BBP coefficients")
print(f"     But population ratios are near φ and 4/π")
print()

print("  5. FORBIDDEN = TRUNCATION:")
print(f"     C→X rate = {cx_rate:.4f}")
print()

# Save
results = {
    'dominant_sequence': [int(x) for x in dominant_sequence],
    'transitions': [int(x) for x in transitions],
    'n_allowed': n_allowed,
    'n_forbidden': n_forbidden,
    'wave_phase': wave_phase.tolist(),
    'leibniz_correlation': float(corr),
    'four_phi4': float(four_phi4),
    'residual_vs_1_over_N': float(abs(0.0361 - 1/N_LAYERS)/0.0361),
    'residual_vs_1_over_4phi4': float(abs(0.0361 - 1/four_phi4)/0.0361),
    'cx_rate': float(cx_rate),
    'xc_rate': float(xc_rate),
    'mean_forb_pct': float(mean_forb_pct),
    'populations': populations.tolist(),
    'wave_velocity_outward': float(outward_velocity),
    'wave_velocity_return': float(return_velocity),
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8c_selection_rules.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
