#!/usr/bin/env python3
"""
Is the 4-State Gate a Real Dimension? (Doc 253/254 + rharithmeticlight + spacetimezeta)

If the 4-state gate dimension (+1, -1, +0, -0) is genuine geometry — not just
a classification convenience — it must obey the same rules we find everywhere:

  Test A: Light-Cone Scaling
    Gate state fluctuations across layers should be BOUNDED when normalized
    by √layer, like G(t) = e^{-t/2}F(t) being bounded implies β ≤ 1/2.
    Analogy: layers are "multiplicative time", gate transitions are "prime events".

  Test B: Base-Collapse (Token Universality)
    Gate state distributions should collapse across tokens when parameterized
    by layer depth, just like primes collapse across numeral bases.

  Test C: Equidistribution Horizon
    The 4 states should equidistribute beyond some layer depth.
    Analogy: primes equidistribute mod q beyond horizon ≈ 2 log(q).

  Test D: φ-Structure in Transition Matrix
    The 4×4 transition matrix between gate states across layers should
    have φ-related eigenvalues or Zipf decay.

  Test E: Self-Similar 4-State (Fractal)
    If the dimension is subject to its own rules, there should be
    sub-4-state structure within each state.

Method: Run tokens through Qwen2-7B, capture pre-SiLU gate activations
at each of 28 layers, classify into 4 states at ±log(φ) boundaries.
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)  # ≈ 0.481

GATE_CONTRACT = 0    # x < -log(φ)
GATE_PRESERVE_N = 1  # -log(φ) ≤ x < 0
GATE_PRESERVE_P = 2  # 0 ≤ x < +log(φ)
GATE_EXPAND = 3      # x ≥ +log(φ)
STATE_NAMES = ['CONTRACT', 'PRESERVE-', 'PRESERVE+', 'EXPAND']

def classify_gate(x):
    """Classify pre-SiLU activations into 4 gate states."""
    codes = np.zeros_like(x, dtype=np.int8)
    codes[x < -LOG_PHI] = GATE_CONTRACT
    codes[(x >= -LOG_PHI) & (x < 0)] = GATE_PRESERVE_N
    codes[(x >= 0) & (x < LOG_PHI)] = GATE_PRESERVE_P
    codes[x >= LOG_PHI] = GATE_EXPAND
    return codes


# ================================================================
# Load model and capture gate activations
# ================================================================
print("=" * 80)
print("IS THE 4-STATE GATE A REAL DIMENSION?")
print("(rharithmeticlight + spacetimezeta applied to neural gate geometry)")
print("=" * 80)
print()

print("Loading Qwen2-7B...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

N_LAYERS = len(model.model.layers)
HIDDEN_DIM = model.config.intermediate_size  # 18944 for Qwen2-7B
print(f"  {N_LAYERS} layers, gate dim = {HIDDEN_DIM}")

# Test tokens: diverse semantic categories
TEST_WORDS = [
    "king", "queen", "man", "woman", "boy", "girl",
    "hot", "cold", "fast", "slow", "big", "small",
    "love", "hate", "light", "dark", "true", "false",
    "cat", "dog", "tree", "water", "fire", "earth",
    "happy", "sad", "strong", "weak", "old", "young",
    "the", "is", "and", "of", "to", "in",  # function words
    "zero", "one", "two", "three", "four", "five",  # numbers
    "red", "blue", "green", "black", "white", "yellow",  # colors
]

# Capture pre-SiLU gate activations at every layer
print(f"\nCapturing gate activations for {len(TEST_WORDS)} tokens across {N_LAYERS} layers...")

gate_activations = {}  # word -> [layer_0_gates, layer_1_gates, ...]

for word in TEST_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    decoded = tokenizer.decode([token_id]).strip()

    # Hook to capture gate_proj output (pre-SiLU)
    layer_gates = []
    hooks = []

    def make_hook(storage):
        def hook_fn(module, input, output):
            storage.append(output.detach().cpu().float().numpy())
        return hook_fn

    for layer_idx in range(N_LAYERS):
        storage = []
        layer_gates.append(storage)
        h = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(
            make_hook(storage)
        )
        hooks.append(h)

    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], device="cuda")
        model(input_ids)

    for h in hooks:
        h.remove()

    # Extract: each storage has one element of shape [1, 1, HIDDEN_DIM]
    gates = []
    for storage in layer_gates:
        g = storage[0].squeeze()  # [HIDDEN_DIM]
        gates.append(g)

    gate_activations[decoded] = np.stack(gates)  # [N_LAYERS, HIDDEN_DIM]

print(f"  Captured {len(gate_activations)} tokens")

# Free model memory
del model
torch.cuda.empty_cache()

# Classify all gate activations into 4 states
gate_codes = {}  # word -> [N_LAYERS, HIDDEN_DIM] int8
for word, gates in gate_activations.items():
    gate_codes[word] = classify_gate(gates)

all_words = sorted(gate_codes.keys())
N_TOKENS = len(all_words)

# Stack for bulk analysis: [N_TOKENS, N_LAYERS, HIDDEN_DIM]
all_codes = np.stack([gate_codes[w] for w in all_words])
all_gates = np.stack([gate_activations[w] for w in all_words])


# ================================================================
# TEST A: Light-Cone Scaling
# ================================================================
print()
print("=" * 80)
print("TEST A: LIGHT-CONE SCALING")
print("Do gate-state transitions respect a 'speed limit' across layers?")
print("=" * 80)
print()
print("Analogy: F(t) = prime counting fluctuation, G(t) = e^{-t/2}F(t)")
print("If bounded, β ≤ 1/2 — no 'tachyonic' modes in gate space.")
print()

# For each token and channel, count cumulative transitions
# F(l) = number of gate-state transitions from layer 0 to layer l
# G(l) = F(l) / √(l+1) — normalized by √layer

transition_counts = np.zeros((N_TOKENS, N_LAYERS), dtype=np.float64)
for l in range(1, N_LAYERS):
    transitions = (all_codes[:, l, :] != all_codes[:, l-1, :]).astype(np.float64)
    per_token_transitions = transitions.mean(axis=1)  # average over channels
    transition_counts[:, l] = transition_counts[:, l-1] + per_token_transitions

# Compute G(l) = F(l) / √(l+1) and H(l) = G(l) / (l+1)
G = np.zeros_like(transition_counts)
H = np.zeros_like(transition_counts)
for l in range(N_LAYERS):
    G[:, l] = transition_counts[:, l] / np.sqrt(l + 1)
    H[:, l] = G[:, l] / (l + 1)

# Average across tokens
F_mean = transition_counts.mean(axis=0)
G_mean = G.mean(axis=0)
H_mean = H.mean(axis=0)
G_std = G.std(axis=0)

print(f"{'Layer':<8} {'F(l) trans':<12} {'G(l)=F/√l':<12} {'H(l)=G/l':<12} {'G std':<10}")
print("-" * 54)
for l in range(N_LAYERS):
    print(f"  {l:<6} {F_mean[l]:<12.4f} {G_mean[l]:<12.4f} {H_mean[l]:<12.4f} {G_std[l]:<10.4f}")

# Is G(l) bounded? Check if it converges or grows
G_growth = G_mean[-1] / G_mean[max(1, N_LAYERS//2)]
G_late_std = G_std[N_LAYERS//2:].mean()
G_late_range = G_mean[N_LAYERS//2:].max() - G_mean[N_LAYERS//2:].min()

print()
print(f"  G growth (last / mid): {G_growth:.4f}")
print(f"  G late-half std:       {G_late_std:.4f}")
print(f"  G late-half range:     {G_late_range:.4f}")
print()
if G_growth < 1.5 and G_late_range < 0.3:
    print("  ★ G(l) is BOUNDED — gate transitions respect a speed limit!")
    print("    This is the arithmetic light cone: β ≤ 1/2 holds in gate space.")
    lightcone_result = "BOUNDED"
else:
    print(f"  → G(l) shows growth ({G_growth:.2f}×) — speed limit unclear.")
    lightcone_result = "UNCLEAR"


# ================================================================
# TEST B: Base-Collapse (Token Universality)
# ================================================================
print()
print("=" * 80)
print("TEST B: BASE-COLLAPSE (TOKEN UNIVERSALITY)")
print("Do gate state distributions collapse across tokens?")
print("=" * 80)
print()
print("Analogy: Primes mod b collapse across different bases.")
print("Here: gate state distributions should be universal across tokens.")
print()

# For each token and layer, compute distribution over 4 states
# Then check if distributions collapse across tokens

per_token_dist = np.zeros((N_TOKENS, N_LAYERS, 4), dtype=np.float64)
for t in range(N_TOKENS):
    for l in range(N_LAYERS):
        codes = all_codes[t, l, :]
        for s in range(4):
            per_token_dist[t, l, s] = (codes == s).mean()

# Mean distribution across tokens (the "universal" curve)
mean_dist = per_token_dist.mean(axis=0)  # [N_LAYERS, 4]

# Total variation distance from mean for each token
tv_distances = np.zeros((N_TOKENS, N_LAYERS))
for t in range(N_TOKENS):
    for l in range(N_LAYERS):
        tv_distances[t, l] = 0.5 * np.abs(per_token_dist[t, l] - mean_dist[l]).sum()

mean_tv = tv_distances.mean(axis=0)
collapse_rms = np.sqrt((tv_distances ** 2).mean())

print(f"{'Layer':<8} {'CONTRACT':<12} {'PRESERVE-':<12} {'PRESERVE+':<12} {'EXPAND':<12} {'TV from mean':<12}")
print("-" * 68)
for l in range(N_LAYERS):
    print(f"  {l:<6} {mean_dist[l,0]*100:>6.1f}%     {mean_dist[l,1]*100:>6.1f}%     "
          f"{mean_dist[l,2]*100:>6.1f}%     {mean_dist[l,3]*100:>6.1f}%     {mean_tv[l]:.4f}")

print()
print(f"  RMS collapse score: {collapse_rms:.4f}")
print(f"  (Lower = better collapse. Primes achieve ≈ 0.10)")
print()
if collapse_rms < 0.10:
    print("  ★ STRONG COLLAPSE — gate distributions are universal across tokens!")
    collapse_result = "STRONG"
elif collapse_rms < 0.20:
    print("  ★ MODERATE COLLAPSE — token-invariant structure detected.")
    collapse_result = "MODERATE"
else:
    print(f"  → Weak collapse (RMS={collapse_rms:.3f})")
    collapse_result = "WEAK"


# ================================================================
# TEST C: Equidistribution Horizon
# ================================================================
print()
print("=" * 80)
print("TEST C: EQUIDISTRIBUTION HORIZON")
print("At what layer do the 4 states reach equidistribution (25% each)?")
print("=" * 80)
print()
print("Analogy: Primes equidistribute mod q beyond horizon ≈ 2 log(q).")
print("4 states → q=4, predicted horizon ≈ 2 log(4) ≈ 2.77 (layer ~3?).")
print()

# Measure distance from uniform [0.25, 0.25, 0.25, 0.25]
uniform = np.array([0.25, 0.25, 0.25, 0.25])
equi_distance = np.zeros(N_LAYERS)
for l in range(N_LAYERS):
    equi_distance[l] = 0.5 * np.abs(mean_dist[l] - uniform).sum()

# Find horizon: first layer where distance < 0.10
horizon = -1
for l in range(N_LAYERS):
    if equi_distance[l] < 0.10:
        horizon = l
        break

print(f"  Layer-by-layer distance from equidistribution:")
for l in range(N_LAYERS):
    bar = "█" * int(equi_distance[l] * 100)
    marker = " ← HORIZON" if l == horizon else ""
    print(f"    L{l:02d}: {equi_distance[l]:.4f} {bar}{marker}")

print()
predicted_horizon = 2 * np.log(4)
print(f"  Predicted horizon (2 log 4): {predicted_horizon:.2f}")
print(f"  Observed horizon (TV < 0.10): {'layer ' + str(horizon) if horizon >= 0 else 'never reached'}")
print()

if horizon >= 0:
    print(f"  ★ Equidistribution horizon EXISTS at layer {horizon}")
    if abs(horizon - predicted_horizon) < 3:
        print(f"    And it's CLOSE to predicted 2 log(q) = {predicted_horizon:.1f}!")
    horizon_result = f"layer_{horizon}"
else:
    # Check if any layer gets close
    min_dist = equi_distance.min()
    min_layer = equi_distance.argmin()
    print(f"  → No equidistribution (min TV = {min_dist:.4f} at layer {min_layer})")
    print(f"    The 4 states do NOT equidistribute — the distribution is structured, not random.")
    horizon_result = "NEVER"


# ================================================================
# TEST D: φ-Structure in Transition Matrix
# ================================================================
print()
print("=" * 80)
print("TEST D: φ-STRUCTURE IN GATE STATE TRANSITIONS")
print("Does the 4×4 transition matrix have φ-eigenvalues?")
print("=" * 80)
print()
print("Analogy: Zeta spacetime geodesic freefall converges to φ ≈ 1.618.")
print("Here: gate-state transitions across layers should exhibit φ-ratios.")
print()

# Build transition matrix: P(state_j at l+1 | state_i at l)
# Aggregate across all tokens and all layer transitions
T = np.zeros((4, 4), dtype=np.float64)
for t in range(N_TOKENS):
    for l in range(N_LAYERS - 1):
        for d in range(all_codes.shape[2]):
            s_from = all_codes[t, l, d]
            s_to = all_codes[t, l+1, d]
            T[s_from, s_to] += 1

# Normalize rows
row_sums = T.sum(axis=1, keepdims=True)
T_norm = T / (row_sums + 1e-10)

print("  Global transition matrix P(to | from):")
print(f"  {'FROM \\ TO':<12}", end="")
for s in STATE_NAMES:
    print(f"{s:<12}", end="")
print()
print("  " + "-" * 60)
for i, name in enumerate(STATE_NAMES):
    print(f"  {name:<12}", end="")
    for j in range(4):
        print(f"{T_norm[i,j]*100:>6.1f}%     ", end="")
    print()

# Eigenvalue analysis
eigenvalues = np.linalg.eigvals(T_norm)
eigenvalues_sorted = sorted(np.abs(eigenvalues), reverse=True)

print()
print("  Eigenvalues (|λ|):")
for i, ev in enumerate(eigenvalues_sorted):
    print(f"    λ_{i}: {ev:.6f}")

# Check for φ-ratios
print()
print("  Eigenvalue ratios:")
phi_hits = 0
for i in range(len(eigenvalues_sorted) - 1):
    if eigenvalues_sorted[i+1] > 1e-10:
        ratio = eigenvalues_sorted[i] / eigenvalues_sorted[i+1]
        phi_diff = abs(ratio - PHI) / PHI * 100
        marker = " ≈ φ!" if phi_diff < 15 else ""
        if phi_diff < 15:
            phi_hits += 1
        print(f"    λ_{i}/λ_{i+1} = {ratio:.4f} ({phi_diff:.1f}% from φ){marker}")

# Self-transition dominance
diag = np.diag(T_norm)
print()
print("  Self-transition (persistence) rates:")
for i, name in enumerate(STATE_NAMES):
    print(f"    {name}: {diag[i]*100:.1f}%")

persistence_mean = diag.mean()
print(f"  Mean persistence: {persistence_mean*100:.1f}%")

if phi_hits > 0:
    print(f"\n  ★ {phi_hits} eigenvalue ratio(s) near φ — transition matrix has φ-structure!")
    transition_result = f"PHI_HITS_{phi_hits}"
else:
    # Check for other structure
    print(f"\n  → No direct φ-eigenvalue ratios, checking other structure...")
    # Zipf analysis of transition frequencies
    flat_T = T.flatten()
    flat_T_sorted = sorted(flat_T, reverse=True)
    if flat_T_sorted[1] > 0:
        zipf_ratio = flat_T_sorted[0] / flat_T_sorted[1]
        print(f"    Top transition / 2nd: {zipf_ratio:.4f} ({abs(zipf_ratio - PHI)/PHI*100:.1f}% from φ)")
    transition_result = "NO_PHI"


# ================================================================
# TEST D2: Per-Layer Transition Analysis
# ================================================================
print()
print("  Per-layer transition rate (fraction of channels changing state):")
per_layer_transition_rate = np.zeros(N_LAYERS - 1)
for l in range(N_LAYERS - 1):
    changes = (all_codes[:, l+1, :] != all_codes[:, l, :]).astype(np.float64)
    per_layer_transition_rate[l] = changes.mean()

print(f"  {'Layer→':<10} {'Rate':<10} {'Rate/prev':<10}")
print("  " + "-" * 30)
for l in range(N_LAYERS - 1):
    ratio_str = ""
    if l > 0 and per_layer_transition_rate[l-1] > 0.001:
        ratio = per_layer_transition_rate[l] / per_layer_transition_rate[l-1]
        phi_diff = abs(ratio - 1/PHI) / (1/PHI) * 100
        ratio_str = f"{ratio:.4f}"
        if phi_diff < 15:
            ratio_str += " ≈ 1/φ!"
    print(f"  {l:>2}→{l+1:<5}  {per_layer_transition_rate[l]:.4f}    {ratio_str}")


# ================================================================
# TEST E: Self-Similar 4-State (Fractal)
# ================================================================
print()
print("=" * 80)
print("TEST E: SELF-SIMILAR 4-STATE (FRACTAL)")
print("Within each state, does a sub-4-state structure emerge?")
print("=" * 80)
print()
print("If the 4-state dimension is subject to its own rules, subdividing")
print("each state at ±log(φ²) should reveal a nested 4-state pattern.")
print()

# For each primary state, look at the distribution of activations
# and subdivide using the NEXT φ-boundary: ±log(φ²) = ±2·log(φ)
LOG_PHI2 = 2 * LOG_PHI  # ≈ 0.962

for state_idx, state_name in enumerate(STATE_NAMES):
    # Collect all activations in this state
    mask = all_codes == state_idx
    state_vals = all_gates[mask]

    if len(state_vals) == 0:
        print(f"  {state_name}: no samples")
        continue

    # Define sub-boundaries based on state
    if state_idx == GATE_CONTRACT:
        # CONTRACT: x < -log(φ). Sub-boundaries at -log(φ²), -log(φ³)
        b1, b2, b3 = -3*LOG_PHI, -2*LOG_PHI, -LOG_PHI
        sub_labels = ['DEEP-C', 'MID-C', 'SHALLOW-C', '(edge)']
    elif state_idx == GATE_PRESERVE_N:
        # PRESERVE-: -log(φ) ≤ x < 0. Sub at -log(φ)/φ, -log(φ)/φ²
        b1 = -LOG_PHI
        b2 = -LOG_PHI / PHI
        b3 = -LOG_PHI / (PHI * PHI)
        sub_labels = ['(edge)', 'OUTER-P-', 'INNER-P-', 'CORE-P-']
    elif state_idx == GATE_PRESERVE_P:
        # PRESERVE+: 0 ≤ x < log(φ). Sub at log(φ)/φ², log(φ)/φ
        b1 = LOG_PHI / (PHI * PHI)
        b2 = LOG_PHI / PHI
        b3 = LOG_PHI
        sub_labels = ['CORE-P+', 'INNER-P+', 'OUTER-P+', '(edge)']
    else:
        # EXPAND: x ≥ log(φ). Sub at log(φ), log(φ²), log(φ³)
        b1, b2, b3 = LOG_PHI, 2*LOG_PHI, 3*LOG_PHI
        sub_labels = ['(edge)', 'SHALLOW-X', 'MID-X', 'DEEP-X']

    # Count sub-states
    sub_counts = np.zeros(4)
    sub_counts[0] = (state_vals < b1).sum()
    sub_counts[1] = ((state_vals >= b1) & (state_vals < b2)).sum()
    sub_counts[2] = ((state_vals >= b2) & (state_vals < b3)).sum()
    sub_counts[3] = (state_vals >= b3).sum()

    total = sub_counts.sum()
    sub_fracs = sub_counts / (total + 1e-10)

    print(f"  {state_name} ({total:.0f} samples):")
    for i, (label, frac) in enumerate(zip(sub_labels, sub_fracs)):
        bar = "█" * int(frac * 40)
        print(f"    {label:<12} {frac*100:>6.1f}% {bar}")

    # Check if sub-distribution has structure (not uniform, not degenerate)
    sub_entropy = -sum(f * np.log(f + 1e-10) for f in sub_fracs if f > 0)
    max_entropy = np.log(4)
    entropy_ratio = sub_entropy / max_entropy

    # Check for φ-ratio in sub-state boundaries
    if sub_fracs[0] > 0.01 and sub_fracs[1] > 0.01:
        ratio_01 = sub_fracs[0] / sub_fracs[1]
        phi_diff = abs(ratio_01 - PHI) / PHI * 100
        inv_phi_diff = abs(ratio_01 - 1/PHI) / (1/PHI) * 100
        if phi_diff < 20 or inv_phi_diff < 20:
            marker = f" ≈ {'φ' if phi_diff < inv_phi_diff else '1/φ'}!"
        else:
            marker = ""
        print(f"    Sub-ratio [0]/[1]: {ratio_01:.3f}{marker}")

    print(f"    Sub-entropy: {sub_entropy:.3f} / {max_entropy:.3f} = {entropy_ratio:.2f}")
    print()

# Check overall self-similarity: do the 4 primary states have φ-ratios?
print("  Primary state population ratios:")
total_per_state = np.zeros(4)
for s in range(4):
    total_per_state[s] = (all_codes == s).sum()

total_all = total_per_state.sum()
for s in range(4):
    print(f"    {STATE_NAMES[s]}: {total_per_state[s]/total_all*100:.1f}%")

# Sorted fractions and their ratios
sorted_fracs = sorted(total_per_state / total_all, reverse=True)
print()
print("  Sorted population ratios:")
for i in range(len(sorted_fracs) - 1):
    if sorted_fracs[i+1] > 0.001:
        ratio = sorted_fracs[i] / sorted_fracs[i+1]
        phi_diff = abs(ratio - PHI) / PHI * 100
        inv_phi_diff = abs(ratio - 1/PHI) / (1/PHI) * 100
        marker = ""
        if phi_diff < 15:
            marker = " ≈ φ!"
        elif inv_phi_diff < 15:
            marker = " ≈ 1/φ!"
        print(f"    [{i}]/[{i+1}] = {ratio:.4f} ({phi_diff:.1f}% from φ){marker}")


# ================================================================
# TEST F: Critical Line σ = 1/2
# ================================================================
print()
print("=" * 80)
print("TEST F: THE CRITICAL LINE σ = 1/2")
print("Does the gate dimension have a 'critical line' at the midpoint?")
print("=" * 80)
print()
print("In spacetimezeta, σ = 1/2 is where zeros lie.")
print("In our system, σ = 0.5 is the projection threshold.")
print("Question: Is there something special about the midpoint of each state?")
print()

# For each state, check if the median activation is at the geometric midpoint
for state_idx, state_name in enumerate(STATE_NAMES):
    mask = all_codes == state_idx
    state_vals = all_gates[mask]
    if len(state_vals) == 0:
        continue

    median_val = np.median(state_vals)
    mean_val = np.mean(state_vals)

    # What are the state boundaries?
    if state_idx == GATE_CONTRACT:
        bounds = (float('-inf'), -LOG_PHI)
        midpoint = -LOG_PHI * PHI  # geometric midpoint
    elif state_idx == GATE_PRESERVE_N:
        bounds = (-LOG_PHI, 0)
        midpoint = -LOG_PHI / 2  # arithmetic midpoint
    elif state_idx == GATE_PRESERVE_P:
        bounds = (0, LOG_PHI)
        midpoint = LOG_PHI / 2
    else:
        bounds = (LOG_PHI, float('inf'))
        midpoint = LOG_PHI * PHI

    # Fraction below median vs above (should be 50/50 by definition)
    # More interesting: fraction below geometric midpoint
    frac_below_mid = (state_vals < midpoint).mean()

    print(f"  {state_name}:")
    print(f"    Bounds: [{bounds[0]:.3f}, {bounds[1]:.3f})")
    print(f"    Median: {median_val:.4f}, Mean: {mean_val:.4f}")
    print(f"    Geometric midpoint: {midpoint:.4f}")
    print(f"    Fraction below midpoint: {frac_below_mid*100:.1f}%")

    # Check if the median splits at σ=1/2 analog
    if 0.45 < frac_below_mid < 0.55:
        print(f"    ★ Median ≈ geometric midpoint (balanced at σ=1/2 analog)")
    print()


# ================================================================
# SYNTHESIS
# ================================================================
print()
print("=" * 80)
print("SYNTHESIS: IS THE 4-STATE GATE A REAL DIMENSION?")
print("=" * 80)
print()
print(f"  Test A (Light-cone scaling):      {lightcone_result}")
print(f"  Test B (Base-collapse):           {collapse_result}")
print(f"  Test C (Equidistribution):        {horizon_result}")
print(f"  Test D (φ-transition structure):  {transition_result}")
print()

# Save results
results = {
    "n_tokens": N_TOKENS,
    "n_layers": N_LAYERS,
    "n_channels": int(all_codes.shape[2]),
    "test_A_lightcone": {
        "result": lightcone_result,
        "G_growth_ratio": float(G_growth),
        "G_late_range": float(G_late_range),
        "G_late_std": float(G_late_std),
    },
    "test_B_collapse": {
        "result": collapse_result,
        "rms_score": float(collapse_rms),
    },
    "test_C_horizon": {
        "result": horizon_result,
        "predicted": float(predicted_horizon),
        "equi_distances": equi_distance.tolist(),
    },
    "test_D_transitions": {
        "result": transition_result,
        "eigenvalues": [float(e) for e in eigenvalues_sorted],
        "persistence_rates": diag.tolist(),
        "transition_matrix": T_norm.tolist(),
    },
    "test_E_fractal": {
        "primary_distribution": (total_per_state / total_all).tolist(),
    },
    "per_layer_distribution": mean_dist.tolist(),
    "per_layer_transition_rate": per_layer_transition_rate.tolist(),
}

out_path = '/home/thorin/truthspace-lcm/experiments/model_reverse_engineering_v2/results/4state_dimension_test.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
