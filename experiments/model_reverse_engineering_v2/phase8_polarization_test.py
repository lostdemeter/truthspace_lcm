#!/usr/bin/env python3
"""
Phase 8: Polarization Physics of the Gate Dimension
====================================================

Three tests to determine if the 4-state gate dimension obeys polarization
physics and whether this enables embarrassingly parallel computation.

Test 1: Standing Wave Prediction
  - Compute mean gate state distribution per layer across tokens
  - For each token, predict gate states using the mean standing wave
  - Measure per-layer and per-channel prediction accuracy
  - Result: how much of the gate state is predictable (→ parallelizable)?

Test 2: Chirality Independence
  - Decompose channels into L (CONTRACT + PRESERVE+) and R (PRESERVE- + EXPAND)
  - Compute mutual information between L and R channel outputs
  - Test statistical independence of the two chirality channels
  - Result: can L and R be processed independently?

Test 3: Malus's Law Quantitative Fit
  - Compute per-layer transition matrices
  - Fit transition probabilities to cos²(θ) (Malus's Law)
  - Extract per-layer angles and verify complementarity
  - Result: is the gate dimension governed by polarization physics?

Requires: Qwen2-7B on GPU (captures gate activations via hooks)
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
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

# Chirality channels (cross-parity pairing from Finding 61)
CHANNEL_L = [GATE_CONTRACT, GATE_PRESERVE_P]   # (-1) + (+0) = 61.3%
CHANNEL_R = [GATE_PRESERVE_N, GATE_EXPAND]      # (-0) + (+1) = 38.7%


def classify_gate(x):
    """Classify pre-SiLU activations into 4 gate states."""
    codes = np.zeros_like(x, dtype=np.int8)
    codes[x < -LOG_PHI] = GATE_CONTRACT
    codes[(x >= -LOG_PHI) & (x < 0)] = GATE_PRESERVE_N
    codes[(x >= 0) & (x < LOG_PHI)] = GATE_PRESERVE_P
    codes[x >= LOG_PHI] = GATE_EXPAND
    return codes


def compute_distribution(codes, n_states=4):
    """Compute probability distribution over states."""
    counts = np.bincount(codes.flatten().astype(int), minlength=n_states)
    return counts / counts.sum()


def mutual_information(x_codes, y_codes, n_states=4):
    """Compute mutual information I(X;Y) between two discrete variables."""
    joint = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            joint[i, j] = np.sum((x_codes == i) & (y_codes == j))
    joint = joint / joint.sum()

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_states):
        for j in range(n_states):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return mi


def entropy(probs):
    """Shannon entropy in bits."""
    p = probs[probs > 0]
    return -np.sum(p * np.log2(p))


# ================================================================
# Load model and capture gate activations
# ================================================================
print("=" * 80)
print("  PHASE 8: POLARIZATION PHYSICS OF THE GATE DIMENSION")
print("  Standing Wave Prediction | Chirality Independence | Malus's Law")
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
HIDDEN_DIM = model.config.intermediate_size  # 18944
print(f"  {N_LAYERS} layers, gate dim = {HIDDEN_DIM}")

# Expanded token set — more diverse than Finding 61 to stress-test universality
TEST_WORDS = [
    # Semantic pairs (opposites)
    "king", "queen", "man", "woman", "boy", "girl",
    "hot", "cold", "fast", "slow", "big", "small",
    "love", "hate", "light", "dark", "true", "false",
    "cat", "dog", "tree", "water", "fire", "earth",
    "happy", "sad", "strong", "weak", "old", "young",
    # Function words
    "the", "is", "and", "of", "to", "in",
    # Numbers
    "zero", "one", "two", "three", "four", "five",
    # Colors
    "red", "blue", "green", "black", "white", "yellow",
    # Technical terms (stress test — unusual tokens)
    "algorithm", "quantum", "geometry", "neural", "vector", "matrix",
    # Proper nouns
    "Paris", "London", "Tokyo", "Einstein", "Newton", "Euler",
    # Punctuation-adjacent (single tokens)
    "hello", "world", "computer", "science", "language", "model",
]

print(f"\nCapturing gate activations for {len(TEST_WORDS)} tokens across {N_LAYERS} layers...")

gate_activations = {}  # word -> [N_LAYERS, HIDDEN_DIM]

for word in TEST_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    decoded = tokenizer.decode([token_id]).strip()

    if decoded in gate_activations:
        continue  # skip duplicates

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

    gates = []
    for storage in layer_gates:
        g = storage[0].squeeze()  # [HIDDEN_DIM]
        gates.append(g)

    gate_activations[decoded] = np.stack(gates)  # [N_LAYERS, HIDDEN_DIM]

print(f"  Captured {len(gate_activations)} unique tokens")

# Free GPU
del model
torch.cuda.empty_cache()

# Classify
gate_codes = {}
for word, gates in gate_activations.items():
    gate_codes[word] = classify_gate(gates)

all_words = sorted(gate_codes.keys())
N_TOKENS = len(all_words)

# Stack: [N_TOKENS, N_LAYERS, HIDDEN_DIM]
all_codes = np.stack([gate_codes[w] for w in all_words])
all_gates = np.stack([gate_activations[w] for w in all_words])

print(f"  Shape: {all_codes.shape} (tokens × layers × channels)")
print()

results = {}

# ================================================================
# TEST 1: Standing Wave Prediction
# ================================================================
print("─" * 80)
print("  TEST 1: STANDING WAVE PREDICTION")
print("  Can the mean gate distribution predict per-token gate states?")
print("─" * 80)
print()

# 1a: Per-layer distribution (the standing wave)
mean_dist = np.zeros((N_LAYERS, 4))
for layer in range(N_LAYERS):
    for state in range(4):
        mean_dist[layer, state] = (all_codes[:, layer, :] == state).mean()

print("  Standing Wave (mean gate distribution per layer):")
print(f"  {'Layer':>5s}  {'CONTRACT':>9s}  {'PRESERVE-':>9s}  {'PRESERVE+':>9s}  {'EXPAND':>9s}  {'Dominant':>10s}")
print("  " + "-" * 60)
for layer in range(N_LAYERS):
    dom = STATE_NAMES[np.argmax(mean_dist[layer])]
    print(f"  {layer:5d}  {mean_dist[layer,0]:9.4f}  {mean_dist[layer,1]:9.4f}  "
          f"{mean_dist[layer,2]:9.4f}  {mean_dist[layer,3]:9.4f}  {dom:>10s}")
print()

# 1b: Per-channel prediction — can we predict the gate state of each channel
# at each layer using the most common state for that channel across tokens?
per_channel_mode = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.int8)
per_channel_mode_accuracy = np.zeros((N_LAYERS, HIDDEN_DIM))

for layer in range(N_LAYERS):
    for ch in range(HIDDEN_DIM):
        channel_codes = all_codes[:, layer, ch]  # [N_TOKENS]
        counts = np.bincount(channel_codes.astype(int), minlength=4)
        mode = counts.argmax()
        per_channel_mode[layer, ch] = mode
        per_channel_mode_accuracy[layer, ch] = counts[mode] / N_TOKENS

# 1c: Per-token prediction error using channel modes
per_token_accuracy = np.zeros((N_TOKENS, N_LAYERS))
for tok_idx in range(N_TOKENS):
    for layer in range(N_LAYERS):
        predicted = per_channel_mode[layer]
        actual = all_codes[tok_idx, layer]
        per_token_accuracy[tok_idx, layer] = (predicted == actual).mean()

# Summary
mean_layer_accuracy = per_token_accuracy.mean(axis=0)  # [N_LAYERS]
overall_accuracy = per_token_accuracy.mean()

print("  Per-Layer Channel-Mode Prediction Accuracy:")
print(f"  {'Layer':>5s}  {'Accuracy':>10s}  {'Error':>8s}  {'Quality':>10s}")
print("  " + "-" * 40)
for layer in range(N_LAYERS):
    acc = mean_layer_accuracy[layer]
    err = 1 - acc
    quality = "★ PARALLEL" if acc > 0.98 else ("  good" if acc > 0.95 else "  needs seq")
    print(f"  {layer:5d}  {acc:10.6f}  {err:8.4f}  {quality}")

print()
print(f"  Overall prediction accuracy: {overall_accuracy:.6f}")
print(f"  Overall prediction error:    {1 - overall_accuracy:.6f}")
print(f"  Fraction of channels needing sequential processing: {1 - overall_accuracy:.4f}")
print()

# Per-token variance (how much do individual tokens deviate from the mean?)
per_token_mean_acc = per_token_accuracy.mean(axis=1)  # [N_TOKENS]
token_acc_std = per_token_mean_acc.std()
worst_token_idx = per_token_mean_acc.argmin()
best_token_idx = per_token_mean_acc.argmax()

print(f"  Per-token accuracy: mean={per_token_mean_acc.mean():.6f} "
      f"± {token_acc_std:.6f}")
print(f"  Best token:  {all_words[best_token_idx]:>15s} ({per_token_mean_acc[best_token_idx]:.6f})")
print(f"  Worst token: {all_words[worst_token_idx]:>15s} ({per_token_mean_acc[worst_token_idx]:.6f})")
print()

results['test1_standing_wave'] = {
    'overall_accuracy': float(overall_accuracy),
    'overall_error': float(1 - overall_accuracy),
    'per_layer_accuracy': mean_layer_accuracy.tolist(),
    'per_token_mean_accuracy': float(per_token_mean_acc.mean()),
    'per_token_std': float(token_acc_std),
    'best_token': all_words[best_token_idx],
    'worst_token': all_words[worst_token_idx],
    'mean_distribution': mean_dist.tolist(),
}

# ================================================================
# TEST 2: CHIRALITY INDEPENDENCE
# ================================================================
print("─" * 80)
print("  TEST 2: CHIRALITY INDEPENDENCE")
print("  Do L (CONTRACT+PRESERVE+) and R (PRESERVE-+EXPAND) carry")
print("  independent information?")
print("─" * 80)
print()

# For each layer, compute mutual information between L-channel and R-channel
# gate states for adjacent channels

# Approach: For each layer, take the raw gate activations and split channels
# into L-type and R-type based on their gate code. Then measure MI between
# the L-type activation pattern and R-type activation pattern across tokens.

mi_per_layer = np.zeros(N_LAYERS)
h_l_per_layer = np.zeros(N_LAYERS)
h_r_per_layer = np.zeros(N_LAYERS)
l_fraction_per_layer = np.zeros(N_LAYERS)

for layer in range(N_LAYERS):
    # For each token, classify channels into L and R
    # Use the per-channel mode to determine "canonical" chirality
    channel_chirality = np.zeros(HIDDEN_DIM, dtype=int)  # 0=L, 1=R
    for ch in range(HIDDEN_DIM):
        mode = per_channel_mode[layer, ch]
        channel_chirality[ch] = 0 if mode in CHANNEL_L else 1

    n_L = (channel_chirality == 0).sum()
    n_R = (channel_chirality == 1).sum()
    l_fraction_per_layer[layer] = n_L / HIDDEN_DIM

    # Compute MI between L-channel and R-channel distributions across tokens
    # For each token, compute the distribution over states for L channels and R channels
    # Then measure MI(L_distribution, R_distribution) across the token population

    # Simplified: for each token, compute a 2-class summary of L and R channels
    # L summary: fraction of L-channels that are actually in their "expected" L state
    # R summary: fraction of R-channels that are actually in their "expected" R state
    l_conformity = np.zeros(N_TOKENS)
    r_conformity = np.zeros(N_TOKENS)

    for tok_idx in range(N_TOKENS):
        tok_codes = all_codes[tok_idx, layer]  # [HIDDEN_DIM]
        l_mask = channel_chirality == 0
        r_mask = channel_chirality == 1

        if l_mask.sum() > 0:
            l_codes = tok_codes[l_mask]
            l_in_l = np.isin(l_codes, CHANNEL_L).mean()
            l_conformity[tok_idx] = l_in_l
        if r_mask.sum() > 0:
            r_codes = tok_codes[r_mask]
            r_in_r = np.isin(r_codes, CHANNEL_R).mean()
            r_conformity[tok_idx] = r_in_r

    # Discretize conformity into bins for MI calculation
    n_bins = 10
    l_binned = np.clip(np.floor(l_conformity * n_bins).astype(int), 0, n_bins - 1)
    r_binned = np.clip(np.floor(r_conformity * n_bins).astype(int), 0, n_bins - 1)

    # Joint distribution
    joint = np.zeros((n_bins, n_bins))
    for tok_idx in range(N_TOKENS):
        joint[l_binned[tok_idx], r_binned[tok_idx]] += 1
    joint /= joint.sum()

    pl = joint.sum(axis=1)
    pr = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and pl[i] > 0 and pr[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (pl[i] * pr[j]))

    h_l = entropy(pl)
    h_r = entropy(pr)

    mi_per_layer[layer] = mi
    h_l_per_layer[layer] = h_l
    h_r_per_layer[layer] = h_r

print("  Per-Layer Chirality Analysis:")
print(f"  {'Layer':>5s}  {'L frac':>8s}  {'MI(L;R)':>8s}  {'H(L)':>6s}  {'H(R)':>6s}  {'MI/H':>6s}  {'Independent?':>12s}")
print("  " + "-" * 65)
for layer in range(N_LAYERS):
    mi = mi_per_layer[layer]
    h_total = h_l_per_layer[layer] + h_r_per_layer[layer]
    mi_frac = mi / h_total if h_total > 0 else 0
    independent = "✓ YES" if mi_frac < 0.10 else ("~ weak" if mi_frac < 0.25 else "✗ NO")
    print(f"  {layer:5d}  {l_fraction_per_layer[layer]:8.4f}  {mi:8.4f}  "
          f"{h_l_per_layer[layer]:6.3f}  {h_r_per_layer[layer]:6.3f}  "
          f"{mi_frac:6.3f}  {independent:>12s}")

mean_mi_frac = np.mean([mi_per_layer[l] / max(h_l_per_layer[l] + h_r_per_layer[l], 1e-10)
                         for l in range(N_LAYERS)])
print()
print(f"  Mean MI/H ratio: {mean_mi_frac:.4f}")
print(f"  → L and R channels share {mean_mi_frac*100:.1f}% of their information")
print(f"  → {(1-mean_mi_frac)*100:.1f}% is independent (target: >90%)")
print()

# Cross-parity population verification
print("  Cross-Parity Population (Finding 61 verification with expanded token set):")
global_dist = compute_distribution(all_codes)
cp_L = global_dist[GATE_CONTRACT] + global_dist[GATE_PRESERVE_P]
cp_R = global_dist[GATE_PRESERVE_N] + global_dist[GATE_EXPAND]
print(f"  Channel L (C + P+): {cp_L:.4f}  (1/φ = {1/PHI:.4f}, error = {abs(cp_L - 1/PHI)/((1/PHI))*100:.2f}%)")
print(f"  Channel R (P- + X): {cp_R:.4f}  (1/φ² = {1/PHI**2:.4f}, error = {abs(cp_R - 1/PHI**2)/(1/PHI**2)*100:.2f}%)")
print()

results['test2_chirality'] = {
    'mi_per_layer': mi_per_layer.tolist(),
    'h_l_per_layer': h_l_per_layer.tolist(),
    'h_r_per_layer': h_r_per_layer.tolist(),
    'l_fraction_per_layer': l_fraction_per_layer.tolist(),
    'mean_mi_fraction': float(mean_mi_frac),
    'cross_parity_L': float(cp_L),
    'cross_parity_R': float(cp_R),
    'phi_target_L': float(1/PHI),
    'phi_target_R': float(1/PHI**2),
}

# ================================================================
# TEST 3: MALUS'S LAW QUANTITATIVE FIT
# ================================================================
print("─" * 80)
print("  TEST 3: MALUS'S LAW QUANTITATIVE FIT")
print("  Do transition probabilities follow cos²(θ) at φ-determined angles?")
print("─" * 80)
print()

# 3a: Per-layer transition matrices
per_layer_tm = np.zeros((N_LAYERS - 1, 4, 4))

for layer in range(N_LAYERS - 1):
    for tok_idx in range(N_TOKENS):
        current = all_codes[tok_idx, layer]
        next_state = all_codes[tok_idx, layer + 1]
        for ch in range(HIDDEN_DIM):
            per_layer_tm[layer, current[ch], next_state[ch]] += 1

    # Normalize rows
    for state in range(4):
        row_sum = per_layer_tm[layer, state].sum()
        if row_sum > 0:
            per_layer_tm[layer, state] /= row_sum

# 3b: Extract persistence rates per layer and fit to cos²(θ)
persistence_C = per_layer_tm[:, GATE_CONTRACT, GATE_CONTRACT]  # [N_LAYERS-1]
persistence_PN = per_layer_tm[:, GATE_PRESERVE_N, GATE_PRESERVE_N]
persistence_PP = per_layer_tm[:, GATE_PRESERVE_P, GATE_PRESERVE_P]
persistence_X = per_layer_tm[:, GATE_EXPAND, GATE_EXPAND]

# Malus's Law: P = cos²(θ) → θ = arccos(√P)
# For valid Malus angles, P must be in [0, 1]
def malus_angle(persistence):
    """Convert persistence rate to Malus angle (degrees)."""
    p = np.clip(persistence, 0, 1)
    return np.degrees(np.arccos(np.sqrt(p)))

angles_C = malus_angle(persistence_C)
angles_PN = malus_angle(persistence_PN)
angles_PP = malus_angle(persistence_PP)
angles_X = malus_angle(persistence_X)

# Expected: θ_C + θ_P = 90° (complementary)
complementarity = angles_C + (angles_PN + angles_PP) / 2

print("  Per-Layer Malus Angles (from persistence rates):")
print(f"  {'Layer':>5s}  {'P(C→C)':>8s}  {'θ_C':>6s}  {'P(P→P)':>8s}  {'θ_P':>6s}  "
      f"{'θ_C+θ_P':>8s}  {'90°?':>5s}")
print("  " + "-" * 60)

complementarity_errors = []
for layer in range(N_LAYERS - 1):
    p_c = persistence_C[layer]
    p_p = (persistence_PN[layer] + persistence_PP[layer]) / 2
    theta_c = angles_C[layer]
    theta_p = malus_angle(p_p)
    comp = theta_c + theta_p
    comp_error = abs(comp - 90.0)
    complementarity_errors.append(comp_error)

    comp_mark = "✓" if comp_error < 5 else ("~" if comp_error < 15 else "✗")
    print(f"  {layer:5d}  {p_c:8.4f}  {theta_c:6.1f}°  {p_p:8.4f}  {theta_p:6.1f}°  "
          f"{comp:8.1f}°  {comp_mark:>5s}")

print()

# Focus on COMB layers (6-22) where the standing wave is active
comb_layers = list(range(6, 23))
comb_comp_errors = [complementarity_errors[l] for l in comb_layers if l < len(complementarity_errors)]
mean_comb_comp_error = np.mean(comb_comp_errors)

print(f"  COMB layers (6-22) complementarity error: {mean_comb_comp_error:.2f}° mean")
print(f"  Expected: θ_C + θ_P = 90° (Malus complementarity)")
print()

# 3c: Check if CONTRACT→EXPAND direct rate matches cos²(~79°) = cos²(θ_C + θ_P - 90°)?
# Actually, check if C→X ≈ cos²(θ_CX) where θ_CX = θ_C + θ_P (they're "crossed")
direct_CX = per_layer_tm[:, GATE_CONTRACT, GATE_EXPAND]  # [N_LAYERS-1]
print("  CONTRACT → EXPAND Direct Transition (\"Crossed Polarizer\" Rate):")
print(f"  {'Layer':>5s}  {'P(C→X)':>8s}  {'Expected cos²':>13s}  {'Match':>6s}")
print("  " + "-" * 40)

for layer in comb_layers:
    if layer >= len(direct_CX):
        continue
    p_cx = direct_CX[layer]
    # "Crossed" angle is θ_C + θ_P from midpoint
    theta_cx = angles_C[layer] + malus_angle((persistence_PN[layer] + persistence_PP[layer]) / 2)
    expected_cx = np.cos(np.radians(theta_cx))**2
    match = "✓" if abs(p_cx - expected_cx) < 0.03 else "~"
    print(f"  {layer:5d}  {p_cx:8.4f}  {expected_cx:13.4f}  {match:>6s}")

print()

# 3d: Global Malus fit — do ALL 16 transition probabilities follow cos²?
print("  Global Malus's Law Fit (all 16 transition probabilities):")
print()

# For the global transition matrix, assign each state an "angle"
# CONTRACT=0, PRESERVE-=θ₁, PRESERVE+=θ₂, EXPAND=θ₃
# and check if T[i,j] ∝ cos²(θ_i - θ_j) for some angles

global_tm = np.zeros((4, 4))
for layer in range(N_LAYERS - 1):
    for tok_idx in range(N_TOKENS):
        for ch in range(HIDDEN_DIM):
            i = all_codes[tok_idx, layer, ch]
            j = all_codes[tok_idx, layer + 1, ch]
            global_tm[i, j] += 1

for state in range(4):
    row_sum = global_tm[state].sum()
    if row_sum > 0:
        global_tm[state] /= row_sum

print("  Global Transition Matrix:")
print(f"  {'':>12s}  {'→ C':>8s}  {'→ P-':>8s}  {'→ P+':>8s}  {'→ X':>8s}")
for i in range(4):
    print(f"  {STATE_NAMES[i]:>12s}  {global_tm[i,0]:8.4f}  {global_tm[i,1]:8.4f}  "
          f"{global_tm[i,2]:8.4f}  {global_tm[i,3]:8.4f}")
print()

# Optimize angles to fit cos² model
# T[i,j] = cos²(θ_i - θ_j) × row_normalization
from scipy.optimize import minimize

def malus_model(angles, n_states=4):
    """Generate transition matrix from Malus's Law with given angles."""
    T = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            T[i, j] = np.cos(np.radians(angles[i] - angles[j]))**2
        T[i] /= T[i].sum()  # normalize rows
    return T

def malus_loss(angles_flat):
    """MSE between model and observed transition matrix."""
    angles = np.array([0.0, angles_flat[0], angles_flat[1], angles_flat[2]])
    T_model = malus_model(angles)
    return np.sum((T_model - global_tm)**2)

# Optimize: fix CONTRACT=0°, fit 3 other angles
result = minimize(malus_loss, [30, 45, 70], method='Nelder-Mead')
best_angles = np.array([0.0, result.x[0], result.x[1], result.x[2]])
T_fit = malus_model(best_angles)
fit_mse = result.fun
fit_r2 = 1 - fit_mse / np.var(global_tm) / 16

print(f"  Best-fit Malus angles (CONTRACT = 0°):")
for i in range(4):
    print(f"    {STATE_NAMES[i]:>12s}: {best_angles[i]:+7.2f}°")
print()

# Check complementarity
theta_C_fit = abs(best_angles[0])
theta_P_fit = (abs(best_angles[1]) + abs(best_angles[2])) / 2
print(f"  θ_CONTRACT = {theta_C_fit:.2f}°")
print(f"  θ_PRESERVE (mean) = {theta_P_fit:.2f}°")
print(f"  θ_C + θ_P = {theta_C_fit + theta_P_fit:.2f}° (target: ~90°)")
print()

print(f"  Fitted vs Observed Transition Matrix:")
print(f"  {'':>12s}  {'Obs→C':>8s} {'Fit→C':>8s}  {'Obs→P-':>8s} {'Fit→P-':>8s}  "
      f"{'Obs→P+':>8s} {'Fit→P+':>8s}  {'Obs→X':>8s} {'Fit→X':>8s}")
for i in range(4):
    print(f"  {STATE_NAMES[i]:>12s}  "
          f"{global_tm[i,0]:8.4f} {T_fit[i,0]:8.4f}  "
          f"{global_tm[i,1]:8.4f} {T_fit[i,1]:8.4f}  "
          f"{global_tm[i,2]:8.4f} {T_fit[i,2]:8.4f}  "
          f"{global_tm[i,3]:8.4f} {T_fit[i,3]:8.4f}")

print()
print(f"  Malus fit MSE: {fit_mse:.6f}")
residuals = np.abs(T_fit - global_tm)
max_residual = residuals.max()
mean_residual = residuals.mean()
print(f"  Mean |residual|: {mean_residual:.4f}")
print(f"  Max |residual|:  {max_residual:.4f}")
print()

# Check φ-structure in fitted angles
print("  φ-Structure in Fitted Angles:")
for i in range(1, 4):
    angle_ratio = abs(best_angles[i]) / abs(best_angles[1]) if abs(best_angles[1]) > 0 else 0
    phi_check = abs(angle_ratio - PHI) / PHI * 100 if angle_ratio > 0 else 999
    phi2_check = abs(angle_ratio - PHI**2) / PHI**2 * 100 if angle_ratio > 0 else 999
    print(f"    θ_{STATE_NAMES[i]} / θ_P- = {angle_ratio:.4f}  "
          f"(φ={PHI:.4f}, err={phi_check:.1f}%)  "
          f"(φ²={PHI**2:.4f}, err={phi2_check:.1f}%)")
print()

results['test3_malus'] = {
    'global_transition_matrix': global_tm.tolist(),
    'fitted_angles': best_angles.tolist(),
    'fitted_transition_matrix': T_fit.tolist(),
    'fit_mse': float(fit_mse),
    'mean_residual': float(mean_residual),
    'max_residual': float(max_residual),
    'complementarity_errors_comb': comb_comp_errors,
    'mean_comb_complementarity_error': float(mean_comb_comp_error),
    'per_layer_persistence_C': persistence_C.tolist(),
    'per_layer_persistence_PN': persistence_PN.tolist(),
    'per_layer_persistence_PP': persistence_PP.tolist(),
    'per_layer_persistence_X': persistence_X.tolist(),
}

# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: POLARIZATION PHYSICS OF THE GATE DIMENSION")
print("=" * 80)
print()

print("  Test 1: Standing Wave Prediction")
print(f"    Overall channel-mode accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"    Parallelizable fraction:       {overall_accuracy:.4f}")
print(f"    Sequential residual:           {1-overall_accuracy:.4f} ({(1-overall_accuracy)*100:.2f}%)")
print()

print("  Test 2: Chirality Independence")
print(f"    Mean MI/H ratio:  {mean_mi_frac:.4f} ({mean_mi_frac*100:.1f}% shared)")
print(f"    Independence:     {(1-mean_mi_frac)*100:.1f}%")
print(f"    Cross-parity L:   {cp_L:.4f} (1/φ = {1/PHI:.4f})")
print(f"    Cross-parity R:   {cp_R:.4f}")
print()

print("  Test 3: Malus's Law")
print(f"    Fit mean residual:  {mean_residual:.4f}")
print(f"    Fit max residual:   {max_residual:.4f}")
print(f"    COMB complementarity error: {mean_comb_comp_error:.2f}°")
print(f"    Fitted angles: C={best_angles[0]:.1f}° P-={best_angles[1]:.1f}° "
      f"P+={best_angles[2]:.1f}° X={best_angles[3]:.1f}°")
print()

# Verdicts
v1 = "CONFIRMED" if overall_accuracy > 0.95 else ("PARTIAL" if overall_accuracy > 0.90 else "WEAK")
v2 = "CONFIRMED" if mean_mi_frac < 0.10 else ("PARTIAL" if mean_mi_frac < 0.25 else "WEAK")
v3 = "CONFIRMED" if mean_residual < 0.02 else ("PARTIAL" if mean_residual < 0.05 else "WEAK")

print(f"  VERDICT:")
print(f"    Standing wave parallelism:  {v1}")
print(f"    Chirality independence:     {v2}")
print(f"    Malus's Law:                {v3}")
print()

# Save results
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, 'phase8_polarization_test.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to {results_path}")
print()
