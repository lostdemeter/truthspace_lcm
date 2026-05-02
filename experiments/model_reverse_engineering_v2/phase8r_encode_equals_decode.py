#!/usr/bin/env python3
"""
Phase 8r: Encode = Decode
==========================

Core principle: encoding and decoding are the SAME operation in opposite
directions, like φ and 1/φ. If the geometry is commutative:

  h → W_gate → g   (encode: hidden → gate, expansion by ~2φ²)
  g → W_gate⁺ → h'  (decode: gate → hidden, compression by ~1/(2φ²))

Tests:
1. ROUND-TRIP: Does h → W_gate → W_gate⁺ → h' reconstruct h?
   And does g → W_gate⁺ → W_gate → g' reconstruct g?
2. INVERSE SCAFFOLD: What does W_gate⁺ · scaffold_gate look like
   in hidden space? Is it the hidden-space scaffold?
3. INVERSE RESIDUALS: Do gate residuals map back to meaningful
   hidden-state directions? Can we predict tokens from inverse space?
4. SVD SYMMETRY: Do W_gate's singular values show φ/1/φ structure?
5. RECIPROCAL FUNNEL: If forward expansion is step^(√φ-1), is the
   inverse compression step^(1-√φ)?
6. COMMUTATIVITY: Does the order of scaffold subtraction and
   W_gate projection matter? (scaffold in gate space vs hidden space)

If encode truly equals decode, the geometry should be self-dual.
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import json
import os

PHI = (1 + np.sqrt(5)) / 2
SQRT_PHI = np.sqrt(PHI)
LOG_PHI = np.log(PHI)

COMB_START = 6
COMB_END = 23

print("=" * 80)
print("  PHASE 8r: ENCODE = DECODE")
print("  If the geometry is commutative, inverse operations should")
print("  reveal the same structure from the opposite direction.")
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
GATE_DIM = model.config.intermediate_size   # 18944
HIDDEN_DIM = model.config.hidden_size       # 3584


# ================================================================
# STEP 0: Extract W_gate matrices and compute pseudo-inverses
# ================================================================
print("-" * 80)
print("  STEP 0: Extract W_gate and compute W_gate⁺ (pseudo-inverse)")
print("-" * 80)

W_gates = {}
W_gate_pinvs = {}
W_gate_svds = {}

for layer in range(COMB_START, COMB_END):
    W = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    W_gates[layer] = W  # Shape: (18944, 3584)

    # Full SVD of W_gate
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_gate_svds[layer] = (U, S, Vt)

    # Pseudo-inverse: W⁺ = V · S⁻¹ · Uᵀ
    S_inv = np.zeros_like(S)
    S_inv[S > 1e-6] = 1.0 / S[S > 1e-6]
    W_pinv = Vt.T @ np.diag(S_inv) @ U.T  # Shape: (3584, 18944)
    W_gate_pinvs[layer] = W_pinv

    if layer == 14:
        print(f"  Layer {layer}: W_gate shape = {W.shape}")
        print(f"  Layer {layer}: W_gate⁺ shape = {W_pinv.shape}")
        print(f"  Layer {layer}: rank = {np.sum(S > 1e-6)} / {min(W.shape)}")
        print(f"  Layer {layer}: condition number = {S[0]/S[-1]:.2f}")

print()


# ================================================================
# STEP 1: Build scaffold in both spaces
# ================================================================
print("-" * 80)
print("  STEP 1: Build scaffolds (gate space and hidden space)")
print("-" * 80)

TRAIN_WORDS = [
    "king", "queen", "man", "woman", "boy", "girl",
    "hot", "cold", "fast", "slow", "big", "small",
    "love", "hate", "light", "dark", "true", "false",
    "cat", "dog", "tree", "water", "fire", "earth",
    "happy", "sad", "strong", "weak", "old", "young",
    "the", "is", "and", "of", "to", "in",
    "zero", "one", "two", "three", "four", "five",
    "red", "blue", "green", "black", "white", "yellow",
    "algorithm", "quantum", "geometry", "neural", "vector", "matrix",
    "Paris", "London", "Tokyo", "Einstein", "Newton", "Euler",
    "hello", "world", "computer", "science", "language", "model",
]

single_gates = {}
single_hs = {}

for word in TRAIN_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    decoded = tokenizer.decode([token_id]).strip()
    if decoded in single_gates:
        continue

    gate_storage = {}
    hs_storage = {}
    hooks = []

    def make_gate_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hs_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(N_LAYERS):
        h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_gate_hook(gate_storage, layer)
        )
        h2 = model.model.layers[layer].mlp.register_forward_hook(
            make_hs_hook(hs_storage, layer)
        )
        hooks.extend([h1, h2])

    with torch.no_grad():
        model(torch.tensor([[token_id]], device="cuda"))

    for h in hooks:
        h.remove()

    single_gates[decoded] = np.stack([gate_storage[l] for l in range(N_LAYERS)])
    single_hs[decoded] = np.stack([hs_storage[l] for l in range(N_LAYERS)])

train_words = sorted(single_gates.keys())
all_gates_single = np.stack([single_gates[w] for w in train_words])
all_hs_single = np.stack([single_hs[w] for w in train_words])

# Scaffold in gate space (mean of all single-token gate outputs)
scaffold_gate = all_gates_single.mean(axis=0)   # (N_LAYERS, 18944)
# Scaffold in hidden space (mean of all single-token hidden states)
scaffold_hidden = all_hs_single.mean(axis=0)     # (N_LAYERS, 3584)

print(f"  Crystal: {len(train_words)} tokens")
print(f"  scaffold_gate shape: {scaffold_gate.shape}")
print(f"  scaffold_hidden shape: {scaffold_hidden.shape}")
print()


# ================================================================
# ANALYSIS 1: ROUND-TRIP TEST
# ================================================================
print("=" * 80)
print("  ANALYSIS 1: ROUND-TRIP TEST")
print("  h → W_gate → W_gate⁺ → h'  (does h' ≈ h?)")
print("  g → W_gate⁺ → W_gate → g'  (does g' ≈ g?)")
print("=" * 80)
print()

layer = 14

# Test with individual token hidden states and gates
print(f"  Layer {layer}:")
print(f"  {'Token':>12s}  {'h→g→h cos':>10s}  {'h→g→h err':>10s}  {'g→h→g cos':>10s}  {'g→h→g err':>10s}")
print("  " + "-" * 55)

round_trip_h = []
round_trip_g = []

for word in train_words[:20]:
    h = single_hs[word][layer]    # (3584,)
    g = single_gates[word][layer] # (18944,)

    # Forward round-trip: h → g' → h'
    g_from_h = W_gates[layer] @ h          # (18944,)
    h_roundtrip = W_gate_pinvs[layer] @ g_from_h  # (3584,)

    cos_h = np.dot(h, h_roundtrip) / (np.linalg.norm(h) * np.linalg.norm(h_roundtrip) + 1e-10)
    err_h = np.linalg.norm(h - h_roundtrip) / (np.linalg.norm(h) + 1e-10)

    # Inverse round-trip: g → h' → g'
    h_from_g = W_gate_pinvs[layer] @ g     # (3584,)
    g_roundtrip = W_gates[layer] @ h_from_g # (18944,)

    cos_g = np.dot(g, g_roundtrip) / (np.linalg.norm(g) * np.linalg.norm(g_roundtrip) + 1e-10)
    err_g = np.linalg.norm(g - g_roundtrip) / (np.linalg.norm(g) + 1e-10)

    round_trip_h.append(cos_h)
    round_trip_g.append(cos_g)

    print(f"  {word:>12s}  {cos_h:10.6f}  {err_h:10.6f}  {cos_g:10.6f}  {err_g:10.6f}")

print()
print(f"  Mean h→g→h cosine: {np.mean(round_trip_h):.6f}")
print(f"  Mean g→h→g cosine: {np.mean(round_trip_g):.6f}")
print()

# KEY QUESTION: Is the forward round-trip perfect?
# h → W_gate → W_gate⁺ should be perfect because W_gate has full row rank (3584 < 18944)
# g → W_gate⁺ → W_gate should NOT be perfect because gate space is bigger
# The difference is the NULL SPACE of W_gate — the directions in gate space
# that have no pre-image in hidden space. WHAT LIVES IN THAT NULL SPACE?

print(f"  Forward (h→g→h) should be PERFECT (W_gate has full column rank)")
print(f"  Inverse (g→h→g) loses the null space component of g")
print()

# Measure what fraction of g lives in the null space vs column space
null_fracs = []
for word in train_words[:20]:
    g = single_gates[word][layer]
    h_from_g = W_gate_pinvs[layer] @ g
    g_in_colspace = W_gates[layer] @ h_from_g
    g_in_nullspace = g - g_in_colspace

    frac_col = np.linalg.norm(g_in_colspace) / (np.linalg.norm(g) + 1e-10)
    frac_null = np.linalg.norm(g_in_nullspace) / (np.linalg.norm(g) + 1e-10)
    null_fracs.append(frac_null)

print(f"  Gate vector decomposition (layer {layer}):")
print(f"    In W_gate column space: {1 - np.mean(null_fracs):.4f} ({(1-np.mean(null_fracs))*100:.1f}%)")
print(f"    In null space:          {np.mean(null_fracs):.4f} ({np.mean(null_fracs)*100:.1f}%)")
print()


# ================================================================
# ANALYSIS 2: INVERSE SCAFFOLD
# ================================================================
print("=" * 80)
print("  ANALYSIS 2: INVERSE SCAFFOLD")
print("  W_gate⁺ · scaffold_gate  vs  scaffold_hidden")
print("  Are they the same object seen from different sides?")
print("=" * 80)
print()

print(f"  {'Layer':>5s}  {'Cosine':>8s}  {'Rel Error':>10s}  {'Gate→H norm':>12s}  {'H norm':>8s}")
print("  " + "-" * 50)

scaffold_cosines = []
for layer in range(COMB_START, COMB_END):
    # Project gate scaffold back to hidden space
    scaffold_gate_to_hidden = W_gate_pinvs[layer] @ scaffold_gate[layer]

    cos = np.dot(scaffold_gate_to_hidden, scaffold_hidden[layer]) / (
        np.linalg.norm(scaffold_gate_to_hidden) * np.linalg.norm(scaffold_hidden[layer]) + 1e-10)
    err = np.linalg.norm(scaffold_gate_to_hidden - scaffold_hidden[layer]) / (
        np.linalg.norm(scaffold_hidden[layer]) + 1e-10)

    scaffold_cosines.append(cos)
    print(f"  {layer:5d}  {cos:8.5f}  {err:10.5f}  "
          f"{np.linalg.norm(scaffold_gate_to_hidden):12.2f}  "
          f"{np.linalg.norm(scaffold_hidden[layer]):8.2f}")

print()
print(f"  Mean cosine: {np.mean(scaffold_cosines):.5f}")
print()

# Also test: does W_gate · scaffold_hidden = scaffold_gate?
print(f"  Forward test: W_gate · scaffold_hidden vs scaffold_gate")
print(f"  {'Layer':>5s}  {'Cosine':>8s}  {'Rel Error':>10s}")
print("  " + "-" * 30)

forward_cosines = []
for layer in range(COMB_START, COMB_END):
    scaffold_h_to_gate = W_gates[layer] @ scaffold_hidden[layer]
    cos = np.dot(scaffold_h_to_gate, scaffold_gate[layer]) / (
        np.linalg.norm(scaffold_h_to_gate) * np.linalg.norm(scaffold_gate[layer]) + 1e-10)
    err = np.linalg.norm(scaffold_h_to_gate - scaffold_gate[layer]) / (
        np.linalg.norm(scaffold_gate[layer]) + 1e-10)
    forward_cosines.append(cos)
    print(f"  {layer:5d}  {cos:8.5f}  {err:10.5f}")

print()
print(f"  Mean forward cosine: {np.mean(forward_cosines):.5f}")
print()


# ================================================================
# ANALYSIS 3: SVD STRUCTURE — φ/1/φ SYMMETRY
# ================================================================
print("=" * 80)
print("  ANALYSIS 3: SVD STRUCTURE OF W_gate")
print("  Do singular values show φ/1/φ symmetry?")
print("=" * 80)
print()

layer = 14
U, S, Vt = W_gate_svds[layer]

print(f"  Layer {layer}: {len(S)} singular values")
print(f"  Range: [{S[-1]:.4f}, {S[0]:.4f}]")
print(f"  Mean: {np.mean(S):.4f}, Median: {np.median(S):.4f}")
print()

# Check ratios of consecutive singular values
print(f"  Top 15 singular values and consecutive ratios:")
print(f"  {'i':>3s}  {'S[i]':>10s}  {'S[i]/S[i+1]':>12s}  {'closest φ':>12s}")
print("  " + "-" * 45)

phi_candidates = {
    'φ': PHI, '1/φ': 1/PHI, '√φ': SQRT_PHI, '1/√φ': 1/SQRT_PHI,
    'φ²': PHI**2, '1/φ²': 1/PHI**2, '1': 1.0,
}

for i in range(15):
    ratio = S[i] / S[i+1] if S[i+1] > 1e-10 else float('inf')
    best_name = min(phi_candidates, key=lambda k: abs(ratio - phi_candidates[k]))
    best_err = abs(ratio - phi_candidates[best_name])
    print(f"  {i:3d}  {S[i]:10.4f}  {ratio:12.6f}  {best_name:>6s} (err={best_err:.4f})")

print()

# Check if S[i] * S[n-1-i] is constant (palindrome symmetry)
n = len(S)
print(f"  Palindrome test: S[i] * S[{n}-1-i]")
products = []
for i in range(10):
    prod = S[i] * S[n-1-i]
    products.append(prod)
    print(f"    S[{i}] * S[{n-1-i}] = {S[i]:.4f} * {S[n-1-i]:.4f} = {prod:.4f}")

print(f"  Product range: [{min(products):.4f}, {max(products):.4f}]")
print(f"  Product CV: {np.std(products)/np.mean(products):.4f}")
print()

# Check the overall distribution shape
# If φ-structured, the singular values might follow S[i] = S[0] * φ^(-αi)
log_S = np.log(S + 1e-10)
indices = np.arange(len(S))
slope_coeffs = np.polyfit(indices[:100], log_S[:100], 1)
decay_rate = -slope_coeffs[0]

print(f"  Spectral decay rate (first 100): {decay_rate:.6f}")
print(f"    vs log(φ)/n: {LOG_PHI/len(S):.6f}")
print(f"    vs log(√φ)/n: {np.log(SQRT_PHI)/len(S):.6f}")
print(f"    × {len(S)} = {decay_rate * len(S):.4f}")
print(f"    vs log(φ) = {LOG_PHI:.4f} (error = {abs(decay_rate * len(S) - LOG_PHI):.4f})")
print(f"    vs log(√φ) = {np.log(SQRT_PHI):.4f} (error = {abs(decay_rate * len(S) - np.log(SQRT_PHI)):.4f})")
print()

# The bulk spectrum shape
quartiles = np.percentile(S, [25, 50, 75])
print(f"  Spectrum quartiles: Q1={quartiles[0]:.4f}, Q2={quartiles[1]:.4f}, Q3={quartiles[2]:.4f}")
print(f"  Q3/Q2 = {quartiles[2]/quartiles[1]:.4f}, Q2/Q1 = {quartiles[1]/quartiles[0]:.4f}")
print(f"  Interquartile ratio: {quartiles[2]/quartiles[0]:.4f}")
print(f"    vs φ: error = {abs(quartiles[2]/quartiles[0] - PHI):.4f}")
print(f"    vs √φ: error = {abs(quartiles[2]/quartiles[0] - SQRT_PHI):.4f}")
print()


# ================================================================
# ANALYSIS 4: INVERSE RESIDUALS — Can we predict from gate⁻¹?
# ================================================================
print("=" * 80)
print("  ANALYSIS 4: INVERSE RESIDUALS")
print("  Map gate residuals back to hidden space. Do they predict?")
print("=" * 80)
print()

# Use multi-token prompts
TEST_PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "Albert Einstein developed the theory of",
    "The speed of light is approximately",
    "In mathematics, pi is approximately equal to",
]

layer = 14

for prompt in TEST_PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tokens = input_ids.shape[1]

    gate_storage = {}
    hs_storage = {}
    hooks = []

    def make_gh(storage, li):
        def hook_fn(module, input, output):
            storage[li] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hh(storage, li):
        def hook_fn(module, input, output):
            storage[li] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for l in range(N_LAYERS):
        h1 = model.model.layers[l].mlp.gate_proj.register_forward_hook(make_gh(gate_storage, l))
        h2 = model.model.layers[l].mlp.register_forward_hook(make_hh(hs_storage, l))
        hooks.extend([h1, h2])

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :]
        pred_token = tokenizer.decode([torch.argmax(logits).item()])

    for h in hooks:
        h.remove()

    gates = gate_storage[layer]   # (n_tokens, 18944)
    hs = hs_storage[layer]        # (n_tokens, 3584)

    # Gate residuals (gate space)
    h_mean = hs.mean(axis=0)
    h_shift = h_mean - scaffold_hidden[layer]
    scaffold_shifted = scaffold_gate[layer] + W_gates[layer] @ h_shift

    gate_resids = gates - scaffold_shifted[np.newaxis, :]

    # Map gate residuals BACK to hidden space via pseudo-inverse
    inverse_resids = np.array([W_gate_pinvs[layer] @ gr for gr in gate_resids])

    # Hidden state residuals (directly in hidden space)
    hidden_resids = hs - scaffold_hidden[layer][np.newaxis, :]

    # Compare: are inverse gate residuals ≈ hidden residuals?
    cosines = []
    for i in range(n_tokens):
        cos = np.dot(inverse_resids[i], hidden_resids[i]) / (
            np.linalg.norm(inverse_resids[i]) * np.linalg.norm(hidden_resids[i]) + 1e-10)
        cosines.append(cos)

    # SVD of inverse residuals vs gate residuals
    U_gate, S_gate, Vt_gate = np.linalg.svd(gate_resids[:n_tokens-1], full_matrices=False)
    U_inv, S_inv, Vt_inv = np.linalg.svd(inverse_resids[:n_tokens-1], full_matrices=False)
    U_hid, S_hid, Vt_hid = np.linalg.svd(hidden_resids[:n_tokens-1], full_matrices=False)

    # Effective dimensionality at 90%
    def d_star(S):
        total = np.sum(S**2)
        cum = np.cumsum(S**2) / (total + 1e-10)
        return int(np.searchsorted(cum, 0.90) + 1)

    d_gate = d_star(S_gate)
    d_inv = d_star(S_inv)
    d_hid = d_star(S_hid)

    # Spectral ratios
    gate_ratio = S_gate[0] / S_gate[1] if len(S_gate) > 1 else float('inf')
    inv_ratio = S_inv[0] / S_inv[1] if len(S_inv) > 1 else float('inf')
    hid_ratio = S_hid[0] / S_hid[1] if len(S_hid) > 1 else float('inf')

    print(f"  '{prompt}' -> '{pred_token.strip()}'")
    print(f"    Inverse ↔ Hidden cosines: {[f'{c:.3f}' for c in cosines]}")
    print(f"    D* (gate={d_gate}, inverse={d_inv}, hidden={d_hid})")
    print(f"    S₀/S₁ (gate={gate_ratio:.3f}, inverse={inv_ratio:.3f}, hidden={hid_ratio:.3f})")
    print()


# ================================================================
# ANALYSIS 5: COMMUTATIVITY TEST
# ================================================================
print("=" * 80)
print("  ANALYSIS 5: COMMUTATIVITY")
print("  Does order matter? scaffold_subtract ∘ W_gate vs W_gate ∘ scaffold_subtract")
print("=" * 80)
print()

# Path A: hidden → subtract h_scaffold → W_gate → gate residual
# Path B: hidden → W_gate → subtract g_scaffold → gate residual
# If commutative, these should be identical (or proportional)

layer = 14
print(f"  Layer {layer}:")
print(f"  Path A: (h - scaffold_h) → W_gate")
print(f"  Path B: W_gate(h) - W_gate(scaffold_h)")
print()
print(f"  {'Token':>12s}  {'Cosine(A,B)':>12s}  {'||A-B||/||A||':>14s}")
print("  " + "-" * 42)

comm_cosines = []
for word in train_words[:20]:
    h = single_hs[word][layer]
    h_resid = h - scaffold_hidden[layer]

    # Path A: subtract first, then project
    pathA = W_gates[layer] @ h_resid

    # Path B: project first, then subtract
    g_from_h = W_gates[layer] @ h
    g_scaffold = W_gates[layer] @ scaffold_hidden[layer]
    pathB = g_from_h - g_scaffold

    cos = np.dot(pathA, pathB) / (np.linalg.norm(pathA) * np.linalg.norm(pathB) + 1e-10)
    err = np.linalg.norm(pathA - pathB) / (np.linalg.norm(pathA) + 1e-10)
    comm_cosines.append(cos)
    print(f"  {word:>12s}  {cos:12.8f}  {err:14.10f}")

print()
print(f"  Mean cosine: {np.mean(comm_cosines):.10f}")
print(f"  >> W_gate is LINEAR, so Path A = Path B exactly (commutativity holds for linear maps)")
print()

# But the ACTUAL scaffold in gate space uses mean of gate outputs, not W_gate · mean of hidden
# The difference is: scaffold_gate = mean(W_gate · h + nonlinear_stuff) ≠ W_gate · mean(h)
# This tests whether the scaffold IS a linear transform of hidden scaffold

print(f"  But is scaffold_gate = W_gate · scaffold_hidden?")
print(f"  (Tests whether nonlinearity breaks commutativity)")
print()
for layer in range(COMB_START, COMB_END):
    linear_scaffold = W_gates[layer] @ scaffold_hidden[layer]
    actual_scaffold = scaffold_gate[layer]

    cos = np.dot(linear_scaffold, actual_scaffold) / (
        np.linalg.norm(linear_scaffold) * np.linalg.norm(actual_scaffold) + 1e-10)
    err = np.linalg.norm(linear_scaffold - actual_scaffold) / (
        np.linalg.norm(actual_scaffold) + 1e-10)

    if layer == COMB_START or layer == 14 or layer == COMB_END - 1:
        print(f"  Layer {layer}: cos = {cos:.6f}, err = {err:.6f}")

print()


# ================================================================
# ANALYSIS 6: φ/1/φ DUALITY IN CONTENT
# ================================================================
print("=" * 80)
print("  ANALYSIS 6: φ/1/φ DUALITY")
print("  If encode ~ φ, does decode ~ 1/φ?")
print("  Check: do gate residual norms relate to hidden norms by φ?")
print("=" * 80)
print()

layer = 14
print(f"  Layer {layer}: norm ratios (gate_resid / hidden_resid)")
print(f"  {'Token':>12s}  {'||g_resid||':>12s}  {'||h_resid||':>12s}  {'Ratio':>8s}")
print("  " + "-" * 50)

norm_ratios = []
for word in train_words[:30]:
    g_resid = single_gates[word][layer] - scaffold_gate[layer]
    h_resid = single_hs[word][layer] - scaffold_hidden[layer]

    g_norm = np.linalg.norm(g_resid)
    h_norm = np.linalg.norm(h_resid)
    ratio = g_norm / (h_norm + 1e-10)
    norm_ratios.append(ratio)

    if word in train_words[:10]:
        print(f"  {word:>12s}  {g_norm:12.2f}  {h_norm:12.2f}  {ratio:8.4f}")

mean_ratio = np.mean(norm_ratios)
print(f"\n  Mean norm ratio: {mean_ratio:.4f} ± {np.std(norm_ratios):.4f}")
print(f"    vs √(18944/3584) = √({GATE_DIM}/{HIDDEN_DIM}) = {np.sqrt(GATE_DIM/HIDDEN_DIM):.4f}")
print(f"    vs √(2φ²) = {np.sqrt(2*PHI**2):.4f}")
print(f"    vs φ = {PHI:.4f}")
print(f"    vs √φ = {SQRT_PHI:.4f}")
print()

# Cross-layer norm ratios
print(f"  Cross-layer mean norm ratios:")
for layer in range(COMB_START, COMB_END):
    ratios = []
    for word in train_words[:30]:
        g_resid = single_gates[word][layer] - scaffold_gate[layer]
        h_resid = single_hs[word][layer] - scaffold_hidden[layer]
        ratios.append(np.linalg.norm(g_resid) / (np.linalg.norm(h_resid) + 1e-10))
    if layer == COMB_START or layer == 14 or layer == COMB_END - 1:
        print(f"    Layer {layer}: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

print()


# ================================================================
# ANALYSIS 7: WHAT LIVES IN THE NULL SPACE?
# ================================================================
print("=" * 80)
print("  ANALYSIS 7: WHAT LIVES IN THE NULL SPACE?")
print("  The gate space (18944) is bigger than hidden (3584).")
print("  15360 dimensions are in W_gate's null space.")
print("  What fraction of gate activations lives there?")
print("=" * 80)
print()

layer = 14

# For single tokens
print(f"  Layer {layer}: Single-token gate decomposition")
print(f"  {'Token':>12s}  {'||col_space||':>14s}  {'||null_space||':>14s}  {'frac_null':>10s}")
print("  " + "-" * 56)

null_fracs_all = []
for word in train_words[:15]:
    g = single_gates[word][layer]

    # Project into column space: W_gate @ W_gate⁺ @ g
    g_col = W_gates[layer] @ (W_gate_pinvs[layer] @ g)
    g_null = g - g_col

    frac_null = np.linalg.norm(g_null) / (np.linalg.norm(g) + 1e-10)
    null_fracs_all.append(frac_null)
    print(f"  {word:>12s}  {np.linalg.norm(g_col):14.2f}  {np.linalg.norm(g_null):14.2f}  {frac_null:10.4f}")

print(f"\n  Mean null-space fraction: {np.mean(null_fracs_all):.4f}")
print()

# Is the null space content the SCAFFOLD or the CONTENT?
print(f"  Null-space decomposition of scaffold vs content:")
g_scaffold = scaffold_gate[layer]
g_scaffold_col = W_gates[layer] @ (W_gate_pinvs[layer] @ g_scaffold)
g_scaffold_null = g_scaffold - g_scaffold_col
frac_null_scaffold = np.linalg.norm(g_scaffold_null) / (np.linalg.norm(g_scaffold) + 1e-10)

resid_null_fracs = []
for word in train_words[:30]:
    g_resid = single_gates[word][layer] - scaffold_gate[layer]
    g_resid_col = W_gates[layer] @ (W_gate_pinvs[layer] @ g_resid)
    g_resid_null = g_resid - g_resid_col
    resid_null_fracs.append(np.linalg.norm(g_resid_null) / (np.linalg.norm(g_resid) + 1e-10))

print(f"  Scaffold null-space fraction: {frac_null_scaffold:.4f}")
print(f"  Content (resid) null-space fraction: {np.mean(resid_null_fracs):.4f}")
print()

if frac_null_scaffold > np.mean(resid_null_fracs):
    print(f"  >> SCAFFOLD lives more in null space than CONTENT")
    print(f"  >> The null space carries the universal/scaffold signal")
else:
    print(f"  >> CONTENT lives more in null space than SCAFFOLD")
    print(f"  >> The null space carries token-specific information")

print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: ENCODE = DECODE")
print("=" * 80)
print()

del model
torch.cuda.empty_cache()

results = {
    'round_trip_h_cosine': float(np.mean(round_trip_h)),
    'round_trip_g_cosine': float(np.mean(round_trip_g)),
    'scaffold_inverse_cosines': [float(c) for c in scaffold_cosines],
    'scaffold_forward_cosines': [float(c) for c in forward_cosines],
    'commutativity_cosines': [float(c) for c in comm_cosines],
    'null_space_fraction': float(np.mean(null_fracs_all)),
    'scaffold_null_fraction': float(frac_null_scaffold),
    'content_null_fraction': float(np.mean(resid_null_fracs)),
    'norm_ratio_mean': float(mean_ratio),
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8r_encode_decode.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
