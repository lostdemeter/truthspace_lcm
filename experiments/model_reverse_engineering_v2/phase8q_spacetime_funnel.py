#!/usr/bin/env python3
"""
Phase 8q: The Spacetime Funnel
================================

From Finding 77: The cone WIDENS during generation. Each token adds
70-95% new directional information. D* grows linearly.

The user's insight: this looks like a spacetime funnel — expansion from
a singularity past an event horizon. W_gate compresses (3584 → through
the horizon), then the gate space EXPANDS (into 18944 dimensions).

Connection to spacetimezeta: the freefall speed in zeta spacetime
approaches φ (the golden ratio). We found φ-structure in the gate's
spectral gaps. Are these the same phenomenon?

Tests:
1. EXPANSION LAW: Does the cone expand linearly, exponentially, or
   as a power law? Measure effective volume, D*, and quality vs step.
2. PHI IN THE EXPANSION: Does the expansion rate relate to φ?
   Check ratios of consecutive quantities for golden ratio signatures.
3. THE HORIZON: W_gate maps 3584→18944 (5.29× expansion). Is this
   expansion ratio φ-structured? What happens at the boundary?
4. SPEED OF LIGHT: Is there a maximum rate at which new directions
   can be added? A "speed limit" on cone expansion?
5. FREEFALL PROFILE: Does the quality decay curve match a geodesic
   in a conformal metric g = e^(2φ) δ ?

Uses data from Phase 8p (multi-step generation captures).
Requires: Qwen2-7B on GPU
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
print("  PHASE 8q: THE SPACETIME FUNNEL")
print("  Does the widening cone follow spacetime expansion laws?")
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
HIDDEN_DIM = model.config.intermediate_size
HIDDEN_STATE_DIM = model.config.hidden_size


# ================================================================
# STEP 1: Build scaffold
# ================================================================
print("-" * 80)
print("  STEP 1: Build scaffold")
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
scaffold_single = all_gates_single.mean(axis=0)
h_mean_single = all_hs_single.mean(axis=0)

print(f"  Crystal: {len(train_words)} tokens")
print()


# ================================================================
# STEP 2: Generate with extended sequence (25 steps)
# ================================================================
print("-" * 80)
print("  STEP 2: Multi-step generation (25 steps for better statistics)")
print("-" * 80)

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "Albert Einstein developed the theory of",
    "The Pythagorean theorem states that",
    "Shakespeare wrote the play Romeo and",
    "The speed of light is approximately",
    "In mathematics, pi is approximately equal to",
]

N_GEN_STEPS = 25

all_gen_data = []

for prompt in PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_prompt = input_ids.shape[1]

    steps = []
    current_ids = input_ids.clone()

    for step in range(N_GEN_STEPS):
        n_total = current_ids.shape[1]

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

        for layer in range(N_LAYERS):
            h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
                make_gh(gate_storage, layer))
            h2 = model.model.layers[layer].mlp.register_forward_hook(
                make_hh(hs_storage, layer))
            hooks.extend([h1, h2])

        with torch.no_grad():
            out = model(current_ids)
            logits = out.logits[0, -1, :]

        for h in hooks:
            h.remove()

        next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        next_word = tokenizer.decode(next_token[0]).strip()

        steps.append({
            'step': step,
            'n_total': n_total,
            'next_token': next_word,
            'gates': {l: gate_storage[l].copy() for l in range(COMB_START, COMB_END)},
            'hs': {l: hs_storage[l].copy() for l in range(COMB_START, COMB_END)},
        })

        current_ids = torch.cat([current_ids, next_token], dim=1)

    generated = tokenizer.decode(current_ids[0][n_prompt:])
    print(f"  '{prompt}' -> '{generated[:60]}...'")

    all_gen_data.append({
        'prompt': prompt,
        'n_prompt': n_prompt,
        'generated': generated,
        'steps': steps,
    })

print()


# ================================================================
# Precompute W_gate for each COMB layer (reuse across analyses)
# ================================================================
W_gates = {}
for layer in range(COMB_START, COMB_END):
    W_gates[layer] = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()


# ================================================================
# ANALYSIS 1: THE EXPANSION LAW
# ================================================================
print("=" * 80)
print("  ANALYSIS 1: THE EXPANSION LAW")
print("  How does the cone expand with each generation step?")
print("=" * 80)
print()

layer = 14

# For each prompt, compute at each step:
# - D* (effective dimensionality)
# - Effective volume (product of top-D* singular values)
# - Total singular value sum (cone "energy")
# - Quality (explained fraction of last pos)
# - g_new (novelty of last position)

all_step_data = []

for gd in all_gen_data:
    steps = gd['steps']
    prompt_step_data = []

    for sd in steps:
        gates = sd['gates'][layer]
        hs = sd['hs'][layer]
        n_total = sd['n_total']

        h_mean = hs.mean(axis=0)
        h_shift = h_mean - h_mean_single[layer]
        scaffold = scaffold_single[layer] + W_gates[layer] @ h_shift

        context_resids = gates[:n_total-1] - scaffold[np.newaxis, :]
        last_resid = gates[n_total-1] - scaffold

        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)
        k = min(n_total - 1, Vt.shape[0])
        dirs_k = Vt[:k]

        # Quality
        proj = dirs_k.T @ (dirs_k @ last_resid)
        resid = last_resid - proj
        quality = 1.0 - (np.linalg.norm(resid) / (np.linalg.norm(last_resid) + 1e-10))
        g_new = np.linalg.norm(resid) / (np.linalg.norm(last_resid) + 1e-10)

        # D* at 90%
        total_var = np.sum(S ** 2)
        cum_var = np.cumsum(S ** 2) / (total_var + 1e-10)
        d_star = int(np.searchsorted(cum_var, 0.90) + 1)

        # Effective volume (geometric mean of top-D* singular values)
        if d_star > 0 and d_star <= len(S):
            eff_vol = np.exp(np.mean(np.log(S[:d_star] + 1e-10)))
        else:
            eff_vol = 0

        # Total energy
        total_energy = np.sum(S)

        # S_conc
        s_conc = S[0] / (total_energy + 1e-10)

        prompt_step_data.append({
            'step': sd['step'],
            'n_total': n_total,
            'quality': quality,
            'g_new': g_new,
            'd_star': d_star,
            'eff_vol': eff_vol,
            'total_energy': total_energy,
            's_conc': s_conc,
            'top_S': S[:min(5, len(S))].tolist(),
        })

    all_step_data.append(prompt_step_data)

# Print one representative prompt in detail
rep = all_step_data[0]
print(f"  '{all_gen_data[0]['prompt']}'  (representative)")
print(f"  {'Step':>5s}  {'n':>3s}  {'D*':>3s}  {'Quality':>8s}  {'g_new':>7s}  "
      f"{'EffVol':>8s}  {'Energy':>8s}  {'S_conc':>7s}")
print("  " + "-" * 60)
for d in rep:
    print(f"  {d['step']:5d}  {d['n_total']:3d}  {d['d_star']:3d}  {d['quality']:8.5f}  "
          f"{d['g_new']:7.5f}  {d['eff_vol']:8.2f}  {d['total_energy']:8.1f}  "
          f"{d['s_conc']:7.4f}")
print()


# ================================================================
# ANALYSIS 2: FIT THE EXPANSION
# ================================================================
print("=" * 80)
print("  ANALYSIS 2: FIT THE EXPANSION")
print("  Linear? Power law? Exponential? φ-related?")
print("=" * 80)
print()

# Aggregate across prompts
from collections import defaultdict
step_aggregates = defaultdict(lambda: {'quality': [], 'g_new': [], 'd_star': [],
                                        'eff_vol': [], 'energy': [], 's_conc': []})

for prompt_data in all_step_data:
    for d in prompt_data:
        s = d['step']
        step_aggregates[s]['quality'].append(d['quality'])
        step_aggregates[s]['g_new'].append(d['g_new'])
        step_aggregates[s]['d_star'].append(d['d_star'])
        step_aggregates[s]['eff_vol'].append(d['eff_vol'])
        step_aggregates[s]['energy'].append(d['total_energy'])
        step_aggregates[s]['s_conc'].append(d['s_conc'])

steps_arr = sorted(step_aggregates.keys())
mean_quality = np.array([np.mean(step_aggregates[s]['quality']) for s in steps_arr])
mean_g_new = np.array([np.mean(step_aggregates[s]['g_new']) for s in steps_arr])
mean_d_star = np.array([np.mean(step_aggregates[s]['d_star']) for s in steps_arr])
mean_energy = np.array([np.mean(step_aggregates[s]['energy']) for s in steps_arr])
mean_s_conc = np.array([np.mean(step_aggregates[s]['s_conc']) for s in steps_arr])

n_arr = np.array(steps_arr, dtype=float) + 1  # step number (1-indexed for fitting)

# Fit quality decay
# Try: quality = 1 - a*n^b  (power law decay from 1)
# And: quality = exp(-a*n)  (exponential decay)
# And: quality = 1 - a*log(n) (logarithmic)

quality_loss = 1.0 - mean_quality  # The "aberration"

# Linear fit: loss = a + b*step
from numpy.polynomial import polynomial as P
linear_coeffs = np.polyfit(n_arr, quality_loss, 1)
linear_pred = np.polyval(linear_coeffs, n_arr)
linear_r2 = 1 - np.sum((quality_loss - linear_pred)**2) / np.sum((quality_loss - np.mean(quality_loss))**2)

# Log fit: loss = a + b*log(step)
log_n = np.log(n_arr + 1)
log_coeffs = np.polyfit(log_n, quality_loss, 1)
log_pred = np.polyval(log_coeffs, log_n)
log_r2 = 1 - np.sum((quality_loss - log_pred)**2) / np.sum((quality_loss - np.mean(quality_loss))**2)

# Power fit: log(loss) = a + b*log(step)
valid = quality_loss > 0
if valid.sum() > 2:
    power_coeffs = np.polyfit(np.log(n_arr[valid]), np.log(quality_loss[valid]), 1)
    power_pred = np.exp(np.polyval(power_coeffs, np.log(n_arr)))
    power_r2 = 1 - np.sum((quality_loss[valid] - power_pred[valid])**2) / np.sum((quality_loss[valid] - np.mean(quality_loss[valid]))**2)
    power_exp = power_coeffs[0]
else:
    power_r2 = 0
    power_exp = 0

print(f"  Quality loss (1 - quality) fitting:")
print(f"    Linear:  R² = {linear_r2:.4f}  (loss = {linear_coeffs[0]:.6f}·step + {linear_coeffs[1]:.6f})")
print(f"    Log:     R² = {log_r2:.4f}  (loss = {log_coeffs[0]:.6f}·log(step) + {log_coeffs[1]:.6f})")
print(f"    Power:   R² = {power_r2:.4f}  (loss ~ step^{power_exp:.3f})")
print()

# What's the power exponent closest to?
print(f"  Power law exponent: {power_exp:.4f}")
print(f"    vs 1.0 (linear):    error = {abs(power_exp - 1.0):.4f}")
print(f"    vs 1/φ (0.618):     error = {abs(power_exp - 1/PHI):.4f}")
print(f"    vs φ-1 (0.618):     error = {abs(power_exp - (PHI-1)):.4f}")
print(f"    vs 1/2:             error = {abs(power_exp - 0.5):.4f}")
print(f"    vs log(φ) (0.481):  error = {abs(power_exp - LOG_PHI):.4f}")
print(f"    vs 2/3:             error = {abs(power_exp - 2/3):.4f}")
print(f"    vs √φ-1 (0.272):   error = {abs(power_exp - (SQRT_PHI-1)):.4f}")
print()


# ================================================================
# ANALYSIS 3: φ IN CONSECUTIVE RATIOS
# ================================================================
print("=" * 80)
print("  ANALYSIS 3: φ IN CONSECUTIVE RATIOS")
print("  Do consecutive values show golden ratio signatures?")
print("=" * 80)
print()

# Check ratios of consecutive quality losses
print(f"  Consecutive quality loss ratios (loss[n+1]/loss[n]):")
loss_ratios = []
for i in range(1, len(quality_loss)):
    if quality_loss[i-1] > 1e-8:
        ratio = quality_loss[i] / quality_loss[i-1]
        loss_ratios.append(ratio)
        print(f"    step {i-1}->{i}: {ratio:.4f}")

if loss_ratios:
    mean_ratio = np.mean(loss_ratios[2:])  # Skip first few (noisy)
    print(f"\n  Mean ratio (step 3+): {mean_ratio:.4f}")
    print(f"    vs φ   (1.618): error = {abs(mean_ratio - PHI):.4f}")
    print(f"    vs √φ  (1.272): error = {abs(mean_ratio - SQRT_PHI):.4f}")
    print(f"    vs 1+1/φ (same as φ): {abs(mean_ratio - PHI):.4f}")
    print(f"    vs 1.0 (constant): error = {abs(mean_ratio - 1.0):.4f}")
print()

# Check ratios of g_new
print(f"  Consecutive g_new ratios (g_new[n+1]/g_new[n]):")
gnew_ratios = []
for i in range(1, len(mean_g_new)):
    if mean_g_new[i-1] > 1e-8:
        ratio = mean_g_new[i] / mean_g_new[i-1]
        gnew_ratios.append(ratio)

if gnew_ratios:
    mean_gnew_ratio = np.mean(gnew_ratios[2:])
    print(f"  Mean g_new ratio (step 3+): {mean_gnew_ratio:.4f}")
    print(f"    vs φ:  error = {abs(mean_gnew_ratio - PHI):.4f}")
    print(f"    vs √φ: error = {abs(mean_gnew_ratio - SQRT_PHI):.4f}")
    print(f"    vs 1.0: error = {abs(mean_gnew_ratio - 1.0):.4f}")
print()

# Check S_conc decay
print(f"  S_conc decay ratios (s_conc[n]/s_conc[n-1]):")
sconc_ratios = []
for i in range(1, len(mean_s_conc)):
    if mean_s_conc[i] > 1e-8:
        ratio = mean_s_conc[i-1] / mean_s_conc[i]  # Inverted because it decays
        sconc_ratios.append(ratio)

if sconc_ratios:
    mean_sconc_ratio = np.mean(sconc_ratios[2:])
    print(f"  Mean S_conc expansion ratio (step 3+): {mean_sconc_ratio:.4f}")
    print(f"    vs φ:  error = {abs(mean_sconc_ratio - PHI):.4f}")
    print(f"    vs √φ: error = {abs(mean_sconc_ratio - SQRT_PHI):.4f}")
    print(f"    vs 1.0: error = {abs(mean_sconc_ratio - 1.0):.4f}")
print()


# ================================================================
# ANALYSIS 4: THE HORIZON — W_gate as event horizon
# ================================================================
print("=" * 80)
print("  ANALYSIS 4: THE HORIZON — W_gate expansion")
print("  Is the 3584→18944 expansion φ-structured?")
print("=" * 80)
print()

# W_gate maps hidden_state (3584) → gate (18944)
# The expansion ratio is 18944/3584 = 5.286

expansion_ratio = HIDDEN_DIM / HIDDEN_STATE_DIM
print(f"  W_gate expansion: {HIDDEN_STATE_DIM} → {HIDDEN_DIM}")
print(f"  Ratio: {expansion_ratio:.4f}")
print()

# Check φ relationships
print(f"  φ-structure of expansion ratio {expansion_ratio:.4f}:")
print(f"    φ³     = {PHI**3:.4f}  (error = {abs(expansion_ratio - PHI**3):.4f})")
print(f"    φ²+φ   = {PHI**2+PHI:.4f}  (error = {abs(expansion_ratio - (PHI**2+PHI)):.4f})")
print(f"    2φ²    = {2*PHI**2:.4f}  (error = {abs(expansion_ratio - 2*PHI**2):.4f})")
print(f"    3φ     = {3*PHI:.4f}  (error = {abs(expansion_ratio - 3*PHI):.4f})")
print(f"    φ³+1   = {PHI**3+1:.4f}  (error = {abs(expansion_ratio - (PHI**3+1)):.4f})")
print(f"    8/φ    = {8/PHI:.4f}  (error = {abs(expansion_ratio - 8/PHI):.4f})")
print(f"    (φ+2)² / φ = {(PHI+2)**2/PHI:.4f}  (error = {abs(expansion_ratio - (PHI+2)**2/PHI):.4f})")
print(f"    16/3   = {16/3:.4f}  (error = {abs(expansion_ratio - 16/3):.4f})")
print()

# Also check the actual dimensions for Fibonacci-like structure
print(f"  Dimension factorizations:")
print(f"    3584 = {3584} = 2^9 × 7 = 512 × 7")
print(f"    18944 = {18944} = 2^9 × 37 = 512 × 37")
print(f"    Ratio of non-power-of-2 parts: 37/7 = {37/7:.4f}")
print(f"    φ³ = {PHI**3:.4f}")
print(f"    37/7 vs φ³: error = {abs(37/7 - PHI**3):.4f} ({abs(37/7 - PHI**3)/PHI**3*100:.2f}%)")
print()


# ================================================================
# ANALYSIS 5: SPEED OF LIGHT — max expansion rate
# ================================================================
print("=" * 80)
print("  ANALYSIS 5: SPEED OF LIGHT")
print("  Is there a max rate at which new directions can be added?")
print("=" * 80)
print()

# For each step, compute how much "new volume" is added
# g_new = fraction of last pos that's orthogonal to cone
# If there's a speed limit, g_new should be bounded

# Also compute: the angle that each new token's CONTEXT POSITION
# (not last pos) makes with the existing cone, across all layers

layer = 14
for gd_idx, gd in enumerate(all_gen_data[:3]):
    steps = gd['steps']
    print(f"  '{gd['prompt'][:50]}'")

    new_dir_norms = []  # Norm of the new direction added at each step
    for i in range(1, len(steps)):
        sd = steps[i]
        gates = sd['gates'][layer]
        hs = sd['hs'][layer]
        n_total = sd['n_total']

        h_mean = hs.mean(axis=0)
        h_shift = h_mean - h_mean_single[layer]
        scaffold = scaffold_single[layer] + W_gates[layer] @ h_shift

        all_resids = gates - scaffold[np.newaxis, :]

        # The new position (index n_total-2) is what was generated at step i-1
        # All positions before it are "old"
        if n_total > 2:
            old_resids = all_resids[:n_total-2]
            new_resid = all_resids[n_total-2]  # The just-generated token's position

            U, S, Vt = np.linalg.svd(old_resids, full_matrices=False)
            k = min(old_resids.shape[0], Vt.shape[0])
            dirs_k = Vt[:k]

            proj = dirs_k.T @ (dirs_k @ new_resid)
            new_component = new_resid - proj
            new_norm = np.linalg.norm(new_component)
            total_norm = np.linalg.norm(new_resid)
            frac_new = new_norm / (total_norm + 1e-10)

            new_dir_norms.append(frac_new)

    if new_dir_norms:
        print(f"    New direction fraction per step: "
              f"{np.mean(new_dir_norms):.4f} ± {np.std(new_dir_norms):.4f}")
        print(f"    Range: [{min(new_dir_norms):.4f}, {max(new_dir_norms):.4f}]")
        print(f"    Trend: first={new_dir_norms[0]:.4f}, last={new_dir_norms[-1]:.4f}")

        # Is it bounded? Decreasing?
        early = np.mean(new_dir_norms[:5])
        late = np.mean(new_dir_norms[-5:]) if len(new_dir_norms) >= 5 else np.mean(new_dir_norms)
        print(f"    Early avg: {early:.4f}, Late avg: {late:.4f}")
        if late < early * 0.8:
            print(f"    >> DECELERATING: expansion slowing down")
        elif late > early * 1.2:
            print(f"    >> ACCELERATING: expansion speeding up")
        else:
            print(f"    >> CONSTANT SPEED: expansion rate stable")
    print()


# ================================================================
# ANALYSIS 6: CROSS-LAYER FUNNEL SHAPE
# ================================================================
print("=" * 80)
print("  ANALYSIS 6: CROSS-LAYER FUNNEL SHAPE")
print("  Does the funnel have a consistent shape across layers?")
print("=" * 80)
print()

# At step 0 (tightest) and step 24 (widest), compare the
# residual subspace volume across layers. This traces the
# "funnel" shape through the model's layer stack.

gd = all_gen_data[0]
steps = gd['steps']

print(f"  '{gd['prompt']}' funnel profile:")
print(f"  {'Layer':>7s}  {'Step0 D*':>9s}  {'Step24 D*':>10s}  {'Step0 S_conc':>12s}  {'Step24 S_conc':>13s}  {'Expansion':>10s}")
print("  " + "-" * 70)

funnel_data = {}

for layer in range(COMB_START, COMB_END):
    row = {}
    for si, step_idx in enumerate([0, -1]):
        sd = steps[step_idx]
        gates = sd['gates'][layer]
        hs = sd['hs'][layer]
        n_total = sd['n_total']

        h_mean = hs.mean(axis=0)
        h_shift = h_mean - h_mean_single[layer]
        scaffold = scaffold_single[layer] + W_gates[layer] @ h_shift

        context_resids = gates[:n_total-1] - scaffold[np.newaxis, :]
        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

        total_var = np.sum(S ** 2)
        cum_var = np.cumsum(S ** 2) / (total_var + 1e-10)
        d_star = int(np.searchsorted(cum_var, 0.90) + 1)
        s_conc = S[0] / (np.sum(S) + 1e-10)

        label = 'early' if si == 0 else 'late'
        row[f'd_star_{label}'] = d_star
        row[f's_conc_{label}'] = s_conc

    expansion = row['d_star_late'] / (row['d_star_early'] + 1e-10)
    funnel_data[layer] = row
    print(f"  {layer:7d}  {row['d_star_early']:9d}  {row['d_star_late']:10d}  "
          f"{row['s_conc_early']:12.4f}  {row['s_conc_late']:13.4f}  {expansion:10.2f}×")

print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: THE SPACETIME FUNNEL")
print("=" * 80)
print()

del model
torch.cuda.empty_cache()

results = {
    'n_prompts': len(PROMPTS),
    'n_gen_steps': N_GEN_STEPS,
    'expansion_ratio': float(expansion_ratio),
    'power_exponent': float(power_exp),
    'mean_quality': mean_quality.tolist(),
    'mean_g_new': mean_g_new.tolist(),
    'mean_d_star': mean_d_star.tolist(),
    'mean_s_conc': mean_s_conc.tolist(),
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8q_spacetime_funnel.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
