#!/usr/bin/env python3
"""
Phase 8j: The Fourth Dimension
================================

Finding 70 showed: rank 1 through 20 ALL give exactly 50%. The ceiling is
NOT about needing more crystal modes - it's about a missing DIMENSION.

The user's geometric insight:
  - 2D triangle: angles sum to 180 degrees
  - 3D tetrahedron: 720 = 4 x 180 (each face is a triangle)
  - 4D: additional constraint that our 3D model misses

Our 3D model: scaffold + direction + alpha = 3 components
The 4-state gate: +1, -1, +0, -0 = 2 axes = needs 4 components

The hypothesis:
  After stereo scaffold correction + rank-1 content, the RESIDUAL
  is the 4th dimension signal. It should:
  1. Separate succeeding vs failing prompts
  2. Correlate with the 4-state gate boundary (+/-0 distinction)
  3. Be capturable as ONE more scalar (beta) per token per layer

If scaffold + alpha*direction + beta*direction2 gives 100%, we've found
the 4th dimension. The crystal needs one more vibrational mode that
lives in the sign/boundary space.

Requires: Qwen2-7B on GPU
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import json
import os

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

COMB_START = 6
COMB_END = 23

GATE_CONTRACT = 0
GATE_PRESERVE_N = 1
GATE_PRESERVE_P = 2
GATE_EXPAND = 3
STATE_NAMES = ['CONTRACT(-1)', 'PRESERVE-(-0)', 'PRESERVE+(+0)', 'EXPAND(+1)']

def classify_gate(x):
    codes = np.zeros_like(x, dtype=np.int8)
    codes[x < -LOG_PHI] = GATE_CONTRACT
    codes[(x >= -LOG_PHI) & (x < 0)] = GATE_PRESERVE_N
    codes[(x >= 0) & (x < LOG_PHI)] = GATE_PRESERVE_P
    codes[x >= LOG_PHI] = GATE_EXPAND
    return codes


print("=" * 80)
print("  PHASE 8j: THE FOURTH DIMENSION")
print("  If 50% = missing one dimension, find it.")
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
# STEP 1: Build 3D crystal from single tokens
# ================================================================
print("-" * 80)
print("  STEP 1: Build 3D crystal (scaffold + direction + alpha)")
print("-" * 80)
print()

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
N_TRAIN = len(train_words)
all_gates = np.stack([single_gates[w] for w in train_words])
all_hs = np.stack([single_hs[w] for w in train_words])

scaffold_single = all_gates.mean(axis=0)
h_mean_single = all_hs.mean(axis=0)

residuals_single = all_gates - scaffold_single[np.newaxis, :, :]
svd_per_layer = {}
for layer in range(COMB_START, COMB_END):
    res = residuals_single[:, layer, :]
    U, S, Vt = np.linalg.svd(res, full_matrices=False)
    svd_per_layer[layer] = {'U': U, 'S': S, 'Vt': Vt}

print(f"  Crystal built: {N_TRAIN} training tokens, {COMB_END - COMB_START} COMB layers")
print()


# ================================================================
# STEP 2: Capture multi-token prompts
# ================================================================
print("-" * 80)
print("  STEP 2: Capture multi-token prompts")
print("-" * 80)
print()

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "The speed of light is approximately",
    "In mathematics, pi is approximately equal to",
    "The chemical symbol for gold is",
    "One plus one equals",
    "The color of the sky is",
    "To solve a quadratic equation you can use the",
    "Albert Einstein developed the theory of",
]

prompt_data = []
for prompt in PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tok = input_ids.shape[1]

    with torch.no_grad():
        base_out = model(input_ids)
        base_logits = base_out.logits[0, -1, :].cpu().float().numpy()

    gate_storage = {}
    hs_storage = {}
    hooks = []

    def make_gate_hook2(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hs_hook2(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(N_LAYERS):
        h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_gate_hook2(gate_storage, layer)
        )
        h2 = model.model.layers[layer].mlp.register_forward_hook(
            make_hs_hook2(hs_storage, layer)
        )
        hooks.extend([h1, h2])

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    prompt_data.append({
        'prompt': prompt,
        'n_tokens': n_tok,
        'base_logits': base_logits,
        'gates': {l: gate_storage[l] for l in range(N_LAYERS)},
        'hs': {l: hs_storage[l] for l in range(N_LAYERS)},
    })
    print(f"  \"{prompt}\" -- {n_tok} tokens")

print()


# ================================================================
# STEP 3: Extract the 4th dimension residual
# ================================================================
print("=" * 80)
print("  STEP 3: EXTRACT THE 4TH DIMENSION")
print("  Residual after stereo scaffold + rank-1 = the missing dimension")
print("=" * 80)
print()

# For each prompt, at each COMB layer, compute the residual at the LAST
# token position after applying stereo scaffold + rank-1 content
residuals_4d = {}  # prompt_idx -> {layer: residual_vector}
rank1_reconstructions = {}

for pi, pd in enumerate(prompt_data):
    residuals_4d[pi] = {}
    rank1_reconstructions[pi] = {}

    for layer in range(COMB_START, COMB_END):
        gates_all_pos = pd['gates'][layer]
        hs_all_pos = pd['hs'][layer]

        # Stereo scaffold correction
        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold_corrected = scaffold_single[layer] + W_gate @ h_shift

        # Rank-1 content using crystal mode 1
        direction = svd_per_layer[layer]['Vt'][0]  # [HIDDEN_DIM]

        # For the LAST token position
        last_pos = gates_all_pos.shape[0] - 1
        gate_true = gates_all_pos[last_pos]  # [HIDDEN_DIM]

        # Rank-1 reconstruction at last position
        residual_from_scaffold = gate_true - scaffold_corrected
        alpha = np.dot(residual_from_scaffold, direction)
        rank1_recon = scaffold_corrected + alpha * direction

        # The 4th dimension residual
        residual_4d = gate_true - rank1_recon
        residuals_4d[pi][layer] = residual_4d
        rank1_reconstructions[pi][layer] = rank1_recon


# ================================================================
# TEST 1: Does the residual separate success vs failure?
# ================================================================
print("-" * 80)
print("  TEST 1: Does the 4th dim residual separate success vs failure?")
print("-" * 80)
print()

# First, determine which prompts succeed and which fail (from Finding 70)
def make_replace_hook(replacement):
    def hook_fn(module, input, output):
        rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
        return rep.reshape(output.shape)
    return hook_fn

successes = []
failures = []

for pi, pd in enumerate(prompt_data):
    input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

    intervened_gates = {}
    for layer in range(COMB_START, COMB_END):
        gates_all_pos = pd['gates'][layer]
        hs_all_pos = pd['hs'][layer]

        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold_corrected = scaffold_single[layer] + W_gate @ h_shift

        direction = svd_per_layer[layer]['Vt'][0]
        residuals = gates_all_pos - scaffold_corrected[np.newaxis, :]
        alphas = residuals @ direction
        reconstruction = scaffold_corrected + np.outer(alphas, direction)
        intervened_gates[layer] = reconstruction

    hooks = []
    for layer in range(COMB_START, COMB_END):
        h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_replace_hook(intervened_gates[layer])
        )
        hooks.append(h)

    with torch.no_grad():
        int_out = model(input_ids)
        int_logits = int_out.logits[0, -1, :].cpu().float().numpy()

    for h in hooks:
        h.remove()

    base_top1 = np.argmax(pd['base_logits'])
    int_top1 = np.argmax(int_logits)
    match = base_top1 == int_top1

    if match:
        successes.append(pi)
    else:
        failures.append(pi)

print(f"  Successes ({len(successes)}): {[prompt_data[i]['prompt'][:30] for i in successes]}")
print(f"  Failures  ({len(failures)}): {[prompt_data[i]['prompt'][:30] for i in failures]}")
print()

# Compare residual magnitudes
for layer in [COMB_START, 14, COMB_END - 1]:
    success_norms = [np.linalg.norm(residuals_4d[pi][layer]) for pi in successes]
    failure_norms = [np.linalg.norm(residuals_4d[pi][layer]) for pi in failures]
    print(f"  Layer {layer}: success residual norm = {np.mean(success_norms):.2f} +/- {np.std(success_norms):.2f}")
    print(f"            failure residual norm = {np.mean(failure_norms):.2f} +/- {np.std(failure_norms):.2f}")
    print(f"            ratio fail/success = {np.mean(failure_norms)/np.mean(success_norms):.2f}")
    print()


# ================================================================
# TEST 2: SVD of the 4th-dimension residuals across prompts
# ================================================================
print("-" * 80)
print("  TEST 2: SVD of the 4th-dimension residuals")
print("  Is there a shared direction in the residual space?")
print("-" * 80)
print()

print(f"  {'Layer':>7s}  {'S0':>8s}  {'S1':>8s}  {'S0/S1':>8s}  {'%var top-1':>12s}  {'S0/S1 vs sqrt(phi)':>18s}")
print("  " + "-" * 70)

svd_4d_per_layer = {}
for layer in range(COMB_START, COMB_END):
    # Stack residuals from ALL prompts (last position only)
    res_stack = np.stack([residuals_4d[pi][layer] for pi in range(len(prompt_data))])

    U, S, Vt = np.linalg.svd(res_stack, full_matrices=False)
    svd_4d_per_layer[layer] = {'U': U, 'S': S, 'Vt': Vt}

    total_var = np.sum(S ** 2)
    top1_var = S[0] ** 2 / total_var * 100
    ratio = S[0] / S[1] if S[1] > 0 else float('inf')
    sqrt_phi = np.sqrt(PHI)
    phi_err = abs(ratio - sqrt_phi) / sqrt_phi * 100

    print(f"  {layer:7d}  {S[0]:8.2f}  {S[1]:8.2f}  {ratio:8.3f}  {top1_var:11.1f}%  {phi_err:10.1f}% from sqrt(phi)")

mean_ratio = np.mean([svd_4d_per_layer[l]['S'][0] / svd_4d_per_layer[l]['S'][1]
                       for l in range(COMB_START, COMB_END)
                       if svd_4d_per_layer[l]['S'][1] > 0])
mean_top1_var = np.mean([svd_4d_per_layer[l]['S'][0]**2 / np.sum(svd_4d_per_layer[l]['S']**2) * 100
                          for l in range(COMB_START, COMB_END)])
print()
print(f"  Mean S0/S1 = {mean_ratio:.3f} (sqrt(phi) = {np.sqrt(PHI):.3f})")
print(f"  Mean top-1 variance = {mean_top1_var:.1f}%")
print()


# ================================================================
# TEST 3: Does the 4th dimension correlate with 4-state boundary?
# ================================================================
print("-" * 80)
print("  TEST 3: Does the 4th dim correlate with 4-state gate boundary?")
print("  The +/-0 distinction from Finding 61")
print("-" * 80)
print()

# For each prompt at each COMB layer, check: does the 4th dim residual
# preferentially affect channels near the SiLU boundary (the +0/-0 zone)?
for layer in [COMB_START, 14, COMB_END - 1]:
    boundary_effects = []
    nonboundary_effects = []

    for pi in range(len(prompt_data)):
        gate_true = prompt_data[pi]['gates'][layer]
        last_pos = gate_true.shape[0] - 1
        gate_last = gate_true[last_pos]
        residual = residuals_4d[pi][layer]

        codes = classify_gate(gate_last)
        boundary_mask = (codes == GATE_PRESERVE_N) | (codes == GATE_PRESERVE_P)
        nonboundary_mask = (codes == GATE_CONTRACT) | (codes == GATE_EXPAND)

        if boundary_mask.sum() > 0:
            boundary_effects.append(np.abs(residual[boundary_mask]).mean())
        if nonboundary_mask.sum() > 0:
            nonboundary_effects.append(np.abs(residual[nonboundary_mask]).mean())

    mean_boundary = np.mean(boundary_effects)
    mean_nonboundary = np.mean(nonboundary_effects)
    ratio = mean_boundary / mean_nonboundary if mean_nonboundary > 0 else float('inf')
    print(f"  Layer {layer}:")
    print(f"    Boundary channels (+/-0):     mean |residual| = {mean_boundary:.4f}")
    print(f"    Non-boundary channels (+/-1): mean |residual| = {mean_nonboundary:.4f}")
    print(f"    Ratio boundary/non-boundary = {ratio:.3f}")

    # Check sign flips: does the residual flip channels across the 0 boundary?
    sign_flips = []
    for pi in range(len(prompt_data)):
        gate_true = prompt_data[pi]['gates'][layer]
        last_pos = gate_true.shape[0] - 1
        gate_last = gate_true[last_pos]
        recon = rank1_reconstructions[pi][layer]
        recon_last = recon[last_pos] if recon.ndim > 1 else recon

        # Sign of true gate vs rank-1 reconstruction
        true_sign = np.sign(gate_last)
        recon_sign = np.sign(recon_last)
        sign_disagreement = (true_sign != recon_sign).mean()
        sign_flips.append(sign_disagreement)

    print(f"    Sign disagreement (true vs rank-1): {np.mean(sign_flips):.3%}")
    print()


# ================================================================
# TEST 4: INTERVENTION - Add the 4th dimension
# ================================================================
print("=" * 80)
print("  TEST 4: INTERVENTION - Adding the 4th dimension")
print("  scaffold + alpha*dir1 + beta*dir2 (where dir2 = 4th dim SVD)")
print("=" * 80)
print()

# Strategy: use the top SVD direction of the 4th-dim residuals as dir2
# For each prompt, project the residual onto dir2 to get beta

print("  A) Oracle 4th dim (true residual projected onto per-layer SVD direction):")
print(f"  {'Prompt':>45s}  {'Cos':>9s}  {'Top1?':>6s}  {'Base':>10s}  {'Int':>10s}")
print("  " + "-" * 88)

oracle_4d_results = []
for pi, pd in enumerate(prompt_data):
    input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

    intervened_gates = {}
    for layer in range(COMB_START, COMB_END):
        gates_all_pos = pd['gates'][layer]
        hs_all_pos = pd['hs'][layer]

        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold_corrected = scaffold_single[layer] + W_gate @ h_shift

        # dir1 = rank-1 crystal mode (from single tokens)
        dir1 = svd_per_layer[layer]['Vt'][0]

        # dir2 = 4th dimension direction (from multi-token residuals)
        dir2 = svd_4d_per_layer[layer]['Vt'][0]

        # Project each position onto both directions
        residuals = gates_all_pos - scaffold_corrected[np.newaxis, :]
        alphas = residuals @ dir1  # [n_tok]
        betas = residuals @ dir2   # [n_tok] -- the 4th dimension

        # Reconstruct with both dimensions
        reconstruction = (scaffold_corrected[np.newaxis, :]
                          + np.outer(alphas, dir1)
                          + np.outer(betas, dir2))
        intervened_gates[layer] = reconstruction

    hooks = []
    for layer in range(COMB_START, COMB_END):
        h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_replace_hook(intervened_gates[layer])
        )
        hooks.append(h)

    with torch.no_grad():
        int_out = model(input_ids)
        int_logits = int_out.logits[0, -1, :].cpu().float().numpy()

    for h in hooks:
        h.remove()

    base_l = pd['base_logits']
    cos = np.dot(base_l, int_logits) / (np.linalg.norm(base_l) * np.linalg.norm(int_logits))
    base_top1_id = np.argmax(base_l)
    int_top1_id = np.argmax(int_logits)
    match = base_top1_id == int_top1_id
    base_tok = tokenizer.decode([base_top1_id]).strip()
    int_tok = tokenizer.decode([int_top1_id]).strip()
    mark = "Y" if match else "N"

    print(f"  {pd['prompt'][:45]:>45s}  {cos:9.4f}  {mark:>6s}  {base_tok:>10s}  {int_tok:>10s}")

    oracle_4d_results.append({
        'prompt': pd['prompt'],
        'cos': float(cos),
        'match': bool(match),
        'base': base_tok,
        'int': int_tok,
    })

oracle_top1 = sum(1 for r in oracle_4d_results if r['match']) / len(oracle_4d_results)
oracle_cos = np.mean([r['cos'] for r in oracle_4d_results])
print()
print(f"  Oracle 4th dim: Top-1 = {oracle_top1:.0%}, Cos = {oracle_cos:.4f}")
print()


# ================================================================
# TEST 5: Check orthogonality of dir1 and dir2
# ================================================================
print("-" * 80)
print("  TEST 5: Orthogonality of crystal mode (dir1) and 4th dim (dir2)")
print("-" * 80)
print()

print(f"  {'Layer':>7s}  {'cos(dir1,dir2)':>15s}  {'|dir1|':>8s}  {'|dir2|':>8s}")
print("  " + "-" * 42)
orthogonality = []
for layer in range(COMB_START, COMB_END):
    dir1 = svd_per_layer[layer]['Vt'][0]
    dir2 = svd_4d_per_layer[layer]['Vt'][0]
    cos_dirs = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
    orthogonality.append(cos_dirs)
    print(f"  {layer:7d}  {cos_dirs:15.4f}  {np.linalg.norm(dir1):8.4f}  {np.linalg.norm(dir2):8.4f}")

mean_orth = np.mean(np.abs(orthogonality))
print()
print(f"  Mean |cos(dir1, dir2)| = {mean_orth:.4f}")
if mean_orth < 0.1:
    print("  The two directions are nearly ORTHOGONAL -- genuine independent dimension!")
elif mean_orth < 0.3:
    print("  Moderate orthogonality -- partially independent dimension")
else:
    print("  Directions are correlated -- may not be a clean independent dimension")
print()


# ================================================================
# TEST 6: Energy accounting - the 4D decomposition
# ================================================================
print("-" * 80)
print("  TEST 6: Energy accounting -- 4D vs 3D decomposition")
print("-" * 80)
print()

print(f"  Layer 14 decomposition for each prompt (last position):")
print(f"  {'Prompt':>30s}  {'|scaffold|':>11s}  {'|alpha*d1|':>11s}  {'|beta*d2|':>11s}  {'|residual|':>11s}  {'Match':>6s}")
print("  " + "-" * 88)

for pi, pd in enumerate(prompt_data):
    layer = 14
    gates_all_pos = pd['gates'][layer]
    hs_all_pos = pd['hs'][layer]
    last_pos = gates_all_pos.shape[0] - 1
    gate_true = gates_all_pos[last_pos]

    h_mean_prompt = hs_all_pos.mean(axis=0)
    h_shift = h_mean_prompt - h_mean_single[layer]
    W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    scaffold_corrected = scaffold_single[layer] + W_gate @ h_shift

    dir1 = svd_per_layer[layer]['Vt'][0]
    dir2 = svd_4d_per_layer[layer]['Vt'][0]

    residual_from_scaffold = gate_true - scaffold_corrected
    alpha = np.dot(residual_from_scaffold, dir1)
    beta = np.dot(residual_from_scaffold, dir2)
    final_residual = gate_true - scaffold_corrected - alpha * dir1 - beta * dir2

    match_str = "Y" if pi in successes else "N"
    print(f"  {pd['prompt'][:30]:>30s}  {np.linalg.norm(scaffold_corrected):11.2f}  "
          f"{np.linalg.norm(alpha * dir1):11.4f}  {np.linalg.norm(beta * dir2):11.4f}  "
          f"{np.linalg.norm(final_residual):11.2f}  {match_str:>6s}")

print()


# ================================================================
# TEST 7: The angle constraint -- does it relate to phi?
# ================================================================
print("-" * 80)
print("  TEST 7: Angular structure of the 4D decomposition")
print("  2D: 180 deg, 3D: 720 = 4*180, 4D: ???")
print("-" * 80)
print()

# Measure angle between the alpha*dir1 and beta*dir2 components
# For a genuine 4D structure, these should be orthogonal (90 degrees)
# The total "angular budget" in 4D would relate to the simplex structure

for layer in [COMB_START, 14, COMB_END - 1]:
    angles = []
    for pi in range(len(prompt_data)):
        gates_all_pos = prompt_data[pi]['gates'][layer]
        hs_all_pos = prompt_data[pi]['hs'][layer]
        last_pos = gates_all_pos.shape[0] - 1
        gate_true = gates_all_pos[last_pos]

        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold_corrected = scaffold_single[layer] + W_gate @ h_shift

        dir1 = svd_per_layer[layer]['Vt'][0]
        dir2 = svd_4d_per_layer[layer]['Vt'][0]

        res = gate_true - scaffold_corrected
        alpha = np.dot(res, dir1)
        beta = np.dot(res, dir2)

        # Angle between the two components in the full space
        comp1 = alpha * dir1
        comp2 = beta * dir2
        if np.linalg.norm(comp1) > 0 and np.linalg.norm(comp2) > 0:
            cos_angle = np.dot(comp1, comp2) / (np.linalg.norm(comp1) * np.linalg.norm(comp2))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle_deg)

    if angles:
        print(f"  Layer {layer}: mean angle(alpha*d1, beta*d2) = {np.mean(angles):.1f} deg +/- {np.std(angles):.1f}")

print()

# Check: alpha values for successes vs failures
print("  Alpha and Beta values (layer 14) for successes vs failures:")
print(f"  {'Prompt':>30s}  {'alpha':>10s}  {'beta':>10s}  {'|beta/alpha|':>13s}  {'Match':>6s}")
print("  " + "-" * 75)

for pi in range(len(prompt_data)):
    layer = 14
    gates_all_pos = prompt_data[pi]['gates'][layer]
    hs_all_pos = prompt_data[pi]['hs'][layer]
    last_pos = gates_all_pos.shape[0] - 1
    gate_true = gates_all_pos[last_pos]

    h_mean_prompt = hs_all_pos.mean(axis=0)
    h_shift = h_mean_prompt - h_mean_single[layer]
    W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    scaffold_corrected = scaffold_single[layer] + W_gate @ h_shift

    dir1 = svd_per_layer[layer]['Vt'][0]
    dir2 = svd_4d_per_layer[layer]['Vt'][0]

    res = gate_true - scaffold_corrected
    alpha = np.dot(res, dir1)
    beta = np.dot(res, dir2)
    ratio = abs(beta / alpha) if abs(alpha) > 1e-8 else float('inf')

    match_str = "Y" if pi in successes else "N"
    print(f"  {prompt_data[pi]['prompt'][:30]:>30s}  {alpha:10.4f}  {beta:10.4f}  {ratio:13.4f}  {match_str:>6s}")

print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: THE FOURTH DIMENSION")
print("=" * 80)
print()

print("  DIMENSIONAL PROGRESSION:")
print(f"    Finding 67 (scaffold only):        0% top-1  (1 component)")
print(f"    Finding 67 (scaffold + rank-1):     100% top-1 for single tokens (3 components)")
print(f"    Finding 68 (on prompts):            0% top-1  (3 components in wrong basis)")
print(f"    Finding 69 (stereo + rank-1):       50% top-1  (3 components, corrected basis)")
print(f"    Finding 70 (stereo + rank-20):      50% top-1  (more modes don't help)")
print(f"    Phase 8j  (stereo + rank-1 + 4th dim): {oracle_top1:.0%} top-1  (4 components)")
print()

print("  4TH DIMENSION PROPERTIES:")
print(f"    Orthogonality |cos(d1,d2)|: {mean_orth:.4f}")
print(f"    SVD gap S0/S1: {mean_ratio:.3f} (sqrt(phi) = {np.sqrt(PHI):.3f})")
print(f"    Mean top-1 variance: {mean_top1_var:.1f}%")
print()

if oracle_top1 > 0.8:
    print("  *** THE 4th DIMENSION EXISTS ***")
    print("  Adding ONE more direction (from multi-token residuals) breaks the 50% ceiling.")
    print("  The gate is genuinely 4-dimensional:")
    print("    dim 1: scaffold (crystal structure)")
    print("    dim 2: direction (rank-1 mode orientation)")
    print("    dim 3: alpha (per-token amplitude along dir1)")
    print("    dim 4: beta (per-token amplitude along dir2, the boundary routing)")
elif oracle_top1 > 0.5:
    print("  ** PARTIAL 4th DIMENSION **")
    print("  The 4th dim improves beyond 50%, but doesn't reach 100%.")
    print("  May need additional dimensions or a different dir2 extraction method.")
else:
    print("  The 4th dimension hypothesis is not confirmed by this approach.")
    print("  The residual may not have a clean directional structure.")

print()

# Free model
del model
torch.cuda.empty_cache()

# Save results
results = {
    'oracle_4d': {
        'top1': float(oracle_top1),
        'cos': float(oracle_cos),
        'per_prompt': oracle_4d_results,
    },
    'orthogonality': {
        'per_layer': {str(l): float(orthogonality[i]) for i, l in enumerate(range(COMB_START, COMB_END))},
        'mean_abs': float(mean_orth),
    },
    'residual_svd': {
        'mean_ratio_s0s1': float(mean_ratio),
        'mean_top1_var': float(mean_top1_var),
    },
    'successes': successes,
    'failures': failures,
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8j_fourth_dimension.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
