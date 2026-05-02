#!/usr/bin/env python3
"""
Phase 8h: Additive Error Gate — Stereo Approach to Scaffold Correction
========================================================================

Finding 68 showed: rank-1 gate works for single tokens (93% top-1) but
fails for multi-token prompts (0% top-1). The scaffold shifts with context.

Additive Error Stereo insight: don't compute the full result — compute
the SHIFT (error) and add it to the base.

The scaffold shift is LINEAR in the hidden state shift:
    scaffold_error = W_gate @ δh_mean
    scaffold_prompt = scaffold_single + scaffold_error

If we can predict δh_mean from the attention output (available before MLP),
we can correct the scaffold without the full 67.9M-op gate matmul.

Tests:
  1. Is the scaffold shift low-rank? (SVD analysis)
  2. Is the per-position residual still rank-1 relative to per-prompt scaffold?
  3. Does corrected scaffold + rank-1 preserve token identity for prompts?
  4. Can we predict scaffold_error from hidden states? (the stereo shortcut)

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

# ================================================================
# SETUP
# ================================================================
print("=" * 80)
print("  PHASE 8h: ADDITIVE ERROR GATE")
print("  Stereo approach: scaffold_error = W_gate @ δh_mean")
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
# STEP 1: Build base scaffold from single tokens (same as phase8f/8g)
# ================================================================
print("─" * 80)
print("  STEP 1: Build base scaffold from single tokens")
print("─" * 80)
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

print(f"  Capturing gates + hidden states for {len(TRAIN_WORDS)} single tokens...")

# Capture BOTH gate outputs AND hidden states for single tokens
single_gates = {}   # word -> [N_LAYERS, HIDDEN_DIM]
single_hs = {}      # word -> [N_LAYERS, HIDDEN_STATE_DIM]  (MLP input)

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

    gates = np.stack([gate_storage[l] for l in range(N_LAYERS)])
    hs = np.stack([hs_storage[l] for l in range(N_LAYERS)])
    single_gates[decoded] = gates
    single_hs[decoded] = hs

train_words = sorted(single_gates.keys())
N_TRAIN = len(train_words)
all_gates = np.stack([single_gates[w] for w in train_words])  # [N_TRAIN, N_LAYERS, HIDDEN_DIM]
all_hs = np.stack([single_hs[w] for w in train_words])        # [N_TRAIN, N_LAYERS, HIDDEN_STATE_DIM]

# Base scaffold and hidden state mean
scaffold_single = all_gates.mean(axis=0)     # [N_LAYERS, HIDDEN_DIM]
h_mean_single = all_hs.mean(axis=0)          # [N_LAYERS, HIDDEN_STATE_DIM]

# Compute rank-1 direction from single tokens
residuals_single = all_gates - scaffold_single[np.newaxis, :, :]
svd_per_layer = {}
for layer in range(COMB_START, COMB_END):
    res = residuals_single[:, layer, :]
    U, S, Vt = np.linalg.svd(res, full_matrices=False)
    svd_per_layer[layer] = {
        'direction': Vt[0],
        'S': S,
    }

print(f"  Built base scaffold from {N_TRAIN} single tokens")
print(f"  h_mean_single shape: {h_mean_single.shape}")
print()


# ================================================================
# STEP 2: Capture multi-token prompts — gates AND hidden states
# ================================================================
print("─" * 80)
print("  STEP 2: Capture multi-token prompts")
print("─" * 80)
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

    # Get baseline output
    with torch.no_grad():
        base_out = model(input_ids)
        base_logits = base_out.logits[0, -1, :].cpu().float().numpy()

    # Capture gate outputs and hidden states for ALL positions
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

    # gate_storage[layer] = [n_tok, HIDDEN_DIM]
    # hs_storage[layer] = [n_tok, HIDDEN_STATE_DIM]
    prompt_data.append({
        'prompt': prompt,
        'n_tokens': n_tok,
        'base_logits': base_logits,
        'gates': {l: gate_storage[l] for l in range(N_LAYERS)},
        'hs': {l: hs_storage[l] for l in range(N_LAYERS)},
    })

    print(f"  \"{prompt}\" — {n_tok} tokens")

print()


# ================================================================
# STEP 3: Analyze scaffold shift
# ================================================================
print("─" * 80)
print("  STEP 3: SCAFFOLD SHIFT ANALYSIS")
print("  Is the shift low-rank? Is it predictable from hidden states?")
print("─" * 80)
print()

print(f"  Per-prompt scaffold shift analysis (COMB layers):")
print(f"  {'Prompt':>45s}  {'Shift norm':>11s}  {'Scaffold norm':>14s}  {'Ratio':>8s}")
print("  " + "-" * 85)

all_scaffold_errors = []  # [n_prompts, n_comb_layers, HIDDEN_DIM]
all_h_shifts = []         # [n_prompts, n_comb_layers, HIDDEN_STATE_DIM]

for pd in prompt_data:
    prompt_scaffold_errors = []
    prompt_h_shifts = []

    for layer in range(COMB_START, COMB_END):
        gates_all_pos = pd['gates'][layer]  # [n_tok, HIDDEN_DIM]
        hs_all_pos = pd['hs'][layer]        # [n_tok, HIDDEN_STATE_DIM]

        # Per-prompt scaffold = mean gate across positions
        scaffold_prompt = gates_all_pos.mean(axis=0)  # [HIDDEN_DIM]
        h_mean_prompt = hs_all_pos.mean(axis=0)       # [HIDDEN_STATE_DIM]

        # Scaffold error (the shift)
        scaffold_error = scaffold_prompt - scaffold_single[layer]
        h_shift = h_mean_prompt - h_mean_single[layer]

        prompt_scaffold_errors.append(scaffold_error)
        prompt_h_shifts.append(h_shift)

    all_scaffold_errors.append(np.array(prompt_scaffold_errors))
    all_h_shifts.append(np.array(prompt_h_shifts))

    # Print summary for this prompt
    shift_norms = [np.linalg.norm(e) for e in prompt_scaffold_errors]
    scaffold_norms = [np.linalg.norm(scaffold_single[l]) for l in range(COMB_START, COMB_END)]
    ratios = [s / n if n > 0 else 0 for s, n in zip(shift_norms, scaffold_norms)]

    print(f"  {pd['prompt'][:45]:>45s}  {np.mean(shift_norms):11.2f}  "
          f"{np.mean(scaffold_norms):14.2f}  {np.mean(ratios):7.2%}")

all_scaffold_errors = np.array(all_scaffold_errors)  # [n_prompts, n_comb, HIDDEN_DIM]
all_h_shifts = np.array(all_h_shifts)                # [n_prompts, n_comb, HIDDEN_STATE_DIM]

print()

# SVD of scaffold errors across prompts — is the shift low-rank?
print(f"  Scaffold shift SVD across {len(PROMPTS)} prompts:")
print(f"  {'Layer':>7s}  {'Top-1 var%':>11s}  {'Top-3 var%':>11s}  {'S₀/S₁':>8s}  {'Low-rank?':>10s}")
print("  " + "-" * 55)

shift_svd_results = []
for li, layer in enumerate(range(COMB_START, COMB_END)):
    errors = all_scaffold_errors[:, li, :]  # [n_prompts, HIDDEN_DIM]
    U, S, Vt = np.linalg.svd(errors, full_matrices=False)
    total_var = (S**2).sum()
    top1_var = S[0]**2 / total_var * 100 if total_var > 0 else 0
    top3_var = (S[:3]**2).sum() / total_var * 100 if total_var > 0 else 0
    ratio = S[0] / S[1] if len(S) > 1 and S[1] > 0 else float('inf')

    low_rank = "YES" if top1_var > 50 else ("PARTIAL" if top1_var > 30 else "NO")

    if li < 5 or li >= 12:
        print(f"  {layer:7d}  {top1_var:10.1f}%  {top3_var:10.1f}%  {ratio:8.2f}  {low_rank:>10s}")

    shift_svd_results.append({
        'layer': layer,
        'top1_var_pct': float(top1_var),
        'top3_var_pct': float(top3_var),
        'sv_ratio': float(ratio),
    })

mean_top1 = np.mean([r['top1_var_pct'] for r in shift_svd_results])
print(f"  {'MEAN':>7s}  {mean_top1:10.1f}%")
print()


# ================================================================
# STEP 4: Per-prompt rank-1 test — is content still 1D within each prompt?
# ================================================================
print("─" * 80)
print("  STEP 4: Per-prompt rank-1 test")
print("  Is the gate content 1D relative to the per-prompt scaffold?")
print("─" * 80)
print()

print(f"  Per-prompt residual analysis:")
print(f"  {'Prompt':>45s}  {'Rank-1 var%':>12s}  {'S₀/S₁':>8s}")
print("  " + "-" * 70)

per_prompt_rank1_results = []
for pd in prompt_data:
    rank1_vars = []
    sv_ratios = []

    for layer in range(COMB_START, COMB_END):
        gates_all_pos = pd['gates'][layer]  # [n_tok, HIDDEN_DIM]
        scaffold_prompt = gates_all_pos.mean(axis=0)
        residuals = gates_all_pos - scaffold_prompt[np.newaxis, :]

        if residuals.shape[0] < 2:
            continue

        U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
        total_var = (S**2).sum()
        if total_var > 0:
            rank1_vars.append(S[0]**2 / total_var * 100)
        if len(S) > 1 and S[1] > 0:
            sv_ratios.append(S[0] / S[1])

    mean_rank1 = np.mean(rank1_vars) if rank1_vars else 0
    mean_ratio = np.mean(sv_ratios) if sv_ratios else 0

    print(f"  {pd['prompt'][:45]:>45s}  {mean_rank1:11.1f}%  {mean_ratio:8.2f}")

    per_prompt_rank1_results.append({
        'prompt': pd['prompt'],
        'mean_rank1_var_pct': float(mean_rank1),
        'mean_sv_ratio': float(mean_ratio),
    })

print()


# ================================================================
# STEP 5: Corrected scaffold intervention
# ================================================================
print("─" * 80)
print("  STEP 5: CORRECTED SCAFFOLD INTERVENTION")
print("  Use per-prompt scaffold + rank-1 direction and test output")
print("─" * 80)
print()

def make_replace_hook(replacement):
    def hook_fn(module, input, output):
        rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
        return rep.reshape(output.shape)
    return hook_fn

print(f"  Intervention results with per-prompt scaffold + rank-1:")
print(f"  {'Prompt':>45s}  {'Cos sim':>9s}  {'Top-1?':>8s}  {'Base':>10s}  {'Rank-1':>10s}")
print("  " + "-" * 88)

intervention_results = []
for pd in prompt_data:
    input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

    # For each COMB layer: compute per-prompt scaffold and rank-1 direction
    intervened_gates = {}
    for layer in range(COMB_START, COMB_END):
        gates_all_pos = pd['gates'][layer]  # [n_tok, HIDDEN_DIM]
        scaffold_prompt = gates_all_pos.mean(axis=0)
        residuals = gates_all_pos - scaffold_prompt[np.newaxis, :]

        if residuals.shape[0] >= 2:
            U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
            direction_prompt = Vt[0]
            # Rank-1 reconstruction
            alphas = residuals @ direction_prompt
            reconstruction = np.outer(alphas, direction_prompt)
            intervened_gates[layer] = scaffold_prompt + reconstruction
        else:
            intervened_gates[layer] = gates_all_pos

    # Run intervention
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

    # Compare
    base_l = pd['base_logits']
    cos = np.dot(base_l, int_logits) / (np.linalg.norm(base_l) * np.linalg.norm(int_logits))
    base_top1_id = np.argmax(base_l)
    int_top1_id = np.argmax(int_logits)
    match = base_top1_id == int_top1_id
    base_top1 = tokenizer.decode([base_top1_id]).strip()
    int_top1 = tokenizer.decode([int_top1_id]).strip()
    mark = "✓" if match else "✗"

    print(f"  {pd['prompt'][:45]:>45s}  {cos:9.4f}  {mark:>8s}  {base_top1:>10s}  {int_top1:>10s}")

    intervention_results.append({
        'prompt': pd['prompt'],
        'cos_sim': float(cos),
        'top1_match': bool(match),
        'base_top1': base_top1,
        'rank1_top1': int_top1,
    })

prompt_top1 = sum(1 for r in intervention_results if r['top1_match']) / len(intervention_results)
prompt_cos = np.mean([r['cos_sim'] for r in intervention_results])
print()
print(f"  Per-prompt scaffold + rank-1: Top-1 = {prompt_top1:.0%}, Cos = {prompt_cos:.4f}")
print()


# ================================================================
# STEP 6: Predicted scaffold correction (the stereo shortcut)
# ================================================================
print("─" * 80)
print("  STEP 6: PREDICTED SCAFFOLD CORRECTION")
print("  Can we predict scaffold_error from hidden state shift?")
print("  scaffold_error ≈ W_gate @ δh_mean (linear prediction)")
print("─" * 80)
print()

# For each prompt, test: predicted_scaffold_error = W_gate @ (h_mean_prompt - h_mean_single)
# Compare to actual scaffold_error
print(f"  Scaffold error prediction quality:")
print(f"  {'Layer':>7s}  {'Cos(pred,actual)':>17s}  {'Relative error':>15s}")
print("  " + "-" * 45)

prediction_quality = []
for li, layer in enumerate(range(COMB_START, COMB_END)):
    W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()

    cos_list = []
    err_list = []

    for pi in range(len(PROMPTS)):
        h_shift = all_h_shifts[pi, li, :]  # [HIDDEN_STATE_DIM]
        actual_error = all_scaffold_errors[pi, li, :]  # [HIDDEN_DIM]

        # Predicted scaffold error via linear projection
        predicted_error = W_gate @ h_shift  # [HIDDEN_DIM]

        # Compare
        actual_norm = np.linalg.norm(actual_error)
        pred_norm = np.linalg.norm(predicted_error)
        if actual_norm > 0 and pred_norm > 0:
            cos = np.dot(actual_error, predicted_error) / (actual_norm * pred_norm)
            cos_list.append(cos)
        rel_err = np.linalg.norm(actual_error - predicted_error) / actual_norm if actual_norm > 0 else 0
        err_list.append(rel_err)

    mean_cos = np.mean(cos_list) if cos_list else 0
    mean_err = np.mean(err_list)

    if li < 5 or li >= 12:
        print(f"  {layer:7d}  {mean_cos:17.4f}  {mean_err:14.2%}")

    prediction_quality.append({
        'layer': layer,
        'mean_cos': float(mean_cos),
        'mean_rel_error': float(mean_err),
    })

mean_pred_cos = np.mean([r['mean_cos'] for r in prediction_quality])
mean_pred_err = np.mean([r['mean_rel_error'] for r in prediction_quality])
print(f"  {'MEAN':>7s}  {mean_pred_cos:17.4f}  {mean_pred_err:14.2%}")
print()


# ================================================================
# STEP 7: Full stereo pipeline intervention
# ================================================================
print("─" * 80)
print("  STEP 7: FULL STEREO PIPELINE INTERVENTION")
print("  scaffold_corrected = scaffold_single + W_gate @ δh_mean")
print("  gate = scaffold_corrected + α · direction_single")
print("  (NO per-prompt SVD — uses only precomputed + hidden state shift)")
print("─" * 80)
print()

print(f"  Stereo pipeline results:")
print(f"  {'Prompt':>45s}  {'Cos sim':>9s}  {'Top-1?':>8s}  {'Base':>10s}  {'Stereo':>10s}")
print("  " + "-" * 88)

stereo_results = []
for pi, pd in enumerate(prompt_data):
    input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

    intervened_gates = {}
    for li, layer in enumerate(range(COMB_START, COMB_END)):
        gates_all_pos = pd['gates'][layer]  # [n_tok, HIDDEN_DIM]
        hs_all_pos = pd['hs'][layer]        # [n_tok, HIDDEN_STATE_DIM]

        # Step A: Predict scaffold correction from hidden state shift
        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold_error = W_gate @ h_shift
        scaffold_corrected = scaffold_single[layer] + scaffold_error

        # Step B: Use SINGLE-TOKEN direction (precomputed, no per-prompt SVD)
        direction = svd_per_layer[layer]['direction']

        # Step C: Per-position residual projected onto direction
        residuals = gates_all_pos - scaffold_corrected[np.newaxis, :]
        alphas = residuals @ direction
        reconstruction = np.outer(alphas, direction)
        intervened_gates[layer] = scaffold_corrected + reconstruction

    # Run intervention
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
    base_top1 = tokenizer.decode([base_top1_id]).strip()
    int_top1 = tokenizer.decode([int_top1_id]).strip()
    mark = "✓" if match else "✗"

    print(f"  {pd['prompt'][:45]:>45s}  {cos:9.4f}  {mark:>8s}  {base_top1:>10s}  {int_top1:>10s}")

    stereo_results.append({
        'prompt': pd['prompt'],
        'cos_sim': float(cos),
        'top1_match': bool(match),
        'base_top1': base_top1,
        'stereo_top1': int_top1,
    })

stereo_top1 = sum(1 for r in stereo_results if r['top1_match']) / len(stereo_results)
stereo_cos = np.mean([r['cos_sim'] for r in stereo_results])
print()
print(f"  Stereo pipeline: Top-1 = {stereo_top1:.0%}, Cos = {stereo_cos:.4f}")
print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: ADDITIVE ERROR GATE")
print("=" * 80)
print()

print(f"  Finding 68 baseline (static scaffold, multi-token prompts):")
print(f"    Top-1 = 0%, Cos = -0.17")
print()
print(f"  Per-prompt scaffold + per-prompt rank-1 (oracle):")
print(f"    Top-1 = {prompt_top1:.0%}, Cos = {prompt_cos:.4f}")
print()
print(f"  Stereo pipeline (predicted scaffold + single-token direction):")
print(f"    Top-1 = {stereo_top1:.0%}, Cos = {stereo_cos:.4f}")
print()

print(f"  Scaffold shift analysis:")
print(f"    Mean top-1 SV var%: {np.mean([r['top1_var_pct'] for r in shift_svd_results]):.1f}%")
print(f"    Prediction cos(pred,actual): {mean_pred_cos:.4f}")
print(f"    Prediction relative error: {mean_pred_err:.2%}")
print()

if stereo_top1 >= 0.6:
    print(f"  VERDICT: STEREO APPROACH WORKS")
    print(f"    The additive error correction fixes the scaffold shift problem.")
    print(f"    scaffold_corrected = scaffold_single + W_gate @ δh_mean")
elif prompt_top1 >= 0.6:
    print(f"  VERDICT: PER-PROMPT SCAFFOLD WORKS, PREDICTION NEEDS IMPROVEMENT")
    print(f"    The rank-1 structure holds per-prompt, but predicting the")
    print(f"    scaffold shift from hidden states needs refinement.")
else:
    print(f"  VERDICT: RANK-1 STRUCTURE DOES NOT HOLD PER-PROMPT")
    print(f"    Multi-token gate content is higher-dimensional than rank-1.")

# Free model
del model
torch.cuda.empty_cache()

# Save results
results = {
    'scaffold_shift_svd': shift_svd_results,
    'per_prompt_rank1': per_prompt_rank1_results,
    'per_prompt_intervention': intervention_results,
    'prediction_quality': prediction_quality,
    'stereo_intervention': stereo_results,
    'summary': {
        'n_prompts': len(PROMPTS),
        'per_prompt_top1': float(prompt_top1),
        'per_prompt_cos': float(prompt_cos),
        'stereo_top1': float(stereo_top1),
        'stereo_cos': float(stereo_cos),
        'mean_pred_cos': float(mean_pred_cos),
        'mean_pred_err': float(mean_pred_err),
    }
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8h_additive_error_gate.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to {results_path}")
