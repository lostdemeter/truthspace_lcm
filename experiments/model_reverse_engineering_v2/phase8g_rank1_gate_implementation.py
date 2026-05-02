#!/usr/bin/env python3
"""
Phase 8g: Rank-1 Gate Implementation — Does it Generalize?
=============================================================

Finding 67 showed that gate content is 1-dimensional: scaffold + α·direction
preserves 100% token identity for 65 training tokens.

But does it generalize to:
  1. HELD-OUT tokens (not in the 65 training set)?
  2. MULTI-TOKEN prompts (the real use case)?
  3. GENERATION (autoregressive output)?

If yes → we can replace the gate matmul (67.9M ops) with a single
dot product (3584 ops) — a 3013× reduction in gate computation.

The key implementation:
  w_alpha = W_gate^T @ direction_normalized   (precomputed)
  α = h · w_alpha - const                     (per-token: 3584 ops)
  gate = scaffold + α · direction              (per-token: 2×18944 ops)

Total: ~22K ops instead of 67.9M ops per layer.

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

GATE_CONTRACT = 0
GATE_PRESERVE_N = 1
GATE_PRESERVE_P = 2
GATE_EXPAND = 3


def classify_gate(x):
    codes = np.zeros_like(x, dtype=np.int8)
    codes[x < -LOG_PHI] = GATE_CONTRACT
    codes[(x >= -LOG_PHI) & (x < 0)] = GATE_PRESERVE_N
    codes[(x >= 0) & (x < LOG_PHI)] = GATE_PRESERVE_P
    codes[x >= LOG_PHI] = GATE_EXPAND
    return codes


# ================================================================
# SETUP
# ================================================================
print("=" * 80)
print("  PHASE 8g: RANK-1 GATE IMPLEMENTATION")
print("  Does the 18944:1 dimensional shift generalize?")
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
COMB_START = 6
COMB_END = 23

# Training tokens (same as phase8f)
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

# Held-out tokens — NOT in the training set
HOLDOUT_WORDS = [
    "moon", "sun", "ocean", "mountain", "river", "desert",
    "music", "painting", "dance", "poetry", "sculpture", "theater",
    "apple", "banana", "grape", "orange", "cherry", "lemon",
    "eagle", "wolf", "bear", "dolphin", "tiger", "elephant",
    "iron", "copper", "silver", "gold", "zinc", "lead",
    "spring", "summer", "autumn", "winter", "morning", "evening",
    "python", "java", "rust", "carbon", "silicon", "hydrogen",
    "mars", "venus", "jupiter", "saturn", "neptune", "mercury",
]


def capture_gates(words):
    """Capture gate activations for single-token words."""
    gate_raw = {}
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            continue
        token_id = ids[0]
        decoded = tokenizer.decode([token_id]).strip()
        if decoded in gate_raw:
            continue

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

        gates = np.stack([s[0].squeeze() for s in layer_gates])
        gate_raw[decoded] = gates

    return gate_raw


# ================================================================
# STEP 1: Build rank-1 model from training tokens
# ================================================================
print("─" * 80)
print("  STEP 1: Build rank-1 gate model from training tokens")
print("─" * 80)
print()

print(f"  Capturing gates for {len(TRAIN_WORDS)} training tokens...")
train_gates = capture_gates(TRAIN_WORDS)
train_words = sorted(train_gates.keys())
N_TRAIN = len(train_words)
train_raw = np.stack([train_gates[w] for w in train_words])  # [N_TRAIN, N_LAYERS, HIDDEN_DIM]

# Compute scaffold (standing wave)
scaffold = train_raw.mean(axis=0)  # [N_LAYERS, HIDDEN_DIM]

# Compute residuals
train_residuals = train_raw - scaffold[np.newaxis, :, :]  # [N_TRAIN, N_LAYERS, HIDDEN_DIM]

# Per-layer SVD to extract direction
svd_per_layer = {}
for layer in range(COMB_START, COMB_END):
    res = train_residuals[:, layer, :]  # [N_TRAIN, HIDDEN_DIM]
    U, S, Vt = np.linalg.svd(res, full_matrices=False)
    svd_per_layer[layer] = {
        'U': U,           # [N_TRAIN, N_TRAIN]
        'S': S,           # [N_TRAIN]
        'Vt': Vt,         # [N_TRAIN, HIDDEN_DIM]
        'direction': Vt[0],  # [HIDDEN_DIM] — the rank-1 direction
        'sv_ratio': S[0] / S[1] if S[1] > 0 else float('inf'),
    }

print(f"  Built rank-1 model: scaffold + α·direction for {COMB_END - COMB_START} COMB layers")
print(f"  Training tokens: {N_TRAIN}")
print(f"  Mean S₀/S₁ across COMB: {np.mean([v['sv_ratio'] for v in svd_per_layer.values()]):.4f}")
print()


# ================================================================
# STEP 2: Test on held-out tokens
# ================================================================
print("─" * 80)
print("  STEP 2: Test rank-1 reconstruction on HELD-OUT tokens")
print("─" * 80)
print()

print(f"  Capturing gates for {len(HOLDOUT_WORDS)} held-out tokens...")
holdout_gates = capture_gates(HOLDOUT_WORDS)
holdout_words = sorted(holdout_gates.keys())
N_HOLDOUT = len(holdout_words)
holdout_raw = np.stack([holdout_gates[w] for w in holdout_words])

# Compute held-out residuals using TRAINING scaffold
holdout_residuals = holdout_raw - scaffold[np.newaxis, :, :]

# For each held-out token, project onto rank-1 direction and reconstruct
print(f"  Reconstruction quality (held-out tokens, COMB layers):")
print(f"  {'Token':>14s}  {'Rank-1 cos':>11s}  {'Rank-1 error%':>14s}  {'α range':>14s}")
print("  " + "-" * 60)

holdout_cos_list = []
holdout_error_list = []

for tok_idx in range(N_HOLDOUT):
    token_cos = []
    token_err = []

    for layer in range(COMB_START, COMB_END):
        direction = svd_per_layer[layer]['direction']
        residual = holdout_residuals[tok_idx, layer, :]

        # Project onto rank-1 direction
        alpha = np.dot(residual, direction)
        reconstruction = alpha * direction

        # Cosine similarity
        r_norm = np.linalg.norm(residual)
        rec_norm = np.linalg.norm(reconstruction)
        if r_norm > 0 and rec_norm > 0:
            cos = np.dot(residual, reconstruction) / (r_norm * rec_norm)
        else:
            cos = 1.0

        # Relative error
        err = np.linalg.norm(residual - reconstruction) / r_norm if r_norm > 0 else 0

        token_cos.append(cos)
        token_err.append(err)

    mean_cos = np.mean(token_cos)
    mean_err = np.mean(token_err) * 100

    holdout_cos_list.append(mean_cos)
    holdout_error_list.append(mean_err)

    if tok_idx < 15:
        print(f"  {holdout_words[tok_idx]:>14s}  {mean_cos:11.4f}  {mean_err:13.1f}%")

print(f"  {'...':>14s}")
print(f"  {'MEAN':>14s}  {np.mean(holdout_cos_list):11.4f}  {np.mean(holdout_error_list):13.1f}%")
print()

# Token discrimination test on held-out
print(f"  Held-out token discrimination:")
# Full residual
holdout_flat = holdout_residuals[:, COMB_START:COMB_END, :].reshape(N_HOLDOUT, -1)
norms = np.linalg.norm(holdout_flat, axis=1, keepdims=True)
norms[norms == 0] = 1
sim_full = (holdout_flat / norms) @ (holdout_flat / norms).T
sim_full_mean = sim_full[~np.eye(N_HOLDOUT, dtype=bool)].mean()

# Rank-1 reconstruction
holdout_rank1 = np.zeros_like(holdout_residuals[:, COMB_START:COMB_END, :])
for li, layer in enumerate(range(COMB_START, COMB_END)):
    direction = svd_per_layer[layer]['direction']
    for tok_idx in range(N_HOLDOUT):
        alpha = np.dot(holdout_residuals[tok_idx, layer, :], direction)
        holdout_rank1[:, li, :][tok_idx] = alpha * direction

holdout_rank1_flat = holdout_rank1.reshape(N_HOLDOUT, -1)
norms_r1 = np.linalg.norm(holdout_rank1_flat, axis=1, keepdims=True)
norms_r1[norms_r1 == 0] = 1
sim_rank1 = (holdout_rank1_flat / norms_r1) @ (holdout_rank1_flat / norms_r1).T
sim_rank1_mean = sim_rank1[~np.eye(N_HOLDOUT, dtype=bool)].mean()

print(f"    Full residual pairwise sim:  {sim_full_mean:.4f}")
print(f"    Rank-1 residual pairwise sim: {sim_rank1_mean:.4f}")
print(f"    → {'Rank-1 discriminates held-out tokens' if sim_rank1_mean < 0.3 else 'DISCRIMINATION FAILS'}")
print()


# ================================================================
# STEP 3: Intervention on held-out tokens
# ================================================================
print("─" * 80)
print("  STEP 3: INTERVENTION — Does rank-1 gate preserve held-out token output?")
print("─" * 80)
print()

print("  Getting baseline outputs for held-out tokens...")
baseline_logits = {}
for word in holdout_words[:15]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], device="cuda")
        out = model(input_ids)
        baseline_logits[word] = out.logits[0, -1, :].cpu().float().numpy()

test_tokens = list(baseline_logits.keys())
print(f"  Baseline captured for {len(test_tokens)} held-out tokens")

def make_replace_hook(replacement):
    def hook_fn(module, input, output):
        rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
        return rep.reshape(output.shape)
    return hook_fn

# Intervention at rank 1
print()
print(f"  Intervention with scaffold + rank-1 residual (COMB layers):")
print(f"  {'Token':>14s}  {'Cos sim':>9s}  {'Top-1 match':>12s}  {'Base top-1':>14s}  {'Rank-1 top-1':>14s}")
print("  " + "-" * 70)

cos_sims = []
top1_matches = 0
top5_overlaps = []

for tok_word in test_tokens:
    tok_idx = holdout_words.index(tok_word)
    ids = tokenizer.encode(tok_word, add_special_tokens=False)
    token_id = ids[0]

    # Compute intervened gates: scaffold + α · direction
    intervened_gates = {}
    for layer in range(COMB_START, COMB_END):
        direction = svd_per_layer[layer]['direction']
        residual = holdout_residuals[tok_idx, layer, :]
        alpha = np.dot(residual, direction)
        intervened_gates[layer] = scaffold[layer] + alpha * direction

    hooks = []
    for layer in range(COMB_START, COMB_END):
        h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_replace_hook(intervened_gates[layer])
        )
        hooks.append(h)

    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], device="cuda")
        out = model(input_ids)
        int_logits = out.logits[0, -1, :].cpu().float().numpy()

    for h in hooks:
        h.remove()

    base_l = baseline_logits[tok_word]
    cos = np.dot(base_l, int_logits) / (np.linalg.norm(base_l) * np.linalg.norm(int_logits))
    cos_sims.append(cos)

    base_top1_id = np.argmax(base_l)
    int_top1_id = np.argmax(int_logits)
    match = base_top1_id == int_top1_id
    if match:
        top1_matches += 1

    base_top5 = set(np.argsort(base_l)[-5:])
    int_top5 = set(np.argsort(int_logits)[-5:])
    top5_overlaps.append(len(base_top5 & int_top5) / 5)

    base_top1 = tokenizer.decode([base_top1_id]).strip()
    int_top1 = tokenizer.decode([int_top1_id]).strip()
    mark = "✓" if match else "✗"

    print(f"  {tok_word:>14s}  {cos:9.4f}  {mark:>12s}  {base_top1:>14s}  {int_top1:>14s}")

mean_cos = np.mean(cos_sims)
top1_rate = top1_matches / len(test_tokens)
mean_top5 = np.mean(top5_overlaps)

print()
print(f"  HELD-OUT INTERVENTION SUMMARY:")
print(f"    Mean cosine similarity: {mean_cos:.4f}")
print(f"    Top-1 agreement:        {top1_rate:.0%}")
print(f"    Top-5 overlap:          {mean_top5:.0%}")
print()


# ================================================================
# STEP 4: Multi-token prompt test
# ================================================================
print("─" * 80)
print("  STEP 4: MULTI-TOKEN PROMPT TEST")
print("  Can rank-1 gate work for real prompts (not just single tokens)?")
print("─" * 80)
print()

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "The speed of light is approximately",
    "In mathematics, pi is approximately equal to",
]

print(f"  Testing {len(PROMPTS)} multi-token prompts...")
print()

prompt_results = []

for prompt in PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tokens_in_prompt = input_ids.shape[1]

    # Get baseline output
    with torch.no_grad():
        base_out = model(input_ids)
        base_logits = base_out.logits[0, -1, :].cpu().float().numpy()

    # Capture gate activations for all tokens in the prompt
    prompt_gates = {}
    hooks = []

    def make_capture_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = output.detach().cpu().float().numpy()
        return hook_fn

    for layer in range(COMB_START, COMB_END):
        h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_capture_hook(prompt_gates, layer)
        )
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # For each COMB layer, for each token position:
    # Compute residual from scaffold, project onto direction, reconstruct
    intervened_prompt_gates = {}
    rank1_cos_per_layer = []

    for layer in range(COMB_START, COMB_END):
        full_gate = prompt_gates[layer].squeeze()  # [n_tokens, HIDDEN_DIM]
        direction = svd_per_layer[layer]['direction']

        # Residual from scaffold (scaffold was trained on single tokens)
        residual = full_gate - scaffold[layer]  # [n_tokens, HIDDEN_DIM]

        # Project each token position onto rank-1 direction
        alphas = residual @ direction  # [n_tokens]
        reconstruction = np.outer(alphas, direction)  # [n_tokens, HIDDEN_DIM]
        reconstructed_gate = scaffold[layer] + reconstruction  # [n_tokens, HIDDEN_DIM]

        # Per-token cosine similarity
        for t in range(full_gate.shape[0]):
            fn = np.linalg.norm(full_gate[t])
            rn = np.linalg.norm(reconstructed_gate[t])
            if fn > 0 and rn > 0:
                c = np.dot(full_gate[t], reconstructed_gate[t]) / (fn * rn)
                rank1_cos_per_layer.append(c)

        intervened_prompt_gates[layer] = reconstructed_gate

    mean_gate_cos = np.mean(rank1_cos_per_layer)

    # Run intervention with rank-1 reconstructed gates
    hooks = []

    def make_prompt_replace_hook(replacement):
        def hook_fn(module, input, output):
            rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
            return rep.reshape(output.shape)
        return hook_fn

    for layer in range(COMB_START, COMB_END):
        h = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_prompt_replace_hook(intervened_prompt_gates[layer])
        )
        hooks.append(h)

    with torch.no_grad():
        int_out = model(input_ids)
        int_logits = int_out.logits[0, -1, :].cpu().float().numpy()

    for h in hooks:
        h.remove()

    # Compare
    cos = np.dot(base_logits, int_logits) / (np.linalg.norm(base_logits) * np.linalg.norm(int_logits))
    base_top1_id = np.argmax(base_logits)
    int_top1_id = np.argmax(int_logits)
    match = base_top1_id == int_top1_id
    base_top1 = tokenizer.decode([base_top1_id]).strip()
    int_top1 = tokenizer.decode([int_top1_id]).strip()

    base_top5 = set(np.argsort(base_logits)[-5:])
    int_top5 = set(np.argsort(int_logits)[-5:])
    top5_ov = len(base_top5 & int_top5) / 5

    mark = "✓" if match else "✗"
    print(f"  Prompt: \"{prompt}\"")
    print(f"    Gate cos sim: {mean_gate_cos:.4f}")
    print(f"    Logit cos:    {cos:.4f}")
    print(f"    Top-1:        base='{base_top1}' → rank1='{int_top1}' {mark}")
    print(f"    Top-5 overlap: {top5_ov:.0%}")
    print()

    prompt_results.append({
        'prompt': prompt,
        'n_tokens': n_tokens_in_prompt,
        'gate_cos': float(mean_gate_cos),
        'logit_cos': float(cos),
        'top1_match': bool(match),
        'base_top1': base_top1,
        'rank1_top1': int_top1,
        'top5_overlap': float(top5_ov),
    })


# ================================================================
# STEP 5: Compute w_alpha — the hidden-state projection vector
# ================================================================
print("─" * 80)
print("  STEP 5: Compute w_alpha — project hidden state directly to α")
print("  If this works, NO gate matmul needed at all")
print("─" * 80)
print()

# w_alpha[layer] = W_gate[layer]^T @ direction_normalized[layer]
# α = h · w_alpha - (scaffold · direction)

w_alpha_per_layer = {}
scaffold_dot_dir = {}

for layer in range(COMB_START, COMB_END):
    W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    direction = svd_per_layer[layer]['direction']
    dir_norm = direction / np.linalg.norm(direction)

    # w_alpha = W_gate^T @ direction_normalized
    # W_gate is [HIDDEN_DIM, HIDDEN_STATE_DIM], direction is [HIDDEN_DIM]
    w_alpha = W_gate.T @ dir_norm  # [HIDDEN_STATE_DIM]
    w_alpha_per_layer[layer] = w_alpha

    # Constant offset
    scaffold_dot_dir[layer] = np.dot(scaffold[layer], dir_norm)

print(f"  Computed w_alpha for {COMB_END - COMB_START} layers")
print(f"  w_alpha shape: {w_alpha.shape} (one vector per layer)")
print(f"  Total parameters: {(COMB_END - COMB_START) * len(w_alpha)} "
      f"(vs {(COMB_END - COMB_START) * HIDDEN_DIM * HIDDEN_STATE_DIM} for full gate_proj)")
print()

# Verify: compute α from hidden state and compare to direct computation
print(f"  Verifying w_alpha on training tokens...")

# Capture hidden states for a few training tokens
verify_words = train_words[:5]
hidden_states = {}

for word in verify_words:
    ids = tokenizer.encode(word, add_special_tokens=False)
    token_id = ids[0]

    hs = {}
    hooks = []

    def make_hs_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            # MLP input is the hidden state after attention + layernorm
            storage[layer_idx] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(COMB_START, COMB_END):
        h = model.model.layers[layer].mlp.register_forward_hook(
            make_hs_hook(hs, layer)
        )
        hooks.append(h)

    with torch.no_grad():
        model(torch.tensor([[token_id]], device="cuda"))

    for h in hooks:
        h.remove()

    hidden_states[word] = hs

print(f"  {'Token':>12s}  {'Layer':>5s}  {'α (direct)':>12s}  {'α (w_alpha)':>12s}  {'Match':>8s}")
print("  " + "-" * 55)

for word in verify_words:
    tok_idx = train_words.index(word)
    for layer in [COMB_START, 14, COMB_END - 1]:
        direction = svd_per_layer[layer]['direction']
        dir_norm = direction / np.linalg.norm(direction)

        # Direct: project gate residual onto direction
        residual = train_residuals[tok_idx, layer, :]
        alpha_direct = np.dot(residual, dir_norm)

        # Via w_alpha: project hidden state
        h = hidden_states[word][layer]
        # h might have shape issues depending on how MLP input is captured
        if h.ndim > 1:
            h = h[-1]  # take last token position
        alpha_walpha = np.dot(h, w_alpha_per_layer[layer]) - scaffold_dot_dir[layer]

        match = abs(alpha_direct - alpha_walpha) / (abs(alpha_direct) + 1e-10) < 0.05
        mark = "✓" if match else "✗"
        print(f"  {word:>12s}  {layer:5d}  {alpha_direct:12.4f}  {alpha_walpha:12.4f}  {mark:>8s}")

print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: RANK-1 GATE GENERALIZATION")
print("=" * 80)
print()

print(f"  HELD-OUT TOKENS ({N_HOLDOUT} tokens, single-token):")
print(f"    Reconstruction cos sim: {np.mean(holdout_cos_list):.4f}")
print(f"    Reconstruction error:   {np.mean(holdout_error_list):.1f}%")
print(f"    Discrimination sim:     {sim_rank1_mean:.4f} (full: {sim_full_mean:.4f})")
print(f"    Intervention cos sim:   {mean_cos:.4f}")
print(f"    Intervention top-1:     {top1_rate:.0%}")
print(f"    Intervention top-5:     {mean_top5:.0%}")
print()

prompt_top1_rate = sum(1 for p in prompt_results if p['top1_match']) / len(prompt_results)
prompt_cos_mean = np.mean([p['logit_cos'] for p in prompt_results])
prompt_top5_mean = np.mean([p['top5_overlap'] for p in prompt_results])

print(f"  MULTI-TOKEN PROMPTS ({len(prompt_results)} prompts):")
print(f"    Logit cos sim:  {prompt_cos_mean:.4f}")
print(f"    Top-1 accuracy: {prompt_top1_rate:.0%}")
print(f"    Top-5 overlap:  {prompt_top5_mean:.0%}")
print()

# Overall verdict
if top1_rate >= 0.8 and prompt_top1_rate >= 0.6:
    verdict = "GENERALIZES — Rank-1 gate works on unseen tokens and prompts"
elif top1_rate >= 0.5 or prompt_top1_rate >= 0.4:
    verdict = "PARTIAL — Works for some tokens/prompts but not reliably"
else:
    verdict = "DOES NOT GENERALIZE — Rank-1 is overfitted to training tokens"

print(f"  VERDICT: {verdict}")
print()

if top1_rate >= 0.8:
    print(f"  COMPUTATION SAVINGS:")
    print(f"    Gate matmul:   {HIDDEN_DIM} × {HIDDEN_STATE_DIM} = {HIDDEN_DIM * HIDDEN_STATE_DIM:,} ops")
    print(f"    Rank-1 gate:   {HIDDEN_STATE_DIM} + {2 * HIDDEN_DIM} = {HIDDEN_STATE_DIM + 2 * HIDDEN_DIM:,} ops")
    print(f"    Speedup:       {HIDDEN_DIM * HIDDEN_STATE_DIM / (HIDDEN_STATE_DIM + 2 * HIDDEN_DIM):.0f}×")
    print()

# Free model
del model
torch.cuda.empty_cache()

# Save results
results = {
    'holdout': {
        'n_tokens': N_HOLDOUT,
        'mean_reconstruction_cos': float(np.mean(holdout_cos_list)),
        'mean_reconstruction_error_pct': float(np.mean(holdout_error_list)),
        'discrimination_sim_full': float(sim_full_mean),
        'discrimination_sim_rank1': float(sim_rank1_mean),
        'intervention_cos': float(mean_cos),
        'intervention_top1': float(top1_rate),
        'intervention_top5': float(mean_top5),
    },
    'prompts': prompt_results,
    'verdict': verdict,
    'summary': {
        'n_train': N_TRAIN,
        'n_holdout': N_HOLDOUT,
        'n_prompts': len(prompt_results),
        'hidden_dim': HIDDEN_DIM,
        'hidden_state_dim': HIDDEN_STATE_DIM,
        'comb_layers': list(range(COMB_START, COMB_END)),
    }
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8g_rank1_gate.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
