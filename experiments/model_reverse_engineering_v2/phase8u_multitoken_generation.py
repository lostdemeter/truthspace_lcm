#!/usr/bin/env python3
"""
Phase 8u: Multi-Token Geometric Generation
=============================================

Building on:
- F67: Gate content is 1D for single tokens (scaffold + α·direction)
- F69: Stereo correction gets scaffold right (0%→50% for multi-token)
- F72: D*=7 with per-prompt SVD gives 100% on multi-token
- F74: Subspace orientation is prompt-specific (24% overlap)
- F75: Last position collapses cone (doesn't add new info)

The approach:
1. Run attention normally → get hidden states h_i at each position
2. Compute per-prompt hidden-state SVD → D* directions in hidden space
3. Project directions through W_gate → gate-space directions
4. Per-position: compute D* alphas from hidden states (cheap)
5. Reconstruct: gate = scaffold_corrected + Σ αᵢ · gate_direction_i

Computational savings:
  Traditional: N × (18944 × 3584) = N × 67.9M ops per layer
  Geometric:   (D*+1) × 67.9M + N × D* × 3584 ops per layer
  Speedup:     ~12× for N=100, ~120× for N=1000

Tests:
A. Oracle (per-prompt gate SVD) — upper bound at each rank
B. Hidden-state SVD projection — does it match oracle?
C. Full intervention — does the model produce correct tokens?
D. Accuracy vs rank — what's the minimum D* for 100%?
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import json
import os

PHI = (1 + np.sqrt(5)) / 2

COMB_START = 6
COMB_END = 23
N_COMB = COMB_END - COMB_START

print("=" * 80)
print("  PHASE 8u: MULTI-TOKEN GEOMETRIC GENERATION")
print("  scaffold + D*-dimensional hidden-state-derived residual")
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
# STEP 0: Build scaffold from single tokens
# ================================================================
print("-" * 80)
print("  STEP 0: Build scaffold from single tokens")
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
scaffold_gate = all_gates_single.mean(axis=0)   # [N_LAYERS, GATE_DIM]
scaffold_hidden = all_hs_single.mean(axis=0)     # [N_LAYERS, HIDDEN_DIM]

print(f"  Scaffold built from {len(train_words)} single tokens")
print()


# ================================================================
# STEP 1: Capture multi-token prompts
# ================================================================
print("-" * 80)
print("  STEP 1: Capture multi-token prompts")
print("-" * 80)

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "Albert Einstein developed the theory of",
    "The speed of light is approximately",
    "In mathematics, pi is approximately equal to",
    "The chemical symbol for gold is",
    "The color of the sky is usually",
    "A triangle has three sides and three",
    "The square root of 144 is",
]

prompt_data = []

for prompt in PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tok = input_ids.shape[1]

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
        base_logits = out.logits[0, -1, :].cpu().float().numpy()

    for h in hooks:
        h.remove()

    base_top1_id = int(np.argmax(base_logits))
    base_top1 = tokenizer.decode([base_top1_id]).strip()

    prompt_data.append({
        'prompt': prompt,
        'n_tokens': n_tok,
        'base_logits': base_logits,
        'base_top1': base_top1,
        'base_top1_id': base_top1_id,
        'gates': {l: gate_storage[l] for l in range(N_LAYERS)},
        'hs': {l: hs_storage[l] for l in range(N_LAYERS)},
    })

    print(f"  \"{prompt}\" ({n_tok} tok) → '{base_top1}'")

print()


# ================================================================
# STEP 2: Test A — Oracle per-prompt gate SVD at different ranks
# ================================================================
print("=" * 80)
print("  TEST A: ORACLE — Per-prompt gate SVD at different ranks")
print("  Upper bound: what accuracy does rank-k gate reconstruction achieve?")
print("=" * 80)
print()

def make_replace_hook(replacement):
    def hook_fn(module, input, output):
        rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
        return rep.reshape(output.shape)
    return hook_fn

TEST_RANKS = [1, 2, 3, 5, 7, 10, 15]

for rank in TEST_RANKS:
    matches = 0
    cos_sims = []

    for pd in prompt_data:
        input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

        intervened_gates = {}
        for layer in range(COMB_START, COMB_END):
            gates_all = pd['gates'][layer]  # [n_tok, GATE_DIM]
            hs_all = pd['hs'][layer]        # [n_tok, HIDDEN_DIM]

            # Per-prompt scaffold correction (exact via W_gate @ δh_mean)
            h_mean = hs_all.mean(axis=0)
            h_shift = h_mean - scaffold_hidden[layer]
            W = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
            scaffold_corrected = scaffold_gate[layer] + W @ h_shift

            # Per-prompt gate residuals and SVD
            gate_resid = gates_all - scaffold_corrected[np.newaxis, :]
            U, S, Vt = np.linalg.svd(gate_resid, full_matrices=False)

            # Rank-k reconstruction
            k = min(rank, len(S))
            reconstruction = (U[:, :k] * S[:k]) @ Vt[:k]
            intervened_gates[layer] = scaffold_corrected + reconstruction

        # Intervention
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

        cos = np.dot(pd['base_logits'], int_logits) / (
            np.linalg.norm(pd['base_logits']) * np.linalg.norm(int_logits) + 1e-10)
        cos_sims.append(cos)
        if int(np.argmax(int_logits)) == pd['base_top1_id']:
            matches += 1

    acc = matches / len(prompt_data)
    mean_cos = np.mean(cos_sims)
    print(f"  Rank {rank:2d}: top-1 = {acc:5.0%} ({matches}/{len(prompt_data)}), cos = {mean_cos:.6f}")

print()


# ================================================================
# STEP 3: Test B — Hidden-state SVD projection (the real approach)
# Instead of SVD on gate residuals (oracle), derive gate directions
# from hidden-state SVD projected through W_gate
# ================================================================
print("=" * 80)
print("  TEST B: HIDDEN-STATE SVD PROJECTION")
print("  Derive gate directions from hidden-state SVD (no gate oracle)")
print("=" * 80)
print()

# Preload W_gate for COMB layers
W_gates = {}
for layer in range(COMB_START, COMB_END):
    W_gates[layer] = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    # Shape: (GATE_DIM, HIDDEN_DIM) = (18944, 3584)

print(f"  W_gate loaded for {N_COMB} COMB layers")
print()

for rank in TEST_RANKS:
    matches = 0
    cos_sims = []

    for pd in prompt_data:
        input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

        intervened_gates = {}
        for layer in range(COMB_START, COMB_END):
            gates_all = pd['gates'][layer]  # [n_tok, GATE_DIM]
            hs_all = pd['hs'][layer]        # [n_tok, HIDDEN_DIM]
            W = W_gates[layer]

            # Scaffold correction (exact)
            h_mean = hs_all.mean(axis=0)
            h_shift = h_mean - scaffold_hidden[layer]
            scaffold_corrected = scaffold_gate[layer] + W @ h_shift

            # Hidden-state SVD
            h_resid = hs_all - h_mean[np.newaxis, :]  # [n_tok, HIDDEN_DIM]
            U_h, S_h, Vt_h = np.linalg.svd(h_resid, full_matrices=False)

            k = min(rank, len(S_h))

            # Project hidden directions through W_gate → gate directions
            # gate_direction_j = W @ Vt_h[j]  (unnormalized)
            # α_j for position i = U_h[i, j] * S_h[j]
            # gate_resid_i = Σ_j α_j * W @ Vt_h[j] = W @ (Σ_j α_j * Vt_h[j])
            #              = W @ (Σ_j U_h[i,j] * S_h[j] * Vt_h[j])
            #              = W @ h_resid_i  (rank-k approximation in hidden space)

            # Rank-k approximation of hidden residuals
            h_resid_approx = (U_h[:, :k] * S_h[:k]) @ Vt_h[:k]  # [n_tok, HIDDEN_DIM]

            # Gate reconstruction via W_gate projection
            gate_resid_approx = h_resid_approx @ W.T  # [n_tok, GATE_DIM]
            intervened_gates[layer] = scaffold_corrected + gate_resid_approx

        # Intervention
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

        cos = np.dot(pd['base_logits'], int_logits) / (
            np.linalg.norm(pd['base_logits']) * np.linalg.norm(int_logits) + 1e-10)
        cos_sims.append(cos)
        if int(np.argmax(int_logits)) == pd['base_top1_id']:
            matches += 1

    acc = matches / len(prompt_data)
    mean_cos = np.mean(cos_sims)
    print(f"  Rank {rank:2d}: top-1 = {acc:5.0%} ({matches}/{len(prompt_data)}), cos = {mean_cos:.6f}")

print()


# ================================================================
# STEP 4: Test C — Comparison: gate oracle vs hidden projection
# Do they give the same result? (they should, by linearity)
# ================================================================
print("=" * 80)
print("  TEST C: GATE ORACLE vs HIDDEN PROJECTION")
print("  Are they mathematically equivalent? (should be, by linearity)")
print("=" * 80)
print()

# For one prompt, compare the reconstructed gates
pd = prompt_data[0]
print(f"  Test prompt: \"{pd['prompt']}\"")
print()

for rank in [1, 3, 7]:
    cos_gate_vs_hidden = []

    for layer in range(COMB_START, COMB_END):
        gates_all = pd['gates'][layer]
        hs_all = pd['hs'][layer]
        W = W_gates[layer]

        h_mean = hs_all.mean(axis=0)
        h_shift = h_mean - scaffold_hidden[layer]
        scaffold_corrected = scaffold_gate[layer] + W @ h_shift

        # Oracle: gate SVD
        gate_resid = gates_all - scaffold_corrected[np.newaxis, :]
        U_g, S_g, Vt_g = np.linalg.svd(gate_resid, full_matrices=False)
        k = min(rank, len(S_g))
        gate_oracle = scaffold_corrected + (U_g[:, :k] * S_g[:k]) @ Vt_g[:k]

        # Hidden projection
        h_resid = hs_all - h_mean[np.newaxis, :]
        U_h, S_h, Vt_h = np.linalg.svd(h_resid, full_matrices=False)
        k = min(rank, len(S_h))
        h_resid_approx = (U_h[:, :k] * S_h[:k]) @ Vt_h[:k]
        gate_hidden = scaffold_corrected + h_resid_approx @ W.T

        # Compare (last position)
        g_o = gate_oracle[-1]
        g_h = gate_hidden[-1]
        cos = np.dot(g_o, g_h) / (np.linalg.norm(g_o) * np.linalg.norm(g_h) + 1e-10)
        cos_gate_vs_hidden.append(cos)

    print(f"  Rank {rank}: mean cos(oracle, hidden_proj) = {np.mean(cos_gate_vs_hidden):.6f}")

print()
print("  Note: if < 1.0, the difference is because SVD of W@H ≠ W@SVD(H)")
print("  (SVD is not commutative with linear maps)")
print()


# ================================================================
# STEP 5: Test D — What if we use FULL hidden residual (no rank cut)?
# This is equivalent to computing W_gate @ h directly but via
# scaffold + W_gate @ (h - h_mean). Should be EXACT.
# ================================================================
print("=" * 80)
print("  TEST D: FULL HIDDEN RESIDUAL (no rank cut)")
print("  scaffold_corrected + W @ (h - h_mean) should be EXACT")
print("=" * 80)
print()

matches = 0
cos_sims = []
gate_cos_sims = []

for pd in prompt_data:
    input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

    intervened_gates = {}
    for layer in range(COMB_START, COMB_END):
        gates_all = pd['gates'][layer]
        hs_all = pd['hs'][layer]
        W = W_gates[layer]

        h_mean = hs_all.mean(axis=0)
        h_shift = h_mean - scaffold_hidden[layer]
        scaffold_corrected = scaffold_gate[layer] + W @ h_shift

        # Full hidden residual (no rank cut)
        h_resid = hs_all - h_mean[np.newaxis, :]
        gate_resid = h_resid @ W.T
        reconstructed = scaffold_corrected + gate_resid
        intervened_gates[layer] = reconstructed

        # Check gate reconstruction quality
        cos_per_pos = []
        for t in range(gates_all.shape[0]):
            c = np.dot(gates_all[t], reconstructed[t]) / (
                np.linalg.norm(gates_all[t]) * np.linalg.norm(reconstructed[t]) + 1e-10)
            cos_per_pos.append(c)
        gate_cos_sims.append(np.mean(cos_per_pos))

    # Intervention
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

    cos = np.dot(pd['base_logits'], int_logits) / (
        np.linalg.norm(pd['base_logits']) * np.linalg.norm(int_logits) + 1e-10)
    cos_sims.append(cos)
    top1_match = int(np.argmax(int_logits)) == pd['base_top1_id']
    if top1_match:
        matches += 1
    mark = "✓" if top1_match else "✗"
    int_top1 = tokenizer.decode([int(np.argmax(int_logits))]).strip()
    print(f"  {pd['prompt'][:50]:>50s} {mark} base='{pd['base_top1']}' recon='{int_top1}' cos={cos:.6f}")

print()
print(f"  Full hidden residual: top-1 = {matches}/{len(prompt_data)} ({matches/len(prompt_data):.0%}), cos = {np.mean(cos_sims):.6f}")
print(f"  Gate reconstruction cos: {np.mean(gate_cos_sims):.6f}")
print()
if matches == len(prompt_data):
    print(f"  CONFIRMED: scaffold + W @ (h - h_mean) is EXACT reconstruction.")
    print(f"  This validates the hidden-state approach — now we need the rank cut.")
else:
    print(f"  WARNING: full reconstruction is not exact. Investigating...")
print()


# ================================================================
# STEP 6: COMPUTATIONAL COST ANALYSIS
# ================================================================
print("=" * 80)
print("  COMPUTATIONAL COST ANALYSIS")
print("=" * 80)
print()

# For each rank, compute theoretical speedup
print(f"  Per-layer costs (gate_proj only):")
print(f"    Traditional: N × {GATE_DIM} × {HIDDEN_DIM} = N × {GATE_DIM * HIDDEN_DIM:,} ops")
print()
print(f"  {'Rank':>5s}  {'Fixed cost':>12s}  {'Per-pos cost':>13s}  {'N=10 speedup':>13s}  {'N=100':>8s}  {'N=1000':>8s}")
print("  " + "-" * 62)

full_cost_per_pos = GATE_DIM * HIDDEN_DIM  # 67.9M

for rank in [1, 2, 3, 5, 7, 10, 15, 20]:
    # Fixed cost: (rank+1) matmuls for scaffold + direction projections
    # Actually: 1 matmul for scaffold correction via W @ δh_mean
    # + for per-position: rank × HIDDEN_DIM (compute alphas) + rank × GATE_DIM (reconstruct)
    # But if we precompute W @ v_k for each direction, the per-position cost is:
    # rank × HIDDEN_DIM (alphas) + rank × GATE_DIM (linear combination of gate directions)

    # Hidden-state SVD approach:
    # Fixed: SVD of n_pos × HIDDEN_DIM matrix + 1 matmul for scaffold correction
    #        + rank matmuls for direction projection (rank × GATE_DIM × HIDDEN_DIM)
    # Per-pos: just alpha computation = rank × HIDDEN_DIM ops
    #          + gate reconstruction = rank × GATE_DIM ops

    fixed_cost = (rank + 1) * full_cost_per_pos  # direction projections + scaffold
    per_pos_cost = rank * (HIDDEN_DIM + GATE_DIM)  # alpha + reconstruction

    for N in [10, 100, 1000]:
        traditional = N * full_cost_per_pos
        geometric = fixed_cost + N * per_pos_cost
        speedup = traditional / geometric

    s10 = 10 * full_cost_per_pos / (fixed_cost + 10 * per_pos_cost)
    s100 = 100 * full_cost_per_pos / (fixed_cost + 100 * per_pos_cost)
    s1000 = 1000 * full_cost_per_pos / (fixed_cost + 1000 * per_pos_cost)

    print(f"  {rank:5d}  {fixed_cost:12,.0f}  {per_pos_cost:13,.0f}  {s10:13.1f}×  {s100:7.1f}×  {s1000:7.1f}×")

print()
print("  Note: Fixed cost amortized across sequence. Speedup grows with N.")
print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: MULTI-TOKEN GEOMETRIC GENERATION")
print("=" * 80)
print()

# Save
del model
torch.cuda.empty_cache()

results = {
    'n_prompts': len(PROMPTS),
    'n_train_tokens': len(train_words),
    'prompts': [{'prompt': pd['prompt'], 'base_top1': pd['base_top1']} for pd in prompt_data],
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8u_multitoken.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
