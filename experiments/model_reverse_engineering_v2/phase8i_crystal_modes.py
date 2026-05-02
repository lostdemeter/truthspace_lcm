#!/usr/bin/env python3
"""
Phase 8i: Crystal Modes — Two Approaches to Multi-Token Gate
==============================================================

Finding 69 showed: stereo scaffold correction jumps 0% → 50% for prompts,
but rank-1 content is insufficient (oracle only 40%).

The Gushurst Crystal insight: the scaffold is a SEED CRYSTAL. Its SVD
modes are VIBRATION MODES. Single tokens excite mode 1 only. Multi-token
prompts, through attention, excite multiple modes.

Two approaches tested:

  A. HIGHER RANK with corrected scaffold
     - Use stereo scaffold correction (W_gate @ δh_mean)
     - Try rank 1, 2, 3, 5, 10 for content extraction
     - Find the minimum rank that gives 100% top-1 for prompts

  B. SPECTROMETER-GUIDED scaffold correction
     - Instead of full W_gate @ δh_mean (67.9M ops), use only the
       structured dimensions from the Spectrometer rules
     - The Spectrometer knows which hidden state dims follow affine/
       gating/quadratic rules — use THOSE to predict scaffold shift
     - If structured dims suffice, we avoid the full gate matmul entirely

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
print("  PHASE 8i: CRYSTAL MODES")
print("  A: Higher rank  |  B: Spectrometer-guided scaffold")
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
HIDDEN_DIM = model.config.intermediate_size     # 18944
HIDDEN_STATE_DIM = model.config.hidden_size      # 3584


# ================================================================
# STEP 1: Build base scaffold + SVD modes from single tokens
# ================================================================
print("─" * 80)
print("  STEP 1: Build crystal (scaffold + modes) from single tokens")
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

# SVD: extract ALL modes (up to N_TRAIN)
residuals_single = all_gates - scaffold_single[np.newaxis, :, :]
svd_per_layer = {}
for layer in range(COMB_START, COMB_END):
    res = residuals_single[:, layer, :]
    U, S, Vt = np.linalg.svd(res, full_matrices=False)
    svd_per_layer[layer] = {
        'U': U,        # [N_TRAIN, N_TRAIN] — token coordinates in mode space
        'S': S,        # [N_TRAIN] — singular values (mode amplitudes)
        'Vt': Vt,      # [N_TRAIN, HIDDEN_DIM] — mode directions in gate space
    }

print(f"  Crystal built: {N_TRAIN} training tokens, {COMB_END - COMB_START} COMB layers")
print(f"  Mode spectrum (layer 14): S = [{', '.join(f'{s:.2f}' for s in svd_per_layer[14]['S'][:10])}...]")
print(f"  S₀/S₁ mean = {np.mean([v['S'][0]/v['S'][1] for v in svd_per_layer.values()]):.3f}")
print()


# ================================================================
# STEP 2: Capture multi-token prompts
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
    print(f"  \"{prompt}\" — {n_tok} tokens")

print()


# ================================================================
# APPROACH A: Higher Rank with Corrected Scaffold
# ================================================================
print("=" * 80)
print("  APPROACH A: HIGHER RANK CRYSTAL MODES")
print("  How many modes does the crystal need for multi-token prompts?")
print("=" * 80)
print()

def make_replace_hook(replacement):
    def hook_fn(module, input, output):
        rep = torch.tensor(replacement, dtype=output.dtype, device=output.device)
        return rep.reshape(output.shape)
    return hook_fn

RANKS_TO_TEST = [1, 2, 3, 5, 10, 20]

approach_a_results = {}

for rank_k in RANKS_TO_TEST:
    top1_matches = 0
    cos_sims = []
    top5_overlaps = []

    for pd in prompt_data:
        input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

        intervened_gates = {}
        for layer in range(COMB_START, COMB_END):
            gates_all_pos = pd['gates'][layer]
            hs_all_pos = pd['hs'][layer]

            # Stereo scaffold correction
            h_mean_prompt = hs_all_pos.mean(axis=0)
            h_shift = h_mean_prompt - h_mean_single[layer]
            W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
            scaffold_corrected = scaffold_single[layer] + W_gate @ h_shift

            # Rank-k reconstruction using precomputed crystal modes
            directions = svd_per_layer[layer]['Vt'][:rank_k]  # [rank_k, HIDDEN_DIM]
            residuals = gates_all_pos - scaffold_corrected[np.newaxis, :]

            # Project onto rank-k modes
            alphas = residuals @ directions.T  # [n_tok, rank_k]
            reconstruction = alphas @ directions  # [n_tok, HIDDEN_DIM]
            intervened_gates[layer] = scaffold_corrected + reconstruction

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
        cos_sims.append(cos)

        base_top1 = np.argmax(base_l)
        int_top1 = np.argmax(int_logits)
        if base_top1 == int_top1:
            top1_matches += 1

        base_top5 = set(np.argsort(base_l)[-5:])
        int_top5 = set(np.argsort(int_logits)[-5:])
        top5_overlaps.append(len(base_top5 & int_top5) / 5)

    top1_rate = top1_matches / len(prompt_data)
    mean_cos = np.mean(cos_sims)
    mean_top5 = np.mean(top5_overlaps)

    print(f"  Rank {rank_k:3d}:  Top-1 = {top1_rate:5.0%}  Cos = {mean_cos:.4f}  Top-5 = {mean_top5:.0%}")

    approach_a_results[rank_k] = {
        'top1': float(top1_rate),
        'cos': float(mean_cos),
        'top5': float(mean_top5),
    }

print()

# Find minimum rank for 100% (or best)
best_rank = max(approach_a_results.keys(), key=lambda k: approach_a_results[k]['top1'])
print(f"  Best rank: {best_rank} → Top-1 = {approach_a_results[best_rank]['top1']:.0%}")
print()


# ================================================================
# APPROACH B: Spectrometer-Guided Scaffold Correction
# ================================================================
print("=" * 80)
print("  APPROACH B: SPECTROMETER-GUIDED SCAFFOLD CORRECTION")
print("  Can we predict scaffold shift from STRUCTURED dims only?")
print("=" * 80)
print()

# Load Spectrometer rules
rules_dir = '/home/thorin/truthspace-lcm/experiments/model_reverse_engineering_v2/results/phase4_rules'
with open(os.path.join(rules_dir, 'summary.json')) as f:
    spec_summary = json.load(f)

# For each COMB layer, identify structured hidden state dimensions
# The Spectrometer rules tell us which dims are affine, gating, etc.
structured_dims_per_layer = {}

for entry in spec_summary:
    layer = entry['layer']
    if layer < COMB_START or layer >= COMB_END:
        continue

    # Load per-dimension rules
    layer_file = os.path.join(rules_dir, f'layer_{layer:02d}.json')
    with open(layer_file) as f:
        layer_data = json.load(f)

    # Identify structured dimensions (R² > 0.5)
    structured = []
    for rule in layer_data['dim_rules']:
        if rule['rule_type'] != 'unstructured' and rule['r_squared'] >= 0.5:
            structured.append(rule['global_dim'])

    structured_dims_per_layer[layer] = sorted(structured)

# Print coverage
for layer in [COMB_START, 14, COMB_END - 1]:
    n_struct = len(structured_dims_per_layer.get(layer, []))
    print(f"  Layer {layer}: {n_struct}/{HIDDEN_STATE_DIM} structured dims "
          f"({n_struct/HIDDEN_STATE_DIM*100:.1f}%)")

print()

# Test: use only structured dims for scaffold correction
# Instead of W_gate @ δh, use W_gate[:, structured] @ δh[structured]
# This uses only the structured columns of W_gate

print(f"  Testing scaffold correction quality with structured dims only:")
print(f"  {'Layer':>7s}  {'Full cos':>10s}  {'Struct cos':>11s}  {'Struct err%':>12s}  {'Coverage':>9s}")
print("  " + "-" * 60)

spec_correction_quality = []
for li, layer in enumerate(range(COMB_START, COMB_END)):
    W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
    struct_dims = structured_dims_per_layer.get(layer, [])

    cos_list = []
    err_list = []

    for pd in prompt_data:
        hs_all_pos = pd['hs'][layer]
        gates_all_pos = pd['gates'][layer]

        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]

        # Full scaffold correction
        full_correction = W_gate @ h_shift

        # Spectrometer-guided correction: only use structured dims
        if len(struct_dims) > 0:
            struct_correction = W_gate[:, struct_dims] @ h_shift[struct_dims]
        else:
            struct_correction = np.zeros(HIDDEN_DIM)

        # Actual scaffold error for reference
        scaffold_prompt = gates_all_pos.mean(axis=0)
        actual_error = scaffold_prompt - scaffold_single[layer]

        # Compare struct correction to actual
        actual_norm = np.linalg.norm(actual_error)
        struct_norm = np.linalg.norm(struct_correction)
        if actual_norm > 0 and struct_norm > 0:
            cos = np.dot(actual_error, struct_correction) / (actual_norm * struct_norm)
            cos_list.append(cos)
        rel_err = np.linalg.norm(actual_error - struct_correction) / actual_norm if actual_norm > 0 else 0
        err_list.append(rel_err)

    mean_cos = np.mean(cos_list) if cos_list else 0
    mean_err = np.mean(err_list)
    coverage = len(struct_dims) / HIDDEN_STATE_DIM * 100

    if li < 3 or li >= 14:
        print(f"  {layer:7d}  {'1.0000':>10s}  {mean_cos:11.4f}  {mean_err*100:11.1f}%  {coverage:8.1f}%")

    spec_correction_quality.append({
        'layer': layer,
        'struct_cos': float(mean_cos),
        'struct_err': float(mean_err),
        'coverage': float(coverage),
    })

mean_struct_cos = np.mean([r['struct_cos'] for r in spec_correction_quality])
mean_struct_err = np.mean([r['struct_err'] for r in spec_correction_quality])
mean_coverage = np.mean([r['coverage'] for r in spec_correction_quality])
print(f"  {'MEAN':>7s}  {'1.0000':>10s}  {mean_struct_cos:11.4f}  {mean_struct_err*100:11.1f}%  {mean_coverage:8.1f}%")
print()


# ================================================================
# APPROACH B intervention: scaffold from structured dims + rank-k content
# ================================================================
print(f"  Intervention: Spectrometer scaffold + rank-{best_rank} content:")
print(f"  {'Prompt':>45s}  {'Cos sim':>9s}  {'Top-1?':>8s}  {'Base':>10s}  {'Spec':>10s}")
print("  " + "-" * 88)

spec_intervention_results = []
for pd in prompt_data:
    input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

    intervened_gates = {}
    for layer in range(COMB_START, COMB_END):
        gates_all_pos = pd['gates'][layer]
        hs_all_pos = pd['hs'][layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        struct_dims = structured_dims_per_layer.get(layer, [])

        # Spectrometer-guided scaffold correction
        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        if len(struct_dims) > 0:
            scaffold_corrected = scaffold_single[layer] + W_gate[:, struct_dims] @ h_shift[struct_dims]
        else:
            scaffold_corrected = scaffold_single[layer].copy()

        # Rank-k content using crystal modes
        directions = svd_per_layer[layer]['Vt'][:best_rank]
        residuals = gates_all_pos - scaffold_corrected[np.newaxis, :]
        alphas = residuals @ directions.T
        reconstruction = alphas @ directions
        intervened_gates[layer] = scaffold_corrected + reconstruction

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

    spec_intervention_results.append({
        'prompt': pd['prompt'],
        'cos_sim': float(cos),
        'top1_match': bool(match),
        'base_top1': base_top1,
        'spec_top1': int_top1,
    })

spec_top1 = sum(1 for r in spec_intervention_results if r['top1_match']) / len(spec_intervention_results)
spec_cos = np.mean([r['cos_sim'] for r in spec_intervention_results])
print()
print(f"  Spectrometer scaffold + rank-{best_rank}: Top-1 = {spec_top1:.0%}, Cos = {spec_cos:.4f}")
print()


# ================================================================
# COMPUTATION COST ANALYSIS
# ================================================================
print("=" * 80)
print("  COMPUTATION COST ANALYSIS")
print("=" * 80)
print()

n_struct_mean = int(np.mean([len(structured_dims_per_layer.get(l, [])) for l in range(COMB_START, COMB_END)]))
n_comb = COMB_END - COMB_START

print(f"  Full gate matmul:     {HIDDEN_DIM} × {HIDDEN_STATE_DIM} = {HIDDEN_DIM * HIDDEN_STATE_DIM:>12,} ops/token/layer")
print(f"  Stereo correction:    {HIDDEN_DIM} × {HIDDEN_STATE_DIM} = {HIDDEN_DIM * HIDDEN_STATE_DIM:>12,} ops (ONCE per prompt)")
print(f"  Spec correction:      {HIDDEN_DIM} × {n_struct_mean} = {HIDDEN_DIM * n_struct_mean:>12,} ops (ONCE per prompt)")
print(f"  Rank-k content:       {best_rank} × {HIDDEN_DIM} = {best_rank * HIDDEN_DIM:>12,} ops/token/layer")
print()

for N in [5, 10, 20, 50]:
    full_cost = N * n_comb * HIDDEN_DIM * HIDDEN_STATE_DIM
    stereo_cost = n_comb * HIDDEN_DIM * HIDDEN_STATE_DIM + N * n_comb * best_rank * HIDDEN_DIM
    spec_cost = n_comb * HIDDEN_DIM * n_struct_mean + N * n_comb * best_rank * HIDDEN_DIM
    print(f"  N={N:3d} tokens:  Full={full_cost/1e9:.2f}G  Stereo={stereo_cost/1e9:.2f}G ({full_cost/stereo_cost:.1f}×)  "
          f"Spec={spec_cost/1e9:.2f}G ({full_cost/spec_cost:.1f}×)")

print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: CRYSTAL MODES")
print("=" * 80)
print()

print(f"  APPROACH A — Higher Rank (corrected scaffold):")
for rank_k in RANKS_TO_TEST:
    r = approach_a_results[rank_k]
    marker = " ←" if rank_k == best_rank else ""
    print(f"    Rank {rank_k:3d}: Top-1 = {r['top1']:5.0%}  Cos = {r['cos']:.4f}  Top-5 = {r['top5']:.0%}{marker}")
print()

print(f"  APPROACH B — Spectrometer-Guided Scaffold:")
print(f"    Structured dims coverage: {mean_coverage:.1f}%")
print(f"    Scaffold correction cos:  {mean_struct_cos:.4f}")
print(f"    Scaffold correction error: {mean_struct_err:.2%}")
print(f"    Intervention Top-1:       {spec_top1:.0%}")
print(f"    Intervention Cos:         {spec_cos:.4f}")
print()

print(f"  COMPARISON:")
print(f"    Finding 68 (static scaffold):     Top-1 = 0%")
print(f"    Finding 69 (stereo + rank-1):     Top-1 = 50%")
if best_rank in approach_a_results:
    print(f"    Approach A (stereo + rank-{best_rank}):    Top-1 = {approach_a_results[best_rank]['top1']:.0%}")
print(f"    Approach B (spec + rank-{best_rank}):      Top-1 = {spec_top1:.0%}")
print()

# Crystal mode interpretation
print(f"  CRYSTAL MODE INTERPRETATION:")
print(f"    The scaffold is a seed crystal with {N_TRAIN} vibrational modes.")
for rank_k in [1, 2, 3, 5]:
    if rank_k in approach_a_results:
        energy = sum(svd_per_layer[14]['S'][:rank_k]**2) / sum(svd_per_layer[14]['S']**2) * 100
        print(f"    Mode 1-{rank_k}: {energy:.1f}% of residual energy → {approach_a_results[rank_k]['top1']:.0%} top-1")
print()

# Free model
del model
torch.cuda.empty_cache()

# Save results
results = {
    'approach_a': approach_a_results,
    'approach_b': {
        'spec_correction_quality': spec_correction_quality,
        'intervention': spec_intervention_results,
        'top1': float(spec_top1),
        'cos': float(spec_cos),
    },
    'crystal_modes': {
        'sv_spectrum_layer14': svd_per_layer[14]['S'].tolist(),
        'sv_ratios': {str(l): float(v['S'][0]/v['S'][1]) for l, v in svd_per_layer.items()},
    },
    'computation': {
        'full_ops_per_token_per_layer': HIDDEN_DIM * HIDDEN_STATE_DIM,
        'struct_dims_mean': n_struct_mean,
        'best_rank': best_rank,
    },
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8i_crystal_modes.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
