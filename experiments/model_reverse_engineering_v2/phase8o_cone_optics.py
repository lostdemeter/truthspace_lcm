#!/usr/bin/env python3
"""
Phase 8o: Cone Optics
======================

From Finding 75: Thinking = cone-building + cone-collapsing.
The last position recombines the basis built by previous positions.

Three optical interventions to test:

1. FOCUS THE CONE: Keep only top-k singular directions, drop the bulk.
   Does a tighter cone give better last-position recombination?
   The 2+5 structure (Finding 72) suggests top-2 carry more signal.

2. CORRECTIVE LENS: Does the cone quality degrade across layers?
   Can we measure "aberration" (how much the cone drifts per layer)?
   Is there a systematic correction (like stereo scaffold for directions)?

3. SHORTEN DISTANCES: Does sequence length blur the cone?
   Is the last position's recombination WORSE for longer sequences?
   Does "refocusing" at intermediate positions help? (= chain-of-thought)

The optical metaphor:
  - Layers = lenses in a telescope
  - Cone = beam of light (the basis directions)
  - Scaffold = lens alignment (fixed)
  - Content directions = beam direction (prompt-specific)
  - Last position = focal point
  - Aberrations = cone quality loss through layers

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
print("  PHASE 8o: CONE OPTICS")
print("  Can we focus, correct, or shorten the beam?")
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
print("  STEP 1: Build scaffold from single tokens")
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
N_SINGLE = len(train_words)
all_gates_single = np.stack([single_gates[w] for w in train_words])
all_hs_single = np.stack([single_hs[w] for w in train_words])

scaffold_single = all_gates_single.mean(axis=0)
h_mean_single = all_hs_single.mean(axis=0)

print(f"  Crystal: {N_SINGLE} tokens")
print()


# ================================================================
# STEP 2: Capture prompts of varying lengths
# ================================================================
print("-" * 80)
print("  STEP 2: Capture prompts (varying lengths)")
print("-" * 80)

PROMPTS = [
    # Short (3-4 tokens)
    "One plus one",
    "The sky is",
    "Water is a",
    "Fire burns hot",
    # Medium (5-6 tokens)
    "The capital of France is",
    "The chemical symbol for gold is",
    "The color of the sky is",
    "The boiling point of water is",
    "The fastest land animal is the",
    "The atomic number of carbon is",
    # Long (7-9 tokens)
    "The largest planet in our solar system is",
    "The first president of the United States was",
    "In mathematics, pi is approximately equal to",
    "The Pythagorean theorem states that",
    "Shakespeare wrote the play Romeo and",
    "DNA stands for deoxyribonucleic",
    "To solve a quadratic equation you can use the",
    "The tallest mountain in the world is",
]

prompt_data = []
for prompt in PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tok = input_ids.shape[1]

    with torch.no_grad():
        base_out = model(input_ids)
        base_logits = base_out.logits[0, -1, :].cpu().float().numpy()

    base_top = int(np.argmax(base_logits))

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
        'base_top': base_top,
        'gates': {l: gate_storage[l] for l in range(N_LAYERS)},
        'hs': {l: hs_storage[l] for l in range(N_LAYERS)},
    })

N_PROMPTS = len(prompt_data)
print(f"  Captured {N_PROMPTS} prompts")
for pd in prompt_data:
    top_word = tokenizer.decode([pd['base_top']]).strip()
    print(f"    n={pd['n_tokens']:2d}  {pd['prompt'][:50]:>50s} -> {top_word}")
print()


# ================================================================
# Precompute stereo scaffolds
# ================================================================
corrected_scaffolds = {}
for pi, pd in enumerate(prompt_data):
    corrected_scaffolds[pi] = {}
    for layer in range(COMB_START, COMB_END):
        hs_all_pos = pd['hs'][layer]
        h_mean_prompt = hs_all_pos.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        corrected_scaffolds[pi][layer] = scaffold_single[layer] + W_gate @ h_shift


# ================================================================
# OPTIC 1: FOCUS THE CONE
# ================================================================
print("=" * 80)
print("  OPTIC 1: FOCUS THE CONE")
print("  Keep only top-k directions. Does a tighter cone improve accuracy?")
print("=" * 80)
print()

# For each prompt at each COMB layer:
# 1. Build cone from positions 0..n-2 (SVD -> directions)
# 2. Project last position onto top-k directions (focused cone)
# 3. Reconstruct gate = scaffold + focused projection
# 4. Measure cosine similarity to true gate at last position
# 5. Compare k=1,2,3,...,D* vs full

layer = 14  # Representative COMB layer
print(f"  Layer {layer}: Focused cone reconstruction quality")
print()

focus_results = {}  # k -> list of cos_sim values

for k_focus in [1, 2, 3, 5, 7, 'all']:
    cos_sims = []
    top1_correct = 0
    total = 0

    for pi, pd in enumerate(prompt_data):
        gates = pd['gates'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        n_pos = gates.shape[0]

        # Build cone from positions 0..n-2
        context_resids = gates[:n_pos-1] - scaffold[np.newaxis, :]  # [n_pos-1, HIDDEN_DIM]
        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

        # The last position's true gate residual
        true_resid = gates[n_pos-1] - scaffold

        if k_focus == 'all':
            k = min(n_pos - 1, Vt.shape[0])
        else:
            k = min(k_focus, n_pos - 1, Vt.shape[0])

        # Project last position onto top-k cone directions
        dirs_k = Vt[:k]  # [k, HIDDEN_DIM]
        coeffs = dirs_k @ true_resid  # [k]
        focused_resid = coeffs @ dirs_k  # [HIDDEN_DIM]

        # Reconstruct
        recon_gate = scaffold + focused_resid
        true_gate = gates[n_pos-1]

        # Cosine similarity
        cos_sim = np.dot(recon_gate, true_gate) / (
            np.linalg.norm(recon_gate) * np.linalg.norm(true_gate) + 1e-10)
        cos_sims.append(cos_sim)

    k_label = str(k_focus) if k_focus != 'all' else 'all'
    focus_results[k_label] = cos_sims
    mean_cos = np.mean(cos_sims)
    print(f"  k={k_label:>3s}: cos_sim = {mean_cos:.6f} ± {np.std(cos_sims):.6f}")

print()

# Now do the REAL test: intervention with focused cone
print(f"  Layer {layer}: Focused cone INTERVENTION (replace last pos gate)")
print()

for k_focus in [1, 2, 3, 5, 'all']:
    correct = 0
    total = 0

    for pi, pd in enumerate(prompt_data):
        n_pos = pd['n_tokens']
        true_top = pd['base_top']

        # We need to do intervention across ALL COMB layers
        reconstructed_gates = {}
        for l in range(COMB_START, COMB_END):
            gates_l = pd['gates'][l]
            scaffold_l = corrected_scaffolds[pi][l]
            n_pos_l = gates_l.shape[0]

            context_resids = gates_l[:n_pos_l-1] - scaffold_l[np.newaxis, :]
            U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

            if k_focus == 'all':
                k = min(n_pos_l - 1, Vt.shape[0])
            else:
                k = min(k_focus, n_pos_l - 1, Vt.shape[0])

            dirs_k = Vt[:k]
            true_resid = gates_l[n_pos_l-1] - scaffold_l
            coeffs = dirs_k @ true_resid
            focused_resid = coeffs @ dirs_k
            reconstructed_gates[l] = scaffold_l + focused_resid

        # Run intervention: replace gate at last position for COMB layers
        input_ids = tokenizer.encode(pd['prompt'], return_tensors="pt").to("cuda")

        def make_intervention_hook(recon_gate, layer_idx):
            recon_tensor = torch.tensor(recon_gate, dtype=torch.bfloat16, device="cuda")
            def hook_fn(module, input, output):
                out = output.clone()
                out[0, -1, :] = recon_tensor
                return out
            return hook_fn

        hooks = []
        for l in range(COMB_START, COMB_END):
            h = model.model.layers[l].mlp.gate_proj.register_forward_hook(
                make_intervention_hook(reconstructed_gates[l], l)
            )
            hooks.append(h)

        with torch.no_grad():
            out = model(input_ids)
            pred_logits = out.logits[0, -1, :].cpu().float().numpy()

        for h in hooks:
            h.remove()

        pred_top = int(np.argmax(pred_logits))
        if pred_top == true_top:
            correct += 1
        total += 1

    k_label = str(k_focus) if k_focus != 'all' else 'all'
    acc = correct / total * 100
    print(f"  k={k_label:>3s}: top-1 accuracy = {acc:.1f}% ({correct}/{total})")

print()


# ================================================================
# OPTIC 2: ABERRATION ACROSS LAYERS (does the cone degrade?)
# ================================================================
print("=" * 80)
print("  OPTIC 2: ABERRATION ACROSS LAYERS")
print("  Does the cone 'blur' through layers? Where is sharpest focus?")
print("=" * 80)
print()

# For each prompt, at each COMB layer:
# 1. Build cone from positions 0..n-2
# 2. How well does the cone explain the last position? (residual fraction)
# 3. Track this "aberration" across layers

print(f"  Cone quality (fraction of last pos explained) per layer:")
print(f"  {'Layer':>7s}  {'Mean explained':>15s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
print("  " + "-" * 55)

layer_quality = {}

for layer in range(COMB_START, COMB_END):
    explained_fracs = []

    for pi, pd in enumerate(prompt_data):
        gates = pd['gates'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        n_pos = gates.shape[0]

        # Build cone from context positions
        context_resids = gates[:n_pos-1] - scaffold[np.newaxis, :]
        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

        # Project last position onto FULL cone
        true_resid = gates[n_pos-1] - scaffold
        k = min(n_pos - 1, Vt.shape[0])
        dirs_k = Vt[:k]
        proj = dirs_k.T @ (dirs_k @ true_resid)
        resid = true_resid - proj

        explained = 1.0 - (np.linalg.norm(resid) / (np.linalg.norm(true_resid) + 1e-10))
        explained_fracs.append(explained)

    mean_exp = np.mean(explained_fracs)
    layer_quality[layer] = mean_exp
    bar = "#" * int(mean_exp * 50)
    print(f"  {layer:7d}  {mean_exp:15.4f}  {np.std(explained_fracs):8.4f}  "
          f"{min(explained_fracs):8.4f}  {max(explained_fracs):8.4f}  {bar}")

# Where is sharpest focus?
best_layer = max(layer_quality, key=layer_quality.get)
worst_layer = min(layer_quality, key=layer_quality.get)
print()
print(f"  Sharpest focus (best lens): layer {best_layer} ({layer_quality[best_layer]:.4f})")
print(f"  Most aberrated (worst lens): layer {worst_layer} ({layer_quality[worst_layer]:.4f})")
print()


# ================================================================
# OPTIC 3: DISTANCE VS FOCUS (sequence length effect)
# ================================================================
print("=" * 80)
print("  OPTIC 3: DISTANCE VS FOCUS")
print("  Does sequence length blur the cone?")
print("=" * 80)
print()

# Group prompts by length, compare cone quality
from collections import defaultdict

layer = 14
length_quality = defaultdict(list)

for pi, pd in enumerate(prompt_data):
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    n_pos = gates.shape[0]

    context_resids = gates[:n_pos-1] - scaffold[np.newaxis, :]
    U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

    true_resid = gates[n_pos-1] - scaffold
    k = min(n_pos - 1, Vt.shape[0])
    dirs_k = Vt[:k]
    proj = dirs_k.T @ (dirs_k @ true_resid)
    resid = true_resid - proj

    explained = 1.0 - (np.linalg.norm(resid) / (np.linalg.norm(true_resid) + 1e-10))
    length_quality[n_pos].append((explained, pd['prompt']))

print(f"  Layer {layer}: Cone quality vs sequence length")
print()

for n_pos in sorted(length_quality.keys()):
    entries = length_quality[n_pos]
    quals = [e[0] for e in entries]
    mean_q = np.mean(quals)
    bar = "#" * int(mean_q * 50)
    print(f"  n_pos={n_pos:2d} ({len(entries):2d} prompts): "
          f"explained = {mean_q:.4f} ± {np.std(quals):.4f}  {bar}")
    for q, p in entries:
        print(f"    {q:.4f}  {p[:50]}")

print()


# ================================================================
# OPTIC 4: REFOCUSING (chain-of-thought in gate space)
# ================================================================
print("=" * 80)
print("  OPTIC 4: REFOCUSING (chain-of-thought analog)")
print("  Build cone from RECENT positions only, not all context")
print("=" * 80)
print()

# Instead of building the cone from ALL positions 0..n-2,
# build it from only the LAST w positions (a sliding window).
# If the cone blurs with distance, a shorter window should help.

layer = 14
print(f"  Layer {layer}: Sliding window cone quality")
print()

# Only test on prompts with enough positions (n_pos >= 6)
long_prompts = [(pi, pd) for pi, pd in enumerate(prompt_data) if pd['n_tokens'] >= 6]

for window in [2, 3, 4, 'all']:
    qualities = []

    for pi, pd in long_prompts:
        gates = pd['gates'][layer]
        scaffold = corrected_scaffolds[pi][layer]
        n_pos = gates.shape[0]

        if window == 'all':
            start = 0
        else:
            start = max(0, n_pos - 1 - window)

        context_resids = gates[start:n_pos-1] - scaffold[np.newaxis, :]
        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

        true_resid = gates[n_pos-1] - scaffold
        k = min(context_resids.shape[0], Vt.shape[0])
        dirs_k = Vt[:k]
        proj = dirs_k.T @ (dirs_k @ true_resid)
        resid = true_resid - proj

        explained = 1.0 - (np.linalg.norm(resid) / (np.linalg.norm(true_resid) + 1e-10))
        qualities.append(explained)

    w_label = str(window) if window != 'all' else 'all'
    mean_q = np.mean(qualities)
    bar = "#" * int(mean_q * 50)
    print(f"  window={w_label:>3s}: explained = {mean_q:.4f} ± {np.std(qualities):.4f}  {bar}")

print()
print("  If shorter windows work as well as 'all', early positions don't")
print("  contribute useful directions — only recent context matters.")
print("  If shorter windows are WORSE, all context contributes.")
print()


# ================================================================
# OPTIC 5: THE LENS EQUATION — can we predict cone quality?
# ================================================================
print("=" * 80)
print("  OPTIC 5: THE LENS EQUATION")
print("  What predicts cone quality? Singular value concentration?")
print("=" * 80)
print()

# For each prompt, compute:
# - Cone quality (explained fraction)
# - Singular value concentration (S0/sum(S))
# - Effective rank (participation ratio)
# - Sequence length

layer = 14
data_points = []

for pi, pd in enumerate(prompt_data):
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    n_pos = gates.shape[0]

    context_resids = gates[:n_pos-1] - scaffold[np.newaxis, :]
    U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

    true_resid = gates[n_pos-1] - scaffold
    k = min(n_pos - 1, Vt.shape[0])
    dirs_k = Vt[:k]
    proj = dirs_k.T @ (dirs_k @ true_resid)
    resid = true_resid - proj
    explained = 1.0 - (np.linalg.norm(resid) / (np.linalg.norm(true_resid) + 1e-10))

    # Singular value concentration
    s_total = np.sum(S)
    s_conc = S[0] / s_total if s_total > 0 else 0

    # Effective rank (participation ratio)
    s_norm = S / (s_total + 1e-10)
    eff_rank = 1.0 / (np.sum(s_norm ** 2) + 1e-10)

    # S0/S1 gap
    gap = S[0] / S[1] if len(S) > 1 and S[1] > 0 else float('inf')

    data_points.append({
        'prompt': pd['prompt'],
        'n_pos': n_pos,
        'explained': explained,
        's_conc': s_conc,
        'eff_rank': eff_rank,
        'gap': gap,
    })

print(f"  Layer {layer}: What predicts cone quality?")
print(f"  {'Prompt':>45s}  {'n':>3s}  {'Expl':>6s}  {'S_conc':>6s}  {'EffRk':>6s}  {'Gap':>6s}")
print("  " + "-" * 80)

for dp in sorted(data_points, key=lambda x: x['explained']):
    print(f"  {dp['prompt'][:45]:>45s}  {dp['n_pos']:3d}  {dp['explained']:6.4f}  "
          f"{dp['s_conc']:6.4f}  {dp['eff_rank']:6.2f}  {dp['gap']:6.3f}")

# Correlations
expl = np.array([dp['explained'] for dp in data_points])
n_pos_arr = np.array([dp['n_pos'] for dp in data_points])
s_conc_arr = np.array([dp['s_conc'] for dp in data_points])
eff_rank_arr = np.array([dp['eff_rank'] for dp in data_points])
gap_arr = np.array([dp['gap'] for dp in data_points])

print()
print(f"  Correlations with cone quality (explained fraction):")
print(f"    vs n_pos:     r = {np.corrcoef(expl, n_pos_arr)[0,1]:.4f}")
print(f"    vs S_conc:    r = {np.corrcoef(expl, s_conc_arr)[0,1]:.4f}")
print(f"    vs eff_rank:  r = {np.corrcoef(expl, eff_rank_arr)[0,1]:.4f}")
print(f"    vs gap:       r = {np.corrcoef(expl, gap_arr)[0,1]:.4f}")
print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: CONE OPTICS")
print("=" * 80)
print()

del model
torch.cuda.empty_cache()

results = {
    'n_prompts': N_PROMPTS,
    'prompts': PROMPTS,
    'focus_cos_sims': {k: [float(v) for v in vals] for k, vals in focus_results.items()},
    'layer_quality': {str(k): float(v) for k, v in layer_quality.items()},
    'data_points': data_points,
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8o_cone_optics.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"  Results saved to {results_path}")
