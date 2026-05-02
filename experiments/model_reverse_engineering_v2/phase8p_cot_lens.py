#!/usr/bin/env python3
"""
Phase 8p: Chain-of-Thought Lens Hypothesis
============================================

From Finding 76: Chain-of-thought = adding lenses to the optical path.
Each generated token becomes a new context position, potentially
refocusing the cone for the next prediction.

Hypothesis: During multi-step generation, each new token acts as a
lens that refocuses the cone. We predict:

1. CONE QUALITY stays high or improves as tokens are generated
   (each new token adds a well-placed lens)
2. The last position's g_new stays near zero throughout generation
   (always recombining, never adding a new direction)
3. D* grows with total sequence length (more tokens = more DOF)
   but cone quality per-step doesn't degrade
4. The SINGULAR VALUE CONCENTRATION may increase step-by-step
   (cone narrows as generation proceeds = more confident)

Method:
- Start with a prompt, generate N tokens autoregressively
- At EACH generation step, capture gates and hidden states
- Measure cone quality, g_new, D*, singular value spectrum
- Compare step 1 vs step N: does the cone sharpen or blur?

Requires: Qwen2-7B on GPU
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

print("=" * 80)
print("  PHASE 8p: CHAIN-OF-THOUGHT LENS HYPOTHESIS")
print("  Does each generated token refocus the cone?")
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
all_gates_single = np.stack([single_gates[w] for w in train_words])
all_hs_single = np.stack([single_hs[w] for w in train_words])
scaffold_single = all_gates_single.mean(axis=0)
h_mean_single = all_hs_single.mean(axis=0)

print(f"  Crystal: {len(train_words)} tokens")
print()


# ================================================================
# STEP 2: Generate tokens and capture gate data at each step
# ================================================================
print("-" * 80)
print("  STEP 2: Multi-step generation with gate capture")
print("-" * 80)

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "Albert Einstein developed the theory of",
    "The Pythagorean theorem states that",
    "Shakespeare wrote the play Romeo and",
]

N_GEN_STEPS = 15  # Generate 15 tokens per prompt

all_generation_data = []

for prompt in PROMPTS:
    print(f"\n  Prompt: '{prompt}'")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_prompt = input_ids.shape[1]

    generation_steps = []
    current_ids = input_ids.clone()

    for step in range(N_GEN_STEPS):
        n_total = current_ids.shape[1]

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
            out = model(current_ids)
            logits = out.logits[0, -1, :]

        for h in hooks:
            h.remove()

        # Get next token (greedy)
        next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        next_word = tokenizer.decode(next_token[0]).strip()

        # Store the gate data for this step
        step_data = {
            'step': step,
            'n_total': n_total,
            'n_prompt': n_prompt,
            'next_token': next_word,
            'gates': {},
            'hs': {},
        }

        for layer in range(COMB_START, COMB_END):
            step_data['gates'][layer] = gate_storage[layer].copy()
            step_data['hs'][layer] = hs_storage[layer].copy()

        generation_steps.append(step_data)

        # Append token for next step
        current_ids = torch.cat([current_ids, next_token], dim=1)

        if step < 5 or step == N_GEN_STEPS - 1:
            print(f"    Step {step:2d}: +'{next_word}' (n_total={n_total})")

    generated_text = tokenizer.decode(current_ids[0][n_prompt:])
    print(f"    Generated: '{generated_text}'")

    all_generation_data.append({
        'prompt': prompt,
        'n_prompt': n_prompt,
        'generated_text': generated_text,
        'steps': generation_steps,
    })

print()


# ================================================================
# ANALYSIS 1: Cone quality at each generation step
# ================================================================
print("=" * 80)
print("  ANALYSIS 1: CONE QUALITY PER GENERATION STEP")
print("  Does the cone stay focused as we generate more tokens?")
print("=" * 80)
print()

layer = 14  # Representative COMB layer

for gd in all_generation_data:
    prompt = gd['prompt']
    n_prompt = gd['n_prompt']
    steps = gd['steps']

    print(f"  {prompt}")

    step_qualities = []
    step_g_news = []
    step_d_stars = []
    step_s_concs = []

    for sd in steps:
        gates = sd['gates'][layer]  # [n_total, HIDDEN_DIM]
        hs = sd['hs'][layer]        # [n_total, HIDDEN_STATE_DIM]
        n_total = sd['n_total']

        # Compute stereo scaffold for this step
        h_mean_prompt = hs.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold = scaffold_single[layer] + W_gate @ h_shift

        # Context positions = all except last
        context_resids = gates[:n_total-1] - scaffold[np.newaxis, :]
        last_resid = gates[n_total-1] - scaffold

        # SVD of context cone
        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)
        k = min(n_total - 1, Vt.shape[0])
        dirs_k = Vt[:k]

        # Cone quality: how much of last pos is explained by context cone?
        proj = dirs_k.T @ (dirs_k @ last_resid)
        resid = last_resid - proj
        explained = 1.0 - (np.linalg.norm(resid) / (np.linalg.norm(last_resid) + 1e-10))
        step_qualities.append(explained)

        # g_new: fraction of last pos orthogonal to context
        g_new = np.linalg.norm(resid) / (np.linalg.norm(last_resid) + 1e-10)
        step_g_news.append(g_new)

        # D* (90% variance of context residuals)
        total_var = np.sum(S ** 2)
        cum_var = np.cumsum(S ** 2) / (total_var + 1e-10)
        d_star = int(np.searchsorted(cum_var, 0.90) + 1)
        step_d_stars.append(d_star)

        # Singular value concentration
        s_total = np.sum(S)
        s_conc = S[0] / (s_total + 1e-10)
        step_s_concs.append(s_conc)

    # Print step-by-step
    print(f"  {'Step':>6s}  {'n_pos':>5s}  {'Token':>10s}  {'Quality':>8s}  "
          f"{'g_new':>8s}  {'D*':>4s}  {'S_conc':>7s}")
    print("  " + "-" * 60)
    for i, sd in enumerate(steps):
        print(f"  {i:6d}  {sd['n_total']:5d}  {sd['next_token'][:10]:>10s}  "
              f"{step_qualities[i]:8.5f}  {step_g_news[i]:8.5f}  "
              f"{step_d_stars[i]:4d}  {step_s_concs[i]:7.4f}")

    # Summary for this prompt
    prompt_quals = step_qualities[:n_prompt-1]  # During prompt context
    gen_quals = step_qualities[n_prompt-1:]      # During generation
    if prompt_quals and gen_quals:
        print(f"  >> Prompt steps avg quality: {np.mean(prompt_quals):.5f}")
        print(f"  >> Generation steps avg quality: {np.mean(gen_quals):.5f}")
        delta = np.mean(gen_quals) - np.mean(prompt_quals)
        print(f"  >> Delta: {delta:+.5f} "
              f"({'SHARPENS' if delta > 0 else 'BLURS' if delta < -0.001 else 'STABLE'})")
    print()


# ================================================================
# ANALYSIS 2: Does S_conc increase during generation? (Cone narrows?)
# ================================================================
print("=" * 80)
print("  ANALYSIS 2: DOES THE CONE NARROW DURING GENERATION?")
print("  Does singular value concentration increase step-by-step?")
print("=" * 80)
print()

layer = 14

for gd in all_generation_data:
    steps = gd['steps']
    n_prompt = gd['n_prompt']

    s_concs_prompt = []
    s_concs_gen = []

    for i, sd in enumerate(steps):
        gates = sd['gates'][layer]
        hs = sd['hs'][layer]
        n_total = sd['n_total']

        h_mean_prompt = hs.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold = scaffold_single[layer] + W_gate @ h_shift

        context_resids = gates[:n_total-1] - scaffold[np.newaxis, :]
        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)

        s_total = np.sum(S)
        s_conc = S[0] / (s_total + 1e-10)

        if i < n_prompt - 1:
            s_concs_prompt.append(s_conc)
        else:
            s_concs_gen.append(s_conc)

    if s_concs_prompt and s_concs_gen:
        print(f"  {gd['prompt'][:50]:>50s}")
        print(f"    Prompt S_conc: {np.mean(s_concs_prompt):.4f} "
              f"(last={s_concs_prompt[-1]:.4f})")
        print(f"    Gen S_conc:    first={s_concs_gen[0]:.4f}, "
              f"last={s_concs_gen[-1]:.4f}")
        trend = s_concs_gen[-1] - s_concs_gen[0]
        print(f"    Gen trend: {trend:+.4f} "
              f"({'NARROWING' if trend > 0.01 else 'WIDENING' if trend < -0.01 else 'STABLE'})")
        print()


# ================================================================
# ANALYSIS 3: Cross-layer consistency during generation
# ================================================================
print("=" * 80)
print("  ANALYSIS 3: CROSS-LAYER CONE QUALITY DURING GENERATION")
print("  Is every layer's cone still ~99%+ during generation?")
print("=" * 80)
print()

# Pick one prompt, measure cone quality across layers at first and last gen step
gd = all_generation_data[0]
steps = gd['steps']
n_prompt = gd['n_prompt']

first_gen_step = steps[n_prompt - 1]  # First step where we're generating
last_gen_step = steps[-1]             # Last generation step

print(f"  Prompt: '{gd['prompt']}'")
print(f"  Generated: '{gd['generated_text']}'")
print()
print(f"  {'Layer':>7s}  {'First gen step':>15s}  {'Last gen step':>15s}  {'Delta':>8s}")
print("  " + "-" * 50)

for layer in range(COMB_START, COMB_END):
    qualities = []
    for sd in [first_gen_step, last_gen_step]:
        gates = sd['gates'][layer]
        hs = sd['hs'][layer]
        n_total = sd['n_total']

        h_mean_prompt = hs.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold = scaffold_single[layer] + W_gate @ h_shift

        context_resids = gates[:n_total-1] - scaffold[np.newaxis, :]
        last_resid = gates[n_total-1] - scaffold
        U, S, Vt = np.linalg.svd(context_resids, full_matrices=False)
        k = min(n_total - 1, Vt.shape[0])
        dirs_k = Vt[:k]
        proj = dirs_k.T @ (dirs_k @ last_resid)
        resid_vec = last_resid - proj
        explained = 1.0 - (np.linalg.norm(resid_vec) / (np.linalg.norm(last_resid) + 1e-10))
        qualities.append(explained)

    delta = qualities[1] - qualities[0]
    print(f"  {layer:7d}  {qualities[0]:15.5f}  {qualities[1]:15.5f}  {delta:+8.5f}")

print()


# ================================================================
# ANALYSIS 4: The "consecutive angle" during generation
# ================================================================
print("=" * 80)
print("  ANALYSIS 4: PATH CURVATURE DURING GENERATION")
print("  Does the ~87° angle hold for generated tokens too?")
print("=" * 80)
print()

layer = 14

for gd in all_generation_data[:3]:
    steps = gd['steps']
    n_prompt = gd['n_prompt']

    print(f"  {gd['prompt'][:50]:>50s}")

    # We need consecutive gate residuals.
    # Each step gives us the FULL gate array up to that point.
    # The angle between the last position's residual at step t
    # and the last position's residual at step t+1 is the "generation curvature"

    prev_resid = None
    angles = []
    labels = []

    for i, sd in enumerate(steps):
        gates = sd['gates'][layer]
        hs = sd['hs'][layer]
        n_total = sd['n_total']

        h_mean_prompt = hs.mean(axis=0)
        h_shift = h_mean_prompt - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold = scaffold_single[layer] + W_gate @ h_shift

        # Last position's residual at this step
        curr_resid = gates[n_total-1] - scaffold

        if prev_resid is not None:
            n1 = np.linalg.norm(prev_resid)
            n2 = np.linalg.norm(curr_resid)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(prev_resid, curr_resid) / (n1 * n2), -1, 1)
                angle = np.degrees(np.arccos(cos_a))
                angles.append(angle)
                phase = "prompt" if i < n_prompt else "gen"
                labels.append(f"step {i-1}->{i} ({phase})")

        prev_resid = curr_resid.copy()

    # Print angles
    for angle, label in zip(angles, labels):
        bar = "#" * int(angle / 3)
        print(f"    {label:>25s}: {angle:5.1f}°  {bar}")

    prompt_angles = [a for a, l in zip(angles, labels) if 'prompt' in l]
    gen_angles = [a for a, l in zip(angles, labels) if 'gen' in l]

    if prompt_angles and gen_angles:
        print(f"    >> Prompt mean angle: {np.mean(prompt_angles):.1f}°")
        print(f"    >> Gen mean angle:    {np.mean(gen_angles):.1f}°")
    print()


# ================================================================
# ANALYSIS 5: Does the cone PREDICT the next token's direction?
# ================================================================
print("=" * 80)
print("  ANALYSIS 5: CONE PREDICTIVE POWER")
print("  Can the cone from step t predict the generated token at step t+1?")
print("=" * 80)
print()

# At step t, we have the cone (context positions 0..n-1).
# The model generates token t+1.
# At step t+1, position n (the generated token) has a gate residual.
# Is that residual WITHIN the cone from step t?
# Or does the generated token bring genuinely new information?

layer = 14

for gd in all_generation_data[:3]:
    steps = gd['steps']
    n_prompt = gd['n_prompt']

    print(f"  {gd['prompt'][:50]:>50s}")

    for i in range(len(steps) - 1):
        sd_curr = steps[i]
        sd_next = steps[i + 1]

        gates_curr = sd_curr['gates'][layer]
        hs_curr = sd_curr['hs'][layer]
        n_curr = sd_curr['n_total']

        gates_next = sd_next['gates'][layer]
        hs_next = sd_next['hs'][layer]
        n_next = sd_next['n_total']

        # Build cone at step t (from ALL positions at step t)
        h_mean = hs_curr.mean(axis=0)
        h_shift = h_mean - h_mean_single[layer]
        W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()
        scaffold_t = scaffold_single[layer] + W_gate @ h_shift

        all_resids_t = gates_curr - scaffold_t[np.newaxis, :]
        U, S, Vt = np.linalg.svd(all_resids_t, full_matrices=False)
        k = min(n_curr, Vt.shape[0])
        dirs_k = Vt[:k]

        # At step t+1, the GENERATED position (index n_curr) has a gate residual
        # Use step t+1's scaffold for the generated position
        h_mean_next = hs_next.mean(axis=0)
        h_shift_next = h_mean_next - h_mean_single[layer]
        scaffold_t1 = scaffold_single[layer] + W_gate @ h_shift_next

        gen_resid = gates_next[n_curr] - scaffold_t1  # The generated token's residual

        # How much of the generated token's residual is within step t's cone?
        proj = dirs_k.T @ (dirs_k @ gen_resid)
        new_info = gen_resid - proj
        frac_within = np.linalg.norm(proj) / (np.linalg.norm(gen_resid) + 1e-10)
        frac_new = np.linalg.norm(new_info) / (np.linalg.norm(gen_resid) + 1e-10)

        phase = "P" if i < n_prompt - 1 else "G"
        token = sd_curr['next_token'][:8]
        print(f"    [{phase}] step {i:2d} -> +'{token:>8s}': "
              f"within_cone={frac_within:.3f}  new={frac_new:.3f}")

    print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: CHAIN-OF-THOUGHT LENS HYPOTHESIS")
print("=" * 80)
print()

del model
torch.cuda.empty_cache()

results = {
    'n_prompts': len(PROMPTS),
    'n_gen_steps': N_GEN_STEPS,
    'prompts': PROMPTS,
    'generated_texts': [gd['generated_text'] for gd in all_generation_data],
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8p_cot_lens.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
