#!/usr/bin/env python3
"""
Phase 8n: What Is Thinking?
============================

From Finding 74 (Marble Geometry):
- Consecutive token positions create ~87° (nearly orthogonal) gate residuals
- This curvature is UNIVERSAL (CV=0.065) across all prompts
- D* = n_pos - 1: each token adds one independent direction

The fundamental question: WHERE does the 87° come from?

Three hypotheses:
A) ATTENTION creates orthogonal hidden states → W_gate just preserves them
B) Hidden states are correlated → W_gate DECORRELATES them to ~87°
C) Both contribute — attention partially decorrelates, W_gate finishes the job

This tells us what "thinking" actually is:
- If A: thinking happens IN attention (constructing orthogonal representations)
- If B: thinking happens IN the gate projection (decorrelating inputs)
- If C: thinking is a two-stage process (diversify then decorrelate)

Additional tests:
- Does the attention pattern have structure that predicts the rotation?
- Is the orthogonality related to the causal mask (later tokens can't see earlier)?
- Does the ~87° have phi-structure?

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
print("  PHASE 8n: WHAT IS THINKING?")
print("  Where does the 87° orthogonality come from?")
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
HIDDEN_DIM = model.config.intermediate_size   # 18944 (gate space)
HIDDEN_STATE_DIM = model.config.hidden_size    # 3584  (hidden state space)


# ================================================================
# STEP 1: Build scaffold from single tokens
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
# STEP 2: Capture prompts with hidden states AND attention
# ================================================================
print("-" * 80)
print("  STEP 2: Capture prompts with gates, hidden states, and attention")
print("-" * 80)

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "The color of the sky is",
    "Albert Einstein developed the theory of",
    "The boiling point of water is",
    "The largest ocean on Earth is the",
    "The first president of the United States was",
    "The atomic number of carbon is",
    "The fastest land animal is the",
    "The Pythagorean theorem states that",
    "DNA stands for deoxyribonucleic",
    "Shakespeare wrote the play Romeo and",
    "The chemical formula for water is",
]

prompt_data = []
for prompt in PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    n_tok = input_ids.shape[1]

    gate_storage = {}
    hs_storage = {}
    attn_storage = {}
    hooks = []

    def make_gate_hook2(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = output.detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_hs_hook2(storage, layer_idx):
        def hook_fn(module, input, output):
            storage[layer_idx] = input[0].detach().cpu().float().numpy().squeeze()
        return hook_fn

    def make_attn_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            # output is (hidden_states, attn_weights, past_key_values) when output_attentions=True
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                storage[layer_idx] = output[1].detach().cpu().float().numpy().squeeze()
        return hook_fn

    for layer in range(N_LAYERS):
        h1 = model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            make_gate_hook2(gate_storage, layer)
        )
        h2 = model.model.layers[layer].mlp.register_forward_hook(
            make_hs_hook2(hs_storage, layer)
        )
        h3 = model.model.layers[layer].self_attn.register_forward_hook(
            make_attn_hook(attn_storage, layer)
        )
        hooks.extend([h1, h2, h3])

    with torch.no_grad():
        model(input_ids, output_attentions=True)

    for h in hooks:
        h.remove()

    prompt_data.append({
        'prompt': prompt,
        'n_tokens': n_tok,
        'gates': {l: gate_storage[l] for l in range(N_LAYERS)},
        'hs': {l: hs_storage[l] for l in range(N_LAYERS)},
        'attn': {l: attn_storage[l] for l in attn_storage},
    })

N_PROMPTS = len(prompt_data)
print(f"  Captured {N_PROMPTS} prompts")
has_attn = any(len(pd['attn']) > 0 for pd in prompt_data)
print(f"  Attention captured: {has_attn}")
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
# TEST 1: THE ORIGIN OF 87° -- hidden state angles vs gate angles
# ================================================================
print("=" * 80)
print("  TEST 1: WHERE DOES THE 87° COME FROM?")
print("  Compare angles in hidden state space vs gate content space")
print("=" * 80)
print()

# For each prompt, compute:
# A) Angles between consecutive HIDDEN STATE deviations (h[p] - h_mean)
# B) Angles between consecutive GATE RESIDUALS (gate[p] - scaffold)
# If A ≈ 87° → attention creates the orthogonality
# If A << 87° but B ≈ 87° → W_gate decorrelates

for layer in [6, 10, 14, 18, 22]:
    h_angles_all = []
    g_angles_all = []

    for pi, pd in enumerate(prompt_data):
        hs = pd['hs'][layer]         # [n_pos, 3584]
        gates = pd['gates'][layer]   # [n_pos, 18944]
        scaffold = corrected_scaffolds[pi][layer]
        n_pos = hs.shape[0]

        h_mean = hs.mean(axis=0)
        h_devs = hs - h_mean[np.newaxis, :]     # Hidden state deviations
        g_resids = gates - scaffold[np.newaxis, :]  # Gate residuals

        for p in range(n_pos - 1):
            # Hidden state angle
            h1, h2 = h_devs[p], h_devs[p + 1]
            hn1, hn2 = np.linalg.norm(h1), np.linalg.norm(h2)
            if hn1 > 1e-10 and hn2 > 1e-10:
                cos_h = np.clip(np.dot(h1, h2) / (hn1 * hn2), -1, 1)
                h_angles_all.append(np.degrees(np.arccos(cos_h)))

            # Gate residual angle
            g1, g2 = g_resids[p], g_resids[p + 1]
            gn1, gn2 = np.linalg.norm(g1), np.linalg.norm(g2)
            if gn1 > 1e-10 and gn2 > 1e-10:
                cos_g = np.clip(np.dot(g1, g2) / (gn1 * gn2), -1, 1)
                g_angles_all.append(np.degrees(np.arccos(cos_g)))

    h_mean_angle = np.mean(h_angles_all)
    g_mean_angle = np.mean(g_angles_all)
    h_std = np.std(h_angles_all)
    g_std = np.std(g_angles_all)

    print(f"  Layer {layer:2d}:")
    print(f"    Hidden state angles:  {h_mean_angle:5.1f}° ± {h_std:5.1f}°  "
          f"(range=[{min(h_angles_all):5.1f}°, {max(h_angles_all):5.1f}°])")
    print(f"    Gate residual angles:  {g_mean_angle:5.1f}° ± {g_std:5.1f}°  "
          f"(range=[{min(g_angles_all):5.1f}°, {max(g_angles_all):5.1f}°])")
    delta = g_mean_angle - h_mean_angle
    print(f"    Delta (gate - hidden): {delta:+5.1f}°")
    if abs(delta) < 5:
        print(f"    >> ATTENTION creates the orthogonality (W_gate preserves it)")
    elif delta > 15:
        print(f"    >> W_gate DECORRELATES (adds {delta:.0f}° of orthogonality)")
    else:
        print(f"    >> BOTH contribute (attention + W_gate)")
    print()


# ================================================================
# TEST 2: NON-CONSECUTIVE ANGLES -- is every pair ~87° or just neighbors?
# ================================================================
print("=" * 80)
print("  TEST 2: ANGLE STRUCTURE -- all pairs, not just consecutive")
print("  Is every pair ~87° or is there structure (decay, clustering)?")
print("=" * 80)
print()

layer = 14
# For each prompt, compute pairwise angles between ALL position residuals
# If all pairs ≈ 87° → positions form an equiangular frame
# If angles increase with distance → there's a path structure

for pi, pd in enumerate(prompt_data[:4]):  # Just a few examples
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    n_pos = gates.shape[0]
    g_resids = gates - scaffold[np.newaxis, :]

    print(f"  {pd['prompt'][:50]:>50s} (n_pos={n_pos})")
    print(f"  {'':>8s}", end="")
    for j in range(n_pos):
        print(f"  pos{j:d}", end="")
    print()

    for i in range(n_pos):
        print(f"  pos{i:d}  ", end="")
        for j in range(n_pos):
            if i == j:
                print(f"    --", end="")
            else:
                gi, gj = g_resids[i], g_resids[j]
                ni, nj = np.linalg.norm(gi), np.linalg.norm(gj)
                if ni > 1e-10 and nj > 1e-10:
                    cos_a = np.clip(np.dot(gi, gj) / (ni * nj), -1, 1)
                    angle = np.degrees(np.arccos(cos_a))
                    print(f"  {angle:4.0f}°", end="")
                else:
                    print(f"    ??", end="")
        print()
    print()


# ================================================================
# TEST 3: ANGLE VS DIMENSION -- does 87° relate to phi?
# ================================================================
print("=" * 80)
print("  TEST 3: IS 87° SPECIAL?")
print("  Check if angle relates to phi, random vectors, or dimension")
print("=" * 80)
print()

# What angle do random unit vectors make in different dimensions?
# In d dimensions, random unit vectors have expected angle arccos(0) = 90°
# The 87° is CLOSE to 90° but consistently below it.

# The deviation from 90° carries information
angle_deviation = 90.0 - 86.8  # ≈ 3.2°
print(f"  Mean angle from Finding 74: 86.8°")
print(f"  Deviation from orthogonal: {angle_deviation:.1f}°")
print(f"  cos(86.8°) = {np.cos(np.radians(86.8)):.6f}")
print()

# Compare with theoretical values
cos_87 = np.cos(np.radians(86.8))
print(f"  Is cos(86.8°) ≈ 1/√d?")
for d in [3584, 18944]:
    expected_cos = 1 / np.sqrt(d)
    expected_angle = np.degrees(np.arccos(expected_cos))
    print(f"    d={d:5d}: 1/√d = {expected_cos:.6f}, angle = {expected_angle:.1f}°")
print()

# Check phi relationships
print(f"  φ-structure checks:")
print(f"    cos(86.8°) = {cos_87:.6f}")
print(f"    1/φ³       = {1/PHI**3:.6f}")
print(f"    1/(2φ)     = {1/(2*PHI):.6f}")
print(f"    log(φ)/π   = {np.log(PHI)/np.pi:.6f}")
print(f"    1/φ⁴       = {1/PHI**4:.6f}")
print()

# Empirical test: generate random vectors and measure angles
print(f"  Monte Carlo comparison (10000 random vector pairs):")
for d in [3584, 18944]:
    v1 = np.random.randn(10000, d).astype(np.float32)
    v2 = np.random.randn(10000, d).astype(np.float32)
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    cos_vals = np.sum((v1/n1) * (v2/n2), axis=1)
    angles = np.degrees(np.arccos(np.clip(cos_vals, -1, 1)))
    print(f"    d={d:5d}: random angle = {np.mean(angles):.2f}° ± {np.std(angles):.2f}°")
print()


# ================================================================
# TEST 4: THE ATTENTION ROTATION
# ================================================================
print("=" * 80)
print("  TEST 4: WHAT ATTENTION DOES TO CREATE THE ROTATION")
print("  Does the attention pattern predict the gate subspace?")
print("=" * 80)
print()

# The hidden state at position p, entering MLP layer L, is:
#   h_L[p] = h_{L-1}[p] + attn_output_L[p]
# The gate residual at position p is:
#   g_resid[p] = W_gate @ (h_L[p] - h_L_mean)
# So the gate residual is a LINEAR function of the hidden state deviation.

# Key insight: the hidden state h[p] encodes ALL previous tokens via attention.
# Position 0 only sees itself. Position p sees tokens 0..p (causal mask).
# Each new position ADDS information from one more token.
# This is why D* ≈ n_pos - 1: each position's "new information" is roughly
# one dimension's worth, because it's the contribution of one new token
# that previous positions couldn't see.

# Let's verify: compute the "new information" at each position
# = the component of h[p] orthogonal to span(h[0]..h[p-1])

layer = 14
print(f"  Layer {layer}: New information per position (orthogonal component)")
print()

for pi, pd in enumerate(prompt_data[:6]):
    hs = pd['hs'][layer]
    n_pos = hs.shape[0]

    # Also compute for gate residuals
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    g_resids = gates - scaffold[np.newaxis, :]

    h_new_frac = []  # Fraction of h[p] that's new (orthogonal to previous)
    g_new_frac = []  # Same for gate residuals

    for p in range(n_pos):
        if p == 0:
            h_new_frac.append(1.0)
            g_new_frac.append(1.0)
            continue

        # Project h[p] onto span of h[0..p-1]
        prev_h = hs[:p]  # [p, 3584]
        Q_h, _ = np.linalg.qr(prev_h.T)  # orthogonal basis
        Q_h = Q_h[:, :p]  # [3584, p]
        proj_h = Q_h @ (Q_h.T @ hs[p])
        resid_h = hs[p] - proj_h
        frac_h = np.linalg.norm(resid_h) / (np.linalg.norm(hs[p]) + 1e-10)
        h_new_frac.append(frac_h)

        # Same for gate residuals
        prev_g = g_resids[:p]
        Q_g, _ = np.linalg.qr(prev_g.T)
        Q_g = Q_g[:, :p]
        proj_g = Q_g @ (Q_g.T @ g_resids[p])
        resid_g = g_resids[p] - proj_g
        frac_g = np.linalg.norm(resid_g) / (np.linalg.norm(g_resids[p]) + 1e-10)
        g_new_frac.append(frac_g)

    print(f"  {pd['prompt'][:45]:>45s}")
    for p in range(n_pos):
        h_bar = "#" * int(h_new_frac[p] * 30)
        g_bar = "#" * int(g_new_frac[p] * 30)
        print(f"    pos {p}: h_new={h_new_frac[p]:.3f} {h_bar}")
        print(f"            g_new={g_new_frac[p]:.3f} {g_bar}")
    print()


# ================================================================
# TEST 5: ATTENTION ENTROPY AND ROTATION
# ================================================================
print("=" * 80)
print("  TEST 5: ATTENTION PATTERN STRUCTURE")
print("  How does attention entropy relate to the marble rotation?")
print("=" * 80)
print()

if has_attn:
    layer = 14
    for pi, pd in enumerate(prompt_data[:4]):
        if layer not in pd['attn']:
            continue
        attn = pd['attn'][layer]  # [n_heads, n_pos, n_pos] or [n_pos, n_pos]
        n_pos = pd['n_tokens']

        print(f"  {pd['prompt'][:50]:>50s}")
        print(f"    Attention shape: {attn.shape}")

        # Average across heads if multi-headed
        if attn.ndim == 3:
            attn_avg = attn.mean(axis=0)  # [n_pos, n_pos]
        else:
            attn_avg = attn

        # Attention entropy per position
        for p in range(n_pos):
            row = attn_avg[p, :p+1]  # causal: only attend to 0..p
            row = row / (row.sum() + 1e-10)
            entropy = -np.sum(row * np.log(row + 1e-10))
            max_entropy = np.log(p + 1)
            norm_entropy = entropy / (max_entropy + 1e-10)
            print(f"    pos {p}: entropy={entropy:.3f}  normalized={norm_entropy:.3f}  "
                  f"max_attn={row.max():.3f} at pos {row.argmax()}")
        print()
else:
    print("  Attention not captured — testing via hidden state geometry only")
    print()

    # Alternative: measure how much each position's hidden state
    # CHANGES from the previous layer (attention contribution)
    print("  Hidden state change per position (proxy for attention effect):")
    print()

    for layer in [10, 14, 18]:
        print(f"  Layer {layer}:")
        for pi, pd in enumerate(prompt_data[:4]):
            hs_prev = pd['hs'][layer - 1] if layer > 0 else pd['hs'][0]
            hs_curr = pd['hs'][layer]
            n_pos = hs_curr.shape[0]

            changes = []
            for p in range(n_pos):
                delta = np.linalg.norm(hs_curr[p] - hs_prev[p])
                base = np.linalg.norm(hs_curr[p])
                changes.append(delta / base)

            print(f"    {pd['prompt'][:40]:>40s}: "
                  f"Δh/|h| = {np.mean(changes):.4f} ± {np.std(changes):.4f}")
        print()


# ================================================================
# TEST 6: THE CAUSAL STRUCTURE
# ================================================================
print("=" * 80)
print("  TEST 6: THE CAUSAL STRUCTURE OF THINKING")
print("  Does the causal mask explain the ~87° / (n_pos-1) DOF?")
print("=" * 80)
print()

# The causal mask means:
#   pos 0: sees only token 0
#   pos 1: sees tokens 0,1
#   pos p: sees tokens 0..p
#
# Each position has access to ONE more token than the previous.
# If each token contributes one "new direction" of information,
# then D* = n_pos - 1 naturally.
#
# But WHY would each new token contribute an orthogonal direction?
# Because the causal mask means:
#   h[p] - proj(h[p] onto span(h[0..p-1])) ≈ "what token p adds"
#   This is roughly orthogonal to h[0..p-1] by construction.
#
# The model is doing implicit Gram-Schmidt through the causal mask!
# Each position can only attend to tokens it hasn't seen yet (the new one).
# The attention output at position p is dominated by the new information
# from token p, which is roughly orthogonal to the previous positions'
# representations.

layer = 14
print(f"  Layer {layer}: Does each position's gate residual approximately")
print(f"  decompose as 'old information + new token direction'?")
print()

for pi, pd in enumerate(prompt_data[:4]):
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    n_pos = gates.shape[0]
    g_resids = gates - scaffold[np.newaxis, :]

    # For each position p, decompose g_resid[p] into:
    # 1. Projection onto span(g_resid[0..p-1])  = "old info"
    # 2. Orthogonal component = "new info from token p"
    # 3. What fraction of the NEW info is explained by ONE new direction?

    print(f"  {pd['prompt'][:50]:>50s}")

    for p in range(n_pos):
        if p == 0:
            print(f"    pos {p}: [first position — defines initial direction]")
            continue

        prev_g = g_resids[:p]  # [p, 18944]
        Q, _ = np.linalg.qr(prev_g.T)
        Q = Q[:, :min(p, prev_g.shape[0])]

        proj = Q @ (Q.T @ g_resids[p])
        new_info = g_resids[p] - proj

        frac_new = np.linalg.norm(new_info) / (np.linalg.norm(g_resids[p]) + 1e-10)
        frac_old = np.linalg.norm(proj) / (np.linalg.norm(g_resids[p]) + 1e-10)

        # What angle does the new info make with the full residual?
        if np.linalg.norm(new_info) > 1e-10:
            cos_new = np.dot(new_info, g_resids[p]) / (
                np.linalg.norm(new_info) * np.linalg.norm(g_resids[p]))
            angle_new = np.degrees(np.arccos(np.clip(cos_new, -1, 1)))
        else:
            angle_new = 90.0

        print(f"    pos {p}: old={frac_old:.3f}  new={frac_new:.3f}  "
              f"angle_from_new={angle_new:.1f}°")

    print()


# ================================================================
# TEST 7: W_gate AMPLIFICATION TEST
# ================================================================
print("=" * 80)
print("  TEST 7: W_gate AS AMPLIFIER/DECORRELATOR")
print("  Does W_gate amplify small angle differences into ~87°?")
print("=" * 80)
print()

# For each prompt at a given layer:
# 1. Compute pairwise angles in hidden state space
# 2. Compute pairwise angles in gate space
# 3. Is there a systematic amplification?

layer = 14
W_gate = model.model.layers[layer].mlp.gate_proj.weight.data.float().cpu().numpy()

all_h_angles = []
all_g_angles = []

for pi, pd in enumerate(prompt_data):
    hs = pd['hs'][layer]
    gates = pd['gates'][layer]
    scaffold = corrected_scaffolds[pi][layer]
    n_pos = hs.shape[0]

    h_mean = hs.mean(axis=0)
    h_devs = hs - h_mean[np.newaxis, :]
    g_resids = gates - scaffold[np.newaxis, :]

    for i in range(n_pos):
        for j in range(i + 1, n_pos):
            # Hidden state angle
            h1, h2 = h_devs[i], h_devs[j]
            hn1, hn2 = np.linalg.norm(h1), np.linalg.norm(h2)
            if hn1 > 1e-10 and hn2 > 1e-10:
                cos_h = np.clip(np.dot(h1, h2) / (hn1 * hn2), -1, 1)
                all_h_angles.append(np.degrees(np.arccos(cos_h)))

                # Corresponding gate angle
                g1, g2 = g_resids[i], g_resids[j]
                gn1, gn2 = np.linalg.norm(g1), np.linalg.norm(g2)
                if gn1 > 1e-10 and gn2 > 1e-10:
                    cos_g = np.clip(np.dot(g1, g2) / (gn1 * gn2), -1, 1)
                    all_g_angles.append(np.degrees(np.arccos(cos_g)))
                else:
                    all_g_angles.append(90.0)

all_h_angles = np.array(all_h_angles)
all_g_angles = np.array(all_g_angles[:len(all_h_angles)])

# Bin by hidden state angle and see the average gate angle
bins = [0, 30, 50, 60, 70, 75, 80, 85, 90, 95, 100, 110, 130, 180]
print(f"  Layer {layer}: Hidden state angle → Gate residual angle")
print(f"  {'H angle range':>20s}  {'N':>6s}  {'Mean G angle':>12s}  {'Std G angle':>12s}  {'Amplification':>14s}")
print("  " + "-" * 70)

for k in range(len(bins) - 1):
    mask = (all_h_angles >= bins[k]) & (all_h_angles < bins[k + 1])
    n = mask.sum()
    if n > 0:
        g_mean = all_g_angles[mask].mean()
        g_std = all_g_angles[mask].std()
        h_mid = (bins[k] + bins[k + 1]) / 2
        amp = g_mean - h_mid
        print(f"  {bins[k]:3d}°-{bins[k+1]:3d}°        {n:6d}  {g_mean:11.1f}°  {g_std:11.1f}°  {amp:+13.1f}°")

print()
corr = np.corrcoef(all_h_angles, all_g_angles)[0, 1]
print(f"  Correlation(h_angle, g_angle) = {corr:.4f}")
if corr > 0.7:
    print(f"  >> STRONG: W_gate preserves angle structure (attention creates orthogonality)")
elif corr > 0.3:
    print(f"  >> MODERATE: both attention and W_gate contribute")
else:
    print(f"  >> WEAK: W_gate substantially reshapes angle structure")
print()


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("  SUMMARY: WHAT IS THINKING?")
print("=" * 80)
print()

# Free model
del model
torch.cuda.empty_cache()

results = {
    'n_prompts': N_PROMPTS,
    'prompts': PROMPTS,
    'h_angles_mean': float(np.mean(all_h_angles)),
    'g_angles_mean': float(np.mean(all_g_angles)),
    'h_g_correlation': float(corr),
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8n_what_is_thinking.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
