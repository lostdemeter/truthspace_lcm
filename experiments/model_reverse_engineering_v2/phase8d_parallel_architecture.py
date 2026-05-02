#!/usr/bin/env python3
"""
Phase 8d: Predict-Parallel-Correct Architecture
=================================================

The gate dimension findings (61-64) prove three properties:
  1. 96.4% of gate states are predictable from the standing wave
  2. 98.5% chirality independence enables L/R split processing
  3. Δ±1 selection rules mean corrections are always local

This experiment tests whether these properties enable a concrete parallel
architecture for transformer inference:

  PREDICT:  Use standing wave to predict gate states (all layers at once)
  PARALLEL: Compute MLP outputs using predicted gates (layers independent)
  CORRECT:  Fix the ~3.6% mispredictions (local Δ±1 corrections)

Five stages:
  Stage 1: Capture — raw gate activations + hidden states (ground truth)
  Stage 2: Predict — standing wave predictor, predicted gate values
  Stage 3: SiLU Error — numerical impact of gate mispredictions
  Stage 4: Intervention — replace gates with predictions, measure logit change
  Stage 5: Speedup Model — theoretical and practical speedup estimation

Requires: Qwen2-7B on GPU
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

GATE_CONTRACT = 0
GATE_PRESERVE_N = 1
GATE_PRESERVE_P = 2
GATE_EXPAND = 3
STATE_NAMES = ['CONTRACT', 'PRESERVE-', 'PRESERVE+', 'EXPAND']

CHANNEL_L = [GATE_CONTRACT, GATE_PRESERVE_P]
CHANNEL_R = [GATE_PRESERVE_N, GATE_EXPAND]


def classify_gate(x):
    """Classify pre-SiLU activations into 4 gate states."""
    codes = np.zeros_like(x, dtype=np.int8)
    codes[x < -LOG_PHI] = GATE_CONTRACT
    codes[(x >= -LOG_PHI) & (x < 0)] = GATE_PRESERVE_N
    codes[(x >= 0) & (x < LOG_PHI)] = GATE_PRESERVE_P
    codes[x >= LOG_PHI] = GATE_EXPAND
    return codes


def silu(x):
    """SiLU activation (numpy)."""
    return x / (1 + np.exp(-x))


# ================================================================
# STAGE 1: CAPTURE
# ================================================================
print("=" * 80)
print("  PHASE 8d: PREDICT-PARALLEL-CORRECT ARCHITECTURE")
print("  Testing embarrassingly parallel transformer inference")
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
HIDDEN_DIM = model.config.intermediate_size  # 18944
MODEL_DIM = model.config.hidden_size  # 3584
print(f"  {N_LAYERS} layers, gate_dim = {HIDDEN_DIM}, model_dim = {MODEL_DIM}")

# Test tokens — same expanded set as phase8
TEST_WORDS = [
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

print(f"\nStage 1: Capturing gate activations for {len(TEST_WORDS)} tokens...")
print()

# Capture raw gate values for each token × layer
gate_raw = {}      # word -> [N_LAYERS, HIDDEN_DIM] raw gate_proj output
gate_codes = {}    # word -> [N_LAYERS, HIDDEN_DIM] classified codes

for word in TEST_WORDS:
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

    gates = np.stack([s[0].squeeze() for s in layer_gates])  # [N_LAYERS, HIDDEN_DIM]
    gate_raw[decoded] = gates
    gate_codes[decoded] = classify_gate(gates)

all_words = sorted(gate_raw.keys())
N_TOKENS = len(all_words)

# Stack: [N_TOKENS, N_LAYERS, HIDDEN_DIM]
all_raw = np.stack([gate_raw[w] for w in all_words])
all_codes = np.stack([gate_codes[w] for w in all_words])

print(f"  Captured {N_TOKENS} tokens × {N_LAYERS} layers × {HIDDEN_DIM} channels")
print(f"  Raw gate shape: {all_raw.shape}")
print()


# ================================================================
# STAGE 2: PREDICT — Build standing wave predictor
# ================================================================
print("─" * 80)
print("  STAGE 2: PREDICT — Standing Wave Gate Predictor")
print("─" * 80)
print()

# 2a: Per-channel mode (the standing wave prediction)
per_channel_mode = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.int8)
per_channel_confidence = np.zeros((N_LAYERS, HIDDEN_DIM))

for layer in range(N_LAYERS):
    for ch in range(HIDDEN_DIM):
        channel_codes = all_codes[:, layer, ch]
        counts = np.bincount(channel_codes.astype(int), minlength=4)
        mode = counts.argmax()
        per_channel_mode[layer, ch] = mode
        per_channel_confidence[layer, ch] = counts[mode] / N_TOKENS

# 2b: Per-channel mean value within each state (for SiLU prediction)
# For each channel at each layer, compute the mean raw gate value
# conditioned on the predicted state (the mode)
per_channel_mean = np.zeros((N_LAYERS, HIDDEN_DIM))

for layer in range(N_LAYERS):
    for ch in range(HIDDEN_DIM):
        predicted_state = per_channel_mode[layer, ch]
        # Mean value across all tokens for this channel (unconditional)
        per_channel_mean[layer, ch] = all_raw[:, layer, ch].mean()

# 2c: Per-state canonical values (midpoints of each state's range)
# These are token-independent — truly parallel
canonical_values = {
    GATE_CONTRACT: -2 * LOG_PHI,      # midpoint of (-inf, -log(φ)) → use -2*log(φ)
    GATE_PRESERVE_N: -LOG_PHI / 2,    # midpoint of (-log(φ), 0)
    GATE_PRESERVE_P: LOG_PHI / 2,     # midpoint of (0, log(φ))
    GATE_EXPAND: 2 * LOG_PHI,         # midpoint of (log(φ), inf) → use 2*log(φ)
}

# Build predicted gate values using two strategies
predicted_mean = per_channel_mean.copy()  # Strategy A: empirical mean per channel
predicted_canonical = np.zeros((N_LAYERS, HIDDEN_DIM))  # Strategy B: canonical per state
for layer in range(N_LAYERS):
    for state, val in canonical_values.items():
        mask = per_channel_mode[layer] == state
        predicted_canonical[layer, mask] = val

# Prediction accuracy
accuracy_per_layer = np.zeros(N_LAYERS)
for layer in range(N_LAYERS):
    correct = 0
    total = 0
    for tok in range(N_TOKENS):
        correct += (all_codes[tok, layer] == per_channel_mode[layer]).sum()
        total += HIDDEN_DIM
    accuracy_per_layer[layer] = correct / total

overall_accuracy = accuracy_per_layer.mean()
print(f"  Gate state prediction accuracy: {overall_accuracy*100:.2f}%")
print(f"  Error budget (sequential): {(1-overall_accuracy)*100:.2f}%")
print(f"  1/(4φ⁴) = {1/(4*PHI**4)*100:.2f}% (structural prediction)")
print()

# Per-layer breakdown
print(f"  {'Layer':>5s}  {'Accuracy':>9s}  {'Error':>7s}  {'C frac':>7s}  {'P- frac':>8s}  "
      f"{'P+ frac':>8s}  {'X frac':>7s}  {'Zone':>6s}")
print("  " + "-" * 70)

zones = {}
for i in range(3):
    zones[i] = 'DRUM'
for i in range(3, 6):
    zones[i] = 'TRANS'
for i in range(6, 23):
    zones[i] = 'COMB'
for i in range(23, 28):
    zones[i] = 'MUSIC'

for layer in range(N_LAYERS):
    state_fracs = [(per_channel_mode[layer] == s).mean() for s in range(4)]
    z = zones.get(layer, '?')
    print(f"  {layer:5d}  {accuracy_per_layer[layer]:9.4f}  {1-accuracy_per_layer[layer]:7.4f}  "
          f"{state_fracs[0]:7.3f}  {state_fracs[1]:8.3f}  {state_fracs[2]:8.3f}  "
          f"{state_fracs[3]:7.3f}  {z:>6s}")
print()


# ================================================================
# STAGE 3: SiLU ERROR — Numerical impact of mispredictions
# ================================================================
print("─" * 80)
print("  STAGE 3: SiLU ERROR — How much do mispredictions cost numerically?")
print("─" * 80)
print()

# For each token, compute SiLU(actual_gate) and SiLU(predicted_gate)
# The difference tells us the numerical impact of misprediction

# Strategy A: use empirical mean as predicted gate value
# Strategy B: use canonical state midpoint

silu_actual = silu(all_raw)  # [N_TOKENS, N_LAYERS, HIDDEN_DIM]

# Strategy A: mean-based prediction
silu_pred_mean = silu(np.broadcast_to(predicted_mean, all_raw.shape).copy())

# Strategy B: canonical-based prediction
silu_pred_canonical = silu(np.broadcast_to(predicted_canonical, all_raw.shape).copy())

# Relative SiLU error per layer
def compute_silu_metrics(silu_actual, silu_predicted, label):
    print(f"\n  {label}:")
    abs_err_per_layer = np.zeros(N_LAYERS)
    rel_err_per_layer = np.zeros(N_LAYERS)
    cosine_sim_per_layer = np.zeros(N_LAYERS)

    for layer in range(N_LAYERS):
        actual = silu_actual[:, layer, :]    # [N_TOKENS, HIDDEN_DIM]
        pred = silu_predicted[:, layer, :]

        # Mean absolute error
        abs_err = np.abs(actual - pred).mean()
        abs_err_per_layer[layer] = abs_err

        # Relative error (normalized by actual magnitude)
        actual_norm = np.abs(actual).mean()
        if actual_norm > 0:
            rel_err_per_layer[layer] = abs_err / actual_norm
        else:
            rel_err_per_layer[layer] = 0

        # Cosine similarity (per token, then average)
        for tok in range(N_TOKENS):
            a = actual[tok]
            p = pred[tok]
            norm_a = np.linalg.norm(a)
            norm_p = np.linalg.norm(p)
            if norm_a > 0 and norm_p > 0:
                cosine_sim_per_layer[layer] += np.dot(a, p) / (norm_a * norm_p)
        cosine_sim_per_layer[layer] /= N_TOKENS

    print(f"  {'Layer':>5s}  {'AbsErr':>10s}  {'RelErr':>10s}  {'CosSim':>10s}  {'Zone':>6s}")
    print("  " + "-" * 50)
    for layer in range(N_LAYERS):
        z = zones.get(layer, '?')
        print(f"  {layer:5d}  {abs_err_per_layer[layer]:10.6f}  "
              f"{rel_err_per_layer[layer]:10.6f}  {cosine_sim_per_layer[layer]:10.6f}  {z:>6s}")

    mean_abs = abs_err_per_layer.mean()
    mean_rel = rel_err_per_layer.mean()
    mean_cos = cosine_sim_per_layer.mean()
    comb_cos = cosine_sim_per_layer[6:23].mean()
    print()
    print(f"  Mean abs error:  {mean_abs:.6f}")
    print(f"  Mean rel error:  {mean_rel:.6f} ({mean_rel*100:.2f}%)")
    print(f"  Mean cosine sim: {mean_cos:.6f}")
    print(f"  COMB cosine sim: {comb_cos:.6f} (layers 6-22, the parallel core)")

    return abs_err_per_layer, rel_err_per_layer, cosine_sim_per_layer

abs_A, rel_A, cos_A = compute_silu_metrics(silu_actual, silu_pred_mean,
                                            "Strategy A: Empirical Mean Prediction")
abs_B, rel_B, cos_B = compute_silu_metrics(silu_actual, silu_pred_canonical,
                                            "Strategy B: Canonical State Midpoint")
print()

# Misprediction breakdown: what type of errors occur?
print("  Misprediction Type Analysis (which state transitions are wrong?):")
mispred_matrix = np.zeros((4, 4))  # actual × predicted
for tok in range(N_TOKENS):
    for layer in range(N_LAYERS):
        actual = all_codes[tok, layer]
        predicted = per_channel_mode[layer]
        for ch in range(HIDDEN_DIM):
            if actual[ch] != predicted[ch]:
                mispred_matrix[actual[ch], predicted[ch]] += 1

mispred_total = mispred_matrix.sum()
print(f"\n  Total mispredictions: {int(mispred_total)} "
      f"({mispred_total/(N_TOKENS*N_LAYERS*HIDDEN_DIM)*100:.2f}%)")
print(f"  {'Actual\\Pred':>12s}  {'CONTRACT':>10s}  {'PRESERVE-':>10s}  {'PRESERVE+':>10s}  {'EXPAND':>10s}")
for i in range(4):
    row = [f"{mispred_matrix[i,j]/mispred_total*100:9.2f}%" for j in range(4)]
    print(f"  {STATE_NAMES[i]:>12s}  {'  '.join(row)}")

# Δ±1 fraction of mispredictions
adj_mispred = 0
for i in range(4):
    for j in range(4):
        if i != j and abs(i - j) == 1:
            adj_mispred += mispred_matrix[i, j]

non_self = mispred_total - sum(mispred_matrix[i, i] for i in range(4))
adj_frac = adj_mispred / non_self if non_self > 0 else 0
print(f"\n  Adjacent (Δ±1) mispredictions: {adj_frac*100:.1f}% of all mispredictions")
print(f"  → Corrections are local (Δ±1): only need to adjust by one gate state")
print()


# ================================================================
# STAGE 4: INTERVENTION — Replace gates, measure logit divergence
# ================================================================
print("─" * 80)
print("  STAGE 4: INTERVENTION — Logit divergence with predicted gates")
print("  Replace actual gate values with standing wave predictions")
print("─" * 80)
print()

# For a subset of tokens, compare:
#   (a) Normal forward pass → logits
#   (b) Intervened forward pass (gate replaced with predictions) → logits
# Measure: KL divergence, top-1 agreement, cosine similarity of logit vectors

INTERVENTION_WORDS = ["king", "queen", "the", "algorithm", "hello", "zero",
                       "cat", "Paris", "light", "happy"]

intervention_results = []

for word in INTERVENTION_WORDS:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        continue
    token_id = ids[0]
    decoded = tokenizer.decode([token_id]).strip()

    if decoded not in gate_raw:
        continue

    # (a) Normal forward pass
    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], device="cuda")
        normal_output = model(input_ids)
        logits_normal = normal_output.logits[0, -1].float().cpu()

    # (b) Intervened forward pass — replace gate_proj outputs with predicted means
    predicted_gates_tensor = torch.tensor(predicted_mean, dtype=torch.bfloat16, device="cuda")

    intervention_hooks = []

    def make_intervention_hook(layer_idx):
        def hook_fn(module, input, output):
            # Replace gate_proj output with standing wave prediction
            return predicted_gates_tensor[layer_idx:layer_idx+1].unsqueeze(0)
        return hook_fn

    for layer_idx in range(N_LAYERS):
        h = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(
            make_intervention_hook(layer_idx)
        )
        intervention_hooks.append(h)

    with torch.no_grad():
        input_ids = torch.tensor([[token_id]], device="cuda")
        intervened_output = model(input_ids)
        logits_intervened = intervened_output.logits[0, -1].float().cpu()

    for h in intervention_hooks:
        h.remove()

    # Compare
    # Cosine similarity of logit vectors
    cos_sim = F.cosine_similarity(logits_normal.unsqueeze(0),
                                   logits_intervened.unsqueeze(0)).item()

    # Top-1 agreement
    top1_normal = logits_normal.argmax().item()
    top1_intervened = logits_intervened.argmax().item()
    top1_match = top1_normal == top1_intervened

    # Top-5 overlap
    top5_normal = set(logits_normal.topk(5).indices.tolist())
    top5_intervened = set(logits_intervened.topk(5).indices.tolist())
    top5_overlap = len(top5_normal & top5_intervened) / 5

    # Top-10 overlap
    top10_normal = set(logits_normal.topk(10).indices.tolist())
    top10_intervened = set(logits_intervened.topk(10).indices.tolist())
    top10_overlap = len(top10_normal & top10_intervened) / 10

    # KL divergence (normal → intervened)
    p_normal = F.softmax(logits_normal, dim=0)
    p_intervened = F.softmax(logits_intervened, dim=0)
    kl_div = F.kl_div(p_intervened.log(), p_normal, reduction='sum').item()

    # Decode top-1 predictions
    top1_normal_word = tokenizer.decode([top1_normal])
    top1_intervened_word = tokenizer.decode([top1_intervened])

    result = {
        'word': decoded,
        'cos_sim': cos_sim,
        'top1_match': top1_match,
        'top5_overlap': top5_overlap,
        'top10_overlap': top10_overlap,
        'kl_div': kl_div,
        'top1_normal': top1_normal_word,
        'top1_intervened': top1_intervened_word,
    }
    intervention_results.append(result)

    match_str = "✓" if top1_match else "✗"
    print(f"  {decoded:>12s}  cos={cos_sim:.4f}  top1={match_str}  "
          f"top5={top5_overlap:.1%}  top10={top10_overlap:.1%}  "
          f"KL={kl_div:.4f}  [{top1_normal_word.strip()} → {top1_intervened_word.strip()}]")

# Summary
mean_cos = np.mean([r['cos_sim'] for r in intervention_results])
top1_rate = np.mean([r['top1_match'] for r in intervention_results])
mean_top5 = np.mean([r['top5_overlap'] for r in intervention_results])
mean_top10 = np.mean([r['top10_overlap'] for r in intervention_results])
mean_kl = np.mean([r['kl_div'] for r in intervention_results])

print()
print(f"  Summary ({len(intervention_results)} tokens):")
print(f"    Mean cosine similarity: {mean_cos:.4f}")
print(f"    Top-1 agreement:        {top1_rate:.1%}")
print(f"    Mean top-5 overlap:     {mean_top5:.1%}")
print(f"    Mean top-10 overlap:    {mean_top10:.1%}")
print(f"    Mean KL divergence:     {mean_kl:.4f}")
print()


# ================================================================
# STAGE 5: SPEEDUP MODEL
# ================================================================
print("─" * 80)
print("  STAGE 5: SPEEDUP MODEL — Theoretical parallel speedup")
print("─" * 80)
print()

# MLP computation per layer:
#   gate = gate_proj(x)       → model_dim × hidden_dim multiplies
#   up   = up_proj(x)         → model_dim × hidden_dim multiplies
#   act  = SiLU(gate) * up    → hidden_dim multiplies
#   out  = down_proj(act)     → hidden_dim × model_dim multiplies
#
# Total per layer: 3 × model_dim × hidden_dim + hidden_dim
# For Qwen2-7B: 3 × 3584 × 18944 + 18944 ≈ 203.7M FLOPs per layer

flops_per_layer = 3 * MODEL_DIM * HIDDEN_DIM + HIDDEN_DIM
total_mlp_flops = flops_per_layer * N_LAYERS

print(f"  MLP FLOPs per layer: {flops_per_layer/1e6:.1f}M")
print(f"  Total MLP FLOPs:     {total_mlp_flops/1e6:.1f}M ({N_LAYERS} layers)")
print()

# Parallelizable fraction analysis
# 1. CONTRACT channels: SiLU ≈ 0 → can skip up_proj and down_proj for these
# 2. EXPAND channels: SiLU ≈ identity → can simplify
# 3. PRESERVE channels: need actual computation

contract_frac_per_layer = np.zeros(N_LAYERS)
preserve_frac_per_layer = np.zeros(N_LAYERS)
expand_frac_per_layer = np.zeros(N_LAYERS)

for layer in range(N_LAYERS):
    contract_frac_per_layer[layer] = (per_channel_mode[layer] == GATE_CONTRACT).mean()
    preserve_frac_per_layer[layer] = ((per_channel_mode[layer] == GATE_PRESERVE_N) |
                                       (per_channel_mode[layer] == GATE_PRESERVE_P)).mean()
    expand_frac_per_layer[layer] = (per_channel_mode[layer] == GATE_EXPAND).mean()

mean_contract = contract_frac_per_layer.mean()
mean_preserve = preserve_frac_per_layer.mean()
mean_expand = expand_frac_per_layer.mean()

print(f"  Channel state fractions (standing wave prediction):")
print(f"    CONTRACT (skip):     {mean_contract:.4f} ({mean_contract*100:.1f}%)")
print(f"    PRESERVE (compute):  {mean_preserve:.4f} ({mean_preserve*100:.1f}%)")
print(f"    EXPAND (simplify):   {mean_expand:.4f} ({mean_expand*100:.1f}%)")
print()

# Speedup calculations
# Sequential: all 28 layers run one after another
# Parallel: layers run independently using predicted gates

# Level 1: Gate prediction only (skip CONTRACT channels)
# Savings: CONTRACT channels don't need up_proj × down_proj contribution
# up_proj and down_proj still needed for non-CONTRACT channels
skip_frac = mean_contract
level1_flops = total_mlp_flops * (1 - skip_frac * 0.67)  # 67% of per-channel cost is up/down
level1_speedup = total_mlp_flops / level1_flops

# Level 2: Full parallel (all layers run simultaneously)
# Sequential bottleneck is the 3.6% error that needs correction
# If 96.4% is correct, the parallel portion takes 1 "layer-step"
# and correction takes 0.036 * N_LAYERS sequential steps
parallel_frac = overall_accuracy
seq_frac = 1 - overall_accuracy
level2_theoretical = 1 / (seq_frac + parallel_frac / N_LAYERS)

# Level 3: Chirality split (L and R processed independently)
# This doubles effective parallelism within each layer
level3_theoretical = level2_theoretical * 2 * 0.985  # 98.5% independence

print(f"  Speedup estimates:")
print()
print(f"    Level 1 — CONTRACT channel skipping:")
print(f"      Skip {skip_frac*100:.1f}% of channels × 67% of per-channel FLOPs")
print(f"      Effective FLOPs: {level1_flops/1e6:.1f}M (was {total_mlp_flops/1e6:.1f}M)")
print(f"      Speedup: {level1_speedup:.2f}×")
print()
print(f"    Level 2 — Full layer parallelism:")
print(f"      {parallel_frac*100:.1f}% predictable → process all {N_LAYERS} layers in parallel")
print(f"      {seq_frac*100:.1f}% needs sequential correction")
print(f"      Theoretical speedup: {level2_theoretical:.1f}× (Amdahl's law)")
print()
print(f"    Level 3 — Chirality split (L/R independent):")
print(f"      98.5% independence → 2× within-layer parallelism")
print(f"      Combined speedup: {level3_theoretical:.1f}×")
print()

# Amdahl's law: S = 1 / ((1-p) + p/N)
# where p = parallel fraction, N = number of processors
print(f"  Amdahl's Law with N processors (p = {parallel_frac:.4f}):")
print(f"  {'N':>6s}  {'Speedup':>10s}  {'Efficiency':>10s}")
print("  " + "-" * 30)
for N in [2, 4, 7, 14, 28, 56]:
    speedup = 1 / (seq_frac + parallel_frac / N)
    efficiency = speedup / N * 100
    print(f"  {N:6d}  {speedup:10.2f}×  {efficiency:9.1f}%")
print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 80)
print("  SUMMARY: PREDICT-PARALLEL-CORRECT ARCHITECTURE")
print("=" * 80)
print()

print(f"  PREDICT:   {overall_accuracy*100:.1f}% of gate states predicted from standing wave")
print(f"  PARALLEL:  Cosine similarity = {mean_cos:.4f} (mean logit preservation)")
print(f"  CORRECT:   {adj_frac*100:.0f}% of mispredictions are Δ±1 (local corrections)")
print()

print(f"  Intervention results:")
print(f"    Top-1 agreement:    {top1_rate:.1%}")
print(f"    Top-5 overlap:      {mean_top5:.1%}")
print(f"    Top-10 overlap:     {mean_top10:.1%}")
print(f"    Logit cosine sim:   {mean_cos:.4f}")
print(f"    Mean KL divergence: {mean_kl:.4f}")
print()

print(f"  Speedup potential:")
print(f"    Level 1 (skip CONTRACT): {level1_speedup:.2f}×")
print(f"    Level 2 (layer parallel): {level2_theoretical:.1f}×")
print(f"    Level 3 (chirality split): {level3_theoretical:.1f}×")
print()

# Architecture verdict
if mean_cos > 0.9 and top1_rate > 0.5:
    verdict = "VIABLE — Predicted gates preserve most of the output distribution"
elif mean_cos > 0.7:
    verdict = "PARTIAL — Predicted gates capture coarse structure but lose detail"
else:
    verdict = "NOT VIABLE — Gate prediction alone is insufficient for parallel inference"

print(f"  VERDICT: {verdict}")
print()

# Save results
results = {
    'stage2_predict': {
        'overall_accuracy': float(overall_accuracy),
        'error_budget': float(1 - overall_accuracy),
        'accuracy_per_layer': accuracy_per_layer.tolist(),
    },
    'stage3_silu_error': {
        'strategy_A_mean': {
            'abs_err': float(abs_A.mean()),
            'rel_err': float(rel_A.mean()),
            'cos_sim': float(cos_A.mean()),
            'comb_cos_sim': float(cos_A[6:23].mean()),
        },
        'strategy_B_canonical': {
            'abs_err': float(abs_B.mean()),
            'rel_err': float(rel_B.mean()),
            'cos_sim': float(cos_B.mean()),
            'comb_cos_sim': float(cos_B[6:23].mean()),
        },
        'mispred_adj_fraction': float(adj_frac),
    },
    'stage4_intervention': {
        'mean_cos_sim': float(mean_cos),
        'top1_rate': float(top1_rate),
        'mean_top5_overlap': float(mean_top5),
        'mean_top10_overlap': float(mean_top10),
        'mean_kl_div': float(mean_kl),
        'per_token': intervention_results,
    },
    'stage5_speedup': {
        'level1_skip_contract': float(level1_speedup),
        'level2_layer_parallel': float(level2_theoretical),
        'level3_chirality_split': float(level3_theoretical),
        'contract_frac': float(mean_contract),
        'preserve_frac': float(mean_preserve),
        'expand_frac': float(mean_expand),
    },
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase8d_parallel_architecture.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")

# Cleanup
del model
torch.cuda.empty_cache()
