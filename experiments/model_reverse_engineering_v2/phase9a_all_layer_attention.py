#!/usr/bin/env python3
"""
Phase 9a: All-Layer Attention Head Characterization
=====================================================

Phase 4 (Findings 38-47) characterized Layer 23's attention heads:
- 20/28 FIXED heads (always attend to pos 0)
- 8 ROUTING heads with rank-1 MESH
- Two routing families: content (all -1s) vs position (mixed signs)

This experiment extends that analysis to ALL 28 layers:

1. For each layer, for each head:
   - Classify as FIXED vs ROUTING (entropy + argmax stability)
   - Compute MESH SVD → rank, condition number, Zipf α
   - Extract d_k direction → check if all-negative (content) or mixed (position)

2. Produce a map: which layers need geometric resonators?
   - Layers where ALL heads are fixed → skip attention entirely
   - Layers with routing heads → need resonator(s)

3. Estimate total compute savings if resonator works at all layers.

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

print("=" * 80)
print("  PHASE 9a: ALL-LAYER ATTENTION HEAD CHARACTERIZATION")
print("  Extending Finding 38 from Layer 23 → all 28 layers")
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
    attn_implementation="eager",
)
model.eval()

N_LAYERS = len(model.model.layers)       # 28
NUM_HEADS = model.config.num_attention_heads      # 28
NUM_KV_HEADS = model.config.num_key_value_heads   # 4
HEAD_DIM = model.config.hidden_size // NUM_HEADS  # 128
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS          # 7
HIDDEN_DIM = model.config.hidden_size              # 3584

print(f"  {N_LAYERS} layers, {NUM_HEADS} heads, {NUM_KV_HEADS} KV heads, head_dim={HEAD_DIM}")
print()


# ================================================================
# Calibration prompts — diverse set to measure head stability
# ================================================================
CAL_PROMPTS = [
    "The capital of France is",
    "The largest ocean is the",
    "The color of grass is",
    "Barack Obama was the",
    "To be or not to",
    "Roses are red, violets are",
    "1 + 1 =",
    "2 + 2 =",
    "The sky is",
    "Water is made of",
    "The sun rises in the",
    "Once upon a time",
    "She walked into the room and",
    "The quick brown fox",
    "In machine learning",
    "Python is a programming",
    "The largest planet is",
    "Albert Einstein developed the",
    "Shakespeare wrote many",
    "The speed of light is",
]


# ================================================================
# STEP 1: Collect attention patterns across all layers
# ================================================================
print("-" * 80)
print("  STEP 1: Collect attention patterns for all layers")
print("-" * 80)
print()

# For each prompt, collect attention weights at every layer
# attention_weights[layer][head] = list of (argmax_pos, entropy) per prompt
layer_head_data = {}
for layer in range(N_LAYERS):
    layer_head_data[layer] = {h: {'argmaxes': [], 'entropies': []} for h in range(NUM_HEADS)}

for pi, prompt in enumerate(CAL_PROMPTS):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model(input_ids, output_attentions=True, return_dict=True)

    for layer in range(N_LAYERS):
        attn = out.attentions[layer]  # (1, num_heads, seq_len, seq_len)
        for h in range(NUM_HEADS):
            w = attn[0, h, -1, :].float().cpu().numpy()  # last token's attention
            am = int(np.argmax(w))
            ent = float(-np.sum(w * np.log(w + 1e-20)))
            layer_head_data[layer][h]['argmaxes'].append(am)
            layer_head_data[layer][h]['entropies'].append(ent)

    if (pi + 1) % 5 == 0:
        print(f"  Processed {pi + 1}/{len(CAL_PROMPTS)} prompts")

print()


# ================================================================
# STEP 2: Classify heads as FIXED vs ROUTING
# ================================================================
print("-" * 80)
print("  STEP 2: Classify heads across all layers")
print("-" * 80)
print()

layer_summary = []

print(f"  {'Layer':>5s}  {'Fixed':>5s}  {'Routing':>7s}  {'Mean H(fixed)':>13s}  {'Mean H(route)':>13s}  {'Routing heads':>30s}")
print("  " + "-" * 80)

for layer in range(N_LAYERS):
    fixed = []
    routing = []

    for h in range(NUM_HEADS):
        ams = layer_head_data[layer][h]['argmaxes']
        ents = layer_head_data[layer][h]['entropies']
        unique = len(set(ams))
        always_0 = sum(1 for a in ams if a == 0) / len(ams)
        mean_ent = np.mean(ents)

        if always_0 > 0.85 and mean_ent < 0.5:
            fixed.append(h)
        elif unique <= 2 and always_0 > 0.75:
            fixed.append(h)
        else:
            routing.append(h)

    mean_h_fixed = np.mean([np.mean(layer_head_data[layer][h]['entropies']) for h in fixed]) if fixed else 0
    mean_h_route = np.mean([np.mean(layer_head_data[layer][h]['entropies']) for h in routing]) if routing else 0

    routing_str = str(routing) if routing else "[]"
    print(f"  {layer:5d}  {len(fixed):5d}  {len(routing):7d}  {mean_h_fixed:13.3f}  {mean_h_route:13.3f}  {routing_str:>30s}")

    layer_summary.append({
        'layer': layer,
        'n_fixed': len(fixed),
        'n_routing': len(routing),
        'fixed_heads': fixed,
        'routing_heads': routing,
        'mean_entropy_fixed': float(mean_h_fixed),
        'mean_entropy_routing': float(mean_h_route),
    })

print()
total_fixed = sum(s['n_fixed'] for s in layer_summary)
total_routing = sum(s['n_routing'] for s in layer_summary)
total_heads = N_LAYERS * NUM_HEADS
print(f"  Total: {total_fixed}/{total_heads} fixed ({total_fixed/total_heads:.0%}), "
      f"{total_routing}/{total_heads} routing ({total_routing/total_heads:.0%})")
print()


# ================================================================
# STEP 3: MESH SVD for routing heads — rank and structure
# ================================================================
print("-" * 80)
print("  STEP 3: MESH SVD for routing heads")
print("  (MESH = W_q_head @ W_k_group^T in head-space, with bias)")
print("-" * 80)
print()

mesh_results = []

for ls in layer_summary:
    layer = ls['layer']
    if not ls['routing_heads']:
        continue

    layer_obj = model.model.layers[layer]
    attn = layer_obj.self_attn

    # Extract W_q, W_k with bias for routing heads
    for h in ls['routing_heads']:
        kv_group = h // HEADS_PER_KV

        # Extract via probing (same approach as phase4_geometric_selector.py)
        W_q_head = torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32)
        W_k_group = torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32)

        chunk_size = 512
        identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)

        for start in range(0, HIDDEN_DIM, chunk_size):
            end = min(start + chunk_size, HIDDEN_DIM)
            chunk = identity[start:end].unsqueeze(0)

            with torch.no_grad():
                q_out = attn.q_proj(chunk.float() if attn.q_proj.weight.dtype == torch.float32 else chunk)
                k_out = attn.k_proj(chunk.float() if attn.k_proj.weight.dtype == torch.float32 else chunk)

            q_out = q_out.float()
            k_out = k_out.float()

            q_reshaped = q_out[0].reshape(-1, NUM_HEADS, HEAD_DIM)
            k_reshaped = k_out[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)

            W_q_head[:, start:end] = q_reshaped[:, h, :].T
            W_k_group[:, start:end] = k_reshaped[:, kv_group, :].T

        # MESH in head-space
        MESH = (W_q_head @ W_k_group.T).cpu().numpy()
        U, S, Vt = np.linalg.svd(MESH)

        # Key metrics
        sv_ratio = float(S[0] / S[1]) if S[1] > 0 else float('inf')
        rank1_var = float(S[0]**2 / (S**2).sum() * 100)
        condition = float(S[0] / S[-1]) if S[-1] > 0 else float('inf')

        # Zipf α
        ranks = np.arange(1, min(21, len(S)) + 1)
        log_ranks = np.log(ranks)
        log_svs = np.log(S[:len(ranks)] + 1e-20)
        alpha = float(-np.polyfit(log_ranks, log_svs, 1)[0])

        # d_k direction (from MESH top singular vector projected to hidden space)
        d_k = (W_k_group.T @ torch.tensor(Vt[0], device="cuda", dtype=torch.float32)).cpu().numpy()
        all_negative = bool((d_k < 0).all())
        neg_frac = float((d_k < 0).sum() / len(d_k))

        family = "content" if all_negative or neg_frac > 0.95 else "position"

        mesh_results.append({
            'layer': layer,
            'head': h,
            'sv_ratio': sv_ratio,
            'rank1_var_pct': rank1_var,
            'condition_number': condition,
            'zipf_alpha': alpha,
            'all_negative': all_negative,
            'neg_fraction': neg_frac,
            'family': family,
        })

    print(f"  Layer {layer:2d}: {len(ls['routing_heads'])} routing heads analyzed")

print()

# Print MESH summary
print(f"  {'Layer':>5s}  {'Head':>4s}  {'S₀/S₁':>10s}  {'Rank-1%':>8s}  {'Zipf α':>7s}  {'Neg%':>5s}  {'Family':>10s}")
print("  " + "-" * 55)

for mr in mesh_results:
    sv_str = f"{mr['sv_ratio']:.0f}" if mr['sv_ratio'] < 1e6 else f"{mr['sv_ratio']:.1e}"
    print(f"  {mr['layer']:5d}  {mr['head']:4d}  {sv_str:>10s}  {mr['rank1_var_pct']:7.1f}%  {mr['zipf_alpha']:7.3f}  {mr['neg_fraction']*100:4.0f}%  {mr['family']:>10s}")

print()


# ================================================================
# STEP 4: Summary statistics
# ================================================================
print("=" * 80)
print("  SUMMARY: ALL-LAYER ATTENTION CHARACTERIZATION")
print("=" * 80)
print()

# Layers with zero routing heads
zero_routing = [s for s in layer_summary if s['n_routing'] == 0]
some_routing = [s for s in layer_summary if s['n_routing'] > 0]

print(f"  Layers with ALL fixed heads (skip attention): {len(zero_routing)}")
for s in zero_routing:
    print(f"    Layer {s['layer']}")

print(f"\n  Layers with routing heads (need resonator): {len(some_routing)}")
for s in some_routing:
    heads_str = ', '.join(str(h) for h in s['routing_heads'])
    print(f"    Layer {s['layer']:2d}: {s['n_routing']} routing heads [{heads_str}]")

# MESH rank-1 statistics
if mesh_results:
    high_rank1 = [m for m in mesh_results if m['sv_ratio'] > 100]
    low_rank1 = [m for m in mesh_results if m['sv_ratio'] <= 100]
    content_heads = [m for m in mesh_results if m['family'] == 'content']
    position_heads = [m for m in mesh_results if m['family'] == 'position']

    print(f"\n  MESH rank-1 structure:")
    print(f"    High rank-1 (S₀/S₁ > 100): {len(high_rank1)}/{len(mesh_results)} routing heads")
    print(f"    Low rank-1 (S₀/S₁ ≤ 100):  {len(low_rank1)}/{len(mesh_results)} routing heads")
    print(f"    Content family (all -1s):    {len(content_heads)}")
    print(f"    Position family (mixed):     {len(position_heads)}")

# Compute savings estimate
print(f"\n  Compute savings estimate (per forward pass):")
full_attn_flops = N_LAYERS * NUM_HEADS * HEAD_DIM  # simplified per-position
fixed_savings = total_fixed / total_heads
print(f"    Fixed heads (zero compute): {total_fixed}/{total_heads} = {fixed_savings:.0%}")
print(f"    Routing heads (need resonator): {total_routing}/{total_heads} = {total_routing/total_heads:.0%}")
print(f"    If resonator works: attention reduced to {total_routing/total_heads:.0%} of original")

print()

# ================================================================
# Save results
# ================================================================
del model
torch.cuda.empty_cache()

results = {
    'n_layers': N_LAYERS,
    'n_heads_per_layer': NUM_HEADS,
    'n_cal_prompts': len(CAL_PROMPTS),
    'layer_summary': layer_summary,
    'mesh_results': mesh_results,
    'totals': {
        'fixed_heads': total_fixed,
        'routing_heads': total_routing,
        'total_heads': total_heads,
        'layers_all_fixed': len(zero_routing),
        'layers_with_routing': len(some_routing),
    },
}

results_path = os.path.join(os.path.dirname(__file__), 'results', 'phase9a_all_layer_attention.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {results_path}")
