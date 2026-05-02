#!/usr/bin/env python3
"""
Phase 9b: End-to-End Geometric Resonator Stack
================================================

Finding 83 showed ALL 302 routing heads have rank-1 MESH across all 28 layers.
This experiment tests: can we replace attention at EVERY layer with geometric
routing and still predict the correct next token?

The resonator replacement for each layer:
  1. Compute V = W_v @ h_normed (still needed for value extraction)
  2. For each head at each query position:
     - FIXED heads: attend to position 0
     - ROUTING heads: attend to argmax(h_normed[j] · d_k) for j in [0..i] (causal)
  3. Gather V at selected positions, project through W_o
  4. MLP runs normally (unchanged)

Key question: does this produce the correct next token?

Requires: Qwen2-7B on GPU, results from phase9a
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import time
import gc

print("=" * 80)
print("  PHASE 9b: END-TO-END GEOMETRIC RESONATOR STACK")
print("  Replace attention at ALL 28 layers with geometric routing")
print("=" * 80)
print()

# Load phase9a classification
results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)

# Build layer classification lookup
layer_classification = {}
for ls in phase9a['layer_summary']:
    layer_classification[ls['layer']] = {
        'fixed': set(ls['fixed_heads']),
        'routing': set(ls['routing_heads']),
    }

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

N_LAYERS = len(model.model.layers)
NUM_HEADS = model.config.num_attention_heads
NUM_KV_HEADS = model.config.num_key_value_heads
HEAD_DIM = model.config.hidden_size // NUM_HEADS
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS
HIDDEN_DIM = model.config.hidden_size

print(f"  {N_LAYERS} layers, {NUM_HEADS} heads, head_dim={HEAD_DIM}")
print()


# ================================================================
# STEP 1: Extract d_k for all routing heads
# ================================================================
print("-" * 80)
print("  STEP 1: Extract d_k routing directions for all routing heads")
print("-" * 80)
print()

# d_k[layer][head] = (HIDDEN_DIM,) tensor on CPU
d_k_vectors = {}
d_k_families = {}  # 'content' or 'position'

for layer_idx in range(N_LAYERS):
    routing_heads = layer_classification[layer_idx]['routing']
    if not routing_heads:
        continue

    d_k_vectors[layer_idx] = {}
    d_k_families[layer_idx] = {}

    layer_obj = model.model.layers[layer_idx]
    attn = layer_obj.self_attn

    identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)

    # Extract W_q and W_k (with bias) for routing heads
    # Collect in chunks
    W_q_heads = {h: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for h in routing_heads}
    W_k_groups = {}
    needed_kv = set(h // HEADS_PER_KV for h in routing_heads)
    for g in needed_kv:
        W_k_groups[g] = torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32)

    chunk_size = 512
    for start in range(0, HIDDEN_DIM, chunk_size):
        end = min(start + chunk_size, HIDDEN_DIM)
        chunk = identity[start:end].unsqueeze(0)

        with torch.no_grad():
            q_out = attn.q_proj(chunk).float()
            k_out = attn.k_proj(chunk).float()

        q_reshaped = q_out[0].reshape(-1, NUM_HEADS, HEAD_DIM)
        k_reshaped = k_out[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)

        for h in routing_heads:
            W_q_heads[h][:, start:end] = q_reshaped[:, h, :].T
        for g in needed_kv:
            W_k_groups[g][:, start:end] = k_reshaped[:, g, :].T

    # Compute MESH → SVD → d_k for each routing head
    for h in routing_heads:
        g = h // HEADS_PER_KV
        MESH = W_q_heads[h] @ W_k_groups[g].T  # (HEAD_DIM, HEAD_DIM)
        _, _, Vt = torch.linalg.svd(MESH)
        d_k = (W_k_groups[g].T @ Vt[0]).cpu()  # (HIDDEN_DIM,)
        d_k_vectors[layer_idx][h] = d_k

        all_neg = bool((d_k < 0).sum() > 0.95 * HIDDEN_DIM)
        d_k_families[layer_idx][h] = 'content' if all_neg else 'position'

    # Clean up layer-specific tensors
    del W_q_heads, W_k_groups
    torch.cuda.empty_cache()

    n_content = sum(1 for h in routing_heads if d_k_families[layer_idx][h] == 'content')
    n_position = len(routing_heads) - n_content
    print(f"  Layer {layer_idx:2d}: {len(routing_heads):2d} routing heads "
          f"({n_content} content, {n_position} position)")

print()


# ================================================================
# STEP 2: Define geometric attention forward pass
# ================================================================
print("-" * 80)
print("  STEP 2: Geometric attention replacement")
print("-" * 80)
print()


def geometric_attention(layer_idx, h_normed, attn_module):
    """
    Replace standard attention with geometric routing.

    h_normed: (1, seq_len, hidden_dim) — already layernormed
    Returns: attn_output (1, seq_len, hidden_dim)
    """
    batch, seq_len, hidden_dim = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']

    # Compute V for all positions (we still need the value vectors)
    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)  # (1, seq_len, kv_heads*head_dim)

    # Reshape V: (1, seq_len, kv_heads, head_dim)
    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    # Expand for GQA: (1, seq_len, num_heads, head_dim)
    V_expanded = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)

    # Compute routing scores for routing heads
    # score_h[pos] = h_normed[0, pos, :] · d_k_h
    routing_selected = {}  # head -> (seq_len,) tensor of selected positions
    for h in routing:
        d_k = d_k_vectors[layer_idx][h].to(h_normed.device, dtype=torch.float32)
        # Compute score for each key position
        scores = (h_normed[0].float() @ d_k)  # (seq_len,)

        # For content heads (all-negative d_k), score = -sum(h)
        # argmax(score) = position with most negative sum = argmin(sum)
        # This is correct as-is because d_k encodes the sign

        # Causal: for each query position i, select from 0..i
        selected = torch.zeros(seq_len, dtype=torch.long, device=h_normed.device)
        for i in range(seq_len):
            selected[i] = scores[:i+1].argmax()

        routing_selected[h] = selected

    # Build attention output: for each head and query position, gather V
    # attn_per_head: (1, seq_len, num_heads, head_dim)
    attn_per_head = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                                device=h_normed.device, dtype=h_normed.dtype)

    for h in range(NUM_HEADS):
        if h in fixed:
            # Always attend to position 0
            attn_per_head[0, :, h, :] = V_expanded[0, 0, h, :]  # broadcast pos 0
        else:
            # Routing: gather V at selected positions
            sel = routing_selected[h]  # (seq_len,)
            for i in range(seq_len):
                attn_per_head[0, i, h, :] = V_expanded[0, sel[i], h, :]

    # Reshape to (1, seq_len, num_heads * head_dim) and project through W_o
    combined = attn_per_head.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)

    with torch.no_grad():
        attn_output = attn_module.o_proj(combined)

    return attn_output


def run_geometric_forward(input_ids):
    """
    Run the full model with geometric attention replacement at every layer.
    Returns logits.
    """
    with torch.no_grad():
        # Embedding
        hidden = model.model.embed_tokens(input_ids)

        # Each layer
        for layer_idx in range(N_LAYERS):
            layer = model.model.layers[layer_idx]

            # Attention block
            residual = hidden
            h_normed = layer.input_layernorm(hidden)
            attn_output = geometric_attention(layer_idx, h_normed, layer.self_attn)
            hidden = residual + attn_output

            # MLP block (unchanged)
            residual = hidden
            h_normed_mlp = layer.post_attention_layernorm(hidden)
            mlp_output = layer.mlp(h_normed_mlp)
            hidden = residual + mlp_output

        # Final norm + LM head
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

    return logits


# ================================================================
# STEP 3: Test prompts — compare baseline vs geometric
# ================================================================
print("-" * 80)
print("  STEP 3: End-to-end accuracy test")
print("-" * 80)
print()

TEST_PROMPTS = [
    "The capital of France is",
    "The largest ocean is the",
    "The color of grass is",
    "Barack Obama was the",
    "To be or not to",
    "Roses are red, violets are",
    "The speed of light is approximately",
    "Albert Einstein developed the theory of",
    "Water freezes at zero degrees",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "Shakespeare wrote many",
    "The square root of 144 is",
    "In mathematics, pi is approximately equal to",
    "The color of the sky is usually",
]

print(f"  {'Prompt':>50s}  {'Baseline':>10s}  {'Geometric':>10s}  {'Match':>5s}")
print("  " + "-" * 80)

n_match = 0
n_total = 0
cos_sims = []

for prompt in TEST_PROMPTS:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Baseline
    with torch.no_grad():
        baseline_out = model(input_ids, return_dict=True)
        baseline_logits = baseline_out.logits[0, -1, :].float()

    baseline_token_id = baseline_logits.argmax().item()
    baseline_token = tokenizer.decode([baseline_token_id])

    # Geometric
    geo_logits = run_geometric_forward(input_ids)[0, -1, :].float()
    geo_token_id = geo_logits.argmax().item()
    geo_token = tokenizer.decode([geo_token_id])

    # Compare
    match = baseline_token_id == geo_token_id
    if match:
        n_match += 1
    n_total += 1

    # Logit cosine similarity
    cos = F.cosine_similarity(baseline_logits.unsqueeze(0),
                              geo_logits.unsqueeze(0)).item()
    cos_sims.append(cos)

    mark = "✓" if match else "✗"
    print(f"  {prompt:>50s}  {baseline_token:>10s}  {geo_token:>10s}  {mark:>5s}  cos={cos:.4f}")

print()
print(f"  Top-1 accuracy: {n_match}/{n_total} ({n_match/n_total:.0%})")
print(f"  Mean logit cosine similarity: {np.mean(cos_sims):.4f}")
print()


# ================================================================
# STEP 4: Per-layer ablation — which layers matter?
# ================================================================
print("-" * 80)
print("  STEP 4: Per-layer ablation (replace ONE layer at a time)")
print("  Which layers tolerate geometric attention?")
print("-" * 80)
print()


def run_single_layer_replacement(input_ids, target_layer):
    """Replace attention at only target_layer; all others run normally.
    Uses a hook to intercept and replace the attention output."""

    replacement_output = [None]

    def attn_hook(module, args, kwargs, output):
        """Replace attention output with geometric version."""
        # Get the layernormed input (first positional arg or from hidden_states kwarg)
        h_normed = args[0] if args else kwargs.get('hidden_states')
        if h_normed is None:
            return output
        geo_out = geometric_attention(target_layer, h_normed, module)
        # Return same tuple structure but with replaced attention output
        if isinstance(output, tuple):
            return (geo_out,) + output[1:]
        return geo_out

    # Register hook on target layer's self_attn
    hook = model.model.layers[target_layer].self_attn.register_forward_hook(
        attn_hook, with_kwargs=True
    )

    try:
        with torch.no_grad():
            out = model(input_ids, return_dict=True)
        logits = out.logits
    finally:
        hook.remove()

    return logits


# Test each layer individually on a subset of prompts
ablation_prompts = TEST_PROMPTS[:6]  # First 6 for speed

# Get baseline tokens
baseline_tokens = []
for prompt in ablation_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(input_ids, return_dict=True)
    baseline_tokens.append(out.logits[0, -1, :].float().argmax().item())

print(f"  {'Layer':>5s}  {'Score':>7s}  {'Routing':>7s}  {'Fixed':>5s}")
print("  " + "-" * 30)

layer_ablation = []
for target_layer in range(N_LAYERS):
    n_pass = 0
    for pi, prompt in enumerate(ablation_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        logits = run_single_layer_replacement(input_ids, target_layer)
        token_id = logits[0, -1, :].float().argmax().item()
        if token_id == baseline_tokens[pi]:
            n_pass += 1

    n_routing = len(layer_classification[target_layer]['routing'])
    n_fixed = len(layer_classification[target_layer]['fixed'])
    print(f"  {target_layer:5d}  {n_pass}/{len(ablation_prompts)}     {n_routing:7d}  {n_fixed:5d}")
    layer_ablation.append({
        'layer': target_layer,
        'score': n_pass,
        'total': len(ablation_prompts),
        'n_routing': n_routing,
        'n_fixed': n_fixed,
    })

print()

# ================================================================
# STEP 5: Summary
# ================================================================
print("=" * 80)
print("  SUMMARY: GEOMETRIC RESONATOR STACK")
print("=" * 80)
print()
print(f"  Full stack (all 28 layers replaced):")
print(f"    Top-1 accuracy: {n_match}/{n_total} ({n_match/n_total:.0%})")
print(f"    Mean logit cosine: {np.mean(cos_sims):.4f}")
print()

perfect_layers = [la for la in layer_ablation if la['score'] == la['total']]
failing_layers = [la for la in layer_ablation if la['score'] < la['total']]
print(f"  Per-layer ablation ({len(ablation_prompts)} prompts each):")
print(f"    Perfect (6/6): {len(perfect_layers)} layers")
print(f"    Imperfect:     {len(failing_layers)} layers")
if failing_layers:
    for la in failing_layers:
        print(f"      Layer {la['layer']:2d}: {la['score']}/{la['total']} "
              f"({la['n_routing']} routing heads)")

# Save results
results = {
    'n_test_prompts': n_total,
    'full_stack_accuracy': n_match / n_total,
    'full_stack_matches': n_match,
    'mean_logit_cos': float(np.mean(cos_sims)),
    'per_prompt': [
        {'prompt': p, 'match': cos_sims[i] > 0, 'cos': cos_sims[i]}
        for i, p in enumerate(TEST_PROMPTS)
    ],
    'layer_ablation': layer_ablation,
}

save_path = os.path.join(results_dir, 'phase9b_resonator_stack.json')
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {save_path}")
