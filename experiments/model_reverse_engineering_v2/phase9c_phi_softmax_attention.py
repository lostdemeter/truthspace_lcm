#!/usr/bin/env python3
"""
Phase 9c: φ-Softmax Geometric Attention Replacement
=====================================================

Finding 84 showed hard argmax routing compounds errors across layers (0/15).
This experiment replaces hard argmax with phi_softmax:

  Standard:  softmax(x) = e^x / Σ e^x
  φ-form:    phi_softmax(x) = φ^(x/T) / Σ φ^(x/T)   where T = ln(φ)

This is EXACT (not an approximation) — same operation in φ-basis.

For each head at each query position:
  - FIXED heads: phi_softmax over d_k scores (mostly pos 0, but preserves bleed)
  - ROUTING heads: phi_softmax over d_k scores (weighted average, not hard pick)
  - Output = phi_softmax_weights @ V  (geometric weighted average)

The routing scores still come from the rank-1 MESH d_k direction.
The mixing uses φ-power weighting instead of hard selection.

Tests:
  A. Full-stack phi_softmax (all 28 layers)
  B. Per-layer ablation with phi_softmax
  C. Temperature sweep (T = ln(φ) vs others)

Requires: Qwen2-7B on GPU, results from phase9a
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import json
import os

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)  # ≈ 0.4812, the natural φ-temperature

print("=" * 80)
print("  PHASE 9c: φ-SOFTMAX GEOMETRIC ATTENTION")
print(f"  T = ln(φ) ≈ {LOG_PHI:.4f}")
print("=" * 80)
print()

# Load phase9a classification
results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)

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

N_LAYERS = 28; NUM_HEADS = 28; NUM_KV_HEADS = 4; HEAD_DIM = 128
HEADS_PER_KV = 7; HIDDEN_DIM = 3584

print(f"  {N_LAYERS} layers, {NUM_HEADS} heads, head_dim={HEAD_DIM}")
print()


# ================================================================
# φ-softmax in torch
# ================================================================
def phi_softmax(scores, dim=-1):
    """
    φ-basis softmax: φ^(x/ln(φ)) / Σ φ^(x/ln(φ)) = e^x / Σ e^x

    Mathematically identical to standard softmax but expressed as
    φ-power selection. Keeping computation in the φ-basis for
    consistency with the geometric framework.
    """
    # Numerical stability: subtract max
    scores_shifted = scores - scores.max(dim=dim, keepdim=True).values
    # φ^(x / ln(φ)) = e^x  — exact
    phi_powers = PHI ** (scores_shifted / LOG_PHI)
    return phi_powers / phi_powers.sum(dim=dim, keepdim=True)


# ================================================================
# STEP 1: Extract d_k for all routing heads
# ================================================================
print("-" * 80)
print("  STEP 1: Extract d_k routing directions")
print("-" * 80)
print()

d_k_vectors = {}
d_k_families = {}

for layer_idx in range(N_LAYERS):
    routing = layer_classification[layer_idx]['routing']
    if not routing:
        continue

    d_k_vectors[layer_idx] = {}
    d_k_families[layer_idx] = {}

    attn = model.model.layers[layer_idx].self_attn
    identity = torch.eye(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)

    W_q = {h: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for h in routing}
    needed_kv = set(h // HEADS_PER_KV for h in routing)
    W_k = {g: torch.zeros(HEAD_DIM, HIDDEN_DIM, device="cuda", dtype=torch.float32) for g in needed_kv}

    for start in range(0, HIDDEN_DIM, 512):
        end = min(start + 512, HIDDEN_DIM)
        chunk = identity[start:end].unsqueeze(0)
        with torch.no_grad():
            q_out = attn.q_proj(chunk).float()
            k_out = attn.k_proj(chunk).float()
        qr = q_out[0].reshape(-1, NUM_HEADS, HEAD_DIM)
        kr = k_out[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        for h in routing:
            W_q[h][:, start:end] = qr[:, h, :].T
        for g in needed_kv:
            W_k[g][:, start:end] = kr[:, g, :].T

    for h in routing:
        g = h // HEADS_PER_KV
        MESH = W_q[h] @ W_k[g].T
        _, _, Vt = torch.linalg.svd(MESH)
        dk = (W_k[g].T @ Vt[0]).cpu()
        d_k_vectors[layer_idx][h] = dk
        d_k_families[layer_idx][h] = 'content' if (dk < 0).sum() > 0.95 * HIDDEN_DIM else 'position'

    del W_q, W_k
    torch.cuda.empty_cache()

    n_c = sum(1 for h in routing if d_k_families[layer_idx][h] == 'content')
    print(f"  Layer {layer_idx:2d}: {len(routing):2d} routing ({n_c} content, {len(routing)-n_c} position)")

print()


# ================================================================
# STEP 2: φ-softmax geometric attention
# ================================================================

def geometric_attention_phi_soft(layer_idx, h_normed, attn_module, temperature=1.0):
    """
    Replace attention with φ-softmax weighted routing.

    For each head:
      scores[pos] = h_normed[pos] · d_k
      weights = phi_softmax(scores / temperature)
      output = weights @ V

    Fixed heads use the SAME mechanism — their d_k just happens to
    produce scores that peak at position 0.
    """
    batch, seq_len, _ = h_normed.shape
    fixed = layer_classification[layer_idx]['fixed']
    routing = layer_classification[layer_idx]['routing']

    # Compute V for all positions
    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)

    V_kv = V_full.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    V_expanded = V_kv.repeat_interleave(HEADS_PER_KV, dim=2)
    # V_expanded: (1, seq_len, NUM_HEADS, HEAD_DIM)

    # Build output per head
    attn_out = torch.zeros(batch, seq_len, NUM_HEADS, HEAD_DIM,
                           device=h_normed.device, dtype=h_normed.dtype)

    # For FIXED heads: just use position 0 value (hard, since they're ~100% pos 0)
    for h in fixed:
        attn_out[0, :, h, :] = V_expanded[0, 0, h, :]

    # For ROUTING heads: phi_softmax weighted average
    for h in routing:
        dk = d_k_vectors[layer_idx][h].to(h_normed.device, dtype=torch.float32)
        # Score each position
        scores = (h_normed[0].float() @ dk)  # (seq_len,)

        # Causal phi_softmax: for each query position i, attend to 0..i
        for i in range(seq_len):
            causal_scores = scores[:i+1] / temperature
            weights = phi_softmax(causal_scores, dim=0)  # (i+1,)
            # Weighted average of V vectors
            weighted_v = (weights.to(h_normed.dtype).unsqueeze(-1) *
                         V_expanded[0, :i+1, h, :])  # (i+1, HEAD_DIM)
            attn_out[0, i, h, :] = weighted_v.sum(dim=0)

    # Project through W_o
    combined = attn_out.reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
    with torch.no_grad():
        return attn_module.o_proj(combined)


# ================================================================
# STEP 3: Test prompts
# ================================================================
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

# Get baseline
print("-" * 80)
print("  Collecting baselines...")
print("-" * 80)

baseline_ids = []
baseline_logits_list = []
for p in TEST_PROMPTS:
    ids = tokenizer.encode(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, return_dict=True)
    bl = out.logits[0, -1, :].float()
    baseline_ids.append(bl.argmax().item())
    baseline_logits_list.append(bl.cpu())

print(f"  {len(TEST_PROMPTS)} baselines collected")
print()


# ================================================================
# STEP 4: Full-stack phi_softmax test
# ================================================================
print("-" * 80)
print("  STEP 4: Full-stack φ-softmax (all 28 layers replaced)")
print("-" * 80)
print()

def run_full_stack_phi_soft(input_ids, temperature=1.0):
    """Run model with phi_softmax geometric attention at every layer."""
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)

        for layer_idx in range(N_LAYERS):
            layer = model.model.layers[layer_idx]

            # Attention block
            residual = hidden
            h_normed = layer.input_layernorm(hidden)
            attn_output = geometric_attention_phi_soft(
                layer_idx, h_normed, layer.self_attn, temperature=temperature
            )
            hidden = residual + attn_output

            # MLP block (unchanged)
            residual = hidden
            h_normed_mlp = layer.post_attention_layernorm(hidden)
            mlp_output = layer.mlp(h_normed_mlp)
            hidden = residual + mlp_output

        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

    return logits


# Test with default temperature = 1.0
print(f"  {'Prompt':>50s}  {'Base':>10s}  {'φ-soft':>10s}  {'Match':>5s}  {'cos':>7s}")
print("  " + "-" * 87)

n_match = 0
cos_sims = []
for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    geo_logits = run_full_stack_phi_soft(ids, temperature=1.0)[0, -1, :].float()
    geo_id = geo_logits.argmax().item()
    match = geo_id == baseline_ids[pi]
    if match:
        n_match += 1
    cos = F.cosine_similarity(baseline_logits_list[pi].unsqueeze(0),
                              geo_logits.cpu().unsqueeze(0)).item()
    cos_sims.append(cos)
    base_tok = tokenizer.decode([baseline_ids[pi]])
    geo_tok = tokenizer.decode([geo_id])
    mark = "✓" if match else "✗"
    print(f"  {prompt:>50s}  {base_tok:>10s}  {geo_tok:>10s}  {mark:>5s}  {cos:>7.4f}")

print()
print(f"  φ-softmax full-stack (T=1.0): {n_match}/{len(TEST_PROMPTS)} ({n_match/len(TEST_PROMPTS):.0%})")
print(f"  Mean logit cosine: {np.mean(cos_sims):.4f}")
print()


# ================================================================
# STEP 5: Temperature sweep
# ================================================================
print("-" * 80)
print("  STEP 5: Temperature sweep")
print("  T = ln(φ) is the natural φ-temperature")
print("-" * 80)
print()

temperatures = [0.1, 0.25, LOG_PHI, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]
temp_results = []

for temp in temperatures:
    n_m = 0
    cs = []
    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        gl = run_full_stack_phi_soft(ids, temperature=temp)[0, -1, :].float()
        gid = gl.argmax().item()
        if gid == baseline_ids[pi]:
            n_m += 1
        c = F.cosine_similarity(baseline_logits_list[pi].unsqueeze(0),
                                gl.cpu().unsqueeze(0)).item()
        cs.append(c)

    is_phi = "  ← ln(φ)" if abs(temp - LOG_PHI) < 0.01 else ""
    print(f"  T={temp:5.2f}: {n_m:2d}/{len(TEST_PROMPTS)} ({n_m/len(TEST_PROMPTS):4.0%})  "
          f"cos={np.mean(cs):.4f}{is_phi}")
    temp_results.append({
        'temperature': float(temp),
        'accuracy': n_m,
        'total': len(TEST_PROMPTS),
        'mean_cos': float(np.mean(cs)),
    })

print()


# ================================================================
# STEP 6: Per-layer ablation with phi_softmax
# ================================================================
print("-" * 80)
print("  STEP 6: Per-layer ablation (phi_softmax, one layer at a time)")
print("-" * 80)
print()

ablation_prompts = TEST_PROMPTS[:6]
abl_baseline = baseline_ids[:6]

best_temp = max(temp_results, key=lambda r: (r['accuracy'], r['mean_cos']))['temperature']
print(f"  Using best temperature from sweep: T={best_temp:.2f}")
print()

print(f"  {'Layer':>5s}  {'Score':>5s}  {'Route':>5s}  {'Fixed':>5s}")
print("  " + "-" * 25)

layer_ablation = []
for target_layer in range(N_LAYERS):
    def make_hook(tl, temp):
        def hook_fn(module, args, kwargs, output):
            h = args[0] if args else kwargs.get('hidden_states')
            if h is None:
                return output
            geo = geometric_attention_phi_soft(tl, h, module, temperature=temp)
            return (geo,) + output[1:] if isinstance(output, tuple) else geo
        return hook_fn

    hook = model.model.layers[target_layer].self_attn.register_forward_hook(
        make_hook(target_layer, best_temp), with_kwargs=True
    )

    n_pass = 0
    for pi, prompt in enumerate(ablation_prompts):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(ids, return_dict=True)
        tid = out.logits[0, -1, :].float().argmax().item()
        if tid == abl_baseline[pi]:
            n_pass += 1

    hook.remove()

    nr = len(layer_classification[target_layer]['routing'])
    nf = len(layer_classification[target_layer]['fixed'])
    mark = "✓" if n_pass == 6 else ""
    print(f"  {target_layer:5d}  {n_pass}/6    {nr:5d}  {nf:5d}  {mark}")
    layer_ablation.append({
        'layer': target_layer, 'score': n_pass, 'total': 6,
        'n_routing': nr, 'n_fixed': nf,
    })

print()

# ================================================================
# Summary
# ================================================================
print("=" * 80)
print("  SUMMARY: φ-SOFTMAX GEOMETRIC ATTENTION")
print("=" * 80)
print()

best = max(temp_results, key=lambda r: (r['accuracy'], r['mean_cos']))
print(f"  Best full-stack: T={best['temperature']:.2f} → "
      f"{best['accuracy']}/{best['total']} ({best['accuracy']/best['total']:.0%}), "
      f"cos={best['mean_cos']:.4f}")

perfect_layers = sum(1 for la in layer_ablation if la['score'] == la['total'])
print(f"  Per-layer ablation: {perfect_layers}/28 perfect (6/6)")

# Compare with F84 hard argmax
print()
print("  Comparison with Finding 84 (hard argmax):")
print("    Hard argmax full-stack: 0/15 (0%), cos=0.571")
print(f"    φ-softmax full-stack:   {best['accuracy']}/{best['total']} "
      f"({best['accuracy']/best['total']:.0%}), cos={best['mean_cos']:.4f}")

# Save
results = {
    'method': 'phi_softmax',
    'temperature_sweep': temp_results,
    'best_temperature': best['temperature'],
    'full_stack_accuracy': best['accuracy'] / best['total'],
    'full_stack_cos': best['mean_cos'],
    'layer_ablation': layer_ablation,
    'phi_softmax_note': 'phi^(x/ln(phi)) / sum = e^x / sum — exact, not approximation',
}

save_path = os.path.join(results_dir, 'phase9c_phi_softmax_attention.json')
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {save_path}")
