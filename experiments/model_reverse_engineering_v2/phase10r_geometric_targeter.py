#!/usr/bin/env python3
"""
Phase 10r: Geometric Targeter Prototype — Doc 263
Build a φ-Filter replacement for L26-27 and verify it matches the real model.

Tests:
  A: Real attention + sparse FFN (isolate FFN sparsification)
  B: Skip attention + full FFN (isolate attention irrelevance)
  C: Skip attention + sparse FFN (full geometric targeter)
  D: Skip attention + bias-only gate + sparse FFN (maximum simplification)
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10r: GEOMETRIC TARGETER PROTOTYPE (Doc 263)")
print("="*80)
results_dir = os.path.join(os.path.dirname(__file__), 'results')

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
NL = 28; HDIM = 3584; D_INT = 18944
TARGETER_LAYERS = [26, 27]

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
    "In the year 2024, the president of the United States was",
    "The square root of 144 is",
    "Photosynthesis converts sunlight into",
    "The longest river in Africa is the",
    "Shakespeare wrote the play Romeo and",
]

# ================================================================
# STEP 1: Collect gate activations to classify channels
# ================================================================
print("\nStep 1: Collecting gate activations from real model runs...")

# Qwen2 gate_proj has NO bias, so gate_proj(0)=0. We must classify channels
# from ACTUAL hidden states. Run calibration prompts to find which channels
# are consistently EXPAND/CONTRACT.
gate_activations = {li: [] for li in TARGETER_LAYERS}

for prompt in TEST_PROMPTS[:10]:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    hooks = []
    gate_acts = {}
    for li in TARGETER_LAYERS:
        def make_hook(idx):
            def hk(mod, inp, output):
                gate_acts[idx] = output[0, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].mlp.gate_proj.register_forward_hook(make_hook(li)))
    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()
    for li in TARGETER_LAYERS:
        gate_activations[li].append(gate_acts[li])

print("  Classifying channels from real gate activations...")
phi_filters = {}
for li in TARGETER_LAYERS:
    # Stack gate activations: [n_prompts, D_INT]
    G = torch.stack(gate_activations[li])
    mean_gate = G.mean(dim=0)

    # Classify by MEAN activation (the "effective bias")
    expand_mask = mean_gate > LOG_PHI
    contract_mask = mean_gate < -LOG_PHI
    preserve_mask = ~(expand_mask | contract_mask)

    n_expand = expand_mask.sum().item()
    n_preserve = preserve_mask.sum().item()
    n_contract = contract_mask.sum().item()

    # Check stability: fraction of prompts agreeing with mean classification
    per_prompt_expand = (G > LOG_PHI).float().mean(dim=0)
    per_prompt_contract = (G < -LOG_PHI).float().mean(dim=0)
    expand_stability = per_prompt_expand[expand_mask].mean().item() if n_expand > 0 else 0
    contract_stability = per_prompt_contract[contract_mask].mean().item() if n_contract > 0 else 0

    # Active mask: EXPAND + PRESERVE
    active_mask = mean_gate > -LOG_PHI
    n_active = active_mask.sum().item()

    print(f"  L{li}: EXPAND={n_expand} ({n_expand/D_INT*100:.1f}%, stability={expand_stability:.3f}), "
          f"PRESERVE={n_preserve} ({n_preserve/D_INT*100:.1f}%), "
          f"CONTRACT={n_contract} ({n_contract/D_INT*100:.1f}%, stability={contract_stability:.3f})")
    print(f"       Active (non-CONTRACT): {n_active} ({n_active/D_INT*100:.1f}%)")

    expand_idx = expand_mask.nonzero().squeeze(-1)
    active_idx = active_mask.nonzero().squeeze(-1)

    with torch.no_grad():
        gate_w_full = model.model.layers[li].mlp.gate_proj.weight.float().cpu()
        up_w_full = model.model.layers[li].mlp.up_proj.weight.float().cpu()
        down_w_full = model.model.layers[li].mlp.down_proj.weight.float().cpu()

    phi_filters[li] = {
        'mean_gate': mean_gate,
        'expand_mask': expand_mask,
        'expand_idx': expand_idx,
        'active_mask': active_mask,
        'active_idx': active_idx,
        'n_expand': n_expand,
        'n_active': n_active,
        'expand_stability': expand_stability,
        'contract_stability': contract_stability,
        # Sparse weights for EXPAND-only
        'gate_w_expand': gate_w_full[expand_idx].to(torch.bfloat16).cuda() if n_expand > 0 else None,
        'up_w_expand': up_w_full[expand_idx].to(torch.bfloat16).cuda() if n_expand > 0 else None,
        'down_w_expand': down_w_full[:, expand_idx].to(torch.bfloat16).cuda() if n_expand > 0 else None,
        # Sparse weights for EXPAND+PRESERVE
        'gate_w_active': gate_w_full[active_idx].to(torch.bfloat16).cuda() if n_active > 0 else None,
        'up_w_active': up_w_full[active_idx].to(torch.bfloat16).cuda() if n_active > 0 else None,
        'down_w_active': down_w_full[:, active_idx].to(torch.bfloat16).cuda() if n_active > 0 else None,
    }

    del gate_w_full, up_w_full, down_w_full
    torch.cuda.empty_cache()

# ================================================================
# STEP 2: Define Geometric Targeter variants
# ================================================================

def silu(x):
    return x * torch.sigmoid(x)

def apply_phi_filter(h_normed, li, mode='expand'):
    """Apply sparse FFN for layer li using only active channels.
    h_normed: [1, seq_len, HDIM] — post-LayerNorm hidden state.
    Returns: [1, seq_len, HDIM] — FFN output.
    """
    pf = phi_filters[li]
    # h_normed is [1, S, D], work with [S, D]
    x = h_normed[0].to(torch.bfloat16)  # [S, D]

    if mode == 'expand' and pf['gate_w_expand'] is not None:
        # Only EXPAND channels (~8%)
        gate_vals = (x @ pf['gate_w_expand'].T)  # [S, n_expand]
        up_vals = (x @ pf['up_w_expand'].T)      # [S, n_expand]
        active = silu(gate_vals) * up_vals         # [S, n_expand]
        ffn_out = active @ pf['down_w_expand'].T   # [S, D]
    elif mode == 'active' and pf['gate_w_active'] is not None:
        # EXPAND + PRESERVE channels
        gate_vals = (x @ pf['gate_w_active'].T)  # [S, n_active]
        up_vals = (x @ pf['up_w_active'].T)      # [S, n_active]
        active = silu(gate_vals) * up_vals         # [S, n_active]
        ffn_out = active @ pf['down_w_active'].T   # [S, D]
    else:
        # Fallback: zero output
        ffn_out = torch.zeros_like(x)

    return ffn_out.unsqueeze(0).to(h_normed.dtype)  # [1, S, D]

def geometric_targeter(h_after_l25, variant='C', ffn_mode='expand'):
    """Apply geometric targeter (L26-27) to hidden state from L25 output.

    Variants:
      B: Skip attention + full FFN (test attn irrelevance)
      C: Skip attention + sparse FFN (the φ-Filter)
      E: Full layer (real attn + real FFN, via model forward from L25)
    """
    h = h_after_l25.clone()

    for li in TARGETER_LAYERS:
        layer = model.model.layers[li]
        residual = h

        # Input LayerNorm (damper)
        h_normed = layer.input_layernorm(h)

        # Skip attention (lever is irrelevant per Finding 98)
        h = residual

        residual = h

        # Post-attention LayerNorm (damper)
        h_normed2 = layer.post_attention_layernorm(h)

        if variant == 'B':
            # Full FFN (real)
            with torch.no_grad():
                ffn_out = layer.mlp(h_normed2)
        elif variant == 'C':
            # Sparse FFN (the φ-Filter)
            ffn_out = apply_phi_filter(h_normed2, li, mode=ffn_mode)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        h = residual + ffn_out

    return h

# ================================================================
# STEP 3: Run experiment
# ================================================================
print("\nStep 3: Running experiments...")

# For each prompt: get baseline, then test each variant
results = {v: {'top1_match': [], 'cos_sim': [], 'logit_rmse': [], 'angle': []}
           for v in ['baseline', 'B', 'C_expand', 'C_active']}

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]

    # Capture L25 output and baseline logits
    l25_output = {}
    def capture_l25(mod, args, output):
        h_out = output[0] if isinstance(output, tuple) else output
        l25_output['h'] = h_out.detach().clone()
    hook = model.model.layers[25].register_forward_hook(capture_l25)

    with torch.no_grad():
        baseline_out = model(ids, return_dict=True)
    baseline_logits = baseline_out.logits[0, -1].float().cpu()
    baseline_top1 = baseline_logits.argmax().item()
    hook.remove()

    h_l25 = l25_output['h']  # [1, seq_len, HDIM]
    results['baseline']['top1_match'].append(True)
    results['baseline']['cos_sim'].append(1.0)
    results['baseline']['logit_rmse'].append(0.0)
    results['baseline']['angle'].append(0.0)

    # We need to handle position_ids for attention in variant A
    # For simplicity, use the last token's hidden state for variants that skip attention
    # But we need the full sequence for proper evaluation

    # Apply each variant using the captured L25 hidden state
    for vname, variant, ffn_mode in [
        ('B', 'B', None),
        ('C_expand', 'C', 'expand'),
        ('C_active', 'C', 'active'),
    ]:
        try:
            h_target = geometric_targeter(h_l25, variant=variant, ffn_mode=ffn_mode)

            # Apply final LayerNorm + lm_head
            with torch.no_grad():
                h_final = model.model.norm(h_target)
                logits = model.lm_head(h_final)[0, -1].float().cpu()

            top1 = logits.argmax().item()
            cos = F.cosine_similarity(logits.unsqueeze(0), baseline_logits.unsqueeze(0)).item()
            rmse = (logits - baseline_logits).pow(2).mean().sqrt().item()
            angle = math.degrees(math.acos(max(-1, min(1, cos))))

            results[vname]['top1_match'].append(top1 == baseline_top1)
            results[vname]['cos_sim'].append(cos)
            results[vname]['logit_rmse'].append(rmse)
            results[vname]['angle'].append(angle)
        except Exception as e:
            print(f"  ERROR in {vname} for prompt {pi}: {e}")
            results[vname]['top1_match'].append(False)
            results[vname]['cos_sim'].append(0.0)
            results[vname]['logit_rmse'].append(999.0)
            results[vname]['angle'].append(90.0)

    if pi % 3 == 0:
        print(f"  Prompt {pi}/{len(TEST_PROMPTS)}: baseline='{tokenizer.decode(baseline_top1)}'")

# ================================================================
# STEP 4: Results
# ================================================================
print("\n" + "="*70)
print("  GEOMETRIC TARGETER RESULTS")
print("="*70)

print(f"\n  {'Variant':>12s} | {'Description':>35s} | {'Top1%':>5s} | {'cos':>6s} | {'RMSE':>7s} | {'Angle':>6s}")
print("  " + "-"*90)

variant_desc = {
    'baseline':  'Real model (reference)',
    'B':         'Skip attn + full FFN',
    'C_expand':  'Skip attn + sparse FFN (EXPAND)',
    'C_active':  'Skip attn + sparse FFN (EXPAND+PRES)',
}

for vn in ['baseline', 'B', 'C_expand', 'C_active']:
    r = results[vn]
    n = len(r['top1_match'])
    top1 = sum(r['top1_match']) / n * 100 if n > 0 else 0
    cos = np.mean(r['cos_sim']) if r['cos_sim'] else 0
    rmse = np.mean(r['logit_rmse']) if r['logit_rmse'] else 0
    angle = np.mean(r['angle']) if r['angle'] else 0
    desc = variant_desc.get(vn, vn)
    print(f"  {vn:>12s} | {desc:>35s} | {top1:5.1f} | {cos:6.4f} | {rmse:7.2f} | {angle:5.2f}°")

# Compute savings
for li in TARGETER_LAYERS:
    pf = phi_filters[li]
    full_ops = 3 * HDIM * D_INT  # 3 matmuls (gate, up, down)
    expand_ops = 3 * HDIM * pf['n_expand']
    active_ops = 3 * HDIM * pf['n_active']
    print(f"\n  L{li} compute: full={full_ops/1e6:.1f}M, expand={expand_ops/1e6:.1f}M "
          f"({expand_ops/full_ops*100:.1f}%), active={active_ops/1e6:.1f}M "
          f"({active_ops/full_ops*100:.1f}%)")

# Per-prompt details for best variant
best_v = 'C_expand'  # Our target
print(f"\n  Per-prompt details for {best_v}:")
print(f"  {'Prompt':>50s} | {'Match':>5s} | {'cos':>6s} | {'BL tok':>10s} | {'GeoTarg':>10s}")
print("  " + "-"*95)
for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    bl_tok = tokenizer.decode(tokenizer.encode(prompt, return_tensors="pt").to("cuda")[0][-1:])
    # Get baseline and variant predictions for display
    r = results[best_v]
    match = "✓" if r['top1_match'][pi] else "✗"
    cos = r['cos_sim'][pi]

    # Re-run baseline to get token names
    with torch.no_grad():
        bl_out = model(ids, return_dict=True)
    bl_top = tokenizer.decode(bl_out.logits[0, -1].argmax().item())
    # We need the variant's top token too - approximate from stored data
    print(f"  {prompt:>50s} | {match:>5s} | {cos:6.4f} | {bl_top:>10s} |")

# Save results
save_data = {
    'phi_filter_params': {
        str(li): {
            'n_expand': phi_filters[li]['n_expand'],
            'n_active': phi_filters[li]['n_active'],
            'expand_pct': phi_filters[li]['n_expand'] / D_INT,
            'active_pct': phi_filters[li]['n_active'] / D_INT,
        }
        for li in TARGETER_LAYERS
    },
    'results': {
        vn: {
            'top1_accuracy': sum(r['top1_match']) / max(len(r['top1_match']), 1),
            'mean_cos_sim': float(np.mean(r['cos_sim'])),
            'mean_logit_rmse': float(np.mean(r['logit_rmse'])),
            'mean_angle': float(np.mean(r['angle'])),
            'per_prompt_match': r['top1_match'],
        }
        for vn, r in results.items()
    },
    'variant_descriptions': variant_desc,
}

out_path = os.path.join(results_dir, 'phase10r_geometric_targeter.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\n  Saved to {out_path}")
print("="*70)
print("  PHASE 10r COMPLETE")
print("="*70)
