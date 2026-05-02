#!/usr/bin/env python3
"""
Phase 10s: Compressor Decomposition (L0-3)
Weight Decomposition Protocol Steps 1-5 applied to the Compressor sub-machine.

Step 1: Boundary Lock — capture I/O hidden states
Step 2: Gate Census — 4-state classification per layer
Step 3: Simple Machines — lever/damper/wedge/spring per layer
Step 4: Independence Test — skip attention, skip FFN variants
Step 5: Transfer Function — classify oscillatory/convergent/step
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10s: COMPRESSOR DECOMPOSITION (L0-3)")
print("  Weight Decomposition Protocol — Steps 1-5")
print("="*80)
results_dir = os.path.join(os.path.dirname(__file__), 'results')

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
NL = 28; HDIM = 3584; D_INT = 18944
COMP_LAYERS = list(range(0, 4))  # L0, L1, L2, L3

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

all_results = {}

# ================================================================
# STEP 1: BOUNDARY LOCK
# ================================================================
print("\n" + "="*60)
print("  STEP 1: BOUNDARY LOCK")
print("="*60)

step1_data = []

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Capture: embedding output (input to L0), and L3 output
    captures = {}
    def hook_embed(mod, args, output):
        captures['embed'] = output.detach().clone()  # after embedding, before L0
    def hook_l3_out(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captures['l3_out'] = h.detach().clone()

    h_embed = model.model.embed_tokens.register_forward_hook(hook_embed)
    h_l3 = model.model.layers[3].register_forward_hook(hook_l3_out)

    with torch.no_grad():
        out = model(ids, return_dict=True)

    h_embed.remove()
    h_l3.remove()

    # Last token
    h_in = captures['embed'][0, -1].float()
    h_out = captures['l3_out'][0, -1].float()

    norm_in = h_in.norm().item()
    norm_out = h_out.norm().item()
    cos = F.cosine_similarity(h_in.unsqueeze(0), h_out.unsqueeze(0)).item()
    angle = math.degrees(math.acos(max(-1, min(1, cos))))

    step1_data.append({
        'prompt': prompt,
        'norm_in': norm_in, 'norm_out': norm_out,
        'norm_ratio': norm_out / norm_in,
        'cos_sim': cos, 'angle': angle,
    })

mean_angle = np.mean([d['angle'] for d in step1_data])
mean_norm_ratio = np.mean([d['norm_ratio'] for d in step1_data])
print(f"  Compressor (L0-3): mean angle change = {mean_angle:.2f}°")
print(f"  Mean norm ratio (out/in) = {mean_norm_ratio:.4f}")
print(f"  Range: {min(d['angle'] for d in step1_data):.1f}° — {max(d['angle'] for d in step1_data):.1f}°")
all_results['step1'] = {
    'mean_angle': mean_angle,
    'mean_norm_ratio': mean_norm_ratio,
    'per_prompt': step1_data,
}

# ================================================================
# STEP 2: GATE CENSUS
# ================================================================
print("\n" + "="*60)
print("  STEP 2: GATE CENSUS")
print("="*60)

gate_activations = {li: [] for li in COMP_LAYERS}

for prompt in TEST_PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    hooks = []
    gate_acts = {}
    for li in COMP_LAYERS:
        def make_hook(idx):
            def hk(mod, inp, output):
                gate_acts[idx] = output[0, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].mlp.gate_proj.register_forward_hook(make_hook(li)))
    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()
    for li in COMP_LAYERS:
        gate_activations[li].append(gate_acts[li])

step2_data = {}
print(f"\n  {'Layer':>5s} | {'EXPAND':>12s} | {'PRESERVE+':>12s} | {'PRESERVE-':>12s} | {'CONTRACT':>12s} | {'Stab_E':>6s} | {'Stab_C':>6s}")
print("  " + "-"*80)

for li in COMP_LAYERS:
    G = torch.stack(gate_activations[li])
    mean_gate = G.mean(dim=0)

    expand_mask = mean_gate > LOG_PHI
    preserve_p = (mean_gate >= 0) & (mean_gate <= LOG_PHI)
    preserve_n = (mean_gate < 0) & (mean_gate >= -LOG_PHI)
    contract_mask = mean_gate < -LOG_PHI

    n_e = expand_mask.sum().item()
    n_pp = preserve_p.sum().item()
    n_pn = preserve_n.sum().item()
    n_c = contract_mask.sum().item()

    per_prompt_e = (G > LOG_PHI).float().mean(dim=0)
    per_prompt_c = (G < -LOG_PHI).float().mean(dim=0)
    stab_e = per_prompt_e[expand_mask].mean().item() if n_e > 0 else 0
    stab_c = per_prompt_c[contract_mask].mean().item() if n_c > 0 else 0

    print(f"  L{li:>3d} | {n_e:>5d} ({n_e/D_INT*100:4.1f}%) | {n_pp:>5d} ({n_pp/D_INT*100:4.1f}%) | "
          f"{n_pn:>5d} ({n_pn/D_INT*100:4.1f}%) | {n_c:>5d} ({n_c/D_INT*100:4.1f}%) | "
          f"{stab_e:5.3f} | {stab_c:5.3f}")

    step2_data[li] = {
        'n_expand': n_e, 'n_preserve_p': n_pp, 'n_preserve_n': n_pn, 'n_contract': n_c,
        'pct_expand': n_e/D_INT, 'pct_preserve_p': n_pp/D_INT,
        'pct_preserve_n': n_pn/D_INT, 'pct_contract': n_c/D_INT,
        'stability_expand': stab_e, 'stability_contract': stab_c,
        'mean_gate_mean': mean_gate.mean().item(),
        'mean_gate_std': mean_gate.std().item(),
    }

all_results['step2'] = {str(k): v for k, v in step2_data.items()}

# ================================================================
# STEP 3: SIMPLE MACHINES
# ================================================================
print("\n" + "="*60)
print("  STEP 3: SIMPLE MACHINES")
print("="*60)

step3_data = {li: [] for li in COMP_LAYERS}

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Capture all sublayer outputs for L0-3
    layer_caps = {}
    hooks = []

    for li in COMP_LAYERS:
        layer_caps[li] = {}
        layer = model.model.layers[li]

        # Pre-layer hidden state (input_layernorm input = layer input)
        def make_layer_in_hook(idx):
            def hk(mod, args):
                h = args[0] if isinstance(args, tuple) else args
                layer_caps[idx]['layer_in'] = h[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.register_forward_pre_hook(make_layer_in_hook(li)))

        # After input_layernorm
        def make_ln1_hook(idx):
            def hk(mod, inp, output):
                layer_caps[idx]['post_ln1'] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.input_layernorm.register_forward_hook(make_ln1_hook(li)))

        # After attention
        def make_attn_hook(idx):
            def hk(mod, inp, output):
                attn_out = output[0] if isinstance(output, tuple) else output
                layer_caps[idx]['attn_out'] = attn_out[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))

        # After post_attention_layernorm
        def make_ln2_hook(idx):
            def hk(mod, inp, output):
                layer_caps[idx]['post_ln2'] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_ln2_hook(li)))

        # After MLP
        def make_mlp_hook(idx):
            def hk(mod, inp, output):
                layer_caps[idx]['mlp_out'] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))

        # Layer output
        def make_layer_out_hook(idx):
            def hk(mod, args, output):
                h = output[0] if isinstance(output, tuple) else output
                layer_caps[idx]['layer_out'] = h[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.register_forward_hook(make_layer_out_hook(li)))

    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()

    # Compute simple machine metrics per layer
    for li in COMP_LAYERS:
        c = layer_caps[li]
        h_in = c['layer_in']
        h_out = c['layer_out']

        # Damper (LN1): compression ratio
        damper1 = c['post_ln1'].norm().item() / max(h_in.norm().item(), 1e-8)

        # Lever (attention): magnification
        lever = c['attn_out'].norm().item() / max(h_in.norm().item(), 1e-8)

        # h_mid = h_in + attn_out
        h_mid = h_in + c['attn_out']

        # Damper (LN2): compression ratio
        damper2 = c['post_ln2'].norm().item() / max(h_mid.norm().item(), 1e-8)

        # Wedge (FFN): force multiplication
        wedge = c['mlp_out'].norm().item() / max(h_in.norm().item(), 1e-8)

        # Spring (residual): dilution = ||residual|| / ||h_out||
        spring = h_mid.norm().item() / max(h_out.norm().item(), 1e-8)

        # Drift: angle change this layer
        cos_drift = F.cosine_similarity(h_in.unsqueeze(0), h_out.unsqueeze(0)).item()
        drift = math.degrees(math.acos(max(-1, min(1, cos_drift))))

        step3_data[li].append({
            'damper1': damper1, 'lever': lever, 'damper2': damper2,
            'wedge': wedge, 'spring': spring, 'drift': drift,
        })

print(f"\n  {'Layer':>5s} | {'Damper1':>7s} | {'Lever':>7s} | {'Damper2':>7s} | {'Wedge':>7s} | {'Spring':>7s} | {'Drift':>7s}")
print("  " + "-"*60)

step3_summary = {}
for li in COMP_LAYERS:
    metrics = step3_data[li]
    avgs = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
    print(f"  L{li:>3d} | {avgs['damper1']:7.4f} | {avgs['lever']:7.4f} | {avgs['damper2']:7.4f} | "
          f"{avgs['wedge']:7.4f} | {avgs['spring']:7.4f} | {avgs['drift']:6.2f}°")
    step3_summary[li] = avgs

all_results['step3'] = {str(k): v for k, v in step3_summary.items()}

# ================================================================
# STEP 4: INDEPENDENCE TEST
# ================================================================
print("\n" + "="*60)
print("  STEP 4: INDEPENDENCE TEST")
print("="*60)

# Variants:
# A: Real model (baseline)
# B: Skip attention for L0-3
# C: Skip FFN for L0-3
# D: Skip both attention and FFN for L0-3 (pure residual passthrough)

step4_results = {v: {'top1_match': [], 'cos_sim': [], 'angle': []}
                 for v in ['baseline', 'skip_attn', 'skip_ffn', 'skip_both']}

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Baseline
    with torch.no_grad():
        bl_out = model(ids, return_dict=True)
    bl_logits = bl_out.logits[0, -1].float().cpu()
    bl_top1 = bl_logits.argmax().item()
    step4_results['baseline']['top1_match'].append(True)
    step4_results['baseline']['cos_sim'].append(1.0)
    step4_results['baseline']['angle'].append(0.0)

    # Capture embedding output
    embed_cap = {}
    def cap_embed(mod, args, output):
        embed_cap['h'] = output.detach().clone()
    hk = model.model.embed_tokens.register_forward_hook(cap_embed)
    with torch.no_grad(): model(ids)
    hk.remove()
    h_embed = embed_cap['h']

    # For each variant, manually run L0-3 with modifications, then run L4-27 via model
    for vname, skip_attn, skip_ffn in [
        ('skip_attn', True, False),
        ('skip_ffn', False, True),
        ('skip_both', True, True),
    ]:
        h = h_embed.clone()

        # Manual forward through L0-3
        for li in COMP_LAYERS:
            layer = model.model.layers[li]
            residual = h

            # Input LN
            with torch.no_grad():
                h_normed = layer.input_layernorm(h)

            if not skip_attn:
                # We need position embeddings and attention mask for real attention
                # This is complex — use hooks to zero out instead
                pass  # handled below
            # Skip attention: h stays as residual
            h_mid = residual  # if skip_attn, no attention added

            if not skip_attn:
                # Can't easily call self_attn outside forward — use hook approach instead
                h_mid = residual  # placeholder, fixed below

            residual2 = h_mid

            with torch.no_grad():
                h_normed2 = layer.post_attention_layernorm(h_mid)

            if not skip_ffn:
                with torch.no_grad():
                    ffn_out = layer.mlp(h_normed2)
                h = residual2 + ffn_out
            else:
                h = residual2

        # Run remaining layers L4-27 + final norm + lm_head
        # Use hooks to inject our modified hidden state at L4 input
        injected = {'done': False}
        def inject_at_l4(mod, args):
            if not injected['done']:
                injected['done'] = True
                # args is a tuple; first element is hidden_states
                new_args = list(args)
                new_args[0] = h
                return tuple(new_args)
        hk_inject = model.model.layers[4].register_forward_pre_hook(inject_at_l4)

        # Also need to skip L0-3 computation — use hooks to make them passthrough
        passthrough_hooks = []
        for li in COMP_LAYERS:
            def make_passthrough(idx):
                def pt(mod, args, output):
                    # Return input unchanged (passthrough)
                    h_in = args[0] if isinstance(args, tuple) else args
                    if isinstance(output, tuple):
                        return (h_in,) + output[1:]
                    return h_in
                return pt
            passthrough_hooks.append(model.model.layers[li].register_forward_hook(make_passthrough(li)))

        with torch.no_grad():
            var_out = model(ids, return_dict=True)
        var_logits = var_out.logits[0, -1].float().cpu()

        hk_inject.remove()
        for phk in passthrough_hooks: phk.remove()

        var_top1 = var_logits.argmax().item()
        cos = F.cosine_similarity(var_logits.unsqueeze(0), bl_logits.unsqueeze(0)).item()
        angle = math.degrees(math.acos(max(-1, min(1, cos))))

        step4_results[vname]['top1_match'].append(var_top1 == bl_top1)
        step4_results[vname]['cos_sim'].append(cos)
        step4_results[vname]['angle'].append(angle)

    if pi % 5 == 0:
        print(f"  Prompt {pi}/{len(TEST_PROMPTS)}")

print(f"\n  {'Variant':>12s} | {'Top-1%':>6s} | {'cos':>6s} | {'Angle':>6s}")
print("  " + "-"*40)
for vn in ['baseline', 'skip_attn', 'skip_ffn', 'skip_both']:
    r = step4_results[vn]
    n = len(r['top1_match'])
    top1 = sum(r['top1_match']) / n * 100
    cos = np.mean(r['cos_sim'])
    angle = np.mean(r['angle'])
    print(f"  {vn:>12s} | {top1:5.1f}% | {cos:6.4f} | {angle:5.2f}°")

all_results['step4'] = {
    vn: {
        'top1_pct': sum(r['top1_match']) / max(len(r['top1_match']), 1) * 100,
        'mean_cos': float(np.mean(r['cos_sim'])),
        'mean_angle': float(np.mean(r['angle'])),
    }
    for vn, r in step4_results.items()
}

# ================================================================
# STEP 5: TRANSFER FUNCTION
# ================================================================
print("\n" + "="*60)
print("  STEP 5: TRANSFER FUNCTION")
print("="*60)

# Use Step 3 drift data to classify
drifts = [step3_summary[li]['drift'] for li in COMP_LAYERS]
print(f"\n  Per-layer drifts: {['L{}: {:.2f}°'.format(li, d) for li, d in zip(COMP_LAYERS, drifts)]}")

# Fit linear recurrence: drift(l+1) = α·drift(l) + β
if len(drifts) >= 3:
    # Use least squares: [drift(l), 1] @ [α, β]^T = drift(l+1)
    X = np.array([[drifts[i], 1.0] for i in range(len(drifts)-1)])
    y = np.array([drifts[i+1] for i in range(len(drifts)-1)])
    result = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = result[0]

    # Classify
    if alpha < -0.1:
        tf_type = "oscillatory"
    elif 0 <= alpha < 0.9:
        tf_type = "convergent"
        eq = beta / (1 - alpha) if abs(1 - alpha) > 1e-6 else float('inf')
    else:
        tf_type = "step"

    print(f"  Linear recurrence: drift(l+1) = {alpha:.3f}·drift(l) + {beta:.3f}")
    print(f"  Transfer function type: {tf_type}")
    if tf_type == "convergent":
        print(f"  Equilibrium drift: {eq:.2f}°")

    # Also compute cumulative angle from embedding
    cum_angles = []
    for pi, prompt in enumerate(TEST_PROMPTS[:5]):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        caps = {}
        hooks_ca = []

        def cap_emb(mod, args, output):
            caps['emb'] = output[0, -1].detach().float().cpu()
        hooks_ca.append(model.model.embed_tokens.register_forward_hook(cap_emb))

        for li in COMP_LAYERS:
            def make_cap(idx):
                def hk(mod, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    caps[f'l{idx}'] = h[0, -1].detach().float().cpu()
                return hk
            hooks_ca.append(model.model.layers[li].register_forward_hook(make_cap(li)))

        with torch.no_grad(): model(ids)
        for hk in hooks_ca: hk.remove()

        h0 = caps['emb']
        angles_from_emb = []
        for li in COMP_LAYERS:
            h_li = caps[f'l{li}']
            cos_a = F.cosine_similarity(h0.unsqueeze(0), h_li.unsqueeze(0)).item()
            angles_from_emb.append(math.degrees(math.acos(max(-1, min(1, cos_a)))))
        cum_angles.append(angles_from_emb)

    mean_cum = np.mean(cum_angles, axis=0)
    print(f"\n  Cumulative angle from embedding:")
    for li, a in zip(COMP_LAYERS, mean_cum):
        print(f"    After L{li}: {a:.2f}°")

    all_results['step5'] = {
        'drifts': drifts,
        'alpha': float(alpha), 'beta': float(beta),
        'transfer_type': tf_type,
        'cumulative_angles': mean_cum.tolist(),
    }

# ================================================================
# SAVE RESULTS
# ================================================================
out_path = os.path.join(results_dir, 'phase10s_compressor_decompose.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n  Saved to {out_path}")

print("\n" + "="*80)
print("  PHASE 10s COMPLETE — COMPRESSOR DECOMPOSITION STEPS 1-5")
print("="*80)
