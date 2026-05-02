#!/usr/bin/env python3
"""
Phase 10s: Processor Decomposition (L4-25)
Weight Decomposition Protocol Steps 1-5 applied to the Processor sub-machine.
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10s: PROCESSOR DECOMPOSITION (L4-25)")
print("  Weight Decomposition Protocol — Steps 1-5")
print("="*80)
results_dir = os.path.join(os.path.dirname(__file__), 'results')

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
HDIM = 3584; D_INT = 18944
PROC_LAYERS = list(range(4, 26))  # L4-L25
SAMPLE_LAYERS = [4, 8, 12, 16, 20, 25]  # Representative sample for Step 3

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
    captures = {}
    def hook_in(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captures['in'] = h[0, -1].detach().float().cpu()
    def hook_out(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captures['out'] = h[0, -1].detach().float().cpu()
    h1 = model.model.layers[3].register_forward_hook(hook_in)   # L3 output = Processor input
    h2 = model.model.layers[25].register_forward_hook(hook_out)  # L25 output = Processor output
    with torch.no_grad(): model(ids)
    h1.remove(); h2.remove()

    h_in, h_out = captures['in'], captures['out']
    norm_in, norm_out = h_in.norm().item(), h_out.norm().item()
    cos = F.cosine_similarity(h_in.unsqueeze(0), h_out.unsqueeze(0)).item()
    angle = math.degrees(math.acos(max(-1, min(1, cos))))
    step1_data.append({'norm_in': norm_in, 'norm_out': norm_out,
                       'norm_ratio': norm_out/norm_in, 'angle': angle})

mean_angle = np.mean([d['angle'] for d in step1_data])
mean_norm_ratio = np.mean([d['norm_ratio'] for d in step1_data])
print(f"  Processor (L4-25): mean angle change = {mean_angle:.2f}°")
print(f"  Mean norm ratio (out/in) = {mean_norm_ratio:.4f}")
all_results['step1'] = {'mean_angle': mean_angle, 'mean_norm_ratio': mean_norm_ratio}

# ================================================================
# STEP 2: GATE CENSUS
# ================================================================
print("\n" + "="*60)
print("  STEP 2: GATE CENSUS")
print("="*60)

gate_acts = {li: [] for li in PROC_LAYERS}
for prompt in TEST_PROMPTS[:10]:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    hooks = []; acts = {}
    for li in PROC_LAYERS:
        def make_hook(idx):
            def hk(mod, inp, output):
                acts[idx] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].mlp.gate_proj.register_forward_hook(make_hook(li)))
    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()
    for li in PROC_LAYERS: gate_acts[li].append(acts[li])

print(f"\n  {'Layer':>5s} | {'EXPAND':>12s} | {'PRESERVE+':>12s} | {'PRESERVE-':>12s} | {'CONTRACT':>12s} | {'Stab_C':>6s}")
print("  " + "-"*75)

step2_data = {}
for li in PROC_LAYERS:
    G = torch.stack(gate_acts[li])
    mg = G.mean(dim=0)
    n_e = (mg > LOG_PHI).sum().item()
    n_pp = ((mg >= 0) & (mg <= LOG_PHI)).sum().item()
    n_pn = ((mg < 0) & (mg >= -LOG_PHI)).sum().item()
    n_c = (mg < -LOG_PHI).sum().item()
    stab_c = (G < -LOG_PHI).float().mean(dim=0)[mg < -LOG_PHI].mean().item() if n_c > 0 else 0
    print(f"  L{li:>3d} | {n_e:>5d} ({n_e/D_INT*100:4.1f}%) | {n_pp:>5d} ({n_pp/D_INT*100:4.1f}%) | "
          f"{n_pn:>5d} ({n_pn/D_INT*100:4.1f}%) | {n_c:>5d} ({n_c/D_INT*100:4.1f}%) | {stab_c:5.3f}")
    step2_data[li] = {'n_e': n_e, 'n_pp': n_pp, 'n_pn': n_pn, 'n_c': n_c,
                      'pct_e': n_e/D_INT, 'pct_pp': n_pp/D_INT,
                      'pct_pn': n_pn/D_INT, 'pct_c': n_c/D_INT, 'stab_c': stab_c}

all_results['step2'] = {str(k): v for k, v in step2_data.items()}

# ================================================================
# STEP 3: SIMPLE MACHINES (sampled layers)
# ================================================================
print("\n" + "="*60)
print("  STEP 3: SIMPLE MACHINES (sampled)")
print("="*60)

step3_data = {li: [] for li in SAMPLE_LAYERS}
for pi, prompt in enumerate(TEST_PROMPTS[:10]):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    caps = {}; hooks = []

    for li in SAMPLE_LAYERS:
        caps[li] = {}
        layer = model.model.layers[li]
        def make_pre(idx):
            def hk(mod, args):
                h = args[0] if isinstance(args, tuple) else args
                caps[idx]['in'] = h[0, -1].detach().float().cpu()
            return hk
        def make_attn(idx):
            def hk(mod, inp, output):
                a = output[0] if isinstance(output, tuple) else output
                caps[idx]['attn'] = a[0, -1].detach().float().cpu()
            return hk
        def make_mlp(idx):
            def hk(mod, inp, output):
                caps[idx]['mlp'] = output[0, -1].detach().float().cpu()
            return hk
        def make_post(idx):
            def hk(mod, args, output):
                h = output[0] if isinstance(output, tuple) else output
                caps[idx]['out'] = h[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.register_forward_pre_hook(make_pre(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp(li)))
        hooks.append(layer.register_forward_hook(make_post(li)))

    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()

    for li in SAMPLE_LAYERS:
        c = caps[li]
        h_in, h_out = c['in'], c['out']
        lever = c['attn'].norm().item() / max(h_in.norm().item(), 1e-8)
        wedge = c['mlp'].norm().item() / max(h_in.norm().item(), 1e-8)
        spring = h_in.norm().item() / max(h_out.norm().item(), 1e-8)
        cos_d = F.cosine_similarity(h_in.unsqueeze(0), h_out.unsqueeze(0)).item()
        drift = math.degrees(math.acos(max(-1, min(1, cos_d))))
        step3_data[li].append({'lever': lever, 'wedge': wedge, 'spring': spring, 'drift': drift})

print(f"\n  {'Layer':>5s} | {'Lever':>7s} | {'Wedge':>7s} | {'Spring':>7s} | {'Drift':>7s}")
print("  " + "-"*45)
step3_summary = {}
for li in SAMPLE_LAYERS:
    avgs = {k: np.mean([m[k] for m in step3_data[li]]) for k in step3_data[li][0]}
    print(f"  L{li:>3d} | {avgs['lever']:7.4f} | {avgs['wedge']:7.4f} | {avgs['spring']:7.4f} | {avgs['drift']:6.2f}°")
    step3_summary[li] = avgs
all_results['step3'] = {str(k): v for k, v in step3_summary.items()}

# ================================================================
# STEP 4: INDEPENDENCE TEST
# ================================================================
print("\n" + "="*60)
print("  STEP 4: INDEPENDENCE TEST")
print("="*60)

def zero_attn_hook(mod, inp, output):
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + output[1:]
    return torch.zeros_like(output)

def zero_mlp_hook(mod, inp, output):
    return torch.zeros_like(output)

# Define test variants: (name, attn_zero_layers, ffn_zero_layers)
variants = [
    ('skip_attn_all',   PROC_LAYERS,       []),
    ('skip_ffn_all',    [],                 PROC_LAYERS),
    ('skip_both_all',   PROC_LAYERS,        PROC_LAYERS),
    ('skip_attn_4-9',   list(range(4,10)),  []),
    ('skip_attn_10-17', list(range(10,18)), []),
    ('skip_attn_18-25', list(range(18,26)), []),
    ('skip_ffn_4-9',    [],                 list(range(4,10))),
    ('skip_ffn_10-17',  [],                 list(range(10,18))),
    ('skip_ffn_18-25',  [],                 list(range(18,26))),
]

step4_results = {vn: {'top1_match': [], 'cos_sim': [], 'angle': []} for vn, _, _ in variants}
step4_results['baseline'] = {'top1_match': [], 'cos_sim': [], 'angle': []}

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        bl_out = model(ids, return_dict=True)
    bl_logits = bl_out.logits[0, -1].float().cpu()
    bl_top1 = bl_logits.argmax().item()
    step4_results['baseline']['top1_match'].append(True)
    step4_results['baseline']['cos_sim'].append(1.0)
    step4_results['baseline']['angle'].append(0.0)

    for vname, attn_z, ffn_z in variants:
        hooks = []
        for li in attn_z:
            hooks.append(model.model.layers[li].self_attn.register_forward_hook(zero_attn_hook))
        for li in ffn_z:
            hooks.append(model.model.layers[li].mlp.register_forward_hook(zero_mlp_hook))
        with torch.no_grad():
            var_out = model(ids, return_dict=True)
        var_logits = var_out.logits[0, -1].float().cpu()
        for hk in hooks: hk.remove()

        var_top1 = var_logits.argmax().item()
        cos = F.cosine_similarity(var_logits.unsqueeze(0), bl_logits.unsqueeze(0)).item()
        angle = math.degrees(math.acos(max(-1, min(1, cos))))
        step4_results[vname]['top1_match'].append(var_top1 == bl_top1)
        step4_results[vname]['cos_sim'].append(cos)
        step4_results[vname]['angle'].append(angle)

    if pi % 5 == 0:
        print(f"  Prompt {pi}/{len(TEST_PROMPTS)}")

print(f"\n  {'Variant':>18s} | {'Top-1%':>6s} | {'cos':>6s} | {'Angle':>6s}")
print("  " + "-"*45)
for vn in ['baseline'] + [v[0] for v in variants]:
    r = step4_results[vn]
    n = len(r['top1_match'])
    top1 = sum(r['top1_match']) / n * 100
    cos = np.mean(r['cos_sim'])
    angle = np.mean(r['angle'])
    print(f"  {vn:>18s} | {top1:5.1f}% | {cos:6.4f} | {angle:5.2f}°")

all_results['step4'] = {
    vn: {'top1_pct': sum(r['top1_match'])/max(len(r['top1_match']),1)*100,
         'mean_cos': float(np.mean(r['cos_sim'])),
         'mean_angle': float(np.mean(r['angle']))}
    for vn, r in step4_results.items()
}

# ================================================================
# STEP 5: TRANSFER FUNCTION
# ================================================================
print("\n" + "="*60)
print("  STEP 5: TRANSFER FUNCTION")
print("="*60)

# Compute per-layer drift and cumulative angle for a few prompts
all_drifts = {li: [] for li in PROC_LAYERS}
all_cum_angles = {li: [] for li in PROC_LAYERS}

for prompt in TEST_PROMPTS[:5]:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    caps = {}; hooks = []

    # Capture L3 output (processor input) and each layer output
    def cap_proc_in(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        caps['proc_in'] = h[0, -1].detach().float().cpu()
    hooks.append(model.model.layers[3].register_forward_hook(cap_proc_in))

    for li in PROC_LAYERS:
        def make_cap(idx):
            def hk(mod, args, output):
                h = output[0] if isinstance(output, tuple) else output
                caps[f'l{idx}'] = h[0, -1].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].register_forward_hook(make_cap(li)))

    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()

    h_prev = caps['proc_in']
    h_ref = caps['proc_in']
    for li in PROC_LAYERS:
        h_cur = caps[f'l{li}']
        # Drift: angle from previous layer
        cos_d = F.cosine_similarity(h_prev.unsqueeze(0), h_cur.unsqueeze(0)).item()
        drift = math.degrees(math.acos(max(-1, min(1, cos_d))))
        all_drifts[li].append(drift)
        # Cumulative: angle from processor input
        cos_c = F.cosine_similarity(h_ref.unsqueeze(0), h_cur.unsqueeze(0)).item()
        cum = math.degrees(math.acos(max(-1, min(1, cos_c))))
        all_cum_angles[li].append(cum)
        h_prev = h_cur

mean_drifts = {li: np.mean(all_drifts[li]) for li in PROC_LAYERS}
mean_cum = {li: np.mean(all_cum_angles[li]) for li in PROC_LAYERS}

print("\n  Per-layer drift and cumulative angle:")
print(f"  {'Layer':>5s} | {'Drift':>7s} | {'Cumul':>7s}")
print("  " + "-"*25)
for li in PROC_LAYERS:
    print(f"  L{li:>3d} | {mean_drifts[li]:6.2f}° | {mean_cum[li]:6.2f}°")

# Fit linear recurrence
drifts_list = [mean_drifts[li] for li in PROC_LAYERS]
X = np.array([[drifts_list[i], 1.0] for i in range(len(drifts_list)-1)])
y = np.array([drifts_list[i+1] for i in range(len(drifts_list)-1)])
result = np.linalg.lstsq(X, y, rcond=None)
alpha, beta = result[0]

if alpha < -0.1:
    tf_type = "oscillatory"
elif 0 <= alpha < 0.9:
    tf_type = "convergent"
    eq = beta / (1 - alpha) if abs(1-alpha) > 1e-6 else float('inf')
else:
    tf_type = "step_or_linear"

print(f"\n  Linear recurrence: drift(l+1) = {alpha:.3f}·drift(l) + {beta:.3f}")
print(f"  Transfer function type: {tf_type}")
if tf_type == "convergent":
    print(f"  Equilibrium drift: {eq:.2f}°")

all_results['step5'] = {
    'drifts': {str(li): mean_drifts[li] for li in PROC_LAYERS},
    'cumulative_angles': {str(li): mean_cum[li] for li in PROC_LAYERS},
    'alpha': float(alpha), 'beta': float(beta), 'transfer_type': tf_type,
}

# Save
out_path = os.path.join(results_dir, 'phase10s_processor_decompose.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n  Saved to {out_path}")
print("\n" + "="*80)
print("  PHASE 10s COMPLETE — PROCESSOR DECOMPOSITION STEPS 1-5")
print("="*80)
