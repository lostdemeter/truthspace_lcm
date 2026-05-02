#!/usr/bin/env python3
"""Phase 10p Refinement: LN damper ratio + L27 output angle."""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, json, os, math
PHI = (1 + np.sqrt(5)) / 2
results_dir = os.path.join(os.path.dirname(__file__), 'results')
print("="*70)
print("  PHASE 10p REFINEMENT")
print("="*70)

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
NL=28; HDIM=3584

PROMPTS = [
    "The capital of France is", "The largest ocean is the",
    "The color of grass is", "Barack Obama was the",
    "To be or not to", "Roses are red, violets are",
    "The speed of light is approximately",
    "Albert Einstein developed the theory of",
    "Water freezes at zero degrees", "The chemical symbol for gold is",
]

# Measure LN1 compression: ||LN(h+ε)-LN(h)||/||ε|| for ε along -h
print("\nMeasuring LN1 damper compression ratio...")
ln1_ratios = {li: [] for li in range(NL)}

for pi, prompt in enumerate(PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    # Capture layer inputs
    layer_h = {}
    hooks = []
    for li in range(NL):
        def mk(idx):
            def hk(m, a):
                h = a[0] if isinstance(a[0], torch.Tensor) else a[0][0]
                layer_h[idx] = h[0, -1, :].detach().clone()
            return hk
        hooks.append(model.model.layers[li].register_forward_pre_hook(mk(li)))
    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()

    for li in range(NL):
        h = layer_h[li]  # bfloat16 on GPU
        eps = -h * 0.01  # 1% perturbation along -h
        ln = model.model.layers[li].input_layernorm
        with torch.no_grad():
            out_h = ln(h.unsqueeze(0).unsqueeze(0))
            out_hp = ln((h + eps).unsqueeze(0).unsqueeze(0))
        delta_ln = (out_hp - out_h).float().norm().item()
        eps_norm = eps.float().norm().item()
        ln1_ratios[li].append(delta_ln / (eps_norm + 1e-10))

    if pi % 3 == 0: print(f"  Prompt {pi}")

def zone(li):
    if li <= 3: return "DRUM"
    if li <= 25: return "COMB"
    return "MUSIC"

print(f"\n  LN1 DAMPER COMPRESSION RATIO (1.0=no damping, <1=compression)")
print(f"  {'Lyr':>3s} {'Zone':>5s} {'Ratio':>7s} {'Compress%':>10s}")
print("  " + "-"*30)
for li in range(NL):
    r = np.mean(ln1_ratios[li])
    print(f"  L{li:2d} {zone(li):>5s} {r:7.4f} {(1-r)*100:8.1f}%")

# Zone averages
print(f"\n  Zone averages:")
for zn, zr in [("DRUM",range(0,4)),("COMB",range(4,26)),("MUSIC",range(26,28))]:
    r = np.mean([np.mean(ln1_ratios[li]) for li in zr])
    print(f"    {zn}: ratio={r:.4f}, compression={((1-r)*100):.1f}%")

# L27 output angle analysis from phase10p data
print(f"\n" + "="*70)
print("  L27 ANALYSIS: THE MUSIC LAYER AS PRECISION MACHINE")
print("="*70)
with open(os.path.join(results_dir, 'phase10p_simple_machines.json')) as f:
    p10p = json.load(f)

l27 = p10p['per_layer']['27']
phi_angle = math.degrees(math.acos(1.0 / PHI**2))

print(f"\n  L27 INPUT angle:           {l27['angle_in']:.2f} deg")
print(f"  L27 lever magnitude:       {l27['lever_mag']:.4f}")
print(f"  L27 wedge magnitude:       {l27['wedge_mag']:.4f}  (HIGHEST in network)")
print(f"  L27 spring constant:       {l27['spring_k']:.4f}  (SOFTEST in network)")
print(f"  L27 drift_in:              {l27['drift_in']:.4f}")
print(f"  L27 drift_out:             {l27['drift_out']:.4f}")
print(f"  L27 cos(attn_delta, h):    {l27['cos_attn_h']:+.4f}")
print(f"  L27 cos(ffn_delta, h):     {l27['cos_ffn_h']:+.4f}")

# L27 adds ~11.5 deg (from 56.9 to ~68.4 from phase10o)
phase10o_angle = 68.39  # from previous experiment
l27_contribution = phase10o_angle - l27['angle_in']
print(f"\n  L27 CONTRIBUTION:")
print(f"    Input angle:   {l27['angle_in']:.2f} deg")
print(f"    Output angle:  ~{phase10o_angle:.2f} deg (from phase10o)")
print(f"    L27 adds:      ~{l27_contribution:.2f} deg")
print(f"    Target (phi):  {phi_angle:.2f} deg")
print(f"    Overshoot:     {phase10o_angle - phi_angle:.2f} deg")

# Machine decomposition at L27
total_force = l27['lever_mag'] + l27['wedge_mag']
print(f"\n  L27 MACHINE DECOMPOSITION:")
print(f"    Lever (attn): {l27['lever_mag']:.4f} ({l27['lever_mag']/total_force*100:.1f}%)")
print(f"    Wedge (FFN):  {l27['wedge_mag']:.4f} ({l27['wedge_mag']/total_force*100:.1f}%)")
print(f"    → FFN at L27 is {l27['wedge_mag']/l27['lever_mag']:.1f}x stronger than attention")

# Linear recurrence model
print(f"\n" + "="*70)
print("  LINEAR RECURRENCE MODEL: drift(l+1) = α·drift(l) + β")
print("="*70)
# Fit α,β per zone from consecutive drift values
for zn, zr in [("DRUM",range(0,4)),("COMB",range(4,26)),("MUSIC",range(26,28))]:
    drifts_in = [p10p['per_layer'][str(li)]['drift_in'] for li in zr]
    drifts_out = [p10p['per_layer'][str(li)]['drift_out'] for li in zr]
    # drift_out[li] becomes drift_in[li+1], so fit: drift_out = α·drift_in + β
    if len(drifts_in) >= 2:
        x = np.array(drifts_in); y = np.array(drifts_out)
        # Filter out L0 (drift_in=0)
        mask = x > 0.01
        if mask.sum() >= 2:
            xm, ym = x[mask], y[mask]
            A = np.vstack([xm, np.ones(len(xm))]).T
            alpha, beta = np.linalg.lstsq(A, ym, rcond=None)[0]
            eq = beta / (1 - alpha) if abs(1-alpha) > 0.01 else float('inf')
            print(f"  {zn}: α={alpha:.4f} β={beta:+.4f} → equilibrium={eq:.4f}")
        else:
            print(f"  {zn}: insufficient non-zero data")
    else:
        print(f"  {zn}: insufficient data")

# Save refinement results
refine = {
    'ln1_compression': {li: float(np.mean(ln1_ratios[li])) for li in range(NL)},
    'l27_contribution_deg': l27_contribution,
    'l27_overshoot_deg': phase10o_angle - phi_angle,
    'l27_ffn_to_attn_ratio': l27['wedge_mag'] / l27['lever_mag'],
}
out_path = os.path.join(results_dir, 'phase10p_refine.json')
with open(out_path, 'w') as f: json.dump(refine, f, indent=2)
print(f"\n  Saved to {out_path}")
print("="*70)
print("  REFINEMENT COMPLETE")
print("="*70)
