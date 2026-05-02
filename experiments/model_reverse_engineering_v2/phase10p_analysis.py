# phase10p_analysis.py — Data collection and analysis
# Exec'd from phase10p_simple_machines.py (shares its namespace)

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
]

# ================================================================
# COLLECT DATA
# ================================================================
print("Collecting real and approximate states for all prompts...")
all_data = []

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    real_map = {li: attn_real_qk for li in range(NL)}
    _, real_inputs, real_attn, real_outputs = run_with_capture(ids, real_map)
    approx_map = {li: attn_bias_aware for li in range(NL)}
    _, approx_inputs, approx_attn, approx_outputs = run_with_capture(ids, approx_map)

    prompt_data = {'prompt': prompt, 'layers': {}}
    for li in range(NL):
        h_real = real_inputs[li]; h_approx = approx_inputs[li]
        a_real = real_attn[li]; a_approx = approx_attn[li]
        o_real = real_outputs[li]; o_approx = approx_outputs[li]

        eps_in = h_approx - h_real
        eps_in_norm = eps_in.norm().item()
        h_real_norm = h_real.norm().item()

        # Mid-layer: h + attn_output
        mid_real = h_real + a_real; mid_approx = h_approx + a_approx
        eps_mid = mid_approx - mid_real
        eps_out = o_approx - o_real

        # LEVER (Attention)
        attn_delta = a_approx - a_real
        lever_mag = attn_delta.norm().item() / (h_real_norm + 1e-10)
        cos_attn_eps = (F.cosine_similarity(attn_delta.unsqueeze(0),
            eps_in.unsqueeze(0)).item() if eps_in_norm > 1e-6 else 0.0)
        cos_attn_h = F.cosine_similarity(
            attn_delta.unsqueeze(0), h_real.unsqueeze(0)).item()

        # DAMPER (LN nonlinearity residual)
        eps_mid_ideal = eps_in + attn_delta
        damper_residual = eps_mid - eps_mid_ideal
        damper_mag = damper_residual.norm().item() / (eps_in_norm + 1e-10)

        # WEDGE (FFN)
        ffn_delta = eps_out - eps_mid
        wedge_mag = ffn_delta.norm().item() / (h_real_norm + 1e-10)
        cos_ffn_eps = (F.cosine_similarity(ffn_delta.unsqueeze(0),
            eps_mid.unsqueeze(0)).item() if eps_mid.norm().item() > 1e-6 else 0.0)
        cos_ffn_h = F.cosine_similarity(
            ffn_delta.unsqueeze(0), h_real.unsqueeze(0)).item()

        # SPRING (Residual)
        delta_total = eps_out - eps_in
        spring_k = h_real_norm / (delta_total.norm().item() + 1e-10)

        # Overall metrics
        drift_in = eps_in_norm / (h_real_norm + 1e-10)
        drift_out = eps_out.norm().item() / (o_real.norm().item() + 1e-10)
        cos_eps_h = (F.cosine_similarity(eps_in.unsqueeze(0),
            h_real.unsqueeze(0)).item() if eps_in_norm > 1e-6 else 0.0)
        angle_in = (math.degrees(math.acos(max(-1, min(1,
            F.cosine_similarity(h_real.unsqueeze(0),
                h_approx.unsqueeze(0)).item()
        )))) if eps_in_norm > 1e-6 else 0.0)

        prompt_data['layers'][li] = {
            'lever_mag': lever_mag, 'cos_attn_eps': cos_attn_eps,
            'cos_attn_h': cos_attn_h,
            'damper_mag': damper_mag,
            'wedge_mag': wedge_mag, 'cos_ffn_eps': cos_ffn_eps,
            'cos_ffn_h': cos_ffn_h,
            'spring_k': spring_k,
            'drift_in': drift_in, 'drift_out': drift_out,
            'cos_eps_h': cos_eps_h, 'angle_in': angle_in,
            'eps_in_norm': eps_in_norm, 'h_norm': h_real_norm,
        }
    all_data.append(prompt_data)
    print(f"  Prompt {pi}: '{prompt}'")

print(f"  {len(TEST_PROMPTS)} prompts done.\n")

# ================================================================
# AGGREGATE
# ================================================================
machine_stats = defaultdict(lambda: defaultdict(list))
for pd in all_data:
    for li in range(NL):
        for key, val in pd['layers'][li].items():
            machine_stats[li][key].append(val)

def zone(li):
    if li <= 3: return "DRUM"
    if li <= 25: return "COMB"
    return "MUSIC"

# ================================================================
# PRINT MACHINE PROFILES
# ================================================================
print("="*80)
print("  MACHINE PARAMETER PROFILES")
print("="*80)

print("\n  THE GEOMETRIC LEVER (Attention)")
print(f"  {'Lyr':>3s} {'Zone':>5s} {'Lever':>7s} {'cos(d,e)':>8s} {'cos(d,h)':>8s}")
print("  " + "-"*35)
for li in range(NL):
    lm = np.mean(machine_stats[li]['lever_mag'])
    ce = np.mean(machine_stats[li]['cos_attn_eps'])
    ch = np.mean(machine_stats[li]['cos_attn_h'])
    print(f"  L{li:2d} {zone(li):>5s} {lm:7.4f} {ce:+8.4f} {ch:+8.4f}")

print("\n  THE GEOMETRIC DAMPER (LN non-ideality)")
print(f"  {'Lyr':>3s} {'Zone':>5s} {'Damper':>7s}  (0=ideal)")
print("  " + "-"*25)
for li in range(NL):
    dm = np.mean(machine_stats[li]['damper_mag'])
    print(f"  L{li:2d} {zone(li):>5s} {dm:7.4f}")

print("\n  THE GEOMETRIC WEDGE (FFN)")
print(f"  {'Lyr':>3s} {'Zone':>5s} {'Wedge':>7s} {'cos(d,e)':>8s} {'cos(d,h)':>8s}")
print("  " + "-"*35)
for li in range(NL):
    wm = np.mean(machine_stats[li]['wedge_mag'])
    ce = np.mean(machine_stats[li]['cos_ffn_eps'])
    ch = np.mean(machine_stats[li]['cos_ffn_h'])
    print(f"  L{li:2d} {zone(li):>5s} {wm:7.4f} {ce:+8.4f} {ch:+8.4f}")

print("\n  THE GEOMETRIC SPRING (Residual)")
print(f"  {'Lyr':>3s} {'Zone':>5s} {'k':>7s} {'Drift':>7s} {'Angle':>7s}")
print("  " + "-"*30)
for li in range(NL):
    sk = np.mean(machine_stats[li]['spring_k'])
    di = np.mean(machine_stats[li]['drift_in'])
    ai = np.mean(machine_stats[li]['angle_in'])
    print(f"  L{li:2d} {zone(li):>5s} {sk:7.2f} {di:7.4f} {ai:6.1f}d")

# ================================================================
# ZONE SUMMARY
# ================================================================
print("\n" + "="*80)
print("  ZONE-BY-ZONE MACHINE SUMMARY")
print("="*80)
zones = [("DRUM", range(0,4)), ("COMB", range(4,26)), ("MUSIC", range(26,28))]
print(f"\n  {'Zone':<6s} {'Lever':>7s} {'Wedge':>7s} {'Damper':>7s} {'Spring':>7s}"
      f" {'cos_a,h':>7s} {'cos_f,h':>7s}")
print("  " + "-"*50)
for zn, zr in zones:
    lm = np.mean([np.mean(machine_stats[li]['lever_mag']) for li in zr])
    wm = np.mean([np.mean(machine_stats[li]['wedge_mag']) for li in zr])
    dm = np.mean([np.mean(machine_stats[li]['damper_mag']) for li in zr])
    sk = np.mean([np.mean(machine_stats[li]['spring_k']) for li in zr])
    ca = np.mean([np.mean(machine_stats[li]['cos_attn_h']) for li in zr])
    cf = np.mean([np.mean(machine_stats[li]['cos_ffn_h']) for li in zr])
    print(f"  {zn:<6s} {lm:7.4f} {wm:7.4f} {dm:7.4f} {sk:7.2f} {ca:+7.4f} {cf:+7.4f}")

# ================================================================
# ANGLE DECOMPOSITION
# ================================================================
print("\n" + "="*80)
print("  ANGLE DECOMPOSITION: WHICH MACHINE MOVES THE ANGLE?")
print("="*80)
print(f"\n  {'Lyr':>3s} {'Zone':>5s} {'Angle':>6s} {'dAngle':>7s}"
      f" {'Lever%':>7s} {'Wedge%':>7s} {'Damp%':>7s}")
print("  " + "-"*48)
prev_angle = 0; lever_total = 0; wedge_total = 0; damper_total = 0
for li in range(NL):
    angle = np.mean(machine_stats[li]['angle_in'])
    lm = np.mean(machine_stats[li]['lever_mag'])
    wm = np.mean(machine_stats[li]['wedge_mag'])
    dm = np.mean(machine_stats[li]['damper_mag'])
    total = lm + wm + dm + 1e-10
    delta_angle = angle - prev_angle
    lever_total += lm; wedge_total += wm; damper_total += dm
    print(f"  L{li:2d} {zone(li):>5s} {angle:5.1f}d {delta_angle:+6.1f}d"
          f" {lm/total*100:6.1f}% {wm/total*100:6.1f}% {dm/total*100:6.1f}%")
    prev_angle = angle

total_m = lever_total + wedge_total + damper_total
print(f"\n  MACHINE CONTRIBUTION TOTALS:")
print(f"    Lever (Attention):  {lever_total:.4f}  ({lever_total/total_m*100:.1f}%)")
print(f"    Wedge (FFN):        {wedge_total:.4f}  ({wedge_total/total_m*100:.1f}%)")
print(f"    Damper (LN):        {damper_total:.4f}  ({damper_total/total_m*100:.1f}%)")

# ================================================================
# THE 0.85° ERROR ANALYSIS
# ================================================================
print("\n" + "="*80)
print("  THE 0.85 DEG ERROR ANALYSIS")
print("="*80)

# Compute from conserved quantities at L27 output
all_r = [pd['layers'][NL-1]['drift_out'] for pd in all_data]
all_cos = [pd['layers'][NL-1]['cos_eps_h'] for pd in all_data]
mean_r = np.mean(all_r); mean_c = np.mean(all_cos)

# Predicted norm ratio from r and c
norm_ratio_pred = math.sqrt(1 + 2*mean_r*mean_c + mean_r**2)
# Predicted angle from cos law: cos(theta) = (1 + r*c) / norm_ratio
cos_theta_pred = (1 + mean_r * mean_c) / norm_ratio_pred
angle_pred = math.degrees(math.acos(max(-1, min(1, cos_theta_pred))))

phi_angle = math.degrees(math.acos(1.0 / PHI**2))
final_angles = [pd['layers'][NL-1]['angle_in'] for pd in all_data]
measured_angle = np.mean(final_angles)

print(f"\n  Measured mean angle at L27:     {measured_angle:.2f} deg")
print(f"  arccos(1/phi^2):                {phi_angle:.2f} deg")
print(f"  Error (measured - phi):         {measured_angle - phi_angle:.2f} deg")
print(f"\n  Mean drift ratio r:             {mean_r:.4f}")
print(f"  Mean cos(eps,h):                {mean_c:+.4f}")
print(f"  Predicted norm ratio:           {norm_ratio_pred:.4f}")
print(f"  Predicted angle (from r,c):     {angle_pred:.2f} deg")
print(f"  Gap (predicted - measured):     {angle_pred - measured_angle:.2f} deg")

# Per-zone contribution to the angle
print(f"\n  PER-ZONE ANGLE CONTRIBUTION:")
for zn, zr in zones:
    zr_list = list(zr)
    a_start = np.mean(machine_stats[zr_list[0]]['angle_in'])
    a_end = np.mean(machine_stats[zr_list[-1]]['angle_in'])
    # For DRUM starting at L0, start angle is 0
    if zn == "DRUM": a_start = 0.0
    delta = a_end - a_start
    print(f"    {zn}: {a_start:.1f}d -> {a_end:.1f}d  (contributes {delta:+.1f} deg)")

# ================================================================
# IDEAL MACHINE SIMULATION
# ================================================================
print("\n" + "="*80)
print("  IDEAL MACHINE SIMULATION")
print("="*80)

lever_profile = [np.mean(machine_stats[li]['lever_mag']) for li in range(NL)]
wedge_profile = [np.mean(machine_stats[li]['wedge_mag']) for li in range(NL)]
spring_profile = [np.mean(machine_stats[li]['spring_k']) for li in range(NL)]

# Model A: Fully constant (zone-agnostic)
mean_lever = np.mean(lever_profile); mean_wedge = np.mean(wedge_profile)
mean_spring = np.mean(spring_profile)
print(f"\n  Model A (constant): lever={mean_lever:.4f} wedge={mean_wedge:.4f} k={mean_spring:.2f}")
sim_drift_A = [0.0]
for li in range(NL):
    f = mean_lever + mean_wedge
    sim_drift_A.append(sim_drift_A[-1] + f / max(mean_spring, 0.1))

# Model B: Zone-constant (different per zone)
print(f"  Model B (zone-constant):")
zone_params = {}
for zn, zr in zones:
    zl = np.mean([lever_profile[li] for li in zr])
    zw = np.mean([wedge_profile[li] for li in zr])
    zk = np.mean([spring_profile[li] for li in zr])
    zone_params[zn] = (zl, zw, zk)
    print(f"    {zn}: lever={zl:.4f} wedge={zw:.4f} k={zk:.2f}")
sim_drift_B = [0.0]
for li in range(NL):
    zn = zone(li)
    zl, zw, zk = zone_params[zn]
    sim_drift_B.append(sim_drift_B[-1] + (zl + zw) / max(zk, 0.1))

# Model C: Exact per-layer (upper bound)
sim_drift_C = [0.0]
for li in range(NL):
    f = lever_profile[li] + wedge_profile[li]
    sim_drift_C.append(sim_drift_C[-1] + f / max(spring_profile[li], 0.1))

real_drift_profile = [np.mean(machine_stats[li]['drift_in']) for li in range(NL)]

print(f"\n  {'Lyr':>3s} {'Real':>7s} {'Const':>7s} {'Zone':>7s} {'Exact':>7s}")
print("  " + "-"*35)
for li in range(NL):
    rd = real_drift_profile[li]
    print(f"  L{li:2d} {rd:7.4f} {sim_drift_A[li]:7.4f} {sim_drift_B[li]:7.4f} {sim_drift_C[li]:7.4f}")

print(f"\n  Final drift: real={real_drift_profile[-1]:.4f}"
      f" const={sim_drift_A[-1]:.4f} zone={sim_drift_B[-1]:.4f}"
      f" exact={sim_drift_C[-1]:.4f}")

# ================================================================
# SAVE RESULTS
# ================================================================
save_data = {
    'prompts': [pd['prompt'] for pd in all_data],
    'n_layers': NL,
    'phi_angle_deg': phi_angle,
    'measured_angle_deg': measured_angle,
    'error_deg': measured_angle - phi_angle,
    'mean_drift_ratio': mean_r,
    'mean_cos_eps_h': mean_c,
    'per_layer': {},
    'zone_summary': {},
    'machine_totals': {
        'lever': lever_total, 'wedge': wedge_total, 'damper': damper_total,
        'lever_pct': lever_total/total_m*100,
        'wedge_pct': wedge_total/total_m*100,
        'damper_pct': damper_total/total_m*100,
    },
}
for li in range(NL):
    save_data['per_layer'][li] = {
        k: float(np.mean(v)) for k, v in machine_stats[li].items()
    }
    save_data['per_layer'][li]['zone'] = zone(li)
for zn, zr in zones:
    save_data['zone_summary'][zn] = {
        'lever': float(np.mean([np.mean(machine_stats[li]['lever_mag']) for li in zr])),
        'wedge': float(np.mean([np.mean(machine_stats[li]['wedge_mag']) for li in zr])),
        'damper': float(np.mean([np.mean(machine_stats[li]['damper_mag']) for li in zr])),
        'spring': float(np.mean([np.mean(machine_stats[li]['spring_k']) for li in zr])),
    }

out_path = os.path.join(results_dir, 'phase10p_simple_machines.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\n  Results saved to {out_path}")
print("\n" + "="*80)
print("  PHASE 10p COMPLETE")
print("="*80)
