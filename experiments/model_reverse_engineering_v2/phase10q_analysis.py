# phase10q_analysis.py — Compound Machine Tests (exec'd from phase10q_compound_machine.py)

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

def angle_between(a, b):
    if a.norm() < 1e-8 or b.norm() < 1e-8: return 0.0
    c = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return math.degrees(math.acos(max(-1, min(1, c))))

def effective_rank(vec_list):
    if len(vec_list) < 2: return 1.0
    M = torch.stack(vec_list)
    M = M - M.mean(dim=0, keepdim=True)
    try:
        S = torch.linalg.svdvals(M.float())
        S = S[S > 1e-10]
        if len(S) == 0: return 0.0
        p = S / S.sum()
        entropy = -(p * p.log()).sum().item()
        return math.exp(entropy)
    except: return float('nan')

# ================================================================
# TEST 1: INDEPENDENT LINEARIZATION
# ================================================================
print("\n" + "="*70)
print("  TEST 1: INDEPENDENT LINEARIZATION")
print("  (Approximate each machine separately vs all at once)")
print("="*70)

# 5 configurations:
# 0: all real (baseline)
# 1: all approximate (global)
# 2: only COMPRESSOR approximate (L0-3)
# 3: only PROCESSOR approximate (L4-25)
# 4: only TARGETER approximate (L26-27)
configs = {
    'baseline':    {},  # all real (default in run_with_capture)
    'global':      {li: attn_bias_aware for li in range(NL)},
    'compressor':  {li: attn_bias_aware for li in COMPRESSOR},
    'processor':   {li: attn_bias_aware for li in PROCESSOR},
    'targeter':    {li: attn_bias_aware for li in TARGETER},
}

# Boundary layers for measurement: end of each machine
BOUNDARIES = {
    'comp_out': 3,    # last layer of Compressor
    'proc_in': 4,     # first layer of Processor
    'proc_out': 25,   # last layer of Processor
    'targ_in': 26,    # first layer of Targeter
    'targ_out': 27,   # last layer of Targeter (final)
}

# Collect per-config boundary angles and states
config_results = {cn: {'angles': defaultdict(list), 'drifts': defaultdict(list),
                       'boundary_errors': defaultdict(list), 'top1_match': []}
                  for cn in configs}

# Also collect error vectors at interfaces for Test 2
interface_errors = {cn: {'comp_proc': [], 'proc_targ': []} for cn in configs if cn != 'baseline'}

print(f"\nRunning {len(configs)} configs × {len(TEST_PROMPTS)} prompts...")

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Run baseline first
    bl_logits, bl_inputs, bl_attn, bl_outputs = run_with_capture(ids, configs['baseline'])
    bl_top1 = bl_logits[0, -1].argmax().item()

    for cn, fn_map in configs.items():
        if cn == 'baseline':
            config_results[cn]['top1_match'].append(True)
            continue

        logits, inputs, attn_outs, outputs = run_with_capture(ids, fn_map)
        top1 = logits[0, -1].argmax().item()
        config_results[cn]['top1_match'].append(top1 == bl_top1)

        # Measure angle and drift at each boundary
        for bname, bli in BOUNDARIES.items():
            h_real = bl_outputs[bli]
            h_approx = outputs[bli]
            ang = angle_between(h_real, h_approx)
            eps = h_approx - h_real
            drift = eps.norm().item() / (h_real.norm().item() + 1e-10)
            config_results[cn]['angles'][bname].append(ang)
            config_results[cn]['drifts'][bname].append(drift)
            config_results[cn]['boundary_errors'][bname].append(eps)

        # Collect interface error vectors for Test 2
        if cn in interface_errors:
            # Compressor→Processor interface: error at L3 output
            eps_cp = outputs[3] - bl_outputs[3]
            interface_errors[cn]['comp_proc'].append(eps_cp)
            # Processor→Targeter interface: error at L25 output
            eps_pt = outputs[25] - bl_outputs[25]
            interface_errors[cn]['proc_targ'].append(eps_pt)

    if pi % 3 == 0:
        print(f"  Prompt {pi}/{len(TEST_PROMPTS)}")

# Print Test 1 results
print(f"\n  {'Config':>12s} | {'Top1%':>5s} | {'CompOut':>7s} | {'ProcOut':>7s} | {'TargOut':>7s} | {'Final°':>7s}")
print("  " + "-"*65)
for cn in ['global', 'compressor', 'processor', 'targeter']:
    r = config_results[cn]
    top1 = sum(r['top1_match']) / len(r['top1_match']) * 100
    co = np.mean(r['angles']['comp_out']) if r['angles']['comp_out'] else 0
    po = np.mean(r['angles']['proc_out']) if r['angles']['proc_out'] else 0
    to = np.mean(r['angles']['targ_out']) if r['angles']['targ_out'] else 0
    print(f"  {cn:>12s} | {top1:5.1f} | {co:6.2f}° | {po:6.2f}° | {to:6.2f}° | {to:6.2f}°")

# Additivity test: do individual machine angles sum to global?
global_final = np.mean(config_results['global']['angles']['targ_out'])
comp_final = np.mean(config_results['compressor']['angles']['targ_out'])
proc_final = np.mean(config_results['processor']['angles']['targ_out'])
targ_final = np.mean(config_results['targeter']['angles']['targ_out'])
linear_sum = comp_final + proc_final + targ_final

print(f"\n  ADDITIVITY TEST:")
print(f"    Global approx final angle:              {global_final:.2f}°")
print(f"    Sum of individual machine final angles:  {linear_sum:.2f}°")
print(f"    Ratio (sum/global):                      {linear_sum/global_final:.3f}")
print(f"    If ratio ≈ 1.0 → machines are independent (linear)")
print(f"    If ratio ≠ 1.0 → machines interact (nonlinear composition)")

# Isolation test: which machine contributes most to global error?
print(f"\n  MACHINE CONTRIBUTION TO FINAL ANGLE:")
print(f"    Compressor only:  {comp_final:6.2f}° ({comp_final/global_final*100:.1f}% of global)")
print(f"    Processor only:   {proc_final:6.2f}° ({proc_final/global_final*100:.1f}% of global)")
print(f"    Targeter only:    {targ_final:6.2f}° ({targ_final/global_final*100:.1f}% of global)")

# ================================================================
# TEST 2: INTERFACE DIMENSIONALITY
# ================================================================
print(f"\n" + "="*70)
print("  TEST 2: INTERFACE DIMENSIONALITY")
print("  (Effective rank of error vectors at machine boundaries)")
print("="*70)

for cn in ['global', 'compressor', 'processor', 'targeter']:
    r = config_results[cn]
    print(f"\n  Config: {cn}")
    for bname in ['comp_out', 'proc_out', 'targ_out']:
        errs = r['boundary_errors'][bname]
        if len(errs) >= 2:
            erank = effective_rank(errs)
            mean_norm = np.mean([e.norm().item() for e in errs])
            print(f"    {bname:>8s}: eff_rank={erank:.2f}, mean_‖ε‖={mean_norm:.4f}")
        else:
            print(f"    {bname:>8s}: insufficient data")

# Cross-interface comparison for global approx
if len(interface_errors['global']['comp_proc']) >= 2:
    cp_rank = effective_rank(interface_errors['global']['comp_proc'])
    pt_rank = effective_rank(interface_errors['global']['proc_targ'])
    # Also measure bulk (mid-processor, L12)
    bulk_errs = config_results['global']['boundary_errors'].get('proc_out', [])
    bulk_rank = effective_rank(bulk_errs) if len(bulk_errs) >= 2 else float('nan')
    print(f"\n  INTERFACE vs BULK (global approx):")
    print(f"    Compressor→Processor (L3):  eff_rank = {cp_rank:.2f}")
    print(f"    Processor→Targeter (L25):   eff_rank = {pt_rank:.2f}")
    print(f"    Processor output (L25):     eff_rank = {bulk_rank:.2f}")
    print(f"    Lower rank at interfaces → simpler interfaces → easier to model")

# ================================================================
# TEST 3: TRANSFER FUNCTION PER MACHINE
# ================================================================
print(f"\n" + "="*70)
print("  TEST 3: TRANSFER FUNCTION PER MACHINE")
print("  (How does error propagate within each machine?)")
print("="*70)

# Use the global approx data to trace error propagation layer by layer
# For each prompt, measure angle(h_real, h_approx) at every layer output
per_layer_angles = defaultdict(list)

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    bl_logits, bl_inputs, bl_attn, bl_outputs = run_with_capture(ids, configs['baseline'])
    ap_logits, ap_inputs, ap_attn, ap_outputs = run_with_capture(ids, configs['global'])

    for li in range(NL):
        ang = angle_between(bl_outputs[li], ap_outputs[li])
        per_layer_angles[li].append(ang)

    if pi % 5 == 0: print(f"  Prompt {pi}")

# Compute per-machine transfer: angle growth rate within each machine
print(f"\n  {'Layer':>5s} {'Machine':>12s} {'Angle°':>7s} {'Δ from prev':>12s}")
print("  " + "-"*45)
prev_ang = 0.0
machine_growth = {'COMPRESSOR': [], 'PROCESSOR': [], 'TARGETER': []}
for li in range(NL):
    ang = np.mean(per_layer_angles[li])
    delta = ang - prev_ang
    m = machine_of(li)
    machine_growth[m].append(delta)
    if li < 6 or li > 23 or li % 4 == 0:
        print(f"  L{li:2d}   {m:>12s} {ang:6.2f}° {delta:+8.2f}°")
    prev_ang = ang

print(f"\n  TRANSFER FUNCTION SUMMARY:")
for m in ['COMPRESSOR', 'PROCESSOR', 'TARGETER']:
    deltas = machine_growth[m]
    total = sum(deltas)
    mean_delta = np.mean(deltas)
    n_layers = len(deltas)
    # Fit linear model: angle(l) = α·angle(l-1) + β within each machine
    print(f"    {m:>12s}: {n_layers:2d} layers, total Δ={total:+6.2f}°, mean Δ/layer={mean_delta:+5.2f}°")

# Fit per-machine recurrence from per-layer angles
print(f"\n  PER-MACHINE RECURRENCE: angle(l) = α·angle(l-1) + β")
for m_name, m_layers in [("COMPRESSOR", COMPRESSOR), ("PROCESSOR", PROCESSOR), ("TARGETER", TARGETER)]:
    if len(m_layers) < 2: continue
    x_vals = [np.mean(per_layer_angles[li]) for li in m_layers[:-1]]
    y_vals = [np.mean(per_layer_angles[li]) for li in m_layers[1:]]
    x = np.array(x_vals); y = np.array(y_vals)
    if len(x) >= 2 and np.std(x) > 0.01:
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        alpha, beta = result[0]
        eq = beta / (1 - alpha) if abs(1 - alpha) > 0.01 else float('inf')
        residuals = y - (alpha * x + beta)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"    {m_name:>12s}: α={alpha:+.4f} β={beta:+.4f} → eq={eq:.2f}° RMSE={rmse:.3f}°")
        if alpha < 0:
            print(f"                   → OSCILLATORY (α<0): overshoots and corrects")
        elif alpha < 1:
            print(f"                   → CONVERGENT (0<α<1): smoothly approaches equilibrium")
        else:
            print(f"                   → DIVERGENT (α≥1): error grows")
    else:
        print(f"    {m_name:>12s}: insufficient variance for fit")

# ================================================================
# TEST 4: GATE MEDIUM VERIFICATION
# ================================================================
print(f"\n" + "="*70)
print("  TEST 4: GATE MEDIUM VERIFICATION")
print("  (Gate state distribution matches machine boundaries)")
print("="*70)

# Measure gate state distribution at each layer
gate_distributions = defaultdict(lambda: {'C': 0, 'P-': 0, 'P+': 0, 'X': 0, 'total': 0})

for pi, prompt in enumerate(TEST_PROMPTS[:5]):  # 5 prompts for speed
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    # Capture gate activations
    gate_acts = {}
    hooks = []
    for li in range(NL):
        def make_gate_hook(idx):
            def hk(mod, inp, output):
                # output of gate_proj = pre-SiLU gate activation
                gate_acts[idx] = output[0, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].mlp.gate_proj.register_forward_hook(make_gate_hook(li)))

    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()

    for li in range(NL):
        g = gate_acts[li]
        boundary = LOG_PHI  # ±log(φ) ≈ ±0.481
        n_total = g.numel()
        n_contract = (g < -boundary).sum().item()
        n_preserve_neg = ((g >= -boundary) & (g < 0)).sum().item()
        n_preserve_pos = ((g >= 0) & (g <= boundary)).sum().item()
        n_expand = (g > boundary).sum().item()
        gate_distributions[li]['C'] += n_contract
        gate_distributions[li]['P-'] += n_preserve_neg
        gate_distributions[li]['P+'] += n_preserve_pos
        gate_distributions[li]['X'] += n_expand
        gate_distributions[li]['total'] += n_total

print(f"\n  {'Lyr':>3s} {'Machine':>12s} | {'CONTRACT':>8s} {'PRES-':>6s} {'PRES+':>6s} {'EXPAND':>6s} | {'Dominant':>8s}")
print("  " + "-"*65)
for li in range(NL):
    d = gate_distributions[li]
    t = d['total'] if d['total'] > 0 else 1
    pc = d['C']/t*100; ppn = d['P-']/t*100; ppp = d['P+']/t*100; px = d['X']/t*100
    m = machine_of(li)
    dominant = max([('C', pc), ('P-', ppn), ('P+', ppp), ('X', px)], key=lambda x: x[1])
    if li < 6 or li > 23 or li % 3 == 0:
        print(f"  L{li:2d} {m:>12s} | {pc:7.1f}% {ppn:5.1f}% {ppp:5.1f}% {px:5.1f}% | {dominant[0]:>6s} {dominant[1]:.0f}%")

# Zone summaries
print(f"\n  GATE MEDIUM BY MACHINE:")
for m_name, m_layers in [("COMPRESSOR", COMPRESSOR), ("PROCESSOR", PROCESSOR), ("TARGETER", TARGETER)]:
    totals = {'C': 0, 'P-': 0, 'P+': 0, 'X': 0, 'total': 0}
    for li in m_layers:
        for k in totals: totals[k] += gate_distributions[li][k]
    t = totals['total'] if totals['total'] > 0 else 1
    pc = totals['C']/t*100; ppn = totals['P-']/t*100
    ppp = totals['P+']/t*100; px = totals['X']/t*100
    preserve_total = ppn + ppp
    print(f"    {m_name:>12s}: CONTRACT={pc:.1f}%  PRESERVE={preserve_total:.1f}%  EXPAND={px:.1f}%")

# ================================================================
# SYNTHESIS
# ================================================================
print(f"\n" + "="*70)
print("  SYNTHESIS: IS THE COMPOUND MACHINE HYPOTHESIS SUPPORTED?")
print("="*70)

# Check: does independent linearization outperform?
global_acc = sum(config_results['global']['top1_match']) / len(config_results['global']['top1_match']) * 100
comp_acc = sum(config_results['compressor']['top1_match']) / len(config_results['compressor']['top1_match']) * 100
proc_acc = sum(config_results['processor']['top1_match']) / len(config_results['processor']['top1_match']) * 100
targ_acc = sum(config_results['targeter']['top1_match']) / len(config_results['targeter']['top1_match']) * 100

print(f"\n  PREDICTION 1: Individual machines should have higher accuracy than global")
print(f"    Global approx:       {global_acc:.0f}%")
print(f"    Compressor only:     {comp_acc:.0f}%")
print(f"    Processor only:      {proc_acc:.0f}%")
print(f"    Targeter only:       {targ_acc:.0f}%")
individual_better = all(x >= global_acc for x in [comp_acc, proc_acc, targ_acc])
print(f"    → {'SUPPORTED' if individual_better else 'MIXED'}: individual ≥ global = {individual_better}")

print(f"\n  PREDICTION 2: Machines operate in different gate media")
# Already printed above — summarize
print(f"    → Check gate distributions above: CONTRACT should dominate Compressor & Targeter,")
print(f"      PRESERVE should dominate Processor")

print(f"\n  PREDICTION 3: Error sum ≠ global (nonlinear composition)")
ratio = linear_sum / global_final if global_final > 0 else float('inf')
nonlinear = abs(ratio - 1.0) > 0.1
print(f"    Sum/Global ratio: {ratio:.3f}")
print(f"    → {'SUPPORTED (nonlinear)' if nonlinear else 'NOT SUPPORTED (linear)'}")

print(f"\n  PREDICTION 4: Different transfer functions per machine")
print(f"    → Check recurrence fits above: Compressor & Targeter should be oscillatory,")
print(f"      Processor should be convergent")

# Save results
save_data = {
    'test1_independent_linearization': {
        cn: {
            'top1_accuracy': sum(config_results[cn]['top1_match']) / max(len(config_results[cn]['top1_match']), 1),
            'angles': {bname: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                       for bname, vals in config_results[cn]['angles'].items()},
            'drifts': {bname: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                       for bname, vals in config_results[cn]['drifts'].items()},
        }
        for cn in configs if cn != 'baseline'
    },
    'test1_additivity': {
        'global_final_angle': float(global_final),
        'sum_individual_angles': float(linear_sum),
        'ratio': float(ratio),
    },
    'test2_interface_dimensionality': {
        cn: {
            bname: {
                'effective_rank': float(effective_rank(config_results[cn]['boundary_errors'][bname]))
                    if len(config_results[cn]['boundary_errors'][bname]) >= 2 else None,
                'mean_error_norm': float(np.mean([e.norm().item() for e in config_results[cn]['boundary_errors'][bname]]))
                    if config_results[cn]['boundary_errors'][bname] else None,
            }
            for bname in ['comp_out', 'proc_out', 'targ_out']
        }
        for cn in configs if cn != 'baseline'
    },
    'test3_per_layer_angles': {
        str(li): {'mean': float(np.mean(per_layer_angles[li])),
                  'std': float(np.std(per_layer_angles[li]))}
        for li in range(NL)
    },
    'test3_machine_growth': {
        m: {'total_delta': float(sum(deltas)), 'mean_delta': float(np.mean(deltas)),
            'n_layers': len(deltas)}
        for m, deltas in machine_growth.items()
    },
    'test4_gate_distributions': {
        str(li): {
            k: gate_distributions[li][k] / max(gate_distributions[li]['total'], 1)
            for k in ['C', 'P-', 'P+', 'X']
        }
        for li in range(NL)
    },
}

out_path = os.path.join(results_dir, 'phase10q_compound_machine.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\n  Saved to {out_path}")
print("="*70)
print("  PHASE 10q COMPLETE")
print("="*70)
