#!/usr/bin/env python3
"""
Phase 10o: Critical Angle Threshold & Lagrange Points
======================================================

The Shadow Orbit (Finding 96, Doc 260) shows that the residual stream absorbs
approximate attention into a stable displaced orbit at ~78°. With zone-aware
anchoring (Finding 92), accuracy goes from 0/15 to 12/15.

Key questions:
1. CRITICAL ANGLE: What angle(h, h') separates correct from incorrect prediction?
2. ANGLE vs ACCURACY: Is there a sharp phase transition (Lagrange boundary)?
3. LAGRANGE POINTS: Are there discrete stable angles (like L4/L5 at 60°)?
4. ZONE CONTRIBUTION: How much angle does each zone contribute?

Method:
- Run model with all-real QK to get reference hidden states at every layer
- Run model with various anchor configs to get approximate hidden states
- Measure angle(h_real, h_approx) at L27 for each config
- Correlate angle with prediction accuracy across 15 prompts

If L4/L5 exist, we expect:
- Discrete "plateaus" of angle (not a smooth curve)
- Sharp transition from correct to incorrect at a specific angle
- The critical angle may relate to φ-constants (60° for L4/L5 in celestial mechanics)
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10o: CRITICAL ANGLE THRESHOLD & LAGRANGE POINTS")
print("="*80)

results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)
layer_cls = {}
for ls in phase9a['layer_summary']:
    layer_cls[ls['layer']] = {'fixed': set(ls['fixed_heads']), 'routing': set(ls['routing_heads'])}

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()

NL=28; NH=28; NKV=4; HD=128; HPK=7; HDIM=3584; MAXS=64; ROPE_THETA=1e6

def phi_softmax(s, dim=-1):
    s = s - s.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI); return p / p.sum(dim=dim, keepdim=True)

def rope_cache(slen, dev, dt):
    inv = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,device=dev,dtype=torch.float32)/HD))
    pos = torch.arange(slen, device=dev, dtype=torch.float32)
    f = torch.outer(pos, inv); e = torch.cat((f,f), dim=-1)
    return e.cos().to(dt)[None,None], e.sin().to(dt)[None,None]

def apply_rope(x, cos, sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return x*cos + torch.cat((-x2, x1), dim=-1)*sin

inv_freq = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,dtype=torch.float32)/HD))

def rope_rotate_vector(v, delta, inv_freq):
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    v1, v2 = v[:len(v)//2], v[len(v)//2:]
    return v * cos_d + torch.cat((-v2, v1)) * sin_d

def rope_rotate_matrix_cols(M, delta, inv_freq):
    freqs = delta * inv_freq
    cos_d = torch.cat((freqs.cos(), freqs.cos()))
    sin_d = torch.cat((freqs.sin(), freqs.sin()))
    M1, M2 = M[:HD//2, :], M[HD//2:, :]
    return M * cos_d.unsqueeze(1) + torch.cat((-M2, M1), dim=0) * sin_d.unsqueeze(1)

# ================================================================
# Build bias-aware tables
# ================================================================
print("\nExtracting weights & building tables...")
head_tables = {}
for li in range(NL):
    attn = model.model.layers[li].self_attn
    ident = torch.eye(HDIM, device="cuda", dtype=torch.bfloat16)
    Wq = torch.zeros(NH,HD,HDIM,dtype=torch.float32)
    Wk = torch.zeros(NKV,HD,HDIM,dtype=torch.float32)
    for s in range(0,HDIM,512):
        e = min(s+512,HDIM); chunk = ident[s:e].unsqueeze(0)
        with torch.no_grad():
            qo=attn.q_proj(chunk).float(); ko=attn.k_proj(chunk).float()
        qr=qo[0].reshape(-1,NH,HD); kr=ko[0].reshape(-1,NKV,HD)
        for h in range(NH): Wq[h,:,s:e]=qr[:,h,:].T
        for g in range(NKV): Wk[g,:,s:e]=kr[:,g,:].T
    zi = torch.zeros(1,1,HDIM,device="cuda",dtype=torch.bfloat16)
    with torch.no_grad():
        qb=attn.q_proj(zi).float()[0,0]; kb=attn.k_proj(zi).float()[0,0]
    bq = qb.reshape(NH,HD).cpu(); bk = kb.reshape(NKV,HD).cpu()
    for h in range(NH): Wq[h] -= bq[h].unsqueeze(1)
    for g in range(NKV): Wk[g] -= bk[g].unsqueeze(1)

    routing = layer_cls[li]['routing']
    for h in routing:
        g = h // HPK
        sc = 1.0/math.sqrt(HD)
        bl=torch.zeros(MAXS); cq=torch.zeros(MAXS,HDIM); ck=torch.zeros(MAXS,HDIM)
        for delta in range(MAXS):
            bk_rot = rope_rotate_vector(bk[g], delta, inv_freq)
            Wk_rot = rope_rotate_matrix_cols(Wk[g], delta, inv_freq)
            bl[delta] = (bq[h] @ bk_rot) * sc
            cq[delta] = (Wq[h].T @ bk_rot) * sc
            ck[delta] = (Wk_rot.T @ bq[h]) * sc
        head_tables[(li, h)] = {'baseline': bl, 'c_q': cq, 'c_k': ck}
    del Wq, Wk; torch.cuda.empty_cache()
    if li % 7 == 0: print(f"  Layer {li} done")
print(f"  {len(head_tables)} head tables ready\n")


# ================================================================
# Attention functions
# ================================================================
def attn_real_qk(layer_idx, h_normed, attn_module):
    batch, seq_len, _ = h_normed.shape
    with torch.no_grad():
        Q = attn_module.q_proj(h_normed).to(torch.bfloat16)
        K = attn_module.k_proj(h_normed).to(torch.bfloat16)
        V_full = attn_module.v_proj(h_normed)
    Q = Q.reshape(batch, seq_len, NH, HD).transpose(1, 2)
    K = K.reshape(batch, seq_len, NKV, HD).transpose(1, 2)
    V_kv = V_full.reshape(batch, seq_len, NKV, HD)
    V_exp = V_kv.repeat_interleave(HPK, dim=2)
    cos, sin = rope_cache(seq_len, h_normed.device, torch.bfloat16)
    Q = apply_rope(Q, cos, sin); K = apply_rope(K, cos, sin)
    K_exp = K.repeat_interleave(HPK, dim=1)
    attn_out = torch.zeros(batch, seq_len, NH, HD, device=h_normed.device, dtype=h_normed.dtype)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
    for hd in range(NH):
        sc = (Q[0, hd] @ K_exp[0, hd].T / math.sqrt(HD)).float()
        sc.masked_fill_(mask, float('-inf'))
        w = phi_softmax(sc, dim=-1)
        attn_out[0, :, hd, :] = (w.to(torch.bfloat16) @ V_exp[0, :, hd, :]).to(h_normed.dtype)
    combined = attn_out.reshape(batch, seq_len, NH * HD)
    with torch.no_grad():
        return attn_module.o_proj(combined)


def attn_bias_aware(layer_idx, h_normed, attn_module):
    batch, seq_len, _ = h_normed.shape
    fixed = layer_cls[layer_idx]['fixed']
    routing = layer_cls[layer_idx]['routing']
    with torch.no_grad():
        V_full = attn_module.v_proj(h_normed)
    V_kv = V_full.reshape(batch, seq_len, NKV, HD)
    V_exp = V_kv.repeat_interleave(HPK, dim=2)
    attn_out = torch.zeros(batch, seq_len, NH, HD, device=h_normed.device, dtype=h_normed.dtype)
    for h in fixed:
        attn_out[0, :, h, :] = V_exp[0, 0, h, :]
    h_float = h_normed[0].float().cpu()
    for h in routing:
        tbl = head_tables[(layer_idx, h)]
        scores = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1):
                d = i - j
                bl = tbl['baseline'][d].item()
                cq_val = (h_float[i] @ tbl['c_q'][d]).item()
                ck_val = (tbl['c_k'][d] @ h_float[j]).item()
                scores[i, j] = bl + cq_val + ck_val
        scores = scores.to(h_normed.device)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h_normed.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = phi_softmax(scores.float(), dim=-1)
        v_h = V_exp[0, :, h, :].float()
        attn_out[0, :, h, :] = (weights @ v_h).to(h_normed.dtype)
    combined = attn_out.reshape(batch, seq_len, NH * HD)
    with torch.no_grad():
        return attn_module.o_proj(combined)


# ================================================================
# Hook-based forward with hidden state capture
# ================================================================
def run_with_hooks_capture(input_ids, attn_fn_map, capture_layers=None):
    """Run model with attention hooks AND capture hidden states at specified layers."""
    if capture_layers is None:
        capture_layers = {NL - 1}  # default: capture after last layer
    hooks = []
    captured = {}

    # Attention replacement hooks
    for layer_idx, attn_fn in attn_fn_map.items():
        def make_attn_hook(li, fn):
            def hook_fn(module, args, kwargs, output):
                h = args[0] if args else kwargs.get('hidden_states')
                if h is None: return output
                geo = fn(li, h, module)
                return (geo,) + output[1:] if isinstance(output, tuple) else geo
            return hook_fn
        hk = model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_attn_hook(layer_idx, attn_fn), with_kwargs=True)
        hooks.append(hk)

    # Hidden state capture hooks (after the full layer, not just attention)
    for layer_idx in capture_layers:
        def make_capture_hook(li):
            def hook_fn(module, args, output):
                # output is the hidden state after the full layer (attn + FFN + residual)
                h_out = output[0] if isinstance(output, tuple) else output
                captured[li] = h_out[0, -1, :].detach().float().cpu()  # last position
            return hook_fn
        hk = model.model.layers[layer_idx].register_forward_hook(
            make_capture_hook(layer_idx))
        hooks.append(hk)

    try:
        with torch.no_grad():
            out = model(input_ids, return_dict=True)
        logits = out.logits
    finally:
        for hk in hooks:
            hk.remove()
    return logits, captured


# ================================================================
# Test prompts
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

CAPTURE_LAYERS = {7, 13, 19, 26, 27}  # checkpoints through the network

# ================================================================
# Collect reference: all-real-QK hidden states
# ================================================================
print("Collecting reference (all real QK) hidden states...")
ref_data = {}
for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    all_real = {li: attn_real_qk for li in range(NL)}
    logits, captured = run_with_hooks_capture(ids, all_real, CAPTURE_LAYERS)
    token = logits[0, -1, :].float().argmax().item()
    ref_data[pi] = {'token': token, 'hidden': captured, 'prompt': prompt}
    if pi % 5 == 0:
        print(f"  Prompt {pi}: '{prompt}' -> {tokenizer.decode([token])}")
print(f"  {len(TEST_PROMPTS)} references ready.\n")


# ================================================================
# Evaluate a config: accuracy + angle at each checkpoint
# ================================================================
def evaluate_config(name, anchor_layers):
    """Run a config, measure accuracy AND angle(h_real, h_approx) at checkpoints."""
    anchor_set = set(anchor_layers)
    approx_set = set(range(NL)) - anchor_set
    cfg = {}
    for li in anchor_set: cfg[li] = attn_real_qk
    for li in approx_set: cfg[li] = attn_bias_aware

    n_match = 0
    angles = defaultdict(list)  # layer -> list of angles across prompts
    drifts = defaultdict(list)  # layer -> list of ||eps||/||h||
    cos_eh = defaultdict(list)  # layer -> list of cos(eps, h)

    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        logits, captured = run_with_hooks_capture(ids, cfg, CAPTURE_LAYERS)

        if logits[0, -1, :].float().argmax().item() == ref_data[pi]['token']:
            n_match += 1

        for li in CAPTURE_LAYERS:
            h_real = ref_data[pi]['hidden'][li]
            h_approx = captured[li]
            eps = h_approx - h_real

            # Angle between real and approximate
            cos_theta = F.cosine_similarity(h_real.unsqueeze(0), h_approx.unsqueeze(0)).item()
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle = math.degrees(math.acos(cos_theta))
            angles[li].append(angle)

            # Drift magnitude
            r = eps.norm().item() / (h_real.norm().item() + 1e-10)
            drifts[li].append(r)

            # cos(eps, h)
            c = F.cosine_similarity(eps.unsqueeze(0), h_real.unsqueeze(0)).item()
            cos_eh[li].append(c)

    return {
        'name': name,
        'n_anchors': len(anchor_set),
        'anchor_layers': sorted(anchor_set),
        'accuracy': n_match,
        'total': len(TEST_PROMPTS),
        'angles': {li: angles[li] for li in CAPTURE_LAYERS},
        'drifts': {li: drifts[li] for li in CAPTURE_LAYERS},
        'cos_eh': {li: cos_eh[li] for li in CAPTURE_LAYERS},
    }


# ================================================================
# TEST 1: Full sweep — 0 to 28 anchors
# ================================================================
print("="*80)
print("  TEST 1: ACCURACY vs ANGLE — Anchor Density Sweep")
print("  Measuring angle(h_real, h_approx) at L27 for each config")
print("="*80)
print()

all_configs = []

# 1a. All approximate (0 anchors = full shadow orbit)
all_configs.append(("All approximate", []))

# 1b. Single anchor at each zone boundary
for li in [0, 3, 7, 14, 21, 27]:
    all_configs.append((f"Single anchor L{li}", [li]))

# 1c. DRUM only (L0-3)
all_configs.append(("DRUM only (L0-3)", list(range(4))))

# 1d. DRUM + MUSIC
all_configs.append(("DRUM + MUSIC (L0-3,27)", list(range(4)) + [27]))

# 1e. DRUM + mid + MUSIC
all_configs.append(("DRUM+mid+MUSIC (L0-3,14,27)", list(range(4)) + [14, 27]))

# 1f. Zone-aware configs from phase10e
all_configs.append(("DRUM+every4+MUSIC", list(range(4)) + [7, 11, 15, 19, 23] + [27]))
all_configs.append(("DRUM+every3+MUSIC", list(range(4)) + [7, 10, 13, 16, 19, 22, 25] + [27]))
all_configs.append(("DRUM+every2+MUSIC", list(range(4)) + list(range(5, 27, 2)) + [27]))

# 1g. Uniform stride
for stride in [14, 7, 4, 3, 2]:
    layers = sorted(set(list(range(0, NL, stride)) + [0, 27]))
    all_configs.append((f"Uniform stride {stride}", layers))

# 1h. All real (28 anchors = no shadow orbit)
all_configs.append(("All real QK", list(range(NL))))

# Sort by anchor count for nice display
all_configs.sort(key=lambda x: len(x[1]))

print(f"  {'Config':<40s}  {'Anch':>4s}  {'Acc':>5s}  {'L27 angle':>9s}  {'L27 drift':>9s}  {'L27 cos_eh':>10s}")
print("  " + "-" * 90)

results_list = []
for name, anchor_layers in all_configs:
    r = evaluate_config(name, anchor_layers)
    results_list.append(r)

    # Get L27 stats
    a27 = r['angles'].get(27, [0])
    d27 = r['drifts'].get(27, [0])
    c27 = r['cos_eh'].get(27, [0])
    mean_angle = np.mean(a27)
    mean_drift = np.mean(d27)
    mean_cos = np.mean(c27)

    print(f"  {name:<40s}  {r['n_anchors']:2d}/28  {r['accuracy']:2d}/15  "
          f"{mean_angle:7.2f} deg  {mean_drift:7.4f}    {mean_cos:+.4f}")

print()


# ================================================================
# TEST 2: PHASE TRANSITION — Is there a sharp angle boundary?
# ================================================================
print("="*80)
print("  TEST 2: PHASE TRANSITION ANALYSIS")
print("  Is there a sharp critical angle separating correct from incorrect?")
print("="*80)
print()

# Collect all (angle, correct) pairs across all configs and prompts
angle_correct_pairs = []
for r in results_list:
    for pi in range(len(TEST_PROMPTS)):
        if 27 in r['angles'] and pi < len(r['angles'][27]):
            angle = r['angles'][27][pi]
            correct = 1 if (r['accuracy'] > 0 and
                           # Re-derive per-prompt correctness from the data
                           True) else 0
            angle_correct_pairs.append((angle, r['accuracy'] / r['total'],
                                        r['n_anchors'], r['name']))

# Per-config summary: angle vs accuracy
print(f"  {'Config':<40s}  {'Anchors':>7s}  {'Accuracy':>8s}  {'Mean angle':>10s}  {'Std angle':>9s}")
print("  " + "-" * 80)

angle_accuracy_data = []
for r in results_list:
    a27 = r['angles'].get(27, [0])
    mean_a = np.mean(a27)
    std_a = np.std(a27)
    acc_pct = r['accuracy'] / r['total'] * 100
    print(f"  {r['name']:<40s}  {r['n_anchors']:2d}/28   {acc_pct:5.1f}%     {mean_a:7.2f} deg  {std_a:6.2f} deg")
    angle_accuracy_data.append({
        'name': r['name'],
        'n_anchors': r['n_anchors'],
        'accuracy': r['accuracy'],
        'mean_angle_L27': float(mean_a),
        'std_angle_L27': float(std_a),
        'mean_drift_L27': float(np.mean(r['drifts'].get(27, [0]))),
        'mean_cos_eh_L27': float(np.mean(r['cos_eh'].get(27, [0]))),
    })

print()


# ================================================================
# TEST 3: LAGRANGE POINT SEARCH
# ================================================================
print("="*80)
print("  TEST 3: LAGRANGE POINT SEARCH")
print("  Are there discrete stable angles (plateaus)?")
print("  L4/L5 in celestial mechanics sit at 60 degrees.")
print("="*80)
print()

# Sort configs by angle
sorted_by_angle = sorted(angle_accuracy_data, key=lambda x: x['mean_angle_L27'])

print("  Configs sorted by increasing angle:")
print(f"  {'Config':<40s}  {'Angle':>8s}  {'Accuracy':>8s}  {'Delta angle':>11s}")
print("  " + "-" * 75)

prev_angle = 0
for d in sorted_by_angle:
    delta = d['mean_angle_L27'] - prev_angle
    marker = ""
    # Check for plateaus (small angle change between configs)
    if delta < 2.0 and prev_angle > 0:
        marker = " <-- PLATEAU?"
    # Check for jumps (large angle change)
    if delta > 15.0 and prev_angle > 0:
        marker = " <-- JUMP"
    acc_pct = d['accuracy'] / 15 * 100
    print(f"  {d['name']:<40s}  {d['mean_angle_L27']:7.2f}   {acc_pct:5.1f}%      {delta:+7.2f}{marker}")
    prev_angle = d['mean_angle_L27']

print()

# Check for the critical transition
print("  CRITICAL TRANSITION SEARCH:")
print("  Looking for the angle where accuracy drops below 50%...")
print()

transition_found = False
for i in range(len(sorted_by_angle) - 1):
    a1 = sorted_by_angle[i]
    a2 = sorted_by_angle[i+1]
    acc1 = a1['accuracy'] / 15 * 100
    acc2 = a2['accuracy'] / 15 * 100
    if acc1 >= 50 and acc2 < 50:
        mid_angle = (a1['mean_angle_L27'] + a2['mean_angle_L27']) / 2
        print(f"  TRANSITION between:")
        print(f"    {a1['name']:<35s}  angle={a1['mean_angle_L27']:6.2f}  acc={acc1:.0f}%")
        print(f"    {a2['name']:<35s}  angle={a2['mean_angle_L27']:6.2f}  acc={acc2:.0f}%")
        print(f"    Critical angle estimate: ~{mid_angle:.1f} deg")

        # Check phi-related angles
        print(f"\n  PHI-RELATED ANGLE CANDIDATES:")
        candidates = [
            (60.0, "60 deg (L4/L5 classical)"),
            (180/PHI, f"180/phi = {180/PHI:.2f} deg"),
            (180/PHI**2, f"180/phi^2 = {180/PHI**2:.2f} deg"),
            (math.degrees(math.acos(1/PHI)), f"arccos(1/phi) = {math.degrees(math.acos(1/PHI)):.2f} deg"),
            (math.degrees(math.acos(1/PHI**2)), f"arccos(1/phi^2) = {math.degrees(math.acos(1/PHI**2)):.2f} deg"),
            (math.degrees(math.atan(PHI)), f"arctan(phi) = {math.degrees(math.atan(PHI)):.2f} deg"),
            (math.degrees(1/PHI), f"1/phi radians = {math.degrees(1/PHI):.2f} deg"),
            (math.degrees(math.acos(PHI-1)), f"arccos(phi-1) = arccos(1/phi) = {math.degrees(math.acos(PHI-1)):.2f} deg"),
            (45.0, "45 deg (pi/4)"),
            (36.0, "36 deg (pi/5, golden angle related)"),
            (72.0, "72 deg (2*pi/5, golden angle related)"),
        ]
        for angle_val, desc in sorted(candidates, key=lambda x: abs(x[0] - mid_angle)):
            delta = abs(angle_val - mid_angle)
            marker = " <-- CLOSE" if delta < 3 else ""
            print(f"    {desc:<45s}  delta = {delta:.2f} deg{marker}")

        transition_found = True
        break

if not transition_found:
    print("  No sharp 50% transition found. Checking for any accuracy drop...")
    for i in range(len(sorted_by_angle) - 1):
        a1 = sorted_by_angle[i]
        a2 = sorted_by_angle[i+1]
        if a1['accuracy'] > a2['accuracy']:
            print(f"    Drop: {a1['name']} ({a1['accuracy']}/15, {a1['mean_angle_L27']:.1f} deg) "
                  f"-> {a2['name']} ({a2['accuracy']}/15, {a2['mean_angle_L27']:.1f} deg)")

print()


# ================================================================
# TEST 4: ANGLE PROFILE THROUGH LAYERS
# ================================================================
print("="*80)
print("  TEST 4: ANGLE PROFILE THROUGH NETWORK")
print("  How does angle evolve at checkpoints L7, L13, L19, L26, L27?")
print("="*80)
print()

# Show angle at each checkpoint for a few key configs
key_configs = [r for r in results_list if r['name'] in [
    "All approximate", "DRUM only (L0-3)", "DRUM + MUSIC (L0-3,27)",
    "DRUM+every4+MUSIC", "DRUM+every2+MUSIC", "All real QK"
]]

if not key_configs:
    key_configs = results_list[:6]

chk_layers = sorted(CAPTURE_LAYERS)
header = f"  {'Config':<35s}"
for li in chk_layers:
    header += f"  {'L'+str(li):>7s}"
header += "  Accuracy"
print(header)
print("  " + "-" * (40 + 10 * len(chk_layers)))

for r in key_configs:
    line = f"  {r['name']:<35s}"
    for li in chk_layers:
        a = r['angles'].get(li, [0])
        line += f"  {np.mean(a):6.1f}d"
    line += f"  {r['accuracy']:2d}/15"
    print(line)

print()


# ================================================================
# TEST 5: PER-PROMPT ANGLE AT L27 — INDIVIDUAL TRANSITION
# ================================================================
print("="*80)
print("  TEST 5: PER-PROMPT ANALYSIS")
print("  Does each prompt have the same critical angle?")
print("="*80)
print()

# For each config, show per-prompt angle and correctness
# Focus on configs near the transition
near_transition = [r for r in results_list
                   if 3 <= r['accuracy'] <= 13 and r['n_anchors'] > 0]

if not near_transition:
    near_transition = [r for r in results_list if r['n_anchors'] > 0][:3]

for r in near_transition[:3]:
    print(f"  Config: {r['name']} ({r['accuracy']}/15)")
    print(f"  {'Prompt':<45s}  {'Angle':>7s}  {'Drift':>7s}  {'cos_eh':>7s}")
    print("  " + "-" * 70)

    ids_list = [tokenizer.encode(p, return_tensors="pt").to("cuda") for p in TEST_PROMPTS]
    # Re-run to get per-prompt correctness
    anchor_set = set(r['anchor_layers'])
    approx_set = set(range(NL)) - anchor_set
    cfg = {}
    for li in anchor_set: cfg[li] = attn_real_qk
    for li in approx_set: cfg[li] = attn_bias_aware

    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        logits, captured = run_with_hooks_capture(ids, cfg, {27})
        pred = logits[0, -1, :].float().argmax().item()
        correct = "OK" if pred == ref_data[pi]['token'] else "FAIL"

        h_real = ref_data[pi]['hidden'][27]
        h_approx = captured[27]
        eps = h_approx - h_real
        cos_theta = F.cosine_similarity(h_real.unsqueeze(0), h_approx.unsqueeze(0)).item()
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = math.degrees(math.acos(cos_theta))
        drift = eps.norm().item() / (h_real.norm().item() + 1e-10)
        c_eh = F.cosine_similarity(eps.unsqueeze(0), h_real.unsqueeze(0)).item()

        print(f"  {prompt:<45s}  {angle:6.1f}d  {drift:6.3f}   {c_eh:+.3f}  [{correct}]")

    print()


# ================================================================
# SYNTHESIS
# ================================================================
print("="*80)
print("  SYNTHESIS: THE LAGRANGE BOUNDARY")
print("="*80)
print()

# Sort by angle, identify the critical region
sorted_data = sorted(angle_accuracy_data, key=lambda x: x['mean_angle_L27'])

print("  ANGLE -> ACCURACY MAP:")
print(f"  {'Angle (deg)':>11s}  {'Accuracy':>8s}  {'Anchors':>7s}  Config")
print("  " + "-" * 70)
for d in sorted_data:
    acc = d['accuracy'] / 15 * 100
    bar = "#" * int(acc / 5)
    print(f"  {d['mean_angle_L27']:9.2f} deg  {acc:5.1f}%   {d['n_anchors']:2d}/28   {d['name'][:35]:<35s}  |{bar}|")

# Summary statistics
angles_all = [d['mean_angle_L27'] for d in sorted_data]
accs_all = [d['accuracy'] / 15 * 100 for d in sorted_data]

# Find angle ranges for different accuracy bands
print()
print("  ACCURACY BANDS:")
for threshold in [100, 80, 60, 40, 20, 0]:
    matching = [d for d in sorted_data if d['accuracy'] / 15 * 100 >= threshold]
    if matching:
        max_angle = max(d['mean_angle_L27'] for d in matching)
        print(f"    >={threshold:3d}%: angle <= {max_angle:.1f} deg ({len(matching)} configs)")

# Check specific phi angles
print()
print("  KEY ANGLE REFERENCE POINTS:")
phi_angles = {
    "0 deg (perfect)": 0.0,
    "arccos(1/phi) = 51.8 deg": math.degrees(math.acos(1/PHI)),
    "60 deg (L4/L5)": 60.0,
    "arccos(1/phi^2) = 67.5 deg": math.degrees(math.acos(1/PHI**2)),
    "72 deg (golden angle)": 72.0,
    "78 deg (shadow orbit)": 78.0,
    "90 deg (orthogonal)": 90.0,
}
for name, angle in sorted(phi_angles.items(), key=lambda x: x[1]):
    # Find closest config
    closest = min(sorted_data, key=lambda d: abs(d['mean_angle_L27'] - angle))
    print(f"    {name:<35s}  closest config: {closest['mean_angle_L27']:.1f} deg = "
          f"{closest['accuracy']}/15 ({closest['name'][:25]})")

print()

# Save results
save_data = {
    'configs': angle_accuracy_data,
    'phi_constants': {
        'arccos_1_over_phi': math.degrees(math.acos(1/PHI)),
        'arccos_1_over_phi2': math.degrees(math.acos(1/PHI**2)),
        'golden_angle': 72.0,
        'shadow_orbit': 78.0,
        '180_over_phi': 180/PHI,
        '180_over_phi2': 180/PHI**2,
    }
}
save_path = os.path.join(results_dir, 'phase10o_critical_angle.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=float)
print(f"  Saved to {save_path}")
print()
print("="*80)
print("  DONE -- Phase 10o Critical Angle & Lagrange Points")
print("="*80)
